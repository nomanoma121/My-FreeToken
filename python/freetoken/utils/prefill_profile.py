"""``--prefill-profile``: where one prefill forward's wall time goes, on each pipeline rank.

An offloaded MoE streams every layer's whole expert bank to the GPU once per prefill chunk, so
a chunk's time is mostly data movement, and which part of it dominates depends on the host:
the page cache holding (or not) the bank's file-backed rows, the disk behind them, the PCIe
link, and under ``--pp-size`` the other rank. None of that shows in the "Prefill batch" line,
whose throughput is the wall time between two such lines on the first rank.

One line per forward, per rank. The first four parts add up to the wall time of the forward:

* ``peers`` -- blocked in a pipeline send/recv, i.e. waiting for the neighbouring rank
* ``bank read`` -- host copies of expert rows that are not registered with the GPU (the
  --moe-bank-ram remainder) into the pinned staging buffers. Page faults on the file-backed
  rows happen inside these copies, so a disk that cannot keep up lands here
* ``PLE`` -- the per-layer-embedding rows read from disk before the forward launches
* ``GPU + rest`` -- everything else: waiting for the GPU (compute and transfers), kernel
  launches, the scheduler's own host work inside the forward

and the rest of the line explains the bank part: how many GiB went through the staging copy
and at what rate, the process's major page faults and storage reads over the forward, how much
of the file-backed rows the page cache held when the forward began, and the GPU-side duration
of the registered rows' transfers (their achieved GiB/s is the PCIe link under load).

Kept free of the engine and the kernels (torch, resource and /proc only) so the offload cache
and the pipeline transport can report into it without importing either.
"""

from __future__ import annotations

import resource
import time
from contextlib import contextmanager
from typing import Callable

import torch

_ACTIVE: "PrefillProfile | None" = None
_GIB = float(1 << 30)


def active() -> "PrefillProfile | None":
    """The profile of the prefill forward running on this process, or None."""
    return _ACTIVE


def _major_faults() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_majflt


def read_bytes(path: str = "/proc/self/io") -> int | None:
    """Bytes this process caused to be read from storage (page-cache fills included)."""
    try:
        with open(path, encoding="ascii") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


class PrefillProfile:
    def __init__(
        self,
        rank: int = 0,
        size: int = 1,
        residency: Callable[[], float] | None = None,
        log: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.label = f" [rank {rank}/{size}]" if size > 1 else ""
        self.residency = residency
        self.log = log or print
        self.clock = clock
        self._last_end: float | None = None
        self._reset(0)

    def _reset(self, tokens: int) -> None:
        self.tokens = tokens
        self.peers = 0.0
        self.bank_read = 0.0
        self.bank_read_bytes = 0
        self.bank_wait = 0.0
        self.bank_faults = 0
        self.ple = 0.0
        self.hot_bytes = 0
        self._hot_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.cold_share: float | None = None

    # ---- the forward ------------------------------------------------------------------------

    def begin(self, tokens: int) -> None:
        global _ACTIVE
        self._reset(tokens)
        if self.residency is not None:
            try:
                self.cold_share = self.residency()
            except Exception:  # noqa: BLE001 -- a diagnostic must not fail the forward
                self.residency = None
        self._faults0 = _major_faults()
        self._io0 = read_bytes()
        self.t0 = self.clock()
        _ACTIVE = self

    def abort(self) -> None:
        global _ACTIVE
        _ACTIVE = None

    def end(self, device: torch.device | None = None) -> str:
        global _ACTIVE
        _ACTIVE = None
        if device is not None and device.type == "cuda":
            # the forward's own readback has already synchronized the compute stream; this
            # waits out the copy stream so the transfer events below can be read
            torch.cuda.synchronize(device)
        now = self.clock()
        total = now - self.t0
        since = None if self._last_end is None else self.t0 - self._last_end
        self._last_end = now
        faults = _major_faults() - self._faults0
        io1 = read_bytes()
        disk = None if self._io0 is None or io1 is None else io1 - self._io0
        hot_s = sum(s.elapsed_time(e) for s, e in self._hot_events) / 1000.0
        line = self.format(total, faults=faults, disk=disk, hot_seconds=hot_s, since=since)
        self.log(line)
        return line

    def format(self, total: float, *, faults: int, disk: int | None, hot_seconds: float,
               since: float | None) -> str:
        rest = max(0.0, total - self.peers - self.bank_read - self.ple)
        rate = f" ({self.tokens / total:.0f} tok/s)" if total > 0 else ""
        parts = [f"peers {self.peers:.2f} s", f"bank read {self.bank_read:.2f} s"]
        if self.ple:
            parts.append(f"PLE {self.ple:.2f} s")
        parts.append(f"GPU + rest {rest:.2f} s")
        head = f"prefill profile{self.label}: {self.tokens} tokens in {total:.2f} s{rate} = " + " | ".join(parts)

        notes = []
        if self.bank_read_bytes:
            gib = self.bank_read_bytes / _GIB
            speed = f" at {gib / self.bank_read:.2f} GiB/s" if self.bank_read > 0 else ""
            notes.append(
                f"unregistered rows {gib:.2f} GiB copied{speed} ({self.bank_faults} major faults"
                + (f", {self.bank_wait:.2f} s more waiting on the GPU between pieces" if self.bank_wait >= 0.005 else "")
                + ")"
            )
        notes.append(f"process: {faults} major faults" + (f", {disk / _GIB:.2f} GiB read from storage" if disk is not None else ""))
        if self.cold_share is not None:
            notes.append(f"page cache held {100 * self.cold_share:.0f}% of the file-backed rows at start")
        if self.hot_bytes:
            gib = self.hot_bytes / _GIB
            speed = f" ({gib / hot_seconds:.1f} GiB/s)" if hot_seconds > 0 else ""
            notes.append(f"registered rows {gib:.2f} GiB transferred in {hot_seconds:.2f} s on the GPU{speed}")
        if since is not None:
            notes.append(f"{since:.2f} s since the previous prefill forward")
        return head + " || " + "; ".join(notes)

    # ---- reports from the movement code -----------------------------------------------------

    @contextmanager
    def peer_wait(self):
        t = self.clock()
        try:
            yield
        finally:
            self.peers += self.clock() - t

    @contextmanager
    def ple_fill(self):
        t = self.clock()
        try:
            yield
        finally:
            self.ple += self.clock() - t

    def staged_piece(self, wait_s: float, copy_s: float, nbytes: int, faults: int) -> None:
        self.bank_wait += wait_s
        self.bank_read += copy_s
        self.bank_read_bytes += nbytes
        self.bank_faults += faults

    def hot_copy_begin(self, stream) -> "torch.cuda.Event":
        start = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        return start

    def hot_copy_end(self, start: "torch.cuda.Event", stream, nbytes: int) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record(stream)
        self._hot_events.append((start, end))
        self.hot_bytes += nbytes


major_faults = _major_faults
