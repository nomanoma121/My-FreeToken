"""Read a prefill chunk's non-resident bank rows with parallel direct reads, not page faults.

A prefill chunk streams every layer's whole expert bank to the GPU. The rows past the
--moe-bank-ram resident prefix are not registered with the device, so they go through pinned
staging buffers first (``OffloadMoeCache._staged_h2d``), and copying them out of the mapping
turns every row the page cache does not hold into a page fault: one thread, one readahead
window per fault, waiting on each. When the page cache is smaller than a rank's non-resident
rows -- the 64 GB host -- reading one layer evicts the one before, so every chunk reads nearly
all of them from the disk again that way. Measured on an RTX 2060 host (Ornith, 11 GiB of
non-resident rows, the server held to 13 GiB): 10.6 GiB re-read per chunk at 1.2 GiB/s with an
8192 kB readahead window, 9.2 s of a 13 s chunk; 0.11 GiB/s with no window at all. The window
that suits decode (256 kB) sits between the two.

This reads the same bytes from the file with ``pread`` on an ``O_DIRECT`` descriptor, from
several threads at once, straight into the pinned staging buffers:

* the rate is the drive's parallel rate, not one fault at a time, and does not depend on
  ``read_ahead_kb`` -- so the window can stay where decode wants it;
* nothing goes through the page cache, so a prefill no longer evicts the non-resident rows
  decode has been accumulating there (the reason a long generation speeds up as it goes).

A range the page cache already holds (at least ``FREETOKEN_BANK_PREAD_CACHED``, default 0.9 of
its pages, by ``mincore``) is still copied out of the mapping: from RAM that is a memcpy at
10+ GiB/s, which no disk read beats. ``FREETOKEN_BANK_PREAD=0`` turns the reader off;
``FREETOKEN_BANK_READ_THREADS`` (default 8) and ``FREETOKEN_BANK_READ_PIECE_MB`` (default 16)
size it. Where the filesystem refuses ``O_DIRECT`` the reads are buffered: still parallel and
still off the fault path, but they do fill the page cache.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import torch

ALIGN = 4096


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, "") or default))
    except ValueError:
        return default


class DirectRangeReader:
    """Parallel reads of file ranges into reusable page-aligned host buffers.

    Device-independent (the pinned-ness of the buffers is the caller's ``alloc``), so the read
    path is testable on any host. ``sets`` groups of ``threads`` buffers alternate, so a group
    can be refilled while the device still drains the previous one.
    """

    def __init__(self, path: str, *, threads: int = 8, piece_bytes: int = 16 << 20, sets: int = 2,
                 alloc: Callable[[int], torch.Tensor] | None = None, direct: bool = True) -> None:
        self.path = path
        self.threads = max(1, threads)
        self.piece = max(ALIGN, piece_bytes // ALIGN * ALIGN)
        self.direct = False
        fd = -1
        o_direct = getattr(os, "O_DIRECT", 0)
        if direct and o_direct:
            try:
                fd = os.open(path, os.O_RDONLY | o_direct)
                self.direct = True
            except OSError:
                fd = -1
        self._fd = fd if fd >= 0 else os.open(path, os.O_RDONLY)
        alloc = alloc or (lambda n: torch.empty(n, dtype=torch.uint8))
        # a buffer holds one piece plus the page on either side the aligned read spills into
        span = self.piece + 2 * ALIGN
        self.buffers: list[list[tuple[torch.Tensor, memoryview]]] = []
        for _ in range(sets):
            group = []
            for _ in range(self.threads):
                raw = alloc(span + ALIGN)
                skew = (-raw.data_ptr()) % ALIGN
                view = raw[skew:skew + span]
                group.append((view, memoryview(view.numpy()).cast("B")))
            self.buffers.append(group)
        self._pool = ThreadPoolExecutor(self.threads, thread_name_prefix="bank-read")
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._pool.shutdown(wait=True)
            os.close(self._fd)

    def _read(self, buf: memoryview, offset: int, nbytes: int) -> tuple[int, int]:
        """Read ``[offset, offset + nbytes)`` into ``buf``; returns (start of the data in buf, bytes)."""
        if self.direct:
            start = offset // ALIGN * ALIGN
            skew = offset - start
            want = -(-(skew + nbytes) // ALIGN) * ALIGN
        else:
            start, skew, want = offset, 0, nbytes
        got = 0
        while got < skew + nbytes:
            n = os.preadv(self._fd, [buf[got:want]], start + got)
            if n <= 0:
                raise OSError(f"{self.path}: short read at {start + got} ({got} of {skew + nbytes} bytes)")
            got += n
        return skew, nbytes

    def read(self, offset: int, nbytes: int, sink: Callable[[int, torch.Tensor, int], None],
             before_group: Callable[[int], None] | None = None) -> None:
        """Read ``nbytes`` from ``offset`` in pieces, ``threads`` at a time, and hand each piece to
        ``sink(position, host view, group set)`` in file order. ``before_group(set)`` runs before
        a set's buffers are refilled (the caller waits there for whatever last read them)."""
        pieces = [(p, min(self.piece, nbytes - p)) for p in range(0, nbytes, self.piece)]
        for g in range(0, len(pieces), self.threads):
            s = (g // self.threads) % len(self.buffers)
            if before_group is not None:
                before_group(s)
            group = pieces[g:g + self.threads]
            futures = [
                self._pool.submit(self._read, self.buffers[s][j][1], offset + p, n)
                for j, (p, n) in enumerate(group)
            ]
            for j, ((p, n), fut) in enumerate(zip(group, futures)):
                skew, _ = fut.result()
                sink(p, self.buffers[s][j][0][skew:skew + n], s)


class BankReader:
    """``OffloadMoeCache.bank_reader``: the prefill copy of a mapped bank's non-resident rows."""

    def __init__(self, banks, *, threads: int | None = None, piece_bytes: int | None = None,
                 cached_share: float | None = None) -> None:
        from freetoken.kernel.pinned import alloc_pinned_tensor

        self.banks = banks
        self._lo = banks._base
        self._hi = banks._base + len(banks._map)
        self._file_base = banks._map_offset - banks._base
        self.cached_share = float(os.environ.get("FREETOKEN_BANK_PREAD_CACHED", "") or 0.9) \
            if cached_share is None else cached_share
        self.reader = DirectRangeReader(
            banks.path,
            threads=threads or _env_int("FREETOKEN_BANK_READ_THREADS", 8),
            piece_bytes=piece_bytes or _env_int("FREETOKEN_BANK_READ_PIECE_MB", 16) << 20,
            alloc=lambda n: alloc_pinned_tensor(n, dtype=torch.uint8),
        )
        self._events = [torch.cuda.Event() for _ in self.reader.buffers]
        self._libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self._libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
        self._lock = threading.Lock()
        self.bytes_read = 0
        self.bytes_cached = 0

    def describe(self) -> str:
        r = self.reader
        return (f"{r.threads} threads x {r.piece >> 20} MiB, {'O_DIRECT' if r.direct else 'buffered (no O_DIRECT here)'}, "
                f"mapping copy when {self.cached_share:.0%} of a range is cached")

    def _share_cached(self, addr: int, nbytes: int) -> float:
        start = addr // mmap.PAGESIZE * mmap.PAGESIZE
        length = addr + nbytes - start
        count = -(-length // mmap.PAGESIZE)
        vec = ctypes.create_string_buffer(count)
        if self._libc.mincore(ctypes.c_void_p(start), length, vec) != 0:
            return 1.0  # cannot tell: the mapping copy is the path that always works
        raw = vec.raw
        return (count - raw.count(0)) / count

    def h2d(self, dst: torch.Tensor, src: torch.Tensor, prof=None) -> bool:
        """Copy ``src`` (a view of the bank mapping) to ``dst`` on the current stream by direct
        reads. False when ``src`` is not in the mapping or the page cache already holds it, and
        the caller copies it the usual way."""
        addr = src.data_ptr()
        nbytes = src.numel() * src.element_size()
        if nbytes == 0 or addr < self._lo or addr + nbytes > self._hi:
            return False
        if self._share_cached(addr, nbytes) >= self.cached_share:
            self.bytes_cached += nbytes
            return False
        d = dst.reshape(-1).view(torch.uint8)
        stream = torch.cuda.current_stream(dst.device)
        events = self._events
        waited = [0.0]

        def before_group(s: int) -> None:
            t = time.perf_counter()
            events[s].synchronize()  # the DMA that last read this set is done
            waited[0] += time.perf_counter() - t

        def sink(p: int, host: torch.Tensor, s: int) -> None:
            d[p:p + host.numel()].copy_(host, non_blocking=True)  # async DMA on the stream
            events[s].record(stream)

        started = time.perf_counter()
        with self._lock:
            self.reader.read(addr + self._file_base, nbytes, sink, before_group)
        if prof is not None:
            elapsed = time.perf_counter() - started
            prof.staged_piece(waited[0], elapsed - waited[0], nbytes, 0)
        self.bytes_read += nbytes
        return True

    def close(self) -> None:
        self.reader.close()
