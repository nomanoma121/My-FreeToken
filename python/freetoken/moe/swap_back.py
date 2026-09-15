"""Page this process's swapped-out memory back in while the server is idle.

``--moe-bank-rewarm`` reads a mapped bank's cold rows back into the page cache after something
took them. The same pressure also pushes the server's own anonymous memory to swap, and that is
not page cache: the walk does not touch it, and the next request faults it back in one page at a
time. Measured on the two RTX 3060s (``guides/33`` §6.1): with the bank back at 100%, the short
prompt still took 5.8 s against 2.9-3.1 warm, in the run where the pressure had put 4.7 GiB of the
server in swap. Whether that swap is the whole remainder is what this module lets a run decide.

Only pages that are actually in swap are read, found through ``/proc/self/pagemap`` (bit 62 of an
entry is "swapped"; a process may read its own flags without privilege). Walking every readable
mapping instead would populate address space nobody ever touched -- CUDA and the pinned-memory
allocator reserve far more virtual memory than they use -- and ``/proc/self/smaps`` narrows the
search to the mappings whose ``Swap:`` is non-zero before any pagemap entry is read.

A run is brought in with ``madvise(MADV_POPULATE_READ)`` (Linux 5.14): a synchronous read fault
over the range, done in the kernel without holding the GIL, which is what makes "finished" mean
the pages are mapped again rather than queued (``MADV_WILLNEED`` on anonymous memory only starts
swap readahead and returns). The address space can change between the scan and the fault -- the
allocator may have unmapped a run, or mapped something new at its address -- so a failing range is
counted and skipped, and a replaced one costs at most the size of the run that used to be there.
There is deliberately no fallback that reads the pages from Python: on an unmapped run that is a
SIGSEGV in the server, where madvise returns ENOMEM.
"""

from __future__ import annotations

import ctypes
import errno
import mmap
import os
import struct
import threading
import time
from collections.abc import Iterator

MADV_POPULATE_READ = 22
_PAGE = mmap.PAGESIZE
_PM_SWAPPED = 1 << 62
_PM_ENTRIES_PER_READ = 1 << 16  # 512 KiB of pagemap per read, 256 MiB of address space


def swapped_bytes(status_path: str = "/proc/self/status") -> int:
    """``VmSwap`` of a process, in bytes (0 when the kernel does not report it)."""
    try:
        with open(status_path) as f:
            for line in f:
                if line.startswith("VmSwap:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def swapped_mappings(smaps_path: str = "/proc/self/smaps") -> list[tuple[int, int]]:
    """``(start, end)`` of every readable mapping whose ``Swap:`` is non-zero."""
    out = []
    start = end = 0
    readable = False
    with open(smaps_path) as f:
        for line in f:
            head = line.split(None, 1)[0]
            if "-" in head and ":" not in head:
                lo, hi = head.split("-")
                start, end = int(lo, 16), int(hi, 16)
                perms = line.split()[1] if len(line.split()) > 1 else ""
                readable = perms.startswith("r")
            elif head == "Swap:" and readable and int(line.split()[1]) > 0:
                out.append((start, end))
    return out


def swapped_runs(mappings, pagemap_path: str = "/proc/self/pagemap") -> Iterator[tuple[int, int]]:
    """Coalesced ``(address, bytes)`` runs of swapped pages inside ``mappings``."""
    entry = struct.Struct("<Q")
    with open(pagemap_path, "rb", buffering=0) as pm:
        for start, end in mappings:
            run_start = None
            addr = start
            while addr < end:
                count = min(_PM_ENTRIES_PER_READ, (end - addr) // _PAGE)
                pm.seek((addr // _PAGE) * entry.size)
                raw = pm.read(count * entry.size)
                if not raw:
                    break
                for i, (value,) in enumerate(entry.iter_unpack(raw[: len(raw) // 8 * 8])):
                    page = addr + i * _PAGE
                    if value & _PM_SWAPPED:
                        if run_start is None:
                            run_start = page
                    elif run_start is not None:
                        yield run_start, page - run_start
                        run_start = None
                addr += (len(raw) // entry.size) * _PAGE
            if run_start is not None:
                yield run_start, addr - run_start


class SelfSwap:
    """This process's swap, as ``BankRewarm`` sees it: how much, and bringing it back."""

    def __init__(self, step_bytes: int = 64 << 20):
        self.step_bytes = int(step_bytes)
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self._libc.madvise.restype = ctypes.c_int
        self.unsupported = False  # set when MADV_POPULATE_READ is refused before anything worked
        self.done_bytes = self.failed_bytes = 0
        self.last_errno = 0

    def swapped_bytes(self) -> int:
        return swapped_bytes()

    def swap_back(self, cancel: threading.Event) -> tuple[int, float, bool]:
        """Fault the swapped pages back in, a step at a time, stopping between steps when
        ``cancel`` is set. Returns (bytes requested, seconds, finished); ``failed_bytes`` says
        how much of it the kernel refused (unmapped since the scan, or no POPULATE_READ)."""
        started = time.perf_counter()
        requested = 0
        self.done_bytes = self.failed_bytes = 0
        for addr, nbytes in swapped_runs(swapped_mappings()):
            pos, end = addr, addr + nbytes
            while pos < end:
                if cancel.is_set() or self.unsupported:
                    return requested, time.perf_counter() - started, False
                step = min(self.step_bytes, end - pos)
                self._bring_in(pos, step)
                requested += step
                pos += step
        return requested, time.perf_counter() - started, True

    def _bring_in(self, addr: int, nbytes: int) -> None:
        # No fallback that reads the memory from here: a run the allocator unmapped after the
        # scan would be a SIGSEGV in the server, where madvise just returns ENOMEM.
        if self._libc.madvise(ctypes.c_void_p(addr), nbytes, MADV_POPULATE_READ) != 0:
            err = ctypes.get_errno()
            self.failed_bytes += nbytes
            self.last_errno = err
            if err == errno.EINVAL and self.done_bytes == 0:
                # EINVAL before anything succeeded: a kernel older than 5.14, most likely.
                self.unsupported = True
        else:
            self.done_bytes += nbytes


def enabled_from_env() -> bool:
    """On with ``--moe-bank-rewarm`` unless ``FREETOKEN_REWARM_SWAP=0``.

    Measured with the server's own anonymous memory pushed to swap and nothing else disturbed (the
    bank's page cache stayed at 100%): a short prompt after 90 s idle waited +15.9 s on an RTX 2060
    (Ornith, 2 GiB swapped) and +0.3 s on two RTX 3060s (Flash-Next, 4 GiB) beyond an undisturbed
    idle; paging it back in during the idle took both to the undisturbed figure. It only reads pages
    the kernel reports as swapped, so with nothing in swap it costs one read of /proc/self/status."""
    return os.environ.get("FREETOKEN_REWARM_SWAP", "1").strip() not in ("", "0")
