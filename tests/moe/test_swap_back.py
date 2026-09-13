"""--moe-bank-rewarm with FREETOKEN_REWARM_SWAP=1: page this process's swap back in while idle.

What is pinned here: only pages the kernel reports as swapped are asked for (a walk over every
readable mapping would populate the address space CUDA reserves and never uses), a request stops
the pass, a pass that loses ground backs off like the bank walk does, and -- on a kernel and a
host with swap -- memory pushed out with MADV_PAGEOUT really comes back.
"""

from __future__ import annotations

import ctypes
import mmap
import struct
import threading
import time

import pytest

from freetoken.moe import swap_back
from freetoken.moe.bank_rewarm import BankRewarm

PAGE = mmap.PAGESIZE
MADV_PAGEOUT = 21


def test_only_readable_mappings_with_swap_are_scanned(tmp_path):
    smaps = tmp_path / "smaps"
    smaps.write_text(
        "7f0000000000-7f0000010000 rw-p 00000000 00:00 0 \n"
        "Size:                 64 kB\n"
        "Swap:                 12 kB\n"
        "VmFlags: rd wr mr mw me ac \n"
        "7f0000010000-7f0000020000 rw-p 00000000 00:00 0 \n"
        "Swap:                  0 kB\n"
        "7f0000020000-7f0000030000 ---p 00000000 00:00 0 \n"
        "Swap:                  8 kB\n"
        "7f0000030000-7f0000040000 r--p 00000000 00:00 0                          [heap]\n"
        "Swap:                  4 kB\n"
    )
    assert swap_back.swapped_mappings(str(smaps)) == [
        (0x7F0000000000, 0x7F0000010000),
        (0x7F0000030000, 0x7F0000040000),
    ]


def test_swapped_pages_coalesce_into_runs(tmp_path):
    base = 0x100000 * PAGE
    flags = [0, 1, 1, 0, 1, 1, 1, 0]  # page i swapped?
    pagemap = tmp_path / "pagemap"
    with open(pagemap, "wb") as f:
        f.seek((base // PAGE) * 8)
        for swapped in flags:
            f.write(struct.pack("<Q", (1 << 62) if swapped else (1 << 63)))
    runs = list(swap_back.swapped_runs([(base, base + len(flags) * PAGE)], str(pagemap)))
    assert runs == [(base + PAGE, 2 * PAGE), (base + 4 * PAGE, 3 * PAGE)]


def test_a_run_reaching_the_end_of_a_mapping_is_not_lost(tmp_path):
    base = 0x200000 * PAGE
    pagemap = tmp_path / "pagemap"
    with open(pagemap, "wb") as f:
        f.seek((base // PAGE) * 8)
        f.write(struct.pack("<QQ", 0, 1 << 62))
    assert list(swap_back.swapped_runs([(base, base + 2 * PAGE)], str(pagemap))) == [(base + PAGE, PAGE)]


def test_vmswap_is_read_from_status(tmp_path):
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmRSS:\t  100 kB\nVmSwap:\t  2048 kB\n")
    assert swap_back.swapped_bytes(str(status)) == 2048 * 1024
    assert swap_back.swapped_bytes(str(tmp_path / "missing")) == 0


def _swap_total() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("SwapTotal:"):
                return int(line.split()[1]) * 1024
    return 0


def test_memory_pushed_to_swap_comes_back():
    if _swap_total() < (256 << 20):
        pytest.skip("no swap on this host")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    n = 32 << 20
    # private, like the server's heap: Python's default mmap(-1) is shared (shmem), whose swapped
    # pages are not counted in VmSwap and leave no swap entry in the page table
    buf = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    head = ctypes.c_char.from_buffer(buf)  # an export of buf: dropped before buf.close()
    try:
        for off in range(0, n, PAGE):
            buf[off] = 0x5A  # distinct from the zero page, so it has to be swapped to be reclaimed
        addr = ctypes.addressof(head)
        before = swap_back.swapped_bytes()
        if libc.madvise(ctypes.c_void_p(addr), n, MADV_PAGEOUT) != 0:
            pytest.skip(f"MADV_PAGEOUT refused (errno {ctypes.get_errno()})")
        pushed = swap_back.swapped_bytes() - before
        if pushed < n // 2:
            pytest.skip(f"the kernel paged out only {pushed >> 20} MiB of {n >> 20}")
        ours = [(a, b) for a, b in swap_back.swapped_mappings() if a <= addr < b]
        assert ours, "the paged-out mapping should report Swap: > 0"
        swapped = sum(nb for _, nb in swap_back.swapped_runs(ours))
        assert swapped >= pushed * 0.9

        s = swap_back.SelfSwap(step_bytes=4 << 20)
        requested, _, finished = s.swap_back(threading.Event())
        if s.unsupported:
            pytest.skip("MADV_POPULATE_READ needs Linux 5.14")
        assert finished and requested >= swapped
        assert swap_back.swapped_bytes() - before < n // 8
        assert all(buf[off] == 0x5A for off in range(0, n, 97 * PAGE))
    finally:
        del head
        buf.close()


def test_a_request_stops_the_swap_pass():
    s = swap_back.SelfSwap()
    cancel = threading.Event()
    cancel.set()
    requested, _, finished = s.swap_back(cancel)
    assert requested == 0 or not finished


class _WarmBanks:
    """The bank is fine; only the process swap is interesting."""

    def cold_residency(self):
        return 1.0

    def rewarm(self, cancel, step_bytes=0):
        raise AssertionError("the bank is resident; nothing should walk it")


class _FakeSwap:
    def __init__(self, swapped, keeps=False):
        self.swapped = swapped
        self.keeps = keeps
        self.passes = 0
        self.stopped = False
        self.failed_bytes = 0
        self.unsupported = False

    def swapped_bytes(self):
        return self.swapped

    def swap_back(self, cancel):
        self.passes += 1
        for _ in range(20):
            if cancel.is_set():
                self.stopped = True
                return 0, 0.0, False
            time.sleep(0.01)
        got = self.swapped
        if not self.keeps:
            self.swapped = 0
        return got, 0.2, True


def _run(rw, seconds):
    rw.idle()
    time.sleep(seconds)
    rw.busy()
    rw.join(2)


def test_swap_is_paged_back_even_when_the_bank_is_warm():
    swap, lines = _FakeSwap(2 << 30), []
    rw = BankRewarm(_WarmBanks(), delay_s=0.02, log=lines.append, swap=swap)
    _run(rw, 0.5)
    assert swap.passes == 1 and swap.swapped == 0
    assert any("had 2.00 GiB in swap; paged 2.00 GiB back" in line for line in lines)


def test_a_little_swap_is_left_alone():
    swap = _FakeSwap(64 << 20)
    _run(BankRewarm(_WarmBanks(), delay_s=0.02, log=lambda _: None, swap=swap), 0.3)
    assert swap.passes == 0


def test_swap_that_does_not_go_down_backs_off():
    swap, lines = _FakeSwap(1 << 30, keeps=True), []
    rw = BankRewarm(_WarmBanks(), delay_s=0.02, log=lines.append, swap=swap, max_backoff_s=10)
    _run(rw, 0.7)
    # 0.02 s checks with 0.2 s passes would be ~3 passes in 0.7 s; backing off 0.08 -> 0.32 -> ...
    assert 1 <= swap.passes <= 3
    assert any("still under pressure" in line for line in lines)


def test_busy_interrupts_a_swap_pass():
    swap = _FakeSwap(1 << 30)
    rw = BankRewarm(_WarmBanks(), delay_s=0.01, log=lambda _: None, swap=swap)
    rw.idle()
    time.sleep(0.08)
    rw.busy()
    rw.join(2)
    assert swap.stopped


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("FREETOKEN_REWARM_SWAP", raising=False)
    assert BankRewarm(_WarmBanks(), delay_s=1.0).swap is None
    monkeypatch.setenv("FREETOKEN_REWARM_SWAP", "1")
    assert isinstance(BankRewarm(_WarmBanks(), delay_s=1.0).swap, swap_back.SelfSwap)
