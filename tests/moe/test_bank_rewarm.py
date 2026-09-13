"""--moe-bank-rewarm: read a mapped bank's file-backed rows back in while the server is idle.

Measured on an RTX 2060: after the page cache lost the cold side, the first follow-up request
waited 31 s for its first token instead of 1 s. What is worth pinning here is the mechanism
that makes the fix safe to leave on: the cold spans are exactly the non-resident rows, the
walk stops as soon as a request arrives, and the idle thread never runs past ``busy()``.
Actual eviction is not reproducible in a unit test (the pages are mapped by the test itself),
so the end-to-end number lives in tools/cold_first_request.py.
"""

from __future__ import annotations

import threading
import time

import torch

from freetoken.moe.bank_file import BankFile, layout_from_sample
from freetoken.moe.bank_rewarm import BankRewarm
from freetoken.moe.mapped_bank import MappedBanks


def _built(tmp_path, num_layers=3, num_experts=64, hot=16, cols=131072):
    # Rows large enough that a block is well past _WHOLE_BLOCK_BYTES (4 MiB) and so split
    # into a resident prefix and a file-backed remainder.
    src = {"packed": [torch.full((num_experts, cols), layer + 1, dtype=torch.uint8)
                      for layer in range(num_layers)]}
    layers = list(range(num_layers))
    lay = layout_from_sample({"packed": src["packed"][0]}, layers, num_experts)
    path = str(tmp_path / "b.ftmb")
    with BankFile.create(path, lay) as w:
        for layer in layers:
            w.write_layer(layer, {"packed": src["packed"][layer]}, list(range(num_experts)))
    return lay, path


def test_cold_spans_are_the_rows_past_the_resident_prefix(tmp_path):
    lay, path = _built(tmp_path)
    banks = MappedBanks(path, register=False, hot_per_layer=16)
    try:
        row = 131072
        assert len(banks.cold_spans) == 3  # one per layer of the one bank
        for layer, (off, nbytes) in zip(lay.layers, banks.cold_spans):
            assert off == lay.offset_of("packed", layer) + 16 * row
            assert nbytes == (64 - 16) * row
    finally:
        banks.close()


def test_residency_is_a_share_and_a_full_walk_leaves_it_whole(tmp_path):
    _, path = _built(tmp_path)
    banks = MappedBanks(path, register=False, hot_per_layer=16)
    try:
        before = banks.cold_residency()
        assert 0.0 <= before <= 1.0
        walked, seconds, finished = banks.rewarm(threading.Event(), step_bytes=1 << 20)
        assert finished and seconds >= 0.0
        assert walked == sum(n for _, n in banks.cold_spans)
        assert banks.cold_residency() == 1.0  # just read, and nothing is pressing on it
    finally:
        banks.close()


def test_a_request_stops_the_walk_before_it_reads_anything_more(tmp_path):
    _, path = _built(tmp_path)
    banks = MappedBanks(path, register=False, hot_per_layer=16)
    try:
        cancel = threading.Event()
        cancel.set()
        walked, _, finished = banks.rewarm(cancel, step_bytes=1 << 20)
        assert walked == 0 and not finished
    finally:
        banks.close()


class _FakeBanks:
    """Residency below the threshold until a walk runs; the walk takes long enough to be
    interrupted, and records whether it was."""

    def __init__(self):
        self.residency = 0.3
        self.walks = 0
        self.stopped = False

    def cold_residency(self):
        return self.residency

    def rewarm(self, cancel, step_bytes=0):
        self.walks += 1
        for _ in range(200):
            if cancel.is_set():
                self.stopped = True
                return 0, 0.0, False
            time.sleep(0.01)
        self.residency = 1.0
        return 1 << 30, 2.0, True


def test_idle_reads_back_after_the_delay_and_logs_it():
    banks, lines = _FakeBanks(), []
    rw = BankRewarm(banks, delay_s=0.05, log=lines.append)
    rw.idle()
    deadline = time.time() + 5
    while rw.last is None and time.time() < deadline:
        time.sleep(0.02)
    rw.busy()
    rw.join(2)
    assert banks.walks == 1 and rw.last["finished"]
    assert any("walked 1.0 GiB" in line for line in lines)


def test_busy_interrupts_a_walk_in_progress():
    banks = _FakeBanks()
    rw = BankRewarm(banks, delay_s=0.01, log=lambda _: None)
    rw.idle()
    time.sleep(0.2)  # the walk has started and is sleeping in its loop
    rw.busy()
    rw.join(2)
    assert banks.stopped


def test_nothing_is_read_when_the_cache_still_holds_it():
    banks = _FakeBanks()
    banks.residency = 0.99
    rw = BankRewarm(banks, delay_s=0.02, log=lambda _: None)
    rw.idle()
    time.sleep(0.15)
    rw.busy()
    rw.join(2)
    assert banks.walks == 0


def test_a_request_before_the_delay_means_no_walk_at_all():
    banks = _FakeBanks()
    rw = BankRewarm(banks, delay_s=10.0, log=lambda _: None)
    rw.idle()
    rw.busy()
    rw.join(2)
    assert banks.walks == 0


class _PressedBanks(_FakeBanks):
    """Every walk finishes, and the cache holds less afterwards than before it."""

    def rewarm(self, cancel, step_bytes=0):
        self.walks += 1
        self.residency = max(0.0, self.residency - 0.1)
        return 1 << 30, 0.0, True


def test_a_walk_that_loses_ground_backs_off_instead_of_fighting():
    banks, lines = _PressedBanks(), []
    rw = BankRewarm(banks, delay_s=0.05, log=lines.append, max_backoff_s=0.8)
    rw.idle()
    time.sleep(0.6)
    rw.busy()
    rw.join(2)
    # at 0.05 s a check there would be ~10 walks in 0.6 s; backing off 0.2 -> 0.8 leaves 2-3
    assert 1 <= banks.walks <= 3
    assert any("still under pressure; next check in 0.2 s" in line for line in lines)
