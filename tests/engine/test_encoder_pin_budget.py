"""The vision tower's streamed block bank is pinned host memory: it is charged to the same pin
quota the expert bank residency planner budgets (engine._host_tables_bytes)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_a_streamed_tower_reports_its_pinned_bank():
    engine = pytest.importorskip("freetoken.engine.engine")
    bank = torch.empty((27, 1000), dtype=torch.uint8)
    model = SimpleNamespace(visual=SimpleNamespace(_streamer=SimpleNamespace(bank=bank)))
    assert engine._encoder_bank_bytes(model) == 27_000


def test_a_resident_tower_or_no_tower_reports_nothing():
    engine = pytest.importorskip("freetoken.engine.engine")
    assert engine._encoder_bank_bytes(SimpleNamespace(visual=SimpleNamespace(_streamer=None))) == 0
    assert engine._encoder_bank_bytes(SimpleNamespace()) == 0
