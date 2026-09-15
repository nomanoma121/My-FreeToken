"""Under --pp-size the KV pool is one number for the whole pipeline.

Every rank runs its own scheduler over the same request stream, so the pool size its admission,
prefix matching and eviction read cannot be a per-rank answer.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.engine import engine as engine_mod
from freetoken.engine.engine import Engine


def _rank(monkeypatch, other_rank_pages: int, seen: dict) -> Engine:
    def _all_reduce(tensor, op=None, group=None):
        seen["op"] = op
        torch.minimum(tensor, torch.tensor([other_rank_pages], dtype=torch.int64), out=tensor)

    monkeypatch.setattr(engine_mod.torch.distributed, "all_reduce", _all_reduce)
    engine = Engine.__new__(Engine)  # bypass __init__/GPU
    engine.tp_cpu_group = None
    return engine


def test_every_pipeline_rank_takes_the_smallest_pool(monkeypatch):
    """--moe-cache-auto gives each rank its reserve plus what its expert fill left over, so the
    ranks' pools differ by up to an expert slot's bytes -- enough for one rank to evict a
    prefix the other still matches once the cache is full."""
    seen = {}
    engine = _rank(monkeypatch, other_rank_pages=131_072, seen=seen)
    config = SimpleNamespace(num_page_override=131_391)

    assert engine._agree_pipeline_num_pages(config, 131_391) == 131_072
    assert seen["op"] is torch.distributed.ReduceOp.MIN
    # written back, so anything that re-reads the override later (a rebuild's fit check) agrees
    assert config.num_page_override == 131_072


def test_the_smallest_rank_keeps_its_pool_and_its_config(monkeypatch):
    engine = _rank(monkeypatch, other_rank_pages=131_391, seen={})
    config = SimpleNamespace(num_page_override=None)

    assert engine._agree_pipeline_num_pages(config, 131_072) == 131_072
    assert config.num_page_override is None
