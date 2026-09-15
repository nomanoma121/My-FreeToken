"""Startup collectives wait for a late rank instead of dying on the serving timeout.

Two RTX 3060s, --pp-size 2, a cold page cache: rank 1 reached the KV page count all_reduce more than
60 s after rank 0, and rank 0's gloo recv gave up (four starts out of four). Reproduced here with
two CPU processes, a 2 s group timeout and a rank that arrives 5 s late: without the barrier the
early rank's all_reduce times out, with it both ranks agree.
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

LATE_S = 5.0
GROUP_TIMEOUT_S = 2.0


def _worker(rank: int, init_file: str, use_barrier: bool, result_dir: str) -> None:
    os.environ["FREETOKEN_RANK_JOIN_TIMEOUT_SECONDS"] = "30"
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
        timeout=timedelta(seconds=GROUP_TIMEOUT_S),
    )
    try:
        if rank == 1:
            time.sleep(LATE_S)  # still loading its half of the model
        waited = 0.0
        if use_barrier:
            from freetoken.distributed.rendezvous import wait_for_ranks

            waited = wait_for_ranks(dist.group.WORLD, "agreeing on the KV page count")
        x = torch.tensor([100 + rank], dtype=torch.int64)
        outcome = "ok"
        try:
            dist.all_reduce(x, op=dist.ReduceOp.MIN)
        except RuntimeError as exc:
            outcome = "timeout" if "imed out" in str(exc) else f"error: {exc}"
        with open(os.path.join(result_dir, f"rank{rank}"), "w") as f:
            f.write(f"{outcome} {int(x.item())} {waited:.1f}")
    finally:
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001 -- a timed-out group may not tear down cleanly
            pass


def _run(use_barrier: bool) -> dict[int, list[str]]:
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "init")
        mp.spawn(_worker, args=(init_file, use_barrier, d), nprocs=2, join=True)
        out = {}
        for rank in (0, 1):
            path = os.path.join(d, f"rank{rank}")
            out[rank] = open(path).read().split() if os.path.exists(path) else ["missing"]
        return out


def test_a_late_rank_is_waited_for():
    out = _run(use_barrier=True)
    assert out[0][:2] == ["ok", "100"] and out[1][:2] == ["ok", "100"], out
    assert float(out[0][2]) >= LATE_S - 1.0  # the early rank did the waiting, in the barrier


def test_without_it_the_early_rank_times_out():
    # the failure the barrier is for, so the test above cannot pass for the wrong reason
    out = _run(use_barrier=False)
    assert out[0][0] == "timeout", out


def test_one_rank_does_not_wait():
    from freetoken.distributed.rendezvous import wait_for_ranks

    assert wait_for_ranks(None, "nothing") == 0.0
