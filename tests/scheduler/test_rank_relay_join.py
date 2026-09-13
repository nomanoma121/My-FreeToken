"""The request relay between ranks must not drop the first request (upstream #364).

Rank 0 relays every request to the other ranks over ZeroMQ PUB/SUB, and a PUB drops what no
registered subscription matches. These tests run the real SchedulerIOMixin on two processes with a
real gloo group and real ipc sockets -- the scheduler minus the engine -- and publish the first
request the moment __init__ returns, which is what a client firing at "ready" does.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import tempfile
import time
from types import SimpleNamespace

import pytest

PAYLOAD = b"the first request"


def _config(tmp: str, rank: int, broadcast: str) -> SimpleNamespace:
    from freetoken.distributed.info import DistributedInfo

    return SimpleNamespace(
        tp_info=DistributedInfo(rank=rank, size=2),
        offline_mode=False,
        zmq_backend_addr=f"ipc://{tmp}/backend",
        zmq_detokenizer_addr=f"ipc://{tmp}/detok",
        backend_create_detokenizer_link=True,
        zmq_scheduler_broadcast_addr=f"ipc://{tmp}/{broadcast}",
    )


def _worker(rank: int, tmp: str, scenario: str) -> None:
    import torch.distributed as dist

    import freetoken.scheduler.io as io_mod
    from freetoken.scheduler.io import SchedulerIOMixin

    # a wedge must fail the test, not hang it
    faulthandler.dump_traceback_later(45, exit=True, file=sys.stderr)
    dist.init_process_group("gloo", init_method=f"file://{tmp}/init", rank=rank, world_size=2)
    out = open(f"{tmp}/rank{rank}.txt", "w")

    broadcast = "broadcast"
    if scenario == "no-join":
        SchedulerIOMixin._join_rank_relay = lambda self, tp_info: None  # the relay before the fix
    if scenario == "never-subscribes":
        io_mod._RELAY_JOIN_TIMEOUT_S = 0.5
        if rank == 1:
            broadcast = "elsewhere"  # rank 1's SUB can never hear rank 0's PUB
    if rank == 1 and scenario in ("rank1-late", "no-join"):
        time.sleep(1.0)  # rank 1 builds its SUB a second after rank 0 is ready

    io = SchedulerIOMixin.__new__(SchedulerIOMixin)
    try:
        SchedulerIOMixin.__init__(io, _config(tmp, rank, broadcast), dist.group.WORLD)
    except RuntimeError as exc:
        out.write(f"raised: {exc}")
        out.close()
        return  # no collective after this: the other rank raised in the same round
    if rank == 0:
        io._send_into_ranks.put_raw(PAYLOAD)  # the first request, at ready + 0 s
        io.sync_all_ranks()
        out.write("sent")
    else:
        io.sync_all_ranks()
        sub = io._recv_from_rank0.socket
        got = sub.recv() if sub.poll(timeout=2000) else None
        out.write("got: " + (got.decode() if got is not None else "nothing"))
    out.close()
    dist.destroy_process_group()


def _run(scenario: str) -> dict[int, str]:
    import torch.multiprocessing as mp

    # ipc paths must stay short (AF_UNIX); a pytest tmp_path can exceed the limit
    with tempfile.TemporaryDirectory(prefix="relay", dir="/tmp") as tmp:
        mp.spawn(_worker, args=(tmp, scenario), nprocs=2, join=True)
        return {r: open(os.path.join(tmp, f"rank{r}.txt")).read() for r in (0, 1)}


pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="ipc:// sockets")


@pytest.mark.parametrize("scenario", ["together", "rank1-late"])
def test_the_first_request_reaches_the_other_rank(scenario):
    out = _run(scenario)
    assert out[0] == "sent"
    assert out[1] == "got: " + PAYLOAD.decode()


def test_without_the_join_a_late_subscriber_loses_the_first_request():
    """The mechanism the join exists for, made deterministic: rank 0 publishes before rank 1's SUB
    exists, and the PUB drops the frame. In a served engine the window is the 100-200 ms between
    the SUB existing and its subscription registering, which is why it was intermittent."""
    out = _run("no-join")
    assert out[0] == "sent"
    assert out[1] == "got: nothing", "the relay delivered without the join; the race did not reproduce"


def test_a_subscription_that_never_lands_fails_every_rank_together():
    """If rank 0 gave up alone, rank 1 would wait in the collective for a round that never comes --
    the hang the join is there to remove. Both ranks must raise."""
    out = _run("never-subscribes")
    assert out[0].startswith("raised: rank relay"), out[0]
    assert out[1].startswith("raised: rank relay"), out[1]
    assert "heard none" in out[1]
