"""The rank-wait watchdog: a blocking send/recv between ranks that outlasts the threshold is
logged (and nothing else happens to it). Fake clock for the bookkeeping; no process group."""

from __future__ import annotations

import time

import torch


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _watchdog(warn_after=60.0):
    from freetoken.distributed.watchdog import RankWaitWatchdog

    clock, warns, infos = _Clock(), [], []
    wd = RankWaitWatchdog(warn_after, clock=clock, warn=warns.append, info=infos.append,
                          start_thread=False)
    return wd, clock, warns, infos


def test_warns_after_threshold_repeats_and_reports_the_end():
    wd, clock, warns, infos = _watchdog()
    clock.t = 10.0
    wd.begin("the hidden states ({detail} rows) from rank {peer}", 0, 3)
    clock.t = 69.0
    wd.tick()
    assert not warns
    clock.t = 70.0
    wd.tick()
    assert len(warns) == 1
    assert "blocked for 60s waiting for the hidden states (3 rows) from rank 0" in warns[0]
    assert "normal while the other rank runs a long prefill chunk" in warns[0]
    clock.t = 129.0
    wd.tick()
    assert len(warns) == 1
    clock.t = 130.0
    wd.tick()
    assert len(warns) == 2 and "blocked for 120s" in warns[1]
    clock.t = 135.0
    wd.end()
    # (other tests in the session may have set a rank, which prefixes "rank N: ")
    assert len(infos) == 1
    assert infos[0].endswith("the wait for the hidden states (3 rows) from rank 0 ended after 125s")
    clock.t = 500.0
    wd.tick()
    assert len(warns) == 2


def test_short_waits_say_nothing():
    wd, clock, warns, infos = _watchdog()
    for i in range(100):
        wd.begin("this step's message count from rank {peer}", 0)
        clock.t += 1.0
        wd.tick()
        wd.end()
    assert not warns and not infos


def test_a_new_wait_restarts_the_clock_and_peer_may_be_unknown():
    wd, clock, warns, _ = _watchdog()
    wd.begin("rank {peer} to take the sampled tokens ({detail})", 1, 1)
    clock.t = 59.0
    wd.end()
    wd.begin("another rank to take a message count sent {detail} sends ago", None, 64)
    clock.t = 118.0
    wd.tick()
    assert not warns
    clock.t = 119.0
    wd.tick()
    assert warns and "another rank to take a message count sent 64 sends ago" in warns[0]


def test_thread_reports_a_real_wait():
    from freetoken.distributed.watchdog import RankWaitWatchdog

    warns, infos = [], []
    wd = RankWaitWatchdog(0.02, warn=warns.append, info=infos.append)
    wd.begin("the hidden states ({detail} rows) from rank {peer}", 0, 1)
    deadline = time.monotonic() + 5.0
    while not warns and time.monotonic() < deadline:
        time.sleep(0.01)
    wd.end()
    assert len(warns) == 1 and len(infos) == 1


class _Recorder:
    def __init__(self):
        self.events = []

    def begin(self, what, peer=None, detail=None):
        self.events.append(("begin", what.format(peer=peer, detail=detail)))

    def end(self):
        self.events.append(("end",))


def test_msg_count_recv_and_relayed_gets_are_bracketed(monkeypatch):
    from freetoken.scheduler import io

    rec = _Recorder()
    monkeypatch.setattr(io, "rank_wait_watchdog", lambda: rec)

    class _Work:
        def wait(self):
            rec.events.append(("wait",))

    class _Group:
        def recv(self, bufs, src, tag):
            bufs[0].fill_(2)
            return _Work()

    class _Sub:
        def get(self):
            rec.events.append(("get",))
            return "msg"

    mixin = io.SchedulerIOMixin.__new__(io.SchedulerIOMixin)
    mixin.tp_cpu_group = _Group()
    mixin._recv_from_rank0 = _Sub()
    assert mixin._recv_msg_multi_rank1(blocking=False) == ["msg", "msg"]
    assert rec.events == [
        ("begin", "this step's message count from rank 0"), ("wait",), ("end",),
        ("begin", "relayed request 1 of this step from rank 0"), ("get",), ("end",),
        ("begin", "relayed request 2 of this step from rank 0"), ("get",), ("end",),
    ]


def test_pipeline_recv_is_bracketed_even_when_it_raises(monkeypatch):
    from freetoken.distributed import pipeline
    from freetoken.distributed.info import PipelineInfo

    rec = _Recorder()
    monkeypatch.setattr(pipeline, "rank_wait_watchdog", lambda: rec)

    def _recv(buf, src, group, tag):
        raise RuntimeError("peer went away")

    monkeypatch.setattr(pipeline.dist, "recv", _recv)
    comm = pipeline.PipelineComm(PipelineInfo(1, 2, 5, 10, 10), None, torch.device("cpu"))
    comm.configure(hidden_width=4, hidden_dtype=torch.bfloat16)
    try:
        comm.recv_hidden(3)
    except RuntimeError:
        pass
    assert rec.events == [("begin", "the hidden states (3 rows) from rank 0"), ("end",)]
