"""A head of the prefill queue that keeps being refused is reported, with the reason the
admission check had (upstream #453: the engine sat at active=1, 0 tok/s and logged nothing).
CPU, real CacheManager/TableManager; no engine."""

from __future__ import annotations

import time

import torch

from freetoken.core import SamplingParams


class _Log:
    def __init__(self):
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def warning(self, msg, *a, **k):
        self.warnings.append(msg)

    def info(self, msg, *a, **k):
        self.infos.append(msg)


def _setup_context() -> None:
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))


def _managers(monkeypatch, *, num_pages=64, page_size=16, max_running=2, type="radix", pool=None):
    from freetoken.scheduler import prefill
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.table import TableManager

    _setup_context()
    log = _Log()
    monkeypatch.setattr(prefill, "logger", log)
    pt = torch.zeros(max_running + 1, 1024, dtype=torch.int32)
    cm = CacheManager(num_pages, page_size, pt, type, linear_state_pool=pool)
    tm = TableManager(max_running_reqs=max_running, page_table=pt)
    # a threshold of one millisecond; the repeat interval is still floored at 60 s
    pm = prefill.PrefillManager(cm, tm, DecodeManager(page_size), stall=prefill.AdmissionStall(0.001))
    return cm, tm, pm, log


def _pending(uid, prompt_len, max_tokens):
    from freetoken.scheduler.utils import PendingReq

    return PendingReq(uid, torch.arange(prompt_len, dtype=torch.int32), SamplingParams(max_tokens=max_tokens))


def test_stall_clock_warns_after_threshold_then_every_interval():
    from freetoken.scheduler.prefill import AdmissionStall

    s = AdmissionStall(30.0)
    assert not s.refused(1, 100.0)  # first refusal starts the clock
    assert not s.refused(1, 129.0)
    assert s.refused(1, 130.0)
    assert not s.refused(1, 131.0)
    assert not s.refused(1, 189.0)
    assert s.refused(1, 190.0)  # repeats every 60 s
    assert s.warnings == 2 and s.since == 100.0
    assert not s.refused(2, 191.0)  # a different head restarts the clock
    assert s.uid == 2 and s.warnings == 0 and not s.refused(2, 220.0) and s.refused(2, 221.0)


def test_stall_clock_waits_out_running_progress():
    from freetoken.scheduler.prefill import AdmissionStall

    s = AdmissionStall(30.0)
    assert not s.refused(1, 0.0, progress=100)
    # (a) ten minutes behind a request whose length keeps growing: FIFO, never a warning
    assert not any(s.refused(1, float(t), progress=100 + t) for t in range(1, 601))
    # (b) it stops advancing: warn warn_after later, then on the repeat interval
    assert not s.refused(1, 629.0, progress=700)
    assert s.refused(1, 630.0, progress=700)
    assert s.since == 0.0 and s.progress_at == 600.0  # total wait kept, stall measured apart
    assert not s.refused(1, 689.0, progress=700) and s.refused(1, 690.0, progress=700)
    # it moves again: quiet until it has stood still for a full warn_after
    assert not s.refused(1, 700.0, progress=701) and not s.refused(1, 729.0, progress=701)
    assert s.refused(1, 730.0, progress=701)
    # (c) nothing running (None): the plain clock, repeating from the last warning
    assert not s.refused(1, 789.0) and s.refused(1, 790.0)
    s = AdmissionStall(30.0)
    assert not s.refused(2, 0.0) and not s.refused(2, 29.0) and s.refused(2, 30.0)


def test_stall_clock_off_at_zero():
    from freetoken.scheduler.prefill import AdmissionStall

    s = AdmissionStall(0.0)
    assert not any(s.refused(1, t) for t in (0.0, 1e3, 1e6))


def test_request_larger_than_the_pool_is_reported_with_the_kv_numbers(monkeypatch):
    cm, tm, pm, log = _managers(monkeypatch)  # 64 pages x 16 = 1024 KV tokens
    pm.pending_list = [_pending(7, 100, 2000), _pending(8, 10, 4)]
    assert pm.schedule_next_batch(512) is None
    assert not log.warnings  # the first refusal only starts the clock
    time.sleep(0.01)
    assert pm.schedule_next_batch(512) is None
    assert len(log.warnings) == 1
    msg = log.warnings[0]
    assert "request 7 (prompt 100 tokens, max_tokens 2000)" in msg
    assert "holding 1 queued request(s)" in msg
    assert "needs 2112 KV tokens" in msg and "1024 were available" in msg
    assert "1024 free + 0 evictable" in msg and "of 1024 in the pool" in msg
    assert "more than the whole pool" in msg and "It can never be admitted" in msg
    assert "Nothing is running" not in msg  # not something a finishing request would fix
    assert pm.schedule_next_batch(512) is None
    assert len(log.warnings) == 1  # rate-limited

    assert pm.abort_req(7) is None
    assert log.infos and "request 7 aborted after" in log.infos[0]
    assert pm.stall.uid is None
    assert pm.schedule_next_batch(512) is not None  # request 8 goes


def test_no_request_slot_then_admitted_logs_the_wait(monkeypatch):
    cm, tm, pm, log = _managers(monkeypatch, max_running=1)
    held = tm.allocate()
    pm.pending_list = [_pending(3, 20, 4)]
    assert pm.schedule_next_batch(512) is None
    time.sleep(0.01)
    assert pm.schedule_next_batch(512) is None
    assert len(log.warnings) == 1
    assert "no free request slot (all 1 of --max-running-requests are taken)" in log.warnings[0]
    assert "Nothing is running that could free it" in log.warnings[0]

    tm.free(held)
    batch = pm.schedule_next_batch(512)
    assert batch is not None and batch.reqs[0].uid == 3
    assert len(log.infos) == 1 and "request 3 admitted after" in log.infos[0]
    assert pm.stall.uid is None


def test_brief_refusal_says_nothing(monkeypatch):
    cm, tm, pm, log = _managers(monkeypatch, max_running=1)
    pm.stall = type(pm.stall)(30.0)
    held = tm.allocate()
    pm.pending_list = [_pending(3, 20, 4)]
    assert pm.schedule_next_batch(512) is None
    tm.free(held)
    assert pm.schedule_next_batch(512) is not None
    assert not log.warnings and not log.infos and pm.stall.uid is None


def test_gdn_slot_refusal_names_the_slot_count(monkeypatch):
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(group=g, num_slots=5, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)
    cm, tm, pm, log = _managers(monkeypatch, type="hybrid_radix", pool=pool)
    pool.alloc(3)  # 4 usable slots, 1 left; nothing in the tree to evict
    pm.pending_list = [_pending(5, 40, 4)]
    assert pm.schedule_next_batch(512) is None
    time.sleep(0.01)
    assert pm.schedule_next_batch(512) is None
    assert len(log.warnings) == 1
    assert "needs 3 GDN state slots and 1 were free" in log.warnings[0]
    assert "pool of 4" in log.warnings[0]


def test_running_requests_change_the_outlook(monkeypatch):
    from freetoken.scheduler.prefill import describe_refusal

    cm, tm, pm, _ = _managers(monkeypatch)
    pm.decode_manager.running_reqs = {object(), object()}
    msg = describe_refusal(("kv", 64, 32, 16), _pending(1, 10, 4), 450.0, 0, cm, tm, pm.decode_manager,
                           no_progress_for=45.0)
    assert "refused admission for 450s" in msg
    assert "queued behind 2 running request(s) that have made no progress for 45s" in msg
    assert "can never" not in msg
    assert "no reason was recorded" in describe_refusal(
        None, _pending(1, 10, 4), 45.0, 0, cm, tm, pm.decode_manager
    )


class _Running:
    """A decoding request as the admission path sees it: a length and an output budget."""

    def __init__(self, device_len):
        self.device_len = device_len
        self.remain_len = 100


class _FakeTime:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


def test_side_request_behind_a_generation_warns_only_when_it_stops(monkeypatch):
    """--max-running-requests 1 with Open WebUI: the title request waits out a long generation
    (no warning), and is reported only once that generation stops being stepped."""
    from freetoken.scheduler import prefill

    cm, tm, pm, log = _managers(monkeypatch, max_running=1)
    clock = _FakeTime()
    monkeypatch.setattr(prefill, "time", clock)
    pm.stall = prefill.AdmissionStall(30.0)
    tm.allocate()  # the generation's slot
    gen = _Running(500)
    pm.decode_manager.running_reqs = {gen}
    pm.pending_list = [_pending(9, 20, 4)]
    for step in range(600 * 20):  # 10 minutes of 50 ms decode steps
        clock.t = step * 0.05
        gen.device_len += 1
        assert pm.schedule_next_batch(512) is None
    assert not log.warnings
    stopped = clock.t
    clock.t = stopped + 29.0
    assert pm.schedule_next_batch(512) is None
    assert not log.warnings
    clock.t = stopped + 30.0
    assert pm.schedule_next_batch(512) is None
    assert len(log.warnings) == 1
    msg = log.warnings[0]
    assert "no free request slot" in msg
    assert "refused admission for 630s" in msg
    assert "queued behind 1 running request(s) that have made no progress for 30s" in msg
