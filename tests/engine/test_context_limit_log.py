"""The boot log says how long a request can be, and what caps it.

/v1/models advertises the model's own limit; the KV pool can hold less, and before this the
only place that number appeared was the "prompt is too long" refusal of the first long chat.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.engine import engine as engine_mod
from freetoken.engine.engine import _log_context_limit


class _Log:
    def __init__(self):
        self.info, self.warning = [], []

    def info_rank0(self, msg):
        self.info.append(msg)

    def warning_rank0(self, msg):
        self.warning.append(msg)


@pytest.fixture
def log(monkeypatch):
    rec = _Log()
    monkeypatch.setattr(engine_mod, "logger", rec)
    return rec


def test_a_pool_below_the_model_limit_is_a_warning_that_names_the_flag(log):
    """gpt-oss-20b started with no context flags: --moe-cache-auto keeps 8192 tokens of KV,
    /v1/models says 131072, and the first long chat was refused with "prompt is too long"."""
    config = SimpleNamespace(max_seq_len=131_072, moe_cache_auto=True)
    _log_context_limit(config, 8192)

    assert not log.info
    (msg,) = log.warning
    assert "context limit 8192 tokens" in msg
    assert "--kv-reserve-tokens, up to 131072" in msg  # not the model limit as a value to pass
    assert "--max-seq-len-override 8192" in msg


def test_without_auto_sizing_the_warning_points_at_the_pool_flags(log):
    config = SimpleNamespace(max_seq_len=65_536, moe_cache_auto=False)
    _log_context_limit(config, 40_000)
    (msg,) = log.warning
    assert "--kv-reserve-tokens" not in msg
    assert "--num-tokens" in msg


def test_a_pool_that_covers_the_model_limit_is_one_info_line(log):
    config = SimpleNamespace(max_seq_len=65_536, moe_cache_auto=True)
    _log_context_limit(config, 65_537)
    assert log.info == ["context limit 65536 tokens (KV pool 65537 tokens)"]
    assert not log.warning
