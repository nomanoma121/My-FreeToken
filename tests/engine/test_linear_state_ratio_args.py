"""--linear-state-cache-ratio: the knob that bounds hybrid-GDN prefix reuse.

It existed as a config field with no flag, so the one lever that decides how many
conversations stay reusable could not be turned from the command line. What is worth pinning
is that the flag reaches the config, and the slot arithmetic the help text promises (cached
snapshots = max(4, ratio * max_running_req), on top of 4 per running request + 1 padding).
"""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

import pytest


def _parse(argv):
    pytest.importorskip("freetoken.server.args")
    from freetoken.server.args import parse_args

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"architectures": ["Qwen4ExpForConditionalGeneration"],
                       "model_type": "qwen4_exp", "torch_dtype": "bfloat16"}, f)
        args, _ = parse_args(
            ["--model", d, "--dtype", "bfloat16", "--tool-call-parser", "llama3",
             "--reasoning-parser", "off", *argv],
            False,
        )
    return args


def test_the_default_is_unchanged():
    assert _parse([]).linear_state_cache_ratio == 2.0


def test_the_flag_reaches_the_config():
    assert _parse(["--linear-state-cache-ratio", "8"]).linear_state_cache_ratio == 8.0


@pytest.mark.parametrize("ratio, slots", [(2.0, 9), (1.0, 9), (8.0, 13), (16.0, 21)])
def test_slot_arithmetic_matches_the_help_text(ratio, slots):
    from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots

    cfg = SimpleNamespace(max_running_req=1, cache_type="hybrid_radix",
                          linear_state_cache_ratio=ratio)
    # 4 to run one request + max(4, ratio) cached snapshots + 1 padding sink
    assert _linear_pool_num_slots(cfg) == slots
