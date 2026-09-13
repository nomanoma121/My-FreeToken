"""--moe-bank-prefetch reaches the config, off by default (experimental)."""

from __future__ import annotations

import json
import os
import tempfile

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


def test_off_by_default():
    assert _parse([]).moe_bank_prefetch is False


def test_flag_turns_it_on():
    assert _parse(["--moe-bank-prefetch"]).moe_bank_prefetch is True
