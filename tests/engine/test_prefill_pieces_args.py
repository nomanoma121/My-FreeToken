"""--prefill-mixer-pieces reaches the config, off (1) by default."""

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
    assert _parse([]).prefill_mixer_pieces == 1


def test_takes_a_count():
    assert _parse(["--prefill-mixer-pieces", "4"]).prefill_mixer_pieces == 4
