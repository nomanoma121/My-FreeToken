"""--moe-bank-ram auto becomes a size in the launcher; --moe-bank-readahead is off by default."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

GiB = 2**30


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


@pytest.fixture
def host(monkeypatch):
    from freetoken.moe import disk_probe

    mem = {"MemTotal": 62 * GiB, "MemAvailable": 60 * GiB}
    monkeypatch.setattr(disk_probe, "meminfo", lambda proc="/proc": dict(mem))
    return mem


def test_auto_is_resolved_to_a_size_once(host):
    from freetoken.moe.bank_disk import parse_size

    got = parse_size(_parse(["--moe-bank-ram", "auto"]).moe_bank_ram)
    # 60 - 4.5 (one rank) - 3.1 (5% of 62) GiB
    assert got == pytest.approx(60 * GiB - 4.5 * GiB - 0.05 * 62 * GiB, abs=0.01 * GiB)


def test_auto_is_split_the_same_way_as_a_size_across_ranks(host):
    """The pipeline split subtracts the rest of the server once per rank; the engine then halves."""
    from freetoken.moe import disk_probe

    one = disk_probe.auto_bank_ram(host, 1).total_bytes
    two = disk_probe.auto_bank_ram(host, 2).total_bytes
    assert one - two == disk_probe.NONBANK_PER_RANK_BYTES


def test_auto_with_too_little_memory_stops_at_parse_time(host):
    host["MemAvailable"] = 6 * GiB
    with pytest.raises(SystemExit):
        _parse(["--moe-bank-ram", "auto"])


def test_an_explicit_size_is_left_alone(host):
    assert _parse(["--moe-bank-ram", "48G"]).moe_bank_ram == "48G"


def test_readahead_is_off_by_default():
    assert _parse([]).moe_bank_readahead == "off"


@pytest.mark.parametrize("value,expected", [("auto", "auto"), ("OFF", "off"), ("256", "256")])
def test_readahead_values(value, expected):
    assert _parse(["--moe-bank-readahead", value]).moe_bank_readahead == expected


@pytest.mark.parametrize("value", ["0", "-5", "fast"])
def test_readahead_rejects_nonsense(value):
    with pytest.raises(SystemExit):
        _parse(["--moe-bank-readahead", value])
