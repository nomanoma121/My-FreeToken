"""--prefill-profile: the per-forward line splits wall time into parts that add up, and the
profile never outlives its forward (the movement code reports into whatever ``active()`` is)."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from freetoken.utils import prefill_profile as pp


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _profile(**kw):
    clock = _Clock()
    lines = []
    prof = pp.PrefillProfile(log=lines.append, clock=clock, **kw)
    return prof, clock, lines


def test_parts_add_up_to_the_forward(monkeypatch):
    monkeypatch.setattr(pp, "read_bytes", lambda path="/proc/self/io": 0)
    prof, clock, lines = _profile(rank=1, size=2, residency=lambda: 0.12)
    prof.begin(8192)
    assert pp.active() is prof
    with prof.peer_wait():
        clock.t += 2.0
    prof.staged_piece(0.5, 3.0, 1 << 30, 10)
    with prof.ple_fill():
        clock.t += 1.0
    clock.t += 7.0  # everything else, the bank read included (reported, not clocked here)
    prof.end()
    assert pp.active() is None
    (line,) = lines
    assert line.startswith("prefill profile [rank 1/2]: 8192 tokens in 10.00 s (819 tok/s)")
    assert "peers 2.00 s | bank read 3.00 s | PLE 1.00 s | GPU + rest 4.00 s" in line
    assert "unregistered rows 1.00 GiB copied at 0.33 GiB/s (10 major faults, 0.50 s more waiting" in line
    assert "page cache held 12% of the file-backed rows at start" in line
    assert "since the previous" not in line


def test_second_forward_reports_the_gap_and_starts_from_zero():
    prof, clock, lines = _profile()
    prof.begin(100)
    prof.staged_piece(0.0, 1.0, 1 << 20, 0)
    clock.t += 2.0
    prof.end()
    clock.t += 0.25
    prof.begin(100)
    clock.t += 1.0
    prof.end()
    assert "[rank" not in lines[1]  # one rank: no label
    assert "bank read 0.00 s" in lines[1] and "unregistered rows" not in lines[1]
    assert "0.25 s since the previous prefill forward" in lines[1]


def test_abort_clears_without_logging():
    prof, _, lines = _profile()
    prof.begin(1)
    prof.abort()
    assert pp.active() is None and lines == []


def test_a_failing_residency_probe_is_dropped():
    def boom():
        raise OSError("mincore failed")

    prof, _, lines = _profile(residency=boom)
    prof.begin(1)
    prof.end()
    assert prof.residency is None
    assert "page cache held" not in lines[0]


def test_read_bytes(tmp_path):
    io = tmp_path / "io"
    io.write_text("rchar: 5\nread_bytes: 4096\nwrite_bytes: 0\n")
    assert pp.read_bytes(str(io)) == 4096
    assert pp.read_bytes(str(tmp_path / "missing")) is None


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


def test_flag():
    assert _parse([]).prefill_profile is False
    assert _parse(["--prefill-profile"]).prefill_profile is True
