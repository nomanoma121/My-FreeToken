"""``ft bank info`` / ``reorder``: the offline side of the bank file, wired to the ``ft`` entry point."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from freetoken.moe import bank_cli  # noqa: E402
from freetoken.moe.bank_file import BankFile, layout_from_sample  # noqa: E402

E, L = 5, 3


def _banks(layer):
    return {"packed": torch.stack([torch.full((6,), layer * 10 + e, dtype=torch.uint8) for e in range(E)])}


@pytest.fixture
def bank_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(bank_cli, "_model_config", lambda _p: SimpleNamespace(first_k_dense_replace=0))
    d = tmp_path / "bankmap"
    d.mkdir()
    with BankFile.create(str(d / "bank.ftmb"), layout_from_sample(_banks(0), range(L), E, {"kind": "nvfp4", "kernel": "triton"})) as bank:
        for layer in range(L):
            bank.write_layer(layer, _banks(layer), list(range(E)))
    (tmp_path / "model").mkdir()
    return tmp_path, d


def test_info_describes_the_file(bank_dir, capsys):
    tmp, d = bank_dir
    assert bank_cli.main(["info", "--model-path", str(tmp / "model"), "--moe-bank-dir", str(d)]) == 0
    out = capsys.readouterr().out
    assert "nvfp4 / triton, 5 experts x 3 MoE layers" in out
    assert "committed layers: 0-2 (complete)" in out
    assert "checkpoint order (no placement applied): layers 0-2" in out


def test_reorder_applies_every_ranks_histogram(bank_dir, capsys):
    tmp, d = bank_dir
    stats = []
    for rank, (start, rows) in enumerate(((0, [[0, 0, 0, 0, 9]]), (1, [[0, 9, 0, 0, 0], [0, 0, 9, 0, 1]]))):
        p = tmp / f"s.rank{rank}.json"
        p.write_text(json.dumps({"layer_range": [start, start + len(rows)], "decode_freq": rows}), encoding="utf-8")
        stats.append(str(p))
    args = ["reorder", "--model-path", str(tmp / "model"), "--moe-bank-dir", str(d), "--moe-bank-stats", *stats]
    assert bank_cli.main(args) == 0
    with BankFile.open(str(d / "bank.ftmb")) as bank:
        assert [bank.layer_state(l)[0][0] for l in range(L)] == [4, 1, 2]
        for layer in range(L):
            bank.check_digest(layer)
    assert bank_cli.main(args) == 0
    assert "already in that order" in capsys.readouterr().out


def test_ft_knows_the_bank_command():
    from freetoken import cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["bank", "--help"])
    assert exc.value.code == 0
