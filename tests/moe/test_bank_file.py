"""The bank file as the only copy: commits, torn manifests, reorders in place and their journal.

Once ``ft bank pack`` has run there is no checkpoint to rebuild a damaged layer from, so what is
pinned here is that no crash leaves the manifest vouching for bytes that are not there: a layer
exists only after its blocks are synced, a torn manifest write falls back to the previous one,
and a reorder interrupted at any point either finishes or rolls back from its journal.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

import freetoken.moe.bank_file as bf  # noqa: E402
from freetoken.moe.bank_file import BankFile, BankFileError, layout_from_sample  # noqa: E402
from freetoken.moe.mapped_bank import MappedBanks  # noqa: E402

E = 8


def _rows(layer, cols=12, dtype=torch.uint8):
    t = torch.empty((E, cols), dtype=torch.uint8)
    for e in range(E):
        t[e] = (layer * 31 + e * 7) % 251
    return t.view(dtype) if dtype is not torch.uint8 else t


def _banks(layer):
    return {"packed": _rows(layer), "scale": _rows(layer + 100, cols=4).view(torch.float16)}


def _layout(layers=(0, 1, 2, 3)):
    return layout_from_sample(_banks(0), list(layers), E, {"kind": "test"})


def _file(tmp_path, layers=(0, 1, 2, 3), write=(0, 1, 2, 3), order=None):
    path = str(tmp_path / "bank.ftmb")
    bank = BankFile.create(path, _layout(layers))
    for layer in write:
        bank.write_layer(layer, _banks(layer), order or list(range(E)))
    return bank, path


def _physical(bank, name, layer):
    return bank.read_block(name, layer).view(E, -1)


# ----- commits -----------------------------------------------------------------------------
def test_a_committed_layer_carries_its_order_and_block_hashes(tmp_path):
    order = [3, 1, 0, 2, 7, 6, 5, 4]
    bank, _ = _file(tmp_path, write=(1,), order=order)
    with bank:
        got_order, digest = bank.layer_state(1)
        assert got_order == order
        packed = _banks(1)["packed"].index_select(0, torch.as_tensor(order))
        assert digest["packed"] == hashlib.sha256(packed.numpy().tobytes()).hexdigest()
        assert bank.present_layers() == [1]
        bank.check_digest(1)


def test_a_write_that_dies_leaves_the_layer_absent_not_stale(tmp_path, monkeypatch):
    """Overwriting a committed layer forgets it first: a half-new block must not keep the old hash."""
    bank, _ = _file(tmp_path, write=(0,))
    with bank:
        real = bf._pwrite_all
        calls = []

        def dying(fd, data, offset):
            calls.append(offset)
            if len(calls) == 2:  # the second bank of the layer
                raise OSError("disk went away")
            return real(fd, data, offset)

        monkeypatch.setattr(bf, "_pwrite_all", dying)
        with pytest.raises(OSError):
            bank.write_layer(0, _banks(5), list(range(E)))
        monkeypatch.setattr(bf, "_pwrite_all", real)
        assert bank.layer_state(0) is None


def test_a_torn_manifest_write_falls_back_to_the_previous_one(tmp_path):
    bank, _ = _file(tmp_path, write=(0, 1))
    with bank:
        seq, slot, _ = bank._read_slots()
        # tear the newest slot: flip a byte inside its body
        pos = bank._slot_pos(slot) + bf._SLOT_HEAD.size + 3
        os.pwrite(bank._fd, bytes([os.pread(bank._fd, 1, pos)[0] ^ 0xFF]), pos)
        seq2, slot2, manifest = bank._read_slots()
        assert slot2 == 1 - slot and seq2 == seq - 1
        assert sorted(manifest["layers"]) == ["0"]  # the state before layer 1 was committed


def test_the_largest_manifest_fits_its_slot(tmp_path):
    # every layer committed with the widest order the geometry allows
    bank, _ = _file(tmp_path)
    with bank:
        bank.set_canonical_for("/some/rather/long/path/to/a/packed/checkpoint" * 4)
        for layer in range(4):
            bank.reorder_layer(layer, list(reversed(range(E))))
        assert bank.present_layers() == [0, 1, 2, 3]


_RANK = """
import sys, torch
from freetoken.moe.bank_file import layout_from_sample
from freetoken.moe.mapped_bank import MappedTier
E = 8
def rows(layer, cols=12):
    t = torch.empty((E, cols), dtype=torch.uint8)
    for e in range(E):
        t[e] = (layer * 31 + e * 7) % 251
    return t
def banks(layer):
    return {"packed": rows(layer), "scale": rows(layer + 100, cols=4).view(torch.float16)}
path, layers = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
tier = MappedTier(path, layers, all_layers=range(4), num_experts=E, hot_per_layer=2,
                  layout=layout_from_sample(banks(0), range(4), E, {"kind": "test"}))
assert tier.prepare() is False
for i, layer in enumerate(layers):
    tier.sink(i, banks(layer))
"""


def test_two_processes_write_disjoint_layers_of_one_new_file(tmp_path):
    """The ranks of a --pp-size run: both may find no file, both write their own layers."""
    path = str(tmp_path / "bank.ftmb")
    procs = [
        subprocess.Popen([sys.executable, "-c", _RANK, path, layers], stderr=subprocess.PIPE)
        for layers in ("0,1", "2,3")
    ]
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err.decode()
    with BankFile.open(path) as bank:
        assert bank.present_layers() == [0, 1, 2, 3]
        for layer in range(4):
            bank.check_digest(layer)
            assert torch.equal(_physical(bank, "packed", layer), _banks(layer)["packed"])


def test_writing_into_a_canonical_file_is_refused(tmp_path):
    bank, _ = _file(tmp_path)
    with bank:
        bank.set_canonical_for("/slim")
        with pytest.raises(BankFileError, match="only copy"):
            bank.write_layer(0, _banks(0), list(range(E)))


# ----- reorder in place ----------------------------------------------------------------------
def test_reorder_moves_rows_and_keeps_every_byte(tmp_path):
    first = [7, 6, 5, 4, 3, 2, 1, 0]
    bank, path = _file(tmp_path, order=first)
    new = [2, 0, 1, 3, 5, 4, 7, 6]
    with bank:
        assert bank.reorder_layer(2, new) is True
        assert bank.layer_state(2)[0] == new
        bank.check_digest(2)
        for name, src in _banks(2).items():
            want = src.contiguous().view(torch.uint8).view(E, -1).index_select(0, torch.as_tensor(new))
            assert torch.equal(_physical(bank, name, 2), want)
        assert not os.path.exists(bank.journal_path(2))
        assert bank.reorder_layer(2, new) is False  # already there
        # the other layers are untouched
        assert bank.layer_state(1)[0] == first


def test_a_damaged_layer_is_not_reordered(tmp_path):
    """Permuting bad bytes and committing fresh hashes over them would bless the damage."""
    bank, _ = _file(tmp_path)
    with bank:
        off = bank.layout.offset_of("packed", 1) + 5
        os.pwrite(bank._fd, b"\xAA", off)
        with pytest.raises(BankFileError, match="damaged"):
            bank.reorder_layer(1, list(reversed(range(E))))
        assert bank.layer_state(1)[0] == list(range(E))


def test_a_reorder_that_dies_before_its_commit_rolls_back(tmp_path, monkeypatch):
    bank, _ = _file(tmp_path)
    with bank:
        before = {n: _physical(bank, n, 3).clone() for n in ("packed", "scale")}

        def dying(*_a, **_k):
            raise KeyboardInterrupt  # killed between the in-place write and the commit

        monkeypatch.setattr(bank, "_commit_layer", dying)
        with pytest.raises(KeyboardInterrupt):
            bank.reorder_layer(3, list(reversed(range(E))))
        monkeypatch.undo()
        # the blocks are half new, the manifest still says the old order: the journal decides
        assert os.path.exists(bank.journal_path(3))
        assert not torch.equal(_physical(bank, "packed", 3), before["packed"])
        assert bank.recover_journals(log=lambda _m: None) == [3]
        assert not os.path.exists(bank.journal_path(3))
        assert bank.layer_state(3)[0] == list(range(E))
        for n in before:
            assert torch.equal(_physical(bank, n, 3), before[n])
        bank.check_digest(3)


def test_a_reorder_that_dies_after_its_commit_keeps_the_new_order(tmp_path, monkeypatch):
    bank, _ = _file(tmp_path)
    new = list(reversed(range(E)))
    with bank:
        real_unlink = os.unlink

        def dying(path, *a, **k):
            if ".journal." in str(path):
                raise KeyboardInterrupt
            return real_unlink(path, *a, **k)

        monkeypatch.setattr(os, "unlink", dying)
        with pytest.raises(KeyboardInterrupt):
            bank.reorder_layer(0, new)
        monkeypatch.undo()
        assert os.path.exists(bank.journal_path(0))
        assert bank.recover_journals(log=lambda _m: None) == [0]
        assert bank.layer_state(0)[0] == new
        bank.check_digest(0)
        assert not os.path.exists(bank.journal_path(0))


def test_a_torn_journal_is_discarded_because_nothing_was_written_yet(tmp_path):
    bank, _ = _file(tmp_path)
    with bank:
        with open(bank.journal_path(1), "wb") as f:
            f.write(b"FTMJ" + b"\0" * 40)  # the crash hit while the journal itself was written
        assert bank.recover_journals(log=lambda _m: None) == []
        assert not os.path.exists(bank.journal_path(1))
        bank.check_digest(1)


def test_a_mapping_after_reorder_serves_the_new_order(tmp_path):
    bank, path = _file(tmp_path)
    new = [1, 0, 3, 2, 5, 4, 7, 6]
    with bank:
        bank.reorder_layer(1, new)
    banks = MappedBanks(path, register=False, layers=[1], hot_per_layer=2)
    try:
        src = _banks(1)["packed"]
        for physical, logical in enumerate(new):
            assert torch.equal(banks.sources["packed"][0][physical], src[logical])
    finally:
        banks.close()
