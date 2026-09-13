"""Expert placement: the renumbering, the budget arithmetic and the router remap.

What is being guarded is that "resident" stays expressible as a contiguous physical prefix,
and that the same histogram always produces the same order -- a reordering between runs would
leave the previous run's bank file describing a placement that no longer holds, with every
shape still matching.
"""

from __future__ import annotations

import json

import pytest

from freetoken.moe.bank_disk import (
    apply_permutation,
    bank_file_path,
    hot_per_layer_for_budget,
    load_freq,
    parse_size,
    permutation_tensor,
    plan_placement,
)


# ----- placement ------------------------------------------------------------------------
def test_plan_orders_each_layer_by_frequency():
    freq = {0: [1, 90, 5, 0], 1: [4, 4, 100, 1]}
    p = plan_placement([0, 1], 4, 2, freq)
    assert p.order[0] == [1, 2, 0, 3]
    # ties (both 4) resolve by logical id, so the same histogram always gives the same file
    assert p.order[1] == [2, 0, 1, 3]
    assert p.cold_per_layer == 2


def test_plan_without_a_histogram_is_identity():
    p = plan_placement([0, 1], 4, 3, None)
    assert p.order[0] == [0, 1, 2, 3] and p.order[1] == [0, 1, 2, 3]


def test_plan_is_reproducible_when_everything_ties():
    freq = {0: [0, 0, 0, 0]}
    assert plan_placement([0], 4, 2, freq).order == plan_placement([0], 4, 2, freq).order


def test_plan_rejects_a_histogram_of_the_wrong_width():
    with pytest.raises(ValueError, match="model has 4"):
        plan_placement([0], 4, 2, {0: [1, 2, 3]})


def test_plan_clamps_the_resident_count():
    assert plan_placement([0], 4, 99, None).hot_per_layer == 4
    assert plan_placement([0], 4, -1, None).hot_per_layer == 0


def test_to_physical_is_the_inverse_of_order():
    p = plan_placement([0], 4, 2, {0: [1, 90, 5, 0]})
    for logical, physical in enumerate(p.to_physical(0)):
        assert p.order[0][physical] == logical


def test_hot_per_layer_for_budget():
    # 4 layers x 100 B cells; 900 B budget -> 225 B per layer -> 2 experts each
    assert hot_per_layer_for_budget(4, 8, 100, 900) == 2
    assert hot_per_layer_for_budget(4, 8, 100, 10_000) == 8  # clamped to num_experts
    assert hot_per_layer_for_budget(4, 8, 100, 0) == 0
    assert hot_per_layer_for_budget(0, 8, 100, 900) == 8  # no layers: nothing to solve


# ----- histograms -----------------------------------------------------------------------
def test_load_freq_merges_ranks_by_global_layer_id(tmp_path):
    for rank, start in ((0, 0), (1, 2)):
        (tmp_path / f"s.rank{rank}.json").write_text(json.dumps({
            "layer_range": [start, start + 2],
            "decode_freq": [[rank, 1], [rank, 2]],
        }), encoding="utf-8")
    freq = load_freq(sorted(str(p) for p in tmp_path.glob("*.json")))
    assert sorted(freq) == [0, 1, 2, 3]
    assert freq[0] == [0, 1] and freq[2] == [1, 1] and freq[3] == [1, 2]


def test_load_freq_sums_sessions_for_the_same_layer(tmp_path):
    # pooling sessions of different work is how a placement stops being fitted to one
    # conversation; overwriting would keep only the last file
    for i, counts in enumerate(([1, 2, 3], [10, 0, 5])):
        (tmp_path / f"s{i}.json").write_text(
            json.dumps({"layer_range": [0, 1], "decode_freq": [counts]}), encoding="utf-8")
    assert load_freq([str(tmp_path / "s0.json"), str(tmp_path / "s1.json")])[0] == [11, 2, 8]


def test_load_freq_rejects_mismatched_widths(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps({"layer_range": [0, 1], "decode_freq": [[1, 2, 3]]}), encoding="utf-8")
    (tmp_path / "b.json").write_text(
        json.dumps({"layer_range": [0, 1], "decode_freq": [[1, 2]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="an earlier file had"):
        load_freq([str(tmp_path / "a.json"), str(tmp_path / "b.json")])


def test_load_freq_rejects_a_graph_captured_run(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"layer_range": [0, 1], "decode_freq": None}), encoding="utf-8")
    with pytest.raises(ValueError, match="disable-cuda-graph"):
        load_freq([str(p)])


# ----- router remap ---------------------------------------------------------------------
def test_apply_permutation_rewrites_ids_in_place():
    torch = pytest.importorskip("torch")
    p = plan_placement([0], 6, 3, {0: [0, 9, 8, 0, 7, 0]})
    # order = [1, 2, 4, 0, 3, 5] -> logical 1 is physical 0, logical 4 is physical 2, ...
    ids = torch.tensor([[1, 4, 5], [0, 2, 3]], dtype=torch.int32)
    before = ids.data_ptr()
    apply_permutation(ids, permutation_tensor(p, 0))
    assert ids.tolist() == [[0, 2, 5], [3, 1, 4]]
    assert ids.data_ptr() == before  # in place: the decode path reuses this buffer
    assert ids.dtype == torch.int32


def test_apply_permutation_is_a_noop_without_a_placement():
    torch = pytest.importorskip("torch")
    ids = torch.tensor([[3, 1]], dtype=torch.int32)
    apply_permutation(ids, None)
    assert ids.tolist() == [[3, 1]]


def test_permutation_is_a_bijection():
    torch = pytest.importorskip("torch")
    p = plan_placement([0], 8, 4, {0: [3, 1, 4, 1, 5, 9, 2, 6]})
    ids = torch.arange(8, dtype=torch.int32).reshape(1, 8)
    apply_permutation(ids, permutation_tensor(p, 0))
    assert sorted(ids.flatten().tolist()) == list(range(8))
    for logical, physical in enumerate(ids.flatten().tolist()):
        assert p.order[0][physical] == logical


def test_offload_layer_has_the_permutation_default_without_init():
    """The routed entry points run on instances built with __new__ (the short-prefill
    dispatch test does exactly that), so the remap's attribute cannot live only in __init__.
    """
    pytest.importorskip("torch")
    try:
        from freetoken.layers import moe as moe_layers
    except Exception:  # needs the compiled kernels; covered on the target machine
        pytest.skip("freetoken.layers.moe needs the built kernel extensions")
    layer = moe_layers.OffloadMoELayer.__new__(moe_layers.OffloadMoELayer)
    assert layer.expert_perm is None


# ----- sizing ---------------------------------------------------------------------------
def test_parse_size():
    assert parse_size("50G") == 50 * 2**30
    assert parse_size("50GiB") == 50 * 2**30
    assert parse_size("512M") == 512 * 2**20
    assert parse_size("0.5T") == 2**39
    assert parse_size("1024") == 1024
    assert parse_size(None) is None and parse_size("") is None
    with pytest.raises(ValueError, match="could not parse"):
        parse_size("lots")


def test_bank_file_path_is_per_rank(tmp_path):
    a = bank_file_path("/models/Flash-Next", 0, 2, str(tmp_path))
    b = bank_file_path("/models/Flash-Next", 1, 2, str(tmp_path))
    assert a != b and a.endswith("bank.rank0of2.ftmb")


def test_bank_file_path_defaults_outside_the_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    p = bank_file_path("/models/Flash-Next", 0, 1, None)
    assert str(tmp_path) in p and "Flash-Next" in p
    assert "/models/" not in p.replace("\\", "/")


def _rank_files(tmp_path, width=4):
    # rank 0 owns decoder layers 0..1, rank 1 owns 2..3; every layer's histogram is distinct,
    # hottest expert = the global layer id, so a placement shows which layer it was sorted by
    paths = []
    for rank, start in ((0, 0), (1, 2)):
        rows = [[100 if e == start + i else e for e in range(width)] for i in range(2)]
        p = tmp_path / f"moe-stats.rank{rank}.json"
        p.write_text(json.dumps({"layer_range": [start, start + 2], "decode_freq": rows}),
                     encoding="utf-8")
        paths.append(str(p))
    return paths


class _Cfg:
    num_experts = 4
    hidden_size = 8
    moe_intermediate_size = 4


def test_a_later_pipeline_rank_sorts_by_its_own_layers(tmp_path, monkeypatch):
    from freetoken.moe import bank_disk

    monkeypatch.setattr(bank_disk, "cell_bytes_from_config", lambda _cfg: 100)
    paths = _rank_files(tmp_path)
    # rank 1: local layers 0..1 are the model's 2..3
    placement, _ = bank_disk.plan_from_config(_Cfg(), 200, [0, 1], paths, first_bank_layer=2)
    assert placement.order[0][0] == 2 and placement.order[1][0] == 3
    # rank 0 is unchanged
    placement, _ = bank_disk.plan_from_config(_Cfg(), 200, [0, 1], paths, first_bank_layer=0)
    assert placement.order[0][0] == 0 and placement.order[1][0] == 1


def test_missing_histograms_for_a_rank_are_said(tmp_path, monkeypatch):
    from freetoken.moe import bank_disk

    monkeypatch.setattr(bank_disk, "cell_bytes_from_config", lambda _cfg: 100)
    rank0_only = _rank_files(tmp_path)[:1]
    said = []
    placement, _ = bank_disk.plan_from_config(
        _Cfg(), 200, [0, 1], rank0_only, first_bank_layer=2, warn=said.append
    )
    assert placement.order[0] == [0, 1, 2, 3]  # no histogram: expert-id order, not rank 0's
    assert said and "0 of this rank's 2 MoE layers (2..3)" in said[0]


def test_leading_dense_layers_are_not_bank_layers(tmp_path):
    p = tmp_path / "moe-stats.rank1.json"
    p.write_text(json.dumps({"layer_range": [3, 5], "decode_freq": [[1, 2], [3, 4]]}),
                 encoding="utf-8")
    assert sorted(load_freq([str(p)], first_k_dense=1)) == [2, 3]
