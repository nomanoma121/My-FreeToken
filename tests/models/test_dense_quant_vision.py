"""--dense-quant fp8 leaves the vision tower bf16: its blocks are streamed from host banks, so fp8
buys no VRAM there, and the tower is where image features are decided."""

from __future__ import annotations

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info


@pytest.fixture(autouse=True)
def _single_rank():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def test_the_load_time_wrapper_names_no_tower_projection():
    from freetoken.layers.quantization import LoadTimeFp8Config

    q = LoadTimeFp8Config(None)
    for prefix in ("visual.blocks.0.attn.qkv", "visual.blocks.3.mlp.linear_fc1", "visual.merger.linear_fc2",
                   "model.visual.blocks.0.attn.proj"):
        assert q.scheme_for(prefix) is None, prefix
    assert q.scheme_for("model.layers.0.self_attn.o_proj") is not None  # the text tower still narrows


def test_a_tower_built_under_dense_quant_holds_no_fp8():
    from freetoken.layers.quantization import LoadTimeFp8Config
    from freetoken.models.qwen3_vl import Qwen3VLVisionModel, VisionConfig

    vc = VisionConfig(
        hidden_size=64, depth=2, num_heads=4, intermediate_size=128, patch_size=16, temporal_patch_size=2,
        spatial_merge_size=2, num_position_embeddings=16, out_hidden_size=32, in_channels=3,
        deepstack_visual_indexes=(),
    )
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            tower = Qwen3VLVisionModel(vc, quant_config=LoadTimeFp8Config(None), prefix="visual")
    finally:
        torch.set_default_dtype(torch.float32)
    fp8 = [k for k, t in tower.state_dict().items() if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)]
    assert fp8 == []
