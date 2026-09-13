"""``--kv-cache-dtype`` on the hybrid full/sliding-window pool (gpt-oss).

Both groups narrow: the attention backend reads one ``kv_quant`` off the pool and hands the
SWA layers the window pool's slabs through the same call, so a window group left at 16 bits
would be read as codes. The CPU tests pin the layout and the bookkeeping around it (rebuild,
the per-token bytes the budget and the cache-status slider see); the CUDA test pins the store,
including the full -> window slot translation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.kvcache.kv_quant import Q4_0, Q8_0, dequantize_rows, quantize_rows


def _init_tp() -> None:
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _gpt_oss_model_config(head_dim: int = 64, num_kv_heads: int = 8):
    """gpt-oss's shape, four layers deep: alternating window / full, one KV geometry."""
    from freetoken.models.config import (
        FullAttentionGroupConfig,
        ModelConfig,
        RotaryConfig,
        SWAAttentionGroupConfig,
    )

    rope = RotaryConfig(
        head_dim=head_dim, rotary_dim=head_dim, max_position=4096, base=150_000.0, scaling=None
    )
    return ModelConfig(
        num_layers=4,
        num_qo_heads=64,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=2880,
        vocab_size=32000,
        intermediate_size=2880,
        rms_norm_eps=1e-5,
        rotary_config=rope,
        hidden_act="swiglu",
        tie_word_embeddings=False,
        num_experts=32,
        num_experts_per_tok=4,
        moe_intermediate_size=2880,
        norm_topk_prob=True,
        model_type="gpt_oss",
        architectures=["GptOssForCausalLM"],
        moe_enabled=True,
        attention_groups=(
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=(0, 2),
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_config=rope,
                sliding_window=128,
            ),
            FullAttentionGroupConfig(
                name="full",
                layer_ids=(1, 3),
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_config=rope,
            ),
        ),
    )


def _pool(spec, *, num_pages=8, page_size=4, num_swa_tokens=24, device="cpu"):
    from freetoken.kvcache import create_kvcache_pool

    _init_tp()
    return create_kvcache_pool(
        model_config=_gpt_oss_model_config(),
        num_pages=num_pages,
        page_size=page_size,
        num_swa_tokens=num_swa_tokens,
        dtype=torch.bfloat16,
        device=torch.device(device),
        kv_quant=spec,
    )


@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
def test_gpt_oss_is_accepted_and_both_groups_hold_codes(spec):
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    pool = _pool(spec)
    assert isinstance(pool, HybridSWAKVCache)
    assert pool.kv_quant is spec
    assert pool.dtype is torch.bfloat16, "the compute dtype must not change"
    assert pool.store_dtype is torch.uint8

    width, blocks = spec.code_bytes_per_row(64), spec.blocks_per_row(64)
    # layer 1 is full: [pages, page_size, kv heads, code bytes]; layer 0 is window: [tokens, 1, ...]
    assert pool.group_of(1) == "full" and pool.group_of(0) == "swa"
    assert pool.k_cache(1).shape == (8, 4, 8, width)
    assert pool.v_cache(1).dtype is torch.uint8
    assert pool.k_scales(1).shape == (8, 4, 8, blocks)
    assert pool.k_cache(0).shape == (24, 1, 8, width)
    assert pool.v_cache(0).dtype is torch.uint8
    assert pool.v_scales(0).shape == (24, 1, 8, blocks)
    assert pool.v_scales(0).dtype is torch.float16
    # the window pool's slot 0 is the sentinel out-of-window tokens read; it must decode to 0
    assert not pool.k_cache(0).any() and not pool.k_scales(0).any()


def test_the_unquantized_pool_is_unchanged():
    pool = _pool(None)
    assert pool.kv_quant is None
    assert pool.store_dtype is torch.bfloat16
    assert pool.k_cache(1).shape == (8, 4, 8, 64)
    assert pool.k_cache(0).shape == (24, 1, 8, 64)
    with pytest.raises(AssertionError, match="not quantized"):
        pool.k_scales(0)


@pytest.mark.parametrize("spec", [None, Q4_0], ids=["bf16", "q4_0"])
def test_rebuild_keeps_the_layout(spec):
    """The 16-bit rebuild read head_dim and dtype back off the buffer; a code slab has neither."""
    pool = _pool(spec)
    before = (pool.k_cache(1).shape[-1], pool.store_dtype)
    pool.rebuild(num_full_pages=12, num_swa_tokens=40)

    assert (pool.k_cache(1).shape[-1], pool.store_dtype) == before
    assert pool.dtype is torch.bfloat16
    assert pool.k_cache(1).shape[0] == 12
    assert pool.k_cache(0).shape[0] == 40
    assert pool.full_kv_pool.head_dim == pool.swa_kv_pool.head_dim == 64
    if spec is not None:
        assert pool.k_scales(1).shape == (12, 4, 8, spec.blocks_per_row(64))
        assert pool.v_scales(0).shape == (40, 1, 8, spec.blocks_per_row(64))
        # and a second rebuild does not drift
        pool.rebuild(num_full_pages=6, num_swa_tokens=24)
        assert pool.k_cache(0).shape == (24, 1, 8, spec.code_bytes_per_row(64))


@pytest.mark.parametrize("spec", [None, Q8_0, Q4_0], ids=["bf16", "q8_0", "q4_0"])
def test_unit_bytes_agree_with_the_budget(spec):
    """What the pool reports per token (cache-status slider) is what the budget priced it at."""
    from freetoken.distributed import DistributedInfo
    from freetoken.kvcache.base import spec_kv_bytes_per_token

    pool = _pool(spec)
    config = SimpleNamespace(
        kv_cache_dtype=None if spec is None else spec.name,
        dtype=torch.bfloat16,
        tp_info=DistributedInfo(rank=0, size=1),
    )
    priced = {
        s.name: spec_kv_bytes_per_token(s, config)
        for s in _gpt_oss_model_config().kv_cache_group_specs()
    }
    assert pool.unit_bytes() == (priced["full"], priced["swa"])
    if spec is Q4_0:
        # 2 layers x K,V x 8 heads x (32 code bytes + 2 fp16 scales)
        assert priced["full"] == 2 * 2 * 8 * 36


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("spec", [Q8_0, Q4_0], ids=lambda s: s.name)
def test_store_lands_in_the_right_group_and_slot(spec):
    """A window layer's store goes through the full -> window slot map; a full layer's does not."""
    torch.manual_seed(11)
    pool = _pool(spec, device="cuda")
    dev = torch.device("cuda")

    full_slots = torch.tensor([5, 9, 30], dtype=torch.int32, device=dev)
    pool.alloc_swa(full_slots)
    swa_slots = pool.translate_loc_from_full_to_swa(full_slots).long()
    assert (swa_slots > 0).all()

    k = torch.randn(3, 8, 64, device=dev, dtype=torch.bfloat16)
    v = torch.randn(3, 8, 64, device=dev, dtype=torch.bfloat16)
    want_kc, want_ks = quantize_rows(k.float().cpu(), spec)
    want_vc, want_vs = quantize_rows(v.float().cpu(), spec)

    for layer in (0, 1):  # window, full
        pool.store_kv(k.view(3, -1), v.view(3, -1), full_slots, layer)

    width, blocks = spec.code_bytes_per_row(64), spec.blocks_per_row(64)
    cases = (
        (0, swa_slots),  # window layer: the translated slots
        (1, full_slots.long()),  # full layer: the page-table slots as given
    )
    for layer, rows in cases:
        kc = pool.k_cache(layer).reshape(-1, 8, width)[rows].cpu()
        ks = pool.k_scales(layer).reshape(-1, 8, blocks)[rows].cpu()
        vc = pool.v_cache(layer).reshape(-1, 8, width)[rows].cpu()
        vs = pool.v_scales(layer).reshape(-1, 8, blocks)[rows].cpu()
        assert torch.equal(kc, want_kc) and torch.equal(vc, want_vc), f"layer {layer} codes"
        assert torch.equal(ks, want_ks) and torch.equal(vs, want_vs), f"layer {layer} scales"
        got_k = dequantize_rows(kc, ks, spec, torch.float32)
        assert torch.allclose(got_k, k.float().cpu(), atol=float(ks.max()) * 1.01)

    # nothing else in the window pool was touched, the sentinel included
    untouched = torch.ones(pool.swa_num_tokens, dtype=torch.bool)
    untouched[swa_slots.cpu()] = False
    assert not pool.k_cache(0).reshape(-1, 8, width)[untouched.to(dev)].any()
