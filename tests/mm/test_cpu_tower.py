"""--mm-encoder-weights cpu: transformers' Qwen VL vision tower on the CPU turns an item's processor
features into its embeddings, which the engine gathers like a GPU tower's output."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

tcfg = pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe")
tmod = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
safetensors_torch = pytest.importorskip("safetensors.torch")

from freetoken.message import MMItem  # noqa: E402
from freetoken.mm import cpu_tower  # noqa: E402
from freetoken.mm.cpu_tower import (  # noqa: E402
    CPU_VISION_MODEL_TYPES,
    CpuImageEncoder,
    CpuVisionTower,
    check_cpu_tower,
    load_prefixed_state,
    vision_model_class,
)


def _tiny_cfg():
    cfg = tcfg.Qwen3_5MoeVisionConfig(depth=1, hidden_size=64, intermediate_size=128, num_heads=4,
                                      out_hidden_size=32, num_position_embeddings=16)
    cfg._attn_implementation = "sdpa"
    return cfg


def _item(grid, *, hash=7, seed=0):
    t, h, w = grid
    g = torch.Generator().manual_seed(seed)
    n = (t * h * w) // 4
    return MMItem(modality="image", hash=hash, pad_value=1_000_000 + hash, offsets=[[3, 3 + n]],
                  feature=torch.randn(t * h * w, 3 * 2 * 16 * 16, generator=g).to(torch.bfloat16),
                  model_specific_data={"grid_thw": list(grid)})


def test_model_types_and_classes():
    assert set(CPU_VISION_MODEL_TYPES) == {"qwen4_exp", "qwen3_5_moe", "qwen3_5"}
    assert vision_model_class("qwen3_5_moe") is tmod.Qwen3_5MoeVisionModel
    with pytest.raises(ValueError, match="no vision tower"):
        vision_model_class("qwen3_vl")


def test_the_checkpoint_is_checked_from_its_config():
    cfg = _tiny_cfg()
    assert check_cpu_tower(SimpleNamespace(model_type="qwen3_5_moe", vision_config=cfg)) is cfg
    with pytest.raises(ValueError, match="no vision_config"):
        check_cpu_tower(SimpleNamespace(model_type="qwen3_5_moe"))
    cfg.deepstack_visual_indexes = [5, 11, 17]
    with pytest.raises(ValueError, match="DeepStack"):
        check_cpu_tower(SimpleNamespace(model_type="qwen3_5_moe", vision_config=cfg))


def test_the_tower_reads_the_shards_and_matches_the_hf_module(tmp_path, monkeypatch):
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    ref = tmod.Qwen3_5MoeVisionModel(cfg).eval()
    state = {f"model.visual.{k}": v.to(torch.bfloat16) for k, v in ref.state_dict().items()}
    state["model.language_model.layers.0.dummy"] = torch.zeros(2)
    safetensors_torch.save_file(state, str(tmp_path / "model-00003-of-00003.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model-00003-of-00003.safetensors" for k in state}}))
    loaded = load_prefixed_state(str(tmp_path))
    assert set(loaded) == set(ref.state_dict())

    monkeypatch.setattr("transformers.AutoConfig.from_pretrained",
                        lambda *_a, **_k: SimpleNamespace(model_type="qwen3_5_moe", vision_config=_tiny_cfg()))
    tower = CpuVisionTower.from_checkpoint(str(tmp_path))
    ref.load_state_dict({k: v.float() for k, v in loaded.items()}, strict=True)
    item = _item((1, 4, 6))
    out = tower.encode(item.feature, item.grid_thw)
    with torch.no_grad():
        want = ref(item.feature.float(), grid_thw=torch.tensor([[1, 4, 6]])).pooler_output
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (6, 32)
    torch.testing.assert_close(out.float(), want, rtol=2e-2, atol=2e-2)


class _CountingTower:
    def __init__(self):
        self.calls = 0

    def encode(self, feature, grid):
        self.calls += 1
        t, h, w = grid
        return torch.full(((t * h * w) // 4, 8), float(self.calls), dtype=torch.bfloat16)


def test_items_leave_with_embeddings_and_without_features():
    tower = _CountingTower()
    enc = CpuImageEncoder(tower, cache_entries=2)
    a, b = _item((1, 4, 4), hash=1), _item((1, 2, 6), hash=2)
    enc.encode_items([a, b])
    for item in (a, b):
        assert item.feature is None and item.precomputed_embeddings.shape == (item.num_tokens, 8)
        item.validate()  # exactly one of feature / precomputed_embeddings
    assert tower.calls == 2


def test_a_resent_image_is_encoded_once():
    tower = _CountingTower()
    enc = CpuImageEncoder(tower, cache_entries=1)
    first, again, other = _item((1, 4, 4), hash=5), _item((1, 4, 4), hash=5), _item((1, 4, 4), hash=6)
    enc.encode_items([first])
    enc.encode_items([again])
    assert tower.calls == 1 and enc.hits == 1
    assert torch.equal(again.precomputed_embeddings, first.precomputed_embeddings)
    enc.encode_items([other])  # capacity 1: evicts hash 5
    enc.encode_items([_item((1, 4, 4), hash=5)])
    assert tower.calls == 3


def test_a_token_count_the_prompt_does_not_have_is_refused():
    class Short(_CountingTower):
        def encode(self, feature, grid):
            return torch.zeros(1, 8, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="embeddings for an image of 4 tokens"):
        CpuImageEncoder(Short(), cache_entries=0).encode_items([_item((1, 4, 4))])


def test_the_tower_loads_at_the_first_image(monkeypatch):
    built = []

    def fake(path, dtype=torch.float32):
        built.append(path)
        return _CountingTower()

    monkeypatch.setattr(cpu_tower.CpuVisionTower, "from_checkpoint", staticmethod(fake))
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained",
                        lambda *_a, **_k: SimpleNamespace(model_type="qwen4_exp", vision_config=_tiny_cfg()))
    enc = CpuImageEncoder.for_checkpoint("/ckpt")
    assert built == [], "no RAM spent before an image arrives"
    enc.encode_items([_item((1, 4, 4))])
    enc.encode_items([_item((1, 4, 4), hash=9)])
    assert built == ["/ckpt"]


def test_the_tokenizer_worker_encodes_and_isolates_a_failure():
    from freetoken.message import UserMsg
    from freetoken.core import SamplingParams
    from freetoken.tokenizer.server import _tokenize_requests

    class Tokenizer:
        def tokenize(self, msgs):
            (msg,) = msgs
            items = [_item((1, 4, 4), hash=msg.uid)]
            return [UserMsg(uid=msg.uid, input_ids=torch.arange(8, dtype=torch.int32),
                            sampling_params=SamplingParams(), mm_items=items)]

    class Encoder:
        def encode_items(self, items):
            if items[0].hash == 2:
                raise ValueError("bad image")
            for item in items:
                item.precomputed_embeddings, item.feature = torch.zeros(item.num_tokens, 8), None

    class Log:
        def warning(self, *_a, **_k):
            pass

    msgs = [SimpleNamespace(uid=1), SimpleNamespace(uid=2)]
    backend, errors = _tokenize_requests(Tokenizer(), msgs, Log(), Encoder())
    assert [m.uid for m in backend] == [1] and backend[0].mm_items[0].feature is None
    assert [e.uid for e in errors] == [2] and "bad image" in errors[0].error
