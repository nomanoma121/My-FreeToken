"""--mm-encoder-weights cpu: the Qwen VL vision tower on the CPU, in the tokenizer worker.

On a 6 GB card the GPU tower (streamed blocks plus the resident merger and staging buffers) takes
VRAM the expert cache and the KV pool need, and its pinned block bank takes the pin quota the
expert banks need. This runs transformers' own vision module on the CPU instead, image by image,
and hands the scheduler each item's final embeddings (``MMItem.precomputed_embeddings``): the
engine then only gathers them into the prompt rows, exactly as it does for a tower of its own.
One image is a few seconds of CPU work. The tower and its merger are fp32 (about 1.7 GB of RAM).
"""

from __future__ import annotations

import glob
import importlib
import json
import os
from collections import OrderedDict
from typing import Any, Dict, Iterable, List

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

VISUAL_PREFIXES = ("model.visual.", "visual.")

# HF model_type -> (module, class) of the vision tower transformers implements for it.
_VISION_CLASSES = {
    "qwen4_exp": ("transformers.models.qwen4_exp.modeling_qwen4_exp", "Qwen4ExpVisionModel"),
    "qwen3_5_moe": ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeVisionModel"),
    "qwen3_5": ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5VisionModel"),
}
CPU_VISION_MODEL_TYPES = tuple(_VISION_CLASSES)


def vision_model_class(model_type: str):
    """The transformers vision tower class for ``model_type`` (ValueError if there is none here)."""
    try:
        module, name = _VISION_CLASSES[model_type]
    except KeyError:
        raise ValueError(
            f"--mm-encoder-weights cpu has no vision tower for model_type {model_type!r} "
            f"(supported: {', '.join(CPU_VISION_MODEL_TYPES)})"
        ) from None
    return getattr(importlib.import_module(module), name)


def load_prefixed_state(model_path: str, prefixes: Iterable[str] = VISUAL_PREFIXES) -> Dict[str, torch.Tensor]:
    """Every checkpoint tensor under one of ``prefixes`` (prefix stripped), read from the
    safetensors shards on the CPU: through ``model.safetensors.index.json`` when present, else by
    scanning every shard header."""
    from safetensors import safe_open

    prefixes = tuple(prefixes)
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    by_shard: Dict[str, List[str]] = {}
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        for name, shard in weight_map.items():
            if name.startswith(prefixes):
                by_shard.setdefault(os.path.join(model_path, shard), []).append(name)
    else:
        for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
            with safe_open(shard, framework="pt", device="cpu") as f:
                names = [k for k in f.keys() if k.startswith(prefixes)]
            if names:
                by_shard[shard] = names
    state: Dict[str, torch.Tensor] = {}
    for shard, names in by_shard.items():
        with safe_open(shard, framework="pt", device="cpu") as f:
            for name in names:
                key = next(name[len(p):] for p in prefixes if name.startswith(p))
                state[key] = f.get_tensor(name)
    if not state:
        raise FileNotFoundError(f"no tensors under {prefixes} in {model_path}: this checkpoint has no vision tower")
    return state


def check_cpu_tower(hf: Any, model_path: str = "") -> Any:
    """The checkpoint config's vision_config, if --mm-encoder-weights cpu can serve it (ValueError
    otherwise) -- from the config alone, so a server refuses at start rather than at its first image."""
    vc = getattr(hf, "vision_config", None)
    if vc is None:
        raise ValueError(f"{model_path}: config has no vision_config")
    if getattr(vc, "deepstack_visual_indexes", None):
        # DeepStack feeds vision features into several decoder layers; precomputed embeddings
        # carry only the merger output the prompt rows take
        raise ValueError(f"{model_path}: --mm-encoder-weights cpu does not serve DeepStack towers")
    vision_model_class(getattr(hf, "model_type", ""))
    return vc


class CpuVisionTower:
    """The HF vision tower + merger, resident on the CPU."""

    def __init__(self, model: Any, dtype: torch.dtype = torch.float32) -> None:
        self.model = model
        self.dtype = dtype

    @classmethod
    def from_checkpoint(cls, model_path: str, dtype: torch.dtype = torch.float32) -> "CpuVisionTower":
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained(model_path)
        vc = check_cpu_tower(hf, model_path)
        vc._attn_implementation = "sdpa"
        model = vision_model_class(getattr(hf, "model_type", ""))(vc).to(dtype).eval()
        model.load_state_dict(load_prefixed_state(model_path), strict=True)
        return cls(model, dtype)

    @torch.inference_mode()
    def encode(self, pixel_values: torch.Tensor, grid_thw: List[int]) -> torch.Tensor:
        """One image: ``pixel_values [t*h*w, C*T*P*P]`` -> ``[t*h*w / merge**2, text_hidden]`` bf16."""
        grid = torch.tensor([list(grid_thw)], dtype=torch.long)
        out = self.model(pixel_values.to(self.dtype), grid_thw=grid)
        return out.pooler_output.to(torch.bfloat16).contiguous()


class CpuImageEncoder:
    """Turns items that carry processor features into items that carry their embeddings.

    Chat clients resend a conversation's images every turn, and again for title and tag
    generation, so the last ``FT_IMAGE_EMBED_CACHE`` (default 32, 0 disables) images stay cached
    by their content hash -- the same hash the radix cache and the engine's encoder cache use.
    """

    def __init__(self, tower: Any = None, cache_entries: int | None = None, model_path: str | None = None) -> None:
        # the tower loads at the first image (``model_path``) unless one is given: a server that
        # never sees an image keeps the RAM
        self.tower = tower
        self.model_path = model_path
        if cache_entries is None:
            cache_entries = int(os.environ.get("FT_IMAGE_EMBED_CACHE", "32"))
        self.capacity = max(0, cache_entries)
        self._cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    @classmethod
    def for_checkpoint(cls, model_path: str) -> "CpuImageEncoder":
        """An encoder whose tower loads at the first image; the checkpoint is checked now."""
        from transformers import AutoConfig

        check_cpu_tower(AutoConfig.from_pretrained(model_path), model_path)
        return cls(model_path=model_path)

    def _tower(self) -> Any:
        if self.tower is None:
            logger.info("--mm-encoder-weights cpu: loading the vision tower on the CPU (%s)", self.model_path)
            self.tower = CpuVisionTower.from_checkpoint(self.model_path)
            logger.info("--mm-encoder-weights cpu: vision tower ready")
        return self.tower

    def _embed(self, item: Any) -> torch.Tensor:
        hit = self._cache.get(item.hash)
        if hit is not None:
            self._cache.move_to_end(item.hash)
            self.hits += 1
            return hit
        self.misses += 1
        emb = self._tower().encode(item.feature, item.grid_thw)
        if emb.shape[0] != item.num_tokens:
            raise ValueError(f"vision tower gave {emb.shape[0]} embeddings for an image of {item.num_tokens} tokens")
        if self.capacity:
            self._cache[item.hash] = emb
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
        return emb

    def encode_items(self, items: List[Any] | None) -> None:
        """In place: each item that still carries a feature gets its embeddings and drops the feature."""
        for item in items or ():
            if item.feature is None:
                continue
            item.precomputed_embeddings = self._embed(item)
            item.feature = None


__all__ = [
    "CPU_VISION_MODEL_TYPES",
    "check_cpu_tower",
    "CpuImageEncoder",
    "CpuVisionTower",
    "VISUAL_PREFIXES",
    "load_prefixed_state",
    "vision_model_class",
]
