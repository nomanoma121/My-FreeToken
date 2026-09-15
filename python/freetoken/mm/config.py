"""Runtime knobs of the multimodal path. The architecture side (vision_config, mrope) lives in ModelConfig."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# encoder tower kinds a family can register; --mm-disable <kind> leaves that tower unbuilt and refuses its inputs
ENCODER_KINDS = ("vision", "audio")
# checkpoint config sections that describe encoder towers; the parser never sees the section of a tower this process does not build
ENCODER_SECTIONS = ("vision_config", "audio_config")


@dataclass(frozen=True)
class MultimodalConfig:
    # encoder kinds (ENCODER_KINDS) whose tower this process does not build; --text-model-only names them all
    disabled_encoders: frozenset[str] = frozenset()
    # Where encoded items wait between prefill chunks. "cpu": pinned host memory, "cuda": the device.
    embed_cache_device: Literal["cpu", "cuda"] = "cpu"
    # Encoder tower block weights. "host": pinned host banks streamed two blocks at a time behind the compute, "gpu": resident,
    # "cpu": the tower runs on the CPU in the tokenizer worker and the engine builds none (mm/cpu_tower.py).
    encoder_weights: Literal["gpu", "host", "cpu"] = "host"
    # Encoder tower compute dtype on the GPU (Kai). "auto": float32 when the engine runs bfloat16 -- the Qwen VL
    # tower loses ~9% of its output in bf16 (7 fraction bits), in transformers' own code as in this one --
    # else the engine dtype. Or float32 / float16 / bfloat16 explicitly.
    encoder_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"

    def resolve_encoder_dtype(self, engine_dtype: Any) -> str:
        """The torch dtype name the GPU encoder towers are built and run in."""
        import torch

        if self.encoder_dtype != "auto":
            return self.encoder_dtype
        return "float32" if engine_dtype == torch.bfloat16 else str(engine_dtype).removeprefix("torch.")
    # per-image token budget; the family's MMProcessor converts it to its image processor's own limits, None keeps the checkpoint defaults
    image_min_tokens: int | None = None
    image_max_tokens: int | None = None
    # extra keyword arguments the family's MMProcessor passes to the image processor call, after the token budget
    processor_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def text_model_only(self) -> bool:
        return set(ENCODER_KINDS) <= self.disabled_encoders


__all__ = ["ENCODER_KINDS", "ENCODER_SECTIONS", "MultimodalConfig"]
