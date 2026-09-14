"""Without torchvision (AutoImageProcessor requires it) the Qwen VL processor still loads, as the
PIL backend of the checkpoint's image processor class."""

from __future__ import annotations

import json

import pytest


def _checkpoint(tmp_path, kind="Qwen2VLImageProcessorFast"):
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({
        "size": {"longest_edge": 16777216, "shortest_edge": 65536},
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "image_processor_type": kind,
    }))
    return str(tmp_path)


def _no_torchvision(monkeypatch):
    transformers = pytest.importorskip("transformers")

    class Gated:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise ImportError("AutoImageProcessor requires the Torchvision library but it was not found")

    monkeypatch.setattr(transformers, "AutoImageProcessor", Gated)


def test_the_pil_backend_stands_in(monkeypatch, tmp_path):
    pytest.importorskip("transformers.models.qwen2_vl.image_processing_pil_qwen2_vl")
    from freetoken.mm.processor import _load_image_processor

    _no_torchvision(monkeypatch)
    proc = _load_image_processor(_checkpoint(tmp_path))
    assert type(proc).__name__ == "Qwen2VLImageProcessorPil"
    assert dict(proc.size) == {"longest_edge": 16777216, "shortest_edge": 65536}


def test_an_unknown_processor_still_says_torchvision_is_missing(monkeypatch, tmp_path):
    from freetoken.mm.processor import _load_image_processor

    _no_torchvision(monkeypatch)
    with pytest.raises(ImportError, match="Torchvision.*no PIL backend"):
        _load_image_processor(_checkpoint(tmp_path, kind="SiglipImageProcessorFast"))
