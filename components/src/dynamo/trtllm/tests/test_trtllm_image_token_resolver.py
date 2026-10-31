# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TensorRT-LLM rc21 image-marker resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA/GPU unavailable; importing the TensorRT-LLM worker requires CUDA.",
        allow_module_level=True,
    )

from dynamo.trtllm.workers import llm_worker
from dynamo.trtllm.workers.llm_worker import (
    _MM_ROUTING_MODEL_TYPES,
    _resolve_image_token_id,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.trtllm,
    pytest.mark.multimodal,
    pytest.mark.unit,
    pytest.mark.gpu_1,
]

# Intentionally unprofiled: these import-heavy, zero-VRAM tests run in the
# sequential GPU stage so TensorRT-LLM initialization is shared.


def _config(model: str = "org/model") -> SimpleNamespace:
    return SimpleNamespace(model=model, revision=None)


@pytest.mark.parametrize("model_type", sorted(_MM_ROUTING_MODEL_TYPES))
@pytest.mark.parametrize("model_path", [None, "org/model"])
def test_supported_family_uses_registry_marker(
    model_type: str, model_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm_worker, "_resolve_model_dir", lambda _: "/models/cached")
    monkeypatch.setattr(llm_worker, "resolve_routing_image_token_id", lambda *_: 163605)

    assert _resolve_image_token_id(model_type, _config(), model_path) == 163605


def test_registry_miss_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_worker, "resolve_routing_image_token_id", lambda *_: None)

    assert _resolve_image_token_id("qwen3_vl", _config()) is None


def test_selected_snapshot_preserves_model_id(tmp_path, monkeypatch) -> None:
    def fail(_):
        raise AssertionError("must not re-resolve the broken cache reference")

    def resolve(model_id, model_dir):
        assert model_id == "org/model"
        assert model_dir == str(tmp_path)
        return 163605

    monkeypatch.setattr(llm_worker, "_resolve_model_dir", fail)
    monkeypatch.setattr(llm_worker, "resolve_routing_image_token_id", resolve)

    assert _resolve_image_token_id("qwen3_vl", _config(), str(tmp_path)) == 163605


def test_missing_binding_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_worker, "resolve_routing_image_token_id", None)

    assert _resolve_image_token_id("kimi_k25", _config()) is None


@pytest.mark.parametrize("model_type", ["llava_next", "step3p7_vl", "unknown_vlm"])
def test_unvalidated_family_does_not_use_registry(
    model_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_: object) -> int:
        raise AssertionError("registry must not be called")

    monkeypatch.setattr(llm_worker, "resolve_routing_image_token_id", fail)

    assert _resolve_image_token_id(model_type, _config()) is None
