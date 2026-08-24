# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for multimodal encode-worker initialization."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import dynamo.vllm.multimodal_handlers.encode_worker_handler as handler_module
from dynamo.vllm.constants import EmbeddingTransferMode

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


@pytest.mark.asyncio
async def test_init_accepts_vision_model_without_out_hidden_size(monkeypatch):
    """A LLaVA-like vision model must not crash encode-worker initialization."""
    vision_model = SimpleNamespace(config=SimpleNamespace(hidden_size=4096))
    engine_args = SimpleNamespace(
        model="llava-hf/llava-1.5-7b-hf",
        enforce_eager=True,
        trust_remote_code=False,
    )
    monkeypatch.setattr(
        handler_module.AutoImageProcessor,
        "from_pretrained",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        handler_module, "load_vision_model", MagicMock(return_value=vision_model)
    )
    monkeypatch.setattr(
        handler_module,
        "get_encoder_components",
        MagicMock(return_value=(MagicMock(), MagicMock())),
    )
    monkeypatch.setattr(handler_module, "ENABLE_ENCODER_CACHE", False)

    assert not hasattr(vision_model, "out_hidden_size")

    handler = handler_module.EncodeWorkerHandler(
        engine_args, EmbeddingTransferMode.LOCAL
    )
    try:
        assert handler.vision_model is vision_model
    finally:
        handler.cleanup()
        await handler.send_complete_checker_task
