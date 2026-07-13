# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("vllm.sampling_params")

from dynamo.common.constants import DisaggregationMode  # noqa: E402
from dynamo.vllm import encode_engine as encode_engine_mod  # noqa: E402
from dynamo.vllm.encode_engine import VllmEncodeEngine  # noqa: E402
from dynamo.vllm.multimodal_handlers import EncodeWorkerHandler  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.unified,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _Context:
    def id(self):
        return "encode-request"


class _Handler:
    def __init__(self):
        self.requests = []

    async def generate(self, request, context):
        self.requests.append((request, context))
        yield json.dumps({"multimodal_inputs": []})


@pytest.mark.asyncio
async def test_encode_engine_emits_versioned_terminal():
    engine = VllmEncodeEngine(SimpleNamespace())
    handler = _Handler()
    engine._handler = handler

    chunks = [
        chunk
        async for chunk in engine.generate(
            {
                "token_ids": [1],
                "multi_modal_data": {
                    "image_url": [{"Url": "https://example.com/image.png"}]
                },
                "mm_processor_kwargs": {
                    "min_pixels": 3136,
                    "max_pixels": 1003520,
                },
            },
            _Context(),
        )
    ]

    assert chunks == [
        {
            "token_ids": [],
            "index": 0,
            "finish_reason": "stop",
            "encoder_result": {
                "schema_version": 1,
                "multimodal_inputs": [],
            },
        }
    ]
    assert handler.requests[0][0].multimodal_inputs[0].multimodal_input.image_url == (
        "https://example.com/image.png"
    )
    assert handler.requests[0][0].mm_processor_kwargs == {
        "min_pixels": 3136,
        "max_pixels": 1003520,
    }


def test_encode_worker_merges_and_filters_processor_kwargs():
    class _ImageProcessor:
        def __call__(
            self,
            images,
            return_tensors=None,
            *,
            min_pixels=None,
            max_pixels=None,
        ):
            del images, return_tensors, min_pixels, max_pixels

    handler = object.__new__(EncodeWorkerHandler)
    handler.engine_args = SimpleNamespace(
        mm_processor_kwargs={
            "min_pixels": 3136,
            "max_pixels": 1003520,
            "unsupported": True,
        }
    )
    handler.image_processor = _ImageProcessor()

    actual = handler._effective_mm_processor_kwargs(
        {"max_pixels": 501760, "unsupported": False}
    )

    assert actual == {"min_pixels": 3136, "max_pixels": 501760}


def test_encode_worker_cache_key_includes_processor_kwargs():
    image_url = "https://example.com/image.png"

    first = EncodeWorkerHandler._embedding_cache_key(
        image_url, {"min_pixels": 3136, "max_pixels": 1003520}
    )
    reordered = EncodeWorkerHandler._embedding_cache_key(
        image_url, {"max_pixels": 1003520, "min_pixels": 3136}
    )
    resized = EncodeWorkerHandler._embedding_cache_key(
        image_url, {"min_pixels": 3136, "max_pixels": 501760}
    )

    assert first == reordered
    assert first != resized


@pytest.mark.asyncio
async def test_encode_engine_rejects_decoded_images():
    engine = VllmEncodeEngine(SimpleNamespace())
    engine._handler = _Handler()
    with pytest.raises(ValueError, match="URL-based images only"):
        _ = [
            chunk
            async for chunk in engine.generate(
                {
                    "token_ids": [1],
                    "multi_modal_data": {
                        "image_url": [{"Decoded": {"shape": [1, 1, 3]}}]
                    },
                },
                _Context(),
            )
        ]


@pytest.mark.asyncio
async def test_encode_engine_from_args_builds_hidden_worker_config(monkeypatch):
    config = SimpleNamespace(
        disaggregation_mode=DisaggregationMode.ENCODE,
        enable_multimodal=True,
        route_to_encoder=False,
        served_model_name="model",
        model="model",
        engine_args=SimpleNamespace(served_model_name=["model"]),
    )
    worker_config = object()
    from_runtime_config = MagicMock(return_value=worker_config)
    monkeypatch.setattr(
        encode_engine_mod.WorkerConfig, "from_runtime_config", from_runtime_config
    )

    engine, actual_worker_config = await VllmEncodeEngine.from_args([], config)

    assert actual_worker_config is worker_config
    assert engine._config is config
    assert from_runtime_config.call_args.kwargs["enable_kv_routing"] is False


@pytest.mark.asyncio
async def test_encode_engine_from_args_normalizes_served_name_fallback(monkeypatch):
    config = SimpleNamespace(
        disaggregation_mode=DisaggregationMode.ENCODE,
        enable_multimodal=True,
        route_to_encoder=False,
        served_model_name=None,
        model="model",
        engine_args=SimpleNamespace(served_model_name=None),
    )
    from_runtime_config = MagicMock(return_value=object())
    monkeypatch.setattr(
        encode_engine_mod.WorkerConfig, "from_runtime_config", from_runtime_config
    )

    await VllmEncodeEngine.from_args([], config)

    assert config.served_model_name == "model"
    assert config.engine_args.served_model_name == ["model"]
    assert from_runtime_config.call_args.kwargs["served_model_name"] == "model"


@pytest.mark.asyncio
async def test_encode_engine_cleanup_is_idempotent():
    engine = VllmEncodeEngine(SimpleNamespace())
    handler = SimpleNamespace(
        cleanup=MagicMock(),
        send_complete_checker_task=AsyncMock(return_value=None)(),
    )
    engine._handler = handler

    await engine.cleanup()
    await engine.cleanup()

    handler.cleanup.assert_called_once_with()
