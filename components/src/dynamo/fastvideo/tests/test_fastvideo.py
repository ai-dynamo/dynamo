# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the built-in FastVideo backend."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, patch

import pytest

from dynamo.fastvideo.args import parse_fastvideo_args
from dynamo.fastvideo.backend import FastVideoHandler, register_fastvideo_model
from dynamo.llm import ModelInput, ModelType

pytestmark = [
    pytest.mark.unit,
    pytest.mark.fastvideo,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
]


class _FakeContext:
    def id(self) -> str:
        return "ctx-123"


class _FakeGenerator:
    def generate_video(self, prompt: str, **kwargs) -> dict[str, float]:
        assert prompt == "A test prompt"
        with open(kwargs["output_path"], "wb") as file:
            file.write(b"fake-mp4-bytes")
        return {"generation_time": 1.0, "e2e_latency": 1.5}


def test_parse_fastvideo_args_uses_builtin_defaults():
    config = parse_fastvideo_args(
        [
            "--model-path",
            "org/model",
            "--num-gpus",
            "2",
            "--optimization-profile",
            "latency",
            "--attention-backend",
            "FLASH_ATTN",
        ]
    )

    assert config.model_path == "org/model"
    assert config.served_model_name == "org/model"
    assert config.num_gpus == 2
    assert config.optimization_profile == "latency"
    assert config.attention_backend == "FLASH_ATTN"
    assert config.namespace == "dynamo"
    assert config.component == "backend"
    assert config.endpoint == "generate"
    assert config.output_modalities == ["video"]


@pytest.mark.asyncio
async def test_register_fastvideo_model_uses_video_discovery_type():
    config = parse_fastvideo_args(["--model-path", "org/model"])
    endpoint = object()

    with patch(
        "dynamo.fastvideo.backend.register_model",
        new=AsyncMock(),
    ) as register_model_mock:
        await register_fastvideo_model(endpoint, config)

    register_model_mock.assert_awaited_once_with(
        ModelInput.Text,
        ModelType.Videos,
        endpoint,
        "org/model",
        "org/model",
    )


@pytest.mark.asyncio
async def test_fastvideo_handler_generates_minimal_video_response():
    config = parse_fastvideo_args(["--model-path", "org/model"])
    handler = FastVideoHandler(config, generator=_FakeGenerator())

    results = []
    async for result in handler.generate(
        {
            "prompt": "A test prompt",
            "model": "org/model",
        },
        _FakeContext(),
    ):
        results.append(result)

    assert len(results) == 1
    response = results[0]
    assert response["id"] == "video_ctx-123"
    assert response["model"] == "org/model"
    assert response["status"] == "completed"
    assert response["progress"] == 100
    assert response["error"] is None
    assert response["data"] == [
        {"url": None, "b64_json": base64.b64encode(b"fake-mp4-bytes").decode("utf-8")}
    ]
