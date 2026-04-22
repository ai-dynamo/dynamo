# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.llm import ModelInput, ModelType
from dynamo.sglang.args import parse_args
from dynamo.sglang.register import (
    register_image_diffusion_model,
    register_video_generation_model,
)
from dynamo.sglang.tests.conftest import make_cli_args_fixture

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_1,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]

mock_sglang_cli = make_cli_args_fixture("dynamo.sglang")


@pytest.mark.asyncio
async def test_register_image_diffusion_model_uses_served_model_name(monkeypatch):
    register_model_mock = AsyncMock()
    monkeypatch.setattr("dynamo.sglang.register.register_model", register_model_mock)

    endpoint = object()
    readiness_gate = asyncio.Event()
    server_args = SimpleNamespace(
        model_path="black-forest-labs/FLUX.1-dev",
        served_model_name="flux-alias",
    )

    await register_image_diffusion_model(
        object(),
        endpoint,
        server_args,
        readiness_gate=readiness_gate,
    )

    register_model_mock.assert_awaited_once_with(
        ModelInput.Text,
        ModelType.Images,
        endpoint,
        "black-forest-labs/FLUX.1-dev",
        "flux-alias",
    )
    assert readiness_gate.is_set()


@pytest.mark.asyncio
async def test_register_video_generation_model_falls_back_to_model_path(monkeypatch):
    register_model_mock = AsyncMock()
    monkeypatch.setattr("dynamo.sglang.register.register_model", register_model_mock)

    endpoint = object()
    readiness_gate = asyncio.Event()
    server_args = SimpleNamespace(
        model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        served_model_name=None,
    )

    await register_video_generation_model(
        object(),
        endpoint,
        server_args,
        readiness_gate=readiness_gate,
    )

    register_model_mock.assert_awaited_once_with(
        ModelInput.Text,
        ModelType.Videos,
        endpoint,
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    )
    assert readiness_gate.is_set()


@pytest.mark.asyncio
async def test_register_image_diffusion_model_falls_back_to_model_path(monkeypatch):
    register_model_mock = AsyncMock()
    monkeypatch.setattr("dynamo.sglang.register.register_model", register_model_mock)

    endpoint = object()
    readiness_gate = asyncio.Event()
    server_args = SimpleNamespace(
        model_path="black-forest-labs/FLUX.1-dev",
        served_model_name="",
    )

    await register_image_diffusion_model(
        object(),
        endpoint,
        server_args,
        readiness_gate=readiness_gate,
    )

    register_model_mock.assert_awaited_once_with(
        ModelInput.Text,
        ModelType.Images,
        endpoint,
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-dev",
    )
    assert readiness_gate.is_set()


@pytest.mark.asyncio
async def test_register_video_generation_model_uses_served_model_name(monkeypatch):
    register_model_mock = AsyncMock()
    monkeypatch.setattr("dynamo.sglang.register.register_model", register_model_mock)

    endpoint = object()
    readiness_gate = asyncio.Event()
    server_args = SimpleNamespace(
        model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        served_model_name="my-video-model",
    )

    await register_video_generation_model(
        object(),
        endpoint,
        server_args,
        readiness_gate=readiness_gate,
    )

    register_model_mock.assert_awaited_once_with(
        ModelInput.Text,
        ModelType.Videos,
        endpoint,
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "my-video-model",
    )
    assert readiness_gate.is_set()


@pytest.mark.asyncio
async def test_parse_args_video_worker_preserves_custom_served_model_name(
    tmp_path, mock_sglang_cli
):
    model_dir = tmp_path / "video-model"
    model_dir.mkdir()

    mock_sglang_cli(
        "--model",
        str(model_dir),
        "--served-model-name",
        "my-video-model",
        "--video-generation-worker",
    )

    config = await parse_args(sys.argv[1:])

    assert config.server_args.model_path == str(model_dir)
    assert config.server_args.served_model_name == "my-video-model"


@pytest.mark.asyncio
async def test_parse_args_video_worker_defaults_served_model_name_to_model_path(
    tmp_path, mock_sglang_cli
):
    model_dir = tmp_path / "video-model"
    model_dir.mkdir()

    mock_sglang_cli(
        "--model",
        str(model_dir),
        "--video-generation-worker",
    )

    config = await parse_args(sys.argv[1:])

    assert config.server_args.model_path == str(model_dir)
    assert config.server_args.served_model_name == str(model_dir)
