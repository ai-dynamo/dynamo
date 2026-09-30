# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for resolving NGC sources before vLLM engine configuration."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import huggingface_hub.constants
import pytest

from dynamo.vllm import args as vllm_args

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.usefixtures("vllm_cpu_platform_when_no_accelerator"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("load_format", ["auto", "mx", "modelexpress"])
async def test_ngc_is_resolved_before_offline_engine_args(
    monkeypatch, tmp_path, load_format
):
    """Resolve NGC sources offline for native and ModelExpress weight loaders."""
    model = "ngc://example/team/model:1"
    fetch = AsyncMock(return_value=str(tmp_path))
    monkeypatch.setattr(vllm_args, "fetch_model", fetch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)

    config = await vllm_args.parse_args_with_model_fetch(
        ["--model", model, "--load-format", load_format]
    )

    fetch.assert_awaited_once_with(model)
    assert config.model == model
    assert config.engine_args.model == str(tmp_path)
    assert config.served_model_name == model
    assert config.engine_args.served_model_name == [model]
    assert config.model_source_path == str(tmp_path)
    engine_config = SimpleNamespace(model_config=SimpleNamespace(model_weights=""))
    vllm_main = importlib.import_module("dynamo.vllm.main")
    assert vllm_main._register_model_source_path(config, engine_config) == str(tmp_path)


@pytest.mark.asyncio
async def test_ngc_preserves_explicit_served_names(monkeypatch, tmp_path):
    """Keep the requested public model name and aliases after resolving NGC."""
    monkeypatch.setattr(vllm_args, "fetch_model", AsyncMock(return_value=str(tmp_path)))

    config = await vllm_args.parse_args_with_model_fetch(
        [
            "--model",
            "ngc://example/team/model:1",
            "--served-model-name",
            "public-model",
            "model-alias",
        ]
    )

    assert config.served_model_name == "public-model"
    assert config.served_model_aliases == ["model-alias"]
    assert config.engine_args.served_model_name == ["public-model", "model-alias"]


@pytest.mark.asyncio
async def test_hf_model_keeps_engine_and_registration_source(monkeypatch):
    """Leave HF acquisition to the existing path and retain the repository ID."""
    model = "Qwen/Qwen3-0.6B"
    fetch = AsyncMock()
    monkeypatch.setattr(vllm_args, "fetch_model", fetch)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)

    config = await vllm_args.parse_args_with_model_fetch(["--model", model])

    fetch.assert_not_awaited()
    assert config.model == model
    assert config.engine_args.model == model
    assert config.model_source_path == model
