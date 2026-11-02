# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for resolving NGC sources before vLLM engine configuration."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

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
    model = "ngc://example/team/model:1"
    fetch = AsyncMock(return_value=str(tmp_path))
    monkeypatch.setattr(vllm_args, "fetch_model", fetch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)

    config = await vllm_args.parse_args_with_model_fetch(
        ["--model", model, "--load-format", load_format]
    )

    fetch.assert_awaited_once_with(model, ignore_weights=True)
    assert config.model == model
    assert config.engine_args.model == str(tmp_path)
    assert config.served_model_name == model
    assert config.engine_args.served_model_name == [model]
    assert config.model_source_path == str(tmp_path)
    engine_config = SimpleNamespace(model_config=SimpleNamespace(model_weights=""))
    vllm_main = importlib.import_module("dynamo.vllm.main")
    assert vllm_main._register_model_source_path(config, engine_config) == model


@pytest.mark.parametrize("scheme", ["ngc", "s3", "gs", "az"])
def test_registration_source_across_local_caches(tmp_path, scheme):
    model = f"{scheme}://example/team/model:1"
    vllm_main = importlib.import_module("dynamo.vllm.main")
    for worker in ("prefill", "decode"):
        local_path = str(tmp_path / worker / "model")
        config = vllm_args.Config()
        config.model = model
        config.engine_args = SimpleNamespace(model=local_path)
        engine_config = SimpleNamespace(
            model_config=SimpleNamespace(
                model=local_path,
                model_weights="" if scheme == "ngc" else model,
            )
        )

        source = vllm_main._register_model_source_path(config, engine_config)

        assert source == (model if scheme == "ngc" else local_path)


@pytest.mark.asyncio
async def test_ngc_preserves_explicit_served_names(monkeypatch, tmp_path):
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
    model = "Qwen/Qwen3-0.6B"
    fetch = AsyncMock()
    monkeypatch.setattr(vllm_args, "fetch_model", fetch)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)

    config = await vllm_args.parse_args_with_model_fetch(["--model", model])

    fetch.assert_not_awaited()
    assert config.model == model
    assert config.engine_args.model == model
    assert config.model_source_path == model


@pytest.mark.parametrize("load_format", ["auto", "mx", "modelexpress"])
async def test_ngc_worker_fetches_weights_before_engine_setup(
    monkeypatch, tmp_path, load_format
):
    model = "ngc://example/team/model:1"
    metadata_path = tmp_path / "metadata"
    metadata_path.mkdir()
    weights_path = tmp_path / "weights"
    weights_path.mkdir()
    fetch = AsyncMock(side_effect=[str(metadata_path), str(weights_path)])
    vllm_main = importlib.import_module("dynamo.vllm.main")
    setup = AsyncMock(side_effect=RuntimeError("stop before engine setup"))
    monkeypatch.setattr(vllm_args, "fetch_model", fetch)
    monkeypatch.setattr(vllm_main, "fetch_model", fetch)
    monkeypatch.setattr(vllm_main, "prepare_snapshot_engine", setup)

    with pytest.raises(RuntimeError, match="stop before engine setup"):
        await vllm_main.worker(["--model", model, "--load-format", load_format])

    assert fetch.await_args_list == [call(model, ignore_weights=True), call(model)]
    config = setup.call_args.args[0]
    assert config.engine_args.model == str(weights_path)
    assert config.model == model


@pytest.mark.parametrize(
    "extra_args,snapshot,error",
    [
        (["--realtime", "--enable-lora"], False, "--enable-lora"),
        (
            ["--embedding-worker", "--embedding-worker-processes", "2"],
            True,
            "checkpoint mode",
        ),
        (["--enable-rl", "--logprobs-mode", "raw_logits"], False, "logprobs_mode"),
    ],
)
async def test_ngc_invalid_worker_config_does_not_fetch_weights(
    monkeypatch, tmp_path, extra_args, snapshot, error
):
    model = "ngc://example/team/model:1"
    fetch = AsyncMock(return_value=str(tmp_path))
    vllm_main = importlib.import_module("dynamo.vllm.main")
    monkeypatch.setattr(vllm_args, "fetch_model", fetch)
    monkeypatch.setattr(vllm_main, "fetch_model", fetch)
    if snapshot:
        monkeypatch.setenv("DYN_SNAPSHOT_CONTROL_DIR", str(tmp_path / "snapshot"))

    with pytest.raises(ValueError, match=error):
        await vllm_main.worker(["--model", model, *extra_args])

    fetch.assert_awaited_once_with(model, ignore_weights=True)
