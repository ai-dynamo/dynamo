# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the built-in FastVideo backend."""

from __future__ import annotations

import base64
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


FASTVIDEO_AVAILABLE = _can_import("fastvideo.api")
DYNAMO_AVAILABLE = all(
    _can_import(module_name)
    for module_name in (
        "dynamo.fastvideo.args",
        "dynamo.fastvideo.backend",
        "dynamo.fastvideo.health_check",
    )
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.fastvideo,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.skipif(
        not (FASTVIDEO_AVAILABLE and DYNAMO_AVAILABLE),
        reason="fastvideo or dynamo runtime bindings are not installed",
    ),
]


class _FakeContext:
    def id(self) -> str:
        return "ctx-123"


class _FakeGenerator:
    def __init__(self) -> None:
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        assert request.prompt == "A test prompt"
        with open(request.output.output_path, "wb") as file:
            file.write(b"fake-mp4-bytes")
        return SimpleNamespace(
            video_path=request.output.output_path,
            generation_time=1.0,
        )


class _GeneratorShouldNotRun:
    def generate(self, request):
        raise AssertionError("generate should not be called")


def test_parse_fastvideo_args_uses_builtin_defaults():
    from dynamo.fastvideo.args import parse_fastvideo_args

    config = parse_fastvideo_args(["--model", "org/model"])

    assert config.model_path == "org/model"
    assert config.served_model_name == "org/model"
    assert config.num_gpus == 1
    assert config.attention_backend == "TORCH_SDPA"
    assert config.dit_cpu_offload is True
    assert config.dit_layerwise_offload is True
    assert config.use_fsdp_inference is False
    assert config.vae_cpu_offload is True
    assert config.image_encoder_cpu_offload is True
    assert config.text_encoder_cpu_offload is True
    assert config.pin_cpu_memory is True
    assert config.disable_autocast is False
    assert config.enable_torch_compile is False
    assert config.torch_compile_backend == "inductor"
    assert config.torch_compile_mode == "max-autotune-no-cudagraphs"
    assert config.torch_compile_fullgraph is True
    assert config.torch_compile_dynamic is False
    assert config.enable_fp4_quantization is False
    assert config.namespace == "dynamo"
    assert config.component == "backend"
    assert config.endpoint == "generate"
    assert config.output_modalities == ["video"]


def test_parse_fastvideo_args_applies_explicit_overrides():
    from dynamo.fastvideo.args import parse_fastvideo_args

    config = parse_fastvideo_args(
        [
            "--model-path",
            "org/model",
            "--served-model-name",
            "served",
            "--num-gpus",
            "2",
            "--attention-backend",
            "FLASH_ATTN",
            "--no-dit-cpu-offload",
            "--no-dit-layerwise-offload",
            "--use-fsdp-inference",
            "--no-vae-cpu-offload",
            "--no-image-encoder-cpu-offload",
            "--no-text-encoder-cpu-offload",
            "--no-pin-cpu-memory",
            "--disable-autocast",
            "--torch-compile",
            "--torch-compile-backend",
            "inductor",
            "--torch-compile-mode",
            "reduce-overhead",
            "--no-torch-compile-fullgraph",
            "--torch-compile-dynamic",
            "--fp4-quantization",
        ]
    )

    assert config.model_path == "org/model"
    assert config.served_model_name == "served"
    assert config.num_gpus == 2
    assert config.attention_backend == "FLASH_ATTN"
    assert config.dit_cpu_offload is False
    assert config.dit_layerwise_offload is False
    assert config.use_fsdp_inference is True
    assert config.vae_cpu_offload is False
    assert config.image_encoder_cpu_offload is False
    assert config.text_encoder_cpu_offload is False
    assert config.pin_cpu_memory is False
    assert config.disable_autocast is True
    assert config.enable_torch_compile is True
    assert config.torch_compile_mode == "reduce-overhead"
    assert config.torch_compile_fullgraph is False
    assert config.torch_compile_dynamic is True
    assert config.enable_fp4_quantization is True
    assert config.namespace == "dynamo"
    assert config.component == "backend"
    assert config.endpoint == "generate"
    assert config.output_modalities == ["video"]


def test_fastvideo_attention_backend_choices_are_hardcoded():
    from dynamo.fastvideo.args import get_attention_backend_choices

    assert get_attention_backend_choices() == (
        "FLASH_ATTN",
        "TORCH_SDPA",
        "SAGE_ATTN",
        "SAGE_ATTN_THREE",
        "VIDEO_SPARSE_ATTN",
        "VMOBA_ATTN",
        "SLA_ATTN",
        "SAGE_SLA_ATTN",
    )


def test_config_to_generator_config_mapping():
    from dynamo.fastvideo.args import parse_fastvideo_args
    from fastvideo.api import (
        CompileConfig,
        EngineConfig,
        GeneratorConfig,
        OffloadConfig,
        PipelineSelection,
        QuantizationConfig,
    )

    config = parse_fastvideo_args(
        [
            "--model",
            "org/model",
            "--num-gpus",
            "2",
            "--use-fsdp-inference",
            "--no-dit-cpu-offload",
            "--no-dit-layerwise-offload",
            "--no-text-encoder-cpu-offload",
            "--no-vae-cpu-offload",
            "--no-pin-cpu-memory",
            "--disable-autocast",
            "--torch-compile",
            "--torch-compile-mode",
            "max-autotune",
            "--fp4-quantization",
        ]
    )

    generator_config = config.to_generator_config()

    assert isinstance(generator_config, GeneratorConfig)
    assert isinstance(generator_config.engine, EngineConfig)
    assert isinstance(generator_config.engine.offload, OffloadConfig)
    assert isinstance(generator_config.engine.compile, CompileConfig)
    assert isinstance(generator_config.engine.quantization, QuantizationConfig)
    assert isinstance(generator_config.pipeline, PipelineSelection)
    assert generator_config.model_path == "org/model"
    assert generator_config.engine.num_gpus == 2
    assert generator_config.engine.use_fsdp_inference is True
    assert generator_config.engine.disable_autocast is True
    assert generator_config.engine.offload.dit is False
    assert generator_config.engine.offload.dit_layerwise is False
    assert generator_config.engine.offload.text_encoder is False
    assert generator_config.engine.offload.vae is False
    assert generator_config.engine.offload.pin_cpu_memory is False
    assert generator_config.engine.compile.enabled is True
    assert generator_config.engine.compile.backend == "inductor"
    assert generator_config.engine.compile.mode == "max-autotune"
    assert generator_config.engine.quantization.transformer_quant == "FP4"
    assert generator_config.pipeline.experimental["fp4_quantization"] is True


@pytest.mark.asyncio
async def test_register_fastvideo_model_uses_video_discovery_type():
    from dynamo.fastvideo.args import parse_fastvideo_args
    from dynamo.fastvideo.backend import register_fastvideo_model
    from dynamo.llm import ModelInput, ModelType

    config = parse_fastvideo_args(["--model", "org/model"])
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


def test_fastvideo_health_check_payload_uses_minimal_video_request():
    from dynamo.fastvideo.health_check import FastVideoHealthCheckPayload

    payload = FastVideoHealthCheckPayload("org/model").to_dict()

    assert payload["model"] == "org/model"
    assert payload["prompt"] == "test"
    assert payload["seconds"] == 1
    assert payload["size"] == "256x256"
    assert payload["response_format"] == "b64_json"
    assert payload["nvext"] == {
        "fps": 8,
        "num_frames": 8,
        "num_inference_steps": 1,
        "guidance_scale": 5.0,
    }


@pytest.mark.asyncio
async def test_fastvideo_handler_generates_minimal_video_response(tmp_path):
    from dynamo.fastvideo.args import parse_fastvideo_args
    from dynamo.fastvideo.backend import FastVideoHandler

    config = parse_fastvideo_args(
        [
            "--model",
            "org/model",
            "--media-output-fs-url",
            f"file://{tmp_path / 'media'}",
        ]
    )
    fake_generator = _FakeGenerator()
    handler = FastVideoHandler(config, generator=fake_generator)

    results = []
    async for result in handler.generate(
        {
            "prompt": "A test prompt",
            "model": "org/model",
            "size": "320x192",
            "response_format": "b64_json",
            "nvext": {
                "fps": 12,
                "num_frames": 8,
                "num_inference_steps": 1,
                "guidance_scale": 2.0,
                "seed": 42,
                "negative_prompt": "blur",
            },
        },
        _FakeContext(),
    ):
        results.append(result)

    assert len(results) == 1
    response = results[0]
    assert response["id"] == "video_ctx-123"
    assert response["object"] == "video"
    assert response["model"] == "org/model"
    assert response["status"] == "completed"
    assert response["progress"] == 100
    assert response["error"] is None
    assert response["data"] == [
        {
            "output_format": "mp4",
            "url": None,
            "b64_json": base64.b64encode(b"fake-mp4-bytes").decode("utf-8"),
        }
    ]

    request = fake_generator.requests[0]
    assert request.prompt == "A test prompt"
    assert request.negative_prompt == "blur"
    assert request.sampling.width == 320
    assert request.sampling.height == 192
    assert request.sampling.fps == 12
    assert request.sampling.num_frames == 8
    assert request.sampling.num_inference_steps == 1
    assert request.sampling.guidance_scale == 2.0
    assert request.sampling.seed == 42
    assert request.output.save_video is True
    assert request.output.return_frames is False


@pytest.mark.asyncio
async def test_fastvideo_handler_generates_url_video_response(tmp_path):
    from dynamo.fastvideo.args import parse_fastvideo_args
    from dynamo.fastvideo.backend import FastVideoHandler

    media_dir = tmp_path / "media"
    config = parse_fastvideo_args(
        [
            "--model",
            "org/model",
            "--media-output-fs-url",
            f"file://{media_dir}",
            "--media-output-http-url",
            "http://media.example",
        ]
    )
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
    assert response["data"] == [
        {
            "output_format": "mp4",
            "url": "http://media.example/videos/video_ctx-123.mp4",
            "b64_json": None,
        }
    ]
    assert (media_dir / "videos" / "video_ctx-123.mp4").read_bytes() == (
        b"fake-mp4-bytes"
    )


@pytest.mark.asyncio
async def test_fastvideo_handler_rejects_input_reference_requests():
    from dynamo.fastvideo.args import parse_fastvideo_args
    from dynamo.fastvideo.backend import FastVideoHandler

    config = parse_fastvideo_args(["--model", "org/model"])
    handler = FastVideoHandler(config, generator=_GeneratorShouldNotRun())

    results = []
    async for result in handler.generate(
        {
            "prompt": "A test prompt",
            "model": "org/model",
            "input_reference": "/tmp/reference.png",
        },
        _FakeContext(),
    ):
        results.append(result)

    assert len(results) == 1
    response = results[0]
    assert response["status"] == "failed"
    assert response["progress"] == 0
    assert response["data"] == []
    assert "input_reference" in response["error"]
