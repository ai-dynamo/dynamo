# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the built-in FastVideo backend."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from dynamo.fastvideo.args import get_attention_backend_choices, parse_fastvideo_args
from dynamo.fastvideo.backend import FastVideoHandler, register_fastvideo_model
from dynamo.fastvideo.health_check import FastVideoHealthCheckPayload
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


class _GeneratorShouldNotRun:
    def generate_video(self, prompt: str, **kwargs) -> dict[str, float]:
        raise AssertionError("generate_video should not be called")


def test_parse_fastvideo_args_uses_builtin_defaults():
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
    assert config.torch_compile_mode == "max-autotune-no-cudagraphs"
    assert config.torch_compile_fullgraph is True
    assert config.enable_fp4_quantization is False
    assert config.extra_generator_args_file == ""
    assert config.override_generator_args_json == ""
    assert config.namespace == "dynamo"
    assert config.component == "backend"
    assert config.endpoint == "generate"
    assert config.output_modalities == ["video"]


def test_parse_fastvideo_args_applies_explicit_overrides():
    config = parse_fastvideo_args(
        [
            "--model",
            "org/model",
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
            "--torch-compile-mode",
            "max-autotune",
            "--no-torch-compile-fullgraph",
            "--fp4-quantization",
        ]
    )

    assert config.model_path == "org/model"
    assert config.served_model_name == "org/model"
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
    assert config.torch_compile_mode == "max-autotune"
    assert config.torch_compile_fullgraph is False
    assert config.enable_fp4_quantization is True
    assert config.namespace == "dynamo"
    assert config.component == "backend"
    assert config.endpoint == "generate"
    assert config.output_modalities == ["video"]


def test_fastvideo_attention_backend_choices_are_hardcoded():
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


def test_parse_fastvideo_args_keeps_generator_args_for_backend_validation(tmp_path):
    tmp_file = tmp_path / "nonexistent-generator-args.json"
    config = parse_fastvideo_args(
        [
            "--model",
            "org/model",
            "--extra-generator-args-file",
            str(tmp_file),
            "--override-generator-args-json",
            '["not", "an", "object"]',
        ]
    )

    assert config.extra_generator_args_file == str(tmp_file)
    assert config.override_generator_args_json == '["not", "an", "object"]'


@pytest.mark.asyncio
async def test_register_fastvideo_model_uses_video_discovery_type():
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


def test_parse_fastvideo_args_accepts_legacy_model_path_flag():
    config = parse_fastvideo_args(["--model-path", "org/model"])

    assert config.model_path == "org/model"
    assert config.served_model_name == "org/model"


@pytest.mark.asyncio
async def test_fastvideo_handler_generates_minimal_video_response():
    config = parse_fastvideo_args(["--model", "org/model"])
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
    context_id = _FakeContext().id()
    expected_url = f"{config.media_output_fs_url}/videos/{context_id}.mp4"
    assert response["data"] == [{"url": expected_url, "b64_json": None}]


@pytest.mark.asyncio
async def test_fastvideo_handler_rejects_input_reference_requests():
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
    assert (
        response["error"] == "FastVideo backend does not support input_reference "
        "(image-to-video) requests"
    )


def test_fastvideo_handler_builds_generator_kwargs_from_generator_args_file_and_generator_args_json(
    tmp_path,
):
    extra_generator_args_file = tmp_path / "generator_args.json"
    extra_generator_args_file.write_text(
        json.dumps(
            {
                "custom_option": 1,
                "ltx2_vae_tiling": True,
                "torch_compile_kwargs": {
                    "mode": "reduce-overhead",
                    "dynamic": True,
                },
                "nested": {"from_file": True},
            }
        ),
        encoding="utf-8",
    )

    config = parse_fastvideo_args(
        [
            "--model",
            "org/model",
            "--no-dit-cpu-offload",
            "--no-dit-layerwise-offload",
            "--no-vae-cpu-offload",
            "--no-text-encoder-cpu-offload",
            "--torch-compile",
            "--extra-generator-args-file",
            str(extra_generator_args_file),
            "--override-generator-args-json",
            '{"torch_compile_kwargs": {"fullgraph": false}, "custom_option": 3, "nested": {"from_override": true}}',
        ]
    )
    handler = FastVideoHandler(config, generator=_FakeGenerator())

    generator_kwargs = handler._build_generator_kwargs()

    assert generator_kwargs["dit_cpu_offload"] is False
    assert generator_kwargs["dit_layerwise_offload"] is False
    assert generator_kwargs["use_fsdp_inference"] is False
    assert generator_kwargs["vae_cpu_offload"] is False
    assert generator_kwargs["image_encoder_cpu_offload"] is True
    assert generator_kwargs["text_encoder_cpu_offload"] is False
    assert generator_kwargs["pin_cpu_memory"] is True
    assert generator_kwargs["disable_autocast"] is False
    assert generator_kwargs["enable_torch_compile"] is True
    assert generator_kwargs["custom_option"] == 3
    assert generator_kwargs["ltx2_vae_tiling"] is True
    assert generator_kwargs["nested"] == {
        "from_file": True,
        "from_override": True,
    }
    assert generator_kwargs["torch_compile_kwargs"] == {
        "backend": "inductor",
        "fullgraph": False,
        "mode": "reduce-overhead",
        "dynamic": True,
    }


def test_fastvideo_handler_rejects_non_object_override_generator_args_json():
    config = parse_fastvideo_args(
        [
            "--model",
            "org/model",
            "--override-generator-args-json",
            '["not", "an", "object"]',
        ]
    )
    handler = FastVideoHandler(config, generator=_FakeGenerator())

    with pytest.raises(
        ValueError,
        match="--override-generator-args-json must decode to a JSON object",
    ):
        handler._build_generator_kwargs()


def test_fastvideo_handler_rejects_missing_generator_args_file(tmp_path):
    config = parse_fastvideo_args(
        [
            "--model",
            "org/model",
            "--extra-generator-args-file",
            str(tmp_path / "definitely-missing-generator-args.yaml"),
        ]
    )
    handler = FastVideoHandler(config, generator=_FakeGenerator())

    with pytest.raises(FileNotFoundError):
        handler._build_generator_kwargs()
