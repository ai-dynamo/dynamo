# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
from dataclasses import dataclass, field

import pytest

try:
    from dynamo.vllm.omni.args import OmniConfig  # noqa: F401
except Exception:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.engine_process import EngineConfig
from tests.utils.payloads import (
    AudioSpeechPayload,
    I2VPayload,
    ImageGenerationPayload,
    VideoGenerationPayload,
)

logger = logging.getLogger(__name__)

vllm_dir = os.environ.get("VLLM_DIR") or os.path.join(
    WORKSPACE_DIR, "examples/backends/vllm"
)


@dataclass
class VLLMOmniConfig(EngineConfig):
    """Configuration for vLLM-Omni test scenarios."""

    stragglers: list[str] = field(default_factory=lambda: ["VLLM:EngineCore"])


vllm_omni_configs = {
    "omni_image": VLLMOmniConfig(
        name="omni_image",
        directory=vllm_dir,
        script_name="xpu/agg_omni_image_xpu.sh",
        script_args=[
            "--vae-use-slicing",
            "--vae-use-tiling",
        ],
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.post_merge,
            pytest.mark.timeout(1200),
            pytest.mark.skip(
                reason="Qwen/Qwen-Image requires ~40GB GPU memory, exceeds CI capacity (22GB)"
            ),
        ],
        model="Qwen/Qwen-Image",
        request_payloads=[
            ImageGenerationPayload(
                body={
                    "prompt": "A red apple on a table",
                    "size": "512x512",
                    "num_inference_steps": 20,
                    "response_format": "url",
                },
                repeat_count=1,
                expected_response=[],
                expected_log=[],
            ),
        ],
    ),
    "omni_i2v": VLLMOmniConfig(
        name="omni_i2v",
        directory=vllm_dir,
        script_name="xpu/agg_omni_i2v_xpu.sh",
        script_args=[
            "--vae-use-slicing",
            "--vae-use-tiling",
            "--enforce-eager",
            "--enable-cpu-offload",
        ],
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.pre_merge,
            pytest.mark.timeout(1200),
        ],
        model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        request_payloads=[
            I2VPayload(
                body={
                    "prompt": "Make it dance",
                    "size": "320x192",
                    "response_format": "url",
                    "nvext": {
                        "num_inference_steps": 5,
                        "num_frames": 9,
                        "guidance_scale": 1.0,
                        "boundary_ratio": 0.875,
                        "guidance_scale_2": 1.0,
                        "seed": 42,
                    },
                },
                repeat_count=1,
                expected_response=[],
                expected_log=[],
            ),
        ],
    ),
    "omni_audio": VLLMOmniConfig(
        name="omni_audio",
        directory=vllm_dir,
        script_name="xpu/agg_omni_audio_xpu.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.pre_merge,
            pytest.mark.timeout(1200),
            pytest.mark.skip(
                reason=(
                    "vLLM-Omni Qwen3-TTS XPU code predictor still hard-codes "
                    "CUDA graph capture and fails at runtime"
                )
            ),
        ],
        model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        request_payloads=[
            AudioSpeechPayload(
                body={
                    "input": "Hello, this is a test of Dynamo audio generation.",
                    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "voice": "vivian",
                    "language": "English",
                },
                repeat_count=1,
                expected_response=[],
                expected_log=[],
            ),
        ],
    ),
    "omni_t2v": VLLMOmniConfig(
        name="omni_t2v",
        directory=vllm_dir,
        script_name="xpu/agg_omni_video_xpu.sh",
        script_args=[
            "--vae-use-slicing",
            "--vae-use-tiling",
            "--enforce-eager",
        ],
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.pre_merge,
            pytest.mark.timeout(1200),
        ],
        model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        request_payloads=[
            VideoGenerationPayload(
                body={
                    "prompt": "Dog running on a beach",
                    "size": "480x272",
                    "response_format": "url",
                    "nvext": {
                        "num_inference_steps": 10,
                        "num_frames": 17,
                    },
                },
                repeat_count=1,
                expected_response=[],
                expected_log=[],
            ),
            VideoGenerationPayload(
                body={
                    "prompt": "Dog running on a beach",
                    "size": "480x272",
                    "response_format": "url",
                    "nvext": {
                        "num_inference_steps": 10,
                        "num_frames": 17,
                    },
                },
                repeat_count=1,
                http_stream=True,
                expected_response=[],
                expected_log=[],
            ),
        ],
    ),
}


@pytest.fixture(params=params_with_model_mark(vllm_omni_configs))
def vllm_omni_config_test(request):
    """Fixture that provides different vLLM-Omni test configurations."""
    return vllm_omni_configs[request.param]


@pytest.mark.vllm
@pytest.mark.e2e
def test_omni_serve_deployment(
    vllm_omni_config_test,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_models,
):
    """Test dynamo serve deployments with vLLM-Omni configurations."""
    config = dataclasses.replace(
        vllm_omni_config_test, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)
