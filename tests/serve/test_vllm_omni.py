# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

import pytest

try:
    from dynamo.vllm.omni.args import OmniConfig  # noqa: F401
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.engine_process import EngineConfig
from tests.utils.payloads import BasePayload, ChatPayload

logger = logging.getLogger(__name__)

vllm_dir = os.environ.get("VLLM_DIR") or os.path.join(
    WORKSPACE_DIR, "examples/backends/vllm"
)


@dataclass
class ImageGenerationPayload(BasePayload):
    """Payload for /v1/images/generations endpoint."""

    endpoint: str = "/v1/images/generations"
    timeout: int = 300

    def response_handler(self, response: Any) -> str:
        response.raise_for_status()
        result = response.json()
        assert (
            "data" in result
        ), f"Missing 'data' in response. Keys: {list(result.keys())}"
        assert len(result["data"]) > 0, "Empty data in image response"
        entry = result["data"][0]
        if "url" in entry:
            assert entry["url"], "Image response url is empty"
            return entry["url"]
        assert entry.get("b64_json"), "Image response b64_json is empty"
        return "b64_image_returned"


@dataclass
class VideoGenerationPayload(BasePayload):
    """Payload for /v1/videos endpoint."""

    endpoint: str = "/v1/videos"
    timeout: int = 600

    def response_handler(self, response: Any) -> str:
        response.raise_for_status()
        result = response.json()
        assert result.get("status") == "completed", (
            f"Video generation not completed. Status: {result.get('status')}, "
            f"Error: {result.get('error', 'none')}"
        )
        assert (
            "data" in result
        ), f"Missing 'data' in response. Keys: {list(result.keys())}"
        assert len(result["data"]) > 0, "Empty data in video response"
        entry = result["data"][0]
        if "url" in entry:
            assert entry["url"], "Video response url is empty"
            return entry["url"]
        assert entry.get("b64_json"), "Video response b64_json is empty"
        return "b64_video_returned"

    def validate(self, response: Any, content: str) -> None:
        assert content, "Video response content is empty"
        if self.expected_response and not any(
            expected.lower() in content.lower() for expected in self.expected_response
        ):
            raise AssertionError(
                f"Expected at least one of {self.expected_response} in {content!r}"
            )


@dataclass
class I2VPayload(VideoGenerationPayload):
    """Payload for image-to-video via /v1/videos with input_reference."""

    _tmp_dir: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        from PIL import Image

        self._tmp_dir = tempfile.TemporaryDirectory()
        path = os.path.join(self._tmp_dir.name, "input.png")
        Image.new("RGB", (64, 64), color="red").save(path)
        self.body["input_reference"] = path


@dataclass
class VLLMOmniConfig(EngineConfig):
    """Configuration for vLLM-Omni test scenarios."""

    stragglers: list[str] = field(default_factory=lambda: ["VLLM:EngineCore"])


vllm_omni_configs = {
    "omni_text": VLLMOmniConfig(
        name="omni_text",
        directory=vllm_dir,
        script_name="agg_omni.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.post_merge,
            pytest.mark.timeout(1200),
            pytest.mark.skip(
                reason="Qwen2.5-Omni-7B requires ~80GB GPU memory, exceeds CI capacity (22GB)"
            ),
        ],
        model="Qwen/Qwen2.5-Omni-7B",
        request_payloads=[
            ChatPayload(
                body={
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "max_tokens": 32,
                    "temperature": 0.0,
                },
                repeat_count=1,
                expected_response=["hello", "Hello"],
                expected_log=[],
            ),
        ],
    ),
    "omni_image": VLLMOmniConfig(
        name="omni_image",
        directory=vllm_dir,
        script_name="agg_omni_image.sh",
        script_args=[
            "--vae-use-slicing",
            "--vae-use-tiling",
            "--enforce-eager",
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
        script_name="agg_omni_i2v.sh",
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
    "omni_t2v": VLLMOmniConfig(
        name="omni_t2v",
        directory=vllm_dir,
        script_name="agg_omni_video.sh",
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
