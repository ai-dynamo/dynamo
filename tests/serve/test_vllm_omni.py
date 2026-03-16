# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
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
from tests.utils.payload_builder import chat_payload
from tests.utils.payloads import BasePayload

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
        assert (
            "b64_json" in entry or "url" in entry
        ), f"Expected 'b64_json' or 'url' in data entry. Keys: {list(entry.keys())}"
        return entry.get("url") or "b64_image_returned"


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
        assert (
            "url" in entry or "b64_json" in entry
        ), f"Expected 'url' or 'b64_json' in data entry. Keys: {list(entry.keys())}"
        return entry.get("url") or "b64_video_returned"

    def validate(self, response: Any, content: str) -> None:
        if self.expected_response:
            for expected in self.expected_response:
                if expected.lower() in content.lower():
                    return
        assert content, "Video response content is empty"


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
            pytest.mark.pre_merge,
            pytest.mark.timeout(600),
        ],
        model="Qwen/Qwen2.5-Omni-7B",
        request_payloads=[
            chat_payload(
                "Say hello",
                repeat_count=1,
                expected_response=["hello", "Hello"],
                temperature=0.0,
                max_tokens=32,
            ),
        ],
    ),
    "omni_image": VLLMOmniConfig(
        name="omni_image",
        directory=vllm_dir,
        script_name="agg_omni_image.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.pre_merge,
            pytest.mark.timeout(600),
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
    "omni_video": VLLMOmniConfig(
        name="omni_video",
        directory=vllm_dir,
        script_name="agg_omni_video.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.pre_merge,
            pytest.mark.timeout(900),
        ],
        model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        request_payloads=[
            VideoGenerationPayload(
                body={
                    "prompt": "Dog running on a beach",
                    "size": "832x480",
                    "response_format": "url",
                    "nvext": {
                        "num_inference_steps": 20,
                        "num_frames": 30,
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
