# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import pytest

from tests.serve.common import EngineConfig
from tests.serve.common import create_payload_for_config as base_create_payload
from tests.utils.deployment_graph import (
    Payload,
    chat_completions_response_handler,
    completions_response_handler,
)
from tests.utils.engine_process import EngineProcess

logger = logging.getLogger(__name__)


def create_payload_for_config(config: "VLLMConfig") -> Payload:
    """Create a payload using the model from the vLLM config"""
    if config.name in ["multimodal_agg_llava", "multimodal_agg_qwen"]:
        # Special handling for multimodal models
        return Payload(
            payload_chat={
                "model": config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "http://images.cocodataset.org/test2017/000000155781.jpg"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.0,
                "stream": False,
            },
            repeat_count=1,
            expected_log=[],
            expected_response=["bus"],
        )
    elif config.name == "multimodal_video_agg":
        # Special handling for multimodal models
        return Payload(
            payload_chat={
                "model": config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the video in detail"},
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": "https://storage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 300,
                "stream": False,
            },
            repeat_count=1,
            expected_log=[],
            expected_response=["rabbit"],
        )
    else:
        # Use base implementation for standard text models
        return base_create_payload(config)


@dataclass
class VLLMConfig(EngineConfig):
    """Configuration for vLLM test scenarios"""

    args: Optional[List[str]] = None


class VLLMProcess(EngineProcess):
    """Simple process manager for vllm shell scripts"""

    def __init__(self, config: VLLMConfig, request):
        self.port = 8080
        self.config = config
        self.dir = config.directory
        script_path = os.path.join(self.dir, "launch", config.script_name)

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"vLLM script not found: {script_path}")

        command = ["bash", script_path]
        if config.args:
            command.extend(config.args)

        super().__init__(
            command=command,
            timeout=config.timeout,
            display_output=True,
            working_dir=self.dir,
            health_check_ports=[],  # Disable port health check
            health_check_urls=[
                (f"http://localhost:{self.port}/v1/models", self._check_models_api)
            ],
            delayed_start=config.delayed_start,
            terminate_existing=False,  # If true, will call all bash processes including myself
            stragglers=[],  # Don't kill any stragglers automatically
            log_dir=request.node.name,
        )


# vLLM test configurations
vllm_configs = {
    "aggregated": VLLMConfig(
        name="aggregated",
        directory="/workspace/components/backends/vllm",
        script_name="agg.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        endpoints=["v1/chat/completions", "v1/completions"],
        response_handlers=[
            chat_completions_response_handler,
            completions_response_handler,
        ],
        model="Qwen/Qwen3-0.6B",
        delayed_start=0,
        timeout=360,
    ),
    "agg-router": VLLMConfig(
        name="agg-router",
        directory="/workspace/components/backends/vllm",
        script_name="agg_router.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.vllm],
        endpoints=["v1/chat/completions", "v1/completions"],
        response_handlers=[
            chat_completions_response_handler,
            completions_response_handler,
        ],
        model="Qwen/Qwen3-0.6B",
        delayed_start=0,
        timeout=360,
    ),
    "disaggregated": VLLMConfig(
        name="disaggregated",
        directory="/workspace/components/backends/vllm",
        script_name="disagg.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.vllm],
        endpoints=["v1/chat/completions", "v1/completions"],
        response_handlers=[
            chat_completions_response_handler,
            completions_response_handler,
        ],
        model="Qwen/Qwen3-0.6B",
        delayed_start=0,
        timeout=360,
    ),
    "deepep": VLLMConfig(
        name="deepep",
        directory="/workspace/components/backends/vllm",
        script_name="dsr1_dep.sh",
        marks=[
            pytest.mark.gpu_2,
            pytest.mark.vllm,
            pytest.mark.h100,
        ],
        endpoints=["v1/chat/completions", "v1/completions"],
        response_handlers=[
            chat_completions_response_handler,
            completions_response_handler,
        ],
        model="deepseek-ai/DeepSeek-V2-Lite",
        delayed_start=0,
        args=[
            "--model",
            "deepseek-ai/DeepSeek-V2-Lite",
            "--num-nodes",
            "1",
            "--node-rank",
            "0",
            "--gpus-per-node",
            "2",
        ],
        timeout=560,
    ),
    "multimodal_agg_llava": VLLMConfig(
        name="multimodal_agg_llava",
        directory="/workspace/examples/multimodal",
        script_name="agg.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.vllm],
        endpoints=["v1/chat/completions"],
        response_handlers=[
            chat_completions_response_handler,
        ],
        model="llava-hf/llava-1.5-7b-hf",
        delayed_start=0,
        args=["--model", "llava-hf/llava-1.5-7b-hf"],
        timeout=360,
    ),
    "multimodal_agg_qwen": VLLMConfig(
        name="multimodal_agg_qwen",
        directory="/workspace/examples/multimodal",
        script_name="agg.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.vllm],
        endpoints=["v1/chat/completions"],
        response_handlers=[
            chat_completions_response_handler,
        ],
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        delayed_start=0,
        args=["--model", "Qwen/Qwen2.5-VL-7B-Instruct"],
        timeout=360,
    ),
    "multimodal_video_agg": VLLMConfig(
        name="multimodal_video_agg",
        directory="/workspace/examples/multimodal",
        script_name="video_agg.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.vllm],
        endpoints=["v1/chat/completions"],
        response_handlers=[
            chat_completions_response_handler,
        ],
        model="llava-hf/LLaVA-NeXT-Video-7B-hf",
        delayed_start=0,
        args=["--model", "llava-hf/LLaVA-NeXT-Video-7B-hf"],
        timeout=360,
    ),
    # TODO: Enable this test case when we have 4 GPUs runners.
    # "multimodal_disagg": VLLMConfig(
    #     name="multimodal_disagg",
    #     directory="/workspace/examples/multimodal",
    #     script_name="disagg.sh",
    #     marks=[pytest.mark.gpu_4, pytest.mark.vllm],
    #     endpoints=["v1/chat/completions"],
    #     response_handlers=[
    #         chat_completions_response_handler,
    #     ],
    #     model="llava-hf/llava-1.5-7b-hf",
    #     delayed_start=45,
    #     args=["--model", "llava-hf/llava-1.5-7b-hf"],
    # ),
}


@pytest.fixture(
    params=[
        pytest.param(config_name, marks=config.marks)
        for config_name, config in vllm_configs.items()
    ]
)
def vllm_config_test(request):
    """Fixture that provides different vLLM test configurations"""
    return vllm_configs[request.param]


@pytest.mark.e2e
def test_serve_deployment(vllm_config_test, request, runtime_services):
    """
    Test dynamo serve deployments with different graph configurations.
    """

    # runtime_services is used to start nats and etcd

    logger = logging.getLogger(request.node.name)
    logger.info("Starting test_deployment")

    config = vllm_config_test
    payload = create_payload_for_config(config)

    logger.info("Using model: %s", config.model)
    logger.info("Script: %s", config.script_name)

    with VLLMProcess(config, request) as server_process:
        for endpoint, response_handler in zip(
            config.endpoints, config.response_handlers
        ):
            url = f"http://localhost:{server_process.port}/{endpoint}"
            start_time = time.time()
            elapsed = 0.0

            request_body = (
                payload.payload_chat
                if endpoint == "v1/chat/completions"
                else payload.payload_completions
            )

            for _ in range(payload.repeat_count):
                elapsed = time.time() - start_time

                response = server_process.send_request(
                    url, payload=request_body, timeout=config.timeout - elapsed
                )
                server_process.check_response(payload, response, response_handler)
