# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from dataclasses import dataclass, field

import pytest

from tests.serve.common import run_serve_deployment
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import (
    chat_payload,
    chat_payload_default,
    completion_payload_default,
    metric_payload_default,
)

logger = logging.getLogger(__name__)


@dataclass
class VLLMConfig(EngineConfig):
    """Configuration for vLLM test scenarios"""

    stragglers: list[str] = field(default_factory=lambda: ["VLLM:EngineCore"])


vllm_dir = os.environ.get("VLLM_DIR", "/workspace/components/backends/vllm")

# vLLM test configurations
vllm_configs = {
    "aggregated": VLLMConfig(
        name="aggregated",
        directory=vllm_dir,
        script_name="agg.sh",
        marks=[pytest.mark.gpu_1],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            metric_payload_default(min_num_requests=6),
        ],
    ),
    "agg-router": VLLMConfig(
        name="agg-router",
        directory=vllm_dir,
        script_name="agg_router.sh",
        marks=[pytest.mark.gpu_2],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload_default(
                expected_log=[
                    r"ZMQ listener .* received batch with \d+ events \(seq=\d+\)",
                    r"Event processor for worker_id \d+ processing event: Stored\(",
                    r"Selected worker: \d+, logit: ",
                ]
            )
        ],
        env={
            "DYN_LOG": "dynamo_llm::kv_router::publisher=trace,dynamo_llm::kv_router::scheduler=info",
        },
    ),
    "disaggregated": VLLMConfig(
        name="disaggregated",
        directory=vllm_dir,
        script_name="disagg.sh",
        marks=[pytest.mark.gpu_2],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "deepep": VLLMConfig(
        name="deepep",
        directory=vllm_dir,
        script_name="dsr1_dep.sh",
        marks=[
            pytest.mark.gpu_2,
            pytest.mark.vllm,
            pytest.mark.h100,
        ],
        model="deepseek-ai/DeepSeek-V2-Lite",
        script_args=[
            "--model",
            "deepseek-ai/DeepSeek-V2-Lite",
            "--num-nodes",
            "1",
            "--node-rank",
            "0",
            "--gpus-per-node",
            "2",
        ],
        timeout=700,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "multimodal_agg_llava": VLLMConfig(
        name="multimodal_agg_llava",
        directory="/workspace/examples/multimodal",
        script_name="agg.sh",
        marks=[pytest.mark.gpu_2],
        model="llava-hf/llava-1.5-7b-hf",
        script_args=["--model", "llava-hf/llava-1.5-7b-hf"],
        request_payloads=[
            chat_payload(
                [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "http://images.cocodataset.org/test2017/000000155781.jpg"
                        },
                    },
                ],
                repeat_count=1,
                expected_response=["bus"],
                temperature=0.0,
            )
        ],
    ),
    "multimodal_agg_qwen": VLLMConfig(
        name="multimodal_agg_qwen",
        directory="/workspace/examples/multimodal",
        script_name="agg.sh",
        marks=[pytest.mark.gpu_2],
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        delayed_start=0,
        script_args=["--model", "Qwen/Qwen2.5-VL-7B-Instruct"],
        timeout=360,
        request_payloads=[
            chat_payload(
                [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "http://images.cocodataset.org/test2017/000000155781.jpg"
                        },
                    },
                ],
                repeat_count=1,
                expected_response=["bus"],
            )
        ],
    ),
    "multimodal_video_agg": VLLMConfig(
        name="multimodal_video_agg",
        directory="/workspace/examples/multimodal",
        script_name="video_agg.sh",
        marks=[pytest.mark.gpu_2],
        model="llava-hf/LLaVA-NeXT-Video-7B-hf",
        delayed_start=0,
        script_args=["--model", "llava-hf/LLaVA-NeXT-Video-7B-hf"],
        timeout=360,
        request_payloads=[
            chat_payload(
                [
                    {"type": "text", "text": "Describe the video in detail"},
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": "https://storage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
                        },
                    },
                ],
                repeat_count=1,
                expected_response=["rabbit"],
            )
        ],
    ),
    # TODO: Enable this test case when we have 4 GPUs runners.
    # "multimodal_disagg": VLLMConfig(
    #     name="multimodal_disagg",
    #     directory="/workspace/examples/multimodal",
    #     script_name="disagg.sh",
    #     marks=[pytest.mark.gpu_4, pytest.mark.vllm],
    #     model="llava-hf/llava-1.5-7b-hf",
    #     delayed_start=45,
    #     script_args=["--model", "llava-hf/llava-1.5-7b-hf"],
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


@pytest.mark.vllm
@pytest.mark.e2e
def test_serve_deployment(vllm_config_test, request, runtime_services):
    """
    Test dynamo serve deployments with different graph configurations.
    """
    config = vllm_config_test
    run_serve_deployment(config, request)
