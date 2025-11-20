# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from dataclasses import dataclass, field

import pytest

from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
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


vllm_dir = os.environ.get("VLLM_DIR") or os.path.join(
    WORKSPACE_DIR, "examples/backends/vllm"
)

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
            metric_payload_default(min_num_requests=6, backend="vllm"),
        ],
    ),
    "aggregated_lmcache": VLLMConfig(
        name="aggregated_lmcache",
        directory=vllm_dir,
        script_name="agg_lmcache.sh",
        marks=[pytest.mark.gpu_1],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            metric_payload_default(min_num_requests=6, backend="vllm"),
            metric_payload_default(min_num_requests=6, backend="lmcache"),
        ],
    ),
    "agg-request-plane-tcp": VLLMConfig(
        name="agg-request-plane-tcp",
        directory=vllm_dir,
        script_name="agg_request_planes.sh",
        marks=[pytest.mark.gpu_1],
        model="Qwen/Qwen3-0.6B",
        script_args=["--tcp"],
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "agg-request-plane-http": VLLMConfig(
        name="agg-request-plane-http",
        directory=vllm_dir,
        script_name="agg_request_planes.sh",
        marks=[pytest.mark.gpu_1],
        model="Qwen/Qwen3-0.6B",
        script_args=["--http"],
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
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
                    r"ZMQ listener .* received batch with \d+ events \(seq=\d+(?:, [^)]*)?\)",
                    r"Event processor for worker_id \d+ processing event: Stored\(",
                    r"Selected worker: worker_id=\d+ dp_rank=.*?, logit: ",
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
            chat_payload_default(expected_response=["joke"]),
            completion_payload_default(expected_response=["joke"]),
        ],
    ),
    "multimodal_agg_llava_epd": VLLMConfig(
        name="multimodal_agg_llava_epd",
        directory=vllm_dir,
        script_name="agg_multimodal_epd.sh",
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
    "multimodal_agg_qwen_epd": VLLMConfig(
        name="multimodal_agg_qwen_epd",
        directory=vllm_dir,
        script_name="agg_multimodal_epd.sh",
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
    "multimodal_agg_qwen": VLLMConfig(
        name="multimodal_agg_qwen",
        directory=vllm_dir,
        script_name="agg_multimodal.sh",
        marks=[pytest.mark.gpu_2],
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        script_args=["--model", "Qwen/Qwen2.5-VL-7B-Instruct"],
        delayed_start=0,
        timeout=360,
        request_payloads=[
            # HTTP URL test
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
            ),
            # Base64 data URL test (1x1 PNG inline, avoids network fetch)
            chat_payload(
                [
                    {"type": "text", "text": "What do you see in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNoAAAAggCBd81ytgAAAABJRU5ErkJggg=="
                        },
                    },
                ],
                repeat_count=1,
                expected_response=[],  # Just validate no error
            ),
        ],
    ),
    "multimodal_agg_llava": VLLMConfig(
        name="multimodal_agg_llava",
        directory=vllm_dir,
        script_name="agg_multimodal.sh",
        marks=[
            pytest.mark.gpu_2,
            # https://github.com/ai-dynamo/dynamo/issues/4501
            pytest.mark.xfail(strict=False),
        ],
        model="llava-hf/llava-1.5-7b-hf",
        script_args=["--model", "llava-hf/llava-1.5-7b-hf"],
        delayed_start=0,
        timeout=360,
        request_payloads=[
            # HTTP URL test
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
            ),
            # String content test - verifies string → array conversion for multimodal templates
            chat_payload_default(
                repeat_count=1,
                expected_response=[],  # Just validate no error
            ),
        ],
    ),
    # TODO: Update this test case when we have video multimodal support in vllm official components
    "multimodal_video_agg": VLLMConfig(
        name="multimodal_video_agg",
        directory=os.path.join(WORKSPACE_DIR, "examples/multimodal"),
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
                temperature=0.7,
            )
        ],
    ),
    "multimodal_audio_agg": VLLMConfig(
        name="multimodal_audio_agg",
        directory="/workspace/examples/multimodal",
        script_name="audio_agg.sh",
        marks=[pytest.mark.gpu_2],
        model="Qwen/Qwen2-Audio-7B-Instruct",
        delayed_start=0,
        script_args=["--model", "Qwen/Qwen2-Audio-7B-Instruct"],
        timeout=500,
        request_payloads=[
            chat_payload(
                [
                    {"type": "text", "text": "What is recited in the audio?"},
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": "https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav"
                        },
                    },
                ],
                repeat_count=1,
                expected_response=[
                    "The original content of this audio is:'yet these thoughts affected Hester Pynne less with hope than apprehension.'"
                ],
                temperature=0.8,
            )
        ],
    ),
    # TODO: Enable this test case when we have 4 GPUs runners.
    # "multimodal_disagg": VLLMConfig(
    #     name="multimodal_disagg",
    #     directory=os.path.join(WORKSPACE_DIR, "examples/multimodal"),
    #     script_name="disagg.sh",
    #     marks=[pytest.mark.gpu_4, pytest.mark.vllm],
    #     model="llava-hf/llava-1.5-7b-hf",
    #     delayed_start=45,
    #     script_args=["--model", "llava-hf/llava-1.5-7b-hf"],
    # ),
}


@pytest.fixture(params=params_with_model_mark(vllm_configs))
def vllm_config_test(request):
    """Fixture that provides different vLLM test configurations"""
    return vllm_configs[request.param]


@pytest.mark.vllm
@pytest.mark.e2e
def test_serve_deployment(
    vllm_config_test, request, runtime_services, predownload_models
):
    """
    Test dynamo serve deployments with different graph configurations.
    """
    config = vllm_config_test
    run_serve_deployment(config, request)
