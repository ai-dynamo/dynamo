# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
import os
import random
from dataclasses import dataclass, field

import pytest

from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.serve.conftest import MULTIMODAL_IMG_PATH, MULTIMODAL_IMG_URL
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import (
    chat_payload,
    chat_payload_default,
    chat_payload_with_logprobs,
    completion_payload_default,
    completion_payload_with_logprobs,
    metric_payload_default,
)
from tests.utils.payloads import ToolCallingChatPayload

logger = logging.getLogger(__name__)


@dataclass
class VLLMConfig(EngineConfig):
    """Configuration for vLLM test scenarios"""

    stragglers: list[str] = field(default_factory=lambda: ["VLLM:EngineCore"])


vllm_dir = os.environ.get("VLLM_DIR") or os.path.join(
    WORKSPACE_DIR, "examples/backends/vllm"
)


# vLLM test configurations
# NOTE: pytest.mark.gpu_1 tests take ~5.5 minutes total to run sequentially (with models pre-cached)
# TODO: Parallelize these tests to reduce total execution time
vllm_configs = {
    "aggregated": VLLMConfig(
        name="aggregated",
        directory=vllm_dir,
        script_name="agg.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.pre_merge,
            pytest.mark.timeout(300),  # 3x measured time (43s) + download time (150s)
        ],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            metric_payload_default(min_num_requests=6, backend="vllm"),
        ],
    ),
    "aggregated_logprobs": VLLMConfig(
        name="aggregated_logprobs",
        directory=vllm_dir,
        script_name="agg.sh",
        marks=[pytest.mark.gpu_1],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload_with_logprobs(
                repeat_count=2,
                expected_response=["AI", "knock", "joke"],
                max_tokens=30,
                temperature=0.0,
                top_logprobs=3,
            ),
            completion_payload_with_logprobs(
                repeat_count=2,
                expected_response=["AI", "knock", "joke"],
                max_tokens=30,
                temperature=0.0,
                logprobs=5,
            ),
        ],
    ),
    "aggregated_lmcache": VLLMConfig(
        name="aggregated_lmcache",
        directory=vllm_dir,
        script_name="agg_lmcache.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.pre_merge,
            pytest.mark.timeout(360),  # 3x estimated time (70s) + download time (150s)
        ],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            metric_payload_default(min_num_requests=6, backend="vllm"),
            metric_payload_default(min_num_requests=6, backend="lmcache"),
        ],
    ),
    "aggregated_lmcache_multiproc": VLLMConfig(
        name="aggregated_lmcache_multiproc",
        directory=vllm_dir,
        script_name="agg_lmcache_multiproc.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.timeout(360),  # 3x estimated time (70s) + download time (150s)
        ],
        model="Qwen/Qwen3-0.6B",
        env={
            "PROMETHEUS_MULTIPROC_DIR": f"/tmp/prometheus_multiproc_test_{os.getpid()}_{random.randint(0, 10000)}"
        },
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
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.pre_merge,
            pytest.mark.timeout(300),  # 3x measured time (43s) + download time (150s)
        ],
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
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.pre_merge,
            pytest.mark.timeout(300),  # 3x measured time (43s) + download time (150s)
        ],
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
        marks=[pytest.mark.gpu_2, pytest.mark.post_merge],
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
        marks=[pytest.mark.gpu_2, pytest.mark.post_merge],
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
            pytest.mark.nightly,
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
    "multimodal_agg_llava_epd": VLLMConfig(
        name="multimodal_agg_llava_epd",
        directory=vllm_dir,
        script_name="agg_multimodal_epd.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.nightly],
        model="llava-hf/llava-1.5-7b-hf",
        script_args=["--model", "llava-hf/llava-1.5-7b-hf"],
        request_payloads=[
            chat_payload(
                [
                    {
                        "type": "text",
                        "text": "What colors are in the following image? Respond only with the colors.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": MULTIMODAL_IMG_URL},
                    },
                ],
                repeat_count=1,
                expected_response=["purple"],
                temperature=0.0,
                max_tokens=100,
            )
        ],
    ),
    "multimodal_agg_qwen_epd": VLLMConfig(
        name="multimodal_agg_qwen_epd",
        directory=vllm_dir,
        script_name="agg_multimodal_epd.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.nightly],
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        delayed_start=0,
        script_args=["--model", "Qwen/Qwen2.5-VL-7B-Instruct"],
        timeout=360,
        request_payloads=[
            chat_payload(
                [
                    {
                        "type": "text",
                        "text": "What colors are in the following image? Respond only with the colors.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": MULTIMODAL_IMG_URL},
                    },
                ],
                repeat_count=1,
                expected_response=["purple"],
                max_tokens=100,
            )
        ],
    ),
    "multimodal_agg_qwen": VLLMConfig(
        name="multimodal_agg_qwen",
        directory=vllm_dir,
        script_name="agg_multimodal.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.nightly],
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        script_args=["--model", "Qwen/Qwen2.5-VL-7B-Instruct"],
        delayed_start=0,
        timeout=360,
        request_payloads=[
            chat_payload(
                [
                    {
                        "type": "text",
                        "text": "What colors are in the following image? Respond only with the colors.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": MULTIMODAL_IMG_URL},
                    },
                ],
                repeat_count=1,
                expected_response=["purple"],
                max_tokens=100,
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
        marks=[pytest.mark.gpu_2, pytest.mark.nightly],
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
    "aggregated_toolcalling": VLLMConfig(
        name="aggregated_toolcalling",
        directory=vllm_dir,
        script_name="agg_multimodal.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.multimodal],
        model="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        script_args=[
            "--model",
            "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
            "--max-model-len",
            "10000",
            "--dyn-tool-call-parser",
            "hermes",
        ],
        delayed_start=0,
        timeout=600,
        request_payloads=[
            ToolCallingChatPayload(
                body={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe what you see in this image in detail.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": MULTIMODAL_IMG_URL},
                                },
                            ],
                        }
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "describe_image",
                                "description": "Provides detailed description of objects and scenes in an image",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "objects": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "List of objects detected in the image",
                                        },
                                        "scene": {
                                            "type": "string",
                                            "description": "Overall scene description",
                                        },
                                    },
                                    "required": ["objects", "scene"],
                                },
                            },
                        }
                    ],
                    "tool_choice": "auto",
                    "max_tokens": 1024,
                },
                repeat_count=1,
                expected_response=["purple"],  # Validate image understanding
                expected_log=[],
                expected_tool_name="describe_image",  # Validate tool call happened
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
    "completions_only": VLLMConfig(
        name="completions_only",
        directory=vllm_dir,
        script_name="agg.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.timeout(
                420
            ),  # 3x estimated time (60s) + download time (240s) for 7B model
        ],
        model="deepseek-ai/deepseek-llm-7b-base",
        script_args=[
            "--model",
            "deepseek-ai/deepseek-llm-7b-base",
            "--dyn-endpoint-types",
            "completions",
        ],
        request_payloads=[
            completion_payload_default(),
        ],
    ),
    "guided_decoding_json": VLLMConfig(
        name="guided_decoding_json",
        directory=vllm_dir,
        script_name="agg.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.pre_merge],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload(
                "Generate a person with name and age",
                repeat_count=1,
                expected_response=['"name"', '"age"'],
                temperature=0.0,
                max_tokens=100,
                extra_body={
                    "guided_json": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name", "age"],
                    }
                },
            )
        ],
    ),
    "guided_decoding_regex": VLLMConfig(
        name="guided_decoding_regex",
        directory=vllm_dir,
        script_name="agg.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.pre_merge],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload(
                "Generate a color name (red, blue, or green)",
                repeat_count=1,
                expected_response=["red", "blue", "green"],
                temperature=0.0,
                max_tokens=20,
                extra_body={"guided_regex": r"(red|blue|green)"},
            )
        ],
    ),
    "guided_decoding_choice": VLLMConfig(
        name="guided_decoding_choice",
        directory=vllm_dir,
        script_name="agg.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.pre_merge],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload(
                "Generate a color name (red, blue, or green)",
                repeat_count=1,
                expected_response=["red", "blue", "green"],
                temperature=0.0,
                max_tokens=20,
                extra_body={"guided_choice": ["red", "blue", "green"]},
            )
        ],
    ),
}


@pytest.fixture(params=params_with_model_mark(vllm_configs))
def vllm_config_test(request):
    """Fixture that provides different vLLM test configurations"""
    return vllm_configs[request.param]


@pytest.mark.vllm
@pytest.mark.e2e
@pytest.mark.nightly
def test_serve_deployment(
    vllm_config_test, request, runtime_services, predownload_models, image_server
):
    """
    Test dynamo serve deployments with different graph configurations.
    """
    config = vllm_config_test
    run_serve_deployment(config, request)


@pytest.mark.vllm
@pytest.mark.e2e
@pytest.mark.gpu_2
def test_multimodal_b64(request, runtime_services, predownload_models):
    """
    Test multimodal inference with base64 url passthrough.

    This test is separate because it loads the required image at runtime
    (not collection time), ensuring it only fails when actually executed.
    """
    # Load B64 image at test execution time
    with open(MULTIMODAL_IMG_PATH, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode()

    # Create payload with B64 image
    b64_payload = chat_payload(
        [
            {
                "type": "text",
                "text": "What colors are in the following image? Respond only with the colors.",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_img}"},
            },
        ],
        repeat_count=1,
        expected_response=["purple"],
        max_tokens=100,
    )

    # Create test config
    config = VLLMConfig(
        name="test_multimodal_b64",
        directory=vllm_dir,
        script_name="agg_multimodal.sh",
        marks=[],  # markers at function-level
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        script_args=["--model", "Qwen/Qwen2.5-VL-7B-Instruct"],
        delayed_start=0,
        timeout=360,
        request_payloads=[b64_payload],
    )

    run_serve_deployment(config, request)
