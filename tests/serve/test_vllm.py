# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import dataclasses
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import pytest

from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.serve.conftest import MULTIMODAL_IMG_PATH, MULTIMODAL_IMG_URL
from tests.serve.lora_utils import MinioLoraConfig
from tests.utils.constants import DefaultPort
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import (
    cached_tokens_chat_payload,
    chat_payload,
    chat_payload_default,
    chat_payload_with_logprobs,
    completion_payload_default,
    completion_payload_with_logprobs,
    metric_payload_default,
)
from tests.utils.payloads import LoraTestChatPayload, ToolCallingChatPayload

logger = logging.getLogger(__name__)


def _is_cuda13() -> bool:
    v = os.environ.get("CUDA_VERSION", "")
    # handles "13", "13.0", "13.0.1", etc.
    return v.startswith("13")


@dataclass
class VLLMConfig(EngineConfig):
    """Configuration for vLLM test scenarios"""

    stragglers: list[str] = field(default_factory=lambda: ["VLLM:EngineCore"])


vllm_dir = os.environ.get("VLLM_DIR") or os.path.join(
    WORKSPACE_DIR, "examples/backends/vllm"
)


# vLLM test configurations
# NOTE: pytest.mark.gpu_1 tests take ~5.5 minutes total to run sequentially (with models pre-cached)
# TODO: Now that these tests use dynamic ports, optimize the runtime by bin-packing and running
# multiple engine deployments in parallel (while keeping GPU contention under control). This may
# require annotating each config with approximate GPU RAM usage so a future collector/launcher can
# bin-pack safely.
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
            chat_payload(
                "Can you write me a song?",
                repeat_count=1,
                expected_response=["song"],
                temperature=0.0,
                max_tokens=32,
                extra_body={
                    "stop": ["song"],
                    "include_stop_str_in_output": True,
                },
            ),
            metric_payload_default(min_num_requests=6, backend="vllm"),
        ],
    ),
    "aggregated_logprobs": VLLMConfig(
        name="aggregated_logprobs",
        directory=vllm_dir,
        script_name="agg.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.post_merge],
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
            pytest.mark.xfail(
                _is_cuda13(),
                reason="lmcache does not support CUDA 13 as of v0.3.11",
                strict=False,
            ),
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
            pytest.mark.pre_merge,
            pytest.mark.timeout(360),  # 3x estimated time (70s) + download time (150s)
            pytest.mark.xfail(
                _is_cuda13(),
                reason="lmcache does not support CUDA 13 as of v0.3.11",
                strict=False,
            ),
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
    "agg-router-approx": VLLMConfig(
        name="agg-router-approx",
        directory=vllm_dir,
        script_name="agg_router_approx.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.post_merge],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            # Test approximate KV routing (--no-kv-events mode)
            # Repeated requests should show cache-aware routing in logs
            chat_payload_default(
                repeat_count=3,
                expected_log=[
                    # Verify scheduler is selecting workers with cache awareness
                    r"Selected worker: worker_id=\d+ dp_rank=.*?, logit: ",
                    # After first request, should see cached blocks being tracked
                    r"with \d+ cached blocks",
                ],
            ),
            # Also test with cached tokens payload to verify usage field
            cached_tokens_chat_payload(
                repeat_count=3,
                expected_log=[
                    # Verify routing decision shows cache hits
                    r"with \d+ cached blocks",
                ],
            ),
        ],
        env={
            "DYN_LOG": "dynamo_llm::kv_router::scheduler=info",
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
    "multimodal_agg_qwen2vl_2b_epd": VLLMConfig(
        name="multimodal_agg_qwen2vl_2b_epd",
        directory=vllm_dir,
        script_name="agg_multimodal_epd.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.pre_merge],
        model="Qwen/Qwen2-VL-2B-Instruct",
        script_args=["--model", "Qwen/Qwen2-VL-2B-Instruct", "--single-gpu"],
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
        marks=[pytest.mark.gpu_1, pytest.mark.pre_merge],
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
            pytest.mark.gpu_1,
            pytest.mark.nightly,
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
        marks=[pytest.mark.gpu_2, pytest.mark.nightly],
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
        marks=[pytest.mark.gpu_2, pytest.mark.multimodal, pytest.mark.nightly],
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
            pytest.mark.post_merge,
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
def test_serve_deployment(
    vllm_config_test,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_models,
    image_server,
):
    """
    Test dynamo serve deployments with different graph configurations.
    """
    config = dataclasses.replace(
        vllm_config_test, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)


@pytest.mark.vllm
@pytest.mark.e2e
@pytest.mark.gpu_2
@pytest.mark.nightly
@pytest.mark.timeout(360)  # Match VLLMConfig.timeout for this multimodal deployment
def test_multimodal_b64(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_models,
):
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

    config = dataclasses.replace(
        config, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)


# LoRA Test Directory
lora_dir = os.path.join(vllm_dir, "launch/lora")


def lora_chat_payload(
    lora_name: str,
    s3_uri: str,
    system_port: int = DefaultPort.SYSTEM1.value,
    repeat_count: int = 2,
    expected_response: Optional[list] = None,
    expected_log: Optional[list] = None,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> LoraTestChatPayload:
    """Create a LoRA-enabled chat payload for testing"""
    return LoraTestChatPayload(
        body={
            "model": lora_name,
            "messages": [
                {
                    "role": "user",
                    "content": "What is deep learning? Answer in one sentence.",
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        },
        lora_name=lora_name,
        s3_uri=s3_uri,
        system_port=system_port,
        repeat_count=repeat_count,
        expected_response=expected_response
        or ["learning", "neural", "network", "AI", "model"],
        expected_log=expected_log or [],
    )


@pytest.mark.vllm
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model("Qwen/Qwen3-0.6B")
@pytest.mark.timeout(600)
@pytest.mark.post_merge
def test_lora_aggregated(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    minio_lora_service,
    dynamo_dynamic_ports,
):
    """
    Test LoRA inference with aggregated vLLM deployment.

    This test:
    1. Uses MinIO fixture to provide S3-compatible storage with uploaded LoRA
    2. Starts vLLM with LoRA support enabled
    3. Loads the LoRA adapter via system API
    4. Runs inference with the LoRA model
    """
    minio_config: MinioLoraConfig = minio_lora_service

    # Create payload that loads LoRA and tests inference
    lora_payload = lora_chat_payload(
        lora_name=minio_config.lora_name,
        s3_uri=minio_config.get_s3_uri(),
        system_port=DefaultPort.SYSTEM1.value,
        repeat_count=2,
    )

    # Create test config with MinIO environment variables
    config = VLLMConfig(
        name="test_lora_aggregated",
        directory=vllm_dir,
        script_name="lora/agg_lora.sh",
        marks=[],  # markers at function-level
        model="Qwen/Qwen3-0.6B",
        timeout=600,
        env=minio_config.get_env_vars(),
        request_payloads=[lora_payload],
    )

    config = dataclasses.replace(
        config, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(
        config,
        request,
        ports=dynamo_dynamic_ports,
        extra_env=minio_config.get_env_vars(),
    )


@pytest.mark.vllm
@pytest.mark.e2e
@pytest.mark.gpu_2
@pytest.mark.model("Qwen/Qwen3-0.6B")
@pytest.mark.timeout(600)
@pytest.mark.post_merge
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_lora_aggregated_router(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    minio_lora_service,
    dynamo_dynamic_ports,
    num_system_ports,
):
    """
    Test LoRA inference with aggregated vLLM deployment using KV router.

    This test:
    1. Uses MinIO fixture to provide S3-compatible storage with uploaded LoRA
    2. Starts multiple vLLM workers with LoRA support and KV router
    3. Loads the LoRA adapter on both workers via system API
    4. Runs inference with the LoRA model, verifying KV cache routing
    """
    assert (
        num_system_ports >= 2
    ), "serve tests require at least SYSTEM_PORT1 + SYSTEM_PORT2"
    minio_config: MinioLoraConfig = minio_lora_service

    # Create payloads that load LoRA on both workers and test inference
    # Worker 1 (DefaultPort.SYSTEM1)
    lora_payload_worker1 = lora_chat_payload(
        lora_name=minio_config.lora_name,
        s3_uri=minio_config.get_s3_uri(),
        system_port=DefaultPort.SYSTEM1.value,
        repeat_count=1,
    )

    # Worker 2 (DefaultPort.SYSTEM2)
    lora_payload_worker2 = lora_chat_payload(
        lora_name=minio_config.lora_name,
        s3_uri=minio_config.get_s3_uri(),
        system_port=DefaultPort.SYSTEM2.value,
        repeat_count=1,
    )

    # Additional inference payload to test routing (LoRA already loaded)
    inference_payload = chat_payload(
        content="Explain machine learning in simple terms.",
        repeat_count=2,
        expected_response=["learn", "data", "algorithm", "model", "pattern"],
        max_tokens=150,
        temperature=0.0,
    ).with_model(minio_config.lora_name)

    # Add env vars including PYTHONHASHSEED for deterministic KV event IDs
    env_vars = minio_config.get_env_vars()
    env_vars["PYTHONHASHSEED"] = "0"

    # Create test config with MinIO environment variables
    config = VLLMConfig(
        name="test_lora_aggregated_router",
        directory=vllm_dir,
        script_name="lora/agg_lora_router.sh",
        marks=[],  # markers at function-level
        model="Qwen/Qwen3-0.6B",
        timeout=600,
        env=env_vars,
        request_payloads=[
            lora_payload_worker1,
            lora_payload_worker2,
            inference_payload,
        ],
    )

    config = dataclasses.replace(
        config, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(
        config, request, ports=dynamo_dynamic_ports, extra_env=env_vars
    )
