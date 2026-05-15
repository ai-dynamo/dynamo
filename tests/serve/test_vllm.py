# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
import platform
import random
from dataclasses import dataclass, field
from typing import Optional

import pytest

from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.serve.conftest import MULTIMODAL_IMG_URL
from tests.serve.lora_utils import MinioLoraConfig
from tests.serve.multimodal_profiles.vllm import (
    VLLM_MULTIMODAL_PROFILES,
    VLLM_TOPOLOGY_SCRIPTS,
)
from tests.utils.constants import DefaultPort
from tests.utils.engine_process import EngineConfig
from tests.utils.multimodal import make_multimodal_configs
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


def _is_aarch64() -> bool:
    arch = os.environ.get("TARGETARCH") or os.environ.get("ARCH") or platform.machine()
    return arch in ("aarch64", "arm64")


def _xfail_lmcache_upstream_container():
    return pytest.mark.xfail(
        _is_cuda13() or _is_aarch64(),
        reason=(
            "LMCache is provided by the upstream vLLM image. The CUDA 13 image "
            "ships LMCache c_ops linked against libcudart.so.12, and LMCache "
            "does not publish aarch64 wheels yet."
        ),
        strict=False,
    )


@dataclass
class VLLMConfig(EngineConfig):
    """Configuration for vLLM test scenarios"""

    stragglers: list[str] = field(default_factory=lambda: ["VLLM:EngineCore"])


vllm_dir = os.environ.get("VLLM_DIR") or os.path.join(
    WORKSPACE_DIR, "examples/backends/vllm"
)

# Generated multimodal configs from profile definitions
_mm_configs: dict[str, VLLMConfig] = {}
for _profile in VLLM_MULTIMODAL_PROFILES:
    _mm_configs.update(
        make_multimodal_configs(_profile, VLLMConfig, vllm_dir, VLLM_TOPOLOGY_SCRIPTS)
    )

# vLLM test configurations
# NOTE: pytest.mark.gpu_1 tests take ~5.5 minutes total to run sequentially (with models pre-cached)
# TODO: Now that these tests use dynamic ports and each config has VRAM markers,
# optimize the runtime by bin-packing multiple engine deployments in parallel on the same GPU.
# A future collector/launcher can sum profiled_vram_gib values to decide how many tests fit
# concurrently without exceeding available VRAM.
vllm_configs = {
    **_mm_configs,
    "aggregated": VLLMConfig(
        name="aggregated",
        directory=vllm_dir,
        script_name="agg.sh",
        # Forwarded through agg.sh -> dynamo.vllm. Required for the
        # max_thinking_tokens payload below: vLLM only enables the thinking-
        # budget logits processor when reasoning_config is populated.
        script_args=["--reasoning-parser", "qwen3"],
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(3.8),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                1_119_388_000
            ),  # KV cache cap (2x safety over min=559_693_824)
            pytest.mark.timeout(
                480
            ),  # vLLM 0.20.x startup can exceed 360s on contended CI runners
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload_default(),
            chat_payload(
                "Name one color in a short sentence.",
                repeat_count=1,
                expected_response=[],
                max_tokens=16,
                extra_body={"n": 2},
                expected_num_choices=2,
            ),
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
            # Smoke: nvext.max_thinking_tokens reaches vLLM's
            # SamplingParams.thinking_token_budget without erroring. Requires
            # the worker to be started with `--reasoning-parser qwen3`
            # (see script_args above).
            chat_payload(
                "Solve: 1+1.",
                repeat_count=1,
                expected_response=[],
                max_tokens=64,
                extra_body={"nvext": {"max_thinking_tokens": 16}},
            ),
            metric_payload_default(min_num_requests=6, backend="vllm"),
        ],
    ),
    "aggregated_unified": VLLMConfig(
        name="aggregated_unified",
        directory=vllm_dir,
        script_name="agg.sh",
        script_args=["--unified"],
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(3.8),
            pytest.mark.requested_vllm_kv_cache_bytes(1_119_388_000),
            pytest.mark.timeout(480),
            pytest.mark.pre_merge,
            pytest.mark.unified,
        ],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "aggregated_logprobs": VLLMConfig(
        name="aggregated_logprobs",
        directory=vllm_dir,
        script_name="agg.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(3.8),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                1_119_388_000
            ),  # KV cache cap (2x safety over min=559_693_824)
            pytest.mark.timeout(120),  # ~5x observed 24.3s; CI machines are slower
            pytest.mark.post_merge,
        ],
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
            pytest.mark.core,
            pytest.mark.lmcache,
            _xfail_lmcache_upstream_container(),
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(3.8),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                1_119_388_000
            ),  # KV cache cap (2x safety over min=559_693_824)
            pytest.mark.timeout(360),  # ~7x observed 49.0s; old value before profiling
            pytest.mark.pre_merge,
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
            pytest.mark.core,
            pytest.mark.lmcache,
            _xfail_lmcache_upstream_container(),
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(3.8),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                1_119_388_000
            ),  # KV cache cap (2x safety over min=559_693_824)
            pytest.mark.timeout(360),  # ~7x observed 49.3s; old value before profiling
            pytest.mark.pre_merge,
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
            pytest.mark.core,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(3.8),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                1_119_388_000
            ),  # KV cache cap (2x safety over min=559_693_824)
            pytest.mark.timeout(
                360
            ),  # ~8x observed 43.0s; bumped for GPU-parallel headroom
            pytest.mark.pre_merge,
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
            pytest.mark.core,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(3.8),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                1_119_388_000
            ),  # KV cache cap (2x safety over min=559_693_824)
            pytest.mark.timeout(
                360
            ),  # ~8.5x observed 42.3s; bumped for GPU-parallel headroom
            pytest.mark.pre_merge,
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
        marks=[
            pytest.mark.gpu_2,
            pytest.mark.router,
            pytest.mark.pre_merge,
            pytest.mark.skip(reason="DYN-2263"),
        ],  # TODO: profile to get max_vram and timeout
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload_default(
                expected_log=[
                    r"ZMQ listener .* received batch with \d+ events \(seq=\d+(?:, [^)]*)?\)",
                    r"Event processor for worker_id \d+ processing event: Stored\(",
                    r"Selected worker: worker_type=\w+, worker_id=\d+ dp_rank=.*?, logit: ",
                ]
            )
        ],
        env={
            "DYN_LOG": "dynamo_llm::kv_router::publisher=trace,dynamo_kv_router::scheduling::selector=info",
        },
    ),
    "agg-router-approx": VLLMConfig(
        name="agg-router-approx",
        directory=vllm_dir,
        script_name="agg_router_approx.sh",
        marks=[
            pytest.mark.gpu_2,
            pytest.mark.router,
            pytest.mark.pre_merge,
            pytest.mark.skip(reason="DYN-2264"),
        ],  # TODO: profile to get max_vram and timeout
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            # Test approximate KV routing (--no-kv-events mode)
            # Repeated requests should show cache-aware routing in logs
            chat_payload_default(
                repeat_count=3,
                expected_log=[
                    # Verify scheduler is selecting workers with cache awareness
                    r"Selected worker: worker_type=\w+, worker_id=\d+ dp_rank=.*?, logit: ",
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
            "DYN_LOG": "dynamo_kv_router::scheduling::selector=info",
        },
    ),
    "disaggregated": VLLMConfig(
        name="disaggregated",
        directory=vllm_dir,
        script_name="disagg.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_2,
            pytest.mark.pre_merge,
        ],  # TODO: profile to get max_vram and timeout
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "disaggregated_same_gpu": VLLMConfig(
        name="disaggregated_same_gpu",
        directory=vllm_dir,
        script_name="disagg_same_gpu.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(7.3),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                1_023_525_000
            ),  # KV cache cap (2x safety over min=511_762_432)
            pytest.mark.timeout(300),  # ~6x observed 50s
            # post_merge: cumulative sequential test time exceeds 35-min job budget.
            # Move back to pre_merge once GPU tests run in parallel.
            pytest.mark.post_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        delayed_start=10,
        health_check_workers=True,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "disaggregated_same_gpu_chat_processor": VLLMConfig(
        name="disaggregated_same_gpu_chat_processor",
        directory=vllm_dir,
        script_name="disagg_same_gpu.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(7.3),
            pytest.mark.requested_vllm_kv_cache_bytes(1_023_525_000),
            pytest.mark.timeout(300),
            pytest.mark.post_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        delayed_start=10,
        health_check_workers=True,
        env={"DYN_CHAT_PROCESSOR": "vllm"},
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "disaggregated_same_gpu_chat_processor_kv_router": VLLMConfig(
        name="disaggregated_same_gpu_chat_processor_kv_router",
        directory=vllm_dir,
        script_name="disagg_same_gpu.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(7.3),
            pytest.mark.requested_vllm_kv_cache_bytes(1_023_525_000),
            pytest.mark.timeout(300),
            pytest.mark.post_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        delayed_start=10,
        health_check_workers=True,
        env={
            "DYN_CHAT_PROCESSOR": "vllm",
            "DYN_ROUTER_MODE": "kv",
            # Deterministic hash for KV event IDs, matches disagg_router.sh.
            "PYTHONHASHSEED": "0",
        },
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
            pytest.mark.core,
            pytest.mark.gpu_2,
            pytest.mark.vllm,
            pytest.mark.h100,
            pytest.mark.nightly,
            # TODO: profile to get max_vram and timeout
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
    "aggregated_toolcalling": VLLMConfig(
        name="aggregated_toolcalling",
        directory=vllm_dir,
        script_name="agg_multimodal.sh",
        marks=[
            pytest.mark.gpu_1,  # agg_multimodal.sh uses single GPU
            pytest.mark.multimodal,
            pytest.mark.nightly,
            pytest.mark.profiled_vram_gib(
                19.9
            ),  # align with multimodal_agg_qwen (7B VLM)
            pytest.mark.requested_vllm_kv_cache_bytes(
                922_354_000
            ),  # KV cache cap (2x safety over min=461_176_832)
        ],
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        script_args=[
            "--model",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "--max-model-len",
            "8192",
            "--dyn-tool-call-parser",
            "hermes",
        ],
        env={"DYN_MM_ALLOW_INTERNAL": "1"},
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
                    "tool_choice": "required",
                    "max_tokens": 1024,
                    "temperature": 0,
                },
                repeat_count=1,
                expected_response=[
                    "purple",
                    "green",
                    "lavender",
                    "violet",
                ],
                expected_log=[],
                expected_tool_name="describe_image",  # Validate tool call happened
            )
        ],
    ),
    "completions_only": VLLMConfig(
        name="completions_only",
        directory=vllm_dir,
        script_name="agg.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(18.3),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                4_074_898_000
            ),  # KV cache cap (2x safety over min=2_037_448_704)
            pytest.mark.timeout(
                420
            ),  # 7B model loads ~48s on CI (A10G/L4) vs ~15s locally
            pytest.mark.post_merge,
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
    "multi_node_tp_headless": VLLMConfig(
        name="multi_node_tp_headless",
        directory=os.path.join(WORKSPACE_DIR, "tests/serve"),
        script_name="multi_node_tp_headless.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_2,
            pytest.mark.pre_merge,
            # TODO: profile to get max_vram
            pytest.mark.timeout(300),
        ],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "guided_decoding": VLLMConfig(
        name="guided_decoding",
        directory=vllm_dir,
        script_name="agg.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(3.8),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                1_119_388_000
            ),  # KV cache cap (2x safety over min=559_693_824)
            pytest.mark.timeout(180),  # vLLM 0.20.x needs more CI headroom
            pytest.mark.pre_merge,
        ],
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
            ),
            chat_payload(
                "Generate a color name (red, blue, or green)",
                repeat_count=1,
                expected_response=["red", "blue", "green"],
                temperature=0.0,
                max_tokens=20,
                extra_body={"guided_regex": r"(red|blue|green)"},
            ),
            chat_payload(
                "Generate a color name (red, blue, or green)",
                repeat_count=1,
                expected_response=["red", "blue", "green"],
                temperature=0.0,
                max_tokens=20,
                extra_body={"guided_choice": ["red", "blue", "green"]},
            ),
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
@pytest.mark.core
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model("Qwen/Qwen3-0.6B")
@pytest.mark.model("codelion/Qwen3-0.6B-accuracy-recovery-lora")
@pytest.mark.profiled_vram_gib(4.0)  # actual nvidia-smi peak with kv-bytes cap
@pytest.mark.requested_vllm_kv_cache_bytes(
    941_712_000
)  # 2x safety over min=470_855_680
@pytest.mark.timeout(300)  # LoRA setup adds overhead; L4 machines are slower
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
@pytest.mark.router
@pytest.mark.e2e
@pytest.mark.gpu_2
@pytest.mark.model("Qwen/Qwen3-0.6B")
@pytest.mark.model("codelion/Qwen3-0.6B-accuracy-recovery-lora")
@pytest.mark.timeout(600)
@pytest.mark.pre_merge
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
