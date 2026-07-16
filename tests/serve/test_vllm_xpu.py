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
from tests.serve.conftest import MULTIMODAL_IMG_URL, get_multimodal_test_image_bytes
from tests.serve.lora_utils import MinioLoraConfig
from tests.serve.multimodal_profiles.vllm_xpu import (
    VLLM_MULTIMODAL_PROFILES,
    VLLM_TOPOLOGY_SCRIPTS,
)
from tests.utils.constants import DefaultPort
from tests.utils.engine_process import EngineConfig
from tests.utils.multimodal import (
    IMAGE_COLOR_PROMPT,
    LOCAL_VIDEO_TEST_URI,
    make_multimodal_configs,
)
from tests.utils.payload_builder import (
    chat_payload,
    chat_payload_default,
    chat_payload_with_logprobs,
    completion_payload_default,
    completion_payload_with_logprobs,
    embedding_payload,
    embedding_payload_default,
    kv_events_metrics_payload,
    metric_payload_default,
    router_cached_tokens_chat_payload,
    router_selection_chat_payload_default,
)
from tests.utils.payloads import (
    EmbeddingMultiWorkerDispatchPayload,
    EmbeddingPayload,
    LoraTestChatPayload,
    ToolCallingChatPayload,
)

logger = logging.getLogger(__name__)


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
vllm_configs = {
    **_mm_configs,
    "aggregated": VLLMConfig(
        name="aggregated_xpu",
        directory=vllm_dir,
        script_name="xpu/agg_xpu.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.xpu_1,
            pytest.mark.profiled_vram_gib(3.8),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                1_119_388_000
            ),  # KV cache cap (2x safety over min=559_693_824)
            pytest.mark.timeout(
                360
            ),  # ~8.5x observed 42.2s; bumped for GPU-parallel headroom
            pytest.mark.pre_merge,
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
        name="aggregated_logprobs_xpu",
        directory=vllm_dir,
        script_name="xpu/agg_xpu.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.xpu_1,
            pytest.mark.profiled_vram_gib(3.8),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                1_119_388_000
            ),  # KV cache cap (2x safety over min=559_693_824)
            pytest.mark.timeout(420),
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
        name="aggregated_lmcache_xpu",
        directory=vllm_dir,
        script_name="xpu/agg_lmcache_xpu.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.lmcache,
            pytest.mark.xpu_1,
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
        name="aggregated_lmcache_multiproc_xpu",
        directory=vllm_dir,
        script_name="xpu/agg_lmcache_multiproc_xpu.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.lmcache,
            pytest.mark.xpu_1,
            pytest.mark.profiled_vram_gib(3.8),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                1_119_388_000
            ),  # KV cache cap (2x safety over min=559_693_824)
            pytest.mark.timeout(360),  # ~7x observed 49.3s; old value before profiling
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        env={
            "PROMETHEUS_MULTIPROC_DIR": f"/tmp/prometheus_multiproc_test_{os.getpid()}_{random.randint(0, 10000)}",
        },
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            metric_payload_default(min_num_requests=6, backend="vllm"),
            metric_payload_default(min_num_requests=6, backend="lmcache"),
        ],
    ),
    "agg-request-plane-tcp": VLLMConfig(
        name="agg-request-plane-tcp-xpu",
        directory=vllm_dir,
        script_name="xpu/agg_request_planes_xpu.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.xpu_1,
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
    "agg-router": VLLMConfig(
        name="agg-router-xpu",
        directory=vllm_dir,
        script_name="xpu/agg_router_xpu.sh",
        marks=[
            pytest.mark.xpu_2,
            pytest.mark.router,
            pytest.mark.profiled_vram_gib(7.6),  # 2x 3.8 GiB (one per GPU)
            pytest.mark.requested_vllm_kv_cache_bytes(
                1_119_388_000
            ),  # KV cache cap per worker (2x safety over min=559_693_824)
            pytest.mark.timeout(
                420
            ),  # 2 workers + router startup; bumped for GPU-parallel headroom
            pytest.mark.post_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            router_selection_chat_payload_default(),
            kv_events_metrics_payload(system_ports=[DefaultPort.SYSTEM2.value]),
        ],
        env={},
    ),
    "agg-router-approx": VLLMConfig(
        name="agg-router-approx-xpu",
        directory=vllm_dir,
        script_name="xpu/agg_router_approx_xpu.sh",
        marks=[
            pytest.mark.xpu_2,
            pytest.mark.router,
            pytest.mark.profiled_vram_gib(7.6),  # 2x 3.8 GiB (one per GPU)
            pytest.mark.requested_vllm_kv_cache_bytes(
                1_119_388_000
            ),  # KV cache cap per worker (2x safety over min=559_693_824)
            pytest.mark.timeout(
                420
            ),  # 2 workers + router startup; bumped for GPU-parallel headroom
            pytest.mark.post_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            # Test approximate KV routing (--no-kv-events mode)
            # Repeated requests should show cache-aware routing in nvext.
            router_selection_chat_payload_default(repeat_count=3),
            # Also test with cached tokens payload to verify usage field
            router_cached_tokens_chat_payload(repeat_count=3),
        ],
        env={},
    ),
    "multimodal_agg_frontend_decoding": VLLMConfig(
        name="multimodal_agg_frontend_decoding_xpu",
        directory=vllm_dir,
        script_name="xpu/agg_multimodal_xpu.sh",
        marks=[
            pytest.mark.xpu_1,
            pytest.mark.multimodal,
            pytest.mark.profiled_vram_gib(9.6),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                1_710_490_000
            ),  # KV cache cap (2x safety over min=855_244_800)
            pytest.mark.timeout(
                360
            ),  # XPU engine init (CCL + model load) needs more time
            pytest.mark.post_merge,
        ],
        model="Qwen/Qwen3-VL-2B-Instruct",
        timeout=360,
        env={"DYN_MM_ALLOW_INTERNAL": "1"},
        script_args=[
            "--model",
            "Qwen/Qwen3-VL-2B-Instruct",
            "--frontend-decoding",
        ],
        request_payloads=[
            chat_payload(
                [
                    {
                        "type": "text",
                        "text": IMAGE_COLOR_PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": MULTIMODAL_IMG_URL},
                    },
                ],
                repeat_count=1,
                expected_response=["purple", "blue", "white", "green"],
                temperature=0.0,
                max_tokens=100,
            )
        ],
    ),
    "multimodal_agg_qwen": VLLMConfig(
        name="multimodal_agg_qwen_xpu",
        directory=vllm_dir,
        script_name="xpu/agg_multimodal_xpu.sh",
        marks=[
            pytest.mark.xpu_1,
            pytest.mark.multimodal,
            pytest.mark.profiled_vram_gib(19.9),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                922_354_000
            ),  # KV cache cap (2x safety over min=461_176_832)
            pytest.mark.timeout(
                360
            ),  # ~7x observed 50.0s; 7B model loads ~48s on CI (A10G/L4)
            pytest.mark.post_merge,
        ],
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        script_args=["--model", "Qwen/Qwen2.5-VL-7B-Instruct"],
        delayed_start=0,
        timeout=360,
        env={"DYN_MM_ALLOW_INTERNAL": "1"},
        request_payloads=[
            chat_payload(
                [
                    {
                        "type": "text",
                        "text": IMAGE_COLOR_PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": MULTIMODAL_IMG_URL},
                    },
                ],
                repeat_count=1,
                expected_response=["purple", "blue", "white", "green"],
                max_tokens=100,
            ),
        ],
    ),
    "multimodal_agg_llava": VLLMConfig(
        name="multimodal_agg_llava_xpu",
        directory=vllm_dir,
        script_name="xpu/agg_multimodal_xpu.sh",
        marks=[
            pytest.mark.xpu_1,
            pytest.mark.multimodal,
            pytest.mark.profiled_vram_gib(14.9),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                922_354_000
            ),  # KV cache cap (2x safety over min=461_176_832)
            pytest.mark.timeout(
                300
            ),  # ~7x observed 42.7s; 7B model loads ~48s on CI (A10G/L4)
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
    "aggregated_toolcalling": VLLMConfig(
        name="aggregated_toolcalling_xpu",
        directory=vllm_dir,
        script_name="xpu/agg_multimodal_xpu.sh",
        marks=[
            pytest.mark.xpu_2,
            pytest.mark.multimodal,
            pytest.mark.nightly,
        ],
        model="Qwen/Qwen3-VL-8B-Instruct",
        script_args=[
            "--model",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--max-model-len",
            "4096",
            "--gpu-memory-utilization",
            "0.90",
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
                    "tool_choice": "auto",
                    "max_tokens": 1024,
                },
                repeat_count=1,
                expected_response=[
                    "green",
                    "purple",
                    "llm",
                    "optimize",
                    "deploy",
                ],  # OR: pass if any keyword found in tool args
                expected_log=[],
                expected_tool_name="describe_image",  # Validate tool call happened
            )
        ],
    ),
    # Video multimodal tests for CI using the vLLM video launch scripts.
    "multimodal_video_agg": VLLMConfig(
        name="multimodal_video_agg_xpu",
        directory=vllm_dir,
        script_name="xpu/agg_multimodal_xpu.sh",
        marks=[
            pytest.mark.skip(
                reason="flaky test, local video file downloading may fail due to network issues"
            ),
            pytest.mark.xpu_1,
            pytest.mark.multimodal,
            pytest.mark.nightly,
            pytest.mark.timeout(600),  # TODO: profile to get tighter timeout
        ],  # TODO: profile to get max_vram
        model="Qwen/Qwen3-VL-2B-Instruct",
        delayed_start=60,  # Video models require longer loading time
        script_args=["--model", "Qwen/Qwen3-VL-2B-Instruct"],
        timeout=600,  # 10 minutes for video processing overhead
        env={"DYN_MM_LOCAL_PATH": WORKSPACE_DIR},
        request_payloads=[
            chat_payload(
                [
                    {"type": "text", "text": "Describe the video in detail"},
                    {
                        "type": "video_url",
                        "video_url": {"url": LOCAL_VIDEO_TEST_URI},
                    },
                ],
                repeat_count=1,
                expected_response=["red", "static", "still"],
                temperature=0.0,
                max_tokens=100,
            )
        ],
    ),
    "completions_only": VLLMConfig(
        name="completions_only_xpu",
        directory=vllm_dir,
        script_name="xpu/agg_xpu.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.xpu_1,
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
    "guided_decoding": VLLMConfig(
        name="guided_decoding_xpu",
        directory=vllm_dir,
        script_name="xpu/agg_xpu.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.xpu_1,
            pytest.mark.profiled_vram_gib(3.8),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                1_119_388_000
            ),  # KV cache cap (2x safety over min=559_693_824)
            pytest.mark.timeout(110),  # ~5x observed 22.3s; CI machines are slower
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
    "embedding_agg": VLLMConfig(
        name="embedding_agg",
        directory=vllm_dir,
        script_name="agg_embed.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.xpu_1,
            # Qwen3-Embedding-0.6B at float32 = ~2.4 GiB params + vLLM overhead.
            # Refine after first CI run reports the actual profiled peak.
            pytest.mark.profiled_vram_gib(5.0),
            # Pooling models do not use a KV cache, but the test harness still
            # needs a non-zero allocation budget. Use the minimum vLLM accepts.
            pytest.mark.requested_vllm_kv_cache_bytes(559_693_824),
            # Cold model load + vLLM startup + warmup for embedding pooling.
            # Mirrors SGLang's 300s embedding-test timeout; refine after profiling.
            # 360 already >= 3x observed 92s (job-log 2026-05-29); left as headroom.
            pytest.mark.timeout(360),
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-Embedding-0.6B",
        request_payloads=[
            # Default helper sends two pre-defined inputs.
            embedding_payload_default(
                repeat_count=2,
                expected_response=["Generated 2 embeddings with dimension"],
            ),
            # Single string input — exercises the str path in
            # EmbeddingWorkerHandler.generate.
            embedding_payload(
                input_text="Hello, world!",
                repeat_count=1,
                expected_response=["Generated 1 embeddings with dimension"],
            ),
            # Batched list input — exercises the per-input loop and index
            # preservation in EmbeddingWorkerHandler._transform_response.
            embedding_payload(
                input_text=[
                    "The quick brown fox jumps over the lazy dog.",
                    "Machine learning is transforming technology.",
                    "Natural language processing enables computers to understand text.",
                ],
                repeat_count=1,
                expected_response=["Generated 3 embeddings with dimension"],
            ),
            # `dimensions` reduction (Matryoshka). Qwen3-Embedding-0.6B has a
            # hidden dim of 1024, so the reduced vector should be exactly 128.
            # The worker forwards `dimensions` to vLLM's pooler (truncate +
            # re-normalize); `agg_embed.sh` launches this model with
            # `--hf-overrides '{"is_matryoshka": true}'` so vLLM accepts the
            # request (Qwen3-Embedding's config doesn't declare Matryoshka).
            # Built inline because the `embedding_payload()` helper doesn't
            # expose an `extra_body` kwarg yet.
            EmbeddingPayload(
                body={"input": ["Hello, world!"], "dimensions": 128},
                repeat_count=1,
                expected_log=[],
                expected_response=["Generated 1 embeddings with dimension 128"],
            ),
            # encoding_format=base64. The Python handler base64-encodes the
            # vector and the Rust frontend deserializes it as a string.
            # The validator decodes back to floats so the dimension
            # assertion stays uniform across both shapes.
            EmbeddingPayload(
                body={
                    "input": ["Hello, world!"],
                    "dimensions": 128,
                    "encoding_format": "base64",
                },
                repeat_count=1,
                expected_log=[],
                expected_response=["Generated 1 embeddings with dimension 128"],
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
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_serve_deployment(
    vllm_config_test,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    num_system_ports,
    predownload_models,
    image_server,
):
    """
    Test dynamo serve deployments with different graph configurations.
    """
    assert (
        num_system_ports >= 2
    ), "serve tests require at least SYSTEM_PORT1 + SYSTEM_PORT2"

    # Start the media HTTP server only for multimodal configs that need it,
    # avoiding the TOCTOU port-allocation race for non-multimodal tests.
    if any(
        getattr(m, "name", None) == "multimodal"
        or getattr(getattr(m, "mark", None), "name", None) == "multimodal"
        for m in vllm_config_test.marks
    ):
        request.getfixturevalue("image_server")

    config = dataclasses.replace(
        vllm_config_test, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)


@pytest.mark.vllm
@pytest.mark.multimodal
@pytest.mark.e2e
@pytest.mark.xpu_2
@pytest.mark.nightly
@pytest.mark.multimodal
@pytest.mark.model("Qwen/Qwen2.5-VL-7B-Instruct")
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

    Uses ``@pytest.mark.model`` so nightly multi-GPU jobs (xpu_2 without the
    xpu_1 multimodal_agg_qwen param) still predownload Qwen2.5-VL-7B before
    ``HF_HUB_OFFLINE=1``.
    """
    # Load B64 image at test execution time (uses real PNG even if MULTIMODAL_IMG is LFS pointer)
    b64_img = base64.b64encode(get_multimodal_test_image_bytes()).decode()

    # Create payload with B64 image
    b64_payload = chat_payload(
        [
            {
                "type": "text",
                "text": IMAGE_COLOR_PROMPT,
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_img}"},
            },
        ],
        repeat_count=1,
        expected_response=["purple", "blue", "white", "green"],
        max_tokens=100,
    )

    # Create test config
    config = VLLMConfig(
        name="test_multimodal_b64_xpu",
        directory=vllm_dir,
        script_name="xpu/agg_multimodal_xpu.sh",
        marks=[],  # markers at function-level
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        script_args=[
            "--model",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "--max-model-len",
            "4096",
            "--gpu-memory-utilization",
            "0.90",
        ],
        delayed_start=0,
        timeout=360,
        request_payloads=[b64_payload],
    )

    config = dataclasses.replace(
        config, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)


@pytest.mark.vllm
@pytest.mark.multimodal
@pytest.mark.e2e
@pytest.mark.xpu_1
@pytest.mark.pre_merge
@pytest.mark.multimodal
@pytest.mark.model("Qwen/Qwen3-VL-2B-Instruct")
@pytest.mark.timeout(360)
def test_multimodal_b64_frontend_decoding(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_models,
):
    """
    Test multimodal inference with base64 images through frontend decoding path.

    This exercises the Rust frontend image decode + NIXL RDMA transfer path
    with inline base64 data: URIs (not HTTP URLs). Verifies that the
    strip_inline_data_urls optimization does not break correctness.

    HF predownload: same model is already listed via ``@pytest.mark.model`` on
    ``test_serve_deployment[multimodal_video_agg]`` (pre_merge + xpu_1), so no
    extra ``model`` mark is needed here for PR CI.
    """
    b64_img = base64.b64encode(get_multimodal_test_image_bytes()).decode()

    b64_payload = chat_payload(
        [
            {
                "type": "text",
                "text": IMAGE_COLOR_PROMPT,
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_img}"},
            },
        ],
        repeat_count=1,
        expected_response=["purple", "blue", "white", "green"],
        temperature=0.0,
        max_tokens=100,
    )

    config = VLLMConfig(
        name="test_multimodal_b64_frontend_decoding",
        directory=vllm_dir,
        script_name="xpu/agg_multimodal_xpu.sh",
        marks=[],
        model="Qwen/Qwen3-VL-2B-Instruct",
        script_args=[
            "--model",
            "Qwen/Qwen3-VL-2B-Instruct",
            "--frontend-decoding",
        ],
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
@pytest.mark.core
@pytest.mark.e2e
@pytest.mark.xpu_1
@pytest.mark.model("Qwen/Qwen3-0.6B", "codelion/Qwen3-0.6B-accuracy-recovery-lora")
@pytest.mark.timeout(600)
@pytest.mark.post_merge
@pytest.mark.xfail(
    reason="XPU LoRA dtype mismatch: PunicaWrapperXPU requires inputs dtype to "
    "match lora_b_weights dtype, pending fix in vLLM XPU backend",
    strict=False,
)
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
    env_vars = minio_config.get_env_vars()
    config = VLLMConfig(
        name="test_lora_aggregated_xpu",
        directory=vllm_dir,
        script_name="lora/xpu/agg_lora_xpu.sh",
        marks=[],  # markers at function-level
        model="Qwen/Qwen3-0.6B",
        timeout=600,
        env=env_vars,
        request_payloads=[lora_payload],
    )

    config = dataclasses.replace(
        config, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(
        config,
        request,
        ports=dynamo_dynamic_ports,
        extra_env=env_vars,
    )


@pytest.mark.vllm
@pytest.mark.router
@pytest.mark.e2e
@pytest.mark.xpu_2
@pytest.mark.model("Qwen/Qwen3-0.6B")
@pytest.mark.timeout(600)
@pytest.mark.post_merge
@pytest.mark.xfail(
    reason="XPU LoRA dtype mismatch: PunicaWrapperXPU requires inputs dtype to "
    "match lora_b_weights dtype, pending fix in vLLM XPU backend",
    strict=False,
)
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_lora_aggregated_router(
    request,
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
        name="test_lora_aggregated_router_xpu",
        directory=vllm_dir,
        script_name="lora/xpu/agg_lora_router_xpu.sh",
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

# ─────────────────────────────────────────────────────────────────────────────
# Multi-worker embedding tests
#
# Verify that Dynamo's routing layer correctly:
#   1. Load-balances across N workers serving the SAME embedding model.
#   2. Dispatches by request `model` field across workers serving
#      DIFFERENT embedding models.
#
# The routing code (`get_embeddings_engine` + `select_worker_set_with`) is
# already exercised by chat-completions through identical machinery, so the
# code is verified by construction; what these tests add is **explicit
# exercise of the embedding code path**.
# ─────────────────────────────────────────────────────────────────────────────


def _embedding_warmup_payload(model: str) -> EmbeddingPayload:
    """One quick embedding request used as a smoke check before the burst."""
    return EmbeddingPayload(
        body={"model": model, "input": "warmup"},
        expected_response=["Generated 1 embeddings with dimension"],
        expected_log=[],
        repeat_count=1,
    )


def _embedding_dispatch_burst(
    *,
    model: str,
    repeat_count: int,
    expected_worker_indices_with_delta: set[int],
    min_total_delta: int,
) -> EmbeddingMultiWorkerDispatchPayload:
    """One burst payload that drives the dispatch assertion.

    ``system_ports`` is fixed at ``[SYSTEM_PORT1, SYSTEM_PORT2]`` — the
    harness remaps those placeholders to the per-test dynamic ports.
    Dispatch expectations are expressed as INDICES into that list (index 0
    = first worker = XPU 0, index 1 = second worker = XPU 1).
    """
    return EmbeddingMultiWorkerDispatchPayload(
        body={"model": model, "input": "Hello, world!"},
        expected_response=["Generated 1 embeddings with dimension"],
        expected_log=[],
        repeat_count=repeat_count,
        system_ports=[DefaultPort.SYSTEM1.value, DefaultPort.SYSTEM2.value],
        expected_worker_indices_with_delta=expected_worker_indices_with_delta,
        min_total_delta=min_total_delta,
    )


# Same model on both GPUs — verifies weighted-random selection in
# `select_worker_set_with` fans out across both registered workers.
_EMBED_SAME_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Multi-model setup: each model is served by exactly one worker, so a
# request whose `model` field names model A must never reach model B's
# worker. BGE-small-en is intentionally small (33M params, fits alongside
# Qwen3-Embedding-0.6B on stock CI nodes).
_EMBED_MODEL_A = "Qwen/Qwen3-Embedding-0.6B"
_EMBED_MODEL_B = "BAAI/bge-small-en-v1.5"


@pytest.mark.vllm
@pytest.mark.core
@pytest.mark.e2e
@pytest.mark.xpu_2
@pytest.mark.model(_EMBED_SAME_MODEL)
@pytest.mark.profiled_vram_gib(5.0)  # per GPU; mirrors single-worker embedding_agg
@pytest.mark.requested_vllm_kv_cache_bytes(559_693_824)
@pytest.mark.timeout(
    420
)  # 2x cold-load vs single-worker embedding_agg (2 GPUs in parallel)
@pytest.mark.pre_merge
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_embedding_multi_worker_same_model_load_balance(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    num_system_ports,
    predownload_models,
):
    """Two workers serving the same model: a burst of requests should be
    weighted-randomly distributed so both workers' /metrics counters > 0.

    Burst size is deliberately small (20) so pure chance of all-to-one-
    worker is negligible (≈ 1 in 2^19) while keeping test runtime tight
    on 2-XPU CI nodes.
    """
    assert num_system_ports >= 2, "Requires SYSTEM_PORT1 + SYSTEM_PORT2"

    # 20 repeats inside the burst; the payload uses repeat 1 as its
    # baseline snapshot and asserts the delta across repeats 2..20.
    burst = _embedding_dispatch_burst(
        model=_EMBED_SAME_MODEL,
        repeat_count=20,
        # Both workers (indices 0 and 1) should see delta > 0.
        expected_worker_indices_with_delta={0, 1},
        # 19 post-baseline requests; loose lower bound absorbs any frontend
        # health probes that the worker happens to count.
        min_total_delta=15,
    )

    config = VLLMConfig(
        name="embedding_multi_worker_same_model",
        directory=vllm_dir,
        script_name="xpu/agg_embed_multiworker_xpu.sh",
        script_args=[_EMBED_SAME_MODEL, _EMBED_SAME_MODEL],
        marks=[],  # markers at function level
        model=_EMBED_SAME_MODEL,
        timeout=420,
        # ``DYN_HEALTH_CHECK_ENABLED=true`` flips the runtime's canary
        # on. Without it ``/health`` returns 200 the moment the endpoint
        # is registered (before the engine has produced anything), so
        # ``health_check_workers=True`` would gate on a constant-true
        # signal and we'd race startup just like the old ``delayed_start``
        # path. Setting the flag plus the embedding-shaped probe payload
        # in ``_create_embedding_worker`` is what actually makes
        # readiness mean "engine produced an embedding".
        health_check_workers=True,
        env={"DYN_HEALTH_CHECK_ENABLED": "true"},
        request_payloads=[
            _embedding_warmup_payload(_EMBED_SAME_MODEL),
            burst,
        ],
    )

    config = dataclasses.replace(
        config, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)


@pytest.mark.vllm
@pytest.mark.core
@pytest.mark.e2e
@pytest.mark.xpu_2
@pytest.mark.model(_EMBED_MODEL_A)
@pytest.mark.model(_EMBED_MODEL_B)
@pytest.mark.profiled_vram_gib(5.0)  # Qwen3-Embed (0.6B) is the larger of the two
@pytest.mark.requested_vllm_kv_cache_bytes(559_693_824)
@pytest.mark.timeout(420)
@pytest.mark.pre_merge
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_embedding_multi_worker_multi_model_dispatch(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    num_system_ports,
    predownload_models,
):
    """Two workers, two different models: requests for model A must reach
    only worker A; symmetric for model B. Verifies name-keyed dispatch in
    ``get_embeddings_engine`` for embedding traffic.
    """
    assert num_system_ports >= 2, "Requires SYSTEM_PORT1 + SYSTEM_PORT2"

    # Worker A → SYSTEM_PORT1 (GPU 0, model A, payload index 0)
    # Worker B → SYSTEM_PORT2 (GPU 1, model B, payload index 1)
    #
    # Each burst takes its own baseline snapshot and checks the DELTA
    # over its repeats — so burst_b's check is independent of burst_a's
    # absolute count, and "wrong-model traffic stays out" can actually
    # be expressed (no delta on the wrong worker during this burst).
    burst_a = _embedding_dispatch_burst(
        model=_EMBED_MODEL_A,
        repeat_count=10,
        expected_worker_indices_with_delta={0},  # only worker A
        min_total_delta=5,
    )
    burst_b = _embedding_dispatch_burst(
        model=_EMBED_MODEL_B,
        repeat_count=10,
        expected_worker_indices_with_delta={1},  # only worker B
        min_total_delta=5,
    )

    config = VLLMConfig(
        name="embedding_multi_worker_multi_model",
        directory=vllm_dir,
        script_name="xpu/agg_embed_multiworker_xpu.sh",
        script_args=[_EMBED_MODEL_A, _EMBED_MODEL_B],
        marks=[],  # markers at function level
        # ``model`` here is just metadata for the test runner; the real
        # per-request model is set in each payload's body.
        model=_EMBED_MODEL_A,
        # BGE-small-en-v1.5's architecture caps at ``max_position_embeddings=512``
        # — applying the script's default ``MAX_MODEL_LEN=2048`` to it crashes
        # the second worker at engine init. Drop the cap to BGE's native max;
        # Qwen3-Embedding-0.6B happily accepts the lower cap.
        # ``DYN_HEALTH_CHECK_ENABLED=true`` is required for
        # ``health_check_workers=True`` below to gate on the canary
        # rather than on endpoint registration (see same-model test for
        # the full rationale).
        env={"MAX_MODEL_LEN": "512", "DYN_HEALTH_CHECK_ENABLED": "true"},
        timeout=420,
        health_check_workers=True,
        request_payloads=[
            _embedding_warmup_payload(_EMBED_MODEL_A),
            _embedding_warmup_payload(_EMBED_MODEL_B),
            burst_a,
            burst_b,
        ],
    )

    config = dataclasses.replace(
        config, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)
