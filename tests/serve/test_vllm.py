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
    run_prefill_drain_deployment,
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
    chat_payload,
    chat_payload_default,
    chat_payload_with_logprobs,
    completion_payload_default,
    completion_payload_with_logprobs,
    elastic_ep_scale_payload,
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


def _is_cuda12() -> bool:
    v = os.environ.get("CUDA_VERSION", "")
    # handles "12", "12.9", "12.9.1", etc.
    return v.startswith("12")


def _is_aarch64() -> bool:
    arch = os.environ.get("TARGETARCH") or os.environ.get("ARCH") or platform.machine()
    return arch in ("aarch64", "arm64")


def _xfail_lmcache_upstream_container():
    return pytest.mark.xfail(
        _is_cuda12() or _is_aarch64(),
        reason=(
            "LMCache is provided by the upstream vLLM image. The CUDA 12 image "
            "ships LMCache c_ops linked against libcudart.so.13, and LMCache "
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
        # vLLM #43808 (0.23.0) moved the MRv2 thinking_token_budget check from
        # startup to request time, removing the auto-fallback to V1. Force V1
        # runner here until native MRv2 support is added.
        env={"VLLM_USE_V2_MODEL_RUNNER": "0"},
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(3.8),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                1_119_388_000
            ),  # KV cache cap (2x safety over min=559_693_824)
            pytest.mark.timeout(610),  # 3x ~203s under new scheduler (3d1554f)
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
            pytest.mark.timeout(610),  # 3x ~203s unified, new scheduler (3d1554f)
            pytest.mark.pre_merge,
            pytest.mark.unified,
        ],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "elastic_ep_unified": VLLMConfig(
        name="elastic_ep_unified",
        directory=vllm_dir,
        script_name="elastic_ep.sh",
        # Start at DP=2 on a 4-GPU node so scale-up to 4 has headroom; the Ray
        # DP backend places the new DP-worker actors on the free GPUs.
        script_args=["--dp-size", "2"],
        marks=[
            pytest.mark.vllm,
            pytest.mark.gpu_4,
            pytest.mark.elastic_ep,
            pytest.mark.unified,
            pytest.mark.nightly,
            # MoE weights + two live reconfigures (scale up then down); generous
            # ceiling over model download + engine-core respawn.
            pytest.mark.timeout(1800),
            pytest.mark.model("Qwen/Qwen3-30B-A3B"),
            # DISABLED pending CI hardware. Elastic EP needs a real MoE with EPLB
            # (`--enable-elastic-ep` requires `--enable-eplb`), and vLLM's EPLB
            # only supports unquantized or FP8 experts — GPTQ/AWQ raise
            # `NotImplementedError: EPLB is not supported ...`. The smallest
            # EPLB-capable MoE (Qwen3-30B-A3B, bf16) needs ~57 GiB of weights per
            # replica at TP=1, and DP=2->4 puts a full replica on each GPU, so it
            # requires ~80 GiB/GPU. CI's only 4-GPU runner
            # (prod-tester-amd-gpu-4-v2) has 24 GiB GPUs and OOMs at model load.
            # Locally validated end-to-end on H200 GPUs with the pinned vLLM
            # 0.24.0: initial serving, DP=2->4->2, and inference after each scale
            # transition all pass. Keep this skipped until CI has a >=80 GiB
            # 4-GPU runner.
            pytest.mark.skip(
                reason="Locally passes DP=2->4->2 on H200 with vLLM 0.24.0, but "
                "needs a >=80 GiB 4-GPU runner (real EPLB MoE is ~57 GiB/GPU at "
                "TP=1); CI's only 4-GPU runner is 24 GiB and OOMs at model load."
            ),
        ],
        # MoE that exercises real expert parallelism; must match elastic_ep.sh's
        # default so payload model-name injection resolves to the served model.
        model="Qwen/Qwen3-30B-A3B",
        request_payloads=[
            # Serving works at the initial DP size...
            chat_payload_default(repeat_count=1),
            # ...scale up to 4 DP ranks and confirm generation survives...
            elastic_ep_scale_payload(new_data_parallel_size=4),
            # ...then scale back down to 2 and confirm generation survives again.
            elastic_ep_scale_payload(new_data_parallel_size=2),
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
            pytest.mark.timeout(600),  # 3x ~200s under new scheduler (3d1554f)
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
            pytest.mark.timeout(600),  # 3x ~199s multiproc, new scheduler (3d1554f)
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
    "aggregated_lmcache_mp": VLLMConfig(
        name="aggregated_lmcache_mp",
        directory=vllm_dir,
        script_name="agg_lmcache_mp.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.lmcache,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(3.8),
            pytest.mark.requested_vllm_kv_cache_bytes(
                1_119_388_000
            ),  # KV cache cap (2x safety over min=559_693_824)
            pytest.mark.timeout(640),  # 3x ~213s under new scheduler (3d1554f)
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        env={"LMCACHE_L1_SIZE_GB": "8"},
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            metric_payload_default(min_num_requests=6, backend="vllm"),
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
            pytest.mark.timeout(600),  # 3x ~199s under new scheduler (3d1554f)
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
            router_selection_chat_payload_default(),
            kv_events_metrics_payload(system_ports=[DefaultPort.SYSTEM2.value]),
        ],
        env={},
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
            # Repeated requests should show cache-aware routing in nvext.
            router_selection_chat_payload_default(repeat_count=3),
            # Also test with cached tokens payload to verify usage field
            router_cached_tokens_chat_payload(repeat_count=3),
        ],
        env={},
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
            # Verifies dynamo+backend can serve a model that ships NO chat
            # template, via the completions endpoint. The model is NOT
            # incidental: it must be a base model without a chat template.
            # TinyLlama-1.1B (intermediate base checkpoint) is a small Llama-
            # family base without a chat template (replaces deepseek-llm-7b-base,
            # 7B) -- keeps the coverage, cuts VRAM. TinyLlama-1.1B-Chat is
            # already used in the router e2e suite.
            # VRAM + KV cap profiled locally on an RTX 6000 Ada.
            pytest.mark.profiled_vram_gib(3.9),  # actual nvidia-smi peak
            pytest.mark.requested_vllm_kv_cache_bytes(
                530_432_000
            ),  # KV cache cap (2x safety over profiled min=265_216_000)
            pytest.mark.timeout(
                300
            ),  # 1.1B loads quickly; margin covers CI model download
            pytest.mark.post_merge,
        ],
        model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        # TinyLlama-1.1B caps at 2048 positions; agg.sh defaults to 4096.
        env={"MAX_MODEL_LEN": "2048"},
        script_args=[
            "--model",
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
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
    "multi_node_tp_headless_unified": VLLMConfig(
        name="multi_node_tp_headless_unified",
        directory=os.path.join(WORKSPACE_DIR, "tests/serve"),
        script_name="multi_node_tp_headless.sh",
        # --unified runs both nodes via dynamo.vllm.unified_main, so the
        # headless worker exercises unified_main -> run_dynamo_headless (the
        # unified backend's headless path, mock-only in unit tests).
        script_args=["--unified"],
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_2,
            pytest.mark.pre_merge,
            pytest.mark.unified,
            # No profiled_vram_gib / requested_vllm_kv_cache_bytes here: the
            # single-GPU Qwen3-0.6B values do not transfer to this TP=2 worker.
            # requested_vllm_kv_cache_bytes forces --kv-cache-memory-bytes +
            # --gpu-memory-utilization 0.01 onto BOTH ranks, which hangs the
            # headless multi-node startup past the timeout (CI confirmed). Matches
            # the legacy multi_node_tp_headless sibling, which is also unprofiled;
            # real TP=2 profiling is needed before VRAM markers can be added.
            # TODO: profile to get max_vram for the TP=2 headless topology.
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
            pytest.mark.timeout(220),  # 3x ~72s under new scheduler (3d1554f)
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
            pytest.mark.gpu_1,
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
    config = dataclasses.replace(
        vllm_config_test, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)


# ---------------------------------------------------------------------------
# Prefill drain on graceful shutdown, unified entry point. A concurrent burst
# gives the prefill worker in-flight work; it's then SIGTERMed mid-flight, and
# the test asserts the Rust Worker drove a graceful shutdown (drain -> cleanup).
# vLLM has no is_quiescent() override, so the drain waits the full budget.
#
# Timing: the launch script's wait_any_exit tears down the frontend and decode
# worker as soon as the drained prefill worker exits, so the burst's `ok >= 1`
# floor must be met inside SIGTERM + drain budget + cleanup. The 30s budget
# (matching the sglang/trtllm variants) and the small per-request decode at the
# call site keep that window winnable on L4-class CI GPUs; a 3s budget with
# 256-token decodes lost the race after the vLLM 0.24.0 bump (#11076).
# ---------------------------------------------------------------------------
_PREFILL_DRAIN_CONFIG = VLLMConfig(
    name="prefill_drain_unified",
    directory=vllm_dir,
    script_name="disagg_same_gpu.sh",
    script_args=["--unified"],
    marks=[],  # applied on the test function below
    model="Qwen/Qwen3-0.6B",
    delayed_start=10,
    health_check_workers=True,
    env={
        "DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS": "0",
        "DYN_PREFILL_DRAIN_TIMEOUT_S": "30",
        # Must exceed the drain budget by CLEANUP_RESERVE_S (5s) or the Rust
        # Worker caps the effective drain at timeout - reserve.
        "DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT": "60",
        # The unified decode worker disables its health canary by design
        # (NixlConnector has no local-only bypass), so its system /health is
        # gated on starting status, which defaults to NotReady. Mark it
        # ready-on-liveness so the harness's worker health check passes; the
        # prefill worker still gates on its real canary.
        "DYN_SYSTEM_STARTING_HEALTH_STATUS": "ready",
    },
    # Required by EngineConfig and used to add frontend readiness checks; the
    # burst issues its own requests.
    request_payloads=[chat_payload_default()],
)


@pytest.mark.vllm
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model("Qwen/Qwen3-0.6B")
@pytest.mark.profiled_vram_gib(7.3)
@pytest.mark.requested_vllm_kv_cache_bytes(1_023_525_000)
@pytest.mark.timeout(360)
@pytest.mark.post_merge
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_prefill_drain_unified(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    num_system_ports,
    predownload_models,
):
    """Fire a concurrent burst, SIGTERM the prefill worker mid-flight, and
    assert the Rust Worker drove graceful shutdown (drain -> cleanup). vLLM has
    no is_quiescent() override, so the framework drains prefill for the full
    budget (safe-by-default)."""
    config = dataclasses.replace(
        _PREFILL_DRAIN_CONFIG, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    # Short decodes so first completions land well inside the drain window on
    # slow CI GPUs; the full-size burst still keeps 96 requests in flight when
    # the SIGTERM lands.
    run_prefill_drain_deployment(
        config, request, ports=dynamo_dynamic_ports, burst_max_tokens=32
    )


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
    = first worker = GPU 0, index 1 = second worker = GPU 1).
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
@pytest.mark.gpu_2
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
    on 2-GPU CI nodes.
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
        script_name="agg_embed_multiworker.sh",
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
@pytest.mark.gpu_2
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
        script_name="agg_embed_multiworker.sh",
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
