# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import pytest

from tests.serve.common import (
    SERVE_TEST_DIR,
    WORKSPACE_DIR,
    params_with_model_mark,
    run_prefill_drain_deployment,
    run_serve_deployment,
)
from tests.serve.lora_utils import DEFAULT_LORA_REPO, MinioLoraConfig
from tests.serve.multimodal_profiles.sglang import (
    SGLANG_MULTIMODAL_PROFILES,
    SGLANG_TOPOLOGY_SCRIPTS,
)
from tests.utils.constants import DefaultPort
from tests.utils.engine_process import EngineConfig
from tests.utils.multimodal import make_image_payload_b64, make_multimodal_configs
from tests.utils.payload_builder import (
    anthropic_messages_payload_default,
    anthropic_messages_stream_payload_default,
    chat_payload,
    chat_payload_default,
    completion_payload_default,
    embedding_payload,
    embedding_payload_default,
    guided_decoding_chat_payload_default,
    kv_events_metrics_payload,
    metric_payload_default,
    responses_payload_default,
    responses_stream_payload_default,
    router_selection_chat_payload_default,
)
from tests.utils.payloads import (
    ImageGenerationPayload,
    LoraTestChatPayload,
    VideoGenerationPayload,
)

logger = logging.getLogger(__name__)


def _is_cuda13() -> bool:
    v = os.environ.get("CUDA_VERSION", "")
    return v.startswith("13")


@dataclass
class SGLangConfig(EngineConfig):
    """Configuration for SGLang test scenarios"""

    stragglers: list[str] = field(default_factory=lambda: ["SGLANG:EngineCore"])


sglang_dir = os.environ.get("SGLANG_DIR") or os.path.join(
    WORKSPACE_DIR, "examples/backends/sglang"
)
REMOTE_VIDEO_TEST_URI = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
)

# Generated multimodal configs from profile definitions (mirrors test_vllm.py).
# Each profile expands into one config per MmCase per topology with the
# appropriate marks (gpu_*, timeout, pre/post_merge, requested_sglang_kv_tokens).
_mm_configs: dict[str, SGLangConfig] = {}
for _profile in SGLANG_MULTIMODAL_PROFILES:
    _mm_configs.update(
        make_multimodal_configs(
            _profile, SGLangConfig, sglang_dir, SGLANG_TOPOLOGY_SCRIPTS
        )
    )

# SGLang test configurations
# NOTE: pytest.mark.gpu_1 tests take ~167s (2m 47s) total to run sequentially (with models pre-cached)
# TODO: Now that these tests use dynamic ports and each config has a profiled_vram_gib marker,
# optimize the runtime by bin-packing multiple engine deployments in parallel on the same GPU.
# A future collector/launcher can sum profiled_vram_gib values to decide how many tests fit
# concurrently without exceeding available VRAM.
sglang_configs = {
    **_mm_configs,
    "aggregated": SGLangConfig(
        # Uses backend agg.sh (with metrics enabled) for testing standard
        # aggregated deployment with metrics collection
        name="aggregated",
        directory=sglang_dir,
        script_name="agg.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(
                3.7
            ),  # actual peak at recommended token count
            pytest.mark.requested_sglang_kv_tokens(
                2048
            ),  # >= prompt(~16) + max_tokens(1000) + scheduler reserve;
            # SGLang 0.5.11 silently hangs (no scheduler activity, no error)
            # when prompt+max_tokens nears max_total_tokens. Bisected hang
            # threshold ~1040 for these payloads; 2048 leaves headroom.
            pytest.mark.timeout(360),  # 3x ~119s (sglang gpu_1 log)
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        env={},
        frontend_port=DefaultPort.FRONTEND.value,
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
            responses_payload_default(),
            responses_stream_payload_default(),
            guided_decoding_chat_payload_default(),
            metric_payload_default(min_num_requests=6, backend="sglang"),
        ],
    ),
    "aggregated_unified": SGLangConfig(
        name="aggregated_unified",
        directory=sglang_dir,
        script_name="agg.sh",
        script_args=["--unified"],
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(3.7),
            pytest.mark.requested_sglang_kv_tokens(2048),  # see "aggregated" above
            pytest.mark.timeout(340),  # 3x ~111s (sglang gpu_1 log)
            pytest.mark.pre_merge,
            pytest.mark.unified,
        ],
        model="Qwen/Qwen3-0.6B",
        env={},
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            guided_decoding_chat_payload_default(),
        ],
    ),
    "disaggregated": SGLangConfig(
        name="disaggregated",
        directory=sglang_dir,
        script_name="disagg.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_2,
            pytest.mark.pre_merge,
        ],  # TODO(gpu_2): profile max_vram, timeout, add markers (separate PR)
        model="Qwen/Qwen3-0.6B",
        env={},
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "disaggregated_same_gpu": SGLangConfig(
        # Uses disagg_same_gpu.sh for single-GPU disaggregated testing
        # Validates metrics from both prefill (DefaultPort.SYSTEM1) and decode
        # (DefaultPort.SYSTEM2) workers
        name="disaggregated_same_gpu",
        directory=sglang_dir,
        script_name="disagg_same_gpu.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(
                13.0
            ),  # observed ~12.1 GiB with kv-tokens; rounded up
            pytest.mark.requested_sglang_kv_tokens(
                37472
            ),  # KV cache cap (2x safety over min=18736)
            # Local repro took ~289s wall time with worker readiness reaching
            # "ready" at ~176s on a warm-cache RTX 6000 Ada.
            pytest.mark.timeout(470),  # 3x ~155s (sglang gpu_1 log)
            pytest.mark.pre_merge,
            pytest.mark.skipif(
                _is_cuda13(),
                reason="torch-memory-saver preload .so links libcudart.so.12, missing in cuda13 images",
            ),
        ],
        model="Qwen/Qwen3-0.6B",
        delayed_start=10,
        health_check_workers=True,
        env={},
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            # Disagg workers expose fewer sglang:* metrics (~14 vs ~25 for aggregated)
            # because each only runs half the scheduler pipeline.
            metric_payload_default(
                min_num_requests=6,
                backend="sglang_disagg",
                port=DefaultPort.SYSTEM1.value,
            ),
            metric_payload_default(
                min_num_requests=6,
                backend="sglang_disagg",
                port=DefaultPort.SYSTEM2.value,
            ),
        ],
    ),
    "disaggregated_same_gpu_chat_processor": SGLangConfig(
        # Same as disaggregated_same_gpu but routes chat through the Python
        # sglang_processor (DYN_CHAT_PROCESSOR=sglang) instead of the default
        # Rust pre/post processor.
        name="disaggregated_same_gpu_chat_processor",
        directory=sglang_dir,
        script_name="disagg_same_gpu.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(13.0),
            pytest.mark.requested_sglang_kv_tokens(37472),
            pytest.mark.timeout(470),  # 3x ~156s (sglang gpu_1 log)
            pytest.mark.post_merge,
            pytest.mark.skipif(
                _is_cuda13(),
                reason="torch-memory-saver preload .so links libcudart.so.12, missing in cuda13 images",
            ),
        ],
        model="Qwen/Qwen3-0.6B",
        delayed_start=10,
        health_check_workers=True,
        env={"DYN_CHAT_PROCESSOR": "sglang"},
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "disaggregated_same_gpu_chat_processor_kv_router": SGLangConfig(
        # sglang Python chat processor + KV router.
        name="disaggregated_same_gpu_chat_processor_kv_router",
        directory=sglang_dir,
        script_name="disagg_same_gpu.sh",
        marks=[
            pytest.mark.router,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(13.0),
            pytest.mark.requested_sglang_kv_tokens(37472),
            pytest.mark.timeout(470),  # 3x ~151s (sglang gpu_1 log)
            pytest.mark.post_merge,
            pytest.mark.skipif(
                _is_cuda13(),
                reason="torch-memory-saver preload .so links libcudart.so.12, missing in cuda13 images",
            ),
        ],
        model="Qwen/Qwen3-0.6B",
        delayed_start=10,
        health_check_workers=True,
        env={
            "DYN_CHAT_PROCESSOR": "sglang",
            "DYN_ROUTER_MODE": "kv",
            # Deterministic hash for KV event IDs.
            "PYTHONHASHSEED": "0",
        },
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "kv_events": SGLangConfig(
        name="kv_events",
        directory=sglang_dir,
        script_name="agg_router.sh",
        marks=[
            pytest.mark.router,
            pytest.mark.gpu_2,
            pytest.mark.pre_merge,
        ],  # TODO(gpu_2): profile max_vram, timeout, add markers (separate PR)
        model="Qwen/Qwen3-0.6B",
        env={},
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            router_selection_chat_payload_default(),
            kv_events_metrics_payload(system_ports=[DefaultPort.SYSTEM2.value]),
        ],
    ),
    "template_verification": SGLangConfig(
        # Tests custom jinja template preprocessing by verifying the template
        # marker 'CUSTOM_TEMPLATE_ACTIVE|' is applied to user messages.
        # The backend (launch/template_verifier.*) checks for this marker
        # and returns "Successfully Applied Chat Template" if found.
        # Uses SERVE_TEST_DIR (not sglang_dir) because template_verifier.sh/.py
        # are test-specific mock scripts in tests/serve/launch/
        name="template_verification",
        directory=SERVE_TEST_DIR,  # special directory for test-specific scripts
        script_name="template_verifier.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(0.0),  # no GPU model load
            pytest.mark.timeout(120),  # profiled 12s on RTX 6000 Ada
            pytest.mark.pre_merge,
            pytest.mark.nightly,
        ],
        model="Qwen/Qwen3-0.6B",
        env={},
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(
                expected_response=["Successfully Applied Chat Template"]
            )
        ],
    ),
    # NOTE: Pack all workers on 1 GPU for lower CI resource requirements.
    # KV size is set by requested_sglang_kv_tokens; no fraction overrides.
    "multimodal_e_pd_qwen": SGLangConfig(
        # E/P/D architecture: Encode, Prefill, Decode workers all on GPU 0
        name="multimodal_e_pd_qwen",
        directory=sglang_dir,
        script_name="multimodal_epd.sh",
        marks=[
            pytest.mark.multimodal,
            pytest.mark.gpu_1,
            # Bisected with tests/utils/profile_pytest.py: min=1104, 2x=2208.
            # Keep this unprofiled for now so the GPU-parallel stage leaves it
            # in the sequential stage; parallel E/P/D runs can trip UCX mm
            # transport init on larger single-GPU runners.
            # pytest.mark.profiled_vram_gib(11.1),
            pytest.mark.requested_sglang_kv_tokens(2208),
            pytest.mark.timeout(206),  # profiled 34s on RTX 6000 Ada
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-VL-2B-Instruct",
        script_args=["--model", "Qwen/Qwen3-VL-2B-Instruct", "--single-gpu"],
        timeout=360,
        env={},
        frontend_port=DefaultPort.FRONTEND.value,
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
                # NOTE: The response text may mention 'bus', 'train', 'streetcar', etc.
                # so we need something consistently found in the response, or a different
                # approach to validation for this test to be stable.
                expected_response=["image"],
                temperature=0.0,
                max_tokens=100,
            )
        ],
    ),
    "multimodal_disagg_qwen": SGLangConfig(
        # E/P/D architecture: Encode, Prefill, Decode workers all on GPU 0
        name="multimodal_disagg_qwen",
        directory=sglang_dir,
        script_name="multimodal_disagg.sh",
        marks=[
            pytest.mark.multimodal,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(16.1),  # actual profiled peak
            pytest.mark.requested_sglang_kv_tokens(
                1024
            ),  # KV cache cap (2x safety over min=512)
            pytest.mark.timeout(280),  # 3x ~92s (sglang gpu_1 log)
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-VL-2B-Instruct",
        script_args=["--model", "Qwen/Qwen3-VL-2B-Instruct", "--single-gpu"],
        timeout=360,
        env={},
        frontend_port=DefaultPort.FRONTEND.value,
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
                expected_response=["image", "bus", "train", "streetcar"],
                temperature=0.0,
                max_tokens=100,
            )
        ],
    ),
    "multimodal_agg_fd_qwen": SGLangConfig(
        # Aggregated multimodal with --frontend-decoding: the Rust frontend
        # decodes the image and ships pre-decoded pixels via NIXL RDMA;
        # the SGLang worker consumes Decoded items through ImageLoader and
        # hands PIL Images to sgl.Engine.async_generate(image_data=...).
        # Mirrors the vLLM FD pattern at
        # tests/serve/multimodal_profiles/vllm.py (Qwen3.5-0.8B "b64_frontend_decoding").
        name="multimodal_agg_fd_qwen",
        directory=sglang_dir,
        script_name="agg_vision.sh",
        marks=[
            pytest.mark.multimodal,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(4.7),  # parity with vLLM Qwen3.5-0.8B
            # 4096 covers the b64 PNG image-token expansion (~2198 tokens
            # for our 1999x1125 LFS test asset under Qwen3.5-0.8B's vision
            # processor) + 100-token max response + headroom.
            # TODO: bisect via tests/utils/profile_pytest.py for a tighter bound.
            pytest.mark.requested_sglang_kv_tokens(4096),
            pytest.mark.timeout(320),  # 3x ~104s (sglang gpu_1 log)
            # post_merge: NIXL stubs outside docker can lack the Decoded
            # transport path. Same gating as vLLM's FD case
            # (tests/serve/multimodal_profiles/vllm.py:67-70).
            pytest.mark.post_merge,
        ],
        model="Qwen/Qwen3.5-0.8B",
        script_args=[
            "--model-path",
            "Qwen/Qwen3.5-0.8B",
            # Hybrid Mamba/full-attention VL: SGLang's MambaRadixCache v1
            # asserts page_size == 1. The launch script defaults to 16.
            "--page-size",
            "1",
            "--frontend-decoding",
        ],
        delayed_start=0,
        timeout=360,
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            # Inline-base64 PNG: exercises strip_inline_data_urls in the
            # Rust frontend + NIXL RDMA transfer of decoded pixels — the
            # path that distinguishes FD from the plain URL path.
            make_image_payload_b64(["green"]),
        ],
    ),
    "multimodal_agg_qwen": SGLangConfig(
        # Tests single-process aggregated multimodal inference using DecodeWorkerHandler
        # with in-process vision encoding (no separate encode worker)
        name="multimodal_agg_qwen",
        directory=sglang_dir,
        script_name="agg.sh",
        marks=[
            pytest.mark.multimodal,
            pytest.mark.skip(
                reason="Nightly CI failure: https://linear.app/nvidia/issue/DYN-2602"
            ),
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(
                19.1
            ),  # actual peak at recommended token count
            pytest.mark.requested_sglang_kv_tokens(
                768
            ),  # KV cache cap (2x safety over min=384)
            pytest.mark.timeout(182),  # profiled 30s on RTX 6000 Ada
            pytest.mark.pre_merge,
            pytest.mark.nightly,
        ],
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        script_args=[
            "--model-path",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "--chat-template",
            "qwen2-vl",
        ],
        delayed_start=0,
        timeout=360,
        frontend_port=DefaultPort.FRONTEND.value,
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
                expected_response=["image"],
                temperature=0.0,
                max_tokens=100,
            )
        ],
    ),
    "video_agg_qwen": SGLangConfig(
        # Tests aggregated video inference using DecodeWorkerHandler
        # with in-process vision encoding (no separate encode worker).
        # Reuses agg_vision.sh because image and video share the same aggregated
        # multimodal SGLang request path.
        #
        # VRAM is bounded via requested_sglang_kv_tokens (deterministic, parallel-safe)
        # rather than --mem-fraction-static (fraction of GPU, not portable across GPU
        # sizes and breaks the VRAM-aware scheduler).
        name="video_agg_qwen",
        directory=sglang_dir,
        script_name="agg_vision.sh",
        marks=[
            pytest.mark.multimodal,
            pytest.mark.gpu_1,
            # Bisected with tests/utils/profile_pytest.py: minimum = 4368
            # tokens, 2x safety = 8736. Peak 20.5 GiB at 8736 tokens. Without
            # the cap, sglang's default 65% fraction allocates ~278k tokens
            # and peaks at ~35 GiB (won't fit L4).
            pytest.mark.profiled_vram_gib(20.5),
            pytest.mark.requested_sglang_kv_tokens(8736),
            pytest.mark.timeout(390),  # 3x ~127s (sglang gpu_1 log)
            pytest.mark.post_merge,
        ],
        model="Qwen/Qwen2-VL-7B-Instruct",
        script_args=[
            "--model-path",
            "Qwen/Qwen2-VL-7B-Instruct",
        ],
        timeout=360,
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload(
                [
                    {"type": "text", "text": "Describe the video in detail"},
                    {
                        "type": "video_url",
                        "video_url": {"url": REMOTE_VIDEO_TEST_URI},
                    },
                ],
                repeat_count=1,
                expected_response=["guitar", "tablet", "draw"],
                temperature=0.0,
                max_tokens=100,
            )
        ],
    ),
    "video_e_pd_qwen": SGLangConfig(
        # Tests E/PD video inference path using a separate encode worker
        # and a multimodal PD worker on a single GPU for CI coverage.
        name="video_e_pd_qwen",
        directory=sglang_dir,
        script_name="multimodal_epd.sh",
        marks=[
            pytest.mark.multimodal,
            pytest.mark.gpu_1,
            # No profiled_vram_gib: multimodal_epd.sh uses explicit
            # --mem-fraction-static via DYN_ENCODE_GPU_MEM / DYN_WORKER_GPU_MEM.
            pytest.mark.timeout(360),
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-VL-2B-Instruct",
        script_args=[
            "--model",
            "Qwen/Qwen3-VL-2B-Instruct",
            "--chat-template",
            "qwen2-vl",
            "--single-gpu",
            "--multimodal-embedding-cache-capacity-gb",
            "0.1",
        ],
        timeout=360,
        env={
            "DYN_ENCODE_GPU_MEM": "0.1",
            "DYN_WORKER_GPU_MEM": "0.4",
            "DYN_SGL_EMBEDDING_TRANSFER_MODE": "local",
        },
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload(
                [
                    {"type": "text", "text": "Describe the video in detail"},
                    {
                        "type": "video_url",
                        "video_url": {"url": REMOTE_VIDEO_TEST_URI},
                    },
                ],
                repeat_count=1,
                expected_response=["guitar", "tablet", "draw"],
                temperature=0.0,
                max_tokens=100,
            ),
            chat_payload(
                [
                    {"type": "text", "text": "Describe the video in detail"},
                    {
                        "type": "video_url",
                        "video_url": {"url": REMOTE_VIDEO_TEST_URI},
                    },
                ],
                repeat_count=1,
                expected_response=["guitar", "tablet", "draw"],
                expected_log=["Embedding cache hit for VIDEO URL index 0"],
                temperature=0.0,
                max_tokens=100,
            ),
        ],
    ),
    "embedding_agg": SGLangConfig(
        name="embedding_agg",
        directory=sglang_dir,
        script_name="agg_embed.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_1,
            # Qwen3-Embedding-0.6B runs the same assertions as the 4B variant
            # (batch, Matryoshka-128, base64). Profiled locally.
            pytest.mark.profiled_vram_gib(3.0),  # actual nvidia-smi peak
            pytest.mark.requested_sglang_kv_tokens(
                128
            ),  # KV cache cap (peak is flat vs token count for embeddings)
            # Generous timeout: CI model download dominates startup on a cold runner.
            pytest.mark.timeout(300),
            pytest.mark.pre_merge,
            pytest.mark.nightly,
        ],
        model="Qwen/Qwen3-Embedding-0.6B",
        # agg_embed.sh defaults to the 4B model; model= alone only drives
        # predownload, so set the served model here too.
        script_args=["--model-path", "Qwen/Qwen3-Embedding-0.6B"],
        delayed_start=0,
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            # Test default payload with multiple inputs
            embedding_payload_default(
                repeat_count=2,
                expected_response=["Generated 2 embeddings with dimension"],
            ),
            # Test single string input
            embedding_payload(
                input_text="Hello, world!",
                repeat_count=1,
                expected_response=["Generated 1 embeddings with dimension"],
            ),
            # Test multiple string inputs
            embedding_payload(
                input_text=[
                    "The quick brown fox jumps over the lazy dog.",
                    "Machine learning is transforming technology.",
                    "Natural language processing enables computers to understand text.",
                ],
                repeat_count=1,
                expected_response=["Generated 3 embeddings with dimension"],
            ),
            # Test `dimensions` truncation (Matryoshka). Qwen3-Embedding-0.6B
            # has a hidden dim (1024) well above 128, so the truncated vector
            # should be exactly 128 floats long.
            embedding_payload(
                input_text="Hello, world!",
                repeat_count=1,
                expected_response=["Generated 1 embeddings with dimension 128"],
                extra_body={"dimensions": 128},
            ),
            # Test ``encoding_format=base64`` end-to-end. The Python
            # handler base64-encodes the f32 byte buffer; the Rust
            # frontend deserializes it as a string; the validator decodes
            # it back and asserts the f32 count matches ``dimensions``.
            embedding_payload(
                input_text="Hello, world!",
                repeat_count=1,
                expected_response=["Generated 1 embeddings with dimension 128"],
                extra_body={"dimensions": 128, "encoding_format": "base64"},
            ),
        ],
    ),
    "completions_only": SGLangConfig(
        name="completions_only",
        directory=sglang_dir,
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
            # Profiled locally on an RTX 6000 Ada at the 2048-token KV cap below.
            pytest.mark.profiled_vram_gib(3.9),  # actual nvidia-smi peak
            pytest.mark.requested_sglang_kv_tokens(
                2048
            ),  # >= prompt(~16) + max_tokens(1000) + scheduler reserve;
            # SGLang 0.5.11 silently hangs when prompt+max_tokens nears
            # max_total_tokens (bisected ~1040 for these payloads). Matches
            # the "aggregated" config above.
            pytest.mark.timeout(300),  # 1.1B loads quickly; CI margin
            pytest.mark.post_merge,
        ],
        model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        script_args=[
            "--model-path",
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            "--dyn-endpoint-types",
            "completions",
        ],
        request_payloads=[
            completion_payload_default(),
        ],
    ),
    # ── Diffusion pre_merge smoke tests ─────────────────────────────────
    "diffusion_t2i_z_image_turbo": SGLangConfig(
        name="diffusion_t2i_z_image_turbo",
        directory=sglang_dir,
        script_name="image_diffusion.sh",
        script_args=["--model-path", "Tongyi-MAI/Z-Image-Turbo"],
        marks=[
            pytest.mark.multimodal,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(19.3),
            pytest.mark.requested_sglang_vram_gib(19.3),
            pytest.mark.timeout(240),
            pytest.mark.nightly,
        ],
        model="Tongyi-MAI/Z-Image-Turbo",
        env={},
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            ImageGenerationPayload(
                body={
                    "prompt": "A red apple on a white table",
                    "size": "512x512",
                    "response_format": "url",
                    "nvext": {"num_inference_steps": 4},
                },
                repeat_count=1,
                expected_response=[],
                expected_log=[],
            ),
        ],
    ),
    "diffusion_t2v_wan_1_3b": SGLangConfig(
        name="diffusion_t2v_wan_1_3b",
        directory=sglang_dir,
        script_name="text-to-video-diffusion.sh",
        script_args=[
            "--wan-size",
            "1b",
            "--num-inference-steps",
            "3",
            "--num-frames",
            "9",
            "--height",
            "256",
            "--width",
            "256",
        ],
        marks=[
            pytest.mark.multimodal,
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(17.6),
            pytest.mark.requested_sglang_vram_gib(17.6),
            pytest.mark.timeout(180),
            pytest.mark.nightly,
        ],
        model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        env={},
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            VideoGenerationPayload(
                body={
                    "prompt": "A dog running on a beach",
                    "size": "256x256",
                    "response_format": "url",
                    "nvext": {
                        "num_inference_steps": 3,
                        "num_frames": 9,
                    },
                },
                repeat_count=1,
                expected_response=[],
                expected_log=[],
            ),
        ],
    ),
    "diffusion_llada": SGLangConfig(
        # Diffusion language model (LLaDA2.0): text generation via iterative
        # refinement rather than autoregressive decoding. Unlike the t2i/t2v
        # diffusion configs above (image/video generation), this serves plain
        # text through /v1/chat/completions, so it uses a chat payload.
        name="diffusion_llada",
        directory=sglang_dir,
        script_name="diffusion_llada.sh",
        # diffusion_llada.sh forwards "$@", so VRAM is bounded from the test
        # (not the example default). --mem-fraction-static 0.4 is TOO LOW: the
        # sglang scheduler OOMs at pool sizing ("Not enough memory. Please try
        # to increase --mem-fraction-static") because LLaDA2.0-mini-preview
        # weights + diffusion activation need a large absolute reservation.
        # 0.7 boots cleanly; on the 80 GiB H100 lane that is ~56 GiB. This model
        # far exceeds the 24 GiB gpu_1 lane, so it runs on the H100 lane
        # (pytest.mark.h100, aws-dev-02 runner).
        script_args=["--mem-fraction-static", "0.7"],
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.h100,
            # --mem-fraction-static 0.7 makes sglang reserve ~0.7*80 = 56 GiB on
            # the H100. This is the reservation the VRAM-aware packer must budget
            # (not the ~32 GiB transient peak), so co-packing another large test
            # is correctly prevented.
            pytest.mark.profiled_vram_gib(56.0),
            # Diffusion decoding is iterative; the 32-token H100 smoke runs ~135s.
            # Keep generous headroom for cold starts / model download.
            pytest.mark.timeout(900),
            # Runs on the nightly H100 lane (sglang-h100-test in nightly-ci.yml),
            # not PR/post-merge: the model needs >24 GiB so it can't use the
            # standard gpu_1 lanes.
            pytest.mark.nightly,
        ],
        model="inclusionAI/LLaDA2.0-mini-preview",
        env={},
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            # Diffusion text output is non-deterministic; accept any non-empty
            # response (empty expected_response) and just exercise the path.
            chat_payload(
                "What is the capital of France? Answer in one word.",
                repeat_count=1,
                expected_response=[],
                # Keep the smoke tiny: diffusion decode cost scales with tokens.
                max_tokens=32,
                temperature=0.0,
            ),
        ],
    ),
    "anthropic_messages": SGLangConfig(
        name="anthropic_messages",
        directory=sglang_dir,
        script_name="agg.sh",
        marks=[
            pytest.mark.core,
            pytest.mark.gpu_1,
            pytest.mark.post_merge,
            pytest.mark.timeout(240),
            pytest.mark.skip(reason="DYN-2261"),
            # TODO: profile once DYN-2261 is fixed (uses agg.sh, profiler works)
        ],
        model="Qwen/Qwen3-0.6B",
        env={"DYN_ENABLE_ANTHROPIC_API": "1"},
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            anthropic_messages_payload_default(),
            anthropic_messages_stream_payload_default(),
        ],
    ),
}


@pytest.fixture(params=params_with_model_mark(sglang_configs))
def sglang_config_test(request):
    """Fixture that provides different SGLang test configurations"""
    return sglang_configs[request.param]


@pytest.mark.e2e
@pytest.mark.sglang
# Use 2 system ports because some `sglang_configs` validate metrics on multiple ports.
# This test iterates over all configs via `sglang_config_test`.
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_sglang_deployment(
    sglang_config_test,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    num_system_ports,
    predownload_models,
    image_server,
):
    """Test SGLang deployment scenarios using common helpers"""
    assert (
        num_system_ports >= 2
    ), "serve tests require at least SYSTEM_PORT1 + SYSTEM_PORT2"
    config = dataclasses.replace(
        sglang_config_test, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)


# ---------------------------------------------------------------------------
# Prefill drain on graceful shutdown, unified entry point. A concurrent burst
# gives the prefill worker in-flight work; it's then SIGTERMed mid-flight, and
# the test asserts the Rust Worker drove a graceful shutdown (drain -> cleanup).
# Also covers two SGLang specifics: the signal guard keeping SGLang's SIGTERM
# handler from preempting the Worker, and is_quiescent() counting both the
# bootstrap and completed prefill-stream paths.
# ---------------------------------------------------------------------------
_PREFILL_DRAIN_CONFIG = SGLangConfig(
    name="prefill_drain_unified",
    directory=sglang_dir,
    script_name="disagg_same_gpu.sh",
    script_args=["--unified"],
    marks=[],  # applied on the test function below
    model="Qwen/Qwen3-0.6B",
    delayed_start=10,
    health_check_workers=True,
    env={
        "DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS": "0",
        # Generous budget so the prefill queue can drain (is_quiescent -> True)
        # within it rather than always timing out.
        "DYN_PREFILL_DRAIN_TIMEOUT_S": "30",
        "DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT": "60",
        # Decode worker may disable its health canary; mark ready-on-liveness so
        # the harness's worker health check passes (canary-having workers still
        # gate on their canary).
        "DYN_SYSTEM_STARTING_HEALTH_STATUS": "ready",
        # torch-memory-saver links libcudart.so.12 (absent on the cu13 image);
        # the 48 GiB GPU fits both workers unpacked, so disable it.
        "DYN_SGLANG_DISABLE_MEMORY_SAVER": "1",
    },
    request_payloads=[chat_payload_default()],
)


@pytest.mark.sglang
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model("Qwen/Qwen3-0.6B")
@pytest.mark.profiled_vram_gib(13.0)
@pytest.mark.requested_sglang_kv_tokens(37472)
@pytest.mark.timeout(470)
@pytest.mark.post_merge
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_prefill_drain_unified(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    num_system_ports,
    predownload_models,
):
    """Burst + mid-flight prefill SIGTERM; assert the Rust Worker drove
    graceful shutdown (drain -> cleanup) — proving the signal guard and the
    is_quiescent() bootstrap-path fix both hold."""
    config = dataclasses.replace(
        _PREFILL_DRAIN_CONFIG, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_prefill_drain_deployment(config, request, ports=dynamo_dynamic_ports)


@pytest.mark.e2e
@pytest.mark.sglang
@pytest.mark.core
@pytest.mark.gpu_2
@pytest.mark.nightly
@pytest.mark.skip(
    reason="Requires 4 GPUs - enable when hardware is consistently available"
)
def test_sglang_disagg_dp_attention(
    request, runtime_services_dynamic_ports, dynamo_dynamic_ports, predownload_models
):
    """Test sglang disaggregated with DP attention (requires 4 GPUs)"""

    # Kept for reference; this test uses a different launch path and is skipped


# ── LoRA Tests ──────────────────────────────────────────────────────────────

lora_dir = os.path.join(sglang_dir, "launch/lora")


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


@pytest.mark.sglang
@pytest.mark.core
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model("Qwen/Qwen3-0.6B")
@pytest.mark.model(DEFAULT_LORA_REPO)
@pytest.mark.profiled_vram_gib(4.7)
@pytest.mark.requested_sglang_kv_tokens(2848)
@pytest.mark.timeout(240)  # 3x ~79s (sglang gpu_1 log)
@pytest.mark.pre_merge
def test_sglang_lora_aggregated(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    minio_lora_service,
    dynamo_dynamic_ports,
):
    """
    Test LoRA inference with aggregated SGLang deployment.

    This test:
    1. Uses MinIO fixture to provide S3-compatible storage with uploaded LoRA
    2. Starts SGLang with LoRA support enabled
    3. Loads the LoRA adapter via system API
    4. Runs inference with the LoRA model
    """
    minio_config: MinioLoraConfig = minio_lora_service

    lora_payload = lora_chat_payload(
        lora_name=minio_config.lora_name,
        s3_uri=minio_config.get_s3_uri(),
        system_port=DefaultPort.SYSTEM1.value,
        repeat_count=2,
    )

    config = SGLangConfig(
        name="test_sglang_lora_aggregated",
        directory=sglang_dir,
        script_name="lora/agg_lora.sh",
        marks=[],
        model="Qwen/Qwen3-0.6B",
        timeout=158,
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
