# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
from dataclasses import dataclass, field

import pytest

from tests.serve.common import (
    SERVE_TEST_DIR,
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.constants import DefaultPort
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import (
    anthropic_messages_payload_default,
    anthropic_messages_stream_payload_default,
    chat_payload,
    chat_payload_default,
    completion_payload_default,
    embedding_payload,
    embedding_payload_default,
    metric_payload_default,
    responses_payload_default,
    responses_stream_payload_default,
)

logger = logging.getLogger(__name__)


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

# SGLang test configurations
# NOTE: pytest.mark.gpu_1 tests take ~167s (2m 47s) total to run sequentially (with models pre-cached)
# TODO: Now that these tests use dynamic ports and each config has a profiled_vram_gib marker,
# optimize the runtime by bin-packing multiple engine deployments in parallel on the same GPU.
# A future collector/launcher can sum profiled_vram_gib values to decide how many tests fit
# concurrently without exceeding available VRAM.
sglang_configs = {
    "aggregated": SGLangConfig(
        # Uses backend agg.sh (with metrics enabled) for testing standard
        # aggregated deployment with metrics collection
        name="aggregated",
        directory=sglang_dir,
        script_name="agg.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(
                3.7
            ),  # actual peak at recommended token count
            pytest.mark.requested_sglang_kv_tokens(
                96
            ),  # KV cache cap (2x safety over min=48)
            pytest.mark.timeout(195),  # profiled 33s on RTX 6000 Ada
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        env={},
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            responses_payload_default(),
            responses_stream_payload_default(),
            metric_payload_default(min_num_requests=6, backend="sglang"),
        ],
    ),
    "disaggregated": SGLangConfig(
        name="disaggregated",
        directory=sglang_dir,
        script_name="disagg.sh",
        marks=[
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
            pytest.mark.gpu_1,
            pytest.mark.pre_merge,
            pytest.mark.skip(reason="unstable"),
            # TODO: profile to get max_vram and timeout (currently skipped)
        ],
        model="Qwen/Qwen3-0.6B",
        delayed_start=30,
        env={},
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            # Validate dynamo_component_* and sglang:* metrics from prefill worker
            # (DefaultPort.SYSTEM1)
            metric_payload_default(
                min_num_requests=6,
                backend="sglang",
                port=DefaultPort.SYSTEM1.value,
            ),
            # Validate dynamo_component_* and sglang:* metrics from decode worker
            # (DefaultPort.SYSTEM2)
            metric_payload_default(
                min_num_requests=6,
                backend="sglang",
                port=DefaultPort.SYSTEM2.value,
            ),
        ],
    ),
    "kv_events": SGLangConfig(
        name="kv_events",
        directory=sglang_dir,
        script_name="agg_router.sh",
        marks=[
            pytest.mark.gpu_2,
            pytest.mark.pre_merge,
        ],  # TODO(gpu_2): profile max_vram, timeout, add markers (separate PR)
        model="Qwen/Qwen3-0.6B",
        env={
            "DYN_LOG": "dynamo_llm::kv_router::publisher=trace,dynamo_kv_router::scheduling::selector=info",
        },
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(
                expected_log=[
                    r"ZMQ listener .* received batch with \d+ events \(engine_seq=\d+(?:, [^)]*)?\)",
                    r"Event processor for worker_id \d+ processing event: Stored\(",
                    r"Selected worker: worker_type=\w+, worker_id=\d+ dp_rank=.*?, logit: ",
                ]
            )
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
    # NOTE: multimodal_epd.sh uses explicit --mem-fraction-static via DYN_ENCODE_GPU_MEM
    # / DYN_WORKER_GPU_MEM env vars. The profiler override distributes proportionally
    # but workers combined consistently use ~23.6 GiB regardless of fraction overrides.
    "multimodal_e_pd_qwen": SGLangConfig(
        # E/P/D architecture: Encode, Prefill, Decode workers all on GPU 0
        name="multimodal_e_pd_qwen",
        directory=sglang_dir,
        script_name="multimodal_epd.sh",
        marks=[
            pytest.mark.gpu_1,
            # No profiled_vram_gib: uses hard-coded --mem-fraction-static via
            # DYN_ENCODE_GPU_MEM / DYN_WORKER_GPU_MEM, so VRAM scales with GPU size.
            pytest.mark.timeout(210),  # profiled 35s on RTX 6000 Ada
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-VL-2B-Instruct",
        script_args=["--model", "Qwen/Qwen3-VL-2B-Instruct", "--single-gpu"],
        timeout=360,
        env={
            "DYN_ENCODE_GPU_MEM": "0.1",
            "DYN_WORKER_GPU_MEM": "0.4",
        },
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
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(16.1),  # actual profiled peak
            pytest.mark.requested_sglang_kv_tokens(
                1024
            ),  # KV cache cap (2x safety over min=512)
            pytest.mark.timeout(222),  # profiled 37s on RTX 6000 Ada
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
                expected_response=["image"],
                temperature=0.0,
                max_tokens=100,
            )
        ],
    ),
    "multimodal_agg_qwen": SGLangConfig(
        # Tests single-process aggregated multimodal inference using DecodeWorkerHandler
        # with in-process vision encoding (no separate encode worker)
        name="multimodal_agg_qwen",
        directory=sglang_dir,
        script_name="agg.sh",
        marks=[
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
        name="video_agg_qwen",
        directory=sglang_dir,
        script_name="agg_vision.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(13.3),  # same as multimodal_e_pd_qwen
            pytest.mark.timeout(360),
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen2-VL-7B-Instruct",
        script_args=[
            "--model-path",
            "Qwen/Qwen2-VL-7B-Instruct",
            "--mem-fraction-static",
            "0.8",
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
    "embedding_agg": SGLangConfig(
        name="embedding_agg",
        directory=sglang_dir,
        script_name="agg_embed.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(
                9.8
            ),  # actual peak at recommended token count
            pytest.mark.requested_sglang_kv_tokens(
                128
            ),  # KV cache cap (2x safety over min=64)
            pytest.mark.timeout(147),  # profiled 24s on RTX 6000 Ada
            pytest.mark.pre_merge,
            pytest.mark.nightly,
        ],
        model="Qwen/Qwen3-Embedding-4B",
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
        ],
    ),
    "completions_only": SGLangConfig(
        name="completions_only",
        directory=sglang_dir,
        script_name="agg.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.profiled_vram_gib(
                14.7
            ),  # actual peak at recommended token count
            pytest.mark.requested_sglang_kv_tokens(
                64
            ),  # KV cache cap (2x safety over min=32)
            pytest.mark.timeout(341),  # profiled 57s on RTX 6000 Ada
            pytest.mark.post_merge,
        ],
        model="deepseek-ai/deepseek-llm-7b-base",
        script_args=[
            "--model-path",
            "deepseek-ai/deepseek-llm-7b-base",
            "--dyn-endpoint-types",
            "completions",
        ],
        request_payloads=[
            completion_payload_default(),
        ],
    ),
    "anthropic_messages": SGLangConfig(
        name="anthropic_messages",
        directory=sglang_dir,
        script_name="agg.sh",
        marks=[
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
):
    """Test SGLang deployment scenarios using common helpers"""
    assert (
        num_system_ports >= 2
    ), "serve tests require at least SYSTEM_PORT1 + SYSTEM_PORT2"
    config = dataclasses.replace(
        sglang_config_test, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)


@pytest.mark.e2e
@pytest.mark.sglang
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
