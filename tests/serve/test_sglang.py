# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import (
    chat_payload,
    chat_payload_default,
    completion_payload_default,
    embedding_payload,
    embedding_payload_default,
    metric_payload_default,
)

logger = logging.getLogger(__name__)


@dataclass
class SGLangConfig(EngineConfig):
    """Configuration for SGLang test scenarios"""

    stragglers: list[str] = field(default_factory=lambda: ["SGLANG:EngineCore"])


sglang_dir = os.environ.get("SGLANG_DIR") or os.path.join(
    WORKSPACE_DIR, "examples/backends/sglang"
)

sglang_configs = {
    "aggregated": SGLangConfig(
        # Uses backend agg.sh (with metrics enabled) for testing standard
        # aggregated deployment with metrics collection
        name="aggregated",
        directory=sglang_dir,
        script_name="agg.sh",
        marks=[pytest.mark.gpu_1],
        model="Qwen/Qwen3-0.6B",
        env={},
        models_port=8000,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            metric_payload_default(min_num_requests=6, backend="sglang"),
        ],
    ),
    "disaggregated": SGLangConfig(
        name="disaggregated",
        directory=sglang_dir,
        script_name="disagg.sh",
        marks=[pytest.mark.gpu_2],
        model="Qwen/Qwen3-0.6B",
        env={},
        models_port=8000,
        request_payloads=[chat_payload_default(), completion_payload_default()],
    ),
    "disaggregated_same_gpu": SGLangConfig(
        # Uses disagg_same_gpu.sh for single-GPU disaggregated testing
        # Validates metrics from both prefill (port 8081) and decode (port 8082) workers
        name="disaggregated_same_gpu",
        directory=sglang_dir,
        script_name="disagg_same_gpu.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.skip(reason="unstable")],
        model="Qwen/Qwen3-0.6B",
        env={},
        models_port=8000,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            # Validate dynamo_component_* and sglang:* metrics from prefill worker (port 8081)
            metric_payload_default(min_num_requests=6, backend="sglang", port=8081),
            # Validate dynamo_component_* and sglang:* metrics from decode worker (port 8082)
            metric_payload_default(min_num_requests=6, backend="sglang", port=8082),
        ],
    ),
    "kv_events": SGLangConfig(
        name="kv_events",
        directory=sglang_dir,
        script_name="agg_router.sh",
        marks=[pytest.mark.gpu_2],
        model="Qwen/Qwen3-0.6B",
        env={
            "DYN_LOG": "dynamo_llm::kv_router::publisher=trace,dynamo_llm::kv_router::scheduler=info",
        },
        models_port=8000,
        request_payloads=[
            chat_payload_default(
                expected_log=[
                    r"ZMQ listener .* received batch with \d+ events \(seq=\d+(?:, [^)]*)?\)",
                    r"Event processor for worker_id \d+ processing event: Stored\(",
                    r"Selected worker: worker_id=\d+ dp_rank=.*?, logit: ",
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
        marks=[pytest.mark.gpu_1],
        model="Qwen/Qwen3-0.6B",
        env={},
        models_port=8000,
        request_payloads=[
            chat_payload_default(
                expected_response=["Successfully Applied Chat Template"]
            )
        ],
    ),
    "multimodal_agg_qwen": SGLangConfig(
        name="multimodal_agg_qwen",
        directory=sglang_dir,
        script_name="multimodal_agg.sh",
        marks=[pytest.mark.gpu_2],
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        delayed_start=0,
        timeout=360,
        models_port=8000,
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
            )
        ],
    ),
    "embedding_agg": SGLangConfig(
        name="embedding_agg",
        directory=sglang_dir,
        script_name="agg_embed.sh",
        marks=[pytest.mark.gpu_1],
        model="Qwen/Qwen3-Embedding-4B",
        delayed_start=0,
        timeout=180,
        models_port=8000,
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
}


@pytest.fixture(params=params_with_model_mark(sglang_configs))
def sglang_config_test(request):
    """Fixture that provides different SGLang test configurations"""
    return sglang_configs[request.param]


@pytest.mark.e2e
@pytest.mark.sglang
def test_sglang_deployment(
    sglang_config_test, request, runtime_services, predownload_models
):
    """Test SGLang deployment scenarios using common helpers"""
    config = sglang_config_test
    run_serve_deployment(config, request)


@pytest.mark.skip(
    reason="Requires 4 GPUs - enable when hardware is consistently available"
)
def test_sglang_disagg_dp_attention(request, runtime_services, predownload_models):
    """Test sglang disaggregated with DP attention (requires 4 GPUs)"""

    # Kept for reference; this test uses a different launch path and is skipped
