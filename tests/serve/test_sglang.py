# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from dataclasses import dataclass, field

import pytest

from tests.serve.common import (
    SERVE_TEST_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import chat_payload_default, completion_payload_default

logger = logging.getLogger(__name__)


@dataclass
class SGLangConfig(EngineConfig):
    """Configuration for SGLang test scenarios"""

    stragglers: list[str] = field(default_factory=lambda: ["SGLANG:EngineCore"])


sglang_dir = os.environ.get("SGLANG_DIR", "/workspace/components/backends/sglang")

sglang_configs = {
    "aggregated": SGLangConfig(
        name="aggregated",
        directory=SERVE_TEST_DIR,
        script_name="sglang_agg.sh",
        marks=[pytest.mark.gpu_1],
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        env={},
        models_port=8000,
        request_payloads=[chat_payload_default(), completion_payload_default()],
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
                    r"ZMQ listener .* received batch with \d+ events \(seq=\d+\)",
                    r"Event processor for worker_id \d+ processing event: Stored\(",
                    r"Selected worker: \d+, logit: ",
                ]
            )
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
