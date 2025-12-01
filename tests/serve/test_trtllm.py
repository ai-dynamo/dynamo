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
    chat_payload_default,
    completion_payload_default,
    metric_payload_default,
    multimodal_payload_default,
)

logger = logging.getLogger(__name__)


@dataclass
class TRTLLMConfig(EngineConfig):
    """Configuration for trtllm test scenarios"""

    stragglers: list[str] = field(default_factory=lambda: ["TRTLLM:EngineCore"])


trtllm_dir = os.environ.get("TRTLLM_DIR") or os.path.join(
    WORKSPACE_DIR, "examples/backends/trtllm"
)

# trtllm test configurations
trtllm_configs = {
    "aggregated": TRTLLMConfig(
        name="aggregated",
        directory=trtllm_dir,
        script_name="agg_metrics.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.trtllm, pytest.mark.pre_merge],
        model="Qwen/Qwen3-0.6B",
        models_port=8000,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            metric_payload_default(min_num_requests=6, backend="trtllm"),
        ],
    ),
    "disaggregated": TRTLLMConfig(
        name="disaggregated",
        directory=trtllm_dir,
        script_name="disagg.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.trtllm, pytest.mark.post_merge],
        model="Qwen/Qwen3-0.6B",
        models_port=8000,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "disaggregated_same_gpu": TRTLLMConfig(
        name="disaggregated_same_gpu",
        directory=trtllm_dir,
        script_name="disagg_same_gpu.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.trtllm],
        model="Qwen/Qwen3-0.6B",
        models_port=8000,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            metric_payload_default(port=8081, min_num_requests=6, backend="trtllm"),
            metric_payload_default(port=8082, min_num_requests=6, backend="trtllm"),
        ],
    ),
    "aggregated_router": TRTLLMConfig(
        name="aggregated_router",
        directory=trtllm_dir,
        script_name="agg_router.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.trtllm, pytest.mark.post_merge],
        model="Qwen/Qwen3-0.6B",
        models_port=8000,
        request_payloads=[
            chat_payload_default(
                expected_log=[
                    r"Event processor for worker_id \d+ processing event: Stored\(",
                    r"Selected worker: worker_id=\d+ dp_rank=.*?, logit: ",
                ]
            )
        ],
        env={
            "DYN_LOG": "dynamo_llm::kv_router::publisher=trace,dynamo_llm::kv_router::scheduler=info",
        },
    ),
    "disaggregated_router": TRTLLMConfig(
        name="disaggregated_router",
        directory=trtllm_dir,
        script_name="disagg_router.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.trtllm, pytest.mark.nightly],
        model="Qwen/Qwen3-0.6B",
        models_port=8000,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "disaggregated_multimodal": TRTLLMConfig(
        name="disaggregated_multimodal",
        directory=trtllm_dir,
        script_name="disagg_multimodal.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.trtllm, pytest.mark.multimodal],
        model="Qwen/Qwen2-VL-7B-Instruct",
        models_port=8000,
        timeout=900,
        delayed_start=60,
        request_payloads=[multimodal_payload_default()],
    ),
}


@pytest.fixture(params=params_with_model_mark(trtllm_configs))
def trtllm_config_test(request):
    """Fixture that provides different trtllm test configurations"""
    return request.param


@pytest.mark.trtllm
@pytest.mark.e2e
def test_deployment(trtllm_config_test, request, runtime_services, predownload_models):
    """
    Test dynamo deployments with different configurations.
    """
    config = trtllm_config_test
    extra_env = {"MODEL_PATH": config.model, "SERVED_MODEL_NAME": config.model}
    run_serve_deployment(config, request, extra_env=extra_env)


# TODO make this a normal guy
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.pre_merge
@pytest.mark.trtllm
def test_chat_only_aggregated_with_test_logits_processor(
    request, runtime_services, predownload_models, monkeypatch
):
    """
    Run a single aggregated chat-completions test using Qwen 0.6B with the
    test logits processor enabled, and expect "Hello world" in the response.
    """

    # Enable HelloWorld logits processor only for this test
    monkeypatch.setenv("DYNAMO_ENABLE_TEST_LOGITS_PROCESSOR", "1")

    base = trtllm_configs["aggregated"]
    config = TRTLLMConfig(
        name="aggregated_qwen_chatonly",
        directory=base.directory,
        script_name=base.script_name,  # agg.sh
        marks=[],  # not used by this direct test
        request_payloads=[
            chat_payload_default(expected_response=["Hello world!"]),
        ],
        model="Qwen/Qwen3-0.6B",
        delayed_start=base.delayed_start,
        timeout=base.timeout,
    )

    run_serve_deployment(config, request)
