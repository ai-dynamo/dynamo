# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
from dataclasses import dataclass, field

import pytest

from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.constants import DefaultPort
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import (
    TEXT_PROMPT,
    chat_payload,
    chat_payload_default,
    completion_payload,
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

# TensorRT-LLM test configurations
# NOTE: pytest.mark.gpu_1 tests take ~442s (7m 22s) total to run sequentially (with models pre-cached)
# TODO: Parallelize these tests to reduce total execution time
trtllm_configs = {
    "aggregated": TRTLLMConfig(
        name="aggregated",
        directory=trtllm_dir,
        script_name="agg_metrics.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.pre_merge,
            pytest.mark.trtllm,
            pytest.mark.timeout(
                300
            ),  # 3x measured time (44.66s) + download time (150s)
        ],
        model="Qwen/Qwen3-0.6B",
        frontend_port=DefaultPort.FRONTEND.value,
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
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "disaggregated_same_gpu": TRTLLMConfig(
        name="disaggregated_same_gpu",
        directory=trtllm_dir,
        script_name="disagg_same_gpu.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.pre_merge,
            pytest.mark.trtllm,
            pytest.mark.skip(reason="unstable"),
            pytest.mark.timeout(
                480
            ),  # 3x measured time (103.66s) + download time (150s)
        ],
        model="Qwen/Qwen3-0.6B",
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            metric_payload_default(
                port=DefaultPort.SYSTEM1.value, min_num_requests=6, backend="trtllm"
            ),
            metric_payload_default(
                port=DefaultPort.SYSTEM2.value, min_num_requests=6, backend="trtllm"
            ),
        ],
    ),
    "aggregated_logprobs": TRTLLMConfig(
        name="aggregated_logprobs",
        directory=trtllm_dir,
        script_name="agg.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.pre_merge, pytest.mark.trtllm],
        model="Qwen/Qwen3-0.6B",
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload(content=TEXT_PROMPT, logprobs=True, top_logprobs=5),
            chat_payload(content=TEXT_PROMPT, logprobs=False, top_logprobs=5),
            chat_payload(content=TEXT_PROMPT, logprobs=True, top_logprobs=None),
            chat_payload(content=TEXT_PROMPT, logprobs=True, top_logprobs=0),
        ],
    ),
    "disaggregated_logprobs": TRTLLMConfig(
        name="disaggregated_logprobs",
        directory=trtllm_dir,
        script_name="disagg.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.post_merge, pytest.mark.trtllm],
        model="Qwen/Qwen3-0.6B",
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload(content=TEXT_PROMPT, logprobs=True, top_logprobs=5),
            chat_payload(content=TEXT_PROMPT, logprobs=False, top_logprobs=5),
            chat_payload(content=TEXT_PROMPT, logprobs=True, top_logprobs=None),
            chat_payload(content=TEXT_PROMPT, logprobs=True, top_logprobs=0),
        ],
    ),
    "aggregated_router": TRTLLMConfig(
        name="aggregated_router",
        directory=trtllm_dir,
        script_name="agg_router.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.pre_merge,
            pytest.mark.trtllm,
            pytest.mark.timeout(
                300
            ),  # 3x measured time (37.91s) + download time (180s)
        ],
        model="Qwen/Qwen3-0.6B",
        frontend_port=DefaultPort.FRONTEND.value,
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
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "disaggregated_multimodal": TRTLLMConfig(
        name="disaggregated_multimodal",
        directory=trtllm_dir,
        script_name="disagg_multimodal.sh",
        marks=[
            pytest.mark.gpu_2,
            pytest.mark.trtllm,
            pytest.mark.multimodal,
            pytest.mark.nightly,
        ],
        model="Qwen/Qwen2-VL-7B-Instruct",
        frontend_port=DefaultPort.FRONTEND.value,
        timeout=900,
        delayed_start=60,
        request_payloads=[multimodal_payload_default()],
    ),
    "completions_only": TRTLLMConfig(
        name="completions_only",
        directory=trtllm_dir,
        script_name="agg.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.trtllm,
            pytest.mark.post_merge,
            pytest.mark.timeout(
                480
            ),  # 3x measured time (83.85s) + download time (210s) for 7B model
        ],
        model="deepseek-ai/deepseek-llm-7b-base",
        script_args=["--dyn-endpoint-types", "completions"],
        env={
            "MODEL_PATH": "deepseek-ai/deepseek-llm-7b-base",
            "SERVED_MODEL_NAME": "deepseek-ai/deepseek-llm-7b-base",
        },
        request_payloads=[
            completion_payload_default(),
            completion_payload(prompt=TEXT_PROMPT, logprobs=3),
        ],
    ),
}


@pytest.fixture(params=params_with_model_mark(trtllm_configs))
def trtllm_config_test(request):
    """Fixture that provides different trtllm test configurations"""
    return trtllm_configs[request.param]


@pytest.mark.trtllm
@pytest.mark.e2e
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_deployment(
    trtllm_config_test,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    num_system_ports,
    predownload_models,
):
    """
    Test dynamo deployments with different configurations.
    """
    assert (
        num_system_ports >= 2
    ), "serve tests require at least SYSTEM_PORT1 + SYSTEM_PORT2"
    # Use per-test ports so tests can run safely under pytest-xdist.
    config = dataclasses.replace(
        trtllm_config_test, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    # Non-port env stays here; ports are wired by run_serve_deployment(ports=...).
    config.env.update(
        {
            "MODEL_PATH": config.model,
            "SERVED_MODEL_NAME": config.model,
        }
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)


# TODO make this a normal guy
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.trtllm
@pytest.mark.pre_merge
@pytest.mark.timeout(660)  # 3x measured time (159.68s) + download time (180s)
def test_chat_only_aggregated_with_test_logits_processor(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_models,
    monkeypatch,
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

    config = dataclasses.replace(
        config, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    config.env.update(
        {
            "MODEL_PATH": config.model,
            "SERVED_MODEL_NAME": config.model,
        }
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)
