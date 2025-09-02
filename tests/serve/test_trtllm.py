# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
from dataclasses import dataclass

import pytest

from tests.serve.common import EngineConfig, create_payload_for_config
from tests.utils.deployment_graph import (
    chat_completions_response_handler,
    completions_response_handler,
    metrics_handler,
)
from tests.utils.engine_process import EngineProcess

logger = logging.getLogger(__name__)


@dataclass
class TRTLLMConfig(EngineConfig):
    """Configuration for trtllm test scenarios"""


class TRTLLMProcess(EngineProcess):
    """Simple process manager for trtllm shell scripts"""

    def __init__(self, config: TRTLLMConfig, request):
        self.port = 8000
        self.backend_metrics_port = 8081
        self.config = config
        self.dir = config.directory
        script_path = os.path.join(self.dir, "launch", config.script_name)

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"trtllm script not found: {script_path}")

        # Set these env vars to customize model launched by launch script to match test
        os.environ["MODEL_PATH"] = config.model
        os.environ["SERVED_MODEL_NAME"] = config.model

        command = ["bash", script_path]

        super().__init__(
            command=command,
            timeout=config.timeout,
            display_output=True,
            working_dir=self.dir,
            health_check_ports=[],  # Disable port health check
            health_check_urls=[
                (f"http://localhost:{self.port}/v1/models", self._check_models_api)
            ],
            delayed_start=config.delayed_start,
            terminate_existing=False,  # If true, will call all bash processes including myself
            stragglers=[],  # Don't kill any stragglers automatically
            log_dir=request.node.name,
        )


def run_trtllm_test_case(config: TRTLLMConfig, request) -> None:
    payload = create_payload_for_config(config)

    with TRTLLMProcess(config, request) as server_process:
        assert len(config.endpoints) == len(config.response_handlers)
        for endpoint, response_handler in zip(
            config.endpoints, config.response_handlers
        ):
            url = f"http://localhost:{server_process.port}/{endpoint}"
            start_time = time.time()
            elapsed = 0.0

            request_body = (
                payload.payload_chat
                if endpoint == "v1/chat/completions"
                else payload.payload_completions
            )

            for _ in range(payload.repeat_count):
                if endpoint == "metrics":
                    response = server_process.get_metrics(
                        server_process.backend_metrics_port
                    )
                    response_handler(response)
                else:
                    elapsed = time.time() - start_time
                    response = server_process.send_request(
                        url, payload=request_body, timeout=config.timeout - elapsed
                    )
                    server_process.check_response(payload, response, response_handler)


# trtllm test configurations
trtllm_configs = {
    "aggregated": TRTLLMConfig(
        name="aggregated",
        directory="/workspace/components/backends/trtllm",
        script_name="agg.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.trtllm_marker],
        endpoints=["v1/chat/completions", "v1/completions"],
        response_handlers=[
            chat_completions_response_handler,
            completions_response_handler,
        ],
        model="Qwen/Qwen3-0.6B",
    ),
    "disaggregated": TRTLLMConfig(
        name="disaggregated",
        directory="/workspace/components/backends/trtllm",
        script_name="disagg.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.trtllm_marker],
        endpoints=["v1/chat/completions", "v1/completions"],
        response_handlers=[
            chat_completions_response_handler,
            completions_response_handler,
        ],
        model="Qwen/Qwen3-0.6B",
    ),
    # TODO: These are sanity tests that the kv router examples launch
    # and inference without error, but do not do detailed checks on the
    # behavior of KV routing.
    "aggregated_router": TRTLLMConfig(
        name="aggregated_router",
        directory="/workspace/components/backends/trtllm",
        script_name="agg_router.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.trtllm_marker],
        endpoints=["v1/chat/completions", "v1/completions"],
        response_handlers=[
            chat_completions_response_handler,
            completions_response_handler,
        ],
        model="Qwen/Qwen3-0.6B",
    ),
    "disaggregated_router": TRTLLMConfig(
        name="disaggregated_router",
        directory="/workspace/components/backends/trtllm",
        script_name="disagg_router.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.trtllm_marker],
        endpoints=["v1/chat/completions", "v1/completions"],
        response_handlers=[
            chat_completions_response_handler,
            completions_response_handler,
        ],
        model="Qwen/Qwen3-0.6B",
    ),
    "aggregated_metrics": TRTLLMConfig(
        name="aggregated_metrics",
        directory="/workspace/components/backends/trtllm",
        script_name="agg_metrics.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.trtllm_marker],
        endpoints=[
            "v1/chat/completions",
            "metrics",
        ],  # Make a request to make sure the model is loaded and metrics are published.
        response_handlers=[chat_completions_response_handler, metrics_handler],
        model="Qwen/Qwen3-0.6B",
    ),
}


@pytest.fixture(
    params=[
        pytest.param(config_name, marks=config.marks)
        for config_name, config in trtllm_configs.items()
    ]
)
def trtllm_config_test(request):
    """Fixture that provides different trtllm test configurations"""
    return trtllm_configs[request.param]


@pytest.mark.e2e
@pytest.mark.slow
def test_deployment(trtllm_config_test, request, runtime_services):
    """
    Test dynamo deployments with different configurations.
    """

    # runtime_services is used to start nats and etcd

    logger = logging.getLogger(request.node.name)
    logger.info("Starting test_deployment")

    config = trtllm_config_test
    logger.info(f"Using model: {config.model}")
    logger.info(f"Script: {config.script_name}")
    run_trtllm_test_case(config, request)


@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.trtllm_marker
@pytest.mark.slow
def test_chat_only_aggregated_with_test_logits_processor(
    request, runtime_services, monkeypatch
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
        endpoints=["v1/chat/completions"],
        response_handlers=[chat_completions_response_handler],
        model="Qwen/Qwen3-0.6B",
        delayed_start=base.delayed_start,
        timeout=base.timeout,
    )

    run_trtllm_test_case(config, request)
