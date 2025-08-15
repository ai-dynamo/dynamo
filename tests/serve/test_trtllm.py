# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, List

import pytest
import requests

from tests.utils.deployment_graph import (
    Payload,
    chat_completions_response_handler,
    completions_response_handler,
)
from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)

text_prompt = "Tell me a short joke about AI."


def create_payload_for_config(config: "TRTLLMConfig") -> Payload:
    """Create a payload using the model from the trtllm config"""
    return Payload(
        payload_chat={
            "model": config.model,
            "messages": [
                {
                    "role": "user",
                    "content": text_prompt,
                }
            ],
            "max_tokens": 150,
            "temperature": 0.1,
        },
        payload_completions={
            "model": config.model,
            "prompt": text_prompt,
            "max_tokens": 150,
            "temperature": 0.1,
        },
        repeat_count=1,
        expected_log=[],
        expected_response=["AI"],
    )


# TODO: Unify with vllm/sglang tests to reduce code duplication
@dataclass
class TRTLLMConfig:
    """Configuration for trtllm test scenarios"""

    name: str
    directory: str
    script_name: str
    marks: List[Any]
    endpoints: List[str]
    response_handlers: List[Callable[[Any], str]]
    model: str
    timeout: int = 60
    delayed_start: int = 0


class TRTLLMProcess(ManagedProcess):
    """Simple process manager for trtllm shell scripts"""

    def __init__(self, config: TRTLLMConfig, request):
        self.port = 8000
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

    def _check_models_api(self, response):
        """Check if models API is working and returns models"""
        try:
            if response.status_code != 200:
                return False
            data = response.json()
            return data.get("data") and len(data["data"]) > 0
        except Exception:
            return False

    def _check_url(self, url, timeout=30, sleep=2.0):
        """Override to use a more reasonable retry interval"""
        return super()._check_url(url, timeout, sleep)

    def check_response(
        self, payload, response, response_handler, logger=logging.getLogger()
    ):
        assert response.status_code == 200, "Response Error"
        content = response_handler(response)
        logger.info(f"Received Content: {content}")
        # Check for expected responses
        assert content, "Empty response content"
        for expected in payload.expected_response:
            assert expected in content, f"Expected '{expected}' not found in response"

    def wait_for_ready(self, payload, logger=logging.getLogger()):
        url = f"http://localhost:{self.port}/{self.config.endpoints[0]}"
        start_time = time.time()
        retry_delay = 5
        elapsed = 0.0
        logger.info("Waiting for Deployment Ready")
        json_payload = (
            payload.payload_chat
            if self.config.endpoints[0] == "v1/chat/completions"
            else payload.payload_completions
        )

        while (elapsed := time.time() - start_time) < self.config.timeout:
            try:
                response = requests.post(
                    url,
                    json=json_payload,
                    timeout=self.config.timeout - elapsed,
                )
            except (requests.RequestException, requests.Timeout) as e:
                logger.warning(f"Retrying due to Request failed: {e}")
                time.sleep(retry_delay)
                continue
            logger.info(f"Response: {response}")
            if response.status_code == 500:
                error = response.json().get("error", "")
                if "no instances" in error:
                    logger.warning(
                        f"Retrying due to no instances available for model '{self.config.model}'"
                    )
                    time.sleep(retry_delay)
                    continue
            if response.status_code == 404:
                error = response.json().get("error", "")
                if "Model not found" in error:
                    logger.warning(
                        f"Retrying due to model not found for model '{self.config.model}'"
                    )
                    time.sleep(retry_delay)
                    continue
            # Process the response
            if response.status_code != 200:
                pytest.fail(
                    f"Service returned status code {response.status_code}: {response.text}"
                )
            else:
                break
        else:
            pytest.fail(
                f"Service did not return a successful response within {self.config.timeout} s"
            )

        self.check_response(payload, response, self.config.response_handlers[0], logger)

        logger.info("Deployment Ready")


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
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        delayed_start=60,
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
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        delayed_start=60,
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
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        delayed_start=60,
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
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        delayed_start=60,
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
    payload = create_payload_for_config(config)

    logger.info(f"Using model: {config.model}")
    logger.info(f"Script: {config.script_name}")

    with TRTLLMProcess(config, request) as server_process:
        server_process.wait_for_ready(payload, logger)

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
                elapsed = time.time() - start_time

                response = requests.post(
                    url,
                    json=request_body,
                    timeout=config.timeout - elapsed,
                )
                server_process.check_response(
                    payload, response, response_handler, logger
                )
