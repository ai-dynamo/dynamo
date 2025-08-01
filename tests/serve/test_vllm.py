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


def create_payload_for_config(config: "VLLMConfig") -> Payload:
    """Create a payload using the model from the vLLM config"""
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


@dataclass
class VLLMConfig:
    """Configuration for vLLM test scenarios"""

    name: str
    directory: str
    script_name: str
    marks: List[Any]
    endpoints: List[str]
    response_handlers: List[Callable[[Any], str]]
    model: str
    timeout: int = 120
    delayed_start: int = 0


class VLLMProcess(ManagedProcess):
    """Simple process manager for vllm shell scripts"""

    def __init__(self, config: VLLMConfig, request):
        self.port = 8080
        self.config = config
        self.dir = config.directory
        script_path = os.path.join(self.dir, "launch", config.script_name)

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"vLLM script not found: {script_path}")

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
        logger.info("Received Content: %s", content)
        # Check for expected responses
        assert content, "Empty response content"
        for expected in payload.expected_response:
            assert expected in content, "Expected '%s' not found in response" % expected

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

        while time.time() - start_time < self.config.timeout:
            elapsed = time.time() - start_time
            try:
                response = requests.post(
                    url,
                    json=json_payload,
                    timeout=self.config.timeout - elapsed,
                )
            except (requests.RequestException, requests.Timeout) as e:
                logger.warning("Retrying due to Request failed: %s", e)
                time.sleep(retry_delay)
                continue
            logger.info("Response%r", response)
            if response.status_code == 500:
                error = response.json().get("error", "")
                if "no instances" in error:
                    logger.warning("Retrying due to no instances available")
                    time.sleep(retry_delay)
                    continue
            if response.status_code == 404:
                error = response.json().get("error", "")
                if "Model not found" in error:
                    logger.warning("Retrying due to model not found")
                    time.sleep(retry_delay)
                    continue
            # Process the response
            if response.status_code != 200:
                logger.error(
                    "Service returned status code %s: %s",
                    response.status_code,
                    response.text,
                )
                pytest.fail(
                    "Service returned status code %s: %s"
                    % (response.status_code, response.text)
                )
            else:
                break
        else:
            logger.error(
                "Service did not return a successful response within %s s",
                self.config.timeout,
            )
            pytest.fail(
                "Service did not return a successful response within %s s"
                % self.config.timeout
            )

        self.check_response(payload, response, self.config.response_handlers[0], logger)

        logger.info("Deployment Ready")


# vLLM test configurations
vllm_configs = {
    "aggregated": VLLMConfig(
        name="aggregated",
        directory="/workspace/components/backends/vllm",
        script_name="agg.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        endpoints=["v1/chat/completions", "v1/completions"],
        response_handlers=[
            chat_completions_response_handler,
            completions_response_handler,
        ],
        model="Qwen/Qwen3-0.6B",
        delayed_start=45,
    ),
    "agg-router": VLLMConfig(
        name="agg-router",
        directory="/workspace/components/backends/vllm",
        script_name="agg_router.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.vllm],
        endpoints=["v1/chat/completions", "v1/completions"],
        response_handlers=[
            chat_completions_response_handler,
            completions_response_handler,
        ],
        model="Qwen/Qwen3-0.6B",
        delayed_start=45,
    ),
    "disaggregated": VLLMConfig(
        name="disaggregated",
        directory="/workspace/components/backends/vllm",
        script_name="disagg.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.vllm],
        endpoints=["v1/chat/completions", "v1/completions"],
        response_handlers=[
            chat_completions_response_handler,
            completions_response_handler,
        ],
        model="Qwen/Qwen3-0.6B",
        delayed_start=45,
    ),
}


@pytest.fixture(
    params=[
        pytest.param(config_name, marks=config.marks)
        for config_name, config in vllm_configs.items()
    ]
)
def vllm_config_test(request):
    """Fixture that provides different vLLM test configurations"""
    return vllm_configs[request.param]


@pytest.mark.e2e
@pytest.mark.slow
def test_serve_deployment(vllm_config_test, request, runtime_services):
    """
    Test dynamo serve deployments with different graph configurations.
    """

    # runtime_services is used to start nats and etcd

    logger = logging.getLogger(request.node.name)
    logger.info("Starting test_deployment")

    config = vllm_config_test
    payload = create_payload_for_config(config)

    logger.info("Using model: %s", config.model)
    logger.info("Script: %s", config.script_name)

    with VLLMProcess(config, request) as server_process:
        server_process.wait_for_ready(payload, logger)

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
