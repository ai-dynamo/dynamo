# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests covering reasoning effort behaviour."""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, Optional, Tuple

import pytest
import requests

from tests.conftest import EtcdServer, NatsServer
from tests.utils.constants import GPT_OSS
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api

logger = logging.getLogger(__name__)

TEST_MODEL = GPT_OSS

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, NY",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit",
                },
            },
            "required": ["location", "format"],
        },
    },
}

SYSTEM_HEALTH_TOOL = {
    "type": "function",
    "function": {
        "name": "get_system_health",
        "description": "Returns the current health status of the LLM runtime—use before critical operations to verify the service is live.",
        "parameters": {"type": "object", "properties": {}},
    },
}


class DynamoFrontendProcess(ManagedProcess):
    """Process manager for Dynamo frontend"""

    def __init__(self, request):
        command = ["python", "-m", "dynamo.frontend", "--router-mode", "round-robin"]

        # Unset DYN_SYSTEM_PORT - frontend doesn't use system metrics server
        env = os.environ.copy()
        env.pop("DYN_SYSTEM_PORT", None)

        log_dir = f"{request.node.name}_frontend"

        # Clean up any existing log directory from previous runs
        try:
            shutil.rmtree(log_dir)
            logger.info(f"Cleaned up existing log directory: {log_dir}")
        except FileNotFoundError:
            # Directory doesn't exist, which is fine
            pass

        super().__init__(
            command=command,
            env=env,
            display_output=True,
            terminate_existing=True,
            log_dir=log_dir,
        )


class VllmWorkerProcess(ManagedProcess):
    """Vllm Worker process for GPT-OSS model."""

    def __init__(self, request, worker_id: str = "vllm-worker"):
        self.worker_id = worker_id

        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            TEST_MODEL,
            "--dyn-tool-call-parser",
            "harmony",
            "--dyn-reasoning-parser",
            "gpt_oss",
            "--connector",
            "none",
        ]

        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = "8083"

        log_dir = f"{request.node.name}_{worker_id}"

        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                ("http://localhost:8000/v1/models", check_models_api),
                ("http://localhost:8083/health", self.is_ready),
            ],
            timeout=500,
            display_output=True,
            terminate_existing=False,
            stragglers=["VLLM::EngineCore"],
            straggler_commands=["-m dynamo.vllm"],
            log_dir=log_dir,
        )

    def is_ready(self, response) -> bool:
        try:
            status = (response.json() or {}).get("status")
        except ValueError:
            logger.warning("%s health response is not valid JSON", self.worker_id)
            return False

        is_ready = status == "ready"
        if is_ready:
            logger.info("%s status is ready", self.worker_id)
        else:
            logger.warning("%s status is not ready: %s", self.worker_id, status)
        return is_ready


def _send_chat_request(
    payload: Dict[str, Any],
    timeout: int = 180,
) -> requests.Response:
    """Send a chat completion request with a specific payload."""
    headers = {"Content-Type": "application/json"}

    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    return response


@pytest.fixture(scope="module")
def runtime_services(request):
    """Module-scoped runtime services for this test file."""
    with NatsServer(request) as nats_process:
        with EtcdServer(request) as etcd_process:
            yield nats_process, etcd_process


@pytest.fixture(scope="module")
def start_services(request, runtime_services):
    """Start frontend and worker processes once for this module's tests."""
    with DynamoFrontendProcess(request):
        logger.info("Frontend started for tests")
        with VllmWorkerProcess(request):
            logger.info("Vllm Worker started for tests")
            yield


def _extract_reasoning_metrics(data: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    """Return the reasoning content and optional reasoning token count from a response."""
    choices = data.get("choices") or []
    if not choices:
        raise AssertionError(f"Response missing choices: {data}")

    message = choices[0].get("message") or {}
    reasoning_text = (message.get("reasoning_content") or "").strip()

    usage_block = data.get("usage") or {}
    tokens = usage_block.get("reasoning_tokens")
    reasoning_tokens: Optional[int] = tokens if isinstance(tokens, int) else None

    if not reasoning_text:
        raise AssertionError(f"Response missing reasoning content: {data}")

    return reasoning_text, reasoning_tokens


def _validate_chat_response(response: requests.Response) -> Dict[str, Any]:
    """Ensure the chat completion response is well-formed and return its payload."""
    assert (
        response.status_code == 200
    ), f"Chat request failed with status {response.status_code}: {response.text}"
    response_json = response.json()
    if "choices" not in response_json:
        raise AssertionError(f"Chat response missing 'choices': {response_json}")
    return response_json


@pytest.mark.usefixtures("start_services")
@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(TEST_MODEL)
def test_reasoning_effort(request, runtime_services, predownload_models) -> None:
    """High reasoning effort should yield more detailed reasoning than low effort."""

    prompt = (
        "Outline the critical steps and trade-offs when designing a Mars habitat. "
        "Focus on life-support, energy, and redundancy considerations."
    )

    logger.info("Start to test reasoning effort")
    high_payload = {
        "model": TEST_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": 2000,
        "chat_template_args": {"reasoning_effort": "high"},
    }

    low_payload = {
        "model": TEST_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": 2000,
        "chat_template_args": {"reasoning_effort": "low"},
    }

    high_response = _send_chat_request(high_payload)
    high_reasoning_text, high_reasoning_tokens = _extract_reasoning_metrics(
        _validate_chat_response(high_response)
    )

    low_response = _send_chat_request(low_payload)
    low_reasoning_text, low_reasoning_tokens = _extract_reasoning_metrics(
        _validate_chat_response(low_response)
    )

    logger.info(
        "Low effort reasoning tokens: %s, High effort reasoning tokens: %s",
        low_reasoning_tokens,
        high_reasoning_tokens,
    )

    if low_reasoning_tokens is not None and high_reasoning_tokens is not None:
        assert high_reasoning_tokens >= low_reasoning_tokens, (
            "Expected high reasoning effort to use at least as many reasoning tokens "
            f"as low effort (low={low_reasoning_tokens}, high={high_reasoning_tokens})"
        )
    else:
        assert len(high_reasoning_text) > len(low_reasoning_text), (
            "Expected high reasoning effort response to include longer reasoning "
            "content than low effort"
        )


@pytest.mark.usefixtures("start_services")
@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(TEST_MODEL)
def test_tool_calling(request, runtime_services, predownload_models) -> None:
    """Test tool calling functionality with weather and system health tools."""

    payload = {
        "model": TEST_MODEL,
        "messages": [
            {
                "role": "user",
                "content": "What is the weather like in San Francisco today?",
            }
        ],
        "max_tokens": 2000,
        "tools": [
            WEATHER_TOOL,
            SYSTEM_HEALTH_TOOL,
        ],
        "tool_choice": "auto",
        "response_format": {"type": "text"},
    }

    response = _send_chat_request(payload)
    response_data = _validate_chat_response(response)

    logger.info("Tool call response: %s", response_data)

    choices = response_data.get("choices", [])
    assert choices, "Response missing choices"

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])

    assert tool_calls, "Expected model to generate tool calls for weather query"
    assert any(
        tc.get("function", {}).get("name") == "get_current_weather" for tc in tool_calls
    ), "Expected get_current_weather tool to be called"


@pytest.mark.usefixtures("start_services")
@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(TEST_MODEL)
def test_tool_calling_second_round(
    request, runtime_services, predownload_models
) -> None:
    """Test tool calling with a follow-up message containing assistant's prior tool calls."""

    payload = {
        "model": TEST_MODEL,
        "messages": [
            # First message
            {
                "role": "user",
                "content": "What is the weather like in San Francisco today?",
            },
            # Assistant message with tool calls
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"format":"celsius","location":"San Francisco"}',
                        },
                    }
                ],
            },
            # Tool message with tool call result
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": '{"celsius":"20"}',
            },
        ],
        "max_tokens": 2000,
        "tools": [
            WEATHER_TOOL,
            SYSTEM_HEALTH_TOOL,
        ],
        "tool_choice": "auto",
        "response_format": {"type": "text"},
    }

    response = _send_chat_request(payload)
    response_data = _validate_chat_response(response)

    logger.info("Tool call second round response: %s", response_data)

    choices = response_data.get("choices", [])
    assert choices, "Response missing choices"

    message = choices[0].get("message", {})
    content = message.get("content", "").strip()

    assert content, "Expected model to generate a response with content"
    assert "20" in content and any(
        temp_word in content.lower()
        for temp_word in ["celsius", "temperature", "degrees", "°c", "20°"]
    ), "Expected response to include temperature information from tool call result (20°C)"


@pytest.mark.usefixtures("start_services")
@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(TEST_MODEL)
def test_reasoning(request, runtime_services, predownload_models) -> None:
    """Test reasoning functionality with a mathematical problem."""

    payload = {
        "model": TEST_MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm playing assetto corsa competizione, and I need you to tell me "
                    "how many liters of fuel to take in a race. The qualifying time was "
                    "2:04.317, the race is 20 minutes long, and the car uses 2.73 liters per lap."
                ),
            }
        ],
        "max_tokens": 2000,
    }

    response = _send_chat_request(payload)
    response_data = _validate_chat_response(response)

    logger.info("Reasoning response: %s", response_data)

    choices = response_data.get("choices", [])
    assert choices, "Response missing choices"

    message = choices[0].get("message", {})
    content = message.get("content", "").strip()

    assert content, "Expected model to generate a response with content"
    assert any(
        char.isdigit() for char in content
    ), "Expected response to contain numerical calculations"
