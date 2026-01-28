# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests covering reasoning effort behaviour.

Runtime note:
- `python -m pytest tests/frontend/test_vllm.py -v` took ~228s (3m48s) wall time.
- Measured on: Ubuntu 24.04.2, Intel(R) Core(TM) i9-14900K (32 CPUs), NVIDIA RTX 6000 Ada Generation (1 warmup run + 1 measured run).
- Expect variance depending on model cache state, compilation warmup, and system load.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, Generator, Optional, Tuple

import pytest
import requests

from tests.utils.constants import GPT_OSS
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import ServicePorts

logger = logging.getLogger(__name__)

TEST_MODEL = GPT_OSS

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.e2e,
    pytest.mark.model(TEST_MODEL),
]

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


class VllmWorkerProcess(ManagedProcess):
    """Vllm Worker process for GPT-OSS model."""

    def __init__(
        self,
        request,
        *,
        frontend_port: int,
        system_port: int,
        worker_id: str = "vllm-worker",
    ):
        self.worker_id = worker_id
        self.frontend_port = int(frontend_port)
        self.system_port = int(system_port)

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
        env["DYN_SYSTEM_PORT"] = str(self.system_port)

        log_dir = f"{request.node.name}_{worker_id}"

        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (f"http://localhost:{self.frontend_port}/v1/models", check_models_api),
                (f"http://localhost:{self.system_port}/health", self.is_ready),
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
    *,
    base_url: str,
    timeout: int = 180,
) -> requests.Response:
    """Send a chat completion request with a specific payload."""
    headers = {"Content-Type": "application/json"}

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    return response


@pytest.fixture(scope="function")
def start_services(
    request, runtime_services_dynamic_ports, dynamo_dynamic_ports: ServicePorts
) -> Generator[ServicePorts, None, None]:
    """Start frontend and worker processes for this test.

    `runtime_services_dynamic_ports` ensures NATS/etcd run on per-test ports and sets
    NATS_SERVER/ETCD_ENDPOINTS env vars for Dynamo to discover them.

    This fixture also *returns the exact ports used to launch the services* so tests
    cannot accidentally construct requests against a different `dynamo_dynamic_ports`
    instance (e.g., if fixture scopes/usage are changed in the future).
    """
    _ = runtime_services_dynamic_ports
    frontend_port = dynamo_dynamic_ports.frontend_port
    system_port = dynamo_dynamic_ports.system_ports[0]
    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        # Optional debugging (not enabled on main):
        # If the frontend hits a Rust panic, enabling backtraces makes failures diagnosable
        # from CI logs without needing to repro locally.
        # extra_env={"RUST_BACKTRACE": "1", "TOKIO_BACKTRACE": "1"},
        terminate_existing=False,
    ):
        logger.info("Frontend started for tests")
        with VllmWorkerProcess(
            request,
            frontend_port=frontend_port,
            system_port=system_port,
        ):
            logger.info("Vllm Worker started for tests")
            yield dynamo_dynamic_ports


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


@pytest.mark.timeout(240)  # ~3x measured total (~70s/test), rounded up
@pytest.mark.post_merge
def test_reasoning_effort(
    request, start_services: ServicePorts, predownload_models
) -> None:
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

    base_url = f"http://localhost:{start_services.frontend_port}"
    high_response = _send_chat_request(high_payload, base_url=base_url)
    high_reasoning_text, high_reasoning_tokens = _extract_reasoning_metrics(
        _validate_chat_response(high_response)
    )

    low_response = _send_chat_request(low_payload, base_url=base_url)
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


@pytest.mark.timeout(180)  # ~3x measured total (~50s/test), rounded up
@pytest.mark.post_merge
def test_tool_calling(
    request, start_services: ServicePorts, predownload_models
) -> None:
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

    base_url = f"http://localhost:{start_services.frontend_port}"
    response = _send_chat_request(payload, base_url=base_url)
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


@pytest.mark.timeout(180)  # ~3x measured total (~50s/test), rounded up
@pytest.mark.nightly
def test_tool_calling_second_round(
    request, start_services: ServicePorts, predownload_models
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

    base_url = f"http://localhost:{start_services.frontend_port}"
    response = _send_chat_request(payload, base_url=base_url)
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


@pytest.mark.timeout(180)  # ~3x measured total (~57s/test), rounded up
@pytest.mark.nightly
def test_reasoning(request, start_services: ServicePorts, predownload_models) -> None:
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

    base_url = f"http://localhost:{start_services.frontend_port}"
    response = _send_chat_request(payload, base_url=base_url)
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
