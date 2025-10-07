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

from tests.utils.constants import GPT_OSS
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api

logger = logging.getLogger(__name__)

REASONING_TEST_MODEL = GPT_OSS


class DynamoFrontendProcess(ManagedProcess):
    """Process manager for Dynamo frontend"""

    def __init__(self, request):
        command = ["python", "-m", "dynamo.frontend", "--router-mode", "round-robin"]

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
            display_output=True,
            terminate_existing=True,
            log_dir=log_dir,
        )


class GPTOSSWorkerProcess(ManagedProcess):
    """Worker process for GPT-OSS model."""

    def __init__(self, request, worker_id: str = "reasoning-worker"):
        self.worker_id = worker_id

        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            REASONING_TEST_MODEL,
            "--enforce-eager",
            "--dyn-tool-call-parser",
            "harmony",
            "--dyn-reasoning-parser",
            "gpt_oss",
        ]

        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_ENABLED"] = "true"
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
            timeout=300,
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
    prompt: str,
    reasoning_effort: str,
    timeout: int = 180,
) -> requests.Response:
    """Send a chat completion request with a specific reasoning effort."""

    payload: Dict[str, Any] = {
        "model": REASONING_TEST_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": 2000,
        "chat_template_args": {"reasoning_effort": reasoning_effort},
    }

    headers = {"Content-Type": "application/json"}

    logger.info(
        "Sending chat completion request with reasoning effort '%s'", reasoning_effort
    )
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    logger.info(
        "Received response for reasoning effort '%s' with status code %s",
        reasoning_effort,
        response.status_code,
    )
    return response


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
    payload = response.json()
    if "choices" not in payload:
        raise AssertionError(f"Chat response missing 'choices': {payload}")
    return payload


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(REASONING_TEST_MODEL)
def test_reasoning_effort(request, runtime_services, predownload_models) -> None:
    """High reasoning effort should yield more detailed reasoning than low effort."""

    prompt = (
        "Outline the critical steps and trade-offs when designing a Mars habitat. "
        "Focus on life-support, energy, and redundancy considerations."
    )

    with DynamoFrontendProcess(request):
        logger.info("Frontend started for reasoning effort test")

        with GPTOSSWorkerProcess(request):
            logger.info("Worker started for reasoning effort test")

            low_response = _send_chat_request(prompt, reasoning_effort="low")
            low_payload = _validate_chat_response(low_response)
            low_reasoning_text, low_reasoning_tokens = _extract_reasoning_metrics(
                low_payload
            )

            high_response = _send_chat_request(prompt, reasoning_effort="high")
            high_payload = _validate_chat_response(high_response)
            high_reasoning_text, high_reasoning_tokens = _extract_reasoning_metrics(
                high_payload
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
