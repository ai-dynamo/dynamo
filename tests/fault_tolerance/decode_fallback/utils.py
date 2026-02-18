# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
import time

import pytest
import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import (
    DynamoFrontendProcess as BaseDynamoFrontendProcess,
)
from tests.utils.managed_process import ManagedProcess, terminate_process_tree

logger = logging.getLogger(__name__)


class DynamoFrontendProcess(BaseDynamoFrontendProcess):
    """Frontend wrapper for decode-only fallback tests.

    Starts the frontend without --enforce-disagg so that requests fall back
    to decode-only mode when the prefill worker is unavailable.
    """

    def __init__(self, request):
        extra_env = {
            "DYN_REQUEST_PLANE": request.getfixturevalue("request_plane"),
            # Disable canary health check to avoid interfering with test requests.
            "DYN_HEALTH_CHECK_ENABLED": "false",
        }
        super().__init__(
            request,
            frontend_port=0,  # allocate a free port (xdist-safe)
            router_mode="round-robin",
            migration_limit=3,
            extra_env=extra_env,
            terminate_all_matching_process_names=False,
            display_name="frontend",
        )


def send_request(
    frontend_port: int, use_chat: bool, stream: bool, timeout: int = 240
) -> str:
    """Send an inference request and return the response text.

    Args:
        frontend_port: Port where the frontend is running.
        use_chat: True for chat completions API, False for completions API.
        stream: Whether to use streaming responses.
        timeout: Request timeout in seconds.

    Returns:
        The generated text content.

    Raises:
        AssertionError: If the request fails or returns an error.
    """
    prompt = "Say hello in one sentence."
    headers = {"Content-Type": "application/json"}

    if use_chat:
        url = f"http://localhost:{frontend_port}/v1/chat/completions"
        payload = {
            "model": FAULT_TOLERANCE_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
    else:
        url = f"http://localhost:{frontend_port}/v1/completions"
        payload = {
            "model": FAULT_TOLERANCE_MODEL_NAME,
            "prompt": prompt,
            "stream": stream,
        }

    logger.info(f"Sending {'chat ' if use_chat else ''}completion request (stream={stream})")

    response = requests.post(url, headers=headers, json=payload, timeout=timeout, stream=stream)
    assert response.status_code == 200, (
        f"Request failed with status {response.status_code}: {response.text}"
    )

    if stream:
        # Collect streamed content
        content_parts = []
        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if line_str.startswith("event: error"):
                raise AssertionError(f"SSE error event received: {line_str}")
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                if use_chat:
                    text = chunk["choices"][0]["delta"].get("content")
                else:
                    text = chunk["choices"][0].get("text")
                if text:
                    content_parts.append(text)
            except Exception as e:
                raise AssertionError(f"Error parsing SSE chunk: {e}")
        content = "".join(content_parts)
    else:
        data = response.json()
        if use_chat:
            content = data["choices"][0]["message"]["content"]
        else:
            content = data["choices"][0]["text"]

    assert content, "Response content is empty"
    logger.info(f"Received response: {content[:100]}...")
    return content


def _parse_decode_only_fallback_metric(metrics_text: str, model_name: str) -> int:
    """Parse the decode-only fallback metric from Prometheus text.

    Args:
        metrics_text: Raw Prometheus metrics text.
        model_name: The model name label value.

    Returns:
        The metric count, or 0 if not found.
    """
    # Match: dynamo_frontend_model_decode_only_fallback_total{model="Qwen/Qwen3-0.6B"} 1
    pattern = rf'dynamo_frontend_model_decode_only_fallback_total\{{[^}}]*model="{re.escape(model_name)}"[^}}]*\}}\s+(\d+)'
    match = re.search(pattern, metrics_text)
    if match:
        return int(match.group(1))
    return 0


def verify_decode_only_fallback_metric(
    frontend_port: int, expected_count: int
) -> None:
    """Verify the decode-only fallback metric matches expected count.

    Args:
        frontend_port: Port where the frontend is running.
        expected_count: Expected metric count.
    """
    metrics_url = f"http://localhost:{frontend_port}/metrics"

    try:
        response = requests.get(metrics_url, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        pytest.fail(f"Failed to fetch metrics from {metrics_url}: {e}")

    metrics_text = response.text
    count = _parse_decode_only_fallback_metric(metrics_text, FAULT_TOLERANCE_MODEL_NAME)

    logger.info(f"decode_only_fallback_total: {count} (expected: {expected_count})")

    assert count == expected_count, (
        f"Expected decode_only_fallback_total == {expected_count}, but got {count}"
    )


def run_decode_fallback_test(
    frontend: DynamoFrontendProcess,
    prefill_worker: ManagedProcess,
    decode_worker: ManagedProcess,
    use_chat: bool,
    stream: bool,
) -> None:
    """Run the decode-only fallback test flow.

    Steps:
        1. Send request with both workers up → assert success.
        2. Check decode_only_fallback_total metric == 0.
        3. Kill the prefill worker.
        4. Send another request → assert success (decode-only fallback).
        5. Check decode_only_fallback_total metric >= 1.

    Args:
        frontend: The frontend process.
        prefill_worker: The prefill worker process.
        decode_worker: The decode worker process (kept alive throughout).
        use_chat: Whether to use chat completion API.
        stream: Whether to use streaming responses.
    """
    # Step 1: Send request with both workers running — should go through prefill path
    logger.info("Step 1: Sending request with both workers up")
    send_request(frontend.frontend_port, use_chat=use_chat, stream=stream)

    # Step 2: Verify no fallbacks occurred
    logger.info("Step 2: Verifying decode_only_fallback_total == 0")
    verify_decode_only_fallback_metric(frontend.frontend_port, expected_count=0)

    # Step 3: Kill the prefill worker
    logger.info(
        f"Step 3: Killing prefill worker (PID {prefill_worker.get_pid()})"
    )
    terminate_process_tree(prefill_worker.get_pid(), immediate_kill=True, timeout=0)

    # Brief pause for NATS to detect the disconnection
    time.sleep(2)

    # Step 4: Send another request — should fall back to decode-only
    logger.info("Step 4: Sending request after prefill worker killed")
    send_request(frontend.frontend_port, use_chat=use_chat, stream=stream)

    # Step 5: Verify fallback metric incremented
    logger.info("Step 5: Verifying decode_only_fallback_total >= 1")
    verify_decode_only_fallback_metric(frontend.frontend_port, expected_count=1)
