# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import threading
import time

import pytest
import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import (
    DynamoFrontendProcess as BaseDynamoFrontendProcess,
)
from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)


class DynamoFrontendProcess(BaseDynamoFrontendProcess):
    """Fault-tolerance frontend wrapper (keeps env settings from the historical helper)."""

    def __init__(self, request, enforce_disagg: bool = False):
        extra_env = {
            "DYN_REQUEST_PLANE": request.getfixturevalue("request_plane"),
            # These tests expect full control over requests sent to workers. The canary
            # health check can inject extra requests and cause intermittent failures.
            "DYN_HEALTH_CHECK_ENABLED": "false",
        }
        extra_args = []
        if enforce_disagg:
            extra_args.append("--enforce-disagg")
        super().__init__(
            request,
            frontend_port=0,  # allocate a free port (xdist-safe)
            router_mode="round-robin",
            extra_args=extra_args if extra_args else None,
            extra_env=extra_env,
            terminate_existing=False,
        )


def start_request(frontend_port: int, use_long_prompt: bool = False) -> tuple:
    """
    Start a long-running chat completion request in a separate thread.

    Logs each streamed chunk as (response_string | None | Exception, timestamp) tuple to
    response_list. First entry is (None, start_time) to mark when request was sent.

    Args:
        frontend_port: Port where the frontend is running
        use_long_prompt: Whether to use a long prompt (~8000 tokens)

    Returns:
        tuple: (request_thread, response_list) where response_list contains
               (str | None | Exception, float) tuples
    """
    response_list: list[tuple[str | None | Exception, float]] = []

    def send_request():
        prompt = "Tell me a long long long story about yourself?"
        if use_long_prompt:
            prompt += " Make sure it is" + " long" * 8000 + "!"
        max_tokens = 8000
        timeout = 240  # Extended timeout for long request

        payload = {
            "model": FAULT_TOLERANCE_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        }
        headers = {"Content-Type": "application/json"}

        logger.info(
            f"Sending chat completion request with prompt: '{prompt[:50]}...' and max_tokens: {max_tokens}"
        )

        response_list.append((None, time.time()))  # start timestamp

        try:
            with requests.post(
                f"http://localhost:{frontend_port}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
                stream=True,
            ) as response:
                logger.info(
                    f"Received response stream with status code: {response.status_code}"
                )

                if response.status_code != 200:
                    response_list.append(
                        (
                            Exception(
                                f"Request failed with status {response.status_code}: {response.text}"
                            ),
                            time.time(),
                        )
                    )
                    return

                for line in response.iter_lines():
                    if line:
                        response_list.append((line.decode("utf-8"), time.time()))

        except Exception as e:
            logger.error(f"Request failed with error: {e}")
            response_list.append((e, time.time()))

    request_thread = threading.Thread(target=send_request, daemon=True)
    request_thread.start()

    return request_thread, response_list


def determine_request_receiving_worker(
    worker1: ManagedProcess, worker2: ManagedProcess, receiving_pattern: str
) -> tuple:
    """
    Determine which worker received the request using parallel polling.

    Args:
        worker1: First worker process
        worker2: Second worker process
        receiving_pattern: Log pattern indicating request receipt

    Returns:
        Tuple of (worker_with_request, name_of_worker_with_request)
    """
    worker1_results: list[bool] = []
    worker2_results: list[bool] = []
    # Event to signal all threads to exit when one finds the pattern
    found_event = threading.Event()

    # Poll both workers in parallel
    def poll_worker(worker: ManagedProcess, result_list: list[bool]):
        max_wait_ms = 500
        poll_interval_ms = 5
        max_iterations = max_wait_ms // poll_interval_ms
        iteration = 0

        while iteration < max_iterations and not found_event.is_set():
            # Check if the worker logs contain the pattern
            try:
                with open(worker.log_path, "r") as f:
                    log_content = f.read()
                    if receiving_pattern in log_content:
                        result_list.append(True)
                        found_event.set()  # Signal other thread to exit
                        return
            except Exception as e:
                logger.error(f"Could not read log file {worker.log_path}: {e}")
                return

            time.sleep(poll_interval_ms / 1000.0)
            iteration += 1

    # Look for which worker received the request
    thread1 = threading.Thread(
        target=poll_worker, args=(worker1, worker1_results), daemon=True
    )
    thread2 = threading.Thread(
        target=poll_worker, args=(worker2, worker2_results), daemon=True
    )
    thread1.start()
    thread2.start()
    thread1.join(timeout=1)
    thread2.join(timeout=1)

    # Get results from lists
    worker1_received = worker1_results[0] if worker1_results else False
    worker2_received = worker2_results[0] if worker2_results else False

    if worker1_received and not worker2_received:
        logger.info("Request was received by Worker 1")
        return worker1, "Worker 1"
    elif worker2_received and not worker1_received:
        logger.info("Request was received by Worker 2")
        return worker2, "Worker 2"
    elif worker1_received and worker2_received:
        pytest.fail("Both workers received the request")
    else:
        pytest.fail("Neither worker received the request")


def wait_for_response(
    response_list: list[tuple[str | None | Exception, float]],
    num_responses: int = 5,
    max_wait_time: float = 10.0,
) -> None:
    """
    Block until num_responses new responses are received or max_wait_time is reached.

    Args:
        response_list: List being populated by background thread
        num_responses: Number of new responses to wait for (default 5)
        max_wait_time: Maximum time to wait in seconds (default 10s)
    """
    initial_len = len(response_list)
    target_len = initial_len + num_responses
    poll_interval = 0.001  # 1ms
    elapsed = 0.0

    while elapsed < max_wait_time:
        if len(response_list) >= target_len:
            return
        time.sleep(poll_interval)
        elapsed += poll_interval

    logger.warning(
        f"Only received {len(response_list) - initial_len}/{num_responses} new responses within {max_wait_time}s"
    )


def validate_response(
    request_thread: threading.Thread,
    response_list: list[tuple[str | None | Exception, float]],
) -> None:
    """
    Wait for and validate the streaming response after migration.
    Checks that delay before each response is reasonable (covers both TTFT and TPOT).

    Args:
        request_thread: The thread running the request
        response_list: List of (response_string | None | Exception, timestamp) tuples
    """
    request_thread.join(timeout=240)
    assert not request_thread.is_alive(), "Request did not complete within 240 seconds"

    assert len(response_list) > 0, "Missing first entry with start timestamp"
    assert response_list[0][0] is None, "First entry should be start timestamp only"
    prev_timestamp = response_list[0][1]

    response_words: list[str] = []
    for res, timestamp in response_list[1:]:
        delay = timestamp - prev_timestamp
        # Cold workers can take longer on first token - only warn but don't fail
        if delay > 2.0:
            logger.warning(f"Delay before response: {delay:.3f} secs")
            # Capture cases like migration is blocked by engine graceful shutdown
            assert delay <= 6.0, f"Delay before response > 6 secs, got {delay:.3f} secs"
        prev_timestamp = timestamp

        assert res is not None, "Response entry should not be None"
        if isinstance(res, Exception):
            raise res
        if res.startswith("event: error"):
            raise AssertionError(f"SSE error event received: {res}")

        assert res.startswith("data: "), f"Unexpected SSE line: {res}"
        data_str = res[6:]  # Remove "data: " prefix
        if data_str == "[DONE]":
            continue
        try:
            chunk = json.loads(data_str)
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                response_words.append(content)
        except Exception as e:
            raise AssertionError(f"Error parsing response chunk: {e}") from e

    logger.info(
        f"Received {len(response_words)} responses: {''.join(response_words)[:100]}..."
    )


def verify_migration_occurred(frontend_process: DynamoFrontendProcess) -> None:
    """
    Verify that migration occurred by checking frontend logs for stream disconnection message.

    Args:
        frontend_process: The frontend process to check logs for
    """
    log_path = frontend_process.log_path
    try:
        with open(log_path, "r") as f:
            log_content = f.read()
    except Exception as e:
        pytest.fail(f"Could not read frontend log file {log_path}: {e}")
    assert (
        "Stream disconnected... recreating stream..." in log_content
    ), "'Stream disconnected... recreating stream...' message not found in logs"
    assert (
        "Cannot recreate stream: " not in log_content
    ), "'Cannot recreate stream: ...' error found in logs"


def _parse_migration_metric(
    metrics_text: str, model_name: str, migration_type: str
) -> int:
    """
    Parse the migration metric value from Prometheus metrics text.

    Args:
        metrics_text: Raw Prometheus metrics text
        model_name: The model name label value
        migration_type: The migration_type label value ("ongoing_request" or "new_request")

    Returns:
        The metric count, or 0 if not found
    """
    import re

    # Match pattern like:
    # dynamo_frontend_model_migration_total{migration_type="ongoing_request",model="Qwen/Qwen3-0.6B"} 1
    # Labels can be in any order
    pattern = rf'dynamo_frontend_model_migration_total\{{[^}}]*migration_type="{migration_type}"[^}}]*model="{re.escape(model_name)}"[^}}]*\}}\s+(\d+)'
    match = re.search(pattern, metrics_text)

    if match:
        return int(match.group(1))

    # Try with labels in reverse order
    pattern = rf'dynamo_frontend_model_migration_total\{{[^}}]*model="{re.escape(model_name)}"[^}}]*migration_type="{migration_type}"[^}}]*\}}\s+(\d+)'
    match = re.search(pattern, metrics_text)

    if match:
        return int(match.group(1))

    return 0


def verify_migration_metrics(
    frontend_port: int,
    expected_ongoing_request_count: int = 0,
    expected_new_request_count: int = 0,
) -> None:
    """
    Verify migration metrics by querying the frontend's /metrics endpoint.

    Args:
        frontend_port: Port where the frontend is running
        expected_ongoing_request_count: Expected count of ongoing_request migrations
        expected_new_request_count: Expected count of new_request migrations
    """
    metrics_url = f"http://localhost:{frontend_port}/metrics"

    try:
        response = requests.get(metrics_url, timeout=1)
        response.raise_for_status()
    except requests.RequestException as e:
        pytest.fail(f"Failed to fetch metrics from {metrics_url}: {e}")

    metrics_text = response.text
    logger.info(f"Fetched metrics from {metrics_url}")

    # Parse metrics to find migration counts
    ongoing_count = _parse_migration_metric(
        metrics_text, FAULT_TOLERANCE_MODEL_NAME, "ongoing_request"
    )
    new_request_count = _parse_migration_metric(
        metrics_text, FAULT_TOLERANCE_MODEL_NAME, "new_request"
    )

    logger.info(
        f"Migration metrics - ongoing_request: {ongoing_count}, new_request: {new_request_count}"
    )

    if expected_ongoing_request_count > 0:
        assert ongoing_count >= expected_ongoing_request_count, (
            f"Expected at least {expected_ongoing_request_count} ongoing_request migrations, "
            f"but got {ongoing_count}"
        )

    if expected_new_request_count > 0:
        assert new_request_count >= expected_new_request_count, (
            f"Expected at least {expected_new_request_count} new_request migrations, "
            f"but got {new_request_count}"
        )
