# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import threading
import time

import pytest
import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import (
    DynamoFrontendProcess as BaseDynamoFrontendProcess,
)
from tests.utils.managed_process import ManagedProcess, terminate_process_tree

openai = pytest.importorskip(
    "openai", reason="openai package is required for fault tolerance migration tests"
)
APIError = openai.APIError
OpenAI = openai.OpenAI

logger = logging.getLogger(__name__)


class DynamoFrontendProcess(BaseDynamoFrontendProcess):
    """Fault-tolerance frontend wrapper (keeps env settings from the historical helper)."""

    def __init__(
        self,
        request,
        migration_limit: int,
        migration_max_seq_len: int | None,
    ):
        extra_env = {
            "DYN_REQUEST_PLANE": request.getfixturevalue("request_plane"),
            # These tests expect full control over requests sent to workers. The canary
            # health check can inject extra requests and cause intermittent failures.
            "DYN_HEALTH_CHECK_ENABLED": "false",
        }
        super().__init__(
            request,
            frontend_port=0,  # allocate a free port (xdist-safe)
            router_mode="round-robin",
            migration_limit=migration_limit,
            migration_max_seq_len=migration_max_seq_len,
            extra_env=extra_env,
            terminate_all_matching_process_names=False,
            display_name="frontend",
        )


def _make_client(frontend_port: int) -> OpenAI:
    """Build an OpenAI client pointed at the test frontend.

    max_retries=0 so fault-tolerance tests see the first error instead of
    silent retries; api_key is a placeholder since the frontend doesn't auth.
    """
    return OpenAI(
        base_url=f"http://localhost:{frontend_port}/v1",
        api_key="not-needed",
        max_retries=0,
        timeout=240,
    )


def start_completion_request(
    frontend_port: int, stream: bool, use_long_prompt: bool = False
) -> tuple:
    """
    Start a long-running completion request in a separate thread.

    Responses are processed internally to extract content. First entry is (None, start_time)
    to mark when request was sent. Subsequent entries contain extracted content or exceptions.

    Args:
        frontend_port: Port where the frontend is running
        stream: Whether to use streaming responses
        use_long_prompt: Whether to use a long prompt (~8000 tokens)

    Returns:
        tuple: (request_thread, response_list) where response_list contains
               (str | None | Exception, float) tuples.
               - For streaming: each entry is (content_word, timestamp)
               - For non-streaming: single entry is (full_content, timestamp)
    """
    response_list: list[tuple[str | None | Exception, float]] = []

    def send_request():
        prompt = "Tell me a long long long story about yourself?"
        if use_long_prompt:
            prompt += " Make sure it is" + " long" * 8000 + "!"

        logger.info(
            f"Sending completion request (stream={stream}) with prompt: '{prompt[:50]}...'"
        )

        response_list.append((None, time.time()))  # start timestamp

        try:
            client = _make_client(frontend_port)
            if stream:
                for chunk in client.completions.create(
                    model=FAULT_TOLERANCE_MODEL_NAME,
                    prompt=prompt,
                    stream=True,
                ):
                    text = chunk.choices[0].text if chunk.choices else None
                    # Match the original hand-rolled parser: keep empty strings,
                    # drop only None. Empty chunks (e.g. the first stream frame)
                    # still count as a response arrival for delay measurement.
                    if text is not None:
                        response_list.append((text, time.time()))
            else:
                resp = client.completions.create(
                    model=FAULT_TOLERANCE_MODEL_NAME,
                    prompt=prompt,
                    stream=False,
                )
                response_list.append((resp.choices[0].text, time.time()))
        except Exception as e:
            # openai.APIError subclasses cover HTTP non-200, mid-stream
            # structured `data: {"error": {...}}` frames, connection failures,
            # and timeouts. Non-openai exceptions (network, etc.) also bubble.
            logger.error(f"Request failed with error: {e}")
            response_list.append((e, time.time()))

    request_thread = threading.Thread(target=send_request, daemon=True)
    request_thread.start()

    return request_thread, response_list


def start_chat_completion_request(
    frontend_port: int, stream: bool, use_long_prompt: bool = False
) -> tuple:
    """
    Start a long-running chat completion request in a separate thread.

    Responses are processed internally to extract content. First entry is (None, start_time)
    to mark when request was sent. Subsequent entries contain extracted content or exceptions.

    Args:
        frontend_port: Port where the frontend is running
        stream: Whether to use streaming responses
        use_long_prompt: Whether to use a long prompt (~8000 tokens)

    Returns:
        tuple: (request_thread, response_list) where response_list contains
               (str | None | Exception, float) tuples.
               - For streaming: each entry is (content_word, timestamp)
               - For non-streaming: single entry is (full_content, timestamp)
    """
    response_list: list[tuple[str | None | Exception, float]] = []

    def send_request():
        prompt = "Tell me a long long long story about yourself?"
        if use_long_prompt:
            prompt += " Make sure it is" + " long" * 8000 + "!"

        logger.info(
            f"Sending chat completion request (stream={stream}) with prompt: '{prompt[:50]}...'"
        )

        response_list.append((None, time.time()))  # start timestamp

        try:
            client = _make_client(frontend_port)
            if stream:
                for chunk in client.chat.completions.create(
                    model=FAULT_TOLERANCE_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                ):
                    content = chunk.choices[0].delta.content if chunk.choices else None
                    # Match the original hand-rolled parser: keep empty strings,
                    # drop only None. Empty chunks (e.g. the first `role`-only
                    # stream frame) still count as a response arrival for delay
                    # measurement.
                    if content is not None:
                        response_list.append((content, time.time()))
            else:
                resp = client.chat.completions.create(
                    model=FAULT_TOLERANCE_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                )
                response_list.append((resp.choices[0].message.content, time.time()))
        except Exception as e:
            # openai.APIError subclasses cover HTTP non-200, mid-stream
            # structured `data: {"error": {...}}` frames, connection failures,
            # and timeouts. Non-openai exceptions also bubble for visibility.
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
    validate_delay: bool = True,
) -> None:
    """
    Wait for and validate the response after migration.
    Checks that delay before each response is reasonable (covers both TTFT and TPOT).

    Args:
        request_thread: The thread running the request
        response_list: List of (content_string | None | Exception, timestamp) tuples.
                       Content is already parsed - no SSE format parsing needed.
        validate_delay: Whether to validate delay before each response.
    """
    request_thread.join(timeout=240)
    assert not request_thread.is_alive(), "Request did not complete within 240 seconds"

    assert len(response_list) > 0, "Missing first entry with start timestamp"
    assert response_list[0][0] is None, "First entry should be start timestamp only"
    prev_timestamp = response_list[0][1]

    response_words: list[str] = []
    for res, timestamp in response_list[1:]:
        delay = timestamp - prev_timestamp
        if delay > 2.0 and validate_delay:
            # Cold workers can take longer on first token - only warn but don't fail
            logger.warning(f"Delay before response: {delay:.3f} secs")
            # Capture cases like migration is blocked by engine graceful shutdown
            assert delay <= 6.0, f"Delay before response > 6 secs, got {delay:.3f} secs"
        prev_timestamp = timestamp

        assert res is not None, "Response entry should not be None"
        if isinstance(res, Exception):
            raise res

        # Content is already parsed - just collect it
        response_words.append(res)

    logger.info(
        f"Received {len(response_words)} response(s): {''.join(response_words)[:100]}..."
    )


def verify_migration_occurred(frontend_process: DynamoFrontendProcess) -> None:
    """
    Verify that migration occurred by checking frontend logs for stream disconnection message.

    Args:
        frontend_process: The frontend process to check logs for
    """
    log_path = frontend_process.log_path
    log_content = ""
    for i in range(10):
        try:
            with open(log_path, "r") as f:
                log_content = f.read()
        except Exception as e:
            pytest.fail(f"Could not read frontend log file {log_path}: {e}")
        # Make sure this message is captured if any with the polling
        if "Cannot recreate stream: " in log_content:
            break
        time.sleep(0.005)

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


def _parse_migration_max_seq_len_exceeded_metric(
    metrics_text: str, model_name: str
) -> int:
    """
    Parse the migration max_seq_len exceeded counter from Prometheus metrics text.

    Returns:
        The metric count, or 0 if not found
    """
    pattern = rf'dynamo_frontend_model_migration_max_seq_len_exceeded_total\{{[^}}]*model="{re.escape(model_name)}"[^}}]*\}}\s+(\d+)'
    match = re.search(pattern, metrics_text)
    return int(match.group(1)) if match else 0


def verify_migration_metrics(
    frontend_port: int,
    expected_ongoing_request_count: int = 0,
    expected_new_request_count: int = 0,
    expected_max_seq_len_exceeded_count: int = 0,
) -> None:
    """
    Verify migration metrics by querying the frontend's /metrics endpoint.

    Args:
        frontend_port: Port where the frontend is running
        expected_ongoing_request_count: Expected count of ongoing_request migrations
        expected_new_request_count: Expected count of new_request migrations
        expected_max_seq_len_exceeded_count: Expected count of max_seq_len exceeded events
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
    max_seq_len_exceeded_count = _parse_migration_max_seq_len_exceeded_metric(
        metrics_text, FAULT_TOLERANCE_MODEL_NAME
    )

    logger.info(
        f"Migration metrics - ongoing_request: {ongoing_count}, "
        f"new_request: {new_request_count}, "
        f"max_seq_len_exceeded: {max_seq_len_exceeded_count}"
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

    assert max_seq_len_exceeded_count == expected_max_seq_len_exceeded_count, (
        f"Expected {expected_max_seq_len_exceeded_count} "
        f"max_seq_len_exceeded events, but got {max_seq_len_exceeded_count}"
    )


def run_migration_test(
    frontend: DynamoFrontendProcess,
    worker1: ManagedProcess,
    worker2: ManagedProcess,
    receiving_pattern: str,
    migration_limit: int,
    migration_max_seq_len: int | None,
    immediate_kill: bool,
    use_chat_completion: bool,
    stream: bool,
    use_long_prompt: bool = False,
    wait_for_new_response_before_stop: bool = False,
) -> None:
    """
    Run the common migration test flow after frontend and workers are started.

    Args:
        frontend: The frontend process
        worker1: First worker process
        worker2: Second worker process
        receiving_pattern: Log pattern to identify which worker received the request
        migration_limit: Migration limit setting (0 = disabled)
        migration_max_seq_len: Max sequence length for migration (None = no limit)
        immediate_kill: True for immediate kill, False for graceful shutdown
        use_chat_completion: Whether to use chat completion API (True) or completion API (False)
        stream: Whether to use streaming responses
        use_long_prompt: Whether to use long prompt (for prefill tests)
        wait_for_new_response_before_stop: Whether to wait for response before stopping (for decode tests)
    """
    # Step 1: Send the request
    if use_chat_completion:
        request_thread, response_list = start_chat_completion_request(
            frontend.frontend_port, stream=stream, use_long_prompt=use_long_prompt
        )
    else:
        request_thread, response_list = start_completion_request(
            frontend.frontend_port, stream=stream, use_long_prompt=use_long_prompt
        )

    # Step 2: Determine which worker received the request
    worker, worker_name = determine_request_receiving_worker(
        worker1, worker2, receiving_pattern=receiving_pattern
    )

    # Step 3: Optionally wait for new response before stop (for decode tests)
    if wait_for_new_response_before_stop:
        wait_for_response(response_list)

    # Step 4: Stop the worker (kill or graceful shutdown)
    if immediate_kill:
        logger.info(f"Killing {worker_name} with PID {worker.get_pid()}")
        terminate_process_tree(worker.get_pid(), immediate_kill=True, timeout=0)
    else:
        logger.info(
            f"Gracefully shutting down {worker_name} with PID {worker.get_pid()}"
        )
        terminate_process_tree(worker.get_pid(), immediate_kill=False, timeout=10)

    # Step 5: Validate response and verify migration occurred.
    # migration_enabled and not max_seq_len_exceeded -> migration should succeed
    if migration_limit > 0 and migration_max_seq_len != 1:
        validate_response(request_thread, response_list, validate_delay=stream)
        verify_migration_occurred(frontend)
    else:
        try:
            validate_response(request_thread, response_list, validate_delay=stream)
            pytest.fail(
                "Request succeeded unexpectedly when migration should have failed"
            )
        except APIError as e:
            # Expected: openai.APIError covers mid-stream structured error
            # frames (DIS-1768 contract) and HTTP non-200 responses. A typed
            # check is more robust than matching the exception's stringified
            # message against a specific wire-format prefix.
            logger.info(f"Got expected APIError: {e}")

        try:
            verify_migration_occurred(frontend)
            pytest.fail("Migration unexpectedly succeeded")
        except AssertionError as e:
            assert "'Cannot recreate stream: ...' error found in logs" in str(e)

    # Step 6: Verify migration metrics
    verify_migration_metrics(
        frontend.frontend_port,
        expected_ongoing_request_count=1 if migration_limit > 0 else 0,
        expected_max_seq_len_exceeded_count=1 if migration_max_seq_len == 1 else 0,
    )
