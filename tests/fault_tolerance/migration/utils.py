# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext

import pytest
import requests
from openai import APIError, OpenAI

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import (
    DynamoFrontendProcess as BaseDynamoFrontendProcess,
)
from tests.utils.managed_process import ManagedProcess, terminate_process_tree
from tests.utils.prometheus import sum_metric_samples

logger = logging.getLogger(__name__)


@contextmanager
def managed_processes_concurrently(
    *processes: ManagedProcess,
) -> Iterator[tuple[ManagedProcess, ...]]:
    """Enter independent managed processes concurrently and clean them up safely."""
    if not processes:
        yield ()
        return

    entered: list[ManagedProcess | None] = [None] * len(processes)
    startup_error: BaseException | None = None
    with ThreadPoolExecutor(max_workers=len(processes)) as executor:
        futures = [executor.submit(process.__enter__) for process in processes]
        for index, future in enumerate(futures):
            try:
                entered[index] = future.result()
            except BaseException as error:
                if startup_error is None:
                    startup_error = error

    with ExitStack() as stack:
        for process in entered:
            if process is not None:
                stack.callback(process.__exit__, None, None, None)

        if startup_error is not None:
            raise startup_error

        yield tuple(process for process in entered if process is not None)


class DynamoFrontendProcess(BaseDynamoFrontendProcess):
    """Fault-tolerance frontend wrapper (keeps env settings from the historical helper)."""

    def __init__(
        self,
        request,
        migration_limit: int,
        migration_max_seq_len: int | None,
        startup_timeout_s: int = 300,
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
        self.timeout = startup_timeout_s


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
    frontend_port: int,
    stream: bool,
    use_long_prompt: bool = False,
    max_tokens: int | None = None,
    long_prompt_repetitions: int = 8_000,
    force_max_output_tokens: bool = False,
) -> tuple:
    """
    Start a long-running completion request in a separate thread.

    Responses are processed internally to extract content. First entry is (None, start_time)
    to mark when request was sent. Subsequent entries contain extracted content or exceptions.

    Args:
        frontend_port: Port where the frontend is running
        stream: Whether to use streaming responses
        use_long_prompt: Whether to use a long prompt (~8000 tokens)
        max_tokens: Explicit output-token cap, or the backend default when unset
        long_prompt_repetitions: Number of repeated words in the long prompt
        force_max_output_tokens: Disable EOS and require the full output-token
            budget. Requires max_tokens.

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
            prompt += " Make sure it is" + " long" * long_prompt_repetitions + "!"

        logger.info(
            "Sending completion request (stream=%s) with prompt: '%s...'",
            stream,
            prompt[:50],
        )

        response_list.append((None, time.monotonic()))  # start observation

        try:
            client = _make_client(frontend_port)
            request_args = {
                "model": FAULT_TOLERANCE_MODEL_NAME,
                "prompt": prompt,
                "stream": stream,
                "temperature": 0,
                "seed": 0,
            }
            if max_tokens is not None:
                request_args["max_tokens"] = max_tokens
            if force_max_output_tokens:
                if max_tokens is None:
                    raise ValueError("force_max_output_tokens requires max_tokens")
                request_args["extra_body"] = {
                    "ignore_eos": True,
                    "min_tokens": max_tokens,
                }
            if stream:
                for chunk in client.completions.create(**request_args):
                    text = chunk.choices[0].text if chunk.choices else None
                    # Match the original hand-rolled parser: keep empty strings,
                    # drop only None. Empty chunks (e.g. the first stream frame)
                    # still count as a response arrival for delay measurement.
                    if text is not None:
                        response_list.append((text, time.monotonic()))
            else:
                resp = client.completions.create(**request_args)
                response_list.append((resp.choices[0].text, time.monotonic()))
        except Exception as error:
            # openai.APIError subclasses cover HTTP non-200, mid-stream
            # structured `data: {"error": {...}}` frames, connection failures,
            # and timeouts. Non-openai exceptions (network, etc.) also bubble.
            logger.error("Request failed with error: %s", error)
            response_list.append((error, time.monotonic()))

    request_thread = threading.Thread(target=send_request, daemon=True)
    request_thread.start()

    return request_thread, response_list


def start_chat_completion_request(
    frontend_port: int,
    stream: bool,
    use_long_prompt: bool = False,
    max_tokens: int | None = None,
    long_prompt_repetitions: int = 8_000,
    force_max_output_tokens: bool = False,
) -> tuple:
    """
    Start a long-running chat completion request in a separate thread.

    Responses are processed internally to extract content. First entry is (None, start_time)
    to mark when request was sent. Subsequent entries contain extracted content or exceptions.

    Args:
        frontend_port: Port where the frontend is running
        stream: Whether to use streaming responses
        use_long_prompt: Whether to use a long prompt (~8000 tokens)
        max_tokens: Explicit output-token cap, or the backend default when unset
        long_prompt_repetitions: Number of repeated words in the long prompt
        force_max_output_tokens: Disable EOS and require the full output-token
            budget. Requires max_tokens.

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
            prompt += " Make sure it is" + " long" * long_prompt_repetitions + "!"

        logger.info(
            "Sending chat completion request (stream=%s) with prompt: '%s...'",
            stream,
            prompt[:50],
        )

        response_list.append((None, time.monotonic()))  # start observation

        try:
            client = _make_client(frontend_port)
            request_args = {
                "model": FAULT_TOLERANCE_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream,
                "temperature": 0,
                "seed": 0,
            }
            if max_tokens is not None:
                request_args["max_tokens"] = max_tokens
            if force_max_output_tokens:
                if max_tokens is None:
                    raise ValueError("force_max_output_tokens requires max_tokens")
                request_args["extra_body"] = {
                    "ignore_eos": True,
                    "min_tokens": max_tokens,
                }
            if stream:
                for chunk in client.chat.completions.create(**request_args):
                    content = chunk.choices[0].delta.content if chunk.choices else None
                    # Match the original hand-rolled parser: keep empty strings,
                    # drop only None. Empty chunks (e.g. the first `role`-only
                    # stream frame) still count as a response arrival for delay
                    # measurement.
                    if content is not None:
                        response_list.append((content, time.monotonic()))
            else:
                resp = client.chat.completions.create(**request_args)
                response_list.append(
                    (resp.choices[0].message.content, time.monotonic())
                )
        except Exception as error:
            # openai.APIError subclasses cover HTTP non-200, mid-stream
            # structured `data: {"error": {...}}` frames, connection failures,
            # and timeouts. Non-openai exceptions also bubble for visibility.
            logger.error("Request failed with error: %s", error)
            response_list.append((error, time.monotonic()))

    request_thread = threading.Thread(target=send_request, daemon=True)
    request_thread.start()

    return request_thread, response_list


def determine_request_receiving_worker(
    worker1: ManagedProcess, worker2: ManagedProcess, receiving_pattern: str
) -> tuple[ManagedProcess, str, str]:
    """
    Determine which worker received the request while inspecting both logs together.

    Args:
        worker1: First worker process
        worker2: Second worker process
        receiving_pattern: Log pattern indicating request receipt

    Returns:
        Tuple of (worker_with_request, name_of_worker_with_request, request_id)
    """
    # Engine logs are written asynchronously and can arrive noticeably later
    # under loaded CI nodes. The first request can also spend more than ten
    # seconds in cold frontend preprocessing before it reaches a worker. Keep
    # polling the receipt condition rather than tearing workers down while that
    # request is still being prepared. See the aggregate vLLM migration
    # regression in #9465.
    max_wait_s = 30.0
    poll_interval_s = 0.1
    request_re = re.compile(re.escape(receiving_pattern) + r"(?P<request_id>\S+)")
    poll_event = threading.Event()

    def request_ids(worker: ManagedProcess) -> list[str]:
        try:
            with open(worker.log_path, "r") as log_file:
                return request_re.findall(log_file.read())
        except FileNotFoundError:
            return []
        except OSError as error:
            logger.warning("Could not read log file %s: %s", worker.log_path, error)
            return []

    deadline = time.monotonic() + max_wait_s
    last_worker1_ids: list[str] = []
    last_worker2_ids: list[str] = []
    while time.monotonic() < deadline:
        last_worker1_ids = request_ids(worker1)
        last_worker2_ids = request_ids(worker2)

        if last_worker1_ids and last_worker2_ids:
            pytest.fail(
                "Both candidate workers received a request before fault injection: "
                f"worker1={last_worker1_ids}, worker2={last_worker2_ids}"
            )
        if last_worker1_ids:
            request_id = last_worker1_ids[-1]
            logger.info("Request %s was received by Worker 1", request_id)
            return worker1, "Worker 1", request_id
        if last_worker2_ids:
            request_id = last_worker2_ids[-1]
            logger.info("Request %s was received by Worker 2", request_id)
            return worker2, "Worker 2", request_id

        poll_event.wait(timeout=poll_interval_s)

    pytest.fail(
        f"Neither worker logged {receiving_pattern!r} within {max_wait_s}s; "
        f"worker1_ids={last_worker1_ids}, worker2_ids={last_worker2_ids}"
    )


def wait_for_worker_request_id(
    worker: ManagedProcess,
    receiving_pattern: str,
    request_id: str,
    max_wait_time: float = 30.0,
) -> None:
    """Wait until the replacement worker accepts the exact migrated request."""
    expected = f"{receiving_pattern}{request_id}"
    deadline = time.monotonic() + max_wait_time
    last_error: OSError | None = None
    poll_event = threading.Event()

    while time.monotonic() < deadline:
        try:
            with open(worker.log_path, "r") as log_file:
                if expected in log_file.read():
                    logger.info(
                        "Replacement worker %s accepted request %s",
                        worker.log_path,
                        request_id,
                    )
                    return
        except FileNotFoundError:
            pass
        except OSError as error:
            last_error = error

        poll_event.wait(timeout=0.1)

    pytest.fail(
        f"Replacement worker did not log {expected!r} within {max_wait_time}s; "
        f"last_error={last_error}"
    )


def wait_for_endpoint_instances(
    frontend_port: int,
    expected_counts: dict[tuple[str, str], int],
    max_wait_time: float = 10.0,
) -> None:
    """Wait until the frontend's discovery view contains every required endpoint."""
    deadline = time.monotonic() + max_wait_time
    last_counts: dict[tuple[str, str], int] = {}
    last_error: Exception | None = None
    poll_event = threading.Event()

    while time.monotonic() < deadline:
        try:
            response = requests.get(
                f"http://localhost:{frontend_port}/health",
                timeout=1,
            )
            response.raise_for_status()
            instances = response.json().get("instances", [])
            last_counts = {}
            for instance in instances:
                key = (instance.get("component"), instance.get("endpoint"))
                last_counts[key] = last_counts.get(key, 0) + 1

            if all(
                last_counts.get(endpoint, 0) >= expected
                for endpoint, expected in expected_counts.items()
            ):
                logger.info("Frontend discovery is ready: %s", last_counts)
                return
        except (requests.RequestException, ValueError) as error:
            last_error = error

        poll_event.wait(timeout=0.1)

    pytest.fail(
        "Frontend discovery did not reach the required endpoint counts "
        f"{expected_counts} within {max_wait_time}s; last counts={last_counts}, "
        f"last error={last_error}"
    )


def wait_for_endpoint_instance_reduction(
    frontend_port: int,
    endpoint: tuple[str, str],
    previous_count: int,
    max_wait_time: float = 10.0,
) -> None:
    """Wait until graceful shutdown removes one endpoint from discovery."""
    if previous_count < 1:
        pytest.fail(
            f"Cannot observe removal of {endpoint}: initial count was {previous_count}"
        )

    deadline = time.monotonic() + max_wait_time
    last_count = previous_count
    last_error: Exception | None = None
    poll_event = threading.Event()

    while time.monotonic() < deadline:
        try:
            response = requests.get(
                f"http://localhost:{frontend_port}/health",
                timeout=1,
            )
            response.raise_for_status()
            instances = response.json().get("instances", [])
            last_count = sum(
                1
                for instance in instances
                if (instance.get("component"), instance.get("endpoint")) == endpoint
            )
            if last_count < previous_count:
                logger.info(
                    "Graceful shutdown reduced %s discovery instances: %s -> %s",
                    endpoint,
                    previous_count,
                    last_count,
                )
                return
        except (requests.RequestException, ValueError) as error:
            last_error = error

        poll_event.wait(timeout=0.1)

    pytest.fail(
        f"Graceful shutdown did not reduce {endpoint} discovery instances below "
        f"{previous_count} within {max_wait_time}s; last count={last_count}, "
        f"last error={last_error}"
    )


def wait_for_response(
    response_list: list[tuple[str | None | Exception, float]],
    num_responses: int = 5,
    max_wait_time: float = 10.0,
) -> None:
    """
    Block until at least ``num_responses`` non-empty payload chunks exist.

    Args:
        response_list: List being populated by background thread
        num_responses: Absolute minimum number of non-empty payload chunks (default 5)
        max_wait_time: Maximum time to wait in seconds (default 10s)
    """
    poll_interval = 0.001  # 1ms
    deadline = time.monotonic() + max_wait_time

    while time.monotonic() < deadline:
        content_count = sum(
            1 for response, _ in response_list if isinstance(response, str) and response
        )
        if content_count >= num_responses:
            return
        time.sleep(poll_interval)

    content_count = sum(
        1 for response, _ in response_list if isinstance(response, str) and response
    )
    pytest.fail(
        f"Only observed {content_count}/{num_responses} non-empty response chunks "
        f"within {max_wait_time}s"
    )


def wait_for_worker_generate_completion(
    worker_system_port: int,
    component: str = "backend",
    max_wait_time: float = 10.0,
) -> None:
    """Prove the replacement worker drained one request and emitted response bytes."""
    metrics_url = f"http://localhost:{worker_system_port}/metrics"
    labels = {"dynamo_component": component, "dynamo_endpoint": "generate"}
    deadline = time.monotonic() + max_wait_time
    duration_count = 0.0
    response_bytes = 0.0
    last_error: Exception | None = None
    poll_event = threading.Event()

    while time.monotonic() < deadline:
        try:
            response = requests.get(metrics_url, timeout=1)
            response.raise_for_status()
            duration_count = sum_metric_samples(
                response.text,
                "dynamo_component_request_duration_seconds_count",
                labels,
            )
            response_bytes = sum_metric_samples(
                response.text,
                "dynamo_component_response_bytes_total",
                labels,
            )
            if duration_count == 1 and response_bytes > 0:
                logger.info(
                    "Replacement worker completed one request with %s response bytes",
                    response_bytes,
                )
                return
        except (requests.RequestException, ValueError) as error:
            last_error = error

        poll_event.wait(timeout=0.1)

    pytest.fail(
        "Replacement worker did not complete exactly one generate request with "
        f"response bytes within {max_wait_time}s; duration_count={duration_count}, "
        f"response_bytes={response_bytes}, last_error={last_error}"
    )


def validate_response(
    request_thread: threading.Thread,
    response_list: list[tuple[str | None | Exception, float]],
) -> None:
    """
    Wait for and validate the response after migration.
    Timing observations are logged for diagnosis, but they are not correctness
    assertions. Loaded CI nodes and concurrent GPU tests can legitimately change
    TTFT/TPOT without changing migration behavior.

    Args:
        request_thread: The thread running the request
        response_list: List of (content_string | None | Exception, timestamp) tuples.
                       Content is already parsed - no SSE format parsing needed.
    """
    request_thread.join(timeout=240)
    assert not request_thread.is_alive(), "Request did not complete within 240 seconds"

    assert len(response_list) > 0, "Missing first entry with start timestamp"
    assert response_list[0][0] is None, "First entry should be start timestamp only"
    prev_timestamp = response_list[0][1]

    response_words: list[str] = []
    for res, timestamp in response_list[1:]:
        delay = timestamp - prev_timestamp
        if delay > 2.0:
            logger.info("Observed %.3fs before the next response chunk", delay)
        prev_timestamp = timestamp

        assert res is not None, "Response entry should not be None"
        if isinstance(res, Exception):
            raise res

        # Content is already parsed - just collect it
        response_words.append(res)

    assert response_words, "Request completed without any response content"
    logger.info(
        "Received %s response(s): %s...",
        len(response_words),
        "".join(response_words)[:100],
    )


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
    exact_counts: bool = False,
) -> None:
    """
    Verify migration metrics by querying the frontend's /metrics endpoint.

    Args:
        frontend_port: Port where the frontend is running
        expected_ongoing_request_count: Expected count of ongoing_request migrations
        expected_new_request_count: Expected count of new_request migrations
        expected_max_seq_len_exceeded_count: Expected count of max_seq_len exceeded events
        exact_counts: Require exact ongoing/new-request counts instead of the
            shared helper's historical lower-bound assertions
    """
    metrics_url = f"http://localhost:{frontend_port}/metrics"

    try:
        response = requests.get(metrics_url, timeout=1)
        response.raise_for_status()
    except requests.RequestException as e:
        pytest.fail(f"Failed to fetch metrics from {metrics_url}: {e}")

    metrics_text = response.text
    logger.info("Fetched metrics from %s", metrics_url)

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
        "Migration metrics - ongoing_request: %s, new_request: %s, "
        "max_seq_len_exceeded: %s",
        ongoing_count,
        new_request_count,
        max_seq_len_exceeded_count,
    )

    if exact_counts:
        assert ongoing_count == expected_ongoing_request_count, (
            f"Expected {expected_ongoing_request_count} ongoing_request migrations, "
            f"but got {ongoing_count}"
        )
        assert new_request_count == expected_new_request_count, (
            f"Expected {expected_new_request_count} new_request migrations, "
            f"but got {new_request_count}"
        )
    else:
        if expected_ongoing_request_count > 0:
            assert ongoing_count >= expected_ongoing_request_count, (
                f"Expected at least {expected_ongoing_request_count} "
                f"ongoing_request migrations, but got {ongoing_count}"
            )
        if expected_new_request_count > 0:
            assert new_request_count >= expected_new_request_count, (
                f"Expected at least {expected_new_request_count} "
                f"new_request migrations, but got {new_request_count}"
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
    max_tokens: int | None = None,
    use_long_prompt: bool = False,
    long_prompt_repetitions: int = 8_000,
    wait_for_new_response_before_stop: bool = False,
    expected_ongoing_request_count: int | None = None,
    graceful_shutdown: Callable[[ManagedProcess], AbstractContextManager[None]]
    | None = None,
    verify_replacement_worker: bool = False,
    force_max_output_tokens: bool = False,
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
        max_tokens: Explicit output-token cap, or the backend default when unset
        use_long_prompt: Whether to use long prompt (for prefill tests)
        long_prompt_repetitions: Number of repeated words in the long prompt
        wait_for_new_response_before_stop: Whether to wait for response before stopping (for decode tests)
        expected_ongoing_request_count: Exact expected count for callers that
            opt into strict metric validation. When omitted, preserve the
            shared helper's historical backend-agnostic lower-bound behavior.
        graceful_shutdown: Optional backend-specific context that initiates
            graceful shutdown before response validation and performs final
            cleanup after the request outcome is known.
        verify_replacement_worker: Require the surviving worker to accept the
            exact request ID and expose one completed generate request with
            nonzero response bytes. Intended for isolated per-test workers.
        force_max_output_tokens: Disable EOS and require the request's full
            max_tokens budget so fault injection cannot race an early EOS.
    """
    # Step 1: Send the request
    if use_chat_completion:
        request_thread, response_list = start_chat_completion_request(
            frontend.frontend_port,
            stream=stream,
            use_long_prompt=use_long_prompt,
            max_tokens=max_tokens,
            long_prompt_repetitions=long_prompt_repetitions,
            force_max_output_tokens=force_max_output_tokens,
        )
    else:
        request_thread, response_list = start_completion_request(
            frontend.frontend_port,
            stream=stream,
            use_long_prompt=use_long_prompt,
            max_tokens=max_tokens,
            long_prompt_repetitions=long_prompt_repetitions,
            force_max_output_tokens=force_max_output_tokens,
        )

    # Step 2: Determine which worker received the request
    worker, worker_name, request_id = determine_request_receiving_worker(
        worker1, worker2, receiving_pattern=receiving_pattern
    )
    replacement_worker = worker2 if worker is worker1 else worker1
    assert (
        request_thread.is_alive()
    ), "Request completed before the migration fault could be injected"

    # Step 3: Optionally wait for new response before stop (for decode tests)
    if wait_for_new_response_before_stop:
        wait_for_response(response_list)
        assert (
            request_thread.is_alive()
        ), "Request completed before the worker fault was injected"

    # Step 4: Stop the worker (kill or graceful shutdown)
    shutdown_context: AbstractContextManager[None] = nullcontext()
    if immediate_kill:
        logger.info("Killing %s with PID %s", worker_name, worker.get_pid())
        terminate_process_tree(worker.get_pid(), immediate_kill=True, timeout=0)
    else:
        logger.info(
            "Gracefully shutting down %s with PID %s",
            worker_name,
            worker.get_pid(),
        )
        if graceful_shutdown is None:
            terminate_process_tree(worker.get_pid(), immediate_kill=False, timeout=2)
        else:
            shutdown_context = graceful_shutdown(worker)

    # Step 5: Validate the request outcome via its response (the user-facing
    # contract). Migration is expected to succeed only when it is enabled and the
    # request does not exceed the migration seq-len cap; otherwise the in-flight
    # request must fail.
    with shutdown_context:
        if migration_limit > 0 and migration_max_seq_len != 1:
            if verify_replacement_worker:
                wait_for_worker_request_id(
                    replacement_worker,
                    receiving_pattern,
                    request_id,
                )
            validate_response(request_thread, response_list)
            if verify_replacement_worker:
                worker_system_port = getattr(replacement_worker, "system_port", None)
                assert isinstance(
                    worker_system_port, int
                ), "Replacement-worker verification requires an integer system_port"
                wait_for_worker_generate_completion(worker_system_port)
        else:
            # openai.APIError covers both mid-stream structured error frames and
            # HTTP non-200 responses.
            with pytest.raises(APIError):
                validate_response(request_thread, response_list)

    # Step 6: Verify that migration behaved as expected via the frontend's
    # Prometheus metrics (a stable structured surface) instead of asserting on
    # log strings. `ongoing_request` counts an error from an established
    # stream, including an attempt that cannot retry because migration_limit is
    # zero. It is the structured equivalent of the old "Stream disconnected,
    # recreating stream" log assertion. `max_seq_len_exceeded` records hitting
    # the migration seq-len cap.
    exact_metric_counts = expected_ongoing_request_count is not None
    if expected_ongoing_request_count is None:
        expected_ongoing_request_count = 1 if migration_limit > 0 else 0

    verify_migration_metrics(
        frontend.frontend_port,
        expected_ongoing_request_count=expected_ongoing_request_count,
        expected_max_seq_len_exceeded_count=1 if migration_max_seq_len == 1 else 0,
        exact_counts=exact_metric_counts,
    )
