# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext

import pytest
import requests
from openai import APIError

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import (
    DynamoFrontendProcess as BaseDynamoFrontendProcess,
)
from tests.utils.managed_process import ManagedProcess, terminate_process_tree
from tests.utils.prometheus import sum_metric_samples

from .request_utils import start_request, validate_response, wait_for_response

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
    startup_cancelled = threading.Event()
    entered_lock = threading.Lock()

    def enter_process(index: int, process: ManagedProcess) -> ManagedProcess:
        entered_process = process.__enter__()
        with entered_lock:
            if startup_cancelled.is_set():
                cleanup_immediately = True
            else:
                entered[index] = entered_process
                cleanup_immediately = False

        if cleanup_immediately:
            try:
                entered_process.__exit__(None, None, None)
            except Exception:
                logger.exception("Failed to clean up a late-starting process")
        return entered_process

    executor = ThreadPoolExecutor(max_workers=len(processes))
    futures = [
        executor.submit(enter_process, index, process)
        for index, process in enumerate(processes)
    ]

    def cancel_startup() -> None:
        startup_cancelled.set()
        for future in futures:
            future.cancel()
        with entered_lock:
            started = [process for process in entered if process is not None]
            entered[:] = [None] * len(processes)
        for process in reversed(started):
            try:
                process.__exit__(None, None, None)
            except Exception:
                logger.exception("Failed to clean up a concurrently started process")
        for process, future in zip(processes, futures, strict=True):
            if future.done() or process in started:
                continue
            process.cancel_startup()
        # Wait until every startup task has either observed the terminated child
        # or cleaned up a late start. This prevents fixture/port teardown from
        # racing a worker that is still inside ManagedProcess.__enter__().
        executor.shutdown(wait=True, cancel_futures=True)

    try:
        for future in as_completed(futures):
            future.result()
    except Exception:
        cancel_startup()
        raise
    except BaseException:
        cancel_startup()
        raise
    else:
        executor.shutdown(wait=True)

    with ExitStack() as stack:
        for process in entered:
            if process is not None:
                stack.callback(process.__exit__, None, None, None)
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


def determine_request_receiving_worker(
    worker1: ManagedProcess,
    worker2: ManagedProcess,
    receiving_pattern: str,
    log_offsets: tuple[int, int] = (0, 0),
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

    def request_ids(worker: ManagedProcess, log_offset: int) -> list[str]:
        try:
            with open(worker.log_path, "rb") as log_file:
                log_file.seek(log_offset)
                return request_re.findall(log_file.read().decode(errors="ignore"))
        except FileNotFoundError:
            return []
        except OSError as error:
            logger.warning("Could not read log file %s: %s", worker.log_path, error)
            return []

    deadline = time.monotonic() + max_wait_s
    last_worker1_ids: list[str] = []
    last_worker2_ids: list[str] = []
    while time.monotonic() < deadline:
        last_worker1_ids = request_ids(worker1, log_offsets[0])
        last_worker2_ids = request_ids(worker2, log_offsets[1])

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


def read_worker_generate_metrics(
    worker_system_port: int,
    component: str = "backend",
) -> tuple[float, float]:
    """Read completed-request and response-byte totals for one worker endpoint."""
    response = requests.get(f"http://localhost:{worker_system_port}/metrics", timeout=1)
    response.raise_for_status()
    labels = {"dynamo_component": component, "dynamo_endpoint": "generate"}
    return (
        sum_metric_samples(
            response.text,
            "dynamo_component_request_duration_seconds_count",
            labels,
        ),
        sum_metric_samples(
            response.text,
            "dynamo_component_response_bytes_total",
            labels,
        ),
    )


def wait_for_worker_generate_completion(
    worker_system_port: int,
    baseline_duration_count: float,
    baseline_response_bytes: float,
    component: str = "backend",
    max_wait_time: float = 10.0,
) -> None:
    """Prove the replacement worker drained one new request with response bytes."""
    deadline = time.monotonic() + max_wait_time
    duration_count = 0.0
    response_bytes = 0.0
    last_error: Exception | None = None
    poll_event = threading.Event()

    while time.monotonic() < deadline:
        try:
            duration_count, response_bytes = read_worker_generate_metrics(
                worker_system_port,
                component,
            )
            if (
                duration_count - baseline_duration_count == 1
                and response_bytes - baseline_response_bytes > 0
            ):
                logger.info(
                    "Replacement worker completed one new request with %s response bytes",
                    response_bytes - baseline_response_bytes,
                )
                return
        except (requests.RequestException, ValueError) as error:
            last_error = error

        poll_event.wait(timeout=0.1)

    pytest.fail(
        "Replacement worker did not complete exactly one new generate request with "
        f"response bytes within {max_wait_time}s; "
        f"baseline_duration_count={baseline_duration_count}, "
        f"duration_count={duration_count}, "
        f"baseline_response_bytes={baseline_response_bytes}, "
        f"response_bytes={response_bytes}, last_error={last_error}"
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
    before_worker_fault: Callable[[], None] | None = None,
    force_max_output_tokens: bool = False,
    expected_output_prefix: str | None = None,
) -> tuple[ManagedProcess, str | None]:
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
        before_worker_fault: Optional state-based synchronization callback run
            immediately before fault injection. The request must remain active
            while the callback runs.
        force_max_output_tokens: Disable EOS and require the request's full
            max_tokens budget so state-based fault synchronization cannot race
            an early EOS.
        expected_output_prefix: Stable fault-free output prefix used to prove
            the content oracle extends beyond the fault boundary.

    Returns:
        The surviving replacement worker and the completed response output.
            Failed migration cases return None for the output.
    """
    # Ignore requests already present in the worker logs so the receiving-worker
    # lookup only considers the request started below.
    log_offsets = tuple(
        os.path.getsize(worker.log_path) if worker.log_path else 0
        for worker in (worker1, worker2)
    )

    # Step 1: Send the request
    request_thread, response = start_request(
        frontend.frontend_port,
        use_chat_completion=use_chat_completion,
        stream=stream,
        use_long_prompt=use_long_prompt,
        max_tokens=max_tokens,
        long_prompt_repetitions=long_prompt_repetitions,
        force_max_output_tokens=force_max_output_tokens,
    )

    # Step 2: Determine which worker received the request
    worker, worker_name, request_id = determine_request_receiving_worker(
        worker1,
        worker2,
        receiving_pattern=receiving_pattern,
        log_offsets=log_offsets,
    )
    replacement_worker = worker2 if worker is worker1 else worker1
    assert (
        request_thread.is_alive()
    ), "Request completed before the migration fault could be injected"

    replacement_generate_baseline: tuple[float, float] | None = None
    if verify_replacement_worker:
        worker_system_port = getattr(replacement_worker, "system_port", None)
        assert isinstance(
            worker_system_port, int
        ), "Replacement-worker verification requires an integer system_port"
        replacement_generate_baseline = read_worker_generate_metrics(worker_system_port)

    # Step 3: Optionally wait for new response before stop (for decode tests)
    if wait_for_new_response_before_stop:
        wait_for_response(response)
        assert (
            request_thread.is_alive()
        ), "Request completed before the worker fault was injected"
        if expected_output_prefix is not None:
            output_before_fault = "".join(
                content
                for content, _ in response.observations
                if isinstance(content, str)
            )
            assert expected_output_prefix.startswith(output_before_fault), (
                "Fault-free reference does not match the output emitted before "
                "the fault; the content oracle is not stable"
            )
            assert len(expected_output_prefix) >= len(output_before_fault) + 32, (
                "Fault-free stable prefix must cover at least 32 characters "
                "after the fault boundary"
            )

    if before_worker_fault is not None:
        before_worker_fault()
        assert (
            request_thread.is_alive()
        ), "Request completed while waiting to inject the worker fault"

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
            completed_output = validate_response(
                request_thread,
                response,
                expected_completion_tokens=(
                    max_tokens if force_max_output_tokens else None
                ),
            )
            if verify_replacement_worker:
                worker_system_port = getattr(replacement_worker, "system_port", None)
                assert isinstance(
                    worker_system_port, int
                ), "Replacement-worker verification requires an integer system_port"
                assert replacement_generate_baseline is not None
                wait_for_worker_generate_completion(
                    worker_system_port,
                    *replacement_generate_baseline,
                )
        else:
            # openai.APIError covers both mid-stream structured error frames and
            # HTTP non-200 responses.
            with pytest.raises(APIError):
                validate_response(request_thread, response)
            completed_output = None

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

    return replacement_worker, completed_output
