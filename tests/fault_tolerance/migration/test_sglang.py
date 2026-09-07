# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import signal
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psutil
import pytest
import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME, DynamoPortRange
from tests.utils.gpu_args import build_gpu_mem_args
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import allocate_port, deallocate_ports
from tests.utils.prometheus import sum_metric_samples

# Customized utils for migration tests
from .utils import (
    DynamoFrontendProcess,
    managed_processes_concurrently,
    run_migration_test,
    wait_for_endpoint_instance_reduction,
    wait_for_endpoint_instances,
)

logger = logging.getLogger(__name__)

AGGREGATED_MAX_MODEL_LEN = 1024
AGGREGATED_MAX_TOKENS = 64
DECODE_MAX_MODEL_LEN = 1024
DECODE_MAX_TOKENS = 64
KV_TRANSFER_MAX_MODEL_LEN = 1024
KV_TRANSFER_MAX_TOKENS = 64
KV_TRANSFER_PROMPT_REPETITIONS = 128

SGLANG_MIGRATION_FRONTEND_STARTUP_TIMEOUT_S = 60
# Last-resort ceiling; individual operations have their own bounded waits.
SGLANG_MIGRATION_TEST_TIMEOUT_S = 780


@contextmanager
def _sglang_graceful_shutdown(
    frontend: DynamoFrontendProcess,
    worker: ManagedProcess,
) -> Iterator[None]:
    """Keep the failed worker alive until the request outcome is observable."""
    endpoint = ("backend", "generate")
    response = requests.get(
        f"http://localhost:{frontend.frontend_port}/health",
        timeout=1,
    )
    response.raise_for_status()
    previous_count = sum(
        1
        for instance in response.json().get("instances", [])
        if (instance.get("component"), instance.get("endpoint")) == endpoint
    )

    pid = worker.get_pid()
    parent = psutil.Process(pid)
    process_groups = {os.getpgid(pid)}
    try:
        for child in parent.children(recursive=True):
            try:
                process_groups.add(os.getpgid(child.pid))
            except (ProcessLookupError, OSError):
                pass
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass

    try:
        parent.terminate()
        wait_for_endpoint_instance_reduction(
            frontend.frontend_port,
            endpoint,
            previous_count,
        )
        yield
    finally:
        for process_group in process_groups:
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass


def get_sglang_kv_transfer_metrics(worker_system_port: int) -> tuple[float, float]:
    """Return the completed SGLang KV-transfer count and total size."""
    response = requests.get(f"http://localhost:{worker_system_port}/metrics", timeout=1)
    response.raise_for_status()
    return (
        sum_metric_samples(response.text, "sglang:kv_transfer_total_mb_count"),
        sum_metric_samples(response.text, "sglang:kv_transfer_total_mb_sum"),
    )


def wait_for_sglang_kv_transfer(
    worker_system_port: int,
    baseline_count: float,
    baseline_total_mb: float,
    max_wait_time: float = 10.0,
) -> tuple[float, float]:
    """Wait for a completed, non-empty SGLang KV transfer after the baseline."""
    deadline = time.monotonic() + max_wait_time
    transfer_count = baseline_count
    total_mb = baseline_total_mb
    last_error: Exception | None = None
    poll_event = threading.Event()

    while time.monotonic() < deadline:
        try:
            transfer_count, total_mb = get_sglang_kv_transfer_metrics(
                worker_system_port
            )
            if transfer_count > baseline_count and total_mb > baseline_total_mb:
                logger.info(
                    "Observed %s completed SGLang KV transfer(s) totaling %.3f MB",
                    transfer_count - baseline_count,
                    total_mb - baseline_total_mb,
                )
                return transfer_count, total_mb
        except (requests.RequestException, ValueError) as error:
            last_error = error

        poll_event.wait(timeout=0.01)

    pytest.fail(
        "SGLang did not report a completed, non-empty KV transfer after the "
        f"request started; count_delta={transfer_count - baseline_count}, "
        f"total_mb_delta={total_mb - baseline_total_mb}, last_error={last_error}"
    )


# Cover each distinct migration policy with complementary lifecycle, API,
# response, and transport values. Together these eight rows cover every pair
# of those four binary dimensions without paying for their Cartesian product.
MIGRATION_CASES = [
    pytest.param(
        3,
        None,
        True,
        "chat",
        True,
        "nats",
        id="migration_enabled-no_seq_cap-worker_failure-chat-stream-nats",
    ),
    pytest.param(
        3,
        None,
        False,
        "completion",
        False,
        "tcp",
        id="migration_enabled-no_seq_cap-graceful_shutdown-completion-unary-tcp",
    ),
    pytest.param(
        0,
        None,
        True,
        "chat",
        False,
        "tcp",
        id="migration_disabled-worker_failure-chat-unary-tcp",
    ),
    pytest.param(
        0,
        None,
        False,
        "completion",
        True,
        "nats",
        id="migration_disabled-graceful_shutdown-completion-stream-nats",
    ),
    pytest.param(
        3,
        1,
        True,
        "completion",
        True,
        "tcp",
        id="max_seq_len_exceeded-worker_failure-completion-stream-tcp",
    ),
    pytest.param(
        3,
        1,
        False,
        "chat",
        False,
        "nats",
        id="max_seq_len_exceeded-graceful_shutdown-chat-unary-nats",
    ),
    pytest.param(
        3,
        1_000_000,
        True,
        "completion",
        False,
        "nats",
        id="max_seq_len_not_exceeded-worker_failure-completion-unary-nats",
    ),
    pytest.param(
        3,
        1_000_000,
        False,
        "chat",
        True,
        "tcp",
        id="max_seq_len_not_exceeded-graceful_shutdown-chat-stream-tcp",
    ),
]

# Decode migration must be streaming so the fault can be injected after
# generation starts. Retain one case for every migration-policy outcome while
# balancing shutdown lifecycle, API, and request-plane transport.
DECODE_MIGRATION_CASES = [
    pytest.param(
        3,
        None,
        True,
        "chat",
        "nats",
        id="migration_enabled-worker_failure-chat-stream-nats",
    ),
    pytest.param(
        0,
        None,
        False,
        "completion",
        "nats",
        id="migration_disabled-graceful_shutdown-completion-stream-nats",
    ),
    pytest.param(
        3,
        1,
        True,
        "completion",
        "tcp",
        id="max_seq_len_exceeded-worker-failure-completion-stream-tcp",
    ),
    pytest.param(
        3,
        1_000_000,
        False,
        "chat",
        "tcp",
        id="max_seq_len_not_exceeded-graceful-shutdown-chat-stream-tcp",
    ),
]

# KV transfer is exercised only when migration is enabled and the request is
# under the sequence cap; aggregate migration owns the shared policy outcomes.
# The two rows retain both shutdown lifecycles and request-plane transports.
KV_TRANSFER_CASES = [
    pytest.param(
        3,
        None,
        True,
        "chat",
        True,
        "nats",
        id="worker-failure-chat-stream-nats",
    ),
    pytest.param(
        3,
        None,
        False,
        "chat",
        False,
        "tcp",
        id="graceful-shutdown-chat-unary-tcp",
    ),
]

MIGRATION_PARAMETERS = pytest.mark.parametrize(
    (
        "migration_limit",
        "migration_max_seq_len",
        "immediate_kill",
        "request_api",
        "stream",
        "request_plane",
    ),
    MIGRATION_CASES,
    indirect=["request_plane"],
)

DECODE_MIGRATION_PARAMETERS = pytest.mark.parametrize(
    (
        "migration_limit",
        "migration_max_seq_len",
        "immediate_kill",
        "request_api",
        "request_plane",
    ),
    DECODE_MIGRATION_CASES,
    indirect=["request_plane"],
)

KV_TRANSFER_MIGRATION_PARAMETERS = pytest.mark.parametrize(
    (
        "migration_limit",
        "migration_max_seq_len",
        "immediate_kill",
        "request_api",
        "stream",
        "request_plane",
    ),
    KV_TRANSFER_CASES,
    indirect=["request_plane"],
)

pytestmark = [
    pytest.mark.fault_tolerance,
    pytest.mark.sglang,
    pytest.mark.gpu_1,
    pytest.mark.e2e,
    pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME),
]


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with SGLang backend

    Supports both aggregated mode (single worker) and disaggregated mode
    (separate prefill and decode workers).

    Args:
        request: pytest request fixture
        worker_id: Unique identifier for the worker (e.g., "worker1", "worker2")
        frontend_port: Port where the frontend is running
        disagg_mode: None for aggregated, "prefill" or "decode" for disaggregated
        enable_metrics: Expose SGLang engine metrics through the worker system port
    """

    def __init__(
        self,
        request,
        worker_id: str,
        frontend_port: int,
        log_root: Path,
        disagg_mode: str | None = None,
        max_model_len: int = 8192,
        enable_metrics: bool = False,
    ):
        self.worker_id = worker_id
        allocated_ports: list[int] = []
        request.addfinalizer(lambda ports=allocated_ports: deallocate_ports(ports))

        self.system_port = allocate_port(DynamoPortRange.SERVE.value)
        allocated_ports.append(self.system_port)
        self.nccl_port = allocate_port(DynamoPortRange.NCCL.value)
        allocated_ports.append(self.nccl_port)
        self.bootstrap_port: int | None = None
        self.prefill_port: int | None = None
        self.disagg_mode = disagg_mode

        env = os.environ.copy()
        if "_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS" not in env:
            kv_mark = request.node.get_closest_marker("requested_sglang_kv_tokens")
            if kv_mark:
                env["_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS"] = str(
                    int(kv_mark.args[0])
                )

        gpu_mem_args = build_gpu_mem_args("build_sglang_gpu_mem_args", env=env)
        if not gpu_mem_args:
            gpu_mem_args = [
                "--max-total-tokens",
                "1024",
                "--mem-fraction-static",
                "0.9",
            ]

        command = [
            "python3",
            "-m",
            "dynamo.sglang",
            "--model-path",
            FAULT_TOLERANCE_MODEL_NAME,
            "--served-model-name",
            FAULT_TOLERANCE_MODEL_NAME,
            "--trust-remote-code",
            "--page-size",
            "16",
            "--tp",
            "1",
            "--nccl-port",
            str(self.nccl_port),
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
            "--max-running-requests",
            "1",
            *gpu_mem_args,
            "--context-length",
            str(max_model_len),
        ]
        if disagg_mode is None:
            # Aggregated
            command.append("--skip-tokenizer-init")
        else:
            # Disaggregated
            self.bootstrap_port = allocate_port(DynamoPortRange.BOOTSTRAP.value)
            allocated_ports.append(self.bootstrap_port)
            command.extend(
                [
                    "--disaggregation-mode",
                    disagg_mode,
                    "--disaggregation-bootstrap-port",
                    str(self.bootstrap_port),
                    "--host",
                    "0.0.0.0",
                    "--disaggregation-transfer-backend",
                    "nixl",
                ]
            )
            if disagg_mode == "prefill":
                self.prefill_port = allocate_port(DynamoPortRange.PREFILL.value)
                allocated_ports.append(self.prefill_port)
                command.extend(["--port", str(self.prefill_port)])

        if enable_metrics:
            command.append("--enable-metrics")

        # Set environment variables
        env["DYN_REQUEST_PLANE"] = request.getfixturevalue("request_plane")

        env["DYN_LOG"] = "debug"
        # Disable canary health check - these tests expect full control over requests
        # sent to the workers where canary health check intermittently sends dummy
        # requests to workers interfering with the test process which may cause
        # intermittent failures
        env["DYN_HEALTH_CHECK_ENABLED"] = "false"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = str(self.system_port)
        env["DYN_HTTP_PORT"] = str(frontend_port)

        # Disable backend shutdown grace period for all migration tests
        env["DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS"] = "0"

        # Configure health check based on worker type
        health_check_urls = [
            (f"http://localhost:{self.system_port}/health", self.is_ready)
        ]
        if disagg_mode is None or disagg_mode == "decode":
            health_check_urls.append(
                (f"http://localhost:{frontend_port}/v1/models", check_models_api)
            )

        log_dir = log_root / worker_id

        super().__init__(
            command=command,
            env=env,
            health_check_urls=health_check_urls,
            # Every worker retains a complete per-test log. Avoid interleaving
            # verbose engine output when several GPU tests run concurrently.
            display_output=False,
            terminate_all_matching_process_names=False,
            stragglers=["SGLANG:EngineCore"],
            straggler_commands=["-m dynamo.sglang"],
            log_dir=str(log_dir),
            display_name=worker_id,
        )

    def is_ready(self, response) -> bool:
        """Check the health of the worker process"""
        try:
            data = response.json()
            if data.get("status") == "ready":
                logger.info(f"{self.worker_id} status is ready")
                return True
            logger.warning(
                f"{self.worker_id} status is not ready: {data.get('status')}"
            )
        except ValueError:
            logger.warning(f"{self.worker_id} health response is not valid JSON")
        return False


@pytest.mark.timeout(SGLANG_MIGRATION_TEST_TIMEOUT_S)
@pytest.mark.nightly
@pytest.mark.profiled_vram_gib(5.4)  # measured NVML peak with two workers
@pytest.mark.requested_sglang_kv_tokens(1024)
@MIGRATION_PARAMETERS
def test_request_migration_sglang_aggregated(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    migration_limit,
    migration_max_seq_len,
    immediate_kill,
    request_api,
    stream,
    tmp_path,
):
    """
    End-to-end test for aggregated worker request migration.

    Parameters:
        immediate_kill: True for abrupt kill (SIGKILL), False for graceful shutdown (SIGTERM)
        migration_limit: > 0 to verify migration succeeds, 0 to verify request fails
        migration_max_seq_len: Max sequence length for migration state tracking
        request_api: "chat" for chat completion API, "completion" for completion API
        stream: True for streaming, False for non-streaming
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(
        request,
        migration_limit=migration_limit,
        migration_max_seq_len=migration_max_seq_len,
        startup_timeout_s=SGLANG_MIGRATION_FRONTEND_STARTUP_TIMEOUT_S,
    ) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 independent workers concurrently
        worker1 = DynamoWorkerProcess(
            request,
            "worker1",
            frontend.frontend_port,
            tmp_path,
            max_model_len=AGGREGATED_MAX_MODEL_LEN,
        )
        worker2 = DynamoWorkerProcess(
            request,
            "worker2",
            frontend.frontend_port,
            tmp_path,
            max_model_len=AGGREGATED_MAX_MODEL_LEN,
        )
        with managed_processes_concurrently(worker1, worker2):
            logger.info("Worker 1 PID: %s", worker1.get_pid())
            logger.info("Worker 2 PID: %s", worker2.get_pid())
            wait_for_endpoint_instances(
                frontend.frontend_port,
                {("backend", "generate"): 2},
            )

            # Step 3: Run migration test
            run_migration_test(
                frontend,
                worker1,
                worker2,
                receiving_pattern="New Request ID: ",
                migration_limit=migration_limit,
                migration_max_seq_len=migration_max_seq_len,
                immediate_kill=immediate_kill,
                use_chat_completion=(request_api == "chat"),
                stream=stream,
                max_tokens=AGGREGATED_MAX_TOKENS,
                expected_ongoing_request_count=1,
                graceful_shutdown=lambda worker: _sglang_graceful_shutdown(
                    frontend, worker
                ),
                verify_replacement_worker=True,
            )


@pytest.mark.timeout(SGLANG_MIGRATION_TEST_TIMEOUT_S)
@pytest.mark.nightly
@pytest.mark.profiled_vram_gib(7.8)  # measured NVML peak with three workers
@pytest.mark.requested_sglang_kv_tokens(1024)
@KV_TRANSFER_MIGRATION_PARAMETERS
def test_request_migration_sglang_kv_transfer(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    migration_limit,
    migration_max_seq_len,
    immediate_kill,
    request_api,
    stream,
    tmp_path,
):
    """
    End-to-end test for request migration with KV re-transfer in disaggregated mode.

    Setup: 1 prefill worker + 2 decode workers. The test waits for the initial
    KV transfer to complete, faults the selected decode worker, proves that the
    replacement handles the same request, and requires a second completed,
    non-empty KV transfer after the fault. Synchronization uses transfer state,
    not timing assertions.

    Parameters:
        immediate_kill: True for abrupt kill (SIGKILL), False for graceful shutdown (SIGTERM)
        migration_limit: > 0 to verify migration succeeds, 0 to verify request fails
        request_api: "chat" for chat completion API, "completion" for completion API
        stream: True for streaming, False for non-streaming
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(
        request,
        migration_limit=migration_limit,
        migration_max_seq_len=migration_max_seq_len,
        startup_timeout_s=SGLANG_MIGRATION_FRONTEND_STARTUP_TIMEOUT_S,
    ) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the independent prefill and decode workers concurrently.
        prefill_worker = DynamoWorkerProcess(
            request,
            "worker0",
            frontend.frontend_port,
            tmp_path,
            disagg_mode="prefill",
            max_model_len=KV_TRANSFER_MAX_MODEL_LEN,
            enable_metrics=True,
        )
        decode1 = DynamoWorkerProcess(
            request,
            "worker1",
            frontend.frontend_port,
            tmp_path,
            disagg_mode="decode",
            max_model_len=KV_TRANSFER_MAX_MODEL_LEN,
        )
        decode2 = DynamoWorkerProcess(
            request,
            "worker2",
            frontend.frontend_port,
            tmp_path,
            disagg_mode="decode",
            max_model_len=KV_TRANSFER_MAX_MODEL_LEN,
        )
        with managed_processes_concurrently(prefill_worker, decode1, decode2):
            logger.info("Prefill Worker PID: %s", prefill_worker.get_pid())
            logger.info("Decode Worker 1 PID: %s", decode1.get_pid())
            logger.info("Decode Worker 2 PID: %s", decode2.get_pid())
            wait_for_endpoint_instances(
                frontend.frontend_port,
                {("prefill", "generate"): 1, ("backend", "generate"): 2},
            )
            baseline_transfer_count, baseline_total_mb = get_sglang_kv_transfer_metrics(
                prefill_worker.system_port
            )
            post_initial_transfer: tuple[float, float] | None = None

            def wait_for_initial_transfer() -> None:
                nonlocal post_initial_transfer
                post_initial_transfer = wait_for_sglang_kv_transfer(
                    prefill_worker.system_port,
                    baseline_transfer_count,
                    baseline_total_mb,
                )

            # Step 3: Wait until the selected decode has received the initial
            # KV payload, then fault it and require the replacement request to
            # trigger a second transfer.
            run_migration_test(
                frontend,
                decode1,
                decode2,
                receiving_pattern="New Request ID: ",
                migration_limit=migration_limit,
                migration_max_seq_len=migration_max_seq_len,
                immediate_kill=immediate_kill,
                use_chat_completion=(request_api == "chat"),
                stream=stream,
                max_tokens=KV_TRANSFER_MAX_TOKENS,
                use_long_prompt=True,
                long_prompt_repetitions=KV_TRANSFER_PROMPT_REPETITIONS,
                expected_ongoing_request_count=1,
                graceful_shutdown=lambda worker: _sglang_graceful_shutdown(
                    frontend, worker
                ),
                verify_replacement_worker=True,
                before_worker_fault=wait_for_initial_transfer,
                force_max_output_tokens=True,
            )
            assert post_initial_transfer is not None
            wait_for_sglang_kv_transfer(
                prefill_worker.system_port,
                *post_initial_transfer,
            )


@pytest.mark.timeout(SGLANG_MIGRATION_TEST_TIMEOUT_S)
@pytest.mark.nightly
@pytest.mark.profiled_vram_gib(8.0)  # measured NVML peak with three workers
@pytest.mark.requested_sglang_kv_tokens(1024)
@DECODE_MIGRATION_PARAMETERS
def test_request_migration_sglang_decode(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    migration_limit,
    migration_max_seq_len,
    immediate_kill,
    request_api,
    tmp_path,
):
    """
    End-to-end test for decode worker request migration in disaggregated mode.

    Setup: 1 prefill worker + 2 decode workers.
    The request is streamed so the test can inject a fault after decode starts.

    Parameters:
        immediate_kill: True for abrupt kill (SIGKILL), False for graceful shutdown (SIGTERM)
        migration_limit: > 0 to verify migration succeeds, 0 to verify request fails
        request_api: "chat" for chat completion API, "completion" for completion API
    This target is always streaming; unary responses finish before a decode
    fault can be injected and are already covered by aggregate migration.
    """
    # Step 1: Start the frontend
    with DynamoFrontendProcess(
        request,
        migration_limit=migration_limit,
        migration_max_seq_len=migration_max_seq_len,
        startup_timeout_s=SGLANG_MIGRATION_FRONTEND_STARTUP_TIMEOUT_S,
    ) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the independent prefill and decode workers concurrently.
        prefill_worker = DynamoWorkerProcess(
            request,
            "worker0",
            frontend.frontend_port,
            tmp_path,
            disagg_mode="prefill",
            max_model_len=DECODE_MAX_MODEL_LEN,
        )
        decode1 = DynamoWorkerProcess(
            request,
            "worker1",
            frontend.frontend_port,
            tmp_path,
            disagg_mode="decode",
            max_model_len=DECODE_MAX_MODEL_LEN,
        )
        decode2 = DynamoWorkerProcess(
            request,
            "worker2",
            frontend.frontend_port,
            tmp_path,
            disagg_mode="decode",
            max_model_len=DECODE_MAX_MODEL_LEN,
        )
        with managed_processes_concurrently(prefill_worker, decode1, decode2):
            logger.info("Prefill Worker PID: %s", prefill_worker.get_pid())
            logger.info("Decode Worker 1 PID: %s", decode1.get_pid())
            logger.info("Decode Worker 2 PID: %s", decode2.get_pid())
            wait_for_endpoint_instances(
                frontend.frontend_port,
                {("prefill", "generate"): 1, ("backend", "generate"): 2},
            )

            # Step 3: Run migration test
            run_migration_test(
                frontend,
                decode1,
                decode2,
                receiving_pattern="New Request ID: ",
                migration_limit=migration_limit,
                migration_max_seq_len=migration_max_seq_len,
                immediate_kill=immediate_kill,
                use_chat_completion=(request_api == "chat"),
                stream=True,
                max_tokens=DECODE_MAX_TOKENS,
                wait_for_new_response_before_stop=True,
                expected_ongoing_request_count=1,
                graceful_shutdown=lambda worker: _sglang_graceful_shutdown(
                    frontend, worker
                ),
                verify_replacement_worker=True,
                force_max_output_tokens=True,
            )
