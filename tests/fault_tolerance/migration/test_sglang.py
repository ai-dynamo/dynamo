# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test Execution Times (Last Run: 2026-01-13):
- test_request_migration_sglang_aggregated: ~75s
- test_request_migration_sglang_prefill: N/A
- test_request_migration_sglang_kv_transfer: N/A
- test_request_migration_sglang_decode: ~75s
"""

import logging
import os
import signal
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

pytestmark = [
    pytest.mark.fault_tolerance,
    pytest.mark.sglang,
    pytest.mark.gpu_1,
    pytest.mark.e2e,
    pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME),
]

# The remaining migration targets retain their existing Cartesian collection
# until their own stack layers classify and reduce them.
LEGACY_MIGRATION_PARAMETERS = (
    pytest.mark.parametrize(
        "migration_limit", [3, 0], ids=["migration_enabled", "migration_disabled"]
    ),
    pytest.mark.parametrize(
        "migration_max_seq_len",
        [
            pytest.param(None, id="max_seq_len_disabled"),
            pytest.param(1_000_000, id="max_seq_len_not_exceeded"),
            pytest.param(1, id="max_seq_len_exceeded"),
        ],
    ),
    pytest.mark.parametrize(
        "immediate_kill",
        [
            pytest.param(True, id="worker_failure"),
            pytest.param(False, id="graceful_shutdown"),
        ],
    ),
    pytest.mark.parametrize(
        "request_api",
        [
            pytest.param("chat"),
            pytest.param(
                "completion",
                marks=pytest.mark.skip(reason="Behavior unverified yet"),
            ),
        ],
    ),
    pytest.mark.parametrize(
        "stream",
        [
            pytest.param(True, id="stream"),
            pytest.param(
                False,
                id="unary",
                marks=pytest.mark.skip(reason="Behavior unverified yet"),
            ),
        ],
    ),
    pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True),
)


def legacy_migration_parameters(test):
    for marker in LEGACY_MIGRATION_PARAMETERS:
        test = marker(test)
    return test


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with SGLang backend

    Supports both aggregated mode (single worker) and disaggregated mode
    (separate prefill and decode workers).

    Args:
        request: pytest request fixture
        worker_id: Unique identifier for the worker (e.g., "worker1", "worker2")
        frontend_port: Port where the frontend is running
        disagg_mode: None for aggregated, "prefill" or "decode" for disaggregated
    """

    def __init__(
        self,
        request,
        worker_id: str,
        frontend_port: int,
        log_root: Path,
        disagg_mode: str | None = None,
        max_model_len: int = 8192,
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


@pytest.mark.skip(reason="KV cache transfer may fail")
@pytest.mark.timeout(230)  # 3x average
@pytest.mark.nightly
@legacy_migration_parameters
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
    End-to-end test for request migration during KV transfer in disaggregated mode.

    Setup: 1 prefill worker + 2 decode workers

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
    ) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start prefill worker first
        with DynamoWorkerProcess(
            request,
            "worker0",
            frontend.frontend_port,
            tmp_path,
            disagg_mode="prefill",
        ) as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start 2 decode workers
            with DynamoWorkerProcess(
                request,
                "worker1",
                frontend.frontend_port,
                tmp_path,
                disagg_mode="decode",
            ) as decode1:
                logger.info(f"Decode Worker 1 PID: {decode1.get_pid()}")

                with DynamoWorkerProcess(
                    request,
                    "worker2",
                    frontend.frontend_port,
                    tmp_path,
                    disagg_mode="decode",
                ) as decode2:
                    logger.info(f"Decode Worker 2 PID: {decode2.get_pid()}")

                    # Step 4: Run migration test
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
                        use_long_prompt=True,
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
