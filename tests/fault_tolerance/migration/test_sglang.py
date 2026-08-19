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

import pytest

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME, DynamoPortRange
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import allocate_port, deallocate_port

# Customized utils for migration tests
from .utils import DynamoFrontendProcess, run_migration_test

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.fault_tolerance,
    pytest.mark.sglang,
    pytest.mark.gpu_1,
    pytest.mark.e2e,
    pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME),
    pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True),
]


def migration_control_parameter():
    """Parametrize the migration controls exercised by the proven decode path."""
    migration_limits = ((3, "migration_enabled"), (0, "migration_disabled"))
    max_sequence_lengths = (
        (None, "max_seq_len_disabled"),
        (1_000_000, "max_seq_len_not_exceeded"),
        (1, "max_seq_len_exceeded"),
    )
    shutdown_modes = ((True, "worker_failure"), (False, "graceful_shutdown"))
    cases = tuple(
        pytest.param(
            migration_limit,
            migration_max_seq_len,
            immediate_kill,
            id=f"{limit_id}-{max_seq_id}-{shutdown_id}",
        )
        for migration_limit, limit_id in migration_limits
        for migration_max_seq_len, max_seq_id in max_sequence_lengths
        for immediate_kill, shutdown_id in shutdown_modes
    )
    return pytest.mark.parametrize(
        "migration_limit,migration_max_seq_len,immediate_kill", cases
    )


def representative_worker_failure_parameter():
    """Parametrize the smallest migration-enabled worker-failure scenario."""
    return pytest.mark.parametrize(
        "migration_limit,migration_max_seq_len,immediate_kill",
        [
            pytest.param(
                3,
                1_000_000,
                True,
                id="migration_enabled-max_seq_len_not_exceeded-worker_failure",
            )
        ],
    )


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
        disagg_mode: str | None = None,
    ):
        self.worker_id = worker_id
        self.system_port = allocate_port(DynamoPortRange.SERVE.value)
        request.addfinalizer(lambda port=self.system_port: deallocate_port(port))
        self.bootstrap_port: int | None = None
        self.prefill_port: int | None = None
        self.disagg_mode = disagg_mode

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
            "--mem-fraction-static",
            "0.3",
            "--context-length",
            "8192",
        ]
        if disagg_mode is None:
            # Aggregated
            command.append("--skip-tokenizer-init")
        else:
            # Disaggregated
            self.bootstrap_port = allocate_port(DynamoPortRange.BOOTSTRAP.value)
            request.addfinalizer(lambda port=self.bootstrap_port: deallocate_port(port))
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
                request.addfinalizer(
                    lambda port=self.prefill_port: deallocate_port(port)
                )
                command.extend(["--port", str(self.prefill_port)])

        # Set environment variables
        env = os.environ.copy()
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

        log_dir = request.getfixturevalue("tmp_path") / worker_id

        super().__init__(
            command=command,
            env=env,
            health_check_urls=health_check_urls,
            timeout=300,
            display_output=True,
            terminate_all_matching_process_names=False,
            stragglers=["SGLANG:EngineCore"],
            straggler_commands=["-m dynamo.sglang"],
            log_dir=str(log_dir),
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release allocated ports when worker exits."""
        for port in (self.system_port, self.bootstrap_port, self.prefill_port):
            if port is None:
                continue
            try:
                deallocate_port(port)
            except Exception as e:
                logging.warning(f"Failed to release SGLang worker port {port}: {e}")

        return super().__exit__(exc_type, exc_val, exc_tb)

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


@pytest.mark.timeout(230)  # 3x average
@pytest.mark.post_merge
@representative_worker_failure_parameter()
def test_request_migration_sglang_aggregated(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    migration_limit,
    migration_max_seq_len,
    immediate_kill,
):
    """
    End-to-end test for aggregated worker request migration.

    Parameters:
        immediate_kill: True for abrupt kill (SIGKILL), False for graceful shutdown (SIGTERM)
        migration_limit: > 0 to verify migration succeeds, 0 to verify request fails
        migration_max_seq_len: Max sequence length for migration state tracking
        This representative case uses the chat streaming API. Completion and unary
        migration behavior remains unverified and is not collected as supported coverage.
    """
    # Graceful shutdown remains excluded by OPS-4472. A disabled max-sequence
    # limit remains excluded by OPS-4446's first-token-delay failure.

    # Step 1: Start the frontend
    with DynamoFrontendProcess(
        request,
        migration_limit=migration_limit,
        migration_max_seq_len=migration_max_seq_len,
    ) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 workers
        with DynamoWorkerProcess(request, "worker1", frontend.frontend_port) as worker1:
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")

            with DynamoWorkerProcess(
                request,
                "worker2",
                frontend.frontend_port,
            ) as worker2:
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                # Step 3: Run migration test
                run_migration_test(
                    frontend,
                    worker1,
                    worker2,
                    receiving_pattern="New Request ID: ",
                    migration_limit=migration_limit,
                    migration_max_seq_len=migration_max_seq_len,
                    immediate_kill=immediate_kill,
                    use_chat_completion=True,
                    stream=True,
                )


@pytest.mark.skip(
    reason="SGLang prefill completes before migration can be triggered; DYN-4059"
)
@pytest.mark.timeout(230)  # 3x average
@pytest.mark.nightly
@representative_worker_failure_parameter()
def test_request_migration_sglang_prefill(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    migration_limit,
    migration_max_seq_len,
    immediate_kill,
):
    """
    End-to-end test for prefill worker request migration in disaggregated mode.

    Setup: 1 decode worker + 2 prefill workers

    Parameters:
        immediate_kill: True for abrupt kill (SIGKILL), False for graceful shutdown (SIGTERM)
        migration_limit: > 0 to verify migration succeeds, 0 to verify request fails
        The disabled representative uses the chat streaming API.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(
        request,
        migration_limit=migration_limit,
        migration_max_seq_len=migration_max_seq_len,
    ) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start decode worker first (required for prefill workers to connect)
        with DynamoWorkerProcess(
            request,
            "worker0",
            frontend.frontend_port,
            disagg_mode="decode",
        ) as decode_worker:
            logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

            # Step 3: Start 2 prefill workers
            with DynamoWorkerProcess(
                request,
                "worker1",
                frontend.frontend_port,
                disagg_mode="prefill",
            ) as prefill1:
                logger.info(f"Prefill Worker 1 PID: {prefill1.get_pid()}")

                with DynamoWorkerProcess(
                    request,
                    "worker2",
                    frontend.frontend_port,
                    disagg_mode="prefill",
                ) as prefill2:
                    logger.info(f"Prefill Worker 2 PID: {prefill2.get_pid()}")

                    # Step 4: Run migration test
                    run_migration_test(
                        frontend,
                        prefill1,
                        prefill2,
                        receiving_pattern="New Request ID: ",
                        migration_limit=migration_limit,
                        migration_max_seq_len=migration_max_seq_len,
                        immediate_kill=immediate_kill,
                        use_chat_completion=True,
                        stream=True,
                        use_long_prompt=True,
                    )


@pytest.mark.skip(
    reason="SGLang migration during KV transfer is not reliable; DYN-4059"
)
@pytest.mark.timeout(230)  # 3x average
@pytest.mark.nightly
@representative_worker_failure_parameter()
def test_request_migration_sglang_kv_transfer(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    migration_limit,
    migration_max_seq_len,
    immediate_kill,
):
    """
    End-to-end test for request migration during KV transfer in disaggregated mode.

    Setup: 1 prefill worker + 2 decode workers

    Parameters:
        immediate_kill: True for abrupt kill (SIGKILL), False for graceful shutdown (SIGTERM)
        migration_limit: > 0 to verify migration succeeds, 0 to verify request fails
        The disabled representative uses the chat streaming API.
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
            disagg_mode="prefill",
        ) as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start 2 decode workers
            with DynamoWorkerProcess(
                request,
                "worker1",
                frontend.frontend_port,
                disagg_mode="decode",
            ) as decode1:
                logger.info(f"Decode Worker 1 PID: {decode1.get_pid()}")

                with DynamoWorkerProcess(
                    request,
                    "worker2",
                    frontend.frontend_port,
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
                        use_chat_completion=True,
                        stream=True,
                        use_long_prompt=True,
                    )


@pytest.mark.timeout(230)  # 3x average
@pytest.mark.nightly
@migration_control_parameter()
def test_request_migration_sglang_decode(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    migration_limit,
    migration_max_seq_len,
    immediate_kill,
):
    """
    End-to-end test for decode worker request migration in disaggregated mode.

    Setup: 1 prefill worker + 2 decode workers

    Parameters:
        immediate_kill: True for abrupt kill (SIGKILL), False for graceful shutdown (SIGTERM)
        migration_limit: > 0 to verify migration succeeds, 0 to verify request fails
        The verified decode matrix uses the chat streaming API.
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
            disagg_mode="prefill",
        ) as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start 2 decode workers
            with DynamoWorkerProcess(
                request,
                "worker1",
                frontend.frontend_port,
                disagg_mode="decode",
            ) as decode1:
                logger.info(f"Decode Worker 1 PID: {decode1.get_pid()}")

                with DynamoWorkerProcess(
                    request,
                    "worker2",
                    frontend.frontend_port,
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
                        use_chat_completion=True,
                        stream=True,
                        wait_for_new_response_before_stop=True,
                    )
