# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test Execution Times (Last Run: 2025-12-09):

Aggregated Mode Tests (gpu_1):
- test_request_migration_vllm_worker_failure: ~90s
- test_request_migration_vllm_graceful_shutdown: ~80s
- test_no_request_migration_vllm_worker_failure: ~75s
- test_no_request_migration_vllm_graceful_shutdown: ~75s

Disaggregated Mode Tests (gpu_2):
- test_request_migration_vllm_prefill_failure: ~120s
- test_request_migration_vllm_decode_failure: ~120s
- test_request_migration_vllm_prefill_graceful_shutdown: ~120s
- test_request_migration_vllm_decode_graceful_shutdown: ~120s
"""

import logging
import os
import shutil

import pytest

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import ManagedProcess, terminate_process_tree
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import allocate_port, deallocate_port

# Import utilities from the refactored utils module
from .utils import (
    DynamoFrontendProcess,
    determine_request_receiving_worker,
    start_completion_request,
    validate_completion_response,
    verify_migration_metrics,
    verify_migration_occurred,
)

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.e2e,
    pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME),
    pytest.mark.post_merge,  # post_merge to pinpoint failure commit
    pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True),
]


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with vLLM backend

    Supports both aggregated mode (single worker) and disaggregated mode
    (separate prefill and decode workers).

    Args:
        request: pytest request fixture
        worker_id: Unique identifier for the worker (e.g., "worker1", "prefill1")
        frontend_port: Port where the frontend is running
        migration_limit: Maximum number of migration attempts (default: 3)
        is_prefill: None for aggregated mode, True for prefill worker, False for decode worker
    """

    def __init__(
        self,
        request,
        worker_id: str,
        frontend_port: int,
        migration_limit: int = 3,
        is_prefill: bool | None = None,
    ):
        self.worker_id = worker_id
        self.frontend_port = frontend_port
        self.is_prefill = is_prefill

        # Allocate system port for this worker
        system_port = allocate_port(9100)
        self.system_port = system_port

        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            FAULT_TOLERANCE_MODEL_NAME,
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.3",  # Reduced to support up to 3 workers concurrently
            "--max-model-len",
            "8192",
            "--migration-limit",
            str(migration_limit),
        ]

        # Add worker role flags for disaggregated mode
        if is_prefill is True:
            command.append("--is-prefill-worker")
        elif is_prefill is False:
            command.append("--is-decode-worker")

        # Set environment variables
        env = os.environ.copy()
        env["DYN_REQUEST_PLANE"] = request.getfixturevalue("request_plane")

        # Set KV event and NIXL ports based on worker mode
        # All workers need unique NIXL side channel ports for KV transfer
        env[
            "VLLM_NIXL_SIDE_CHANNEL_PORT"
        ] = f"560{worker_id[-1]}"  # TODO: use dynamic port allocation

        if is_prefill is False:
            # Decode workers don't publish KV events
            env.pop("DYN_VLLM_KV_EVENT_PORT", None)
        else:
            # Aggregated mode and prefill workers publish KV events
            env[
                "DYN_VLLM_KV_EVENT_PORT"
            ] = f"2008{worker_id[-1]}"  # TODO: use dynamic port allocation

        env["DYN_LOG"] = "debug"
        # Disable canary health check - these tests expect full control over requests
        # sent to the workers where canary health check intermittently sends dummy
        # requests to workers interfering with the test process which may cause
        # intermittent failures
        env["DYN_HEALTH_CHECK_ENABLED"] = "false"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = str(system_port)
        env["DYN_HTTP_PORT"] = str(frontend_port)

        # Configure health check based on worker type
        if is_prefill is True:
            # Prefill workers only check their own status endpoint
            health_check_urls = [
                (f"http://localhost:{system_port}/health", self.is_ready)
            ]
        elif is_prefill is False:
            # Decode workers check their own status, then frontend
            health_check_urls = [
                (f"http://localhost:{system_port}/health", self.is_ready),
                (f"http://localhost:{frontend_port}/v1/models", check_models_api),
            ]
        else:
            # Aggregated mode: check frontend models API and worker health
            health_check_urls = [
                (f"http://localhost:{frontend_port}/v1/models", check_models_api),
                (f"http://localhost:{system_port}/health", self.is_ready),
            ]

        # TODO: Have the managed process take a command name explicitly to distinguish
        #       between processes started with the same command.
        log_dir = f"{request.node.name}_{worker_id}"

        # Clean up any existing log directory from previous runs
        try:
            shutil.rmtree(log_dir)
            logger.info(f"Cleaned up existing log directory: {log_dir}")
        except FileNotFoundError:
            # Directory doesn't exist, which is fine
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=health_check_urls,
            timeout=300,
            display_output=True,
            terminate_existing=False,
            stragglers=["VLLM::EngineCore"],
            straggler_commands=["-m dynamo.vllm"],
            log_dir=log_dir,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release allocated port when worker exits."""
        try:
            # system_port is always allocated in __init__
            deallocate_port(self.system_port)
        except Exception as e:
            logging.warning(f"Failed to release vLLM worker port: {e}")

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


# =============================================================================
# Aggregated Migration Tests
# =============================================================================


@pytest.mark.timeout(290)  # 3x average
def test_request_migration_vllm_worker_failure(
    request, runtime_services_dynamic_ports, set_ucx_tls_no_mm, predownload_models
):
    """
    End-to-end test for worker fault tolerance with migration support.

    This test verifies that when a worker is killed during request processing,
    the system can handle the failure gracefully and migrate the request to
    another worker.

    Timing (Last Run: 2025-12-09): ~90s total
    - Engine initialization: ~40s (Worker1: 20s, Worker2: 20s)
    - Test execution (request + migration): ~48s
    - Teardown: ~2s
    """

    # Step 1: Start the frontend (allocates its own frontend_port)
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 workers sequentially (each allocates its own system_port)
        with DynamoWorkerProcess(request, "worker1", frontend.frontend_port) as worker1:
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")

            with DynamoWorkerProcess(
                request, "worker2", frontend.frontend_port
            ) as worker2:
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                # Step 3: Send the request
                request_thread, response_list = start_completion_request(
                    frontend.frontend_port
                )

                # Step 4: Use polling to determine which worker received the request
                worker, worker_name = determine_request_receiving_worker(
                    worker1, worker2, receiving_pattern="Decode Request ID: "
                )

                # Step 5: Kill the worker that has the request
                logger.info(
                    f"Killing {worker_name} with PID {worker.get_pid()} processing the request"
                )
                terminate_process_tree(worker.get_pid(), immediate_kill=True, timeout=0)

                # Step 6: Validate the completion response
                validate_completion_response(request_thread, response_list)

                # Step 7: Verify migration occurred
                verify_migration_occurred(frontend)

                # Step 8: Verify migration metrics
                verify_migration_metrics(
                    frontend.frontend_port, expected_ongoing_request_count=1
                )


@pytest.mark.timeout(280)  # 3x average
def test_request_migration_vllm_graceful_shutdown(
    request, runtime_services_dynamic_ports, set_ucx_tls_no_mm, predownload_models
):
    """
    End-to-end test for worker fault tolerance with graceful shutdown and migration support.

    This test verifies that when a worker receives a graceful shutdown signal (SIGTERM)
    during request processing, the system can handle the shutdown gracefully and migrate
    the request to another worker. Unlike the abrupt kill test, this simulates a more
    controlled shutdown scenario where the worker has time to clean up and notify the
    system about its shutdown.

    Timing (Last Run: 2025-12-09): ~80s total
    - Engine initialization: ~40s (Worker1: 20s, Worker2: 20s)
    - Test execution (graceful shutdown + migration): ~38s
    - Teardown: ~2s
    """

    # Step 1: Start the frontend (allocates its own frontend_port)
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 workers sequentially (each allocates its own system_port)
        with DynamoWorkerProcess(request, "worker1", frontend.frontend_port) as worker1:
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")

            with DynamoWorkerProcess(
                request, "worker2", frontend.frontend_port
            ) as worker2:
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                # Step 3: Send the request
                request_thread, response_list = start_completion_request(
                    frontend.frontend_port
                )

                # Step 4: Use polling to determine which worker received the request
                worker, worker_name = determine_request_receiving_worker(
                    worker1, worker2, receiving_pattern="Decode Request ID: "
                )

                # Step 5: Gracefully shutdown the worker that has the request
                logger.info(
                    f"Gracefully shutting down {worker_name} with PID {worker.get_pid()} processing the request"
                )
                terminate_process_tree(
                    worker.get_pid(), immediate_kill=False, timeout=10
                )

                # Step 6: Validate the completion response
                validate_completion_response(request_thread, response_list)

                # Step 7: Verify migration occurred during graceful shutdown
                verify_migration_occurred(frontend)

                # Step 8: Verify migration metrics
                verify_migration_metrics(
                    frontend.frontend_port, expected_ongoing_request_count=1
                )


@pytest.mark.timeout(150)  # 3x average
def test_no_request_migration_vllm_worker_failure(
    request, runtime_services_dynamic_ports, set_ucx_tls_no_mm, predownload_models
):
    """
    End-to-end test for worker fault tolerance with migration disabled.

    This test verifies that when migration is disabled (migration_limit=0) and a worker
    is killed during request processing, the request fails as expected without migration.
    This is the opposite behavior of test_request_migration_vllm_worker_failure.

    Timing (Last Run: 2025-12-09): ~75s total
    - Engine initialization: ~40s (Worker1: 20s, Worker2: 20s)
    - Test execution (failure validation): ~33s
    - Teardown: ~2s
    """

    # Step 1: Start the frontend (allocates its own frontend_port)
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 workers sequentially with migration disabled (each allocates its own system_port)
        with DynamoWorkerProcess(
            request, "worker1", frontend.frontend_port, migration_limit=0
        ) as worker1:
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")

            with DynamoWorkerProcess(
                request, "worker2", frontend.frontend_port, migration_limit=0
            ) as worker2:
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                # Step 3: Send the request
                request_thread, response_list = start_completion_request(
                    frontend.frontend_port
                )

                # Step 4: Use polling to determine which worker received the request
                worker, worker_name = determine_request_receiving_worker(
                    worker1, worker2, receiving_pattern="Decode Request ID: "
                )

                # Step 5: Kill the worker that has the request
                logger.info(
                    f"Killing {worker_name} with PID {worker.get_pid()} processing the request"
                )
                terminate_process_tree(worker.get_pid(), immediate_kill=True, timeout=0)

                # Step 6: Validate the completion response - should fail without migration
                try:
                    validate_completion_response(request_thread, response_list)
                    pytest.fail(
                        "Request succeeded unexpectedly when migration was disabled"
                    )
                except AssertionError as e:
                    assert "Request failed with status 500: " in str(
                        e
                    ), f"Unexpected request error message: {e}"

                # Step 7: Verify migration did NOT occur - should fail
                try:
                    verify_migration_occurred(frontend)
                    pytest.fail(
                        "Migration verification unexpectedly passed when migration was disabled"
                    )
                except AssertionError as e:
                    assert "'Cannot recreate stream: ...' error found in logs" in str(
                        e
                    ), f"Unexpected migration message: {e}"


@pytest.mark.timeout(140)  # 3x average
def test_no_request_migration_vllm_graceful_shutdown(
    request, runtime_services_dynamic_ports, set_ucx_tls_no_mm, predownload_models
):
    """
    End-to-end test for worker fault tolerance with graceful shutdown and migration disabled.

    This test verifies that when migration is disabled (migration_limit=0) and a worker
    receives a graceful shutdown signal (SIGTERM) during request processing, the request
    fails as expected without migration. This is the opposite behavior of
    test_request_migration_vllm_graceful_shutdown.

    Timing (Last Run: 2025-12-09): ~75s total
    - Engine initialization: ~40s (Worker1: 20s, Worker2: 20s)
    - Test execution (graceful shutdown validation): ~33s
    - Teardown: ~2s
    """

    # Step 1: Start the frontend (allocates its own frontend_port)
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 workers sequentially with migration disabled (each allocates its own system_port)
        with DynamoWorkerProcess(
            request, "worker1", frontend.frontend_port, migration_limit=0
        ) as worker1:
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")

            with DynamoWorkerProcess(
                request, "worker2", frontend.frontend_port, migration_limit=0
            ) as worker2:
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                # Step 3: Send the request
                request_thread, response_list = start_completion_request(
                    frontend.frontend_port
                )

                # Step 4: Use polling to determine which worker received the request
                worker, worker_name = determine_request_receiving_worker(
                    worker1, worker2, receiving_pattern="Decode Request ID: "
                )

                # Step 5: Gracefully shutdown the worker that has the request
                logger.info(
                    f"Gracefully shutting down {worker_name} with PID {worker.get_pid()} processing the request"
                )
                terminate_process_tree(
                    worker.get_pid(), immediate_kill=False, timeout=10
                )

                # Step 6: Validate the completion response - should fail without migration
                try:
                    validate_completion_response(request_thread, response_list)
                    pytest.fail(
                        "Request succeeded unexpectedly when migration was disabled"
                    )
                except AssertionError as e:
                    assert "Request failed with status 500: " in str(
                        e
                    ), f"Unexpected request error message: {e}"

                # Step 7: Verify migration did NOT occur - should fail
                try:
                    verify_migration_occurred(frontend)
                    pytest.fail(
                        "Migration verification unexpectedly passed when migration was disabled"
                    )
                except AssertionError as e:
                    assert "'Cannot recreate stream: ...' error found in logs" in str(
                        e
                    ), f"Unexpected migration message: {e}"


# =============================================================================
# Disaggregated Migration Tests (Prefill/Decode Workers)
# =============================================================================


@pytest.mark.timeout(350)  # Higher timeout for 3 workers
def test_request_migration_vllm_prefill_failure(
    request, runtime_services_dynamic_ports, set_ucx_tls_no_mm, predownload_models
):
    """
    End-to-end test for prefill worker fault tolerance with migration support.

    This test verifies that when a prefill worker is killed during request processing
    in a disaggregated setup (1 decode + 2 prefill workers), the system can handle
    the failure gracefully and migrate the request to another prefill worker.

    Setup: 1 decode worker + 2 prefill workers
    """

    # Step 1: Start the frontend (allocates its own frontend_port)
    with DynamoFrontendProcess(request, enforce_disagg=True) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start decode worker first (required for prefill workers to connect)
        with DynamoWorkerProcess(
            request, "worker0", frontend.frontend_port, is_prefill=False
        ) as decode_worker:
            logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

            # Step 3: Start 2 prefill workers
            with DynamoWorkerProcess(
                request, "worker1", frontend.frontend_port, is_prefill=True
            ) as prefill1:
                logger.info(f"Prefill Worker 1 PID: {prefill1.get_pid()}")

                with DynamoWorkerProcess(
                    request, "worker2", frontend.frontend_port, is_prefill=True
                ) as prefill2:
                    logger.info(f"Prefill Worker 2 PID: {prefill2.get_pid()}")

                    # Step 4: Send the request
                    request_thread, response_list = start_completion_request(
                        frontend.frontend_port
                    )

                    # Step 5: Use polling to determine which prefill worker received the request
                    worker, worker_name = determine_request_receiving_worker(
                        prefill1, prefill2, receiving_pattern="Prefill Request ID: "
                    )

                    # Step 6: Kill the prefill worker that has the request
                    logger.info(
                        f"Killing {worker_name} with PID {worker.get_pid()} processing the request"
                    )
                    terminate_process_tree(
                        worker.get_pid(), immediate_kill=True, timeout=0
                    )

                    # Step 7: Validate the completion response
                    validate_completion_response(request_thread, response_list)

                    # Step 8: Verify migration occurred
                    verify_migration_occurred(frontend)

                    # Step 9: Verify migration metrics
                    verify_migration_metrics(
                        frontend.frontend_port, expected_ongoing_request_count=1
                    )


@pytest.mark.timeout(350)  # Higher timeout for 3 workers
def test_request_migration_vllm_decode_failure(
    request, runtime_services_dynamic_ports, set_ucx_tls_no_mm, predownload_models
):
    """
    End-to-end test for decode worker fault tolerance with migration support.

    This test verifies that when a decode worker is killed during request processing
    in a disaggregated setup (1 prefill + 2 decode workers), the system can handle
    the failure gracefully and migrate the request to another decode worker.

    Setup: 1 prefill worker + 2 decode workers
    """

    # Step 1: Start the frontend (allocates its own frontend_port)
    with DynamoFrontendProcess(request, enforce_disagg=True) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start prefill worker first
        with DynamoWorkerProcess(
            request, "worker0", frontend.frontend_port, is_prefill=True
        ) as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start 2 decode workers
            with DynamoWorkerProcess(
                request, "worker1", frontend.frontend_port, is_prefill=False
            ) as decode1:
                logger.info(f"Decode Worker 1 PID: {decode1.get_pid()}")

                with DynamoWorkerProcess(
                    request, "worker2", frontend.frontend_port, is_prefill=False
                ) as decode2:
                    logger.info(f"Decode Worker 2 PID: {decode2.get_pid()}")

                    # Step 4: Send the request
                    request_thread, response_list = start_completion_request(
                        frontend.frontend_port
                    )

                    # Step 5: Use polling to determine which decode worker received the request
                    worker, worker_name = determine_request_receiving_worker(
                        decode1, decode2, receiving_pattern="Decode Request ID: "
                    )

                    # Step 6: Kill the decode worker that has the request
                    logger.info(
                        f"Killing {worker_name} with PID {worker.get_pid()} processing the request"
                    )
                    terminate_process_tree(
                        worker.get_pid(), immediate_kill=True, timeout=0
                    )

                    # Step 7: Validate the completion response
                    validate_completion_response(request_thread, response_list)

                    # Step 8: Verify migration occurred
                    verify_migration_occurred(frontend)

                    # Step 9: Verify migration metrics
                    verify_migration_metrics(
                        frontend.frontend_port, expected_ongoing_request_count=1
                    )


@pytest.mark.timeout(350)  # Higher timeout for 3 workers
def test_request_migration_vllm_prefill_graceful_shutdown(
    request, runtime_services_dynamic_ports, set_ucx_tls_no_mm, predownload_models
):
    """
    End-to-end test for prefill worker graceful shutdown with migration support.

    This test verifies that when a prefill worker receives a graceful shutdown signal
    (SIGTERM) during request processing in a disaggregated setup (1 decode + 2 prefill
    workers), the system can handle the shutdown gracefully and migrate the request
    to another prefill worker.

    Setup: 1 decode worker + 2 prefill workers
    """

    # Step 1: Start the frontend (allocates its own frontend_port)
    with DynamoFrontendProcess(request, enforce_disagg=True) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start decode worker first (required for prefill workers to connect)
        with DynamoWorkerProcess(
            request, "worker0", frontend.frontend_port, is_prefill=False
        ) as decode_worker:
            logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

            # Step 3: Start 2 prefill workers
            with DynamoWorkerProcess(
                request, "worker1", frontend.frontend_port, is_prefill=True
            ) as prefill1:
                logger.info(f"Prefill Worker 1 PID: {prefill1.get_pid()}")

                with DynamoWorkerProcess(
                    request, "worker2", frontend.frontend_port, is_prefill=True
                ) as prefill2:
                    logger.info(f"Prefill Worker 2 PID: {prefill2.get_pid()}")

                    # Step 4: Send the request
                    request_thread, response_list = start_completion_request(
                        frontend.frontend_port
                    )

                    # Step 5: Use polling to determine which prefill worker received the request
                    worker, worker_name = determine_request_receiving_worker(
                        prefill1, prefill2, receiving_pattern="Prefill Request ID: "
                    )

                    # Step 6: Gracefully shutdown the prefill worker that has the request
                    logger.info(
                        f"Gracefully shutting down {worker_name} with PID {worker.get_pid()} processing the request"
                    )
                    terminate_process_tree(
                        worker.get_pid(), immediate_kill=False, timeout=10
                    )

                    # Step 7: Validate the completion response
                    validate_completion_response(request_thread, response_list)

                    # Step 8: Verify migration occurred during graceful shutdown
                    verify_migration_occurred(frontend)

                    # Step 9: Verify migration metrics
                    verify_migration_metrics(
                        frontend.frontend_port, expected_ongoing_request_count=1
                    )


@pytest.mark.timeout(350)  # Higher timeout for 3 workers
def test_request_migration_vllm_decode_graceful_shutdown(
    request, runtime_services_dynamic_ports, set_ucx_tls_no_mm, predownload_models
):
    """
    End-to-end test for decode worker graceful shutdown with migration support.

    This test verifies that when a decode worker receives a graceful shutdown signal
    (SIGTERM) during request processing in a disaggregated setup (1 prefill + 2 decode
    workers), the system can handle the shutdown gracefully and migrate the request
    to another decode worker.

    Setup: 1 prefill worker + 2 decode workers
    """

    # Step 1: Start the frontend (allocates its own frontend_port)
    with DynamoFrontendProcess(request, enforce_disagg=True) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start prefill worker first
        with DynamoWorkerProcess(
            request, "worker0", frontend.frontend_port, is_prefill=True
        ) as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start 2 decode workers
            with DynamoWorkerProcess(
                request, "worker1", frontend.frontend_port, is_prefill=False
            ) as decode1:
                logger.info(f"Decode Worker 1 PID: {decode1.get_pid()}")

                with DynamoWorkerProcess(
                    request, "worker2", frontend.frontend_port, is_prefill=False
                ) as decode2:
                    logger.info(f"Decode Worker 2 PID: {decode2.get_pid()}")

                    # Step 4: Send the request
                    request_thread, response_list = start_completion_request(
                        frontend.frontend_port
                    )

                    # Step 5: Use polling to determine which decode worker received the request
                    worker, worker_name = determine_request_receiving_worker(
                        decode1, decode2, receiving_pattern="Decode Request ID: "
                    )

                    # Step 6: Gracefully shutdown the decode worker that has the request
                    logger.info(
                        f"Gracefully shutting down {worker_name} with PID {worker.get_pid()} processing the request"
                    )
                    terminate_process_tree(
                        worker.get_pid(), immediate_kill=False, timeout=10
                    )

                    # Step 7: Validate the completion response
                    validate_completion_response(request_thread, response_list)

                    # Step 8: Verify migration occurred during graceful shutdown
                    verify_migration_occurred(frontend)

                    # Step 9: Verify migration metrics
                    verify_migration_metrics(
                        frontend.frontend_port, expected_ongoing_request_count=1
                    )
