# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test Execution Times (Last Run: 2025-12-09):
- test_request_migration_sglang_worker_failure: ~58s (gpu_1)
- test_request_migration_sglang_graceful_shutdown: ~58s (gpu_1, skipped)
- test_no_request_migration_sglang_worker_failure: ~38s (gpu_1)
- test_no_request_migration_sglang_graceful_shutdown: ~38s (gpu_1, skipped)
- Total: 115.71s (0:01:55) for enabled tests
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
    verify_migration_occurred,
)

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.sglang,
    pytest.mark.gpu_1,
    pytest.mark.e2e,
    pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME),
    pytest.mark.post_merge,  # post_merge to pinpoint failure commit
    pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True),
]


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with SGLang backend"""

    def __init__(
        self,
        request,
        worker_id: str,
        system_port: int,
        frontend_port: int,
        migration_limit: int = 3,
    ):
        self.worker_id = worker_id
        self.system_port = system_port

        command = [
            "python3",
            "-m",
            "dynamo.sglang",
            "--model-path",
            FAULT_TOLERANCE_MODEL_NAME,
            "--served-model-name",
            FAULT_TOLERANCE_MODEL_NAME,
            "--trust-remote-code",
            "--skip-tokenizer-init",
            "--mem-fraction-static",
            "0.45",
            "--context-length",
            "8192",
            "--migration-limit",
            str(migration_limit),
        ]

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
        env["DYN_SYSTEM_PORT"] = str(system_port)
        env["DYN_HTTP_PORT"] = str(frontend_port)

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
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", check_models_api),
                (f"http://localhost:{system_port}/health", self.is_ready),
            ],
            timeout=300,
            display_output=True,
            terminate_existing=False,
            stragglers=["SGLANG:EngineCore"],
            straggler_commands=["-m dynamo.sglang"],
            log_dir=log_dir,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release allocated port when worker exits."""
        try:
            # system_port is a required parameter, always set in __init__
            deallocate_port(self.system_port)
        except Exception as e:
            logging.warning(f"Failed to release SGLang worker port: {e}")

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


@pytest.mark.timeout(235)  # 3x average
def test_request_migration_sglang_worker_failure(
    request, runtime_services_dynamic_ports, set_ucx_tls_no_mm, predownload_models
):
    """
    End-to-end test for worker fault tolerance with migration support using SGLang.

    This test verifies that when a worker is killed during request processing,
    the system can handle the failure gracefully and migrate the request to
    another worker.

    Timing (Last Run: 2025-12-09): ~58s total
    - Engine initialization: ~22s (Worker1: 12s, Worker2: 10s)
    - Test execution (request + migration): ~21s
    - Teardown: ~15s
    """

    # Allocate ports to avoid conflicts with parallel tests
    worker1_system_port = allocate_port(9100)
    worker2_system_port = allocate_port(9200)

    # Step 1: Start the frontend (allocates its own port)
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 workers sequentially
        with DynamoWorkerProcess(
            request,
            "worker1",
            system_port=worker1_system_port,
            frontend_port=frontend.frontend_port,
        ) as worker1:
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")

            with DynamoWorkerProcess(
                request,
                "worker2",
                system_port=worker2_system_port,
                frontend_port=frontend.frontend_port,
            ) as worker2:
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                # Step 3: Send the request
                request_thread, response_list = start_completion_request(
                    frontend.frontend_port
                )

                # Step 4: Use polling to determine which worker received the request
                worker, worker_name = determine_request_receiving_worker(
                    worker1, worker2, receiving_pattern="New Request ID: "
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


@pytest.mark.timeout(235)  # 3x average
@pytest.mark.skip(reason="SGLang graceful shutdown not yet implemented")
def test_request_migration_sglang_graceful_shutdown(
    request, runtime_services_dynamic_ports, set_ucx_tls_no_mm, predownload_models
):
    """
    End-to-end test for worker fault tolerance with graceful shutdown and migration support using SGLang.

    This test verifies that when a worker receives a graceful shutdown signal (SIGTERM)
    during request processing, the system can handle the shutdown gracefully and migrate
    the request to another worker. Unlike the abrupt kill test, this simulates a more
    controlled shutdown scenario where the worker has time to clean up and notify the
    system about its shutdown.

    Timing (Last Run: 2025-12-09): ~58s total (estimated, similar to worker_failure)
    - Engine initialization: ~22s (Worker1: 12s, Worker2: 10s)
    - Test execution (request + graceful shutdown + migration): ~21s
    - Teardown: ~15s
    """

    # Allocate ports to avoid conflicts with parallel tests
    worker1_system_port = allocate_port(9100)
    worker2_system_port = allocate_port(9200)

    # Step 1: Start the frontend (allocates its own port)
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 workers sequentially
        with DynamoWorkerProcess(
            request,
            "worker1",
            system_port=worker1_system_port,
            frontend_port=frontend.frontend_port,
        ) as worker1:
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")

            with DynamoWorkerProcess(
                request,
                "worker2",
                system_port=worker2_system_port,
                frontend_port=frontend.frontend_port,
            ) as worker2:
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                # Step 3: Send the request
                request_thread, response_list = start_completion_request(
                    frontend.frontend_port
                )

                # Step 4: Use polling to determine which worker received the request
                worker, worker_name = determine_request_receiving_worker(
                    worker1, worker2, receiving_pattern="New Request ID: "
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


@pytest.mark.timeout(135)  # 3x average
def test_no_request_migration_sglang_worker_failure(
    request, runtime_services_dynamic_ports, set_ucx_tls_no_mm, predownload_models
):
    """
    End-to-end test for worker fault tolerance with migration disabled using SGLang.

    This test verifies that when migration is disabled (migration_limit=0) and a worker
    is killed during request processing, the request fails as expected without migration.
    This is the opposite behavior of test_request_migration_sglang_worker_failure.

    Timing (Last Run: 2025-12-09): ~38s total
    - Engine initialization: ~23s (Worker1: 13s, Worker2: 10s)
    - Test execution (failure validation): <1s
    - Teardown: ~15s
    """

    # Allocate ports to avoid conflicts with parallel tests
    worker1_system_port = allocate_port(9100)
    worker2_system_port = allocate_port(9200)

    # Step 1: Start the frontend (allocates its own port)
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 workers sequentially with migration disabled
        with DynamoWorkerProcess(
            request,
            "worker1",
            system_port=worker1_system_port,
            frontend_port=frontend.frontend_port,
            migration_limit=0,
        ) as worker1:
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")

            with DynamoWorkerProcess(
                request,
                "worker2",
                system_port=worker2_system_port,
                frontend_port=frontend.frontend_port,
                migration_limit=0,
            ) as worker2:
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                # Step 3: Send the request
                request_thread, response_list = start_completion_request(
                    frontend.frontend_port
                )

                # Step 4: Use polling to determine which worker received the request
                worker, worker_name = determine_request_receiving_worker(
                    worker1, worker2, receiving_pattern="New Request ID: "
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


@pytest.mark.timeout(135)  # 3x average
@pytest.mark.skip(reason="SGLang graceful shutdown not yet implemented")
def test_no_request_migration_sglang_graceful_shutdown(
    request, runtime_services_dynamic_ports, set_ucx_tls_no_mm, predownload_models
):
    """
    End-to-end test for worker fault tolerance with graceful shutdown and migration disabled using SGLang.

    This test verifies that when migration is disabled (migration_limit=0) and a worker
    receives a graceful shutdown signal (SIGTERM) during request processing, the request
    fails as expected without migration. This is the opposite behavior of
    test_request_migration_sglang_graceful_shutdown.

    Timing (Last Run: 2025-12-09): ~38s total (estimated, similar to no_migration_worker_failure)
    - Engine initialization: ~23s (Worker1: 13s, Worker2: 10s)
    - Test execution (graceful shutdown + failure validation): <1s
    - Teardown: ~15s
    """

    # Allocate ports to avoid conflicts with parallel tests
    worker1_system_port = allocate_port(9100)
    worker2_system_port = allocate_port(9200)

    # Step 1: Start the frontend (allocates its own port)
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 workers sequentially with migration disabled
        with DynamoWorkerProcess(
            request,
            "worker1",
            system_port=worker1_system_port,
            frontend_port=frontend.frontend_port,
            migration_limit=0,
        ) as worker1:
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")

            with DynamoWorkerProcess(
                request,
                "worker2",
                system_port=worker2_system_port,
                frontend_port=frontend.frontend_port,
                migration_limit=0,
            ) as worker2:
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                # Step 3: Send the request
                request_thread, response_list = start_completion_request(
                    frontend.frontend_port
                )

                # Step 4: Use polling to determine which worker received the request
                worker, worker_name = determine_request_receiving_worker(
                    worker1, worker2, receiving_pattern="New Request ID: "
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
