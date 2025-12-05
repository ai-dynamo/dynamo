# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil

import pytest

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.engine_process import FRONTEND_PORT
from tests.utils.managed_process import ManagedProcess, terminate_process_tree
from tests.utils.payloads import check_models_api

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
    pytest.mark.trtllm,
    pytest.mark.gpu_1,
    pytest.mark.e2e,
    pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME),
    pytest.mark.pre_merge,  # can be moved to nightly once stable for a week
]


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with TRT-LLM backend"""

    def __init__(self, request, worker_id: str, migration_limit: int = 3):
        self.worker_id = worker_id

        command = [
            "python3",
            "-m",
            "dynamo.trtllm",
            "--model",
            FAULT_TOLERANCE_MODEL_NAME,
            "--disaggregation-mode",
            "prefill_and_decode",
            "--free-gpu-memory-fraction",
            "0.45",
            "--max-seq-len",
            "8192",
            "--migration-limit",
            str(migration_limit),
        ]

        # Set debug logging environment
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = f"808{worker_id[-1]}"

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
                (f"http://localhost:{FRONTEND_PORT}/v1/models", check_models_api),
                (f"http://localhost:808{worker_id[-1]}/health", self.is_ready),
            ],
            timeout=300,
            display_output=True,
            terminate_existing=False,
            log_dir=log_dir,
        )

    def get_pid(self):
        """Get the PID of the worker process"""
        return self.proc.pid if self.proc else None

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


@pytest.mark.timeout(290)  # 3x average
@pytest.mark.xfail(
    reason="For some reason both replicas received the request where only one should",
    strict=False,
)
def test_request_migration_trtllm_worker_failure(
    request, runtime_services, predownload_models, set_ucx_tls_no_mm
):
    """
    End-to-end test for worker fault tolerance with migration support using TRT-LLM.

    This test verifies that when a worker is killed during request processing,
    the system can handle the failure gracefully and migrate the request to
    another worker.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 workers sequentially
        with DynamoWorkerProcess(request, "worker1") as worker1:
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")

            with DynamoWorkerProcess(request, "worker2") as worker2:
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                # Step 3: Send the request
                request_thread, response_list = start_completion_request()

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


@pytest.mark.skip(reason="TRT-LLM graceful shutdown not yet implemented")
def test_request_migration_trtllm_graceful_shutdown(
    request, runtime_services, predownload_models, set_ucx_tls_no_mm
):
    """
    End-to-end test for worker fault tolerance with graceful shutdown and migration support using TRT-LLM.

    This test verifies that when a worker receives a graceful shutdown signal (SIGTERM)
    during request processing, the system can handle the shutdown gracefully and migrate
    the request to another worker. Unlike the abrupt kill test, this simulates a more
    controlled shutdown scenario where the worker has time to clean up and notify the
    system about its shutdown.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 workers sequentially
        with DynamoWorkerProcess(request, "worker1") as worker1:
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")

            with DynamoWorkerProcess(request, "worker2") as worker2:
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                # Step 3: Send the request
                request_thread, response_list = start_completion_request()

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


@pytest.mark.timeout(185)  # 3x average
@pytest.mark.xfail(
    reason="For some reason both replicas received the request where only one should",
    strict=False,
)
def test_no_request_migration_trtllm_worker_failure(
    request, runtime_services, predownload_models, set_ucx_tls_no_mm
):
    """
    End-to-end test for worker fault tolerance with migration disabled using TRT-LLM.

    This test verifies that when migration is disabled (migration_limit=0) and a worker
    is killed during request processing, the request fails as expected without migration.
    This is the opposite behavior of test_request_migration_trtllm_worker_failure.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 workers sequentially with migration disabled
        with DynamoWorkerProcess(request, "worker1", migration_limit=0) as worker1:
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")

            with DynamoWorkerProcess(request, "worker2", migration_limit=0) as worker2:
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                # Step 3: Send the request
                request_thread, response_list = start_completion_request()

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


@pytest.mark.skip(reason="TRT-LLM graceful shutdown not yet implemented")
def test_no_request_migration_trtllm_graceful_shutdown(
    request, runtime_services, predownload_models, set_ucx_tls_no_mm
):
    """
    End-to-end test for worker fault tolerance with graceful shutdown and migration disabled using TRT-LLM.

    This test verifies that when migration is disabled (migration_limit=0) and a worker
    receives a graceful shutdown signal (SIGTERM) during request processing, the request
    fails as expected without migration. This is the opposite behavior of
    test_request_migration_trtllm_graceful_shutdown.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 workers sequentially with migration disabled
        with DynamoWorkerProcess(request, "worker1", migration_limit=0) as worker1:
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")

            with DynamoWorkerProcess(request, "worker2", migration_limit=0) as worker2:
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                # Step 3: Send the request
                request_thread, response_list = start_completion_request()

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
