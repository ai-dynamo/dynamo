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


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with vLLM backend"""

    def __init__(self, request, worker_id: str):
        self.worker_id = worker_id

        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            FAULT_TOLERANCE_MODEL_NAME,
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.45",
            "--max-model-len",
            "8192",
            "--migration-limit",
            "3",
        ]

        # Set debug logging environment
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_ENABLED"] = "true"
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
            stragglers=["VLLM::EngineCore"],
            straggler_commands=["-m dynamo.vllm"],
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


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
def test_request_migration_vllm_worker_failure(
    request, runtime_services, predownload_models
):
    """
    End-to-end test for worker fault tolerance with migration support.

    This test verifies that when a worker is killed during request processing,
    the system can handle the failure gracefully and migrate the request to
    another worker.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 workers sequentially

        # Start worker1 first and wait for it to be ready
        logger.info("Starting worker 1...")
        with DynamoWorkerProcess(request, "worker1") as worker1:
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")

            with DynamoWorkerProcess(request, "worker2") as worker2:
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                # Step 3: Send the request
                request_thread, response_list = start_completion_request()

                # Step 4: Use polling to determine which worker received the request
                worker, worker_name = determine_request_receiving_worker(
                    worker1, worker2
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


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
def test_request_migration_vllm_graceful_shutdown(
    request, runtime_services, predownload_models
):
    """
    End-to-end test for worker fault tolerance with graceful shutdown and migration support.

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
                    worker1, worker2
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
