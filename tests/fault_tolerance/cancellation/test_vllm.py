# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import time

import pytest

from tests.fault_tolerance.cancellation.utils import (
    DynamoFrontendProcess,
    read_log_content,
    send_request_and_cancel,
    strip_ansi_codes,
)
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.engine_process import FRONTEND_PORT
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_health_generate, check_models_api

logger = logging.getLogger(__name__)


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with vLLM backend"""

    def __init__(self, request, is_prefill: bool = False):
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

        health_check_urls = [
            (f"http://localhost:{FRONTEND_PORT}/v1/models", check_models_api),
            (f"http://localhost:{FRONTEND_PORT}/health", check_health_generate),
        ]

        # Set port based on worker type
        port = "8082" if is_prefill else "8081"

        # Add prefill worker flag if needed
        if is_prefill:
            command.append("--is-prefill-worker")
            health_check_urls = [(f"http://localhost:{port}/health", self.is_ready)]

        # Set debug logging environment
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_ENABLED"] = "true"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = port

        # Set log directory based on worker type
        worker_type = "prefill_worker" if is_prefill else "worker"
        log_dir = f"{request.node.name}_{worker_type}"

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
            # Ensure any orphaned vLLM engine cores or child helpers are cleaned up
            stragglers=[
                "VLLM::EngineCore",
            ],
            straggler_commands=[
                "-m dynamo.vllm",
            ],
            log_dir=log_dir,
        )

        self.is_prefill = is_prefill

    def get_pid(self):
        """Get the PID of the worker process"""
        return self.proc.pid if self.proc else None

    def is_ready(self, response) -> bool:
        """Check the health of the worker process"""
        try:
            data = response.json()
            if data.get("status") == "ready":
                worker_type = "Prefill worker" if self.is_prefill else "Worker"
                logger.info(f"{worker_type} status is ready")
                return True
            worker_type = "Prefill worker" if self.is_prefill else "Worker"
            logger.warning(f"{worker_type} status is not ready: {data.get('status')}")
        except ValueError:
            worker_type = "Prefill worker" if self.is_prefill else "Worker"
            logger.warning(f"{worker_type} health response is not valid JSON")
        return False


def verify_request_cancelled(
    frontend_process: DynamoFrontendProcess,
    worker_process: DynamoWorkerProcess,
    prefill_worker_process: DynamoWorkerProcess | None = None,
    frontend_log_offset: int = 0,
    worker_log_offset: int = 0,
    assert_cancel_at_prefill: bool = False,
) -> tuple[int, int]:
    """Verify that the worker and frontend logs contain cancellation messages

    Returns:
        tuple: (new_worker_log_length, new_frontend_log_length)
    """

    # Check worker log for cancellation pattern
    worker_log_content = read_log_content(worker_process._log_path)
    new_worker_content = worker_log_content[worker_log_offset:]

    # Find the LAST occurrence of "New Request ID: <id>" line (health checks may log earlier ones)
    request_id = None
    for line in reversed(new_worker_content.split("\n")):
        # Strip ANSI codes and whitespace for pattern matching
        clean_line = strip_ansi_codes(line).strip()
        if "New Request ID: " in clean_line:
            # Extract ID from the last delimiter occurrence on the line
            parts = clean_line.rsplit("New Request ID: ", 1)
            if len(parts) > 1:
                request_id = parts[-1].strip()
                break
    if request_id is None:
        pytest.fail("Could not find 'New Request ID: <id>' pattern in worker log")

    # Check if the same request ID was cancelled
    has_worker_cancellation = False
    cancellation_pattern = (
        f"Aborted Remote Prefill Request ID: {request_id}"
        if assert_cancel_at_prefill
        else f"Aborted Request ID: {request_id}"
    )
    for line in new_worker_content.split("\n"):
        # Strip ANSI codes and whitespace for pattern matching
        clean_line = strip_ansi_codes(line).strip()
        if clean_line.endswith(cancellation_pattern):
            has_worker_cancellation = True
            break
    if not has_worker_cancellation:
        pytest.fail(f"Could not find '{cancellation_pattern}' pattern in worker log")

    # Check prefill worker log if provided
    if prefill_worker_process is not None:
        prefill_worker_log_content = read_log_content(prefill_worker_process._log_path)

        # Check if the same request ID was remote prefilled
        has_remote_prefill = False
        remote_prefill_pattern = f"New Prefill Request ID: {request_id}"
        for line in prefill_worker_log_content.split("\n"):
            clean_line = strip_ansi_codes(line).strip()
            if clean_line.endswith(remote_prefill_pattern):
                has_remote_prefill = True
                break
        if not has_remote_prefill:
            pytest.fail(
                f"Could not find '{remote_prefill_pattern}' pattern in prefill worker log"
            )

        # Check for remote prefill cancellation
        if assert_cancel_at_prefill:
            has_prefill_cancellation = False
            prefill_cancellation_pattern = f"Aborted Prefill Request ID: {request_id}"
            for line in prefill_worker_log_content.split("\n"):
                clean_line = strip_ansi_codes(line).strip()
                if clean_line.endswith(prefill_cancellation_pattern):
                    has_prefill_cancellation = True
                    break
            if not has_prefill_cancellation:
                pytest.fail(
                    f"Could not find '{prefill_cancellation_pattern}' pattern in prefill worker log"
                )

    # Check frontend log for cancellation issued pattern
    frontend_log_content = read_log_content(frontend_process._log_path)
    new_frontend_content = frontend_log_content[frontend_log_offset:]

    has_kill_message = False
    kill_message = "issued control message Kill to sender"
    for line in new_frontend_content.split("\n"):
        # Strip ANSI codes and whitespace for pattern matching
        clean_line = strip_ansi_codes(line).strip()
        if clean_line.endswith(kill_message):
            has_kill_message = True
            break
    if not has_kill_message:
        pytest.fail("Could not find cancellation issued in frontend log")

    return len(frontend_log_content), len(worker_log_content)


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
def test_request_cancellation_vllm(request, runtime_services, predownload_models):
    """
    End-to-end test for request cancellation functionality.

    This test verifies that when a request is cancelled by the client,
    the system properly handles the cancellation and cleans up resources
    on the worker side. Tests three scenarios:
    1. Completion request
    2. Chat completion request (non-streaming)
    3. Chat completion request (streaming)
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start a single worker
        logger.info("Starting worker...")
        worker = DynamoWorkerProcess(request)

        with worker:
            logger.info(f"Worker PID: {worker.get_pid()}")

            # Step 3: Test request cancellation
            frontend_log_offset, worker_log_offset = 0, 0

            test_scenarios = [
                ("completion", "Completion request cancellation"),
                ("chat_completion", "Chat completion request cancellation"),
                (
                    "chat_completion_stream",
                    "Chat completion stream request cancellation",
                ),
            ]

            for i, (request_type, description) in enumerate(test_scenarios, 1):
                logger.info(f"Testing {description.lower()}...")
                send_request_and_cancel(request_type)

                logger.info(
                    "Checking for cancellation messages in worker and frontend logs..."
                )
                time.sleep(0.05)  # time for cancellation to propagate
                frontend_log_offset, worker_log_offset = verify_request_cancelled(
                    frontend,
                    worker,
                    frontend_log_offset=frontend_log_offset,
                    worker_log_offset=worker_log_offset,
                )

                logger.info(f"{description} detected successfully")


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
def test_request_cancellation_vllm_decode(
    request, runtime_services, predownload_models
):
    """
    End-to-end test for request cancellation functionality with remote prefill.

    This test verifies that when a request is cancelled by the client,
    the system properly handles the cancellation and cleans up resources
    on the decode worker side in a disaggregated setup.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the prefill worker
        logger.info("Starting prefill worker...")
        prefill_worker = DynamoWorkerProcess(request, is_prefill=True)

        with prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start the decode worker
            logger.info("Starting decode worker...")
            decode_worker = DynamoWorkerProcess(request, is_prefill=False)

            with decode_worker:
                logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

                # Step 4: Test request cancellation for completion scenario only
                logger.info(
                    "Testing completion request cancellation in decode worker..."
                )
                send_request_and_cancel("completion")

                logger.info(
                    "Checking for cancellation messages in decode and prefill worker and frontend logs..."
                )
                time.sleep(0.05)  # time for cancellation to propagate
                verify_request_cancelled(frontend, decode_worker, prefill_worker)


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.xfail(
    reason="Time-sensitive test: Relies on request timeout (0.1s) to cancel during prefill phase. "
    "May fail if prefill completes too quickly or timeout triggers at a different phase.",
    strict=False,
)
def test_request_cancellation_vllm_prefill(
    request, runtime_services, predownload_models
):
    """
    End-to-end test for request cancellation on remote prefill.

    This test verifies that when a request is cancelled by the client during the
    prefill phase, the system properly handles the cancellation and cleans up
    resources on the prefill worker and decode worker sides in a disaggregated
    setup.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the prefill worker
        logger.info("Starting prefill worker...")
        prefill_worker = DynamoWorkerProcess(request, is_prefill=True)

        with prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start the decode worker
            logger.info("Starting decode worker...")
            decode_worker = DynamoWorkerProcess(request, is_prefill=False)

            with decode_worker:
                logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

                # Step 4: Test request cancellation for completion scenario only
                logger.info(
                    "Testing completion request cancellation in prefill worker..."
                )
                send_request_and_cancel("completion", timeout=0.1, use_long_prompt=True)

                logger.info(
                    "Checking for cancellation messages in decode and prefill worker and frontend logs..."
                )
                time.sleep(0.05)  # time for cancellation to propagate
                verify_request_cancelled(
                    frontend,
                    decode_worker,
                    prefill_worker,
                    assert_cancel_at_prefill=True,
                )
