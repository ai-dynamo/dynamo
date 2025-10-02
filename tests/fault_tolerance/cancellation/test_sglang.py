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
    """Process manager for Dynamo worker with SGLang backend"""

    def __init__(self, request, mode: str = "null"):
        """
        Initialize SGLang worker process.

        Args:
            request: pytest request object
            mode: One of "null", "prefill", "decode"
        """
        command = [
            "python3",
            "-m",
            "dynamo.sglang",
            "--model-path",
            FAULT_TOLERANCE_MODEL_NAME,
            "--served-model-name",
            FAULT_TOLERANCE_MODEL_NAME,
            "--page-size",
            "16",
            "--tp",
            "1",
            "--trust-remote-code",
        ]

        # Add mode-specific arguments
        if mode == "null":
            # Aggregated mode - add skip-tokenizer-init like the serve test
            command.append("--skip-tokenizer-init")
        else:
            # Disaggregated mode - add disaggregation arguments like disagg.sh
            command.extend(
                [
                    "--disaggregation-mode",
                    mode,
                    "--disaggregation-bootstrap-port",
                    "12345",
                    "--host",
                    "0.0.0.0",
                    "--disaggregation-transfer-backend",
                    "nixl",
                ]
            )

        health_check_urls = [
            (f"http://localhost:{FRONTEND_PORT}/v1/models", check_models_api),
            (f"http://localhost:{FRONTEND_PORT}/health", check_health_generate),
        ]

        # Set port based on worker type
        if mode == "prefill":
            port = "8082"
            health_check_urls = [(f"http://localhost:{port}/health", self.is_ready)]
        elif mode == "decode":
            port = "8081"
            health_check_urls = [(f"http://localhost:{port}/health", self.is_ready)]
        else:  # null (aggregated mode)
            port = "8081"

        # Set debug logging environment
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_ENABLED"] = "true"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = port

        # Set GPU assignment for disaggregated mode (like disagg.sh)
        if mode == "decode":
            env["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 for decode worker
        elif mode == "prefill":
            env["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 for prefill worker
        # For null (aggregated) mode, use default GPU assignment

        # Set log directory based on worker type
        log_dir = f"{request.node.name}_{mode}_worker"

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
            # Ensure any orphaned SGLang engine cores or child helpers are cleaned up
            stragglers=[
                "SGLANG:EngineCore",
            ],
            straggler_commands=[
                "-m dynamo.sglang",
            ],
            log_dir=log_dir,
        )

        self.mode = mode

    def get_pid(self):
        """Get the PID of the worker process"""
        return self.proc.pid if self.proc else None

    def is_ready(self, response) -> bool:
        """Check the health of the worker process"""
        try:
            data = response.json()
            if data.get("status") == "ready":
                logger.info(f"{self.mode.capitalize()} worker status is ready")
                return True
            logger.warning(
                f"{self.mode.capitalize()} worker status is not ready: {data.get('status')}"
            )
        except ValueError:
            logger.warning(
                f"{self.mode.capitalize()} worker health response is not valid JSON"
            )
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
        f"Aborted Prefill Request ID: {request_id}"
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


@pytest.mark.e2e
@pytest.mark.sglang
@pytest.mark.gpu_1
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
def test_request_cancellation_sglang_aggregated(
    request, runtime_services, predownload_models
):
    """
    End-to-end test for request cancellation functionality in aggregated mode.

    This test verifies that when a request is cancelled by the client,
    the system properly handles the cancellation and cleans up resources
    on the worker side in aggregated (null) mode.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start an aggregated worker
        logger.info("Starting aggregated worker...")
        worker = DynamoWorkerProcess(request, mode="null")

        with worker:
            logger.info(f"Aggregated Worker PID: {worker.get_pid()}")

            # TODO: Why wait after worker ready fixes frontend 404 / 500 flakiness?
            time.sleep(2)

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
                time.sleep(1)  # time for cancellation to propagate
                frontend_log_offset, worker_log_offset, _ = verify_request_cancelled(
                    frontend,
                    worker,
                    frontend_log_offset=frontend_log_offset,
                    worker_log_offset=worker_log_offset,
                )

                logger.info(f"{description} detected successfully")


# skipping this test for now
@pytest.mark.skip(reason="Skipping this test for now until clarity from SGLANG")
@pytest.mark.e2e
@pytest.mark.sglang
@pytest.mark.gpu_1
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
def test_request_cancellation_sglang_remote_prefill_cancel(
    request, runtime_services, predownload_models
):
    """
    End-to-end test for request cancellation during prefill phase.

    This test verifies that when a request is cancelled by the client during the prefill phase,
    the system properly handles the cancellation and cleans up resources
    on the prefill worker side in a disaggregated setup.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the decode worker
        logger.info("Starting decode worker...")
        decode_worker = DynamoWorkerProcess(request, mode="decode")

        with decode_worker:
            logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

            # Step 3: Start the prefill worker
            logger.info("Starting prefill worker...")
            prefill_worker = DynamoWorkerProcess(request, mode="prefill")

            with prefill_worker:
                logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

                # TODO: Why wait after worker ready fixes frontend 404 / 500 flakiness?
                time.sleep(2)

                # Step 4: Test request cancellation during prefill phase
                logger.info(
                    "Testing completion request cancellation during prefill phase..."
                )
                send_request_and_cancel("completion", timeout=1, use_long_prompt=True)

                logger.info(
                    "Checking for cancellation messages in prefill and decode worker and frontend logs..."
                )
                time.sleep(1)  # time for cancellation to propagate
                verify_request_cancelled(
                    frontend,
                    decode_worker,
                    prefill_worker,
                    assert_cancel_at_prefill=True,
                )


@pytest.mark.e2e
@pytest.mark.sglang
@pytest.mark.gpu_1
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
def test_request_cancellation_sglang_decode_cancel(
    request, runtime_services, predownload_models
):
    """
    End-to-end test for request cancellation during remote decode phase.

    This test verifies that when a request is cancelled by the client during the remote decode phase,
    the system properly handles the cancellation and cleans up resources
    on both the prefill and decode workers in a disaggregated setup.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the decode worker
        logger.info("Starting decode worker...")
        decode_worker = DynamoWorkerProcess(request, mode="decode")

        with decode_worker:
            logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

            # Step 3: Start the prefill worker
            logger.info("Starting prefill worker...")
            prefill_worker = DynamoWorkerProcess(request, mode="prefill")

            with prefill_worker:
                logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

                # TODO: Why wait after worker ready fixes frontend 404 / 500 flakiness?
                time.sleep(2)

                # Step 4: Test request cancellation during remote decode phase
                logger.info(
                    "Testing completion request cancellation during remote decode phase..."
                )
                send_request_and_cancel("completion", timeout=2)

                logger.info(
                    "Checking for cancellation messages in prefill and decode worker and frontend logs..."
                )
                time.sleep(1)  # time for cancellation to propagate
                verify_request_cancelled(frontend, decode_worker, prefill_worker)
