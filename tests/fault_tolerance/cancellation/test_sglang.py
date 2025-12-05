# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import time

import pytest

from tests.fault_tolerance.cancellation.utils import (
    DynamoFrontendProcess,
    poll_for_pattern,
    read_streaming_responses,
    send_cancellable_request,
)
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.engine_process import FRONTEND_PORT
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_health_generate, check_models_api

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.sglang,
    pytest.mark.e2e,
    pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME),
    pytest.mark.post_merge,  # post_merge to pinpoint failure commit
]


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with SGLang backend"""

    def __init__(self, request, mode: str = "agg"):
        """
        Initialize SGLang worker process.

        Args:
            request: pytest request object
            mode: One of "agg", "prefill", "decode"
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
        if mode == "agg":
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
        else:  # agg (aggregated mode)
            port = "8081"

        # Set debug logging environment
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        # Disable canary health check - these tests expect full control over requests
        # sent to the workers where canary health check intermittently sends dummy
        # requests to workers interfering with the test process which may cause
        # intermittent failures
        env["DYN_HEALTH_CHECK_ENABLED"] = "false"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = port

        # Set GPU assignment for disaggregated mode (like disagg.sh)
        if mode == "decode":
            env["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 for decode worker
        elif mode == "prefill":
            env["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 for prefill worker
        # For agg (aggregated) mode, use default GPU assignment

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


@pytest.mark.timeout(160)  # 3x average
@pytest.mark.gpu_1
@pytest.mark.xfail(strict=False)
def test_request_cancellation_sglang_aggregated(
    request, runtime_services, predownload_models
):
    """
    End-to-end test for request cancellation functionality in aggregated mode.

    This test verifies that when a request is cancelled by the client,
    the system properly handles the cancellation and cleans up resources
    on the worker side in aggregated (agg) mode.

    TODO: Test is currently flaky/failing due to SGLang limitations with prefill cancellation.
    See: https://github.com/sgl-project/sglang/issues/11139
    """
    logger.info("Sanity check if latest test is getting executed")
    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start an aggregated worker
        with DynamoWorkerProcess(request, mode="agg") as worker:
            logger.info(f"Aggregated Worker PID: {worker.get_pid()}")
            # TODO: Why wait after worker ready fixes frontend 404 / 500 flakiness?
            time.sleep(2)

            # Step 3: Test request cancellation with polling approach
            frontend_log_offset, worker_log_offset = 0, 0

            test_scenarios = [
                ("completion", "Completion request cancellation"),
                ("chat_completion", "Chat completion request cancellation"),
                (
                    "chat_completion_stream",
                    "Chat completion stream request cancellation",
                ),
            ]

            for request_type, description in test_scenarios:
                logger.info(f"Testing {description.lower()}...")

                # Send the request (non-blocking)
                cancellable_req = send_cancellable_request(request_type)

                # Poll for "New Request ID" pattern (Dynamo context ID)
                request_id, worker_log_offset = poll_for_pattern(
                    process=worker,
                    pattern="New Request ID: ",
                    log_offset=worker_log_offset,
                    match_type="contains",
                )

                # For streaming, read one response first to trigger SGLang ID logging
                if request_type == "chat_completion_stream":
                    read_streaming_responses(cancellable_req, expected_count=1)

                # Wait for SGLang to actually start processing (get SGLang request ID)
                _, worker_log_offset = poll_for_pattern(
                    process=worker,
                    pattern="New SGLang Request ID: ",
                    log_offset=worker_log_offset,
                    match_type="contains",
                )

                # Now we know SGLang has the request, cancel it
                cancellable_req.cancel()
                logger.info(f"Cancelled request ID: {request_id}")

                # Poll for "Aborted Request ID" with matching ID
                _, worker_log_offset = poll_for_pattern(
                    process=worker,
                    pattern=f"Aborted Request ID: {request_id}",
                    log_offset=worker_log_offset,
                    max_wait_ms=2000,
                )

                # Verify frontend log has kill message
                _, frontend_log_offset = poll_for_pattern(
                    process=frontend,
                    pattern="issued control message Kill to sender",
                    log_offset=frontend_log_offset,
                )

                logger.info(f"{description} detected successfully")


@pytest.mark.timeout(185)  # 3x average
@pytest.mark.gpu_2
def test_request_cancellation_sglang_decode_cancel(
    request, runtime_services, predownload_models
):
    """
    End-to-end test for request cancellation during decode phase.

    This test verifies that when a request is cancelled by the client during the decode phase,
    the system properly handles the cancellation and cleans up resources
    on both the prefill and decode workers in a disaggregated setup.

    Note: This test requires 2 GPUs to run decode and prefill workers on separate GPUs.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the decode worker
        with DynamoWorkerProcess(request, mode="decode") as decode_worker:
            logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

            # Step 3: Start the prefill worker
            with DynamoWorkerProcess(request, mode="prefill") as prefill_worker:
                logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

                # TODO: Why wait after worker ready fixes frontend 404 / 500 flakiness?
                time.sleep(2)

                # Step 4: Test request cancellation during decode phase
                logger.info(
                    "Testing chat completion stream request cancellation during decode phase..."
                )

                # Send streaming request (non-blocking)
                cancellable_req = send_cancellable_request("chat_completion_stream")

                # Poll for "New Request ID" pattern in decode worker (Dynamo context ID)
                request_id, decode_log_offset = poll_for_pattern(
                    process=decode_worker,
                    pattern="New Request ID: ",
                    match_type="contains",
                )

                # Verify same request ID reached prefill worker
                _, prefill_log_offset = poll_for_pattern(
                    process=prefill_worker,
                    pattern=f"New Request ID: {request_id}",
                )

                # Read one response first to trigger SGLang ID logging in decode worker
                read_streaming_responses(cancellable_req, expected_count=1)

                # Wait for SGLang to start processing in decode worker
                _, decode_log_offset = poll_for_pattern(
                    process=decode_worker,
                    pattern="New SGLang Request ID: ",
                    log_offset=decode_log_offset,
                    match_type="contains",
                )

                # Now we know SGLang has the request in decode worker, cancel it
                cancellable_req.cancel()
                logger.info(f"Cancelled request ID: {request_id}")

                # Poll for "Aborted Request ID" in decode worker
                _, decode_log_offset = poll_for_pattern(
                    process=decode_worker,
                    pattern=f"Aborted Request ID: {request_id}",
                    log_offset=decode_log_offset,
                )

                # Verify frontend log has kill message
                _, frontend_log_offset = poll_for_pattern(
                    process=frontend,
                    pattern="issued control message Kill to sender",
                )

                logger.info(
                    "Chat completion stream cancellation in decode phase detected successfully"
                )
