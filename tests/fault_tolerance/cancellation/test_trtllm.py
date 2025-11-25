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


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with TensorRT-LLM backend"""

    def __init__(self, request, mode: str = "prefill_and_decode"):
        """
        Initialize TensorRT-LLM worker process.

        Args:
            request: pytest request object
            mode: One of "prefill_and_decode", "prefill", "decode"
        """
        # Prefill workers require migration_limit=0 (no KV cache migration support)
        migration_limit = "0" if mode == "prefill" else "3"

        command = [
            "python3",
            "-m",
            "dynamo.trtllm",
            "--model",
            FAULT_TOLERANCE_MODEL_NAME,
            "--disaggregation-mode",
            mode,
            "--free-gpu-memory-fraction",
            "0.45",
            "--max-seq-len",
            "16384",
            "--max-num-tokens",
            "16384",
            "--migration-limit",
            migration_limit,
        ]
        if mode != "prefill_and_decode":
            with open("test_request_cancellation_trtllm_config.yaml", "w") as f:
                f.write("cache_transceiver_config:\n  backend: DEFAULT\n")
                f.write("disable_overlap_scheduler: true\n")
            command += [
                "--extra-engine-args",
                "test_request_cancellation_trtllm_config.yaml",
            ]

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
        else:  # prefill_and_decode
            port = "8081"

        # Set debug logging environment
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = port

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


@pytest.mark.trtllm_marker
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
def test_request_cancellation_trtllm_aggregated(
    request, runtime_services, predownload_models
):
    """
    End-to-end test for request cancellation functionality in aggregated mode.

    This test verifies that when a request is cancelled by the client,
    the system properly handles the cancellation and cleans up resources
    on the worker side in aggregated (prefill_and_decode) mode.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start an aggregated worker
        with DynamoWorkerProcess(request, mode="prefill_and_decode") as worker:
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

                # Poll for "New Request ID" pattern
                request_id, worker_log_offset = poll_for_pattern(
                    process=worker,
                    pattern="New Request ID: ",
                    log_offset=worker_log_offset,
                    match_type="contains",
                )

                # For streaming, read 5 responses before cancelling
                if request_type == "chat_completion_stream":
                    read_streaming_responses(cancellable_req, expected_count=5)

                # Now cancel the request
                cancellable_req.cancel()
                logger.info(f"Cancelled request ID: {request_id}")

                # Poll for "Aborted Request ID" with matching ID
                _, worker_log_offset = poll_for_pattern(
                    process=worker,
                    pattern=f"Aborted Request ID: {request_id}",
                    log_offset=worker_log_offset,
                )

                # Verify frontend log has kill message
                _, frontend_log_offset = poll_for_pattern(
                    process=frontend,
                    pattern="issued control message Kill to sender",
                    log_offset=frontend_log_offset,
                )

                logger.info(f"{description} detected successfully")


@pytest.mark.trtllm_marker
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
def test_request_cancellation_trtllm_decode_cancel(
    request, runtime_services, predownload_models
):
    """
    End-to-end test for request cancellation during decode phase with unified frontend.

    This test verifies that when a request is cancelled by the client during the decode phase,
    the system properly handles the cancellation and cleans up resources
    on the decode worker side in a disaggregated setup.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the prefill worker
        with DynamoWorkerProcess(request, mode="prefill") as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start the decode worker
            with DynamoWorkerProcess(request, mode="decode") as decode_worker:
                logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

                # TODO: Why wait after worker ready fixes frontend 404 / 500 flakiness?
                time.sleep(2)

                # Step 4: Test request cancellation for streaming scenario
                logger.info(
                    "Testing chat completion stream request cancellation in decode worker (decode phase)..."
                )

                # Send streaming request (non-blocking)
                cancellable_req = send_cancellable_request("chat_completion_stream")

                # Poll for "Prefill Request ID" pattern in prefill worker (frontend routes here first)
                request_id, prefill_log_offset = poll_for_pattern(
                    process=prefill_worker,
                    pattern="Prefill Request ID: ",
                    match_type="contains",
                )

                # Verify same request ID reached decode worker (after prefill completes)
                _, decode_log_offset = poll_for_pattern(
                    process=decode_worker,
                    pattern=f"Decode Request ID: {request_id}",
                )

                # Read 5 streaming responses (decode phase)
                read_streaming_responses(cancellable_req, expected_count=5)

                # Now cancel the request
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


@pytest.mark.trtllm_marker
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
def test_request_cancellation_trtllm_prefill_cancel(
    request, runtime_services, predownload_models
):
    """
    End-to-end test for request cancellation during prefill phase with unified frontend.

    This test verifies that when a request is cancelled by the client during the prefill phase,
    the system properly handles the cancellation and cleans up resources on the prefill worker.
    Since the request is cancelled before prefill completes, the decode worker never receives it.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the prefill worker
        with DynamoWorkerProcess(request, mode="prefill") as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start the decode worker
            with DynamoWorkerProcess(request, mode="decode") as decode_worker:
                logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

                # TODO: Why wait after worker ready fixes frontend 404 / 500 flakiness?
                time.sleep(2)

                # Step 4: Test request cancellation during prefill phase
                logger.info(
                    "Testing completion request cancellation during prefill phase..."
                )

                # Send request with long prompt (non-blocking)
                cancellable_req = send_cancellable_request(
                    "completion", use_long_prompt=True
                )

                # Poll for "Prefill Request ID" pattern in prefill worker (frontend routes here first)
                request_id, prefill_log_offset = poll_for_pattern(
                    process=prefill_worker,
                    pattern="Prefill Request ID: ",
                    match_type="contains",
                )

                # Cancel during prefill phase
                cancellable_req.cancel()
                logger.info(f"Cancelled request ID: {request_id} during prefill")

                # Poll for "Aborted Request ID" in prefill worker (where cancellation happens)
                _, prefill_log_offset = poll_for_pattern(
                    process=prefill_worker,
                    pattern=f"Aborted Request ID: {request_id}",
                    log_offset=prefill_log_offset,
                )

                # Verify frontend log has kill message
                _, frontend_log_offset = poll_for_pattern(
                    process=frontend,
                    pattern="issued control message Kill to sender",
                )

                # Verify decode worker never received the request
                pattern = "Request ID: "
                try:
                    _, decode_log_offset = poll_for_pattern(
                        process=decode_worker,
                        pattern=pattern,
                        max_wait_ms=10,
                        match_type="contains",
                    )
                    pytest.fail(
                        "Decode worker received request cancelled during prefill phase"
                    )
                except AssertionError as e:
                    assert str(e).startswith(
                        f"Failed to find '{pattern}' pattern after 2 iterations "
                    ), f"Unexpected error: {e}"

                logger.info(
                    "Completion request cancellation during prefill phase detected successfully"
                )
