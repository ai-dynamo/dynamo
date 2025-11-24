# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil

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

        # Set port based on worker type
        port = "8082" if is_prefill else "8081"

        # Configure health check based on worker type
        if is_prefill:
            # Prefill workers check their own status endpoint
            command.append("--is-prefill-worker")
            health_check_urls = [(f"http://localhost:{port}/health", self.is_ready)]
        else:
            # Decode workers should also check their own status endpoint first,
            # then verify the frontend sees the model
            health_check_urls = [
                (f"http://localhost:{port}/health", self.is_ready),
                (f"http://localhost:{FRONTEND_PORT}/v1/models", check_models_api),
                (f"http://localhost:{FRONTEND_PORT}/health", check_health_generate),
            ]

        # Set debug logging environment
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = port

        # Set KV event port and NIXL side channel port only for prefill worker
        # to avoid conflicts with decode worker
        if is_prefill:
            env["DYN_VLLM_KV_EVENT_PORT"] = "20082"
            env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "5601"

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


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
def test_request_cancellation_vllm_aggregated(
    request, runtime_services, predownload_models
):
    """
    End-to-end test for request cancellation functionality in aggregated mode.

    This test verifies that when a request is cancelled by the client,
    the system properly handles the cancellation and cleans up resources
    on the worker side in aggregated (single worker) mode. Tests three scenarios:
    1. Completion request
    2. Chat completion request (non-streaming)
    3. Chat completion request (streaming)
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start a single worker
        with DynamoWorkerProcess(request) as worker:
            logger.info(f"Worker PID: {worker.get_pid()}")

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

                # Poll for "Decode Request ID" pattern (vLLM v2 pattern)
                request_id, worker_log_offset = poll_for_pattern(
                    process=worker,
                    pattern="Decode Request ID: ",
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


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
def test_request_cancellation_vllm_decode_cancel(
    request, runtime_services, predownload_models, set_ucx_tls_no_mm
):
    """
    End-to-end test for request cancellation during decode phase.

    This test verifies that when a request is cancelled by the client during the decode phase,
    the system properly handles the cancellation and cleans up resources
    on the decode worker side in a disaggregated setup.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the prefill worker
        with DynamoWorkerProcess(request, is_prefill=True) as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start the decode worker
            with DynamoWorkerProcess(request, is_prefill=False) as decode_worker:
                logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

                # Step 4: Test request cancellation for streaming scenario
                logger.info(
                    "Testing chat completion stream request cancellation in decode worker (decode phase)..."
                )

                # Send streaming request (non-blocking)
                cancellable_req = send_cancellable_request("chat_completion_stream")

                # Poll for "Decode Request ID" pattern in decode worker (vLLM v2 pattern)
                request_id, decode_log_offset = poll_for_pattern(
                    process=decode_worker,
                    pattern="Decode Request ID: ",
                    match_type="contains",
                )

                # Verify same request ID reached prefill worker (as "Prefill Request ID")
                _, prefill_log_offset = poll_for_pattern(
                    process=prefill_worker,
                    pattern=f"Prefill Request ID: {request_id}",
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


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
def test_request_cancellation_vllm_prefill_cancel(
    request, runtime_services, predownload_models, set_ucx_tls_no_mm
):
    """
    End-to-end test for request cancellation during prefill phase.

    This test verifies that when a request is cancelled by the client during the prefill phase,
    the system properly handles the cancellation and cleans up resources
    on both the decode and prefill workers in a disaggregated setup.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the prefill worker
        with DynamoWorkerProcess(request, is_prefill=True) as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start the decode worker
            with DynamoWorkerProcess(request, is_prefill=False) as decode_worker:
                logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

                # Step 4: Test request cancellation during prefill phase
                # Note: With the new architecture, prefill routing happens in the frontend,
                # so the request goes directly to the prefill worker first
                logger.info(
                    "Testing completion request cancellation during prefill phase..."
                )

                # Send request with long prompt (non-blocking)
                cancellable_req = send_cancellable_request(
                    "completion", use_long_prompt=True
                )

                # Poll for "Prefill Request ID" pattern in prefill worker (vLLM v2 pattern)
                # With new architecture, prefill is routed by frontend's internal router
                request_id, prefill_log_offset = poll_for_pattern(
                    process=prefill_worker,
                    pattern="Prefill Request ID: ",
                    match_type="contains",
                )

                # Cancel during prefill phase
                cancellable_req.cancel()
                logger.info(f"Cancelled request ID: {request_id} during prefill")

                # Poll for "Aborted Prefill Request ID" in prefill worker (where cancellation happens)
                _, prefill_log_offset = poll_for_pattern(
                    process=prefill_worker,
                    pattern=f"Aborted Prefill Request ID: {request_id}",
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
