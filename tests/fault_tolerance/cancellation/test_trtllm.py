# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test Execution Times (Last Run: 2025-12-13):
- test_request_cancellation_trtllm_aggregated: ~45s (gpu_1)
- test_request_cancellation_trtllm_decode_cancel: ~65s (gpu_1)
- test_request_cancellation_trtllm_prefill_cancel: ~65s (gpu_1)
- test_request_cancellation_trtllm_kv_transfer_cancel: ~65s (gpu_1)
- Total: ~240s x2 request planes = ~480s (0:08:00)
"""

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
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_health_generate, check_models_api
from tests.utils.port_utils import allocate_port, deallocate_port

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.trtllm,
    pytest.mark.gpu_1,
    pytest.mark.e2e,
    pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME),
    pytest.mark.post_merge,  # post_merge to pinpoint failure commit
    pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True),
    pytest.mark.xfail(reason="Cancellation is temporarily disabled", strict=True),
]


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with TensorRT-LLM backend"""

    def __init__(
        self,
        request,
        frontend_port: int,
        mode: str = "prefill_and_decode",
    ):
        """
        Initialize TensorRT-LLM worker process.

        Args:
            request: pytest request object
            frontend_port: Port for the frontend server
            mode: One of "prefill_and_decode", "prefill", "decode"
        """
        # Allocate system port for this worker
        system_port = allocate_port(9100)
        self.system_port = system_port
        self.frontend_port = frontend_port

        command = [
            "python3",
            "-m",
            "dynamo.trtllm",
            "--model",
            FAULT_TOLERANCE_MODEL_NAME,
            "--disaggregation-mode",
            mode,
            "--max-seq-len",
            "16384",
            "--max-num-tokens",
            "16384",
        ]
        if mode != "prefill_and_decode":
            with open("test_request_cancellation_trtllm_config.yaml", "w") as f:
                f.write(
                    "cache_transceiver_config:\n  backend: DEFAULT\n  max_tokens_in_buffer: 16384\n"
                )
                f.write("disable_overlap_scheduler: true\n")
                f.write("kv_cache_config:\n  max_tokens: 16384\n")
            command += [
                "--extra-engine-args",
                "test_request_cancellation_trtllm_config.yaml",
            ]

        health_check_urls = [
            (f"http://localhost:{frontend_port}/v1/models", check_models_api),
            (f"http://localhost:{frontend_port}/health", check_health_generate),
        ]

        # Set health check based on worker type
        if mode in ["prefill", "decode"]:
            health_check_urls = [
                (f"http://localhost:{system_port}/health", self.is_ready)
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
            terminate_all_matching_process_names=False,
            log_dir=log_dir,
        )

        self.mode = mode

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

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release allocated port when worker exits."""
        try:
            # system_port is always allocated in __init__
            deallocate_port(self.system_port)
        except Exception as e:
            logging.warning(f"Failed to release TRT-LLM worker port: {e}")

        return super().__exit__(exc_type, exc_val, exc_tb)


@pytest.mark.timeout(135)  # 3x average
def test_request_cancellation_trtllm_aggregated(
    request, runtime_services_dynamic_ports, predownload_models
):
    """
    End-to-end test for request cancellation functionality in aggregated mode.

    This test verifies that when a request is cancelled by the client,
    the system properly handles the cancellation and cleans up resources
    on the worker side in aggregated (prefill_and_decode) mode. Tests three scenarios:
    1. Completion request
    2. Chat completion request (non-streaming)
    3. Chat completion request (streaming)

    Timing (Last Run: 2025-12-09): ~45s total
    - Engine initialization: ~27s (frontend + worker)
    - Testing 3 scenarios: ~15s (~5s each)
    - Teardown: ~3s
    """

    # Step 1: Start the frontend (allocates its own frontend_port)
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start an aggregated worker
        # Step 2: Start a single worker (allocates its own system_port)
        with DynamoWorkerProcess(
            request, frontend.frontend_port, mode="prefill_and_decode"
        ) as worker:
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
                cancellable_req = send_cancellable_request(
                    frontend.frontend_port, request_type
                )

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


@pytest.mark.timeout(195)  # 3x average
def test_request_cancellation_trtllm_decode_cancel(
    request, runtime_services_dynamic_ports, predownload_models
):
    """
    End-to-end test for request cancellation during decode phase with unified frontend.

    This test verifies that when a request is cancelled by the client during the decode phase,
    the system properly handles the cancellation and cleans up resources
    on the decode worker side in a disaggregated setup.

    Timing (Last Run: 2025-12-09): ~115s total (2 workers at 45% GPU each)
    - Engine initialization: ~92s (frontend: 2s, prefill worker: 45s, decode worker: 45s sequential)
    - Testing stream cancellation during decode: ~20s
    - Teardown: ~3s
    """

    # Step 1: Start the frontend (allocates its own frontend_port)
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the prefill worker (allocates its own system_port)
        with DynamoWorkerProcess(
            request, frontend.frontend_port, mode="prefill"
        ) as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start the decode worker (allocates its own system_port)
            with DynamoWorkerProcess(
                request, frontend.frontend_port, mode="decode"
            ) as decode_worker:
                logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

                # TODO: Why wait after worker ready fixes frontend 404 / 500 flakiness?
                time.sleep(2)

                # Step 4: Test request cancellation for streaming scenario
                logger.info(
                    "Testing chat completion stream request cancellation in decode worker (decode phase)..."
                )

                # Send streaming request (non-blocking)
                cancellable_req = send_cancellable_request(
                    frontend.frontend_port, "chat_completion_stream"
                )

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


@pytest.mark.timeout(195)  # 3x average
def test_request_cancellation_trtllm_prefill_cancel(
    request, runtime_services_dynamic_ports, predownload_models
):
    """
    End-to-end test for request cancellation during prefill phase with unified frontend.

    This test verifies that when a request is cancelled by the client during the prefill phase,
    the system properly handles the cancellation and cleans up resources on the prefill worker.
    Since the request is cancelled before prefill completes, the decode worker never receives it.

    Timing (Last Run: 2025-12-09): ~115s total (2 workers at 45% GPU each)
    - Engine initialization: ~92s (frontend: 2s, prefill worker: 45s, decode worker: 45s sequential)
    - Testing cancellation during prefill: ~20s
    - Teardown: ~3s
    """

    # Step 1: Start the frontend (allocates its own frontend_port)
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the prefill worker (allocates its own system_port)
        with DynamoWorkerProcess(
            request, frontend.frontend_port, mode="prefill"
        ) as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start the decode worker (allocates its own system_port)
            with DynamoWorkerProcess(
                request, frontend.frontend_port, mode="decode"
            ) as decode_worker:
                logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

                # TODO: Why wait after worker ready fixes frontend 404 / 500 flakiness?
                time.sleep(2)

                # Step 4: Test request cancellation during prefill phase
                logger.info(
                    "Testing completion request cancellation during prefill phase..."
                )

                # Send request with long prompt (non-blocking)
                cancellable_req = send_cancellable_request(
                    frontend.frontend_port, "completion", use_long_prompt=True
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


@pytest.mark.xfail(reason="Test fails only on CI", strict=False)
@pytest.mark.timeout(195)  # 3x average
def test_request_cancellation_trtllm_kv_transfer_cancel(
    request, runtime_services_dynamic_ports, predownload_models
):
    """
    End-to-end test for request cancellation during prefill to decode KV transfer phase.

    This test verifies that when a request is cancelled by the client during the KV transfer phase,
    the system properly handles the cancellation and cleans up resources on the workers.

    Timing (Last Run: 2025-12-09): ~115s total (2 workers at 45% GPU each)
    - Engine initialization: ~92s (frontend: 2s, prefill worker: 45s, decode worker: 45s sequential)
    - Testing KV transfer cancellation: ~20s
    - Teardown: ~3s
    """

    # Step 1: Start the frontend (allocates its own frontend_port)
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the prefill worker (allocates its own system_port)
        with DynamoWorkerProcess(
            request, frontend.frontend_port, mode="prefill"
        ) as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start the decode worker (allocates its own system_port)
            with DynamoWorkerProcess(
                request, frontend.frontend_port, mode="decode"
            ) as decode_worker:
                logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

                # TODO: Why wait after worker ready fixes frontend 404 / 500 flakiness?
                time.sleep(2)

                # Step 4: Test request cancellation during KV transfer phase
                logger.info(
                    "Testing completion request cancellation during KV transfer phase..."
                )

                # Send request with long prompt
                cancellable_req = send_cancellable_request(
                    frontend.frontend_port, "completion", use_long_prompt=True
                )

                # Poll for "Prefill Request ID" pattern in prefill worker
                request_id, prefill_log_offset = poll_for_pattern(
                    process=prefill_worker,
                    pattern="Prefill Request ID: ",
                    match_type="contains",
                )

                # Poll for decode worker entry signaling start of KV transfer phase
                _, decode_log_offset = poll_for_pattern(
                    process=decode_worker,
                    pattern=f"Decode Request ID: {request_id}",
                    poll_interval_ms=2,
                )

                # Cancel during KV transfer phase in decode worker
                cancellable_req.cancel()
                logger.info(
                    f"Cancelled request ID: {request_id} at beginning of decode"
                )

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
                    "Completion request cancellation at beginning of decode detected successfully"
                )

                # Verify the workers are still functional
                cancellable_req = send_cancellable_request(
                    frontend.frontend_port, "chat_completion_stream"
                )
                _, decode_log_offset = poll_for_pattern(
                    process=decode_worker,
                    pattern="Decode Request ID: ",
                    log_offset=decode_log_offset,
                    match_type="contains",
                )
                read_streaming_responses(cancellable_req, expected_count=5)

                logger.info(
                    "Workers are functional after cancellation during KV transfer"
                )
