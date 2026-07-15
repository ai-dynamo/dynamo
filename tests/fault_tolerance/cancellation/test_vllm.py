# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test Execution Times (Last Run: 2025-12-09):
- test_request_cancellation_vllm_aggregated: ~55s (gpu_1)
- test_request_cancellation_vllm_decode_cancel: ~53s (gpu_2)
- test_request_cancellation_vllm_prefill_cancel: ~53s (gpu_2)
- Total: 161.65s (0:02:41)
"""

import json
import logging
import os
import shutil

import pytest

from tests.fault_tolerance.cancellation.utils import (
    DynamoFrontendProcess,
    poll_for_pattern,
    read_streaming_responses,
    read_worker_generate_summary,
    send_cancellable_request,
    verify_frontend_cancellation_metrics,
    verify_runtime_cancellation_metrics,
)
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME, DynamoPortRange
from tests.utils.device import (
    build_nixl_kv_transfer_config_json,
    get_default_vllm_block_size,
)
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_health_generate, check_models_api
from tests.utils.port_utils import allocate_port, deallocate_port

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.fault_tolerance,
    pytest.mark.vllm,
    pytest.mark.e2e,
    pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME),
    pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True),
]


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with vLLM backend"""

    def __init__(
        self,
        request,
        frontend_port: int,
        is_prefill: bool | None = None,
        timeout_s: int = 300,
    ):
        # Allocate system port for this worker
        system_port = allocate_port(DynamoPortRange.SERVE.value)
        request.addfinalizer(lambda port=system_port: deallocate_port(port))
        self.system_port = system_port
        self.frontend_port = frontend_port

        # Determine max-model-len based on worker type:
        # Aggregated mode uses a smaller value (4096) to reduce GPU memory usage on XPU,
        # while disaggregated prefill/decode workers need 16384 for long-context KV transfer tests.
        max_model_len = "4096" if is_prefill is None else "16384"

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
            max_model_len,
            "--block-size",
            str(get_default_vllm_block_size()),
        ]

        # Configure disaggregation mode, KV transfer, and health checks per worker type
        if is_prefill is True:
            # Prefill worker: disaggregated prefill mode; check own status endpoint only
            command.extend(["--disaggregation-mode", "prefill"])
            command.extend(
                [
                    "--kv-transfer-config",
                    build_nixl_kv_transfer_config_json(),
                ]
            )
            health_check_urls = [
                (f"http://localhost:{system_port}/health", self.is_ready)
            ]
        elif is_prefill is False:
            # Decode worker: disaggregated decode mode; also verify frontend sees the model
            command.extend(["--disaggregation-mode", "decode"])
            command.extend(
                [
                    "--kv-transfer-config",
                    build_nixl_kv_transfer_config_json(),
                ]
            )
            health_check_urls = [
                (f"http://localhost:{system_port}/health", self.is_ready),
                (f"http://localhost:{frontend_port}/v1/models", check_models_api),
                (f"http://localhost:{frontend_port}/health", check_health_generate),
            ]
        else:
            # Aggregated worker: no disaggregation mode; verify frontend sees the model
            health_check_urls = [
                (f"http://localhost:{system_port}/health", self.is_ready),
                (f"http://localhost:{frontend_port}/v1/models", check_models_api),
                (f"http://localhost:{frontend_port}/health", check_health_generate),
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

        # Set KV events config and NIXL side channel port only for prefill worker
        # to avoid conflicts with decode worker
        if is_prefill is True:
            command.extend(
                [
                    "--kv-events-config",
                    json.dumps(
                        {
                            "publisher": "zmq",
                            "topic": "kv-events",
                            "endpoint": "tcp://*:20082",
                            "enable_kv_cache_events": True,
                        }
                    ),
                ]
            )
            env[
                "VLLM_NIXL_SIDE_CHANNEL_PORT"
            ] = "5601"  # TODO: use dynamic port allocation

        # Set log directory based on worker type
        if is_prefill is True:
            worker_type = "prefill_worker"
        elif is_prefill is False:
            worker_type = "decode_worker"
        else:
            worker_type = "worker"
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
            timeout=timeout_s,
            display_output=True,
            terminate_all_matching_process_names=False,
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
                worker_type = "Prefill worker" if self.is_prefill else "Worker"
                logger.info(f"{worker_type} status is ready")
                return True
            worker_type = "Prefill worker" if self.is_prefill else "Worker"
            logger.warning(f"{worker_type} status is not ready: {data.get('status')}")
        except ValueError:
            worker_type = "Prefill worker" if self.is_prefill else "Worker"
            logger.warning(f"{worker_type} health response is not valid JSON")
        return False


@pytest.mark.timeout(
    660
)  # worker startup can take up to 600s; allow headroom for test body
@pytest.mark.post_merge
@pytest.mark.gpu_1
@pytest.mark.xpu_1
def test_request_cancellation_vllm_aggregated(
    request, runtime_services_dynamic_ports, predownload_models
):
    """
    End-to-end test for request cancellation functionality in aggregated mode.

    This test verifies that when a request is cancelled by the client,
    the system properly handles the cancellation and cleans up resources
    on the worker side in aggregated (single worker) mode. Tests three scenarios:
    1. Completion request
    2. Chat completion request (non-streaming)
    3. Chat completion request (streaming)

    Timing (Last Run: 2025-12-09): ~55s total
    - Engine initialization: ~15s
    - Testing 3 scenarios: ~38s (~12s each)
    - Teardown: ~2s
    """

    def wait_for_stable_frontend(
        frontend_port: int, stable_seconds: int = 3, timeout_seconds: int = 60
    ):
        """Wait for frontend to reach stable state without errors."""
        import time

        import requests

        start_time = time.time()
        stable_start = None
        while time.time() - start_time < timeout_seconds:
            try:
                response = requests.get(
                    f"http://localhost:{frontend_port}/v1/models", timeout=2
                )
                if response.status_code == 200:
                    if stable_start is None:
                        stable_start = time.time()
                    elif time.time() - stable_start >= stable_seconds:
                        logger.info("Frontend is stable")
                        return
                else:
                    stable_start = None
            except Exception as e:
                logger.debug(f"Frontend health check failed: {e}")
                stable_start = None
            time.sleep(0.5)
        raise TimeoutError(f"Frontend did not stabilize within {timeout_seconds}s")

    # Step 1: Start the frontend (allocates its own frontend_port)
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start a single worker (allocates its own system_port)
        with DynamoWorkerProcess(
            request, frontend.frontend_port, timeout_s=600
        ) as worker:
            logger.info(f"Worker PID: {worker.get_pid()}")
            wait_for_stable_frontend(frontend.frontend_port)

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

            for idx, (request_type, description) in enumerate(test_scenarios):
                logger.info(f"Testing {description.lower()}...")

                # Send the request (non-blocking)
                # 2096 is an empirically chosen repro size that still exercises
                # cancellation while avoiding the XPU crash seen with 16384 tokens.
                cancellable_req = send_cancellable_request(
                    frontend.frontend_port,
                    request_type,
                    max_tokens=2096,
                )

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
                    pattern="issued control message control_msg=Kill",
                    log_offset=frontend_log_offset,
                )

                logger.info(f"{description} detected successfully")

                # Verify cancellation metrics after each scenario
                verify_frontend_cancellation_metrics(
                    frontend_port=frontend.frontend_port,
                    request_type=request_type,
                    expected_count=1,
                )
                verify_runtime_cancellation_metrics(
                    worker_system_port=worker.system_port,
                    expected_count=idx + 1,
                )


@pytest.mark.timeout(150)  # 3x average
@pytest.mark.nightly
@pytest.mark.gpu_2
def test_request_cancellation_vllm_decode_cancel(
    request, runtime_services_dynamic_ports, set_ucx_tls_no_mm, predownload_models
):
    """
    End-to-end test for request cancellation during decode phase.

    This test verifies that when a request is cancelled by the client during the decode phase,
    the system properly handles the cancellation and cleans up resources
    on the decode worker side in a disaggregated setup.

    Timing (Last Run: 2025-12-09): ~53s total (requires 2 GPUs)
    - Engine initialization: ~23s (decode + prefill workers)
    - Testing stream cancellation during decode: ~28s
    - Teardown: ~2s
    """

    # Step 1: Start the frontend (allocates its own frontend_port)
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the prefill worker (allocates its own system_port)
        with DynamoWorkerProcess(
            request, frontend.frontend_port, is_prefill=True
        ) as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start the decode worker (allocates its own system_port)
            with DynamoWorkerProcess(
                request, frontend.frontend_port, is_prefill=False
            ) as decode_worker:
                logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

                # Step 4: Test request cancellation for streaming scenario
                logger.info(
                    "Testing chat completion stream request cancellation in decode worker (decode phase)..."
                )

                # Send streaming request (non-blocking)
                cancellable_req = send_cancellable_request(
                    frontend.frontend_port, "chat_completion_stream"
                )

                # Poll for "Decode Request ID" pattern in decode worker (vLLM v2 pattern)
                request_id, decode_log_offset = poll_for_pattern(
                    process=decode_worker,
                    pattern="Decode Request ID: ",
                    match_type="contains",
                    max_wait_ms=10000,
                    poll_interval_ms=50,
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
                    pattern="issued control message control_msg=Kill",
                )

                logger.info(
                    "Chat completion stream cancellation in decode phase detected successfully"
                )

                # Verify cancellation metrics
                verify_frontend_cancellation_metrics(
                    frontend_port=frontend.frontend_port,
                    request_type="chat_completion_stream",
                    expected_count=1,
                )
                verify_runtime_cancellation_metrics(
                    worker_system_port=decode_worker.system_port,
                    expected_count=1,
                )
                verify_runtime_cancellation_metrics(
                    worker_system_port=prefill_worker.system_port,
                    expected_count=0,
                    component="prefill",
                )


@pytest.mark.timeout(660)  # 3x average (~219s)
@pytest.mark.nightly
@pytest.mark.gpu_2
def test_request_cancellation_vllm_prefill_cancel(
    request, runtime_services_dynamic_ports, set_ucx_tls_no_mm, predownload_models
):
    """
    End-to-end test for request cancellation during prefill phase.

    This test verifies that when a client disconnects during the prefill
    phase in a disaggregated setup, the prefill worker still runs the
    request to completion so KV blocks are released via the normal path
    (rather than leaking on a torn-down NIXL transfer), and decode routing
    still proceeds so the KV-transfer-complete guard can free the blocks.

    Reference: PR ai-dynamo/dynamo#7489

    Timing (Last Run: 2026-05-26): ~219s total (requires 2 GPUs)
    - Engine initialization: ~23s (decode + prefill workers)
    - Testing graceful disconnect during prefill: ~83s
    - Teardown: ~2s
    """

    # Step 1: Start the frontend (allocates its own frontend_port)
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the prefill worker (allocates its own system_port)
        with DynamoWorkerProcess(
            request, frontend.frontend_port, is_prefill=True
        ) as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start the decode worker (allocates its own system_port)
            with DynamoWorkerProcess(
                request, frontend.frontend_port, is_prefill=False
            ) as decode_worker:
                logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

                # Step 4: Test request cancellation during prefill phase
                # Note: With the new architecture, prefill routing happens in the frontend,
                # so the request goes directly to the prefill worker first
                logger.info(
                    "Testing completion request cancellation during prefill phase..."
                )

                # Send request with long prompt (non-blocking)
                cancellable_req = send_cancellable_request(
                    frontend.frontend_port, "completion", use_long_prompt=True
                )

                request_id, prefill_log_offset = poll_for_pattern(
                    process=prefill_worker,
                    pattern="Prefill Request ID: ",
                    match_type="contains",
                )

                # Cancel during prefill phase
                cancellable_req.cancel()
                logger.info(f"Cancelled request ID: {request_id} during prefill")

                # Prefill must complete despite client disconnect.
                poll_for_pattern(
                    process=prefill_worker,
                    pattern=f"Prefill completed for request {request_id}",
                    log_offset=prefill_log_offset,
                    match_type="contains",
                    max_wait_ms=15000,
                    poll_interval_ms=50,
                )

                poll_for_pattern(
                    process=frontend,
                    pattern="Connection closed unexpectedly",
                    match_type="contains",
                    max_wait_ms=2000,
                    poll_interval_ms=50,
                )

                # Wait for the runtime to log "request completed" for our request — this
                # fires on the same RequestMetricsGuard::drop that observes the histogram,
                # so once we see this log line the metric is already up to date.
                poll_for_pattern(
                    process=prefill_worker,
                    pattern=f"request completed request_id={request_id}",
                    log_offset=prefill_log_offset,
                    match_type="contains",
                    max_wait_ms=5000,
                    poll_interval_ms=100,
                )
                summary = read_worker_generate_summary(
                    worker_system_port=prefill_worker.system_port,
                    component="prefill",
                )
                logger.info(f"Prefill generate summary: {summary}")
                assert summary["duration_count"] == 1.0, (
                    f"Prefill histogram count={summary['duration_count']} — "
                    "request was aborted mid-flight."
                )
                assert summary["duration_sum"] >= 0.1, (
                    f"Prefill generate took only {summary['duration_sum']}s — "
                    "suspiciously short."
                )
                assert summary["response_bytes"] > 0, (
                    "Prefill sent 0 response bytes — handler exited before "
                    "yielding KV-transfer params."
                )

                # Verify cancellation metrics. The decode-side counter
                # increments in tcp/client.rs:347 only after the reader loop
                # exits (which lags the prefill drain by: decode dispatch +
                # frontend->decode ControlMessage::Kill + Python handler exit +
                # writer drain). Poll until it matches to avoid scraping before
                # the async chain finishes on slow runners.
                verify_frontend_cancellation_metrics(
                    frontend_port=frontend.frontend_port,
                    request_type="completion",
                    expected_count=1,
                )
                verify_runtime_cancellation_metrics(
                    worker_system_port=decode_worker.system_port,
                    expected_count=1,
                    max_wait_ms=15000,
                )
                verify_runtime_cancellation_metrics(
                    worker_system_port=prefill_worker.system_port,
                    expected_count=0,
                    component="prefill",
                )
