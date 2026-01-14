# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test Execution Times (Last Run: 2026-01-09):
- test_request_migration_vllm_aggregated: ~95s
- test_request_migration_vllm_prefill: N/A
- test_request_migration_vllm_decode: ~115s
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
    start_request,
    validate_response,
    verify_migration_metrics,
    verify_migration_occurred,
    wait_for_response,
)

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.e2e,
    pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME),
    pytest.mark.post_merge,  # post_merge to pinpoint failure commit
    pytest.mark.parametrize(
        "migration_limit", [3, 0], ids=["migration_enabled", "migration_disabled"]
    ),
    pytest.mark.parametrize(
        "immediate_kill", [True, False], ids=["worker_failure", "graceful_shutdown"]
    ),
    pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True),
]


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with vLLM backend

    Supports both aggregated mode (single worker) and disaggregated mode
    (separate prefill and decode workers).

    Args:
        request: pytest request fixture
        worker_id: Unique identifier for the worker (e.g., "worker1", "prefill1")
        frontend_port: Port where the frontend is running
        migration_limit: Maximum number of migration attempts (default: 3)
        is_prefill: None for aggregated mode, True for prefill worker, False for decode worker
    """

    def __init__(
        self,
        request,
        worker_id: str,
        frontend_port: int,
        migration_limit: int = 3,
        is_prefill: bool | None = None,
    ):
        self.worker_id = worker_id
        self.system_port = allocate_port(9100)

        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            FAULT_TOLERANCE_MODEL_NAME,
            "--enforce-eager",
            "--max-model-len",
            "8192",  # input + output tokens
            "--max-num-seqs",
            "1",  # number of requests at a time
            "--num-gpu-blocks-override",  # limit total KV cache allocation
            "512",  # 8192 tokens x 1 context / 16 tokens per block = 512 blocks
            "--gpu-memory-utilization",
            "0.15",  # avoid assertion error on vLLM available memory checks
            "--migration-limit",
            str(migration_limit),
        ]
        if is_prefill is True:
            command.append("--is-prefill-worker")
        elif is_prefill is False:
            command.append("--is-decode-worker")

        # Set environment variables
        env = os.environ.copy()
        env["DYN_REQUEST_PLANE"] = request.getfixturevalue("request_plane")

        # Set KV event and NIXL ports based on worker mode
        # All workers need unique NIXL side channel ports for KV transfer
        env[
            "VLLM_NIXL_SIDE_CHANNEL_PORT"
        ] = f"560{worker_id[-1]}"  # TODO: use dynamic port allocation

        if is_prefill is False:
            # Decode workers don't publish KV events
            env.pop("DYN_VLLM_KV_EVENT_PORT", None)
        else:
            # Aggregated mode and prefill workers publish KV events
            env[
                "DYN_VLLM_KV_EVENT_PORT"
            ] = f"2008{worker_id[-1]}"  # TODO: use dynamic port allocation

        env["DYN_LOG"] = "debug"
        # Disable canary health check - these tests expect full control over requests
        # sent to the workers where canary health check intermittently sends dummy
        # requests to workers interfering with the test process which may cause
        # intermittent failures
        env["DYN_HEALTH_CHECK_ENABLED"] = "false"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = str(self.system_port)
        env["DYN_HTTP_PORT"] = str(frontend_port)

        # Configure health check based on worker type
        health_check_urls = [
            (f"http://localhost:{self.system_port}/health", self.is_ready)
        ]
        if is_prefill is None or is_prefill is False:
            # aggregated or decode
            health_check_urls.append(
                (f"http://localhost:{frontend_port}/v1/models", check_models_api)
            )

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
            health_check_urls=health_check_urls,
            timeout=300,
            display_output=True,
            terminate_existing=False,
            stragglers=["VLLM::EngineCore"],
            straggler_commands=["-m dynamo.vllm"],
            log_dir=log_dir,
        )

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
                logger.info(f"{self.worker_id} status is ready")
                return True
            logger.warning(
                f"{self.worker_id} status is not ready: {data.get('status')}"
            )
        except ValueError:
            logger.warning(f"{self.worker_id} health response is not valid JSON")
        return False


@pytest.mark.timeout(290)  # 3x average
def test_request_migration_vllm_aggregated(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    migration_limit,
    immediate_kill,
):
    """
    End-to-end test for aggregated worker request migration.

    Parameters:
        immediate_kill: True for abrupt kill (SIGKILL), False for graceful shutdown (SIGTERM)
        migration_limit: > 0 to verify migration succeeds, 0 to verify request fails
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 workers
        with DynamoWorkerProcess(
            request, "worker1", frontend.frontend_port, migration_limit=migration_limit
        ) as worker1:
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")

            with DynamoWorkerProcess(
                request,
                "worker2",
                frontend.frontend_port,
                migration_limit=migration_limit,
            ) as worker2:
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                # Step 3: Send the request
                request_thread, response_list = start_request(frontend.frontend_port)

                # Step 4: Determine which worker received the request
                worker, worker_name = determine_request_receiving_worker(
                    worker1, worker2, receiving_pattern="Decode Request ID: "
                )

                # Step 5: Stop the worker (kill or graceful shutdown)
                if immediate_kill:
                    logger.info(f"Killing {worker_name} with PID {worker.get_pid()}")
                    terminate_process_tree(
                        worker.get_pid(), immediate_kill=True, timeout=0
                    )
                else:
                    logger.info(
                        f"Gracefully shutting down {worker_name} with PID {worker.get_pid()}"
                    )
                    terminate_process_tree(
                        worker.get_pid(), immediate_kill=False, timeout=10
                    )

                # Step 6: Validate response based on migration setting
                if migration_limit > 0:
                    validate_response(request_thread, response_list)
                    verify_migration_occurred(frontend)
                    verify_migration_metrics(
                        frontend.frontend_port, expected_ongoing_request_count=1
                    )
                else:
                    try:
                        validate_response(request_thread, response_list)
                        pytest.fail(
                            "Request succeeded unexpectedly when migration was disabled"
                        )
                    except AssertionError as e:
                        assert "SSE error event received: " in str(
                            e
                        ), f"Unexpected error: {e}"

                    try:
                        verify_migration_occurred(frontend)
                        pytest.fail("Migration unexpectedly occurred when disabled")
                    except AssertionError as e:
                        assert (
                            "'Cannot recreate stream: ...' error found in logs"
                            in str(e)
                        )


@pytest.mark.xfail(strict=False, reason="Prefill migration not yet supported")
@pytest.mark.timeout(350)  # 3x average
def test_request_migration_vllm_prefill(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    migration_limit,
    immediate_kill,
):
    """
    End-to-end test for prefill worker request migration in disaggregated mode.

    Setup: 1 decode worker + 2 prefill workers

    Parameters:
        immediate_kill: True for abrupt kill (SIGKILL), False for graceful shutdown (SIGTERM)
        migration_limit: > 0 to verify migration succeeds, 0 to verify request fails
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request, enforce_disagg=True) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start decode worker first (required for prefill workers to connect)
        with DynamoWorkerProcess(
            request,
            "worker0",
            frontend.frontend_port,
            migration_limit=migration_limit,
            is_prefill=False,
        ) as decode_worker:
            logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

            # Step 3: Start 2 prefill workers
            with DynamoWorkerProcess(
                request,
                "worker1",
                frontend.frontend_port,
                migration_limit=migration_limit,
                is_prefill=True,
            ) as prefill1:
                logger.info(f"Prefill Worker 1 PID: {prefill1.get_pid()}")

                with DynamoWorkerProcess(
                    request,
                    "worker2",
                    frontend.frontend_port,
                    migration_limit=migration_limit,
                    is_prefill=True,
                ) as prefill2:
                    logger.info(f"Prefill Worker 2 PID: {prefill2.get_pid()}")

                    # Step 4: Send the request (long prompt to extend prefill time)
                    request_thread, response_list = start_request(
                        frontend.frontend_port, use_long_prompt=True
                    )

                    # Step 5: Determine which prefill worker received the request
                    worker, worker_name = determine_request_receiving_worker(
                        prefill1, prefill2, receiving_pattern="Prefill Request ID: "
                    )

                    # Step 6: Stop the worker (kill or graceful shutdown)
                    if immediate_kill:
                        logger.info(
                            f"Killing {worker_name} with PID {worker.get_pid()}"
                        )
                        terminate_process_tree(
                            worker.get_pid(), immediate_kill=True, timeout=0
                        )
                    else:
                        logger.info(
                            f"Gracefully shutting down {worker_name} with PID {worker.get_pid()}"
                        )
                        terminate_process_tree(
                            worker.get_pid(), immediate_kill=False, timeout=10
                        )

                    # Step 7: Validate response based on migration setting
                    if migration_limit > 0:
                        validate_response(request_thread, response_list)
                        verify_migration_occurred(frontend)
                        verify_migration_metrics(
                            frontend.frontend_port, expected_ongoing_request_count=1
                        )
                    else:
                        try:
                            validate_response(request_thread, response_list)
                            pytest.fail(
                                "Request succeeded unexpectedly when migration was disabled"
                            )
                        except AssertionError as e:
                            assert "SSE error event received: " in str(
                                e
                            ), f"Unexpected error: {e}"

                        try:
                            verify_migration_occurred(frontend)
                            pytest.fail("Migration unexpectedly occurred when disabled")
                        except AssertionError as e:
                            assert (
                                "'Cannot recreate stream: ...' error found in logs"
                                in str(e)
                            )


@pytest.mark.timeout(350)  # 3x average
def test_request_migration_vllm_decode(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    migration_limit,
    immediate_kill,
):
    """
    End-to-end test for decode worker request migration in disaggregated mode.

    Setup: 1 prefill worker + 2 decode workers

    Parameters:
        immediate_kill: True for abrupt kill (SIGKILL), False for graceful shutdown (SIGTERM)
        migration_limit: > 0 to verify migration succeeds, 0 to verify request fails
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request, enforce_disagg=True) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start prefill worker first
        with DynamoWorkerProcess(
            request,
            "worker0",
            frontend.frontend_port,
            migration_limit=migration_limit,
            is_prefill=True,
        ) as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start 2 decode workers
            with DynamoWorkerProcess(
                request,
                "worker1",
                frontend.frontend_port,
                migration_limit=migration_limit,
                is_prefill=False,
            ) as decode1:
                logger.info(f"Decode Worker 1 PID: {decode1.get_pid()}")

                with DynamoWorkerProcess(
                    request,
                    "worker2",
                    frontend.frontend_port,
                    migration_limit=migration_limit,
                    is_prefill=False,
                ) as decode2:
                    logger.info(f"Decode Worker 2 PID: {decode2.get_pid()}")

                    # Step 4: Send the request
                    request_thread, response_list = start_request(
                        frontend.frontend_port
                    )

                    # Step 5: Determine which decode worker received the request
                    worker, worker_name = determine_request_receiving_worker(
                        decode1, decode2, receiving_pattern="Decode Request ID: "
                    )

                    # Wait for a new response making sure worker is actively decoding
                    wait_for_response(response_list)

                    # Step 6: Stop the worker (kill or graceful shutdown)
                    if immediate_kill:
                        logger.info(
                            f"Killing {worker_name} with PID {worker.get_pid()}"
                        )
                        terminate_process_tree(
                            worker.get_pid(), immediate_kill=True, timeout=0
                        )
                    else:
                        logger.info(
                            f"Gracefully shutting down {worker_name} with PID {worker.get_pid()}"
                        )
                        terminate_process_tree(
                            worker.get_pid(), immediate_kill=False, timeout=10
                        )

                    # Step 7: Validate response based on migration setting
                    if migration_limit > 0:
                        validate_response(request_thread, response_list)
                        verify_migration_occurred(frontend)
                        verify_migration_metrics(
                            frontend.frontend_port, expected_ongoing_request_count=1
                        )
                    else:
                        try:
                            validate_response(request_thread, response_list)
                            pytest.fail(
                                "Request succeeded unexpectedly when migration was disabled"
                            )
                        except AssertionError as e:
                            assert "SSE error event received: " in str(
                                e
                            ), f"Unexpected error: {e}"

                        try:
                            verify_migration_occurred(frontend)
                            pytest.fail("Migration unexpectedly occurred when disabled")
                        except AssertionError as e:
                            assert (
                                "'Cannot recreate stream: ...' error found in logs"
                                in str(e)
                            )
