# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test Execution Times (Last Run: 2026-01-09):
- test_request_migration_vllm_aggregated: ~95s
- test_request_migration_vllm_prefill: N/A
- test_request_migration_vllm_kv_transfer: N/A
- test_request_migration_vllm_decode: ~115s
"""

import logging
import os
import shutil

import pytest

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import allocate_port, deallocate_port

# Customized utils for migration tests
from .utils import DynamoFrontendProcess, run_migration_test

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
    pytest.mark.parametrize(
        "request_api",
        [
            pytest.param("chat"),
            pytest.param(
                "completion",
                marks=pytest.mark.skip(reason="Behavior unverified yet"),
            ),
        ],
    ),
    pytest.mark.parametrize(
        "stream",
        [
            pytest.param(True, id="stream"),
            pytest.param(
                False,
                id="unary",
                marks=pytest.mark.skip(reason="Behavior unverified yet"),
            ),
        ],
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
        is_prefill: None for aggregated mode, True for prefill worker, False for decode worker
    """

    def __init__(
        self,
        request,
        worker_id: str,
        frontend_port: int,
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
            terminate_all_matching_process_names=False,
            stragglers=["VLLM::EngineCore"],
            straggler_commands=["-m dynamo.vllm"],
            log_dir=log_dir,
            display_name=worker_id,
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
    request_api,
    stream,
):
    """
    End-to-end test for aggregated worker request migration.

    Parameters:
        immediate_kill: True for abrupt kill (SIGKILL), False for graceful shutdown (SIGTERM)
        migration_limit: > 0 to verify migration succeeds, 0 to verify request fails
        request_api: "chat" for chat completion API, "completion" for completion API
        stream: True for streaming, False for non-streaming
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request, migration_limit=migration_limit) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 workers
        with DynamoWorkerProcess(request, "worker1", frontend.frontend_port) as worker1:
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")

            with DynamoWorkerProcess(
                request,
                "worker2",
                frontend.frontend_port,
            ) as worker2:
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                # Step 3: Run migration test
                run_migration_test(
                    frontend,
                    worker1,
                    worker2,
                    receiving_pattern="Decode Request ID: ",
                    migration_limit=migration_limit,
                    immediate_kill=immediate_kill,
                    use_chat_completion=(request_api == "chat"),
                    stream=stream,
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
    request_api,
    stream,
):
    """
    End-to-end test for prefill worker request migration in disaggregated mode.

    Setup: 1 decode worker + 2 prefill workers

    Parameters:
        immediate_kill: True for abrupt kill (SIGKILL), False for graceful shutdown (SIGTERM)
        migration_limit: > 0 to verify migration succeeds, 0 to verify request fails
        request_api: "chat" for chat completion API, "completion" for completion API
        stream: True for streaming, False for non-streaming
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(
        request, migration_limit=migration_limit, enforce_disagg=True
    ) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start decode worker first (required for prefill workers to connect)
        with DynamoWorkerProcess(
            request,
            "worker0",
            frontend.frontend_port,
            is_prefill=False,
        ) as decode_worker:
            logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

            # Step 3: Start 2 prefill workers
            with DynamoWorkerProcess(
                request,
                "worker1",
                frontend.frontend_port,
                is_prefill=True,
            ) as prefill1:
                logger.info(f"Prefill Worker 1 PID: {prefill1.get_pid()}")

                with DynamoWorkerProcess(
                    request,
                    "worker2",
                    frontend.frontend_port,
                    is_prefill=True,
                ) as prefill2:
                    logger.info(f"Prefill Worker 2 PID: {prefill2.get_pid()}")

                    # Step 4: Run migration test
                    run_migration_test(
                        frontend,
                        prefill1,
                        prefill2,
                        receiving_pattern="Prefill Request ID: ",
                        migration_limit=migration_limit,
                        immediate_kill=immediate_kill,
                        use_chat_completion=(request_api == "chat"),
                        stream=stream,
                        use_long_prompt=True,
                    )


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Migration reuses the same request_id for vLLM, but the prefill worker's "
        "KV cache still holds the request due to delay_free_blocks in disaggregated mode. "
        "With chat completions API, prefix cache hits on chat template tokens cause "
        "an assertion error in vLLM's KV cache manager (save_new_computed_blocks expects "
        "no new computed blocks for existing requests)."
    ),
)
@pytest.mark.timeout(350)  # 3x average
def test_request_migration_vllm_kv_transfer(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    migration_limit,
    immediate_kill,
    request_api,
    stream,
):
    """
    End-to-end test for request migration during KV transfer in disaggregated mode.

    Setup: 1 prefill worker + 2 decode workers

    Parameters:
        immediate_kill: True for abrupt kill (SIGKILL), False for graceful shutdown (SIGTERM)
        migration_limit: > 0 to verify migration succeeds, 0 to verify request fails
        request_api: "chat" for chat completion API, "completion" for completion API
        stream: True for streaming, False for non-streaming
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(
        request, migration_limit=migration_limit, enforce_disagg=True
    ) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start prefill worker first
        with DynamoWorkerProcess(
            request,
            "worker0",
            frontend.frontend_port,
            is_prefill=True,
        ) as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start 2 decode workers
            with DynamoWorkerProcess(
                request,
                "worker1",
                frontend.frontend_port,
                is_prefill=False,
            ) as decode1:
                logger.info(f"Decode Worker 1 PID: {decode1.get_pid()}")

                with DynamoWorkerProcess(
                    request,
                    "worker2",
                    frontend.frontend_port,
                    is_prefill=False,
                ) as decode2:
                    logger.info(f"Decode Worker 2 PID: {decode2.get_pid()}")

                    # Step 4: Run migration test
                    run_migration_test(
                        frontend,
                        decode1,
                        decode2,
                        receiving_pattern="Decode Request ID: ",
                        migration_limit=migration_limit,
                        immediate_kill=immediate_kill,
                        use_chat_completion=(request_api == "chat"),
                        stream=stream,
                        use_long_prompt=True,
                    )


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Migration reuses the same request_id for vLLM, but the prefill worker's "
        "KV cache still holds the request due to delay_free_blocks in disaggregated mode. "
        "With chat completions API, prefix cache hits on chat template tokens cause "
        "an assertion error in vLLM's KV cache manager (save_new_computed_blocks expects "
        "no new computed blocks for existing requests)."
    ),
)
@pytest.mark.timeout(350)  # 3x average
def test_request_migration_vllm_decode(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    migration_limit,
    immediate_kill,
    request_api,
    stream,
):
    """
    End-to-end test for decode worker request migration in disaggregated mode.

    Setup: 1 prefill worker + 2 decode workers

    Parameters:
        immediate_kill: True for abrupt kill (SIGKILL), False for graceful shutdown (SIGTERM)
        migration_limit: > 0 to verify migration succeeds, 0 to verify request fails
        request_api: "chat" for chat completion API, "completion" for completion API
        stream: True for streaming, False for non-streaming
    """
    if not stream:
        pytest.skip(
            "Decode test requires streaming to wait for response before stopping worker"
        )

    # Step 1: Start the frontend
    with DynamoFrontendProcess(
        request, migration_limit=migration_limit, enforce_disagg=True
    ) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start prefill worker first
        with DynamoWorkerProcess(
            request,
            "worker0",
            frontend.frontend_port,
            is_prefill=True,
        ) as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start 2 decode workers
            with DynamoWorkerProcess(
                request,
                "worker1",
                frontend.frontend_port,
                is_prefill=False,
            ) as decode1:
                logger.info(f"Decode Worker 1 PID: {decode1.get_pid()}")

                with DynamoWorkerProcess(
                    request,
                    "worker2",
                    frontend.frontend_port,
                    is_prefill=False,
                ) as decode2:
                    logger.info(f"Decode Worker 2 PID: {decode2.get_pid()}")

                    # Step 4: Run migration test
                    run_migration_test(
                        frontend,
                        decode1,
                        decode2,
                        receiving_pattern="Decode Request ID: ",
                        migration_limit=migration_limit,
                        immediate_kill=immediate_kill,
                        use_chat_completion=(request_api == "chat"),
                        stream=stream,
                        wait_for_new_response_before_stop=True,
                    )
