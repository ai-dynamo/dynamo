# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test Execution Times (Last Run: 2026-01-09):
- test_request_migration_vllm_aggregated: ~95s
- test_request_migration_vllm_prefill: N/A
- test_request_migration_vllm_kv_transfer: N/A
- test_request_migration_vllm_decode: ~115s
"""

import json
import logging
import os
from pathlib import Path

import pytest

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME, DynamoPortRange
from tests.utils.gpu_args import build_gpu_mem_args
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import allocate_port, deallocate_ports

# Customized utils for migration tests
from .utils import (
    DynamoFrontendProcess,
    managed_processes_concurrently,
    run_migration_test,
)

logger = logging.getLogger(__name__)

AGGREGATED_MAX_MODEL_LEN = 1024
AGGREGATED_MAX_TOKENS = 512

# Cover each distinct migration policy with complementary lifecycle, API,
# response, and transport values. Together these eight rows cover every pair
# of those four binary dimensions without paying for their Cartesian product.
MIGRATION_CASES = [
    pytest.param(
        3,
        None,
        True,
        "chat",
        True,
        "nats",
        id="migration_enabled-no_seq_cap-worker_failure-chat-stream-nats",
    ),
    pytest.param(
        3,
        None,
        False,
        "completion",
        False,
        "tcp",
        id="migration_enabled-no_seq_cap-graceful_shutdown-completion-unary-tcp",
    ),
    pytest.param(
        0,
        None,
        True,
        "chat",
        False,
        "tcp",
        id="migration_disabled-worker_failure-chat-unary-tcp",
    ),
    pytest.param(
        0,
        None,
        False,
        "completion",
        True,
        "nats",
        id="migration_disabled-graceful_shutdown-completion-stream-nats",
    ),
    pytest.param(
        3,
        1,
        True,
        "completion",
        True,
        "tcp",
        id="max_seq_len_exceeded-worker_failure-completion-stream-tcp",
    ),
    pytest.param(
        3,
        1,
        False,
        "chat",
        False,
        "nats",
        id="max_seq_len_exceeded-graceful_shutdown-chat-unary-nats",
    ),
    pytest.param(
        3,
        1_000_000,
        True,
        "completion",
        False,
        "nats",
        id="max_seq_len_not_exceeded-worker_failure-completion-unary-nats",
    ),
    pytest.param(
        3,
        1_000_000,
        False,
        "chat",
        True,
        "tcp",
        id="max_seq_len_not_exceeded-graceful_shutdown-chat-stream-tcp",
    ),
]

pytestmark = [
    pytest.mark.fault_tolerance,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.e2e,
    pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME),
    pytest.mark.parametrize(
        (
            "migration_limit",
            "migration_max_seq_len",
            "immediate_kill",
            "request_api",
            "stream",
            "request_plane",
        ),
        MIGRATION_CASES,
        indirect=["request_plane"],
    ),
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
        max_model_len: Maximum input-plus-output context exposed by the worker
    """

    def __init__(
        self,
        request,
        worker_id: str,
        frontend_port: int,
        log_root: Path,
        is_prefill: bool | None = None,
        max_model_len: int = 8192,
    ):
        self.worker_id = worker_id
        allocated_ports: list[int] = []
        request.addfinalizer(lambda ports=allocated_ports: deallocate_ports(ports))

        self.system_port = allocate_port(DynamoPortRange.SERVE.value)
        allocated_ports.append(self.system_port)

        self.nixl_side_channel_port = allocate_port(DynamoPortRange.NIXL.value)
        allocated_ports.append(self.nixl_side_channel_port)

        # vLLM defaults every engine to the same torch.distributed rendezvous
        # port (29501). TP=1 does not always bind it, but assigning it explicitly
        # avoids startup races when several engine processes initialize together.
        self.master_port = allocate_port(DynamoPortRange.BOOTSTRAP.value)
        allocated_ports.append(self.master_port)

        env = os.environ.copy()
        if "_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES" not in env:
            kv_mark = request.node.get_closest_marker("requested_vllm_kv_cache_bytes")
            if kv_mark:
                env["_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES"] = str(int(kv_mark.args[0]))

        gpu_mem_args = build_gpu_mem_args("build_vllm_gpu_mem_args", env=env)
        if not gpu_mem_args:
            gpu_mem_args = [
                "--num-gpu-blocks-override",
                "512",  # 8192 tokens / 16 tokens per block
                "--gpu-memory-utilization",
                "0.15",
            ]

        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            FAULT_TOLERANCE_MODEL_NAME,
            "--enforce-eager",
            "--max-model-len",
            str(max_model_len),
            "--max-num-seqs",
            "1",  # number of requests at a time
            "--master-port",
            str(self.master_port),
            *gpu_mem_args,
        ]
        if is_prefill is True:
            command.extend(["--disaggregation-mode", "prefill"])
        elif is_prefill is False:
            command.extend(["--disaggregation-mode", "decode"])

        # Aggregated mode and prefill workers publish KV events
        if is_prefill is not False:
            kv_event_port = allocate_port(DynamoPortRange.SERVE.value)
            allocated_ports.append(kv_event_port)
            command.extend(
                [
                    "--kv-events-config",
                    json.dumps(
                        {
                            "publisher": "zmq",
                            "topic": "kv-events",
                            "endpoint": f"tcp://*:{kv_event_port}",
                            "enable_kv_cache_events": True,
                        }
                    ),
                ]
            )

        # Set environment variables
        env["DYN_REQUEST_PLANE"] = request.getfixturevalue("request_plane")

        # All workers need unique NIXL side channel ports for KV transfer
        env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(self.nixl_side_channel_port)

        env["DYN_LOG"] = "debug"
        # Disable canary health check - these tests expect full control over requests
        # sent to the workers where canary health check intermittently sends dummy
        # requests to workers interfering with the test process which may cause
        # intermittent failures
        env["DYN_HEALTH_CHECK_ENABLED"] = "false"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = str(self.system_port)
        env["DYN_HTTP_PORT"] = str(frontend_port)

        # Disable backend shutdown grace period for all migration tests
        env["DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS"] = "0"

        # Configure health check based on worker type
        health_check_urls = [
            (f"http://localhost:{self.system_port}/health", self.is_ready)
        ]
        if is_prefill is None or is_prefill is False:
            # aggregated or decode
            health_check_urls.append(
                (f"http://localhost:{frontend_port}/v1/models", check_models_api)
            )

        log_dir = log_root / worker_id

        super().__init__(
            command=command,
            env=env,
            health_check_urls=health_check_urls,
            timeout=300,
            # Every worker retains a complete per-test log. Avoid interleaving
            # verbose engine output when several GPU tests run concurrently.
            display_output=False,
            terminate_all_matching_process_names=False,
            stragglers=["VLLM::EngineCore"],
            straggler_commands=["-m dynamo.vllm"],
            log_dir=str(log_dir),
            display_name=worker_id,
        )

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
@pytest.mark.post_merge
@pytest.mark.profiled_vram_gib(4.8)
@pytest.mark.requested_vllm_kv_cache_bytes(331_711_000)
def test_request_migration_vllm_aggregated(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    migration_limit,
    migration_max_seq_len,
    immediate_kill,
    request_api,
    stream,
    tmp_path,
):
    """
    End-to-end test for aggregated worker request migration.

    Parameters:
        immediate_kill: True for abrupt kill (SIGKILL), False for graceful shutdown (SIGTERM)
        migration_limit: > 0 to verify migration succeeds, 0 to verify request fails
        migration_max_seq_len: Max sequence length for migration state tracking
        request_api: "chat" for chat completion API, "completion" for completion API
        stream: True for streaming, False for non-streaming
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(
        request,
        migration_limit=migration_limit,
        migration_max_seq_len=migration_max_seq_len,
    ) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 independent workers concurrently
        worker1 = DynamoWorkerProcess(
            request,
            "worker1",
            frontend.frontend_port,
            tmp_path,
            max_model_len=AGGREGATED_MAX_MODEL_LEN,
        )
        worker2 = DynamoWorkerProcess(
            request,
            "worker2",
            frontend.frontend_port,
            tmp_path,
            max_model_len=AGGREGATED_MAX_MODEL_LEN,
        )
        with managed_processes_concurrently(worker1, worker2):
            logger.info(f"Worker 1 PID: {worker1.get_pid()}")
            logger.info(f"Worker 2 PID: {worker2.get_pid()}")

            # Step 3: Run migration test
            run_migration_test(
                frontend,
                worker1,
                worker2,
                receiving_pattern="Decode Request ID: ",
                migration_limit=migration_limit,
                migration_max_seq_len=migration_max_seq_len,
                immediate_kill=immediate_kill,
                use_chat_completion=(request_api == "chat"),
                stream=stream,
                max_tokens=AGGREGATED_MAX_TOKENS,
            )


@pytest.mark.skip(reason="Prefill migration not yet supported")
@pytest.mark.timeout(350)  # 3x average
@pytest.mark.nightly
def test_request_migration_vllm_prefill(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    migration_limit,
    migration_max_seq_len,
    immediate_kill,
    request_api,
    stream,
    tmp_path,
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
        request,
        migration_limit=migration_limit,
        migration_max_seq_len=migration_max_seq_len,
    ) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start decode worker first (required for prefill workers to connect)
        with DynamoWorkerProcess(
            request,
            "worker0",
            frontend.frontend_port,
            tmp_path,
            is_prefill=False,
        ) as decode_worker:
            logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

            # Step 3: Start 2 prefill workers
            with DynamoWorkerProcess(
                request,
                "worker1",
                frontend.frontend_port,
                tmp_path,
                is_prefill=True,
            ) as prefill1:
                logger.info(f"Prefill Worker 1 PID: {prefill1.get_pid()}")

                with DynamoWorkerProcess(
                    request,
                    "worker2",
                    frontend.frontend_port,
                    tmp_path,
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
                        migration_max_seq_len=migration_max_seq_len,
                        immediate_kill=immediate_kill,
                        use_chat_completion=(request_api == "chat"),
                        stream=stream,
                        use_long_prompt=True,
                    )


@pytest.mark.skip(
    reason=(
        "Migration reuses the same request_id for vLLM, but the prefill worker's "
        "KV cache still holds the request due to delay_free_blocks in disaggregated mode. "
        "With chat completions API, prefix cache hits on chat template tokens cause "
        "an assertion error in vLLM's KV cache manager (save_new_computed_blocks expects "
        "no new computed blocks for existing requests)."
    ),
)
@pytest.mark.timeout(350)  # 3x average
@pytest.mark.nightly
def test_request_migration_vllm_kv_transfer(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    migration_limit,
    migration_max_seq_len,
    immediate_kill,
    request_api,
    stream,
    tmp_path,
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
        request,
        migration_limit=migration_limit,
        migration_max_seq_len=migration_max_seq_len,
    ) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start prefill worker first
        with DynamoWorkerProcess(
            request,
            "worker0",
            frontend.frontend_port,
            tmp_path,
            is_prefill=True,
        ) as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start 2 decode workers
            with DynamoWorkerProcess(
                request,
                "worker1",
                frontend.frontend_port,
                tmp_path,
                is_prefill=False,
            ) as decode1:
                logger.info(f"Decode Worker 1 PID: {decode1.get_pid()}")

                with DynamoWorkerProcess(
                    request,
                    "worker2",
                    frontend.frontend_port,
                    tmp_path,
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
                        migration_max_seq_len=migration_max_seq_len,
                        immediate_kill=immediate_kill,
                        use_chat_completion=(request_api == "chat"),
                        stream=stream,
                        use_long_prompt=True,
                    )


@pytest.mark.skip(
    reason=(
        "Migration reuses the same request_id for vLLM, but the prefill worker's "
        "KV cache still holds the request due to delay_free_blocks in disaggregated mode. "
        "With chat completions API, prefix cache hits on chat template tokens cause "
        "an assertion error in vLLM's KV cache manager (save_new_computed_blocks expects "
        "no new computed blocks for existing requests)."
    ),
)
@pytest.mark.timeout(350)  # 3x average
@pytest.mark.nightly
def test_request_migration_vllm_decode(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    migration_limit,
    migration_max_seq_len,
    immediate_kill,
    request_api,
    stream,
    tmp_path,
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
        request,
        migration_limit=migration_limit,
        migration_max_seq_len=migration_max_seq_len,
    ) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start prefill worker first
        with DynamoWorkerProcess(
            request,
            "worker0",
            frontend.frontend_port,
            tmp_path,
            is_prefill=True,
        ) as prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start 2 decode workers
            with DynamoWorkerProcess(
                request,
                "worker1",
                frontend.frontend_port,
                tmp_path,
                is_prefill=False,
            ) as decode1:
                logger.info(f"Decode Worker 1 PID: {decode1.get_pid()}")

                with DynamoWorkerProcess(
                    request,
                    "worker2",
                    frontend.frontend_port,
                    tmp_path,
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
                        migration_max_seq_len=migration_max_seq_len,
                        immediate_kill=immediate_kill,
                        use_chat_completion=(request_api == "chat"),
                        stream=stream,
                        wait_for_new_response_before_stop=True,
                    )
