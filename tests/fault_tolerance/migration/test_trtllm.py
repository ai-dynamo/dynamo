# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM request-migration fault-tolerance tests."""

import logging
import os
import time
from pathlib import Path

import pytest
import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME, DynamoPortRange
from tests.utils.gpu_args import build_trtllm_override_args
from tests.utils.managed_process import ManagedProcess, check_health_ready
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import allocate_port, deallocate_ports
from tests.utils.prometheus import sum_metric_samples

# Customized utils for migration tests
from .request_utils import (
    OutputContinuityError,
    assert_output_prefix,
    request_to_completion,
)
from .utils import (
    DynamoFrontendProcess,
    managed_processes_concurrently,
    run_migration_test,
    wait_for_endpoint_instances,
)

logger = logging.getLogger(__name__)

AGGREGATED_MAX_SEQ_LEN = 4096
AGGREGATED_MAX_TOKENS = 3072
AGGREGATED_PROMPT_REPETITIONS = 128
DECODE_MAX_SEQ_LEN = 4096
DECODE_MAX_TOKENS = 3072
KV_TRANSFER_MAX_SEQ_LEN = 2048
KV_TRANSFER_MAX_TOKENS = 1536
KV_TRANSFER_BASELINE_MAX_TOKENS = 256
KV_TRANSFER_BASELINE_REQUESTS = 3
KV_TRANSFER_PROMPT_REPETITIONS = 128

# Keep one active worker-failure case for every migration-policy outcome while
# varying API, response, and request plane.
MIGRATION_CASES = [
    pytest.param(
        3,
        None,
        True,
        "completion",
        True,
        "nats",
        id="migration_enabled-no_seq_cap-worker_failure-completion-stream-nats",
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
        1_000_000,
        True,
        "completion",
        False,
        "nats",
        id="max_seq_len_not_exceeded-worker_failure-completion-unary-nats",
    ),
]

# Decode migration must be streaming so the fault can be injected after
# generation starts.
DECODE_MIGRATION_CASES = [
    pytest.param(
        3,
        None,
        True,
        "completion",
        "nats",
        id="migration_enabled-worker_failure-completion-stream-nats",
    ),
    pytest.param(
        0,
        None,
        True,
        "completion",
        "tcp",
        id="migration_disabled-worker_failure-completion-stream-tcp",
    ),
    pytest.param(
        3,
        1,
        True,
        "chat",
        "tcp",
        id="max_seq_len_exceeded-worker-failure-chat-stream-tcp",
    ),
    pytest.param(
        3,
        1_000_000,
        True,
        "chat",
        "nats",
        id="max_seq_len_not_exceeded-worker-failure-chat-stream-nats",
    ),
]

# KV re-transfer is exercised after the first decode token proves the initial
# transfer completed. These cases isolate API and request-plane behavior;
# graceful shutdown remains skipped until TRT-LLM emits a retryable error.
KV_TRANSFER_CASES = [
    pytest.param(
        3, None, True, "chat", True, "nats", id="worker-failure-chat-stream-nats"
    ),
    pytest.param(
        3,
        None,
        True,
        "completion",
        True,
        "tcp",
        marks=pytest.mark.xfail(
            reason=(
                "Known TensorRT-LLM 1.3.0rc25 KV-migration output-continuity "
                "failure; tracked in https://github.com/ai-dynamo/dynamo/pull/14609"
            ),
            raises=OutputContinuityError,
            strict=False,
        ),
        id="worker-failure-completion-stream-tcp",
    ),
]

MIGRATION_PARAMETERS = pytest.mark.parametrize(
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
)

DECODE_MIGRATION_PARAMETERS = pytest.mark.parametrize(
    (
        "migration_limit",
        "migration_max_seq_len",
        "immediate_kill",
        "request_api",
        "request_plane",
    ),
    DECODE_MIGRATION_CASES,
    indirect=["request_plane"],
)

KV_TRANSFER_MIGRATION_PARAMETERS = pytest.mark.parametrize(
    (
        "migration_limit",
        "migration_max_seq_len",
        "immediate_kill",
        "request_api",
        "stream",
        "request_plane",
    ),
    KV_TRANSFER_CASES,
    indirect=["request_plane"],
)


def read_trtllm_kv_transfer_metrics(worker_system_port: int) -> tuple[float, float]:
    """Read the worker's successful-transfer count and transferred-byte sum."""
    response = requests.get(f"http://localhost:{worker_system_port}/metrics", timeout=1)
    response.raise_for_status()
    return (
        sum_metric_samples(response.text, "trtllm_kv_transfer_success_total"),
        sum_metric_samples(response.text, "trtllm_kv_transfer_bytes_sum"),
    )


def wait_for_trtllm_kv_transfer_success(
    worker_system_port: int,
    baseline_count: float,
    baseline_bytes: float,
    max_wait_time: float = 10.0,
) -> None:
    """Require the replacement decode worker to complete one KV transfer."""
    deadline = time.monotonic() + max_wait_time
    transfer_count = 0.0
    transfer_bytes = 0.0
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            transfer_count, transfer_bytes = read_trtllm_kv_transfer_metrics(
                worker_system_port
            )
            if (
                transfer_count - baseline_count == 1
                and transfer_bytes - baseline_bytes > 0
            ):
                return
        except (requests.RequestException, ValueError) as error:
            last_error = error
        time.sleep(0.1)

    pytest.fail(
        "Replacement decode worker did not complete exactly one positive-byte "
        f"KV transfer; baseline_count={baseline_count}, "
        f"transfer_count={transfer_count}, baseline_bytes={baseline_bytes}, "
        f"transfer_bytes={transfer_bytes}, last_error={last_error}"
    )


pytestmark = [
    pytest.mark.fault_tolerance,
    pytest.mark.trtllm,
    pytest.mark.gpu_1,
    pytest.mark.e2e,
    pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME),
]


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with TRT-LLM backend

    Supports both aggregated mode (single worker) and disaggregated mode
    (separate prefill and decode workers).

    Args:
        request: pytest request fixture
        worker_id: Unique identifier for the worker (e.g., "worker1", "prefill1")
        frontend_port: Port where the frontend is running
        mode: "agg" for aggregated, "prefill" or "decode" for disaggregated
    """

    def __init__(
        self,
        request,
        worker_id: str,
        frontend_port: int,
        log_root: Path,
        mode: str = "agg",
        max_seq_len: int = 8192,
        publish_metrics: bool = False,
        enable_block_reuse: bool = True,
    ):
        self.worker_id = worker_id
        allocated_ports: list[int] = []
        request.addfinalizer(lambda ports=allocated_ports: deallocate_ports(ports))

        self.system_port = allocate_port(DynamoPortRange.SERVE.value)
        allocated_ports.append(self.system_port)
        self.mode = mode

        env = os.environ.copy()
        if "_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS" not in env:
            kv_mark = request.node.get_closest_marker("requested_trtllm_kv_tokens")
            if kv_mark:
                env["_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS"] = str(
                    int(kv_mark.args[0])
                )

        command = [
            "python3",
            "-m",
            "dynamo.trtllm",
            "--model",
            FAULT_TOLERANCE_MODEL_NAME,
            "--disaggregation-mode",
            mode,
            "--max-seq-len",
            str(max_seq_len),
            "--max-num-tokens",
            str(max_seq_len),
            "--free-gpu-memory-fraction",
            "0.15",  # avoid validation error on TRT-LLM available memory checks
        ]
        if mode != "agg":
            config_file = log_root / f"trtllm_config_{self.system_port}.yaml"
            with config_file.open("w") as f:
                f.write(
                    "cache_transceiver_config:\n"
                    "  backend: DEFAULT\n"
                    f"  max_tokens_in_buffer: {max_seq_len}\n"
                )
                f.write("disable_overlap_scheduler: true\n")
                f.write("kv_cache_config:\n" f"  max_tokens: {max_seq_len}\n")
                if not enable_block_reuse:
                    f.write("  enable_block_reuse: false\n")
            command += ["--extra-engine-args", str(config_file)]
        if publish_metrics:
            command.append("--publish-metrics")
        command.extend(build_trtllm_override_args(env))

        # Set environment variables
        env["DYN_REQUEST_PLANE"] = request.getfixturevalue("request_plane")

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
            (f"http://localhost:{self.system_port}/health", check_health_ready)
        ]
        if mode in ["decode", "agg"]:
            health_check_urls.append(
                (f"http://localhost:{frontend_port}/v1/models", check_models_api)
            )

        log_dir = log_root / worker_id

        super().__init__(
            command=command,
            env=env,
            health_check_urls=health_check_urls,
            timeout=300,
            display_output=False,
            terminate_all_matching_process_names=False,
            log_dir=str(log_dir),
            display_name=worker_id,
        )


@pytest.mark.timeout(290)  # 3x average
@pytest.mark.nightly
@pytest.mark.profiled_vram_gib(7.2)
@pytest.mark.requested_trtllm_kv_tokens(4096)
@MIGRATION_PARAMETERS
def test_request_migration_trtllm_aggregated(
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

        worker1 = DynamoWorkerProcess(
            request,
            "worker1",
            frontend.frontend_port,
            tmp_path,
            max_seq_len=AGGREGATED_MAX_SEQ_LEN,
        )
        worker2 = DynamoWorkerProcess(
            request,
            "worker2",
            frontend.frontend_port,
            tmp_path,
            max_seq_len=AGGREGATED_MAX_SEQ_LEN,
        )
        with managed_processes_concurrently(worker1, worker2):
            logger.info("Worker 1 PID: %s", worker1.get_pid())
            logger.info("Worker 2 PID: %s", worker2.get_pid())
            wait_for_endpoint_instances(
                frontend.frontend_port,
                {("backend", "generate"): 2},
            )

            run_migration_test(
                frontend,
                worker1,
                worker2,
                receiving_pattern="AggregatedHandler Request ID: ",
                migration_limit=migration_limit,
                migration_max_seq_len=migration_max_seq_len,
                immediate_kill=immediate_kill,
                use_chat_completion=(request_api == "chat"),
                stream=stream,
                max_tokens=AGGREGATED_MAX_TOKENS,
                use_long_prompt=True,
                long_prompt_repetitions=AGGREGATED_PROMPT_REPETITIONS,
                expected_ongoing_request_count=1,
                verify_replacement_worker=True,
                force_max_output_tokens=True,
            )


@pytest.mark.timeout(350)  # 3x average
@pytest.mark.nightly
@pytest.mark.profiled_vram_gib(10.4)
@pytest.mark.requested_trtllm_kv_tokens(2048)
@KV_TRANSFER_MIGRATION_PARAMETERS
def test_request_migration_trtllm_kv_transfer(
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

        prefill_worker = DynamoWorkerProcess(
            request,
            "worker0",
            frontend.frontend_port,
            tmp_path,
            mode="prefill",
            max_seq_len=KV_TRANSFER_MAX_SEQ_LEN,
            enable_block_reuse=False,
        )
        decode1 = DynamoWorkerProcess(
            request,
            "worker1",
            frontend.frontend_port,
            tmp_path,
            mode="decode",
            max_seq_len=KV_TRANSFER_MAX_SEQ_LEN,
            publish_metrics=True,
            enable_block_reuse=False,
        )
        decode2 = DynamoWorkerProcess(
            request,
            "worker2",
            frontend.frontend_port,
            tmp_path,
            mode="decode",
            max_seq_len=KV_TRANSFER_MAX_SEQ_LEN,
            publish_metrics=True,
            enable_block_reuse=False,
        )
        with managed_processes_concurrently(prefill_worker, decode1, decode2):
            logger.info("Prefill Worker PID: %s", prefill_worker.get_pid())
            logger.info("Decode Worker 1 PID: %s", decode1.get_pid())
            logger.info("Decode Worker 2 PID: %s", decode2.get_pid())
            wait_for_endpoint_instances(
                frontend.frontend_port,
                {("prefill", "generate"): 1, ("backend", "generate"): 2},
            )

            # GPU model output can diverge despite fixed sampling parameters.
            # Characterize the stable prefix across both round-robin decode
            # workers, then require migration to preserve it past the fault.
            baseline_outputs = [
                request_to_completion(
                    frontend.frontend_port,
                    use_chat_completion=(request_api == "chat"),
                    stream=stream,
                    max_tokens=KV_TRANSFER_BASELINE_MAX_TOKENS,
                    use_long_prompt=True,
                    long_prompt_repetitions=KV_TRANSFER_PROMPT_REPETITIONS,
                    force_max_output_tokens=True,
                )
                for _ in range(KV_TRANSFER_BASELINE_REQUESTS)
            ]
            stable_output_prefix = os.path.commonprefix(baseline_outputs)

            transfer_baselines = {
                worker.system_port: read_trtllm_kv_transfer_metrics(worker.system_port)
                for worker in (decode1, decode2)
            }

            replacement_worker, migrated_output = run_migration_test(
                frontend,
                decode1,
                decode2,
                receiving_pattern="Decode Request ID: ",
                migration_limit=migration_limit,
                migration_max_seq_len=migration_max_seq_len,
                immediate_kill=immediate_kill,
                use_chat_completion=(request_api == "chat"),
                stream=stream,
                max_tokens=KV_TRANSFER_MAX_TOKENS,
                use_long_prompt=True,
                long_prompt_repetitions=KV_TRANSFER_PROMPT_REPETITIONS,
                wait_for_new_response_before_stop=True,
                expected_ongoing_request_count=1,
                verify_replacement_worker=True,
                force_max_output_tokens=True,
                expected_output_prefix=stable_output_prefix,
            )
            wait_for_trtllm_kv_transfer_success(
                replacement_worker.system_port,
                *transfer_baselines[replacement_worker.system_port],
            )
            assert migrated_output is not None
            assert_output_prefix(migrated_output, stable_output_prefix)


@pytest.mark.timeout(350)  # 3x average
@pytest.mark.nightly
@pytest.mark.profiled_vram_gib(12.6)
@pytest.mark.requested_trtllm_kv_tokens(4096)
@DECODE_MIGRATION_PARAMETERS
def test_request_migration_trtllm_decode(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    migration_limit,
    migration_max_seq_len,
    immediate_kill,
    request_api,
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
    # Step 1: Start the frontend
    with DynamoFrontendProcess(
        request,
        migration_limit=migration_limit,
        migration_max_seq_len=migration_max_seq_len,
    ) as frontend:
        logger.info("Frontend started successfully")

        prefill_worker = DynamoWorkerProcess(
            request,
            "worker0",
            frontend.frontend_port,
            tmp_path,
            mode="prefill",
            max_seq_len=DECODE_MAX_SEQ_LEN,
        )
        decode1 = DynamoWorkerProcess(
            request,
            "worker1",
            frontend.frontend_port,
            tmp_path,
            mode="decode",
            max_seq_len=DECODE_MAX_SEQ_LEN,
        )
        decode2 = DynamoWorkerProcess(
            request,
            "worker2",
            frontend.frontend_port,
            tmp_path,
            mode="decode",
            max_seq_len=DECODE_MAX_SEQ_LEN,
        )
        with managed_processes_concurrently(prefill_worker, decode1, decode2):
            logger.info("Prefill Worker PID: %s", prefill_worker.get_pid())
            logger.info("Decode Worker 1 PID: %s", decode1.get_pid())
            logger.info("Decode Worker 2 PID: %s", decode2.get_pid())
            wait_for_endpoint_instances(
                frontend.frontend_port,
                {("prefill", "generate"): 1, ("backend", "generate"): 2},
            )

            run_migration_test(
                frontend,
                decode1,
                decode2,
                receiving_pattern="Decode Request ID: ",
                migration_limit=migration_limit,
                migration_max_seq_len=migration_max_seq_len,
                immediate_kill=immediate_kill,
                use_chat_completion=(request_api == "chat"),
                stream=True,
                max_tokens=DECODE_MAX_TOKENS,
                wait_for_new_response_before_stop=True,
                expected_ongoing_request_count=1,
                verify_replacement_worker=True,
                force_max_output_tokens=True,
            )
