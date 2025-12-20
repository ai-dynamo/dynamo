# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test for TRT-LLM worker rollout restart recovery with continuous background requests.
"""

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field

import pytest

from tests.fault_tolerance.cancellation.utils import (
    DynamoFrontendProcess,
    send_chat_completion_request,
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
    pytest.mark.post_merge,
    pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True),
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
            "--max-seq-len",
            "16384",
            "--max-num-tokens",
            "16384",
            "--migration-limit",
            migration_limit,
        ]
        if mode != "prefill_and_decode":
            with open("test_trtllm_rollout_restart_config.yaml", "w") as f:
                f.write(
                    "cache_transceiver_config:\n  backend: DEFAULT\n  max_tokens_in_buffer: 16384\n"
                )
                f.write("disable_overlap_scheduler: true\n")
                f.write("kv_cache_config:\n  max_tokens: 16384\n")
            command += [
                "--extra-engine-args",
                "test_trtllm_rollout_restart_config.yaml",
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

        env["DYN_LOG"] = "warn"
        # Disable canary health check - these tests expect full control over requests
        # sent to the workers where canary health check intermittently sends dummy
        # requests to workers interfering with the test process which may cause
        # intermittent failures
        env["DYN_HEALTH_CHECK_ENABLED"] = "false"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = str(system_port)
        env["TLLM_LOG_LEVEL"] = "info"

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


def test_worker_restart_recovery_trtllm(
    request, runtime_services_dynamic_ports, predownload_models
):
    """
    End-to-end test for worker restart recovery in disaggregated mode with rolling restarts.

    This test verifies that the system can recover when workers are stopped
    and restarted in a disaggregated setup. A background thread continuously sends
    requests to verify the system remains responsive during rolling restarts.

    The test performs rolling restart cycles where each cycle:
    1. Restarts decode worker
    2. Restarts prefill worker

    A background thread continuously sends requests throughout the test to verify
    the system remains responsive during worker transitions.

    Timing (Estimated): ~300s total for 2 cycles
    - Engine initialization (first cycle): ~92s (frontend: 2s, prefill: 45s, decode: 45s)
    - Per cycle: ~100s (decode restart: 45s + prefill restart: 45s)
    """

    @dataclass
    class RequestStats:
        """Thread-safe statistics for background request tracking."""

        success: int = 0
        failure: int = 0
        _lock: threading.Lock = field(default_factory=threading.Lock)

        def record_success(self):
            with self._lock:
                self.success += 1
                return self.success

        def record_failure(self):
            with self._lock:
                self.failure += 1
                return self.failure

        def get_stats(self):
            with self._lock:
                return self.success, self.failure

    def background_request_sender(
        frontend_port: int, stop_event: threading.Event, stats: RequestStats
    ):
        """Background thread that continuously sends requests to the frontend."""
        while not stop_event.is_set():
            try:
                # Send a simple streaming request
                cancellable_req = send_chat_completion_request(
                    prompt="Count from 1 to 5",
                    max_tokens=20,
                    frontend_port=frontend_port,
                    stream=True,
                )
                # Wait a bit for the request to start
                time.sleep(0.5)

                # Check if we should stop before reading
                if stop_event.is_set():
                    break

                # Read responses manually to avoid pytest.fail in thread
                response = cancellable_req.response
                if response is None:
                    raise RuntimeError(
                        "Response is None - request may not have completed"
                    )
                if response.status_code != 200:
                    raise RuntimeError(f"Bad status code: {response.status_code}")

                # Read at least 3 streaming responses
                response_count = 0
                for line in response.iter_lines():
                    if stop_event.is_set():
                        break
                    response_count += 1
                    if response_count >= 3:
                        break

                if response_count >= 3:
                    total_success = stats.record_success()
                    logger.info(
                        f"Background request succeeded. Total success: {total_success}"
                    )
                else:
                    raise RuntimeError(f"Only got {response_count} responses")

            except Exception as e:
                total_failure = stats.record_failure()
                logger.warning(
                    f"Background request failed: {e}. Total failures: {total_failure}"
                )

    # Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Dictionary to track active workers
        workers = {"prefill": None, "decode": None}

        # Initialize both workers
        workers["prefill"] = DynamoWorkerProcess(
            request, frontend.frontend_port, mode="prefill"
        )
        workers["prefill"].__enter__()
        logger.info(f"Initial Prefill Worker PID: {workers['prefill'].get_pid()}")

        workers["decode"] = DynamoWorkerProcess(
            request, frontend.frontend_port, mode="decode"
        )
        workers["decode"].__enter__()
        logger.info(f"Initial Decode Worker PID: {workers['decode'].get_pid()}")

        time.sleep(2)

        # Start background request senders
        num_threads = 8
        stop_event = threading.Event()
        stats = RequestStats()
        background_threads = []
        for i in range(num_threads):
            thread = threading.Thread(
                target=background_request_sender,
                args=(frontend.frontend_port, stop_event, stats),
                daemon=True,
                name=f"request-sender-{i}",
            )
            thread.start()
            background_threads.append(thread)
        logger.info(f"Started {num_threads} background request sender threads")

        # Perform rolling restart cycles
        num_cycles = 2
        try:
            for cycle in range(1, num_cycles + 1):
                success, failure = stats.get_stats()
                logger.info(
                    f"=== Starting rolling restart cycle {cycle}/{num_cycles} === "
                    f"(Success: {success}, Failures: {failure})"
                )

                # Rolling restart: Decode worker (start new before stopping old)
                logger.info(f"Cycle {cycle}: Starting new decode worker...")
                old_decode_worker = workers["decode"]
                workers["decode"] = DynamoWorkerProcess(
                    request, frontend.frontend_port, mode="decode"
                )
                workers["decode"].__enter__()
                logger.info(
                    f"Cycle {cycle}: New Decode Worker started with PID: {workers['decode'].get_pid()}"
                )

                logger.info(f"Cycle {cycle}: Stopping old decode worker...")
                old_decode_worker.__exit__(None, None, None)
                logger.info(f"Cycle {cycle}: Old decode worker stopped")

                time.sleep(2)

                # Rolling restart: Prefill worker (start new before stopping old)
                logger.info(f"Cycle {cycle}: Starting new prefill worker...")
                old_prefill_worker = workers["prefill"]
                workers["prefill"] = DynamoWorkerProcess(
                    request, frontend.frontend_port, mode="prefill"
                )
                workers["prefill"].__enter__()
                logger.info(
                    f"Cycle {cycle}: New Prefill Worker started with PID: {workers['prefill'].get_pid()}"
                )

                logger.info(f"Cycle {cycle}: Stopping old prefill worker...")
                old_prefill_worker.__exit__(None, None, None)
                logger.info(f"Cycle {cycle}: Old prefill worker stopped")

                time.sleep(2)

                success, failure = stats.get_stats()
                logger.info(
                    f"=== Completed rolling restart cycle {cycle}/{num_cycles} === "
                    f"(Success: {success}, Failures: {failure})"
                )
        finally:
            # Stop all background threads
            stop_event.set()
            for thread in background_threads:
                thread.join(timeout=5)
            logger.info(f"Stopped {num_threads} background request sender threads")

        # Clean up workers
        logger.info("Cleaning up workers...")
        for mode, worker in workers.items():
            if worker is not None:
                worker.__exit__(None, None, None)
                logger.info(f"Cleaned up {mode} worker")

        success, failure = stats.get_stats()
        logger.info(
            f"Worker restart recovery test completed - "
            f"system recovered from {num_cycles} complete rolling restart cycles. "
            f"Final stats: Success={success}, Failures={failure}"
        )

        # Assert that at least some requests succeeded during rolling restarts
        assert success > 0, "Expected at least some background requests to succeed"
