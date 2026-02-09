# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import time

import pytest
import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.engine_process import FRONTEND_PORT
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.payloads import check_models_api, completions_response_handler

logger = logging.getLogger(__name__)


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with vLLM backend"""

    def __init__(self, request, worker_id: str):
        self.worker_id = worker_id

        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            FAULT_TOLERANCE_MODEL_NAME,
            "--enforce-eager",
            "--max-model-len",
            "8192",
        ]

        # Set debug logging environment
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = "9345"

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
            health_check_urls=[
                (f"http://localhost:{FRONTEND_PORT}/v1/models", check_models_api),
                ("http://localhost:9345/health", self.is_ready),
            ],
            timeout=300,
            display_output=True,
            terminate_all_matching_process_names=False,
            stragglers=["VLLM::EngineCore"],
            straggler_commands=["-m dynamo.vllm"],
            log_dir=log_dir,
        )

    def get_pid(self) -> int | None:
        """Get the PID of the worker process"""
        return self.proc.pid if hasattr(self, "proc") and self.proc else None

    def is_ready(self, response) -> bool:
        """Check the health of the worker process"""
        try:
            data = response.json()
            if data.get("status") == "ready":
                logger.info(
                    f"{self.__class__.__name__} {{ name: {self.worker_id} }} status is ready"
                )
                return True
            logger.warning(
                f"{self.__class__.__name__} {{ name: {self.worker_id} }} status is not ready: {data.get('status')}"
            )
        except ValueError:
            logger.warning(
                f"{self.__class__.__name__} {{ name: {self.worker_id} }} health response is not valid JSON"
            )
        return False


def send_completion_request(
    prompt: str, max_tokens: int, timeout: int = 120
) -> requests.Response:
    """Send a completion request to the frontend"""
    payload = {
        "model": FAULT_TOLERANCE_MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
    }

    headers = {"Content-Type": "application/json"}

    logger.info(
        f"Sending completion request with prompt: '{prompt[:50]}...' and max_tokens: {max_tokens}"
    )

    try:
        response = requests.post(
            "http://localhost:8000/v1/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        logger.info(f"Received response with status code: {response.status_code}")
        return response
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out after {timeout} seconds")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed with error: {e}")
        raise


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.nightly
@pytest.mark.timeout(160)  # 3x average (~50s)
@pytest.mark.skip(reason="Flaky, temporarily disabled")
def test_vllm_health_check_active(request, runtime_services):
    """
    End-to-end test for worker fault tolerance with migration support.

    This test verifies that when a worker is killed during request processing,
    the system can handle the failure gracefully and migrate the request to
    another worker.
    """

    # Step 1: Start the frontend
    logger.info("Starting frontend...")
    with DynamoFrontendProcess(request):
        logger.info("Frontend started.")

        # Step 2: Start a worker
        logger.info("Starting worker...")
        with DynamoWorkerProcess(request, "decode") as worker:
            logger.info(f"Worker PID: {worker.get_pid()}")

            time.sleep(12)  # Give the model some time to get started.

            # Step 3: Send a test request to prove the worker is live.
            test_response = send_completion_request("Who are you?", 100, timeout=60)
            completions_response_handler(test_response)
            logger.info("Test request completed successfully")

            # Step 4: Find and kill vLLM engine processes to force the EngineDeadError condition.
            children = worker.subprocesses()
            logger.info(f"Worker children: {[child.pid for child in children]}")
            for child in children:
                cmdline = child.cmdline()
                if len(cmdline) > 0 and cmdline[0] == "VLLM::EngineCore":
                    logger.warning(
                        f"Killing vLLM engine process {{ pid: {child.pid}, cmdline: '{' '.join(cmdline)}' }}"
                    )
                    child.kill()
                    break

            time.sleep(2)  # Give some time for the worker to stabilize

            # Step 5: Send a request triggering the handler to shutdown everything.
            test_response = send_completion_request("How old are you?", 100, timeout=60)
            logger.error(f"Test request failed: {test_response}")

            # Step 6: Ensure the worker process has been stopped as a result of the EngineDeadError condition.
            if worker.is_running():
                pytest.fail(
                    "Worker should not be running after killing vLLM engine process."
                )


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.nightly
@pytest.mark.timeout(160)  # 3x average (~50s)
def test_vllm_health_check_passive(request, runtime_services, predownload_models):
    """
    End-to-end test for worker fault tolerance with migration support.

    This test verifies that when a worker is killed during request processing,
    the system can handle the failure gracefully and migrate the request to
    another worker.
    """

    # Step 1: Start the frontend
    logger.info("Starting frontend...")
    with DynamoFrontendProcess(request):
        logger.info("Frontend started.")

        # Step 2: Start a worker
        logger.info("Starting worker...")
        with DynamoWorkerProcess(request, "decode") as worker:
            logger.info(f"Worker PID: {worker.get_pid()}")

            time.sleep(12)  # Give the model some time to get started.

            # Step 3: Send a test request to prove the worker is live.
            test_response = send_completion_request("Who are you?", 100, timeout=60)
            completions_response_handler(test_response)
            logger.info("Test request completed successfully")

            # Step 4: Find and kill vLLM engine processes to force the EngineDeadError condition.
            children = worker.subprocesses()
            logger.info(f"Worker children: {[child.pid for child in children]}")
            for child in children:
                cmdline = child.cmdline()
                if len(cmdline) > 0 and cmdline[0] == "VLLM::EngineCore":
                    logger.warning(
                        f"Killing vLLM engine process {{ pid: {child.pid}, cmdline: '{' '.join(cmdline)}' }}"
                    )
                    child.kill()
                    break

            time.sleep(6)  # Give some time for the worker to stabilize

            # Step 5: Ensure the worker process has been stopped as a result of the EngineDeadError condition.
            if worker.is_running():
                pytest.fail(
                    "Worker should not be running after killing vLLM engine process."
                )
