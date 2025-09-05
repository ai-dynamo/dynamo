# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import time

import pytest
import requests
from huggingface_hub import snapshot_download

from tests.utils.deployment_graph import completions_response_handler
from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)


class DynamoFrontendProcess(ManagedProcess):
    """Process manager for Dynamo frontend"""

    def __init__(self, request):
        command = ["python", "-m", "dynamo.frontend", "--router-mode", "round-robin"]

        log_dir = f"{request.node.name}_frontend"

        # Clean up any existing log directory from previous runs
        try:
            shutil.rmtree(log_dir)
            logger.info(f"Cleaned up existing log directory: {log_dir}")
        except FileNotFoundError:
            # Directory doesn't exist, which is fine
            pass

        super().__init__(
            command=command,
            display_output=True,
            terminate_existing=True,
            log_dir=log_dir,
        )

    def get_pid(self) -> int | None:
        """Get the PID of the worker process"""
        return self.proc.pid if self.proc else None


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with vLLM backend"""

    def __init__(self, request, worker_id: str):
        self.worker_id = worker_id

        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.45",
            "--max-model-len",
            "8192",
            "--migration-limit",
            "3",
        ]

        # Set debug logging environment
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_ENABLED"] = "true"
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
            health_check_urls=[("http://localhost:9345/health", self.is_ready)],
            timeout=300,
            display_output=True,
            terminate_existing=False,
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


def download_model() -> None:
    """
    Download the DeepSeek-R1-Distill-Llama-8B model from HuggingFace Hub if not already cached.
    """
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    logger.info(f"Caching model {model_id}...")

    max_retries = 5
    retry_delay = 30  # seconds

    for attempt in range(max_retries):
        try:
            # Download the model to the default cache directory
            # This will skip download if the model is already cached
            snapshot_download(
                repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                repo_type="model",
                local_files_only=False,
            )
            logger.info(f"Model {model_id} is ready for use")
            return  # Success, exit the function
        except Exception as e:
            if attempt < max_retries - 1:  # Not the last attempt
                logger.warning(
                    f"Failed to download model {model_id} (attempt {attempt + 1}/{max_retries}): {e}"
                )
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:  # Last attempt failed
                logger.error(
                    f"Failed to download model {model_id} after {max_retries} attempts: {e}"
                )
                raise


def send_completion_request(
    prompt: str, max_tokens: int, timeout: int = 120
) -> requests.Response:
    """Send a completion request to the frontend"""
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
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
@pytest.mark.slow
def test_vllm_health_check_active(request, runtime_services):
    """
    End-to-end test for worker fault tolerance with migration support.

    This test verifies that when a worker is killed during request processing,
    the system can handle the failure gracefully and migrate the request to
    another worker.
    """
    # Step 0: Download the model from HuggingFace if not already cached
    download_model()

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
@pytest.mark.slow
def test_vllm_health_check_passive(request, runtime_services):
    """
    End-to-end test for worker fault tolerance with migration support.

    This test verifies that when a worker is killed during request processing,
    the system can handle the failure gracefully and migrate the request to
    another worker.
    """
    # Step 0: Download the model from HuggingFace if not already cached
    download_model()

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
