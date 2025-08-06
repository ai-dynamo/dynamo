# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import time

import pytest
import requests
from huggingface_hub import snapshot_download

from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)


class DynamoFrontendProcess(ManagedProcess):
    """Process manager for Dynamo frontend"""

    def __init__(self, request):
        command = ["python", "-m", "dynamo.frontend"]

        # Set debug logging environment
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"

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
            env=env,
            display_output=True,
            terminate_existing=True,
            log_dir=log_dir,
        )


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with vLLM backend"""

    def __init__(self, request):
        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "--enforce-eager",
            "--max-model-len",
            "16384",
            "--migration-limit",
            "3",
        ]

        # Set debug logging environment
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_ENABLED"] = "true"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = "8081"

        log_dir = f"{request.node.name}_worker"

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
            health_check_urls=[("http://localhost:8081/health", self.is_ready)],
            timeout=300,
            display_output=True,
            terminate_existing=True,
            log_dir=log_dir,
        )

    def get_pid(self):
        """Get the PID of the worker process"""
        return self.proc.pid if self.proc else None

    def is_ready(self, response) -> bool:
        """Check the health of the worker process"""
        try:
            data = response.json()
            if data.get("status") == "ready":
                logger.info("Worker status is ready")
                return True
            logger.warning(f"Worker status is not ready: {data.get('status')}")
        except ValueError:
            logger.warning("Worker health response is not valid JSON")
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

    session = requests.Session()
    try:
        response = session.post(
            "http://localhost:8080/v1/completions",
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


def send_chat_completion_request(
    prompt: str, max_tokens: int, timeout: int = 120, stream: bool = False
) -> requests.Response:
    """Send a chat completion request to the frontend"""
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
    }

    headers = {"Content-Type": "application/json"}

    logger.info(
        f"Sending chat completion request (stream={stream}) with prompt: '{prompt[:50]}...' and max_tokens: {max_tokens}"
    )

    session = requests.Session()
    try:
        response = session.post(
            "http://localhost:8080/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
            stream=stream,
        )
        logger.info(f"Received response with status code: {response.status_code}")
        return response
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out after {timeout} seconds")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed with error: {e}")
        raise


def send_request_and_cancel(request_type: str = "completion"):
    """Send a request with short timeout to trigger cancellation"""
    logger.info(f"Sending {request_type} request to be cancelled...")

    prompt = "Tell me a very long and detailed story about the history of artificial intelligence, including all major milestones, researchers, and breakthroughs?"
    timeout = 1  # Short timeout to trigger cancellation
    try:
        if request_type == "completion":
            response = send_completion_request(prompt, 16000, timeout)
        elif request_type == "chat_completion":
            response = send_chat_completion_request(prompt, 16000, timeout, False)
        elif request_type == "chat_completion_stream":
            response = send_chat_completion_request(prompt, 16000, timeout, True)
            # Read a few responses and then disconnect
            if response.status_code == 200:
                itr_count, max_itr = 0, 5
                try:
                    for res in response.iter_lines():
                        logger.info(f"Received response {itr_count + 1}: {res[:50]}...")
                        itr_count += 1
                        if itr_count >= max_itr:
                            break
                        time.sleep(0.1)
                except Exception as e:
                    pytest.fail(f"Stream reading failed: {e}")

            response.close()
            raise Exception("Closed response")
        else:
            pytest.fail(f"Unknown request type: {request_type}")

        pytest.fail(
            f"{request_type} request completed unexpectedly - should have been cancelled"
        )
    except Exception as e:
        logger.info(f"{request_type} request was cancelled: {e}")


def read_log_content(log_path: str | None) -> str:
    """Read log content from a file"""
    if log_path is None:
        pytest.fail("Log path is None - cannot read log content")

    try:
        with open(log_path, "r") as f:
            return f.read()
    except Exception as e:
        pytest.fail(f"Could not read log file {log_path}: {e}")


def verify_request_cancelled(
    worker_process: DynamoWorkerProcess,
    frontend_process: DynamoFrontendProcess,
    worker_log_offset: int = 0,
    frontend_log_offset: int = 0,
) -> tuple[int, int]:
    """Verify that the worker and frontend logs contain cancellation messages

    Returns:
        tuple: (new_worker_log_length, new_frontend_log_length)
    """

    # Check worker log for cancellation pattern
    worker_log_content = read_log_content(worker_process._log_path)
    new_worker_content = worker_log_content[worker_log_offset:]
    has_worker_cancellation = (
        "finished processing python async generator stream" in new_worker_content
    )

    # Check frontend log for "issued control message Kill to sender" message
    frontend_log_content = read_log_content(frontend_process._log_path)
    new_frontend_content = frontend_log_content[frontend_log_offset:]
    has_kill_message = "issued control message Kill to sender" in new_frontend_content

    logger.info(f"Kill message found in frontend log: {has_kill_message}")
    logger.info(f"Worker cancellation pattern found: {has_worker_cancellation}")

    if not (has_kill_message and has_worker_cancellation):
        pytest.fail("Request cancellation was not detected in worker and frontend logs")

    return len(worker_log_content), len(frontend_log_content)


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.slow
def test_request_cancellation_vllm(request, runtime_services):
    """
    End-to-end test for request cancellation functionality.

    This test verifies that when a request is cancelled by the client,
    the system properly handles the cancellation and cleans up resources
    on the worker side. Tests three scenarios:
    1. Completion request
    2. Chat completion request (non-streaming)
    3. Chat completion request (streaming)
    """
    # Step 0: Download the model from HuggingFace if not already cached
    download_model()

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start a single worker
        logger.info("Starting worker...")
        worker = DynamoWorkerProcess(request)

        with worker:
            logger.info(f"Worker PID: {worker.get_pid()}")

            # TODO: Why the model is not immediately available at the frontend after health check
            #       returns success.
            time.sleep(2)

            # Step 3: Test request cancellation
            worker_log_offset, frontend_log_offset = 0, 0

            test_scenarios = [
                ("completion", "Completion request cancellation"),
                ("chat_completion", "Chat completion request cancellation"),
                (
                    "chat_completion_stream",
                    "Chat completion stream request cancellation",
                ),
            ]

            for i, (request_type, description) in enumerate(test_scenarios, 1):
                logger.info(f"Testing {description.lower()}...")
                send_request_and_cancel(request_type)

                logger.info(
                    "Checking for cancellation messages in worker and frontend logs..."
                )
                time.sleep(0.5)  # Make sure logs are written before proceeding
                worker_log_offset, frontend_log_offset = verify_request_cancelled(
                    worker, frontend, worker_log_offset, frontend_log_offset
                )

                logger.info(f"{description} detected successfully")

            logger.info(
                "All request cancellation tests completed successfully - request cancellation is working correctly"
            )
