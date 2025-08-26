# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
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

    def __init__(self, request, is_prefill: bool = False):
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

        # Add prefill worker flag if needed
        if is_prefill:
            command.append("--is-prefill-worker")

        # Set port based on worker type
        port = "8082" if is_prefill else "8081"

        # Set debug logging environment
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_ENABLED"] = "true"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = port

        # Set log directory based on worker type
        worker_type = "prefill_worker" if is_prefill else "worker"
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
            health_check_urls=[(f"http://localhost:{port}/health", self.is_ready)],
            timeout=300,
            display_output=True,
            terminate_existing=False,
            log_dir=log_dir,
        )

        self.is_prefill = is_prefill

    def get_pid(self):
        """Get the PID of the worker process"""
        return self.proc.pid if self.proc else None

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


def send_request_and_cancel(request_type: str = "completion", timeout: int = 1):
    """Send a request with short timeout to trigger cancellation"""
    logger.info(f"Sending {request_type} request to be cancelled...")

    prompt = "Tell me a very long and detailed story about the history of artificial intelligence, including all major milestones, researchers, and breakthroughs?"
    try:
        if request_type == "completion":
            response = send_completion_request(prompt, 8000, timeout)
        elif request_type == "chat_completion":
            response = send_chat_completion_request(prompt, 8000, timeout, False)
        elif request_type == "chat_completion_stream":
            response = send_chat_completion_request(prompt, 8000, timeout, True)
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


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes from text"""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def verify_request_cancelled(
    frontend_process: DynamoFrontendProcess,
    worker_process: DynamoWorkerProcess,
    prefill_worker_process: DynamoWorkerProcess | None = None,
    frontend_log_offset: int = 0,
    worker_log_offset: int = 0,
    prefill_worker_log_offset: int = 0,
) -> tuple[int, int]:
    """Verify that the worker and frontend logs contain cancellation messages

    Returns:
        tuple: (new_worker_log_length, new_frontend_log_length)
    """

    # Check worker log for cancellation pattern
    worker_log_content = read_log_content(worker_process._log_path)
    new_worker_content = worker_log_content[worker_log_offset:]

    # Find request ID from "New Request ID: <id>" line
    request_id = None
    for line in new_worker_content.split("\n"):
        # Strip ANSI codes and whitespace for pattern matching
        clean_line = strip_ansi_codes(line).strip()
        if "New Request ID: " in clean_line:
            # Extract ID from the end of the line
            parts = clean_line.split("New Request ID: ")
            if len(parts) > 1:
                request_id = parts[-1].strip()
                break
    if request_id is None:
        pytest.fail("Could not find 'New Request ID: <id>' pattern in worker log")

    # Check if the same request ID was cancelled
    has_worker_cancellation = False
    cancellation_pattern = f"Aborted Request ID: {request_id}"
    for line in new_worker_content.split("\n"):
        # Strip ANSI codes and whitespace for pattern matching
        clean_line = strip_ansi_codes(line).strip()
        if clean_line.endswith(cancellation_pattern):
            has_worker_cancellation = True
            break
    if not has_worker_cancellation:
        pytest.fail(
            f"Could not find 'Aborted Request ID: {request_id}' pattern in worker log"
        )

    # Check if the same request ID was remote prefilled
    if prefill_worker_process is not None:
        prefill_worker_log_content = read_log_content(prefill_worker_process._log_path)
        new_prefill_worker_content = prefill_worker_log_content[
            prefill_worker_log_offset:
        ]

        has_remote_prefill = False
        remote_prefill_pattern = f"New Prefill Request ID: {request_id}"
        for line in new_prefill_worker_content.split("\n"):
            clean_line = strip_ansi_codes(line).strip()
            if clean_line.endswith(remote_prefill_pattern):
                has_remote_prefill = True
                break
        if not has_remote_prefill:
            pytest.fail(
                f"Could not find 'New Prefill Request ID: {request_id}' pattern in prefill worker log"
            )

    # Check frontend log for cancellation issued pattern
    frontend_log_content = read_log_content(frontend_process._log_path)
    new_frontend_content = frontend_log_content[frontend_log_offset:]

    has_kill_message = False
    kill_message = "issued control message Kill to sender"
    for line in new_frontend_content.split("\n"):
        # Strip ANSI codes and whitespace for pattern matching
        clean_line = strip_ansi_codes(line).strip()
        if clean_line.endswith(kill_message):
            has_kill_message = True
            break
    if not has_kill_message:
        pytest.fail("Could not find cancellation issued in frontend log")

    return len(frontend_log_content), len(worker_log_content)


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
            frontend_log_offset, worker_log_offset = 0, 0

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
                frontend_log_offset, worker_log_offset = verify_request_cancelled(
                    frontend,
                    worker,
                    frontend_log_offset=frontend_log_offset,
                    worker_log_offset=worker_log_offset,
                )

                logger.info(f"{description} detected successfully")

            logger.info(
                "All request cancellation tests completed successfully - request cancellation is working correctly"
            )


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.slow
def test_request_cancellation_vllm_decode(request, runtime_services):
    """
    End-to-end test for request cancellation functionality with remote prefill.

    This test verifies that when a request is cancelled by the client,
    the system properly handles the cancellation and cleans up resources
    on the decode worker side in a disaggregated setup.
    """
    # Step 0: Download the model from HuggingFace if not already cached
    download_model()

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start the prefill worker
        logger.info("Starting prefill worker...")
        prefill_worker = DynamoWorkerProcess(request, is_prefill=True)

        with prefill_worker:
            logger.info(f"Prefill Worker PID: {prefill_worker.get_pid()}")

            # Step 3: Start the decode worker
            logger.info("Starting decode worker...")
            decode_worker = DynamoWorkerProcess(request, is_prefill=False)

            with decode_worker:
                logger.info(f"Decode Worker PID: {decode_worker.get_pid()}")

                # TODO: Why the model is not immediately available at the frontend after health check
                #       returns success.
                time.sleep(2)

                # Step 4: Test request cancellation for completion scenario only
                logger.info(
                    "Testing completion request cancellation in disaggregated mode..."
                )
                send_request_and_cancel("completion")

                logger.info(
                    "Checking for cancellation messages in decode worker, prefill worker, and frontend logs..."
                )
                time.sleep(0.5)  # Make sure logs are written before proceeding
                verify_request_cancelled(frontend, decode_worker, prefill_worker)

                logger.info(
                    "Completion request cancellation detected successfully in disaggregated mode"
                )

                logger.info(
                    "Request cancellation test completed successfully in disaggregated mode - request cancellation is working correctly"
                )


@pytest.mark.skip(reason="require cancel support before receiving 1st response")
@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.slow
def test_request_cancellation_vllm_prefill(request, runtime_services):
    """
    End-to-end test for request cancellation on remote prefill.

    This test verifies that when a request is cancelled by the client during the
    prefill phase, the system properly handles the cancellation and cleans up
    resources on the prefill worker and decode worker sides in a disaggregated
    setup.
    """
