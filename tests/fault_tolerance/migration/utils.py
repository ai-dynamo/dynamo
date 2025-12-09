# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import threading
import time

import pytest
import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.engine_process import FRONTEND_PORT
from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)


class DynamoFrontendProcess(ManagedProcess):
    """Process manager for Dynamo frontend"""

    def __init__(self, request):
        command = ["python", "-m", "dynamo.frontend", "--router-mode", "round-robin"]

        # Unset DYN_SYSTEM_PORT - frontend doesn't use system metrics server
        env = os.environ.copy()
        # Disable canary health check - these tests expect full control over requests
        # sent to the workers where canary health check intermittently sends dummy
        # requests to workers interfering with the test process which may cause
        # intermittent failures
        env["DYN_HEALTH_CHECK_ENABLED"] = "false"
        env.pop("DYN_SYSTEM_PORT", None)

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


def start_completion_request() -> tuple:
    """
    Start a long-running completion request in a separate thread.

    Returns:
        tuple: (request_thread, response_list)
    """
    response_list = []  # Thread safe is not required as only one thread writes to it

    def send_request():
        prompt = "Tell me a long long long story about yourself?"
        max_tokens = 8000
        timeout = 240  # Extended timeout for long request

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
                f"http://localhost:{FRONTEND_PORT}/v1/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            logger.info(f"Received response with status code: {response.status_code}")
            response_list.append(response)
        except Exception as e:
            logger.error(f"Request failed with error: {e}")

    request_thread = threading.Thread(target=send_request, daemon=True)
    request_thread.start()

    return request_thread, response_list


def determine_request_receiving_worker(
    worker1: ManagedProcess, worker2: ManagedProcess, receiving_pattern: str
) -> tuple:
    """
    Determine which worker received the request using parallel polling.

    Args:
        worker1: First worker process
        worker2: Second worker process
        receiving_pattern: Log pattern indicating request receipt

    Returns:
        Tuple of (worker_with_request, name_of_worker_with_request)
    """
    worker1_results: list[bool] = []
    worker2_results: list[bool] = []

    # Poll both workers in parallel
    def poll_worker(worker: ManagedProcess, result_list: list[bool]):
        max_wait_ms = 500
        poll_interval_ms = 5
        max_iterations = max_wait_ms // poll_interval_ms
        iteration = 0

        while iteration < max_iterations:
            # Check if the worker logs contain 'New Request ID:' message
            try:
                with open(worker.log_path, "r") as f:
                    log_content = f.read()
                    if receiving_pattern in log_content:
                        result_list.append(True)
                        return
            except Exception as e:
                logger.error(f"Could not read log file {worker.log_path}: {e}")
                return

            time.sleep(poll_interval_ms / 1000.0)
            iteration += 1

    # Look for which worker received the request
    thread1 = threading.Thread(
        target=poll_worker, args=(worker1, worker1_results), daemon=True
    )
    thread2 = threading.Thread(
        target=poll_worker, args=(worker2, worker2_results), daemon=True
    )
    thread1.start()
    thread2.start()
    thread1.join(timeout=1)
    thread2.join(timeout=1)

    # Get results from lists
    worker1_received = worker1_results[0] if worker1_results else False
    worker2_received = worker2_results[0] if worker2_results else False

    if worker1_received and not worker2_received:
        logger.info("Request was received by Worker 1")
        return worker1, "Worker 1"
    elif worker2_received and not worker1_received:
        logger.info("Request was received by Worker 2")
        return worker2, "Worker 2"
    elif worker1_received and worker2_received:
        pytest.fail("Both workers received the request")
    else:
        pytest.fail("Neither worker received the request")


def validate_completion_response(
    request_thread: threading.Thread, response_list: list
) -> None:
    """
    Wait for and validate the completion response after worker failure.

    Args:
        request_thread: The thread running the completion request
        response_list: List containing the response from the request
    """
    request_thread.join(timeout=240)
    if request_thread.is_alive():
        pytest.fail("Request did not complete within 240 seconds")

    # Get the response
    if len(response_list) != 1:
        pytest.fail(f"Received {len(response_list)} responses, expected 1")
    response = response_list[0]

    assert (
        response.status_code == 200
    ), f"Request failed with status {response.status_code}: {response.text}"

    try:
        data = response.json()
    except ValueError:
        pytest.fail(f"Response is not valid JSON: {response.text}")

    # Validate OpenAI completion response structure
    assert "choices" in data, f"Response missing 'choices' field: {data}"
    assert len(data["choices"]) > 0, f"Response has empty 'choices': {data}"
    assert "text" in data["choices"][0], f"Response choice missing 'text' field: {data}"
    assert data["choices"][0]["text"], f"Response text is empty: {data}"

    logger.info(
        f"Received valid completion response: {data['choices'][0]['text'][:100]}..."
    )
    logger.info("Request completed successfully")


def verify_migration_occurred(frontend_process: DynamoFrontendProcess) -> None:
    """
    Verify that migration occurred by checking frontend logs for stream disconnection message.

    Args:
        frontend_process: The frontend process to check logs for
    """
    log_path = frontend_process.log_path
    try:
        with open(log_path, "r") as f:
            log_content = f.read()
    except Exception as e:
        pytest.fail(f"Could not read frontend log file {log_path}: {e}")
    assert (
        "Stream disconnected... recreating stream..." in log_content
    ), "'Stream disconnected... recreating stream...' message not found in logs"
    assert (
        "Cannot recreate stream: " not in log_content
    ), "'Cannot recreate stream: ...' error found in logs"
