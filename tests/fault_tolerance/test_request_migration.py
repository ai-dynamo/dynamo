# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import queue
import shutil
import threading
import time

import pytest
import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.engine_process import FRONTEND_PORT
from tests.utils.managed_process import ManagedProcess, terminate_process_tree
from tests.utils.payloads import check_models_api

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
        env["DYN_SYSTEM_PORT"] = f"808{worker_id[-1]}"

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
                (f"http://localhost:808{worker_id[-1]}/health", self.is_ready),
            ],
            timeout=300,
            display_output=True,
            terminate_existing=False,
            stragglers=["VLLM::EngineCore"],
            straggler_commands=["-m dynamo.vllm"],
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
                logger.info(f"{self.worker_id} status is ready")
                return True
            logger.warning(
                f"{self.worker_id} status is not ready: {data.get('status')}"
            )
        except ValueError:
            logger.warning(f"{self.worker_id} health response is not valid JSON")
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


def validate_openai_response(response: requests.Response) -> None:
    """Validate that the response is a proper OpenAI completion response"""
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


def check_worker_received_request(worker_process: DynamoWorkerProcess) -> bool:
    """Check if the worker logs contain 'New Request ID:' message indicating it received a request"""
    log_path = worker_process._log_path
    if log_path and os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                log_content = f.read()
                return "New Request ID: " in log_content
        except Exception as e:
            logger.warning(f"Could not read worker log file {log_path}: {e}")
    return False


def determine_worker_roles(worker1: DynamoWorkerProcess, worker2: DynamoWorkerProcess):
    """Determine primary and backup workers based on which worker handled the test request"""
    worker1_received_test = check_worker_received_request(worker1)
    worker2_received_test = check_worker_received_request(worker2)

    if worker1_received_test and not worker2_received_test:
        primary_worker = (worker2, "Worker 2")
        backup_worker = (worker1, "Worker 1")
        logger.info("Test request was handled by Worker 1")
        return primary_worker, backup_worker
    elif worker2_received_test and not worker1_received_test:
        primary_worker = (worker1, "Worker 1")
        backup_worker = (worker2, "Worker 2")
        logger.info("Test request was handled by Worker 2")
        return primary_worker, backup_worker
    else:
        pytest.fail(
            f"Could not determine which worker handled the test request. Worker1: {worker1_received_test}, Worker2: {worker2_received_test}"
        )


def start_completion_request(primary_worker_name: str):
    """
    Start a request in a separate thread.

    Args:
        primary_worker_name: Name of the primary worker expected to handle the request

    Returns:
        tuple: (request_thread, response_queue)
    """
    response_queue: queue.Queue[requests.Response] = queue.Queue()

    def send_formal_request():
        response = send_completion_request(
            "Tell me a long long long story about yourself?",
            8000,
            timeout=240,  # Extended timeout for long request
        )
        response_queue.put(response)

    request_thread = threading.Thread(target=send_formal_request)
    request_thread.start()

    return request_thread, response_queue


def validate_completion_response(
    request_thread: threading.Thread, response_queue: queue.Queue
):
    """
    Wait for and validate the completion response after worker failure.

    Args:
        request_thread: The thread running the completion request
        response_queue: Queue containing the response from the request
    """
    request_thread.join(timeout=300)
    if request_thread.is_alive():
        pytest.fail("Request did not complete within timeout")

    # Get the response
    if response_queue.empty():
        pytest.fail("No response received for request")
    response = response_queue.get()

    # Validate the response
    validate_openai_response(response)
    logger.info("✓ Request completed successfully after worker failure")


def verify_migration_occurred(frontend_process: DynamoFrontendProcess) -> None:
    """
    Verify that migration occurred by checking frontend logs for stream disconnection message.

    Args:
        frontend_process: The frontend process to check logs for

    Raises:
        pytest.fail: If migration message is not found in logs
    """
    log_path = frontend_process._log_path
    if not log_path or not os.path.exists(log_path):
        pytest.fail(f"Frontend log file not found at {log_path}")

    try:
        with open(log_path, "r") as f:
            log_content = f.read()
            if "Stream disconnected... recreating stream..." in log_content:
                logger.info(
                    "✓ Migration detected: Found migration message in frontend logs"
                )
                return
            else:
                pytest.fail(
                    "Expected migration did not occur - migration message not found in frontend logs"
                )
    except Exception as e:
        pytest.fail(f"Could not read frontend log file {log_path}: {e}")


@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
def test_request_migration_vllm(
    request, runtime_services, predownload_models, set_ucx_tls_no_mm
):
    """
    End-to-end test for worker fault tolerance with migration support.

    This test verifies that when a worker is killed during request processing,
    the system can handle the failure gracefully and migrate the request to
    another worker.
    """

    # Step 1: Start the frontend
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Step 2: Start 2 workers sequentially

        # Start worker1 first and wait for it to be ready
        logger.info("Starting worker 1...")
        worker1 = DynamoWorkerProcess(request, "worker1")

        with worker1:
            # Start worker2 after worker1 is ready
            logger.info("Starting worker 2...")
            worker2 = DynamoWorkerProcess(request, "worker2")

            with worker2:
                logger.info(f"Worker 1 PID: {worker1.get_pid()}")
                logger.info(f"Worker 2 PID: {worker2.get_pid()}")

                # Step 3: Send a test request to see which worker handles it
                logger.info("Sending test request to determine worker assignment...")
                test_response = send_completion_request("Who are you?", 100, timeout=60)
                validate_openai_response(test_response)
                logger.info("Test request completed successfully")

                # Step 4: Determine worker roles based on test request handling
                # Frontend must use round-robin for the detection to work correctly
                primary_worker, backup_worker = determine_worker_roles(worker1, worker2)

                # Step 5: Send the formal request (expected to be received by the primary worker)
                logger.info(
                    f"Sending formal request - expected to be handled by {primary_worker[1]}"
                )
                request_thread, response_queue = start_completion_request(
                    primary_worker[1]
                )

                # Step 6: Wait 0.5 seconds after sending the formal request, then kill the primary worker
                logger.info(
                    f"Killing {primary_worker[1]} with PID {primary_worker[0].get_pid()}"
                )
                time.sleep(0.5)
                terminate_process_tree(
                    primary_worker[0].get_pid(), immediate_kill=True, timeout=0
                )

                # Step 7: Validate the completion response
                logger.info("Waiting for formal request to complete")
                validate_completion_response(request_thread, response_queue)

                # Step 8: Verify migration occurred
                logger.info("Checking for migration message in frontend logs")
                verify_migration_occurred(frontend)

                logger.info(
                    "Test completed successfully - migration is detected and the request was successful"
                )
