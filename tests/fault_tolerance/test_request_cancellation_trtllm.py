# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field

import pytest
import requests

from tests.utils.engine_process import FRONTEND_PORT, EngineProcess
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_health_generate, check_models_api

logger = logging.getLogger(__name__)


class DynamoFrontendProcess(ManagedProcess):
    """Process manager for Dynamo frontend"""

    def __init__(self, request):
        command = ["python", "-m", "dynamo.frontend", f"--http-port={FRONTEND_PORT}"]

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


@dataclass
class TRTLLMConfig:
    """Configuration for TRTLLM cancellation test scenarios"""

    name: str
    directory: str
    script_name: str
    model: str
    marks: list = field(default_factory=list)
    endpoints: list = field(default_factory=list)
    response_handlers: list = field(default_factory=list)
    env: dict = field(default_factory=dict)
    timeout: int = 300
    delayed_start: int = 0


class TRTLLMProcess(EngineProcess):
    """Process manager for TRTLLM backend cancellation tests"""

    def __init__(self, config: TRTLLMConfig, request):
        self.port = FRONTEND_PORT
        self.config = config
        self.dir = config.directory
        script_path = os.path.join(self.dir, "launch", config.script_name)

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"TRTLLM script not found: {script_path}")

        # Set environment variables for the TRTLLM backend
        env = os.environ.copy()
        env.update(
            {
                "MODEL_PATH": config.model,
                "SERVED_MODEL_NAME": config.model,
                "DYN_LOG": "debug",
            }
        )
        env.update(config.env)

        command = ["bash", script_path]

        super().__init__(
            command=command,
            timeout=config.timeout,
            display_output=True,
            working_dir=self.dir,
            health_check_urls=[
                (f"http://localhost:{self.port}/v1/models", check_models_api),
                (f"http://localhost:{self.port}/health", check_health_generate),
            ],
            delayed_start=config.delayed_start,
            terminate_existing=False,
            stragglers=["TRTLLM:EngineCore"],
            log_dir=request.node.name,
            env=env,
        )


def send_completion_request(
    prompt: str, max_tokens: int, timeout: int = 120, model: str = "Qwen/Qwen3-0.6B"
) -> requests.Response:
    """Send a completion request to the frontend"""
    payload = {
        "model": model,
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
            f"http://localhost:{FRONTEND_PORT}/v1/completions",
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
    prompt: str,
    max_tokens: int,
    timeout: int = 120,
    stream: bool = False,
    model: str = "Qwen/Qwen3-0.6B",
) -> requests.Response:
    """Send a chat completion request to the frontend"""
    payload = {
        "model": model,
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
            f"http://localhost:{FRONTEND_PORT}/v1/chat/completions",
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


def send_request_and_cancel(
    request_type: str = "completion", timeout: int = 1, model: str = "Qwen/Qwen3-0.6B"
):
    """Send a request with short timeout to trigger cancellation"""
    logger.info(f"Sending {request_type} request to be cancelled...")

    prompt = "Tell me a very long and detailed story about the history of artificial intelligence, including all major milestones, researchers, and breakthroughs?"
    try:
        if request_type == "completion":
            response = send_completion_request(prompt, 8000, timeout, model)
        elif request_type == "chat_completion":
            response = send_chat_completion_request(prompt, 8000, timeout, False, model)
        elif request_type == "chat_completion_stream":
            response = send_chat_completion_request(prompt, 8000, timeout, True, model)
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


def verify_request_cancelled_trtllm_simple(
    trtllm_process: TRTLLMProcess,
    trtllm_log_offset: int = 0,
) -> int:
    """Simplified verification for TRTLLM cancellation messages

    Returns:
        int: new_trtllm_log_length
    """

    # Check TRTLLM log for cancellation pattern
    trtllm_log_content = read_log_content(trtllm_process._log_path)
    new_trtllm_content = trtllm_log_content[trtllm_log_offset:]

    # Look for TRTLLM-specific cancellation pattern
    has_trtllm_cancellation = False
    cancellation_pattern = "Aborted Request ID: "

    for line in new_trtllm_content.split("\n"):
        clean_line = strip_ansi_codes(line).strip()
        if cancellation_pattern in clean_line:
            has_trtllm_cancellation = True
            logger.info(f"Found TRTLLM cancellation message: {clean_line}")
            break

    if not has_trtllm_cancellation:
        pytest.fail(f"Could not find '{cancellation_pattern}' in TRTLLM backend log")

    return len(trtllm_log_content)


@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.trtllm_marker
@pytest.mark.slow
def test_request_cancellation_trtllm_aggregated(request, runtime_services):
    """
    End-to-end test for request cancellation functionality with TRTLLM aggregated backend.

    This test verifies that when a request is cancelled by the client,
    the system properly handles the cancellation and cleans up resources
    on the TRTLLM backend side. Tests three scenarios:
    1. Completion request
    2. Chat completion request (non-streaming)
    3. Chat completion request (streaming)
    """

    trtllm_dir = os.environ.get("TRTLLM_DIR", "/workspace/components/backends/trtllm")

    # Configuration for TRTLLM aggregated cancellation test
    trtllm_config = TRTLLMConfig(
        name="aggregated_cancellation_test",
        directory=trtllm_dir,
        script_name="agg.sh",
        model="Qwen/Qwen3-0.6B",
    )

    # Start TRTLLM aggregated backend
    logger.info("Starting TRTLLM aggregated backend...")
    with TRTLLMProcess(trtllm_config, request) as trtllm_process:
        logger.info("TRTLLM aggregated backend started successfully")

        # Test request cancellation scenarios
        test_scenarios = [
            ("completion", "Completion request cancellation"),
            ("chat_completion", "Chat completion request cancellation"),
            (
                "chat_completion_stream",
                "Chat completion streaming request cancellation",
            ),
        ]

        trtllm_log_offset = 0

        for i, (request_type, description) in enumerate(test_scenarios, 1):
            logger.info(f"Testing {description.lower()} with TRTLLM...")
            send_request_and_cancel(request_type, timeout=1, model="Qwen/Qwen3-0.6B")

            logger.info("Checking for cancellation messages in TRTLLM logs...")
            time.sleep(1)  # Give more time for TRTLLM logs to be written
            trtllm_log_offset = verify_request_cancelled_trtllm_simple(
                trtllm_process,
                trtllm_log_offset=trtllm_log_offset,
            )

            logger.info(f"{description} with TRTLLM detected successfully")

        logger.info("All TRTLLM request cancellation tests completed successfully")


@pytest.mark.e2e
@pytest.mark.gpu_2
@pytest.mark.trtllm_marker
@pytest.mark.slow
def test_request_cancellation_trtllm_disaggregated(request, runtime_services):
    """
    End-to-end test for request cancellation functionality with TRTLLM disaggregated backend.

    This test verifies that when a request is cancelled by the client,
    the system properly handles the cancellation and cleans up resources
    on the TRTLLM disaggregated backend side. Tests three scenarios:
    1. Completion request
    2. Chat completion request (non-streaming)
    3. Chat completion request (streaming)
    """

    trtllm_dir = os.environ.get("TRTLLM_DIR", "/workspace/components/backends/trtllm")

    # Configuration for TRTLLM disaggregated cancellation test
    trtllm_config = TRTLLMConfig(
        name="disaggregated_cancellation_test",
        directory=trtllm_dir,
        script_name="disagg.sh",
        model="Qwen/Qwen3-0.6B",
    )

    # Start TRTLLM disaggregated backend
    logger.info("Starting TRTLLM disaggregated backend...")
    with TRTLLMProcess(trtllm_config, request) as trtllm_process:
        logger.info("TRTLLM disaggregated backend started successfully")

        # Test request cancellation scenarios
        test_scenarios = [
            ("completion", "Completion request cancellation"),
            ("chat_completion", "Chat completion request cancellation"),
            (
                "chat_completion_stream",
                "Chat completion streaming request cancellation",
            ),
        ]

        trtllm_log_offset = 0

        for i, (request_type, description) in enumerate(test_scenarios, 1):
            logger.info(f"Testing {description.lower()} with TRTLLM disaggregated...")
            send_request_and_cancel(request_type, timeout=1, model="Qwen/Qwen3-0.6B")

            logger.info("Checking for cancellation messages in TRTLLM logs...")
            time.sleep(1)  # Give more time for TRTLLM logs to be written
            trtllm_log_offset = verify_request_cancelled_trtllm_simple(
                trtllm_process,
                trtllm_log_offset=trtllm_log_offset,
            )

            logger.info(
                f"{description} with TRTLLM disaggregated detected successfully"
            )

        logger.info(
            "All TRTLLM disaggregated request cancellation tests completed successfully"
        )
