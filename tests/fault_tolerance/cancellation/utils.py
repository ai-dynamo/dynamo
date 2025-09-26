# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
import shutil
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


def send_completion_request(
    prompt: str, max_tokens: int, timeout: int | float = 120
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
    prompt: str, max_tokens: int, timeout: int | float = 120, stream: bool = False
) -> requests.Response:
    """Send a chat completion request to the frontend"""
    payload = {
        "model": FAULT_TOLERANCE_MODEL_NAME,
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
    request_type: str = "completion",
    timeout: int | float = 1,
    use_long_prompt: bool = False,
):
    """Send a request with short timeout to trigger cancellation"""
    logger.info(f"Sending {request_type} request to be cancelled...")

    prompt = "Tell me a very long and detailed story about the history of artificial intelligence, including all major milestones, researchers, and breakthroughs?"
    if use_long_prompt:
        prompt += " Make sure it is" + " long" * 8000 + "!"

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
