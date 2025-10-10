# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests covering reasoning effort behaviour."""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, Sequence, Union

import pytest
import requests

from tests.utils.constants import GPT_OSS
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api
from tests.conftest import NatsServer, EtcdServer

logger = logging.getLogger(__name__)

REASONING_TEST_MODEL = GPT_OSS


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


class GPTOSSWorkerProcess(ManagedProcess):
    """Worker process for GPT-OSS model."""

    def __init__(self, request, worker_id: str = "reasoning-worker"):
        self.worker_id = worker_id

        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            REASONING_TEST_MODEL,
            "--enforce-eager",
            "--dyn-tool-call-parser",
            "harmony",
            "--dyn-reasoning-parser",
            "gpt_oss",
        ]

        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_ENABLED"] = "true"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = "8083"

        log_dir = f"{request.node.name}_{worker_id}"

        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                ("http://localhost:8000/v1/models", check_models_api),
                ("http://localhost:8083/health", self.is_ready),
            ],
            timeout=300,
            display_output=True,
            terminate_existing=False,
            stragglers=["VLLM::EngineCore"],
            straggler_commands=["-m dynamo.vllm"],
            log_dir=log_dir,
        )

    def is_ready(self, response) -> bool:
        try:
            status = (response.json() or {}).get("status")
        except ValueError:
            logger.warning("%s health response is not valid JSON", self.worker_id)
            return False

        is_ready = status == "ready"
        if is_ready:
            logger.info("%s status is ready", self.worker_id)
        else:
            logger.warning("%s status is not ready: %s", self.worker_id, status)
        return is_ready



def _send_completion_request(
    payload: Dict[str, Any],
    timeout: int = 180,
) -> requests.Response:
    """Send a text completion request"""

    headers = {"Content-Type": "application/json"}

    response = requests.post(
        "http://localhost:8000/v1/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    return response


@pytest.fixture(scope="module")
def runtime_services(request):
    """Module-scoped runtime services for this test file."""
    with NatsServer(request) as nats_process:
        with EtcdServer(request) as etcd_process:
            yield nats_process, etcd_process


@pytest.fixture(scope="module")
def start_services(request, runtime_services, predownload_models):
    """Start frontend and worker processes once for this module's tests."""
    with DynamoFrontendProcess(request):
        logger.info("Frontend started for tests")
        with GPTOSSWorkerProcess(request):
            logger.info("Worker started for tests")
            yield


@pytest.mark.usefixtures('start_services')
@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(REASONING_TEST_MODEL)
def test_completion_single_element_array_prompt() -> None:
    """Exercise completions with reasoning effort and prompt."""
    reasoning_effort = 'low'

    payload: Dict[str, Any] = {
        "model": REASONING_TEST_MODEL,
        "prompt": ["Tell me about Mars"],
        "max_tokens": 2000,
        "chat_template_args": {"reasoning_effort": reasoning_effort},
    }

    response = _send_completion_request(payload)

    assert response.status_code == 200, (
        f"Completion request ({reasoning_effort}) failed with status "
        f"{response.status_code}: {response.text}"
    )


@pytest.mark.usefixtures('start_services')
@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(REASONING_TEST_MODEL)
def test_completion_multi_string_prompt() -> None:
    """Exercise completions with reasoning effort and prompt."""
    reasoning_effort = 'low'

    payload: Dict[str, Any] = {
        "model": REASONING_TEST_MODEL,
        "prompt": "Tell me about Mars",
        "max_tokens": 2000,
        "chat_template_args": {"reasoning_effort": reasoning_effort},
    }

    response = _send_completion_request(payload)

    assert response.status_code == 200, (
        f"Completion request ({reasoning_effort}) failed with status "
        f"{response.status_code}: {response.text}"
    )

@pytest.mark.usefixtures('start_services')
@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.model(REASONING_TEST_MODEL)
def test_completion_multi_element_array_prompt() -> None:
    """Exercise completions with reasoning effort and prompt."""
    reasoning_effort = 'low'

    payload: Dict[str, Any] = {
        "model": REASONING_TEST_MODEL,
        "prompt": ["Tell me about Mars", "Tell me about Ceres"],
        "max_tokens": 2000,
        "chat_template_args": {"reasoning_effort": reasoning_effort},
    }

    response = _send_completion_request(payload)

    # request should fail because we are sending multiple prompts
    assert response.status_code == 500, (
        f"Completion request ({reasoning_effort}) failed with status "
        f"{response.status_code}: {response.text}"
    )
