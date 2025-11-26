# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Frontend completion tests with mock backend engine.

Parallel Execution:
-------------------
Install pytest-xdist for parallel test execution:
    pip install pytest-xdist

Most tests MUST run serially (default - no marker needed).
Tests that are safe to run in parallel should use @pytest.mark.parallel.

Usage Examples:
    # Run all tests serially (safe default)
    pytest tests/frontend/test_completion_mocker_engine.py -v

    # Run only serial tests (exclude parallel)
    pytest tests/frontend/test_completion_mocker_engine.py -v -m "not parallel"

    # Run only parallel tests serially
    pytest tests/frontend/test_completion_mocker_engine.py -v -m parallel

    # Run only parallel tests in parallel with 2 workers
    pytest tests/frontend/test_completion_mocker_engine.py -v -n 2 -m parallel

    # Run only parallel tests in parallel with auto workers
    pytest tests/frontend/test_completion_mocker_engine.py -v -n auto -m parallel

Marker Examples:
    # Serial test (default - most tests, no marker needed)
    @pytest.mark.e2e
    def test_needs_serial():
        pass

    # Parallel test (uses dynamic ports, isolated fixtures)
    @pytest.mark.e2e
    @pytest.mark.parallel
    def test_can_run_parallel():
        pass

Performance Comparison (32-core machine, parallel tests only):
    Serial:              101.24s (1:41)
    Parallel (-n 2):     51.14s  (50% faster)
    Parallel (-n auto):  27.37s  (73% faster, 24 workers)
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from typing import Any, Dict

import pytest
import requests

from tests.utils.constants import QWEN
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api

logger = logging.getLogger(__name__)

TEST_MODEL = QWEN

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu_1,
    pytest.mark.post_merge,
    pytest.mark.parallel,
    pytest.mark.model(TEST_MODEL),
]


class DynamoFrontendProcess(ManagedProcess):
    """Process manager for Dynamo frontend"""

    def __init__(self, request, nats_port, etcd_port, http_port):
        self.http_port = http_port

        command = ["python", "-m", "dynamo.frontend", "--router-mode", "round-robin"]

        # Unset DYN_SYSTEM_PORT - frontend doesn't use system metrics server
        env = os.environ.copy()
        env.pop("DYN_SYSTEM_PORT", None)

        # pytest-xdist safe: workers are separate processes with isolated os.environ.
        env["DYN_HTTP_PORT"] = str(http_port)
        env["NATS_SERVER"] = f"nats://localhost:{nats_port}"
        env["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_port}"

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
            terminate_existing=False,  # Disabled for parallel test execution
            log_dir=log_dir,
        )


class MockWorkerProcess(ManagedProcess):
    def __init__(
        self,
        request,
        nats_port,
        etcd_port,
        frontend_http_port,
        system_port,
        worker_id: str = "mocker-worker",
    ):
        self.worker_id = worker_id
        self.system_port = system_port

        command = [
            "python3",
            "-m",
            "dynamo.mocker",
            "--model-path",
            TEST_MODEL,
            "--speedup-ratio",
            "100",
        ]

        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'

        # pytest-xdist safe: workers are separate processes with isolated os.environ.
        env["DYN_SYSTEM_PORT"] = str(system_port)
        env["NATS_SERVER"] = f"nats://localhost:{nats_port}"
        env["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_port}"

        log_dir = f"{request.node.name}_{worker_id}"

        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (f"http://localhost:{frontend_http_port}/v1/models", check_models_api),
                (f"http://localhost:{system_port}/health", self.is_ready),
            ],
            timeout=300,
            display_output=True,
            terminate_existing=False,
            stragglers=["VLLM::EngineCore"],
            straggler_commands=["-m dynamo.mocker"],
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
    frontend_http_port: int,
    timeout: int = 180,
) -> requests.Response:
    """Send a text completion request"""

    headers = {"Content-Type": "application/json"}
    print(f"Sending request: {time.time()}")

    response = requests.post(
        f"http://localhost:{frontend_http_port}/v1/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    return response


@pytest.fixture(scope="function")
def start_services(request, runtime_services):
    """Start frontend and worker processes for each test (for parallel execution)."""
    from tests.utils.port_utils import allocate_free_port, free_ports

    nats_process, etcd_process = runtime_services

    # Allocate ports for HTTP and system servers
    # Frontend starts at 8100, worker/backend starts at 9100
    http_port = allocate_free_port(8100)
    system_port = allocate_free_port(9100)
    logger.info(f"Allocated ports - HTTP: {http_port}, System: {system_port}")

    try:
        with DynamoFrontendProcess(
            request, nats_process.port, etcd_process.port, http_port
        ) as frontend:
            logger.info(f"Frontend started on port {http_port}")
            with MockWorkerProcess(
                request, nats_process.port, etcd_process.port, http_port, system_port
            ):
                logger.info(f"Worker started with system port {system_port}")
                yield frontend
    finally:
        # Release ports when test completes
        free_ports([http_port, system_port])
        logger.info(f"Released ports - HTTP: {http_port}, System: {system_port}")


@pytest.mark.usefixtures("start_services")
def test_completion_string_prompt(start_services) -> None:
    payload: Dict[str, Any] = {
        "model": TEST_MODEL,
        "prompt": "Tell me about Mars",
        "max_tokens": 2000,
    }

    response = _send_completion_request(payload, start_services.http_port)

    assert response.status_code == 200, (
        f"Completion request failed with status "
        f"{response.status_code}: {response.text}"
    )


@pytest.mark.usefixtures("start_services")
def test_completion_empty_array_prompt(start_services) -> None:
    payload: Dict[str, Any] = {
        "model": TEST_MODEL,
        "prompt": [],
        "max_tokens": 2000,
    }

    response = _send_completion_request(payload, start_services.http_port)

    assert response.status_code == 400, (
        f"Completion request should failed with status 400 but got"
        f"{response.status_code}: {response.text}"
    )


@pytest.mark.usefixtures("start_services")
def test_completion_single_element_array_prompt(start_services) -> None:
    payload: Dict[str, Any] = {
        "model": TEST_MODEL,
        "prompt": ["Tell me about Mars"],
        "max_tokens": 2000,
    }

    response = _send_completion_request(payload, start_services.http_port)

    assert response.status_code == 200, (
        f"Completion request failed with status "
        f"{response.status_code}: {response.text}"
    )


@pytest.mark.usefixtures("start_services")
def test_completion_multi_element_array_prompt(start_services) -> None:
    payload: Dict[str, Any] = {
        "model": TEST_MODEL,
        "prompt": [
            "Tell me about Mars",
            "Tell me about Ceres",
            "Tell me about Jupiter",
        ],
        "max_tokens": 300,
    }

    response = _send_completion_request(payload, start_services.http_port)
    response_data = response.json()

    assert response.status_code == 200, (
        f"Completion request failed with status "
        f"{response.status_code}: {response.text}"
    )

    expected_choices = len(payload.get("prompt"))  # type: ignore
    choices = len(response_data.get("choices", []))

    assert (
        expected_choices == choices
    ), f"Expected {expected_choices} choices, got {choices}"
