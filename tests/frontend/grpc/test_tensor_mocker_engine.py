# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
import os
import shutil

import pytest
import triton_echo_client

from tests.conftest import EtcdServer, NatsServer
from tests.utils.constants import QWEN
from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)

TEST_MODEL = QWEN


class DynamoFrontendProcess(ManagedProcess):
    """Process manager for Dynamo frontend"""

    def __init__(self, request):
        command = ["python", "-m", "dynamo.frontend", "--kserve-grpc-server"]

        # Unset DYN_SYSTEM_PORT - frontend doesn't use system metrics server
        env = os.environ.copy()
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


class MockWorkerProcess(ManagedProcess):
    def __init__(self, request, worker_id: str = "mocker-worker"):
        self.worker_id = worker_id

        command = [
            "python3",
            os.path.join(os.path.dirname(__file__), "echo_tensor_worker.py"),
        ]

        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
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
                # gRPC doesn't expose endpoint for listing models, so skip this check
                # ("http://localhost:8000/v1/models", check_models_api),
                ("http://localhost:8083/health", self.is_ready),
            ],
            timeout=300,
            display_output=True,
            terminate_existing=False,
            stragglers=[],
            straggler_commands=["echo_tensor_worker.py"],
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


@pytest.fixture(scope="module")
def runtime_services(request):
    """Module-scoped runtime services for this test file."""
    with NatsServer(request) as nats_process:
        with EtcdServer(request) as etcd_process:
            yield nats_process, etcd_process


@pytest.fixture(scope="module")
def start_services(request, runtime_services):
    """Start frontend and worker processes once for this module's tests."""
    with DynamoFrontendProcess(request):
        logger.info("Frontend started for tests")
        with MockWorkerProcess(request):
            logger.info("Worker started for tests")
            yield


@pytest.mark.usefixtures("start_services")
@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.integration
@pytest.mark.model(TEST_MODEL)
def test_echo() -> None:
    triton_echo_client.check_health()
    triton_echo_client.run_infer()
    triton_echo_client.get_config()
