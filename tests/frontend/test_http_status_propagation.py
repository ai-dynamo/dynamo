# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end client-error propagation checks for a Python engine.

The tests launch a real worker/frontend pair and prove that typed HTTP and
SSRF policy failures survive the TCP/etcd request plane as 4xx responses.
The SSRF case also guards the network boundary with a connection canary.
"""

from __future__ import annotations

import os
from typing import Generator

import pytest
import requests

from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.network_canary import ConnectionCanary, running_canary
from tests.utils.port_utils import ServicePorts

MODEL_NAME = "test-http-status-prop"
ENDPOINT_PATH = "test.http_status_prop.generate"
EXPECTED_STATUS = 415
EXPECTED_MESSAGE = "unsupported-media-via-wire"

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
]


class _WorkerProcess(ManagedProcess):
    def __init__(self, request, *, frontend_port: int) -> None:
        env = os.environ.copy()
        env["DYN_MM_ALLOW_INTERNAL"] = "0"
        super().__init__(
            command=["python3", "-m", "tests.frontend.http_status_propagation_worker"],
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", self._model_listed)
            ],
            timeout=60,
            display_output=True,
            terminate_all_matching_process_names=False,
            straggler_commands=["-m tests.frontend.http_status_propagation_worker"],
            log_dir=f"{request.node.name}_worker",
            env=env,
        )

    @staticmethod
    def _model_listed(response: requests.Response) -> bool:
        try:
            if response.status_code != 200:
                return False
            data = response.json()
        except (ValueError, KeyError):
            return False
        return any(m.get("id") == MODEL_NAME for m in data.get("data", []))


@pytest.fixture(scope="function")
def services(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports: ServicePorts,
) -> Generator[int, None, None]:
    _ = runtime_services_dynamic_ports
    frontend_port = dynamo_dynamic_ports.frontend_port
    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_args=["--discovery-backend", "etcd", "--request-plane", "tcp"],
        extra_env={"DYN_MM_ALLOW_INTERNAL": "0"},
        terminate_all_matching_process_names=False,
    ):
        with _WorkerProcess(request, frontend_port=frontend_port):
            yield frontend_port


@pytest.fixture
def outbound_canary() -> Generator[ConnectionCanary, None, None]:
    with running_canary() as server:
        yield server


def test_http_status_propagates_through_wire(services: int) -> None:
    response = requests.post(
        f"http://localhost:{services}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hello"}],
        },
        timeout=30,
    )
    assert response.status_code == EXPECTED_STATUS, response.text
    assert EXPECTED_MESSAGE in response.text


@pytest.mark.timeout(120)
def test_python_backend_ssrf_rejection_is_4xx_with_zero_egress(
    services: int,
    outbound_canary: ConnectionCanary,
) -> None:
    """Public HTTP -> Python batch loader must reject before any connection."""
    response = requests.post(
        f"http://localhost:{services}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {"url": outbound_canary.blocked_url()},
                        },
                    ],
                }
            ],
        },
        timeout=30,
    )

    assert 400 <= response.status_code < 500, response.text
    assert "blocked range" in response.text
    outbound_canary.assert_no_connection()


@pytest.mark.timeout(30)
def test_outbound_canary_counts_a_real_connection(
    outbound_canary: ConnectionCanary,
) -> None:
    """Prove the shared canary records a real connection."""
    assert outbound_canary.connection_count == 0

    outbound_canary.touch()

    assert outbound_canary.await_connection() == 1
