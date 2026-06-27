# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end check that an HTTP status raised inside a Python engine
propagates through the wire transport to the frontend's HTTP response.

Launches a real worker + frontend subprocess pair, posts one
chat-completions request, asserts the response is 415 — proving the
boundary fix survives the TCP/etcd request plane, not just the
in-process pipeline."""

from __future__ import annotations

from typing import Generator

import pytest
import requests

from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
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
        terminate_all_matching_process_names=False,
    ):
        with _WorkerProcess(request, frontend_port=frontend_port):
            yield frontend_port


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
