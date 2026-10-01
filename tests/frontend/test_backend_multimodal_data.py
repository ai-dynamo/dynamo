# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end transport checks for custom-modality completion payloads.

A released frontend rejects an unknown top-level field at admission, so a
client cannot hand a backend its own modality inputs. These tests launch a real
frontend and a stub worker and prove the channel end to end: the schema admits
the payload, the wire carries it unchanged, and it arrives in its own field
rather than in the frontend-owned media map.

The payload is deliberately opaque — the frontend must not interpret, reorder,
coerce, or normalize any of it. Worker-side installation as vLLM's engine
``multi_modal_data`` is covered by the vLLM request-processor unit tests.
"""

from __future__ import annotations

from typing import Generator

import pytest
import requests

from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import ServicePorts

MODEL_NAME = "test-backend-mm-data"
ENDPOINT_PATH = "test.backend_mm_data.generate"
PLAIN_MODEL_NAME = "test-backend-mm-data-plain"
PLAIN_ENDPOINT_PATH = "test.backend_mm_data.generate_plain"
MISMATCH_STATUS = 422

# Response tokens are arbitrary; the assertion is the request-side round trip.
RESPONSE_TOKEN_IDS = (9707, 1879)

# Exercises every JSON type the channel must preserve, including nesting,
# float/int distinction, an explicit null, and a base64 blob standing in for a
# serialized tensor.
EXPECTED_PAYLOAD = {
    "custom_input": {
        "dtype": "float32-le",
        "shape": [2, 4],
        "data_base64": "AAAAAAAAgD8AAABAAABAQAAAgEAAAKBAAADAQAAA4EA=",
        "scale": 0.5,
        "normalized": False,
        "cache_key": None,
        "tags": ["alpha", "beta"],
    },
    "second_modality": [{"index": 0}, {"index": 1}],
}

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    # Each case starts a frontend and a worker subprocess and waits on real HTTP
    # health checks, so bound the whole case rather than hanging CI on a service
    # that never comes up.
    pytest.mark.timeout(180),
]


class _WorkerProcess(ManagedProcess):
    def __init__(self, request, *, frontend_port: int) -> None:
        super().__init__(
            command=["python3", "-m", "tests.frontend.backend_multimodal_worker"],
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", self._models_listed)
            ],
            timeout=60,
            display_output=True,
            terminate_all_matching_process_names=False,
            straggler_commands=["-m tests.frontend.backend_multimodal_worker"],
            log_dir=f"{request.node.name}_worker",
        )

    @staticmethod
    def _models_listed(response: requests.Response) -> bool:
        try:
            if response.status_code != 200:
                return False
            data = response.json()
        except ValueError:
            return False
        listed = {m.get("id") for m in data.get("data", [])}
        return {MODEL_NAME, PLAIN_MODEL_NAME} <= listed


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


def _completions(port: int, payload: dict) -> requests.Response:
    return requests.post(
        f"http://localhost:{port}/v1/completions",
        json=payload,
        timeout=60,
    )


def _assert_stub_answered(response: requests.Response) -> None:
    """The stub only yields tokens once its payload comparison passed."""
    assert response.status_code == 200, response.text
    choices = response.json()["choices"]
    assert len(choices) == 1, response.text
    assert choices[0]["finish_reason"] == "stop", response.text


def test_custom_modality_payload_reaches_the_worker_unchanged(services: int) -> None:
    response = _completions(
        services,
        {
            "model": MODEL_NAME,
            "prompt": [1, 2, 3],
            "max_tokens": 4,
            "multi_modal_data": EXPECTED_PAYLOAD,
        },
    )
    _assert_stub_answered(response)


def test_omitted_payload_stays_absent(services: int) -> None:
    response = _completions(
        services,
        {
            "model": PLAIN_MODEL_NAME,
            "prompt": [1, 2, 3],
            "max_tokens": 4,
        },
    )
    _assert_stub_answered(response)
