# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real frontend transport tests for arbitrary JSON chat content and mixed media."""

from __future__ import annotations

from typing import Generator

import pytest
import requests

from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import ServicePorts

MODEL_NAME = "test-json-mm-data"
ENDPOINT_PATH = "test.json_mm_data.generate"
PLAIN_MODEL_NAME = "test-json-mm-data-plain"
PLAIN_ENDPOINT_PATH = "test.json_mm_data.generate_plain"
MISMATCH_STATUS = 422
IMAGE_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

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
    pytest.mark.parametrize("request_plane", ["tcp"], indirect=True),
    pytest.mark.parametrize("event_plane", ["zmq"], indirect=True),
]


class _WorkerProcess(ManagedProcess):
    def __init__(self, request, *, frontend_port: int) -> None:
        super().__init__(
            command=["python3", "-m", "tests.frontend.json_multimodal_worker"],
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", self._models_listed)
            ],
            timeout=60,
            display_output=True,
            terminate_all_matching_process_names=False,
            straggler_commands=["-m tests.frontend.json_multimodal_worker"],
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
        extra_args=[
            "--discovery-backend",
            "etcd",
            "--request-plane",
            "tcp",
            "--router-mode",
            "round-robin",
        ],
        terminate_all_matching_process_names=False,
    ):
        with _WorkerProcess(request, frontend_port=frontend_port):
            yield frontend_port


def _chat(port: int, payload: dict) -> requests.Response:
    return requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        json=payload,
        timeout=60,
    )


def _assert_stub_answered(response: requests.Response) -> None:
    """The stub only yields tokens once its payload comparison passed."""
    assert response.status_code == 200, response.text
    choices = response.json()["choices"]
    assert len(choices) == 1, response.text
    assert choices[0]["finish_reason"] == "stop", response.text


@pytest.mark.parametrize("role", ["user", "tool"])
def test_custom_modality_payload_reaches_the_worker_unchanged(
    services: int, role: str
) -> None:
    content = [{"type": "text", "text": "Predict this"}]
    content.extend(
        {"type": kind, kind: value} for kind, value in EXPECTED_PAYLOAD.items()
    )
    messages = [{"role": role, "content": content}]
    if role == "tool":
        messages[0]["tool_call_id"] = "input-1"
        messages = [
            {"role": "user", "content": "Predict this"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "input-1",
                        "type": "function",
                        "function": {"name": "input", "arguments": "{}"},
                    }
                ],
            },
            *messages,
        ]
    response = _chat(
        services,
        {
            "model": MODEL_NAME,
            "nvext": {"token_data": [1, 2, 3]},
            "messages": messages,
            "max_tokens": 4,
        },
    )
    _assert_stub_answered(response)


def test_omitted_payload_stays_absent(services: int) -> None:
    response = _chat(
        services,
        {
            "model": PLAIN_MODEL_NAME,
            "nvext": {"token_data": [1, 2, 3]},
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Predict this"}]}
            ],
            "max_tokens": 4,
        },
    )
    _assert_stub_answered(response)


def test_worker_without_json_capability_rejects_custom_content(services: int) -> None:
    response = _chat(
        services,
        {
            "model": PLAIN_MODEL_NAME,
            "nvext": {"token_data": [1, 2, 3]},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Predict this"},
                        {"type": "chemistry", "chemistry": {"data_base64": "..."}},
                    ],
                }
            ],
            "max_tokens": 4,
        },
    )
    assert response.status_code == 400, response.text
    assert "do not support custom JSON" in response.text


def test_json_and_url_media_reach_the_same_worker_field(services: int) -> None:
    content = [
        {"type": "text", "text": "Predict this"},
        {"type": "image_url", "image_url": {"url": IMAGE_URL}},
    ]
    content.extend(
        {"type": kind, kind: value} for kind, value in EXPECTED_PAYLOAD.items()
    )
    response = _chat(
        services,
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": content}],
            "nvext": {"token_data": [1, 2, 4]},
            "max_tokens": 4,
        },
    )
    _assert_stub_answered(response)


def test_custom_json_uses_the_normal_chat_template(services: int) -> None:
    content = [{"type": "text", "text": "Predict this <custom_input>"}]
    content.extend(
        {"type": kind, kind: value} for kind, value in EXPECTED_PAYLOAD.items()
    )
    response = _chat(
        services,
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 4,
        },
    )
    _assert_stub_answered(response)
