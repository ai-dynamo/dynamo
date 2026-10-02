# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""System One HTTP contract coverage over the SGLang mocker transport."""

import math
from types import SimpleNamespace

import pytest
import requests

from tests.frontend.conftest import MockerWorkerProcess, wait_for_http_completions_ready
from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.model(QWEN),
    pytest.mark.timeout(180),
]


def _payload(model=QWEN):
    return {
        "model": model,
        "state": "A customer was charged twice and requests a refund today.",
        "chat_template_kwargs": {"enable_thinking": False},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Choose the responsible team.",
                "criteria": {"billing": "Payments", "technical": "Software"},
            },
            "urgent": {"type": "noul", "instructions": "Action is needed today."},
            "severity": {
                "type": "score",
                "instructions": "Rate the impact.",
                "criteria": ["Low", "Moderate", "High"],
            },
        },
    }


@pytest.fixture
def systemone_server(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
):
    config = getattr(request, "param", {})
    enabled = config.get("enabled", True)
    extra_args = ["--enable-systemone-api"] if enabled else []
    extra_env = {
        "DYN_SYSTEMONE_MAX_INFLIGHT_BRANCHES": str(config.get("branch_limit", 256)),
        "DYN_HTTP_OVERLOAD_STATUS_CODE": str(config.get("overload_status", 529)),
    }
    ports = dynamo_dynamic_ports
    with (
        DynamoFrontendProcess(
            request,
            frontend_port=ports.frontend_port,
            extra_args=extra_args,
            extra_env=extra_env,
        ),
        MockerWorkerProcess(
            request,
            QWEN,
            ports.frontend_port,
            ports.system_ports[0],
            extra_args=["--engine-type", "sglang", "--sglang-generate"],
        ),
    ):
        wait_for_http_completions_ready(frontend_port=ports.frontend_port, model=QWEN)
        yield f"http://localhost:{ports.frontend_port}"


def _assert_distribution(probabilities):
    assert all(
        math.isfinite(value) and 0 <= value <= 1 for value in probabilities.values()
    )
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-12)


def test_systemone_mixed_results_and_existing_apis(systemone_server):
    response = requests.post(
        f"{systemone_server}/v1/systemone", json=_payload(), timeout=30
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-dynamo-systemone-version"] == "1"
    assert response.headers["x-request-id"]
    body = response.json()
    assert body["model"] == QWEN
    assert list(body["answers"]) == ["route", "urgent", "severity"]
    route, urgent, severity = (body["answers"][key] for key in body["answers"])
    assert route["type"] == "choice"
    assert route["choice"] in ["billing", "technical"]
    assert list(route["probabilities"]) == ["billing", "technical"]
    assert urgent["type"] == "noul"
    assert math.isfinite(urgent["noul"]) and 0 <= urgent["noul"] <= 1
    assert severity["type"] == "score"
    assert math.isfinite(severity["score"]) and 0 <= severity["score"] <= 2
    assert list(severity["legend"]) == ["0", "1", "2"]
    for answer in [route, severity]:
        _assert_distribution(answer["probabilities"])
        assert math.isfinite(answer["confidence"]) and 0 <= answer["confidence"] <= 1
    for answer in [route, urgent, severity]:
        assert (
            math.isfinite(answer["x_label_mass"]) and 0 <= answer["x_label_mass"] <= 1
        )
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] == 0

    for endpoint, fields in [
        ("chat/completions", {"messages": [{"role": "user", "content": "Hello"}]}),
        ("completions", {"prompt": "Hello"}),
    ]:
        response = requests.post(
            f"{systemone_server}/v1/{endpoint}",
            json={"model": QWEN, "max_tokens": 4, "stream": False, **fields},
            timeout=30,
        )
        assert response.status_code == 200, response.text
        assert len(response.json()["choices"]) == 1
        assert response.json()["usage"]["completion_tokens"] > 0

    alias = requests.post(
        f"{systemone_server}/v1/systemone", json=_payload("jev-latest"), timeout=30
    )
    assert alias.status_code == 200, alias.text
    assert alias.json()["model"] == QWEN
    unknown = requests.post(
        f"{systemone_server}/v1/systemone", json=_payload("missing-model"), timeout=30
    )
    assert unknown.status_code == 404, unknown.text


def test_systemone_validation_and_body_limit(systemone_server):
    url = f"{systemone_server}/v1/systemone"
    malformed = requests.post(
        url, data='{"model":', headers={"Content-Type": "application/json"}, timeout=30
    )
    assert malformed.status_code == 400, malformed.text
    nested = _payload()
    nested["questions"]["urgent"]["unknown"] = True
    temperature = {**_payload(), "temperature": 0}
    for payload in [nested, temperature]:
        response = requests.post(url, json=payload, timeout=30)
        assert response.status_code == 422, response.text
        assert response.json()["error"]["message"]
    oversized = {**_payload(), "state": "x" * (4 * 1024 * 1024)}
    response = requests.post(url, json=oversized, timeout=30)
    assert response.status_code == 413, response.text


@pytest.mark.parametrize("systemone_server", [{"enabled": False}], indirect=True)
def test_systemone_disabled_route(systemone_server):
    response = requests.post(
        f"{systemone_server}/v1/systemone", json=_payload(), timeout=30
    )
    assert response.status_code == 404, response.text


@pytest.mark.parametrize(
    "systemone_server", [{"branch_limit": 2, "overload_status": 503}], indirect=True
)
def test_systemone_impossible_admission_and_recovery(systemone_server):
    url = f"{systemone_server}/v1/systemone"
    response = requests.post(url, json=_payload(), timeout=30)
    assert response.status_code == 422, response.text
    assert "retry-after" not in response.headers
    assert "branch limit" in response.json()["error"]["message"]
    payload = _payload()
    payload["questions"] = {"urgent": payload["questions"]["urgent"]}
    recovered = requests.post(url, json=payload, timeout=30)
    assert recovered.status_code == 200, recovered.text
    assert list(recovered.json()["answers"]) == ["urgent"]


def test_systemone_client_retries_only_bounded_capacity_errors(monkeypatch):
    from tests.serve import test_systemone_sglang as client

    sleeps = []
    monkeypatch.setattr(client.time, "sleep", sleeps.append)
    responses = iter(
        SimpleNamespace(status_code=status, headers={"retry-after": "1"}, text="")
        for status in [529, 503, 200]
    )
    monkeypatch.setattr(
        client.requests, "post", lambda *args, **kwargs: next(responses)
    )
    assert client._post_with_capacity_retry("unused", {}).status_code == 200
    assert sleeps == [1, 1]
    overloaded = SimpleNamespace(status_code=529, headers={"retry-after": "1"}, text="")
    responses = iter([overloaded, overloaded, overloaded])
    monkeypatch.setattr(
        client.requests, "post", lambda *args, **kwargs: next(responses)
    )
    assert client._post_with_capacity_retry("unused", {}).status_code == 529
    assert sleeps == [1, 1, 1, 1]
    unavailable = SimpleNamespace(
        status_code=503, headers={}, text="worker unavailable"
    )
    monkeypatch.setattr(client.requests, "post", lambda *args, **kwargs: unavailable)
    with pytest.raises(AssertionError, match="worker unavailable"):
        client._post_with_capacity_retry("unused", {})
