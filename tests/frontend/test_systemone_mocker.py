# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decision API contracts over the SGLang mocker transport."""

import math
from types import SimpleNamespace

import pytest
import requests

from dynamo import prometheus_names
from tests.frontend.conftest import MockerWorkerProcess, wait_for_http_completions_ready
from tests.utils.constants import QWEN
from tests.utils.decision_api import (
    assert_decision_body,
    assert_request_headers,
    decision_payload,
)
from tests.utils.managed_process import DynamoFrontendProcess
from tests.utils.prometheus import find_metric_samples

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.model(QWEN),
    pytest.mark.timeout(180),
]


def _payload(model=QWEN):
    return decision_payload(model, "systemone")


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
        "DYN_DISABLE_FRONTEND_NVEXT": str(config.get("disable_nvext", False)).lower(),
    }
    ports = dynamo_dynamic_ports
    with (
        DynamoFrontendProcess(
            request,
            frontend_port=ports.frontend_port,
            extra_args=extra_args,
            extra_env=extra_env,
            terminate_all_matching_process_names=False,
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


@pytest.fixture
def worker_request_count(dynamo_dynamic_ports):
    def read():
        response = requests.get(
            f"http://localhost:{dynamo_dynamic_ports.system_ports[0]}/metrics",
            timeout=30,
        )
        response.raise_for_status()
        samples = find_metric_samples(
            response.text,
            f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.work_handler.REQUESTS_TOTAL}",
        )
        assert samples, "Worker ingress request counter must be registered"
        return sum(samples)

    return read


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
        if "x_label_mass" in answer:
            assert (
                math.isfinite(answer["x_label_mass"])
                and 0 <= answer["x_label_mass"] <= 1
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
    assert alias.status_code == 404, alias.text
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
        assert response.json()["detail"][0]["msg"]
    oversized = {**_payload(), "state": "x" * (4 * 1024 * 1024)}
    response = requests.post(url, json=oversized, timeout=30)
    assert response.status_code == 413, response.text


@pytest.mark.parametrize("systemone_server", [{"enabled": False}], indirect=True)
def test_systemone_disabled_route(systemone_server):
    for dialect in ("systemone", "oai"):
        route = "systemone" if dialect == "systemone" else "decisions"
        response = requests.post(
            f"{systemone_server}/v1/{route}",
            json=decision_payload(QWEN, dialect),
            timeout=30,
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
    assert "branch limit" in response.json()["detail"][0]["msg"]
    payload = _payload()
    payload["questions"] = {"urgent": payload["questions"]["urgent"]}
    recovered = requests.post(url, json=payload, timeout=30)
    assert recovered.status_code == 200, recovered.text
    assert list(recovered.json()["answers"]) == ["urgent"]


def test_decisions_and_systemone_preserve_their_contracts(systemone_server):
    results = {}
    for dialect in ("oai", "systemone"):
        route = "systemone" if dialect == "systemone" else "decisions"
        response = requests.post(
            f"{systemone_server}/v1/{route}",
            json=decision_payload(QWEN, dialect),
            timeout=30,
        )
        assert response.status_code == 200, response.text
        assert_request_headers(response)
        results[dialect] = response.json()
        assert_decision_body(results[dialect], QWEN, dialect)
    oai, jev = (results[k] for k in ("oai", "systemone"))
    for index, key in ((0, "route"), (2, "severity")):
        expected = list(jev["answers"][key]["probabilities"].values())
        assert [
            p["probability"] for p in oai["answers"][index]["probabilities"]
        ] == pytest.approx(expected)
    assert oai["answers"][1]["probability"] == pytest.approx(
        jev["answers"]["urgent"]["noul"]
    )


@pytest.mark.parametrize(
    "systemone_server",
    [{"disable_nvext": False}, {"disable_nvext": True}],
    indirect=True,
)
def test_decisions_reject_extensions_independently_of_global_policy(
    systemone_server, worker_request_count
):
    baseline = worker_request_count()
    extensions = (
        {},
        {"format": "oai"},
        {"format": "sglang_native"},
        {"format": "unknown"},
        None,
        "oai",
        7,
        [],
    )
    for dialect, status in (("oai", 400), ("systemone", 422)):
        route = "systemone" if dialect == "systemone" else "decisions"
        for extension in extensions:
            response = requests.post(
                f"{systemone_server}/v1/{route}",
                json={**decision_payload(QWEN, dialect), "nvext": extension},
                timeout=30,
            )
            assert response.status_code == status, response.text
            body = response.json()
            assert "nvext" in str(body)
            assert "object" not in body
            assert_request_headers(response)
    assert worker_request_count() == baseline


def test_decisions_mixed_schema_rejections(systemone_server, worker_request_count):
    baseline = worker_request_count()
    url = f"{systemone_server}/v1/decisions"
    native = {
        "model": QWEN,
        "input": "Choose a team.",
        "questions": [
            {
                "type": "choice",
                "id": "team",
                "question": "Team?",
                "options": [{"name": "billing"}, {"name": "technical"}],
            }
        ],
    }
    for route, status in (("decisions", 400), ("systemone", 422)):
        for payload in (native, {**native, "nvext": {"format": "sglang_native"}}):
            response = requests.post(
                f"{systemone_server}/v1/{route}", json=payload, timeout=30
            )
            assert response.status_code == status, response.text
            body = response.json()
            assert body["error"]["message"] if route == "decisions" else body["detail"]
            assert "object" not in body
            assert_request_headers(response)
    duplicate = requests.post(
        url,
        data='{"model":"first","model":"second","input":"x","questions":[]}',
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    assert duplicate.status_code == 400, duplicate.text
    assert worker_request_count() == baseline


def test_decisions_preserves_typed_choices_and_unnamed_order(systemone_server):
    payload = decision_payload(QWEN)
    questions = [
        {
            "type": "choice",
            "instructions": "Choose the boolean value, not its spelling.",
            "choices": [
                {"value": True, "description": "Boolean true"},
                {"value": "true", "description": "The string true"},
            ],
        },
        payload["questions"][1],
        {key: value for key, value in payload["questions"][2].items() if key != "name"},
    ]
    response = requests.post(
        f"{systemone_server}/v1/decisions",
        json={**payload, "questions": questions},
        timeout=30,
    )
    assert response.status_code == 200, response.text
    answers = response.json()["answers"]
    assert [answer["name"] for answer in answers] == [None, "urgent", None]
    assert [answer["type"] for answer in answers] == ["choice", "predicate", "score"]
    values = [entry["value"] for entry in answers[0]["probabilities"]]
    assert values[0] is True
    assert type(values[1]) is str and values[1] == "true"


def test_systemone_client_retries_only_bounded_capacity_errors(monkeypatch):
    from tests.utils import decision_api as client

    sleeps = []
    monkeypatch.setattr(client.time, "sleep", sleeps.append)
    responses = iter(
        SimpleNamespace(status_code=status, headers={"retry-after": "1"}, text="")
        for status in [429, 429, 200]
    )
    monkeypatch.setattr(
        client.requests, "post", lambda *args, **kwargs: next(responses)
    )
    assert client.post_with_capacity_retry("unused", {}).status_code == 200
    assert sleeps == [1, 1]
    overloaded = SimpleNamespace(status_code=429, headers={"retry-after": "1"}, text="")
    responses = iter([overloaded, overloaded, overloaded])
    monkeypatch.setattr(
        client.requests, "post", lambda *args, **kwargs: next(responses)
    )
    assert client.post_with_capacity_retry("unused", {}).status_code == 429
    assert sleeps == [1, 1, 1, 1]
    unavailable = SimpleNamespace(
        status_code=429, headers={}, text="missing retry contract"
    )
    monkeypatch.setattr(client.requests, "post", lambda *args, **kwargs: unavailable)
    with pytest.raises(AssertionError, match="missing retry contract"):
        client.post_with_capacity_retry("unused", {})
    unavailable = SimpleNamespace(
        status_code=503, headers={}, text="worker unavailable"
    )
    monkeypatch.setattr(client.requests, "post", lambda *args, **kwargs: unavailable)
    assert client.post_with_capacity_retry("unused", {}) is unavailable
