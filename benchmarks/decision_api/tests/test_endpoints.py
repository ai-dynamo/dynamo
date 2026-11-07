# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from types import SimpleNamespace

import pytest
from aiperf.common.models import TextResponse
from dynamo_decision_perf.contracts import DecisionContractError
from dynamo_decision_perf.endpoints import (
    NativeScoreEndpoint,
    OpenAIDecisionEndpoint,
    payload_hash,
)
from test_contracts import fixture, native_fixture

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def endpoint(cls=OpenAIDecisionEndpoint):
    config = SimpleNamespace(headers=[], api_key=None)
    return cls(model_endpoint=SimpleNamespace(endpoint=config))


def info(body, request_id="measurement-1", raw=False):
    return SimpleNamespace(
        x_request_id=request_id,
        payload_bytes=None if raw else json.dumps(body).encode(),
        turns=[SimpleNamespace(raw_payload=body)],
        endpoint_headers={},
    )


@pytest.mark.parametrize("raw", [False, True])
def test_native_salt_changes_per_replay_without_modifying_frozen_body(raw):
    body, _ = native_fixture()
    original = copy.deepcopy(body)
    adapter = endpoint(NativeScoreEndpoint)
    first, second = info(body, raw=raw), info(body, "measurement-2", raw=raw)
    headers = adapter.get_endpoint_headers(first)
    adapter.get_endpoint_headers(second)
    sent = json.loads(first.payload_bytes)
    assert sent["cache_salt"] != json.loads(second.payload_bytes)["cache_salt"]
    assert headers["X-Decision-Measurement-ID"] == "measurement-1"
    assert "X-Request-ID" not in headers
    assert headers["X-Decision-Payload-SHA256"] == payload_hash(original)
    assert body == original and "cache_salt" not in body
    assert {k: v for k, v in sent.items() if k != "cache_salt"} == original


def test_public_payload_preserved_and_no_generated_text():
    request, body = fixture()
    adapter, request_info = endpoint(), info(request, raw=True)
    assert adapter.format_payload(request_info) == request
    adapter.get_endpoint_headers(request_info)
    assert json.loads(request_info.payload_bytes) == request
    parsed = adapter.parse_response(TextResponse(perf_ns=2, text=json.dumps(body)))
    assert parsed.data.body == body and parsed.data.get_text() == ""
    assert parsed.metadata["question_count"] == 1
    assert parsed.usage["output_tokens"] == 0


@pytest.mark.parametrize(
    "bad", ["{}", "not-json", '{"model":"a","model":"b"}', "[DONE]"]
)
def test_http_200_malformed_body_is_not_silent_success(bad):
    with pytest.raises(DecisionContractError):
        endpoint().parse_response(TextResponse(perf_ns=2, text=bad))


def record(request, body, headers=None):
    return SimpleNamespace(
        _parsed_responses_cache=None,
        request_info=info(request),
        responses=[TextResponse(perf_ns=2, text=json.dumps(body))],
        trace_data=SimpleNamespace(
            response_headers=headers
            or {
                "x-request-id": "measurement-1",
                "x-typesafe-request-id": "measurement-1",
            }
        ),
    )


def test_record_correlates_request_count_and_request_ids():
    request, body = fixture()
    adapter = endpoint()
    good = record(request, body)
    parsed = adapter.extract_response_data(good)
    assert parsed[0].metadata["request_id"] == "measurement-1"
    assert adapter.extract_response_data(good) is parsed
    request["questions"].append(copy.deepcopy(request["questions"][0]))
    invalid = record(request, body)
    assert adapter.extract_response_data(invalid) == []
    assert invalid.error.type == "DecisionContractError"
    request, body = fixture()
    invalid = record(request, body, {"x-request-id": "other"})
    assert adapter.extract_response_data(invalid) == []
    assert invalid.error.type == "DecisionContractError"


def test_server_request_id_is_separate_from_measurement_request_id():
    request, body = fixture()
    parsed = endpoint().extract_response_data(
        record(
            request,
            body,
            {
                "x-request-id": "server-generated",
                "x-typesafe-request-id": "server-generated",
            },
        )
    )
    assert parsed[0].metadata["request_id"] == "measurement-1"
    assert parsed[0].metadata["server_request_id"] == "server-generated"


def test_missing_usage_does_not_invent_token_accounting():
    request, body = fixture()
    body.pop("usage")
    parsed = endpoint().extract_response_data(record(request, body))[0]
    assert parsed.usage is None
    assert parsed.metadata["input_tokens"] is None
    assert parsed.metadata["output_tokens"] is None


@pytest.mark.parametrize("kind", ["HTTPStatusError", "TimeoutError"])
def test_transport_errors_are_preserved_without_schema_reclassification(kind):
    request, body = fixture()
    failed = record(request, body)
    failed.responses = []
    original_error = SimpleNamespace(type=kind, message="transport failure")
    failed.error = original_error
    assert endpoint().extract_response_data(failed) == []
    assert failed.error is original_error


def test_schema_failure_retains_original_response_and_request_evidence():
    request, _ = fixture()
    invalid = record(request, {"invalid": "HTTP 200 response"})
    response, request_info = invalid.responses[0], invalid.request_info
    assert endpoint().extract_response_data(invalid) == []
    assert invalid.error.type == "DecisionContractError"
    assert invalid.responses[0] is response
    assert invalid.request_info is request_info


def test_native_skips_nonterminal_but_requires_one_terminal():
    request, body = native_fixture()
    adapter = endpoint(NativeScoreEndpoint)
    pending = copy.deepcopy(body)
    pending["meta_info"]["finish_reason"] = None
    assert (
        adapter.parse_response(TextResponse(perf_ns=1, text=json.dumps(pending)))
        is None
    )
    assert adapter.parse_response(TextResponse(perf_ns=3, text="[DONE]")) is None
    assert (
        adapter.parse_response(
            TextResponse(perf_ns=2, text=json.dumps(body))
        ).data.get_text()
        == ""
    )
    invalid = record(request, pending)
    assert adapter.extract_response_data(invalid) == []
    assert invalid.error.type == "DecisionContractError"
    duplicate = record(request, body)
    duplicate.responses *= 2
    assert adapter.extract_response_data(duplicate) == []
    assert duplicate.error.type == "DecisionContractError"


def test_decision_transport_omits_wire_request_id_only(monkeypatch):
    from aiperf.transports.aiohttp_transport import AioHttpTransport
    from dynamo_decision_perf.transport import DecisionHttpTransport

    original = {
        "X-Request-ID": "measurement-1",
        "x-request-id": "override",
        "X-Decision-Measurement-ID": "measurement-1",
        "X-Correlation-ID": "logical-1",
        "Authorization": "test-only",
    }
    monkeypatch.setattr(
        AioHttpTransport, "build_headers", lambda self, request: original
    )
    request = info(fixture()[0])
    transport = object.__new__(DecisionHttpTransport)
    headers = transport.build_headers(request)
    assert headers == {k: v for k, v in original.items() if k.lower() != "x-request-id"}
    assert request.x_request_id == "measurement-1"
    assert "X-Request-ID" in original
