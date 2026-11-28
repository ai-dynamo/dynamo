# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import requests

from tests.utils.engine_process import EngineProcess, EngineResponseError
from tests.utils.payloads import HttpErrorPayload
from tests.utils.structured_output import (
    StructuredOutputErrorPayload,
    structured_output_validation_payloads,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.timeout(5),
]

_MESSAGE = "Invalid JSON schema: enum array must not be empty"
_JSON_ERROR = json.dumps({"message": _MESSAGE, "type": "Bad Request", "code": 400})


def _sse_error(code=400):
    return {
        "error": {"message": _MESSAGE, "type": "invalid_request_error", "code": code}
    }


def _sse(*chunks):
    return (
        "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        + "data: [DONE]\n\n"
    )


def _response(status_code, content):
    response = requests.Response()
    response.status_code = status_code
    response.encoding = "utf-8"
    response._content = content.encode("utf-8")
    response._content_consumed = True
    return response


def _payload(streaming, *, rejection):
    return next(
        payload
        for payload in structured_output_validation_payloads(())
        if payload.http_stream == streaming
        and isinstance(payload, StructuredOutputErrorPayload) == rejection
    )


@pytest.fixture
def engine_process():
    return EngineProcess.__new__(EngineProcess)


@pytest.fixture
def rejection_payload(monkeypatch):
    def prepare(streaming):
        payload = _payload(streaming, rejection=True)
        counts = iter([0.0, 1.0])
        monkeypatch.setattr(payload, "_worker_requests", lambda: next(counts))
        payload.before_request()
        return payload

    return prepare


@pytest.mark.parametrize("expected_status", [200, 400])
def test_ordinary_payload_status_remains_exact(engine_process, expected_status):
    payload = HttpErrorPayload(
        body={},
        expected_response=[],
        expected_log=[],
        expected_status_code=expected_status,
    )
    engine_process.check_response(payload, _response(expected_status, _JSON_ERROR))

    other_status = 400 if expected_status == 200 else 200
    with pytest.raises(EngineResponseError) as caught:
        engine_process.check_response(
            payload, _response(other_status, json.dumps(_sse_error()))
        )
    assert f"got {other_status}, expected {expected_status}" in str(caught.value)
    assert "Error details:" in str(caught.value)
    assert _MESSAGE in str(caught.value)


@pytest.mark.parametrize(
    ("streaming", "status_code", "content"),
    [
        pytest.param(False, 400, _JSON_ERROR, id="unary-json-400"),
        pytest.param(True, 400, _JSON_ERROR, id="stream-json-400"),
        pytest.param(True, 200, _sse(_sse_error()), id="stream-sse-400"),
    ],
)
def test_structured_output_error_accepts_both_commit_boundaries(
    engine_process, rejection_payload, streaming, status_code, content
):
    engine_process.check_response(
        rejection_payload(streaming), _response(status_code, content)
    )


@pytest.mark.parametrize(
    ("streaming", "status_code", "content"),
    [
        pytest.param(False, 200, _sse(_sse_error()), id="unary-http-200"),
        pytest.param(True, 503, _JSON_ERROR, id="stream-http-503"),
        pytest.param(True, 200, _JSON_ERROR, id="stream-json-200"),
        pytest.param(True, 200, _sse(_sse_error(500)), id="stream-error-code-500"),
        pytest.param(
            True,
            200,
            _sse(
                {"choices": [{"delta": {"content": "unexpected output"}}]}, _sse_error()
            ),
            id="content-before-error",
        ),
        pytest.param(
            True,
            200,
            _sse(_sse_error(), {"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            id="success-after-error",
        ),
    ],
)
def test_structured_output_error_rejects_wrong_status_or_body(
    engine_process, rejection_payload, streaming, status_code, content
):
    with pytest.raises(EngineResponseError):
        engine_process.check_response(
            rejection_payload(streaming), _response(status_code, content)
        )


@pytest.mark.parametrize("streaming", [False, True])
def test_structured_output_positive_control_still_requires_http_200(
    engine_process, streaming
):
    with pytest.raises(EngineResponseError, match="got 400, expected 200"):
        engine_process.check_response(
            _payload(streaming, rejection=False), _response(400, _JSON_ERROR)
        )
