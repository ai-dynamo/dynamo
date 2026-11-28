# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the public-error oracle without a model or a running server."""

import json
from types import SimpleNamespace

import pytest

from tests.utils.payloads import PublicErrorPayload

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]


def _payload(stream: bool = False) -> PublicErrorPayload:
    """Use explicit observations rather than an engine-specific expectation table."""
    return PublicErrorPayload(
        body={"stream": stream},
        expected_response=[],
        expected_log=[],
        expected_status_code=200 if stream else 400,
        expected_error_code=400,
        expected_message="Invalid request",
    )


def _response(document: dict, stream: bool = False) -> SimpleNamespace:
    """Expose only the response fields the real payload reads."""
    text = json.dumps(document)
    if stream:
        text = f"data: {text}\n\ndata: [DONE]\n\n"
    return SimpleNamespace(
        status_code=200 if stream else 400,
        headers={
            "content-type": (
                "text/event-stream; charset=utf-8" if stream else "application/json"
            )
        },
        text=text,
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_public_error_accepts_unary_envelopes(wrapped):
    error = {"code": 400, "message": "Invalid request"}
    response = _response({"error": error} if wrapped else error)

    assert _payload().process_response(response) == response.text


def test_public_error_accepts_committed_sse():
    response = _response({"error": {"code": 400, "message": "Invalid request"}}, True)

    _payload(True).process_response(response)


def test_public_error_rejects_early_frontend_message():
    response = _response({"code": 400, "message": "Unsupported image URL"})

    with pytest.raises(AssertionError, match="Expected message"):
        _payload().process_response(response)


def test_public_error_rejects_private_sse_details():
    response = _response(
        {
            "error": {
                "code": 400,
                "message": "Invalid request",
                "details": {"diagnostic": "OBJECT AT 0xdeadbeef"},
            }
        },
        True,
    )

    with pytest.raises(AssertionError, match="Private detail"):
        _payload(True).process_response(response)


@pytest.mark.parametrize("failure", ["header", "duplicate"])
def test_public_error_requires_sse_framing(failure):
    response = _response({"error": {"code": 400, "message": "Invalid request"}}, True)
    if failure == "header":
        response.headers["content-type"] = "application/json"
    else:
        response.text += response.text

    with pytest.raises(AssertionError, match="Expected.*SSE"):
        _payload(True).process_response(response)


@pytest.mark.parametrize("failure", ["status", "code"])
def test_public_error_requires_exact_status_and_code(failure):
    response = _response(
        {"code": 500 if failure == "code" else 400, "message": "Invalid request"}
    )
    if failure == "status":
        response.status_code = 500

    with pytest.raises(AssertionError, match="Expected (HTTP|error code)"):
        _payload().process_response(response)
