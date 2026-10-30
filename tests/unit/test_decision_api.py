# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import requests

from tests.utils import decision_api as client

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def _response(status, dialect="oai", code="rate_limit_exceeded", retry_after="1"):
    response = requests.Response()
    response.status_code = status
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    if dialect == "oai":
        body = {"error": {"code": code, "type": code}}
    else:
        body = {"detail": "capacity", "code": code}
    response._content = json.dumps(body).encode()
    return response


@pytest.mark.parametrize("status", [429, 503, 529])
@pytest.mark.parametrize("dialect", ["oai", "systemone"])
def test_systemone_client_retries_only_bounded_capacity_errors(
    monkeypatch, status, dialect
):
    sleeps = []
    calls = []
    monkeypatch.setattr(client.time, "sleep", sleeps.append)

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return next(responses)

    monkeypatch.setattr(client.requests, "post", post)
    overloaded = _response(status, dialect)
    responses = iter([overloaded, overloaded, _response(200, dialect)])
    assert client.post_with_capacity_retry("unused", {}).status_code == 200
    assert sleeps == [1, 1]
    assert len(calls) == 3
    calls.clear()
    responses = iter([overloaded, overloaded, overloaded])
    assert client.post_with_capacity_retry("unused", {}).status_code == status
    assert sleeps == [1, 1, 1, 1]
    assert len(calls) == 3


@pytest.mark.parametrize("retry_after", [None, "0", "2", "invalid"])
@pytest.mark.parametrize("status", [429, 503, 529])
def test_capacity_retry_rejects_invalid_retry_metadata(
    monkeypatch, status, retry_after
):
    response = _response(status, retry_after=retry_after)
    monkeypatch.setattr(client.requests, "post", lambda *args, **kwargs: response)
    monkeypatch.setattr(client.time, "sleep", lambda _: pytest.fail("must not sleep"))
    with pytest.raises(AssertionError, match="rate_limit_exceeded"):
        client.post_with_capacity_retry("unused", {})


@pytest.mark.parametrize("dialect", ["oai", "systemone"])
@pytest.mark.parametrize("status", [400, 429, 503, 529])
def test_non_capacity_errors_are_not_retried_even_with_retry_after(
    monkeypatch, dialect, status
):
    response = _response(status, dialect, code="server_error")
    calls = []

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    monkeypatch.setattr(client.requests, "post", post)
    monkeypatch.setattr(client.time, "sleep", lambda _: pytest.fail("must not sleep"))
    assert client.post_with_capacity_retry("unused", {}) is response
    assert len(calls) == 1


@pytest.mark.parametrize(
    "body", [b"not-json", b"[]", b'{"error":null}', b'{"error":"capacity"}']
)
def test_malformed_error_is_not_retried(monkeypatch, body):
    response = _response(503)
    response._content = body
    monkeypatch.setattr(client.requests, "post", lambda *args, **kwargs: response)
    monkeypatch.setattr(client.time, "sleep", lambda _: pytest.fail("must not sleep"))
    assert client.post_with_capacity_retry("unused", {}) is response
