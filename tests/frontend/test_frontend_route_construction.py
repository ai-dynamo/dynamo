# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the FrontendRoute / FrontendResponse construction contract.

The initial extension surface is static-path GET routes with a synchronous
handler. These assertions pin that contract at construction time so malformed
routes never reach (and panic) the axum router.
"""

from __future__ import annotations

import pytest

from dynamo.llm import FrontendResponse, FrontendRoute

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


def _handler(ctx):
    return {"ok": True}


def test_accepts_static_get_route():
    route = FrontendRoute("GET", "/v1/health/live", _handler)
    assert route.method == "GET"
    assert route.path == "/v1/health/live"


def test_get_is_case_insensitive():
    assert FrontendRoute("get", "/ok", _handler).method == "GET"


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "HEAD"])
def test_rejects_non_get_methods(method):
    with pytest.raises(ValueError):
        FrontendRoute(method, "/ok", _handler)


@pytest.mark.parametrize(
    "path",
    [
        "no-leading-slash",
        "/items/{id}",  # path parameter
        "/files/{*rest}",  # wildcard
        "/{}",  # malformed
        "/has space",  # whitespace
        "/tab\tchar",  # control char
    ],
)
def test_rejects_non_static_paths(path):
    with pytest.raises(ValueError):
        FrontendRoute("GET", path, _handler)


def test_rejects_async_handler():
    async def _async_handler(ctx):
        return {}

    with pytest.raises(ValueError):
        FrontendRoute("GET", "/ok", _async_handler)


def test_rejects_non_callable_handler():
    with pytest.raises(TypeError):
        FrontendRoute("GET", "/ok", object())


def test_frontend_response_rejects_invalid_status():
    # In u16 range but not a valid HTTP status code -> our StatusCode check.
    with pytest.raises(ValueError):
        FrontendResponse(9999, {"x": 1})


def test_frontend_response_accepts_valid_status():
    # Construction must not raise for a valid status.
    FrontendResponse(503, {"status": "not ready"})
