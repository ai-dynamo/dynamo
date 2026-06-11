# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``HttpxClient`` exception mapping + redirect parsing.

The client singleton is held on the instance (``client._client``) and
replaced via ``patch.object``; the autouse ``_close_shared_http_client``
fixture in ``conftest.py`` resets the process-wide singleton between
tests.

Redirect parsing is exercised through the public ``fetch_bytes(...,
policy=...)`` path so we don't reach into the
``_fetch_body_or_redirect`` abstract-method seam from tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from dynamo.common import http as mm_http
from dynamo.common.http import HttpxClient
from dynamo.common.http.url_validator import UrlValidationPolicy

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _async_returning(value):
    """Coroutine factory used as ``side_effect`` for an awaited async call."""

    async def _coro(*args, **kwargs):
        return value

    return _coro


def _async_raising(exc):
    async def _coro(*args, **kwargs):
        raise exc

    return _coro


def _make_client_with_inner(inner) -> HttpxClient:
    client = HttpxClient()
    client._client = inner
    return client


_PERMISSIVE = UrlValidationPolicy(allow_http=True, allow_private_ips=True)


async def test_fetch_bytes_returns_body_on_200() -> None:
    response = MagicMock(spec=httpx.Response)
    response.content = b"hello"
    response.raise_for_status = MagicMock(return_value=None)
    inner = MagicMock(spec=httpx.AsyncClient)
    inner.is_closed = False
    inner.get = MagicMock(side_effect=_async_returning(response))
    client = _make_client_with_inner(inner)
    result = await client.fetch_bytes("https://h/x", 30.0)
    assert result == b"hello"


async def test_fetch_bytes_maps_timeout() -> None:
    inner = MagicMock(spec=httpx.AsyncClient)
    inner.is_closed = False
    inner.get = MagicMock(side_effect=_async_raising(httpx.ConnectTimeout("timeout")))
    client = _make_client_with_inner(inner)
    with pytest.raises(mm_http.HttpTimeoutError) as exc:
        await client.fetch_bytes("https://h/x", 30.0)
    assert isinstance(exc.value.__cause__, httpx.ConnectTimeout)


async def test_fetch_bytes_maps_status() -> None:
    response = MagicMock(spec=httpx.Response)
    response.status_code = 404
    response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=response
        )
    )
    inner = MagicMock(spec=httpx.AsyncClient)
    inner.is_closed = False
    inner.get = MagicMock(side_effect=_async_returning(response))
    client = _make_client_with_inner(inner)
    with pytest.raises(mm_http.HttpStatusError) as exc:
        await client.fetch_bytes("https://h/x", 30.0)
    assert exc.value.status == 404


async def test_fetch_bytes_maps_connection_error() -> None:
    inner = MagicMock(spec=httpx.AsyncClient)
    inner.is_closed = False
    inner.get = MagicMock(side_effect=_async_raising(httpx.ConnectError("refused")))
    client = _make_client_with_inner(inner)
    with pytest.raises(mm_http.HttpConnectionError):
        await client.fetch_bytes("https://h/x", 30.0)


async def test_redirect_resolved_through_policy_path() -> None:
    """302 → absolute next URL is parsed correctly when the SSRF policy
    drives the redirect loop. Verifies relative-Location resolution
    against the response URL through the public API."""
    redirect_response = MagicMock(spec=httpx.Response)
    redirect_response.is_redirect = True
    redirect_response.headers = {"location": "/next.png"}
    redirect_response.url = httpx.URL("https://h/x.png")
    redirect_response.aclose = AsyncMock(return_value=None)

    final_response = MagicMock(spec=httpx.Response)
    final_response.is_redirect = False
    final_response.content = b"final"
    final_response.raise_for_status = MagicMock(return_value=None)
    final_response.aclose = AsyncMock(return_value=None)

    responses_by_url = {
        "https://h/x.png": redirect_response,
        "https://h/next.png": final_response,
    }

    async def _send(request, *args, **kwargs):
        return responses_by_url[str(request.url)]

    inner = MagicMock(spec=httpx.AsyncClient)
    inner.is_closed = False
    inner.build_request = MagicMock(
        side_effect=lambda method, url, **kw: MagicMock(url=httpx.URL(url))
    )
    inner.send = MagicMock(side_effect=_send)

    client = _make_client_with_inner(inner)
    body = await client.fetch_bytes("https://h/x.png", 30.0, policy=_PERMISSIVE)
    assert body == b"final"
