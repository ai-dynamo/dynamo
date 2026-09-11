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
from dynamo.common.http.url_validator import UrlValidationError, UrlValidationPolicy

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


def _streaming(body: bytes):
    """``aiter_bytes`` stand-in — the seam the download cap reads through."""

    async def _iter(*args, **kwargs):
        _iter.call_kwargs = kwargs
        for i in range(0, len(body), kwargs.get("chunk_size") or len(body) or 1):
            yield body[i : i + (kwargs.get("chunk_size") or len(body))]

    _iter.call_kwargs = {}
    return _iter


def _inner_sending(response_or_exc) -> MagicMock:
    """AsyncClient stand-in for the streaming path (``build_request`` + ``send``)."""
    inner = MagicMock(spec=httpx.AsyncClient)
    inner.is_closed = False
    inner.build_request = MagicMock(
        side_effect=lambda method, url, **kw: MagicMock(url=httpx.URL(url))
    )
    if isinstance(response_or_exc, BaseException):
        inner.send = MagicMock(side_effect=_async_raising(response_or_exc))
    else:
        inner.send = MagicMock(side_effect=_async_returning(response_or_exc))
    return inner


def _make_client_with_inner(inner) -> HttpxClient:
    client = HttpxClient()
    client._client = inner
    return client


_PERMISSIVE = UrlValidationPolicy(allow_http=True, allow_private_ips=True)


async def test_fetch_bytes_returns_body_on_200() -> None:
    response = MagicMock(spec=httpx.Response)
    response.raise_for_status = MagicMock(return_value=None)
    response.aiter_bytes = _streaming(b"hello")
    response.aclose = AsyncMock(return_value=None)
    client = _make_client_with_inner(_inner_sending(response))
    result = await client.fetch_bytes("https://h/x", 30.0)
    assert result == b"hello"
    # Without stream=True httpx reads the whole body into response.content
    # before the cap ever looks at it; the cap still raises, so nothing else
    # here would notice.
    assert client._client.send.call_args.kwargs["stream"] is True


async def test_fetch_bytes_refuses_a_body_over_the_cap() -> None:
    response = MagicMock(spec=httpx.Response)
    response.raise_for_status = MagicMock(return_value=None)
    response.aiter_bytes = _streaming(b"x" * 2048)
    response.aclose = AsyncMock(return_value=None)
    client = _make_client_with_inner(_inner_sending(response))
    with pytest.raises(UrlValidationError, match="download limit"):
        await client.fetch_bytes("https://h/x", 30.0, max_bytes=512)
    # The reader has to be handed a granularity, or one decompressed chunk is
    # buffered whole before the running total is checked.
    assert response.aiter_bytes.call_kwargs["chunk_size"] > 0


async def test_fetch_bytes_maps_timeout() -> None:
    client = _make_client_with_inner(_inner_sending(httpx.ConnectTimeout("timeout")))
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
    response.aclose = AsyncMock(return_value=None)
    client = _make_client_with_inner(_inner_sending(response))
    with pytest.raises(mm_http.HttpStatusError) as exc:
        await client.fetch_bytes("https://h/x", 30.0)
    assert exc.value.status == 404


async def test_fetch_bytes_maps_connection_error() -> None:
    client = _make_client_with_inner(_inner_sending(httpx.ConnectError("refused")))
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
    final_response.aiter_bytes = _streaming(b"final")
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
