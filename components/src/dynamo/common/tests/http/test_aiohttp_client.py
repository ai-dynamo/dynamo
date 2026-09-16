# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``AiohttpClient`` exception mapping + redirect parsing.

The session singleton is held on the client instance (``client._session``)
and replaced via ``patch.object``; the autouse
``_close_shared_http_client`` fixture in ``conftest.py`` resets the
process-wide singleton between tests.

Redirect parsing is exercised through the public ``fetch_bytes(...,
policy=...)`` path so we don't reach into the
``_fetch_body_or_redirect`` abstract-method seam from tests.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import aiohttp
import pytest
from yarl import URL

from dynamo.common import http as mm_http
from dynamo.common.http import AiohttpClient
from dynamo.common.http._ssrf_resolver import BlocklistResolver
from dynamo.common.http.url_validator import UrlValidationPolicy

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _FakeContent:
    """``response.content`` stand-in: the reader the size cap streams from."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    async def iter_chunked(self, n: int):
        for i in range(0, len(self._body), n) or [0]:
            yield self._body[i : i + n]


class _FakeResponse:
    """Minimal aiohttp response stand-in for ``async with session.get(...) as r``."""

    def __init__(self, *, status=200, headers=None, url=None, body=b"") -> None:
        self.status = status
        self.headers = headers or {}
        self.url = url
        self._body = body
        self.content = _FakeContent(body)

    def raise_for_status(self) -> None:
        return None

    async def read(self) -> bytes:
        return self._body


def _cm_returning(response):
    """``session.get`` stand-in: async-CM whose ``__aenter__`` yields ``response``."""

    class _CM:
        async def __aenter__(self):
            return response

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def _get(url, **kwargs):
        return _CM()

    return _get


def _cm_per_url(response_for_url):
    """``session.get`` stand-in that picks the response by URL.

    ``response_for_url`` maps URL → ``_FakeResponse``.
    """

    class _CM:
        def __init__(self, response):
            self._response = response

        async def __aenter__(self):
            return self._response

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def _get(url, **kwargs):
        return _CM(response_for_url[str(url)])

    return _get


def _cm_raising(exc_factory):
    """``session.get`` stand-in: async-CM whose ``__aenter__`` raises."""

    class _CM:
        async def __aenter__(self):
            raise exc_factory()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def _get(url, **kwargs):
        return _CM()

    return _get


def _make_client_with_session(session) -> AiohttpClient:
    client = AiohttpClient()
    client._session = session
    return client


_PERMISSIVE = UrlValidationPolicy(allow_http=True, allow_private_ips=True)

# Building the real connector trips aiohttp's notice that ``enable_cleanup_closed``
# is a no-op on Python >= 3.12.7. That flag is pre-existing on main, so this is
# scoped to the tests that build a real connector rather than ignored repo-wide.
_allows_cleanup_closed_notice = pytest.mark.filterwarnings(
    "ignore:enable_cleanup_closed ignored because:DeprecationWarning"
)


async def test_fetch_bytes_returns_body_on_200() -> None:
    response = _FakeResponse(status=200, body=b"hello")
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False
    session.get = _cm_returning(response)
    client = _make_client_with_session(session)
    result = await client.fetch_bytes("https://h/x", 30.0)
    assert result == b"hello"


async def test_fetch_bytes_maps_timeout() -> None:
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False
    session.get = _cm_raising(lambda: asyncio.TimeoutError())
    client = _make_client_with_session(session)
    with pytest.raises(mm_http.HttpTimeoutError):
        await client.fetch_bytes("https://h/x", 30.0)


async def test_fetch_bytes_maps_status() -> None:
    def _mk_error():
        return aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=404, message="Not Found"
        )

    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False
    session.get = _cm_raising(_mk_error)
    client = _make_client_with_session(session)
    with pytest.raises(mm_http.HttpStatusError) as exc:
        await client.fetch_bytes("https://h/x", 30.0)
    assert exc.value.status == 404


async def test_fetch_bytes_maps_connection_error() -> None:
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False
    session.get = _cm_raising(lambda: aiohttp.ClientConnectionError("refused"))
    client = _make_client_with_session(session)
    with pytest.raises(mm_http.HttpConnectionError):
        await client.fetch_bytes("https://h/x", 30.0)


async def test_redirect_resolved_through_policy_path() -> None:
    """302 → absolute next URL is parsed correctly when the SSRF policy
    drives the redirect loop. Verifies relative-Location resolution
    against the response URL through the public API."""
    responses = {
        "https://h/x.png": _FakeResponse(
            status=302,
            headers={"Location": "/next.png"},
            url=URL("https://h/x.png"),
        ),
        "https://h/next.png": _FakeResponse(status=200, body=b"final"),
    }
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False
    session.get = _cm_per_url(responses)
    client = _make_client_with_session(session)

    body = await client.fetch_bytes("https://h/x.png", 30.0, policy=_PERMISSIVE)
    assert body == b"final"


@_allows_cleanup_closed_notice
async def test_build_session_installs_the_connect_time_resolver() -> None:
    """The shared connector carries the connect-time resolver.

    Every fetch goes through this one connector, so losing the ``resolver=``
    wiring silently drops the connect-time check for the whole process while
    the resolver's own unit tests stay green.
    """
    client = AiohttpClient()
    session = client._build_session()
    try:
        assert isinstance(session.connector._resolver, BlocklistResolver)
    finally:
        await session.close()


@_allows_cleanup_closed_notice
async def test_connector_resolver_ignores_a_permissive_per_call_policy(
    monkeypatch,
) -> None:
    """A per-request ``allow_private_ips=True`` must not loosen the connector.

    The connector is shared across every caller, so it is keyed to the
    deployment-wide env baseline instead. ``_PERMISSIVE`` above is a per-call
    policy and deliberately has no bearing here.
    """
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    client = AiohttpClient()
    session = client._build_session()
    try:
        assert session.connector._resolver._allow_private_ips is False
    finally:
        await session.close()

    monkeypatch.setenv("DYN_MM_ALLOW_INTERNAL", "1")
    session = client._build_session()
    try:
        assert session.connector._resolver._allow_private_ips is True
    finally:
        await session.close()
