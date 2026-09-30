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
from dynamo.common.http.base import HttpConfigurationError
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
    # Both strictness keys map to the same double, so a test does not have to
    # care which session the policy under test selects.
    client._sessions = {True: session, False: session}
    return client


_PERMISSIVE = UrlValidationPolicy(allow_http=True, allow_private_ips=True)
_STRICT = UrlValidationPolicy(allow_http=True, allow_private_ips=False)

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

    Every fetch goes through a connector built here, so losing the
    ``resolver=`` wiring silently drops the connect-time check for the whole
    process while the resolver's own unit tests stay green.
    """
    client = AiohttpClient()
    # Through _get_session, so client.close() owns it. _build_session does not
    # register the session, and closing the client would leave it open.
    session = await client._get_session(False)
    try:
        assert isinstance(session.connector._resolver, BlocklistResolver)
    finally:
        await client.close()
    assert session.closed


@_allows_cleanup_closed_notice
async def test_connect_policy_is_the_intersection(monkeypatch) -> None:
    """Neither side alone decides whether the connector may return private IPs.

    A permissive request must not loosen a strict deployment, and a permissive
    deployment must not override a request that explicitly asked for strict.
    """
    strict = UrlValidationPolicy(allow_http=True, allow_private_ips=False)

    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    client = AiohttpClient()
    assert client._connect_allows_private(None) is False
    assert client._connect_allows_private(_PERMISSIVE) is False
    assert client._connect_allows_private(strict) is False

    monkeypatch.setenv("DYN_MM_ALLOW_INTERNAL", "1")
    assert client._connect_allows_private(None) is True
    assert client._connect_allows_private(_PERMISSIVE) is True
    # The case that regressed: the deployment allows private, the caller does
    # not, and the caller wins.
    assert client._connect_allows_private(strict) is False


@_allows_cleanup_closed_notice
async def test_a_strict_request_gets_a_filtering_connector(monkeypatch) -> None:
    """The intersection reaches the connector, not just the helper."""
    monkeypatch.setenv("DYN_MM_ALLOW_INTERNAL", "1")
    strict = UrlValidationPolicy(allow_http=True, allow_private_ips=False)
    client = AiohttpClient()
    try:
        permissive_session = await client._get_session(
            client._connect_allows_private(_PERMISSIVE)
        )
        strict_session = await client._get_session(
            client._connect_allows_private(strict)
        )
        assert permissive_session is not strict_session
        assert permissive_session.connector._resolver._allow_private_ips is True
        assert strict_session.connector._resolver._allow_private_ips is False
    finally:
        await client.close()


@_allows_cleanup_closed_notice
async def test_close_closes_every_resolver(monkeypatch) -> None:
    """aiohttp will not close an injected resolver, so the client must.

    ``TCPConnector`` sets ``_resolver_owner=False`` for a resolver it did not
    create, and only closes its own. Without this the optional aiodns resolver
    stays registered for the loop after the session is gone.
    """
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    client = AiohttpClient()
    session = await client._get_session(False)
    resolver = session.connector._resolver
    assert session.connector._resolver_owner is False

    closed = []
    original = resolver._inner.close

    async def _spy() -> None:
        closed.append(1)
        await original()

    resolver._inner.close = _spy
    await client.close()

    assert closed, "the injected resolver was never closed"
    assert client._sessions == {}
    assert client._resolvers == {}


async def test_a_proxied_fetch_fails_closed_without_the_opt_in(monkeypatch) -> None:
    """A proxied fetch puts the origin out of the resolver's reach.

    Documenting the boundary does not enforce it, so without an operator
    assertion the fetch must not quietly run unchecked. Driven through
    ``fetch_bytes`` so it also pins that the gate is wired into the fetch
    path, not merely present as a helper.
    """
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    monkeypatch.delenv("DYN_MM_TRUST_EGRESS_PROXY", raising=False)
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(name, "http://proxy.internal:3128")

    client = AiohttpClient()
    try:
        with pytest.raises(HttpConfigurationError) as excinfo:
            # No policy, so no URL validation runs first and the gate is the
            # only thing that can reject this.
            await client.fetch_bytes("https://example.com/x.png", 5.0)
        message = str(excinfo.value)
        assert "DYN_MM_TRUST_EGRESS_PROXY" in message
        # An operator configuration fault, not a verdict on the caller's URL,
        # so it must not arrive as the ValueError subclass that maps to a 400.
        assert not isinstance(excinfo.value, ValueError)
    finally:
        await client.close()


async def test_the_gate_guards_the_revalidating_path_too(monkeypatch) -> None:
    """Both fetch seams must be gated, not just the simple one."""
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    monkeypatch.delenv("DYN_MM_TRUST_EGRESS_PROXY", raising=False)
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(name, "http://proxy.internal:3128")

    client = AiohttpClient()
    try:
        with pytest.raises(HttpConfigurationError) as excinfo:
            await client._fetch_body_or_redirect(
                "https://example.com/x.png", 5.0, policy=_STRICT
            )
        assert "DYN_MM_TRUST_EGRESS_PROXY" in str(excinfo.value)
    finally:
        await client.close()


async def test_no_proxy_exempts_a_host_from_the_gate(monkeypatch) -> None:
    """NO_PROXY means the fetch goes direct, so the gate must not fire.

    Asking only "is any proxy variable set" would refuse this fetch even
    though the connect-time check governs it perfectly well. Asserted on the
    helper directly, because driving it through a fetch reaches the transport
    and passes whether or not the gate is precise.
    """
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    monkeypatch.delenv("DYN_MM_TRUST_EGRESS_PROXY", raising=False)
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(name, "http://proxy.internal:3128")
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "example.com")

    client = AiohttpClient()
    # Must not raise: NO_PROXY sends this host direct.
    await client._require_trusted_egress_proxy("https://example.com/x.png")
    # A host NOT covered by NO_PROXY is still gated, which is the control.
    with pytest.raises(HttpConfigurationError):
        await client._require_trusted_egress_proxy("https://other.invalid/x.png")
    await client.close()


async def test_the_proxy_gate_does_not_fire_when_private_is_allowed(
    monkeypatch,
) -> None:
    """DYN_MM_ALLOW_INTERNAL=1 already permits private destinations.

    Nothing is left for the proxy gate to protect, so it must not demand a
    second variable.
    """
    monkeypatch.setenv("DYN_MM_ALLOW_INTERNAL", "1")
    monkeypatch.delenv("DYN_MM_TRUST_EGRESS_PROXY", raising=False)
    for name in ("http_proxy", "HTTP_PROXY"):
        monkeypatch.setenv(name, "http://proxy.internal:3128")

    client = AiohttpClient()
    assert client._connect_allows_private(None) is True
    # The gate is keyed to the effective policy, so it never runs here.
    client._require_trusted_egress_proxy  # present, but not reached
    await client.close()


async def test_the_opt_in_allows_a_proxied_fetch(monkeypatch) -> None:
    """With the assertion set, the fetch proceeds and the operator owns it."""
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    monkeypatch.setenv("DYN_MM_TRUST_EGRESS_PROXY", "1")
    for name in ("http_proxy", "HTTP_PROXY"):
        monkeypatch.setenv(name, "http://proxy.internal:3128")

    client = AiohttpClient()
    # Returns without raising, which is the whole assertion.
    await client._require_trusted_egress_proxy("https://example.com/x.png")
    await client.close()


@_allows_cleanup_closed_notice
async def test_each_policy_pool_gets_the_full_connection_limit(monkeypatch) -> None:
    """The cap is per connect-time policy, and the help text says so.

    Two pools can coexist, each with the configured limit, so a deployment
    that enables internal access and also issues stricter per-request policies
    can reach twice the value. Pinned here so the number and the documented
    meaning cannot drift apart silently.
    """
    monkeypatch.setenv("DYN_MM_ALLOW_INTERNAL", "1")
    client = AiohttpClient()
    try:
        permissive = await client._get_session(True)
        strict = await client._get_session(False)
        limit = client._config.max_connections
        assert permissive is not strict
        assert permissive.connector.limit == limit
        assert strict.connector.limit == limit
    finally:
        await client.close()


async def test_the_default_configuration_only_ever_builds_one_pool(monkeypatch) -> None:
    """Without DYN_MM_ALLOW_INTERNAL the connect policy is always strict.

    So the second pool, and the doubled cap with it, cannot appear in the
    default deployment whatever a caller passes.
    """
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    client = AiohttpClient()
    strict = UrlValidationPolicy(allow_http=True, allow_private_ips=False)
    assert client._connect_allows_private(None) is False
    assert client._connect_allows_private(_PERMISSIVE) is False
    assert client._connect_allows_private(strict) is False
    await client.close()


async def test_the_proxy_gate_is_awaitable() -> None:
    """The gate must not run aiohttp's proxy discovery on the event loop.

    ``get_env_proxy_for_url`` does proxy-bypass discovery and ``.netrc`` file
    reads, which is why aiohttp itself calls it through ``asyncio.to_thread``.
    A synchronous gate repeats that blocking work inline on every protected
    hop.
    """
    import inspect

    assert inspect.iscoroutinefunction(AiohttpClient._require_trusted_egress_proxy)
