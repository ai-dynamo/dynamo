# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The configured egress proxy is exempt from the connect-time address check
only on a connection that actually dials the proxy.

These tests run real aiohttp connections to stub servers on 127.0.0.1. The
connect-time DNS answer for every host name is loopback. It stands for a proxy
on a private address, or for a name that resolved to a public address when it
was validated and to loopback when the client connected. URL validation does
not run here, so the connect-time check is the only one in play, and no real
DNS lookup happens.

A fetch without a URL policy follows redirects itself, one hop at a time. The
tests at the end also pin its time budget, its limit of 10 redirects, and the
redirect targets that it takes and refuses, which are the ones aiohttp took
and refused.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket

import aiohttp
import pytest

from dynamo.common.http import (
    AiohttpClient,
    HttpConnectionError,
    HttpStatusError,
    HttpTimeoutError,
    _ssrf_resolver,
)
from dynamo.common.http.url_validator import UrlValidationPolicy

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    # Real connections to stub servers on loopback.
    pytest.mark.timeout(30),
]

_PROXY_HOST = "proxy.test"
_BLOCKED = "resolves only to blocked IPs"
_BODY = b"stub-ok"
_STRICT = UrlValidationPolicy(allow_http=True, allow_private_ips=False)


class _LoopbackInner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def resolve(self, host, port=0, family=socket.AF_INET):
        return [
            {
                "hostname": host,
                "host": "127.0.0.1",
                "port": port,
                "family": socket.AF_INET,
                "proto": 0,
                "flags": 0,
            }
        ]

    async def close(self) -> None:
        pass


class _LoopbackServer:
    """Records the first line of each connection, and answers a GET with 200.

    With ``redirect``, a request line that starts with its prefix gets a 302
    to its ``Location`` template, where ``{port}`` is this server's port.
    ``delay`` holds each answer back for that many seconds.
    """

    def __init__(
        self,
        redirect: tuple[bytes, str] | None = None,
        delay: float = 0.0,
        redirect_header: bytes = b"Location",
    ) -> None:
        self.first_lines: list[bytes] = []
        self._redirect = redirect
        self._delay = delay
        self._redirect_header = redirect_header

    async def _handle(self, reader, writer) -> None:
        try:
            try:
                data = await reader.read(65536)
            except ConnectionError:
                data = b""
            line = data.split(b"\r\n", 1)[0]
            self.first_lines.append(line)
            await asyncio.sleep(self._delay)
            if self._redirect and line.startswith(self._redirect[0]):
                location = self._redirect[1].format(port=self.port).encode()
                writer.write(
                    b"HTTP/1.1 302 Found\r\n%s: %s\r\nContent-Length: 0\r\n"
                    b"Connection: close\r\n\r\n" % (self._redirect_header, location)
                )
            elif data.startswith(b"GET "):
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n"
                    b"Connection: close\r\n\r\n%s" % (len(_BODY), _BODY)
                )
            # The client can be gone already, for example after a timeout.
            with contextlib.suppress(ConnectionError):
                await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

    async def __aenter__(self) -> "_LoopbackServer":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc_info) -> None:
        self._server.close()
        await self._server.wait_closed()


@pytest.fixture(autouse=True)
def _strict_deployment_with_loopback_dns(monkeypatch):
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    monkeypatch.delenv("DYN_MM_TRUST_EGRESS_PROXY", raising=False)
    monkeypatch.setattr(_ssrf_resolver, "DefaultResolver", _LoopbackInner)


def _configure_proxy(monkeypatch, port: int) -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.setenv(name, f"http://{_PROXY_HOST}:{port}")
    monkeypatch.setenv("NO_PROXY", _PROXY_HOST)


@pytest.mark.parametrize(
    ("scheme", "on_the_proxy_port", "seam"),
    [
        ("https", False, "simple"),
        ("https", False, "revalidating"),
        ("http", True, "simple"),
    ],
    ids=["https", "https-revalidating", "http-proxy-port"],
)
async def test_a_direct_fetch_to_the_proxy_host_is_filtered(
    monkeypatch, scheme, on_the_proxy_port, seam
) -> None:
    """NO_PROXY sends this fetch direct, so it gets no proxy exemption.

    The host name is the configured proxy's, and the private server must not
    receive the connection: not on another port, and not on the proxy's own
    port either, where the fetch would reach the proxy as an origin.
    """
    async with _LoopbackServer() as server:
        _configure_proxy(monkeypatch, server.port if on_the_proxy_port else 3128)
        url = f"{scheme}://{_PROXY_HOST}:{server.port}/x"
        client = AiohttpClient()
        try:
            with pytest.raises(HttpConnectionError, match=_BLOCKED):
                if seam == "simple":
                    await client.fetch_bytes(url, 5.0)
                else:
                    await client._fetch_body_or_redirect(url, 5.0, policy=_STRICT)
        finally:
            await client.close()
    assert server.first_lines == []


async def test_a_trusted_private_proxy_still_carries_the_fetch(monkeypatch) -> None:
    monkeypatch.setenv("DYN_MM_TRUST_EGRESS_PROXY", "1")
    async with _LoopbackServer() as proxy:
        _configure_proxy(monkeypatch, proxy.port)
        client = AiohttpClient()
        try:
            body = await client.fetch_bytes("http://origin.test/x", 5.0)
        finally:
            await client.close()
    assert body == _BODY
    assert proxy.first_lines == [b"GET http://origin.test/x HTTP/1.1"]


async def test_a_cached_proxy_answer_is_not_reused_for_a_direct_fetch(
    monkeypatch,
) -> None:
    """aiohttp caches DNS answers per connector, keyed by (host, port) alone.

    So the proxied fetch below leaves the proxy's unfiltered answer in its
    connector's cache. A direct fetch to the same host and port must not get
    that answer, and must not reach the proxy.
    """
    monkeypatch.setenv("DYN_MM_TRUST_EGRESS_PROXY", "1")
    async with _LoopbackServer() as proxy:
        _configure_proxy(monkeypatch, proxy.port)
        client = AiohttpClient()
        try:
            await client.fetch_bytes("http://origin.test/x", 5.0)
            with pytest.raises(HttpConnectionError, match=_BLOCKED):
                await client.fetch_bytes(f"http://{_PROXY_HOST}:{proxy.port}/x", 5.0)
        finally:
            await client.close()
    assert proxy.first_lines == [b"GET http://origin.test/x HTTP/1.1"]


async def test_the_proxy_host_name_without_a_proxy_setting_is_filtered() -> None:
    async with _LoopbackServer() as server:
        client = AiohttpClient()
        try:
            with pytest.raises(HttpConnectionError, match=_BLOCKED):
                await client.fetch_bytes(f"http://{_PROXY_HOST}:{server.port}/x", 5.0)
        finally:
            await client.close()
    assert server.first_lines == []


# --- Redirects without a URL policy ---


async def test_a_redirect_to_the_proxy_gets_its_own_connector(monkeypatch) -> None:
    """Each redirect picks its connector, like the first hop.

    Here a proxied fetch is redirected to the proxy's own host and port, which
    NO_PROXY sends direct. aiohttp used to follow the redirect on the first
    hop's connector, whose resolver exempts the proxy.
    """
    monkeypatch.setenv("DYN_MM_TRUST_EGRESS_PROXY", "1")
    redirect = (b"GET http://origin.test/", f"http://{_PROXY_HOST}:{{port}}/admin")
    async with _LoopbackServer(redirect) as proxy:
        _configure_proxy(monkeypatch, proxy.port)
        client = AiohttpClient()
        try:
            with pytest.raises(HttpConnectionError, match=_BLOCKED):
                await client.fetch_bytes("http://origin.test/x", 5.0)
        finally:
            await client.close()
    assert proxy.first_lines == [b"GET http://origin.test/x HTTP/1.1"]


async def test_a_redirect_from_a_direct_fetch_goes_through_the_proxy(
    monkeypatch,
) -> None:
    """The other direction: a direct origin redirects to a URL that a trusted
    proxy carries, and that hop goes through the proxy."""
    monkeypatch.setenv("DYN_MM_TRUST_EGRESS_PROXY", "1")
    async with _LoopbackServer() as proxy, _LoopbackServer(
        (b"GET /start", "http://origin.test/y")
    ) as origin:
        _configure_proxy(monkeypatch, proxy.port)
        # aiohttp skips the resolver for an IP literal, so this first hop goes
        # direct and passes the connect-time check.
        monkeypatch.setenv("NO_PROXY", f"{_PROXY_HOST},127.0.0.1")
        client = AiohttpClient()
        try:
            body = await client.fetch_bytes(
                f"http://127.0.0.1:{origin.port}/start", 5.0
            )
        finally:
            await client.close()
    assert body == _BODY
    assert origin.first_lines == [b"GET /start HTTP/1.1"]
    assert proxy.first_lines == [b"GET http://origin.test/y HTTP/1.1"]


async def test_one_time_budget_covers_every_redirect_hop() -> None:
    """Each hop gets only the time that is left of the request's budget.

    Each hop alone fits in the budget, but the two together do not.
    """
    async with _LoopbackServer((b"GET /start", "/next"), delay=0.5) as server:
        client = AiohttpClient()
        try:
            with pytest.raises(HttpTimeoutError):
                await client.fetch_bytes(f"http://127.0.0.1:{server.port}/start", 0.8)
        finally:
            await client.close()
    assert server.first_lines == [b"GET /start HTTP/1.1", b"GET /next HTTP/1.1"]


async def test_redirects_without_a_policy_stop_at_ten() -> None:
    """The limit and the error are aiohttp's, as before this loop existed."""
    async with _LoopbackServer((b"GET ", "/again")) as server:
        client = AiohttpClient()
        try:
            with pytest.raises(HttpStatusError) as excinfo:
                await client.fetch_bytes(f"http://127.0.0.1:{server.port}/start", 5.0)
        finally:
            await client.close()
    assert excinfo.value.status == 0
    assert isinstance(excinfo.value.__cause__, aiohttp.TooManyRedirects)
    assert len(server.first_lines) == 10


@pytest.mark.parametrize(
    ("location", "cause"),
    [
        ("http://[::1", aiohttp.InvalidUrlRedirectClientError),
        ("ws://127.0.0.1:{port}/x", aiohttp.NonHttpUrlRedirectClientError),
        ("https:x", aiohttp.InvalidUrlRedirectClientError),
    ],
    ids=["unparsable", "not-http", "no-host"],
)
async def test_a_redirect_target_that_aiohttp_refused_is_refused(
    monkeypatch, location, cause
) -> None:
    """aiohttp refused these targets with a client error, which the fetch
    reported as HttpConnectionError. The loop refuses them before the next
    hop, so the proxy gate does not see a URL without a host either."""
    # The fetch goes direct, and a proxy applies to a URL without a host.
    for name in ("HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.setenv(name, f"http://{_PROXY_HOST}:3128")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    async with _LoopbackServer((b"GET /start", location)) as server:
        client = AiohttpClient()
        try:
            with pytest.raises(HttpConnectionError) as excinfo:
                await client.fetch_bytes(f"http://127.0.0.1:{server.port}/start", 5.0)
        finally:
            await client.close()
    assert isinstance(excinfo.value.__cause__, cause)
    assert len(server.first_lines) == 1


async def test_a_redirect_in_the_uri_header_is_followed() -> None:
    """aiohttp took the obsolete URI header when Location was missing."""
    async with _LoopbackServer(
        (b"GET /start", "/next"), redirect_header=b"URI"
    ) as server:
        client = AiohttpClient()
        try:
            body = await client.fetch_bytes(
                f"http://127.0.0.1:{server.port}/start", 5.0
            )
        finally:
            await client.close()
    assert body == _BODY
    assert len(server.first_lines) == 2


@pytest.mark.parametrize(
    "trusted_proxy", [True, False], ids=["trusted-proxy", "no-proxy"]
)
async def test_a_url_that_does_not_parse_is_a_connection_error(
    monkeypatch, trusted_proxy
) -> None:
    """aiohttp refuses a URL that it cannot parse, and the fetch reports
    HttpConnectionError. The proxy lookup that runs first must not raise the
    parse error itself, with or without the trusted-proxy opt-in."""
    if trusted_proxy:
        monkeypatch.setenv("DYN_MM_TRUST_EGRESS_PROXY", "1")
        _configure_proxy(monkeypatch, 3128)
    client = AiohttpClient()
    try:
        with pytest.raises(HttpConnectionError) as excinfo:
            await client.fetch_bytes("http://[::1", 5.0)
    finally:
        await client.close()
    assert isinstance(excinfo.value.__cause__, aiohttp.InvalidUrlClientError)
