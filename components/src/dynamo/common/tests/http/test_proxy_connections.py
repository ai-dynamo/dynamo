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
"""

from __future__ import annotations

import asyncio
import contextlib
import socket

import pytest

from dynamo.common.http import AiohttpClient, HttpConnectionError, _ssrf_resolver
from dynamo.common.http.url_validator import UrlValidationPolicy

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

_PROXY_HOST = "proxy.test"
_BLOCKED = "resolves only to blocked IPs"
_BODY = b"stub-ok"
_STRICT = UrlValidationPolicy(allow_http=True, allow_private_ips=False)


class _LoopbackInner:
    """Connect-time DNS that answers every host name with 127.0.0.1."""

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
    """Records the first line of each connection, and answers a GET with 200."""

    def __init__(self) -> None:
        self.first_lines: list[bytes] = []

    async def _handle(self, reader, writer) -> None:
        try:
            try:
                data = await reader.read(65536)
            except ConnectionError:
                data = b""
            self.first_lines.append(data.split(b"\r\n", 1)[0])
            if data.startswith(b"GET "):
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n"
                    b"Connection: close\r\n\r\n%s" % (len(_BODY), _BODY)
                )
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
        ("http", False, "simple"),
        ("http", True, "simple"),
    ],
    ids=["https", "https-revalidating", "http", "http-proxy-port"],
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
    """Control: a trusted proxy on a private address still works."""
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
    """Control: with no proxy configured, the same host name is refused."""
    async with _LoopbackServer() as server:
        client = AiohttpClient()
        try:
            with pytest.raises(HttpConnectionError, match=_BLOCKED):
                await client.fetch_bytes(f"http://{_PROXY_HOST}:{server.port}/x", 5.0)
        finally:
            await client.close()
    assert server.first_lines == []
