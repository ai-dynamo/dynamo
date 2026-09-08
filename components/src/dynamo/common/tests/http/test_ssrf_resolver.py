# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the connect-time SSRF backstop (``_ssrf_resolver``).

These exercise the DNS-rebinding case deterministically: the resolver is fed a
mix of public and blocked addresses (as if a rebinding server flipped its
answer between ``validate_url`` and connect) and must drop the blocked ones so
the backend never dials an internal address. No network is used.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dynamo.common.http._ssrf_resolver import (
    BlocklistResolver,
    SsrfBlockedAddress,
    resolve_allowed_ip,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


class _FakeInner:
    """Stand-in for aiohttp's DefaultResolver returning canned entries."""

    def __init__(self, ips: list[str]) -> None:
        self._ips = ips
        self.closed = False

    async def resolve(self, host, port=0, family=0):
        return [{"hostname": host, "host": ip, "port": port} for ip in self._ips]

    async def close(self):
        self.closed = True


def _resolver_with(ips: list[str], *, allow_private_ips: bool) -> BlocklistResolver:
    r = BlocklistResolver(allow_private_ips=allow_private_ips)
    r._inner = _FakeInner(ips)  # bypass real DNS
    return r


# ---------------------------------------------------------------------------
# aiohttp BlocklistResolver
# ---------------------------------------------------------------------------


async def test_resolver_filters_blocked_ip_at_connect() -> None:
    # DNS answers with a public IP *and* the metadata IP (rebinding): the
    # blocked one must be dropped, the public one kept.
    resolver = _resolver_with(["93.184.216.34", "169.254.169.254"], allow_private_ips=False)
    out = await resolver.resolve("evil.example.com")
    assert [h["host"] for h in out] == ["93.184.216.34"]


async def test_resolver_raises_when_only_blocked() -> None:
    # A pure rebind to an internal address leaves nothing to dial -> fail closed.
    resolver = _resolver_with(["169.254.169.254"], allow_private_ips=False)
    with pytest.raises(SsrfBlockedAddress):
        await resolver.resolve("evil.example.com")


async def test_resolver_passthrough_when_internal_allowed() -> None:
    resolver = _resolver_with(["10.0.0.5"], allow_private_ips=True)
    out = await resolver.resolve("internal.svc")
    assert [h["host"] for h in out] == ["10.0.0.5"]


# ---------------------------------------------------------------------------
# httpx resolve_allowed_ip (pin helper)
# ---------------------------------------------------------------------------


def _fake_getaddrinfo(addrs: list[str]):
    async def _impl(host, *_a, **_k):
        return [(2, 1, 6, "", (addr, 0)) for addr in addrs]

    return _impl


async def test_resolve_allowed_ip_picks_non_blocked() -> None:
    with patch("asyncio.get_running_loop") as gl:
        gl.return_value.getaddrinfo = _fake_getaddrinfo(
            ["169.254.169.254", "93.184.216.34"]
        )
        ip = await resolve_allowed_ip("evil.example.com", allow_private_ips=False)
    assert ip == "93.184.216.34"


async def test_resolve_allowed_ip_raises_when_only_blocked() -> None:
    with patch("asyncio.get_running_loop") as gl:
        gl.return_value.getaddrinfo = _fake_getaddrinfo(["10.0.0.5"])
        with pytest.raises(SsrfBlockedAddress):
            await resolve_allowed_ip("evil.example.com", allow_private_ips=False)


async def test_resolve_allowed_ip_blocks_ip_literal() -> None:
    with pytest.raises(SsrfBlockedAddress):
        await resolve_allowed_ip("169.254.169.254", allow_private_ips=False)


async def test_resolve_allowed_ip_returns_public_literal_unchanged() -> None:
    # Public IP literal: no lookup, dialed as-is.
    assert await resolve_allowed_ip("8.8.8.8", allow_private_ips=False) == "8.8.8.8"
