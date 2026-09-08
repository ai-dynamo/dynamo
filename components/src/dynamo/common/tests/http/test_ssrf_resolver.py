# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the connect-time SSRF backstop (``_ssrf_resolver``).

Exercise the DNS-rebinding case deterministically: a resolver fed a mix of
public and blocked answers (as if a rebinding server flipped between check and
connect) must drop the blocked ones and fail closed if none remain. No network.
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

    async def resolve(self, host, port=0, family=0):
        return [{"hostname": host, "host": ip, "port": port} for ip in self._ips]

    async def close(self):
        pass


def _resolver_with(ips: list[str]) -> BlocklistResolver:
    r = BlocklistResolver(allow_private_ips=False)
    r._inner = _FakeInner(ips)  # bypass real DNS
    return r


async def test_resolver_drops_blocked_ip_at_connect() -> None:
    # Rebinding answer (public + metadata IP): the blocked one is dropped.
    resolver = _resolver_with(["93.184.216.34", "169.254.169.254"])
    out = await resolver.resolve("evil.example.com")
    assert [h["host"] for h in out] == ["93.184.216.34"]


async def test_resolver_fails_closed_when_only_blocked() -> None:
    resolver = _resolver_with(["169.254.169.254"])
    with pytest.raises(SsrfBlockedAddress):
        await resolver.resolve("evil.example.com")


async def test_resolve_allowed_ip_picks_non_blocked() -> None:
    async def fake_getaddrinfo(host, *_a, **_k):
        return [(2, 1, 6, "", (ip, 0)) for ip in ["169.254.169.254", "93.184.216.34"]]

    with patch("asyncio.get_running_loop") as gl:
        gl.return_value.getaddrinfo = fake_getaddrinfo
        ip = await resolve_allowed_ip("evil.example.com", allow_private_ips=False)
    assert ip == "93.184.216.34"


async def test_resolve_allowed_ip_blocks_ip_literal() -> None:
    with pytest.raises(SsrfBlockedAddress):
        await resolve_allowed_ip("169.254.169.254", allow_private_ips=False)
