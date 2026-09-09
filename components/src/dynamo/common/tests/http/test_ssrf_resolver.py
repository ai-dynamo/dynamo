# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the aiohttp connect-time SSRF resolver (``_ssrf_resolver``).

Exercise the DNS-rebinding case deterministically: the resolver is fed a mix of
public and blocked answers (as if a rebinding server flipped between check and
connect) and must drop the blocked ones, keep the rest, and fail closed when
none remain. No network.
"""

from __future__ import annotations

import pytest

from dynamo.common.http._ssrf_resolver import BlocklistResolver, SsrfBlockedAddress

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
    r._inner = _FakeInner(ips)
    return r


async def test_resolver_drops_blocked_and_keeps_the_rest() -> None:
    # Rebinding answer (public + metadata IP): the blocked one is dropped, and
    # every non-blocked address survives for multi-address fallback.
    resolver = _resolver_with(["93.184.216.34", "169.254.169.254", "8.8.8.8"])
    out = await resolver.resolve("evil.example.com")
    assert [h["host"] for h in out] == ["93.184.216.34", "8.8.8.8"]


async def test_resolver_fails_closed_when_only_blocked() -> None:
    resolver = _resolver_with(["169.254.169.254"])
    with pytest.raises(SsrfBlockedAddress):
        await resolver.resolve("evil.example.com")
