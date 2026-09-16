# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the aiohttp connect-time SSRF resolver (``_ssrf_resolver``).

Exercise the DNS-rebinding case deterministically: the resolver is fed a mix of
public and blocked answers (as if a rebinding server flipped between check and
connect) and must drop the blocked ones, keep the rest, and fail closed when
none remain. No network.
"""

from __future__ import annotations

import errno

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


async def test_allow_private_ips_bypasses_filtering() -> None:
    # The env baseline (DYN_MM_ALLOW_INTERNAL=1) opts the deployment into
    # internal targets, so the resolver returns every answer unfiltered.
    resolver = BlocklistResolver(allow_private_ips=True)
    resolver._inner = _FakeInner(["169.254.169.254", "10.0.0.5"])
    out = await resolver.resolve("internal.svc")
    assert [h["host"] for h in out] == ["169.254.169.254", "10.0.0.5"]


async def test_configured_egress_proxy_is_not_filtered(monkeypatch) -> None:
    """A corporate proxy on a private address must not be filtered out.

    The session runs with ``trust_env=True``, so with a proxy configured the
    connector dials the proxy and it is the proxy's own address that reaches
    this resolver. Filtering it turns every fetch into a connection error --
    measured before the exemption, HTTP_PROXY at a private host failed them all.
    """
    # aiohttp's proxies_from_env() prefers the lowercase name, and CI images can
    # carry one already -- clear both so this test controls what is configured.
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.internal:3128")

    resolver = BlocklistResolver(allow_private_ips=False)
    resolver._inner = _FakeInner(["10.1.2.3"])

    hosts = await resolver.resolve("proxy.internal")

    assert [h["host"] for h in hosts] == ["10.1.2.3"]


async def test_a_non_proxy_host_is_still_filtered(monkeypatch) -> None:
    """Control: the exemption is for the configured proxy only."""
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.internal:3128")

    resolver = BlocklistResolver(allow_private_ips=False)
    resolver._inner = _FakeInner(["10.1.2.3"])

    with pytest.raises(SsrfBlockedAddress):
        await resolver.resolve("origin.example.com")


async def test_blocked_message_bounds_the_hostname() -> None:
    """The host is caller-supplied; this text reaches a response and a log."""
    resolver = _resolver_with(["169.254.169.254"])

    with pytest.raises(SsrfBlockedAddress) as excinfo:
        await resolver.resolve("h" * 200_000 + ".example.com")

    assert len(str(excinfo.value)) < 500
    assert "h" * 200 not in str(excinfo.value)


async def test_blocked_reason_survives_into_strerror() -> None:
    """aiohttp renders ``strerror``, not ``str(exc)``, for a connector error.

    A one-argument OSError leaves ``strerror`` as None, and the caller then
    receives ``Cannot connect to host ... [None]`` with the reason dropped at
    the one place an operator reads it.
    """
    resolver = _resolver_with(["169.254.169.254"])

    with pytest.raises(SsrfBlockedAddress) as excinfo:
        await resolver.resolve("evil.example.com")

    assert excinfo.value.strerror == (
        "host evil.example.com resolves only to blocked IPs"
    )
    assert excinfo.value.errno == errno.EHOSTUNREACH
