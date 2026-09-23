# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connect-time SSRF backstop for the aiohttp backend.

``validate_url`` checks a hostname's resolved IPs, but the client re-resolves at
connect, so a DNS-rebinding server can return a public IP on the check and an
internal one at connect. aiohttp's ``TCPConnector(resolver=...)`` hook lets us
resolve + filter once and hand the connector the validated addresses to dial,
while the hostname is still used for TLS SNI / certificate verification — the
same mechanism the Rust frontend uses on reqwest's ``dns_resolver``
(``lib/llm/src/preprocessor/media/loader.rs``). Reuses
:func:`url_validator.is_blocked_ip`.

Scope: this governs **direct** connections. When an egress proxy is configured,
the proxy resolves the origin, so SSRF must be enforced at the proxy / network
layer instead (true of the Rust path as well).
"""

from __future__ import annotations

import errno
import socket

from aiohttp.abc import AbstractResolver, ResolveResult
from aiohttp.helpers import proxies_from_env
from aiohttp.resolver import DefaultResolver

from .url_validator import describe_media_source, is_blocked_ip


class SsrfBlockedAddress(OSError):
    """Raised at connect time when every resolved IP is in a blocked range."""


def _env_proxy_hosts() -> frozenset[str]:
    """Hosts of any egress proxy configured in the environment.

    The session runs with ``trust_env=True``, so when a proxy is configured the
    connector dials *the proxy*, and it is the proxy's own address that reaches
    this resolver -- the origin is resolved by the proxy, out of our sight. A
    corporate proxy on a private address would therefore be filtered here and
    take out every fetch: measured, HTTP_PROXY at a private host turns each one
    into ClientConnectorDNSError. Exempt the configured proxy so proxied
    deployments keep working, and see the module docstring for what that means
    for enforcement.
    """
    return frozenset(
        info.proxy.host
        for info in proxies_from_env().values()
        if info.proxy.host is not None
    )


class BlocklistResolver(AbstractResolver):
    """aiohttp resolver that drops blocked IPs before the connector dials.

    Returns the full set of non-blocked addresses (not just the first) so
    aiohttp keeps its normal multi-address / Happy-Eyeballs fallback.

    Note aiohttp short-circuits IP literals in ``TCPConnector._resolve_host``
    and never calls a resolver for them, so literal blocked addresses are
    ``validate_url``'s job, not this one. This covers the hostname case, which
    is the one that rebinds.
    """

    def __init__(self, *, allow_private_ips: bool) -> None:
        self._inner = DefaultResolver()
        self._allow_private_ips = allow_private_ips
        self._proxy_hosts = _env_proxy_hosts()

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_INET,
    ) -> list[ResolveResult]:
        hosts = await self._inner.resolve(host, port, family)
        if self._allow_private_ips or host in self._proxy_hosts:
            return hosts
        allowed = [h for h in hosts if not is_blocked_ip(h["host"])]
        if not allowed:
            # ``host`` is caller-supplied and unbounded; this text reaches an
            # error response and a log line through the facade's connection
            # error, so bound it the way every other message on this path is.
            #
            # Two-argument OSError on purpose: aiohttp wraps this in
            # ClientConnectorDNSError and renders ``strerror``, so the
            # one-argument form reaches the caller as a bare "[None]" and the
            # reason is lost exactly where an operator would read it.
            raise SsrfBlockedAddress(
                errno.EHOSTUNREACH,
                f"host {describe_media_source(host)} resolves only to blocked IPs",
            )
        return allowed

    async def close(self) -> None:
        await self._inner.close()
