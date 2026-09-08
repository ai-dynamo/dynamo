# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connect-time SSRF backstop shared by the HTTP backends.

``validate_url`` resolves a hostname and checks its IPs against the blocklist,
but the HTTP backend re-resolves the name when it actually connects. A DNS
server the attacker controls can answer with a public IP during the check and
an internal one at connect (DNS rebinding), so the pre-check alone has a
time-of-check/time-of-use gap.

These shims re-apply the blocklist at resolve time, right before the socket is
dialed, so the backend never learns a blocked address. They mirror the Rust
frontend's ``BlocklistResolver`` (``lib/llm/src/preprocessor/media/loader.rs``),
which attaches the same filter to reqwest's ``dns_resolver`` — same division of
labour: ``validate_url`` is the pre-check, the resolver is the connect-time
backstop. Both reuse :func:`url_validator.is_blocked_ip`, so the two paths share
one blocklist.

``allow_private_ips`` is read from ``UrlValidationPolicy.from_env()`` at client
construction, matching how the Rust ``MediaFetcher::from_env`` builds its client
and how the backend callers build their per-request policy.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Any

from .url_validator import is_blocked_ip


class SsrfBlockedAddress(OSError):
    """Raised at connect time when every resolved IP is in a blocked range."""


def _filter_allowed(ips: list[str], *, allow_private_ips: bool) -> list[str]:
    if allow_private_ips:
        return ips
    return [ip for ip in ips if not is_blocked_ip(ip)]


try:  # aiohttp is the default backend; guard the import so httpx-only envs load.
    from aiohttp.abc import AbstractResolver
    from aiohttp.resolver import DefaultResolver

    class BlocklistResolver(AbstractResolver):
        """aiohttp resolver that drops blocked IPs before the connector dials.

        Wraps aiohttp's default resolver: resolve as usual, then filter the
        answers. If nothing survives, raise so the fetch fails closed instead of
        connecting to an internal address.
        """

        def __init__(self, *, allow_private_ips: bool) -> None:
            self._inner = DefaultResolver()
            self._allow_private_ips = allow_private_ips

        async def resolve(
            self, host: str, port: int = 0, family: int = socket.AF_INET
        ) -> list[dict[str, Any]]:
            hosts = await self._inner.resolve(host, port, family)
            if self._allow_private_ips:
                return hosts
            allowed = [h for h in hosts if not is_blocked_ip(h["host"])]
            if not allowed:
                raise SsrfBlockedAddress(
                    f"host {host!r} resolves only to blocked addresses"
                )
            return allowed

        async def close(self) -> None:
            await self._inner.close()

except ImportError:  # pragma: no cover - aiohttp always present in practice
    BlocklistResolver = None  # type: ignore[assignment,misc]


async def resolve_allowed_ip(host: str, *, allow_private_ips: bool) -> str:
    """Resolve ``host`` and return one non-blocked IP for a pinned connect.

    Used by the httpx backend, which has no resolver hook: we resolve here,
    filter, and hand the backend a specific IP to dial (with the original host
    preserved for SNI / ``Host`` / cert verification). An IP literal is checked
    directly without a lookup. Raises :class:`SsrfBlockedAddress` if nothing
    non-blocked remains.
    """
    # An IP literal needs no lookup: check it directly and dial it as-is.
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not allow_private_ips and is_blocked_ip(host):
            raise SsrfBlockedAddress(f"IP literal {host!r} is in a blocked range")
        return host

    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None)
    ips = [info[4][0] for info in infos]
    allowed = _filter_allowed(ips, allow_private_ips=allow_private_ips)
    if not allowed:
        raise SsrfBlockedAddress(f"host {host!r} resolves only to blocked addresses")
    return allowed[0]
