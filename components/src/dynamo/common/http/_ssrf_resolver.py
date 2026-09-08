# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connect-time SSRF backstop shared by the HTTP backends.

``validate_url`` checks a hostname's resolved IPs, but the backend re-resolves
at connect, so a DNS-rebinding server can return a public IP on the check and an
internal one at connect. These shims re-apply the blocklist at resolve time,
mirroring the Rust frontend's ``BlocklistResolver``
(``lib/llm/src/preprocessor/media/loader.rs``). Both reuse
:func:`url_validator.is_blocked_ip`.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Any

from .url_validator import is_blocked_ip


class SsrfBlockedAddress(OSError):
    """Raised at connect time when every resolved IP is in a blocked range."""


try:  # aiohttp is the default backend; httpx-only envs still import this module.
    from aiohttp.abc import AbstractResolver
    from aiohttp.resolver import DefaultResolver

    class BlocklistResolver(AbstractResolver):
        """aiohttp resolver that drops blocked IPs before the connector dials."""

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
                raise SsrfBlockedAddress(f"host {host!r} resolves only to blocked IPs")
            return allowed

        async def close(self) -> None:
            await self._inner.close()

except ImportError:  # pragma: no cover - aiohttp always present in practice
    BlocklistResolver = None  # type: ignore[assignment,misc]


async def resolve_allowed_ip(host: str, *, allow_private_ips: bool) -> str:
    """Resolve ``host`` to one non-blocked IP for a pinned connect (httpx path).

    httpx has no resolver hook, so we resolve + filter here and hand the backend
    a specific IP to dial. An IP literal is checked without a lookup. Raises
    :class:`SsrfBlockedAddress` if nothing non-blocked remains.
    """
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not allow_private_ips and is_blocked_ip(host):
            raise SsrfBlockedAddress(f"IP literal {host!r} is in a blocked range")
        return host

    infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    for info in infos:
        ip = info[4][0]
        if allow_private_ips or not is_blocked_ip(ip):
            return ip
    raise SsrfBlockedAddress(f"host {host!r} resolves only to blocked IPs")
