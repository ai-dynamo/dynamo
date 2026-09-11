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

import socket
from typing import Any

from .url_validator import is_blocked_ip


class SsrfBlockedAddress(OSError):
    """Raised at connect time when every resolved IP is in a blocked range."""


try:  # guard the aiohttp import so the module still loads if aiohttp is absent.
    from aiohttp.abc import AbstractResolver
    from aiohttp.resolver import DefaultResolver

    class BlocklistResolver(AbstractResolver):
        """aiohttp resolver that drops blocked IPs before the connector dials.

        Returns the full set of non-blocked addresses (not just the first) so
        aiohttp keeps its normal multi-address / Happy-Eyeballs fallback.
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
                raise SsrfBlockedAddress(f"host {host!r} resolves only to blocked IPs")
            return allowed

        async def close(self) -> None:
            await self._inner.close()

except ImportError:  # pragma: no cover - aiohttp always present in practice
    BlocklistResolver = None  # type: ignore[assignment,misc]
