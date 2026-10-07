# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aiohttp implementation of :class:`HttpClient` — the default backend.

See ``README.md`` (sibling file) for why aiohttp is the default and
the perf data behind that choice. Operator-tunable knobs live in
:mod:`.args`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import aiohttp
from aiohttp.helpers import get_env_proxy_for_url
from yarl import URL

from ._ssrf_resolver import BlocklistResolver
from .base import (
    HttpClient,
    HttpConfigurationError,
    HttpConnectionError,
    HttpStatusError,
    HttpTimeoutError,
    collect_capped,
)
from .url_validator import (
    UrlValidationPolicy,
    describe_error_detail,
    describe_media_source,
)

logger = logging.getLogger(__name__)


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

# Read granularity for the capped reader. Chunks are joined, so this only
# bounds how far past the limit a single read can carry.
_READ_CHUNK = 64 * 1024

# aiohttp's default ``max_redirects``, kept for a fetch without a URL policy.
_MAX_SIMPLE_REDIRECTS = 10

# The redirect targets that aiohttp follows: http(s), or a relative reference.
_REDIRECT_SCHEMES = frozenset({"http", "https", ""})

# Set to "1" to assert that the configured egress proxy enforces destination
# policy itself. Spelled like DYN_MM_ALLOW_INTERNAL, which it sits beside.
DYN_MM_TRUST_EGRESS_PROXY = "DYN_MM_TRUST_EGRESS_PROXY"


def _redirect_target(base: URL, location: str) -> str:
    """The next hop of a redirect without a URL policy, checked like aiohttp.

    aiohttp followed these redirects before the client did. A target that it
    refused raises the same client error here, so the fetch reports it as
    HttpConnectionError, as before.
    """
    try:
        target = URL(location)
    except ValueError as e:
        raise aiohttp.InvalidUrlRedirectClientError(
            location,
            "Server attempted redirecting to a location that does not look like a URL",
        ) from e
    if target.scheme not in _REDIRECT_SCHEMES:
        raise aiohttp.NonHttpUrlRedirectClientError(location)
    target = base.join(target)
    try:
        target.origin()
    except ValueError as e:
        raise aiohttp.InvalidUrlRedirectClientError(
            target, "Invalid redirect URL origin"
        ) from e
    return str(target)


class AiohttpClient(HttpClient):
    """aiohttp-backed concrete client."""

    def __init__(self, config=None) -> None:
        super().__init__(config)
        # Keyed by (whether the connector may return private addresses, whether
        # it dials an egress proxy). A request that asks for a stricter policy
        # than the deployment baseline gets a connector that actually enforces
        # it, and only a proxied fetch gets the connector that resolves the
        # proxy unfiltered. At most three entries: a permissive connector
        # filters nothing, so it is never split by proxy.
        self._sessions: dict[tuple[bool, bool], aiohttp.ClientSession] = {}
        # aiohttp marks an injected resolver as externally owned, so closing a
        # session never closes it. Hold each one and close it ourselves.
        self._resolvers: dict[tuple[bool, bool], BlocklistResolver] = {}
        # One cookie jar per connect policy, shared by its direct and proxied
        # sessions. A redirect between them then keeps its cookies, as it did
        # when one session served both.
        self._cookie_jars: dict[bool, aiohttp.CookieJar] = {}

    def _effective_timeout(
        self, timeout: float, read_timeout: Optional[float] = None
    ) -> aiohttp.ClientTimeout:
        # The override caps ``total`` (whole request). ``sock_connect``
        # bounds just the TCP+TLS handshake, so a stuck origin fast-fails at
        # the connect budget instead of burning the full ``total`` before any
        # byte arrives. ``sock_read`` fails a server that stops sending, and
        # the override does not change it.
        total = (
            self._config.per_call_timeout_override
            if self._config.per_call_timeout_override is not None
            else timeout
        )
        return aiohttp.ClientTimeout(
            total=total,
            sock_connect=self._config.connect_timeout,
            sock_read=read_timeout,
        )

    @staticmethod
    def _connect_allows_private(policy: Optional[UrlValidationPolicy]) -> bool:
        """Whether the connector may return private addresses for this fetch.

        The intersection of the deployment baseline and the request policy.
        Taking the baseline alone lets ``DYN_MM_ALLOW_INTERNAL=1`` override a
        caller that explicitly asked for ``allow_private_ips=False``, and
        taking the request alone lets any caller opt out of the deployment
        setting. Only when both allow it does the backstop stand down.
        """
        env_allows = UrlValidationPolicy.from_env().allow_private_ips
        if policy is None:
            return env_allows
        return env_allows and policy.allow_private_ips

    @staticmethod
    async def _require_trusted_egress_proxy(url: str) -> bool:
        """Refuse a protected fetch that a proxy would put out of our reach.

        When a proxy applies, the connector dials the proxy and the proxy
        resolves the origin, so the origin hostname never reaches the
        connect-time check. Measured: the resolver sees only the proxy host,
        while the proxy receives the absolute URI. Documenting that boundary
        does not enforce it, so fail closed unless an operator asserts that
        the proxy enforces destination policy itself.

        Asks aiohttp which proxy applies to *this* URL rather than whether any
        proxy variable is set, so ``NO_PROXY`` is honored and a fetch that
        would go direct is not refused.

        Returns whether a trusted proxy carries the fetch, so the caller can
        give only that fetch the connector that resolves the proxy unfiltered.
        """
        try:
            # aiohttp runs this same helper through asyncio.to_thread because
            # it does proxy-bypass discovery and .netrc file reads. Match that
            # rather than repeating the blocking work on the event loop.
            await asyncio.to_thread(get_env_proxy_for_url, URL(url))
        except LookupError:
            # No proxy for this URL, so aiohttp dials the origin and the
            # connect-time check governs it.
            return False
        if os.getenv(DYN_MM_TRUST_EGRESS_PROXY, "").strip() == "1":
            return True
        raise HttpConfigurationError(
            f"{describe_media_source(url)} would be fetched through an egress "
            "proxy, so the connect-time address check cannot govern the "
            f"origin; set {DYN_MM_TRUST_EGRESS_PROXY}=1 to assert that the "
            "proxy enforces destination policy, or unset the proxy"
        )

    def _build_session(
        self, allow_private_ips: bool, via_proxy: bool
    ) -> aiohttp.ClientSession:
        resolver = BlocklistResolver(
            allow_private_ips=allow_private_ips, via_proxy=via_proxy
        )
        self._resolvers[(allow_private_ips, via_proxy)] = resolver
        connector = aiohttp.TCPConnector(
            limit=self._config.max_connections,
            # Single-origin fan-out is the whole point; capping per-host
            # would defeat it. Hard-coded rather than env-tunable.
            limit_per_host=0,
            keepalive_timeout=self._config.keepalive_timeout,
            enable_cleanup_closed=True,
            # Connect-time SSRF backstop against DNS rebinding (see
            # _ssrf_resolver). See _connect_allows_private for how the flag is
            # derived. DYN_MM_ALLOW_INTERNAL is the deployment knob.
            resolver=resolver,
        )
        cookie_jar = self._cookie_jars.get(allow_private_ips)
        if cookie_jar is None:
            cookie_jar = self._cookie_jars[allow_private_ips] = aiohttp.CookieJar()
        return aiohttp.ClientSession(
            connector=connector, trust_env=True, cookie_jar=cookie_jar
        )

    async def _get_session(
        self, allow_private_ips: bool, via_proxy: bool = False
    ) -> aiohttp.ClientSession:
        key = (allow_private_ips, via_proxy)
        async with self._lock:
            session = self._sessions.get(key)
            if session is None or session.closed:
                session = self._build_session(allow_private_ips, via_proxy)
                self._sessions[key] = session
                logger.info(
                    "aiohttp backend initialized: limit=%d, limit_per_host=0, "
                    "keepalive_timeout=%.1fs%s",
                    self._config.max_connections,
                    self._config.keepalive_timeout,
                    f", total timeout forced to {self._config.per_call_timeout_override:.1f}s via env"
                    if self._config.per_call_timeout_override is not None
                    else "; total timeout set per-request",
                )
        return session

    async def _fetch_simple(
        self,
        url: str,
        timeout: float,
        *,
        max_bytes: Optional[int] = None,
        policy: Optional[UrlValidationPolicy] = None,
        read_timeout: Optional[float] = None,
    ) -> bytes:
        allow_private = self._connect_allows_private(policy)
        client_timeout = self._effective_timeout(timeout, read_timeout)
        # Follow redirects here, not in aiohttp. aiohttp keeps the first hop's
        # session for every hop, so a hop that NO_PROXY sends direct could use
        # the resolver that exempts the proxy. Each hop runs the proxy gate and
        # picks its own session, and one deadline covers all hops.
        loop = asyncio.get_running_loop()
        started: Optional[float] = None
        current = url
        redirects = 0
        try:
            while True:
                via_proxy = False
                # Only when the check is meant to bite. If private destinations
                # are already permitted for this fetch, the gate protects nothing.
                if not allow_private:
                    via_proxy = await self._require_trusted_egress_proxy(current)
                session = await self._get_session(allow_private, via_proxy)
                hop_timeout = client_timeout
                if started is None:
                    # The budget starts with the first request, as it did when
                    # aiohttp followed the redirects.
                    started = loop.time()
                elif client_timeout.total is not None:
                    remaining = client_timeout.total - (loop.time() - started)
                    if remaining <= 0:
                        raise asyncio.TimeoutError()
                    hop_timeout = aiohttp.ClientTimeout(
                        total=remaining,
                        sock_connect=client_timeout.sock_connect,
                        sock_read=client_timeout.sock_read,
                    )
                async with session.get(
                    current, timeout=hop_timeout, allow_redirects=False
                ) as response:
                    # aiohttp also took the obsolete URI header.
                    headers = response.headers
                    location = headers.get("Location") or headers.get("URI")
                    if response.status in _REDIRECT_STATUSES and location:
                        redirects += 1
                        if redirects >= _MAX_SIMPLE_REDIRECTS:
                            # What aiohttp raised when it followed redirects.
                            raise aiohttp.TooManyRedirects(
                                response.request_info, (response,)
                            )
                        current = _redirect_target(response.url, location)
                        continue
                    response.raise_for_status()
                    return await collect_capped(
                        response.content.iter_chunked(_READ_CHUNK), url, max_bytes
                    )
        except aiohttp.ClientResponseError as e:
            raise HttpStatusError(e.status, e.message or "", url) from e
        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError) as e:
            raise HttpTimeoutError(
                f"Timeout loading {describe_media_source(url)}"
            ) from e
        except (
            aiohttp.ClientConnectionError,
            aiohttp.ClientConnectorError,
            aiohttp.ServerDisconnectedError,
        ) as e:
            raise HttpConnectionError(
                f"Connection error loading {describe_media_source(url)}: {describe_error_detail(str(e))}"
            ) from e
        except aiohttp.ClientError as e:
            raise HttpConnectionError(
                f"HTTP error loading {describe_media_source(url)}: {describe_error_detail(str(e))}"
            ) from e

    async def _fetch_body_or_redirect(
        self,
        url: str,
        timeout: float,
        *,
        max_bytes: Optional[int] = None,
        policy: Optional[UrlValidationPolicy] = None,
        read_timeout: Optional[float] = None,
    ) -> tuple[bytes | None, str | None]:
        allow_private = self._connect_allows_private(policy)
        via_proxy = False
        # Only when the check is meant to bite. If private destinations are
        # already permitted for this fetch, the proxy gate protects nothing.
        if not allow_private:
            via_proxy = await self._require_trusted_egress_proxy(url)
        session = await self._get_session(allow_private, via_proxy)
        client_timeout = self._effective_timeout(timeout, read_timeout)
        try:
            async with session.get(
                url, timeout=client_timeout, allow_redirects=False
            ) as response:
                if response.status in _REDIRECT_STATUSES:
                    location = response.headers.get("Location")
                    if location:
                        next_url = str(response.url.join(URL(location)))
                        return None, next_url
                    return (
                        await collect_capped(
                            response.content.iter_chunked(_READ_CHUNK), url, max_bytes
                        ),
                        None,
                    )

                try:
                    response.raise_for_status()
                except aiohttp.ClientResponseError as e:
                    raise HttpStatusError(e.status, e.message or "", url) from e
                return (
                    await collect_capped(
                        response.content.iter_chunked(_READ_CHUNK), url, max_bytes
                    ),
                    None,
                )
        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError) as e:
            raise HttpTimeoutError(
                f"Timeout loading {describe_media_source(url)}"
            ) from e
        except (
            aiohttp.ClientConnectionError,
            aiohttp.ClientConnectorError,
            aiohttp.ServerDisconnectedError,
        ) as e:
            raise HttpConnectionError(
                f"Connection error loading {describe_media_source(url)}: {describe_error_detail(str(e))}"
            ) from e
        except aiohttp.ClientError as e:
            raise HttpConnectionError(
                f"HTTP error loading {describe_media_source(url)}: {describe_error_detail(str(e))}"
            ) from e

    async def close(self) -> None:
        async with self._lock:
            for session in self._sessions.values():
                if not session.closed:
                    await session.close()
            self._sessions = {}
            # TCPConnector sets _resolver_owner=False for an injected resolver
            # and only closes the one it made itself, so closing the session
            # above leaves ours open.
            for resolver in self._resolvers.values():
                await resolver.close()
            self._resolvers = {}
            self._cookie_jars = {}
