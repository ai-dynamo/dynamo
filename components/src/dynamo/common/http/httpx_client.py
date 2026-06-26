# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""httpx implementation of :class:`HttpClient`.

Behavior notes (operator-tunable knobs live in :mod:`.args`):

  - ``httpx.Timeout`` is built per-request so each fetch sees its
    caller-supplied read timeout instead of the first caller's value
    getting baked into the singleton. ``write`` is unbounded — fan-out
    GET workloads have no meaningful write phase.
  - ``max_keepalive_connections`` is sized to match ``max_connections``
    so idle connections get reused under fan-out instead of churning
    TLS handshakes.
  - A process-wide semaphore (``DYN_HTTP_CONCURRENCY``) wraps every
    fetch so a burst can't push ``PoolTimeout`` up the stack. aiohttp
    has no equivalent because its connector queues natively.

An operator who flips ``DYN_HTTP_BACKEND=httpx`` gets a working backend,
not the original PoolTimeout bug.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from .base import HttpClient, HttpConnectionError, HttpStatusError, HttpTimeoutError

logger = logging.getLogger(__name__)


class HttpxClient(HttpClient):
    """httpx-backed concrete client."""

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Process-wide concurrency cap on in-flight httpx fetches.

        Lazy-init so the bound (``DYN_HTTP_CONCURRENCY``) is read once
        and the ``asyncio.Semaphore`` binds to whichever loop first
        awaits it.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._config.concurrency)
        return self._semaphore

    def _per_call_timeout(self, read_timeout: float) -> httpx.Timeout:
        # Override targets ``read`` only — ``connect`` / ``pool`` stay
        # independent so a stuck handshake or saturated pool still
        # fast-fails on its own budget.
        effective_read = (
            self._config.per_call_timeout_override
            if self._config.per_call_timeout_override is not None
            else read_timeout
        )
        return httpx.Timeout(
            connect=self._config.connect_timeout,
            read=effective_read,
            write=None,
            pool=self._config.pool_timeout,
        )

    def _build_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._per_call_timeout(60.0),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=self._config.max_connections,
                max_keepalive_connections=self._config.max_keepalive,
            ),
        )

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._lock:
            if self._client is None or self._client.is_closed:
                self._client = self._build_client()
                logger.info(
                    "httpx backend initialized: max_connections=%d, max_keepalive=%d, "
                    "timeout(connect=%.1fs, write=None, pool=%.1fs); read timeout %s",
                    self._config.max_connections,
                    self._config.max_keepalive,
                    self._config.connect_timeout,
                    self._config.pool_timeout,
                    f"forced to {self._config.per_call_timeout_override:.1f}s via env"
                    if self._config.per_call_timeout_override is not None
                    else "set per-request",
                )
        return self._client

    async def _fetch_simple(self, url: str, timeout: float) -> bytes:
        client = await self._get_client()
        async with self._get_semaphore():
            try:
                response = await client.get(
                    url, timeout=self._per_call_timeout(timeout)
                )
                response.raise_for_status()
                return response.content
            except httpx.HTTPStatusError as e:
                raise HttpStatusError(e.response.status_code, str(e), url) from e
            except httpx.TimeoutException as e:
                raise HttpTimeoutError(f"Timeout loading {url}") from e
            except (httpx.ConnectError, httpx.NetworkError) as e:
                raise HttpConnectionError(f"Connection error loading {url}: {e}") from e
            except httpx.HTTPError as e:
                raise HttpConnectionError(f"HTTP error loading {url}: {e}") from e

    async def _fetch_body_or_redirect(
        self, url: str, timeout: float
    ) -> tuple[bytes | None, str | None]:
        client = await self._get_client()
        async with self._get_semaphore():
            try:
                request = client.build_request(
                    "GET", url, timeout=self._per_call_timeout(timeout)
                )
                response = await client.send(
                    request,
                    follow_redirects=False,
                )
            except httpx.TimeoutException as e:
                raise HttpTimeoutError(f"Timeout loading {url}") from e
            except (httpx.ConnectError, httpx.NetworkError) as e:
                raise HttpConnectionError(f"Connection error loading {url}: {e}") from e
            except httpx.HTTPError as e:
                raise HttpConnectionError(f"HTTP error loading {url}: {e}") from e

            try:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if location:
                        next_url = str(response.url.join(location))
                        return None, next_url
                    # 3xx without Location: treat as terminal, surface the body.
                    return response.content, None

                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    raise HttpStatusError(e.response.status_code, str(e), url) from e
                return response.content, None
            finally:
                await response.aclose()

    async def close(self) -> None:
        async with self._lock:
            if self._client is not None and not self._client.is_closed:
                await self._client.aclose()
            self._client = None
            self._semaphore = None
