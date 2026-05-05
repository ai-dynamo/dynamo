# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aiohttp implementation of :class:`MmHttpClient` — the default backend.

See ``README.md`` (sibling file) for why aiohttp is the default and
the perf data behind that choice. Operator-tunable knobs live in
:mod:`.args`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp
from yarl import URL

from .base import MmHttpClient, MmHttpConnectionError, MmHttpStatusError, MmHttpTimeout

logger = logging.getLogger(__name__)


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class AiohttpClient(MmHttpClient):
    """aiohttp-backed concrete client."""

    def __init__(self, args=None) -> None:
        super().__init__(args)
        self._session: Optional[aiohttp.ClientSession] = None

    def _effective_timeout(self, timeout: float) -> aiohttp.ClientTimeout:
        total = (
            self._args.read_timeout_override
            if self._args.read_timeout_override is not None
            else timeout
        )
        return aiohttp.ClientTimeout(total=total)

    def _build_session(self) -> aiohttp.ClientSession:
        connector = aiohttp.TCPConnector(
            limit=self._args.max_connections,
            # Single-origin fan-out is the whole point; capping per-host
            # would defeat it. Hard-coded rather than env-tunable.
            limit_per_host=0,
            keepalive_timeout=self._args.keepalive_timeout,
            enable_cleanup_closed=True,
        )
        return aiohttp.ClientSession(connector=connector, trust_env=True)

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._lock:
            if self._session is None or self._session.closed:
                self._session = self._build_session()
                logger.info(
                    "aiohttp backend initialized: limit=%d, limit_per_host=0, "
                    "keepalive_timeout=%.1fs%s",
                    self._args.max_connections,
                    self._args.keepalive_timeout,
                    f", read timeout forced to {self._args.read_timeout_override:.1f}s via env"
                    if self._args.read_timeout_override is not None
                    else "; total timeout set per-request",
                )
        return self._session

    async def _fetch_simple(self, url: str, timeout: float) -> bytes:
        session = await self._get_session()
        client_timeout = self._effective_timeout(timeout)
        try:
            async with session.get(
                url, timeout=client_timeout, allow_redirects=True
            ) as response:
                response.raise_for_status()
                return await response.read()
        except aiohttp.ClientResponseError as e:
            raise MmHttpStatusError(e.status, e.message or "", url) from e
        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError) as e:
            raise MmHttpTimeout(f"Timeout loading {url}") from e
        except (
            aiohttp.ClientConnectionError,
            aiohttp.ClientConnectorError,
            aiohttp.ServerDisconnectedError,
        ) as e:
            raise MmHttpConnectionError(f"Connection error loading {url}: {e}") from e
        except aiohttp.ClientError as e:
            raise MmHttpConnectionError(f"HTTP error loading {url}: {e}") from e

    async def fetch_body_or_redirect(
        self, url: str, timeout: float
    ) -> tuple[bytes | None, str | None]:
        session = await self._get_session()
        client_timeout = self._effective_timeout(timeout)
        try:
            async with session.get(
                url, timeout=client_timeout, allow_redirects=False
            ) as response:
                if response.status in _REDIRECT_STATUSES:
                    location = response.headers.get("Location")
                    if location:
                        next_url = str(response.url.join(URL(location)))
                        return None, next_url
                    return await response.read(), None

                try:
                    response.raise_for_status()
                except aiohttp.ClientResponseError as e:
                    raise MmHttpStatusError(e.status, e.message or "", url) from e
                return await response.read(), None
        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError) as e:
            raise MmHttpTimeout(f"Timeout loading {url}") from e
        except (
            aiohttp.ClientConnectionError,
            aiohttp.ClientConnectorError,
            aiohttp.ServerDisconnectedError,
        ) as e:
            raise MmHttpConnectionError(f"Connection error loading {url}: {e}") from e
        except aiohttp.ClientError as e:
            raise MmHttpConnectionError(f"HTTP error loading {url}: {e}") from e

    async def close(self) -> None:
        async with self._lock:
            if self._session is not None and not self._session.closed:
                await self._session.close()
            self._session = None
