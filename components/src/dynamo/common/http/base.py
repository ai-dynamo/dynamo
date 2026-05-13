# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class + unified exception classes for the HTTP facade.

Each concrete subclass (:class:`AiohttpClient`, :class:`HttpxClient`)
owns a backend-specific session/client singleton on the instance.
Backend-neutral logic (the SSRF redirect loop) lives here so it isn't
duplicated across subclasses.
"""

from __future__ import annotations

import abc
import asyncio
from typing import Optional

from dynamo.common.configuration.groups.http_args import HttpConfigBase, from_env

from .url_validator import (
    _MAX_REDIRECTS,
    UrlValidationError,
    UrlValidationPolicy,
    validate_url,
)


class HttpError(Exception):
    """Base class for all HTTP fetch failures."""


class HttpTimeoutError(HttpError):
    """Timeout during connect / read / pool-wait."""


class HttpConnectionError(HttpError):
    """Network-layer failure: DNS, refused, reset, half-close."""


class HttpStatusError(HttpError):
    """Server responded with a non-2xx status."""

    def __init__(self, status: int, message: str, url: str) -> None:
        super().__init__(f"HTTP {status} for {url}: {message}")
        self.status = status
        self.message = message
        self.url = url


class HttpClient(abc.ABC):
    """Backend-neutral HTTP client.

    Subclasses own a backend-specific session/client singleton on the
    instance. Callers reach the public surface via :meth:`fetch_bytes`
    and the unified exception classes above; the concrete backend type
    is invisible past instantiation.
    """

    def __init__(self, config: Optional[HttpConfigBase] = None) -> None:
        self._config: HttpConfigBase = config if config is not None else from_env()
        self._lock = asyncio.Lock()

    async def fetch_bytes(
        self,
        url: str,
        timeout: float,
        *,
        policy: Optional[UrlValidationPolicy] = None,
    ) -> bytes:
        """Fetch ``url`` and return the response body.

        Single-shot: no retries. Raises one of the unified exception
        classes above; callers never see native httpx/aiohttp classes.

        ``policy=None``: use the backend's built-in redirect handling.

        ``policy`` set: follow redirects manually and revalidate each
        hop against the policy via :func:`url_validator.validate_url`.
        This is the SSRF-safe path; raises :class:`UrlValidationError`
        if any hop fails or the chain exceeds ``_MAX_REDIRECTS``.
        """
        if policy is None:
            return await self._fetch_simple(url, timeout)
        return await self._fetch_with_revalidation(url, timeout, policy)

    async def _fetch_with_revalidation(
        self,
        url: str,
        timeout: float,
        policy: UrlValidationPolicy,
    ) -> bytes:
        """Manual redirect loop with per-hop SSRF validation (backend-neutral)."""
        current = url
        hops_remaining = _MAX_REDIRECTS
        visited: list[str] = []
        while True:
            await validate_url(current, policy)
            visited.append(current)

            body, redirect_to = await self._fetch_body_or_redirect(current, timeout)

            if redirect_to is None:
                if body is None:
                    raise HttpError(
                        f"Backend returned (None, None) for {current}; "
                        "expected bytes on a terminal (2xx or 3xx-without-Location) response"
                    )
                return body

            if hops_remaining <= 0:
                raise UrlValidationError(
                    f"Too many redirects (max={_MAX_REDIRECTS}); chain={visited}"
                )
            hops_remaining -= 1
            current = redirect_to

    @abc.abstractmethod
    async def _fetch_simple(self, url: str, timeout: float) -> bytes:
        """Backend's native redirect-following GET (no SSRF policy applied)."""

    @abc.abstractmethod
    async def _fetch_body_or_redirect(
        self, url: str, timeout: float
    ) -> tuple[bytes | None, str | None]:
        """Single hop with redirects disabled.

        Subclass contract used by :meth:`_fetch_with_revalidation`.
        Returns ``(body, None)`` for a terminal response (2xx, or 3xx
        without a ``Location`` header), or ``(None, absolute_next_url)``
        for a followable redirect. Raises :class:`HttpStatusError`
        for 4xx/5xx and the usual timeout / connection classes on
        transport failure.
        """

    @abc.abstractmethod
    async def close(self) -> None:
        """Close the backend session/client. Idempotent."""
