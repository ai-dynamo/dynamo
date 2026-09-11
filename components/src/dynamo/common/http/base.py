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
from typing import AsyncIterator, Optional

from dynamo.common.configuration.groups.http_args import HttpConfigBase, from_env

from .url_validator import (
    _MAX_REDIRECTS,
    UrlValidationError,
    UrlValidationPolicy,
    describe_error_detail,
    describe_media_source,
    validate_url,
)

# Longest a client-supplied reason may render as inside an HttpStatusError.
# Matches ``dynamo.llm.exceptions.HttpError._MAX_MESSAGE_LENGTH``, which the
# binding applies to the other class it forwards on a 4xx. Generous on purpose:
# callers build real guidance here -- the trtllm decoder hint is ~480 characters
# and its test allows 2000 -- so a tight bound silently deletes the actionable
# part and leaves only a truncated prefix.
_MAX_MESSAGE_LENGTH = 8192


class HttpError(Exception):
    """Base class for all HTTP fetch failures."""


class HttpTimeoutError(HttpError):
    """Timeout during connect / read / pool-wait."""


class HttpConnectionError(HttpError):
    """Network-layer failure: DNS, refused, reset, half-close."""


class HttpStatusError(HttpError):
    """Server responded with a non-2xx status."""

    def __init__(self, status: int, message: str, url: str) -> None:
        # Both halves are client-supplied: ``url`` directly, and ``message``
        # because httpx's own HTTPStatusError text repeats the URL. The video
        # diffusion handler puts str(exc) in its response body.
        #
        # ``message`` is bounded in the *attribute*, not just in the rendered
        # string: errors.rs::extract_http_like_error reads ``.status`` and
        # ``.message`` off this class by name and, per the SECURITY note there,
        # forwards ``.message`` verbatim to the client on a 4xx. Nothing in the
        # rendered text reaches that path. ``url`` is not part of that protocol,
        # so it keeps its full value for debugging.
        message = describe_error_detail(message, _MAX_MESSAGE_LENGTH)
        super().__init__(f"HTTP {status} for {describe_media_source(url)}: {message}")
        self.status = status
        self.message = message
        self.url = url


async def collect_capped(
    chunks: AsyncIterator[bytes], url: str, max_bytes: Optional[int]
) -> bytes:
    """Join ``chunks`` into one body, refusing to buffer past ``max_bytes``.

    The check is per chunk rather than on the finished body: a declared
    Content-Length is attacker-controlled and absent on a chunked response, and
    a single capped read is not enough either — aiohttp's ``read(n)`` returns
    *at most* n bytes, so a short read would look like a body under the limit.
    """
    out: list[bytes] = []
    total = 0
    async for chunk in chunks:
        total += len(chunk)
        if max_bytes is not None and total > max_bytes:
            raise UrlValidationError(
                f"Media exceeds the {max_bytes} byte download limit: "
                f"{describe_media_source(url)}"
            )
        out.append(chunk)
    return b"".join(out)


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
        max_bytes: Optional[int] = None,
    ) -> bytes:
        """Fetch ``url`` and return the response body.

        Single-shot: no retries. Raises one of the unified exception
        classes above; callers never see native httpx/aiohttp classes.

        ``policy=None``: use the backend's built-in redirect handling.

        ``max_bytes`` set: refuse a body larger than that while it is being
        read, so an attacker-chosen URL cannot buffer an unbounded response.
        Raises :class:`UrlValidationError` — it is a verdict on a
        client-supplied source, like the redirect cap below.

        ``policy`` set: follow redirects manually and revalidate each
        hop against the policy via :func:`url_validator.validate_url`.
        This is the SSRF-safe path; raises :class:`UrlValidationError`
        if any hop fails or the chain exceeds ``_MAX_REDIRECTS``.
        """
        if policy is None:
            return await self._fetch_simple(url, timeout, max_bytes=max_bytes)
        return await self._fetch_with_revalidation(
            url, timeout, policy, max_bytes=max_bytes
        )

    async def _fetch_with_revalidation(
        self,
        url: str,
        timeout: float,
        policy: UrlValidationPolicy,
        *,
        max_bytes: Optional[int] = None,
    ) -> bytes:
        """Manual redirect loop with per-hop SSRF validation (backend-neutral)."""
        current = url
        hops_remaining = _MAX_REDIRECTS
        visited: list[str] = []
        while True:
            await validate_url(current, policy)
            visited.append(current)

            body, redirect_to = await self._fetch_body_or_redirect(
                current, timeout, max_bytes=max_bytes
            )

            if redirect_to is None:
                if body is None:
                    raise HttpError(
                        f"Backend returned (None, None) for {describe_media_source(current)}; "
                        "expected bytes on a terminal (2xx or 3xx-without-Location) response"
                    )
                return body

            if hops_remaining <= 0:
                # ``visited`` holds attacker-chosen URLs; describe each one.
                chain = [describe_media_source(hop) for hop in visited]
                raise UrlValidationError(
                    f"Too many redirects (max={_MAX_REDIRECTS}); chain={chain}"
                )
            hops_remaining -= 1
            current = redirect_to

    @abc.abstractmethod
    async def _fetch_simple(
        self, url: str, timeout: float, *, max_bytes: Optional[int] = None
    ) -> bytes:
        """Backend's native redirect-following GET (no SSRF policy applied)."""

    @abc.abstractmethod
    async def _fetch_body_or_redirect(
        self, url: str, timeout: float, *, max_bytes: Optional[int] = None
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
