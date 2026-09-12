# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Facade for Dynamo's HTTP fetch client (aiohttp).

:func:`fetch_bytes` runs over a process-wide :class:`AiohttpClient` singleton.
Callers catch the unified exception classes (``HttpTimeoutError``,
``HttpConnectionError``, ``HttpStatusError``). For SSRF-safe fetches (e.g. from
``ImageLoader``) pass a ``UrlValidationPolicy`` — :func:`fetch_bytes` then
follows redirects manually and revalidates each hop against the policy.

``DYN_HTTP_BACKEND`` accepts only ``aiohttp``; any other value logs a warning
and uses aiohttp. aiohttp scales well under fan-out and exposes a native
DNS-resolver hook used for SSRF pinning (see README).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from dynamo.common.configuration.groups.http_args import (
    HttpArgGroup,
    HttpConfigBase,
    _apply_legacy_env_aliases,
    from_env,
)

from .aiohttp_client import AiohttpClient
from .base import (
    HttpClient,
    HttpConnectionError,
    HttpError,
    HttpStatusError,
    HttpTimeoutError,
)

logger = logging.getLogger(__name__)


_default: Optional[HttpClient] = None


def _create_client() -> HttpClient:
    """Instantiate the aiohttp client (the only supported backend).

    Mirrors any legacy ``DYN_MM_HTTP_*`` env vars to their canonical
    ``DYN_HTTP_*`` names first, so the client's lazy ``from_env()``
    inside ``HttpClient.__init__`` sees the migrated values. A stale
    ``DYN_HTTP_BACKEND`` other than ``aiohttp`` is ignored with a warning.
    """
    _apply_legacy_env_aliases()
    name = os.environ.get("DYN_HTTP_BACKEND", "aiohttp").lower()
    if name not in ("", "aiohttp"):
        logger.warning(
            "DYN_HTTP_BACKEND=%r is not supported; using aiohttp.",
            name,
        )
    return AiohttpClient()


def get_default_client() -> HttpClient:
    """Return the process-wide singleton client, instantiating on first call."""
    global _default
    if _default is None:
        _default = _create_client()
        logger.info("HTTP backend resolved: %s", type(_default).__name__)
    return _default


async def fetch_bytes(url, timeout, *, policy=None) -> bytes:
    """Singleton-backed convenience wrapper over :meth:`HttpClient.fetch_bytes`."""
    return await get_default_client().fetch_bytes(url, timeout, policy=policy)


async def close_http_client() -> None:
    """Close the active singleton. Idempotent. Safe across resets.

    Clears the resolved client so a fresh env-var reading happens on
    the next call (primarily useful in tests that vary
    ``DYN_HTTP_BACKEND``).
    """
    global _default
    if _default is None:
        return
    await _default.close()
    _default = None


__all__ = [
    "HttpClient",
    "AiohttpClient",
    "HttpError",
    "HttpTimeoutError",
    "HttpConnectionError",
    "HttpStatusError",
    "HttpConfigBase",
    "HttpArgGroup",
    "from_env",
    "fetch_bytes",
    "close_http_client",
    "get_default_client",
]
