# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral facade for Dynamo's HTTP fetch client.

Env var ``DYN_HTTP_BACKEND`` selects the implementation behind the
process-wide singleton used by :func:`fetch_bytes`:

  - ``aiohttp`` (default) — :class:`AiohttpClient` over an
    ``aiohttp.ClientSession``.
  - ``httpx`` — :class:`HttpxClient` over an ``httpx.AsyncClient``
    (fixed config: decoupled pool timeout, ``max_keepalive_connections``
    matches ``max_connections``).

Callers use :func:`fetch_bytes` and catch the unified exception classes
(``HttpTimeoutError``, ``HttpConnectionError``, ``HttpStatusError``).
For SSRF-safe fetches (e.g. from ``ImageLoader``) pass a
``UrlValidationPolicy`` — :func:`fetch_bytes` then follows redirects
manually and revalidates each hop against the policy.

Tests / advanced callers can also instantiate :class:`AiohttpClient` or
:class:`HttpxClient` directly and bypass the singleton.
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
from .httpx_client import HttpxClient

logger = logging.getLogger(__name__)


_VALID_BACKENDS = ("aiohttp", "httpx")
_default: Optional[HttpClient] = None


def _create_client() -> HttpClient:
    """Pick a concrete client class based on ``DYN_HTTP_BACKEND``.

    Mirrors any legacy ``DYN_MM_HTTP_*`` env vars to their canonical
    ``DYN_HTTP_*`` names first, so the client's lazy ``from_env()``
    inside ``HttpClient.__init__`` sees the migrated values.
    """
    _apply_legacy_env_aliases()
    name = os.environ.get("DYN_HTTP_BACKEND", "aiohttp").lower()
    if name == "aiohttp":
        return AiohttpClient()
    if name == "httpx":
        return HttpxClient()
    raise ValueError(
        f"DYN_HTTP_BACKEND={name!r} is invalid; must be one of {_VALID_BACKENDS}"
    )


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
    "HttpxClient",
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
