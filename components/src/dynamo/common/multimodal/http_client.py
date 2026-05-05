# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import asyncio
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# The shared client has ``follow_redirects=False`` to prevent redirect-based
# SSRF filter bypass. Callers must follow redirects manually via
# :func:`dynamo.common.multimodal.url_validator.fetch_with_revalidation` so
# that each hop is re-validated against the SSRF policy.
_global_http_client: Optional[httpx.AsyncClient] = None
_global_http_semaphore: Optional[asyncio.Semaphore] = None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None and raw != "" else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None and raw != "" else default


def get_http_client(timeout: float = 60.0) -> httpx.AsyncClient:
    """Return a shared async HTTP client for media fetches.

    The client intentionally disables automatic redirect following. Callers
    that need to follow redirects must route the request through
    :func:`fetch_with_revalidation`, which revalidates every redirect hop
    against the SSRF policy.

    Pool sizing and per-field timeouts are configurable via environment
    variables so operators can tune without patching source:

    - ``DYN_MM_HTTP_CONNECT_TIMEOUT`` (default 5s)
    - ``DYN_MM_HTTP_READ_TIMEOUT`` (default: value of ``timeout`` argument)
    - ``DYN_MM_HTTP_POOL_TIMEOUT`` (default 60s) — decoupled from read so a
      saturated pool surfaces quickly instead of waiting the read timeout.
    - ``DYN_MM_HTTP_MAX_CONNECTIONS`` (default 100)
    - ``DYN_MM_HTTP_MAX_KEEPALIVE`` (default 20)
    """
    global _global_http_client

    if _global_http_client is None or _global_http_client.is_closed:
        connect_timeout = _env_float("DYN_MM_HTTP_CONNECT_TIMEOUT", 5.0)
        read_timeout = _env_float("DYN_MM_HTTP_READ_TIMEOUT", timeout)
        pool_timeout = _env_float("DYN_MM_HTTP_POOL_TIMEOUT", 60.0)
        max_connections = _env_int("DYN_MM_HTTP_MAX_CONNECTIONS", 100)
        max_keepalive = _env_int("DYN_MM_HTTP_MAX_KEEPALIVE", 20)

        _global_http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=None,
                pool=pool_timeout,
            ),
            follow_redirects=False,
            limits=httpx.Limits(
                max_keepalive_connections=max_keepalive,
                max_connections=max_connections,
            ),
        )
        logger.info(
            "Shared HTTP client initialized ("
            "connect=%ss, read=%ss, write=None, pool=%ss, "
            "max_connections=%d, max_keepalive=%d, follow_redirects=False)",
            connect_timeout,
            read_timeout,
            pool_timeout,
            max_connections,
            max_keepalive,
        )

    return _global_http_client


def get_http_semaphore() -> asyncio.Semaphore:
    """Return the process-global semaphore that bounds in-flight media fetches.

    Acts as backpressure in front of :data:`_global_http_client`: caps the
    number of concurrent HTTP fetches across all media loaders (image, video,
    audio) so a burst of requests cannot saturate the pool and push
    ``PoolTimeout`` errors up the stack.

    Tunable via ``DYN_MM_HTTP_CONCURRENCY`` (default 50).
    """
    global _global_http_semaphore
    if _global_http_semaphore is None:
        bound = _env_int("DYN_MM_HTTP_CONCURRENCY", 50)
        _global_http_semaphore = asyncio.Semaphore(bound)
    return _global_http_semaphore
