# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for ``DYN_MM_HTTP_*`` env-var parsing.

Both backends import :class:`HttpArgs` and call :func:`from_env`; each
consumes the fields that apply to it. Overlapping knobs
(``MAX_CONNECTIONS``, ``READ_TIMEOUT``) live as one field so they don't
get parsed twice with drift-prone defaults.

Operator-tunable env vars
-------------------------

Shared (consumed by both backends):

  - ``DYN_MM_HTTP_MAX_CONNECTIONS`` (default 100) — total pool size cap.
  - ``DYN_MM_HTTP_READ_TIMEOUT`` (unset by default) — when set, replaces
    the caller's per-call ``timeout`` arg on every fetch. Useful as a
    global ceiling without editing call sites.

httpx-only:

  - ``DYN_MM_HTTP_MAX_KEEPALIVE`` (default = ``MAX_CONNECTIONS``) — cap
    on idle keepalive connections kept warm.
  - ``DYN_MM_HTTP_CONNECT_TIMEOUT`` (default 5.0s) — TCP-connect budget.
  - ``DYN_MM_HTTP_POOL_TIMEOUT`` (default 60.0s) — wait-for-free-slot
    budget. Decoupled from read so a saturated pool surfaces fast.
  - ``DYN_MM_HTTP_CONCURRENCY`` (default 50) — process-wide cap on
    concurrent in-flight HTTP fetches. Acts as backpressure in front
    of the httpx pool so a burst can't push ``PoolTimeout`` up the
    stack. aiohttp has no equivalent because its connector queues
    natively.

aiohttp-only:

  - ``DYN_MM_HTTP_KEEPALIVE_TIMEOUT`` (default 15.0s) — how long an
    idle connection stays warm in the pool.

Fields on :class:`HttpArgs` are grouped by which backend consumes them
— see the section comments inside the class.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None and raw != "" else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None and raw != "" else default


@dataclass(frozen=True)
class HttpArgs:
    """Operator-tunable knobs for the multimodal HTTP backends.

    Frozen so a session that captured the args at startup never sees
    mid-process drift. Use :func:`from_env` to construct.
    """

    # --- Shared (consumed by httpx + aiohttp) ----------------------------

    # Total pool size cap.
    #   httpx: ``Limits.max_connections``
    #   aiohttp: ``TCPConnector.limit``
    # Bigger pool → more concurrent connections allowed, at the cost of more
    # open file descriptors.
    max_connections: int

    # Override for the per-call ``timeout`` arg passed to ``fetch_bytes``.
    # ``None`` → caller's value wins (the common case). When set, every
    # fetch uses this value regardless of what the caller asked for —
    # useful for forcing a global ceiling without editing call sites.
    #   httpx: read budget (component of ``Timeout``)
    #   aiohttp: total budget (``ClientTimeout.total``)
    read_timeout_override: Optional[float]

    # --- httpx-only -------------------------------------------------------

    # Cap on idle keepalive connections kept warm in the pool. Raising
    # this to match ``max_connections`` prevents TLS re-handshake churn
    # under fan-out. Maps to ``Limits.max_keepalive_connections``.
    # Default = ``max_connections``.
    max_keepalive: int

    # TCP-connect timeout in seconds; ``Timeout.connect``. Default 5.0.
    connect_timeout: float

    # How long to wait for a free pool slot before raising
    # ``PoolTimeout``; ``Timeout.pool``. Decoupled from the read budget
    # so a saturated pool surfaces quickly. Default 60.0.
    pool_timeout: float

    # Process-wide cap on concurrent in-flight HTTP fetches via the
    # httpx backend. The semaphore acts as backpressure in front of the
    # pool so a burst of requests can't push ``PoolTimeout`` up the
    # stack. aiohttp has no equivalent because its connector queues
    # natively. Default 50.
    concurrency: int

    # --- aiohttp-only -----------------------------------------------------

    # How long an idle connection stays warm in the pool, in seconds;
    # ``TCPConnector.keepalive_timeout``. Larger value rides through
    # request bursts without re-establishing the connection. Default 15.0.
    keepalive_timeout: float


def from_env() -> HttpArgs:
    """Build an :class:`HttpArgs` from the current environment."""
    max_connections = _env_int("DYN_MM_HTTP_MAX_CONNECTIONS", 100)
    raw_read = _env_float("DYN_MM_HTTP_READ_TIMEOUT", -1.0)
    return HttpArgs(
        # shared
        max_connections=max_connections,
        read_timeout_override=raw_read if raw_read >= 0 else None,
        # httpx-only
        max_keepalive=_env_int("DYN_MM_HTTP_MAX_KEEPALIVE", max_connections),
        connect_timeout=_env_float("DYN_MM_HTTP_CONNECT_TIMEOUT", 5.0),
        pool_timeout=_env_float("DYN_MM_HTTP_POOL_TIMEOUT", 60.0),
        concurrency=_env_int("DYN_MM_HTTP_CONCURRENCY", 50),
        # aiohttp-only
        keepalive_timeout=_env_float("DYN_MM_HTTP_KEEPALIVE_TIMEOUT", 15.0),
    )
