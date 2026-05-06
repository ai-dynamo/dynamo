# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for ``DYN_MM_HTTP_*`` configuration.

Both backends import :class:`HttpConfigBase` and call :func:`from_env`;
each consumes the fields that apply to it. The class extends the shared
:class:`dynamo.common.configuration.config_base.ConfigBase`, and
:class:`HttpArgGroup` registers the matching CLI flags + ``DYN_MM_HTTP_*``
env vars via :func:`dynamo.common.configuration.utils.add_argument` so
this group plays nicely with components that build their own argparse
surface (frontend, planner, etc.).

Why this group ships :func:`from_env` while sibling groups don't:
the HTTP client's primary callers — ``image_loader.py`` /
``audio_loader.py`` / ``video_loader.py`` — don't own an argparse,
so they need an env-only construction path. The
``dynamo.common.http`` singleton boots through this entry point. Other
groups (router, kv-router, runtime) are always materialized off a
parent component's parsed CLI args, so an env-only helper would be
redundant there.

Operator-tunable env vars
-------------------------

Shared (consumed by both backends):

  - ``DYN_MM_HTTP_MAX_CONNECTIONS`` (default 100) — total pool size cap.
  - ``DYN_MM_HTTP_TIMEOUT`` (unset by default) — when set, replaces the
    caller's per-call ``timeout`` arg on every fetch. Useful as a global
    ceiling without editing call sites. Per-backend semantics differ:
    on httpx it caps the ``read`` component only (``connect`` / ``pool``
    keep their independent budgets so a stuck handshake or saturated
    pool still fast-fails); on aiohttp it caps ``total`` because aiohttp
    doesn't expose separate connect/read components.
    ``DYN_MM_HTTP_READ_TIMEOUT`` is accepted as a deprecated alias.
  - ``DYN_MM_HTTP_CONNECT_TIMEOUT`` (default 5.0s) — TCP+TLS-handshake
    budget. Independent of ``DYN_MM_HTTP_TIMEOUT`` so a stuck origin
    fast-fails on its own. httpx → ``Timeout.connect``;
    aiohttp → ``ClientTimeout.sock_connect``.

httpx-only:

  - ``DYN_MM_HTTP_MAX_KEEPALIVE`` (default = ``MAX_CONNECTIONS``) — cap
    on idle keepalive connections kept warm.
  - ``DYN_MM_HTTP_POOL_TIMEOUT`` (default 60.0s) — wait-for-free-slot
    budget. Decoupled from read so a saturated pool surfaces fast.
    aiohttp has no equivalent because its connector queues natively.
  - ``DYN_MM_HTTP_CONCURRENCY`` (default 50) — process-wide cap on
    concurrent in-flight HTTP fetches. Acts as backpressure in front
    of the httpx pool so a burst can't push ``PoolTimeout`` up the
    stack. aiohttp has no equivalent because its connector queues
    natively.

aiohttp-only:

  - ``DYN_MM_HTTP_KEEPALIVE_TIMEOUT`` (default 15.0s) — how long an
    idle connection stays warm in the pool.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Self

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument


def _nullable_float(value: str) -> Optional[float]:
    """Parse a float; treat empty/'None' as unset."""
    if value is None or value == "" or value == "None":
        return None
    return float(value)


class HttpConfigBase(ConfigBase):
    """Operator-tunable knobs for the multimodal HTTP backends.

    Field accesses (e.g. ``self._config.max_connections``) are how
    backends read their tunables. Per-backend semantics for each field
    are documented inline below.
    """

    # --- Shared (consumed by httpx + aiohttp) ----------------------------

    # Total pool size cap.
    #   httpx: ``Limits.max_connections``
    #   aiohttp: ``TCPConnector.limit``
    max_connections: int

    # Override for the per-call ``timeout`` arg passed to ``fetch_bytes``.
    # ``None`` → caller's value wins (the common case). When set, every
    # fetch uses this value regardless of what the caller asked for.
    # Per-backend semantics:
    #   httpx → ``Timeout.read`` (``connect`` / ``pool`` stay independent
    #           so a stuck handshake or saturated pool still fast-fails).
    #   aiohttp → ``ClientTimeout.total`` (aiohttp has no separate read
    #             component; the override caps the whole request).
    per_call_timeout_override: Optional[float]

    # TCP+TLS-handshake budget in seconds. Independent of the per-call /
    # read budget so a stuck origin fast-fails on its own.
    #   httpx: ``Timeout.connect``
    #   aiohttp: ``ClientTimeout.sock_connect``
    connect_timeout: float

    # --- httpx-only -------------------------------------------------------

    # Cap on idle keepalive connections kept warm in the pool. Raising
    # this to match ``max_connections`` prevents TLS re-handshake churn
    # under fan-out. Maps to ``Limits.max_keepalive_connections``.
    max_keepalive: int

    # Wait-for-free-slot timeout; ``Timeout.pool``. Decoupled from the
    # read budget so a saturated pool surfaces quickly.
    pool_timeout: float

    # Process-wide cap on concurrent in-flight HTTP fetches via the
    # httpx backend. The semaphore acts as backpressure in front of the
    # pool so a burst of requests can't push ``PoolTimeout`` up the
    # stack. aiohttp has no equivalent because its connector queues
    # natively.
    concurrency: int

    # --- aiohttp-only -----------------------------------------------------

    # How long an idle connection stays warm in the pool, in seconds;
    # ``TCPConnector.keepalive_timeout``.
    keepalive_timeout: float

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> Self:
        config = super().from_cli_args(args)
        # ``--mm-http-max-keepalive 0`` is the documented sentinel for
        # "match max_connections" — resolve here so the httpx backend
        # always gets a non-zero keepalive cap regardless of which
        # construction path the operator used.
        if config.max_keepalive == 0:
            config.max_keepalive = config.max_connections
        return config


class HttpArgGroup(ArgGroup):
    """CLI / env-var registration for the multimodal HTTP backends."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("Multimodal HTTP Options")

        add_argument(
            g,
            flag_name="--mm-http-max-connections",
            env_var="DYN_MM_HTTP_MAX_CONNECTIONS",
            default=100,
            arg_type=int,
            dest="max_connections",
            help="Total pool size cap (httpx Limits.max_connections / aiohttp TCPConnector.limit).",
        )
        add_argument(
            g,
            flag_name="--mm-http-timeout",
            env_var="DYN_MM_HTTP_TIMEOUT",
            default=None,
            arg_type=_nullable_float,
            dest="per_call_timeout_override",
            help=(
                "Per-call timeout override (seconds). When set, replaces "
                "the caller's timeout on every fetch. httpx caps Timeout.read; "
                "aiohttp caps ClientTimeout.total."
            ),
        )
        add_argument(
            g,
            flag_name="--mm-http-connect-timeout",
            env_var="DYN_MM_HTTP_CONNECT_TIMEOUT",
            default=5.0,
            arg_type=float,
            dest="connect_timeout",
            help=(
                "TCP+TLS-handshake budget (seconds). Independent of the "
                "per-call timeout so a stuck origin fast-fails on its own."
            ),
        )
        add_argument(
            g,
            flag_name="--mm-http-max-keepalive",
            env_var="DYN_MM_HTTP_MAX_KEEPALIVE",
            default=0,  # 0 → match max_connections; resolved in from_env.
            arg_type=int,
            dest="max_keepalive",
            help=(
                "[httpx-only] Cap on idle keepalive connections in the pool. "
                "0 → match --mm-http-max-connections."
            ),
        )
        add_argument(
            g,
            flag_name="--mm-http-pool-timeout",
            env_var="DYN_MM_HTTP_POOL_TIMEOUT",
            default=60.0,
            arg_type=float,
            dest="pool_timeout",
            help="[httpx-only] Wait-for-free-slot timeout (seconds).",
        )
        add_argument(
            g,
            flag_name="--mm-http-concurrency",
            env_var="DYN_MM_HTTP_CONCURRENCY",
            default=50,
            arg_type=int,
            dest="concurrency",
            help=(
                "[httpx-only] Process-wide cap on concurrent in-flight fetches. "
                "Acts as backpressure in front of the pool."
            ),
        )
        add_argument(
            g,
            flag_name="--mm-http-keepalive-timeout",
            env_var="DYN_MM_HTTP_KEEPALIVE_TIMEOUT",
            default=15.0,
            arg_type=float,
            dest="keepalive_timeout",
            help="[aiohttp-only] How long an idle connection stays warm (seconds).",
        )


def from_env() -> HttpConfigBase:
    """Build an :class:`HttpConfigBase` from the current environment.

    Spins up an internal parser, registers :class:`HttpArgGroup`, and
    materializes the config from an empty argv. Defaults flow through
    ``DYN_MM_HTTP_*`` env vars via :func:`add_argument`'s
    ``env_or_default`` plumbing — same code path as a component that
    embeds the group in its CLI surface.

    Backwards compat: ``DYN_MM_HTTP_READ_TIMEOUT`` is honored as a
    deprecated alias for ``DYN_MM_HTTP_TIMEOUT`` (which wins if both
    are set).
    """
    if (
        os.environ.get("DYN_MM_HTTP_TIMEOUT") is None
        and os.environ.get("DYN_MM_HTTP_READ_TIMEOUT") is not None
    ):
        os.environ["DYN_MM_HTTP_TIMEOUT"] = os.environ["DYN_MM_HTTP_READ_TIMEOUT"]

    parser = argparse.ArgumentParser(add_help=False)
    HttpArgGroup().add_arguments(parser)
    return HttpConfigBase.from_cli_args(parser.parse_args([]))
