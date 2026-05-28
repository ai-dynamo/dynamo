# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the ``dynamo.common.http`` client.

:class:`HttpConfigBase` carries the operator-tunable knobs;
:class:`HttpArgGroup` registers the matching ``--http-*`` CLI flags
and ``DYN_HTTP_*`` env vars. :func:`from_env` is an env-only
construction path used by the singleton client in
``dynamo.common.http`` (whose primary callers don't own an argparse).

Legacy ``DYN_MM_HTTP_*`` env vars are still honored for backward
compatibility with deployments that predate the rename — see
:func:`_apply_legacy_env_aliases`. See per-field comments below for
behavior details.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional, Self

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument

logger = logging.getLogger(__name__)


# Map of canonical env var → list of legacy aliases (priority order).
# Only the six env vars that shipped via the now-deleted
# ``dynamo.common.multimodal.http_client`` module are kept for back
# compat. ``DYN_HTTP_TIMEOUT`` accepts ``DYN_MM_HTTP_READ_TIMEOUT``
# because the read-vs-total semantic was clarified during the rename.
_LEGACY_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "DYN_HTTP_MAX_CONNECTIONS": ("DYN_MM_HTTP_MAX_CONNECTIONS",),
    "DYN_HTTP_TIMEOUT": ("DYN_MM_HTTP_READ_TIMEOUT",),
    "DYN_HTTP_CONNECT_TIMEOUT": ("DYN_MM_HTTP_CONNECT_TIMEOUT",),
    "DYN_HTTP_MAX_KEEPALIVE": ("DYN_MM_HTTP_MAX_KEEPALIVE",),
    "DYN_HTTP_POOL_TIMEOUT": ("DYN_MM_HTTP_POOL_TIMEOUT",),
    "DYN_HTTP_CONCURRENCY": ("DYN_MM_HTTP_CONCURRENCY",),
}

_legacy_warned: set[str] = set()


def _apply_legacy_env_aliases() -> None:
    """Mirror legacy ``DYN_MM_HTTP_*`` env vars to ``DYN_HTTP_*``.

    If the canonical ``DYN_HTTP_*`` name isn't set but a legacy alias
    is, copy the value across so downstream ``env_or_default`` lookups
    see it. Idempotent. Emits a one-time deprecation warning per
    legacy name observed.
    """
    for canonical, legacy_names in _LEGACY_ENV_ALIASES.items():
        if os.environ.get(canonical) is not None:
            continue
        for legacy in legacy_names:
            value = os.environ.get(legacy)
            if value is None:
                continue
            os.environ[canonical] = value
            if legacy not in _legacy_warned:
                _legacy_warned.add(legacy)
                logger.warning("%s is deprecated; use %s instead.", legacy, canonical)
            break


def _nullable_float(value: str) -> Optional[float]:
    """Parse a float; treat empty/'None' as unset."""
    if value is None or value == "" or value == "None":
        return None
    return float(value)


class HttpConfigBase(ConfigBase):
    """Operator-tunable knobs for the HTTP fetch client.

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
        # ``--http-max-keepalive 0`` is the documented sentinel for
        # "match max_connections" — resolve here so the httpx backend
        # always gets a non-zero keepalive cap regardless of which
        # construction path the operator used.
        if config.max_keepalive == 0:
            config.max_keepalive = config.max_connections
        return config


class HttpArgGroup(ArgGroup):
    """CLI / env-var registration for the HTTP fetch client."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("HTTP Fetch Options")

        add_argument(
            g,
            flag_name="--http-max-connections",
            env_var="DYN_HTTP_MAX_CONNECTIONS",
            default=100,
            arg_type=int,
            dest="max_connections",
            help="Total pool size cap (httpx Limits.max_connections / aiohttp TCPConnector.limit).",
        )
        add_argument(
            g,
            flag_name="--http-timeout",
            env_var="DYN_HTTP_TIMEOUT",
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
            flag_name="--http-connect-timeout",
            env_var="DYN_HTTP_CONNECT_TIMEOUT",
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
            flag_name="--http-max-keepalive",
            env_var="DYN_HTTP_MAX_KEEPALIVE",
            default=0,  # 0 → match max_connections; resolved in from_env.
            arg_type=int,
            dest="max_keepalive",
            help=(
                "[httpx-only] Cap on idle keepalive connections in the pool. "
                "0 → match --http-max-connections."
            ),
        )
        add_argument(
            g,
            flag_name="--http-pool-timeout",
            env_var="DYN_HTTP_POOL_TIMEOUT",
            default=60.0,
            arg_type=float,
            dest="pool_timeout",
            help="[httpx-only] Wait-for-free-slot timeout (seconds).",
        )
        add_argument(
            g,
            flag_name="--http-concurrency",
            env_var="DYN_HTTP_CONCURRENCY",
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
            flag_name="--http-keepalive-timeout",
            env_var="DYN_HTTP_KEEPALIVE_TIMEOUT",
            default=15.0,
            arg_type=float,
            dest="keepalive_timeout",
            help="[aiohttp-only] How long an idle connection stays warm (seconds).",
        )


def from_env() -> HttpConfigBase:
    """Build an :class:`HttpConfigBase` from the current environment.

    Spins up an internal parser, registers :class:`HttpArgGroup`, and
    materializes the config from an empty argv. Defaults flow through
    ``DYN_HTTP_*`` env vars via :func:`add_argument`'s
    ``env_or_default`` plumbing — same code path as a component that
    embeds the group in its CLI surface. Legacy ``DYN_MM_HTTP_*`` names
    are mirrored to their canonical ``DYN_HTTP_*`` form first via
    :func:`_apply_legacy_env_aliases`.
    """
    _apply_legacy_env_aliases()
    parser = argparse.ArgumentParser(add_help=False)
    HttpArgGroup().add_arguments(parser)
    return HttpConfigBase.from_cli_args(parser.parse_args([]))
