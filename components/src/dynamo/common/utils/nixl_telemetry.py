# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exporter ports for NIXL telemetry when several ranks share one pod.

NIXL's Prometheus exporter binds one fixed TCP port, read from
``NIXL_TELEMETRY_PROMETHEUS_PORT`` when the NIXL agent is constructed. The
operator injects that variable once per container, which is correct for a
backend that builds a single agent in the worker process, and wrong for one
that builds an agent per co-located rank: every rank inherits the same value,
the first exporter wins the port, and each remaining rank aborts with
``Failed to create exporter: null context when constructing CivetServer``.
The process that aborts is the rank itself, so the container never reaches
Ready.

Deriving ``base + local_rank`` is the fix rather than asking NIXL for an
ephemeral port. NIXL does accept ``0`` and binds an ephemeral port, but it
logs the port it was *asked* for and offers no accessor for the port it got,
so nothing can name that port as a ``containerPort`` or as a PodMonitor
target -- the same reason ``dynamo.vllm.embedding_worker_processes`` gives for
rejecting ephemeral ports for its own child listeners. A derived port is
predictable, so the operator can reserve the whole range on the pod and the
PodMonitor can scrape every rank.

This module deliberately imports no inference engine: the derivation is pure
arithmetic over a base port and a rank index, and the callers that know how to
find a rank index live in the per-backend packages.
"""

from __future__ import annotations

import os
from typing import Mapping

MAX_PORT = 65535

NIXL_TELEMETRY_ENABLE_ENV = "NIXL_TELEMETRY_ENABLE"
NIXL_TELEMETRY_EXPORTER_ENV = "NIXL_TELEMETRY_EXPORTER"
NIXL_TELEMETRY_PROMETHEUS_PORT_ENV = "NIXL_TELEMETRY_PROMETHEUS_PORT"

DEFAULT_NIXL_PROMETHEUS_PORT = 19090

# Keep in sync with DynamoMaxNixlPorts in deploy/operator/internal/consts/consts.go:
# a rank deriving a port past the reserved range would bind a port nothing scrapes.
MAX_COLOCATED_NIXL_EXPORTERS = 8

# Listeners the container already owns, as (env var, default base). Each is
# treated as a MAX_COLOCATED_NIXL_EXPORTERS-wide range: both are per-rank bases elsewhere.
_COLLIDING_PORT_ENVS = ("DYN_SYSTEM_PORT", "DYN_FORWARDPASS_METRIC_PORT")


def configured_fixed_port(
    env_name: str,
    *,
    default: int | None = None,
    env: Mapping[str, str] | None = None,
) -> int | None:
    """Return a configured fixed TCP port, ignoring disabled/invalid values.

    Mirrors ``dynamo.vllm.backend_args._configured_fixed_port`` so that a value
    this package accepts is exactly a value that package accepts.
    """
    environ = os.environ if env is None else env
    raw = environ.get(env_name)
    if raw is None:
        return default
    try:
        port = int(raw)
    except ValueError:
        return None
    return port if 0 < port <= MAX_PORT else None


def nixl_prometheus_base_port(env: Mapping[str, str] | None = None) -> int | None:
    """Return the base NIXL Prometheus port, or None when it is not in use."""
    environ = os.environ if env is None else env
    enabled = environ.get(NIXL_TELEMETRY_ENABLE_ENV, "").strip().lower()
    exporter = environ.get(NIXL_TELEMETRY_EXPORTER_ENV, "prometheus")
    if enabled != "y" or exporter.strip().lower() != "prometheus":
        return None
    return configured_fixed_port(
        NIXL_TELEMETRY_PROMETHEUS_PORT_ENV,
        default=DEFAULT_NIXL_PROMETHEUS_PORT,
        env=environ,
    )


def reserved_port_ranges(
    env: Mapping[str, str] | None = None,
    *,
    width: int = MAX_COLOCATED_NIXL_EXPORTERS,
) -> list[tuple[str, int, int]]:
    """Inclusive port ranges already claimed by other listeners in this container."""
    environ = os.environ if env is None else env
    ranges: list[tuple[str, int, int]] = []
    for env_name in _COLLIDING_PORT_ENVS:
        if env_name not in environ:
            continue
        base = configured_fixed_port(env_name, env=environ)
        if base is None:
            continue
        ranges.append((env_name, base, min(base + width - 1, MAX_PORT)))
    return ranges


def derive_nixl_prometheus_port(
    base_port: int,
    local_rank: int,
    *,
    max_ranks: int = MAX_COLOCATED_NIXL_EXPORTERS,
    env: Mapping[str, str] | None = None,
) -> int:
    """Return the exporter port for one node-local rank.

    ``local_rank`` is the rank's index *within its pod*, not its global rank: a
    port range is reserved per pod, so a multi-node deployment restarts the
    offset on every node.

    Raises ValueError rather than returning a port that would leave the
    reserved range or land on another listener. A rank that cannot be given a
    port of its own must say so; falling back to the base port would recreate
    the bind collision this module exists to prevent.
    """
    if local_rank < 0 or local_rank >= max_ranks:
        raise ValueError(
            f"node-local rank {local_rank} is outside the reserved NIXL exporter "
            f"range of {max_ranks} ports starting at {base_port}. The pod reserves "
            f"one port per co-located rank; a rank beyond that range has no "
            f"declared container port and would not be scraped."
        )

    port = base_port + local_rank
    if not 0 < port <= MAX_PORT:
        raise ValueError(
            f"{NIXL_TELEMETRY_PROMETHEUS_PORT_ENV}={base_port} with {max_ranks} "
            f"co-located ranks needs ports {base_port}-{base_port + max_ranks - 1}, "
            f"which exceeds the maximum port {MAX_PORT}. Lower "
            f"{NIXL_TELEMETRY_PROMETHEUS_PORT_ENV}."
        )

    for env_name, start, end in reserved_port_ranges(env, width=max_ranks):
        if start <= port <= end:
            raise ValueError(
                f"NIXL exporter port {port} for node-local rank {local_rank} "
                f"collides with {env_name}, which reserves {start}-{end}. "
                f"Configure non-overlapping ports."
            )
    return port
