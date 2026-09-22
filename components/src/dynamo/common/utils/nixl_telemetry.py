# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exporter ports for NIXL telemetry when several ranks share one pod.

NIXL's Prometheus exporter binds one fixed TCP port, read from
``NIXL_TELEMETRY_PROMETHEUS_PORT`` when the NIXL agent is constructed, so a
backend that builds one agent per co-located rank needs one port per rank.
The port is derived as ``base + local_rank`` rather than left ephemeral
because it has to be predictable: the operator declares the whole range as
container ports and the PodMonitor scrapes each one by name. NIXL accepts
``0`` and binds an ephemeral port, but exposes no accessor for the port it
actually got, so nothing could name that port as a scrape target.

This module deliberately imports no inference engine: the derivation is pure
arithmetic over a base port and a rank index, and the callers that know how to
find a rank index live in the per-backend packages.

Only environment configuration is inspected here. NIXL can also read these
settings from ``NIXL_CONFIG_FILE``, ``~/.nixl.cfg``, or ``/etc/nixl.cfg``;
Dynamo cannot derive or declare per-rank ports from those files.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

MAX_PORT = 65535

NIXL_TELEMETRY_ENABLE_ENV = "NIXL_TELEMETRY_ENABLE"
NIXL_TELEMETRY_EXPORTER_ENV = "NIXL_TELEMETRY_EXPORTER"
NIXL_TELEMETRY_PROMETHEUS_PORT_ENV = "NIXL_TELEMETRY_PROMETHEUS_PORT"

# NIXL's Prometheus exporter uses this port when no override is configured.
DEFAULT_NIXL_PROMETHEUS_PORT = 9090

# NIXL compares these tokens case-insensitively, without trimming whitespace.
_NIXL_TRUE_VALUES = frozenset({"y", "1", "yes", "on", "true", "enable"})
_NIXL_FALSE_VALUES = frozenset({"n", "0", "no", "off", "false", "disable"})

_NIXL_UINT16 = re.compile(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)")

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
    """Return a configured fixed TCP port, ignoring disabled/invalid values."""
    environ = os.environ if env is None else env
    raw = environ.get(env_name)
    if raw is None:
        return default
    try:
        port = int(raw)
    except ValueError:
        return None
    return port if 0 < port <= MAX_PORT else None


def _parse_nixl_uint16(env_name: str, raw: str) -> int:
    """Parse one present value with NIXL's unsigned 16-bit syntax."""
    # Unlike int(), NIXL rejects signs, whitespace, separators, and partial parses.
    if _NIXL_UINT16.fullmatch(raw) is None:
        raise ValueError(f"{env_name}={raw!r} is not a valid unsigned 16-bit integer")

    is_hex = raw.startswith(("0x", "0X"))
    digits = (raw[2:] if is_hex else raw).lstrip("0") or "0"
    max_digits = format(MAX_PORT, "x") if is_hex else str(MAX_PORT)
    if len(digits) > len(max_digits) or (
        len(digits) == len(max_digits) and digits.lower() > max_digits
    ):
        raise ValueError(f"{env_name}={raw!r} is outside the range 0-{MAX_PORT}")
    # Parse only the significant digits so Python's configurable integer-string
    # limit cannot misclassify a valid value with many leading zeroes.
    return int(digits, 16 if is_hex else 10)


def _parse_nixl_bool(env_name: str, raw: str) -> bool:
    """Parse one present value with NIXL's case-insensitive boolean syntax."""
    normalized = raw.lower()
    if normalized in _NIXL_TRUE_VALUES:
        return True
    if normalized in _NIXL_FALSE_VALUES:
        return False
    raise ValueError(f"{env_name}={raw!r} is not a boolean value recognized by NIXL")


def nixl_prometheus_base_port(env: Mapping[str, str] | None = None) -> int | None:
    """Return the fixed Prometheus port selected by the NIXL environment.

    ``None`` means this environment does not select the Prometheus exporter.
    Invalid NIXL values raise, as they do during NIXL agent construction.
    """
    environ = os.environ if env is None else env
    enabled = environ.get(NIXL_TELEMETRY_ENABLE_ENV)
    if enabled is None or not _parse_nixl_bool(NIXL_TELEMETRY_ENABLE_ENV, enabled):
        return None

    exporter = environ.get(NIXL_TELEMETRY_EXPORTER_ENV, "")
    # TODO: NIXL main also has a prometheus_mp exporter that shares this port
    # variable but aggregates all ranks behind one endpoint. Add separate
    # reservation semantics when Dynamo moves to a NIXL release containing it.
    if exporter != "prometheus":
        return None

    raw_port = environ.get(NIXL_TELEMETRY_PROMETHEUS_PORT_ENV)
    port = (
        DEFAULT_NIXL_PROMETHEUS_PORT
        if raw_port is None
        else _parse_nixl_uint16(NIXL_TELEMETRY_PROMETHEUS_PORT_ENV, raw_port)
    )
    if port == 0:
        raise ValueError(
            f"{NIXL_TELEMETRY_PROMETHEUS_PORT_ENV}=0 asks NIXL for an ephemeral "
            "port, which Dynamo cannot declare or configure as a scrape target. "
            f"Set a fixed port between 1 and {MAX_PORT}."
        )
    return port


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

    ``max_ranks`` is how wide that reservation is. A caller that knows how many
    ranks its launch actually places on a node should pass that count rather
    than the maximum, because every check below measures that whole span.

    Raises ValueError rather than returning a port outside the reservation.
    """
    # A narrower reservation is the caller describing its own launch, but a
    # wider one would hand out ports past the range the operator declares as
    # container ports, which is the one thing no caller may narrow away.
    if max_ranks < 1 or max_ranks > MAX_COLOCATED_NIXL_EXPORTERS:
        raise ValueError(
            f"a pod reserves between 1 and {MAX_COLOCATED_NIXL_EXPORTERS} NIXL "
            f"exporter ports, so {max_ranks} co-located ranks cannot each be given "
            f"a declared port to be scraped on."
        )

    if base_port < 1:
        raise ValueError(
            f"{NIXL_TELEMETRY_PROMETHEUS_PORT_ENV}={base_port} is not a usable TCP "
            f"port. Set it between 1 and {MAX_PORT}."
        )

    # Reject the whole range up front, not just this rank's port. A base that
    # leaves room for rank 0 alone would start one scheduler and fail the next,
    # leaving the pod with a partially started scheduler set and never Ready.
    last_port = base_port + max_ranks - 1
    if last_port > MAX_PORT:
        raise ValueError(
            f"{NIXL_TELEMETRY_PROMETHEUS_PORT_ENV}={base_port} with {max_ranks} "
            f"co-located ranks needs ports {base_port}-{last_port}, "
            f"which exceeds the maximum port {MAX_PORT}. Lower "
            f"{NIXL_TELEMETRY_PROMETHEUS_PORT_ENV}."
        )

    # Compare the whole reserved range against each listener, not just this
    # rank's port. A base one below another listener leaves rank 0 clear and
    # collides from rank 1 on, so checking a single port lets the pod start one
    # scheduler and then fail every later one.
    for env_name, start, end in reserved_port_ranges(env, width=max_ranks):
        if base_port <= end and start <= last_port:
            raise ValueError(
                f"the NIXL exporter range {base_port}-{last_port} overlaps "
                f"{env_name}, which reserves {start}-{end}. Configure "
                f"non-overlapping ports."
            )

    if local_rank < 0 or local_rank >= max_ranks:
        raise ValueError(
            f"node-local rank {local_rank} is outside the reserved NIXL exporter "
            f"range of {max_ranks} ports starting at {base_port}. The pod reserves "
            f"one port per co-located rank; a rank beyond that range has no "
            f"declared container port and would not be scraped."
        )

    return base_port + local_rank
