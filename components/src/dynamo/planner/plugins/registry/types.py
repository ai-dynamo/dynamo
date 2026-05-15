# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry-internal data types.

``RegisteredPlugin`` is the in-memory record the ``PluginRegistryServer``
owns for each active plugin. It holds the union of:

- The ``RegisterRequest`` fields (identity, priority, scheduling, needs,
  protocol negotiation, auth metadata).
- Runtime fields (``registered_at``, ``last_heartbeat_at``,
  ``last_call_at``, ``evaluations_total``, ``enabled``) that the
  scheduler / heartbeat monitor read + update.
- The constructed ``PluginTransport`` and a tag ``transport_type``
  driving the heartbeat-skip rule.

CircuitBreaker state is **not** stored on ``RegisteredPlugin`` — it lives
in ``CircuitBreaker`` keyed by ``plugin_id`` (separable lifecycle:
restart clears circuit, back to CLOSED).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from dynamo.planner.plugins.transport.base import PluginTransport
from dynamo.planner.plugins.types import HoldPolicy


TransportType = Literal["in_process", "grpc"]


def derive_transport_type(endpoint: str) -> TransportType:
    """Classify an endpoint URL by scheme.

    Raises ``ValueError`` for unknown schemes so configuration errors
    fail at register time, not at the first ``call()``.
    """
    if endpoint.startswith("inproc://"):
        return "in_process"
    if endpoint.startswith("grpc://"):
        return "grpc"
    raise ValueError(
        f"derive_transport_type: unknown endpoint scheme in {endpoint!r}; "
        f"expected 'inproc://' or 'grpc://'"
    )


@dataclass
class RegisteredPlugin:
    """Internal record the registry owns for every registered plugin.

    Mutable: ``last_heartbeat_at`` / ``last_call_at`` / ``evaluations_total``
    / ``enabled`` are updated in place by the heartbeat monitor,
    orchestrator pipeline driver, and admin endpoint respectively.
    """

    plugin_id: str
    plugin_type: Literal["predict", "propose", "reconcile", "constrain"]
    priority: int
    endpoint: str
    version: str
    protocol_version: str
    execution_interval_seconds: float
    hold_policy: HoldPolicy
    needs: list[str]
    fpm_encoding: str
    request_timeout_seconds: float
    is_builtin: bool
    transport: PluginTransport
    transport_type: TransportType
    registered_at: float
    last_heartbeat_at: float = field(default=-math.inf)
    last_call_at: float = field(default=-math.inf)
    evaluations_total: int = 0
    enabled: bool = True


__all__ = [
    "TransportType",
    "RegisteredPlugin",
    "derive_transport_type",
]
