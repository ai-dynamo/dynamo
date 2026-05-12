# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin registry + scheduler.

- ``PluginRegistryServer`` hosts the Register / Heartbeat / Unregister /
  ListPlugins RPCs. Invokable in-process (for builtin / in_process user
  plugins via ``register_internal``) or behind a gRPC / UDS server.
- ``HeartbeatMonitor`` evicts plugins that miss heartbeats (skips
  ``transport_type == "in_process"``).
- ``CircuitBreaker`` tracks per-plugin failure counts and fans out OPEN
  transitions to the scheduler for cache invalidation.
- ``PluginScheduler`` computes per-tick active set (triggered vs inherited
  via HOLD_LAST cache) and honours the cache invalidation 6-row table.
- ``auth/`` hosts ``AuthValidator`` implementations (static_secret,
  allow_unauthenticated, K8s SA, SPIFFE JWT).

The orchestrator composes these with the merge algorithms and
transport/clock primitives into the planner pipeline.
"""

from dynamo.planner.plugins.registry.errors import (
    AuthError,
    DuplicatePluginIdError,
    ProtocolVersionError,
    RegistryError,
)
from dynamo.planner.plugins.registry.types import (
    RegisteredPlugin,
    derive_transport_type,
)

__all__ = [
    "RegisteredPlugin",
    "derive_transport_type",
    "RegistryError",
    "AuthError",
    "ProtocolVersionError",
    "DuplicatePluginIdError",
]
