# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry configuration schema + factories.

Schema shape
------------

``planner.plugin_registration.*``
  - ``auth`` (trusted_sources + per-source config)
  - ``transport`` (TransportConfig; see ``transport/config.py``)
  - ``protocol_version_min`` / ``_max``
  - ``heartbeat_timeout_seconds`` / ``heartbeat_missed_threshold``
  - ``in_process_plugins`` — lives next to other "how plugins register"
    settings
  - ``admin`` (simplified — ``AllowAllAdminAuth`` default)

``planner.scheduling.*`` lives in ``config/planner_config.py``
(``SchedulingConfig`` + ``GatewayConfig``); this module does not own
that subtree. The clock + transport timeouts referenced by
``SchedulingConfig`` come from ``plugins/transport/config.py``.

Auth scope: PR #1 wires ``static_secret`` + ``allow_unauthenticated``.
``k8s_sa`` and ``spiffe_jwt`` land in a follow-up PR alongside their
cluster-side configuration and end-to-end smoke tests.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamo.planner.plugins.transport.config import TransportConfig

# ``PluginRegistryServer`` (and through it ``plugin_pb2`` / ``plugin_pb2_grpc``)
# is only needed at *runtime* by the build helpers below — not by the
# Pydantic config schema this module exposes for import-time deserialisation.
# Keeping the heavy import out of the module top-level lets
# ``PlannerConfig.scheduling.plugin_registration`` resolve to its schema in
# a default ``use_orchestrator=False`` deployment without requiring the
# generated proto stubs to be present on disk (the stubs are generated at
# install / dev-time only; PSM-only deployments must still parse the
# config tree).
if TYPE_CHECKING:
    from dynamo.planner.plugins.clock import Clock
    from dynamo.planner.plugins.registry.auth.base import AuthValidator
    from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
    from dynamo.planner.plugins.registry.server import PluginRegistryServer
    from dynamo.planner.plugins.transport.base import PluginTransport

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Auth
# ----------------------------------------------------------------------------


AuthSource = Literal["static_secret", "allow_unauthenticated"]


class AuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trusted_sources: list[AuthSource] = Field(default_factory=list)
    """Empty default = fail-closed; ``build_auth_validator`` raises."""

    static_secrets: dict[str, str] = Field(default_factory=dict)
    """``secret_value -> subject_label`` map."""


# ----------------------------------------------------------------------------
# In-process plugin spec
# ----------------------------------------------------------------------------


class InProcessPluginSpec(BaseModel):
    """Spec for an in-process plugin entry — lives under
    PluginRegistrationConfig so all "how plugins come to exist"
    settings live together.

    ``extra="forbid"`` rejects unknown fields (including
    ``protocol_version``, which is nonsensical for in-process plugins).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    module: str
    class_: str = Field(..., alias="class")
    """Python class name in ``module``; aliased since ``class`` is a keyword."""

    plugin_id: str
    plugin_type: Literal["predict", "propose", "reconcile", "constrain"]
    priority: int
    execution_interval_seconds: float = 0.0
    hold_policy: Literal["ACCEPT_WHEN_IDLE", "HOLD_LAST"] = "ACCEPT_WHEN_IDLE"
    needs: list[str] = Field(default_factory=list)
    """Capability list (consumed by type-aware merge / lazy-traffic-pull)."""

    requires_produced_fields: list[str] = Field(default_factory=list)
    """Hard dependency on earlier-stage produced fields. Each entry is a
    dot-path into ``PipelineContext`` (e.g. ``"predictions"``,
    ``"observations.traffic"``). Scheduler skips this plugin for the
    tick if any listed field is unset; skipping does NOT advance the
    plugin's anchor. Empty = no gating."""

    observation_window_seconds: float = Field(default=0.0, ge=0)
    """Aggregation window the plugin wants for windowed observation
    types in ``needs``. ``0.0`` = ``scale_interval`` freshness;
    ``N > 0`` = aggregate over the last ``N`` seconds."""

    kwargs: dict[str, Any] = Field(default_factory=dict)


# ----------------------------------------------------------------------------
# Admin
# ----------------------------------------------------------------------------


class AdminAuthConfig(BaseModel):
    """Admin (ListPlugins) RBAC config.

    **PR #1 status — config is parsed but inert**: the gRPC
    ``PluginRegistry.ListPlugins`` RPC currently default-denies with
    ``PERMISSION_DENIED`` regardless of ``mode`` (see
    ``plugins/registry/gateway.py:ListPlugins``). The admin RBAC path
    that consumes this field — including ``k8s_rbac`` resolution — lands
    together with the broader auth follow-up (PR 1.5). Setting ``mode``
    in YAML today has no effect; in-process callers
    (``PluginRegistryServer.list_plugins`` direct method) are
    unaffected by the gateway default-deny.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["allow_all", "k8s_rbac"] = "allow_all"


# ----------------------------------------------------------------------------
# Top-level aggregate
# ----------------------------------------------------------------------------


class PluginRegistrationConfig(BaseModel):
    """``planner.plugin_registration.*`` root config tree (v11)."""

    model_config = ConfigDict(extra="forbid")

    auth: AuthConfig = Field(default_factory=AuthConfig)
    transport: TransportConfig = Field(default_factory=TransportConfig)
    protocol_version_min: str = "1.0"
    protocol_version_max: str = "1.0"
    heartbeat_timeout_seconds: float = 15.0
    heartbeat_missed_threshold: int = 2
    in_process_plugins: list[InProcessPluginSpec] = Field(default_factory=list)
    admin: AdminAuthConfig = Field(default_factory=AdminAuthConfig)


# ----------------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------------


def build_auth_validator(config: AuthConfig) -> "AuthValidator":
    # Heavy auth/registry imports deferred to call time so this module
    # stays importable in PSM-only deployments without the generated
    # plugin_pb2 stubs (see TYPE_CHECKING block at module top).
    from dynamo.planner.plugins.registry.auth import (
        AllowUnauthenticatedAuth,
        MultiSourceAuth,
        StaticSecretAuth,
    )

    """Construct the composed auth validator from ``AuthConfig``.

    Raises ``ValueError`` on empty ``trusted_sources`` (fail-closed) —
    PR #1 supports ``static_secret`` and ``allow_unauthenticated``.
    """
    if not config.trusted_sources:
        raise ValueError(
            "AuthConfig.trusted_sources is empty; registry would reject "
            "every token. Configure at least one source (e.g. "
            "['static_secret']) or ['allow_unauthenticated'] for dev."
        )
    sources: list[AuthValidator] = []
    for source_name in config.trusted_sources:
        if source_name == "static_secret":
            if not config.static_secrets:
                log.warning(
                    "AuthConfig.static_secrets is empty but 'static_secret' "
                    "listed in trusted_sources — StaticSecretAuth will reject "
                    "every token."
                )
            sources.append(StaticSecretAuth(config.static_secrets))
        elif source_name == "allow_unauthenticated":
            sources.append(AllowUnauthenticatedAuth())
        else:  # pragma: no cover — schema Literal prevents reaching here
            raise ValueError(f"unknown auth source: {source_name!r}")
    return MultiSourceAuth(sources)


def build_registry_from_config(
    config: PluginRegistrationConfig,
    clock: "Clock",
) -> tuple["PluginRegistryServer", "CircuitBreaker"]:
    """Construct and wire the registry + circuit breaker.

    Returns the pair so the caller (orchestrator) can hand the circuit
    breaker to other subsystems (scheduler, heartbeat monitor).
    """
    # Heavy registry imports deferred to call time so this module stays
    # importable in PSM-only deployments without the generated
    # plugin_pb2 stubs (see TYPE_CHECKING block at module top).
    from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
    from dynamo.planner.plugins.registry.server import PluginRegistryServer

    auth = build_auth_validator(config.auth)
    cb = CircuitBreaker(clock)

    transport_factory = functools.partial(
        _transport_factory_shim, transport_config=config.transport
    )

    server = PluginRegistryServer(
        clock=clock,
        auth=auth,
        circuit_breaker=cb,
        transport_factory=transport_factory,
        protocol_versions=(config.protocol_version_min, config.protocol_version_max),
    )
    return server, cb


def _transport_factory_shim(
    plugin_id: str,
    endpoint: str,
    *,
    in_process_instance: Any = None,
    transport_config: TransportConfig,
) -> "PluginTransport":
    """Adapter: ``make_transport_for_endpoint`` takes ``config`` as the
    third positional argument; the registry's factory protocol is
    ``(plugin_id, endpoint, *, in_process_instance=None)``."""
    # Heavy transport import deferred (see TYPE_CHECKING block above).
    from dynamo.planner.plugins.transport.config import make_transport_for_endpoint

    return make_transport_for_endpoint(
        plugin_id,
        endpoint,
        transport_config,
        in_process_instance=in_process_instance,
    )


__all__ = [
    "AuthSource",
    "AuthConfig",
    "InProcessPluginSpec",
    "AdminAuthConfig",
    "PluginRegistrationConfig",
    "build_auth_validator",
    "build_registry_from_config",
]
