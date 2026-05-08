# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry configuration schema + factories (DEP-XXXX PR 3 sub-task 3-10).

Schema shape (v11)
------------------

``planner.plugin_registration.*``
  - ``endpoints`` (UDS socket + optional gRPC listen addr)
  - ``auth`` (trusted_sources + per-source config)
  - ``transport`` (TransportConfig; see PR 2 ``transport/config.py``)
  - ``protocol_version_min`` / ``_max``
  - ``heartbeat_timeout_seconds`` / ``heartbeat_missed_threshold``
  - ``in_process_plugins`` (v11 § C-2: moved from SchedulingConfig to
    live next to other "how plugins register" settings)
  - ``admin`` (Q5 v1: simplified — ``AllowAllAdminAuth`` default)

``planner.scheduling.*``
  - ``clock`` (see PR 2 ``transport/config.py``)
  - ``request_timeout_seconds`` / ``tick_max_duration_seconds``
  - ``builtins`` — per-builtin-plugin toggles (M-6 v11: actual default for
    ``enabled`` is the ``enable_*_scaling`` toggle of the parent planner
    config; overriden here)

Auth scope: all four sources (``static_secret`` / ``allow_unauthenticated``
/ ``k8s_sa`` / ``spiffe_jwt``) are wired. ``k8s_sa`` and ``spiffe_jwt``
were deferred from v1 to PR 3.5 to avoid blocking the registry ship on
K8s RBAC and SPIRE integration; they share the same schema shape so no
config migration is required.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from dynamo.planner.plugins.clock import Clock
from dynamo.planner.plugins.registry.auth import (
    AllowUnauthenticatedAuth,
    AuthValidator,
    K8sSATokenAuth,
    MultiSourceAuth,
    SpiffeJwtAuth,
    StaticSecretAuth,
)
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.server import PluginRegistryServer
from dynamo.planner.plugins.transport.base import PluginTransport
from dynamo.planner.plugins.transport.config import (
    TransportConfig,
    make_transport_for_endpoint,
)

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------


class EndpointsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uds_socket_path: str = "/var/run/dynamo/planner/registry.sock"
    grpc_listen_addr: Optional[str] = None
    """``None`` by default — gRPC listener is OPT-IN to prevent accidental
    network exposure. Set to e.g. ``":50051"`` to enable (M-8 v11)."""


# ----------------------------------------------------------------------------
# Auth
# ----------------------------------------------------------------------------


AuthSource = Literal[
    "static_secret", "k8s_sa", "spiffe_jwt", "allow_unauthenticated"
]


class K8sSAConfig(BaseModel):
    """Config for ``k8s_sa`` auth source. Consumed by
    ``K8sSATokenAuth`` via ``build_auth_validator``."""

    model_config = ConfigDict(extra="forbid")

    audience: str = "dynamo-planner"
    trusted_service_accounts: list[str] = Field(default_factory=list)
    """List of ``namespace/serviceaccount`` strings."""


class SpiffeConfig(BaseModel):
    """Config for ``spiffe_jwt`` auth source. Consumed by
    ``SpiffeJwtAuth`` via ``build_auth_validator``."""

    model_config = ConfigDict(extra="forbid")

    jwks_endpoint: str
    audience: str = "dynamo-planner"
    trust_domain: str
    trusted_spiffe_ids: list[str] = Field(default_factory=list)


class AuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trusted_sources: list[AuthSource] = Field(default_factory=list)
    """Empty default = fail-closed; ``build_auth_validator`` raises."""

    static_secrets: dict[str, str] = Field(default_factory=dict)
    """``secret_value -> subject_label`` map."""

    k8s_sa: Optional[K8sSAConfig] = None
    spiffe: Optional[SpiffeConfig] = None


# ----------------------------------------------------------------------------
# Scheduling + in-process + builtins
# ----------------------------------------------------------------------------


class BuiltinPluginToggle(BaseModel):
    """Override for a builtin plugin's scheduling (M-6 v11).

    ``enabled`` default ``True`` is overridden by the parent planner's
    ``enable_*_scaling`` toggle at bootstrap time; runtime config edits
    here take effect on next ``config.reload()`` (row 5 of the v11 cache
    invalidation table)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    priority: Optional[int] = None
    execution_interval_seconds: Optional[float] = None


class InProcessPluginSpec(BaseModel):
    """v11 § C-2: moved from SchedulingConfig to PluginRegistrationConfig
    so all "how plugins come to exist" settings live together.
    v11 § M-5: ``extra="forbid"`` rejects unknown fields (including
    ``protocol_version``, which is nonsensical for in-process plugins)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    module: str
    class_: str = Field(..., alias="class")
    """Python class name in ``module``; aliased since ``class`` is a keyword."""

    plugin_id: str
    plugin_type: Literal["predict", "propose", "reconcile", "constrain"]
    priority: int
    execution_interval_seconds: float = 0.0
    hold_policy: Literal["ACCEPT_WHEN_IDLE", "HOLD_LAST"] = "ACCEPT_WHEN_IDLE"
    kwargs: dict[str, Any] = Field(default_factory=dict)


class SchedulingConfig(BaseModel):
    """``planner.scheduling.*`` config tree. Note: ``in_process_plugins``
    is NOT here in v11 (moved to PluginRegistrationConfig)."""

    model_config = ConfigDict(extra="forbid")

    request_timeout_seconds: float = 5.0
    tick_max_duration_seconds: float = 30.0
    builtins: dict[str, BuiltinPluginToggle] = Field(default_factory=dict)


# ----------------------------------------------------------------------------
# Admin (Q5 v1: simplified)
# ----------------------------------------------------------------------------


class AdminAuthConfig(BaseModel):
    """Admin (ListPlugins) RBAC config. v1 default = allow_all (dev); Q5
    leaves K8s RBAC admin to PR 3.5 follow-up."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["allow_all", "k8s_rbac"] = "allow_all"


# ----------------------------------------------------------------------------
# Top-level aggregate
# ----------------------------------------------------------------------------


class PluginRegistrationConfig(BaseModel):
    """``planner.plugin_registration.*`` root config tree (v11)."""

    model_config = ConfigDict(extra="forbid")

    endpoints: EndpointsConfig = Field(default_factory=EndpointsConfig)
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


def build_auth_validator(config: AuthConfig) -> AuthValidator:
    """Construct the composed auth validator from ``AuthConfig``.

    Raises ``ValueError`` on empty ``trusted_sources`` (fail-closed) or
    on a listed source whose per-source config is missing (e.g.
    ``'k8s_sa'`` in ``trusted_sources`` but ``k8s_sa`` field is ``None``).
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
        elif source_name == "k8s_sa":
            if config.k8s_sa is None:
                raise ValueError(
                    "AuthConfig.trusted_sources lists 'k8s_sa' but "
                    "AuthConfig.k8s_sa is None; populate the K8sSAConfig "
                    "sub-tree (audience + trusted_service_accounts)."
                )
            sources.append(_build_k8s_sa(config.k8s_sa))
        elif source_name == "spiffe_jwt":
            if config.spiffe is None:
                raise ValueError(
                    "AuthConfig.trusted_sources lists 'spiffe_jwt' but "
                    "AuthConfig.spiffe is None; populate the SpiffeConfig "
                    "sub-tree (jwks_endpoint / trust_domain / etc)."
                )
            sources.append(_build_spiffe(config.spiffe))
        else:  # pragma: no cover — schema Literal prevents reaching here
            raise ValueError(f"unknown auth source: {source_name!r}")
    return MultiSourceAuth(sources)


def _build_k8s_sa(cfg: K8sSAConfig) -> K8sSATokenAuth:
    """Build ``K8sSATokenAuth`` from its sub-tree, loading the
    in-cluster ``kubernetes`` client on demand so environments without
    K8s libs only hit the import when this source is actually enabled.
    """
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config

    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        # Fallback for out-of-cluster testing / dev (``~/.kube/config``).
        k8s_config.load_kube_config()
    return K8sSATokenAuth(
        kube_client=k8s_client.AuthenticationV1Api(),
        audience=cfg.audience,
        trusted_service_accounts=cfg.trusted_service_accounts,
    )


def _build_spiffe(cfg: SpiffeConfig) -> SpiffeJwtAuth:
    return SpiffeJwtAuth(
        jwks_endpoint=cfg.jwks_endpoint,
        audience=cfg.audience,
        trust_domain=cfg.trust_domain,
        trusted_spiffe_ids=cfg.trusted_spiffe_ids,
    )


def build_registry_from_config(
    config: PluginRegistrationConfig,
    clock: Clock,
) -> tuple[PluginRegistryServer, CircuitBreaker]:
    """Construct and wire the registry + circuit breaker.

    Returns the pair so caller (orchestrator / PR 5) can hand the
    circuit breaker to other subsystems (scheduler, heartbeat monitor).
    """
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
) -> PluginTransport:
    """Adapter: ``make_transport_for_endpoint`` takes ``config`` as the
    third positional argument; the registry's factory protocol is
    ``(plugin_id, endpoint, *, in_process_instance=None)``."""
    return make_transport_for_endpoint(
        plugin_id,
        endpoint,
        transport_config,
        in_process_instance=in_process_instance,
    )


__all__ = [
    "EndpointsConfig",
    "AuthSource",
    "K8sSAConfig",
    "SpiffeConfig",
    "AuthConfig",
    "BuiltinPluginToggle",
    "InProcessPluginSpec",
    "SchedulingConfig",
    "AdminAuthConfig",
    "PluginRegistrationConfig",
    "build_auth_validator",
    "build_registry_from_config",
]
