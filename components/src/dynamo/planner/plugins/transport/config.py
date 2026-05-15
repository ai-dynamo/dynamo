# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport / Clock configuration schema and factories.

Configures both:
- ``planner.plugin_registration.transport.*`` — TransportConfig (mTLS, timeouts, etc.)
- ``planner.scheduling.clock.*`` — ClockConfig (wall vs virtual)

Factory functions:
- ``make_transport_for_endpoint(plugin_id, endpoint, config, instance=None)``
- ``make_clock(config)`` — production refuses ``virtual`` unless test override env set
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamo.planner.plugins.clock import Clock, VirtualClock, WallClock
from dynamo.planner.plugins.transport._mtls import MtlsConfig
from dynamo.planner.plugins.transport.base import PluginTransport
from dynamo.planner.plugins.transport.grpc_remote import GrpcTransport
from dynamo.planner.plugins.transport.in_process import InProcessTransport

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------


class GrpcMtlsConfig(BaseModel):
    """v11 C-3: reuse dynamo platform cert-manager / certificateSecret convention."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    secret_mount_path: str = "/var/run/dynamo/planner-tls"
    """K8s Secret mount with three keys: tls.crt / tls.key / ca.crt"""

    cert_reload_inotify: bool = False
    """v1: not implemented (cert-manager triggers Pod restart for rotation);
    v2 follow-up may add inotify-based hot reload."""


class TransportConfig(BaseModel):
    """``planner.plugin_registration.transport.*`` config tree."""

    model_config = ConfigDict(extra="forbid")

    allow_insecure_grpc: bool = False
    """Default refuse plaintext grpc:// channels; set true + WARNING log for dev."""

    grpc_mtls: GrpcMtlsConfig | None = None
    """Required for grpc:// endpoint unless allow_insecure_grpc=True."""

    request_timeout_seconds: float = 5.0
    """Per-RPC timeout default; can be overridden per-plugin via
    ``RegisterRequest.request_timeout_seconds``."""

    keepalive_time_ms: int = 30_000
    max_message_size_bytes: int = 10 * 1024 * 1024  # 10 MB


class ClockConfig(BaseModel):
    """``planner.scheduling.clock.*`` config tree."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["wall", "virtual"] = "wall"
    """Production must be ``wall``; ``virtual`` only allowed when env
    ``DYNAMO_PLANNER_TEST=1`` is set."""

    virtual_start_now: float = 0.0
    """Initial epoch time for VirtualClock (only used when type=virtual)."""

    virtual_start_mono: float = 0.0
    """Initial monotonic time for VirtualClock (only used when type=virtual)."""


# ----------------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------------


def make_transport_for_endpoint(
    plugin_id: str,
    endpoint: str,
    config: TransportConfig,
    *,
    in_process_instance: Any | None = None,
) -> PluginTransport:
    """Construct a ``PluginTransport`` from endpoint scheme + config.

    Args:
        plugin_id: identifier passed through to the transport
        endpoint: must start with ``inproc://`` or ``grpc://``
        config: TransportConfig (timeouts, mTLS, etc.)
        in_process_instance: required when ``endpoint`` starts with ``inproc://``;
            ignored otherwise. Bridges the ``register_internal`` path.

    Raises:
        ValueError: invalid endpoint scheme, missing instance for inproc,
            or grpc without mTLS / allow_insecure_grpc
    """
    timeout = config.request_timeout_seconds

    if endpoint.startswith("inproc://"):
        if in_process_instance is None:
            raise ValueError(
                f"make_transport_for_endpoint(plugin_id={plugin_id!r}, "
                f"endpoint={endpoint!r}): in_process_instance required for inproc://"
            )
        return InProcessTransport(plugin_id, in_process_instance, timeout_seconds=timeout)

    if endpoint.startswith("grpc://"):
        mtls = None
        if config.grpc_mtls is not None and config.grpc_mtls.enabled:
            mtls = MtlsConfig.from_k8s_secret_mount(config.grpc_mtls.secret_mount_path)
        if mtls is None and not config.allow_insecure_grpc:
            raise ValueError(
                f"make_transport_for_endpoint(plugin_id={plugin_id!r}, "
                f"endpoint={endpoint!r}): grpc:// requires mTLS or "
                f"allow_insecure_grpc=True"
            )
        return GrpcTransport(
            plugin_id,
            endpoint,
            mtls_config=mtls,
            timeout_seconds=timeout,
            allow_insecure=config.allow_insecure_grpc,
        )

    raise ValueError(
        f"make_transport_for_endpoint(plugin_id={plugin_id!r}): "
        f"unknown endpoint scheme in {endpoint!r}; expected one of "
        f"'inproc://', 'grpc://'"
    )


_TEST_OVERRIDE_ENV = "DYNAMO_PLANNER_TEST"


def make_clock(config: ClockConfig) -> Clock:
    """Construct a Clock from config.

    Production safety: ``type="virtual"`` is rejected unless
    ``DYNAMO_PLANNER_TEST=1`` is set in the environment. Replay /
    test code paths set the env var explicitly.
    """
    if config.type == "wall":
        return WallClock()
    if config.type == "virtual":
        if os.environ.get(_TEST_OVERRIDE_ENV) != "1":
            raise ValueError(
                f"make_clock: clock.type=virtual requires environment "
                f"variable {_TEST_OVERRIDE_ENV}=1 (production safety check). "
                f"VirtualClock must not be used in production."
            )
        return VirtualClock(
            start_now=config.virtual_start_now,
            start_mono=config.virtual_start_mono,
        )
    raise ValueError(f"make_clock: unknown clock.type={config.type!r}")


__all__ = [
    "GrpcMtlsConfig",
    "TransportConfig",
    "ClockConfig",
    "make_transport_for_endpoint",
    "make_clock",
]
