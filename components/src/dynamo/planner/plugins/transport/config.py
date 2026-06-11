# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport / Clock configuration schema and factories.

Configures both:
- ``planner.plugin_registration.transport.*`` — TransportConfig (timeouts, etc.)
- ``planner.scheduling.clock.*`` — ClockConfig (wall vs virtual)

mTLS support lands in a follow-up PR; PR #1 ships plaintext gRPC only,
gated behind ``allow_insecure_grpc=true`` (DEV ONLY).

Factory functions:
- ``make_transport_for_endpoint(plugin_id, endpoint, config, instance=None)``
- ``make_clock(config)`` — production refuses ``virtual`` unless test override env set
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamo.planner.plugins.clock import Clock, VirtualClock, WallClock
from dynamo.planner.plugins.transport.base import PluginTransport

# ``GrpcTransport`` import deferred to ``make_transport_for_endpoint``
# where it's actually constructed.  Module-top import would pull in
# ``_grpc_base`` → ``_proto_bridge`` → ``plugin_pb2``, which is generated
# at install time and absent from the source tree before proto generation.


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------


class TransportConfig(BaseModel):
    """``planner.plugin_registration.transport.*`` config tree."""

    model_config = ConfigDict(extra="forbid")

    allow_insecure_grpc: bool = False
    """Default refuse plaintext grpc:// channels; set true + WARNING log for dev.

    PR #1 only supports plaintext gRPC behind this flag. mTLS support
    (cert-manager / Secret mount) lands in a follow-up PR."""

    request_timeout_seconds: float = Field(default=5.0, gt=0)
    """Per-RPC timeout — applies uniformly to every plugin's ``call()``.
    Per-plugin override is not shipped in PR #1; a future PR may add it
    by plumbing a new ``RegisterRequest`` field through
    ``make_transport_for_endpoint``."""

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
        config: TransportConfig (timeouts + ``allow_insecure_grpc``)
        in_process_instance: required when ``endpoint`` starts with ``inproc://``;
            ignored otherwise. Bridges the ``register_internal`` path.

    Raises:
        ValueError: invalid endpoint scheme, missing instance for inproc,
            or ``grpc://`` endpoint without ``allow_insecure_grpc=True``
            (mTLS support lands in a follow-up PR).
    """
    timeout = config.request_timeout_seconds

    if endpoint.startswith("inproc://"):
        # Local import keeps the module top free of plugin_pb2-dependent
        # transports — see comment at top of file.
        from dynamo.planner.plugins.transport.in_process import InProcessTransport

        if in_process_instance is None:
            raise ValueError(
                f"make_transport_for_endpoint(plugin_id={plugin_id!r}, "
                f"endpoint={endpoint!r}): in_process_instance required for inproc://"
            )
        return InProcessTransport(
            plugin_id, in_process_instance, timeout_seconds=timeout
        )

    if endpoint.startswith("grpc://"):
        # Same deferred-import pattern as ``InProcessTransport`` above.
        from dynamo.planner.plugins.transport.grpc_remote import GrpcTransport

        if not config.allow_insecure_grpc:
            raise ValueError(
                f"make_transport_for_endpoint(plugin_id={plugin_id!r}, "
                f"endpoint={endpoint!r}): plaintext grpc:// requires "
                f"allow_insecure_grpc=True; mTLS support lands in a "
                f"follow-up PR."
            )
        return GrpcTransport(
            plugin_id,
            endpoint,
            timeout_seconds=timeout,
            allow_insecure=config.allow_insecure_grpc,
            keepalive_time_ms=config.keepalive_time_ms,
            max_message_size_bytes=config.max_message_size_bytes,
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
    "TransportConfig",
    "ClockConfig",
    "make_transport_for_endpoint",
    "make_clock",
]
