# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport abstractions for plugin invocation.

Two transports under one ``PluginTransport`` ABC:
- ``InProcessTransport``: direct Python call (``inproc://<plugin_id>``)
- ``GrpcTransport``: grpc + optional mTLS (``grpc://host:port``)

All transports satisfy the same ``call(method, request)`` contract;
the contract test enforces byte-equality across them.
"""

from dynamo.planner.plugins.transport._mtls import MtlsConfig
from dynamo.planner.plugins.transport.base import PluginTransport
from dynamo.planner.plugins.transport.errors import (
    PluginCallError,
    PluginConnectionError,
    PluginSerializationError,
    PluginTimeoutError,
    PluginUnknownMethodError,
)
from dynamo.planner.plugins.transport.grpc_remote import GrpcTransport
from dynamo.planner.plugins.transport.in_process import InProcessTransport

__all__ = [
    "PluginTransport",
    "InProcessTransport",
    "GrpcTransport",
    "MtlsConfig",
    "PluginCallError",
    "PluginConnectionError",
    "PluginSerializationError",
    "PluginTimeoutError",
    "PluginUnknownMethodError",
]
