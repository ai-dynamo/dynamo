# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport abstractions for plugin invocation.

Three transports under one ``PluginTransport`` ABC:
- ``InProcessTransport``: direct Python call (``inproc://<plugin_id>``)
- ``UdsTransport``: grpc-over-uds (``unix:///path/to/sock``)
- ``GrpcTransport``: grpc + optional mTLS (``grpc://host:port``)

All three satisfy the same ``call(method, request)`` contract;
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
from dynamo.planner.plugins.transport.uds import UdsTransport

__all__ = [
    "PluginTransport",
    "InProcessTransport",
    "UdsTransport",
    "GrpcTransport",
    "MtlsConfig",
    "PluginCallError",
    "PluginConnectionError",
    "PluginSerializationError",
    "PluginTimeoutError",
    "PluginUnknownMethodError",
]
