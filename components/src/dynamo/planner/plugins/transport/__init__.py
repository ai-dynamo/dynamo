# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport abstractions for plugin invocation.

Two transports under one ``PluginTransport`` ABC:
- ``InProcessTransport``: direct Python call (``inproc://<plugin_id>``)
- ``GrpcTransport``: plaintext grpc (``grpc://host:port``)

All transports satisfy the same ``call(method, request)`` contract;
the contract test enforces byte-equality across them.

mTLS support lands in a follow-up PR; PR #1 ships plaintext gRPC only,
gated behind ``allow_insecure_grpc=true`` (DEV ONLY).
"""

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
    "PluginCallError",
    "PluginConnectionError",
    "PluginSerializationError",
    "PluginTimeoutError",
    "PluginUnknownMethodError",
]
