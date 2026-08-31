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

from typing import TYPE_CHECKING, Any

from dynamo.planner.plugins.transport.base import PluginTransport
from dynamo.planner.plugins.transport.errors import (
    PluginCallError,
    PluginConnectionError,
    PluginSerializationError,
    PluginTimeoutError,
    PluginUnknownMethodError,
)
from dynamo.planner.plugins.transport.in_process import InProcessTransport

# ``GrpcTransport`` is eagerly *re-exported* via PEP 562 lazy
# ``__getattr__`` so ``from ... transport import GrpcTransport`` still
# works for consumers that want it.  Eager import would pull in
# ``_grpc_base`` → ``_proto_bridge`` → ``plugin_pb2``, which is generated
# at install time and not on disk in source-tree deployments before proto
# generation. See ``planner_config.py`` lazy-import refactor for the matching
# change at the registry layer.
if TYPE_CHECKING:
    from dynamo.planner.plugins.transport.grpc_remote import GrpcTransport

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


def __getattr__(name: str) -> Any:
    if name == "GrpcTransport":
        from dynamo.planner.plugins.transport.grpc_remote import GrpcTransport

        return GrpcTransport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
