# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin RPC method → gRPC stub method dispatch table.

Shared by ``UdsTransport`` and ``GrpcTransport``: both maintain one
gRPC channel per plugin, then dispatch ``call(method, ...)`` to the
correct stub method.

Dispatch table is built lazily per channel — each plugin only needs the
stage-specific stub it serves (a propose plugin only needs ProposePlugin
service; no need to instantiate all 6 service stubs).
"""

from __future__ import annotations

from typing import Any, Callable

import grpc

from dynamo.planner.plugins.proto.v1 import plugin_pb2_grpc as pbg

# Method name → (stub class, method attribute on stub instance)
_METHOD_STUB_MAP: dict[str, tuple[type[Any], str]] = {
    # Stage RPCs
    "Predict": (pbg.PredictPluginStub, "Predict"),
    "Propose": (pbg.ProposePluginStub, "Propose"),
    "Reconcile": (pbg.ReconcilePluginStub, "Reconcile"),
    "Constrain": (pbg.ConstrainPluginStub, "Constrain"),
    # PluginLifecycle RPCs
    "Bootstrap": (pbg.PluginLifecycleStub, "Bootstrap"),
    "Reset": (pbg.PluginLifecycleStub, "Reset"),
}


class StubDispatcher:
    """Lazy stub-cache per gRPC channel.

    Instantiates a service stub only on first use; caches the bound
    method (``stub.Predict``) for direct invocation.
    """

    def __init__(self, channel: grpc.aio.Channel) -> None:
        self._channel = channel
        self._stub_cache: dict[type[Any], Any] = {}
        self._method_cache: dict[str, Callable[..., Any]] = {}

    def get_method(self, method_name: str) -> Callable[..., Any] | None:
        """Resolve ``method_name`` → bound stub method, or None if unknown."""
        if method_name in self._method_cache:
            return self._method_cache[method_name]
        entry = _METHOD_STUB_MAP.get(method_name)
        if entry is None:
            return None
        stub_cls, attr = entry
        stub = self._stub_cache.get(stub_cls)
        if stub is None:
            stub = stub_cls(self._channel)
            self._stub_cache[stub_cls] = stub
        bound = getattr(stub, attr)
        self._method_cache[method_name] = bound
        return bound


__all__ = ["StubDispatcher", "_METHOD_STUB_MAP"]
