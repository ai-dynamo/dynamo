# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``PluginTransport`` ABC — unified contract for plugin RPC invocation.

PR #1 ships two transports (in-process / grpc) under this interface; the
orchestrator's pipeline driver treats them uniformly via
``await plugin.transport.call(method, request)``. A dedicated
``UdsTransport`` is deferred to a follow-up PR — see
``plugins/transport/README.md`` "Deferred" section.
"""

from __future__ import annotations

import abc
from typing import Any


class PluginTransport(abc.ABC):
    """Abstract transport interface for plugin RPC invocation.

    **Lifecycle**:
    - Constructed once per plugin (during register / register_internal)
    - ``call(method, request)`` invoked many times across ticks
    - ``close()`` called once during plugin unregister or orchestrator shutdown
        (must be idempotent — orchestrator may call multiple times defensively)

    **Concurrency**:
    - Single-threaded asyncio model
    - ``call()`` is async; multiple concurrent calls to the SAME transport
      from different ``asyncio.gather`` branches are safe (gRPC channel
      multiplexing handles it)
    - Concurrent calls from MULTIPLE event loops is UB

    **Error contract**:
    - ALL failures MUST raise a ``PluginCallError`` subclass
    - Specifically: timeout → ``PluginTimeoutError``; connection failure →
      ``PluginConnectionError``; method not found → ``PluginUnknownMethodError``;
      (de)serialization → ``PluginSerializationError``
    - Subclasses MUST NOT swallow exceptions or return error sentinels
    """

    plugin_id: str
    """Plugin identifier (matches ``RegisterRequest.plugin_id``)."""

    endpoint: str
    """Endpoint URL — ``inproc://<plugin_id>`` for in-process plugins,
    ``grpc://host:port`` for out-of-process. ``make_transport_for_endpoint``
    rejects other schemes."""

    timeout_seconds: float
    """Per-RPC timeout. The **transport** enforces this internally — each
    ``call()`` wraps its own dispatch in ``asyncio.wait_for(...,
    timeout_seconds)`` (see ``in_process.py`` / ``_grpc_base.py``). The
    pipeline driver deliberately does NOT wrap calls (it uses a bare
    ``asyncio.gather``) so the per-plugin deadline isn't double-counted."""

    @abc.abstractmethod
    async def call(self, method: str, request: Any) -> Any:
        """Invoke a plugin RPC by method name.

        Args:
            method: RPC method name. One of ``"Predict"`` / ``"Propose"`` /
                ``"Reconcile"`` / ``"Constrain"`` / ``"Bootstrap"`` / ``"Reset"``.
                For ``InProcessTransport``: must be a method name on the
                Python plugin instance. For ``GrpcTransport``: must be a
                registered stub method.
            request: proto generated message instance (e.g.
                ``ProposeStageRequest``) — or a Pydantic mirror; the gRPC
                transport accepts both and converts at the wire boundary
                (see ``_GrpcTransportBase`` for the bridge).

        Returns:
            proto generated response message (e.g. ``ProposeStageResponse``).

        Raises:
            PluginTimeoutError: ``asyncio.wait_for(timeout=self.timeout_seconds)`` expired
            PluginUnknownMethodError: method not registered on this plugin
            PluginConnectionError: transport-layer failure (gRPC channel
                disconnected, unreachable endpoint, ...)
            PluginSerializationError: request / response (de)serialization failure
            PluginCallError: catch-all for plugin-internal exceptions
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self) -> None:
        """Release transport resources.

        MUST be idempotent — orchestrator shutdown may invoke multiple times.

        - ``InProcessTransport``: no-op (plugin instance lifecycle owned by orchestrator)
        - ``GrpcTransport``: close the gRPC channel
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}(plugin_id={self.plugin_id!r}, endpoint={self.endpoint!r})"


__all__ = ["PluginTransport"]
