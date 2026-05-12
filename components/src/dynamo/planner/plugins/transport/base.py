# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``PluginTransport`` ABC — unified contract for plugin RPC invocation.

All three transports (in-process / uds / grpc) implement this interface.
The orchestrator's pipeline driver treats them uniformly via
``await plugin.transport.call(method, request)``.
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
    """Endpoint URL (``inproc://...`` | ``unix:///...`` | ``grpc://...``)."""

    timeout_seconds: float
    """Per-RPC timeout (orchestrator wraps each ``call()`` in ``asyncio.wait_for``)."""

    @abc.abstractmethod
    async def call(self, method: str, request: Any) -> Any:
        """Invoke a plugin RPC by method name.

        Args:
            method: RPC method name. One of ``"Predict"`` / ``"Propose"`` /
                ``"Reconcile"`` / ``"Constrain"`` / ``"Bootstrap"`` / ``"Reset"``.
                For ``InProcessTransport``: must be a method name on the
                Python plugin instance. For UDS / gRPC: must be a registered
                stub method.
            request: proto generated message instance (e.g.
                ``ProposeStageRequest``). NOT a Pydantic mirror —
                orchestrator converts before calling.

        Returns:
            proto generated response message (e.g. ``ProposeStageResponse``).

        Raises:
            PluginTimeoutError: ``asyncio.wait_for(timeout=self.timeout_seconds)`` expired
            PluginUnknownMethodError: method not registered on this plugin
            PluginConnectionError: transport-layer failure (UDS socket missing,
                gRPC channel disconnected, mTLS handshake failed, ...)
            PluginSerializationError: request / response (de)serialization failure
            PluginCallError: catch-all for plugin-internal exceptions
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self) -> None:
        """Release transport resources.

        MUST be idempotent — orchestrator shutdown may invoke multiple times.

        - ``InProcessTransport``: no-op (plugin instance lifecycle owned by orchestrator)
        - ``UdsTransport`` / ``GrpcTransport``: close the gRPC channel
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}(plugin_id={self.plugin_id!r}, endpoint={self.endpoint!r})"


__all__ = ["PluginTransport"]
