# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin call error hierarchy.

All transport ``call()`` failures MUST raise a ``PluginCallError``
subclass — no naked exceptions, no silent return. The subtype hierarchy
lets the orchestrator decide selectively: timeout → circuit breaker;
connection → reconnect; unknown method → contract violation;
serialization → plugin bug audit.
"""

from __future__ import annotations


class PluginCallError(Exception):
    """Base for all plugin transport / RPC errors.

    Attributes:
        plugin_id: which plugin raised
        method: which RPC method was invoked
        cause: original exception (if any), preserved via ``raise ... from cause``
    """

    def __init__(
        self,
        message: str,
        *,
        plugin_id: str = "",
        method: str = "",
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.plugin_id = plugin_id
        self.method = method
        self.cause = cause


class PluginTimeoutError(PluginCallError):
    """``asyncio.wait_for`` exceeded ``request_timeout_seconds`` for this RPC.

    Orchestrator increments circuit breaker failure count.
    """


class PluginConnectionError(PluginCallError):
    """Transport-layer connection failure: socket missing, channel down,
    DNS error, mTLS handshake failed, etc.

    Orchestrator may attempt reconnection on next tick (UDS / gRPC); for
    in-process this should never occur.
    """


class PluginSerializationError(PluginCallError):
    """Request / response (de)serialization failed.

    Common causes:
    - proto schema mismatch between orchestrator and plugin
    - bytes encoding mismatch in ``FpmData`` (e.g. msgspec vs proto)

    Note: a plugin returning a response with an empty ``oneof result`` is
    NOT a serialization error — it is treated as silent ACCEPT for graceful
    degradation (see ``plugins/proto/v1/README.md`` and the inline note in
    ``pipeline.py:_response_to_plugin_result``). Only true bytes-level
    decode failures land here.
    """


class PluginUnknownMethodError(PluginCallError):
    """Requested method name not found on plugin.

    For ``InProcessTransport``: ``getattr(instance, method)`` returned None.
    For ``GrpcTransport``: stub map has no entry for method.

    This indicates a programming bug in the orchestrator (calling wrong stage)
    or a plugin missing required RPC handlers.
    """


__all__ = [
    "PluginCallError",
    "PluginTimeoutError",
    "PluginConnectionError",
    "PluginSerializationError",
    "PluginUnknownMethodError",
]
