# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared gRPC channel base for ``GrpcTransport``.

Factored as a base class so that follow-up transports (e.g. a future
``UdsTransport`` over the gRPC ``unix:`` URI scheme, or an mTLS variant)
can reuse the call dispatch + error mapping path and only override
channel construction. In PR #1 only ``GrpcTransport`` (plaintext, gated
by ``allow_insecure_grpc``) subclasses this.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import grpc
from google.protobuf.message import Message as ProtoMessage
from pydantic import BaseModel

from dynamo.planner.plugins._proto_bridge import proto_to_pydantic, pydantic_to_proto
from dynamo.planner.plugins.transport._method_dispatch import StubDispatcher
from dynamo.planner.plugins.transport.base import PluginTransport
from dynamo.planner.plugins.transport.errors import (
    PluginCallError,
    PluginConnectionError,
    PluginSerializationError,
    PluginTimeoutError,
    PluginUnknownMethodError,
)

log = logging.getLogger(__name__)

_DEFAULT_KEEPALIVE_TIME_MS = 30_000
_DEFAULT_MAX_MESSAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def grpc_channel_options(
    *,
    keepalive_time_ms: int = _DEFAULT_KEEPALIVE_TIME_MS,
    max_message_size_bytes: int = _DEFAULT_MAX_MESSAGE_SIZE_BYTES,
) -> list[tuple[str, int]]:
    """Build the per-channel gRPC option list for a plugin transport.

    Centralised so individual plugins can't quietly override the
    common options (keepalive timing, max ping behaviour), but the
    operator-tunable knobs (``keepalive_time_ms``,
    ``max_message_size_bytes``) are honoured per the user's
    ``TransportConfig``.
    """
    return [
        ("grpc.keepalive_time_ms", keepalive_time_ms),
        ("grpc.keepalive_timeout_ms", 10_000),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.max_send_message_length", max_message_size_bytes),
        ("grpc.max_receive_message_length", max_message_size_bytes),
    ]


class _GrpcTransportBase(PluginTransport):
    """Shared call/close logic for gRPC-based transports.

    Subclasses must:
    - set ``self.plugin_id`` / ``self.endpoint`` / ``self.timeout_seconds``
      in ``__init__``
    - implement ``_build_channel()`` to construct the appropriate
      ``grpc.aio.Channel`` (insecure UDS / insecure TCP / secure mTLS TCP)
    """

    def __init__(
        self,
        plugin_id: str,
        endpoint: str,
        timeout_seconds: float,
        *,
        keepalive_time_ms: int = _DEFAULT_KEEPALIVE_TIME_MS,
        max_message_size_bytes: int = _DEFAULT_MAX_MESSAGE_SIZE_BYTES,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                f"{type(self).__name__}(plugin_id={plugin_id!r}): "
                f"timeout_seconds must be positive, got {timeout_seconds}"
            )
        self.plugin_id = plugin_id
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.keepalive_time_ms = keepalive_time_ms
        self.max_message_size_bytes = max_message_size_bytes
        self._channel: grpc.aio.Channel | None = None
        self._dispatcher: StubDispatcher | None = None
        self._closed = False
        self._channel_lock = asyncio.Lock()

    def _build_channel(self) -> grpc.aio.Channel:  # pragma: no cover (abstract)
        raise NotImplementedError

    async def _ensure_channel(self) -> StubDispatcher:
        if self._dispatcher is not None:
            return self._dispatcher
        async with self._channel_lock:
            if self._dispatcher is None:
                if self._closed:
                    raise PluginConnectionError(
                        f"plugin {self.plugin_id!r}: transport already closed",
                        plugin_id=self.plugin_id,
                    )
                try:
                    self._channel = self._build_channel()
                except Exception as e:
                    raise PluginConnectionError(
                        f"plugin {self.plugin_id!r}: failed to build gRPC channel "
                        f"to {self.endpoint!r}: {type(e).__name__}: {e}",
                        plugin_id=self.plugin_id,
                        cause=e,
                    ) from e
                self._dispatcher = StubDispatcher(self._channel)
        return self._dispatcher

    async def call(self, method: str, request: Any) -> Any:
        if self._closed:
            raise PluginConnectionError(
                f"plugin {self.plugin_id!r}: cannot call {method!r} — transport closed",
                plugin_id=self.plugin_id,
                method=method,
            )
        dispatcher = await self._ensure_channel()
        rpc = dispatcher.get_method(method)
        if rpc is None:
            raise PluginUnknownMethodError(
                f"method {method!r} not in dispatch table; "
                f"plugin {self.plugin_id!r} cannot serve it",
                plugin_id=self.plugin_id,
                method=method,
            )
        # Pydantic ↔ proto bridging at the wire boundary.
        # The pipeline emits Pydantic stage requests (so it can keep
        # using attribute-style access on the way back). gRPC stubs
        # need proto messages. Convert here, mirror back on the way
        # out so callers always see whatever they sent — Pydantic in →
        # Pydantic out; proto in (e.g. transport contract test) →
        # proto out. Without this, every external gRPC plugin call fails
        # at gRPC serialisation — found while writing the first real
        # external-plugin e2e test.
        request_was_pyd = isinstance(request, BaseModel)
        try:
            wire_request: Any = (
                pydantic_to_proto(request) if request_was_pyd else request
            )
        except KeyError as e:
            # Unmapped Pydantic request type — surface as a typed
            # transport error so callers see the documented contract
            # (PluginCallError hierarchy) rather than a raw KeyError.
            raise PluginSerializationError(
                f"plugin {self.plugin_id!r} method {method!r}: unmapped "
                f"Pydantic request class {type(request).__name__} ({e})",
                plugin_id=self.plugin_id,
                method=method,
                cause=e,
            ) from e
        try:
            wire_response = await asyncio.wait_for(
                rpc(wire_request), self.timeout_seconds
            )
        except asyncio.TimeoutError as e:
            raise PluginTimeoutError(
                f"plugin {self.plugin_id!r} method {method!r} exceeded "
                f"timeout_seconds={self.timeout_seconds}",
                plugin_id=self.plugin_id,
                method=method,
                cause=e,
            ) from e
        except grpc.aio.AioRpcError as e:
            code = e.code()
            details = e.details() or ""
            # Map gRPC status codes to typed call errors
            if code == grpc.StatusCode.UNAVAILABLE:
                raise PluginConnectionError(
                    f"plugin {self.plugin_id!r} method {method!r}: "
                    f"endpoint unreachable ({details})",
                    plugin_id=self.plugin_id,
                    method=method,
                    cause=e,
                ) from e
            if code == grpc.StatusCode.UNIMPLEMENTED:
                raise PluginUnknownMethodError(
                    f"plugin {self.plugin_id!r} did not implement method {method!r}",
                    plugin_id=self.plugin_id,
                    method=method,
                    cause=e,
                ) from e
            if code in (grpc.StatusCode.INTERNAL, grpc.StatusCode.DATA_LOSS):
                raise PluginSerializationError(
                    f"plugin {self.plugin_id!r} method {method!r}: "
                    f"serialization or internal error ({code.name}: {details})",
                    plugin_id=self.plugin_id,
                    method=method,
                    cause=e,
                ) from e
            raise PluginCallError(
                f"plugin {self.plugin_id!r} method {method!r}: "
                f"gRPC error {code.name}: {details}",
                plugin_id=self.plugin_id,
                method=method,
                cause=e,
            ) from e
        except PluginCallError:
            raise
        except Exception as e:
            raise PluginCallError(
                f"plugin {self.plugin_id!r} method {method!r} raised "
                f"{type(e).__name__}: {e}",
                plugin_id=self.plugin_id,
                method=method,
                cause=e,
            ) from e

        # Symmetric conversion on the response. Treat anything that
        # isn't a proto Message as already-Pydantic / unknown and
        # leave it alone (defensive — should never happen in practice
        # because gRPC stubs always return proto). If the caller gave
        # us proto in, give proto back: this preserves the transport
        # contract test's roundtrip-equivalence assertions.
        if request_was_pyd and isinstance(wire_response, ProtoMessage):
            try:
                return proto_to_pydantic(wire_response)
            except KeyError as e:
                raise PluginSerializationError(
                    f"plugin {self.plugin_id!r} method {method!r}: "
                    f"unmapped response proto class {type(wire_response).__name__} "
                    f"({e})",
                    plugin_id=self.plugin_id,
                    method=method,
                    cause=e,
                ) from e
        return wire_response

    async def close(self) -> None:
        # Idempotent
        if self._closed:
            return
        self._closed = True
        if self._channel is not None:
            try:
                await self._channel.close()
            except Exception as exc:  # noqa: BLE001 — close must be idempotent
                # ``close`` must not raise to caller (it's the planner
                # shutdown path; one buggy plugin must not stall the
                # remaining cleanup). Surface the failure via the
                # logger so it isn't silently lost.
                log.warning(
                    "plugin %r transport close raised %s: %s",
                    self.plugin_id,
                    type(exc).__name__,
                    exc,
                )
        self._channel = None
        self._dispatcher = None


__all__ = ["_GrpcTransportBase", "grpc_channel_options"]
