# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public gRPC entry point for the plugin registry.

Wraps a :class:`PluginRegistryServer` (single-process Python methods)
and exposes its 4 RPCs over the network so external plugin processes
can register / heartbeat / unregister / list themselves without
needing to be inside the planner Python process.

The class is deliberately thin: every RPC just converts proto →
Pydantic via ``_proto_bridge``, calls the underlying
``PluginRegistryServer`` method (which already enforces auth /
protocol / dedup / endpoint-scheme), and converts the response back.
That keeps the auth + reject-reason contract identical between the
in-process call site and the gRPC call site — operators never have
two diverging code paths to reason about.

Lifecycle
---------

``start_gateway_server`` returns the bound ``grpc.aio.Server`` so the
caller (``NativePlannerBase``-equivalent startup hook) controls
``await server.stop(grace=...)`` on shutdown. Don't park the gRPC
server's lifecycle inside this module — it has to coordinate with the
planner's own shutdown sequence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import grpc

from dynamo.planner.plugins._proto_bridge import (
    proto_to_pydantic,
    pydantic_to_proto,
)
from dynamo.planner.plugins.proto.v1 import plugin_pb2 as pb
from dynamo.planner.plugins.proto.v1 import plugin_pb2_grpc as pbg
from dynamo.planner.plugins.registry.server import PluginRegistryServer
from dynamo.planner.plugins.types import (
    HeartbeatRequest,
    HeartbeatResponse,
    ListPluginsRequest,
    ListPluginsResponse,
    PluginInfo,
    RegisterRequest,
    RegisterResponse,
    UnregisterRequest,
    UnregisterResponse,
)

log = logging.getLogger(__name__)


class PluginRegistryGatewayServicer(pbg.PluginRegistryServicer):
    """Thin proto adapter over :class:`PluginRegistryServer`.

    All 4 RPCs follow the same shape:

    1. ``proto_to_pydantic`` the request (failure → INVALID_ARGUMENT)
    2. ``await self._server.<method>(pyd_request)`` (the underlying
       method is the same one in-process callers use, so auth / dedup
       / circuit-breaker contracts are identical)
    3. ``pydantic_to_proto`` the response

    Authentication is performed *inside* ``server.register()`` (it
    consults the configured ``AuthValidator``) — the gateway does NOT
    duplicate that logic. Keeps a single source of truth for "what is
    accepted".
    """

    def __init__(self, server: PluginRegistryServer) -> None:
        self._server = server

    async def Register(
        self,
        request: pb.RegisterRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.RegisterResponse:
        try:
            pyd_req: RegisterRequest = proto_to_pydantic(request)
        except Exception as exc:  # pragma: no cover (defensive)
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"register: malformed request: {type(exc).__name__}: {exc}",
            )
        pyd_resp: RegisterResponse = await self._server.register(pyd_req)
        return pydantic_to_proto(pyd_resp)

    async def Heartbeat(
        self,
        request: pb.HeartbeatRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.HeartbeatResponse:
        try:
            pyd_req: HeartbeatRequest = proto_to_pydantic(request)
        except Exception as exc:  # pragma: no cover
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"heartbeat: malformed request: {type(exc).__name__}: {exc}",
            )
        # ``server.heartbeat`` is a historical helper that takes
        # plugin_id directly and returns bool. Wrap the bool into
        # ``HeartbeatResponse`` for the gRPC contract.
        ok = await self._server.heartbeat(pyd_req.plugin_id)
        return pydantic_to_proto(HeartbeatResponse(ok=ok))

    async def Unregister(
        self,
        request: pb.UnregisterRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.UnregisterResponse:
        try:
            pyd_req: UnregisterRequest = proto_to_pydantic(request)
        except Exception as exc:  # pragma: no cover
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"unregister: malformed request: {type(exc).__name__}: {exc}",
            )
        # ``server.unregister`` takes (plugin_id, *, reason) — match
        # the in-process method's signature; it returns bool, not a
        # Pydantic message (historical), so we wrap.
        ok = await self._server.unregister(
            pyd_req.plugin_id, reason=pyd_req.reason
        )
        return pydantic_to_proto(UnregisterResponse(ok=ok))

    async def ListPlugins(
        self,
        request: pb.ListPluginsRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.ListPluginsResponse:
        try:
            pyd_req: ListPluginsRequest = proto_to_pydantic(request)
        except Exception as exc:  # pragma: no cover
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"list_plugins: malformed request: {type(exc).__name__}: {exc}",
            )
        infos: list[PluginInfo] = self._server.list_plugins(pyd_req)
        # ListPluginsResponse wraps a repeated PluginInfo — build the
        # Pydantic mirror first then convert. Per-PluginInfo conversion
        # is exercised by _proto_bridge round-trip tests.
        pyd_resp = ListPluginsResponse(plugins=list(infos))
        return pydantic_to_proto(pyd_resp)


# ---------------------------------------------------------------------------
# Server lifecycle helper
# ---------------------------------------------------------------------------


async def start_gateway_server(
    server: PluginRegistryServer,
    *,
    listen: str,
    server_credentials: Optional[grpc.ServerCredentials] = None,
) -> tuple[grpc.aio.Server, str]:
    """Build and start a gRPC server hosting :class:`PluginRegistryGatewayServicer`.

    Args:
        server: the in-process registry the gateway should delegate to.
        listen: bind address.
            - ``unix:///path/to/sock`` for same-Pod sidecar plugins
            - ``host:port`` for cross-Pod (use ``0.0.0.0:N`` to bind
              all interfaces; ``:0`` for an ephemeral port)
            - ``[::]:N`` for IPv6
        server_credentials: optional ``ssl_server_credentials`` for
            mTLS. ``None`` uses an insecure port — only acceptable in
            same-Pod sidecar deployments where the Pod boundary is the
            trust boundary, OR in tests.

    Returns:
        ``(grpc_server, actual_listen)``. The caller is responsible
        for ``await grpc_server.stop(grace=...)`` on planner shutdown.
        ``actual_listen`` echoes ``listen`` unless an ephemeral port
        was requested (``:0``), in which case it carries the bound port.
    """
    grpc_server = grpc.aio.server()
    pbg.add_PluginRegistryServicer_to_server(
        PluginRegistryGatewayServicer(server), grpc_server
    )
    if server_credentials is not None:
        port = grpc_server.add_secure_port(listen, server_credentials)
    else:
        port = grpc_server.add_insecure_port(listen)
    await grpc_server.start()
    actual_listen = listen
    if listen.endswith(":0"):
        host = listen.rsplit(":", 1)[0]
        actual_listen = f"{host}:{port}"
    log.info(
        "plugin registry gateway listening at %s (secure=%s)",
        actual_listen,
        server_credentials is not None,
    )
    return grpc_server, actual_listen


__all__ = [
    "PluginRegistryGatewayServicer",
    "start_gateway_server",
]
