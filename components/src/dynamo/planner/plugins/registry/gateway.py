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
from typing import Optional

import grpc

from dynamo.planner.plugins._proto_bridge import proto_to_pydantic, pydantic_to_proto
from dynamo.planner.plugins.proto.v1 import plugin_pb2 as pb
from dynamo.planner.plugins.proto.v1 import plugin_pb2_grpc as pbg
from dynamo.planner.plugins.registry.server import PluginRegistryServer
from dynamo.planner.plugins.types import (
    HeartbeatRequest,
    HeartbeatResponse,
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
        ok, reject = await self._server.authenticated_heartbeat(
            pyd_req.plugin_id, pyd_req.auth_token
        )
        if reject == "auth_failed":
            await context.abort(
                grpc.StatusCode.UNAUTHENTICATED, "heartbeat: auth_failed"
            )
        if reject == "permission_denied":
            await context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "heartbeat: caller subject does not match registered plugin",
            )
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
        ok, reject = await self._server.authenticated_unregister(
            pyd_req.plugin_id, pyd_req.auth_token, reason=pyd_req.reason
        )
        if reject == "auth_failed":
            await context.abort(
                grpc.StatusCode.UNAUTHENTICATED, "unregister: auth_failed"
            )
        if reject == "permission_denied":
            await context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "unregister: caller subject does not match registered plugin",
            )
        return pydantic_to_proto(UnregisterResponse(ok=ok))

    async def ListPlugins(  # type: ignore[return]
        self,
        request: pb.ListPluginsRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.ListPluginsResponse:
        # ListPlugins exposes the full plugin inventory (ids + endpoints +
        # circuit state). The plugin-level ``auth_token`` model used for
        # Register / Heartbeat / Unregister is not the right gate here —
        # that's an admin-level RBAC concern. Until ``AdminAuthConfig`` is
        # wired (PR 1.5, alongside k8s_sa / mTLS), the gateway default-denies
        # this RPC. In-process callers (Prometheus exporter,
        # ``PluginRegistryServer.list_plugins``) are unaffected.
        await context.abort(
            grpc.StatusCode.PERMISSION_DENIED,
            "list_plugins: admin authentication is not yet wired over gRPC; "
            "use the in-process registry method until admin RBAC lands.",
        )


# ---------------------------------------------------------------------------
# Server lifecycle helper
# ---------------------------------------------------------------------------


async def start_gateway_server(
    server: PluginRegistryServer,
    *,
    listen: str,
    server_credentials: Optional[grpc.ServerCredentials] = None,
    allow_insecure: bool = False,
) -> tuple[grpc.aio.Server, str]:
    """Build and start a gRPC server hosting :class:`PluginRegistryGatewayServicer`.

    Args:
        server: the in-process registry the gateway should delegate to.
        listen: bind address, passed verbatim to
            ``grpc.aio.server.add_insecure_port`` /
            ``add_secure_port``. Both accept gRPC's URI scheme:
            - ``unix:/abs/path`` (or ``unix:///abs/path``) for a Unix
              domain socket — used when plugins register from the same
              Pod and the Pod boundary is the trust boundary.
            - ``host:port`` for TCP (use ``0.0.0.0:N`` to bind all
              interfaces; ``:0`` for an ephemeral port).
            - ``[::]:N`` for IPv6.
            Note that this is the *gateway listen address*; the
            plugins' own callback endpoints (``RegisterRequest.endpoint``)
            are restricted to ``inproc://`` and ``grpc://`` by
            ``make_transport_for_endpoint``.
        server_credentials: optional ``ssl_server_credentials`` for
            mTLS. mTLS itself lands in a follow-up PR — PR #1 callers
            pass ``None`` (insecure port). Insecure is acceptable for
            UDS bind (Pod-local trust) or tests; for cross-Pod TCP it
            relies on K8s NetworkPolicy / Pod-to-Pod identity until
            mTLS lands.

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
        # Plaintext bind. The gateway receives plugins' shared-secret
        # ``auth_token``, so an insecure TCP listen would leak it on the
        # wire. Fail closed on TCP unless the operator explicitly opts in
        # via ``allow_insecure`` — mirroring the outbound transport's
        # ``allow_insecure_grpc`` gate. ``unix:`` (Pod-local) listens are
        # always allowed: the Pod boundary is the trust boundary.
        is_unix = listen.startswith("unix:")
        if not is_unix and not allow_insecure:
            raise RuntimeError(
                f"refusing to bind plaintext (no-TLS) gRPC gateway on TCP "
                f"listen {listen!r}: it would expose plugin auth tokens on "
                f"the wire. Use a ``unix:`` listen, supply TLS credentials, "
                f"or set ``gateway.allow_insecure=true`` to accept the risk."
            )
        if not is_unix:
            log.warning(
                "plugin registry gateway binding PLAINTEXT (no TLS) on TCP "
                "%r — plugin auth tokens cross the wire unencrypted; "
                "allow_insecure=true was set. Prefer a unix: socket or mTLS.",
                listen,
            )
        port = grpc_server.add_insecure_port(listen)
    # ``add_*_port`` returns 0 when the bind fails (port in use, bad
    # address, permission denied on a unix socket path, etc). Catch this
    # before starting so the operator sees a clear error instead of a
    # silently-running gateway that never accepts connections.
    if port == 0:
        raise RuntimeError(
            f"plugin registry gateway failed to bind {listen!r} — "
            "address may be in use, the path may be unwritable, or the "
            "scheme may be malformed"
        )
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
