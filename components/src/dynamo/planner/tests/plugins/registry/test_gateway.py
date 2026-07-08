# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``PluginRegistryGatewayServicer``.

Focuses on the auth-gating contract for ``Heartbeat`` / ``Unregister`` /
``ListPlugins`` — the in-process methods are exercised by
``test_server.py``; this file checks that the gateway layer maps server
``(ok, reject)`` results onto the correct gRPC status codes and does NOT
let unauthenticated callers reach destructive operations.

Uses a stub ``ServicerContext`` that records ``abort`` and raises
``grpc.aio.AbortError`` so the servicer's ``raise # unreachable`` lines
behave the same way as on a real gRPC connection.
"""

from __future__ import annotations

from typing import Any

import grpc
import pytest

from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.proto.v1 import plugin_pb2 as pb
from dynamo.planner.plugins.registry.auth import AuthIdentity, AuthValidator
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.errors import AuthError
from dynamo.planner.plugins.registry.gateway import PluginRegistryGatewayServicer
from dynamo.planner.plugins.registry.server import PluginRegistryServer
from dynamo.planner.plugins.transport.base import PluginTransport

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _StubTransport(PluginTransport):
    def __init__(self, plugin_id, endpoint, *, in_process_instance=None):
        self.plugin_id = plugin_id
        self.endpoint = endpoint
        self.timeout_seconds = 1.0
        self.instance = in_process_instance
        self.closed = False

    async def call(self, method, request):
        return None

    async def close(self):
        self.closed = True


class _PerTokenAuth(AuthValidator):
    """token → subject; ``"bad"`` raises so auth_failed is reachable."""

    async def validate(self, token: str) -> AuthIdentity:
        if token == "bad":
            raise AuthError("invalid")
        return AuthIdentity(source="static_secret", subject=f"subj-{token}")


class _FakeContext:
    """Minimal ``grpc.aio.ServicerContext`` substitute.

    Records the ``abort`` call (code + message) and raises
    ``grpc.aio.AbortError`` so the servicer's post-abort ``raise`` is
    consistent with real RPC behaviour.
    """

    def __init__(self) -> None:
        self.aborted_code: Any = None
        self.aborted_message: str = ""

    async def abort(self, code, message: str) -> None:
        self.aborted_code = code
        self.aborted_message = message
        raise grpc.aio.AbortError()


def _make_servicer():
    clock = VirtualClock()
    cb = CircuitBreaker(clock)

    def factory(plugin_id, endpoint, *, in_process_instance=None):
        return _StubTransport(
            plugin_id, endpoint, in_process_instance=in_process_instance
        )

    server = PluginRegistryServer(
        clock=clock,
        auth=_PerTokenAuth(),
        circuit_breaker=cb,
        transport_factory=factory,
        protocol_versions=("1.0", "1.0"),
    )
    servicer = PluginRegistryGatewayServicer(server)
    return server, servicer


async def _register(server, plugin_id="p1", auth_token="A"):
    req = pb.RegisterRequest(
        plugin_id=plugin_id,
        plugin_type="propose",
        priority=10,
        endpoint="grpc://127.0.0.1:9000",
        version="1.0.0",
        protocol_version="1.0",
        auth_token=auth_token,
    )
    # Use the in-process method (already exercised in test_server.py); the
    # gateway's Register path goes through this same method too.
    from dynamo.planner.plugins._proto_bridge import proto_to_pydantic

    resp = await server.register(proto_to_pydantic(req))
    assert resp.accepted is True


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_matching_token_returns_ok():
    server, svc = _make_servicer()
    await _register(server)
    resp = await svc.Heartbeat(
        pb.HeartbeatRequest(plugin_id="p1", auth_token="A"), _FakeContext()
    )
    assert resp.ok is True


@pytest.mark.asyncio
async def test_heartbeat_invalid_token_aborts_unauthenticated():
    server, svc = _make_servicer()
    await _register(server)
    ctx = _FakeContext()
    with pytest.raises(grpc.aio.AbortError):
        await svc.Heartbeat(pb.HeartbeatRequest(plugin_id="p1", auth_token="bad"), ctx)
    assert ctx.aborted_code == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.asyncio
async def test_heartbeat_subject_mismatch_aborts_permission_denied():
    server, svc = _make_servicer()
    await _register(server, auth_token="A")
    ctx = _FakeContext()
    with pytest.raises(grpc.aio.AbortError):
        # token "B" validates but maps to a different subject.
        await svc.Heartbeat(pb.HeartbeatRequest(plugin_id="p1", auth_token="B"), ctx)
    assert ctx.aborted_code == grpc.StatusCode.PERMISSION_DENIED


# ---------------------------------------------------------------------------
# Unregister — the destructive RPC. Forged calls MUST NOT evict.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unregister_matching_token_removes_plugin():
    server, svc = _make_servicer()
    await _register(server)
    resp = await svc.Unregister(
        pb.UnregisterRequest(plugin_id="p1", auth_token="A", reason="shutdown"),
        _FakeContext(),
    )
    assert resp.ok is True
    assert server.get_plugin("p1") is None


@pytest.mark.asyncio
async def test_unregister_invalid_token_aborts_and_keeps_plugin():
    server, svc = _make_servicer()
    await _register(server)
    ctx = _FakeContext()
    with pytest.raises(grpc.aio.AbortError):
        await svc.Unregister(
            pb.UnregisterRequest(plugin_id="p1", auth_token="bad"), ctx
        )
    assert ctx.aborted_code == grpc.StatusCode.UNAUTHENTICATED
    assert server.get_plugin("p1") is not None  # NOT evicted


@pytest.mark.asyncio
async def test_unregister_subject_mismatch_aborts_and_keeps_plugin():
    """Core security guarantee: a *valid* token from a different subject
    cannot evict another caller's plugin."""
    server, svc = _make_servicer()
    await _register(server, auth_token="A")
    ctx = _FakeContext()
    with pytest.raises(grpc.aio.AbortError):
        await svc.Unregister(pb.UnregisterRequest(plugin_id="p1", auth_token="B"), ctx)
    assert ctx.aborted_code == grpc.StatusCode.PERMISSION_DENIED
    assert server.get_plugin("p1") is not None  # NOT evicted


# ---------------------------------------------------------------------------
# ListPlugins — default-deny over gRPC until admin auth lands (PR 1.5).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_plugins_over_gateway_default_denied():
    _, svc = _make_servicer()
    ctx = _FakeContext()
    with pytest.raises(grpc.aio.AbortError):
        await svc.ListPlugins(pb.ListPluginsRequest(), ctx)
    assert ctx.aborted_code == grpc.StatusCode.PERMISSION_DENIED
    # Error message must direct the operator to the in-process path so they
    # have an escape hatch until admin RBAC lands.
    assert "in-process" in ctx.aborted_message.lower()


# ---------------------------------------------------------------------------
# start_gateway_server bind-failure handling.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_gateway_server_raises_when_port_zero():
    """``add_insecure_port`` returns 0 on bind failure (port in use,
    bad address, etc). The helper must catch that BEFORE starting the
    server so operators see a clear RuntimeError instead of a silently
    running gateway that accepts no connections."""
    from dynamo.planner.plugins.registry import gateway as gw_mod
    from dynamo.planner.plugins.registry.gateway import start_gateway_server

    server, _ = _make_servicer()

    class _StubAioServer:
        def __init__(self) -> None:
            self.started = False

        def add_generic_rpc_handlers(self, _handlers: Any) -> None:
            # Called by ``add_PluginRegistryServicer_to_server``;
            # no-op stub for the bind-failure test.
            pass

        def add_registered_method_handlers(
            self, _service_name: str, _method_handlers: Any
        ) -> None:
            # Newer grpc.aio adds this alongside add_generic_rpc_handlers;
            # we accept it as a no-op so the servicer install succeeds.
            pass

        def add_insecure_port(self, _listen: str) -> int:  # noqa: D401
            # Simulate a bind failure.
            return 0

        def add_secure_port(self, _listen: str, _creds: Any) -> int:
            return 0

        async def start(self) -> None:
            self.started = True

        async def stop(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    stub = _StubAioServer()
    # Monkeypatch grpc.aio.server() to return our stub.
    real_factory = gw_mod.grpc.aio.server
    gw_mod.grpc.aio.server = lambda: stub  # type: ignore[assignment]
    try:
        # allow_insecure=True so we exercise the port==0 bind-failure path
        # rather than the plaintext-TCP fail-closed guard (a separate test).
        with pytest.raises(RuntimeError, match="failed to bind"):
            await start_gateway_server(server, listen="0.0.0.0:1", allow_insecure=True)
    finally:
        gw_mod.grpc.aio.server = real_factory  # type: ignore[assignment]
    assert stub.started is False, (
        "start_gateway_server must fail fast BEFORE calling grpc_server.start() "
        "when add_*_port() reports a bind failure"
    )


# ---------------------------------------------------------------------------
# Plaintext-TCP fail-closed gate (review #6).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_gateway_server_refuses_plaintext_tcp_without_allow_insecure():
    """A TCP listen with no TLS creds and allow_insecure=False must fail
    closed — the gateway receives plugins' shared-secret auth tokens, so a
    plaintext TCP bind would leak them. Mirrors the outbound
    allow_insecure_grpc gate. The check raises BEFORE any bind."""
    from dynamo.planner.plugins.registry.gateway import start_gateway_server

    server, _ = _make_servicer()
    with pytest.raises(RuntimeError, match="refusing to bind plaintext"):
        await start_gateway_server(server, listen="0.0.0.0:9099", allow_insecure=False)


@pytest.mark.asyncio
async def test_start_gateway_server_allows_plaintext_tcp_when_opted_in():
    """allow_insecure=True permits the plaintext TCP bind (operator accepted
    the risk); it must not hit the fail-closed guard."""
    from dynamo.planner.plugins.registry import gateway as gw_mod
    from dynamo.planner.plugins.registry.gateway import start_gateway_server

    server, _ = _make_servicer()

    class _StubAioServer:
        def __init__(self) -> None:
            self.started = False

        def add_generic_rpc_handlers(self, _handlers: Any) -> None:
            pass

        def add_registered_method_handlers(self, _s: str, _m: Any) -> None:
            pass

        def add_insecure_port(self, _listen: str) -> int:
            return 9099  # simulate a successful bind

        async def start(self) -> None:
            self.started = True

        async def stop(self, *_a: Any, **_k: Any) -> None:
            pass

    stub = _StubAioServer()
    real_factory = gw_mod.grpc.aio.server
    gw_mod.grpc.aio.server = lambda: stub  # type: ignore[assignment]
    try:
        srv, listen = await start_gateway_server(
            server, listen="0.0.0.0:9099", allow_insecure=True
        )
    finally:
        gw_mod.grpc.aio.server = real_factory  # type: ignore[assignment]
    assert stub.started is True
    assert listen == "0.0.0.0:9099"
