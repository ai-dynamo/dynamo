# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PluginRegistryServer."""

from __future__ import annotations

from typing import Any

import pytest

from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.registry.auth import (
    AuthIdentity,
    AuthValidator,
    StaticSecretAuth,
)
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.errors import AuthError
from dynamo.planner.plugins.registry.server import PluginRegistryServer
from dynamo.planner.plugins.transport.base import PluginTransport
from dynamo.planner.plugins.types import HoldPolicy, ListPluginsRequest, RegisterRequest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubTransport(PluginTransport):
    """Records lifecycle; never hits the network."""

    def __init__(self, plugin_id, endpoint, *, in_process_instance=None):
        self.plugin_id = plugin_id
        self.endpoint = endpoint
        self.timeout_seconds = 1.0
        self.instance = in_process_instance
        self.closed = False
        self.calls: list[tuple[str, Any]] = []

    async def call(self, method, request):
        self.calls.append((method, request))
        return None

    async def close(self):
        self.closed = True


def _stub_factory():
    """Returns (factory, created) where ``created`` is populated with the
    transports built through the factory, so tests can assert on them."""
    created: list[_StubTransport] = []

    def factory(plugin_id, endpoint, *, in_process_instance=None):
        t = _StubTransport(plugin_id, endpoint, in_process_instance=in_process_instance)
        created.append(t)
        return t

    return factory, created


class _AcceptAllAuth(AuthValidator):
    async def validate(self, token):
        return AuthIdentity(source="static_secret", subject="test")


def _make_server(auth=None, protocol_versions=("1.0", "1.0")):
    clock = VirtualClock()
    cb = CircuitBreaker(clock)
    factory, created = _stub_factory()
    server = PluginRegistryServer(
        clock=clock,
        auth=auth or _AcceptAllAuth(),
        circuit_breaker=cb,
        transport_factory=factory,
        protocol_versions=protocol_versions,
    )
    return server, clock, cb, created


def _req(
    plugin_id="p1",
    plugin_type="propose",
    endpoint="grpc://127.0.0.1:9000",
    auth_token="",
    protocol_version="1.0",
    **kwargs,
):
    return RegisterRequest(
        plugin_id=plugin_id,
        plugin_type=plugin_type,
        endpoint=endpoint,
        auth_token=auth_token,
        protocol_version=protocol_version,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_happy_path_creates_plugin_and_transport():
    server, _, _, created = _make_server()
    resp = await server.register(_req(priority=10))
    assert resp.accepted is True
    assert resp.negotiated_protocol_version == "1.0"
    assert resp.reject_reason == ""

    plugin = server.get_plugin("p1")
    assert plugin is not None
    assert plugin.plugin_type == "propose"
    assert plugin.priority == 10
    assert plugin.transport_type == "grpc"
    assert plugin.endpoint == "grpc://127.0.0.1:9000"
    assert plugin.is_builtin is False
    assert len(created) == 1


@pytest.mark.asyncio
async def test_heartbeat_updates_timestamp_and_returns_true():
    server, clock, _, _ = _make_server()
    await server.register(_req())
    clock.advance(3.0)
    ok = await server.heartbeat("p1")
    assert ok is True
    assert server.get_plugin("p1").last_heartbeat_at == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_heartbeat_for_unknown_plugin_returns_false():
    server, _, _, _ = _make_server()
    ok = await server.heartbeat("ghost")
    assert ok is False


class _PerTokenAuth(AuthValidator):
    """Maps token → subject so subject-mismatch can be exercised.

    Token ``"bad"`` raises AuthError to cover the auth-failed branch.
    """

    async def validate(self, token):
        if token == "bad":
            raise AuthError("invalid")
        return AuthIdentity(source="static_secret", subject=f"subj-{token}")


@pytest.mark.asyncio
async def test_authenticated_heartbeat_matching_subject_ok():
    server, clock, _, _ = _make_server(auth=_PerTokenAuth())
    await server.register(_req(auth_token="A"))
    assert server.get_plugin("p1").auth_subject == "subj-A"
    clock.advance(3.0)
    ok, reject = await server.authenticated_heartbeat("p1", "A")
    assert (ok, reject) == (True, None)
    assert server.get_plugin("p1").last_heartbeat_at == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_authenticated_heartbeat_invalid_token_returns_auth_failed():
    server, _, _, _ = _make_server(auth=_PerTokenAuth())
    await server.register(_req(auth_token="A"))
    ok, reject = await server.authenticated_heartbeat("p1", "bad")
    assert (ok, reject) == (False, "auth_failed")


@pytest.mark.asyncio
async def test_authenticated_heartbeat_subject_mismatch_returns_permission_denied():
    server, _, _, _ = _make_server(auth=_PerTokenAuth())
    await server.register(_req(auth_token="A"))
    # token "B" validates but maps to a different subject — caller cannot
    # manage plugins they did not register.
    ok, reject = await server.authenticated_heartbeat("p1", "B")
    assert (ok, reject) == (False, "permission_denied")
    # Plugin must NOT be touched on rejection.
    assert server.get_plugin("p1").last_heartbeat_at == -float("inf")


@pytest.mark.asyncio
async def test_authenticated_heartbeat_unknown_plugin_returns_permission_denied():
    """Unknown plugin_id collapses to ``permission_denied`` — same response
    as a wrong-subject probe — so a token-holder cannot enumerate registered
    plugin_ids by observing distinct return codes."""
    server, _, _, _ = _make_server(auth=_PerTokenAuth())
    ok, reject = await server.authenticated_heartbeat("ghost", "A")
    assert (ok, reject) == (False, "permission_denied")


@pytest.mark.asyncio
async def test_authenticated_heartbeat_in_process_plugin_returns_permission_denied():
    """``register_internal`` plugins have ``auth_subject == ""`` by design —
    they are not reachable via the gateway.  Heartbeat against one must
    collapse to ``permission_denied`` (NOT silently succeed because
    ``"" == ""`` if some buggy auth backend ever returned an empty subject)."""
    server, _, _, _ = _make_server(auth=_PerTokenAuth())
    server.register_internal(
        plugin_id="builtin",
        plugin_type="propose",
        priority=10,
        instance=object(),
    )
    assert server.get_plugin("builtin").auth_subject == ""
    ok, reject = await server.authenticated_heartbeat("builtin", "A")
    assert (ok, reject) == (False, "permission_denied")


@pytest.mark.asyncio
async def test_authenticated_unregister_matching_subject_removes_plugin():
    server, _, _, created = _make_server(auth=_PerTokenAuth())
    await server.register(_req(auth_token="A"))
    ok, reject = await server.authenticated_unregister("p1", "A", reason="shutdown")
    assert (ok, reject) == (True, None)
    assert server.get_plugin("p1") is None
    assert created[0].closed is True


@pytest.mark.asyncio
async def test_authenticated_unregister_invalid_token_returns_auth_failed():
    server, _, _, _ = _make_server(auth=_PerTokenAuth())
    await server.register(_req(auth_token="A"))
    ok, reject = await server.authenticated_unregister("p1", "bad")
    assert (ok, reject) == (False, "auth_failed")
    # Plugin must remain registered when auth fails.
    assert server.get_plugin("p1") is not None


@pytest.mark.asyncio
async def test_authenticated_unregister_subject_mismatch_does_not_evict():
    """Forged Unregister with a *valid* token whose subject doesn't match
    the registered plugin's subject MUST NOT evict the plugin. This is the
    core security guarantee of the gateway auth model."""
    server, _, _, _ = _make_server(auth=_PerTokenAuth())
    await server.register(_req(auth_token="A"))
    ok, reject = await server.authenticated_unregister("p1", "B")
    assert (ok, reject) == (False, "permission_denied")
    assert server.get_plugin("p1") is not None  # NOT evicted


@pytest.mark.asyncio
async def test_authenticated_unregister_unknown_plugin_returns_permission_denied():
    """Same existence-oracle hardening as the heartbeat case — unknown
    plugin and wrong subject return the same code."""
    server, _, _, _ = _make_server(auth=_PerTokenAuth())
    ok, reject = await server.authenticated_unregister("ghost", "A")
    assert (ok, reject) == (False, "permission_denied")


@pytest.mark.asyncio
async def test_authenticated_unregister_in_process_plugin_not_reachable_via_gateway():
    """``register_internal`` plugins are not removable via the gateway —
    in-process invariant enforced explicitly so the data shape isn't the
    only line of defense."""
    server, _, _, _ = _make_server(auth=_PerTokenAuth())
    server.register_internal(
        plugin_id="builtin",
        plugin_type="propose",
        priority=10,
        instance=object(),
    )
    ok, reject = await server.authenticated_unregister("builtin", "A")
    assert (ok, reject) == (False, "permission_denied")
    assert server.get_plugin("builtin") is not None  # NOT evicted


@pytest.mark.asyncio
async def test_unregister_removes_plugin_and_closes_transport():
    server, _, _, created = _make_server()
    await server.register(_req())
    ok = await server.unregister("p1", reason="client_shutdown")
    assert ok is True
    assert server.get_plugin("p1") is None
    assert created[0].closed is True


@pytest.mark.asyncio
async def test_unregister_unknown_plugin_idempotent_false():
    server, _, _, _ = _make_server()
    ok = await server.unregister("ghost", reason="")
    assert ok is False


@pytest.mark.asyncio
async def test_unregister_fans_out_to_subscribers():
    server, _, _, _ = _make_server()
    events: list[tuple[str, str]] = []
    server.on_unregister(lambda pid, reason: events.append((pid, reason)))
    await server.register(_req())
    await server.unregister("p1", reason="heartbeat_missed")
    assert events == [("p1", "heartbeat_missed")]


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_plugin_id_rejected_no_upsert():
    server, _, _, created = _make_server()
    first = await server.register(_req(priority=1))
    second = await server.register(_req(priority=99))  # same plugin_id
    assert first.accepted is True
    assert second.accepted is False
    assert "duplicate_plugin_id" in second.reject_reason
    # Original plugin priority unchanged (no upsert).
    assert server.get_plugin("p1").priority == 1
    assert len(created) == 1  # second call did NOT build a second transport


@pytest.mark.asyncio
async def test_protocol_version_out_of_range_rejected():
    server, _, _, _ = _make_server(protocol_versions=("1.0", "1.0"))
    resp = await server.register(_req(protocol_version="0.9"))
    assert resp.accepted is False
    assert "protocol_version_unsupported" in resp.reject_reason

    resp2 = await server.register(_req(plugin_id="p2", protocol_version="1.1"))
    assert resp2.accepted is False
    assert "protocol_version_unsupported" in resp2.reject_reason


@pytest.mark.asyncio
async def test_protocol_version_semantic_compare_not_lexicographic():
    """``1.10`` must be greater than ``1.2`` (semantic version),
    even though lexicographically ``"1.10" < "1.2"`` because the
    second character ``1`` < ``2``. The register check must use
    ``packaging.version.Version`` (or equivalent) — not raw string
    compare — otherwise valid plugins get rejected once any component
    reaches 10."""
    server, _, _, _ = _make_server(protocol_versions=("1.0", "1.10"))

    # 1.10 must be accepted under max=1.10 (semantic compare).
    # With lexicographic compare, "1.10" > "1.2" would be False and
    # the upper-bound check ``req <= max`` would still pass (because
    # "1.10" <= "1.10" lexicographically too); the failure mode is
    # different. The real lex bug: requested="1.2", max="1.10" —
    # lex says "1.2" > "1.10" so the register rejects something
    # that should be accepted. Cover both that and the "above max"
    # rejection that semantic compare must respect.
    accepted = await server.register(_req(plugin_id="ok-1.10", protocol_version="1.10"))
    assert accepted.accepted is True, accepted.reject_reason

    accepted_mid = await server.register(
        _req(plugin_id="ok-1.2", protocol_version="1.2")
    )
    assert accepted_mid.accepted is True, accepted_mid.reject_reason

    rejected = await server.register(_req(plugin_id="too-new", protocol_version="2.0"))
    assert rejected.accepted is False
    assert "protocol_version_unsupported" in rejected.reject_reason


@pytest.mark.asyncio
async def test_protocol_version_malformed_rejected_clearly():
    """A non-semver protocol_version string surfaces as a distinct
    ``protocol_version_malformed`` reject reason — not silently treated
    as out-of-range."""
    server, _, _, _ = _make_server(protocol_versions=("1.0", "2.0"))
    resp = await server.register(_req(protocol_version="not-a-version"))
    assert resp.accepted is False
    assert "protocol_version_malformed" in resp.reject_reason


def _make_server_with_scale_interval(scale_interval_seconds: float):
    clock = VirtualClock()
    cb = CircuitBreaker(clock)
    factory, _ = _stub_factory()
    return PluginRegistryServer(
        clock=clock,
        auth=_AcceptAllAuth(),
        circuit_breaker=cb,
        transport_factory=factory,
        scale_interval_seconds=scale_interval_seconds,
    )


@pytest.mark.asyncio
async def test_observation_window_zero_accepted():
    """Default ``observation_window_seconds=0.0`` means
    "per-tick freshness" — always accepted."""
    server = _make_server_with_scale_interval(5.0)
    resp = await server.register(_req(observation_window_seconds=0.0))
    assert resp.accepted is True, resp.reject_reason


@pytest.mark.asyncio
async def test_observation_window_multiple_of_scale_interval_accepted():
    """``N * scale_interval`` aligns to tick boundaries — accepted."""
    server = _make_server_with_scale_interval(5.0)
    resp = await server.register(_req(plugin_id="p1", observation_window_seconds=5.0))
    assert resp.accepted is True, resp.reject_reason
    resp2 = await server.register(_req(plugin_id="p2", observation_window_seconds=15.0))
    assert resp2.accepted is True, resp2.reject_reason


@pytest.mark.asyncio
async def test_observation_window_non_multiple_rejected():
    """Non-multiple windows drive Prometheus queries that cross tick
    boundaries — reject with a clear reason."""
    server = _make_server_with_scale_interval(5.0)
    resp = await server.register(_req(observation_window_seconds=7.0))
    assert resp.accepted is False
    assert "observation_window_misaligned" in resp.reject_reason


@pytest.mark.asyncio
async def test_observation_window_negative_rejected():
    server = _make_server_with_scale_interval(5.0)
    resp = await server.register(_req(observation_window_seconds=-1.0))
    assert resp.accepted is False
    assert "observation_window_misaligned" in resp.reject_reason


@pytest.mark.asyncio
async def test_observation_window_unverifiable_without_scale_interval():
    """When ``scale_interval_seconds == 0.0`` (older test fixtures construct the
    server without one), the alignment constraint can't be verified —
    accept any value rather than reject erroneously."""
    server = _make_server_with_scale_interval(0.0)
    resp = await server.register(_req(observation_window_seconds=7.0))
    assert resp.accepted is True, resp.reject_reason


@pytest.mark.asyncio
async def test_auth_failure_rejected_with_generic_reason():
    server, _, _, _ = _make_server(auth=StaticSecretAuth({"good": "alice"}))
    resp = await server.register(_req(auth_token="bad"))
    assert resp.accepted is False
    # Generic reason — no leak of specific failure mode.
    assert resp.reject_reason == "auth_failed"


@pytest.mark.asyncio
async def test_auth_success_accepts():
    server, _, _, _ = _make_server(auth=StaticSecretAuth({"good": "alice"}))
    resp = await server.register(_req(auth_token="good"))
    assert resp.accepted is True


@pytest.mark.asyncio
async def test_inproc_endpoint_over_rpc_rejected():
    # Clients MUST NOT use inproc:// endpoints via the network RPC —
    # that's what register_internal is for.
    server, _, _, _ = _make_server()
    resp = await server.register(_req(endpoint="inproc://sneaky"))
    assert resp.accepted is False
    assert "inproc" in resp.reject_reason


@pytest.mark.asyncio
async def test_unknown_endpoint_scheme_rejected():
    server, _, _, _ = _make_server()
    resp = await server.register(_req(endpoint="http://bad"))
    assert resp.accepted is False
    assert "transport_build_failed" in resp.reject_reason


# ---------------------------------------------------------------------------
# register_internal
# ---------------------------------------------------------------------------


def test_register_internal_skips_auth_and_wraps_inproc():
    server, _, _, created = _make_server()

    class Echo:
        async def Propose(self, req):
            return req

    plugin = server.register_internal(
        plugin_id="builtin_echo",
        plugin_type="propose",
        priority=5,
        instance=Echo(),
        execution_interval_seconds=10.0,
        hold_policy=HoldPolicy.HOLD_LAST,
    )
    assert plugin.transport_type == "in_process"
    assert plugin.endpoint == "inproc://builtin_echo"
    assert plugin.is_builtin is True
    assert len(created) == 1
    assert created[0].instance is not None  # factory received the instance


def test_register_internal_duplicate_raises():
    server, _, _, _ = _make_server()
    server.register_internal("p1", "propose", 1, object())
    with pytest.raises(ValueError, match="already registered"):
        server.register_internal("p1", "propose", 1, object())


def test_register_internal_rejects_misaligned_observation_window():
    server = _make_server_with_scale_interval(5.0)

    with pytest.raises(ValueError, match="observation_window_misaligned"):
        server.register_internal(
            plugin_id="p1",
            plugin_type="propose",
            priority=1,
            instance=object(),
            observation_window_seconds=7.0,
        )


def test_register_internal_can_mark_user_inprocess_plugin():
    server, _, _, _ = _make_server()
    plugin = server.register_internal(
        plugin_id="user_inproc",
        plugin_type="predict",
        priority=1,
        instance=object(),
        is_builtin=False,
    )
    assert plugin.is_builtin is False
    # G-3: transport_type=in_process even when is_builtin=False.
    assert plugin.transport_type == "in_process"


# ---------------------------------------------------------------------------
# list_plugins
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_plugins_filters_and_reports_fields():
    server, _, _, _ = _make_server()
    await server.register(_req(plugin_id="p1", plugin_type="propose"))
    await server.register(
        _req(plugin_id="p2", plugin_type="predict", endpoint="grpc://127.0.0.1:9000")
    )
    # no filter
    out = server.list_plugins(ListPluginsRequest())
    assert {p.plugin_id for p in out} == {"p1", "p2"}
    # stage filter
    out_propose = server.list_plugins(ListPluginsRequest(stage_filter="propose"))
    assert {p.plugin_id for p in out_propose} == {"p1"}
    # disabled filter
    server.get_plugin("p2").enabled = False
    out_default = server.list_plugins(ListPluginsRequest())
    assert {p.plugin_id for p in out_default} == {"p1"}
    out_all = server.list_plugins(ListPluginsRequest(include_disabled=True))
    assert {p.plugin_id for p in out_all} == {"p1", "p2"}


# ---------------------------------------------------------------------------
# CircuitBreaker reset on register/unregister
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_resets_circuit_breaker_for_plugin_id():
    server, _, cb, _ = _make_server()
    # Manually seed some failures under plugin_id
    cb.record_failure("p1")
    cb.record_failure("p1")
    cb.record_failure("p1")
    cb.record_failure("p1")
    cb.record_failure("p1")  # threshold default 5 -> OPEN
    await server.register(_req())
    # After register the breaker state should be fresh.
    from dynamo.planner.plugins.types import CircuitState

    assert cb.state("p1") == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_unregister_resets_circuit_breaker():
    server, _, cb, _ = _make_server()
    await server.register(_req())
    for _ in range(5):
        cb.record_failure("p1")
    await server.unregister("p1")
    from dynamo.planner.plugins.types import CircuitState

    assert cb.state("p1") == CircuitState.CLOSED
