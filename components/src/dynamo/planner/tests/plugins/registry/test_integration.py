# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end registry integration test.

Exercises the full registry stack wired together — config → auth →
registry → circuit breaker → scheduler → heartbeat monitor — through
realistic lifecycle scenarios:

1. Register happy path: config-driven factory + full metadata in
   ``list_plugins``.
2. Tick + record_result + HOLD_LAST inheritance across multiple ticks.
3. Unregister → cache invalidation (row 1 of the table).
4. Circuit breaker OPEN → cache invalidation + plugin drop from
   active set (row 3), then HALF_OPEN → recovery.
5. Heartbeat missed eviction → full cleanup path (v11 row 2).
6. Client-driven version upgrade: unregister + re-register → fresh state
   (v11 row 4 + Q6).

Uses in-memory stubs for transport + VirtualClock for determinism. A
real UDS / gRPC socket round-trip test belongs in a ``tests/integration``
e2e suite; this file keeps coverage at the component-wiring level so it
runs fast under the ``pre_merge`` CI marker.
"""

from __future__ import annotations

import pytest

from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.registry.config import (
    AuthConfig,
    PluginRegistrationConfig,
    build_registry_from_config,
)
from dynamo.planner.plugins.registry.heartbeat_monitor import HeartbeatMonitor
from dynamo.planner.plugins.scheduler import PluginScheduler
from dynamo.planner.plugins.transport.base import PluginTransport
from dynamo.planner.plugins.types import (
    CircuitState,
    ComponentTarget,
    HoldPolicy,
    ListPluginsRequest,
    OverrideResult,
    OverrideType,
    RegisterRequest,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# The real transport factory (make_transport_for_endpoint) opens sockets;
# we monkeypatch at the module level so builds succeed without touching
# the filesystem / network. The orchestrator e2e suite wires these
# end-to-end with real UDS.
@pytest.fixture
def stub_transport(monkeypatch):
    class _Stub(PluginTransport):
        def __init__(self, plugin_id, endpoint, *, in_process_instance=None, **_):
            self.plugin_id = plugin_id
            self.endpoint = endpoint
            self.timeout_seconds = 1.0
            self.closed = False

        async def call(self, method, request):
            return None

        async def close(self):
            self.closed = True

    def _factory(plugin_id, endpoint, config, *, in_process_instance=None):
        return _Stub(plugin_id, endpoint)

    monkeypatch.setattr(
        "dynamo.planner.plugins.registry.config.make_transport_for_endpoint",
        _factory,
    )


def _ovr(replicas):
    return OverrideResult(
        targets=[
            ComponentTarget(
                sub_component_type="prefill",
                replicas=replicas,
                type=OverrideType.SET,
            )
        ]
    )


def _config(trusted_sources=("static_secret",), static_secrets=None):
    return PluginRegistrationConfig(
        auth=AuthConfig(
            trusted_sources=list(trusted_sources),
            static_secrets=dict(static_secrets or {"secret-a": "alice"}),
        ),
    )


def _assemble(clock):
    server, cb = build_registry_from_config(_config(), clock)
    scheduler = PluginScheduler(server, cb, clock)
    monitor = HeartbeatMonitor(
        server, clock, timeout_seconds=10.0, missed_threshold=2, check_interval_seconds=5.0
    )
    return server, scheduler, cb, monitor


@pytest.mark.asyncio
async def test_full_lifecycle_register_tick_unregister(stub_transport):
    clock = VirtualClock()
    server, scheduler, _, _ = _assemble(clock)

    # 1. Register
    resp = await server.register(
        RegisterRequest(
            plugin_id="load-scaler",
            plugin_type="propose",
            priority=10,
            endpoint="grpc://127.0.0.1:9000",
            auth_token="secret-a",
            protocol_version="1.0",
            execution_interval_seconds=10.0,
            hold_policy=HoldPolicy.HOLD_LAST,
            version="v1",
        )
    )
    assert resp.accepted

    # ListPlugins reports a complete picture.
    (info,) = server.list_plugins(ListPluginsRequest())
    assert info.plugin_id == "load-scaler"
    assert info.transport == "grpc"
    assert info.circuit_state == CircuitState.CLOSED
    assert info.is_builtin is False

    # 2. First tick: plugin is triggered.
    active = scheduler.compute_active_set(clock.monotonic(), "propose")
    assert [p.plugin_id for p in active.triggered] == ["load-scaler"]
    scheduler.record_result("load-scaler", "propose", _ovr(4), clock.monotonic())

    # 3. Mid-interval tick: not triggered, but HOLD_LAST inherits.
    clock.advance(5.0)
    active = scheduler.compute_active_set(clock.monotonic(), "propose")
    assert active.triggered == []
    (inherited,) = active.inherited
    assert inherited.result.targets[0].replicas == 4

    # 4. Second interval: triggers again.
    clock.advance(5.0)
    active = scheduler.compute_active_set(clock.monotonic(), "propose")
    assert [p.plugin_id for p in active.triggered] == ["load-scaler"]
    scheduler.record_result("load-scaler", "propose", _ovr(6), clock.monotonic())
    assert scheduler.cache_entries_count() == 1

    # 5. Unregister: cache drops.
    ok = await server.unregister("load-scaler", reason="client_shutdown")
    assert ok
    assert scheduler.cache_entries_count() == 0
    assert server.list_plugins(ListPluginsRequest()) == []


@pytest.mark.asyncio
async def test_circuit_open_removes_plugin_then_half_open_recovers(stub_transport):
    clock = VirtualClock()
    server, scheduler, cb, _ = _assemble(clock)
    cb._failure_threshold = 3  # tighter threshold for this test
    cb._cooldown = 10.0

    await server.register(
        RegisterRequest(
            plugin_id="p",
            plugin_type="propose",
            priority=1,
            endpoint="grpc://127.0.0.1:9000",
            auth_token="secret-a",
            protocol_version="1.0",
            execution_interval_seconds=10.0,
            hold_policy=HoldPolicy.HOLD_LAST,
        )
    )
    scheduler.compute_active_set(clock.monotonic(), "propose")
    scheduler.record_result("p", "propose", _ovr(3), clock.monotonic())

    # Three failures → OPEN. Cache cleared by on_open fan-out.
    for _ in range(3):
        cb.record_failure("p")
    assert cb.state("p") == CircuitState.OPEN
    assert scheduler.cache_entries_count() == 0
    # Plugin drops out of active set (cannot call + cache empty).
    active = scheduler.compute_active_set(clock.monotonic(), "propose")
    assert active.triggered == []
    assert active.inherited == []

    # After cooldown → HALF_OPEN, plugin re-admitted.
    clock.advance(10.0)
    assert cb.state("p") == CircuitState.HALF_OPEN
    active = scheduler.compute_active_set(clock.monotonic(), "propose")
    assert [x.plugin_id for x in active.triggered] == ["p"]
    # One success → CLOSED.
    cb.record_success("p")
    assert cb.state("p") == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_heartbeat_missed_eviction_full_path(stub_transport):
    clock = VirtualClock()
    server, scheduler, _, monitor = _assemble(clock)

    await server.register(
        RegisterRequest(
            plugin_id="flaky",
            plugin_type="propose",
            priority=1,
            endpoint="grpc://127.0.0.1:9000",
            auth_token="secret-a",
            protocol_version="1.0",
            execution_interval_seconds=5.0,
            hold_policy=HoldPolicy.HOLD_LAST,
        )
    )
    scheduler.compute_active_set(clock.monotonic(), "propose")
    scheduler.record_result("flaky", "propose", _ovr(9), clock.monotonic())
    assert scheduler.cache_entries_count() == 1

    # Advance past eviction deadline (timeout*threshold = 10*2 = 20s).
    clock.advance(21.0)
    await monitor._check_once()

    assert server.get_plugin("flaky") is None
    assert scheduler.cache_entries_count() == 0
    active = scheduler.compute_active_set(clock.monotonic(), "propose")
    assert active.triggered == []
    assert active.inherited == []


@pytest.mark.asyncio
async def test_client_driven_version_upgrade(stub_transport):
    clock = VirtualClock()
    server, scheduler, _, _ = _assemble(clock)

    # v1 registers, ticks, caches.
    await server.register(
        RegisterRequest(
            plugin_id="p",
            plugin_type="propose",
            priority=1,
            endpoint="grpc://127.0.0.1:9000",
            auth_token="secret-a",
            protocol_version="1.0",
            execution_interval_seconds=10.0,
            hold_policy=HoldPolicy.HOLD_LAST,
            version="v1",
        )
    )
    scheduler.compute_active_set(clock.monotonic(), "propose")
    scheduler.record_result("p", "propose", _ovr(3), clock.monotonic())

    # Attempt to re-register without unregistering → rejected (Q6).
    dup = await server.register(
        RegisterRequest(
            plugin_id="p",
            plugin_type="propose",
            priority=1,
            endpoint="grpc://127.0.0.1:9000",
            auth_token="secret-a",
            protocol_version="1.0",
            execution_interval_seconds=10.0,
            hold_policy=HoldPolicy.HOLD_LAST,
            version="v2",  # note: upgrade attempt
        )
    )
    assert dup.accepted is False
    assert "duplicate_plugin_id" in dup.reject_reason

    # Client-driven upgrade: Unregister then Register.
    await server.unregister("p", reason="version_upgrade")
    assert scheduler.cache_entries_count() == 0
    v2 = await server.register(
        RegisterRequest(
            plugin_id="p",
            plugin_type="propose",
            priority=1,
            endpoint="grpc://127.0.0.1:9000",
            auth_token="secret-a",
            protocol_version="1.0",
            execution_interval_seconds=10.0,
            hold_policy=HoldPolicy.HOLD_LAST,
            version="v2",
        )
    )
    assert v2.accepted
    assert server.get_plugin("p").version == "v2"
    assert scheduler.cache_entries_count() == 0  # fresh; needs new record_result


@pytest.mark.asyncio
async def test_in_process_user_plugin_survives_heartbeat_monitor(stub_transport):
    # Regression at integration level: in-process plugins must not be
    # evicted by the heartbeat monitor.
    clock = VirtualClock()
    server, _, _, monitor = _assemble(clock)
    server.register_internal(
        plugin_id="user_inproc",
        plugin_type="predict",
        priority=1,
        instance=object(),
        is_builtin=False,
    )
    clock.advance(10_000.0)
    await monitor._check_once()
    assert server.get_plugin("user_inproc") is not None
