# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HeartbeatMonitor."""

from __future__ import annotations

import asyncio

import pytest

from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.registry.auth import (
    AuthIdentity,
    AuthValidator,
)
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.heartbeat_monitor import HeartbeatMonitor
from dynamo.planner.plugins.registry.server import PluginRegistryServer
from dynamo.planner.plugins.transport.base import PluginTransport
from dynamo.planner.plugins.types import RegisterRequest

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
        self.closed = False

    async def call(self, method, request):
        return None

    async def close(self):
        self.closed = True


class _AcceptAllAuth(AuthValidator):
    async def validate(self, token):
        return AuthIdentity(source="static_secret", subject="test")


def _make_ctx():
    clock = VirtualClock()
    cb = CircuitBreaker(clock)

    def factory(plugin_id, endpoint, *, in_process_instance=None):
        return _StubTransport(plugin_id, endpoint, in_process_instance=in_process_instance)

    server = PluginRegistryServer(
        clock=clock,
        auth=_AcceptAllAuth(),
        circuit_breaker=cb,
        transport_factory=factory,
    )
    return server, clock


async def _register_uds(server, plugin_id="p1", endpoint=None):
    resp = await server.register(
        RegisterRequest(
            plugin_id=plugin_id,
            plugin_type="propose",
            endpoint=endpoint or f"grpc://127.0.0.1:9000",
            auth_token="",
            protocol_version="1.0",
        )
    )
    assert resp.accepted, f"register rejected: {resp.reject_reason}"
    return resp


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evicts_uds_plugin_after_deadline_without_heartbeat():
    server, clock = _make_ctx()
    monitor = HeartbeatMonitor(server, clock, timeout_seconds=15.0, missed_threshold=2)
    await _register_uds(server)
    # last_heartbeat_at starts at -inf, so ANY advance past 0 will exceed deadline;
    # also advance some time so it's realistic.
    clock.advance(monitor.eviction_deadline_seconds + 1.0)
    await monitor._check_once()
    assert server.get_plugin("p1") is None


@pytest.mark.asyncio
async def test_regular_heartbeats_keep_plugin_alive():
    server, clock = _make_ctx()
    monitor = HeartbeatMonitor(
        server, clock, timeout_seconds=10.0, missed_threshold=2, check_interval_seconds=5.0
    )
    await _register_uds(server)
    # Send heartbeat every 5s for 60s; eviction deadline is 20s.
    for _ in range(12):
        clock.advance(5.0)
        await server.heartbeat("p1")
        await monitor._check_once()
    assert server.get_plugin("p1") is not None


@pytest.mark.asyncio
async def test_missed_threshold_requires_exceeding_full_window():
    server, clock = _make_ctx()
    monitor = HeartbeatMonitor(
        server, clock, timeout_seconds=10.0, missed_threshold=2
    )
    await _register_uds(server)
    # Immediately hand in a heartbeat so last_heartbeat_at is concrete.
    await server.heartbeat("p1")
    # Advance up to but not past 20s deadline.
    clock.advance(19.0)
    await monitor._check_once()
    assert server.get_plugin("p1") is not None
    # Advance past deadline.
    clock.advance(2.0)
    await monitor._check_once()
    assert server.get_plugin("p1") is None


@pytest.mark.asyncio
async def test_late_heartbeat_resets_deadline():
    server, clock = _make_ctx()
    monitor = HeartbeatMonitor(server, clock, timeout_seconds=10.0, missed_threshold=2)
    await _register_uds(server)
    await server.heartbeat("p1")
    clock.advance(15.0)  # not yet at 20s deadline
    await server.heartbeat("p1")  # resets last_heartbeat_at
    clock.advance(15.0)  # now at 30s total but only 15s since last heartbeat
    await monitor._check_once()
    assert server.get_plugin("p1") is not None


# ---------------------------------------------------------------------------
# In-process skip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_builtin_inprocess_plugin_never_evicted():
    server, clock = _make_ctx()
    monitor = HeartbeatMonitor(server, clock, timeout_seconds=10.0, missed_threshold=2)
    server.register_internal(
        plugin_id="builtin",
        plugin_type="propose",
        priority=1,
        instance=object(),
        is_builtin=True,
    )
    clock.advance(1_000_000.0)  # arbitrarily long — should still be alive
    await monitor._check_once()
    assert server.get_plugin("builtin") is not None


@pytest.mark.asyncio
async def test_user_inprocess_plugin_never_evicted():
    # G-3 v11 regression: is_builtin=False + transport_type=in_process must still skip.
    server, clock = _make_ctx()
    monitor = HeartbeatMonitor(server, clock, timeout_seconds=10.0, missed_threshold=2)
    server.register_internal(
        plugin_id="user_inproc",
        plugin_type="predict",
        priority=1,
        instance=object(),
        is_builtin=False,
    )
    clock.advance(1_000_000.0)
    await monitor._check_once()
    assert server.get_plugin("user_inproc") is not None


# ---------------------------------------------------------------------------
# Eviction integration: goes through unregister path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_eviction_fires_on_unregister_callbacks_with_reason():
    server, clock = _make_ctx()
    monitor = HeartbeatMonitor(server, clock, timeout_seconds=10.0, missed_threshold=1)
    events: list[tuple[str, str]] = []
    server.on_unregister(lambda pid, reason: events.append((pid, reason)))
    await _register_uds(server)
    clock.advance(11.0)
    await monitor._check_once()
    assert events == [("p1", "heartbeat_missed")]
    assert server.get_plugin("p1") is None


# ---------------------------------------------------------------------------
# run() loop integration — one round trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_loop_sleeps_and_stops_cleanly():
    server, clock = _make_ctx()
    monitor = HeartbeatMonitor(
        server,
        clock,
        timeout_seconds=10.0,
        missed_threshold=1,
        check_interval_seconds=5.0,
    )
    await _register_uds(server)
    task = asyncio.create_task(monitor.run())
    # Yield so the task reaches its first clock.sleep.
    for _ in range(5):
        await asyncio.sleep(0)
    # Advance past deadline AND past one check interval. The sleep future
    # wakes, monitor re-checks, plugin is evicted.
    clock.advance(11.0)
    for _ in range(10):
        await asyncio.sleep(0)
    assert server.get_plugin("p1") is None
    monitor.stop()
    clock.advance(5.0)  # wake any pending sleep so run() exits
    for _ in range(5):
        await asyncio.sleep(0)
    await asyncio.wait_for(task, timeout=1.0)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_invalid_config_rejected():
    server, clock = _make_ctx()
    with pytest.raises(ValueError):
        HeartbeatMonitor(server, clock, timeout_seconds=0)
    with pytest.raises(ValueError):
        HeartbeatMonitor(server, clock, missed_threshold=0)
    with pytest.raises(ValueError):
        HeartbeatMonitor(server, clock, check_interval_seconds=0)
