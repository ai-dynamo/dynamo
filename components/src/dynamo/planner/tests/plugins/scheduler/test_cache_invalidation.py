# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""6-row cache-invalidation must-pass tests.

Each of the 6 rows in the cache invalidation table gets its own dedicated
test. These are MUST-PASS — any future change touching the scheduler
must keep all six green.
"""

from __future__ import annotations

import pytest

from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.registry.auth import AllowUnauthenticatedAuth
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.heartbeat_monitor import HeartbeatMonitor
from dynamo.planner.plugins.registry.server import PluginRegistryServer
from dynamo.planner.plugins.scheduler import PluginScheduler
from dynamo.planner.plugins.transport.base import PluginTransport
from dynamo.planner.plugins.types import (
    ComponentTarget,
    HoldPolicy,
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


class _StubTransport(PluginTransport):
    def __init__(self, plugin_id, endpoint, *, in_process_instance=None):
        self.plugin_id = plugin_id
        self.endpoint = endpoint
        self.timeout_seconds = 1.0

    async def call(self, method, request):
        return None

    async def close(self):
        pass


def _make_ctx():
    clock = VirtualClock()
    cb = CircuitBreaker(clock, failure_threshold=3, cooldown_seconds=30.0)

    def factory(plugin_id, endpoint, *, in_process_instance=None):
        return _StubTransport(plugin_id, endpoint)

    server = PluginRegistryServer(
        clock=clock, auth=AllowUnauthenticatedAuth(),
        circuit_breaker=cb, transport_factory=factory,
    )
    scheduler = PluginScheduler(server, cb, clock)
    return server, scheduler, cb, clock


async def _register_hold_last(server, plugin_id="p1"):
    resp = await server.register(RegisterRequest(
        plugin_id=plugin_id,
        plugin_type="propose",
        priority=10,
        endpoint=f"grpc://127.0.0.1:9000",
        protocol_version="1.0",
        execution_interval_seconds=10.0,
        hold_policy=HoldPolicy.HOLD_LAST,
    ))
    assert resp.accepted, resp.reject_reason


def _ovr(replicas):
    return OverrideResult(targets=[
        ComponentTarget(sub_component_type="prefill", replicas=replicas, type=OverrideType.SET)
    ])


async def _seed_cache(server, scheduler, plugin_id="p1"):
    """Register + tick + record → scheduler holds a cache entry for plugin_id."""
    await _register_hold_last(server, plugin_id=plugin_id)
    scheduler.compute_active_set(0.0, "propose")
    scheduler.record_result(plugin_id, "propose", _ovr(5), 0.0)
    assert scheduler.cache_entries_count() >= 1


# ---------------------------------------------------------------------------
# Row 1 — explicit Unregister
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_row_1_explicit_unregister_clears_cache():
    server, scheduler, _, _ = _make_ctx()
    await _seed_cache(server, scheduler)
    await server.unregister("p1", reason="client_shutdown")
    assert scheduler.cache_entries_count() == 0


# ---------------------------------------------------------------------------
# Row 2 — heartbeat missed eviction (auto-unregister path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_row_2_heartbeat_missed_eviction_clears_cache():
    server, scheduler, _, clock = _make_ctx()
    monitor = HeartbeatMonitor(server, clock, timeout_seconds=10.0, missed_threshold=1)
    await _seed_cache(server, scheduler)
    clock.advance(11.0)
    await monitor._check_once()
    assert server.get_plugin("p1") is None
    assert scheduler.cache_entries_count() == 0


# ---------------------------------------------------------------------------
# Row 3 — circuit breaker OPEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_row_3_circuit_open_clears_cache():
    server, scheduler, cb, _ = _make_ctx()
    await _seed_cache(server, scheduler)
    for _ in range(3):  # failure_threshold=3
        cb.record_failure("p1")
    assert scheduler.cache_entries_count() == 0


# ---------------------------------------------------------------------------
# Row 4 — client-driven version upgrade (Unregister + Register)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_row_4_client_driven_version_upgrade_path_is_fresh_cache():
    server, scheduler, _, _ = _make_ctx()
    await _seed_cache(server, scheduler)
    # Version upgrade: client Unregister then Register new version.
    await server.unregister("p1", reason="version_upgrade")
    assert scheduler.cache_entries_count() == 0  # cleared by the unregister step
    # Fresh Register does NOT inherit the old cache; entry stays at 0 until
    # a new record_result lands.
    await _register_hold_last(server, plugin_id="p1")
    assert scheduler.cache_entries_count() == 0
    scheduler.compute_active_set(0.0, "propose")
    scheduler.record_result("p1", "propose", _ovr(99), 0.0)
    assert scheduler.cache_entries_count() == 1


# ---------------------------------------------------------------------------
# Row 5 — explicit config-reload full clear
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_row_5_config_reload_clears_every_cache():
    server, scheduler, _, _ = _make_ctx()
    await _seed_cache(server, scheduler, plugin_id="p1")
    await _seed_cache(server, scheduler, plugin_id="p2")
    assert scheduler.cache_entries_count() == 2
    scheduler.invalidate_cache(reason="config_reload")
    assert scheduler.cache_entries_count() == 0


# ---------------------------------------------------------------------------
# Row 6 — orchestrator restart (in-memory only; new scheduler is empty)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_row_6_restart_equivalent_fresh_scheduler_has_no_cache():
    # Simulating restart: construct a new Scheduler bound to the same
    # registry + circuit_breaker. The old scheduler's cache dies with it.
    server, old_scheduler, cb, clock = _make_ctx()
    await _seed_cache(server, old_scheduler)
    assert old_scheduler.cache_entries_count() == 1

    # "Restart" — discard the old scheduler, spin up a new one.
    del old_scheduler
    new_scheduler = PluginScheduler(server, cb, clock)
    assert new_scheduler.cache_entries_count() == 0
