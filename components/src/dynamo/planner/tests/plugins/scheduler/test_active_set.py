# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PluginScheduler.compute_active_set."""

from __future__ import annotations

import pytest

from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.registry.auth import AllowUnauthenticatedAuth
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
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


async def _register(server, plugin_id, plugin_type, priority,
                    execution_interval_seconds=0.0,
                    hold_policy=HoldPolicy.ACCEPT_WHEN_IDLE):
    resp = await server.register(RegisterRequest(
        plugin_id=plugin_id,
        plugin_type=plugin_type,
        priority=priority,
        endpoint=f"grpc://127.0.0.1:9000",
        protocol_version="1.0",
        execution_interval_seconds=execution_interval_seconds,
        hold_policy=hold_policy,
    ))
    assert resp.accepted, resp.reject_reason


def _ovr(replicas):
    return OverrideResult(targets=[
        ComponentTarget(sub_component_type="prefill", replicas=replicas, type=OverrideType.SET)
    ])


# ---------------------------------------------------------------------------
# Basic triggering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_tick_triggers_even_with_positive_interval():
    server, scheduler, _, clock = _make_ctx()
    await _register(server, "p1", "propose", 10, execution_interval_seconds=10.0)
    active = scheduler.compute_active_set(clock.monotonic(), "propose")
    assert [p.plugin_id for p in active.triggered] == ["p1"]
    assert active.inherited == []


@pytest.mark.asyncio
async def test_zero_interval_triggers_every_tick():
    server, scheduler, _, clock = _make_ctx()
    await _register(server, "p1", "propose", 10, execution_interval_seconds=0.0)
    active = scheduler.compute_active_set(clock.monotonic(), "propose")
    assert [p.plugin_id for p in active.triggered] == ["p1"]
    scheduler.record_result("p1", "propose", _ovr(5), clock.monotonic())
    clock.advance(0.001)
    active = scheduler.compute_active_set(clock.monotonic(), "propose")
    assert [p.plugin_id for p in active.triggered] == ["p1"]


@pytest.mark.asyncio
async def test_not_triggered_inside_interval_window():
    server, scheduler, _, clock = _make_ctx()
    await _register(server, "p1", "propose", 10, execution_interval_seconds=10.0)
    active = scheduler.compute_active_set(clock.monotonic(), "propose")
    scheduler.record_result("p1", "propose", _ovr(5), clock.monotonic())
    clock.advance(5.0)
    active = scheduler.compute_active_set(clock.monotonic(), "propose")
    assert active.triggered == []
    assert active.inherited == []  # ACCEPT_WHEN_IDLE -> skip, no inherited


@pytest.mark.asyncio
async def test_triggered_again_after_interval_elapses():
    server, scheduler, _, clock = _make_ctx()
    await _register(server, "p1", "propose", 10, execution_interval_seconds=10.0)
    active = scheduler.compute_active_set(clock.monotonic(), "propose")
    scheduler.record_result("p1", "propose", _ovr(5), clock.monotonic())
    clock.advance(10.0)
    active = scheduler.compute_active_set(clock.monotonic(), "propose")
    assert [p.plugin_id for p in active.triggered] == ["p1"]


@pytest.mark.asyncio
async def test_hold_last_inherits_between_triggers():
    server, scheduler, _, clock = _make_ctx()
    await _register(server, "p1", "propose", 10,
                    execution_interval_seconds=10.0,
                    hold_policy=HoldPolicy.HOLD_LAST)
    # First tick triggers.
    scheduler.compute_active_set(clock.monotonic(), "propose")
    scheduler.record_result("p1", "propose", _ovr(7), clock.monotonic())
    clock.advance(5.0)
    active = scheduler.compute_active_set(clock.monotonic(), "propose")
    assert active.triggered == []
    assert len(active.inherited) == 1
    assert active.inherited[0].plugin_id == "p1"
    assert active.inherited[0].priority == 10
    assert active.inherited[0].result.targets[0].replicas == 7


# ---------------------------------------------------------------------------
# Filtering: stage / enabled / circuit breaker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage_filter_only_returns_matching_plugin_type():
    server, scheduler, _, clock = _make_ctx()
    await _register(server, "p1", "propose", 10)
    await _register(server, "p2", "predict", 20)
    active_propose = scheduler.compute_active_set(clock.monotonic(), "propose")
    assert [p.plugin_id for p in active_propose.triggered] == ["p1"]
    active_predict = scheduler.compute_active_set(clock.monotonic(), "predict")
    assert [p.plugin_id for p in active_predict.triggered] == ["p2"]


@pytest.mark.asyncio
async def test_disabled_plugin_excluded_from_active_set():
    server, scheduler, _, clock = _make_ctx()
    await _register(server, "p1", "propose", 10)
    server.get_plugin("p1").enabled = False
    active = scheduler.compute_active_set(clock.monotonic(), "propose")
    assert active.triggered == []
    assert active.inherited == []


@pytest.mark.asyncio
async def test_circuit_open_excludes_plugin_from_active_set():
    server, scheduler, cb, clock = _make_ctx()
    await _register(server, "p1", "propose", 10,
                    execution_interval_seconds=10.0,
                    hold_policy=HoldPolicy.HOLD_LAST)
    # Seed the cache so inherited would otherwise be possible.
    scheduler.compute_active_set(clock.monotonic(), "propose")
    scheduler.record_result("p1", "propose", _ovr(5), clock.monotonic())
    # Open the circuit.
    for _ in range(3):
        cb.record_failure("p1")
    clock.advance(5.0)
    active = scheduler.compute_active_set(clock.monotonic(), "propose")
    assert active.triggered == []
    assert active.inherited == []  # OPEN skips even HOLD_LAST
