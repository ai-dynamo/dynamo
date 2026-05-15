# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for list_plugins end-to-end with scheduler cache_age wiring.

Basic filter tests live in test_server.py; this file focuses on the
observability fields (``circuit_state``, ``cache_age_seconds``,
``last_call_at_seconds_ago``) that require the scheduler + circuit
breaker to be attached.
"""

from __future__ import annotations

import pytest

from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.registry.auth import AllowUnauthenticatedAuth
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.server import PluginRegistryServer
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


async def _register(server, plugin_id, plugin_type="propose", priority=10,
                    execution_interval_seconds=10.0,
                    hold_policy=HoldPolicy.HOLD_LAST):
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


@pytest.mark.asyncio
async def test_cache_age_seconds_reports_scheduler_cache_age():
    server, scheduler, _, clock = _make_ctx()
    await _register(server, "p1")
    scheduler.compute_active_set(0.0, "propose")
    scheduler.record_result("p1", "propose", _ovr(5), 0.0)
    clock.advance(4.0)
    out = server.list_plugins(ListPluginsRequest())
    (info,) = out
    assert info.cache_age_seconds == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_circuit_state_field_reflects_breaker():
    server, scheduler, cb, clock = _make_ctx()
    await _register(server, "p1")
    (info,) = server.list_plugins(ListPluginsRequest())
    assert info.circuit_state == CircuitState.CLOSED

    for _ in range(3):
        cb.record_failure("p1")
    (info,) = server.list_plugins(ListPluginsRequest())
    assert info.circuit_state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_last_call_at_seconds_ago_reports_staleness():
    server, scheduler, _, clock = _make_ctx()
    await _register(server, "p1")
    (info_never,) = server.list_plugins(ListPluginsRequest())
    assert info_never.last_call_at_seconds_ago == 0.0

    scheduler.compute_active_set(0.0, "propose")
    scheduler.record_result("p1", "propose", _ovr(5), 0.0)
    clock.advance(7.0)
    (info_called,) = server.list_plugins(ListPluginsRequest())
    assert info_called.last_call_at_seconds_ago == pytest.approx(7.0)


@pytest.mark.asyncio
async def test_evaluations_total_increments_on_record_result():
    server, scheduler, _, _ = _make_ctx()
    await _register(server, "p1")
    scheduler.compute_active_set(0.0, "propose")
    scheduler.record_result("p1", "propose", _ovr(5), 0.0)
    scheduler.record_result("p1", "propose", _ovr(6), 1.0)
    (info,) = server.list_plugins(ListPluginsRequest())
    assert info.evaluations_total == 2


@pytest.mark.asyncio
async def test_transport_label_matches_transport_type():
    server, _, _, _ = _make_ctx()
    await _register(server, "p_grpc")
    server.register_internal(
        plugin_id="p_inproc",
        plugin_type="propose",
        priority=1,
        instance=object(),
    )
    out = {p.plugin_id: p.transport for p in server.list_plugins(ListPluginsRequest())}
    assert out == {"p_grpc": "grpc", "p_inproc": "in_process"}


@pytest.mark.asyncio
async def test_is_builtin_propagates_through_list_plugins():
    server, _, _, _ = _make_ctx()
    await _register(server, "user_uds")
    server.register_internal(
        plugin_id="builtin_inproc",
        plugin_type="propose",
        priority=1,
        instance=object(),
        is_builtin=True,
    )
    out = {p.plugin_id: p.is_builtin for p in server.list_plugins(ListPluginsRequest())}
    assert out == {"user_uds": False, "builtin_inproc": True}
