# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative-dependency gating tests — ``compute_active_set(ctx=...)``.

The scale_interval cadence model adds a second gate to ``is_due``:
``RegisterRequest.requires_produced_fields`` lists dot-paths into
``PipelineContext`` that must be non-None at fire time.  This lets a
plugin declare "I only run when upstream stage produced predictions"
without writing the gate check inside the plugin itself.

When the gate fails, scheduler increments
``tick_requires_unsatisfied_total`` and skips the plugin (no cache
inherit — the plugin chose to opt into strict dependency, so a stale
cached result would violate its own declared contract).
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
    HoldPolicy,
    ObservationData,
    PipelineContext,
    PredictionData,
    RegisterRequest,
    TrafficMetrics,
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
        clock=clock,
        auth=AllowUnauthenticatedAuth(),
        circuit_breaker=cb,
        transport_factory=factory,
    )
    scheduler = PluginScheduler(server, cb, clock)
    return server, scheduler, clock


async def _register(
    server,
    plugin_id,
    plugin_type="propose",
    requires=None,
):
    resp = await server.register(
        RegisterRequest(
            plugin_id=plugin_id,
            plugin_type=plugin_type,
            priority=10,
            endpoint="grpc://127.0.0.1:9000",
            protocol_version="1.0",
            execution_interval_seconds=0.0,  # every tick
            hold_policy=HoldPolicy.ACCEPT_WHEN_IDLE,
            requires_produced_fields=list(requires or []),
        )
    )
    assert resp.accepted, resp.reject_reason


# ---------------------------------------------------------------------------
# _ctx_get dot-path walker
# ---------------------------------------------------------------------------


def test_ctx_get_top_level_attribute():
    ctx = PipelineContext(
        observations=ObservationData(
            traffic=TrafficMetrics(duration_s=5, num_req=1, isl=1, osl=1),
        ),
    )
    assert PluginScheduler._ctx_get(ctx, "observations") is not None


def test_ctx_get_nested_dot_path():
    ctx = PipelineContext(
        observations=ObservationData(
            traffic=TrafficMetrics(duration_s=5, num_req=1, isl=1, osl=1),
        ),
    )
    traffic = PluginScheduler._ctx_get(ctx, "observations.traffic")
    assert traffic is not None
    assert PluginScheduler._ctx_get(ctx, "observations.traffic.num_req") == 1


def test_ctx_get_returns_none_for_missing_intermediate():
    ctx = PipelineContext(observations=None)
    assert PluginScheduler._ctx_get(ctx, "observations.traffic") is None


def test_ctx_get_returns_none_for_missing_attribute():
    ctx = PipelineContext()
    assert PluginScheduler._ctx_get(ctx, "predictions.predicted_num_req") is None


def test_ctx_get_returns_none_when_ctx_is_none():
    assert PluginScheduler._ctx_get(None, "anything") is None


# ---------------------------------------------------------------------------
# Requires gating in compute_active_set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plugin_without_requires_fires_normally():
    """Plugin declaring no requires fires regardless of ctx state.
    Backward compat: existing plugins (no requires field set) are
    unaffected by the new gate."""
    server, scheduler, clock = _make_ctx()
    await _register(server, "p1")

    active = scheduler.compute_active_set(clock.monotonic(), "propose", ctx=None)
    assert [p.plugin_id for p in active.triggered] == ["p1"]


@pytest.mark.asyncio
async def test_plugin_with_satisfied_requires_fires():
    server, scheduler, clock = _make_ctx()
    await _register(server, "p1", requires=["predictions"])
    ctx = PipelineContext(
        predictions=PredictionData(
            predicted_num_req=42.0, predicted_isl=10, predicted_osl=20
        ),
    )

    active = scheduler.compute_active_set(clock.monotonic(), "propose", ctx=ctx)
    assert [p.plugin_id for p in active.triggered] == ["p1"]


@pytest.mark.asyncio
async def test_plugin_with_unsatisfied_requires_skipped():
    server, scheduler, clock = _make_ctx()
    await _register(server, "p1", requires=["predictions"])
    ctx = PipelineContext(predictions=None)

    active = scheduler.compute_active_set(clock.monotonic(), "propose", ctx=ctx)
    assert active.triggered == []
    assert active.inherited == []  # no inherit for requires-gated skip


@pytest.mark.asyncio
async def test_plugin_with_nested_requires_path():
    server, scheduler, clock = _make_ctx()
    await _register(server, "p1", requires=["observations.traffic"])
    ctx_missing = PipelineContext(observations=None)
    ctx_present = PipelineContext(
        observations=ObservationData(
            traffic=TrafficMetrics(duration_s=5, num_req=1, isl=1, osl=1),
        ),
    )

    assert (
        scheduler.compute_active_set(
            clock.monotonic(), "propose", ctx=ctx_missing
        ).triggered
        == []
    )
    assert [
        p.plugin_id
        for p in scheduler.compute_active_set(
            clock.monotonic(), "propose", ctx=ctx_present
        ).triggered
    ] == ["p1"]


@pytest.mark.asyncio
async def test_multiple_requires_all_must_be_present():
    server, scheduler, clock = _make_ctx()
    await _register(server, "p1", requires=["predictions", "observations.traffic"])

    # Only predictions present — traffic missing → skip
    ctx_partial = PipelineContext(
        predictions=PredictionData(
            predicted_num_req=42, predicted_isl=10, predicted_osl=20
        ),
        observations=None,
    )
    assert (
        scheduler.compute_active_set(
            clock.monotonic(), "propose", ctx=ctx_partial
        ).triggered
        == []
    )

    # Both present → fire
    ctx_both = PipelineContext(
        predictions=PredictionData(
            predicted_num_req=42, predicted_isl=10, predicted_osl=20
        ),
        observations=ObservationData(
            traffic=TrafficMetrics(duration_s=5, num_req=1, isl=1, osl=1),
        ),
    )
    assert [
        p.plugin_id
        for p in scheduler.compute_active_set(
            clock.monotonic(), "propose", ctx=ctx_both
        ).triggered
    ] == ["p1"]


@pytest.mark.asyncio
async def test_requires_unsatisfied_when_ctx_is_none_conservative_skip():
    """If caller didn't supply ctx and plugin declares requires,
    conservative default = skip (don't fire a plugin without the data
    it asked for).  This is the contract for callers that haven't
    threaded ctx through yet (e.g. test fixtures).  Plugins without
    requires still fire when ctx is None (see
    ``test_plugin_without_requires_fires_normally``)."""
    server, scheduler, clock = _make_ctx()
    await _register(server, "p1", requires=["predictions"])

    active = scheduler.compute_active_set(clock.monotonic(), "propose", ctx=None)
    assert active.triggered == []


# ---------------------------------------------------------------------------
# Metric emission
# ---------------------------------------------------------------------------


def _stub_metrics():
    """Minimal stand-in for PluginFrameworkMetrics — just records the
    counter inc calls."""

    class _BoundCounter:
        def __init__(self, parent, labels_kw):
            self._parent = parent
            self._labels_kw = labels_kw

        def inc(self):
            self._parent.calls.append(self._labels_kw)

    class _Counter:
        def __init__(self):
            self.calls = []

        def labels(self, **kw):
            return _BoundCounter(self, kw)

    class _BoundGauge:
        def set(self, _v):
            pass

    class _Gauge:
        def labels(self, **kw):
            return _BoundGauge()

    class M:
        tick_skipped_total = _Counter()
        tick_requires_unsatisfied_total = _Counter()
        tick_lag_seconds = _Gauge()

    return M()


@pytest.mark.asyncio
async def test_tick_requires_unsatisfied_metric_emits_with_missing_field():
    server, scheduler, clock = _make_ctx()
    scheduler._metrics = _stub_metrics()
    await _register(server, "p1", requires=["predictions", "observations.traffic"])

    # Both missing — should record the FIRST missing field, not both.
    scheduler.compute_active_set(
        clock.monotonic(),
        "propose",
        ctx=PipelineContext(predictions=None, observations=None),
    )
    calls = scheduler._metrics.tick_requires_unsatisfied_total.calls
    assert calls == [{"plugin_id": "p1", "missing_field": "predictions"}]

    # Now predictions present, traffic missing → records "observations.traffic"
    scheduler._metrics.tick_requires_unsatisfied_total.calls.clear()
    scheduler.compute_active_set(
        clock.monotonic(),
        "propose",
        ctx=PipelineContext(
            predictions=PredictionData(
                predicted_num_req=42, predicted_isl=10, predicted_osl=20
            ),
            observations=None,
        ),
    )
    calls = scheduler._metrics.tick_requires_unsatisfied_total.calls
    assert calls == [{"plugin_id": "p1", "missing_field": "observations.traffic"}]
