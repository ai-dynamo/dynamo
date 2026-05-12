# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration test for plugin invocation metric emissions.

Drives a real ``LocalPlannerOrchestrator`` through a tick with stub
plugins and asserts the plugin invocation metrics land with the
expected labels and counts.
"""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from dynamo.planner.monitoring.planner_metrics import (
    CIRCUIT_STATE_CLOSED,
    PluginFrameworkMetrics,
)
from dynamo.planner.plugins.merge.types import ComponentKey
from dynamo.planner.plugins.types import (
    AcceptResult,
    ComponentTarget,
    HoldPolicy,
    OverrideResult,
    OverrideType,
    PipelineContext,
    ProposeStageResponse,
    ReconcileStageResponse,
    ConstrainStageResponse,
    RejectResult,
)

from .conftest import StubPlugin

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


PREFILL = ComponentKey(sub_component_type="prefill", component_name="worker")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def metrics():
    return PluginFrameworkMetrics(registry=CollectorRegistry())


def _register_stub(ctx, *, plugin_id, plugin_type, priority, instance):
    ctx["registry"].register_internal(
        plugin_id=plugin_id,
        plugin_type=plugin_type,
        priority=priority,
        instance=instance,
        execution_interval_seconds=0.0,
        hold_policy=HoldPolicy.ACCEPT_WHEN_IDLE,
        is_builtin=True,
    )


async def _drive_with_metrics(ctx_factory, metrics, *, stubs):
    """Build orchestrator with metrics injected, register stubs, tick."""
    ctx = ctx_factory()
    ctx["orchestrator"]._metrics = metrics  # override None default
    for s in stubs:
        _register_stub(ctx, **s)
    outcome = await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    return ctx, outcome


def _gauge_value(metric, **labels):
    return metric.labels(**labels)._value.get()


def _counter_value(metric, **labels):
    return metric.labels(**labels)._value.get()


# ---------------------------------------------------------------------------
# plugin_evaluations_total + plugin_latency_seconds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accept_plugin_increments_eval_counter_with_accept_label(
    ctx_factory, metrics
):
    accept_stub = StubPlugin(
        propose=lambda req: ProposeStageResponse(
            result_kind="accept", accept=AcceptResult()
        ),
    )
    await _drive_with_metrics(
        ctx_factory,
        metrics,
        stubs=[
            dict(
                plugin_id="acceptor",
                plugin_type="propose",
                priority=1,
                instance=accept_stub,
            )
        ],
    )
    assert (
        _counter_value(
            metrics.plugin_evaluations_total,
            plugin_id="acceptor",
            stage="propose",
            result="accept",
        )
        == 1
    )


@pytest.mark.asyncio
async def test_set_override_emits_set_result_label_and_override_gauge(
    ctx_factory, metrics
):
    set_stub = StubPlugin(
        propose=lambda req: ProposeStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        component_name="worker",
                        replicas=5,
                        type=OverrideType.SET,
                    )
                ],
            ),
        ),
    )
    await _drive_with_metrics(
        ctx_factory,
        metrics,
        stubs=[
            dict(
                plugin_id="setter",
                plugin_type="propose",
                priority=1,
                instance=set_stub,
            )
        ],
    )

    # eval counter
    assert (
        _counter_value(
            metrics.plugin_evaluations_total,
            plugin_id="setter",
            stage="propose",
            result="set",
        )
        == 1
    )
    # override gauge: SET=1, other types=0
    assert (
        _gauge_value(
            metrics.plugin_override_active,
            plugin_id="setter",
            stage="propose",
            override_type="SET",
        )
        == 1
    )
    assert (
        _gauge_value(
            metrics.plugin_override_active,
            plugin_id="setter",
            stage="propose",
            override_type="AT_LEAST",
        )
        == 0
    )


@pytest.mark.asyncio
async def test_latency_histogram_records_count_for_successful_call(
    ctx_factory, metrics
):
    """Successful call → latency observation with matching (plugin_id, stage)
    labels. The exact bucket split isn't asserted (too brittle); the count
    is.
    """
    stub = StubPlugin(
        propose=lambda req: ProposeStageResponse(
            result_kind="accept", accept=AcceptResult()
        ),
    )
    await _drive_with_metrics(
        ctx_factory,
        metrics,
        stubs=[
            dict(
                plugin_id="timed",
                plugin_type="propose",
                priority=1,
                instance=stub,
            )
        ],
    )
    samples = list(metrics.plugin_latency_seconds.collect())[0].samples
    counts = [
        s.value
        for s in samples
        if s.name.endswith("_count")
        and s.labels.get("plugin_id") == "timed"
        and s.labels.get("stage") == "propose"
    ]
    assert counts and counts[0] == 1.0


# ---------------------------------------------------------------------------
# plugin_circuit_state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_state_gauge_is_closed_after_successful_tick(
    ctx_factory, metrics
):
    stub = StubPlugin(
        propose=lambda req: ProposeStageResponse(
            result_kind="accept", accept=AcceptResult()
        ),
    )
    await _drive_with_metrics(
        ctx_factory,
        metrics,
        stubs=[
            dict(
                plugin_id="stable",
                plugin_type="propose",
                priority=1,
                instance=stub,
            )
        ],
    )
    assert (
        _gauge_value(metrics.plugin_circuit_state, plugin_id="stable")
        == CIRCUIT_STATE_CLOSED
    )


# ---------------------------------------------------------------------------
# metrics=None path: existing tests must keep working (regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_metrics_registered_when_metrics_is_none(ctx_factory):
    """With ``metrics=None`` the pipeline must run identically to before
    8-2 — no emission side effects, no exceptions."""
    stub = StubPlugin(
        propose=lambda req: ProposeStageResponse(
            result_kind="accept", accept=AcceptResult()
        ),
    )
    ctx = ctx_factory()
    assert ctx["orchestrator"]._metrics is None  # default unchanged
    _register_stub(
        ctx,
        plugin_id="no_metric",
        plugin_type="propose",
        priority=1,
        instance=stub,
    )
    outcome = await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    # Just care it didn't raise; a real assertion on the outcome shape
    # lives in test_pipeline.py.
    assert outcome is not None


# ---------------------------------------------------------------------------
# plugin_held_over_total + plugin_cache_age_seconds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_held_over_plugin_emits_held_over_counter(ctx_factory, metrics):
    """A plugin with ``execution_interval_seconds > 0`` + ``HOLD_LAST``
    hold_policy gets its first-tick result cached; on the second tick
    the scheduler replays the cached result and we should see
    ``plugin_held_over_total`` increment."""
    stub = StubPlugin(
        propose=lambda req: ProposeStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        component_name="worker",
                        replicas=4,
                        type=OverrideType.SET,
                    )
                ],
            ),
        ),
    )
    ctx = ctx_factory()
    ctx["orchestrator"]._metrics = metrics
    ctx["registry"].register_internal(
        plugin_id="cached",
        plugin_type="propose",
        priority=1,
        instance=stub,
        execution_interval_seconds=60.0,  # not due again for 60s
        hold_policy=HoldPolicy.HOLD_LAST,
        is_builtin=True,
    )

    # Tick 1: plugin evaluates, result cached.
    await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    # VirtualClock advances 1s (much less than 60s interval)
    ctx["clock"].advance(1.0)
    # Tick 2: plugin not due, cached result inherited → held_over.
    await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})

    held = _counter_value(
        metrics.plugin_held_over_total, plugin_id="cached", stage="propose"
    )
    assert held == 1
    cache_age = _gauge_value(metrics.plugin_cache_age_seconds, plugin_id="cached")
    # Cache was stored on tick 1, read ~1s later.  Exact value depends on
    # VirtualClock steps; tolerate >=0 and <= advance amount.
    assert 0 <= cache_age <= 2.0


# ---------------------------------------------------------------------------
# Family-3 metrics (pipeline integration)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_clamp_emits_reconcile_clamped_total(
    ctx_factory, metrics
):
    """Two plugins at RECONCILE: one sets replicas=10, the other
    says AT_MOST=4. Merge clamps to 4; we expect
    ``reconcile_clamped_total{source='cap'}`` to increment once.
    """
    set_stub = StubPlugin(
        reconcile=lambda req: ReconcileStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        component_name="worker",
                        replicas=10,
                        type=OverrideType.SET,
                    )
                ]
            ),
        ),
    )
    cap_stub = StubPlugin(
        reconcile=lambda req: ReconcileStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        component_name="worker",
                        replicas=4,
                        type=OverrideType.AT_MOST,
                    )
                ]
            ),
        ),
    )
    ctx, outcome = await _drive_with_metrics(
        ctx_factory,
        metrics,
        stubs=[
            dict(
                plugin_id="setter",
                plugin_type="reconcile",
                priority=1,
                instance=set_stub,
            ),
            dict(
                plugin_id="cap",
                plugin_type="reconcile",
                priority=2,
                instance=cap_stub,
            ),
        ],
    )
    assert outcome is not None
    # Exactly one clamp event emitted, sourced to the cap plugin.
    v = _counter_value(
        metrics.reconcile_clamped_total,
        sub_component_type="prefill",
        component_name="worker",
        source="cap",
    )
    assert v == 1
    # constrain counter untouched this tick.
    all_samples = list(metrics.constrain_capped_total.collect())[0].samples
    # Counter with no inc()s has no samples other than the _created.
    assert not any(
        s.name.endswith("_total") and s.value > 0 for s in all_samples
    )


@pytest.mark.asyncio
async def test_constrain_clamp_emits_constrain_capped_total(ctx_factory, metrics):
    """A CONSTRAIN-stage AT_MOST lowering the baseline is tracked as
    a ``constrain_capped_total`` event (not reconcile_clamped_total)."""
    # Baseline prefill=10; budget plugin caps to 4.
    cap_stub = StubPlugin(
        constrain=lambda req: ConstrainStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        component_name="worker",
                        replicas=4,
                        type=OverrideType.AT_MOST,
                    )
                ]
            ),
        ),
    )
    ctx = ctx_factory()
    ctx["orchestrator"]._metrics = metrics
    _register_stub(
        ctx,
        plugin_id="budget",
        plugin_type="constrain",
        priority=1,
        instance=cap_stub,
    )
    # Use a baseline that's above the AT_MOST so the cap actually fires.
    outcome = await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 10})
    assert outcome is not None
    v = _counter_value(
        metrics.constrain_capped_total,
        sub_component_type="prefill",
        component_name="worker",
        source="budget",
    )
    assert v == 1


@pytest.mark.asyncio
async def test_reject_emits_reject_short_circuited_total(ctx_factory, metrics):
    reject_stub = StubPlugin(
        propose=lambda req: ProposeStageResponse(
            result_kind="reject", reject=RejectResult(reason="nope")
        ),
    )
    ctx, outcome = await _drive_with_metrics(
        ctx_factory,
        metrics,
        stubs=[
            dict(
                plugin_id="safety",
                plugin_type="propose",
                priority=1,
                instance=reject_stub,
            )
        ],
    )
    assert outcome.execute_action == "skip_short_circuit"
    v = _counter_value(metrics.reject_short_circuited_total, plugin_id="safety")
    assert v == 1


# ---------------------------------------------------------------------------
# Family-6 pipeline integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_duration_seconds_observed_per_tick(ctx_factory, metrics):
    stub = StubPlugin(
        propose=lambda req: ProposeStageResponse(
            result_kind="accept", accept=AcceptResult()
        ),
    )
    await _drive_with_metrics(
        ctx_factory,
        metrics,
        stubs=[
            dict(
                plugin_id="p",
                plugin_type="propose",
                priority=1,
                instance=stub,
            )
        ],
    )
    # One tick → one observation
    samples = list(metrics.tick_duration_seconds.collect())[0].samples
    counts = [s.value for s in samples if s.name.endswith("_count")]
    assert counts and counts[0] == 1.0


@pytest.mark.asyncio
async def test_tick_skipped_total_fires_when_plugin_not_due(ctx_factory, metrics):
    """A plugin with execution_interval_seconds > 0 that produces an
    OverrideResult gets its ``last_call_at`` bumped by ``record_result``.
    On the next tick (before the interval elapses) it is deferred and
    we expect ``tick_skipped_total`` to increment.

    Note: only OverrideResult triggers ``record_result`` (HOLD_LAST
    caching semantics). Accept-only plugins stay at -inf forever so
    they're always "due" — out-of-scope for this test.
    """
    stub = StubPlugin(
        propose=lambda req: ProposeStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        component_name="worker",
                        replicas=4,
                        type=OverrideType.SET,
                    )
                ]
            ),
        ),
    )
    ctx = ctx_factory()
    ctx["orchestrator"]._metrics = metrics
    # Also wire the scheduler's metrics since tick_skipped_total is
    # emitted from there (the orchestrator owns a single metrics
    # instance shared across layers).
    ctx["scheduler"]._metrics = metrics
    ctx["registry"].register_internal(
        plugin_id="cadenced",
        plugin_type="propose",
        priority=1,
        instance=stub,
        execution_interval_seconds=60.0,  # not due again for 60s
        hold_policy=HoldPolicy.ACCEPT_WHEN_IDLE,
        is_builtin=True,
    )

    # Tick 1: first-ever call, is_due=True → triggered, no skip
    await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    assert (
        _counter_value(metrics.tick_skipped_total, plugin_id="cadenced") == 0
    )

    # Tick 2: advance 1s only (way short of 60s interval) → not due → skipped
    ctx["clock"].advance(1.0)
    await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    assert (
        _counter_value(metrics.tick_skipped_total, plugin_id="cadenced") == 1
    )


@pytest.mark.asyncio
async def test_tick_lag_seconds_set_when_plugin_evaluated(ctx_factory, metrics):
    """tick_lag_seconds is set per evaluation; first tick has lag=0
    (no prior due_at), subsequent ticks after the interval has
    elapsed show positive lag proportional to delay past schedule.

    Same caveat as tick_skipped_total: only OverrideResult bumps
    ``last_call_at``, so the stub must return one for subsequent ticks
    to have a meaningful due_at.
    """
    stub = StubPlugin(
        propose=lambda req: ProposeStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        component_name="worker",
                        replicas=4,
                        type=OverrideType.SET,
                    )
                ]
            ),
        ),
    )
    ctx = ctx_factory()
    ctx["orchestrator"]._metrics = metrics
    ctx["scheduler"]._metrics = metrics
    ctx["registry"].register_internal(
        plugin_id="timed",
        plugin_type="propose",
        priority=1,
        instance=stub,
        execution_interval_seconds=5.0,  # due every 5s
        hold_policy=HoldPolicy.ACCEPT_WHEN_IDLE,
        is_builtin=True,
    )

    # Tick 1: first-ever call → lag=0
    await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    assert _gauge_value(metrics.tick_lag_seconds, plugin_id="timed") == 0.0

    # Advance 7s → due was at 5s, we're 2s late
    ctx["clock"].advance(7.0)
    await ctx["orchestrator"].tick(PipelineContext(), {PREFILL: 3})
    lag = _gauge_value(metrics.tick_lag_seconds, plugin_id="timed")
    assert lag == pytest.approx(2.0, abs=0.1)


@pytest.mark.asyncio
async def test_no_clamp_when_recommendation_within_bounds(ctx_factory, metrics):
    """Regression guard: bounds that don't change the final value MUST
    NOT emit the counter — otherwise dashboards would over-count."""
    set_stub = StubPlugin(
        reconcile=lambda req: ReconcileStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        component_name="worker",
                        replicas=5,
                        type=OverrideType.SET,
                    )
                ]
            ),
        ),
    )
    ceiling_stub = StubPlugin(
        reconcile=lambda req: ReconcileStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=[
                    ComponentTarget(
                        sub_component_type="prefill",
                        component_name="worker",
                        replicas=8,  # larger than SET=5, no clamp
                        type=OverrideType.AT_MOST,
                    )
                ]
            ),
        ),
    )
    await _drive_with_metrics(
        ctx_factory,
        metrics,
        stubs=[
            dict(
                plugin_id="setter",
                plugin_type="reconcile",
                priority=1,
                instance=set_stub,
            ),
            dict(
                plugin_id="loose_ceiling",
                plugin_type="reconcile",
                priority=2,
                instance=ceiling_stub,
            ),
        ],
    )
    # No clamp event emitted.
    samples = list(metrics.reconcile_clamped_total.collect())[0].samples
    assert not any(
        s.name.endswith("_total") and s.value > 0 for s in samples
    )
