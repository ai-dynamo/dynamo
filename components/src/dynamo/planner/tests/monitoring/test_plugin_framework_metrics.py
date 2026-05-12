# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PluginFrameworkMetrics."""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from dynamo.planner.monitoring.planner_metrics import (
    CIRCUIT_STATE_CLOSED,
    CIRCUIT_STATE_HALF_OPEN,
    CIRCUIT_STATE_OPEN,
    PluginFrameworkMetrics,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def metrics():
    """Fresh isolated registry per test so we can instantiate the
    metrics container repeatedly without ``Duplicated timeseries``."""
    return PluginFrameworkMetrics(registry=CollectorRegistry())


def _sample_value(metric, **labels):
    """Read a single sample value from a labelled metric — the
    `_sum`/`_count`/actual-value depending on metric type. Tests use
    ``.labels(...)._value.get()`` on counters/gauges; for histograms
    we read `_count` via iteration."""
    collected = list(metric.collect())[0]
    for s in collected.samples:
        if labels.items() <= s.labels.items() and s.name.endswith(
            ("_total", "_count", "_bucket")
        ):
            pass
    # Simpler: for Counter/Gauge, use the internal _value
    labelled = metric.labels(**labels)
    return labelled._value.get()


# ---------------------------------------------------------------------------
# plugin_evaluations_total
# ---------------------------------------------------------------------------


def test_plugin_evaluations_total_increments_per_call(metrics):
    metrics.plugin_evaluations_total.labels(
        plugin_id="p1", stage="propose", result="accept"
    ).inc()
    metrics.plugin_evaluations_total.labels(
        plugin_id="p1", stage="propose", result="accept"
    ).inc()
    metrics.plugin_evaluations_total.labels(
        plugin_id="p1", stage="propose", result="set"
    ).inc()
    metrics.plugin_evaluations_total.labels(
        plugin_id="p2", stage="constrain", result="at_most"
    ).inc()

    assert _sample_value(
        metrics.plugin_evaluations_total,
        plugin_id="p1",
        stage="propose",
        result="accept",
    ) == 2
    assert _sample_value(
        metrics.plugin_evaluations_total,
        plugin_id="p1",
        stage="propose",
        result="set",
    ) == 1
    assert _sample_value(
        metrics.plugin_evaluations_total,
        plugin_id="p2",
        stage="constrain",
        result="at_most",
    ) == 1


# ---------------------------------------------------------------------------
# plugin_latency_seconds
# ---------------------------------------------------------------------------


def test_plugin_latency_seconds_observes_values(metrics):
    h = metrics.plugin_latency_seconds.labels(plugin_id="p1", stage="propose")
    h.observe(0.003)
    h.observe(0.045)
    h.observe(0.8)

    # Histogram buckets are (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
    # 3 observations total.
    samples = {
        s.name: s.value
        for s in list(metrics.plugin_latency_seconds.collect())[0].samples
        if s.labels.get("plugin_id") == "p1" and s.labels.get("stage") == "propose"
    }
    # Total count across buckets = 3
    count_samples = [v for k, v in samples.items() if k.endswith("_count")]
    assert any(v == 3.0 for v in count_samples)


def test_plugin_latency_buckets_span_in_process_to_timeout(metrics):
    """Verify the bucket boundaries: the family should be useful both
    for in-process plugins (~1ms) and up to the default request_timeout
    (5s)."""
    # Reading buckets out of the collected samples is the cleanest way.
    h = metrics.plugin_latency_seconds.labels(plugin_id="any", stage="propose")
    h.observe(0.0)
    bucket_bounds = []
    for s in list(metrics.plugin_latency_seconds.collect())[0].samples:
        if s.name.endswith("_bucket") and s.labels.get("plugin_id") == "any":
            bucket_bounds.append(s.labels["le"])
    assert "0.001" in bucket_bounds
    assert "5.0" in bucket_bounds or "5" in bucket_bounds
    assert "+Inf" in bucket_bounds


# ---------------------------------------------------------------------------
# plugin_circuit_state
# ---------------------------------------------------------------------------


def test_plugin_circuit_state_encoding(metrics):
    metrics.plugin_circuit_state.labels(plugin_id="p1").set(CIRCUIT_STATE_CLOSED)
    metrics.plugin_circuit_state.labels(plugin_id="p2").set(CIRCUIT_STATE_HALF_OPEN)
    metrics.plugin_circuit_state.labels(plugin_id="p3").set(CIRCUIT_STATE_OPEN)

    assert _sample_value(metrics.plugin_circuit_state, plugin_id="p1") == 0.0
    assert _sample_value(metrics.plugin_circuit_state, plugin_id="p2") == 0.5
    assert _sample_value(metrics.plugin_circuit_state, plugin_id="p3") == 1.0


def test_circuit_state_constants_are_monotonic():
    """Encoding must be monotonic closed→half_open→open so dashboards
    can compute ``max_over_time()`` without inversions."""
    assert CIRCUIT_STATE_CLOSED < CIRCUIT_STATE_HALF_OPEN < CIRCUIT_STATE_OPEN


# ---------------------------------------------------------------------------
# plugin_held_over_total
# ---------------------------------------------------------------------------


def test_plugin_held_over_total_increments(metrics):
    metrics.plugin_held_over_total.labels(plugin_id="p1", stage="propose").inc()
    metrics.plugin_held_over_total.labels(plugin_id="p1", stage="propose").inc()
    metrics.plugin_held_over_total.labels(plugin_id="p2", stage="predict").inc()

    assert (
        _sample_value(metrics.plugin_held_over_total, plugin_id="p1", stage="propose")
        == 2
    )
    assert (
        _sample_value(metrics.plugin_held_over_total, plugin_id="p2", stage="predict")
        == 1
    )


# ---------------------------------------------------------------------------
# plugin_cache_age_seconds
# ---------------------------------------------------------------------------


def test_plugin_cache_age_seconds_set_per_plugin(metrics):
    metrics.plugin_cache_age_seconds.labels(plugin_id="p1").set(4.2)
    metrics.plugin_cache_age_seconds.labels(plugin_id="p2").set(60.0)
    assert _sample_value(metrics.plugin_cache_age_seconds, plugin_id="p1") == 4.2
    assert _sample_value(metrics.plugin_cache_age_seconds, plugin_id="p2") == 60.0


# ---------------------------------------------------------------------------
# plugin_override_active + reset_overrides
# ---------------------------------------------------------------------------


def test_plugin_override_active_per_label(metrics):
    metrics.plugin_override_active.labels(
        plugin_id="p1", stage="propose", override_type="SET"
    ).set(1)
    metrics.plugin_override_active.labels(
        plugin_id="p1", stage="propose", override_type="AT_LEAST"
    ).set(0)

    assert (
        _sample_value(
            metrics.plugin_override_active,
            plugin_id="p1",
            stage="propose",
            override_type="SET",
        )
        == 1
    )
    assert (
        _sample_value(
            metrics.plugin_override_active,
            plugin_id="p1",
            stage="propose",
            override_type="AT_LEAST",
        )
        == 0
    )


def test_reset_overrides_zeroes_all_types(metrics):
    # Set all four override types active for the same plugin/stage.
    for t in ("SET", "AT_LEAST", "AT_MOST", "REJECT"):
        metrics.plugin_override_active.labels(
            plugin_id="p1", stage="propose", override_type=t
        ).set(1)

    metrics.reset_overrides("p1", "propose")

    for t in ("SET", "AT_LEAST", "AT_MOST", "REJECT"):
        assert (
            _sample_value(
                metrics.plugin_override_active,
                plugin_id="p1",
                stage="propose",
                override_type=t,
            )
            == 0
        )


def test_reset_overrides_isolates_per_plugin_stage(metrics):
    """``reset_overrides`` zeros only the (plugin_id, stage) pair, not
    other plugins' overrides."""
    metrics.plugin_override_active.labels(
        plugin_id="p1", stage="propose", override_type="SET"
    ).set(1)
    metrics.plugin_override_active.labels(
        plugin_id="p2", stage="propose", override_type="SET"
    ).set(1)

    metrics.reset_overrides("p1", "propose")

    assert (
        _sample_value(
            metrics.plugin_override_active,
            plugin_id="p1",
            stage="propose",
            override_type="SET",
        )
        == 0
    )
    assert (
        _sample_value(
            metrics.plugin_override_active,
            plugin_id="p2",
            stage="propose",
            override_type="SET",
        )
        == 1
    )


# ---------------------------------------------------------------------------
# Registry isolation
# ---------------------------------------------------------------------------


def test_default_registry_construction_succeeds():
    """Without an explicit registry, metrics land on the global REGISTRY.
    The production path must be usable without changing anything.

    Note: ``OrchestratorEngineAdapter.__init__`` claims these metric
    names on the global REGISTRY at first construction. Other tests
    that construct the adapter therefore race with this one. We skip
    when names are already registered — the production path is the
    same either way.
    """
    from prometheus_client import REGISTRY

    try:
        m = PluginFrameworkMetrics()
    except ValueError:
        pytest.skip(
            "PluginFrameworkMetrics already registered on REGISTRY by "
            "another test in this session; test-only collision, not a bug."
        )
    try:
        m.plugin_evaluations_total.labels(
            plugin_id="x", stage="propose", result="accept"
        ).inc()
        value = _sample_value(
            m.plugin_evaluations_total,
            plugin_id="x",
            stage="propose",
            result="accept",
        )
        assert value == 1
    finally:
        for metric in (
            m.plugin_evaluations_total,
            m.plugin_latency_seconds,
            m.plugin_circuit_state,
            m.plugin_held_over_total,
            m.plugin_cache_age_seconds,
            m.plugin_override_active,
        ):
            try:
                REGISTRY.unregister(metric)
            except KeyError:
                pass


# ---------------------------------------------------------------------------
# Family-3 metrics (unit-level)
# ---------------------------------------------------------------------------


def test_reconcile_clamped_total_increments(metrics):
    metrics.reconcile_clamped_total.labels(
        sub_component_type="prefill",
        component_name="worker",
        source="budget_constrain",
    ).inc()
    metrics.reconcile_clamped_total.labels(
        sub_component_type="prefill",
        component_name="worker",
        source="budget_constrain",
    ).inc()
    metrics.reconcile_clamped_total.labels(
        sub_component_type="decode",
        component_name="",
        source="user_plugin",
    ).inc()
    assert (
        _sample_value(
            metrics.reconcile_clamped_total,
            sub_component_type="prefill",
            component_name="worker",
            source="budget_constrain",
        )
        == 2
    )
    assert (
        _sample_value(
            metrics.reconcile_clamped_total,
            sub_component_type="decode",
            component_name="",
            source="user_plugin",
        )
        == 1
    )


def test_constrain_capped_total_increments(metrics):
    metrics.constrain_capped_total.labels(
        sub_component_type="prefill",
        component_name="",
        source="budget_constrain",
    ).inc()
    assert (
        _sample_value(
            metrics.constrain_capped_total,
            sub_component_type="prefill",
            component_name="",
            source="budget_constrain",
        )
        == 1
    )


# ---------------------------------------------------------------------------
# Family-6 tick metrics (unit-level)
# ---------------------------------------------------------------------------


def test_tick_skipped_total_increments_per_plugin(metrics):
    metrics.tick_skipped_total.labels(plugin_id="p1").inc()
    metrics.tick_skipped_total.labels(plugin_id="p1").inc()
    metrics.tick_skipped_total.labels(plugin_id="p2").inc()
    assert _sample_value(metrics.tick_skipped_total, plugin_id="p1") == 2
    assert _sample_value(metrics.tick_skipped_total, plugin_id="p2") == 1


def test_tick_lag_seconds_is_last_set_value(metrics):
    metrics.tick_lag_seconds.labels(plugin_id="p1").set(0.0)
    metrics.tick_lag_seconds.labels(plugin_id="p1").set(3.2)
    metrics.tick_lag_seconds.labels(plugin_id="p2").set(0.5)
    assert _sample_value(metrics.tick_lag_seconds, plugin_id="p1") == 3.2
    assert _sample_value(metrics.tick_lag_seconds, plugin_id="p2") == 0.5


def test_tick_duration_seconds_observes_and_counts(metrics):
    metrics.tick_duration_seconds.observe(0.5)
    metrics.tick_duration_seconds.observe(2.0)
    metrics.tick_duration_seconds.observe(0.1)
    # Histogram: no labels on this one → one set of samples
    samples = list(metrics.tick_duration_seconds.collect())[0].samples
    counts = [s.value for s in samples if s.name.endswith("_count")]
    assert counts and counts[0] == 3.0


def test_tick_timeout_total_increments(metrics):
    metrics.tick_timeout_total.inc()
    metrics.tick_timeout_total.inc()
    # Unlabelled counter: value is on the Counter itself
    samples = list(metrics.tick_timeout_total.collect())[0].samples
    values = [s.value for s in samples if s.name.endswith("_total")]
    assert values and values[0] == 2.0


def test_tick_duration_buckets_span_healthy_to_deadline(metrics):
    """Buckets must cover the full range from healthy tick (~50ms) to
    the default ``tick_max_duration_seconds=30`` — operators scale
    ``tick_max_duration_seconds`` higher by default when their pipeline
    has slow user plugins, so 30s is the upper interesting bucket."""
    metrics.tick_duration_seconds.observe(0.0)
    samples = list(metrics.tick_duration_seconds.collect())[0].samples
    buckets = {s.labels["le"] for s in samples if s.name.endswith("_bucket")}
    assert "0.01" in buckets
    assert "30.0" in buckets or "30" in buckets
    assert "+Inf" in buckets


def test_reject_short_circuited_total_increments(metrics):
    metrics.reject_short_circuited_total.labels(plugin_id="safety_plugin").inc()
    metrics.reject_short_circuited_total.labels(plugin_id="safety_plugin").inc()
    metrics.reject_short_circuited_total.labels(plugin_id="other_plugin").inc()
    assert (
        _sample_value(
            metrics.reject_short_circuited_total, plugin_id="safety_plugin"
        )
        == 2
    )
    assert (
        _sample_value(metrics.reject_short_circuited_total, plugin_id="other_plugin")
        == 1
    )


def test_two_instances_with_fresh_registries_coexist():
    r1 = CollectorRegistry()
    r2 = CollectorRegistry()
    m1 = PluginFrameworkMetrics(registry=r1)
    m2 = PluginFrameworkMetrics(registry=r2)
    m1.plugin_evaluations_total.labels(
        plugin_id="a", stage="propose", result="accept"
    ).inc()
    m2.plugin_evaluations_total.labels(
        plugin_id="a", stage="propose", result="accept"
    ).inc(5)
    assert (
        _sample_value(
            m1.plugin_evaluations_total,
            plugin_id="a",
            stage="propose",
            result="accept",
        )
        == 1
    )
    assert (
        _sample_value(
            m2.plugin_evaluations_total,
            plugin_id="a",
            stage="propose",
            result="accept",
        )
        == 5
    )
