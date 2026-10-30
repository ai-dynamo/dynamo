# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``dynamo_planner_retune_*`` metrics: names, labels and values per design
§4.10, collectors shared per registry, and methods that never raise."""

from __future__ import annotations

import math

import pytest
from prometheus_client import REGISTRY, CollectorRegistry, generate_latest

from dynamo.planner.core.types import RETUNE_OUTCOMES
from dynamo.planner.retune.live.contracts import NullMetrics, RetuneMetricsSink
from dynamo.planner.retune.live.metrics import PREFIX, RetuneMetrics

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.fixture
def registry() -> CollectorRegistry:
    return CollectorRegistry()


def _value(registry: CollectorRegistry, name: str, **labels: str) -> float | None:
    return registry.get_sample_value(f"{PREFIX}_{name}", labels)


def test_metrics_implement_the_sink_protocol(registry):
    metrics: RetuneMetricsSink = RetuneMetrics(registry)
    null: RetuneMetricsSink = NullMetrics()
    for name in (
        "decision",
        "failure",
        "skipped",
        "supervisor_state",
        "scores",
        "router_converged",
        "patch_error",
        "trace_sink_missing",
        "dropped_members",
    ):
        assert callable(getattr(metrics, name))
        assert callable(getattr(null, name))


def test_counters_count_by_label(registry):
    metrics = RetuneMetrics(registry)

    metrics.decision("applied")
    metrics.decision("applied")
    metrics.decision("kept")
    metrics.failure("crn_mismatch")
    metrics.skipped("not_leader")
    metrics.patch_error("409")
    metrics.patch_error("timeout")
    metrics.patch_error("409")

    assert _value(registry, "decisions_total", outcome="applied") == 2
    assert _value(registry, "decisions_total", outcome="kept") == 1
    assert _value(registry, "failures_total", reason="crn_mismatch") == 1
    assert _value(registry, "skipped_total", reason="not_leader") == 1
    assert _value(registry, "router_patch_errors_total", code="409") == 2
    assert _value(registry, "router_patch_errors_total", code="timeout") == 1


def test_known_outcomes_and_skip_reasons_start_at_zero(registry):
    RetuneMetrics(registry)

    for outcome in (*RETUNE_OUTCOMES, "promoted", "rolled_back"):
        assert _value(registry, "decisions_total", outcome=outcome) == 0
    for reason in ("not_leader", "not_ready", "already_decided"):
        assert _value(registry, "skipped_total", reason=reason) == 0
    # Gauges of the latest decision read NaN until one is scored.
    for name in ("baseline_score", "best_score", "sweep_seconds"):
        value = _value(registry, name)
        assert value is not None and math.isnan(value)


def test_supervisor_state_gauges(registry):
    metrics = RetuneMetrics(registry)

    metrics.supervisor_state(
        leader=True,
        ready=False,
        initialized=True,
        rollback_pending=True,
        generation=3,
    )
    assert _value(registry, "leader") == 1
    assert _value(registry, "ready") == 0
    assert _value(registry, "initialized") == 1
    assert _value(registry, "rollback_pending") == 1
    assert _value(registry, "controller_generation") == 3

    metrics.supervisor_state(
        leader=False,
        ready=True,
        initialized=False,
        rollback_pending=False,
        generation=4,
    )
    assert _value(registry, "leader") == 0
    assert _value(registry, "ready") == 1
    assert _value(registry, "initialized") == 0
    assert _value(registry, "rollback_pending") == 0
    assert _value(registry, "controller_generation") == 4


def test_scores_set_nan_for_unknown_values(registry):
    metrics = RetuneMetrics(registry)

    metrics.scores(baseline=1.5, best=2.25, sweep_seconds=3600.0, missing_candidates=2)
    assert _value(registry, "baseline_score") == 1.5
    assert _value(registry, "best_score") == 2.25
    assert _value(registry, "sweep_seconds") == 3600.0
    assert _value(registry, "missing_candidates") == 2

    metrics.scores(baseline=None, best=None, sweep_seconds=None, missing_candidates=0)
    for name in ("baseline_score", "best_score", "sweep_seconds"):
        value = _value(registry, name)
        assert value is not None and math.isnan(value)
    assert _value(registry, "missing_candidates") == 0


def test_router_converged_gauge(registry):
    metrics = RetuneMetrics(registry)

    metrics.router_converged(True)
    assert _value(registry, "router_converged") == 1
    metrics.router_converged(False)
    assert _value(registry, "router_converged") == 0


def test_trace_sink_missing_drops_the_series_once_the_sink_is_back(registry):
    metrics = RetuneMetrics(registry)

    metrics.trace_sink_missing("frontend-0", True)
    metrics.trace_sink_missing("frontend-1", True)
    assert _value(registry, "trace_sink_missing", pod="frontend-0") == 1
    assert _value(registry, "trace_sink_missing", pod="frontend-1") == 1

    metrics.trace_sink_missing("frontend-0", False)
    assert _value(registry, "trace_sink_missing", pod="frontend-0") is None
    assert _value(registry, "trace_sink_missing", pod="frontend-1") == 1
    # A pod that never had a series is not an error.
    metrics.trace_sink_missing("frontend-9", False)
    assert _value(registry, "trace_sink_missing", pod="frontend-9") is None


def test_dropped_members_counts_only_positive_counts(registry):
    metrics = RetuneMetrics(registry)

    metrics.dropped_members(3)
    metrics.dropped_members(0)
    metrics.dropped_members(-1)
    metrics.dropped_members(2)
    assert _value(registry, "trace_dropped_members_total") == 5


def test_exposition_uses_the_design_names(registry):
    metrics = RetuneMetrics(registry)
    metrics.trace_sink_missing("frontend-0", True)
    text = generate_latest(registry).decode()

    for name in (
        "decisions_total",
        "failures_total",
        "skipped_total",
        "leader",
        "ready",
        "initialized",
        "rollback_pending",
        "controller_generation",
        "baseline_score",
        "best_score",
        "sweep_seconds",
        "missing_candidates",
        "router_converged",
        "router_patch_errors_total",
        "trace_sink_missing",
        "trace_dropped_members_total",
    ):
        assert f"\n# HELP {PREFIX}_{name} " in f"\n{text}", name


def test_instances_on_one_registry_share_collectors(registry):
    first = RetuneMetrics(registry)
    second = RetuneMetrics(registry)

    first.decision("applied")
    second.decision("applied")
    assert _value(registry, "decisions_total", outcome="applied") == 2


def test_default_registry_tolerates_reinstantiation():
    # The planner serves prometheus_client.REGISTRY; a controller rebuild or
    # a second test must not hit "Duplicated timeseries".
    first = RetuneMetrics()
    second = RetuneMetrics()
    before = (
        REGISTRY.get_sample_value(f"{PREFIX}_failures_total", {"reason": "probe"})
        or 0.0
    )
    first.failure("probe")
    second.failure("probe")
    assert (
        REGISTRY.get_sample_value(f"{PREFIX}_failures_total", {"reason": "probe"})
        == before + 2
    )


def test_empty_labels_read_unknown(registry):
    metrics = RetuneMetrics(registry)

    metrics.failure("")
    metrics.patch_error("")
    assert _value(registry, "failures_total", reason="unknown") == 1
    assert _value(registry, "router_patch_errors_total", code="unknown") == 1


class _Broken:
    def labels(self, *args, **kwargs):
        raise RuntimeError("broken")

    def set(self, *args, **kwargs):
        raise RuntimeError("broken")

    def inc(self, *args, **kwargs):
        raise RuntimeError("broken")

    def remove(self, *args, **kwargs):
        raise RuntimeError("broken")


def test_methods_never_raise(registry, monkeypatch):
    metrics = RetuneMetrics(registry)
    for name in vars(metrics._m):
        monkeypatch.setattr(metrics._m, name, _Broken())

    metrics.decision("applied")
    metrics.failure("k8s_error")
    metrics.skipped("not_ready")
    metrics.supervisor_state(
        leader=True, ready=True, initialized=True, rollback_pending=False, generation=1
    )
    metrics.scores(baseline=1.0, best=2.0, sweep_seconds=3.0, missing_candidates=0)
    metrics.router_converged(True)
    metrics.patch_error("503")
    metrics.trace_sink_missing("frontend-0", True)
    metrics.trace_sink_missing("frontend-0", False)
    metrics.dropped_members(1)


def test_bad_values_do_not_raise(registry):
    metrics = RetuneMetrics(registry)

    metrics.scores(
        baseline="not a number",  # type: ignore[arg-type]
        best=None,
        sweep_seconds=None,
        missing_candidates=0,
    )
    metrics.dropped_members(None)  # type: ignore[arg-type]
    metrics.supervisor_state(
        leader=True,
        ready=True,
        initialized=True,
        rollback_pending=False,
        generation="x",  # type: ignore[arg-type]
    )
