# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``dynamo_planner_retune_*`` metrics: the names the live retune guide
lists, the series that start at zero, the ``trace_sink_missing`` series a
recovered pod drops, and collectors shared per registry."""

from __future__ import annotations

import math

import pytest
from prometheus_client import REGISTRY, CollectorRegistry, generate_latest

from dynamo.planner.core.types import RETUNE_OUTCOMES
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


def test_exposition_uses_the_documented_names(registry):
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


def test_trace_sink_missing_drops_the_series_once_the_sink_is_back(registry):
    metrics = RetuneMetrics(registry)

    metrics.trace_sink_missing("frontend-0", True)
    metrics.trace_sink_missing("frontend-1", True)
    metrics.trace_sink_missing("frontend-0", False)
    assert _value(registry, "trace_sink_missing", pod="frontend-0") is None
    assert _value(registry, "trace_sink_missing", pod="frontend-1") == 1
    # A pod that never had a series is not an error.
    metrics.trace_sink_missing("frontend-9", False)


def test_instances_share_the_collectors_of_their_registry(registry):
    first = RetuneMetrics(registry)
    second = RetuneMetrics(registry)
    first.decision("applied")
    second.decision("applied")
    assert _value(registry, "decisions_total", outcome="applied") == 2

    # The planner serves prometheus_client.REGISTRY; a controller rebuild or
    # a second test must not hit "Duplicated timeseries".
    before = (
        REGISTRY.get_sample_value(f"{PREFIX}_failures_total", {"reason": "probe"})
        or 0.0
    )
    RetuneMetrics().failure("probe")
    RetuneMetrics().failure("probe")
    assert (
        REGISTRY.get_sample_value(f"{PREFIX}_failures_total", {"reason": "probe"})
        == before + 2
    )
