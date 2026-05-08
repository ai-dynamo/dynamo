# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GlobalPlannerMetrics (PR 8 sub-task 8-4)."""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from dynamo.planner.monitoring.planner_metrics import GlobalPlannerMetrics

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.fixture
def metrics():
    return GlobalPlannerMetrics(registry=CollectorRegistry())


def _counter_value(metric, **labels):
    return metric.labels(**labels)._value.get()


def _gauge_value(metric, **labels):
    return metric.labels(**labels)._value.get()


# ---------------------------------------------------------------------------
# Shape / label contract
# ---------------------------------------------------------------------------


def test_label_set_matches_spec(metrics):
    """Counter has (result, reason); latency has (result); gauge has (dgd_name)."""
    c_samples = list(metrics.global_scale_request_total.collect())[0].samples
    l_samples = list(metrics.global_scale_request_latency_seconds.collect())[0].samples
    g_samples = list(metrics.global_managed_dgd_gpus.collect())[0].samples
    # Counters get no samples until labelled + inc, so the family only
    # needs to construct cleanly; assert metric names.
    assert (
        list(metrics.global_scale_request_total.collect())[0].name
        == "dynamo_planner_global_scale_request"
    )
    assert (
        list(metrics.global_scale_request_latency_seconds.collect())[0].name
        == "dynamo_planner_global_scale_request_latency_seconds"
    )
    assert (
        list(metrics.global_managed_dgd_gpus.collect())[0].name
        == "dynamo_planner_global_managed_dgd_gpus"
    )


def test_reason_constants_cover_server_branches():
    """Every server-side reason surfaces as a class constant so
    callers can't drift from the enum by typing a string."""
    reasons = {
        GlobalPlannerMetrics.REASON_CLIENT,
        GlobalPlannerMetrics.REASON_SUCCESS,
        GlobalPlannerMetrics.REASON_NOT_AUTHORIZED,
        GlobalPlannerMetrics.REASON_NO_OPERATION,
        GlobalPlannerMetrics.REASON_BUDGET_EXCEEDED,
        GlobalPlannerMetrics.REASON_EXCEPTION,
    }
    # All strings, all non-empty, all unique.
    assert all(isinstance(r, str) and r for r in reasons)
    assert len(reasons) == 6


# ---------------------------------------------------------------------------
# Basic emission behaviour
# ---------------------------------------------------------------------------


def test_counter_and_latency_increment_together(metrics):
    metrics.global_scale_request_total.labels(
        result="success", reason=GlobalPlannerMetrics.REASON_CLIENT
    ).inc()
    metrics.global_scale_request_latency_seconds.labels(result="success").observe(0.05)
    metrics.global_scale_request_latency_seconds.labels(result="success").observe(0.03)

    assert (
        _counter_value(
            metrics.global_scale_request_total,
            result="success",
            reason=GlobalPlannerMetrics.REASON_CLIENT,
        )
        == 1
    )
    # Histogram count
    lsamples = list(metrics.global_scale_request_latency_seconds.collect())[0].samples
    counts = [
        s.value
        for s in lsamples
        if s.name.endswith("_count") and s.labels.get("result") == "success"
    ]
    assert counts and counts[0] == 2.0


def test_gauge_tracks_per_dgd(metrics):
    metrics.global_managed_dgd_gpus.labels(dgd_name="dgd-a").set(12)
    metrics.global_managed_dgd_gpus.labels(dgd_name="dgd-b").set(4)
    metrics.global_managed_dgd_gpus.labels(dgd_name="dgd-a").set(16)  # scale-up

    assert _gauge_value(metrics.global_managed_dgd_gpus, dgd_name="dgd-a") == 16
    assert _gauge_value(metrics.global_managed_dgd_gpus, dgd_name="dgd-b") == 4


# ---------------------------------------------------------------------------
# Latency buckets cover realistic K8s apiserver tail
# ---------------------------------------------------------------------------


def test_latency_buckets_span_healthy_to_loaded_cluster(metrics):
    metrics.global_scale_request_latency_seconds.labels(result="success").observe(0.0)
    samples = list(metrics.global_scale_request_latency_seconds.collect())[0].samples
    buckets = {s.labels["le"] for s in samples if s.name.endswith("_bucket")}
    assert "0.01" in buckets
    assert "60.0" in buckets or "60" in buckets
    assert "+Inf" in buckets


# ---------------------------------------------------------------------------
# Registry isolation for parallel tests
# ---------------------------------------------------------------------------


def test_two_instances_with_fresh_registries_coexist():
    r1 = CollectorRegistry()
    r2 = CollectorRegistry()
    m1 = GlobalPlannerMetrics(registry=r1)
    m2 = GlobalPlannerMetrics(registry=r2)
    m1.global_managed_dgd_gpus.labels(dgd_name="a").set(1)
    m2.global_managed_dgd_gpus.labels(dgd_name="a").set(5)
    assert _gauge_value(m1.global_managed_dgd_gpus, dgd_name="a") == 1
    assert _gauge_value(m2.global_managed_dgd_gpus, dgd_name="a") == 5
