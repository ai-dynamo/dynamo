# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PlannerPrometheusMetrics label injection.

When ``model_name`` is supplied, every gauge and enum exposed by the
planner should carry a ``model_name`` static label. When omitted, the
metric output must be byte-identical to the prior unlabeled behaviour
(back-compat for existing scrapers / dashboards).
"""

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from dynamo.planner.monitoring.planner_metrics import (
    PlannerPrometheusMetrics,
    _PreLabeledMetric,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _scrape(registry: CollectorRegistry) -> str:
    return generate_latest(registry).decode()


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Each test gets a fresh default registry to avoid duplicate metric
    name collisions across tests."""
    fresh = CollectorRegistry()
    monkeypatch.setattr(
        "dynamo.planner.monitoring.planner_metrics.Gauge",
        _wrap_gauge_for_registry(fresh),
    )
    monkeypatch.setattr(
        "dynamo.planner.monitoring.planner_metrics.Enum",
        _wrap_enum_for_registry(fresh),
    )
    yield fresh


def _wrap_gauge_for_registry(registry: CollectorRegistry):
    from prometheus_client import Gauge as _G

    def _ctor(name, doc, labelnames=()):
        return _G(name, doc, labelnames=labelnames, registry=registry)

    return _ctor


def _wrap_enum_for_registry(registry: CollectorRegistry):
    from prometheus_client import Enum as _E

    def _ctor(name, doc, *, states, labelnames=()):
        return _E(name, doc, states=states, labelnames=labelnames, registry=registry)

    return _ctor


def test_no_model_name_keeps_metrics_unlabeled(_isolated_registry):
    pm = PlannerPrometheusMetrics()
    pm.num_prefill_replicas.set(3)
    pm.engine_queued_prefill_tokens.labels(worker_id="w1", dp_rank="0").set(42)
    pm.load_scaling_decision.state("scale_up")

    out = _scrape(_isolated_registry)
    assert "dynamo_planner_num_prefill_replicas 3.0" in out
    assert (
        'dynamo_planner_engine_queued_prefill_tokens{dp_rank="0",worker_id="w1"} 42.0'
        in out
    )
    # No model_name label present anywhere
    assert "model_name=" not in out


def test_empty_string_model_name_still_labels(_isolated_registry):
    # ``model_name=""`` is an explicit (if unusual) request to label every
    # metric with an empty model_name value.  A truthiness check would treat
    # it as "unset" and produce unlabeled metrics, which is the opposite of
    # the caller's intent.  Guard against that regression.
    pm = PlannerPrometheusMetrics(model_name="")
    pm.num_prefill_replicas.set(1)

    out = _scrape(_isolated_registry)
    assert 'dynamo_planner_num_prefill_replicas{model_name=""} 1.0' in out


def test_model_name_labels_simple_gauges(_isolated_registry):
    model = "google/gemma-3-27b-it-disagg"
    pm = PlannerPrometheusMetrics(model_name=model)
    pm.num_prefill_replicas.set(1)
    pm.predicted_num_decode_replicas.set(2)
    pm.observed_ttft_ms.set(123.4)

    out = _scrape(_isolated_registry)
    assert f'dynamo_planner_num_prefill_replicas{{model_name="{model}"}} 1.0' in out
    assert (
        f'dynamo_planner_predicted_num_decode_replicas{{model_name="{model}"}} 2.0'
        in out
    )
    assert f'dynamo_planner_observed_ttft_ms{{model_name="{model}"}} 123.4' in out


def test_model_name_merges_with_existing_engine_labels(_isolated_registry):
    model = "google/gemma-3-27b-it-disagg"
    pm = PlannerPrometheusMetrics(model_name=model)
    pm.engine_queued_prefill_tokens.labels(worker_id="w1", dp_rank="0").set(99)
    pm.engine_queued_decode_kv_tokens.labels(worker_id="w2", dp_rank="1").set(11)

    out = _scrape(_isolated_registry)
    # Labels are emitted alphabetically: dp_rank, model_name, worker_id.
    assert (
        "dynamo_planner_engine_queued_prefill_tokens"
        f'{{dp_rank="0",model_name="{model}",worker_id="w1"}} 99.0'
    ) in out
    assert (
        "dynamo_planner_engine_queued_decode_kv_tokens"
        f'{{dp_rank="1",model_name="{model}",worker_id="w2"}} 11.0'
    ) in out


def test_model_name_labels_enums(_isolated_registry):
    model = "m1"
    pm = PlannerPrometheusMetrics(model_name=model)
    pm.load_scaling_decision.state("scale_down")

    out = _scrape(_isolated_registry)
    # Enum sets one series per state; the chosen state has value 1.0.
    assert (
        "dynamo_planner_load_scaling_decision"
        f'{{dynamo_planner_load_scaling_decision="scale_down",model_name="{model}"}}'
        " 1.0"
    ) in out
    # Other states should be 0.0 with the same model_name label.
    assert f'model_name="{model}"' in out


def test_pre_labeled_metric_forwards_inc_dec_clear(_isolated_registry):
    model = "m"
    pm = PlannerPrometheusMetrics(model_name=model)
    # gpu_hours is a simple Gauge; .inc / .dec should accumulate through
    # the wrapper.
    pm.gpu_hours.inc(2.5)
    pm.gpu_hours.inc(1.5)
    pm.gpu_hours.dec(0.5)

    out = _scrape(_isolated_registry)
    assert f'dynamo_planner_gpu_hours{{model_name="{model}"}} 3.5' in out

    # clear() on an engine gauge should drop all per-worker children.
    pm.engine_queued_prefill_tokens.labels(worker_id="w", dp_rank="0").set(7)
    out = _scrape(_isolated_registry)
    assert "dynamo_planner_engine_queued_prefill_tokens{" in out
    pm.engine_queued_prefill_tokens.clear()
    out = _scrape(_isolated_registry)
    assert "dynamo_planner_engine_queued_prefill_tokens{" not in out


def test_labels_accepts_positional_values(_isolated_registry):
    # Mirror prometheus_client's labels(*values) signature so existing
    # positional callers (and any future ones in the planner code base)
    # keep working when model_name is configured.
    pm = PlannerPrometheusMetrics(model_name="m")
    # engine_queued_prefill_tokens declares user labels [worker_id, dp_rank];
    # passing them positionally must produce the labeled child series.
    pm.engine_queued_prefill_tokens.labels("w7", "3").set(42)
    out = _scrape(_isolated_registry)
    assert (
        "dynamo_planner_engine_queued_prefill_tokens"
        '{dp_rank="3",model_name="m",worker_id="w7"} 42.0'
    ) in out


def test_labels_positional_wrong_arity_raises(_isolated_registry):
    pm = PlannerPrometheusMetrics(model_name="m")
    with pytest.raises(ValueError, match="positional label values"):
        # engine gauge needs 2 user labels (worker_id, dp_rank); 1 is wrong.
        pm.engine_queued_prefill_tokens.labels("only-one")


def test_labels_passthrough_when_no_static(_isolated_registry):
    # With no model_name, the wrapper must not invent label ordering or
    # arity checks -- forward to the underlying metric unchanged.
    pm = PlannerPrometheusMetrics()
    pm.engine_queued_prefill_tokens.labels("w1", "0").set(5)
    pm.engine_queued_prefill_tokens.labels(worker_id="w2", dp_rank="1").set(6)
    out = _scrape(_isolated_registry)
    assert 'dp_rank="0",worker_id="w1"} 5.0' in out
    assert 'dp_rank="1",worker_id="w2"} 6.0' in out


def test_observe_forwards_through_static_label(_isolated_registry, monkeypatch):
    # _PreLabeledMetric.observe() forwards Histogram/Summary observations
    # through the labeled child so the static model_name label is preserved.
    # Use a stub metric (no Prometheus registry interaction) since the
    # planner doesn't ship any Histograms today; this guards future use.
    from dynamo.planner.monitoring.planner_metrics import _PreLabeledMetric

    class _StubLabeled:
        def __init__(self):
            self.observed: list[float] = []

        def observe(self, amount: float) -> None:
            self.observed.append(amount)

    class _StubHistogram:
        def __init__(self):
            self.child = _StubLabeled()
            self.unlabeled_observations: list[float] = []

        def labels(self, **kwargs: str):
            assert kwargs == {"model_name": "m"}
            return self.child

        def observe(self, amount: float) -> None:
            self.unlabeled_observations.append(amount)

    h = _StubHistogram()
    wrapped = _PreLabeledMetric(h, model_name="m")
    wrapped.observe(1.5)
    assert h.child.observed == [1.5]
    # Unlabeled path must not be touched when a static label is bound.
    assert h.unlabeled_observations == []


def test_pre_labeled_metric_isinstance_and_attr_passthrough(_isolated_registry):
    pm = PlannerPrometheusMetrics(model_name="m")
    # Wrapper exposes the underlying metric attributes via __getattr__.
    inner = pm.num_prefill_replicas._metric  # type: ignore[attr-defined]
    assert isinstance(pm.num_prefill_replicas, _PreLabeledMetric)
    # Doc string is on the underlying metric.
    assert "Current number of prefill replicas" in (
        getattr(inner, "_documentation", "") or ""
    )
