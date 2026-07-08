# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for throughput-cadence Prometheus gauge gating.

Companion to ``TestReportDiagnosticsEnumGating`` in
``test_metric_publication.py`` (added by #8575). That PR guarded the two
scaling-decision *Enum* gauges on ``tick.run_{load,throughput}_scaling`` so
load-only ticks stop clobbering them. The same ``_report_diagnostics`` method
left the *numeric* throughput-stage gauges unguarded:

  - ``predicted_requests_per_second``
  - ``predicted_input_sequence_tokens``
  - ``predicted_output_sequence_tokens``
  - ``engine_prefill_capacity_requests_per_second``
  - ``engine_decode_capacity_requests_per_second``

All five are populated only by the throughput stage
(``advance_throughput_from_prediction``); ``_reset_diag()`` nulls them at the
start of every tick. Published unconditionally, they were written ``0`` on the
~35 of every 36 load-only ticks (default load=5s / throughput=180s), so Grafana
reads ~0 essentially always while the planner logs real predictions every 180s.

These tests pin the per-gauge publication convention (review follow-up on
#10804): each numeric gauge is written only when its own diagnostic field is
present. ``None`` = no new observation this tick -> leave the gauge unchanged;
a concrete value (including ``0.0``) = asserted observation -> publish it.
``PredictionData`` explicitly supports partial predictions, so a tick that
asserts only some fields must not zero the sibling gauges. The
throughput-decision *Enum* stays gated on ``tick.run_throughput_scaling``.
"""

import os
from unittest.mock import Mock, patch

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.core.types import ScheduledTick, TickDiagnostics

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _make_planner(prometheus_enabled: bool = True) -> NativePlannerBase:
    """Minimal NativePlannerBase with a mocked Prometheus metrics object."""
    with patch(
        "dynamo.planner.core.base.PlannerPrometheusMetrics"
    ) as mock_metrics, patch("dynamo.planner.core.base.start_http_server"), patch(
        "dynamo.planner.connectors.kubernetes.KubernetesAPI"
    ), patch.dict(
        os.environ, {"DYN_PARENT_DGD_K8S_NAME": "test-graph"}
    ):
        mock_metrics.return_value = Mock()
        config = PlannerConfig.model_construct(
            throughput_adjustment_interval_seconds=180,
            prefill_engine_num_gpu=2,
            decode_engine_num_gpu=4,
            min_endpoint=1,
            max_gpu_budget=-1,
            ttft_ms=500.0,
            itl_ms=50.0,
            backend="vllm",
            no_operation=True,
            metric_pulling_prometheus_endpoint="http://localhost:9090",
            metric_reporting_prometheus_port=0,
            load_predictor="constant",
            environment="kubernetes",
            namespace="test-namespace",
            mode="disagg",
            enable_load_scaling=True,
            enable_throughput_scaling=True,
            load_adjustment_interval_seconds=5,
            max_num_fpm_samples=50,
            fpm_sample_bucket_size=16,
            load_scaling_down_sensitivity=80,
            load_min_observations=5,
        )
        planner = NativePlannerBase(None, config)
    planner.prometheus_port = 1 if prometheus_enabled else 0
    return planner


def _tick(run_load: bool, run_throughput: bool) -> ScheduledTick:
    """A ScheduledTick that runs the load and/or throughput scaling stages."""
    return ScheduledTick(
        at_s=0.0,
        run_load_scaling=run_load,
        run_throughput_scaling=run_throughput,
        need_worker_states=True,
        need_worker_fpm=run_load,
        need_traffic_metrics=run_throughput,
    )


def _diag_with_prediction() -> TickDiagnostics:
    """A diagnostics snapshot as produced on a throughput tick (predictions set)."""
    return TickDiagnostics(
        predicted_num_req=1330.0,
        predicted_isl=8008.0,
        predicted_osl=946.0,
        engine_rps_prefill=2.5,
        engine_rps_decode=4.0,
        throughput_decision_reason="scale_up",
    )


def _diag_load_only() -> TickDiagnostics:
    """A diagnostics snapshot as produced on a load-only tick.

    ``_reset_diag()`` ran at tick start and the throughput stage did not, so all
    predicted/engine fields are None; only the load-stage estimates are present.
    """
    return TickDiagnostics(
        estimated_ttft_ms=171.0,
        estimated_itl_ms=12.0,
        load_decision_reason="no_change",
    )


_PREDICTED_GAUGES = (
    "predicted_requests_per_second",
    "predicted_input_sequence_tokens",
    "predicted_output_sequence_tokens",
    "engine_prefill_capacity_requests_per_second",
    "engine_decode_capacity_requests_per_second",
)


class TestReportDiagnosticsThroughputGauges:
    """Throughput-cadence numeric gauges must be written only when a prediction
    is actually present (the throughput loop is due, or a PREDICT plugin
    produced one), so the ~35/36 empty load-only ticks don't zero the last real
    value while a genuine off-cadence prediction is still published."""

    def test_load_only_tick_does_not_touch_throughput_gauges(self):
        """REGRESSION: a load-only tick must not write the predicted/engine
        gauges (pre-fix it wrote 0 to all five, wiping the prior value)."""
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=True, run_throughput=False), _diag_load_only()
        )

        for name in _PREDICTED_GAUGES:
            getattr(pm, name).set.assert_not_called()

    def test_throughput_tick_publishes_throughput_gauges(self):
        """A throughput tick writes all five gauges with their real values."""
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=False, run_throughput=True), _diag_with_prediction()
        )

        # predicted rps = num_req / throughput_adjustment_interval_seconds
        pm.predicted_requests_per_second.set.assert_called_once_with(1330.0 / 180)
        pm.predicted_input_sequence_tokens.set.assert_called_once_with(8008.0)
        pm.predicted_output_sequence_tokens.set.assert_called_once_with(946.0)
        pm.engine_prefill_capacity_requests_per_second.set.assert_called_once_with(2.5)
        pm.engine_decode_capacity_requests_per_second.set.assert_called_once_with(4.0)

    def test_estimated_gauges_still_written_on_load_only_tick(self):
        """Common path unchanged: load-stage estimated_* gauges are published
        on the ticks that produce them (the load stage)."""
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=True, run_throughput=False), _diag_load_only()
        )

        pm.estimated_ttft_ms.set.assert_called_once_with(171.0)
        pm.estimated_itl_ms.set.assert_called_once_with(12.0)

    def test_prediction_present_without_throughput_flag_publishes_gauges(self):
        """A tick that did not set ``run_throughput_scaling`` but still carries a
        prediction (e.g. an independently-scheduled PREDICT plugin produced one)
        must publish the five numeric gauges -- gating on the flag alone would
        drop a real value. The throughput-decision enum stays gated on the flag,
        so it must NOT be written on such a tick."""
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=True, run_throughput=False), _diag_with_prediction()
        )

        pm.predicted_requests_per_second.set.assert_called_once_with(1330.0 / 180)
        pm.predicted_input_sequence_tokens.set.assert_called_once_with(8008.0)
        pm.predicted_output_sequence_tokens.set.assert_called_once_with(946.0)
        pm.engine_prefill_capacity_requests_per_second.set.assert_called_once_with(2.5)
        pm.engine_decode_capacity_requests_per_second.set.assert_called_once_with(4.0)
        # enum is throughput-cadence only, not prediction-presence driven
        pm.throughput_scaling_decision.state.assert_not_called()

    def test_skipped_when_prometheus_disabled(self):
        """No gauge writes at all when prometheus_port=0."""
        planner = _make_planner(prometheus_enabled=False)
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=True, run_throughput=True), _diag_with_prediction()
        )

        for name in _PREDICTED_GAUGES:
            getattr(pm, name).set.assert_not_called()
        pm.estimated_ttft_ms.set.assert_not_called()


class TestPerGaugePublicationConvention:
    """Review follow-up on #10804: ``PredictionData`` supports partial
    predictions, so gauge writes must be gated per field. ``None`` = no new
    observation -> leave that gauge unchanged; a concrete value (including
    ``0.0``) = a real observed value -> publish it."""

    def test_partial_prediction_updates_only_asserted_gauge(self):
        """REGRESSION (review probe): a prediction asserting only
        ``predicted_num_req`` must update the RPS gauge and leave the other
        four untouched. Pre-fix, the group gate opened on any present field and
        wrote 0 to the four missing ones, erasing latched real values."""
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=True, run_throughput=False),
            TickDiagnostics(predicted_num_req=1800.0),
        )

        pm.predicted_requests_per_second.set.assert_called_once_with(1800.0 / 180)
        pm.predicted_input_sequence_tokens.set.assert_not_called()
        pm.predicted_output_sequence_tokens.set.assert_not_called()
        pm.engine_prefill_capacity_requests_per_second.set.assert_not_called()
        pm.engine_decode_capacity_requests_per_second.set.assert_not_called()

    def test_asserted_zero_is_published(self):
        """0.0 is a real observation, not a missing one: an asserted zero must
        be published (an ``or 0`` fallback cannot distinguish the two)."""
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=True, run_throughput=False),
            TickDiagnostics(predicted_num_req=0.0, predicted_isl=0.0),
        )

        pm.predicted_requests_per_second.set.assert_called_once_with(0.0)
        pm.predicted_input_sequence_tokens.set.assert_called_once_with(0.0)
        pm.predicted_output_sequence_tokens.set.assert_not_called()

    def test_throughput_tick_with_missing_field_skips_that_gauge(self):
        """Even on a builtin throughput tick, a field the stage did not produce
        (e.g. no prefill capacity in aggregated mode) must not be zeroed --
        the throughput flag alone no longer forces all five writes."""
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=False, run_throughput=True),
            TickDiagnostics(
                predicted_num_req=1330.0,
                predicted_isl=8008.0,
                predicted_osl=946.0,
                engine_rps_decode=4.0,
                throughput_decision_reason="scale_up",
            ),
        )

        pm.engine_prefill_capacity_requests_per_second.set.assert_not_called()
        pm.engine_decode_capacity_requests_per_second.set.assert_called_once_with(4.0)
        pm.throughput_scaling_decision.state.assert_called_once_with("scale_up")

    def test_estimated_gauges_not_zeroed_when_absent(self):
        """The load-stage estimated_* gauges follow the same convention: a tick
        without fresh estimates must not overwrite the last real value with 0."""
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=False, run_throughput=True), _diag_with_prediction()
        )

        pm.estimated_ttft_ms.set.assert_not_called()
        pm.estimated_itl_ms.set.assert_not_called()

    def test_nonpositive_interval_leaves_rps_unchanged(self):
        """If the RPS gauge cannot be derived (interval <= 0), treat it as no
        observation rather than publishing 0."""
        planner = _make_planner()
        planner.config.throughput_adjustment_interval_seconds = 0
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=False, run_throughput=True),
            TickDiagnostics(predicted_num_req=1800.0, predicted_isl=8008.0),
        )

        pm.predicted_requests_per_second.set.assert_not_called()
        pm.predicted_input_sequence_tokens.set.assert_called_once_with(8008.0)
