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

These tests pin that those five gauges are written only when a prediction is
actually present -- on a builtin throughput tick, or on any tick whose
diagnostics still carry a prediction -- and never on an empty load-only tick,
while the load-stage ``estimated_*`` gauges remain written every tick. The
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
        """Common path unchanged: load-stage estimated_* gauges are still set
        every tick (they are produced by the load stage, not guarded)."""
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
