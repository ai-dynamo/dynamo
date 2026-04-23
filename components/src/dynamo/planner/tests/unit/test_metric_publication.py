# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Prometheus metric publication in NativePlannerBase.

Covers:
- ``_publish_inventory_and_gpu_hours``: replica counts + cumulative
  gpu_hours must publish on every tick (not just throughput ticks) and
  accumulate using wall-clock deltas.
- ``_report_diagnostics``: scaling-decision Enum gauges must only be
  written for the scaling path that actually ran this tick, so
  load-only ticks don't wipe the throughput Enum (and vice versa).
"""

import os
from unittest.mock import Mock, patch

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.core.types import (
    ScheduledTick,
    TickDiagnostics,
    TickInput,
    WorkerCounts,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _make_planner(prometheus_enabled: bool = True) -> NativePlannerBase:
    """Build a minimal NativePlannerBase with a mock Prometheus metrics object.

    ``start_http_server`` is patched so no real HTTP port is bound (tests
    would otherwise fail in parallel or on shared hosts). ``prometheus_port``
    is toggled post-init as an enabled/disabled gate for the methods under
    test; the actual port value is never used for I/O.
    """
    with patch(
        "dynamo.planner.core.base.PlannerPrometheusMetrics"
    ) as mock_metrics, patch("dynamo.planner.core.base.start_http_server"), patch(
        "dynamo.planner.connectors.kubernetes.KubernetesAPI"
    ), patch.dict(
        os.environ, {"DYN_PARENT_DGD_K8S_NAME": "test-graph"}
    ):
        mock_metrics.return_value = Mock()
        config = PlannerConfig.model_construct(
            throughput_adjustment_interval=60,
            prefill_engine_num_gpu=2,
            decode_engine_num_gpu=4,
            min_endpoint=1,
            max_gpu_budget=-1,
            ttft=500.0,
            itl=50.0,
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
            load_adjustment_interval=5,
            max_num_fpm_samples=50,
            fpm_sample_bucket_size=16,
            load_scaling_down_sensitivity=80,
            load_metric_samples=10,
            load_min_observations=5,
        )
        planner = NativePlannerBase(None, config)
    # Gate the methods under test without binding a real port.
    planner.prometheus_port = 1 if prometheus_enabled else 0
    return planner


def _tick_input(now_s: float, num_p: int = 2, num_d: int = 3) -> TickInput:
    return TickInput(
        now_s=now_s,
        worker_counts=WorkerCounts(ready_num_prefill=num_p, ready_num_decode=num_d),
    )


# ── Bug 1: inventory & gpu_hours publish every tick ─────────────────


class TestPublishInventoryAndGpuHours:
    """Inventory/gpu_hours gauges must publish on every tick regardless of
    whether the throughput-scaling path is enabled."""

    def test_replica_gauges_set_on_first_call(self):
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._publish_inventory_and_gpu_hours(_tick_input(now_s=1000.0))

        pm.num_prefill_replicas.set.assert_called_once_with(2)
        pm.num_decode_replicas.set.assert_called_once_with(3)

    def test_first_call_contributes_zero_gpu_hours(self):
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._publish_inventory_and_gpu_hours(_tick_input(now_s=1000.0))

        # First call has no prior timestamp -> delta is zero.
        assert planner._cumulative_gpu_hours == 0.0
        pm.gpu_hours.set.assert_called_once_with(0.0)

    def test_cumulative_gpu_hours_uses_wall_clock_delta(self):
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._publish_inventory_and_gpu_hours(_tick_input(now_s=1000.0))
        planner._publish_inventory_and_gpu_hours(
            _tick_input(now_s=1180.0, num_p=2, num_d=3)
        )

        # dt = 180s, gpu_count = 2*2 + 3*4 = 16, gpu_hours = 16 * 180 / 3600 = 0.8
        assert planner._cumulative_gpu_hours == pytest.approx(0.8)
        pm.gpu_hours.set.assert_called_with(pytest.approx(0.8))

    def test_accumulates_across_multiple_ticks(self):
        planner = _make_planner()
        # Two 5-second ticks with 1 prefill + 1 decode worker each on single GPUs.
        planner.config.prefill_engine_num_gpu = 1
        planner.config.decode_engine_num_gpu = 1

        planner._publish_inventory_and_gpu_hours(
            _tick_input(now_s=0.0, num_p=1, num_d=1)
        )
        planner._publish_inventory_and_gpu_hours(
            _tick_input(now_s=5.0, num_p=1, num_d=1)
        )
        planner._publish_inventory_and_gpu_hours(
            _tick_input(now_s=10.0, num_p=1, num_d=1)
        )

        # Two deltas of 5s each, 2 GPUs total -> 2 * (2 * 5 / 3600) = 20/3600
        assert planner._cumulative_gpu_hours == pytest.approx(20.0 / 3600.0)

    def test_prometheus_disabled_still_accumulates_gpu_hours(self):
        # Prometheus export off, but the HTML recorder / live dashboard
        # still consumes _cumulative_gpu_hours, so accumulation must
        # continue. Only the gauge publishes should be skipped.
        planner = _make_planner(prometheus_enabled=False)
        pm = planner.prometheus_metrics

        planner._publish_inventory_and_gpu_hours(_tick_input(now_s=1000.0))
        planner._publish_inventory_and_gpu_hours(
            _tick_input(now_s=1180.0, num_p=2, num_d=3)
        )

        # dt = 180s, gpu_count = 2*2 + 3*4 = 16, gpu_hours = 16 * 180 / 3600 = 0.8
        assert planner._cumulative_gpu_hours == pytest.approx(0.8)
        pm.num_prefill_replicas.set.assert_not_called()
        pm.num_decode_replicas.set.assert_not_called()
        pm.gpu_hours.set.assert_not_called()

    def test_handles_none_worker_counts(self):
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._publish_inventory_and_gpu_hours(
            TickInput(now_s=1000.0, worker_counts=None)
        )

        pm.num_prefill_replicas.set.assert_not_called()
        pm.num_decode_replicas.set.assert_not_called()


# ── Bug 3: per-path enum publishes ──────────────────────────────────


def _diag(
    load_reason: str | None = None, throughput_reason: str | None = None
) -> TickDiagnostics:
    return TickDiagnostics(
        load_decision_reason=load_reason,
        throughput_decision_reason=throughput_reason,
    )


def _tick(run_load: bool, run_throughput: bool) -> ScheduledTick:
    return ScheduledTick(
        at_s=0.0,
        run_load_scaling=run_load,
        run_throughput_scaling=run_throughput,
        need_worker_states=True,
        need_worker_fpm=run_load,
        need_traffic_metrics=run_throughput,
    )


class TestReportDiagnosticsEnumGating:
    """Scaling-decision Enum gauges must only be written for the path that
    actually ran this tick, so load-only ticks don't clobber the
    throughput Enum (and vice versa)."""

    def test_load_only_tick_does_not_touch_throughput_enum(self):
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=True, run_throughput=False),
            _diag(load_reason="scale_up"),
        )

        pm.load_scaling_decision.state.assert_called_once_with("scale_up")
        pm.throughput_scaling_decision.state.assert_not_called()

    def test_throughput_only_tick_does_not_touch_load_enum(self):
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=False, run_throughput=True),
            _diag(throughput_reason="scale"),
        )

        pm.throughput_scaling_decision.state.assert_called_once_with("scale")
        pm.load_scaling_decision.state.assert_not_called()

    def test_combined_tick_publishes_both(self):
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=True, run_throughput=True),
            _diag(load_reason="no_change", throughput_reason="set_lower_bound"),
        )

        pm.load_scaling_decision.state.assert_called_once_with("no_change")
        pm.throughput_scaling_decision.state.assert_called_once_with("set_lower_bound")

    def test_run_tick_with_no_reason_still_writes_unset(self):
        # Defensive: if a scaling path ran but didn't populate a reason,
        # explicitly write "unset" so stale state from a prior tick
        # doesn't linger.
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._report_diagnostics(_tick(run_load=True, run_throughput=False), _diag())

        pm.load_scaling_decision.state.assert_called_once_with("unset")
        pm.throughput_scaling_decision.state.assert_not_called()

    def test_skipped_when_prometheus_disabled(self):
        planner = _make_planner(prometheus_enabled=False)
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=True, run_throughput=True),
            _diag(load_reason="scale_up", throughput_reason="scale"),
        )

        pm.load_scaling_decision.state.assert_not_called()
        pm.throughput_scaling_decision.state.assert_not_called()
