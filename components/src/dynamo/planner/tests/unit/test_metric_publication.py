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
- SLA target gauges (``sla_target_ttft_ms``, ``sla_target_itl_ms``):
  must be written once at planner startup with the configured SLA
  values so Grafana dashboards can compare observed/estimated latency
  against the operator-defined target.
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.core.types import (
    ScheduledTick,
    TickDiagnostics,
    TickInput,
    WorkerCounts,
)
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.errors import DeploymentValidationError
from dynamo.planner.monitoring.planner_metrics import PREFIX, PlannerPrometheusMetrics
from dynamo.planner.monitoring.traffic_metrics import Metrics
from dynamo.planner.monitoring.worker_info import WorkerInfo

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _make_environment(config: PlannerConfig) -> Mock:
    state = DeploymentState()
    state.prefill.num_gpus = config.prefill_engine_num_gpu
    state.decode.num_gpus = config.decode_engine_num_gpu

    environment = Mock()
    environment.deployment_state.return_value = state
    environment.runtime_namespace.return_value = config.namespace
    environment.metrics_state.return_value = Metrics()
    return environment


def _make_planner(prometheus_enabled: bool = True) -> NativePlannerBase:
    """Build a minimal NativePlannerBase with a mock Prometheus metrics object.

    ``start_http_server`` is patched so no real HTTP port is bound (tests
    would otherwise fail in parallel or on shared hosts). ``prometheus_port``
    is toggled post-init as an enabled/disabled gate for the methods under
    test; the actual port value is never used for I/O.
    """
    with (
        patch("dynamo.planner.core.base.PlannerPrometheusMetrics") as mock_metrics,
        patch("dynamo.planner.core.base.start_http_server"),
        patch("dynamo.planner.connectors.kubernetes.KubernetesAPI"),
        patch.dict(os.environ, {"DYN_PARENT_DGD_K8S_NAME": "test-graph"}),
    ):
        mock_metrics.return_value = Mock()
        config = PlannerConfig.model_construct(
            throughput_adjustment_interval_seconds=60,
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
        planner = NativePlannerBase(None, config, _make_environment(config))
    # Gate the methods under test without binding a real port.
    planner.prometheus_port = 1 if prometheus_enabled else 0
    return planner


def _tick_input(now_s: float, num_p: int = 2, num_d: int = 3) -> TickInput:
    """Build a TickInput with the given wall-clock time and worker counts."""
    return TickInput(
        now_s=now_s,
        worker_counts=WorkerCounts(ready_num_prefill=num_p, ready_num_decode=num_d),
    )


# ── Bug 1: inventory & gpu_hours publish every tick ─────────────────


class TestPublishInventoryAndGpuHours:
    """Inventory/gpu_hours gauges must publish on every tick regardless of
    whether the throughput-scaling path is enabled."""

    def test_replica_gauges_set_on_first_call(self):
        """Prefill and decode replica gauges are written with current counts."""
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._publish_inventory_and_gpu_hours(_tick_input(now_s=1000.0))

        pm.num_prefill_replicas.set.assert_called_once_with(2)
        pm.num_decode_replicas.set.assert_called_once_with(3)

    def test_first_call_contributes_zero_gpu_hours(self):
        """First tick has no prior timestamp so the gpu_hours delta is zero."""
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._publish_inventory_and_gpu_hours(_tick_input(now_s=1000.0))

        # First call has no prior timestamp -> delta is zero.
        assert planner._cumulative_gpu_hours == 0.0
        pm.gpu_hours.set.assert_called_once_with(0.0)

    def test_cumulative_gpu_hours_uses_wall_clock_delta(self):
        """gpu_hours accumulates using wall-clock delta between ticks."""
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
        """gpu_hours increases monotonically across successive ticks."""
        planner = _make_planner()
        # Two 5-second ticks with 1 prefill + 1 decode worker each on single GPUs.
        state = planner.environment.deployment_state()
        state.prefill.num_gpus = 1
        state.decode.num_gpus = 1

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
        """gpu_hours accumulates internally even when Prometheus export is off."""
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
        """No gauge writes when worker_counts is None (states not yet available)."""
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
    """Build a TickDiagnostics with optional load/throughput decision reasons."""
    return TickDiagnostics(
        load_decision_reason=load_reason,
        throughput_decision_reason=throughput_reason,
    )


def _tick(run_load: bool, run_throughput: bool) -> ScheduledTick:
    """Build a ScheduledTick with the given load/throughput scaling flags."""
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
        """A load-only tick writes the load Enum but leaves the throughput Enum untouched."""
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=True, run_throughput=False),
            _diag(load_reason="scale_up"),
        )

        pm.load_scaling_decision.state.assert_called_once_with("scale_up")
        pm.throughput_scaling_decision.state.assert_not_called()

    def test_scale_down_refused_consolidation_state_registered(self):
        """The new consolidation-refusal reason must be a registered
        Enum state; otherwise ``Enum.state(...)`` raises ValueError and
        crashes the diagnostic publication path on Prometheus-backed
        planners.
        """
        # Mocked Enum's state() doesn't validate, so guard against the
        # registration list drifting away from the diag stamps by checking
        # the constant directly.
        from dynamo.planner.monitoring.planner_metrics import LOAD_DECISION_STATES

        assert "scale_down_refused_consolidation" in LOAD_DECISION_STATES

        # And verify the publication path actually writes it on tick.
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=True, run_throughput=False),
            _diag(load_reason="scale_down_refused_consolidation"),
        )

        pm.load_scaling_decision.state.assert_called_once_with(
            "scale_down_refused_consolidation"
        )

    def test_throughput_only_tick_does_not_touch_load_enum(self):
        """A throughput-only tick writes the throughput Enum but leaves the load Enum untouched."""
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=False, run_throughput=True),
            _diag(throughput_reason="scale"),
        )

        pm.throughput_scaling_decision.state.assert_called_once_with("scale")
        pm.load_scaling_decision.state.assert_not_called()

    def test_combined_tick_publishes_both(self):
        """A tick with both paths enabled writes both Enum gauges."""
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=True, run_throughput=True),
            _diag(load_reason="no_change", throughput_reason="set_lower_bound"),
        )

        pm.load_scaling_decision.state.assert_called_once_with("no_change")
        pm.throughput_scaling_decision.state.assert_called_once_with("set_lower_bound")

    def test_run_tick_with_no_reason_still_writes_unset(self):
        """A scaling path that ran without populating a reason writes 'unset'."""
        # Defensive: if a scaling path ran but didn't populate a reason,
        # explicitly write "unset" so stale state from a prior tick
        # doesn't linger.
        planner = _make_planner()
        pm = planner.prometheus_metrics

        planner._report_diagnostics(_tick(run_load=True, run_throughput=False), _diag())

        pm.load_scaling_decision.state.assert_called_once_with("unset")
        pm.throughput_scaling_decision.state.assert_not_called()

    def test_skipped_when_prometheus_disabled(self):
        """No Enum writes when prometheus_port=0."""
        planner = _make_planner(prometheus_enabled=False)
        pm = planner.prometheus_metrics

        planner._report_diagnostics(
            _tick(run_load=True, run_throughput=True),
            _diag(load_reason="scale_up", throughput_reason="scale"),
        )

        pm.load_scaling_decision.state.assert_not_called()
        pm.throughput_scaling_decision.state.assert_not_called()


# ── SLA target gauges: published once at __init__ ───────────────────


def _make_planner_with_port(
    ttft_ms: float = 500.0,
    itl_ms: float = 50.0,
    prometheus_enabled: bool = True,
) -> NativePlannerBase:
    """Build a NativePlannerBase with prometheus_port set at config time.

    Unlike ``_make_planner``, this helper sets ``metric_reporting_prometheus_port``
    in the PlannerConfig itself (not post-init) so that __init__-time Prometheus
    publication — such as the one-shot SLA target gauge writes — is exercised.
    ``start_http_server`` is still patched to avoid binding a real port.
    """
    port = 9091 if prometheus_enabled else 0
    with (
        patch("dynamo.planner.core.base.PlannerPrometheusMetrics") as mock_metrics,
        patch("dynamo.planner.core.base.start_http_server"),
        patch("dynamo.planner.connectors.kubernetes.KubernetesAPI"),
        patch.dict(os.environ, {"DYN_PARENT_DGD_K8S_NAME": "test-graph"}),
    ):
        mock_metrics.return_value = Mock()
        config = PlannerConfig.model_construct(
            throughput_adjustment_interval_seconds=60,
            prefill_engine_num_gpu=2,
            decode_engine_num_gpu=4,
            min_endpoint=1,
            max_gpu_budget=-1,
            ttft_ms=ttft_ms,
            itl_ms=itl_ms,
            backend="vllm",
            no_operation=True,
            metric_pulling_prometheus_endpoint="http://localhost:9090",
            metric_reporting_prometheus_port=port,
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
            load_metric_samples=10,
            load_min_observations=5,
        )
        planner = NativePlannerBase(None, config, _make_environment(config))
    return planner


class TestSlaTargetGaugesInit:
    """sla_target_ttft_ms and sla_target_itl_ms must be published once
    during NativePlannerBase.__init__ with the values from PlannerConfig."""

    def test_sla_targets_set_at_init_with_default_config_values(self):
        """Default config (ttft_ms=500, itl_ms=50) is pushed to Prometheus on startup."""
        planner = _make_planner_with_port()
        pm = planner.prometheus_metrics

        pm.sla_target_ttft_ms.set.assert_called_once_with(500.0)
        pm.sla_target_itl_ms.set.assert_called_once_with(50.0)

    def test_sla_targets_not_set_when_prometheus_disabled(self):
        """No gauge writes when prometheus_port=0 (Prometheus export disabled)."""
        planner = _make_planner_with_port(prometheus_enabled=False)
        pm = planner.prometheus_metrics

        pm.sla_target_ttft_ms.set.assert_not_called()
        pm.sla_target_itl_ms.set.assert_not_called()

    def test_sla_targets_reflect_custom_config_values(self):
        """Non-default SLA targets from PlannerConfig are passed through unchanged."""
        planner = _make_planner_with_port(ttft_ms=1500.0, itl_ms=75.0)
        pm = planner.prometheus_metrics

        pm.sla_target_ttft_ms.set.assert_called_once_with(1500.0)
        pm.sla_target_itl_ms.set.assert_called_once_with(75.0)


class TestPlannerPrometheusMetricsHasSlaTargetGauges:
    """PlannerPrometheusMetrics must declare sla_target_ttft_ms and
    sla_target_itl_ms as Gauge attributes with the correct metric names."""

    def test_metrics_object_has_sla_target_gauge_attributes(self):
        """Instance attributes sla_target_ttft_ms and sla_target_itl_ms must exist."""
        with (
            patch("dynamo.planner.monitoring.planner_metrics.Gauge") as mock_gauge,
            patch("dynamo.planner.monitoring.planner_metrics.Enum"),
        ):
            mock_gauge.return_value = MagicMock()
            metrics = PlannerPrometheusMetrics()

        assert hasattr(
            metrics, "sla_target_ttft_ms"
        ), "PlannerPrometheusMetrics is missing sla_target_ttft_ms attribute"
        assert hasattr(
            metrics, "sla_target_itl_ms"
        ), "PlannerPrometheusMetrics is missing sla_target_itl_ms attribute"

        registered_names = [call.args[0] for call in mock_gauge.call_args_list]
        assert (
            f"{PREFIX}_sla_target_ttft_ms" in registered_names
        ), f"Expected Gauge name '{PREFIX}_sla_target_ttft_ms' not registered"
        assert (
            f"{PREFIX}_sla_target_itl_ms" in registered_names
        ), f"Expected Gauge name '{PREFIX}_sla_target_itl_ms' not registered"


def _power_planner(
    prefill_num_gpu=2,
    decode_num_gpu=4,
    total_gpu_power_limit=10000.0,
    prefill_cap=300.0,
    decode_cap=250.0,
):
    """A power-aware planner wired for _publish_power_budget_metrics tests."""
    planner = _make_planner(prometheus_enabled=True)
    planner.config.enable_power_awareness = True
    planner.config.total_gpu_power_limit = total_gpu_power_limit
    planner.config.prefill_engine_gpu_power_limit = prefill_cap
    planner.config.decode_engine_gpu_power_limit = decode_cap
    planner.config.prefill_engine_num_gpu = prefill_num_gpu
    planner.config.decode_engine_num_gpu = decode_num_gpu
    # NativePlannerBase defaults these to False; disagg exercises both roles.
    planner.require_prefill = True
    planner.require_decode = True
    return planner


class TestPublishPowerBudgetMetrics:
    """_publish_power_budget_metrics must read the per-replica GPU totals from
    the init-time cache and perform NO apiserver I/O on the tick loop."""

    def test_publish_uses_cached_counts_without_apiserver_call(self):
        """Cached replica totals feed the projection; the connector is untouched."""
        planner = _power_planner()
        # Simulate a 4-node decode replica resolved once at init (8 GPUs × 4).
        planner._power_projection_gpu_counts = (2, 32)
        planner.environment.controller.get_replica_gpu_counts_for_power_projection = (
            Mock()
        )
        pm = planner.prometheus_metrics

        planner._publish_power_budget_metrics(num_p=1, num_d=1)

        # No blocking apiserver GET on the loop.
        planner.environment.controller.get_replica_gpu_counts_for_power_projection.assert_not_called()
        # projected = 1*300*2 + 1*250*32 = 600 + 8000 = 8600
        pm.power_projected_watts.set.assert_called_once_with(8600.0)
        pm.power_budget_total_watts.set.assert_called_once_with(10000.0)
        pm.power_budget_utilization.set.assert_called_once_with(0.86)

    def test_publish_falls_back_to_per_pod_when_cache_unset(self):
        """With no cache resolved, per-pod config counts are used (never None)."""
        planner = _power_planner(prefill_num_gpu=2, decode_num_gpu=4)
        planner._power_projection_gpu_counts = None
        pm = planner.prometheus_metrics

        planner._publish_power_budget_metrics(num_p=1, num_d=1)

        # projected = 1*300*2 + 1*250*4 = 600 + 1000 = 1600
        pm.power_projected_watts.set.assert_called_once_with(1600.0)

    def test_publish_skipped_when_power_awareness_disabled(self):
        """Disabled power awareness publishes nothing (and touches no connector)."""
        planner = _power_planner()
        planner.config.enable_power_awareness = False
        planner._power_projection_gpu_counts = (2, 32)
        planner.environment.controller.get_replica_gpu_counts_for_power_projection = (
            Mock()
        )
        pm = planner.prometheus_metrics

        planner._publish_power_budget_metrics(num_p=1, num_d=1)

        pm.power_projected_watts.set.assert_not_called()
        planner.environment.controller.get_replica_gpu_counts_for_power_projection.assert_not_called()

    def test_zero_gpu_warning_fires_once(self):
        """The zero/partial projection warning is emitted at most once."""
        planner = _power_planner(prefill_num_gpu=0, decode_num_gpu=0)
        planner._power_projection_gpu_counts = (0, 0)

        with patch("dynamo.planner.core.base.logger") as mock_logger:
            planner._publish_power_budget_metrics(num_p=1, num_d=1)
            planner._publish_power_budget_metrics(num_p=1, num_d=1)

        assert mock_logger.warning.call_count == 1
        assert planner._power_projected_zero_warned is True


class TestResolvePowerProjectionGpuCounts:
    """_resolve_power_projection_gpu_counts caches per-replica totals once and
    falls back to per-pod counts (not a per-tick unit flip) on failure."""

    def test_caches_replica_total_from_connector(self):
        """A successful resolve stores the replica-total tuple."""
        planner = _power_planner()
        planner.environment.controller.get_replica_gpu_counts_for_power_projection = (
            Mock(return_value=(2, 32))
        )

        planner._resolve_power_projection_gpu_counts()

        assert planner._power_projection_gpu_counts == (2, 32)

    def test_falls_back_to_per_pod_counts_on_operational_error(self):
        """An operational resolution failure is logged; per-pod counts are cached."""
        planner = _power_planner(prefill_num_gpu=2, decode_num_gpu=4)
        planner.environment.controller.get_replica_gpu_counts_for_power_projection = (
            Mock(
                side_effect=DeploymentValidationError(["decode GPU count unavailable"])
            )
        )

        with patch("dynamo.planner.core.base.logger") as mock_logger:
            planner._resolve_power_projection_gpu_counts()

        assert planner._power_projection_gpu_counts == (2, 4)
        mock_logger.warning.assert_called_once()

    def test_unexpected_error_propagates(self):
        """Programming errors must abort init, not become a silent fallback."""
        planner = _power_planner()
        planner.environment.controller.get_replica_gpu_counts_for_power_projection = (
            Mock(side_effect=RuntimeError("implementation bug"))
        )

        with pytest.raises(RuntimeError, match="implementation bug"):
            planner._resolve_power_projection_gpu_counts()

    def test_forwards_component_names_from_deployment_state(self):
        """Component names from DeploymentState are forwarded to the connector.

        Ensures that k8s_name values resolved during environment initialisation
        reach get_replica_gpu_counts_for_power_projection as explicit overrides,
        allowing the connector to resolve ambiguous workers by name rather than
        falling back to role-based lookup.
        """
        planner = _power_planner()
        state = planner.environment.deployment_state()
        state.prefill.info = WorkerInfo(k8s_name="CustomPrefillWorker")
        state.decode.info = WorkerInfo(k8s_name="CustomDecodeWorker")
        mock_fn = Mock(return_value=(4, 8))
        planner.environment.controller.get_replica_gpu_counts_for_power_projection = (
            mock_fn
        )

        planner._resolve_power_projection_gpu_counts()

        mock_fn.assert_called_once_with(
            require_prefill=True,
            require_decode=True,
            prefill_component_name="CustomPrefillWorker",
            decode_component_name="CustomDecodeWorker",
        )
        assert planner._power_projection_gpu_counts == (4, 8)
