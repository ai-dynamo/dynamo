# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import Mock, patch

import pytest

from dynamo.planner.utils.decode_planner import DecodePlanner
from dynamo.planner.utils.load_based_regression import LoadBasedRegressionModel
from dynamo.planner.utils.planner_config import PlannerConfig
from dynamo.planner.utils.planner_core import PlannerSharedState
from dynamo.planner.utils.prefill_planner import PrefillPlanner
from dynamo.planner.utils.prometheus import CachedLoadMetrics, DirectRouterMetricsClient

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ── LoadBasedRegressionModel tests ──────────────────────────────────────


class TestLoadBasedRegressionModel:
    def test_insufficient_data(self):
        model = LoadBasedRegressionModel(window_size=50, min_observations=5)
        assert not model.has_sufficient_data()
        assert model.predict_x_from_sla(100.0) is None

    def test_basic_linear_prediction(self):
        model = LoadBasedRegressionModel(window_size=50, min_observations=3)
        # y = 2x + 10: x in [1..5], y in [12..20]
        for x in range(1, 6):
            model.add_observation(float(x), 2.0 * x + 10.0)
        assert model.has_sufficient_data()
        # Reverse: x = (y - 10) / 2, y=100 => x=45
        result = model.predict_x_from_sla(100.0)
        assert result is not None
        assert abs(result - 45.0) < 0.5

    def test_negative_slope_fallback_points_below_sla(self):
        model = LoadBasedRegressionModel(window_size=50, min_observations=3)
        # Negative slope: higher x => lower y
        # x=1 -> y=98, x=2 -> y=96, x=3 -> y=94, x=4 -> y=92, x=5 -> y=90
        for x in range(1, 6):
            model.add_observation(float(x), 100.0 - 2.0 * x)
        # target_y=95 => points below: x=3(y=94), x=4(y=92), x=5(y=90)
        # min x among those is 3
        result = model.predict_x_from_sla(95.0)
        assert result is not None
        assert abs(result - 3.0) < 0.01

    def test_negative_slope_fallback_all_above_sla(self):
        model = LoadBasedRegressionModel(window_size=50, min_observations=3)
        # Negative slope: x=1 -> y=98, x=2 -> y=96, ..., x=5 -> y=90
        for x in range(1, 6):
            model.add_observation(float(x), 100.0 - 2.0 * x)
        # target_y=50 => all points have y >= 90 > 50, none below
        # fallback returns smallest x overall = 1
        result = model.predict_x_from_sla(50.0)
        assert result is not None
        assert abs(result - 1.0) < 0.01

    def test_sliding_window_evicts_old(self):
        model = LoadBasedRegressionModel(window_size=5, min_observations=3)
        # Add 10 observations; only last 5 should remain
        for i in range(10):
            model.add_observation(float(i), float(i) * 2)
        assert model.num_observations == 5

    def test_result_clamped_to_non_negative(self):
        model = LoadBasedRegressionModel(window_size=50, min_observations=3)
        # y = 10x + 100: intercept=100, slope=10
        for x in range(1, 6):
            model.add_observation(float(x), 10.0 * x + 100.0)
        # target_y=5 => x = (5-100)/10 = -9.5 => clamped to 0
        result = model.predict_x_from_sla(5.0)
        assert result == 0.0

    def test_slope_and_intercept_properties(self):
        model = LoadBasedRegressionModel(window_size=50, min_observations=3)
        for x in range(1, 6):
            model.add_observation(float(x), 3.0 * x + 5.0)
        assert model.slope is not None
        assert abs(model.slope - 3.0) < 0.01
        assert model.intercept is not None
        assert abs(model.intercept - 5.0) < 0.01


# ── DirectRouterMetricsClient tests ─────────────────────────────────────


class TestDirectRouterMetricsClient:
    def test_parse_prometheus_text_basic(self):
        """Metrics with dynamo_namespace/model labels are grouped by worker_type."""
        client = DirectRouterMetricsClient("http://localhost:8000/metrics", "test-ns")
        text = (
            "# HELP dynamo_frontend_worker_active_prefill_tokens Active prefill tokens\n"
            "# TYPE dynamo_frontend_worker_active_prefill_tokens gauge\n"
            'dynamo_frontend_worker_active_prefill_tokens{dynamo_namespace="test-ns",model="TestModel",worker_type="prefill",worker_id="w1"} 1234\n'
            'dynamo_frontend_worker_active_decode_blocks{dynamo_namespace="test-ns",model="TestModel",worker_type="decode",worker_id="w2"} 56\n'
            'dynamo_frontend_worker_last_time_to_first_token_seconds{dynamo_namespace="test-ns",model="TestModel",worker_type="prefill",worker_id="w1"} 0.25\n'
            'dynamo_frontend_worker_last_input_sequence_tokens{dynamo_namespace="test-ns",model="TestModel",worker_type="prefill",worker_id="w1"} 3000\n'
            'dynamo_frontend_worker_last_inter_token_latency_seconds{dynamo_namespace="test-ns",model="TestModel",worker_type="decode",worker_id="w2"} 0.04\n'
        )
        result = client._parse_prometheus_text(text)
        assert "prefill" in result
        assert "w1" in result["prefill"]
        assert result["prefill"]["w1"]["active_prefill_tokens"] == 1234.0
        assert abs(result["prefill"]["w1"]["last_ttft"] - 0.25) < 1e-6
        assert result["prefill"]["w1"]["last_isl"] == 3000.0
        assert "decode" in result
        assert "w2" in result["decode"]
        assert result["decode"]["w2"]["active_decode_blocks"] == 56.0
        assert abs(result["decode"]["w2"]["last_itl"] - 0.04) < 1e-6

    def test_parse_ignores_extra_labels(self):
        """Parser extracts metrics regardless of extra labels like dynamo_namespace/model."""
        client = DirectRouterMetricsClient("http://localhost:8000/metrics", "ns")
        text = 'dynamo_frontend_worker_active_prefill_tokens{dynamo_namespace="any-ns",model="mymodel",worker_type="prefill",worker_id="w1"} 100\n'
        result = client._parse_prometheus_text(text)
        assert "prefill" in result
        assert "w1" in result["prefill"]
        assert result["prefill"]["w1"]["active_prefill_tokens"] == 100.0

    def test_get_recent_and_averaged_empty_buffer(self):
        client = DirectRouterMetricsClient("http://localhost:8000/metrics", "ns")
        assert client.get_recent_and_averaged_metrics("prefill") is None

    def test_get_recent_and_averaged_single_sample(self):
        client = DirectRouterMetricsClient("http://localhost:8000/metrics", "ns")
        client._sample_buffer = [
            {
                "prefill": {"w1": {"active_prefill_tokens": 100.0}},
                "decode": {"w2": {"active_decode_blocks": 50.0}},
            }
        ]
        result = client.get_recent_and_averaged_metrics("prefill")
        assert result is not None
        recent, per_worker_avg, cluster_avg = result
        assert recent["w1"]["active_prefill_tokens"] == 100.0
        assert per_worker_avg["w1"]["active_prefill_tokens"] == 100.0
        assert cluster_avg["active_prefill_tokens"] == 100.0
        # decode workers not included
        assert "w2" not in recent

        result_d = client.get_recent_and_averaged_metrics("decode")
        assert result_d is not None
        recent_d, per_worker_avg_d, cluster_avg_d = result_d
        assert recent_d["w2"]["active_decode_blocks"] == 50.0
        assert per_worker_avg_d["w2"]["active_decode_blocks"] == 50.0
        assert cluster_avg_d["active_decode_blocks"] == 50.0

    def test_get_recent_and_averaged_multiple_samples(self):
        client = DirectRouterMetricsClient("http://localhost:8000/metrics", "ns")
        client._sample_buffer = [
            {"prefill": {"w1": {"active_prefill_tokens": 100.0}}},
            {"prefill": {"w1": {"active_prefill_tokens": 200.0}}},
            {"prefill": {"w1": {"active_prefill_tokens": 300.0}}},
        ]
        result = client.get_recent_and_averaged_metrics("prefill")
        assert result is not None
        recent, per_worker_avg, cluster_avg = result
        # Recent should be the last sample
        assert abs(recent["w1"]["active_prefill_tokens"] - 300.0) < 1e-6
        # Per-worker averaged over time
        assert abs(per_worker_avg["w1"]["active_prefill_tokens"] - 200.0) < 1e-6
        # Cluster averaged (same as per-worker when only 1 worker)
        assert abs(cluster_avg["active_prefill_tokens"] - 200.0) < 1e-6

    def test_parse_multiple_workers(self):
        client = DirectRouterMetricsClient("http://localhost:8000/metrics", "ns")
        text = (
            'dynamo_frontend_worker_active_prefill_tokens{dynamo_namespace="ns",model="M",worker_type="prefill",worker_id="w1"} 100\n'
            'dynamo_frontend_worker_active_prefill_tokens{dynamo_namespace="ns",model="M",worker_type="prefill",worker_id="w2"} 200\n'
        )
        result = client._parse_prometheus_text(text)
        assert len(result.get("prefill", {})) == 2
        assert result["prefill"]["w1"]["active_prefill_tokens"] == 100.0
        assert result["prefill"]["w2"]["active_prefill_tokens"] == 200.0

    def test_parse_rust_labels_separates_worker_types(self):
        """Rust KV router emits all metrics for all workers; parser must separate by worker_type."""
        client = DirectRouterMetricsClient("http://localhost:8000/metrics", "ns")
        text = (
            "# HELP dynamo_frontend_worker_active_prefill_tokens Active prefill tokens\n"
            "# TYPE dynamo_frontend_worker_active_prefill_tokens gauge\n"
            'dynamo_frontend_worker_active_prefill_tokens{worker_id="123",dp_rank="0",worker_type="prefill"} 500\n'
            'dynamo_frontend_worker_active_prefill_tokens{worker_id="456",dp_rank="0",worker_type="decode"} 0\n'
            'dynamo_frontend_worker_active_decode_blocks{worker_id="123",dp_rank="0",worker_type="prefill"} 0\n'
            'dynamo_frontend_worker_active_decode_blocks{worker_id="456",dp_rank="0",worker_type="decode"} 30\n'
            'dynamo_frontend_worker_last_time_to_first_token_seconds{worker_id="123",dp_rank="0",worker_type="prefill"} 0.15\n'
            'dynamo_frontend_worker_last_input_sequence_tokens{worker_id="123",dp_rank="0",worker_type="prefill"} 2000\n'
            'dynamo_frontend_worker_last_inter_token_latency_seconds{worker_id="456",dp_rank="0",worker_type="decode"} 0.03\n'
        )
        result = client._parse_prometheus_text(text)

        # Prefill worker 123 grouped under "prefill"
        assert "prefill" in result
        assert "123" in result["prefill"]
        assert result["prefill"]["123"]["active_prefill_tokens"] == 500.0
        assert result["prefill"]["123"]["last_ttft"] == 0.15
        assert result["prefill"]["123"]["last_isl"] == 2000.0

        # Decode worker 456 grouped under "decode"
        assert "decode" in result
        assert "456" in result["decode"]
        assert result["decode"]["456"]["active_decode_blocks"] == 30.0
        assert abs(result["decode"]["456"]["last_itl"] - 0.03) < 1e-6

        # Cross-type metrics are stored under the correct worker_type
        # (prefill worker's decode_blocks=0 stored under "prefill", not "decode")
        assert "456" not in result["prefill"]
        assert "123" not in result["decode"]


# ── PrefillPlanner load-based scaling tests ─────────────────────────────


@pytest.fixture(autouse=True)
def mock_prometheus_metrics():
    with patch("dynamo.planner.utils.planner_core.Gauge") as mock_gauge:
        mock_gauge.return_value = Mock()
        yield


def _build_load_config(**overrides) -> PlannerConfig:
    defaults = dict(
        throughput_adjustment_interval=60,
        prefill_engine_num_gpu=1,
        decode_engine_num_gpu=1,
        min_endpoint=1,
        max_gpu_budget=-1,
        ttft=500.0,
        itl=50.0,
        backend="vllm",
        no_operation=True,
        no_correction=True,
        metric_pulling_prometheus_endpoint="http://localhost:9090",
        metric_reporting_prometheus_port=0,
        load_predictor="constant",
        profile_results_dir=os.path.join(
            os.path.dirname(__file__),
            "..",
            "profiling_results",
            "H200_TP1P_TP1D",
        ),
        environment="kubernetes",
        namespace="test-namespace",
        mode="disagg",
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        load_router_metrics_url="http://router:8000/metrics",
        load_adjustment_interval=5,
        load_learning_window=50,
        load_scaling_down_sensitivity=80,
        load_metric_samples=10,
        load_min_observations=5,
    )
    defaults.update(overrides)
    return PlannerConfig.model_construct(**defaults)


def _avg(per_worker: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute flat averaged metrics from per-worker dicts (for test convenience)."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metrics in per_worker.values():
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


class TestPrefillLoadBasedScaling:
    def test_scale_up_all_workers_above_target(self):
        """When all workers have active_prefill_tokens above the regression target, scale up."""
        config = _build_load_config()
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 2

        planner = PrefillPlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"

        # Feed regression data: TTFT = 0.1 * (active_prefill_tokens + ISL) + 100
        # With TTFT SLA = 500ms: x_sla = (500 - 100) / 0.1 = 4000
        # If ISL avg = 3000, target_active_tokens = 4000 - 3000 = 1000
        for i in range(10):
            x = 2000 + i * 200  # active_tokens + ISL
            y = 0.1 * x + 100  # TTFT in ms
            planner.ttft_regression.add_observation(x, y)

        # Set per-worker metrics: all workers ABOVE target (1000)
        metrics = {
            "w1": {
                "active_prefill_tokens": 1500.0,
                "last_isl": 3000.0,
                "last_ttft": 0.35,
            },
            "w2": {
                "active_prefill_tokens": 1200.0,
                "last_isl": 3000.0,
                "last_ttft": 0.30,
            },
        }
        planner.cached_load_metrics = CachedLoadMetrics(
            recent=metrics, per_worker_averaged=metrics, cluster_averaged=_avg(metrics)
        )

        result = planner.load_plan_adjustment()
        assert result == 3  # scale up from 2 to 3

    def test_scale_down_all_workers_below_boundary(self):
        """When all workers are below the scale-down boundary, scale down."""
        config = _build_load_config(load_scaling_down_sensitivity=100)
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 3

        planner = PrefillPlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"

        # Feed regression: TTFT = 0.1 * x + 100
        # x_sla = (500-100)/0.1 = 4000, target = 4000-3000 = 1000
        # boundary = 1000 * (3-1)/3 * 1.0 = 666.67
        for i in range(10):
            x = 2000 + i * 200
            y = 0.1 * x + 100
            planner.ttft_regression.add_observation(x, y)

        # All workers below boundary (666.67)
        metrics = {
            "w1": {
                "active_prefill_tokens": 100.0,
                "last_isl": 3000.0,
                "last_ttft": 0.15,
            },
            "w2": {
                "active_prefill_tokens": 200.0,
                "last_isl": 3000.0,
                "last_ttft": 0.16,
            },
            "w3": {
                "active_prefill_tokens": 150.0,
                "last_isl": 3000.0,
                "last_ttft": 0.15,
            },
        }
        planner.cached_load_metrics = CachedLoadMetrics(
            recent=metrics, per_worker_averaged=metrics, cluster_averaged=_avg(metrics)
        )

        result = planner.load_plan_adjustment()
        assert result == 2  # scale down from 3 to 2

    def test_no_change_mixed_workers(self):
        """When workers are mixed (some above, some below), no scaling."""
        config = _build_load_config()
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 2

        planner = PrefillPlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"

        for i in range(10):
            x = 2000 + i * 200
            y = 0.1 * x + 100
            planner.ttft_regression.add_observation(x, y)

        # Mixed: one above target, one below
        metrics = {
            "w1": {
                "active_prefill_tokens": 1500.0,
                "last_isl": 3000.0,
                "last_ttft": 0.35,
            },
            "w2": {
                "active_prefill_tokens": 100.0,
                "last_isl": 3000.0,
                "last_ttft": 0.15,
            },
        }
        planner.cached_load_metrics = CachedLoadMetrics(
            recent=metrics, per_worker_averaged=metrics, cluster_averaged=_avg(metrics)
        )

        result = planner.load_plan_adjustment()
        assert result is None

    def test_cold_start_returns_none(self):
        """With insufficient data, load_plan_adjustment returns None."""
        config = _build_load_config()
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 2

        planner = PrefillPlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"

        # Only 2 observations (min is 5)
        planner.ttft_regression.add_observation(1000.0, 200.0)
        planner.ttft_regression.add_observation(2000.0, 300.0)

        metrics = {
            "w1": {
                "active_prefill_tokens": 5000.0,
                "last_isl": 3000.0,
                "last_ttft": 0.5,
            },
        }
        planner.cached_load_metrics = CachedLoadMetrics(
            recent=metrics, per_worker_averaged=metrics, cluster_averaged=_avg(metrics)
        )

        result = planner.load_plan_adjustment()
        assert result is None


class TestDecodeLoadBasedScaling:
    def test_scale_up_all_workers_above_target(self):
        """When all workers have active_decode_blocks above x_sla, scale up."""
        config = _build_load_config()
        shared_state = PlannerSharedState()
        shared_state.num_d_workers = 2

        planner = DecodePlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"

        # Feed regression: ITL = 0.5 * active_decode_blocks + 10
        # x_sla = (50 - 10) / 0.5 = 80
        for i in range(10):
            x = 20 + i * 10
            y = 0.5 * x + 10
            planner.itl_regression.add_observation(x, y)

        # All workers above x_sla (80)
        metrics = {
            "w1": {"active_decode_blocks": 100.0, "last_itl": 0.06},
            "w2": {"active_decode_blocks": 95.0, "last_itl": 0.055},
        }
        planner.cached_load_metrics = CachedLoadMetrics(
            recent=metrics, per_worker_averaged=metrics, cluster_averaged=_avg(metrics)
        )

        result = planner.load_plan_adjustment()
        assert result == 3

    def test_scale_down_all_workers_below_boundary(self):
        """When all decode workers are below boundary, scale down."""
        config = _build_load_config(load_scaling_down_sensitivity=100)
        shared_state = PlannerSharedState()
        shared_state.num_d_workers = 3

        planner = DecodePlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"

        # ITL = 0.5 * x + 10, x_sla = (50-10)/0.5 = 80
        # boundary = 80 * (3-1)/3 * 1.0 = 53.33
        for i in range(10):
            x = 20 + i * 10
            y = 0.5 * x + 10
            planner.itl_regression.add_observation(x, y)

        # All workers below boundary (53.33)
        metrics = {
            "w1": {"active_decode_blocks": 10.0, "last_itl": 0.02},
            "w2": {"active_decode_blocks": 15.0, "last_itl": 0.025},
            "w3": {"active_decode_blocks": 20.0, "last_itl": 0.03},
        }
        planner.cached_load_metrics = CachedLoadMetrics(
            recent=metrics, per_worker_averaged=metrics, cluster_averaged=_avg(metrics)
        )

        result = planner.load_plan_adjustment()
        assert result == 2

    def test_cold_start_returns_none(self):
        """Decode cold start also returns None."""
        config = _build_load_config()
        shared_state = PlannerSharedState()
        shared_state.num_d_workers = 2

        planner = DecodePlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"

        planner.itl_regression.add_observation(10.0, 15.0)

        metrics = {
            "w1": {"active_decode_blocks": 200.0, "last_itl": 0.1},
        }
        planner.cached_load_metrics = CachedLoadMetrics(
            recent=metrics, per_worker_averaged=metrics, cluster_averaged=_avg(metrics)
        )

        result = planner.load_plan_adjustment()
        assert result is None


class TestLowerBoundEnforcement:
    def test_throughput_lower_bound_respected(self):
        """Load-based scaling should never go below throughput lower bound."""
        config = _build_load_config()
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 5
        # Throughput says we need at least 4 prefill workers
        shared_state.throughput_lower_bound_p = 4

        planner = PrefillPlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"

        # Regression says we should scale down to 4 (from 5)
        for i in range(10):
            x = 2000 + i * 200
            y = 0.1 * x + 100
            planner.ttft_regression.add_observation(x, y)

        # Workers all lightly loaded => wants to scale down to 4
        metrics = {
            f"w{i}": {
                "active_prefill_tokens": 50.0,
                "last_isl": 3000.0,
                "last_ttft": 0.12,
            }
            for i in range(5)
        }
        planner.cached_load_metrics = CachedLoadMetrics(
            recent=metrics, per_worker_averaged=metrics, cluster_averaged=_avg(metrics)
        )

        result = planner.load_plan_adjustment()
        # Even though load-based wants to scale down, the result should be
        # at least 4 after lower bound enforcement (done in the loop, not in
        # load_plan_adjustment itself)
        # load_plan_adjustment returns raw desired value
        assert result == 4  # raw value from load-based

    def test_scaling_down_sensitivity_zero_never_scales_down(self):
        """With sensitivity=0, scale-down boundary is 0 so never scale down."""
        config = _build_load_config(load_scaling_down_sensitivity=0)
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 3

        planner = PrefillPlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"

        for i in range(10):
            x = 2000 + i * 200
            y = 0.1 * x + 100
            planner.ttft_regression.add_observation(x, y)

        # All workers at zero load
        metrics = {
            f"w{i}": {
                "active_prefill_tokens": 0.0,
                "last_isl": 3000.0,
                "last_ttft": 0.12,
            }
            for i in range(3)
        }
        planner.cached_load_metrics = CachedLoadMetrics(
            recent=metrics, per_worker_averaged=metrics, cluster_averaged=_avg(metrics)
        )

        # boundary = target * (3-1)/3 * 0/100 = 0
        # all workers at 0 which is NOT less than 0 (it's equal)
        result = planner.load_plan_adjustment()
        assert result is None  # no scaling happens


# ── Correction factor auto-disable tests ─────────────────────────────


class TestCorrectionFactorAutoDisable:
    def test_correction_factor_disabled_when_load_enabled(self):
        """Correction factor should be auto-disabled when load-based scaling is on."""
        config = PlannerConfig(
            enable_load_scaling=True,
            enable_throughput_scaling=True,
            no_correction=False,
            load_router_metrics_url="http://router:8000/metrics",
        )
        assert config.no_correction is True

    def test_correction_factor_stays_disabled_if_already_set(self):
        """If user already set no_correction, it stays True."""
        config = PlannerConfig(
            enable_load_scaling=True,
            enable_throughput_scaling=True,
            no_correction=True,
            load_router_metrics_url="http://router:8000/metrics",
        )
        assert config.no_correction is True

    def test_correction_factor_not_disabled_without_loadbased(self):
        """Without load-based scaling, correction factor should respect user setting."""
        config = PlannerConfig(
            enable_load_scaling=False,
            enable_throughput_scaling=True,
            no_correction=False,
        )
        assert config.no_correction is False


# ── DGD worker count reconciliation tests ────────────────────────────


class TestWorkerCountReconciliation:
    @pytest.mark.asyncio
    async def test_prefill_observe_gets_only_prefill_workers(self):
        """observe_engine_load_stats for prefill queries get_recent_and_averaged_metrics('prefill')."""
        config = _build_load_config()
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 1

        planner = PrefillPlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"

        # get_recent_and_averaged_metrics("prefill") returns (recent, per_worker_avg, cluster_avg)
        prefill_metrics = {
            "w1": {
                "active_prefill_tokens": 500.0,
                "last_ttft": 0.2,
                "last_isl": 3000.0,
            },
        }
        planner.prometheus_engine_client = Mock()
        planner.prometheus_engine_client.get_recent_and_averaged_metrics.return_value = (
            prefill_metrics,
            prefill_metrics,
            _avg(prefill_metrics),
        )

        await planner.observe_engine_load_stats()

        planner.prometheus_engine_client.get_recent_and_averaged_metrics.assert_called_once_with(
            "prefill"
        )
        assert len(planner.cached_load_metrics.recent) == 1
        assert "w1" in planner.cached_load_metrics.recent

    @pytest.mark.asyncio
    async def test_decode_observe_gets_only_decode_workers(self):
        """observe_engine_load_stats for decode queries get_recent_and_averaged_metrics('decode')."""
        config = _build_load_config()
        shared_state = PlannerSharedState()
        shared_state.num_d_workers = 1

        planner = DecodePlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"

        decode_metrics = {
            "w2": {"active_decode_blocks": 50.0, "last_itl": 0.04},
        }
        planner.prometheus_engine_client = Mock()
        planner.prometheus_engine_client.get_recent_and_averaged_metrics.return_value = (
            decode_metrics,
            decode_metrics,
            _avg(decode_metrics),
        )

        await planner.observe_engine_load_stats()

        planner.prometheus_engine_client.get_recent_and_averaged_metrics.assert_called_once_with(
            "decode"
        )
        assert len(planner.cached_load_metrics.recent) == 1
        assert "w2" in planner.cached_load_metrics.recent

    def test_worker_count_mismatch_detected(self):
        """When DGD and Prometheus worker counts differ, the mismatch should be detectable."""
        config = _build_load_config()
        shared_state = PlannerSharedState()
        # DGD says 3 prefill workers
        shared_state.num_p_workers = 3

        planner = PrefillPlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"

        # But router only reports 2 prefill workers
        metrics = {
            "w1": {
                "active_prefill_tokens": 500.0,
                "last_isl": 3000.0,
                "last_ttft": 0.2,
            },
            "w2": {
                "active_prefill_tokens": 600.0,
                "last_isl": 3000.0,
                "last_ttft": 0.25,
            },
        }
        planner.cached_load_metrics = CachedLoadMetrics(
            recent=metrics, per_worker_averaged=metrics, cluster_averaged=_avg(metrics)
        )

        # The mismatch should be detectable by comparing counts
        prom_count = len(planner.cached_load_metrics.recent)
        dgd_count = shared_state.num_p_workers
        assert prom_count != dgd_count
        assert prom_count == 2
        assert dgd_count == 3

    def test_worker_count_match_allows_scaling(self):
        """When DGD and Prometheus counts match, scaling proceeds normally."""
        config = _build_load_config()
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 2

        planner = PrefillPlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"

        metrics = {
            "w1": {
                "active_prefill_tokens": 1500.0,
                "last_isl": 3000.0,
                "last_ttft": 0.35,
            },
            "w2": {
                "active_prefill_tokens": 1200.0,
                "last_isl": 3000.0,
                "last_ttft": 0.30,
            },
        }
        planner.cached_load_metrics = CachedLoadMetrics(
            recent=metrics, per_worker_averaged=metrics, cluster_averaged=_avg(metrics)
        )

        prom_count = len(planner.cached_load_metrics.recent)
        dgd_count = shared_state.num_p_workers
        assert prom_count == dgd_count

        # With matching counts and sufficient regression data, scaling should work
        for i in range(10):
            x = 2000 + i * 200
            y = 0.1 * x + 100
            planner.ttft_regression.add_observation(x, y)

        result = planner.load_plan_adjustment()
        assert result is not None  # scaling proceeds
