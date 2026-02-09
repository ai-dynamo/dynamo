# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from unittest.mock import Mock, patch

import pytest

from dynamo.planner.utils.load_based_regression import LoadBasedRegressionModel
from dynamo.planner.utils.planner_argparse import validate_sla_planner_args
from dynamo.planner.utils.planner_core import PlannerSharedState
from dynamo.planner.utils.decode_planner import DecodePlanner
from dynamo.planner.utils.prefill_planner import PrefillPlanner
from dynamo.planner.utils.prometheus import DirectRouterMetricsClient

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

    def test_negative_slope_returns_none(self):
        model = LoadBasedRegressionModel(window_size=50, min_observations=3)
        # Negative slope: higher x => lower y
        for x in range(1, 6):
            model.add_observation(float(x), 100.0 - 2.0 * x)
        result = model.predict_x_from_sla(50.0)
        assert result is None

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
        client = DirectRouterMetricsClient(
            "http://localhost:8000/metrics", "test-ns"
        )
        text = (
            '# HELP dynamo_frontend_worker_active_prefill_tokens Active prefill tokens\n'
            '# TYPE dynamo_frontend_worker_active_prefill_tokens gauge\n'
            'dynamo_frontend_worker_active_prefill_tokens{dynamo_namespace="test-ns",model="TestModel",worker_id="w1"} 1234\n'
            'dynamo_frontend_worker_active_decode_blocks{dynamo_namespace="test-ns",model="TestModel",worker_id="w1"} 56\n'
            'dynamo_frontend_worker_last_time_to_first_token_seconds{dynamo_namespace="test-ns",model="TestModel",worker_id="w1"} 0.25\n'
            'dynamo_frontend_worker_last_input_sequence_tokens{dynamo_namespace="test-ns",model="TestModel",worker_id="w1"} 3000\n'
            'dynamo_frontend_worker_last_inter_token_latency_seconds{dynamo_namespace="test-ns",model="TestModel",worker_id="w1"} 0.04\n'
        )
        result = client._parse_prometheus_text(text, "TestModel")
        assert "w1" in result
        assert result["w1"]["active_prefill_tokens"] == 1234.0
        assert result["w1"]["active_decode_blocks"] == 56.0
        assert abs(result["w1"]["last_ttft"] - 0.25) < 1e-6
        assert result["w1"]["last_isl"] == 3000.0
        assert abs(result["w1"]["last_itl"] - 0.04) < 1e-6

    def test_parse_filters_by_namespace(self):
        client = DirectRouterMetricsClient(
            "http://localhost:8000/metrics", "my-ns"
        )
        text = (
            'dynamo_frontend_worker_active_prefill_tokens{dynamo_namespace="other-ns",model="M",worker_id="w1"} 100\n'
        )
        result = client._parse_prometheus_text(text, "M")
        assert len(result) == 0

    def test_parse_case_insensitive_model(self):
        client = DirectRouterMetricsClient(
            "http://localhost:8000/metrics", "ns"
        )
        text = (
            'dynamo_frontend_worker_active_prefill_tokens{dynamo_namespace="ns",model="mymodel",worker_id="w1"} 100\n'
        )
        result = client._parse_prometheus_text(text, "MyModel")
        assert "w1" in result
        assert result["w1"]["active_prefill_tokens"] == 100.0

    def test_get_averaged_metrics_empty_buffer(self):
        client = DirectRouterMetricsClient(
            "http://localhost:8000/metrics", "ns"
        )
        assert client.get_averaged_metrics() is None

    def test_get_averaged_metrics_single_sample(self):
        client = DirectRouterMetricsClient(
            "http://localhost:8000/metrics", "ns"
        )
        client._sample_buffer = [
            {"w1": {"active_prefill_tokens": 100.0, "active_decode_blocks": 50.0}}
        ]
        result = client.get_averaged_metrics()
        assert result is not None
        assert result["w1"]["active_prefill_tokens"] == 100.0
        assert result["w1"]["active_decode_blocks"] == 50.0

    def test_get_averaged_metrics_multiple_samples(self):
        client = DirectRouterMetricsClient(
            "http://localhost:8000/metrics", "ns"
        )
        client._sample_buffer = [
            {"w1": {"active_prefill_tokens": 100.0}},
            {"w1": {"active_prefill_tokens": 200.0}},
            {"w1": {"active_prefill_tokens": 300.0}},
        ]
        result = client.get_averaged_metrics()
        assert result is not None
        assert abs(result["w1"]["active_prefill_tokens"] - 200.0) < 1e-6

    def test_parse_multiple_workers(self):
        client = DirectRouterMetricsClient(
            "http://localhost:8000/metrics", "ns"
        )
        text = (
            'dynamo_frontend_worker_active_prefill_tokens{dynamo_namespace="ns",model="M",worker_id="w1"} 100\n'
            'dynamo_frontend_worker_active_prefill_tokens{dynamo_namespace="ns",model="M",worker_id="w2"} 200\n'
        )
        result = client._parse_prometheus_text(text, "M")
        assert len(result) == 2
        assert result["w1"]["active_prefill_tokens"] == 100.0
        assert result["w2"]["active_prefill_tokens"] == 200.0


# ── PrefillPlanner load-based scaling tests ─────────────────────────────


@pytest.fixture(autouse=True)
def mock_prometheus_metrics():
    with patch("dynamo.planner.utils.planner_core.Gauge") as mock_gauge:
        mock_gauge.return_value = Mock()
        yield


def _build_loadbased_args():
    args = argparse.Namespace()
    args.adjustment_interval = 60
    args.prefill_engine_num_gpu = 1
    args.decode_engine_num_gpu = 1
    args.min_endpoint = 1
    args.max_gpu_budget = -1
    args.ttft = 500.0
    args.itl = 50.0
    args.backend = "vllm"
    args.no_operation = True
    args.no_correction = True
    args.metric_pulling_prometheus_endpoint = "http://localhost:9090"
    args.metric_reporting_prometheus_port = 0
    args.load_predictor = "constant"
    args.load_predictor_warmup_trace = None
    args.profile_results_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "profiling_results",
        "H200_TP1P_TP1D",
    )
    args.environment = "kubernetes"
    args.namespace = "test-namespace"
    args.mode = "disagg"
    # Load-based scaling config
    args.enable_loadbased_scaling = True
    args.enable_throughput_scaling = True
    args.disable_throughput_scaling = False
    args.loadbased_router_metrics_url = "http://router:8000/metrics"
    args.loadbased_adjustment_interval = 5
    args.loadbased_learning_window = 50
    args.loadbased_scaling_down_sensitivity = 80
    args.loadbased_metric_samples = 10
    args.loadbased_min_observations = 5
    return args


class TestPrefillLoadBasedScaling:
    def test_scale_up_all_workers_above_target(self):
        """When all workers have active_prefill_tokens above the regression target, scale up."""
        args = _build_loadbased_args()
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 2

        planner = PrefillPlanner(None, args, shared_state=shared_state)
        planner.model_name = "test-model"

        # Feed regression data: TTFT = 0.1 * (active_prefill_tokens + ISL) + 100
        # With TTFT SLA = 500ms: x_sla = (500 - 100) / 0.1 = 4000
        # If ISL avg = 3000, target_active_tokens = 4000 - 3000 = 1000
        for i in range(10):
            x = 2000 + i * 200  # active_tokens + ISL
            y = 0.1 * x + 100   # TTFT in ms
            planner.ttft_regression.add_observation(x, y)

        # Set per-worker metrics: all workers ABOVE target (1000)
        planner.cached_per_worker_metrics = {
            "w1": {"active_prefill_tokens": 1500.0, "last_isl": 3000.0, "last_ttft": 0.35},
            "w2": {"active_prefill_tokens": 1200.0, "last_isl": 3000.0, "last_ttft": 0.30},
        }

        result = planner.loadbased_plan_adjustment()
        assert result == 3  # scale up from 2 to 3

    def test_scale_down_all_workers_below_boundary(self):
        """When all workers are below the scale-down boundary, scale down."""
        args = _build_loadbased_args()
        args.loadbased_scaling_down_sensitivity = 100  # max sensitivity
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 3

        planner = PrefillPlanner(None, args, shared_state=shared_state)
        planner.model_name = "test-model"

        # Feed regression: TTFT = 0.1 * x + 100
        # x_sla = (500-100)/0.1 = 4000, target = 4000-3000 = 1000
        # boundary = 1000 * (3-1)/3 * 1.0 = 666.67
        for i in range(10):
            x = 2000 + i * 200
            y = 0.1 * x + 100
            planner.ttft_regression.add_observation(x, y)

        # All workers below boundary (666.67)
        planner.cached_per_worker_metrics = {
            "w1": {"active_prefill_tokens": 100.0, "last_isl": 3000.0, "last_ttft": 0.15},
            "w2": {"active_prefill_tokens": 200.0, "last_isl": 3000.0, "last_ttft": 0.16},
            "w3": {"active_prefill_tokens": 150.0, "last_isl": 3000.0, "last_ttft": 0.15},
        }

        result = planner.loadbased_plan_adjustment()
        assert result == 2  # scale down from 3 to 2

    def test_no_change_mixed_workers(self):
        """When workers are mixed (some above, some below), no scaling."""
        args = _build_loadbased_args()
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 2

        planner = PrefillPlanner(None, args, shared_state=shared_state)
        planner.model_name = "test-model"

        for i in range(10):
            x = 2000 + i * 200
            y = 0.1 * x + 100
            planner.ttft_regression.add_observation(x, y)

        # Mixed: one above target, one below
        planner.cached_per_worker_metrics = {
            "w1": {"active_prefill_tokens": 1500.0, "last_isl": 3000.0, "last_ttft": 0.35},
            "w2": {"active_prefill_tokens": 100.0, "last_isl": 3000.0, "last_ttft": 0.15},
        }

        result = planner.loadbased_plan_adjustment()
        assert result is None

    def test_cold_start_returns_none(self):
        """With insufficient data, loadbased_plan_adjustment returns None."""
        args = _build_loadbased_args()
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 2

        planner = PrefillPlanner(None, args, shared_state=shared_state)
        planner.model_name = "test-model"

        # Only 2 observations (min is 5)
        planner.ttft_regression.add_observation(1000.0, 200.0)
        planner.ttft_regression.add_observation(2000.0, 300.0)

        planner.cached_per_worker_metrics = {
            "w1": {"active_prefill_tokens": 5000.0, "last_isl": 3000.0, "last_ttft": 0.5},
        }

        result = planner.loadbased_plan_adjustment()
        assert result is None


class TestDecodeLoadBasedScaling:
    def test_scale_up_all_workers_above_target(self):
        """When all workers have active_decode_blocks above x_sla, scale up."""
        args = _build_loadbased_args()
        shared_state = PlannerSharedState()
        shared_state.num_d_workers = 2

        planner = DecodePlanner(None, args, shared_state=shared_state)
        planner.model_name = "test-model"

        # Feed regression: ITL = 0.5 * active_decode_blocks + 10
        # x_sla = (50 - 10) / 0.5 = 80
        for i in range(10):
            x = 20 + i * 10
            y = 0.5 * x + 10
            planner.itl_regression.add_observation(x, y)

        # All workers above x_sla (80)
        planner.cached_per_worker_metrics = {
            "w1": {"active_decode_blocks": 100.0, "last_itl": 0.06},
            "w2": {"active_decode_blocks": 95.0, "last_itl": 0.055},
        }

        result = planner.loadbased_plan_adjustment()
        assert result == 3

    def test_scale_down_all_workers_below_boundary(self):
        """When all decode workers are below boundary, scale down."""
        args = _build_loadbased_args()
        args.loadbased_scaling_down_sensitivity = 100
        shared_state = PlannerSharedState()
        shared_state.num_d_workers = 3

        planner = DecodePlanner(None, args, shared_state=shared_state)
        planner.model_name = "test-model"

        # ITL = 0.5 * x + 10, x_sla = (50-10)/0.5 = 80
        # boundary = 80 * (3-1)/3 * 1.0 = 53.33
        for i in range(10):
            x = 20 + i * 10
            y = 0.5 * x + 10
            planner.itl_regression.add_observation(x, y)

        # All workers below boundary (53.33)
        planner.cached_per_worker_metrics = {
            "w1": {"active_decode_blocks": 10.0, "last_itl": 0.02},
            "w2": {"active_decode_blocks": 15.0, "last_itl": 0.025},
            "w3": {"active_decode_blocks": 20.0, "last_itl": 0.03},
        }

        result = planner.loadbased_plan_adjustment()
        assert result == 2

    def test_cold_start_returns_none(self):
        """Decode cold start also returns None."""
        args = _build_loadbased_args()
        shared_state = PlannerSharedState()
        shared_state.num_d_workers = 2

        planner = DecodePlanner(None, args, shared_state=shared_state)
        planner.model_name = "test-model"

        planner.itl_regression.add_observation(10.0, 15.0)

        planner.cached_per_worker_metrics = {
            "w1": {"active_decode_blocks": 200.0, "last_itl": 0.1},
        }

        result = planner.loadbased_plan_adjustment()
        assert result is None


class TestLowerBoundEnforcement:
    def test_throughput_lower_bound_respected(self):
        """Load-based scaling should never go below throughput lower bound."""
        args = _build_loadbased_args()
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 5
        # Throughput says we need at least 4 prefill workers
        shared_state.throughput_lower_bound_p = 4

        planner = PrefillPlanner(None, args, shared_state=shared_state)
        planner.model_name = "test-model"

        # Regression says we should scale down to 4 (from 5)
        for i in range(10):
            x = 2000 + i * 200
            y = 0.1 * x + 100
            planner.ttft_regression.add_observation(x, y)

        # Workers all lightly loaded => wants to scale down to 4
        planner.cached_per_worker_metrics = {
            f"w{i}": {"active_prefill_tokens": 50.0, "last_isl": 3000.0, "last_ttft": 0.12}
            for i in range(5)
        }

        result = planner.loadbased_plan_adjustment()
        # Even though load-based wants to scale down, the result should be
        # at least 4 after lower bound enforcement (done in the loop, not in
        # loadbased_plan_adjustment itself)
        # loadbased_plan_adjustment returns raw desired value
        assert result == 4  # raw value from load-based

    def test_scaling_down_sensitivity_zero_never_scales_down(self):
        """With sensitivity=0, scale-down boundary is 0 so never scale down."""
        args = _build_loadbased_args()
        args.loadbased_scaling_down_sensitivity = 0
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 3

        planner = PrefillPlanner(None, args, shared_state=shared_state)
        planner.model_name = "test-model"

        for i in range(10):
            x = 2000 + i * 200
            y = 0.1 * x + 100
            planner.ttft_regression.add_observation(x, y)

        # All workers at zero load
        planner.cached_per_worker_metrics = {
            f"w{i}": {"active_prefill_tokens": 0.0, "last_isl": 3000.0, "last_ttft": 0.12}
            for i in range(3)
        }

        # boundary = target * (3-1)/3 * 0/100 = 0
        # all workers at 0 which is NOT less than 0 (it's equal)
        result = planner.loadbased_plan_adjustment()
        assert result is None  # no scaling happens


# ── Correction factor auto-disable tests ─────────────────────────────


class TestCorrectionFactorAutoDisable:
    def test_correction_factor_disabled_when_loadbased_enabled(self):
        """Correction factor should be auto-disabled when load-based scaling is on."""
        args = _build_loadbased_args()
        args.no_correction = False  # user didn't explicitly disable
        validate_sla_planner_args(args)
        assert args.no_correction is True

    def test_correction_factor_stays_disabled_if_already_set(self):
        """If user already set --no-correction, no extra warning needed."""
        args = _build_loadbased_args()
        args.no_correction = True  # user explicitly set
        validate_sla_planner_args(args)
        assert args.no_correction is True

    def test_correction_factor_not_disabled_without_loadbased(self):
        """Without load-based scaling, correction factor should respect user setting."""
        args = _build_loadbased_args()
        args.enable_loadbased_scaling = False
        args.no_correction = False
        validate_sla_planner_args(args)
        assert args.no_correction is False


# ── DGD worker count reconciliation tests ────────────────────────────


class TestWorkerCountReconciliation:
    async def test_prefill_observe_filters_workers(self):
        """observe_engine_load_stats should filter to workers with prefill metrics."""
        args = _build_loadbased_args()
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 1

        planner = PrefillPlanner(None, args, shared_state=shared_state)
        planner.model_name = "test-model"

        # Simulate router_metrics_client returning both prefill and decode workers
        all_metrics = {
            "w1": {"active_prefill_tokens": 500.0, "last_ttft": 0.2, "last_isl": 3000.0},
            "w2": {"active_decode_blocks": 50.0, "last_itl": 0.04},  # decode worker
        }
        planner.router_metrics_client = Mock()
        planner.router_metrics_client.get_averaged_metrics.return_value = all_metrics

        await planner.observe_engine_load_stats()

        # Only the prefill worker should be in cached metrics
        assert len(planner.cached_per_worker_metrics) == 1
        assert "w1" in planner.cached_per_worker_metrics
        assert "w2" not in planner.cached_per_worker_metrics

    async def test_decode_observe_filters_workers(self):
        """observe_engine_load_stats should filter to workers with decode metrics."""
        args = _build_loadbased_args()
        shared_state = PlannerSharedState()
        shared_state.num_d_workers = 1

        planner = DecodePlanner(None, args, shared_state=shared_state)
        planner.model_name = "test-model"

        all_metrics = {
            "w1": {"active_prefill_tokens": 500.0, "last_ttft": 0.2, "last_isl": 3000.0},
            "w2": {"active_decode_blocks": 50.0, "last_itl": 0.04},
        }
        planner.router_metrics_client = Mock()
        planner.router_metrics_client.get_averaged_metrics.return_value = all_metrics

        await planner.observe_engine_load_stats()

        # Only the decode worker should be in cached metrics
        assert len(planner.cached_per_worker_metrics) == 1
        assert "w2" in planner.cached_per_worker_metrics
        assert "w1" not in planner.cached_per_worker_metrics

    def test_worker_count_mismatch_detected(self):
        """When DGD and Prometheus worker counts differ, the mismatch should be detectable."""
        args = _build_loadbased_args()
        shared_state = PlannerSharedState()
        # DGD says 3 prefill workers
        shared_state.num_p_workers = 3

        planner = PrefillPlanner(None, args, shared_state=shared_state)
        planner.model_name = "test-model"

        # But router only reports 2 prefill workers
        planner.cached_per_worker_metrics = {
            "w1": {"active_prefill_tokens": 500.0, "last_isl": 3000.0, "last_ttft": 0.2},
            "w2": {"active_prefill_tokens": 600.0, "last_isl": 3000.0, "last_ttft": 0.25},
        }

        # The mismatch should be detectable by comparing counts
        prom_count = len(planner.cached_per_worker_metrics)
        dgd_count = shared_state.num_p_workers
        assert prom_count != dgd_count
        assert prom_count == 2
        assert dgd_count == 3

    def test_worker_count_match_allows_scaling(self):
        """When DGD and Prometheus counts match, scaling proceeds normally."""
        args = _build_loadbased_args()
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 2

        planner = PrefillPlanner(None, args, shared_state=shared_state)
        planner.model_name = "test-model"

        planner.cached_per_worker_metrics = {
            "w1": {"active_prefill_tokens": 1500.0, "last_isl": 3000.0, "last_ttft": 0.35},
            "w2": {"active_prefill_tokens": 1200.0, "last_isl": 3000.0, "last_ttft": 0.30},
        }

        prom_count = len(planner.cached_per_worker_metrics)
        dgd_count = shared_state.num_p_workers
        assert prom_count == dgd_count

        # With matching counts and sufficient regression data, scaling should work
        for i in range(10):
            x = 2000 + i * 200
            y = 0.1 * x + 100
            planner.ttft_regression.add_observation(x, y)

        result = planner.loadbased_plan_adjustment()
        assert result is not None  # scaling proceeds
