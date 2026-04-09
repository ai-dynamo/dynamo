# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for SLA planner replica calculation logic.

These tests focus specifically on the replica calculation formulas without
testing load prediction or regression internals.
"""

import asyncio
import math
import os
from unittest.mock import Mock, patch

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.budget import _apply_global_gpu_budget
from dynamo.planner.core.decode import DecodePlanner
from dynamo.planner.core.prefill import PrefillPlanner
from dynamo.planner.core.state import PlannerSharedState
from dynamo.planner.monitoring.traffic_metrics import Metrics
from dynamo.planner.monitoring.worker_info import WorkerInfo

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]


class PlannerHarness:
    def __init__(self, prefill_planner, decode_planner, shared_state):
        self.prefill_planner = prefill_planner
        self.decode_planner = decode_planner
        self.shared_state = shared_state
        self.last_target_replicas = []

    async def make_adjustments(self):
        if not self.shared_state.last_metrics.is_valid():
            return

        num_p, num_d, is_stable = await self.prefill_planner.get_workers_info()
        self.shared_state.num_p_workers = num_p
        self.shared_state.num_d_workers = num_d

        next_num_p = self.prefill_planner.plan_adjustment()
        next_num_d = self.decode_planner.plan_adjustment()
        if next_num_p is None or next_num_d is None:
            return

        next_num_p, next_num_d = _apply_global_gpu_budget(
            next_num_p, next_num_d, self.prefill_planner.config
        )
        self.prefill_planner.update_predicted_replicas_metric(next_num_p)
        self.decode_planner.update_predicted_replicas_metric(next_num_d)

        target_replicas = [
            {
                "sub_component_type": "prefill",
                "component_name": self.prefill_planner.prefill_worker_info.k8s_name,
                "desired_replicas": next_num_p,
            },
            {
                "sub_component_type": "decode",
                "component_name": self.prefill_planner.decode_worker_info.k8s_name,
                "desired_replicas": next_num_d,
            },
        ]
        self.last_target_replicas = target_replicas

        if not self.prefill_planner.config.no_operation:
            await self.prefill_planner.connector.set_component_replicas(
                target_replicas, blocking=False
            )

    def __getattr__(self, name):
        shared_attrs = {
            "num_req_predictor",
            "isl_predictor",
            "osl_predictor",
            "connector",
            "prometheus_traffic_client",
            "config",
        }
        prefill_attrs = {
            "ttft_regression",
            "prefill_worker_info",
        }
        decode_attrs = {
            "itl_regression",
            "decode_worker_info",
        }
        if name == "last_metrics":
            return self.shared_state.last_metrics
        if name == "get_workers_info":
            return self.prefill_planner.get_workers_info
        if name in shared_attrs:
            return getattr(self.prefill_planner, name)
        if name in prefill_attrs:
            return getattr(self.prefill_planner, name)
        if name in decode_attrs:
            return getattr(self.decode_planner, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in {"prefill_planner", "decode_planner", "shared_state"}:
            return super().__setattr__(name, value)
        shared_attrs = {
            "num_req_predictor",
            "isl_predictor",
            "osl_predictor",
            "connector",
            "prometheus_traffic_client",
            "config",
            "get_workers_info",
        }
        prefill_attrs = {"ttft_regression"}
        decode_attrs = {"itl_regression"}
        if name == "last_metrics":
            self.shared_state.last_metrics = value
            return None
        if name in shared_attrs:
            # Store locally to support patch.object lifecycle (set/del).
            object.__setattr__(self, name, value)
            setattr(self.prefill_planner, name, value)
            setattr(self.decode_planner, name, value)
            return None
        if name in prefill_attrs:
            setattr(self.prefill_planner, name, value)
            return None
        if name in decode_attrs:
            setattr(self.decode_planner, name, value)
            return None
        return super().__setattr__(name, value)


def _replica_count(target_replicas, component_name, default=1):
    for replica in target_replicas:
        if replica.get("component_name") == component_name:
            return replica.get("desired_replicas", default)
    return default


@pytest.fixture
def planner():
    """Set up test environment with mocked dependencies."""
    config = PlannerConfig.model_construct(
        throughput_adjustment_interval=60,
        prefill_engine_num_gpu=1,
        decode_engine_num_gpu=1,
        min_endpoint=1,
        max_gpu_budget=10,
        ttft=80.0,
        itl=10.0,
        backend="vllm",
        no_operation=True,
        metric_pulling_prometheus_endpoint="http://localhost:9090",
        metric_reporting_prometheus_port=0,
        load_predictor="constant",
        profile_results_dir=os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "profiling_results",
            "H200_TP1P_TP1D",
        ),
        environment="kubernetes",
        namespace="test-namespace",
        enable_throughput_scaling=True,
        enable_load_scaling=False,
        load_predictor_warmup_trace=None,
        load_predictor_log1p=False,
        max_num_fpm_samples=50,
        fpm_sample_bucket_size=16,
        load_min_observations=5,
    )

    mock_runtime = Mock()

    with patch("dynamo.planner.monitoring.planner_metrics.Gauge") as mock_gauge:
        mock_gauge.return_value = Mock()

        shared_state = PlannerSharedState()
        prefill_planner = PrefillPlanner(
            mock_runtime, config, shared_state=shared_state
        )
        decode_planner = DecodePlanner(mock_runtime, config, shared_state=shared_state)
        planner = PlannerHarness(prefill_planner, decode_planner, shared_state)

        # Set up WorkerInfo for both planners
        prefill_planner.prefill_worker_info = WorkerInfo(
            k8s_name="VllmPrefillWorker",
            component_name="prefill",
            endpoint="generate",
        )
        prefill_planner.decode_worker_info = WorkerInfo(
            k8s_name="VllmDecodeWorker",
            component_name="backend",
            endpoint="generate",
        )
        decode_planner.prefill_worker_info = prefill_planner.prefill_worker_info
        decode_planner.decode_worker_info = prefill_planner.decode_worker_info

        planner.ttft_regression = Mock()
        # Default: 40000 tokens/s at isl=3000 → 40000/3000 rps
        planner.ttft_regression.find_best_engine_prefill_rps.return_value = (
            40000.0 / 3000.0,
            75.0,
        )
        planner.ttft_regression.has_sufficient_data.return_value = True

        planner.itl_regression = Mock()
        # Default: 10000 tokens/s at osl=150 → 10000/150 rps
        planner.itl_regression.find_best_engine_decode_rps.return_value = (
            10000.0 / 150.0,
            9.5,
        )
        planner.itl_regression.has_sufficient_data.return_value = True

        # Mock the predictors to return fixed values
        planner.num_req_predictor = Mock()
        planner.isl_predictor = Mock()
        planner.osl_predictor = Mock()

        # Mock the connector since we're not testing actual scaling
        planner.connector = Mock()

        # Mock prometheus client
        planner.prometheus_traffic_client = Mock()

        planner.config = config

        yield planner


class TestReplicaCalculation:
    """Test replica calculation formulas in isolation."""

    @pytest.mark.nightly
    @pytest.mark.gpu_2
    @pytest.mark.performance
    def test_prefill_replica_calculation_basic(self, planner):
        """Test basic prefill replica calculation."""
        next_num_req = 10
        next_isl = 3000
        engine_rps = 40000.0 / next_isl

        planner.num_req_predictor.predict_next.return_value = next_num_req
        planner.isl_predictor.predict_next.return_value = next_isl
        planner.osl_predictor.predict_next.return_value = 150

        planner.ttft_regression.find_best_engine_prefill_rps.return_value = (
            engine_rps,
            75.0,
        )
        planner.itl_regression.find_best_engine_decode_rps.return_value = (
            10000.0 / 150.0,
            9.5,
        )

        # Formula: ceil(num_req / interval / engine_rps)
        pred_prefill_demand = (
            next_num_req / planner.config.throughput_adjustment_interval
        )
        expected_prefill_replicas = math.ceil(pred_prefill_demand / engine_rps)

        planner.last_metrics = Metrics(
            num_req=10, isl=3000, osl=150, ttft=80.0, itl=10.0, request_duration=100.0
        )

        async def mock_get_workers_info(*args, **kwargs):
            return (1, 1, True)

        planner.get_workers_info = mock_get_workers_info

        asyncio.run(planner.make_adjustments())

        prefill_component = "VllmPrefillWorker"
        calculated_prefill_replicas = _replica_count(
            planner.last_target_replicas, prefill_component
        )
        print(f"Expected prefill replicas: {expected_prefill_replicas}")
        print(f"Calculated prefill replicas: {calculated_prefill_replicas}")

        assert (
            max(expected_prefill_replicas, planner.config.min_endpoint)
            == calculated_prefill_replicas
        )

    @pytest.mark.nightly
    @pytest.mark.gpu_2
    @pytest.mark.performance
    def test_decode_replica_calculation_basic(self, planner):
        """Test basic decode replica calculation."""
        next_num_req = 10
        next_osl = 150
        engine_rps = 10000.0 / next_osl

        planner.num_req_predictor.predict_next.return_value = next_num_req
        planner.isl_predictor.predict_next.return_value = 3000
        planner.osl_predictor.predict_next.return_value = next_osl

        planner.ttft_regression.find_best_engine_prefill_rps.return_value = (
            40000.0 / 3000.0,
            75.0,
        )
        planner.itl_regression.find_best_engine_decode_rps.return_value = (
            engine_rps,
            9.5,
        )

        # Formula: ceil(num_req / interval / engine_rps)
        expected_decode_replicas = math.ceil(
            next_num_req / planner.config.throughput_adjustment_interval / engine_rps
        )

        planner.last_metrics = Metrics(
            num_req=10, isl=3000, osl=150, ttft=80.0, itl=10.0, request_duration=100.0
        )

        async def mock_get_workers_info(*args, **kwargs):
            return (1, 1, True)

        planner.get_workers_info = mock_get_workers_info

        asyncio.run(planner.make_adjustments())

        decode_component = "VllmDecodeWorker"
        calculated_decode_replicas = _replica_count(
            planner.last_target_replicas, decode_component
        )
        print(f"Expected decode replicas: {expected_decode_replicas}")
        print(f"Calculated decode replicas: {calculated_decode_replicas}")

        assert (
            max(expected_decode_replicas, planner.config.min_endpoint)
            == calculated_decode_replicas
        )

    @pytest.mark.parametrize(
        "num_req,decode_rps,expected_p,expected_d",
        [
            (10, 10000.0 / 150.0, 1, 1),  # low_load_10_req_per_second
            (
                500,
                1000.0 / 150.0,
                1,
                2,
            ),  # high_load_500_req_per_second (lower decode rps)
        ],
    )
    @pytest.mark.nightly
    @pytest.mark.gpu_2
    @pytest.mark.performance
    def test_scaling_scenario_low_to_high_load(
        self, planner, num_req, decode_rps, expected_p, expected_d
    ):
        """Test scaling from low to high load scenarios."""
        planner.num_req_predictor.predict_next.return_value = num_req
        planner.isl_predictor.predict_next.return_value = 3000
        planner.osl_predictor.predict_next.return_value = 150

        planner.ttft_regression.find_best_engine_prefill_rps.return_value = (
            40000.0 / 3000.0,
            75.0,
        )
        planner.itl_regression.find_best_engine_decode_rps.return_value = (
            decode_rps,
            9.5,
        )

        planner.last_metrics = Metrics(
            num_req=num_req,
            isl=3000,
            osl=150,
            ttft=80.0,
            itl=10.0,
            request_duration=100.0,
        )

        async def mock_get_workers_info(*args, **kwargs):
            return (1, 1, True)

        planner.get_workers_info = mock_get_workers_info
        planner.connector.reset_mock()

        asyncio.run(planner.make_adjustments())

        prefill_replicas = _replica_count(
            planner.last_target_replicas, "VllmPrefillWorker"
        )
        decode_replicas = _replica_count(
            planner.last_target_replicas, "VllmDecodeWorker"
        )
        print(f"Load {num_req} req/s: P={prefill_replicas}, D={decode_replicas}")

        assert (
            prefill_replicas == expected_p
        ), f"Prefill replicas mismatch: expected {expected_p}, got {prefill_replicas}"
        assert (
            decode_replicas == expected_d
        ), f"Decode replicas mismatch: expected {expected_d}, got {decode_replicas}"

    @pytest.mark.nightly
    @pytest.mark.gpu_2
    @pytest.mark.performance
    def test_gpu_budget_constraint(self, planner):
        """Test that GPU budget constraints are properly applied."""
        planner.config.max_gpu_budget = 3

        planner.num_req_predictor.predict_next.return_value = 50
        planner.isl_predictor.predict_next.return_value = 3000
        planner.osl_predictor.predict_next.return_value = 150

        planner.ttft_regression.find_best_engine_prefill_rps.return_value = (
            40000.0 / 3000.0,
            75.0,
        )
        planner.itl_regression.find_best_engine_decode_rps.return_value = (
            10000.0 / 150.0,
            9.5,
        )

        planner.last_metrics = Metrics(
            num_req=50, isl=3000, osl=150, ttft=80.0, itl=10.0, request_duration=100.0
        )

        async def mock_get_workers_info(*args, **kwargs):
            return (1, 1, True)

        planner.get_workers_info = mock_get_workers_info

        asyncio.run(planner.make_adjustments())

        prefill_replicas = _replica_count(
            planner.last_target_replicas, "VllmPrefillWorker"
        )
        decode_replicas = _replica_count(
            planner.last_target_replicas, "VllmDecodeWorker"
        )
        total_gpus = (
            prefill_replicas * planner.config.prefill_engine_num_gpu
            + decode_replicas * planner.config.decode_engine_num_gpu
        )

        print(
            f"GPU budget test: P={prefill_replicas}, D={decode_replicas}, Total GPUs={total_gpus}"
        )

        assert (
            total_gpus <= planner.config.max_gpu_budget
        ), "Total GPU usage exceeds budget"

    @pytest.mark.nightly
    @pytest.mark.gpu_2
    @pytest.mark.performance
    def test_min_endpoint_constraint(self, planner):
        """Test that minimum endpoint constraints are respected."""
        planner.config.min_endpoint = 2

        planner.num_req_predictor.predict_next.return_value = 1
        planner.isl_predictor.predict_next.return_value = 100
        planner.osl_predictor.predict_next.return_value = 10

        planner.ttft_regression.find_best_engine_prefill_rps.return_value = (
            40000.0 / 100.0,
            75.0,
        )
        planner.itl_regression.find_best_engine_decode_rps.return_value = (
            10000.0 / 10.0,
            9.5,
        )

        planner.last_metrics = Metrics(
            num_req=1, isl=100, osl=10, ttft=80.0, itl=10.0, request_duration=100.0
        )

        async def mock_get_workers_info(*args, **kwargs):
            return (1, 1, True)

        planner.get_workers_info = mock_get_workers_info

        asyncio.run(planner.make_adjustments())

        prefill_replicas = _replica_count(
            planner.last_target_replicas, "VllmPrefillWorker"
        )
        decode_replicas = _replica_count(
            planner.last_target_replicas, "VllmDecodeWorker"
        )
        print(f"Min endpoint test: P={prefill_replicas}, D={decode_replicas}")

        assert (
            prefill_replicas >= planner.config.min_endpoint
        ), "Prefill replicas below minimum"
        assert (
            decode_replicas >= planner.config.min_endpoint
        ), "Decode replicas below minimum"

    @pytest.mark.nightly
    @pytest.mark.gpu_2
    @pytest.mark.performance
    def test_multi_gpu_engines(self, planner):
        """Test replica calculation with multi-GPU engines."""
        planner.config.prefill_engine_num_gpu = 2
        planner.config.decode_engine_num_gpu = 4

        planner.num_req_predictor.predict_next.return_value = 20
        planner.isl_predictor.predict_next.return_value = 3000
        planner.osl_predictor.predict_next.return_value = 150

        # Engine-level request rate (already accounts for multi-GPU)
        prefill_engine_rps = 40000.0 / 3000.0
        decode_engine_rps = 5000.0 / 150.0
        planner.ttft_regression.find_best_engine_prefill_rps.return_value = (
            prefill_engine_rps,
            75.0,
        )
        planner.itl_regression.find_best_engine_decode_rps.return_value = (
            decode_engine_rps,
            9.5,
        )

        planner.last_metrics = Metrics(
            num_req=20, isl=3000, osl=150, ttft=80.0, itl=10.0, request_duration=100.0
        )

        async def mock_get_workers_info(*args, **kwargs):
            return (1, 1, True)

        planner.get_workers_info = mock_get_workers_info

        # No engine_num_gpu division — regression returns engine-level rps
        expected_prefill_replicas = math.ceil(
            20 / planner.config.throughput_adjustment_interval / prefill_engine_rps
        )
        expected_decode_replicas = math.ceil(
            20 / planner.config.throughput_adjustment_interval / decode_engine_rps
        )

        asyncio.run(planner.make_adjustments())

        prefill_replicas = _replica_count(
            planner.last_target_replicas, "VllmPrefillWorker"
        )
        decode_replicas = _replica_count(
            planner.last_target_replicas, "VllmDecodeWorker"
        )
        print(
            f"Multi-GPU test: P={prefill_replicas} (expected ~{expected_prefill_replicas}), "
            f"D={decode_replicas} (expected ~{expected_decode_replicas})"
        )

        assert prefill_replicas == max(
            expected_prefill_replicas, planner.config.min_endpoint
        )
        assert decode_replicas == max(
            expected_decode_replicas, planner.config.min_endpoint
        )

    @pytest.mark.weekly
    @pytest.mark.gpu_2
    @pytest.mark.performance
    def test_complex_gpu_budget_scaling(self, planner):
        """Test complex GPU budget scaling with proportional reduction."""
        planner.config.max_gpu_budget = 5
        planner.config.prefill_engine_num_gpu = 2
        planner.config.decode_engine_num_gpu = 2
        planner.config.min_endpoint = 1

        planner.num_req_predictor.predict_next.return_value = 100
        planner.isl_predictor.predict_next.return_value = 3000
        planner.osl_predictor.predict_next.return_value = 150

        planner.ttft_regression.find_best_engine_prefill_rps.return_value = (
            10000.0 / 3000.0,
            300.0,
        )
        planner.itl_regression.find_best_engine_decode_rps.return_value = (
            1000.0 / 150.0,
            9.5,
        )

        planner.last_metrics = Metrics(
            num_req=100,
            isl=3000,
            osl=150,
            ttft=80.0,
            itl=10.0,
            request_duration=100.0,
        )

        async def mock_get_workers_info(*args, **kwargs):
            return (1, 1, True)

        planner.get_workers_info = mock_get_workers_info

        asyncio.run(planner.make_adjustments())

        prefill_replicas = _replica_count(
            planner.last_target_replicas, "VllmPrefillWorker"
        )
        decode_replicas = _replica_count(
            planner.last_target_replicas, "VllmDecodeWorker"
        )
        total_gpus = (
            prefill_replicas * planner.config.prefill_engine_num_gpu
            + decode_replicas * planner.config.decode_engine_num_gpu
        )

        print(
            f"Complex GPU budget test: P={prefill_replicas}, D={decode_replicas}, "
            f"Total GPUs={total_gpus}"
        )

        assert (
            total_gpus <= planner.config.max_gpu_budget
        ), "Total GPU usage should not exceed budget"
        assert (
            prefill_replicas >= planner.config.min_endpoint
        ), "Should respect min_endpoint for prefill"
        assert (
            decode_replicas >= planner.config.min_endpoint
        ), "Should respect min_endpoint for decode"
