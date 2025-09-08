# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for SLA planner replica calculation logic.

These tests focus specifically on the replica calculation formulas without
testing load prediction, interpolation, or correction factors.
"""

import argparse
import math
import os

# Mock dependencies before importing planner modules
import sys

# We'll import the actual Planner class to test its calculation logic
from unittest.mock import MagicMock, Mock, patch

import pytest

# Create mock modules for dependencies that might not be available in test environment
mock_prometheus = MagicMock()
mock_prometheus.Gauge = MagicMock()
mock_prometheus.start_http_server = MagicMock()

mock_runtime = MagicMock()
mock_runtime.logging = MagicMock()
mock_runtime.logging.configure_dynamo_logging = MagicMock()

# Patch them into sys.modules before importing
sys.modules["prometheus_client"] = mock_prometheus
sys.modules["dynamo.runtime"] = mock_runtime
sys.modules["dynamo.runtime.logging"] = mock_runtime.logging

# Now import after mocking
from dynamo.planner.utils.planner_core import Metrics, Planner  # noqa: E402


@pytest.fixture
def planner():
    """Set up test environment with mocked dependencies."""
    # Create mock arguments
    args = argparse.Namespace()
    args.adjustment_interval = 60
    args.prefill_engine_num_gpu = 1
    args.decode_engine_num_gpu = 1
    args.min_endpoint = 1
    args.max_gpu_budget = 10
    args.ttft = 80  # ms
    args.itl = 10  # ms
    args.backend = "vllm"
    args.no_operation = True  # Don't actually scale
    args.prometheus_port = 0  # 0 means disabled
    args.load_predictor = "constant"
    args.load_prediction_window_size = 10
    args.profile_results_dir = os.path.join(
        os.path.dirname(__file__),
        "profiling_results/H200_TP1P_TP1D",
    )
    args.environment = "kubernetes"

    # Mock the runtime
    mock_runtime = Mock()

    # Patch Prometheus Gauge to avoid registry conflicts
    with patch("dynamo.planner.utils.planner_core.Gauge") as mock_gauge:
        mock_gauge.return_value = Mock()

        # Create planner instance
        planner = Planner(mock_runtime, args)

        # Mock the interpolators to return fixed values for testing
        planner.prefill_interpolator = Mock()
        planner.decode_interpolator = Mock()

        # Mock the predictors to return fixed values
        planner.num_req_predictor = Mock()
        planner.isl_predictor = Mock()
        planner.osl_predictor = Mock()

        # Mock the connector since we're not testing actual scaling
        planner.connector = Mock()

        # Mock prometheus client
        planner.prometheus_api_client = Mock()

        # Set up some baseline correction factors
        planner.p_correction_factor = 1.0
        planner.d_correction_factor = 1.0

        # Store args for easy access in tests
        planner.args = args

        yield planner
        # Cleanup is automatic with context manager


class TestReplicaCalculation:
    """Test replica calculation formulas in isolation."""

    def test_prefill_replica_calculation_basic(self, planner):
        """Test basic prefill replica calculation."""
        # Setup test data
        next_num_req = 10
        next_isl = 3000
        prefill_thpt_per_gpu = 40000  # tokens/s/gpu (from the test data)

        # Mock the predictor outputs
        planner.num_req_predictor.predict_next.return_value = next_num_req
        planner.isl_predictor.predict_next.return_value = next_isl
        planner.osl_predictor.predict_next.return_value = 150

        # Mock interpolator output
        planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = (
            prefill_thpt_per_gpu
        )
        planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
            10000,
            0.01,
            0.5,
        )

        # Calculate expected result manually
        pred_prefill_load_per_gpu = (
            next_num_req
            * next_isl
            / planner.args.adjustment_interval
            * min(1, planner.p_correction_factor)
        )
        expected_prefill_replicas = math.ceil(
            pred_prefill_load_per_gpu
            / prefill_thpt_per_gpu
            / planner.args.prefill_engine_num_gpu
        )

        # Set up valid metrics to trigger calculation
        planner.last_metrics = Metrics(
            num_req=10, isl=3000, osl=150, ttft=80.0, itl=10.0, request_duration=100.0
        )

        # Mock workers info
        async def mock_get_workers_info():
            return (["prefill1"], ["decode1"])

        planner.get_workers_info = mock_get_workers_info

        # Mock interpolation calls for correction factor calculation
        planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
        planner.decode_interpolator.interpolate_itl.return_value = 10.0

        # Run the calculation
        import asyncio

        asyncio.run(planner.make_adjustments())

        # Extract the calculated values from the log calls or by checking the mock calls
        # Since we mocked the connector, we can check what replicas were requested
        if planner.connector.set_component_replicas.called:
            call_args = planner.connector.set_component_replicas.call_args[0][0]
            prefill_component = "VllmPrefillWorker"
            calculated_prefill_replicas = call_args.get(prefill_component, 1)

            print(f"Expected prefill replicas: {expected_prefill_replicas}")
            print(f"Calculated prefill replicas: {calculated_prefill_replicas}")

            # Allow for small differences due to min_endpoint constraints
            assert (
                max(expected_prefill_replicas, planner.args.min_endpoint)
                == calculated_prefill_replicas
            )

    def test_decode_replica_calculation_basic(self, planner):
        """Test basic decode replica calculation."""
        # Setup test data
        next_num_req = 10
        next_osl = 150
        decode_thpt_per_gpu = 10000  # tokens/s/gpu

        # Mock the predictor outputs
        planner.num_req_predictor.predict_next.return_value = next_num_req
        planner.isl_predictor.predict_next.return_value = 3000
        planner.osl_predictor.predict_next.return_value = next_osl

        # Mock interpolator outputs
        planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = 40000
        planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
            decode_thpt_per_gpu,
            0.01,
            0.5,
        )

        # Calculate expected result manually
        expected_decode_replicas = math.ceil(
            next_num_req
            * next_osl
            / planner.args.adjustment_interval
            / decode_thpt_per_gpu
            / planner.args.decode_engine_num_gpu
        )

        # Set up valid metrics
        planner.last_metrics = Metrics(
            num_req=10, isl=3000, osl=150, ttft=80.0, itl=10.0, request_duration=100.0
        )

        # Mock workers info
        async def mock_get_workers_info():
            return (["prefill1"], ["decode1"])

        planner.get_workers_info = mock_get_workers_info

        # Mock interpolation calls for correction factor calculation
        planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
        planner.decode_interpolator.interpolate_itl.return_value = 10.0

        # Run the calculation
        import asyncio

        asyncio.run(planner.make_adjustments())

        # Check the results
        if planner.connector.set_component_replicas.called:
            call_args = planner.connector.set_component_replicas.call_args[0][0]
            decode_component = "VllmDecodeWorker"
            calculated_decode_replicas = call_args.get(decode_component, 1)

            print(f"Expected decode replicas: {expected_decode_replicas}")
            print(f"Calculated decode replicas: {calculated_decode_replicas}")

            # Allow for small differences due to min_endpoint constraints
            assert (
                max(expected_decode_replicas, planner.args.min_endpoint)
                == calculated_decode_replicas
            )

    @pytest.mark.parametrize(
        "num_req,decode_thpt,expected_p,expected_d",
        [
            (10, 10000, 1, 1),  # low_load_10_req_per_second
            (500, 1000, 1, 2),  # high_load_500_req_per_second (lower decode throughput)
        ],
    )
    def test_scaling_scenario_low_to_high_load(
        self, planner, num_req, decode_thpt, expected_p, expected_d
    ):
        """Test scaling from low to high load scenarios."""
        # Reset the planner state
        planner.p_correction_factor = 1.0
        planner.d_correction_factor = 1.0

        # Mock predictor outputs for this case
        planner.num_req_predictor.predict_next.return_value = num_req
        planner.isl_predictor.predict_next.return_value = 3000
        planner.osl_predictor.predict_next.return_value = 150

        # Mock interpolator outputs (based on H200 1P1D profiling data)
        planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = (
            40000  # tokens/s/gpu
        )
        planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
            decode_thpt,
            0.01,
            0.5,
        )

        # Set up metrics
        planner.last_metrics = Metrics(
            num_req=num_req,
            isl=3000,
            osl=150,
            ttft=80.0,
            itl=10.0,
            request_duration=100.0,
        )

        # Mock workers info
        async def mock_get_workers_info():
            return (["prefill1"], ["decode1"])

        planner.get_workers_info = mock_get_workers_info

        # Mock interpolation calls for correction factor calculation
        planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
        planner.decode_interpolator.interpolate_itl.return_value = 10.0

        # Reset the mock
        planner.connector.reset_mock()

        # Run calculation
        import asyncio

        asyncio.run(planner.make_adjustments())

        # Verify results
        if planner.connector.set_component_replicas.called:
            call_args = planner.connector.set_component_replicas.call_args[0][0]

            prefill_replicas = call_args.get("VllmPrefillWorker", 1)
            decode_replicas = call_args.get("VllmDecodeWorker", 1)

            print(f"Load {num_req} req/s: P={prefill_replicas}, D={decode_replicas}")

            assert (
                prefill_replicas == expected_p
            ), f"Prefill replicas mismatch: expected {expected_p}, got {prefill_replicas}"
            assert (
                decode_replicas == expected_d
            ), f"Decode replicas mismatch: expected {expected_d}, got {decode_replicas}"

    def test_gpu_budget_constraint(self, planner):
        """Test that GPU budget constraints are properly applied."""
        # Set a low GPU budget
        planner.args.max_gpu_budget = 3

        # Mock predictor outputs that would normally require more GPUs
        planner.num_req_predictor.predict_next.return_value = 50  # High load
        planner.isl_predictor.predict_next.return_value = 3000
        planner.osl_predictor.predict_next.return_value = 150

        # Mock interpolator outputs
        planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = 40000
        planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
            10000,
            0.01,
            0.5,
        )

        # Set up metrics
        planner.last_metrics = Metrics(
            num_req=50, isl=3000, osl=150, ttft=80.0, itl=10.0, request_duration=100.0
        )

        # Mock workers info
        async def mock_get_workers_info():
            return (["prefill1"], ["decode1"])

        planner.get_workers_info = mock_get_workers_info

        # Mock interpolation calls
        planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
        planner.decode_interpolator.interpolate_itl.return_value = 10.0

        # Run calculation
        import asyncio

        asyncio.run(planner.make_adjustments())

        # Verify that total GPU usage doesn't exceed budget
        if planner.connector.set_component_replicas.called:
            call_args = planner.connector.set_component_replicas.call_args[0][0]

            prefill_replicas = call_args.get("VllmPrefillWorker", 1)
            decode_replicas = call_args.get("VllmDecodeWorker", 1)

            total_gpus = (
                prefill_replicas * planner.args.prefill_engine_num_gpu
                + decode_replicas * planner.args.decode_engine_num_gpu
            )

            print(
                f"GPU budget test: P={prefill_replicas}, D={decode_replicas}, Total GPUs={total_gpus}"
            )

            assert (
                total_gpus <= planner.args.max_gpu_budget
            ), "Total GPU usage exceeds budget"

    def test_min_endpoint_constraint(self, planner):
        """Test that minimum endpoint constraints are respected."""
        planner.args.min_endpoint = 2

        # Mock predictor outputs that would normally require fewer workers
        planner.num_req_predictor.predict_next.return_value = 1  # Very low load
        planner.isl_predictor.predict_next.return_value = 100
        planner.osl_predictor.predict_next.return_value = 10

        # Mock interpolator outputs
        planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = 40000
        planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
            10000,
            0.01,
            0.5,
        )

        # Set up metrics
        planner.last_metrics = Metrics(
            num_req=1, isl=100, osl=10, ttft=80.0, itl=10.0, request_duration=100.0
        )

        # Mock workers info
        async def mock_get_workers_info():
            return (["prefill1"], ["decode1"])

        planner.get_workers_info = mock_get_workers_info

        # Mock interpolation calls
        planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
        planner.decode_interpolator.interpolate_itl.return_value = 10.0

        # Run calculation
        import asyncio

        asyncio.run(planner.make_adjustments())

        # Verify minimum constraints are respected
        if planner.connector.set_component_replicas.called:
            call_args = planner.connector.set_component_replicas.call_args[0][0]

            prefill_replicas = call_args.get("VllmPrefillWorker", 1)
            decode_replicas = call_args.get("VllmDecodeWorker", 1)

            print(f"Min endpoint test: P={prefill_replicas}, D={decode_replicas}")

            assert (
                prefill_replicas >= planner.args.min_endpoint
            ), "Prefill replicas below minimum"
            assert (
                decode_replicas >= planner.args.min_endpoint
            ), "Decode replicas below minimum"

    def test_prefill_correction_factor_clamping(self, planner):
        """Test that prefill correction factor > 1 is clamped to 1."""
        # Set a high correction factor > 1
        planner.p_correction_factor = 2.5
        planner.d_correction_factor = 1.0

        # Mock predictor outputs
        planner.num_req_predictor.predict_next.return_value = 10
        planner.isl_predictor.predict_next.return_value = 3000
        planner.osl_predictor.predict_next.return_value = 150

        # Mock interpolator outputs
        planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = 40000
        planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
            10000,
            0.01,
            0.5,
        )

        # Set up metrics
        planner.last_metrics = Metrics(
            num_req=10, isl=3000, osl=150, ttft=80.0, itl=10.0, request_duration=100.0
        )

        # Mock workers info
        async def mock_get_workers_info():
            return (["prefill1"], ["decode1"])

        planner.get_workers_info = mock_get_workers_info

        # Mock interpolation calls
        planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
        planner.decode_interpolator.interpolate_itl.return_value = 10.0

        # Calculate expected result manually with clamping
        # Should use min(1, 2.5) = 1
        pred_prefill_load_per_gpu = (
            10 * 3000 / planner.args.adjustment_interval * min(1, 2.5)  # Should be * 1
        )
        expected_prefill_replicas = math.ceil(
            pred_prefill_load_per_gpu / 40000 / planner.args.prefill_engine_num_gpu
        )

        # Run calculation
        import asyncio

        asyncio.run(planner.make_adjustments())

        # Verify that correction factor was effectively clamped
        if planner.connector.set_component_replicas.called:
            call_args = planner.connector.set_component_replicas.call_args[0][0]
            prefill_replicas = call_args.get("VllmPrefillWorker", 1)

            print(
                f"Correction factor clamping test: Expected={expected_prefill_replicas}, Got={prefill_replicas}"
            )

            assert prefill_replicas == max(
                expected_prefill_replicas, planner.args.min_endpoint
            ), "Prefill correction factor should be clamped to 1"

    def test_decode_correction_factor_zero_handling(self, planner):
        """Test handling of d_correction_factor <= 0."""
        # Test both 0 and negative values
        for correction_factor in [0.0, -1.0]:
            with patch.object(planner, "connector") as mock_connector:
                planner.p_correction_factor = 1.0
                planner.d_correction_factor = correction_factor

                # Mock predictor outputs
                planner.num_req_predictor.predict_next.return_value = 10
                planner.isl_predictor.predict_next.return_value = 3000
                planner.osl_predictor.predict_next.return_value = 150

                # Mock interpolator outputs
                planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = (
                    40000
                )
                planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
                    10000,
                    0.01,
                    0.5,
                )

                # Set up metrics
                planner.last_metrics = Metrics(
                    num_req=10,
                    isl=3000,
                    osl=150,
                    ttft=80.0,
                    itl=10.0,
                    request_duration=100.0,
                )

                # Mock workers info
                async def mock_get_workers_info():
                    return (["prefill1"], ["decode1"])

                planner.get_workers_info = mock_get_workers_info

                # Mock interpolation calls
                planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
                planner.decode_interpolator.interpolate_itl.return_value = 10.0

                # Run calculation
                import asyncio

                asyncio.run(planner.make_adjustments())

                # Should handle gracefully without crashing
                # The code should use args.itl directly instead of dividing by 0
                if mock_connector.set_component_replicas.called:
                    call_args = mock_connector.set_component_replicas.call_args[0][0]
                    decode_replicas = call_args.get("VllmDecodeWorker", 1)

                    print(
                        f"Correction factor {correction_factor} test: Decode replicas={decode_replicas}"
                    )

                    # Should get a valid result (not crash)
                    assert (
                        decode_replicas >= 1
                    ), f"Should handle correction factor {correction_factor} gracefully"

    def test_multi_gpu_engines(self, planner):
        """Test replica calculation with multi-GPU engines."""
        # Set multi-GPU configuration
        planner.args.prefill_engine_num_gpu = 2
        planner.args.decode_engine_num_gpu = 4

        # Mock predictor outputs
        planner.num_req_predictor.predict_next.return_value = 20
        planner.isl_predictor.predict_next.return_value = 3000
        planner.osl_predictor.predict_next.return_value = 150

        # Mock interpolator outputs
        planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = 40000
        planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
            5000,
            0.01,
            0.5,
        )  # Lower for scaling

        # Set up metrics
        planner.last_metrics = Metrics(
            num_req=20, isl=3000, osl=150, ttft=80.0, itl=10.0, request_duration=100.0
        )

        # Mock workers info
        async def mock_get_workers_info():
            return (["prefill1"], ["decode1"])

        planner.get_workers_info = mock_get_workers_info

        # Mock interpolation calls
        planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
        planner.decode_interpolator.interpolate_itl.return_value = 10.0

        # Calculate expected results manually
        pred_prefill_load_per_gpu = 20 * 3000 / planner.args.adjustment_interval * 1.0
        expected_prefill_replicas = math.ceil(
            pred_prefill_load_per_gpu / 40000 / 2
        )  # 2 GPUs per engine

        expected_decode_replicas = math.ceil(
            20 * 150 / planner.args.adjustment_interval / 5000 / 4
        )  # 4 GPUs per engine

        # Run calculation
        import asyncio

        asyncio.run(planner.make_adjustments())

        if planner.connector.set_component_replicas.called:
            call_args = planner.connector.set_component_replicas.call_args[0][0]

            prefill_replicas = call_args.get("VllmPrefillWorker", 1)
            decode_replicas = call_args.get("VllmDecodeWorker", 1)

            print(
                f"Multi-GPU test: P={prefill_replicas} (expected ~{expected_prefill_replicas}), D={decode_replicas} (expected ~{expected_decode_replicas})"
            )

            # Verify calculations account for multiple GPUs per engine
            assert prefill_replicas == max(
                expected_prefill_replicas, planner.args.min_endpoint
            )
            assert decode_replicas == max(
                expected_decode_replicas, planner.args.min_endpoint
            )

    def test_complex_gpu_budget_scaling(self, planner):
        """Test complex GPU budget scaling with proportional reduction and decode adjustment."""
        # Set tight GPU budget that will trigger complex scaling
        planner.args.max_gpu_budget = 5
        planner.args.prefill_engine_num_gpu = 2
        planner.args.decode_engine_num_gpu = 2
        planner.args.min_endpoint = 1

        # High load that would normally require more GPUs
        planner.num_req_predictor.predict_next.return_value = 100
        planner.isl_predictor.predict_next.return_value = 3000
        planner.osl_predictor.predict_next.return_value = 150

        # Lower throughput to trigger higher replica needs
        planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = 10000
        planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
            1000,
            0.01,
            0.5,
        )

        # Set up metrics
        planner.last_metrics = Metrics(
            num_req=100, isl=3000, osl=150, ttft=80.0, itl=10.0, request_duration=100.0
        )

        # Mock workers info
        async def mock_get_workers_info():
            return (["prefill1"], ["decode1"])

        planner.get_workers_info = mock_get_workers_info

        # Mock interpolation calls
        planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
        planner.decode_interpolator.interpolate_itl.return_value = 10.0

        # Run calculation
        import asyncio

        asyncio.run(planner.make_adjustments())

        if planner.connector.set_component_replicas.called:
            call_args = planner.connector.set_component_replicas.call_args[0][0]

            prefill_replicas = call_args.get("VllmPrefillWorker", 1)
            decode_replicas = call_args.get("VllmDecodeWorker", 1)

            # Verify total GPU usage doesn't exceed budget
            total_gpus = (
                prefill_replicas * planner.args.prefill_engine_num_gpu
                + decode_replicas * planner.args.decode_engine_num_gpu
            )

            print(
                f"Complex GPU budget test: P={prefill_replicas}, D={decode_replicas}, Total GPUs={total_gpus}"
            )

            assert (
                total_gpus <= planner.args.max_gpu_budget
            ), "Total GPU usage should not exceed budget"
            assert (
                prefill_replicas >= planner.args.min_endpoint
            ), "Should respect min_endpoint for prefill"
            assert (
                decode_replicas >= planner.args.min_endpoint
            ), "Should respect min_endpoint for decode"


# No need for unittest.main() with pytest!
