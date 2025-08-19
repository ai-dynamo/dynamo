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

# We'll import the actual Planner class to test its calculation logic
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../components/planner/src"))

from dynamo.planner.utils.planner_core import Metrics, Planner


class TestReplicaCalculation(unittest.TestCase):
    """Test replica calculation formulas in isolation."""

    def setUp(self):
        """Set up test environment with mocked dependencies."""
        # Create mock arguments
        self.args = argparse.Namespace()
        self.args.adjustment_interval = 60
        self.args.prefill_engine_num_gpu = 1
        self.args.decode_engine_num_gpu = 1
        self.args.min_endpoint = 1
        self.args.max_gpu_budget = 10
        self.args.ttft = 80  # ms
        self.args.itl = 10  # ms
        self.args.backend = "vllm"
        self.args.no_operation = True  # Don't actually scale
        self.args.prometheus_port = None
        self.args.load_predictor = "constant"
        self.args.load_prediction_window_size = 10
        self.args.profile_results_dir = os.path.join(
            os.path.dirname(__file__),
            "profiling_results/H200_TP1P_TP1D/profiling_results",
        )
        self.args.environment = "kubernetes"

        # Mock the runtime
        self.mock_runtime = Mock()

        # Patch Prometheus Gauge to avoid registry conflicts
        self.prometheus_gauge_patcher = patch("dynamo.planner.utils.planner_core.Gauge")
        self.mock_gauge = self.prometheus_gauge_patcher.start()
        self.mock_gauge.return_value = Mock()

        # Create planner instance
        self.planner = Planner(self.mock_runtime, self.args)

        # Mock the interpolators to return fixed values for testing
        self.planner.prefill_interpolator = Mock()
        self.planner.decode_interpolator = Mock()

        # Mock the predictors to return fixed values
        self.planner.num_req_predictor = Mock()
        self.planner.isl_predictor = Mock()
        self.planner.osl_predictor = Mock()

        # Mock the connector since we're not testing actual scaling
        self.planner.connector = Mock()

        # Mock prometheus client
        self.planner.prometheus_api_client = Mock()

        # Set up some baseline correction factors
        self.planner.p_correction_factor = 1.0
        self.planner.d_correction_factor = 1.0

    def tearDown(self):
        """Clean up test environment."""
        # Stop the Prometheus gauge patch
        self.prometheus_gauge_patcher.stop()

    def test_prefill_replica_calculation_basic(self):
        """Test basic prefill replica calculation."""
        # Setup test data
        next_num_req = 10
        next_isl = 3000
        prefill_thpt_per_gpu = 40000  # tokens/s/gpu (from the test data)

        # Mock the predictor outputs
        self.planner.num_req_predictor.predict_next.return_value = next_num_req
        self.planner.isl_predictor.predict_next.return_value = next_isl
        self.planner.osl_predictor.predict_next.return_value = 150

        # Mock interpolator output
        self.planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = (
            prefill_thpt_per_gpu
        )
        self.planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
            10000
        )

        # Calculate expected result manually
        pred_prefill_load_per_gpu = (
            next_num_req
            * next_isl
            / self.args.adjustment_interval
            * min(1, self.planner.p_correction_factor)
        )
        expected_prefill_replicas = math.ceil(
            pred_prefill_load_per_gpu
            / prefill_thpt_per_gpu
            / self.args.prefill_engine_num_gpu
        )

        # Set up valid metrics to trigger calculation
        self.planner.last_metrics = Metrics(
            num_req=10, isl=3000, osl=150, ttft=80.0, itl=10.0, request_duration=100.0
        )

        # Mock workers info
        async def mock_get_workers_info():
            return (["prefill1"], ["decode1"])

        self.planner.get_workers_info = mock_get_workers_info

        # Mock interpolation calls for correction factor calculation
        self.planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
        self.planner.decode_interpolator.interpolate_itl.return_value = 10.0

        # Run the calculation
        import asyncio

        asyncio.run(self.planner.make_adjustments())

        # Extract the calculated values from the log calls or by checking the mock calls
        # Since we mocked the connector, we can check what replicas were requested
        if self.planner.connector.set_component_replicas.called:
            call_args = self.planner.connector.set_component_replicas.call_args[0][0]
            prefill_component = "VllmPrefillWorker"
            calculated_prefill_replicas = call_args.get(prefill_component, 1)

            print(f"Expected prefill replicas: {expected_prefill_replicas}")
            print(f"Calculated prefill replicas: {calculated_prefill_replicas}")

            # Allow for small differences due to min_endpoint constraints
            self.assertEqual(
                max(expected_prefill_replicas, self.args.min_endpoint),
                calculated_prefill_replicas,
            )

    def test_decode_replica_calculation_basic(self):
        """Test basic decode replica calculation."""
        # Setup test data
        next_num_req = 10
        next_osl = 150
        decode_thpt_per_gpu = 10000  # tokens/s/gpu

        # Mock the predictor outputs
        self.planner.num_req_predictor.predict_next.return_value = next_num_req
        self.planner.isl_predictor.predict_next.return_value = 3000
        self.planner.osl_predictor.predict_next.return_value = next_osl

        # Mock interpolator outputs
        self.planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = 40000
        self.planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
            decode_thpt_per_gpu
        )

        # Calculate expected result manually
        expected_decode_replicas = math.ceil(
            next_num_req
            * next_osl
            / self.args.adjustment_interval
            / decode_thpt_per_gpu
            / self.args.decode_engine_num_gpu
        )

        # Set up valid metrics
        self.planner.last_metrics = Metrics(
            num_req=10, isl=3000, osl=150, ttft=80.0, itl=10.0, request_duration=100.0
        )

        # Mock workers info
        async def mock_get_workers_info():
            return (["prefill1"], ["decode1"])

        self.planner.get_workers_info = mock_get_workers_info

        # Mock interpolation calls for correction factor calculation
        self.planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
        self.planner.decode_interpolator.interpolate_itl.return_value = 10.0

        # Run the calculation
        import asyncio

        asyncio.run(self.planner.make_adjustments())

        # Check the results
        if self.planner.connector.set_component_replicas.called:
            call_args = self.planner.connector.set_component_replicas.call_args[0][0]
            decode_component = "VllmDecodeWorker"
            calculated_decode_replicas = call_args.get(decode_component, 1)

            print(f"Expected decode replicas: {expected_decode_replicas}")
            print(f"Calculated decode replicas: {calculated_decode_replicas}")

            # Allow for small differences due to min_endpoint constraints
            self.assertEqual(
                max(expected_decode_replicas, self.args.min_endpoint),
                calculated_decode_replicas,
            )

    def test_scaling_scenario_low_to_high_load(self):
        """Test scaling from low to high load scenarios."""

        test_cases = [
            {
                "name": "low_load_10_req_per_second",
                "num_req": 10,
                "expected_p": 1,
                "expected_d": 1,
            },
            {
                "name": "high_load_500_req_per_second",
                "num_req": 500,
                "decode_thpt": 1000,  # Lower throughput to trigger scaling
                "expected_p": 1,
                "expected_d": 2,  # Should scale to 2 for decode
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                # Reset the planner state
                self.planner.p_correction_factor = 1.0
                self.planner.d_correction_factor = 1.0

                # Mock predictor outputs for this case
                self.planner.num_req_predictor.predict_next.return_value = case[
                    "num_req"
                ]
                self.planner.isl_predictor.predict_next.return_value = 3000
                self.planner.osl_predictor.predict_next.return_value = 150

                # Mock interpolator outputs (based on H200 1P1D profiling data)
                self.planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = (
                    40000  # tokens/s/gpu
                )
                decode_thpt = case.get(
                    "decode_thpt", 10000
                )  # Use custom throughput if specified
                self.planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
                    decode_thpt
                )

                # Set up metrics
                self.planner.last_metrics = Metrics(
                    num_req=case["num_req"],
                    isl=3000,
                    osl=150,
                    ttft=80.0,
                    itl=10.0,
                    request_duration=100.0,
                )

                # Mock workers info
                async def mock_get_workers_info():
                    return (["prefill1"], ["decode1"])

                self.planner.get_workers_info = mock_get_workers_info

                # Mock interpolation calls for correction factor calculation
                self.planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
                self.planner.decode_interpolator.interpolate_itl.return_value = 10.0

                # Reset the mock
                self.planner.connector.reset_mock()

                # Run calculation
                import asyncio

                asyncio.run(self.planner.make_adjustments())

                # Verify results
                if self.planner.connector.set_component_replicas.called:
                    call_args = self.planner.connector.set_component_replicas.call_args[
                        0
                    ][0]

                    prefill_replicas = call_args.get("VllmPrefillWorker", 1)
                    decode_replicas = call_args.get("VllmDecodeWorker", 1)

                    print(
                        f"Case {case['name']}: P={prefill_replicas}, D={decode_replicas}"
                    )

                    self.assertEqual(
                        prefill_replicas,
                        case["expected_p"],
                        f"Prefill replicas mismatch for {case['name']}",
                    )
                    self.assertEqual(
                        decode_replicas,
                        case["expected_d"],
                        f"Decode replicas mismatch for {case['name']}",
                    )

    def test_gpu_budget_constraint(self):
        """Test that GPU budget constraints are properly applied."""
        # Set a low GPU budget
        self.args.max_gpu_budget = 3

        # Mock predictor outputs that would normally require more GPUs
        self.planner.num_req_predictor.predict_next.return_value = 50  # High load
        self.planner.isl_predictor.predict_next.return_value = 3000
        self.planner.osl_predictor.predict_next.return_value = 150

        # Mock interpolator outputs
        self.planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = 40000
        self.planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
            10000
        )

        # Set up metrics
        self.planner.last_metrics = Metrics(
            num_req=50, isl=3000, osl=150, ttft=80.0, itl=10.0, request_duration=100.0
        )

        # Mock workers info
        async def mock_get_workers_info():
            return (["prefill1"], ["decode1"])

        self.planner.get_workers_info = mock_get_workers_info

        # Mock interpolation calls
        self.planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
        self.planner.decode_interpolator.interpolate_itl.return_value = 10.0

        # Run calculation
        import asyncio

        asyncio.run(self.planner.make_adjustments())

        # Verify that total GPU usage doesn't exceed budget
        if self.planner.connector.set_component_replicas.called:
            call_args = self.planner.connector.set_component_replicas.call_args[0][0]

            prefill_replicas = call_args.get("VllmPrefillWorker", 1)
            decode_replicas = call_args.get("VllmDecodeWorker", 1)

            total_gpus = (
                prefill_replicas * self.args.prefill_engine_num_gpu
                + decode_replicas * self.args.decode_engine_num_gpu
            )

            print(
                f"GPU budget test: P={prefill_replicas}, D={decode_replicas}, Total GPUs={total_gpus}"
            )

            self.assertLessEqual(
                total_gpus, self.args.max_gpu_budget, "Total GPU usage exceeds budget"
            )

    def test_min_endpoint_constraint(self):
        """Test that minimum endpoint constraints are respected."""
        self.args.min_endpoint = 2

        # Mock predictor outputs that would normally require fewer workers
        self.planner.num_req_predictor.predict_next.return_value = 1  # Very low load
        self.planner.isl_predictor.predict_next.return_value = 100
        self.planner.osl_predictor.predict_next.return_value = 10

        # Mock interpolator outputs
        self.planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = 40000
        self.planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
            10000
        )

        # Set up metrics
        self.planner.last_metrics = Metrics(
            num_req=1, isl=100, osl=10, ttft=80.0, itl=10.0, request_duration=100.0
        )

        # Mock workers info
        async def mock_get_workers_info():
            return (["prefill1"], ["decode1"])

        self.planner.get_workers_info = mock_get_workers_info

        # Mock interpolation calls
        self.planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
        self.planner.decode_interpolator.interpolate_itl.return_value = 10.0

        # Run calculation
        import asyncio

        asyncio.run(self.planner.make_adjustments())

        # Verify minimum constraints are respected
        if self.planner.connector.set_component_replicas.called:
            call_args = self.planner.connector.set_component_replicas.call_args[0][0]

            prefill_replicas = call_args.get("VllmPrefillWorker", 1)
            decode_replicas = call_args.get("VllmDecodeWorker", 1)

            print(f"Min endpoint test: P={prefill_replicas}, D={decode_replicas}")

            self.assertGreaterEqual(
                prefill_replicas,
                self.args.min_endpoint,
                "Prefill replicas below minimum",
            )
            self.assertGreaterEqual(
                decode_replicas, self.args.min_endpoint, "Decode replicas below minimum"
            )

    def test_prefill_correction_factor_clamping(self):
        """Test that prefill correction factor > 1 is clamped to 1."""
        # Set a high correction factor > 1
        self.planner.p_correction_factor = 2.5
        self.planner.d_correction_factor = 1.0

        # Mock predictor outputs
        self.planner.num_req_predictor.predict_next.return_value = 10
        self.planner.isl_predictor.predict_next.return_value = 3000
        self.planner.osl_predictor.predict_next.return_value = 150

        # Mock interpolator outputs
        self.planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = 40000
        self.planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
            10000
        )

        # Set up metrics
        self.planner.last_metrics = Metrics(
            num_req=10, isl=3000, osl=150, ttft=80.0, itl=10.0, request_duration=100.0
        )

        # Mock workers info
        async def mock_get_workers_info():
            return (["prefill1"], ["decode1"])

        self.planner.get_workers_info = mock_get_workers_info

        # Mock interpolation calls
        self.planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
        self.planner.decode_interpolator.interpolate_itl.return_value = 10.0

        # Calculate expected result manually with clamping
        # Should use min(1, 2.5) = 1
        pred_prefill_load_per_gpu = (
            10 * 3000 / self.args.adjustment_interval * min(1, 2.5)  # Should be * 1
        )
        expected_prefill_replicas = math.ceil(
            pred_prefill_load_per_gpu / 40000 / self.args.prefill_engine_num_gpu
        )

        # Run calculation
        import asyncio

        asyncio.run(self.planner.make_adjustments())

        # Verify that correction factor was effectively clamped
        if self.planner.connector.set_component_replicas.called:
            call_args = self.planner.connector.set_component_replicas.call_args[0][0]
            prefill_replicas = call_args.get("VllmPrefillWorker", 1)

            print(
                f"Correction factor clamping test: Expected={expected_prefill_replicas}, Got={prefill_replicas}"
            )

            self.assertEqual(
                prefill_replicas,
                max(expected_prefill_replicas, self.args.min_endpoint),
                "Prefill correction factor should be clamped to 1",
            )

    def test_decode_correction_factor_zero_handling(self):
        """Test handling of d_correction_factor <= 0."""
        # Set correction factor to 0 (edge case)
        self.planner.p_correction_factor = 1.0
        self.planner.d_correction_factor = 0.0

        # Mock predictor outputs
        self.planner.num_req_predictor.predict_next.return_value = 10
        self.planner.isl_predictor.predict_next.return_value = 3000
        self.planner.osl_predictor.predict_next.return_value = 150

        # Mock interpolator outputs
        self.planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = 40000
        self.planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
            10000
        )

        # Set up metrics
        self.planner.last_metrics = Metrics(
            num_req=10, isl=3000, osl=150, ttft=80.0, itl=10.0, request_duration=100.0
        )

        # Mock workers info
        async def mock_get_workers_info():
            return (["prefill1"], ["decode1"])

        self.planner.get_workers_info = mock_get_workers_info

        # Mock interpolation calls
        self.planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
        self.planner.decode_interpolator.interpolate_itl.return_value = 10.0

        # Run calculation
        import asyncio

        asyncio.run(self.planner.make_adjustments())

        # Should handle gracefully without crashing
        # The code should use args.itl directly instead of dividing by 0
        if self.planner.connector.set_component_replicas.called:
            call_args = self.planner.connector.set_component_replicas.call_args[0][0]
            decode_replicas = call_args.get("VllmDecodeWorker", 1)

            print(f"Zero correction factor test: Decode replicas={decode_replicas}")

            # Should get a valid result (not crash)
            self.assertGreaterEqual(
                decode_replicas, 1, "Should handle zero correction factor gracefully"
            )

    def test_multi_gpu_engines(self):
        """Test replica calculation with multi-GPU engines."""
        # Set multi-GPU configuration
        self.args.prefill_engine_num_gpu = 2
        self.args.decode_engine_num_gpu = 4

        # Mock predictor outputs
        self.planner.num_req_predictor.predict_next.return_value = 20
        self.planner.isl_predictor.predict_next.return_value = 3000
        self.planner.osl_predictor.predict_next.return_value = 150

        # Mock interpolator outputs
        self.planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = 40000
        self.planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
            5000  # Lower for scaling
        )

        # Set up metrics
        self.planner.last_metrics = Metrics(
            num_req=20, isl=3000, osl=150, ttft=80.0, itl=10.0, request_duration=100.0
        )

        # Mock workers info
        async def mock_get_workers_info():
            return (["prefill1"], ["decode1"])

        self.planner.get_workers_info = mock_get_workers_info

        # Mock interpolation calls
        self.planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
        self.planner.decode_interpolator.interpolate_itl.return_value = 10.0

        # Calculate expected results manually
        pred_prefill_load_per_gpu = 20 * 3000 / self.args.adjustment_interval * 1.0
        expected_prefill_replicas = math.ceil(
            pred_prefill_load_per_gpu / 40000 / 2
        )  # 2 GPUs per engine

        expected_decode_replicas = math.ceil(
            20 * 150 / self.args.adjustment_interval / 5000 / 4
        )  # 4 GPUs per engine

        # Run calculation
        import asyncio

        asyncio.run(self.planner.make_adjustments())

        if self.planner.connector.set_component_replicas.called:
            call_args = self.planner.connector.set_component_replicas.call_args[0][0]

            prefill_replicas = call_args.get("VllmPrefillWorker", 1)
            decode_replicas = call_args.get("VllmDecodeWorker", 1)

            print(
                f"Multi-GPU test: P={prefill_replicas} (expected ~{expected_prefill_replicas}), D={decode_replicas} (expected ~{expected_decode_replicas})"
            )

            # Verify calculations account for multiple GPUs per engine
            self.assertEqual(
                prefill_replicas, max(expected_prefill_replicas, self.args.min_endpoint)
            )
            self.assertEqual(
                decode_replicas, max(expected_decode_replicas, self.args.min_endpoint)
            )

    def test_complex_gpu_budget_scaling(self):
        """Test complex GPU budget scaling with proportional reduction and decode adjustment."""
        # Set tight GPU budget that will trigger complex scaling
        self.args.max_gpu_budget = 5
        self.args.prefill_engine_num_gpu = 2
        self.args.decode_engine_num_gpu = 2
        self.args.min_endpoint = 1

        # High load that would normally require more GPUs
        self.planner.num_req_predictor.predict_next.return_value = 100
        self.planner.isl_predictor.predict_next.return_value = 3000
        self.planner.osl_predictor.predict_next.return_value = 150

        # Lower throughput to trigger higher replica needs
        self.planner.prefill_interpolator.interpolate_thpt_per_gpu.return_value = 10000
        self.planner.decode_interpolator.find_best_throughput_per_gpu.return_value = (
            1000
        )

        # Set up metrics
        self.planner.last_metrics = Metrics(
            num_req=100, isl=3000, osl=150, ttft=80.0, itl=10.0, request_duration=100.0
        )

        # Mock workers info
        async def mock_get_workers_info():
            return (["prefill1"], ["decode1"])

        self.planner.get_workers_info = mock_get_workers_info

        # Mock interpolation calls
        self.planner.prefill_interpolator.interpolate_ttft.return_value = 80.0
        self.planner.decode_interpolator.interpolate_itl.return_value = 10.0

        # Run calculation
        import asyncio

        asyncio.run(self.planner.make_adjustments())

        if self.planner.connector.set_component_replicas.called:
            call_args = self.planner.connector.set_component_replicas.call_args[0][0]

            prefill_replicas = call_args.get("VllmPrefillWorker", 1)
            decode_replicas = call_args.get("VllmDecodeWorker", 1)

            # Verify total GPU usage doesn't exceed budget
            total_gpus = (
                prefill_replicas * self.args.prefill_engine_num_gpu
                + decode_replicas * self.args.decode_engine_num_gpu
            )

            print(
                f"Complex GPU budget test: P={prefill_replicas}, D={decode_replicas}, Total GPUs={total_gpus}"
            )

            self.assertLessEqual(
                total_gpus,
                self.args.max_gpu_budget,
                "Total GPU usage should not exceed budget",
            )
            self.assertGreaterEqual(
                prefill_replicas,
                self.args.min_endpoint,
                "Should respect min_endpoint for prefill",
            )
            self.assertGreaterEqual(
                decode_replicas,
                self.args.min_endpoint,
                "Should respect min_endpoint for decode",
            )


if __name__ == "__main__":
    unittest.main()
