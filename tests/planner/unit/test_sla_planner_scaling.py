# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import math
import os
from unittest.mock import Mock, patch

import pytest

from dynamo.planner.utils.exceptions import DeploymentValidationError
from dynamo.planner.utils.planner_core import (
    DecodePlanner,
    PlannerSharedState,
    PrefillPlanner,
    _initialize_gpu_counts,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.fixture(autouse=True)
def mock_prometheus_metrics():
    with patch("dynamo.planner.utils.planner_core.Gauge") as mock_gauge:
        mock_gauge.return_value = Mock()
        yield


def _build_args():
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
    return args


def _build_prometheus_client(samples):
    client = Mock()
    client.get_avg_time_to_first_token.side_effect = [
        s["ttft_ms"] / 1000 for s in samples
    ]
    client.get_avg_inter_token_latency.side_effect = [
        s["itl_ms"] / 1000 for s in samples
    ]
    client.get_avg_request_count.side_effect = [s["num_req"] for s in samples]
    client.get_avg_request_duration.side_effect = [
        s["request_duration"] for s in samples
    ]
    client.get_avg_input_sequence_tokens.side_effect = [s["isl"] for s in samples]
    client.get_avg_output_sequence_tokens.side_effect = [s["osl"] for s in samples]
    return client


def _build_planners(args, prometheus_client):
    shared_state = PlannerSharedState()
    prefill_planner = PrefillPlanner(None, args, shared_state=shared_state)
    decode_planner = DecodePlanner(None, args, shared_state=shared_state)
    prefill_planner.prometheus_api_client = prometheus_client
    decode_planner.prometheus_api_client = prometheus_client
    prefill_planner.model_name = "test-model"
    decode_planner.model_name = "test-model"

    async def mock_get_workers_info(require_prefill=True, require_decode=True):
        return (
            1 if require_prefill else 0,
            1 if require_decode else 0,
            True,  # is_stable
        )

    prefill_planner.get_workers_info = mock_get_workers_info
    decode_planner.get_workers_info = mock_get_workers_info
    return prefill_planner, decode_planner, shared_state


def _expected_prefill(args, prefill_planner, sample):
    pred_prefill_throughput = (
        sample["num_req"] * sample["isl"] / args.adjustment_interval
    )
    thpt_per_gpu = prefill_planner.prefill_interpolator.interpolate_thpt_per_gpu(
        sample["isl"]
    )
    expected = math.ceil(
        pred_prefill_throughput / thpt_per_gpu / args.prefill_engine_num_gpu
    )
    return max(expected, args.min_endpoint)


def _expected_decode(args, decode_planner, sample):
    (
        pred_decode_thpt_per_gpu,
        _,
        _,
    ) = decode_planner.decode_interpolator.find_best_throughput_per_gpu(
        itl=args.itl, context_length=sample["isl"] + sample["osl"] / 2
    )
    pred_decode_throughput = (
        sample["num_req"] * sample["osl"] / args.adjustment_interval
    )
    expected = math.ceil(
        pred_decode_throughput / pred_decode_thpt_per_gpu / args.decode_engine_num_gpu
    )
    return max(expected, args.min_endpoint)


def _run_interval(prefill_planner, decode_planner, shared_state):
    asyncio.run(
        prefill_planner.observe_metrics(require_prefill=True, require_decode=True)
    )
    decode_planner.update_predictors_from_metrics(shared_state.last_metrics)
    next_num_p = prefill_planner.plan_adjustment()
    next_num_d = decode_planner.plan_adjustment()
    return next_num_p, next_num_d


def test_disagg_scale_up():
    args = _build_args()
    samples = [
        {
            "num_req": 10,
            "isl": 3000,
            "osl": 150,
            "ttft_ms": 400.0,
            "itl_ms": 30.0,
            "request_duration": 20.0,
        },
        {
            "num_req": 5000,
            "isl": 3000,
            "osl": 150,
            "ttft_ms": 400.0,
            "itl_ms": 30.0,
            "request_duration": 20.0,
        },
    ]
    client = _build_prometheus_client(samples)
    prefill_planner, decode_planner, shared_state = _build_planners(args, client)

    low_p, low_d = _run_interval(prefill_planner, decode_planner, shared_state)
    high_p, high_d = _run_interval(prefill_planner, decode_planner, shared_state)

    assert low_p == _expected_prefill(args, prefill_planner, samples[0])
    assert low_d == _expected_decode(args, decode_planner, samples[0])
    assert high_p == _expected_prefill(args, prefill_planner, samples[1])
    assert high_d == _expected_decode(args, decode_planner, samples[1])
    assert high_p > low_p
    assert high_d > low_d


def test_disagg_scale_down():
    args = _build_args()
    samples = [
        {
            "num_req": 5000,
            "isl": 3000,
            "osl": 150,
            "ttft_ms": 400.0,
            "itl_ms": 30.0,
            "request_duration": 20.0,
        },
        {
            "num_req": 10,
            "isl": 3000,
            "osl": 150,
            "ttft_ms": 400.0,
            "itl_ms": 30.0,
            "request_duration": 20.0,
        },
    ]
    client = _build_prometheus_client(samples)
    prefill_planner, decode_planner, shared_state = _build_planners(args, client)

    high_p, high_d = _run_interval(prefill_planner, decode_planner, shared_state)
    low_p, low_d = _run_interval(prefill_planner, decode_planner, shared_state)

    assert high_p == _expected_prefill(args, prefill_planner, samples[0])
    assert high_d == _expected_decode(args, decode_planner, samples[0])
    assert low_p == _expected_prefill(args, prefill_planner, samples[1])
    assert low_d == _expected_decode(args, decode_planner, samples[1])
    assert low_p < high_p
    assert low_d < high_d


# Tests for _initialize_gpu_counts
class TestInitializeGpuCounts:
    def test_kubernetes_mode_reads_from_dgd(self):
        """Test that GPU counts are read from DGD in Kubernetes mode"""
        args = argparse.Namespace()
        args.prefill_engine_num_gpu = None
        args.decode_engine_num_gpu = None

        connector = Mock()
        connector.get_gpu_counts = Mock(return_value=(2, 4))

        _initialize_gpu_counts(
            args, connector, require_prefill=True, require_decode=True
        )

        assert args.prefill_engine_num_gpu == 2
        assert args.decode_engine_num_gpu == 4
        connector.get_gpu_counts.assert_called_once_with(
            require_prefill=True, require_decode=True
        )

    def test_kubernetes_mode_prefill_only(self):
        """Test GPU count initialization for prefill-only mode"""
        args = argparse.Namespace()
        args.prefill_engine_num_gpu = None
        args.decode_engine_num_gpu = None

        connector = Mock()
        connector.get_gpu_counts = Mock(return_value=(2, 0))

        _initialize_gpu_counts(
            args, connector, require_prefill=True, require_decode=False
        )

        assert args.prefill_engine_num_gpu == 2
        assert args.decode_engine_num_gpu == 0
        connector.get_gpu_counts.assert_called_once_with(
            require_prefill=True, require_decode=False
        )

    def test_virtual_mode_uses_cli_args(self):
        """Test that GPU counts come from CLI args in virtual mode"""
        args = argparse.Namespace()
        args.prefill_engine_num_gpu = 2
        args.decode_engine_num_gpu = 4

        # Virtual connector doesn't have get_gpu_counts method
        connector = Mock(spec=[])

        _initialize_gpu_counts(
            args, connector, require_prefill=True, require_decode=True
        )

        # Values should remain unchanged
        assert args.prefill_engine_num_gpu == 2
        assert args.decode_engine_num_gpu == 4

    def test_virtual_mode_missing_prefill_raises_error(self):
        """Test that missing prefill GPU flag raises error in virtual mode"""
        args = argparse.Namespace()
        args.prefill_engine_num_gpu = None
        args.decode_engine_num_gpu = 4

        connector = Mock(spec=[])

        with pytest.raises(DeploymentValidationError) as exc_info:
            _initialize_gpu_counts(
                args, connector, require_prefill=True, require_decode=True
            )

        assert "prefill-engine-num-gpu" in str(exc_info.value)

    def test_virtual_mode_missing_decode_raises_error(self):
        """Test that missing decode GPU flag raises error in virtual mode"""
        args = argparse.Namespace()
        args.prefill_engine_num_gpu = 2
        args.decode_engine_num_gpu = None

        connector = Mock(spec=[])

        with pytest.raises(DeploymentValidationError) as exc_info:
            _initialize_gpu_counts(
                args, connector, require_prefill=True, require_decode=True
            )

        assert "decode-engine-num-gpu" in str(exc_info.value)

    def test_virtual_mode_missing_both_raises_error_with_both_messages(self):
        """Test that missing both GPU flags shows both error messages"""
        args = argparse.Namespace()
        args.prefill_engine_num_gpu = None
        args.decode_engine_num_gpu = None

        connector = Mock(spec=[])

        with pytest.raises(DeploymentValidationError) as exc_info:
            _initialize_gpu_counts(
                args, connector, require_prefill=True, require_decode=True
            )

        assert len(exc_info.value.errors) == 2

    def test_virtual_mode_decode_only_no_prefill_error(self):
        """Test decode-only mode doesn't require prefill GPU flag"""
        args = argparse.Namespace()
        args.prefill_engine_num_gpu = None
        args.decode_engine_num_gpu = 4

        connector = Mock(spec=[])

        # Should not raise - prefill not required
        _initialize_gpu_counts(
            args, connector, require_prefill=False, require_decode=True
        )

        assert args.decode_engine_num_gpu == 4

    def test_kubernetes_mode_fallback_to_cli_on_dgd_error(self):
        """Test that K8s mode falls back to CLI flags when DGD parsing fails"""
        args = argparse.Namespace()
        args.prefill_engine_num_gpu = 2
        args.decode_engine_num_gpu = 4

        connector = Mock()
        connector.get_gpu_counts = Mock(
            side_effect=ValueError("No GPU count specified")
        )

        _initialize_gpu_counts(
            args, connector, require_prefill=True, require_decode=True
        )

        # Should use CLI flag values after fallback
        assert args.prefill_engine_num_gpu == 2
        assert args.decode_engine_num_gpu == 4

    def test_kubernetes_mode_fallback_missing_cli_flags_raises_error(self):
        """Test that K8s fallback raises error when CLI flags are also missing"""
        args = argparse.Namespace()
        args.prefill_engine_num_gpu = None
        args.decode_engine_num_gpu = None

        connector = Mock()
        connector.get_gpu_counts = Mock(
            side_effect=ValueError("No GPU count specified")
        )

        with pytest.raises(DeploymentValidationError) as exc_info:
            _initialize_gpu_counts(
                args, connector, require_prefill=True, require_decode=True
            )

        assert len(exc_info.value.errors) == 2

    def test_kubernetes_mode_fallback_partial_cli_flags(self):
        """Test K8s fallback with only one CLI flag provided"""
        args = argparse.Namespace()
        args.prefill_engine_num_gpu = 2
        args.decode_engine_num_gpu = None

        connector = Mock()
        connector.get_gpu_counts = Mock(
            side_effect=ValueError("No GPU count specified")
        )

        with pytest.raises(DeploymentValidationError) as exc_info:
            _initialize_gpu_counts(
                args, connector, require_prefill=True, require_decode=True
            )

        assert "decode-engine-num-gpu" in str(exc_info.value)


# Tests for dryrun GPU defaults
class TestDryrunGpuDefaults:
    def test_dryrun_defaults_gpu_counts_when_none(self):
        """Test that dryrun sets default GPU counts of 1 when None"""
        from dynamo.planner.utils.dryrun import run_sla_planner_dryrun

        args = _build_args()
        args.prefill_engine_num_gpu = None
        args.decode_engine_num_gpu = None
        args.dataset = "nonexistent.jsonl"  # Will fail but we check args first

        # The function will set defaults before trying to load dataset
        try:
            run_sla_planner_dryrun(args)
        except (FileNotFoundError, ValueError):
            pass  # Expected - dataset doesn't exist

        assert args.prefill_engine_num_gpu == 1
        assert args.decode_engine_num_gpu == 1

    def test_dryrun_preserves_cli_gpu_counts(self):
        """Test that dryrun preserves GPU counts provided via CLI"""
        from dynamo.planner.utils.dryrun import run_sla_planner_dryrun

        args = _build_args()
        args.prefill_engine_num_gpu = 2
        args.decode_engine_num_gpu = 4
        args.dataset = "nonexistent.jsonl"

        try:
            run_sla_planner_dryrun(args)
        except (FileNotFoundError, ValueError):
            pass

        assert args.prefill_engine_num_gpu == 2
        assert args.decode_engine_num_gpu == 4
