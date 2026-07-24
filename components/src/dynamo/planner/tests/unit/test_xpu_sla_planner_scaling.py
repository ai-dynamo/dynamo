# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
XPU (Intel B60) unit tests for SLA planner scaling logic.

These tests mirror the H200 tests in test_sla_planner_scaling.py but use
the B60_TP1P_TP1D profiling results, validating that the planner works
correctly with XPU performance characteristics (lower throughput, higher
latency compared to NVIDIA GPUs).
"""

import argparse
import asyncio
import math
import os
from unittest.mock import Mock, patch

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.budget import _initialize_gpu_counts
from dynamo.planner.core.decode import DecodePlanner
from dynamo.planner.core.prefill import PrefillPlanner
from dynamo.planner.core.state import PlannerSharedState
from dynamo.planner.errors import DeploymentValidationError
from dynamo.planner.offline.dryrun import run_sla_planner_dryrun

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

# B60 XPU profiling data directory
B60_PROFILE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "profiling_results",
    "B60_TP1P_TP1D",
)


@pytest.fixture(autouse=True)
def mock_prometheus_metrics():
    with patch("dynamo.planner.monitoring.planner_metrics.Gauge") as mock_gauge:
        mock_gauge.return_value = Mock()
        yield


def _build_xpu_config(**overrides):
    """Build a PlannerConfig using B60 XPU profiling data.

    B60 XPU characteristics vs H200:
    - Prefill throughput: ~5900 tok/s/gpu at ISL=3000 (vs ~50000 on H200)
    - TTFT: ~529ms at ISL=3000 (vs ~63ms on H200)
    - ITL: ~35-39ms (vs ~4-47ms on H200)
    - Max KV tokens: 55000 (vs 945029 on H200)
    """
    defaults = dict(
        throughput_adjustment_interval=60,
        prefill_engine_num_gpu=1,
        decode_engine_num_gpu=1,
        min_endpoint=1,
        max_gpu_budget=-1,
        ttft=1000.0,  # 1000ms SLA (B60 TTFT at ISL=3000 is ~529ms)
        itl=50.0,  # 50ms SLA (B60 ITL is ~35-39ms)
        backend="vllm",
        no_operation=True,
        no_correction=True,
        metric_pulling_prometheus_endpoint="http://localhost:9090",
        metric_reporting_prometheus_port=0,
        load_predictor="constant",
        load_predictor_warmup_trace=None,
        load_predictor_log1p=False,
        profile_results_dir=B60_PROFILE_DIR,
        environment="kubernetes",
        namespace="test-namespace",
        mode="disagg",
        enable_throughput_scaling=True,
        enable_load_scaling=False,
    )
    defaults.update(overrides)
    return PlannerConfig.model_construct(**defaults)


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


def _build_planners(config, prometheus_client):
    shared_state = PlannerSharedState()
    prefill_planner = PrefillPlanner(None, config, shared_state=shared_state)
    decode_planner = DecodePlanner(None, config, shared_state=shared_state)
    prefill_planner.prometheus_traffic_client = prometheus_client
    decode_planner.prometheus_traffic_client = prometheus_client
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


def _expected_prefill(config, prefill_planner, sample):
    pred_prefill_throughput = (
        sample["num_req"] * sample["isl"] / config.throughput_adjustment_interval
    )
    thpt_per_gpu = prefill_planner.prefill_interpolator.interpolate_thpt_per_gpu(
        sample["isl"]
    )
    expected = math.ceil(
        pred_prefill_throughput / thpt_per_gpu / config.prefill_engine_num_gpu
    )
    return max(expected, config.min_endpoint)


def _expected_decode(config, decode_planner, sample):
    (
        pred_decode_thpt_per_gpu,
        _,
        _,
    ) = decode_planner.decode_interpolator.find_best_throughput_per_gpu(
        itl=config.itl, context_length=sample["isl"] + sample["osl"] / 2
    )
    pred_decode_throughput = (
        sample["num_req"] * sample["osl"] / config.throughput_adjustment_interval
    )
    expected = math.ceil(
        pred_decode_throughput / pred_decode_thpt_per_gpu / config.decode_engine_num_gpu
    )
    return max(expected, config.min_endpoint)


def _run_interval(prefill_planner, decode_planner, shared_state):
    asyncio.run(
        prefill_planner.observe_traffic_stats(require_prefill=True, require_decode=True)
    )
    decode_planner.update_predictors_from_metrics(shared_state.last_metrics)
    next_num_p = prefill_planner.plan_adjustment()
    next_num_d = decode_planner.plan_adjustment()
    return next_num_p, next_num_d


# ── XPU Disaggregated Scaling Tests ─────────────────────────────────


def test_xpu_disagg_scale_up():
    """Test scale-up with B60 XPU profiling data.

    Verifies that the planner correctly computes higher replica counts
    when traffic increases, using B60 performance characteristics.
    """
    config = _build_xpu_config()
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
    prefill_planner, decode_planner, shared_state = _build_planners(config, client)

    low_p, low_d = _run_interval(prefill_planner, decode_planner, shared_state)
    high_p, high_d = _run_interval(prefill_planner, decode_planner, shared_state)

    assert low_p == _expected_prefill(config, prefill_planner, samples[0])
    assert low_d == _expected_decode(config, decode_planner, samples[0])
    assert high_p == _expected_prefill(config, prefill_planner, samples[1])
    assert high_d == _expected_decode(config, decode_planner, samples[1])
    assert high_p > low_p
    assert high_d > low_d


def test_xpu_disagg_scale_down():
    """Test scale-down with B60 XPU profiling data.

    Verifies that the planner correctly reduces replica counts
    when traffic decreases, using B60 performance characteristics.
    """
    config = _build_xpu_config()
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
    prefill_planner, decode_planner, shared_state = _build_planners(config, client)

    high_p, high_d = _run_interval(prefill_planner, decode_planner, shared_state)
    low_p, low_d = _run_interval(prefill_planner, decode_planner, shared_state)

    assert high_p == _expected_prefill(config, prefill_planner, samples[0])
    assert high_d == _expected_decode(config, decode_planner, samples[0])
    assert low_p == _expected_prefill(config, prefill_planner, samples[1])
    assert low_d == _expected_decode(config, decode_planner, samples[1])
    assert low_p < high_p
    assert low_d < high_d


def test_xpu_disagg_higher_replicas_than_h200():
    """Verify XPU (B60) requires more replicas than H200 for the same workload.

    Since B60 has lower throughput per GPU, the planner should compute
    higher replica counts for the same traffic load.
    """
    H200_PROFILE_DIR = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "profiling_results",
        "H200_TP1P_TP1D",
    )
    sample = {
        "num_req": 1000,
        "isl": 3000,
        "osl": 150,
        "ttft_ms": 400.0,
        "itl_ms": 30.0,
        "request_duration": 20.0,
    }

    # Run XPU (B60) planner
    xpu_config = _build_xpu_config()
    xpu_client = _build_prometheus_client([sample])
    xpu_prefill, xpu_decode, xpu_state = _build_planners(xpu_config, xpu_client)
    xpu_p, xpu_d = _run_interval(xpu_prefill, xpu_decode, xpu_state)

    expected_xpu_p = _expected_prefill(xpu_config, xpu_prefill, sample)
    expected_xpu_d = _expected_decode(xpu_config, xpu_decode, sample)
    assert xpu_p == expected_xpu_p
    assert xpu_d == expected_xpu_d

    # Run H200 planner with the same workload
    h200_config = _build_xpu_config(profile_results_dir=H200_PROFILE_DIR)
    h200_client = _build_prometheus_client([sample])
    h200_prefill, h200_decode, h200_state = _build_planners(h200_config, h200_client)
    h200_p, h200_d = _run_interval(h200_prefill, h200_decode, h200_state)

    expected_h200_p = _expected_prefill(h200_config, h200_prefill, sample)
    expected_h200_d = _expected_decode(h200_config, h200_decode, sample)
    assert h200_p == expected_h200_p
    assert h200_d == expected_h200_d

    # B60 prefill throughput at ISL=3000 is ~5957 tok/s/gpu
    # H200 prefill throughput at ISL=3000 is ~49846 tok/s/gpu
    # So XPU should need more replicas than H200
    assert xpu_p > h200_p, f"XPU prefill ({xpu_p}) should exceed H200 ({h200_p})"
    assert xpu_d > h200_d, f"XPU decode ({xpu_d}) should exceed H200 ({h200_d})"


def test_xpu_disagg_short_input_sequence():
    """Test XPU planner behavior with short input sequences.

    B60 has lower throughput at short ISL too (~2709 tok/s/gpu at ISL=100).
    """
    config = _build_xpu_config()
    samples = [
        {
            "num_req": 100,
            "isl": 100,
            "osl": 50,
            "ttft_ms": 30.0,
            "itl_ms": 30.0,
            "request_duration": 5.0,
        },
    ]
    client = _build_prometheus_client(samples)
    prefill_planner, decode_planner, shared_state = _build_planners(config, client)

    num_p, num_d = _run_interval(prefill_planner, decode_planner, shared_state)

    assert num_p == _expected_prefill(config, prefill_planner, samples[0])
    assert num_d == _expected_decode(config, decode_planner, samples[0])
    assert num_p >= config.min_endpoint
    assert num_d >= config.min_endpoint


def test_xpu_disagg_long_input_sequence():
    """Test XPU planner behavior with long input sequences.

    B60 throughput degrades with longer sequences (~4163 tok/s/gpu at ISL=16372).
    """
    config = _build_xpu_config()
    samples = [
        {
            "num_req": 500,
            "isl": 10000,
            "osl": 200,
            "ttft_ms": 800.0,
            "itl_ms": 35.0,
            "request_duration": 30.0,
        },
    ]
    client = _build_prometheus_client(samples)
    prefill_planner, decode_planner, shared_state = _build_planners(config, client)

    num_p, num_d = _run_interval(prefill_planner, decode_planner, shared_state)

    assert num_p == _expected_prefill(config, prefill_planner, samples[0])
    assert num_d == _expected_decode(config, decode_planner, samples[0])
    assert num_p >= config.min_endpoint
    assert num_d >= config.min_endpoint


def test_xpu_disagg_min_endpoint_floor():
    """Zero traffic should still respect min_endpoint on XPU."""
    config = _build_xpu_config()
    samples = [
        {
            "num_req": 0,
            "isl": 3000,
            "osl": 150,
            "ttft_ms": 0.0,
            "itl_ms": 0.0,
            "request_duration": 0.0,
        },
    ]
    client = _build_prometheus_client(samples)
    prefill_planner, decode_planner, shared_state = _build_planners(config, client)

    num_p, num_d = _run_interval(prefill_planner, decode_planner, shared_state)

    assert num_p >= config.min_endpoint
    assert num_d >= config.min_endpoint


def test_xpu_disagg_tight_sla():
    """Test XPU planner with tight SLA thresholds.

    With B60's higher latency, a tight SLA should require more replicas.
    """
    config = _build_xpu_config(ttft=600.0, itl=40.0)
    samples = [
        {
            "num_req": 1000,
            "isl": 3000,
            "osl": 150,
            "ttft_ms": 500.0,
            "itl_ms": 35.0,
            "request_duration": 20.0,
        },
    ]
    client = _build_prometheus_client(samples)
    prefill_planner, decode_planner, shared_state = _build_planners(config, client)

    num_p, num_d = _run_interval(prefill_planner, decode_planner, shared_state)

    assert num_p == _expected_prefill(config, prefill_planner, samples[0])
    assert num_d == _expected_decode(config, decode_planner, samples[0])


# ── XPU GPU Count Initialization Tests ──────────────────────────────


class TestXpuInitializeGpuCounts:
    """GPU count initialization tests for XPU deployments."""

    def test_xpu_kubernetes_mode_reads_from_dgd(self):
        """Test that GPU counts are read from DGD in XPU Kubernetes mode"""
        args = argparse.Namespace()
        args.prefill_engine_num_gpu = None
        args.decode_engine_num_gpu = None

        connector = Mock()
        connector.get_gpu_counts = Mock(return_value=(1, 1))

        _initialize_gpu_counts(
            args, connector, require_prefill=True, require_decode=True
        )

        assert args.prefill_engine_num_gpu == 1
        assert args.decode_engine_num_gpu == 1
        connector.get_gpu_counts.assert_called_once_with(
            require_prefill=True, require_decode=True
        )

    def test_xpu_virtual_mode_uses_cli_args(self):
        """Test that GPU counts come from CLI args in XPU virtual mode"""
        args = argparse.Namespace()
        args.prefill_engine_num_gpu = 1
        args.decode_engine_num_gpu = 1

        connector = Mock(spec=[])

        _initialize_gpu_counts(
            args, connector, require_prefill=True, require_decode=True
        )

        assert args.prefill_engine_num_gpu == 1
        assert args.decode_engine_num_gpu == 1

    def test_xpu_virtual_mode_missing_gpu_flags_raises_error(self):
        """Test that missing GPU flags raise error in XPU virtual mode"""
        args = argparse.Namespace()
        args.prefill_engine_num_gpu = None
        args.decode_engine_num_gpu = None

        connector = Mock(spec=[])

        with pytest.raises(DeploymentValidationError) as exc_info:
            _initialize_gpu_counts(
                args, connector, require_prefill=True, require_decode=True
            )

        assert len(exc_info.value.errors) == 2


# ── XPU Dryrun GPU Defaults Tests ───────────────────────────────────


class TestXpuDryrunGpuDefaults:
    @staticmethod
    def _build_dryrun_config(**overrides) -> PlannerConfig:
        defaults = dict(
            throughput_adjustment_interval=60,
            prefill_engine_num_gpu=1,
            decode_engine_num_gpu=1,
            min_endpoint=1,
            max_gpu_budget=-1,
            ttft=1000.0,
            itl=50.0,
            backend="vllm",
            no_operation=True,
            no_correction=True,
            metric_pulling_prometheus_endpoint="http://localhost:9090",
            metric_reporting_prometheus_port=0,
            load_predictor="constant",
            load_predictor_warmup_trace=None,
            load_predictor_log1p=False,
            profile_results_dir=B60_PROFILE_DIR,
            environment="kubernetes",
            namespace="test-namespace",
            mode="disagg",
            enable_throughput_scaling=True,
            enable_load_scaling=False,
        )
        defaults.update(overrides)
        return PlannerConfig.model_construct(**defaults)

    def test_xpu_dryrun_defaults_gpu_counts_when_none(self):
        """Test that dryrun sets default GPU counts of 1 when None on XPU"""
        config = self._build_dryrun_config(
            prefill_engine_num_gpu=None, decode_engine_num_gpu=None
        )

        try:
            run_sla_planner_dryrun(config, dataset="nonexistent.jsonl")
        except (FileNotFoundError, ValueError):
            pass

        assert config.prefill_engine_num_gpu == 1
        assert config.decode_engine_num_gpu == 1

    def test_xpu_dryrun_preserves_cli_gpu_counts(self):
        """Test that dryrun preserves GPU counts provided via config on XPU"""
        config = self._build_dryrun_config(
            prefill_engine_num_gpu=2, decode_engine_num_gpu=4
        )

        try:
            run_sla_planner_dryrun(config, dataset="nonexistent.jsonl")
        except (FileNotFoundError, ValueError):
            pass

        assert config.prefill_engine_num_gpu == 2
        assert config.decode_engine_num_gpu == 4
