# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import math
import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.budget import _initialize_gpu_counts
from dynamo.planner.core.decode import DecodePlanner
from dynamo.planner.core.prefill import PrefillPlanner
from dynamo.planner.core.state import PlannerSharedState
from dynamo.planner.errors import DeploymentValidationError

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

PREFILL_ENGINE_RPS = 10.0
DECODE_ENGINE_RPS = 5.0
DECODE_ACTUAL_ITL_MS = 40.0


@pytest.fixture(autouse=True)
def mock_prometheus_metrics():
    with patch("dynamo.planner.monitoring.planner_metrics.Gauge") as mock_gauge:
        mock_gauge.return_value = Mock()
        yield


def _build_config():
    return PlannerConfig.model_construct(
        throughput_adjustment_interval=60,
        prefill_engine_num_gpu=1,
        decode_engine_num_gpu=1,
        min_endpoint=1,
        max_gpu_budget=-1,
        ttft=500.0,
        itl=50.0,
        backend="vllm",
        no_operation=True,
        metric_pulling_prometheus_endpoint="http://localhost:9090",
        metric_reporting_prometheus_port=0,
        load_predictor="constant",
        load_predictor_warmup_trace=None,
        load_predictor_log1p=False,
        profile_results_dir=os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "profiling_results",
            "H200_TP1P_TP1D",
        ),
        environment="kubernetes",
        namespace="test-namespace",
        mode="disagg",
        enable_throughput_scaling=True,
        enable_load_scaling=False,
    )


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

    prefill_planner.ttft_regression = MagicMock()
    prefill_planner.ttft_regression.find_best_engine_prefill_rps.return_value = (
        PREFILL_ENGINE_RPS,
        75.0,
    )
    prefill_planner.ttft_regression.has_sufficient_data.return_value = True

    decode_planner.itl_regression = MagicMock()
    decode_planner.itl_regression.find_best_engine_decode_rps.return_value = (
        DECODE_ENGINE_RPS,
        DECODE_ACTUAL_ITL_MS,
    )
    decode_planner.itl_regression.has_sufficient_data.return_value = True

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
    demand_rps = sample["num_req"] / config.throughput_adjustment_interval
    engine_rps, _ = prefill_planner.ttft_regression.find_best_engine_prefill_rps(
        ttft_sla=config.ttft, isl=sample["isl"]
    )
    expected = math.ceil(demand_rps / engine_rps)
    return max(expected, config.min_endpoint)


def _expected_decode(config, decode_planner, sample):
    demand_rps = sample["num_req"] / config.throughput_adjustment_interval
    engine_rps, _ = decode_planner.itl_regression.find_best_engine_decode_rps(
        itl=config.itl, context_length=sample["isl"] + sample["osl"] / 2
    )
    expected = math.ceil(demand_rps / engine_rps)
    return max(expected, config.min_endpoint)


def _run_interval(prefill_planner, decode_planner, shared_state):
    asyncio.run(
        prefill_planner.observe_traffic_stats(require_prefill=True, require_decode=True)
    )
    decode_planner.update_predictors_from_metrics(shared_state.last_metrics)
    next_num_p = prefill_planner.plan_adjustment()
    next_num_d = decode_planner.plan_adjustment()
    return next_num_p, next_num_d


def test_disagg_scale_up():
    config = _build_config()
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


def test_disagg_scale_down():
    config = _build_config()
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


class TestInitializeGpuCounts:
    @staticmethod
    def _make_config(**overrides):
        defaults = dict(prefill_engine_num_gpu=None, decode_engine_num_gpu=None)
        defaults.update(overrides)
        return PlannerConfig.model_construct(**defaults)

    def test_kubernetes_mode_reads_from_dgd(self):
        """Test that GPU counts are read from DGD in Kubernetes mode"""
        config = self._make_config()

        connector = Mock()
        connector.get_gpu_counts = Mock(return_value=(2, 4))

        _initialize_gpu_counts(
            config, connector, require_prefill=True, require_decode=True
        )

        assert config.prefill_engine_num_gpu == 2
        assert config.decode_engine_num_gpu == 4
        connector.get_gpu_counts.assert_called_once_with(
            require_prefill=True, require_decode=True
        )

    def test_kubernetes_mode_prefill_only(self):
        """Test GPU count initialization for prefill-only mode"""
        config = self._make_config()

        connector = Mock()
        connector.get_gpu_counts = Mock(return_value=(2, 0))

        _initialize_gpu_counts(
            config, connector, require_prefill=True, require_decode=False
        )

        assert config.prefill_engine_num_gpu == 2
        assert config.decode_engine_num_gpu == 0
        connector.get_gpu_counts.assert_called_once_with(
            require_prefill=True, require_decode=False
        )

    def test_virtual_mode_uses_cli_args(self):
        """Test that GPU counts come from config in virtual mode"""
        config = self._make_config(prefill_engine_num_gpu=2, decode_engine_num_gpu=4)

        connector = Mock(spec=[])

        _initialize_gpu_counts(
            config, connector, require_prefill=True, require_decode=True
        )

        assert config.prefill_engine_num_gpu == 2
        assert config.decode_engine_num_gpu == 4

    def test_virtual_mode_missing_prefill_raises_error(self):
        """Test that missing prefill GPU config raises error in virtual mode"""
        config = self._make_config(decode_engine_num_gpu=4)

        connector = Mock(spec=[])

        with pytest.raises(DeploymentValidationError) as exc_info:
            _initialize_gpu_counts(
                config, connector, require_prefill=True, require_decode=True
            )

        assert "prefill_engine_num_gpu" in str(exc_info.value)

    def test_virtual_mode_missing_decode_raises_error(self):
        """Test that missing decode GPU config raises error in virtual mode"""
        config = self._make_config(prefill_engine_num_gpu=2)

        connector = Mock(spec=[])

        with pytest.raises(DeploymentValidationError) as exc_info:
            _initialize_gpu_counts(
                config, connector, require_prefill=True, require_decode=True
            )

        assert "decode_engine_num_gpu" in str(exc_info.value)

    def test_virtual_mode_missing_both_raises_error_with_both_messages(self):
        """Test that missing both GPU configs shows both error messages"""
        config = self._make_config()

        connector = Mock(spec=[])

        with pytest.raises(DeploymentValidationError) as exc_info:
            _initialize_gpu_counts(
                config, connector, require_prefill=True, require_decode=True
            )

        assert len(exc_info.value.errors) == 2

    def test_virtual_mode_decode_only_no_prefill_error(self):
        """Test decode-only mode doesn't require prefill GPU config"""
        config = self._make_config(decode_engine_num_gpu=4)

        connector = Mock(spec=[])

        _initialize_gpu_counts(
            config, connector, require_prefill=False, require_decode=True
        )

        assert config.decode_engine_num_gpu == 4

    def test_kubernetes_mode_fallback_to_cli_on_dgd_error(self):
        """Test that K8s mode falls back to config when DGD parsing fails"""
        config = self._make_config(prefill_engine_num_gpu=2, decode_engine_num_gpu=4)

        connector = Mock()
        connector.get_gpu_counts = Mock(
            side_effect=ValueError("No GPU count specified")
        )

        _initialize_gpu_counts(
            config, connector, require_prefill=True, require_decode=True
        )

        assert config.prefill_engine_num_gpu == 2
        assert config.decode_engine_num_gpu == 4

    def test_kubernetes_mode_fallback_missing_cli_flags_raises_error(self):
        """Test that K8s fallback raises error when config also missing"""
        config = self._make_config()

        connector = Mock()
        connector.get_gpu_counts = Mock(
            side_effect=ValueError("No GPU count specified")
        )

        with pytest.raises(DeploymentValidationError) as exc_info:
            _initialize_gpu_counts(
                config, connector, require_prefill=True, require_decode=True
            )

        assert len(exc_info.value.errors) == 2

    def test_kubernetes_mode_fallback_partial_cli_flags(self):
        """Test K8s fallback with only one config value provided"""
        config = self._make_config(prefill_engine_num_gpu=2)

        connector = Mock()
        connector.get_gpu_counts = Mock(
            side_effect=ValueError("No GPU count specified")
        )

        with pytest.raises(DeploymentValidationError) as exc_info:
            _initialize_gpu_counts(
                config, connector, require_prefill=True, require_decode=True
            )

        assert "decode_engine_num_gpu" in str(exc_info.value)
