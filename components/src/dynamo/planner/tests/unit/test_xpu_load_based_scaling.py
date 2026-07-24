# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
XPU (Intel B60) unit tests for load-based scaling logic.

These tests mirror the H200 tests in test_load_based_scaling.py but use
the B60_TP1P_TP1D profiling results, validating that load-based scaling
works correctly with XPU performance characteristics.
"""

import os
from unittest.mock import Mock, patch

import pytest

try:
    import msgspec  # noqa: F401
except ImportError:
    pytest.skip("msgspec required for FPM tests", allow_module_level=True)

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
    encode,
)
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.decode import DecodePlanner
from dynamo.planner.core.prefill import PrefillPlanner
from dynamo.planner.core.state import PlannerSharedState
from dynamo.planner.monitoring.worker_info import WorkerInfo

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


def _make_fpm(
    *,
    sum_prefill_tokens: int = 0,
    num_prefill_requests: int = 0,
    sum_decode_kv_tokens: int = 0,
    num_decode_requests: int = 0,
    queued_prefill_tokens: int = 0,
    queued_decode_kv_tokens: int = 0,
    wall_time: float = 0.01,
    worker_id: str = "w1",
    dp_rank: int = 0,
) -> ForwardPassMetrics:
    return ForwardPassMetrics(
        worker_id=worker_id,
        dp_rank=dp_rank,
        wall_time=wall_time,
        scheduled_requests=ScheduledRequestMetrics(
            sum_prefill_tokens=sum_prefill_tokens,
            num_prefill_requests=num_prefill_requests,
            sum_decode_kv_tokens=sum_decode_kv_tokens,
            num_decode_requests=num_decode_requests,
        ),
        queued_requests=QueuedRequestMetrics(
            sum_prefill_tokens=queued_prefill_tokens,
            sum_decode_kv_tokens=queued_decode_kv_tokens,
        ),
    )


@pytest.fixture(autouse=True)
def mock_prometheus_metrics():
    with patch("dynamo.planner.monitoring.planner_metrics.Gauge") as mock_gauge:
        mock_gauge.return_value = Mock()
        yield


def _build_xpu_load_config(**overrides) -> PlannerConfig:
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
        profile_results_dir=B60_PROFILE_DIR,
        environment="kubernetes",
        namespace="test-namespace",
        mode="disagg",
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        load_adjustment_interval=5,
        load_learning_window=50,
        load_scaling_down_sensitivity=80,
        load_metric_samples=10,
        load_min_observations=5,
    )
    defaults.update(overrides)
    return PlannerConfig.model_construct(**defaults)


def _mock_fpm_subscriber(fpm_stats: dict[tuple[str, int], ForwardPassMetrics]):
    """Create a mock FPM subscriber that returns encoded FPM stats."""
    mock = Mock()
    encoded = {k: encode(v) for k, v in fpm_stats.items()}
    mock.get_recent_stats.return_value = encoded
    return mock


class TestXpuPrefillFpmScaling:
    def test_scale_up_all_engines_above_sla(self):
        """All engines have high queued prefill -> estimated TTFT > SLA -> scale up (B60 XPU)."""
        config = _build_xpu_load_config(ttft=5.0)  # 5ms SLA (easy to exceed)
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 2

        planner = PrefillPlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"
        planner.prefill_worker_info = WorkerInfo(max_num_batched_tokens=2048)

        # Train regression: wall_time grows linearly with prefill tokens
        for tokens in range(200, 1200, 100):
            fpm = _make_fpm(
                sum_prefill_tokens=tokens,
                num_prefill_requests=1,
                wall_time=0.001 * tokens,
            )
            planner.ttft_regression.add_observation(fpm)

        # Both engines have heavy queued prefill -> high estimated TTFT
        stats = {
            ("w1", 0): _make_fpm(
                worker_id="w1",
                queued_prefill_tokens=10000,
                sum_prefill_tokens=500,
                num_prefill_requests=1,
                wall_time=0.5,
            ),
            ("w2", 0): _make_fpm(
                worker_id="w2",
                queued_prefill_tokens=8000,
                sum_prefill_tokens=600,
                num_prefill_requests=1,
                wall_time=0.6,
            ),
        }
        planner.fpm_subscriber = _mock_fpm_subscriber(stats)

        result = planner.load_plan_adjustment()
        assert result == 3

    def test_scale_down_all_engines_below_sla(self):
        """All engines have low queued prefill -> estimated TTFT < SLA * sensitivity (B60 XPU)."""
        config = _build_xpu_load_config(ttft=1000.0, load_scaling_down_sensitivity=100)
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 3

        planner = PrefillPlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"
        planner.prefill_worker_info = WorkerInfo(max_num_batched_tokens=2048)

        # Train with short ISL (100 tokens each) so avg_isl stays low.
        for tokens in range(100, 600, 50):
            fpm = _make_fpm(
                sum_prefill_tokens=tokens,
                num_prefill_requests=1,
                wall_time=0.001 * tokens,
            )
            planner.ttft_regression.add_observation(fpm)

        # All engines idle (no queued prefill).
        stats = {
            (f"w{i}", 0): _make_fpm(
                worker_id=f"w{i}",
                queued_prefill_tokens=0,
                sum_prefill_tokens=100,
                num_prefill_requests=1,
                wall_time=0.1,
            )
            for i in range(3)
        }
        planner.fpm_subscriber = _mock_fpm_subscriber(stats)

        result = planner.load_plan_adjustment()
        assert result == 2

    def test_cold_start_returns_none(self):
        config = _build_xpu_load_config()
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 2

        planner = PrefillPlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"
        planner.prefill_worker_info = WorkerInfo(max_num_batched_tokens=2048)

        # Only 2 observations, need 5
        for tokens in [100, 200]:
            fpm = _make_fpm(sum_prefill_tokens=tokens, wall_time=0.01)
            planner.ttft_regression.add_observation(fpm)

        stats = {("w1", 0): _make_fpm(queued_prefill_tokens=5000, wall_time=0.5)}
        planner.fpm_subscriber = _mock_fpm_subscriber(stats)

        result = planner.load_plan_adjustment()
        assert result is None


class TestXpuDecodeFpmScaling:
    def test_scale_up_all_engines_above_sla(self):
        """All engines have high decode load -> estimated ITL > SLA -> scale up (B60 XPU)."""
        config = _build_xpu_load_config(itl=5.0)  # 5ms SLA
        shared_state = PlannerSharedState()
        shared_state.num_d_workers = 2

        planner = DecodePlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"

        for kv in range(1000, 6000, 500):
            fpm = _make_fpm(
                sum_decode_kv_tokens=kv,
                num_decode_requests=10,
                wall_time=0.0001 * kv + 0.001,
            )
            planner.itl_regression.add_observation(fpm)

        stats = {
            ("w1", 0): _make_fpm(
                worker_id="w1",
                sum_decode_kv_tokens=5000,
                queued_decode_kv_tokens=3000,
                num_decode_requests=20,
                wall_time=0.6,
            ),
            ("w2", 0): _make_fpm(
                worker_id="w2",
                sum_decode_kv_tokens=4500,
                queued_decode_kv_tokens=2500,
                num_decode_requests=18,
                wall_time=0.55,
            ),
        }
        planner.fpm_subscriber = _mock_fpm_subscriber(stats)

        result = planner.load_plan_adjustment()
        assert result == 3

    def test_cold_start_returns_none(self):
        config = _build_xpu_load_config()
        shared_state = PlannerSharedState()
        shared_state.num_d_workers = 2

        planner = DecodePlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"

        fpm = _make_fpm(sum_decode_kv_tokens=1000, wall_time=0.01)
        planner.itl_regression.add_observation(fpm)

        stats = {("w1", 0): _make_fpm(sum_decode_kv_tokens=5000, wall_time=0.5)}
        planner.fpm_subscriber = _mock_fpm_subscriber(stats)

        result = planner.load_plan_adjustment()
        assert result is None
