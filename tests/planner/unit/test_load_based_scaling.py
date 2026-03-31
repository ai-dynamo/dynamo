# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from dynamo.planner.core.load.fpm_regression import (
    AggRegressionModel,
    DecodeRegressionModel,
    PrefillRegressionModel,
)
from dynamo.planner.core.prefill import PrefillPlanner
from dynamo.planner.core.state import PlannerSharedState
from dynamo.planner.monitoring.worker_info import WorkerInfo

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


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


# ── PrefillRegressionModel tests ─────────────────────────────────────


class TestPrefillRegressionModel:
    def test_insufficient_data(self):
        model = PrefillRegressionModel(window_size=50, min_observations=5)
        assert not model.has_sufficient_data()
        assert model.estimate_next_ttft(0, 2048) is None

    def test_heartbeat_skipped(self):
        model = PrefillRegressionModel(window_size=50, min_observations=3)
        fpm = _make_fpm(wall_time=0.0, sum_prefill_tokens=100, num_prefill_requests=1)
        model.add_observation(fpm)
        assert model.num_observations == 0

    def test_basic_regression_and_ttft_estimate(self):
        model = PrefillRegressionModel(window_size=50, min_observations=3)
        # wall_time = 0.001 * prefill_tokens + 0.002 (linear relationship)
        for tokens in [500, 1000, 1500, 2000, 2500]:
            fpm = _make_fpm(
                sum_prefill_tokens=tokens,
                num_prefill_requests=1,
                wall_time=0.001 * tokens + 0.002,
            )
            model.add_observation(fpm)

        assert model.has_sufficient_data()

        # Single iteration: queued=0, avg_isl should be mean of [500..2500]=1500
        # total_tokens = 0 + avg_isl ≈ 1500
        # 1 iteration at max_num_batched_tokens=2048 (1500 < 2048)
        est = model.estimate_next_ttft(
            queued_prefill_tokens=0, max_num_batched_tokens=2048
        )
        assert est is not None
        assert est > 0

    def test_chunked_ttft_simulation(self):
        model = PrefillRegressionModel(window_size=50, min_observations=3)
        # Simple: wall_time = 0.001 * prefill_tokens (slope=0.001, intercept≈0)
        for tokens in [100, 200, 300, 400, 500]:
            fpm = _make_fpm(
                sum_prefill_tokens=tokens,
                num_prefill_requests=1,
                wall_time=0.001 * tokens,
            )
            model.add_observation(fpm)

        # avg_isl = mean([100,200,300,400,500]) = 300
        # total_tokens = 5000 (queued) + 300 (next ISL) = 5300
        # max_num_batched_tokens = 2048
        # iterations: ceil(5300/2048) = 3
        # chunk1=2048, chunk2=2048, chunk3=1204
        est = model.estimate_next_ttft(
            queued_prefill_tokens=5000, max_num_batched_tokens=2048
        )
        assert est is not None
        assert est > 0.003  # at least 3 iterations worth

    def test_avg_isl_tracking(self):
        model = PrefillRegressionModel(window_size=50, min_observations=3)
        for isl in [1000, 2000, 3000]:
            fpm = _make_fpm(
                sum_prefill_tokens=isl, num_prefill_requests=1, wall_time=0.01
            )
            model.add_observation(fpm)
        assert abs(model.avg_isl - 2000.0) < 1.0

    def test_sliding_window_eviction(self):
        model = PrefillRegressionModel(window_size=5, min_observations=3)
        for i in range(10):
            fpm = _make_fpm(sum_prefill_tokens=100 * (i + 1), wall_time=0.01)
            model.add_observation(fpm)
        assert model.num_observations == 5


# ── DecodeRegressionModel tests ──────────────────────────────────────


class TestDecodeRegressionModel:
    def test_insufficient_data(self):
        model = DecodeRegressionModel(window_size=50, min_observations=5)
        assert not model.has_sufficient_data()
        assert model.estimate_next_itl(0, 0) is None

    def test_heartbeat_skipped(self):
        model = DecodeRegressionModel(window_size=50, min_observations=3)
        fpm = _make_fpm(wall_time=0.0, sum_decode_kv_tokens=100, num_decode_requests=1)
        model.add_observation(fpm)
        assert model.num_observations == 0

    def test_basic_itl_estimate(self):
        model = DecodeRegressionModel(window_size=50, min_observations=3)
        # wall_time = 0.0001 * decode_kv + 0.001
        for kv in [1000, 2000, 3000, 4000, 5000]:
            fpm = _make_fpm(
                sum_decode_kv_tokens=kv,
                num_decode_requests=10,
                wall_time=0.0001 * kv + 0.001,
            )
            model.add_observation(fpm)

        assert model.has_sufficient_data()
        est = model.estimate_next_itl(scheduled_decode_kv=3000, queued_decode_kv=0)
        assert est is not None
        assert est > 0

    def test_avg_decode_length_tracking(self):
        model = DecodeRegressionModel(window_size=50, min_observations=3)
        for total_kv, num_req in [(1000, 10), (2000, 10), (3000, 10)]:
            fpm = _make_fpm(
                sum_decode_kv_tokens=total_kv,
                num_decode_requests=num_req,
                wall_time=0.01,
            )
            model.add_observation(fpm)
        assert abs(model.avg_decode_length - 200.0) < 1.0


# ── AggRegressionModel tests ─────────────────────────────────────────


class TestAggRegressionModel:
    def test_insufficient_data(self):
        model = AggRegressionModel(window_size=50, min_observations=5)
        assert not model.has_sufficient_data()
        assert model.estimate_next_ttft(0, 2048, 0) is None
        assert model.estimate_next_itl(0, 0) is None

    def test_heartbeat_skipped(self):
        model = AggRegressionModel(window_size=50, min_observations=3)
        fpm = _make_fpm(wall_time=0.0, sum_prefill_tokens=100, sum_decode_kv_tokens=200)
        model.add_observation(fpm)
        assert model.num_observations == 0

    def test_2d_regression(self):
        model = AggRegressionModel(window_size=50, min_observations=3)
        # wall_time = 0.001 * prefill + 0.0001 * decode_kv + 0.001
        for p, d in [(100, 1000), (200, 2000), (300, 3000), (400, 4000), (500, 5000)]:
            fpm = _make_fpm(
                sum_prefill_tokens=p,
                num_prefill_requests=1,
                sum_decode_kv_tokens=d,
                num_decode_requests=10,
                wall_time=0.001 * p + 0.0001 * d + 0.001,
            )
            model.add_observation(fpm)

        assert model.has_sufficient_data()

        ttft = model.estimate_next_ttft(
            queued_prefill_tokens=0,
            max_num_batched_tokens=2048,
            current_decode_kv=3000,
        )
        assert ttft is not None
        assert ttft > 0

        itl = model.estimate_next_itl(scheduled_decode_kv=3000, queued_decode_kv=0)
        assert itl is not None
        assert itl > 0


# ── Planner integration tests (with mocked FPM subscriber) ──────────


@pytest.fixture(autouse=True)
def mock_prometheus_metrics():
    with patch("dynamo.planner.monitoring.planner_metrics.Gauge") as mock_gauge:
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


class TestPrefillFpmScaling:
    def test_scale_up_all_engines_above_sla(self):
        """All engines have high queued prefill -> estimated TTFT > SLA -> scale up."""
        config = _build_load_config(ttft=5.0)  # 5ms SLA (easy to exceed)
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
        """All engines have low queued prefill -> estimated TTFT < SLA * sensitivity."""
        config = _build_load_config(ttft=500.0, load_scaling_down_sensitivity=100)
        shared_state = PlannerSharedState()
        shared_state.num_p_workers = 3

        planner = PrefillPlanner(None, config, shared_state=shared_state)
        planner.model_name = "test-model"
        planner.prefill_worker_info = WorkerInfo(max_num_batched_tokens=2048)

        # Train with short ISL (100 tokens each) so avg_isl stays low.
        # Regression: wall_time ≈ 0.001 * prefill_tokens
        for tokens in range(100, 600, 50):
            fpm = _make_fpm(
                sum_prefill_tokens=tokens,
                num_prefill_requests=1,
                wall_time=0.001 * tokens,
            )
            planner.ttft_regression.add_observation(fpm)

        # All engines idle (no queued prefill).
        # estimate_next_ttft: total = 0 + avg_isl(~100) = ~100 tokens
        # predicted wall_time ≈ 0.001 * 100 = 0.1s = 100ms < 500ms SLA
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
        config = _build_load_config()
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


class TestDecodeFpmScaling:
    def test_scale_up_all_engines_above_sla(self):
        """All engines have high decode load -> estimated ITL > SLA -> scale up."""
        config = _build_load_config(itl=5.0)  # 5ms SLA
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
        config = _build_load_config()
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


# ── Correction factor auto-disable tests ─────────────────────────────


class TestCorrectionFactorAutoDisable:
    def test_correction_factor_disabled_when_load_enabled(self):
        config = PlannerConfig(
            enable_load_scaling=True,
            enable_throughput_scaling=True,
            no_correction=False,
        )
        assert config.no_correction is True

    def test_correction_factor_stays_disabled_if_already_set(self):
        config = PlannerConfig(
            enable_load_scaling=True,
            enable_throughput_scaling=True,
            no_correction=True,
        )
        assert config.no_correction is True

    def test_correction_factor_not_disabled_without_loadbased(self):
        config = PlannerConfig(
            enable_load_scaling=False,
            enable_throughput_scaling=True,
            no_correction=False,
        )
        assert config.no_correction is False
