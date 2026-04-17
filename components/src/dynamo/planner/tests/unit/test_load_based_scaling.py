# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression model unit tests.

These test the perf_model classes directly (PrefillRegressionModel,
DecodeRegressionModel, AggRegressionModel) without any planner adapter.

FPM-driven scaling integration tests live in test_state_machine.py.
"""

from unittest.mock import Mock, patch

import pytest

try:
    import msgspec  # noqa: F401
except ImportError:
    pytest.skip("msgspec required for FPM tests", allow_module_level=True)

try:
    from dynamo.common.forward_pass_metrics import (
        ForwardPassMetrics,
        QueuedRequestMetrics,
        ScheduledRequestMetrics,
    )
except ImportError:
    pytest.skip("forward_pass_metrics not available", allow_module_level=True)
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.core.perf_model import (
    AggRegressionModel,
    DecodeRegressionModel,
    PrefillRegressionModel,
)
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
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=5, bucket_count=16
        )
        assert not model.has_sufficient_data()
        assert model.estimate_next_ttft(0, 2048) is None

    def test_heartbeat_skipped(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        fpm = _make_fpm(wall_time=0.0, sum_prefill_tokens=100, num_prefill_requests=1)
        model.add_observation(fpm)
        assert model.num_observations == 0

    def test_basic_regression_and_ttft_estimate(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        for tokens in [500, 1000, 1500, 2000, 2500]:
            fpm = _make_fpm(
                sum_prefill_tokens=tokens,
                num_prefill_requests=1,
                wall_time=0.001 * tokens + 0.002,
            )
            model.add_observation(fpm)

        assert model.has_sufficient_data()

        est = model.estimate_next_ttft(
            queued_prefill_tokens=0, max_num_batched_tokens=2048
        )
        assert est is not None
        assert est > 0

    def test_chunked_ttft_simulation(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        for tokens in [100, 200, 300, 400, 500]:
            fpm = _make_fpm(
                sum_prefill_tokens=tokens,
                num_prefill_requests=1,
                wall_time=0.001 * tokens,
            )
            model.add_observation(fpm)

        est = model.estimate_next_ttft(
            queued_prefill_tokens=5000, max_num_batched_tokens=2048
        )
        assert est is not None
        assert est > 0.003  # at least 3 iterations worth

    def test_avg_isl_tracking(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        for isl in [1000, 2000, 3000]:
            fpm = _make_fpm(
                sum_prefill_tokens=isl, num_prefill_requests=1, wall_time=0.01
            )
            model.add_observation(fpm)
        assert abs(model.avg_isl - 2000.0) < 1.0

    def test_find_best_engine_prefill_rps(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        for tokens in [500, 1000, 1500, 2000, 2500]:
            fpm = _make_fpm(
                sum_prefill_tokens=tokens,
                num_prefill_requests=1,
                wall_time=0.001 * tokens + 0.002,
            )
            model.add_observation(fpm)

        rps, actual_ttft_ms = model.find_best_engine_prefill_rps(
            ttft_sla=2000.0, isl=1000.0
        )
        assert rps > 0
        assert 0.5 < rps < 2.0
        assert actual_ttft_ms > 0
        assert 1000 < actual_ttft_ms < 2000

    def test_find_best_engine_prefill_rps_zero_isl(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        for tokens in [500, 1000, 1500]:
            fpm = _make_fpm(
                sum_prefill_tokens=tokens,
                num_prefill_requests=1,
                wall_time=0.001 * tokens,
            )
            model.add_observation(fpm)
        rps, _ = model.find_best_engine_prefill_rps(ttft_sla=1000.0, isl=0.0)
        assert rps == 0.0

    def test_load_benchmark_fpms(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        fpms = [
            _make_fpm(sum_prefill_tokens=t, num_prefill_requests=1, wall_time=0.001 * t)
            for t in [500, 1000, 1500, 2000, 2500]
        ]
        model.load_benchmark_fpms(fpms)
        assert model.num_observations == 5
        assert model.has_sufficient_data()
        est = model.estimate_next_ttft(
            queued_prefill_tokens=0, max_num_batched_tokens=2048
        )
        assert est is not None


# ── Bucketed retirement tests ─────────────────────────────────────────


class TestBucketedRetirement:
    def test_total_capped_at_max(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=10, min_observations=3, bucket_count=4
        )
        for i in range(20):
            fpm = _make_fpm(
                sum_prefill_tokens=100 * (i + 1),
                num_prefill_requests=1,
                wall_time=0.01 * (i + 1),
            )
            model.add_observation(fpm)
        assert model.num_observations == 10

    def test_most_populated_bucket_loses_oldest(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=6, min_observations=1, bucket_count=4
        )
        for i in range(3):
            fpm = _make_fpm(
                sum_prefill_tokens=10 + i,
                num_prefill_requests=1,
                wall_time=0.001 * (10 + i),
            )
            model.add_observation(fpm)
        for i in range(3):
            fpm = _make_fpm(
                sum_prefill_tokens=1000 + i * 100,
                num_prefill_requests=1,
                wall_time=0.001 * (1000 + i * 100),
            )
            model.add_observation(fpm)
        assert model.num_observations == 6
        fpm = _make_fpm(sum_prefill_tokens=15, num_prefill_requests=1, wall_time=0.015)
        model.add_observation(fpm)
        assert model.num_observations == 6

    def test_uniform_distribution_preserved(self):
        model = DecodeRegressionModel(
            max_num_fpm_samples=10, min_observations=3, bucket_count=16
        )
        for _ in range(15):
            fpm = _make_fpm(
                num_decode_requests=32, sum_decode_kv_tokens=32000, wall_time=0.01
            )
            model.add_observation(fpm)
        assert model.num_observations == 10
        fpm = _make_fpm(
            num_decode_requests=4, sum_decode_kv_tokens=4000, wall_time=0.005
        )
        model.add_observation(fpm)
        assert model.num_observations == 10

    def test_2d_bucketed_retirement(self):
        model = AggRegressionModel(
            max_num_fpm_samples=8, min_observations=1, bucket_count=16
        )
        for p, d in [(100, 500), (200, 1000), (300, 1500), (400, 2000)]:
            fpm = _make_fpm(
                sum_prefill_tokens=p,
                num_prefill_requests=1,
                sum_decode_kv_tokens=d,
                num_decode_requests=5,
                wall_time=0.001 * p + 0.0001 * d,
            )
            model.add_observation(fpm)
        for _ in range(4):
            fpm = _make_fpm(
                sum_prefill_tokens=100,
                num_prefill_requests=1,
                sum_decode_kv_tokens=500,
                num_decode_requests=5,
                wall_time=0.15,
            )
            model.add_observation(fpm)
        assert model.num_observations == 8
        fpm = _make_fpm(
            sum_prefill_tokens=350,
            num_prefill_requests=1,
            sum_decode_kv_tokens=1800,
            num_decode_requests=5,
            wall_time=0.5,
        )
        model.add_observation(fpm)
        assert model.num_observations == 8


# ── DecodeRegressionModel tests ──────────────────────────────────────


class TestDecodeRegressionModel:
    def _train_2d(self, model):
        for n_req, kv in [(5, 1000), (10, 2000), (15, 3000), (20, 4000), (25, 5000)]:
            fpm = _make_fpm(
                sum_decode_kv_tokens=kv,
                num_decode_requests=n_req,
                wall_time=0.0001 * kv + 0.0005 * n_req + 0.001,
            )
            model.add_observation(fpm)

    def test_insufficient_data(self):
        model = DecodeRegressionModel(
            max_num_fpm_samples=50, min_observations=5, bucket_count=16
        )
        assert not model.has_sufficient_data()
        assert model.estimate_next_itl(0, 0) is None

    def test_heartbeat_skipped(self):
        model = DecodeRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        fpm = _make_fpm(wall_time=0.0, sum_decode_kv_tokens=100, num_decode_requests=1)
        model.add_observation(fpm)
        assert model.num_observations == 0

    def test_basic_itl_estimate(self):
        model = DecodeRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        self._train_2d(model)
        assert model.has_sufficient_data()
        est = model.estimate_next_itl(scheduled_decode_kv=3000, queued_decode_kv=0)
        assert est is not None and est > 0

    def test_avg_decode_length_tracking(self):
        model = DecodeRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        for total_kv, num_req in [(1000, 10), (2000, 10), (3000, 10)]:
            fpm = _make_fpm(
                sum_decode_kv_tokens=total_kv,
                num_decode_requests=num_req,
                wall_time=0.01,
            )
            model.add_observation(fpm)
        assert abs(model.avg_decode_length - 200.0) < 1.0

    def _train_thpt_model(self, model):
        for n_req, kv in [
            (5, 5000),
            (10, 10000),
            (20, 20000),
            (30, 30000),
            (40, 40000),
        ]:
            fpm = _make_fpm(
                sum_decode_kv_tokens=kv,
                num_decode_requests=n_req,
                wall_time=0.00001 * kv + 0.001,
            )
            model.add_observation(fpm)

    def test_find_best_engine_decode_rps(self):
        model = DecodeRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        self._train_thpt_model(model)
        rps, actual_itl = model.find_best_engine_decode_rps(
            itl=50.0, context_length=1000.0, osl=150.0
        )
        assert rps > 0 and actual_itl > 0 and actual_itl <= 50.0

    def test_find_best_engine_decode_rps_zero_context(self):
        model = DecodeRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        self._train_2d(model)
        rps, itl_ms = model.find_best_engine_decode_rps(
            itl=50.0, context_length=0.0, osl=150.0
        )
        assert rps == 0.0 and itl_ms == 0.0

    def test_load_benchmark_fpms(self):
        model = DecodeRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        fpms = [
            _make_fpm(
                num_decode_requests=n,
                sum_decode_kv_tokens=n * 1000,
                wall_time=0.001 * n,
            )
            for n in [5, 10, 15, 20, 25]
        ]
        model.load_benchmark_fpms(fpms)
        assert model.num_observations == 5 and model.has_sufficient_data()


# ── AggRegressionModel tests ─────────────────────────────────────────


class TestAggRegressionModel:
    def _train_agg(self, model):
        for p, d in [(100, 1000), (200, 2000), (300, 3000), (400, 4000), (500, 5000)]:
            fpm = _make_fpm(
                sum_prefill_tokens=p,
                num_prefill_requests=1,
                sum_decode_kv_tokens=d,
                num_decode_requests=10,
                wall_time=0.001 * p + 0.0001 * d + 0.001,
            )
            model.add_observation(fpm)

    def test_insufficient_data(self):
        model = AggRegressionModel(
            max_num_fpm_samples=50, min_observations=5, bucket_count=16
        )
        assert not model.has_sufficient_data()
        assert model.estimate_next_ttft(0, 2048, 0) is None
        assert model.estimate_next_itl(0, 0) is None

    def test_heartbeat_skipped(self):
        model = AggRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        fpm = _make_fpm(wall_time=0.0, sum_prefill_tokens=100, sum_decode_kv_tokens=200)
        model.add_observation(fpm)
        assert model.num_observations == 0

    def test_2d_regression(self):
        model = AggRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        self._train_agg(model)
        assert model.has_sufficient_data()
        ttft = model.estimate_next_ttft(
            queued_prefill_tokens=0, max_num_batched_tokens=2048, current_decode_kv=3000
        )
        assert ttft is not None and ttft > 0
        itl = model.estimate_next_itl(scheduled_decode_kv=3000, queued_decode_kv=0)
        assert itl is not None and itl > 0

    def test_find_best_engine_agg_rps(self):
        model = AggRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        self._train_agg(model)
        thpt, actual_ttft, actual_itl = model.find_best_engine_agg_rps(
            isl=2048.0,
            osl=150.0,
            max_num_batched_tokens=4096,
            ttft_sla=500.0,
            itl_sla=50.0,
        )
        assert thpt > 0 and actual_ttft >= 0 and actual_itl >= 0

    def test_find_best_engine_agg_rps_insufficient_data(self):
        model = AggRegressionModel(
            max_num_fpm_samples=50, min_observations=5, bucket_count=16
        )
        thpt, _, _ = model.find_best_engine_agg_rps(
            isl=2048.0,
            osl=150.0,
            max_num_batched_tokens=4096,
            ttft_sla=500.0,
            itl_sla=50.0,
        )
        assert thpt == 0.0


# ── Connector-driven refresh tests ──────────────────────────────────


class TestRefreshWorkerInfoFromConnector:
    """Tests for NativePlannerBase._refresh_worker_info_from_connector.

    The tick-loop refresh delegates to the connector's ``get_worker_info``,
    which is where each connector implements its own MDC source (K8s CRDs
    for KubernetesConnector, discovery watch for VirtualConnector).  These
    tests exercise the shared refresh plumbing with a mock connector.
    """

    def _make_planner(self, require_prefill=False, require_decode=True):
        """Build a minimal NativePlannerBase with no_operation=True."""
        with patch("dynamo.planner.monitoring.planner_metrics.Gauge") as mock_gauge:
            mock_gauge.return_value = Mock()
            config = PlannerConfig.model_construct(
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
                environment="kubernetes",
                namespace="test-namespace",
                mode="agg",
                enable_load_scaling=True,
                enable_throughput_scaling=True,
                load_adjustment_interval=5,
                max_num_fpm_samples=50,
                fpm_sample_bucket_size=16,
                load_scaling_down_sensitivity=80,
                load_metric_samples=10,
                load_min_observations=5,
            )
            planner = NativePlannerBase(None, config)
        planner.require_prefill = require_prefill
        planner.require_decode = require_decode
        planner.prefill_worker_info = WorkerInfo()
        planner.decode_worker_info = WorkerInfo()
        return planner

    def _install_mock_connector(self, planner, **fresh_info_kwargs):
        """Replace planner.connector with a Mock returning a fresh WorkerInfo."""
        fresh = WorkerInfo(**fresh_info_kwargs)
        mock_connector = Mock()
        mock_connector.get_worker_info.return_value = fresh
        planner.connector = mock_connector
        return mock_connector

    def test_refresh_populates_missing_fields(self):
        """Connector returns a populated WorkerInfo; missing fields backfill."""
        planner = self._make_planner()
        assert planner.decode_worker_info.max_num_batched_tokens is None

        self._install_mock_connector(
            planner,
            max_num_batched_tokens=8192,
            total_kv_blocks=1024,
            max_num_seqs=256,
            kv_cache_block_size=16,
            context_length=4096,
        )

        planner._refresh_worker_info_from_connector()
        assert planner.decode_worker_info.max_num_batched_tokens == 8192
        assert planner.decode_worker_info.total_kv_blocks == 1024
        assert planner.decode_worker_info.max_num_seqs == 256
        assert planner.decode_worker_info.kv_cache_block_size == 16
        assert planner.decode_worker_info.context_length == 4096

    def test_noop_when_already_set(self):
        """Does not re-query once max_num_batched_tokens is populated."""
        planner = self._make_planner()
        planner.decode_worker_info = WorkerInfo(max_num_batched_tokens=2048)

        mock_connector = self._install_mock_connector(
            planner, max_num_batched_tokens=8192
        )

        planner._refresh_worker_info_from_connector()
        assert planner.decode_worker_info.max_num_batched_tokens == 2048
        mock_connector.get_worker_info.assert_not_called()

    def test_noop_when_connector_lacks_get_worker_info(self):
        """Silently does nothing if the connector does not implement get_worker_info."""
        planner = self._make_planner()

        class _StubConnector:
            pass

        planner.connector = _StubConnector()
        planner._refresh_worker_info_from_connector()
        assert planner.decode_worker_info.max_num_batched_tokens is None

    def test_noop_when_connector_returns_none_fields(self):
        """Fresh WorkerInfo with None everywhere does not overwrite anything."""
        planner = self._make_planner()
        self._install_mock_connector(planner)  # All Nones
        planner._refresh_worker_info_from_connector()
        assert planner.decode_worker_info.max_num_batched_tokens is None

    def test_exception_does_not_propagate(self):
        """If connector.get_worker_info throws, refresh is a no-op."""
        planner = self._make_planner()
        mock_connector = Mock()
        mock_connector.get_worker_info.side_effect = RuntimeError("boom")
        planner.connector = mock_connector

        planner._refresh_worker_info_from_connector()  # must not raise
        assert planner.decode_worker_info.max_num_batched_tokens is None

    def test_updates_state_machine_capabilities(self):
        """State machine capabilities are updated via update_capabilities()."""
        planner = self._make_planner()
        _ = planner.state_machine
        assert planner._state_machine is not None

        self._install_mock_connector(planner, max_num_batched_tokens=4096)

        planner._refresh_worker_info_from_connector()
        assert planner.decode_worker_info.max_num_batched_tokens == 4096
        assert (
            planner._state_machine._capabilities.decode.max_num_batched_tokens == 4096
        )

    def test_refresh_skips_unneeded_sub_component(self):
        """Only sub-components with require_* True are refreshed."""
        planner = self._make_planner(require_prefill=False, require_decode=True)

        def _side_effect(sub_type, backend):
            # Should only be called for DECODE.
            assert sub_type.value == "decode"
            return WorkerInfo(max_num_batched_tokens=4096)

        mock_connector = Mock()
        mock_connector.get_worker_info.side_effect = _side_effect
        planner.connector = mock_connector

        planner._refresh_worker_info_from_connector()
        assert planner.prefill_worker_info.max_num_batched_tokens is None
        assert planner.decode_worker_info.max_num_batched_tokens == 4096
