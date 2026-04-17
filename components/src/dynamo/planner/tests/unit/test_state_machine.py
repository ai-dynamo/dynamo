# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core-only planner tests: TickInput -> PlannerEffects, no mocks."""

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
from dynamo.planner.core.state_machine import PlannerStateMachine
from dynamo.planner.core.types import (
    EngineCapabilities,
    FpmObservations,
    ScheduledTick,
    TickInput,
    TrafficObservation,
    WorkerCapabilities,
    WorkerCounts,
)


def _tick_for(tick_input: TickInput) -> ScheduledTick:
    """Build a ScheduledTick matching the data present in a TickInput."""
    has_fpm = tick_input.fpm_observations is not None
    has_traffic = tick_input.traffic is not None
    return ScheduledTick(
        at_s=tick_input.now_s,
        run_load_scaling=has_fpm,
        run_throughput_scaling=has_traffic,
        need_worker_states=True,
        need_worker_fpm=has_fpm,
        need_traffic_metrics=has_traffic,
        traffic_metrics_duration_s=tick_input.traffic.duration_s
        if has_traffic
        else 0.0,
    )


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


def _make_config(**overrides) -> PlannerConfig:
    defaults = dict(
        mode="disagg",
        optimization_target="sla",
        ttft=500.0,
        itl=50.0,
        min_endpoint=1,
        max_gpu_budget=-1,
        throughput_adjustment_interval=60,
        load_adjustment_interval=5,
        load_scaling_down_sensitivity=80,
        max_num_fpm_samples=50,
        fpm_sample_bucket_size=16,
        load_min_observations=5,
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        load_predictor="constant",
        backend="vllm",
        metric_pulling_prometheus_endpoint="http://localhost:9090",
        metric_reporting_prometheus_port=0,
    )
    defaults.update(overrides)
    return PlannerConfig.model_construct(**defaults)


def _default_caps() -> WorkerCapabilities:
    return WorkerCapabilities(
        prefill=EngineCapabilities(num_gpu=1, max_num_batched_tokens=2048),
        decode=EngineCapabilities(num_gpu=1, max_num_batched_tokens=2048),
    )


def _agg_caps() -> WorkerCapabilities:
    return WorkerCapabilities(
        decode=EngineCapabilities(num_gpu=1, max_num_batched_tokens=2048),
    )


def _agg_config(**overrides) -> PlannerConfig:
    return _make_config(mode="agg", **overrides)


def _make_core(config=None, caps=None, **config_overrides) -> PlannerStateMachine:
    cfg = config or _make_config(**config_overrides)
    return PlannerStateMachine(cfg, caps or _default_caps())


def _make_agg_core(config=None, caps=None, **config_overrides) -> PlannerStateMachine:
    cfg = config or _agg_config(**config_overrides)
    return PlannerStateMachine(cfg, caps or _agg_caps())


def _train_prefill_regression(core: PlannerStateMachine) -> None:
    fpms = [
        _make_fpm(
            sum_prefill_tokens=t, num_prefill_requests=1, wall_time=0.001 * t + 0.002
        )
        for t in [500, 1000, 1500, 2000, 2500]
    ]
    core.load_benchmark_fpms(prefill_fpms=fpms)


def _train_decode_regression(core: PlannerStateMachine) -> None:
    fpms = [
        _make_fpm(
            sum_decode_kv_tokens=kv,
            num_decode_requests=n,
            wall_time=0.00001 * kv + 0.001,
        )
        for n, kv in [(5, 5000), (10, 10000), (20, 20000), (30, 30000), (40, 40000)]
    ]
    core.load_benchmark_fpms(decode_fpms=fpms)


# ── Initial ticks ─────────────────────────────────────────────────────


class TestInitialTick:
    def test_both_enabled_returns_earliest(self):
        core = _make_core()
        tick = core.initial_tick(start_s=100.0)
        # Load interval (5s) < throughput interval (60s), so load tick first
        assert tick.at_s == 105.0
        assert tick.need_worker_fpm
        assert not tick.need_traffic_metrics

    def test_load_only(self):
        core = _make_core(enable_throughput_scaling=False)
        tick = core.initial_tick(start_s=0.0)
        assert tick.at_s == 5.0
        assert tick.need_worker_fpm
        assert not tick.need_traffic_metrics

    def test_throughput_only(self):
        core = _make_core(enable_load_scaling=False)
        tick = core.initial_tick(start_s=0.0)
        # Load tick is still scheduled (feeds regression) at 5s < 60s
        assert tick.at_s == 5.0
        assert tick.need_worker_fpm


# ── Load benchmark bootstrapping ──────────────────────────────────────


class TestBenchmarkBootstrap:
    def test_prefill_regression_bootstrapped(self):
        core = _make_core(mode="prefill")
        _train_prefill_regression(core)
        assert core.prefill_regression.has_sufficient_data()

    def test_decode_regression_bootstrapped(self):
        core = _make_core(mode="decode")
        _train_decode_regression(core)
        assert core.decode_regression.has_sufficient_data()


# ── FPM observation via on_tick ───────────────────────────────────────


class TestFpmObservation:
    def test_fpm_feeds_regression(self):
        core = _make_core(mode="prefill")
        assert core.prefill_regression.num_observations == 0

        fpm = _make_fpm(sum_prefill_tokens=500, num_prefill_requests=1, wall_time=0.5)
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(prefill={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        core.on_tick(_tick_for(tick), tick)
        assert core.prefill_regression.num_observations == 1

    def test_next_tick_scheduled_after_fpm(self):
        core = _make_core(mode="prefill")
        tick = TickInput(
            now_s=10.0,
            fpm_observations=FpmObservations(
                prefill={
                    ("w1", 0): _make_fpm(
                        sum_prefill_tokens=500,
                        num_prefill_requests=1,
                        wall_time=0.5,
                    )
                }
            ),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.next_tick is not None
        assert effects.next_tick.at_s == 15.0
        assert effects.next_tick.need_worker_fpm


# ── Load-based scaling (prefill) ──────────────────────────────────────


class TestPrefillLoadScaling:
    def test_scale_up_when_all_above_sla(self):
        core = _make_core(mode="prefill", ttft=5.0)
        _train_prefill_regression(core)

        fpm = _make_fpm(
            worker_id="w1",
            queued_prefill_tokens=10000,
            sum_prefill_tokens=500,
            num_prefill_requests=1,
            wall_time=0.5,
        )
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(prefill={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_prefill is not None
        assert effects.scale_to.num_prefill > 1
        assert effects.diagnostics.estimated_ttft_ms is not None
        assert effects.diagnostics.estimated_ttft_ms > 0
        assert effects.diagnostics.load_decision_reason == "scale_up"

    def test_no_scaling_when_insufficient_data(self):
        core = _make_core(mode="prefill")
        fpm = _make_fpm(
            queued_prefill_tokens=5000, sum_prefill_tokens=100, wall_time=0.1
        )
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(prefill={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is None
        assert effects.diagnostics.load_decision_reason == "insufficient_data"

    def test_no_scaling_when_load_disabled(self):
        core = _make_core(mode="prefill", enable_load_scaling=False)
        _train_prefill_regression(core)

        fpm = _make_fpm(
            queued_prefill_tokens=10000,
            sum_prefill_tokens=500,
            num_prefill_requests=1,
            wall_time=0.5,
        )
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(prefill={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is None
        assert effects.diagnostics.load_decision_reason == "disabled"


# ── Load-based scaling (decode) ───────────────────────────────────────


class TestDecodeLoadScaling:
    def test_scale_up_when_all_above_sla(self):
        core = _make_core(mode="decode", itl=5.0)
        _train_decode_regression(core)

        fpm = _make_fpm(
            worker_id="w1",
            sum_decode_kv_tokens=30000,
            queued_decode_kv_tokens=20000,
            num_decode_requests=30,
            wall_time=0.3,
        )
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(decode={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_decode=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_decode is not None
        assert effects.scale_to.num_decode > 1
        assert effects.diagnostics.estimated_itl_ms is not None
        assert effects.diagnostics.estimated_itl_ms > 0
        assert effects.diagnostics.load_decision_reason == "scale_up"


# ── Disagg load scaling ───────────────────────────────────────────────


class TestDisaggLoadScaling:
    def test_disagg_scale_up(self):
        core = _make_core(ttft=5.0, itl=5.0)
        _train_prefill_regression(core)
        _train_decode_regression(core)

        p_fpm = _make_fpm(
            worker_id="w1",
            queued_prefill_tokens=10000,
            sum_prefill_tokens=500,
            num_prefill_requests=1,
            wall_time=0.5,
        )
        d_fpm = _make_fpm(
            worker_id="w1",
            sum_decode_kv_tokens=5000,
            queued_decode_kv_tokens=3000,
            num_decode_requests=20,
            wall_time=0.6,
        )
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                prefill={("w1", 0): p_fpm},
                decode={("w1", 0): d_fpm},
            ),
            worker_counts=WorkerCounts(ready_num_prefill=1, ready_num_decode=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None


# ── Throughput scaling ────────────────────────────────────────────────


class TestThroughputScaling:
    def test_throughput_only_returns_decision(self):
        core = _make_core(
            mode="prefill", enable_load_scaling=False, enable_throughput_scaling=True
        )
        _train_prefill_regression(core)

        # Warm predictor with traffic
        core._observe_traffic(
            TrafficObservation(duration_s=60, num_req=100, isl=1000, osl=150)
        )

        tick = TickInput(
            now_s=60.0,
            traffic=TrafficObservation(duration_s=60, num_req=100, isl=1000, osl=150),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_prefill is not None
        assert effects.scale_to.num_prefill >= 1
        assert effects.diagnostics.predicted_num_req is not None
        assert effects.diagnostics.engine_rps_prefill is not None
        assert effects.diagnostics.throughput_decision_reason == "scale"

    def test_throughput_sets_lower_bound_when_load_enabled(self):
        core = _make_core(enable_load_scaling=True, enable_throughput_scaling=True)
        _train_prefill_regression(core)
        _train_decode_regression(core)

        core._observe_traffic(
            TrafficObservation(duration_s=60, num_req=100, isl=1000, osl=150)
        )

        tick = TickInput(
            now_s=60.0,
            traffic=TrafficObservation(duration_s=60, num_req=100, isl=1000, osl=150),
            worker_counts=WorkerCounts(ready_num_prefill=1, ready_num_decode=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        # When both modes enabled, throughput tick returns None (just sets lower bound)
        assert effects.scale_to is None
        assert core._throughput_lower_bound_p >= 1
        assert core._throughput_lower_bound_d >= 1
        assert effects.diagnostics.throughput_decision_reason == "set_lower_bound"

    def test_next_tick_scheduled_after_traffic(self):
        core = _make_core(mode="prefill")
        tick = TickInput(
            now_s=60.0,
            traffic=TrafficObservation(duration_s=60, num_req=0, isl=0, osl=0),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.next_tick is not None
        assert effects.next_tick.need_traffic_metrics
        assert effects.next_tick.at_s == 120.0


class TestKvHitRatePlumbing:
    def test_observe_traffic_updates_last_kv_hit_rate(self):
        core = _make_core()
        core._observe_traffic(
            TrafficObservation(
                duration_s=60, num_req=100, isl=1000, osl=150, kv_hit_rate=0.3
            )
        )
        assert core._last_kv_hit_rate == 0.3

    def test_none_kv_hit_rate_leaves_last_value_unchanged(self):
        core = _make_core()
        core._observe_traffic(
            TrafficObservation(
                duration_s=60, num_req=100, isl=1000, osl=150, kv_hit_rate=0.42
            )
        )
        # Subsequent observation without a hit rate (scrape failure / frontend
        # source) must not clobber the sticky value -- the planner keeps
        # using the most recent valid reading.
        core._observe_traffic(
            TrafficObservation(
                duration_s=60, num_req=100, isl=1000, osl=150, kv_hit_rate=None
            )
        )
        assert core._last_kv_hit_rate == 0.42

    def test_nan_kv_hit_rate_is_ignored(self):
        core = _make_core()
        core._observe_traffic(
            TrafficObservation(
                duration_s=60, num_req=100, isl=1000, osl=150, kv_hit_rate=0.5
            )
        )
        core._observe_traffic(
            TrafficObservation(
                duration_s=60,
                num_req=100,
                isl=1000,
                osl=150,
                kv_hit_rate=float("nan"),
            )
        )
        assert core._last_kv_hit_rate == 0.5

    def test_warm_load_predictors_skips_kv_hit_rate(self):
        """kv_hit_rate has no good offline-trace proxy, so it must not
        receive warmup data (only live observations feed it)."""
        core = _make_core()
        observations = [
            TrafficObservation(
                duration_s=60, num_req=50 * i, isl=1000, osl=150, kv_hit_rate=0.1 * i
            )
            for i in range(1, 4)
        ]
        core.warm_load_predictors(observations)
        # Other predictors accumulated their respective series
        assert len(core._num_req_predictor.data_buffer) == 3
        assert len(core._isl_predictor.data_buffer) == 3
        assert len(core._osl_predictor.data_buffer) == 3
        # kv_hit_rate predictor stayed cold
        assert core._kv_hit_rate_predictor.data_buffer == []

    def test_throughput_diagnostics_include_predicted_kv_hit_rate(self):
        core = _make_core(
            mode="prefill", enable_load_scaling=False, enable_throughput_scaling=True
        )
        _train_prefill_regression(core)
        core._observe_traffic(
            TrafficObservation(
                duration_s=60, num_req=100, isl=1000, osl=150, kv_hit_rate=0.4
            )
        )
        tick = TickInput(
            now_s=60.0,
            traffic=TrafficObservation(
                duration_s=60, num_req=100, isl=1000, osl=150, kv_hit_rate=0.4
            ),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        # ConstantPredictor predicts the last value it saw
        assert effects.diagnostics.predicted_kv_hit_rate == 0.4

    def test_high_predicted_hit_rate_reduces_prefill_replicas(self):
        """With the same demand + regression, a high predicted hit rate
        should yield fewer (or at worst equal) prefill replicas than no
        reuse."""
        core_base = _make_core(
            mode="prefill", enable_load_scaling=False, enable_throughput_scaling=True
        )
        _train_prefill_regression(core_base)
        core_hit = _make_core(
            mode="prefill", enable_load_scaling=False, enable_throughput_scaling=True
        )
        _train_prefill_regression(core_hit)

        # Feed several observations so the (constant) predictor locks in.
        traffic_base = TrafficObservation(
            duration_s=60, num_req=500, isl=4000, osl=150, kv_hit_rate=0.0
        )
        traffic_hit = TrafficObservation(
            duration_s=60, num_req=500, isl=4000, osl=150, kv_hit_rate=0.8
        )
        core_base._observe_traffic(traffic_base)
        core_hit._observe_traffic(traffic_hit)

        tick_base = TickInput(
            now_s=60.0,
            traffic=traffic_base,
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        tick_hit = TickInput(
            now_s=60.0,
            traffic=traffic_hit,
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        effects_base = core_base.on_tick(_tick_for(tick_base), tick_base)
        effects_hit = core_hit.on_tick(_tick_for(tick_hit), tick_hit)
        assert effects_base.scale_to is not None
        assert effects_hit.scale_to is not None
        assert effects_hit.scale_to.num_prefill <= effects_base.scale_to.num_prefill


# ── FPM reconciliation ───────────────────────────────────────────────


class TestFpmReconciliation:
    def test_mismatch_skips_scaling(self):
        core = _make_core(mode="prefill", ttft=5.0)
        _train_prefill_regression(core)

        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                prefill={
                    ("w1", 0): _make_fpm(
                        queued_prefill_tokens=10000,
                        sum_prefill_tokens=500,
                        num_prefill_requests=1,
                        wall_time=0.5,
                    ),
                    ("w2", 0): _make_fpm(
                        worker_id="w2",
                        queued_prefill_tokens=8000,
                        sum_prefill_tokens=500,
                        num_prefill_requests=1,
                        wall_time=0.5,
                    ),
                }
            ),
            worker_counts=WorkerCounts(ready_num_prefill=3),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        # FPM reports 2 workers but ready count is 3 -> skip scaling
        assert effects.scale_to is None
        assert effects.diagnostics.load_decision_reason == "worker_count_mismatch"


# ── Agg planner core ──────────────────────────────────────────────────


class TestAggPlannerStateMachine:
    def _train_agg(self, core: PlannerStateMachine) -> None:
        fpms = [
            _make_fpm(
                sum_prefill_tokens=p,
                num_prefill_requests=1,
                sum_decode_kv_tokens=d,
                num_decode_requests=10,
                wall_time=0.001 * p + 0.0001 * d + 0.001,
            )
            for p, d in [
                (100, 1000),
                (200, 2000),
                (300, 3000),
                (400, 4000),
                (500, 5000),
            ]
        ]
        core.load_benchmark_fpms(agg_fpms=fpms)

    def test_initial_tick(self):
        core = _make_agg_core()
        tick = core.initial_tick(start_s=0.0)
        assert tick.at_s == 5.0
        assert tick.need_worker_fpm

    def test_fpm_feeds_regression(self):
        core = _make_agg_core()
        assert core.regression.num_observations == 0
        fpm = _make_fpm(
            sum_prefill_tokens=200,
            num_prefill_requests=1,
            sum_decode_kv_tokens=2000,
            num_decode_requests=10,
            wall_time=0.3,
        )
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(decode={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_decode=1),
        )
        core.on_tick(_tick_for(tick), tick)
        assert core.regression.num_observations == 1

    def test_throughput_only_returns_decision(self):
        core = _make_agg_core(enable_load_scaling=False, enable_throughput_scaling=True)
        self._train_agg(core)

        core._observe_traffic(
            TrafficObservation(duration_s=60, num_req=100, isl=1000, osl=150)
        )

        tick = TickInput(
            now_s=60.0,
            traffic=TrafficObservation(duration_s=60, num_req=100, isl=1000, osl=150),
            worker_counts=WorkerCounts(ready_num_decode=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_decode is not None
        assert effects.scale_to.num_decode >= 1


# ── Diagnostics ──────────────────────────────────────────────────────


class TestDiagnostics:
    """Verify TickDiagnostics is populated correctly across tick types."""

    def test_diagnostics_always_present(self):
        core = _make_core(mode="prefill")
        tick = TickInput(
            now_s=5.0,
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.diagnostics is not None

    def test_diagnostics_reset_each_tick(self):
        core = _make_core(mode="prefill", ttft=5.0)
        _train_prefill_regression(core)

        fpm = _make_fpm(
            queued_prefill_tokens=10000,
            sum_prefill_tokens=500,
            num_prefill_requests=1,
            wall_time=0.5,
        )
        tick1 = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(prefill={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        effects1 = core.on_tick(_tick_for(tick1), tick1)
        assert effects1.diagnostics.estimated_ttft_ms is not None

        tick2 = TickInput(
            now_s=10.0,
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        st2 = ScheduledTick(
            at_s=10.0,
            run_load_scaling=False,
            run_throughput_scaling=False,
            need_worker_states=True,
        )
        effects2 = core.on_tick(st2, tick2)
        assert effects2.diagnostics.estimated_ttft_ms is None
        assert effects2.diagnostics.load_decision_reason is None

    def test_no_fpm_data_reason(self):
        core = _make_core(mode="prefill")
        _train_prefill_regression(core)
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(prefill=None),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.diagnostics.load_decision_reason == "no_fpm_data"

    def test_throughput_predicted_load_populated(self):
        core = _make_core(
            mode="prefill", enable_load_scaling=False, enable_throughput_scaling=True
        )
        _train_prefill_regression(core)
        core._observe_traffic(
            TrafficObservation(duration_s=60, num_req=100, isl=1000, osl=150)
        )

        tick = TickInput(
            now_s=60.0,
            traffic=TrafficObservation(duration_s=60, num_req=100, isl=1000, osl=150),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        diag = effects.diagnostics
        assert diag.predicted_num_req is not None
        assert diag.predicted_isl is not None
        assert diag.predicted_osl is not None
        assert diag.engine_rps_prefill is not None
        assert diag.engine_rps_prefill > 0
