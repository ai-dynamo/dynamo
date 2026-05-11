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
        # Load-only mode rides a kv-hit-rate scrape on the load tick so the
        # planner can discount prefill work by recent prefix reuse.
        assert tick.need_traffic_metrics
        assert tick.traffic_metrics_duration_s == 5.0

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


# ── Consolidation-aware scale-down ────────────────────────────────────


def _decode_caps_with_max_kv(max_kv_tokens: int) -> WorkerCapabilities:
    """Decode-only capabilities advertising a max_kv_tokens budget."""
    return WorkerCapabilities(
        decode=EngineCapabilities(
            num_gpu=1,
            max_num_batched_tokens=2048,
            max_kv_tokens=max_kv_tokens,
        ),
    )


class TestDecodeConsolidationAwareScaleDown:
    """Decode scale-down uses two checks at the survivor's post-consolidation
    KV (current sched+queued scaled by N/(N-1)):

    1. **Cache feasibility** -- post_kv must fit within ``max_kv_tokens``;
       crossing it forces block eviction / queueing, a non-linear regime
       outside the regression's domain.
    2. **SLA check** -- regression-predicted ITL at post_kv must stay
       within ``SLA * sensitivity``.

    Either failure refuses the scale-down.
    """

    def _setup(self, *, itl_sla: float = 100.0, max_kv_tokens: int = 100_000):
        core = _make_core(
            mode="decode",
            itl=itl_sla,
            load_scaling_down_sensitivity=80,
        )
        core._capabilities = _decode_caps_with_max_kv(max_kv_tokens)
        _train_decode_regression(core)
        return core

    def _tick(self, *, num_workers: int, sched_kv_per_worker: int) -> TickInput:
        decode = {}
        for i in range(num_workers):
            decode[(f"w{i}", 0)] = _make_fpm(
                worker_id=f"w{i}",
                sum_decode_kv_tokens=sched_kv_per_worker,
                num_decode_requests=max(1, sched_kv_per_worker // 1000),
                # Match _train_decode_regression's wall_time formula so the
                # post-bootstrap refit on each tick stays monotone in kv;
                # otherwise the regression rejects the fit and decisions skip.
                wall_time=0.00001 * sched_kv_per_worker + 0.001,
            )
        return TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(decode=decode),
            worker_counts=WorkerCounts(ready_num_decode=num_workers),
        )

    def test_post_consolidation_within_sla_permits(self):
        """Light load: post_kv well under cache and SLA -> ALLOW.

        N=2, sched_kv=1500. post_kv = 3000. Predicted ITL ~= 0.001 +
        0.00001 * 4000 ~= 41 ms (with internal avg_decode_len), under
        the 80 ms threshold. No scale-up either (under 100 ms SLA).
        """
        core = self._setup(itl_sla=100.0)
        tick = self._tick(num_workers=2, sched_kv_per_worker=1_500)
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_decode == 1

    def test_post_consolidation_breaches_sla_refuses(self):
        """Cache fine but predicted ITL > SLA*sensitivity -> SLA check refuses.

        N=2, sched_kv=8000. post_kv=16000 (well below 100K cache). Predicted
        ITL ~= 0.001 + 0.00001 * 17000 ~= 171 ms, above the 80 ms threshold.
        """
        core = self._setup(itl_sla=100.0)
        tick = self._tick(num_workers=2, sched_kv_per_worker=8_000)
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is None or effects.scale_to.num_decode == 2
        assert (
            effects.diagnostics.load_decision_reason
            == "scale_down_refused_consolidation"
        )

    def test_post_consolidation_exceeds_max_kv_refuses(self):
        """Hard cache fail-safe: post_kv >= max_kv -> refuse outright.

        N=2, sched_kv=60_000. post_kv=120_000 >= max_kv 100_000. SLA is
        effectively off (10s) so only the cache check can refuse.
        """
        core = self._setup(itl_sla=10_000.0, max_kv_tokens=100_000)
        tick = self._tick(num_workers=2, sched_kv_per_worker=60_000)
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is None or effects.scale_to.num_decode == 2
        assert (
            effects.diagnostics.load_decision_reason
            == "scale_down_refused_consolidation"
        )

    def test_no_max_kv_falls_through_to_sla_check(self):
        """Without max_kv_tokens, only the SLA check governs.

        Cache check is skipped (no denominator); the regression still gates
        scale-down by predicted ITL. Light load passes -> ALLOW.
        """
        core = self._setup(itl_sla=100.0)
        # Erase max_kv: cache check becomes a no-op.
        core._capabilities = WorkerCapabilities(
            decode=EngineCapabilities(num_gpu=1, max_num_batched_tokens=2048),
        )
        tick = self._tick(num_workers=2, sched_kv_per_worker=1_500)
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_decode == 1


def _train_slow_prefill_regression(core: PlannerStateMachine) -> None:
    """Trains a regression with low slope so chunked TTFTs stay tractable.

    Slope 1e-5 (not 1e-6) is the floor where np.linalg keeps the coefficient
    reliably positive across small fits -- 1e-6 sometimes computes as
    slightly negative from floating-point noise and the fit gets rejected.
    """
    fpms = [
        _make_fpm(
            sum_prefill_tokens=t,
            num_prefill_requests=1,
            wall_time=1e-5 * t + 1e-3,
        )
        for t in [500, 1000, 1500, 2000, 2500]
    ]
    core.load_benchmark_fpms(prefill_fpms=fpms)


class TestPrefillConsolidationAwareScaleDown:
    """Prefill scale-down re-runs ``estimate_next_ttft`` with scaled queue.

    The regression's internal ``avg_isl`` (own-request compute) is unchanged
    by consolidation -- only the queue input is scaled. This guards against
    inflating the new request's prefill compute time.
    """

    def _setup(self, ttft: float = 100.0):
        core = _make_core(
            mode="prefill",
            ttft=ttft,
            load_scaling_down_sensitivity=80,
        )
        _train_slow_prefill_regression(core)
        return core

    def _tick(self, *, num_workers: int, queued_per_worker: int) -> TickInput:
        prefill = {}
        for i in range(num_workers):
            prefill[(f"w{i}", 0)] = _make_fpm(
                worker_id=f"w{i}",
                queued_prefill_tokens=queued_per_worker,
                sum_prefill_tokens=500,
                num_prefill_requests=1,
                # Match _train_slow_prefill_regression to keep the per-tick
                # refit monotone (1e-5 * 500 + 1e-3 = 0.006).
                wall_time=1e-5 * 500 + 1e-3,
            )
        return TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(prefill=prefill),
            worker_counts=WorkerCounts(ready_num_prefill=num_workers),
        )

    def test_n2_refuses_when_post_consolidation_breaks_sla(self):
        """High queue at N=2: post-consolidation TTFT exceeds SLA * 0.8."""
        core = self._setup(ttft=100.0)
        # avg_isl=1500 from training; queue 30K per worker doubled to 60K
        # post-consolidation -> ceil((60000+1500)/2048)=31 chunks ~= 95+ ms.
        tick = self._tick(num_workers=2, queued_per_worker=30_000)
        effects = core.on_tick(_tick_for(tick), tick)
        assert (
            effects.scale_to is None
            or effects.scale_to.num_prefill is None
            or effects.scale_to.num_prefill >= 2
        )

    def test_n2_permits_when_queue_empty(self):
        """At N=2 with empty queues, post-consolidation TTFT stays within SLA."""
        core = self._setup(ttft=100.0)
        tick = self._tick(num_workers=2, queued_per_worker=0)
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_prefill == 1

    def test_consolidation_only_inflates_queue_not_compute(self):
        """At N=10 the queue-only scaling lets us scale down at queue sizes
        that would refuse if we'd naively multiplied the whole TTFT by 10/9.

        With queue=2000 per worker: avg_isl=1500 dominates the TTFT (~=3 ms)
        and scaling N->N-1 only inflates the queue portion. Post-consolidation
        queue ~= 2222, total ~= 3722 -> still 2 chunks -> predicted TTFT remains
        well under 100ms * 0.8.
        """
        core = self._setup(ttft=100.0)
        core._num_p_workers = 10  # ensure reconcile sees 10 workers
        tick = self._tick(num_workers=10, queued_per_worker=2_000)
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_prefill == 9


def _train_prefill_regression_high_own_compute(core: PlannerStateMachine) -> None:
    """Trains a steeper regression so own-compute is a sizeable fraction of SLA.

    With wall_time = 1e-5*t + 1e-3 and ISLs in [1500..2500] (avg_isl=2000),
    a single chunk at MBT=2048 is ~= 21.5 ms and T_own (queue=0) is ~= 21 ms.
    """
    fpms = [
        _make_fpm(
            sum_prefill_tokens=t,
            num_prefill_requests=1,
            wall_time=1e-5 * t + 1e-3,
        )
        for t in [1500, 1750, 2000, 2250, 2500]
    ]
    core.load_benchmark_fpms(prefill_fpms=fpms)


class TestPrefillQueueBudgetRefinement:
    """Sensitivity applies to the queue-induced TTFT, not the full TTFT.

    When ``T_own`` (own-compute, fixed cost) is a meaningful fraction of SLA,
    the old `TTFT(post_queue) < SLA * sensitivity` check over-penalises the
    queue budget by spending sensitivity on the unavoidable own-compute.
    The corrected check allows scale-down when the queue-induced TTFT after
    consolidation fits within ``(SLA - T_own) * sensitivity``.
    """

    def _setup(self, ttft: float = 50.0):
        core = _make_core(
            mode="prefill",
            ttft=ttft,
            load_scaling_down_sensitivity=80,
        )
        _train_prefill_regression_high_own_compute(core)
        return core

    def _tick(self, *, num_workers: int, queued_per_worker: int) -> TickInput:
        prefill = {}
        for i in range(num_workers):
            prefill[(f"w{i}", 0)] = _make_fpm(
                worker_id=f"w{i}",
                queued_prefill_tokens=queued_per_worker,
                sum_prefill_tokens=2000,
                num_prefill_requests=1,
                wall_time=0.021,
            )
        return TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(prefill=prefill),
            worker_counts=WorkerCounts(ready_num_prefill=num_workers),
        )

    def test_high_own_compute_permits_scale_down(self):
        """Old check would have refused; new queue-budget check allows.

        N=2, queue=1000, SLA=50ms, T_own~=21ms.
        Post-consolidation TTFT ~= 42ms -> exceeds old SLA*0.8 = 40ms (refuse).
        Queue-induced post ~= 21ms < (50-21)*0.8 = 23.2ms (allow).
        """
        core = self._setup(ttft=50.0)
        tick = self._tick(num_workers=2, queued_per_worker=1_000)
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_prefill == 1

    def test_t_own_above_sla_blocks_scale_down(self):
        """When T_own alone exceeds SLA, queue_budget <= 0 -> refuse.

        SLA=15ms but T_own~=21ms -> new request can't even meet SLA empty;
        scaling down would only worsen contention.
        """
        core = self._setup(ttft=15.0)
        tick = self._tick(num_workers=2, queued_per_worker=0)
        effects = core.on_tick(_tick_for(tick), tick)
        assert (
            effects.scale_to is None
            or effects.scale_to.num_prefill is None
            or effects.scale_to.num_prefill >= 2
        )


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

    def test_ttft_sla_floor_overrides_throughput_ratio_under_backpressure(self):
        """Regression: when demand_rps << engine_rps (backpressure), the latency
        violation must still drive scale-up rather than resolving to 1."""
        # ttft_sla=200ms; regression gives wt(1000 tokens) ≈ 1.002s → ttft≈1002ms.
        # sla_floor = ceil(1002/200) = 6.  With near-zero demand the raw
        # ceil(demand_rps/engine_rps) would return 1 without the fix.
        core = _make_core(
            mode="prefill",
            enable_load_scaling=False,
            enable_throughput_scaling=True,
            ttft=200.0,
        )
        _train_prefill_regression(core)

        # Tiny demand to reproduce the backpressure equilibrium
        core._observe_traffic(
            TrafficObservation(duration_s=60, num_req=1, isl=1000, osl=150)
        )

        tick = TickInput(
            now_s=60.0,
            traffic=TrafficObservation(duration_s=60, num_req=1, isl=1000, osl=150),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        # ceil(demand_rps/engine_rps) = ceil(0.017/1.0) = 1 without the floor.
        # With the SLA floor: ceil(1002/200) = 6.
        assert effects.scale_to.num_prefill is not None
        assert effects.scale_to.num_prefill >= 6

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
    def test_load_only_observe_traffic_updates_last_kv_hit_rate(self):
        core = _make_core(enable_throughput_scaling=False)
        core._observe_traffic(
            TrafficObservation(
                duration_s=5, num_req=100, isl=1000, osl=150, kv_hit_rate=0.3
            )
        )
        assert core._last_kv_hit_rate == 0.3

    def test_load_only_skips_throughput_predictor_feeds(self):
        """In load-only mode the throughput predictors have no consumer; we
        must not pollute their buffers with placeholder zeros."""
        core = _make_core(enable_throughput_scaling=False)
        core._observe_traffic(
            TrafficObservation(duration_s=5, num_req=0, isl=0, osl=0, kv_hit_rate=0.4)
        )
        assert core._num_req_predictor.data_buffer == []
        assert core._isl_predictor.data_buffer == []
        assert core._osl_predictor.data_buffer == []
        # kv predictor also untouched in load-only mode (no prediction needed)
        assert core._kv_hit_rate_predictor.data_buffer == []

    def test_load_only_none_kv_hit_rate_leaves_last_value_unchanged(self):
        core = _make_core(enable_throughput_scaling=False)
        core._observe_traffic(
            TrafficObservation(duration_s=5, num_req=0, isl=0, osl=0, kv_hit_rate=0.42)
        )
        # Subsequent observation without a hit rate (scrape failure / frontend
        # source) must not clobber the sticky value -- the planner keeps
        # using the most recent valid reading.
        core._observe_traffic(
            TrafficObservation(duration_s=5, num_req=0, isl=0, osl=0, kv_hit_rate=None)
        )
        assert core._last_kv_hit_rate == 0.42

    def test_load_only_nan_kv_hit_rate_is_ignored(self):
        core = _make_core(enable_throughput_scaling=False)
        core._observe_traffic(
            TrafficObservation(duration_s=5, num_req=0, isl=0, osl=0, kv_hit_rate=0.5)
        )
        core._observe_traffic(
            TrafficObservation(
                duration_s=5,
                num_req=0,
                isl=0,
                osl=0,
                kv_hit_rate=float("nan"),
            )
        )
        assert core._last_kv_hit_rate == 0.5

    def test_mixed_mode_observe_traffic_feeds_predictor_only(self):
        """In mixed mode the raw observation feeds the predictor; the sticky
        ``_last_kv_hit_rate`` is *not* updated until ``_advance_throughput``
        promotes the predicted value to it."""
        core = _make_core()  # both load + throughput scaling enabled
        assert core._last_kv_hit_rate is None
        core._observe_traffic(
            TrafficObservation(
                duration_s=60, num_req=100, isl=1000, osl=150, kv_hit_rate=0.3
            )
        )
        # Predictor saw the observation
        assert len(core._kv_hit_rate_predictor.data_buffer) == 1
        # Sticky value is *not* set from the raw observation in mixed mode
        assert core._last_kv_hit_rate is None

    def test_mixed_mode_advance_throughput_promotes_predicted_value(self):
        """After a throughput tick fires, ``_last_kv_hit_rate`` should hold
        the predicted value (used by all subsequent load ticks until the
        next throughput tick)."""
        core = _make_core(
            mode="prefill", enable_load_scaling=True, enable_throughput_scaling=True
        )
        _train_prefill_regression(core)
        # ConstantPredictor returns the last observed value once min_data_points=1.
        # Feed a known value and run a throughput tick.
        traffic = TrafficObservation(
            duration_s=60, num_req=100, isl=1000, osl=150, kv_hit_rate=0.6
        )
        tick_input = TickInput(
            now_s=60.0,
            traffic=traffic,
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        core.on_tick(_tick_for(tick_input), tick_input)
        # Constant predictor returns 0.6, which is then promoted to sticky
        assert core._last_kv_hit_rate == pytest.approx(0.6)

    def test_load_only_scheduler_sets_need_traffic_on_load_tick(self):
        core = _make_core(
            mode="prefill",
            enable_load_scaling=True,
            enable_throughput_scaling=False,
            load_adjustment_interval=7,
        )
        tick = core.initial_tick(start_s=0.0)
        # Load-only mode: the load tick should request a kv-hit-rate scrape
        # over the load interval.
        assert tick.run_load_scaling
        assert not tick.run_throughput_scaling
        assert tick.need_traffic_metrics
        assert tick.traffic_metrics_duration_s == 7.0

    def test_throughput_enabled_scheduler_skips_traffic_on_pure_load_tick(self):
        core = _make_core(
            mode="prefill",
            enable_load_scaling=True,
            enable_throughput_scaling=True,
            load_adjustment_interval=5,
            throughput_adjustment_interval=60,
        )
        tick = core.initial_tick(start_s=0.0)
        # First tick is a pure load tick (5s < 60s); traffic scrape is reserved
        # for the throughput tick when both modes are enabled.
        assert tick.run_load_scaling
        assert not tick.run_throughput_scaling
        assert not tick.need_traffic_metrics

    def test_load_only_load_tick_consumes_traffic(self):
        core = _make_core(
            mode="prefill",
            enable_load_scaling=True,
            enable_throughput_scaling=False,
        )
        tick_input = TickInput(
            now_s=5.0,
            traffic=TrafficObservation(
                duration_s=5, num_req=0, isl=0, osl=0, kv_hit_rate=0.7
            ),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        core.on_tick(_tick_for(tick_input), tick_input)
        assert core._last_kv_hit_rate == 0.7

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


def _agg_caps_with_max_kv(max_kv_tokens: int) -> WorkerCapabilities:
    """Agg capabilities advertising max_num_batched_tokens AND max_kv_tokens."""
    return WorkerCapabilities(
        decode=EngineCapabilities(
            num_gpu=1,
            max_num_batched_tokens=2048,
            max_kv_tokens=max_kv_tokens,
        ),
    )


class TestAggConsolidationAwareScaleDown:
    """Agg dispatcher must respect consolidation refusal from either sub-decision.

    Regression test for the dispatcher bug where ``_agg_prefill_scaling``
    returning ``None`` because of an active safety refusal was conflated with
    "no prefill signal", letting ``_advance_load_agg`` fall through to
    decode-only scale-down via its line-327 fallback. Post-fix, the sub-
    decisions return ``num_workers`` on refusal so the dispatcher distinguishes
    "stay at current count" from "no signal at all".
    """

    def _train_agg_high_decode_kv_cost(self, core: PlannerStateMachine) -> None:
        """Regression with a strong decode_kv coefficient.

        ``T_own`` of a zero-queue prefill is ``a*avg_isl + b*decode_kv + c``.
        With ``b=1e-5`` and ``decode_kv=30K`` we get ~0.31s; doubling
        ``decode_kv`` to 60K (post-consolidation) pushes ``T_own`` to ~0.62s,
        well past a 500ms TTFT SLA.
        """
        fpms = [
            _make_fpm(
                sum_prefill_tokens=p,
                num_prefill_requests=1,
                sum_decode_kv_tokens=d,
                num_decode_requests=10,
                wall_time=1e-4 * p + 1e-5 * d + 1e-3,
            )
            for p, d in [
                (100, 5000),
                (200, 15000),
                (300, 25000),
                (400, 35000),
                (500, 45000),
            ]
        ]
        core.load_benchmark_fpms(agg_fpms=fpms)

    def _tick(
        self,
        *,
        num_workers: int,
        sched_decode_kv_per_worker: int,
        queued_decode_kv_per_worker: int = 0,
    ) -> TickInput:
        decode = {}
        for i in range(num_workers):
            decode[(f"w{i}", 0)] = _make_fpm(
                worker_id=f"w{i}",
                sum_prefill_tokens=200,
                num_prefill_requests=1,
                sum_decode_kv_tokens=sched_decode_kv_per_worker,
                num_decode_requests=10,
                queued_prefill_tokens=0,
                queued_decode_kv_tokens=queued_decode_kv_per_worker,
                # Match the regression so per-tick refit stays monotone.
                wall_time=1e-4 * 200 + 1e-5 * sched_decode_kv_per_worker + 1e-3,
            )
        return TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(decode=decode),
            worker_counts=WorkerCounts(ready_num_decode=num_workers),
        )

    def test_prefill_refusal_blocks_decode_only_scale_down(self):
        """Agg dispatcher: prefill safety refusal is NOT overridden.

        Setup at N=2:
          - decode util = 30K / 100K = 0.3, post-consolidation 0.6 < 0.8
            (sensitivity) -> agg-decode would allow scale-down.
          - prefill T_own_post at decode_kv=60K is ~620ms > 500ms SLA
            -> agg-prefill refuses scale-down (queue_budget <= 0).

        Pre-B1-fix the dispatcher would have dropped the prefill veto and
        scaled down 2 -> 1 anyway. Post-fix we stay at 2.
        """
        core = _make_agg_core(
            ttft=500.0,
            itl=1000.0,
            load_scaling_down_sensitivity=80,
        )
        core._capabilities = _agg_caps_with_max_kv(100_000)
        self._train_agg_high_decode_kv_cost(core)

        tick = self._tick(num_workers=2, sched_decode_kv_per_worker=30_000)
        effects = core.on_tick(_tick_for(tick), tick)

        # Must NOT scale down to 1.
        assert (
            effects.scale_to is None
            or effects.scale_to.num_decode is None
            or effects.scale_to.num_decode >= 2
        )
        # Operator-facing reason should distinguish the safety veto from
        # generic "no_change".
        assert (
            effects.diagnostics.load_decision_reason
            == "scale_down_refused_consolidation"
        )

    def test_queued_decode_kv_included_in_consolidation(self):
        """Queued decode KV must be added to the post-consolidation input.

        Scenario: low scheduled decode kv per worker but a sizable queued
        decode backlog. Without summing the queue, the post-consolidation
        ``current_decode_kv`` underestimates the survivor's decode pressure
        and the prefill TTFT prediction lets scale-down through.

        Setup at N=2, ttft=500ms, regression slope 1e-5 on decode_kv:
          - sched_decode_kv = 5K per worker, queued_decode_kv = 25K per worker
          - Combined = 30K; post-consolidation combined = 60K.
          - Sched-only (buggy) ``T_own_post`` at decode_kv=10K is ~121ms,
            queue_budget = (500-121)*0.8 = 303ms -> would ALLOW.
          - Combined (fixed) ``T_own_post`` at decode_kv=60K is ~621ms >
            500ms -> queue_budget <= 0 -> REFUSES.
        """
        core = _make_agg_core(
            ttft=500.0,
            itl=1000.0,
            load_scaling_down_sensitivity=80,
        )
        core._capabilities = _agg_caps_with_max_kv(100_000)
        self._train_agg_high_decode_kv_cost(core)

        tick = self._tick(
            num_workers=2,
            sched_decode_kv_per_worker=5_000,
            queued_decode_kv_per_worker=25_000,
        )
        effects = core.on_tick(_tick_for(tick), tick)

        # Without the fix, agg-prefill would see post_decode_kv=10K and
        # let scale-down through. With the fix, post_decode_kv=60K refuses.
        assert (
            effects.scale_to is None
            or effects.scale_to.num_decode is None
            or effects.scale_to.num_decode >= 2
        )
        assert (
            effects.diagnostics.load_decision_reason
            == "scale_down_refused_consolidation"
        )

    def test_both_sides_safe_permits_scale_down(self):
        """Sanity: when neither side refuses, agg DOES scale down."""
        core = _make_agg_core(
            ttft=2000.0,  # generous TTFT so prefill never refuses
            itl=1000.0,
            load_scaling_down_sensitivity=80,
        )
        core._capabilities = _agg_caps_with_max_kv(100_000)
        self._train_agg_high_decode_kv_cost(core)

        # Light load: decode util 0.1 per worker -> post 0.2 < 0.8 (allow)
        # Prefill T_own_post at decode_kv=20K ~ 0.221s < 2000ms (allow)
        tick = self._tick(num_workers=2, sched_decode_kv_per_worker=10_000)
        effects = core.on_tick(_tick_for(tick), tick)

        assert effects.scale_to is not None
        assert effects.scale_to.num_decode == 1


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
