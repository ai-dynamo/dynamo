# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for easy-mode scaling (optimization_target = throughput | latency)."""

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
    WorkerCapabilities,
    WorkerCounts,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _tick_for(tick_input: TickInput) -> ScheduledTick:
    has_fpm = tick_input.fpm_observations is not None
    return ScheduledTick(
        at_s=tick_input.now_s,
        run_load_scaling=has_fpm,
        run_throughput_scaling=False,
        need_worker_states=True,
        need_worker_fpm=has_fpm,
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


def _easy_config(**overrides) -> PlannerConfig:
    defaults = dict(
        mode="disagg",
        optimization_target="throughput",
        enable_load_scaling=True,
        enable_throughput_scaling=False,
        min_endpoint=1,
        max_gpu_budget=-1,
        load_adjustment_interval=5,
        max_num_fpm_samples=50,
        fpm_sample_bucket_size=16,
        load_min_observations=5,
        load_predictor="constant",
        backend="vllm",
        metric_pulling_prometheus_endpoint="http://localhost:9090",
        metric_reporting_prometheus_port=0,
    )
    defaults.update(overrides)
    return PlannerConfig.model_construct(**defaults)


CONTEXT_LENGTH = 4096
MAX_KV_TOKENS = 100000


def _prefill_caps() -> WorkerCapabilities:
    return WorkerCapabilities(
        prefill=EngineCapabilities(
            num_gpu=1, context_length=CONTEXT_LENGTH, max_num_batched_tokens=2048
        ),
        decode=EngineCapabilities(
            num_gpu=1, max_kv_tokens=MAX_KV_TOKENS, max_num_batched_tokens=2048
        ),
    )


def _decode_caps() -> WorkerCapabilities:
    return WorkerCapabilities(
        decode=EngineCapabilities(
            num_gpu=1, max_kv_tokens=MAX_KV_TOKENS, max_num_batched_tokens=2048
        ),
    )


def _agg_caps() -> WorkerCapabilities:
    return WorkerCapabilities(
        decode=EngineCapabilities(
            num_gpu=1,
            max_kv_tokens=MAX_KV_TOKENS,
            max_num_batched_tokens=2048,
            context_length=CONTEXT_LENGTH,
        ),
    )


def _make_core(config=None, caps=None, **overrides) -> PlannerStateMachine:
    cfg = config or _easy_config(**overrides)
    return PlannerStateMachine(cfg, caps or _prefill_caps())


# ── Config validation ────────────────────────────────────────────────


class TestEasyConfig:
    def test_throughput_forces_load_on_throughput_off(self):
        cfg = PlannerConfig.model_validate(dict(optimization_target="throughput"))
        assert cfg.enable_load_scaling is True
        assert cfg.enable_throughput_scaling is False

    def test_latency_forces_load_on_throughput_off(self):
        cfg = PlannerConfig.model_validate(dict(optimization_target="latency"))
        assert cfg.enable_load_scaling is True
        assert cfg.enable_throughput_scaling is False

    def test_sla_mode_preserves_original_flags(self):
        cfg = PlannerConfig.model_validate(
            dict(
                optimization_target="sla",
                enable_load_scaling=True,
                enable_throughput_scaling=True,
                pre_deployment_sweeping_mode="rapid",
                throughput_adjustment_interval=60,
                load_adjustment_interval=5,
            )
        )
        assert cfg.enable_load_scaling is True
        assert cfg.enable_throughput_scaling is True

    def test_no_regression_in_easy_mode(self):
        core = _make_core(optimization_target="throughput", mode="prefill")
        assert not hasattr(core, "_prefill_regression")

    def test_no_predictors_in_easy_mode(self):
        core = _make_core(optimization_target="throughput", mode="prefill")
        assert not hasattr(core, "_num_req_predictor")


# ── Prefill throughput scaling ───────────────────────────────────────


class TestPrefillThroughputEasy:
    def test_scale_up_at_context_length(self):
        core = _make_core(mode="prefill", optimization_target="throughput")
        fpm = _make_fpm(queued_prefill_tokens=CONTEXT_LENGTH)
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(prefill={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_prefill == 2
        assert effects.diagnostics.load_decision_reason == "scale_up"

    def test_no_change_between_thresholds(self):
        core = _make_core(mode="prefill", optimization_target="throughput")
        # queued = context_length / 2 -> between 0.1 and 1.0
        fpm1 = _make_fpm(worker_id="w1", queued_prefill_tokens=CONTEXT_LENGTH // 2)
        fpm2 = _make_fpm(worker_id="w2", queued_prefill_tokens=CONTEXT_LENGTH // 2)
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                prefill={("w1", 0): fpm1, ("w2", 0): fpm2}
            ),
            worker_counts=WorkerCounts(ready_num_prefill=2),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is None
        assert effects.diagnostics.load_decision_reason == "no_change"

    def test_scale_down_below_tenth(self):
        core = _make_core(mode="prefill", optimization_target="throughput")
        # queued < context_length / 10
        fpm1 = _make_fpm(worker_id="w1", queued_prefill_tokens=CONTEXT_LENGTH // 20)
        fpm2 = _make_fpm(worker_id="w2", queued_prefill_tokens=CONTEXT_LENGTH // 20)
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                prefill={("w1", 0): fpm1, ("w2", 0): fpm2}
            ),
            worker_counts=WorkerCounts(ready_num_prefill=2),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_prefill == 1
        assert effects.diagnostics.load_decision_reason == "scale_down"

    def test_no_scale_down_at_min(self):
        core = _make_core(mode="prefill", optimization_target="throughput")
        fpm = _make_fpm(queued_prefill_tokens=0)
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(prefill={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        # Already at 1 worker, can't scale down further
        assert effects.scale_to is None


# ── Prefill latency scaling ──────────────────────────────────────────


class TestPrefillLatencyEasy:
    def test_scale_up_at_tenth(self):
        core = _make_core(mode="prefill", optimization_target="latency")
        # Use exact tenth (ceil to avoid int division rounding below threshold)
        fpm = _make_fpm(queued_prefill_tokens=CONTEXT_LENGTH // 10 + 1)
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(prefill={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_prefill == 2

    def test_scale_down_at_zero(self):
        core = _make_core(mode="prefill", optimization_target="latency")
        fpm1 = _make_fpm(worker_id="w1", queued_prefill_tokens=0)
        fpm2 = _make_fpm(worker_id="w2", queued_prefill_tokens=0)
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                prefill={("w1", 0): fpm1, ("w2", 0): fpm2}
            ),
            worker_counts=WorkerCounts(ready_num_prefill=2),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_prefill == 1

    def test_no_scale_down_with_any_queued(self):
        core = _make_core(mode="prefill", optimization_target="latency")
        fpm1 = _make_fpm(worker_id="w1", queued_prefill_tokens=10)
        fpm2 = _make_fpm(worker_id="w2", queued_prefill_tokens=10)
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                prefill={("w1", 0): fpm1, ("w2", 0): fpm2}
            ),
            worker_counts=WorkerCounts(ready_num_prefill=2),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is None


# ── Decode throughput scaling ────────────────────────────────────────


class TestDecodeThroughputEasy:
    def test_scale_up_above_100_pct(self):
        core = _make_core(
            mode="decode", optimization_target="throughput", caps=_decode_caps()
        )
        # util = (80000 + 30000) / 100000 = 1.1 > 1.0
        fpm = _make_fpm(sum_decode_kv_tokens=80000, queued_decode_kv_tokens=30000)
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(decode={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_decode=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_decode == 2
        assert effects.diagnostics.load_decision_reason == "scale_up"

    def test_scale_down_below_60_pct(self):
        core = _make_core(
            mode="decode", optimization_target="throughput", caps=_decode_caps()
        )
        # util = (40000 + 0) / 100000 = 0.4 < 0.6
        fpm1 = _make_fpm(
            worker_id="w1", sum_decode_kv_tokens=40000, queued_decode_kv_tokens=0
        )
        fpm2 = _make_fpm(
            worker_id="w2", sum_decode_kv_tokens=40000, queued_decode_kv_tokens=0
        )
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(decode={("w1", 0): fpm1, ("w2", 0): fpm2}),
            worker_counts=WorkerCounts(ready_num_decode=2),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_decode == 1

    def test_no_change_between_thresholds(self):
        core = _make_core(
            mode="decode", optimization_target="throughput", caps=_decode_caps()
        )
        # util = (70000 + 0) / 100000 = 0.7 -> between 0.6 and 1.0
        fpm1 = _make_fpm(
            worker_id="w1", sum_decode_kv_tokens=70000, queued_decode_kv_tokens=0
        )
        fpm2 = _make_fpm(
            worker_id="w2", sum_decode_kv_tokens=70000, queued_decode_kv_tokens=0
        )
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(decode={("w1", 0): fpm1, ("w2", 0): fpm2}),
            worker_counts=WorkerCounts(ready_num_decode=2),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is None


# ── Decode latency scaling ───────────────────────────────────────────


class TestDecodeLatencyEasy:
    def test_scale_up_above_40_pct(self):
        core = _make_core(
            mode="decode", optimization_target="latency", caps=_decode_caps()
        )
        # util = (45000 + 0) / 100000 = 0.45 > 0.4
        fpm = _make_fpm(sum_decode_kv_tokens=45000, queued_decode_kv_tokens=0)
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(decode={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_decode=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_decode == 2

    def test_scale_down_below_10_pct(self):
        core = _make_core(
            mode="decode", optimization_target="latency", caps=_decode_caps()
        )
        # util = (5000 + 0) / 100000 = 0.05 < 0.1
        fpm1 = _make_fpm(
            worker_id="w1", sum_decode_kv_tokens=5000, queued_decode_kv_tokens=0
        )
        fpm2 = _make_fpm(
            worker_id="w2", sum_decode_kv_tokens=5000, queued_decode_kv_tokens=0
        )
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(decode={("w1", 0): fpm1, ("w2", 0): fpm2}),
            worker_counts=WorkerCounts(ready_num_decode=2),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_decode == 1


# ── ANY-up / ALL-down logic ─────────────────────────────────────────


class TestAnyUpAllDown:
    def test_any_engine_above_triggers_scale_up(self):
        """One engine above threshold, one below -> scale up."""
        core = _make_core(
            mode="decode", optimization_target="throughput", caps=_decode_caps()
        )
        fpm_ok = _make_fpm(
            worker_id="w1", sum_decode_kv_tokens=50000, queued_decode_kv_tokens=0
        )
        fpm_hot = _make_fpm(
            worker_id="w2", sum_decode_kv_tokens=80000, queued_decode_kv_tokens=30000
        )
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                decode={("w1", 0): fpm_ok, ("w2", 0): fpm_hot}
            ),
            worker_counts=WorkerCounts(ready_num_decode=2),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_decode == 3

    def test_mixed_no_scale_down(self):
        """One engine below scale-down threshold, one above -> no change."""
        core = _make_core(
            mode="decode", optimization_target="throughput", caps=_decode_caps()
        )
        fpm_low = _make_fpm(
            worker_id="w1", sum_decode_kv_tokens=20000, queued_decode_kv_tokens=0
        )
        fpm_mid = _make_fpm(
            worker_id="w2", sum_decode_kv_tokens=70000, queued_decode_kv_tokens=0
        )
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                decode={("w1", 0): fpm_low, ("w2", 0): fpm_mid}
            ),
            worker_counts=WorkerCounts(ready_num_decode=2),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is None

    def test_all_below_triggers_scale_down(self):
        """All engines below scale-down threshold -> scale down."""
        core = _make_core(
            mode="decode", optimization_target="throughput", caps=_decode_caps()
        )
        fpm1 = _make_fpm(
            worker_id="w1", sum_decode_kv_tokens=20000, queued_decode_kv_tokens=0
        )
        fpm2 = _make_fpm(
            worker_id="w2", sum_decode_kv_tokens=30000, queued_decode_kv_tokens=0
        )
        fpm3 = _make_fpm(
            worker_id="w3", sum_decode_kv_tokens=25000, queued_decode_kv_tokens=0
        )
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                decode={("w1", 0): fpm1, ("w2", 0): fpm2, ("w3", 0): fpm3}
            ),
            worker_counts=WorkerCounts(ready_num_decode=3),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_decode == 2


# ── Disagg mode ──────────────────────────────────────────────────────


class TestDisaggEasy:
    def test_disagg_scale_up_prefill(self):
        core = _make_core(mode="disagg", optimization_target="throughput")
        p_fpm = _make_fpm(queued_prefill_tokens=CONTEXT_LENGTH)
        d_fpm = _make_fpm(sum_decode_kv_tokens=50000, queued_decode_kv_tokens=0)
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
        assert effects.scale_to.num_prefill == 2

    def test_disagg_scale_up_decode(self):
        core = _make_core(mode="disagg", optimization_target="throughput")
        p_fpm = _make_fpm(queued_prefill_tokens=0)
        d_fpm = _make_fpm(sum_decode_kv_tokens=80000, queued_decode_kv_tokens=30000)
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
        assert effects.scale_to.num_decode == 2


# ── Agg mode ─────────────────────────────────────────────────────────


class TestAggEasy:
    def test_agg_scale_up_decode_heavy(self):
        core = _make_core(
            mode="agg", optimization_target="throughput", caps=_agg_caps()
        )
        # util = (80000 + 30000 + 0) / 100000 = 1.1 > 1.0
        fpm = _make_fpm(
            sum_decode_kv_tokens=80000,
            queued_decode_kv_tokens=30000,
            queued_prefill_tokens=0,
        )
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(decode={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_decode=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_decode == 2

    def test_agg_scale_up_prefill_heavy(self):
        """Agg includes queued prefill in utilization calc."""
        core = _make_core(
            mode="agg", optimization_target="throughput", caps=_agg_caps()
        )
        # util = (20000 + 0 + 90000) / 100000 = 1.1 > 1.0
        fpm = _make_fpm(
            sum_decode_kv_tokens=20000,
            queued_decode_kv_tokens=0,
            queued_prefill_tokens=90000,
        )
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(decode={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_decode=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_decode == 2

    def test_agg_scale_down(self):
        core = _make_core(
            mode="agg", optimization_target="throughput", caps=_agg_caps()
        )
        # util = (30000 + 0 + 0) / 100000 = 0.3 < 0.6
        fpm1 = _make_fpm(
            worker_id="w1",
            sum_decode_kv_tokens=30000,
            queued_decode_kv_tokens=0,
            queued_prefill_tokens=0,
        )
        fpm2 = _make_fpm(
            worker_id="w2",
            sum_decode_kv_tokens=30000,
            queued_decode_kv_tokens=0,
            queued_prefill_tokens=0,
        )
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(decode={("w1", 0): fpm1, ("w2", 0): fpm2}),
            worker_counts=WorkerCounts(ready_num_decode=2),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is not None
        assert effects.scale_to.num_decode == 1


# ── Missing capabilities ─────────────────────────────────────────────


class TestMissingCapabilities:
    def test_missing_context_length_skips_prefill(self):
        caps = WorkerCapabilities(
            prefill=EngineCapabilities(num_gpu=1),  # no context_length
            decode=EngineCapabilities(num_gpu=1, max_kv_tokens=MAX_KV_TOKENS),
        )
        core = _make_core(mode="prefill", optimization_target="throughput", caps=caps)
        fpm = _make_fpm(queued_prefill_tokens=10000)
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(prefill={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is None
        assert effects.diagnostics.load_decision_reason == "insufficient_data"

    def test_missing_max_kv_tokens_skips_decode(self):
        caps = WorkerCapabilities(
            decode=EngineCapabilities(num_gpu=1),  # no max_kv_tokens
        )
        core = _make_core(mode="decode", optimization_target="throughput", caps=caps)
        fpm = _make_fpm(sum_decode_kv_tokens=80000, queued_decode_kv_tokens=30000)
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(decode={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_decode=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is None
        assert effects.diagnostics.load_decision_reason == "insufficient_data"


# ── Scaling in progress ──────────────────────────────────────────────


class TestScalingInProgress:
    def test_no_decision_when_scaling(self):
        core = _make_core(mode="prefill", optimization_target="throughput")
        fpm = _make_fpm(queued_prefill_tokens=CONTEXT_LENGTH * 2)
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(prefill={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_prefill=1, expected_num_prefill=2),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        assert effects.scale_to is None
        assert effects.diagnostics.load_decision_reason == "scaling_in_progress"


# ── Budget clamping ──────────────────────────────────────────────────


class TestBudgetClamping:
    def test_min_endpoint_respected(self):
        core = _make_core(
            mode="prefill", optimization_target="throughput", min_endpoint=2
        )
        fpm = _make_fpm(queued_prefill_tokens=0)
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(prefill={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_prefill=2),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        # Can't scale below min_endpoint=2
        assert effects.scale_to is None

    def test_gpu_budget_caps_scale_up(self):
        caps = WorkerCapabilities(
            prefill=EngineCapabilities(
                num_gpu=4, context_length=CONTEXT_LENGTH, max_num_batched_tokens=2048
            ),
        )
        core = _make_core(
            mode="prefill",
            optimization_target="throughput",
            max_gpu_budget=4,
            caps=caps,
        )
        fpm = _make_fpm(queued_prefill_tokens=CONTEXT_LENGTH * 2)
        tick = TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(prefill={("w1", 0): fpm}),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        )
        effects = core.on_tick(_tick_for(tick), tick)
        # Would want 2 replicas (8 GPUs) but budget is 4 -> capped at 1
        assert effects.scale_to is None or (
            effects.scale_to.num_prefill is not None
            and effects.scale_to.num_prefill * 4 <= 4
        )
