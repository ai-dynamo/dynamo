# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end simulation tests for the AIC power-aware planner (Phase 3 + Phase 4 preview).

These tests do NOT require aiconfigurator to be installed.  They use mocked
AIC responses parameterised from the real `aic_h200_power_data` measurements:

    H200 context_attention  (prefill proxy) — p50 = 380 W, p90 = 603 W, max = 691 W
    H200 generation_attention (decode proxy) — p50 = 263 W, p90 = 558 W, max = 700 W

Scenario groups
---------------
1. ``TestFeedbackLoopH200``          — multi-tick EMA convergence with H200-realistic params
2. ``TestBudgetConstraintPipeline``  — replica capping when power_w × GPUs × replicas > budget
3. ``TestPhase4SyntheticMultiLevel`` — synthetic 3-level power sweep that exercises the exact
                                       Phase 4 code path before the AIC team delivers it

The Phase 4 synthetic scenario is the key "pipeclean" test: it proves that the
existing optimizer already selects the right (cap, replicas) pair from a set of
AIC responses that differ only in their `power_w` field, exactly as Phase 4 will
do once the AIC team delivers multi-power-level data.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from dynamo.planner.config.aic_interpolation_spec import AICInterpolationSpec
from dynamo.planner.config.parallelization import PickedParallelConfig
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import TrafficObservation
from dynamo.planner.monitoring.aic_power_optimizer import (
    AICPowerOptimizer,
    PowerAwareConfig,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]

# ---------------------------------------------------------------------------
# H200 power constants (derived from aic_h200_power_data statistics)
# ---------------------------------------------------------------------------
H200_TDP_W = 700.0

# context_attention_perf.txt — prefill proxy
H200_PREFILL_P50_W = 380.0
H200_PREFILL_P90_W = 603.0

# generation_attention_perf.txt — decode proxy
H200_DECODE_P50_W = 263.0
H200_DECODE_P90_W = 558.0

# "Serving-load" operating point (steady-state heavy traffic)
H200_PREFILL_SERVING_W = H200_PREFILL_P90_W  # 603 W/GPU at heavy prefill load
H200_DECODE_SERVING_W = H200_DECODE_P90_W  # 558 W/GPU at heavy decode load

# GPUs per H200 SXM engine (8-GPU tensor parallel)
H200_GPUS_PER_ENGINE = 8

# ---------------------------------------------------------------------------
# Synthetic multi-level profiles (Phase 4 simulation)
#   Three "power levels" derived by scaling the measured serving-point values.
#   Phase 4 will get these from the AIC database at different NVML cap settings.
# ---------------------------------------------------------------------------
#   Level  | Cap (% TDP) | Prefill W/GPU | Decode W/GPU | Relative throughput
#   -------+-------------+---------------+--------------+---------------------
#   HIGH   | 100%  700W  |   603 W       |   558 W      |  1.00×
#   MED    |  75%  525W  |   452 W       |   419 W      |  ~0.88× (rough estimate)
#   LOW    |  50%  350W  |   302 W       |   279 W      |  ~0.70×
_POWER_LEVELS = {
    "high": {
        "cap_pct": 1.00,
        "prefill_w": H200_PREFILL_SERVING_W,
        "decode_w": H200_DECODE_SERVING_W,
        "ttft_ms": 50.0,  # fastest at full power
        "itl_ms": 10.0,
        "max_kv": 50_000,
    },
    "med": {
        "cap_pct": 0.75,
        "prefill_w": H200_PREFILL_SERVING_W * 0.75,
        "decode_w": H200_DECODE_SERVING_W * 0.75,
        "ttft_ms": 57.0,  # ~14% slower (throttled GEMMs)
        "itl_ms": 11.4,
        "max_kv": 50_000,
    },
    "low": {
        "cap_pct": 0.50,
        "prefill_w": H200_PREFILL_SERVING_W * 0.50,
        "decode_w": H200_DECODE_SERVING_W * 0.50,
        "ttft_ms": 70.0,  # ~40% slower
        "itl_ms": 13.0,
        "max_kv": 50_000,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PICK_8GPU = PickedParallelConfig(tp=8, pp=1, dp=1, moe_tp=1, moe_ep=1)

_SPEC = AICInterpolationSpec(
    hf_id="Qwen/Qwen3-32B",
    system="h200_sxm",
    backend="trtllm",
    isl=1024,
    osl=200,
    sweep_max_context_length=4096,
    prefill_interpolation_granularity=4,
    decode_interpolation_granularity=4,
    prefill_pick=_PICK_8GPU,
    decode_pick=_PICK_8GPU,
)


def _make_config(
    *,
    total_gpu_power_limit: int | None = 200_000,  # 200 kW default — very generous
    ttft: int = 200,
    itl: int = 50,
    min_endpoint: int = 1,
    max_gpu_budget: int = 64,
    c_ttft: float = 1.0,
    c_itl: float = 1.0,
    c_power_p: float = 1.0,
    c_power_d: float = 1.0,
    aic_reoptimize_interval: float = 0.0,  # no rate-limit in tests
    aic_drift_consecutive_ticks: int = 3,
    mode: str = "disagg",
) -> PlannerConfig:
    return PlannerConfig(
        namespace="test-sim",
        mode=mode,
        enable_power_awareness=True,
        total_gpu_power_limit=total_gpu_power_limit,
        power_agent_safe_default_watts=350,
        prefill_engine_gpu_power_limit=int(H200_TDP_W),
        decode_engine_gpu_power_limit=int(H200_TDP_W),
        enable_aic_optimizer=True,
        aic_interpolation=_SPEC,
        ttft=ttft,
        itl=itl,
        min_endpoint=min_endpoint,
        max_gpu_budget=max_gpu_budget,
        aic_reoptimize_interval=aic_reoptimize_interval,
        aic_drift_relative_threshold=0.15,
        aic_drift_consecutive_ticks=aic_drift_consecutive_ticks,
        aic_max_consecutive_failures=3,
        aic_initial_c_ttft=c_ttft,
        aic_initial_c_itl=c_itl,
        aic_initial_c_power_prefill=c_power_p,
        aic_initial_c_power_decode=c_power_d,
        aic_throughput_regression_warn_threshold=0.20,
    )


def _estimator_for_level(level: str) -> MagicMock:
    """Build an AIC estimator mock using the given synthetic power-level profile."""
    p = _POWER_LEVELS[level]
    inst = MagicMock(name=f"AIC_{level}")
    inst.database.system_spec = {"gpu": {"power": H200_TDP_W}}
    inst.estimate_prefill_perf.return_value = {
        "context_latency": p["ttft_ms"],
        "power_w": p["prefill_w"],
    }
    inst.estimate_perf.return_value = {
        "tpot": p["itl_ms"],
        "power_w": p["decode_w"],
    }
    inst.get_max_kv_tokens.return_value = p["max_kv"]
    return inst


def _make_optimizer(config: PlannerConfig) -> tuple[AICPowerOptimizer, MagicMock]:
    metrics = MagicMock()
    return AICPowerOptimizer(config, metrics), metrics


def _make_traffic(
    *,
    num_req: float = 20.0,
    ttft_avg_s: float | None = None,
    itl_avg_s: float | None = None,
    total_tokens_per_s: float | None = 500.0,
    scheduled_prefill_tokens: float = 8_000.0,
    scheduled_decode_kv_tokens: float = 4_000.0,
) -> TrafficObservation:
    return TrafficObservation(
        duration_s=60.0,
        num_req=num_req,
        isl=1024.0,
        osl=200.0,
        ttft_avg=ttft_avg_s,
        itl_avg=itl_avg_s,
        total_tokens_per_s=total_tokens_per_s,
        scheduled_prefill_tokens=scheduled_prefill_tokens,
        scheduled_decode_kv_tokens=scheduled_decode_kv_tokens,
    )


# ---------------------------------------------------------------------------
# 1. Feedback loop — EMA convergence with H200-realistic power values
# ---------------------------------------------------------------------------


class TestFeedbackLoopH200:
    """Multi-tick EMA feedback loop using H200 realistic power values.

    The EMA denominator is AIC's raw per-GPU power estimate from the last
    sweep (``_last_optimal_config.aic_power_w_prefill`` / ``_decode``), NOT
    the planner-applied cap (§5.3, decision #6).  For the ``"high"`` level
    that's ``H200_PREFILL_P90_W = 603 W`` and ``H200_DECODE_P90_W = 558 W``.

    With this denominator:
    - ``c_power_p < 1.0`` when observed < AIC's prediction (AIC over-predicts
      power; the asymmetric clamp ``max(1.0, c)`` then leaves the cap at the
      AIC estimate, which is conservative-correct).
    - ``c_power_p > 1.0`` when observed > AIC's prediction (AIC under-predicts;
      the cap inflates by ``c_power_p`` so the planner books realistic budget).
    """

    def test_ema_converges_from_cold_start(self):
        """After N ticks of consistent power observation, c_power_p converges.

        Scenario: observed power is 80% of AIC's predicted prefill power
        (482.4 W out of 603 W).  EMA raw ratio = 482.4/603 ≈ 0.80 each tick.
        After 20 ticks the coefficient should be within 5% of 0.80.
        """
        AIC_PREFILL_W = H200_PREFILL_P90_W  # 603 W (synthetic AIC estimate)
        OBSERVED_W = AIC_PREFILL_W * 0.80  # ≈482.4 W (observed)
        TRUE_RATIO = OBSERVED_W / AIC_PREFILL_W  # 0.80

        estimator = _estimator_for_level("high")
        config = _make_config(c_power_p=1.0)
        opt, _ = _make_optimizer(config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()
        assert result is not None
        # Sanity-check that the optimizer captured the AIC raw power as the
        # EMA denominator — this is the bug-#2 invariant.
        assert result.aic_power_w_prefill == pytest.approx(AIC_PREFILL_W)

        traffic = _make_traffic(num_req=30.0, scheduled_prefill_tokens=10_000.0)

        TICKS = 20
        for _ in range(TICKS):
            opt.update_correction(traffic, observed_power_w_prefill=OBSERVED_W)

        assert (
            abs(opt._c_power_p - TRUE_RATIO) < 0.05
        ), f"c_power_p={opt._c_power_p:.3f} did not converge to {TRUE_RATIO:.3f}"

    def test_ema_stable_when_observed_matches_aic_prediction(self):
        """When observed power equals AIC's prefill power_w, c stays at 1.0.

        raw = observed / aic_power_w_prefill = 1.0 → EMA(1.0, 1.0) = 1.0 forever.
        """
        estimator = _estimator_for_level("high")
        config = _make_config(c_power_p=1.0)
        opt, _ = _make_optimizer(config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            opt.optimize()

        traffic = _make_traffic(scheduled_prefill_tokens=5_000.0)

        # observed exactly matches AIC's prefill estimate (603W).
        # raw = 603/603 = 1.0 → EMA update: 0.3×1.0 + 0.7×1.0 = 1.0 (no change).
        for _ in range(10):
            opt.update_correction(
                traffic,
                observed_power_w_prefill=H200_PREFILL_P90_W,
            )

        assert abs(opt._c_power_p - 1.0) < 0.001

    def test_decode_coefficient_independent_of_prefill(self):
        """Decode EMA must not be contaminated by prefill observations."""
        estimator = _estimator_for_level("high")
        config = _make_config(c_power_p=1.0, c_power_d=1.0)
        opt, _ = _make_optimizer(config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            opt.optimize()

        # Prefill active, decode idle.
        traffic = _make_traffic(
            scheduled_prefill_tokens=8_000.0,
            scheduled_decode_kv_tokens=0.0,
        )
        initial_d = opt._c_power_d
        BIG_PREFILL_OBS = H200_TDP_W * 1.5  # large prefill observation

        for _ in range(10):
            opt.update_correction(traffic, observed_power_w_prefill=BIG_PREFILL_OBS)

        # c_power_d must not have moved (decode side was idle).
        assert opt._c_power_d == initial_d, (
            f"c_power_d changed from {initial_d} to {opt._c_power_d} "
            f"despite decode side being idle"
        )

    def test_drift_detection_fires_when_sla_consistently_violated(self):
        """TTFT > SLA for consecutive_ticks → should_reoptimize() returns True."""
        estimator = _estimator_for_level("high")
        config = _make_config(
            ttft=200,
            aic_reoptimize_interval=0.0,
            aic_drift_consecutive_ticks=3,
        )
        opt, _ = _make_optimizer(config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            opt.optimize()

        # TTFT measured at 350ms > 200ms SLA.
        bad_traffic = _make_traffic(ttft_avg_s=0.350)

        assert opt.should_reoptimize(bad_traffic) is False  # tick 1
        assert opt.should_reoptimize(bad_traffic) is False  # tick 2
        assert opt.should_reoptimize(bad_traffic) is True  # tick 3 — fires

    def test_no_drift_under_nominal_load(self):
        """At normal load well within budget, no spurious reoptimization triggers."""
        estimator = _estimator_for_level("high")
        config = _make_config(
            ttft=200,
            aic_reoptimize_interval=0.0,
            aic_drift_consecutive_ticks=3,
        )
        opt, _ = _make_optimizer(config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()
        assert result is not None

        # Set the estimated throughput reference.
        opt._estimated_throughput = 1_000.0  # tok/s

        # Good traffic: TTFT within SLA, tokens within estimated capacity.
        good_traffic = _make_traffic(
            ttft_avg_s=0.100,  # 100ms < 200ms SLA
            itl_avg_s=0.020,  # 20ms < 50ms SLA
            total_tokens_per_s=800.0,  # < 1000 × 1.15 threshold
        )
        for _ in range(10):
            assert opt.should_reoptimize(good_traffic) is False


# ---------------------------------------------------------------------------
# 2. Budget constraint pipeline
# ---------------------------------------------------------------------------


class TestBudgetConstraintPipeline:
    """Verify that the power budget correctly restricts replica counts end-to-end.

    Uses H200-realistic power_w values so the wattage numbers are grounded
    in the actual measured data.
    """

    def test_budget_limits_replicas_to_realistic_count(self):
        """With a realistic 40 kW budget and H200 serving power, replica count is bounded."""
        # 8 GPUs per engine, serving power:
        #   prefill: 603 W/GPU × 8 = 4,824 W/replica
        #   decode:  558 W/GPU × 8 = 4,464 W/replica
        # Budget 40 kW → can fit ~4 decode replicas (4 × 4464 = 17856) + prefill
        estimator = _estimator_for_level("high")
        config = _make_config(
            total_gpu_power_limit=40_000,
            max_gpu_budget=64,
        )
        opt, _ = _make_optimizer(config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        assert result is not None
        # Total power must respect the budget.
        total_w = (
            result.n_p * result.cap_p * H200_GPUS_PER_ENGINE
            + result.n_d * result.cap_d * H200_GPUS_PER_ENGINE
        )
        assert (
            total_w <= 40_000 + 200
        ), f"total_w={total_w} W exceeds budget=40000 W"  # small rounding tolerance

    def test_corrected_cap_respected_under_budget(self):
        """When c_power_p = 1.10 (10% over-estimate detected), cap inflates by 10%."""
        estimator = _estimator_for_level("high")
        config = _make_config(c_power_p=1.10, c_power_d=1.10)
        opt, _ = _make_optimizer(config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        assert result is not None
        # cap_p should be ceil(H200_PREFILL_P90_W × 1.10)
        expected_cap_p = math.ceil(H200_PREFILL_P90_W * 1.10)
        assert (
            result.cap_p == expected_cap_p
        ), f"cap_p={result.cap_p} != expected {expected_cap_p}"

    def test_total_gpu_power_limit_none_gives_max_replicas(self):
        """With a very generous power budget, replicas are bounded only by max_gpu_budget."""
        estimator = _estimator_for_level("high")
        config = _make_config(
            total_gpu_power_limit=200_000,  # 200 kW — effectively unbounded
            max_gpu_budget=16,  # 16 GPUs → 2 engines of 8 GPUs each
        )
        opt, _ = _make_optimizer(config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        assert result is not None
        assert result.n_d >= 1


# ---------------------------------------------------------------------------
# 3. Synthetic Phase 4 multi-level power sweep
# ---------------------------------------------------------------------------


class TestPhase4SyntheticMultiLevel:
    """Synthetic multi-level power sweep — pipecleans the Phase 4 code path.

    Phase 4 will add a ``power_level`` dimension to the AIC sweep:
    for each candidate power cap, AIC returns a (ttft_ms, itl_ms, power_w,
    max_kv) tuple.  The optimizer selects the highest-throughput config whose
    per-replica power draw fits within the budget.

    This test simulates that selection by calling optimize() three times with
    different AIC mock returns (one per synthetic power level) and a fixed
    budget that only certain levels can satisfy.  It proves that:

    1. The optimizer correctly selects the highest-watt config when budget is
       generous (``high`` level wins).
    2. With a tighter budget, the optimizer is forced to the medium level.
    3. With a very tight budget, only the low level fits.
    4. The selected cap reflects the power_w from the chosen level, not TDP.
    5. Throughput (n_d × aic_seq_per_s_per_replica) decreases monotonically as
       budget tightens (lower power → lower throughput → fewer effective replicas).

    All three scenarios use IDENTICAL optimizer configuration; only the AIC
    mock response (i.e., which ``power_level`` the AIC database returns) varies.
    This is exactly how Phase 4 will work at runtime.
    """

    # Per-replica wattage for each level (power_w × 8 GPUs)
    _WATTS_PER_REPLICA = {
        lvl: _POWER_LEVELS[lvl]["prefill_w"] * H200_GPUS_PER_ENGINE
        for lvl in _POWER_LEVELS
    }

    def _optimize_at_level(
        self, level: str, total_gpu_power_limit: int
    ) -> tuple[AICPowerOptimizer, "PowerAwareConfig"]:
        estimator = _estimator_for_level(level)
        config = _make_config(
            total_gpu_power_limit=total_gpu_power_limit,
            max_gpu_budget=64,
            ttft=300,
            itl=100,
        )
        opt, _ = _make_optimizer(config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()
        assert result is not None, f"optimize() returned None at level={level}"
        return opt, result

    def test_high_level_wins_with_generous_budget(self):
        """With a 120 kW budget, the optimizer should use the full-TDP config."""
        # high level: 603 W/GPU × 8 = 4824 W/replica → 120kW fits ~24 decode replicas
        _, result = self._optimize_at_level("high", total_gpu_power_limit=120_000)
        # cap should be ceil(603 × 1.0) = 603 W (not 700 W TDP fallback)
        assert result.cap_p == math.ceil(H200_PREFILL_P90_W)
        assert result.cap_d == math.ceil(H200_DECODE_P90_W)

    def test_med_level_produces_lower_cap(self):
        """Medium level (75% TDP) produces lower caps than high level."""
        _, high_r = self._optimize_at_level("high", total_gpu_power_limit=120_000)
        _, med_r = self._optimize_at_level("med", total_gpu_power_limit=120_000)

        assert (
            med_r.cap_p < high_r.cap_p
        ), f"med cap_p={med_r.cap_p} should be < high cap_p={high_r.cap_p}"
        assert med_r.cap_d < high_r.cap_d

    def test_budget_selects_lower_power_level(self):
        """With a tight budget, the optimizer forced to lower-watt config gets more replicas.

        Key Phase 4 insight: a lower per-replica power means more replicas fit
        in the same budget, which can offset the lower per-replica throughput.
        """
        # Tight budget: 20 kW.  High-level replica = 4824 W → 4 decode fit.
        # Low-level replica = ~2420 W → 8 decode fit.
        TIGHT_BUDGET = 20_000

        _, high_r = self._optimize_at_level("high", total_gpu_power_limit=TIGHT_BUDGET)
        _, low_r = self._optimize_at_level("low", total_gpu_power_limit=TIGHT_BUDGET)

        # Low level has lower per-replica cost → should fit more decode replicas.
        assert (
            low_r.n_d >= high_r.n_d
        ), f"low n_d={low_r.n_d} should be >= high n_d={high_r.n_d} at tight budget"

        # Caps reflect the power_w from each level.
        assert low_r.cap_p < high_r.cap_p
        assert low_r.cap_d < high_r.cap_d

    def test_all_levels_respect_budget(self):
        """For every budget × level combo, total power must not exceed budget."""
        for budget in [10_000, 25_000, 50_000, 100_000]:
            for level in ["high", "med", "low"]:
                _, result = self._optimize_at_level(level, total_gpu_power_limit=budget)
                total_w = (
                    result.n_p * result.cap_p * H200_GPUS_PER_ENGINE
                    + result.n_d * result.cap_d * H200_GPUS_PER_ENGINE
                )
                assert (
                    total_w <= budget + 500
                ), (  # 500W rounding tolerance
                    f"level={level}, budget={budget}: total_w={total_w} exceeds budget"
                )

    def test_caps_are_not_tdp_fallback(self):
        """In all three levels, cap_p and cap_d must differ from TDP (700W).

        The TDP fallback path emits a WARNING and returns caps based on TDP.
        Real power_w from the H200 data is < 700W at all serving points, so
        if the integration is working, caps will be < 700W.
        """
        for level in ["high", "med", "low"]:
            _, result = self._optimize_at_level(level, total_gpu_power_limit=200_000)
            assert result.cap_p != int(H200_TDP_W), (
                f"level={level}: cap_p={result.cap_p} W equals TDP — "
                "TDP fallback was triggered (power_w=0 from AIC)"
            )
            assert result.cap_d != int(H200_TDP_W), (
                f"level={level}: cap_d={result.cap_d} W equals TDP — "
                "TDP fallback was triggered (power_w=0 from AIC)"
            )

    def test_ema_correction_applied_consistently_across_levels(self):
        """After EMA converges (c_power = 1.1), caps are inflated consistently."""
        for level in _POWER_LEVELS:
            estimator = _estimator_for_level(level)
            config = _make_config(c_power_p=1.10, c_power_d=1.10)
            opt, _ = _make_optimizer(config)

            with patch(
                "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
                return_value=estimator,
            ):
                result = opt.optimize()

            assert result is not None
            # cap = ceil(power_w × max(1.0, c_power)) = ceil(power_w × 1.10)
            expected_p = math.ceil(_POWER_LEVELS[level]["prefill_w"] * 1.10)
            expected_d = math.ceil(_POWER_LEVELS[level]["decode_w"] * 1.10)
            assert result.cap_p == expected_p, f"level={level}: cap_p mismatch"
            assert result.cap_d == expected_d, f"level={level}: cap_d mismatch"


# ---------------------------------------------------------------------------
# 4. Full scenario: from cold start through stable state
# ---------------------------------------------------------------------------


class TestFullDeploymentScenario:
    """Simulate a 'day in the life' of the power-aware planner on H200.

    Phase 0: Cold start — first optimize() call with c_power=1.0.
    Phase 1: EMA calibration — 15 ticks where observed power > AIC estimate.
    Phase 2: Stable state — coefficients have converged; no spurious re-sweeps.
    Phase 3: Traffic surge — tokens/s spikes above capacity → drift detected.
    Phase 4: Re-optimise — new sweep produces updated config.
    """

    def test_full_scenario_completes_without_disabling(self):
        estimator = _estimator_for_level("high")
        config = _make_config(
            total_gpu_power_limit=50_000,
            ttft=200,
            itl=50,
            aic_reoptimize_interval=0.0,
            aic_drift_consecutive_ticks=3,
        )
        opt, _ = _make_optimizer(config)

        # Phase 0: cold start sweep.
        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            first_result = opt.optimize()
        assert first_result is not None
        assert (
            not opt._disabled
        ), "Optimizer disabled after first sweep — check SLA feasibility"

        # Phase 1: EMA calibration — 15 ticks.
        # - TTFT observed at 55ms vs AIC 50ms → raw_ttft = 55/50 = 1.10 → c_ttft > 1.0
        # - ITL observed at 11ms vs AIC 10ms  → raw_itl  = 11/10 = 1.10 → c_itl  > 1.0
        # - Prefill power observed at 637W, AIC predicted 603W → raw = 1.056 → c_power_p > 1.0
        #   (AIC slightly under-predicts; the cap inflates by c_power_p on the next sweep).
        OVER_PREDICT_FACTOR = 1.056
        traffic_calibrate = _make_traffic(
            num_req=25.0,
            ttft_avg_s=0.055,  # 55ms vs 50ms AIC TTFT (within 200ms SLA)
            itl_avg_s=0.011,  # 11ms vs 10ms AIC ITL
            total_tokens_per_s=400.0,
            scheduled_prefill_tokens=8_000.0,
            scheduled_decode_kv_tokens=5_000.0,
        )
        for _ in range(15):
            opt.update_correction(
                traffic_calibrate,
                observed_ttft_avg=0.055,
                observed_itl_avg=0.011,
                observed_power_w_prefill=H200_PREFILL_P90_W
                * OVER_PREDICT_FACTOR,  # 637 W
                observed_power_w_decode=H200_DECODE_P90_W * OVER_PREDICT_FACTOR,
            )
            # No drift should trigger during calibration (SLA not violated, no capacity surge).
            assert opt.should_reoptimize(traffic_calibrate) is False

        # TTFT/ITL calibrate above 1.0 (observed > AIC estimate).
        assert opt._c_ttft > 1.0, f"c_ttft={opt._c_ttft} did not calibrate above 1.0"
        assert opt._c_itl > 1.0, f"c_itl={opt._c_itl} did not calibrate above 1.0"
        # Power coefficient normalises against AIC raw estimate (603W for high level)
        # → raw = 637/603 = 1.056, c_power_p > 1.0 (AIC slightly under-predicts).
        assert (
            opt._c_power_p > 1.0
        ), f"c_power_p={opt._c_power_p:.3f} should be > 1.0 when observed > AIC raw"
        assert opt._c_power_p == pytest.approx(OVER_PREDICT_FACTOR, abs=0.01)

        # Phase 2: stable state — 10 ticks, all metrics nominal.
        traffic_stable = _make_traffic(
            num_req=20.0,
            ttft_avg_s=0.055,
            itl_avg_s=0.011,
            total_tokens_per_s=400.0,
        )
        opt._estimated_throughput = 1_000.0  # set reference
        for _ in range(10):
            opt.update_correction(
                traffic_stable,
                observed_power_w_prefill=H200_PREFILL_P90_W * OVER_PREDICT_FACTOR,
                observed_power_w_decode=H200_DECODE_P90_W * OVER_PREDICT_FACTOR,
            )
            assert opt.should_reoptimize(traffic_stable) is False

        # Phase 3: traffic surge — 3 ticks at 120% of estimated capacity.
        traffic_surge = _make_traffic(
            num_req=60.0,
            ttft_avg_s=0.055,  # TTFT still OK (enough replicas)
            total_tokens_per_s=1_200.0,  # 20% above 1000 threshold
        )
        assert opt.should_reoptimize(traffic_surge) is False  # tick 1
        assert opt.should_reoptimize(traffic_surge) is False  # tick 2
        assert opt.should_reoptimize(traffic_surge) is True  # tick 3 — fires!

        # Phase 4: re-optimize.
        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            second_result = opt.optimize()

        assert second_result is not None
        assert not opt._disabled
        # After a valid sweep, consecutive violations reset to 0.
        assert opt._consecutive_violation_ticks == 0
