# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for AICPowerOptimizer (Phase 3).

The AIConfigurator SDK is mocked throughout so these tests run without a GPU
or an aiconfigurator installation.

Coverage:
    optimize()      — happy path, TDP fallback, measured power_w, budget
                      constraint, SLA infeasibility, startup failures, runtime
                      failures, auto-disable, throughput regression warning,
                      implied admission thresholds.
    update_correction() — EMA update, traffic gating (latency + per-side power),
                          agg vs disagg mode, clamp saturation.
    should_reoptimize() — rate limit, SLA-miss trigger, capacity-exceeded trigger
                          (upward-only), hysteresis, under-load invariance.
    _sweep_replicas() — budget-constrained, unbounded, budget-too-small fallback.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from dynamo.planner.config.aic_interpolation_spec import AICInterpolationSpec
from dynamo.planner.config.parallelization import PickedParallelConfig
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import TrafficObservation
from dynamo.planner.monitoring.aic_power_optimizer import (
    AICPowerOptimizer,
    PowerAwareConfig,
    _EMA_ALPHA,
    _COEFF_MIN,
    _COEFF_MAX,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PICK_8GPU = PickedParallelConfig(tp=8, pp=1, dp=1, moe_tp=1, moe_ep=1)

_AIC_SPEC = AICInterpolationSpec(
    hf_id="Qwen/Qwen3-32B",
    system="h200_sxm",
    backend="trtllm",
    isl=1000,
    osl=200,
    sweep_max_context_length=4096,
    prefill_interpolation_granularity=4,
    decode_interpolation_granularity=4,
    prefill_pick=_PICK_8GPU,
    decode_pick=_PICK_8GPU,
)


def _make_config(
    *,
    mode: str = "disagg",
    total_gpu_power_limit: int | None = 8_000,   # 8 kW — very generous
    prefill_limit: int = 700,
    decode_limit: int = 700,
    ttft: int = 200,
    itl: int = 50,
    min_endpoint: int = 1,
    max_gpu_budget: int = 32,
    aic_reoptimize_interval: float = 300.0,
    aic_drift_relative_threshold: float = 0.15,
    aic_drift_consecutive_ticks: int = 3,
    aic_max_consecutive_failures: int = 3,
    aic_throughput_regression_warn_threshold: float = 0.20,
    c_ttft: float = 1.0,
    c_itl: float = 1.0,
    c_power_p: float = 1.0,
    c_power_d: float = 1.0,
    c_power_agg: float = 1.0,
) -> PlannerConfig:
    return PlannerConfig(
        namespace="test-ns",
        mode=mode,
        enable_power_awareness=True,
        total_gpu_power_limit=total_gpu_power_limit,
        power_agent_safe_default_watts=300,
        prefill_engine_gpu_power_limit=prefill_limit,
        decode_engine_gpu_power_limit=decode_limit,
        enable_aic_optimizer=True,
        aic_interpolation=_AIC_SPEC,
        ttft=ttft,
        itl=itl,
        min_endpoint=min_endpoint,
        max_gpu_budget=max_gpu_budget,
        aic_reoptimize_interval=aic_reoptimize_interval,
        aic_drift_relative_threshold=aic_drift_relative_threshold,
        aic_drift_consecutive_ticks=aic_drift_consecutive_ticks,
        aic_max_consecutive_failures=aic_max_consecutive_failures,
        aic_initial_c_ttft=c_ttft,
        aic_initial_c_itl=c_itl,
        aic_initial_c_power_prefill=c_power_p,
        aic_initial_c_power_decode=c_power_d,
        aic_initial_c_power_agg=c_power_agg,
        aic_throughput_regression_warn_threshold=aic_throughput_regression_warn_threshold,
    )


def _make_metrics() -> MagicMock:
    """Return a MagicMock that satisfies the PlannerPrometheusMetrics interface."""
    m = MagicMock()
    # Prometheus Gauge-like: .set(), Counter-like: .inc()
    # MagicMock handles all of these automatically.
    return m


def _make_estimator_mock(
    *,
    ttft_ms: float = 50.0,
    itl_ms: float = 10.0,
    max_kv_tokens: int = 50_000,
    power_w_prefill: float = 400.0,
    power_w_decode: float = 300.0,
    tdp_w: float = 700.0,
) -> MagicMock:
    """Build a mock AIConfiguratorPerfEstimator with configurable returns."""
    inst = MagicMock(name="AICEstimator")
    inst.database.system_spec = {"gpu": {"power": tdp_w}}
    inst.estimate_prefill_perf.return_value = {
        "context_latency": ttft_ms,
        "power_w": power_w_prefill,
    }
    inst.estimate_perf.return_value = {
        "tpot": itl_ms,
        "power_w": power_w_decode,
    }
    inst.get_max_kv_tokens.return_value = max_kv_tokens
    return inst


def _make_traffic(
    *,
    num_req: float = 10.0,
    isl: float = 1000.0,
    osl: float = 200.0,
    ttft_avg: float | None = None,
    itl_avg: float | None = None,
    total_tokens_per_s: float | None = None,
    scheduled_prefill_tokens: float | None = 5000.0,
    scheduled_decode_kv_tokens: float | None = 2000.0,
) -> TrafficObservation:
    return TrafficObservation(
        duration_s=60.0,
        num_req=num_req,
        isl=isl,
        osl=osl,
        ttft_avg=ttft_avg,
        itl_avg=itl_avg,
        total_tokens_per_s=total_tokens_per_s,
        scheduled_prefill_tokens=scheduled_prefill_tokens,
        scheduled_decode_kv_tokens=scheduled_decode_kv_tokens,
    )


def _make_optimizer_with_mock(
    estimator_inst: MagicMock,
    config: PlannerConfig | None = None,
) -> tuple[AICPowerOptimizer, MagicMock]:
    """Return (optimizer, metrics) with the estimator class patched."""
    if config is None:
        config = _make_config()
    metrics = _make_metrics()
    opt = AICPowerOptimizer(config, metrics)
    return opt, metrics


# ---------------------------------------------------------------------------
# optimize() — happy path
# ---------------------------------------------------------------------------


class TestOptimizeHappyPath:
    """optimize() with a cooperative AIC mock returns a well-formed config."""

    def test_returns_power_aware_config(self):
        estimator = _make_estimator_mock(ttft_ms=50.0, itl_ms=10.0)
        config = _make_config(ttft=200, itl=50)
        opt, _ = _make_optimizer_with_mock(estimator, config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        assert isinstance(result, PowerAwareConfig)
        assert result.aic_ttft_ms == pytest.approx(50.0)
        assert result.aic_itl_ms == pytest.approx(10.0)
        assert result.n_p >= 1
        assert result.n_d >= 1

    def test_uses_measured_power_w_for_caps(self):
        """When AIC returns non-zero power_w, caps are derived from it, not TDP."""
        estimator = _make_estimator_mock(
            ttft_ms=50.0,
            itl_ms=10.0,
            power_w_prefill=400.0,
            power_w_decode=300.0,
            tdp_w=700.0,
        )
        config = _make_config(c_power_p=1.0, c_power_d=1.0)
        opt, _ = _make_optimizer_with_mock(estimator, config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        assert result is not None
        # cap_p should be ceil(400 * max(1.0, 1.0)) = 400, not 700
        assert result.cap_p == 400
        assert result.cap_d == 300

    def test_uses_tdp_fallback_when_power_w_zero(self):
        """When AIC returns power_w=0, the optimizer falls back to nameplate TDP."""
        estimator = _make_estimator_mock(
            ttft_ms=50.0,
            itl_ms=10.0,
            power_w_prefill=0.0,
            power_w_decode=0.0,
            tdp_w=700.0,
        )
        config = _make_config(c_power_p=1.0, c_power_d=1.0)
        opt, _ = _make_optimizer_with_mock(estimator, config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        assert result is not None
        # cap should be TDP (700) * max(1.0, c_power=1.0) = 700
        assert result.cap_p == 700
        assert result.cap_d == 700

    def test_correction_coefficients_inflate_caps(self):
        """c_power > 1.0 inflates caps above the measured power_w."""
        estimator = _make_estimator_mock(power_w_prefill=400.0, power_w_decode=300.0)
        config = _make_config(c_power_p=1.5, c_power_d=1.2)
        opt, _ = _make_optimizer_with_mock(estimator, config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        import math
        assert result is not None
        assert result.cap_p == math.ceil(400.0 * 1.5)   # 600
        assert result.cap_d == math.ceil(300.0 * 1.2)   # 360

    def test_isl_osl_propagated(self):
        estimator = _make_estimator_mock()
        config = _make_config()
        opt, _ = _make_optimizer_with_mock(estimator, config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        assert result is not None
        assert result.isl == _AIC_SPEC.isl
        assert result.osl == _AIC_SPEC.osl

    def test_implied_admission_thresholds_in_unit_interval(self):
        """theta_decode_impl and theta_prefill_frac_impl must always be in [0, 1]."""
        estimator = _make_estimator_mock(ttft_ms=50.0, itl_ms=10.0, max_kv_tokens=100_000)
        config = _make_config()
        opt, _ = _make_optimizer_with_mock(estimator, config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        assert result is not None
        assert 0.0 <= result.theta_decode_impl <= 1.0
        assert 0.0 <= result.theta_prefill_frac_impl <= 1.0

    def test_agg_mode_single_cap(self):
        """In agg mode, cap_p == cap_d (both set from c_power_agg)."""
        estimator = _make_estimator_mock(power_w_prefill=500.0, power_w_decode=500.0)
        config = _make_config(mode="agg", c_power_agg=1.1)
        opt, _ = _make_optimizer_with_mock(estimator, config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        assert result is not None
        assert result.cap_p == result.cap_d


# ---------------------------------------------------------------------------
# optimize() — defensive clamp for non-physical AIC power_w
# ---------------------------------------------------------------------------


class TestAICPowerWClamp:
    """Defensive clamp guarding against non-physical AIC power_w outputs.

    Context: AIC's per-kernel 3D-cubic interpolator extrapolates non-physical
    power values at sparse-grid query points (observed on h200_sxm + vLLM
    0.19.1 + generation_attention at batch >= 256; aggregate power_w returned
    by estimate_perf was 1275 W vs a 700 W TDP and a 721 W measured peak).

    The optimizer must:
      - cap cap_*_per_gpu at ceil(TDP × max(1, c_power_*)) when AIC's raw
        power_w exceeds TDP × 1.1;
      - preserve the unclamped raw value on PowerAwareConfig.aic_power_w_*
        so update_correction() has a true denominator and c_power_*
        coefficients converge below 1.0 instead of being silently masked;
      - increment aic_power_w_clamped_total{side="..."} for each clamped side.
    """

    def test_nonphysical_decode_power_clamps_cap_to_tdp(self):
        """1275 W decode → cap_d clamped at ceil(700 W × c_power_d=1.0) = 700."""
        estimator = _make_estimator_mock(
            power_w_prefill=400.0,
            power_w_decode=1275.0,
            tdp_w=700.0,
        )
        config = _make_config(c_power_p=1.0, c_power_d=1.0)
        opt, metrics = _make_optimizer_with_mock(estimator, config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        assert result is not None
        assert result.cap_d == 700, (
            f"cap_d={result.cap_d} W must be clamped to TDP=700 W, "
            "not derived from the bogus 1275 W AIC value"
        )
        assert result.cap_p == 400  # prefill in range → unchanged

    def test_clamped_side_preserves_raw_value_on_config(self):
        """PowerAwareConfig.aic_power_w_decode keeps the raw 1275 W so EMA
        sees the true mismatch and c_power_decode converges below 1.0."""
        estimator = _make_estimator_mock(
            power_w_prefill=400.0,
            power_w_decode=1275.0,
            tdp_w=700.0,
        )
        opt, _ = _make_optimizer_with_mock(estimator)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        assert result is not None
        assert result.aic_power_w_decode == pytest.approx(1275.0), (
            "raw AIC value must be preserved on the config so update_correction() "
            "can normalise observed power against AIC's actual (over-)prediction"
        )
        assert result.aic_power_w_prefill == pytest.approx(400.0)

    def test_clamp_metric_increments_only_for_offending_side(self):
        estimator = _make_estimator_mock(
            power_w_prefill=400.0,    # in range — should NOT clamp
            power_w_decode=1500.0,    # > 1.1 × 700 — MUST clamp
            tdp_w=700.0,
        )
        opt, metrics = _make_optimizer_with_mock(estimator)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            opt.optimize()

        sides = [
            c.kwargs.get("side")
            for c in metrics.aic_power_w_clamped_total.labels.call_args_list
        ]
        assert "decode" in sides
        assert "prefill" not in sides

    def test_at_threshold_does_not_clamp(self):
        """Boundary: power_w exactly == 1.1 × TDP must NOT clamp (<= rule)."""
        threshold = 700.0 * 1.1
        estimator = _make_estimator_mock(
            power_w_prefill=threshold,
            power_w_decode=threshold,
            tdp_w=700.0,
        )
        config = _make_config(c_power_p=1.0, c_power_d=1.0)
        opt, metrics = _make_optimizer_with_mock(estimator, config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        import math
        assert result is not None
        assert result.cap_p == math.ceil(threshold)   # 770
        assert result.cap_d == math.ceil(threshold)   # 770
        sides = [
            c.kwargs.get("side")
            for c in metrics.aic_power_w_clamped_total.labels.call_args_list
        ]
        assert sides == []

    def test_agg_mode_clamps_via_averaged_path(self):
        """In agg mode the (prefill_raw + decode_raw)/2 path uses the CLAMPED
        per-side values, so a single bad side cannot inflate cap_p == cap_d."""
        estimator = _make_estimator_mock(
            power_w_prefill=400.0,
            power_w_decode=1275.0,    # bogus
            tdp_w=700.0,
        )
        config = _make_config(mode="agg", c_power_agg=1.0)
        opt, metrics = _make_optimizer_with_mock(estimator, config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        import math
        assert result is not None
        # Averaged clamped path: (400 + 700) / 2 = 550
        assert result.cap_p == math.ceil((400.0 + 700.0) / 2.0)
        assert result.cap_p == result.cap_d
        sides = [
            c.kwargs.get("side")
            for c in metrics.aic_power_w_clamped_total.labels.call_args_list
        ]
        assert "decode" in sides


# ---------------------------------------------------------------------------
# optimize() — budget constraint
# ---------------------------------------------------------------------------


class TestOptimizeBudgetConstraint:
    """Budget limits drive replica count selection."""

    def test_budget_limits_replicas(self):
        """Tight budget produces fewer replicas than the maximum."""
        estimator = _make_estimator_mock(power_w_prefill=400.0, power_w_decode=300.0)
        # 8-GPU engines; cap_p = 400*8 = 3200 W/replica, cap_d = 300*8 = 2400 W/replica
        # Budget 10000 W → at most 3 decode (3×2400=7200) + 1 prefill (3200) = 10400 > 10000
        # So expect ≤ 3 decode replicas
        config = _make_config(
            total_gpu_power_limit=10_000,
            max_gpu_budget=32,
        )
        opt, _ = _make_optimizer_with_mock(estimator, config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        assert result is not None
        # Total wattage must stay within budget
        total_w = result.n_p * result.cap_p * 8 + result.n_d * result.cap_d * 8
        assert total_w <= 10_000 + 100  # small rounding allowance

    def test_budget_too_small_returns_min_endpoint(self):
        """When the budget can't cover min_endpoint prefill, fall back to (min_ep, min_ep)."""
        # 8-GPU engine, cap = 700 W/GPU → 5600 W/replica
        # Budget = 1000 W → cannot cover even 1 prefill replica
        estimator = _make_estimator_mock(power_w_prefill=0.0, power_w_decode=0.0, tdp_w=700.0)
        config = _make_config(
            total_gpu_power_limit=1_000,   # impossibly small
            min_endpoint=1,
        )
        opt, _ = _make_optimizer_with_mock(estimator, config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        # Should fall back to (min_ep, min_ep)
        assert result is not None
        assert result.n_p == 1
        assert result.n_d == 1

    def test_unbounded_budget(self):
        """When total_gpu_power_limit is effectively unlimited, replicas are capped only by max_gpu_budget."""
        estimator = _make_estimator_mock(power_w_prefill=400.0, power_w_decode=300.0)
        # Validator forbids None when enable_power_awareness=True; use a very
        # large value to make the power constraint effectively non-binding so
        # max_gpu_budget becomes the binding constraint.
        config = _make_config(
            total_gpu_power_limit=200_000,  # 200 kW — far above any realistic per-DGD budget
            max_gpu_budget=8,
        )
        opt, _ = _make_optimizer_with_mock(estimator, config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        assert result is not None
        # With unlimited budget, n_d should be driven by max_replicas from max_gpu_budget
        assert result.n_d >= 1


# ---------------------------------------------------------------------------
# optimize() — failure handling
# ---------------------------------------------------------------------------


class TestOptimizeFailureHandling:
    """Failure modes: startup disable, runtime keep-last, auto-disable."""

    def test_startup_import_error_disables_optimizer(self):
        """ImportError from aiconfigurator at startup → disabled fail-closed."""
        config = _make_config()
        opt, _ = _make_optimizer_with_mock(MagicMock(), config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            side_effect=ImportError("aiconfigurator not installed"),
        ):
            result = opt.optimize()

        assert result is None
        assert opt._disabled is True

    def test_startup_sla_infeasible_disables_optimizer(self):
        """When corrected TTFT > ttft_sla at first sweep, optimizer is disabled."""
        # AIC estimates 180ms TTFT; SLA is 100ms; c_ttft=1.0 → corrected = 180ms > 100ms
        estimator = _make_estimator_mock(ttft_ms=180.0, itl_ms=5.0)
        config = _make_config(ttft=100, itl=50)  # tight TTFT SLA
        opt, _ = _make_optimizer_with_mock(estimator, config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            result = opt.optimize()

        assert result is None
        assert opt._disabled is True

    def test_runtime_failure_returns_last_config(self):
        """After a successful sweep, a subsequent exception keeps the last config."""
        estimator_ok = _make_estimator_mock(ttft_ms=50.0, itl_ms=10.0)
        config = _make_config()
        opt, _ = _make_optimizer_with_mock(estimator_ok, config)

        # First sweep succeeds.
        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator_ok,
        ):
            first_result = opt.optimize()

        assert first_result is not None
        assert not opt._disabled

        # Second sweep: AIC throws.
        estimator_fail = _make_estimator_mock()
        estimator_fail.estimate_prefill_perf.side_effect = RuntimeError("AIC down")

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator_fail,
        ):
            second_result = opt.optimize()

        assert second_result is first_result  # same object = last good config
        assert not opt._disabled              # not disabled yet (1 of 3 failures)
        assert opt._consecutive_failures == 1

    def test_auto_disable_after_max_consecutive_failures(self):
        """After aic_max_consecutive_failures runtime failures, optimizer disables."""
        estimator_ok = _make_estimator_mock(ttft_ms=50.0, itl_ms=10.0)
        config = _make_config(aic_max_consecutive_failures=2)
        opt, _ = _make_optimizer_with_mock(estimator_ok, config)

        # Bootstrap with one good sweep.
        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator_ok,
        ):
            opt.optimize()

        # Now fail max_consecutive_failures times.
        estimator_fail = _make_estimator_mock()
        estimator_fail.estimate_prefill_perf.side_effect = RuntimeError("AIC down")

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator_fail,
        ):
            for _ in range(2):
                opt.optimize()

        assert opt._disabled is True

    def test_optimize_returns_none_when_already_disabled(self):
        config = _make_config()
        opt, _ = _make_optimizer_with_mock(MagicMock(), config)
        opt._disabled = True

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
        ) as cls_mock:
            result = opt.optimize()

        # The estimator should never be constructed when already disabled.
        cls_mock.assert_not_called()
        assert result is None


# ---------------------------------------------------------------------------
# update_correction() — EMA updates and gating
# ---------------------------------------------------------------------------


class TestUpdateCorrection:
    """EMA updates for latency and power correction coefficients."""

    def _optimizer_with_last_config(self, **config_kwargs) -> AICPowerOptimizer:
        """Return an optimizer with a synthetic _last_optimal_config for EMA tests.

        ``aic_power_w_*`` are pinned to 700W so the EMA-update arithmetic in
        these tests can express the AIC raw-power denominator as 700.  The
        caps differ (400/300) — they intentionally do NOT match the raw power,
        which exercises the bug-#2 fix (denominator must be the raw value,
        not the cap).
        """
        estimator = _make_estimator_mock(ttft_ms=50.0, itl_ms=10.0)
        config = _make_config(**config_kwargs)
        opt, _ = _make_optimizer_with_mock(estimator, config)
        # Inject a plausible last_optimal_config without running optimize().
        opt._last_optimal_config = PowerAwareConfig(
            n_p=1, n_d=2, cap_p=400, cap_d=300,
            aic_ttft_ms=50.0, aic_itl_ms=10.0,
            aic_seq_per_s_per_replica=100.0,
            isl=1000, osl=200,
            theta_decode_impl=0.8, theta_prefill_frac_impl=0.6,
            aic_power_w_prefill=700.0,
            aic_power_w_decode=700.0,
            aic_power_w_agg=700.0,
        )
        return opt

    def test_latency_ema_updates_when_traffic_present(self):
        opt = self._optimizer_with_last_config(c_ttft=1.0, c_itl=1.0)
        traffic = _make_traffic(num_req=10.0)
        initial_c_ttft = opt._c_ttft

        # Observed TTFT = 75 ms = 0.075 s; AIC TTFT = 50 ms = 0.050 s → ratio = 1.5
        opt.update_correction(
            traffic,
            observed_ttft_avg=0.075,
            observed_itl_avg=0.015,
        )

        expected_c_ttft = initial_c_ttft + _EMA_ALPHA * (1.5 - initial_c_ttft)
        assert opt._c_ttft == pytest.approx(expected_c_ttft, rel=1e-3)

    def test_latency_ema_skipped_when_no_traffic(self):
        opt = self._optimizer_with_last_config(c_ttft=1.0, c_itl=1.0)
        traffic = _make_traffic(num_req=0.0)
        initial_c_ttft = opt._c_ttft

        opt.update_correction(traffic, observed_ttft_avg=0.075)

        # No traffic → EMA should NOT update.
        assert opt._c_ttft == initial_c_ttft

    def test_power_prefill_ema_skipped_when_idle(self):
        """Prefill coefficient must not update when scheduled_prefill_tokens == 0."""
        opt = self._optimizer_with_last_config(c_power_p=1.0)
        traffic = _make_traffic(scheduled_prefill_tokens=0.0, num_req=5.0)
        initial_c_power_p = opt._c_power_p

        opt.update_correction(traffic, observed_power_w_prefill=600.0)

        assert opt._c_power_p == initial_c_power_p

    def test_power_prefill_ema_updates_when_active(self):
        """Prefill coefficient updates when the prefill side has scheduled work."""
        opt = self._optimizer_with_last_config(c_power_p=1.0)
        # aic_p_w = config.prefill_engine_gpu_power_limit = 700
        # observed = 840 → ratio = 840/700 = 1.2
        traffic = _make_traffic(scheduled_prefill_tokens=5000.0)
        initial = opt._c_power_p

        opt.update_correction(traffic, observed_power_w_prefill=840.0)

        expected = initial + _EMA_ALPHA * (840.0 / 700.0 - initial)
        assert opt._c_power_p == pytest.approx(expected, rel=1e-3)

    def test_power_decode_ema_gated_independently(self):
        """Decode-side EMA ignores prefill-side scheduled tokens."""
        opt = self._optimizer_with_last_config(c_power_d=1.0)
        # decode side active, prefill idle
        traffic = _make_traffic(scheduled_prefill_tokens=0.0, scheduled_decode_kv_tokens=2000.0)
        initial_d = opt._c_power_d
        initial_p = opt._c_power_p

        # Synthetic last_config has aic_power_w_decode=700.0; observe 850.0 so
        # the EMA actually moves (ratio 850/700 ≈ 1.214 ≠ initial_d=1.0).
        opt.update_correction(
            traffic,
            observed_power_w_prefill=600.0,
            observed_power_w_decode=850.0,
        )

        assert opt._c_power_p == initial_p   # prefill idle → unchanged
        assert opt._c_power_d != initial_d   # decode active → updated

    def test_agg_mode_uses_single_coefficient(self):
        opt = self._optimizer_with_last_config(mode="agg", c_power_agg=1.0)
        traffic = _make_traffic(scheduled_prefill_tokens=3000.0, scheduled_decode_kv_tokens=2000.0)
        initial_agg = opt._c_power_agg
        initial_p = opt._c_power_p

        # aic_power_w_agg from the synthetic last_optimal_config = 700
        # observed = 850 → ratio = 850/700 ≈ 1.214 → EMA shifts upward.
        opt.update_correction(traffic, observed_power_w_agg=850.0)

        expected_agg = initial_agg + _EMA_ALPHA * (850.0 / 700.0 - initial_agg)
        assert opt._c_power_agg == pytest.approx(expected_agg, rel=1e-3)
        assert opt._c_power_agg != initial_agg   # agg mode updates c_power_agg
        # per-component coefficients should remain unchanged in agg mode
        assert opt._c_power_p == initial_p

    def test_coefficient_clamp_max(self):
        """EMA result clamped to _COEFF_MAX = 2.0 when raw ratio is very large."""
        opt = self._optimizer_with_last_config(c_ttft=1.9)
        traffic = _make_traffic(num_req=5.0)
        # observed TTFT = 10x AIC estimate → raw ratio = 10.0
        # EMA = 1.9 + 0.3*(10.0 - 1.9) = 1.9 + 2.43 = 4.33 → clamped to 2.0
        opt.update_correction(traffic, observed_ttft_avg=0.5)  # 500ms vs 50ms AIC

        assert opt._c_ttft == _COEFF_MAX

    def test_coefficient_clamp_min(self):
        """EMA result clamped to _COEFF_MIN = 0.5 when raw ratio is very small."""
        opt = self._optimizer_with_last_config(c_ttft=0.6)
        traffic = _make_traffic(num_req=5.0)
        # observed TTFT = 0.0001ms → ratio ≈ 0 → EMA drives toward 0 → clamp to 0.5
        opt.update_correction(traffic, observed_ttft_avg=0.000001)

        assert opt._c_ttft == _COEFF_MIN

    def test_no_update_when_disabled(self):
        opt = self._optimizer_with_last_config(c_ttft=1.0)
        opt._disabled = True
        opt.update_correction(
            _make_traffic(num_req=10.0),
            observed_ttft_avg=0.1,
        )
        assert opt._c_ttft == 1.0   # unchanged


# ---------------------------------------------------------------------------
# should_reoptimize() — drift detection
# ---------------------------------------------------------------------------


class TestShouldReoptimize:
    """Drift detection: rate limit, SLA miss, capacity exceeded, hysteresis."""

    def _optimizer_with_recent_sweep(self, **config_kwargs) -> AICPowerOptimizer:
        """Return an optimizer whose last sweep was just now (rate-limit applies)."""
        config = _make_config(**config_kwargs)
        opt, _ = _make_optimizer_with_mock(MagicMock(), config)
        opt._time_of_last_optimize = time.monotonic()
        opt._estimated_throughput = 1_000.0  # tok/s
        return opt

    def test_rate_limit_prevents_reoptimize(self):
        opt = self._optimizer_with_recent_sweep(aic_reoptimize_interval=300.0)
        # Simulate SLA violation — but interval hasn't elapsed.
        traffic = _make_traffic(ttft_avg=5.0)   # 5s >> any SLA in ms
        assert opt.should_reoptimize(traffic) is False

    def test_sla_miss_triggers_after_hysteresis(self):
        """3 consecutive ticks with TTFT above target → True on the third tick."""
        opt = self._optimizer_with_recent_sweep(
            aic_reoptimize_interval=0.0,
            aic_drift_consecutive_ticks=3,
            ttft=200,  # 200ms SLA
        )
        traffic = _make_traffic(ttft_avg=0.5)  # 500ms observed > 200ms SLA

        assert opt.should_reoptimize(traffic) is False  # tick 1
        assert opt.should_reoptimize(traffic) is False  # tick 2
        assert opt.should_reoptimize(traffic) is True   # tick 3 — triggers

    def test_no_reoptimize_on_single_sla_miss(self):
        """One tick of SLA miss doesn't trigger when hysteresis=3."""
        opt = self._optimizer_with_recent_sweep(
            aic_reoptimize_interval=0.0,
            aic_drift_consecutive_ticks=3,
        )
        traffic = _make_traffic(ttft_avg=5.0)
        opt.should_reoptimize(traffic)  # tick 1 — violation
        # Clear with a good tick.
        good_traffic = _make_traffic(ttft_avg=0.01)
        assert opt.should_reoptimize(good_traffic) is False
        assert opt._consecutive_violation_ticks == 0

    def test_capacity_exceeded_triggers_upward_only(self):
        """Capacity-exceeded fires only when tokens/s > estimated × (1+threshold)."""
        opt = self._optimizer_with_recent_sweep(
            aic_reoptimize_interval=0.0,
            aic_drift_consecutive_ticks=1,
            aic_drift_relative_threshold=0.15,
        )
        opt._estimated_throughput = 1_000.0

        # Under-load (tokens below estimate) should NOT trigger.
        under = _make_traffic(total_tokens_per_s=200.0)
        assert opt.should_reoptimize(under) is False

        # Over-load (tokens above estimate × (1 + 0.15)) should trigger.
        over = _make_traffic(total_tokens_per_s=1_200.0)
        assert opt.should_reoptimize(over) is True

    def test_disabled_optimizer_never_reoptimizes(self):
        opt = self._optimizer_with_recent_sweep(aic_reoptimize_interval=0.0)
        opt._disabled = True
        traffic = _make_traffic(ttft_avg=10.0, total_tokens_per_s=9_999.0)
        assert opt.should_reoptimize(traffic) is False


# ---------------------------------------------------------------------------
# _sweep_replicas() — direct unit tests
# ---------------------------------------------------------------------------


class TestSweepReplicas:
    """The internal replica-count sweep helper."""

    def _optimizer(self) -> AICPowerOptimizer:
        config = _make_config(min_endpoint=1, max_gpu_budget=64)
        return AICPowerOptimizer(config, _make_metrics())

    def test_budget_constrains_n_d_first(self):
        opt = self._optimizer()
        # p_watts=3200, d_watts=2400, budget=14000, min_ep=1
        # After min_ep=1 prefill: remaining = 14000 - 3200 = 10800 → n_d = 10800//2400 = 4
        # After locking n_d=4: remaining for p = 14000 - 4*2400 = 4400 → n_p = 4400//3200 = 1
        n_p, n_d = opt._sweep_replicas(1, 10, 3200, 2400, 14_000)
        assert n_d == 4
        assert n_p == 1

    def test_no_budget_uses_max_replicas_for_decode(self):
        opt = self._optimizer()
        n_p, n_d = opt._sweep_replicas(1, 5, 3200, 2400, None)
        assert n_d == 5

    def test_min_endpoint_is_lower_bound(self):
        opt = self._optimizer()
        n_p, n_d = opt._sweep_replicas(3, 10, 3200, 2400, 100_000)
        assert n_p >= 3
        assert n_d >= 3

    def test_budget_too_small_returns_min_endpoint(self):
        opt = self._optimizer()
        # budget=100 < min_ep * p_watts=3200 → falls back to (min_ep, min_ep)
        n_p, n_d = opt._sweep_replicas(1, 10, 3200, 2400, 100)
        assert n_p == 1
        assert n_d == 1


# ---------------------------------------------------------------------------
# Throughput regression warning
# ---------------------------------------------------------------------------


class TestThroughputRegressionWarning:
    def test_regression_increments_counter(self):
        """When re-optimization yields lower predicted throughput, counter increments."""
        estimator = _make_estimator_mock(ttft_ms=50.0, itl_ms=10.0, max_kv_tokens=50_000)
        config = _make_config(aic_throughput_regression_warn_threshold=0.05)
        opt, metrics = _make_optimizer_with_mock(estimator, config)

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            first_result = opt.optimize()

        # Inject a "previous" config with much higher throughput so the second
        # sweep looks like a regression.
        assert first_result is not None
        opt._last_optimal_config = PowerAwareConfig(
            n_p=1, n_d=100,                   # 100 decode replicas → huge throughput
            cap_p=400, cap_d=300,
            aic_ttft_ms=50.0, aic_itl_ms=10.0,
            aic_seq_per_s_per_replica=1000.0,
            isl=1000, osl=200,
            theta_decode_impl=0.8, theta_prefill_frac_impl=0.6,
            aic_power_w_prefill=400.0,
            aic_power_w_decode=300.0,
        )

        with patch(
            "dynamo.planner.monitoring.aic_estimator.AIConfiguratorPerfEstimator",
            return_value=estimator,
        ):
            opt.optimize()

        # The regression counter should have been incremented once.
        metrics.aic_throughput_regression_total.inc.assert_called()
