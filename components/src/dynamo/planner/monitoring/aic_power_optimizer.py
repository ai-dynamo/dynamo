# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIC-closed-loop power optimizer (Phase 3).

Wraps AIConfigurator to produce a per-tick ``PowerAwareConfig`` — the
recommended (n_p, n_d, cap_p, cap_d) tuple that the planner applies via
``NativePlannerBase._apply_aic_config()``.

Phase 3 is TDP-only: ``power_w`` columns in the AIC database are zeroed so
we fall back to the nameplate TDP from ``system_spec["gpu"]["tdp_w"]``.
Phase 4 will add a power-sweep dimension once the AIC team backfills measured
per-configuration power data.

Key design decisions (§5.3–§5.6 of powerplanner-design.md):
- EMA-smoothed correction coefficients (c_ttft, c_itl, c_power_p/d/agg)
  bridge AIC offline estimates to live serving behaviour.
- Asymmetric clamp max(1.0, c) is applied when gating feasibility so we
  never *loosen* SLA or under-cap power based on noisy live data.
- EMA updates are gated: latency on ``traffic.num_req > 0``; per-side power
  on ``traffic.scheduled_*_tokens > 0`` to prevent idle-side EMA drag.
- Drift detection uses ONLY the upward-exceeded direction for throughput to
  avoid spurious re-sweeps in under-loaded clusters (§5.6 "Direction matters").
- Failure modes: fail-open at runtime (keep last config), fail-closed at
  startup (scale to min_endpoint, disable optimizer, planner stays alive).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dynamo.planner.config.planner_config import PlannerConfig
    from dynamo.planner.core.types import TrafficObservation
    from dynamo.planner.monitoring.planner_metrics import PlannerPrometheusMetrics

logger = logging.getLogger(__name__)

# EMA smoothing factor — §5.3 "α = 0.3".
_EMA_ALPHA = 0.3
# Coefficient clamp bounds — §5.3 "clamped to [0.5, 2.0] after each update".
_COEFF_MIN = 0.5
_COEFF_MAX = 2.0

# Defensive clamp for AIC's reported per-GPU power_w.  Empirically (see
# .agents/research/aic_power_extrapolation_2026-05-11.md), AIC's per-kernel
# 3D-cubic interpolator can return non-physical power values when the query
# falls into a sparse corner of the offline data lattice — observed on
# h200_sxm + vLLM 0.19.1 + generation_attention at batch >= 256 (returned
# ~1563 W per op vs a 721 W measured peak).  Any aic_power_w above
# TDP × _AIC_POWER_W_NONPHYSICAL_MULTIPLE is treated as bogus and clamped to
# TDP before being used to derive per-GPU caps.  The cap that ultimately
# reaches NVML can never legitimately exceed TDP anyway (the daemon would
# clamp it down silently), so this clamp only loses fidelity in the
# "AIC was wrong" regime — never in the physical regime.
_AIC_POWER_W_NONPHYSICAL_MULTIPLE = 1.1


@dataclass
class PowerAwareConfig:
    """Output of a single AIC sweep.

    Carries everything ``NativePlannerBase._apply_aic_config()`` needs to
    update the planner state: replica counts, per-GPU power caps, drift
    reference, and implied admission thresholds.

    The ``aic_power_w_*`` fields preserve AIC's *raw* per-GPU power estimate
    (before any EMA correction) so ``update_correction()`` can normalise
    observed power against the AIC prediction — not against the planner's
    applied cap.  Comparing against the cap would collapse the EMA toward
    1.0 because the cap = ceil(aic_power_w × max(1, c_power)) by construction.
    """

    n_p: int  # recommended prefill replicas
    n_d: int  # recommended decode replicas
    cap_p: int  # per-GPU power cap for prefill (watts)
    cap_d: int  # per-GPU power cap for decode (watts)
    aic_ttft_ms: float  # AIC-estimated TTFT (before correction)
    aic_itl_ms: float  # AIC-estimated ITL  (before correction)
    aic_seq_per_s_per_replica: float  # saturated decode throughput, for drift ref
    isl: int
    osl: int
    theta_decode_impl: float  # implied decode KV utilization (§5.7)
    theta_prefill_frac_impl: float  # implied prefill fractional utilization
    # Raw AIC per-GPU power estimates used as EMA denominators in
    # update_correction().  Falls back to nameplate TDP when AIC's perf DB
    # has zeroed power_w (Phase 3 typical state — see §6.4).
    aic_power_w_prefill: float = 0.0
    aic_power_w_decode: float = 0.0
    aic_power_w_agg: float = 0.0


class AICPowerOptimizer:
    """Closed-loop AIC optimizer for power-aware planner (Phase 3).

    Instantiated once at planner startup when ``enable_aic_optimizer=True``.
    The public API:

    * ``optimize()`` — blocking AIC sweep; run via ``asyncio.to_thread``.
    * ``update_correction(...)`` — update EMA coefficients after each tick.
    * ``should_reoptimize(traffic)`` — drift + hysteresis check.
    """

    def __init__(
        self,
        config: "PlannerConfig",
        metrics: "PlannerPrometheusMetrics",
    ) -> None:
        self._config = config
        self._metrics = metrics

        spec = config.aic_interpolation
        if spec is None:
            raise ValueError(
                "enable_aic_optimizer=True requires aic_interpolation to be set "
                "in PlannerConfig. See AICInterpolationSpec for required fields."
            )
        self._spec = spec

        # EMA coefficients — initialised from config (cold-start values).
        self._c_ttft: float = config.aic_initial_c_ttft
        self._c_itl: float = config.aic_initial_c_itl
        self._c_power_p: float = config.aic_initial_c_power_prefill
        self._c_power_d: float = config.aic_initial_c_power_decode
        self._c_power_agg: float = config.aic_initial_c_power_agg

        # Drift detection state (§5.6).
        self._estimated_throughput: float = (
            0.0  # tokens/s; updated by _apply_aic_config
        )
        self._consecutive_violation_ticks: int = 0
        self._time_of_last_optimize: float = 0.0

        # Failure-handling state (§8).
        self._consecutive_failures: int = 0
        self._last_optimal_config: Optional[PowerAwareConfig] = None
        self._disabled: bool = False
        self._disabled_reason: str = ""

        # Emit initial coefficient gauges so dashboards show cold-start values.
        self._emit_coefficient_metrics()

        # Optional testbed injection: when set, replaces the real
        # AIConfiguratorPerfEstimator instantiation inside optimize().
        # Type: Optional[Callable[[str, str, str], Any]]  (hf_id, system, backend) → estimator
        # Set only by the synthetic testbed; never touched in production.
        self._aic_estimator_factory = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self) -> Optional[PowerAwareConfig]:
        """Run a full AIC sweep and return the recommended config.

        Always call via ``asyncio.to_thread`` — the first call per process
        takes 3-6 s (perf-database load from disk, dominated by parsing
        ~10-100 K rows of TXT data and building cubic interpolators).
        Subsequent calls reuse ``aiconfigurator.sdk.perf_database.databases_cache``
        and complete in ~10-15 ms.  Measured 2026-05-11 against the
        in-repo H200 vLLM 0.19.1 and B200 TRT-LLM 1.3.0rc6 sandboxes
        (see .aic_bench.py).

        Closed-loop budget at the default ``aic_reoptimize_interval=300s``
        is therefore well under 0.01 % planner-CPU duty cycle at steady
        state — the rate-limit + hysteresis guards exist for downstream
        impact (annotation PATCH rate, frontend ``/busy_threshold`` POST
        fanout, config flapping), not for AIC sweep CPU cost.

        Returns ``None`` when the optimizer is auto-disabled (§8 rows 1, 4).
        """
        if self._disabled:
            return None

        spec = self._spec
        system = self._config.aic_system or spec.system

        try:
            if self._aic_estimator_factory is not None:
                estimator = self._aic_estimator_factory(
                    hf_id=spec.hf_id, system=system, backend=spec.backend
                )
            else:
                from dynamo.planner.monitoring.aic_estimator import (
                    AIConfiguratorPerfEstimator,
                )

                estimator = AIConfiguratorPerfEstimator(
                    hf_id=spec.hf_id,
                    system=system,
                    backend=spec.backend,
                )
            from dynamo.planner.config.parallelization import (
                picked_to_aic_model_config_kwargs,
            )
        except ImportError as exc:
            self._handle_sweep_failure(
                f"aiconfigurator not installed: {exc}",
                at_startup=(self._last_optimal_config is None),
            )
            return self._last_optimal_config

        except Exception as exc:
            self._handle_sweep_failure(
                f"Failed to initialise AIC estimator: {exc}",
                at_startup=(self._last_optimal_config is None),
            )
            return self._last_optimal_config

        prefill_kwargs = picked_to_aic_model_config_kwargs(spec.prefill_pick)
        decode_kwargs = picked_to_aic_model_config_kwargs(spec.decode_pick)

        p_gpu = spec.prefill_pick.num_gpus
        d_gpu = spec.decode_pick.num_gpus

        # ---------- Nameplate TDP (used as fallback when power_w is unavailable) ----------
        try:
            tdp_w = float(estimator.database.system_spec["gpu"]["power"])
            if tdp_w <= 0:
                raise ValueError(f"tdp_w={tdp_w} is not positive")
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "AIC optimizer: could not read GPU power from system spec (%s); "
                "falling back to prefill_engine_gpu_power_limit=%d W.",
                exc,
                self._config.prefill_engine_gpu_power_limit,
            )
            tdp_w = float(self._config.prefill_engine_gpu_power_limit)

        # ---------- Single-engine AIC calls (TTFT, ITL, and per-config power_w) ----------
        try:
            prefill_perf = estimator.estimate_prefill_perf(
                isl=spec.isl, **prefill_kwargs
            )
            aic_ttft_ms = prefill_perf.get("context_latency")
            if aic_ttft_ms is None or aic_ttft_ms <= 0:
                raise ValueError(f"AIC returned invalid TTFT: {aic_ttft_ms}")
            aic_ttft_ms = float(aic_ttft_ms)
            aic_power_w_prefill = float(prefill_perf.get("power_w") or 0.0)
        except Exception as exc:
            self._handle_sweep_failure(
                f"AIC prefill estimate failed: {exc}",
                at_startup=(self._last_optimal_config is None),
            )
            return self._last_optimal_config

        try:
            attention_dp = max(1, spec.decode_pick.dp)
            per_rank_max_kv = estimator.get_max_kv_tokens(
                spec.isl, spec.osl, **decode_kwargs
            )
            max_kv_aggregate = max(1, per_rank_max_kv * attention_dp)
            max_concurrency = max(1, max_kv_aggregate // (spec.isl + spec.osl))
            batch_size_per_rank = max(1, max_concurrency // attention_dp)

            decode_perf = estimator.estimate_perf(
                spec.isl,
                spec.osl,
                batch_size_per_rank,
                mode="decode",
                **decode_kwargs,
            )
            aic_itl_ms = decode_perf.get("tpot")
            if aic_itl_ms is None or aic_itl_ms <= 0:
                raise ValueError(f"AIC returned invalid ITL (tpot): {aic_itl_ms}")
            aic_itl_ms = float(aic_itl_ms)
            aic_power_w_decode = float(decode_perf.get("power_w") or 0.0)
        except Exception as exc:
            self._handle_sweep_failure(
                f"AIC decode estimate failed: {exc}",
                at_startup=(self._last_optimal_config is None),
            )
            return self._last_optimal_config

        # ---------- Power caps: use measured per-config power_w (Phase 4) or TDP (Phase 3) ----------
        # Resolve the raw AIC power estimate per side; this value is what
        # update_correction() will use as the EMA denominator (§5.3).  When
        # AIC has not backfilled measured power_w yet, fall back to nameplate
        # TDP so the closed loop remains well-defined.
        mode = self._config.mode
        if aic_power_w_prefill > 0 and aic_power_w_decode > 0:
            logger.debug(
                "AIC optimizer: using measured power_w — prefill %.0f W, decode %.0f W.",
                aic_power_w_prefill,
                aic_power_w_decode,
            )
            raw_power_w_prefill = aic_power_w_prefill
            raw_power_w_decode = aic_power_w_decode
        else:
            logger.warning(
                "AIC optimizer: power_w unavailable in AIC result "
                "(database may not have power data yet); using nameplate TDP %.0f W.",
                tdp_w,
            )
            raw_power_w_prefill = tdp_w
            raw_power_w_decode = tdp_w

        # Defensive clamp: AIC's per-kernel power interpolator can extrapolate
        # to non-physical values at sparse-grid query points (observed on H200
        # vLLM generation_attention at batch >= 256).  The clamped values feed
        # cap derivation; the *raw* values are stored on PowerAwareConfig as
        # EMA denominators so c_power_* coefficients visibly converge below 1.0
        # whenever AIC over-predicts, giving operators an independent signal
        # alongside the aic_power_w_clamped_total counter.
        cap_power_w_prefill = self._clamp_aic_power_w(
            raw_power_w_prefill, tdp_w, side="prefill"
        )
        cap_power_w_decode = self._clamp_aic_power_w(
            raw_power_w_decode, tdp_w, side="decode"
        )

        if mode == "agg":
            raw_power_w_agg = (raw_power_w_prefill + raw_power_w_decode) / 2.0
            cap_power_w_agg = (cap_power_w_prefill + cap_power_w_decode) / 2.0
            cap_p_per_gpu = math.ceil(cap_power_w_agg * max(1.0, self._c_power_agg))
            cap_d_per_gpu = cap_p_per_gpu
        else:
            raw_power_w_agg = 0.0
            cap_p_per_gpu = math.ceil(cap_power_w_prefill * max(1.0, self._c_power_p))
            cap_d_per_gpu = math.ceil(cap_power_w_decode * max(1.0, self._c_power_d))

        # ---------- SLA feasibility (single-engine check) ----------
        corrected_ttft_ms = aic_ttft_ms * max(1.0, self._c_ttft)
        corrected_itl_ms = aic_itl_ms * max(1.0, self._c_itl)

        ttft_sla_ms = self._config.ttft_ms  # SLA in ms
        itl_sla_ms = self._config.itl_ms

        if corrected_ttft_ms > ttft_sla_ms or corrected_itl_ms > itl_sla_ms:
            msg = (
                f"AIC optimizer: infeasible SLA at single-engine level — "
                f"corrected TTFT {corrected_ttft_ms:.1f}ms vs target {ttft_sla_ms:.1f}ms, "
                f"corrected ITL {corrected_itl_ms:.1f}ms vs target {itl_sla_ms:.1f}ms."
            )
            self._handle_sweep_failure(
                msg,
                at_startup=(self._last_optimal_config is None),
            )
            return self._last_optimal_config

        # ---------- Replica-count sweep ----------
        # Throughput per decode replica at saturation (seq/s).
        # Formula: max_concurrency sequences complete in osl decode steps,
        # each step taking aic_itl_ms ms → seq/s = max_concurrency*1000 / (itl_ms*osl).
        aic_seq_per_s_per_replica = (
            max_concurrency * 1000.0 / (aic_itl_ms * max(1, spec.osl))
        )

        budget = self._config.total_gpu_power_limit  # None means unbounded
        min_ep = self._config.min_endpoint
        max_replicas = max(
            min_ep,
            (self._config.max_gpu_budget // max(1, d_gpu))
            if self._config.max_gpu_budget > 0
            else 256,
        )

        p_watts = cap_p_per_gpu * p_gpu  # per prefill replica
        d_watts = cap_d_per_gpu * d_gpu  # per decode replica

        best_n_p, best_n_d = self._sweep_replicas(
            min_ep, max_replicas, p_watts, d_watts, budget
        )

        # ---------- Implied admission thresholds (§5.7) ----------
        # θ_decode_impl: KV utilization at max concurrency per decode replica.
        kv_total_tokens_per_replica = max_kv_aggregate  # aggregated across attention-DP
        theta_decode_impl = min(
            1.0,
            max_concurrency
            * (spec.isl + spec.osl / 2.0)
            / max(1, kv_total_tokens_per_replica),
        )
        # θ_prefill_frac_impl: fraction of peak prefill capacity consumed.
        peak_seq_s_per_replica = 1000.0 / max(0.001, aic_ttft_ms)  # at zero queueing
        achieved_seq_s = aic_seq_per_s_per_replica  # per decode replica at saturation
        theta_prefill_frac_impl = min(
            1.0, achieved_seq_s / max(0.001, peak_seq_s_per_replica)
        )

        config = PowerAwareConfig(
            n_p=best_n_p,
            n_d=best_n_d,
            cap_p=cap_p_per_gpu,
            cap_d=cap_d_per_gpu,
            aic_ttft_ms=aic_ttft_ms,
            aic_itl_ms=aic_itl_ms,
            aic_seq_per_s_per_replica=aic_seq_per_s_per_replica,
            isl=spec.isl,
            osl=spec.osl,
            theta_decode_impl=theta_decode_impl,
            theta_prefill_frac_impl=theta_prefill_frac_impl,
            aic_power_w_prefill=raw_power_w_prefill,
            aic_power_w_decode=raw_power_w_decode,
            aic_power_w_agg=raw_power_w_agg,
        )

        # Throughput-regression check (§8 row 5) — apply regardless, just warn.
        if self._last_optimal_config is not None:
            old_tp = (
                self._last_optimal_config.aic_seq_per_s_per_replica
                * self._last_optimal_config.n_d
                * (self._last_optimal_config.isl + self._last_optimal_config.osl)
            )
            new_tp = aic_seq_per_s_per_replica * best_n_d * (spec.isl + spec.osl)
            regress_pct = (old_tp - new_tp) / max(1.0, old_tp)
            if (
                old_tp > 0
                and new_tp < old_tp
                and regress_pct > self._config.aic_throughput_regression_warn_threshold
            ):
                logger.warning(
                    "AIC optimizer: re-optimization produced %.1f%% lower predicted "
                    "throughput (old=%.0f tok/s, new=%.0f tok/s). Applying anyway — "
                    "correction coefficients already factored in.",
                    regress_pct * 100,
                    old_tp,
                    new_tp,
                )
                self._metrics.aic_throughput_regression_total.inc()

        self._consecutive_failures = 0
        self._consecutive_violation_ticks = 0
        self._last_optimal_config = config
        self._time_of_last_optimize = time.monotonic()
        self._metrics.aic_consecutive_failures.set(0)
        logger.info(
            "AIC sweep complete: n_p=%d n_d=%d cap_p=%dW cap_d=%dW "
            "corrected_TTFT=%.1fms corrected_ITL=%.1fms "
            "θ_decode=%.3f θ_prefill_frac=%.3f",
            best_n_p,
            best_n_d,
            cap_p_per_gpu,
            cap_d_per_gpu,
            corrected_ttft_ms,
            corrected_itl_ms,
            theta_decode_impl,
            theta_prefill_frac_impl,
        )
        return config

    def update_correction(
        self,
        traffic: "TrafficObservation",
        observed_ttft_avg: Optional[float] = None,
        observed_itl_avg: Optional[float] = None,
        observed_power_w_prefill: Optional[float] = None,
        observed_power_w_decode: Optional[float] = None,
        observed_power_w_agg: Optional[float] = None,
    ) -> None:
        """Update EMA correction coefficients after each planner tick (§5.3).

        All observed latency values are in **seconds** (matching
        ``PrometheusAPIClient.get_avg_*`` conventions); AIC TTFT/ITL are in ms.
        Gating rules:
        - Latency coefficients: updated only when ``traffic.num_req > 0``.
        - Power coefficients:   updated only when the matching component had
          scheduled work (per-side idle-EMA guard).
        """
        if self._disabled or self._last_optimal_config is None:
            return

        aic_ttft_s = self._last_optimal_config.aic_ttft_ms / 1000.0
        aic_itl_s = self._last_optimal_config.aic_itl_ms / 1000.0

        changed = False

        # -- Latency coefficients (gated on num_req > 0) --
        if traffic.num_req > 0:
            if (
                observed_ttft_avg is not None
                and observed_ttft_avg > 0
                and aic_ttft_s > 0
            ):
                raw = observed_ttft_avg / aic_ttft_s
                self._c_ttft = _ema_update(self._c_ttft, raw, "ttft", self._metrics)
                changed = True

            if observed_itl_avg is not None and observed_itl_avg > 0 and aic_itl_s > 0:
                raw = observed_itl_avg / aic_itl_s
                self._c_itl = _ema_update(self._c_itl, raw, "itl", self._metrics)
                changed = True

        mode = self._config.mode

        # Power EMA denominator is AIC's *raw* per-GPU power estimate from the
        # last sweep — NOT the planner-applied cap.  The cap is derived as
        # ceil(aic_power_w × max(1, c_power)); using it as the denominator
        # collapses the EMA toward 1.0 by construction (§5.3).  The raw value
        # is preserved on _last_optimal_config so subsequent ticks normalise
        # against the same reference between sweeps.
        if mode == "agg":
            # Aggregated mode — single coefficient.
            sched_total = (traffic.scheduled_prefill_tokens or 0) + (
                traffic.scheduled_decode_kv_tokens or 0
            )
            if (
                sched_total > 0
                and observed_power_w_agg is not None
                and observed_power_w_agg > 0
            ):
                aic_power_w = self._last_optimal_config.aic_power_w_agg
                if aic_power_w > 0:
                    raw = observed_power_w_agg / aic_power_w
                    self._c_power_agg = _ema_update(
                        self._c_power_agg, raw, "power_agg", self._metrics
                    )
                    changed = True
        else:
            # Disaggregated mode — per-component.
            if (
                (traffic.scheduled_prefill_tokens or 0) > 0
                and observed_power_w_prefill is not None
                and observed_power_w_prefill > 0
            ):
                aic_p_w = self._last_optimal_config.aic_power_w_prefill
                if aic_p_w > 0:
                    raw = observed_power_w_prefill / aic_p_w
                    self._c_power_p = _ema_update(
                        self._c_power_p, raw, "power_prefill", self._metrics
                    )
                    changed = True

            if (
                (traffic.scheduled_decode_kv_tokens or 0) > 0
                and observed_power_w_decode is not None
                and observed_power_w_decode > 0
            ):
                aic_d_w = self._last_optimal_config.aic_power_w_decode
                if aic_d_w > 0:
                    raw = observed_power_w_decode / aic_d_w
                    self._c_power_d = _ema_update(
                        self._c_power_d, raw, "power_decode", self._metrics
                    )
                    changed = True

        if changed:
            self._emit_coefficient_metrics()

    def should_reoptimize(self, traffic: "TrafficObservation") -> bool:
        """Return True when drift detection + hysteresis says to re-sweep (§5.6).

        Rate-limited by ``aic_reoptimize_interval``.
        Hysteresis requires ``aic_drift_consecutive_ticks`` of sustained signal.
        """
        if self._disabled:
            return False

        elapsed = time.monotonic() - self._time_of_last_optimize
        if elapsed < self._config.aic_reoptimize_interval:
            return False

        sla_violated = False
        if traffic.ttft_avg is not None:
            sla_violated = sla_violated or traffic.ttft_avg > (
                self._config.ttft_ms / 1000.0
            )
        if traffic.itl_avg is not None:
            sla_violated = sla_violated or traffic.itl_avg > (
                self._config.itl_ms / 1000.0
            )

        # Capacity-exceeded: upward direction only (§5.6 "Direction matters").
        capacity_exceeded = False
        if (
            self._estimated_throughput > 0
            and traffic.total_tokens_per_s is not None
            and traffic.total_tokens_per_s
            > self._estimated_throughput
            * (1.0 + self._config.aic_drift_relative_threshold)
        ):
            capacity_exceeded = True

        needs_reopt = sla_violated or capacity_exceeded
        self._consecutive_violation_ticks = (
            self._consecutive_violation_ticks + 1 if needs_reopt else 0
        )
        return (
            self._consecutive_violation_ticks
            >= self._config.aic_drift_consecutive_ticks
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clamp_aic_power_w(
        self, raw_power_w: float, tdp_w: float, *, side: str
    ) -> float:
        """Defensive clamp for AIC's reported per-GPU power_w (§6.4, §8 row 7).

        Returns ``raw_power_w`` unchanged when it is at or below
        ``tdp_w × _AIC_POWER_W_NONPHYSICAL_MULTIPLE``.  Otherwise logs a
        WARNING with both the raw and clamped values, increments the
        ``aic_power_w_clamped_total{side=...}`` counter, and returns
        ``tdp_w``.  Callers must always use the returned value when sizing
        per-GPU caps so a single bad AIC interpolation cannot produce a
        nominal cap larger than the GPU can physically draw.
        """
        threshold = tdp_w * _AIC_POWER_W_NONPHYSICAL_MULTIPLE
        if raw_power_w <= threshold:
            return raw_power_w
        logger.warning(
            "AIC optimizer: %s power_w=%.1f W exceeds TDP×%.2f (%.1f W) — "
            "treating as non-physical interpolation artefact; clamping to TDP %.0f W "
            "for cap computation. EMA denominator preserves the raw value so "
            "c_power_%s converges to live/raw (typically <1) and gives an "
            "independent signal alongside aic_power_w_clamped_total.",
            side,
            raw_power_w,
            _AIC_POWER_W_NONPHYSICAL_MULTIPLE,
            threshold,
            tdp_w,
            side,
        )
        try:
            self._metrics.aic_power_w_clamped_total.labels(side=side).inc()
        except Exception:
            pass
        return tdp_w

    def _sweep_replicas(
        self,
        min_ep: int,
        max_replicas: int,
        p_watts: int,
        d_watts: int,
        budget: Optional[int],
    ) -> tuple[int, int]:
        """Find the (n_p, n_d) that maximises throughput within budget.

        Strategy: maximise n_d first (decode is the throughput bottleneck for
        OSL-heavy workloads), then use remaining budget for n_p.
        Ties broken by fewer total GPUs, then lower cap power (§5.4).
        """
        # Maximum n_d given we must keep at least min_ep prefill replicas.
        if budget is not None:
            remaining_after_min_p = budget - min_ep * p_watts
            if remaining_after_min_p <= 0:
                logger.warning(
                    "AIC optimizer: budget %dW cannot cover min_endpoint prefill "
                    "replicas (%d × %dW); falling back to min_endpoint.",
                    budget,
                    min_ep,
                    p_watts,
                )
                return min_ep, min_ep

            best_n_d = min(max_replicas, remaining_after_min_p // max(1, d_watts))
        else:
            best_n_d = max_replicas

        best_n_d = max(min_ep, best_n_d)

        # Remaining budget after locking in best_n_d.
        if budget is not None:
            remaining_for_p = budget - best_n_d * d_watts
            best_n_p = min(
                max_replicas, max(min_ep, remaining_for_p // max(1, p_watts))
            )
        else:
            best_n_p = min_ep  # conservative default when unbounded

        return best_n_p, best_n_d

    def _handle_sweep_failure(self, msg: str, *, at_startup: bool) -> None:
        """Central failure handler for both startup and runtime sweep failures."""
        if at_startup:
            # §8 rows 1 and 4: disable the optimizer on startup failure.
            reason = (
                "infeasible_at_startup" if "infeasible" in msg else "startup_exception"
            )
            logger.error(
                "AIC optimizer disabled (reason=%s): %s. "
                "Planner will use static _apply_power_budget() enforcement.",
                reason,
                msg,
            )
            self._metrics.aic_optimizer_exceptions_total.inc()
            self._metrics.aic_optimizer_disabled_reason.labels(reason=reason).set(1)
            self._disabled = True
            self._disabled_reason = reason
        else:
            # §8 rows 2 and 3: keep last config, increment failure counter.
            self._consecutive_failures += 1
            self._metrics.aic_consecutive_failures.set(self._consecutive_failures)
            self._metrics.aic_optimizer_exceptions_total.inc()
            logger.error(
                "AIC sweep failure (%d/%d consecutive): %s",
                self._consecutive_failures,
                self._config.aic_max_consecutive_failures,
                msg,
            )
            if self._consecutive_failures >= self._config.aic_max_consecutive_failures:
                logger.error(
                    "AIC optimizer: %d consecutive failures — auto-disabling. "
                    "Restart planner pod with fixed configuration to re-enable.",
                    self._consecutive_failures,
                )
                self._metrics.aic_optimizer_disabled_reason.labels(
                    reason="max_consecutive_failures"
                ).set(1)
                self._disabled = True
                self._disabled_reason = "max_consecutive_failures"

    def _emit_coefficient_metrics(self) -> None:
        """Push current EMA coefficient values to Prometheus gauges."""
        self._metrics.aic_c_ttft.set(self._c_ttft)
        self._metrics.aic_c_itl.set(self._c_itl)
        mode = self._config.mode
        if mode == "agg":
            self._metrics.aic_c_power.labels(component="agg").set(self._c_power_agg)
        else:
            self._metrics.aic_c_power.labels(component="prefill").set(self._c_power_p)
            self._metrics.aic_c_power.labels(component="decode").set(self._c_power_d)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _ema_update(
    prev: float,
    raw: float,
    name: str,
    metrics: "PlannerPrometheusMetrics",
) -> float:
    """Apply one EMA step, clamp to [0.5, 2.0], increment peg counter if saturated."""
    updated = _EMA_ALPHA * raw + (1.0 - _EMA_ALPHA) * prev
    clamped = max(_COEFF_MIN, min(_COEFF_MAX, updated))
    if clamped != updated:
        metrics.aic_correction_pegged_total.labels(coefficient=name).inc()
        logger.critical(
            "AIC correction coefficient '%s' pegged at clamp (raw=%.3f computed=%.3f "
            "clamped=%.3f). AIC calibration may be far from live silicon — investigate.",
            name,
            raw,
            updated,
            clamped,
        )
    return clamped
