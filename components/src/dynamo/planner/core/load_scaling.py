# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="attr-defined"

"""Load-based scaling logic (FPM-driven, reactive).

Mixin consumed by ``PlannerStateMachine``.  All methods access state
via ``self._config``, ``self._capabilities``, and regression models.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from dynamo.planner.core.types import FpmObservations, ScalingDecision

if TYPE_CHECKING:
    from dynamo.common.forward_pass_metrics import ForwardPassMetrics

logger = logging.getLogger(__name__)

# -- Easy-mode static thresholds (optimization_target != "sla") -----------
# Prefill: ratio of queued_prefill_tokens / context_length
_PREFILL_THROUGHPUT_SCALE_UP = 1.0  # queued >= context_length
_PREFILL_THROUGHPUT_SCALE_DOWN = 0.1  # queued < context_length / 10
_PREFILL_LATENCY_SCALE_UP = 0.1  # queued >= context_length / 10
_PREFILL_LATENCY_SCALE_DOWN = 0.0  # queued == 0

# Decode/Agg: KV cache utilization (scheduled + queued) / max_kv_tokens
_DECODE_THROUGHPUT_SCALE_UP = 1.0  # util > 100%
_DECODE_THROUGHPUT_SCALE_DOWN = 0.6  # util < 60%
_DECODE_LATENCY_SCALE_UP = 0.4  # util > 40%
_DECODE_LATENCY_SCALE_DOWN = 0.1  # util < 10%


class LoadScalingMixin:
    """FPM-driven load-based scaling decisions."""

    # Scratch fields owned by PlannerStateMachine, declared here for mypy
    _diag_estimated_ttft_ms: Optional[float]
    _diag_estimated_itl_ms: Optional[float]
    _diag_load_reason: Optional[str]
    _diag_load_reason_prefill: Optional[str]
    _diag_load_reason_decode: Optional[str]

    def _advance_load(self, obs: FpmObservations) -> Optional[ScalingDecision]:
        if not self._config.enable_load_scaling:
            self._diag_load_reason = "disabled"
            return None
        mode = self._config.mode
        if mode == "agg":
            return self._advance_load_agg(obs)
        if mode == "disagg":
            return self._advance_load_disagg(obs)
        return self._advance_load_single(obs, mode)

    def _advance_load_single(
        self, obs: FpmObservations, component: str
    ) -> Optional[ScalingDecision]:
        if self._scaling_in_progress(component):
            logger.info(f"Scaling in progress for {component}, observing only")
            self._diag_load_reason = "scaling_in_progress"
            return None

        fpm_stats = obs.prefill if component == "prefill" else obs.decode
        num_workers = (
            self._num_p_workers if component == "prefill" else self._num_d_workers
        )

        if not fpm_stats:
            self._diag_load_reason = "no_fpm_data"
            return None
        if not self._reconcile_fpm_worker_count(fpm_stats, num_workers, component):
            self._diag_load_reason = "worker_count_mismatch"
            return None

        easy = self._config.optimization_target != "sla"
        if easy:
            desired = (
                self._prefill_easy_decision(fpm_stats, num_workers)
                if component == "prefill"
                else self._decode_easy_decision(fpm_stats, num_workers)
            )
        else:
            desired = (
                self._prefill_load_decision(fpm_stats, num_workers)
                if component == "prefill"
                else self._decode_load_decision(fpm_stats, num_workers)
            )
        if desired is None:
            return None

        original_desired = desired
        if self._config.enable_throughput_scaling:
            bound = (
                self._throughput_lower_bound_p
                if component == "prefill"
                else self._throughput_lower_bound_d
            )
            desired = max(desired, bound)

        desired = self._apply_single_budget(desired, component)

        if desired < num_workers:
            if desired > original_desired:
                self._diag_load_reason = "scale_down_capped_by_throughput"
            else:
                self._diag_load_reason = "scale_down"
        elif desired > num_workers:
            self._diag_load_reason = "scale_up"
        else:
            self._diag_load_reason = "no_change"

        return (
            ScalingDecision(num_prefill=desired)
            if component == "prefill"
            else ScalingDecision(num_decode=desired)
        )

    def _advance_load_disagg(self, obs: FpmObservations) -> Optional[ScalingDecision]:
        p_stats, d_stats = obs.prefill, obs.decode

        if not p_stats and not d_stats:
            logger.warning("No FPM data for either prefill or decode, skipping")
            self._diag_load_reason = "no_fpm_data"
            self._diag_load_reason_prefill = "no_fpm_data"
            self._diag_load_reason_decode = "no_fpm_data"
            return None
        if p_stats and not self._reconcile_fpm_worker_count(
            p_stats, self._num_p_workers, "prefill"
        ):
            self._diag_load_reason = "worker_count_mismatch"
            self._diag_load_reason_prefill = "worker_count_mismatch"
            self._diag_load_reason_decode = "worker_count_mismatch"
            return None
        if d_stats and not self._reconcile_fpm_worker_count(
            d_stats, self._num_d_workers, "decode"
        ):
            self._diag_load_reason = "worker_count_mismatch"
            self._diag_load_reason_prefill = "worker_count_mismatch"
            self._diag_load_reason_decode = "worker_count_mismatch"
            return None

        easy = self._config.optimization_target != "sla"

        # Sub-decisions may set self._diag_load_reason to an informative
        # value (e.g. "insufficient_data") before returning None. The
        # per-component aggregation below only emits {scale_up,
        # scale_down, scale_down_capped_by_throughput, no_change}, which
        # would silently overwrite them. Isolate each component's
        # contribution so both can be restored in the no-scaling-needed
        # branch; otherwise sequential sub-decision calls would clobber
        # each other on the shared field.
        p_reason: Optional[str] = None
        p_desired: Optional[int] = None
        if p_stats:
            self._diag_load_reason = None
            p_desired = (
                self._prefill_easy_decision(p_stats, self._num_p_workers)
                if easy
                else self._prefill_load_decision(p_stats, self._num_p_workers)
            )
            p_reason = self._diag_load_reason

        d_reason: Optional[str] = None
        d_desired: Optional[int] = None
        if d_stats:
            self._diag_load_reason = None
            d_desired = (
                self._decode_easy_decision(d_stats, self._num_d_workers)
                if easy
                else self._decode_load_decision(d_stats, self._num_d_workers)
            )
            d_reason = self._diag_load_reason

        final_p = p_desired if p_desired is not None else self._num_p_workers
        final_d = d_desired if d_desired is not None else self._num_d_workers

        # Enforce bounds first so "no change" comparison is against the
        # post-floor target, not the raw load decision.  Otherwise a load
        # decision of "no change" would skip the floor and let replicas
        # stay below a throughput-scaling lower bound that was raised on
        # a previous (or same) tick.
        original_p, original_d = final_p, final_d
        # Apply throughput floor first and track the post-floor value so we
        # can attribute later lifts to their real source -- throughput
        # capping is a distinct diagnostic from min_endpoint / global-budget
        # lifts, which should not be labelled "scale_down_capped_by_throughput".
        if self._config.enable_throughput_scaling:
            final_p = max(final_p, self._throughput_lower_bound_p)
            final_d = max(final_d, self._throughput_lower_bound_d)
        post_floor_p, post_floor_d = final_p, final_d

        final_p = max(final_p, self._config.min_endpoint)
        final_d = max(final_d, self._config.min_endpoint)
        final_p, final_d = self._apply_global_budget(final_p, final_d)

        # Per-component reasons
        def _reason(final: int, original: int, post_floor: int, current: int) -> str:
            # Only classify as throughput-capped when the throughput floor
            # itself lifted the load decision; later min_endpoint / budget
            # adjustments don't count.
            floor_capped = post_floor > original and original < current
            if final > current:
                return "scale_up"
            if final < current:
                return (
                    "scale_down_capped_by_throughput" if floor_capped else "scale_down"
                )
            return "scale_down_capped_by_throughput" if floor_capped else "no_change"

        self._diag_load_reason_prefill = _reason(
            final_p, original_p, post_floor_p, self._num_p_workers
        )
        self._diag_load_reason_decode = _reason(
            final_d, original_d, post_floor_d, self._num_d_workers
        )

        # Aggregate reason: prioritise "most interesting" across components.
        _PRIORITY = {
            "scale_up": 4,
            "scale_down_capped_by_throughput": 3,
            "scale_down": 2,
            "no_change": 1,
        }
        self._diag_load_reason = max(
            (self._diag_load_reason_prefill, self._diag_load_reason_decode),
            key=lambda r: _PRIORITY.get(r or "", 0),
        )

        if final_p == self._num_p_workers and final_d == self._num_d_workers:
            logger.info("Load-based scaling: no scaling needed")
            # Restore per-component sub-decision reasons that the
            # aggregation step overwrote with "no_change", so operators
            # can tell which side is stalled (e.g. prefill
            # insufficient_data while decode is fine).
            if p_reason is not None and p_reason != "no_change":
                self._diag_load_reason_prefill = p_reason
            if d_reason is not None and d_reason != "no_change":
                self._diag_load_reason_decode = d_reason
            # Aggregate reason: surface the most informative of the two
            # so the non-per-component Enum/HTML view also reflects it.
            for candidate in (p_reason, d_reason):
                if candidate is not None and candidate != "no_change":
                    self._diag_load_reason = candidate
                    break
            return None

        logger.info(
            f"Load-based disagg scaling: prefill {self._num_p_workers}->{final_p}, "
            f"decode {self._num_d_workers}->{final_d}"
        )
        return ScalingDecision(num_prefill=final_p, num_decode=final_d)

    def _advance_load_agg(self, obs: FpmObservations) -> Optional[ScalingDecision]:
        fpm_stats = obs.decode
        if not fpm_stats:
            self._diag_load_reason = "no_fpm_data"
            return None
        num_workers = self._num_d_workers

        if self._scaling_in_progress("decode"):
            logger.info(
                f"Scaling in progress ({num_workers} -> {self._expected_num_d}), observing only"
            )
            self._diag_load_reason = "scaling_in_progress"
            return None
        if not self._reconcile_fpm_worker_count(fpm_stats, num_workers, "agg"):
            self._diag_load_reason = "worker_count_mismatch"
            return None

        easy = self._config.optimization_target != "sla"
        if easy:
            desired = self._agg_easy_decision(fpm_stats, num_workers)
            # For agg easy mode, we directly get a single decision
            # _agg_easy_decision already sets _diag_load_reason before returning None
            if desired is None:
                return None

            original_desired = desired
            desired = max(desired, self._config.min_endpoint)
            if self._config.enable_throughput_scaling:
                desired = max(desired, self._throughput_lower_bound_d)
            desired = self._apply_single_budget(desired, "decode")

            if desired < num_workers:
                if desired > original_desired:
                    self._diag_load_reason = "scale_down_capped_by_throughput"
                else:
                    self._diag_load_reason = "scale_down"
            elif desired > num_workers:
                self._diag_load_reason = "scale_up"
            else:
                self._diag_load_reason = "no_change"

            logger.info(f"Agg easy-mode scaling: {num_workers} -> {desired}")
            return ScalingDecision(num_decode=desired)

        if not self._agg_regression.has_sufficient_data():
            logger.info(
                f"Agg regression: insufficient data "
                f"({self._agg_regression.num_observations}/{self._agg_regression.min_observations})"
            )
            self._diag_load_reason = "insufficient_data"
            return None

        d_caps = self._capabilities.decode
        max_tokens = d_caps.max_num_batched_tokens if d_caps else None
        if not max_tokens or max_tokens <= 0:
            logger.warning(
                "max_num_batched_tokens not available, skipping agg prefill scaling"
            )
            p_desired = None
        else:
            p_desired = self._agg_prefill_scaling(fpm_stats, num_workers, max_tokens)
        # Capture the prefill sub-decision's reason before _agg_decode_scaling
        # potentially overwrites it. ``scale_down_refused_consolidation`` from
        # either side must survive into the dispatcher's final diagnostic stamp.
        p_refused = self._diag_load_reason == "scale_down_refused_consolidation"
        d_desired = self._agg_decode_scaling(fpm_stats, num_workers)
        d_refused = self._diag_load_reason == "scale_down_refused_consolidation"
        any_refused = p_refused or d_refused

        logger.info(
            f"Agg scaling decisions: prefill={p_desired}, decode={d_desired} (current={num_workers})"
        )

        if p_desired is not None and p_desired > num_workers:
            desired = p_desired
        elif d_desired is not None and d_desired > num_workers:
            desired = d_desired
        elif p_desired is None and d_desired is not None and d_desired < num_workers:
            # Prefill signal unavailable: allow decode-only scale-down.
            desired = d_desired
        elif (
            p_desired is not None
            and p_desired < num_workers
            and d_desired is not None
            and d_desired < num_workers
        ):
            desired = max(p_desired, d_desired)
        else:
            # Load scaling sees "no change" -- but the throughput floor may
            # still require scaling up, so keep processing rather than
            # returning early.
            desired = num_workers

        original_desired = desired
        desired = max(desired, self._config.min_endpoint)
        if self._config.enable_throughput_scaling:
            desired = max(desired, self._throughput_lower_bound_d)
        desired = self._apply_single_budget(desired, "decode")

        # Preserve "load wanted to scale down but floor lifted it" as a
        # distinct diagnostic reason even when the net result is no change.
        floor_capped = desired > original_desired and original_desired < num_workers

        if desired == num_workers:
            logger.info("Agg scaling: no scaling needed")
            if any_refused and not floor_capped:
                # A sub-decision actively vetoed scale-down on consolidation
                # safety grounds; surface that distinct from "no_change".
                self._diag_load_reason = "scale_down_refused_consolidation"
            else:
                self._diag_load_reason = (
                    "scale_down_capped_by_throughput" if floor_capped else "no_change"
                )
            return None

        if desired < num_workers:
            self._diag_load_reason = (
                "scale_down_capped_by_throughput" if floor_capped else "scale_down"
            )
        else:  # desired > num_workers (equality returned above)
            self._diag_load_reason = "scale_up"

        logger.info(f"Agg load-based scaling: {num_workers} -> {desired}")
        return ScalingDecision(num_decode=desired)

    # ------------------------------------------------------------------
    # Per-engine latency estimation
    # ------------------------------------------------------------------

    def _prefill_load_decision(
        self, fpm_stats: dict[tuple[str, int], ForwardPassMetrics], num_workers: int
    ) -> Optional[int]:
        if not self._prefill_regression.has_sufficient_data():
            logger.info(
                f"TTFT regression: insufficient data "
                f"({self._prefill_regression.num_observations}/{self._prefill_regression.min_observations})"
            )
            self._diag_load_reason = "insufficient_data"
            return None
        if num_workers == 0:
            self._diag_load_reason = "insufficient_data"
            return None

        p_caps = self._capabilities.prefill
        max_tokens = p_caps.max_num_batched_tokens if p_caps else None
        if not max_tokens or max_tokens <= 0:
            logger.warning(
                "max_num_batched_tokens not available, skipping prefill load scaling"
            )
            self._diag_load_reason = "insufficient_data"
            return None

        kv_hit_rate = self._last_kv_hit_rate
        sensitivity = self._config.load_scaling_down_sensitivity / 100.0
        consolidation = num_workers / (num_workers - 1) if num_workers > 1 else 1.0

        # Own-compute baseline TTFT (queue=0): the new request's avg_isl chunks
        # alone. This portion does not change with consolidation, so we apply
        # sensitivity only to the queue-side budget ``SLA - T_own``.
        t_own_s: Optional[float] = None
        if num_workers > 1:
            t_own_s = self._prefill_regression.estimate_next_ttft(
                queued_prefill_tokens=0,
                max_num_batched_tokens=max_tokens,
                kv_hit_rate=kv_hit_rate,
            )

        estimates: list[float] = []
        # Consolidation-aware scale-down: re-predict TTFT with the queue
        # scaled by N/(N-1) and check the *queue-induced* portion against
        # ``(SLA - T_own) * sensitivity``. The new request's own forward-pass
        # time (``T_own``) does not shrink with more workers, so it is
        # excluded from the safety-margin budget.
        can_scale_down = num_workers > 1 and t_own_s is not None
        consolidation_refused = False
        if can_scale_down:
            t_own_ms = t_own_s * 1000  # type: ignore[operator]
            queue_budget_ms = (self._config.ttft_ms - t_own_ms) * sensitivity
            if queue_budget_ms <= 0:
                # Own compute alone already exceeds SLA -- never safe to lose
                # a worker (would only make queue contention worse).
                can_scale_down = False
                consolidation_refused = True
        for (wid, dp), fpm in fpm_stats.items():
            queued = fpm.queued_requests.sum_prefill_tokens
            est = self._prefill_regression.estimate_next_ttft(
                queued_prefill_tokens=queued,
                max_num_batched_tokens=max_tokens,
                kv_hit_rate=kv_hit_rate,
            )
            if est is not None:
                est_ms = est * 1000
                estimates.append(est_ms)
                logger.info(
                    f"Prefill engine {wid}:dp{dp}: estimated TTFT {est_ms:.2f}ms "
                    f"(queued={queued}, "
                    f"avg_isl={self._prefill_regression.avg_isl:.1f}, "
                    f"kv_hit_rate={kv_hit_rate if kv_hit_rate is not None else 'n/a'})"
                )

            if can_scale_down:
                post_queued = int(queued * consolidation)
                post_est = self._prefill_regression.estimate_next_ttft(
                    queued_prefill_tokens=post_queued,
                    max_num_batched_tokens=max_tokens,
                    kv_hit_rate=kv_hit_rate,
                )
                if post_est is None:
                    can_scale_down = False
                else:
                    queue_induced_ms = post_est * 1000 - t_own_ms
                    if queue_induced_ms >= queue_budget_ms:
                        can_scale_down = False
                        consolidation_refused = True

        if estimates:
            self._diag_estimated_ttft_ms = max(estimates)

        decision = self._scale_decision(
            estimates,
            self._config.ttft_ms,
            num_workers,
            "prefill TTFT",
            can_scale_down=can_scale_down,
        )
        if decision is None and consolidation_refused:
            self._diag_load_reason = "scale_down_refused_consolidation"
        return decision

    def _decode_load_decision(
        self, fpm_stats: dict[tuple[str, int], ForwardPassMetrics], num_workers: int
    ) -> Optional[int]:
        if not self._decode_regression.has_sufficient_data():
            logger.info(
                f"ITL regression: insufficient data "
                f"({self._decode_regression.num_observations}/{self._decode_regression.min_observations})"
            )
            self._diag_load_reason = "insufficient_data"
            return None
        if num_workers == 0:
            self._diag_load_reason = "insufficient_data"
            return None

        sensitivity = self._config.load_scaling_down_sensitivity / 100.0
        consolidation = num_workers / (num_workers - 1) if num_workers > 1 else 1.0
        d_caps = self._capabilities.decode
        max_kv = d_caps.max_kv_tokens if d_caps else None

        estimates: list[float] = []
        # Consolidation-aware scale-down. Two safety checks per worker:
        #  1. Hard cache-feasibility: post-consolidation KV must fit within
        #     ``max_kv_tokens``. Exceeding the cache forces request queueing
        #     / block eviction, a non-linear regime the regression cannot
        #     model, so refuse outright when crossed.
        #  2. SLA check: predicted ITL at the survivor's post-consolidation KV
        #     must stay within ``SLA * sensitivity``. Decouples from cache
        #     size -- engines often saturate latency well before cache.
        can_scale_down = num_workers > 1
        consolidation_refused = False
        for (wid, dp), fpm in fpm_stats.items():
            sched_kv = fpm.scheduled_requests.sum_decode_kv_tokens
            queued_kv = fpm.queued_requests.sum_decode_kv_tokens
            est = self._decode_regression.estimate_next_itl(
                scheduled_decode_kv=sched_kv,
                queued_decode_kv=queued_kv,
            )
            if est is not None:
                est_ms = est * 1000
                estimates.append(est_ms)
                logger.info(
                    f"Decode engine {wid}:dp{dp}: estimated ITL {est_ms:.2f}ms "
                    f"(sched_kv={sched_kv}, queued_kv={queued_kv})"
                )

            if can_scale_down:
                post_sched_kv = int((sched_kv + queued_kv) * consolidation)
                # (1) cache feasibility
                if max_kv is not None and max_kv > 0 and post_sched_kv >= max_kv:
                    can_scale_down = False
                    consolidation_refused = True
                    continue
                # (2) SLA check via regression at post-consolidation kv
                post_itl = self._decode_regression.estimate_next_itl(
                    scheduled_decode_kv=post_sched_kv,
                    queued_decode_kv=0,
                )
                if post_itl is None:
                    can_scale_down = False
                elif post_itl * 1000 >= self._config.itl_ms * sensitivity:
                    can_scale_down = False
                    consolidation_refused = True

        if estimates:
            self._diag_estimated_itl_ms = max(estimates)

        decision = self._scale_decision(
            estimates,
            self._config.itl_ms,
            num_workers,
            "decode ITL",
            can_scale_down=can_scale_down,
        )
        if decision is None and consolidation_refused:
            self._diag_load_reason = "scale_down_refused_consolidation"
        return decision

    def _agg_prefill_scaling(
        self,
        fpm_stats: dict[tuple[str, int], ForwardPassMetrics],
        num_workers: int,
        max_tokens: int,
    ) -> Optional[int]:
        kv_hit_rate = self._last_kv_hit_rate
        sensitivity = self._config.load_scaling_down_sensitivity / 100.0
        consolidation = num_workers / (num_workers - 1) if num_workers > 1 else 1.0

        estimates: list[float] = []
        # Agg-prefill consolidation: queued prefill and decode KV are both
        # absorbed by the survivor (scale both by N/(N-1)). Sensitivity is
        # applied only to the queue-induced portion of TTFT -- ``T_own`` (a
        # zero-queue prefill at the post-consolidation decode_kv) is treated
        # as the fixed cost the new request must pay regardless of N.
        can_scale_down = num_workers > 1
        consolidation_refused = False
        for fpm in fpm_stats.values():
            queued = fpm.queued_requests.sum_prefill_tokens
            sched_decode_kv = fpm.scheduled_requests.sum_decode_kv_tokens
            queued_decode_kv = fpm.queued_requests.sum_decode_kv_tokens
            # Pre-consolidation prediction uses scheduled-only because the
            # regression's training feature is sum_decode_kv_tokens (in the
            # scheduled batch). Queued decode is not part of training space.
            est = self._agg_regression.estimate_next_ttft(
                queued_prefill_tokens=queued,
                max_num_batched_tokens=max_tokens,
                current_decode_kv=sched_decode_kv,
                kv_hit_rate=kv_hit_rate,
            )
            if est is not None:
                estimates.append(est * 1000)

            if can_scale_down:
                post_queued = int(queued * consolidation)
                # Post-consolidation: survivors inherit both the killed
                # worker's scheduled AND its queued decode KV (queued
                # migrates and eventually schedules). Use the combined
                # backlog for the steady-state prediction so we don't
                # under-estimate the survivor's prefill TTFT.
                post_decode_kv = int(
                    (sched_decode_kv + queued_decode_kv) * consolidation
                )
                t_own_post = self._agg_regression.estimate_next_ttft(
                    queued_prefill_tokens=0,
                    max_num_batched_tokens=max_tokens,
                    current_decode_kv=post_decode_kv,
                    kv_hit_rate=kv_hit_rate,
                )
                post_est = self._agg_regression.estimate_next_ttft(
                    queued_prefill_tokens=post_queued,
                    max_num_batched_tokens=max_tokens,
                    current_decode_kv=post_decode_kv,
                    kv_hit_rate=kv_hit_rate,
                )
                if t_own_post is None or post_est is None:
                    can_scale_down = False
                else:
                    t_own_ms = t_own_post * 1000
                    queue_budget_ms = (self._config.ttft_ms - t_own_ms) * sensitivity
                    queue_induced_ms = post_est * 1000 - t_own_ms
                    if queue_budget_ms <= 0 or queue_induced_ms >= queue_budget_ms:
                        can_scale_down = False
                        consolidation_refused = True

        if estimates:
            self._diag_estimated_ttft_ms = max(estimates)

        decision = self._scale_decision(
            estimates,
            self._config.ttft_ms,
            num_workers,
            "agg TTFT",
            can_scale_down=can_scale_down,
        )
        if decision is None and consolidation_refused:
            self._diag_load_reason = "scale_down_refused_consolidation"
            # Return num_workers (not None) so ``_advance_load_agg`` does NOT
            # mistake the safety refusal for a missing prefill signal and grant
            # decode-only scale-down via its line 327 fallback.
            return num_workers
        return decision

    def _agg_decode_scaling(
        self,
        fpm_stats: dict[tuple[str, int], ForwardPassMetrics],
        num_workers: int,
    ) -> Optional[int]:
        sensitivity = self._config.load_scaling_down_sensitivity / 100.0
        consolidation = num_workers / (num_workers - 1) if num_workers > 1 else 1.0
        d_caps = self._capabilities.decode
        max_kv = d_caps.max_kv_tokens if d_caps else None

        estimates: list[float] = []
        # Agg-decode consolidation. Combined cache pressure (sched_decode_kv
        # + queued_decode_kv + queued_prefill_tokens, since queued prefill
        # eventually becomes decode KV) scaled by N/(N-1). Two safety checks:
        #  1. Hard cache-feasibility against ``max_kv_tokens`` (regression
        #     can't model block eviction past the cache).
        #  2. SLA check via ``agg_regression.estimate_next_itl`` at the
        #     post-consolidation combined kv.
        can_scale_down = num_workers > 1
        consolidation_refused = False
        for fpm in fpm_stats.values():
            sched_kv = fpm.scheduled_requests.sum_decode_kv_tokens
            queued_kv = fpm.queued_requests.sum_decode_kv_tokens
            queued_prefill = fpm.queued_requests.sum_prefill_tokens
            est = self._agg_regression.estimate_next_itl(
                scheduled_decode_kv=sched_kv,
                queued_decode_kv=queued_kv,
            )
            if est is not None:
                estimates.append(est * 1000)

            if can_scale_down:
                post_combined_kv = int(
                    (sched_kv + queued_kv + queued_prefill) * consolidation
                )
                # (1) cache feasibility
                if max_kv is not None and max_kv > 0 and post_combined_kv >= max_kv:
                    can_scale_down = False
                    consolidation_refused = True
                    continue
                # (2) SLA check via regression at post-consolidation combined kv
                post_itl = self._agg_regression.estimate_next_itl(
                    scheduled_decode_kv=post_combined_kv,
                    queued_decode_kv=0,
                )
                if post_itl is None:
                    can_scale_down = False
                elif post_itl * 1000 >= self._config.itl_ms * sensitivity:
                    can_scale_down = False
                    consolidation_refused = True

        if estimates:
            self._diag_estimated_itl_ms = max(estimates)

        decision = self._scale_decision(
            estimates,
            self._config.itl_ms,
            num_workers,
            "agg ITL",
            can_scale_down=can_scale_down,
        )
        if decision is None and consolidation_refused:
            self._diag_load_reason = "scale_down_refused_consolidation"
            # Return num_workers (not None) so ``_advance_load_agg`` does NOT
            # mistake the safety refusal for a missing decode signal in its
            # combine logic.
            return num_workers
        return decision

    # ------------------------------------------------------------------
    # Easy-mode decision methods (optimization_target != "sla")
    # ------------------------------------------------------------------

    def _prefill_easy_decision(
        self, fpm_stats: dict[tuple[str, int], ForwardPassMetrics], num_workers: int
    ) -> Optional[int]:
        if num_workers == 0:
            self._diag_load_reason = "insufficient_data"
            return None

        target = self._config.optimization_target
        is_load = target == "load"
        is_latency = target == "latency"

        values: list[float] = []
        if is_load:
            up_thresh = self._config.prefill_scale_up_queue_tokens
            down_thresh = self._config.prefill_scale_down_queue_tokens
            if up_thresh is None or down_thresh is None:
                logger.warning(
                    "prefill load thresholds not available, skipping prefill scaling"
                )
                self._diag_load_reason = "insufficient_data"
                return None
            for (wid, dp), fpm in fpm_stats.items():
                queued = fpm.queued_requests.sum_prefill_tokens
                values.append(float(queued))
                logger.info(
                    f"Load prefill {wid}:dp{dp}: queued={queued}, "
                    f"scale_up_tokens={up_thresh}, scale_down_tokens={down_thresh}"
                )
        else:
            p_caps = self._capabilities.prefill
            ctx_len = p_caps.context_length if p_caps else None
            if not ctx_len or ctx_len <= 0:
                logger.warning(
                    "context_length not available, skipping easy prefill scaling"
                )
                self._diag_load_reason = "insufficient_data"
                return None

            up_thresh = (
                _PREFILL_LATENCY_SCALE_UP
                if is_latency
                else _PREFILL_THROUGHPUT_SCALE_UP
            )
            down_thresh = (
                _PREFILL_LATENCY_SCALE_DOWN
                if is_latency
                else _PREFILL_THROUGHPUT_SCALE_DOWN
            )

            for (wid, dp), fpm in fpm_stats.items():
                queued = fpm.queued_requests.sum_prefill_tokens
                ratio = queued / ctx_len
                values.append(ratio)
                logger.info(
                    f"Easy prefill {wid}:dp{dp}: queued={queued}, "
                    f"context_length={ctx_len}, ratio={ratio:.3f}"
                )

        if not values:
            self._diag_load_reason = "insufficient_data"
            return None

        # Scale up if ANY engine above threshold
        if any(v >= up_thresh for v in values):
            logger.info(
                f"Easy prefill: engine(s) above scale-up threshold "
                f"({up_thresh}), scaling up to {num_workers + 1}"
            )
            return num_workers + 1

        # Scale down if ALL engines below threshold
        if num_workers > 1:
            if is_load:
                if all(v <= down_thresh for v in values):
                    desired = max(num_workers - 1, self._config.min_endpoint)
                    logger.info(
                        f"Load prefill: all engines at or below scale-down "
                        f"threshold ({down_thresh}), -> {desired}"
                    )
                    return desired
            elif is_latency:
                # For latency mode, scale down when ALL queues are empty
                if all(v <= down_thresh for v in values):
                    desired = max(num_workers - 1, self._config.min_endpoint)
                    logger.info(
                        f"Easy prefill: all engines at zero queue, -> {desired}"
                    )
                    return desired
            else:
                if all(v < down_thresh for v in values):
                    desired = max(num_workers - 1, self._config.min_endpoint)
                    logger.info(
                        f"Easy prefill: all engines below scale-down threshold "
                        f"({down_thresh}), -> {desired}"
                    )
                    return desired

        self._diag_load_reason = "no_change"
        return None

    def _decode_easy_decision(
        self, fpm_stats: dict[tuple[str, int], ForwardPassMetrics], num_workers: int
    ) -> Optional[int]:
        d_caps = self._capabilities.decode
        max_kv = d_caps.max_kv_tokens if d_caps else None
        if not max_kv or max_kv <= 0:
            logger.warning("max_kv_tokens not available, skipping easy decode scaling")
            self._diag_load_reason = "insufficient_data"
            return None
        if num_workers == 0:
            self._diag_load_reason = "insufficient_data"
            return None

        target = self._config.optimization_target
        is_load = target == "load"
        is_latency = target == "latency"
        if is_load:
            if (
                self._config.decode_scale_up_kv_rate is None
                or self._config.decode_scale_down_kv_rate is None
            ):
                logger.warning(
                    "decode load thresholds not available, skipping decode scaling"
                )
                self._diag_load_reason = "insufficient_data"
                return None
            up_thresh = self._config.decode_scale_up_kv_rate / 100.0
            down_thresh = self._config.decode_scale_down_kv_rate / 100.0
        else:
            up_thresh = (
                _DECODE_LATENCY_SCALE_UP if is_latency else _DECODE_THROUGHPUT_SCALE_UP
            )
            down_thresh = (
                _DECODE_LATENCY_SCALE_DOWN
                if is_latency
                else _DECODE_THROUGHPUT_SCALE_DOWN
            )

        utils: list[float] = []
        for (wid, dp), fpm in fpm_stats.items():
            sched_kv = fpm.scheduled_requests.sum_decode_kv_tokens
            queued_kv = fpm.queued_requests.sum_decode_kv_tokens
            util = (sched_kv + queued_kv) / max_kv
            utils.append(util)
            logger.info(
                f"Easy decode {wid}:dp{dp}: sched_kv={sched_kv}, "
                f"queued_kv={queued_kv}, max_kv={max_kv}, util={util:.3f}"
            )

        if not utils:
            self._diag_load_reason = "insufficient_data"
            return None

        if any(u >= up_thresh if is_load else u > up_thresh for u in utils):
            logger.info(
                f"Easy decode: engine(s) above scale-up threshold "
                f"({up_thresh}), scaling up to {num_workers + 1}"
            )
            return num_workers + 1

        scale_down = (
            all(u <= down_thresh for u in utils)
            if is_load
            else all(u < down_thresh for u in utils)
        )
        if num_workers > 1 and scale_down:
            desired = max(num_workers - 1, self._config.min_endpoint)
            logger.info(
                f"Easy decode: all engines below scale-down threshold "
                f"({down_thresh}), -> {desired}"
            )
            return desired

        self._diag_load_reason = "no_change"
        return None

    def _agg_easy_decision(
        self, fpm_stats: dict[tuple[str, int], ForwardPassMetrics], num_workers: int
    ) -> Optional[int]:
        """Easy-mode decision for agg: uses combined KV utilization including queued prefill."""
        d_caps = self._capabilities.decode
        max_kv = d_caps.max_kv_tokens if d_caps else None
        if not max_kv or max_kv <= 0:
            logger.warning("max_kv_tokens not available, skipping easy agg scaling")
            self._diag_load_reason = "insufficient_data"
            return None
        if num_workers == 0:
            self._diag_load_reason = "insufficient_data"
            return None

        target = self._config.optimization_target
        is_load = target == "load"
        is_latency = target == "latency"
        if is_load:
            if (
                self._config.decode_scale_up_kv_rate is None
                or self._config.decode_scale_down_kv_rate is None
            ):
                logger.warning(
                    "decode load thresholds not available, skipping agg scaling"
                )
                self._diag_load_reason = "insufficient_data"
                return None
            up_thresh = self._config.decode_scale_up_kv_rate / 100.0
            down_thresh = self._config.decode_scale_down_kv_rate / 100.0
        else:
            up_thresh = (
                _DECODE_LATENCY_SCALE_UP if is_latency else _DECODE_THROUGHPUT_SCALE_UP
            )
            down_thresh = (
                _DECODE_LATENCY_SCALE_DOWN
                if is_latency
                else _DECODE_THROUGHPUT_SCALE_DOWN
            )

        utils: list[float] = []
        for (wid, dp), fpm in fpm_stats.items():
            sched_kv = fpm.scheduled_requests.sum_decode_kv_tokens
            queued_kv = fpm.queued_requests.sum_decode_kv_tokens
            queued_prefill = fpm.queued_requests.sum_prefill_tokens
            util = (sched_kv + queued_kv + queued_prefill) / max_kv
            utils.append(util)
            logger.info(
                f"Easy agg {wid}:dp{dp}: sched_kv={sched_kv}, queued_kv={queued_kv}, "
                f"queued_prefill={queued_prefill}, max_kv={max_kv}, util={util:.3f}"
            )

        if not utils:
            self._diag_load_reason = "insufficient_data"
            return None

        if any(u >= up_thresh if is_load else u > up_thresh for u in utils):
            logger.info(
                f"Easy agg: engine(s) above scale-up threshold "
                f"({up_thresh}), scaling up to {num_workers + 1}"
            )
            return num_workers + 1

        scale_down = (
            all(u <= down_thresh for u in utils)
            if is_load
            else all(u < down_thresh for u in utils)
        )
        if num_workers > 1 and scale_down:
            desired = max(num_workers - 1, self._config.min_endpoint)
            logger.info(
                f"Easy agg: all engines below scale-down threshold "
                f"({down_thresh}), -> {desired}"
            )
            return desired

        self._diag_load_reason = "no_change"
        return None

    # ------------------------------------------------------------------
    # SLA-based per-engine latency estimation
    # ------------------------------------------------------------------

    def _scale_decision(
        self,
        estimates: list[float],
        sla: float,
        num_workers: int,
        label: str,
        *,
        can_scale_down: bool,
    ) -> Optional[int]:
        """Combine scale-up (latency > SLA on every worker) with the
        caller-precomputed ``can_scale_down`` signal.

        Scale-down checks live in the per-component decision functions because
        they are consolidation-aware:

        - prefill / agg-prefill: re-predict TTFT with ``queued * N/(N-1)``
          (and ``current_decode_kv * N/(N-1)`` for agg) so the new request's
          own compute time is not double-counted.
        - decode / agg-decode: per-worker KV utilisation * ``N/(N-1)`` against
          ``sensitivity`` (i.e. fraction of full cache).

        Both refuse the scale-down when the predicted post-survival metric
        exceeds ``sensitivity``-fraction of capacity, fixing the 2->1
        oscillation where flat-fraction thresholds let scale-down through but
        the survivor immediately violates SLA.
        """
        if not estimates:
            self._diag_load_reason = "insufficient_data"
            return None

        logger.info(
            f"Load-based {label}: workers={num_workers}, sla={sla:.1f}ms, "
            f"estimates={[f'{t:.1f}' for t in estimates]}"
        )

        if all(t > sla for t in estimates):
            logger.info(
                f"Load-based {label}: ALL above SLA, scaling up to {num_workers + 1}"
            )
            return num_workers + 1

        if num_workers > 1 and can_scale_down:
            desired = max(num_workers - 1, self._config.min_endpoint)
            logger.info(
                f"Load-based {label}: post-consolidation prediction within "
                f"sensitivity, -> {desired}"
            )
            return desired

        self._diag_load_reason = "no_change"
        return None
