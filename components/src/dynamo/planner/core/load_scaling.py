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


class LoadScalingMixin:
    """FPM-driven load-based scaling decisions."""

    # Scratch fields owned by PlannerStateMachine, declared here for mypy
    _diag_estimated_ttft_ms: Optional[float]
    _diag_estimated_itl_ms: Optional[float]
    _diag_load_reason: Optional[str]

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
            return None
        if p_stats and not self._reconcile_fpm_worker_count(
            p_stats, self._num_p_workers, "prefill"
        ):
            self._diag_load_reason = "worker_count_mismatch"
            return None
        if d_stats and not self._reconcile_fpm_worker_count(
            d_stats, self._num_d_workers, "decode"
        ):
            self._diag_load_reason = "worker_count_mismatch"
            return None

        p_desired = (
            self._prefill_load_decision(p_stats, self._num_p_workers)
            if p_stats
            else None
        )
        d_desired = (
            self._decode_load_decision(d_stats, self._num_d_workers)
            if d_stats
            else None
        )

        final_p = p_desired if p_desired is not None else self._num_p_workers
        final_d = d_desired if d_desired is not None else self._num_d_workers

        if final_p == self._num_p_workers and final_d == self._num_d_workers:
            logger.info("Load-based scaling: no scaling needed")
            self._diag_load_reason = "no_change"
            return None

        original_p, original_d = final_p, final_d
        if self._config.enable_throughput_scaling:
            final_p = max(final_p, self._throughput_lower_bound_p)
            final_d = max(final_d, self._throughput_lower_bound_d)

        final_p = max(final_p, self._config.min_endpoint)
        final_d = max(final_d, self._config.min_endpoint)
        final_p, final_d = self._apply_global_budget(final_p, final_d)

        if (final_p > original_p or final_d > original_d) and (
            original_p < self._num_p_workers or original_d < self._num_d_workers
        ):
            self._diag_load_reason = "scale_down_capped_by_throughput"
        elif final_p > self._num_p_workers or final_d > self._num_d_workers:
            self._diag_load_reason = "scale_up"
        elif final_p < self._num_p_workers or final_d < self._num_d_workers:
            self._diag_load_reason = "scale_down"
        else:
            self._diag_load_reason = "no_change"

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
            logger.warning("max_num_batched_tokens not available, skipping agg scaling")
            self._diag_load_reason = "insufficient_data"
            return None

        p_desired = self._agg_prefill_scaling(fpm_stats, num_workers, max_tokens)
        d_desired = self._agg_decode_scaling(fpm_stats, num_workers)

        logger.info(
            f"Agg scaling decisions: prefill={p_desired}, decode={d_desired} (current={num_workers})"
        )

        if p_desired is not None and p_desired > num_workers:
            desired = p_desired
        elif d_desired is not None and d_desired > num_workers:
            desired = d_desired
        elif (
            p_desired is not None
            and p_desired < num_workers
            and d_desired is not None
            and d_desired < num_workers
        ):
            desired = max(p_desired, d_desired)
        else:
            logger.info("Agg scaling: no scaling needed")
            self._diag_load_reason = "no_change"
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

        estimates: list[float] = []
        for (wid, dp), fpm in fpm_stats.items():
            est = self._prefill_regression.estimate_next_ttft(
                queued_prefill_tokens=fpm.queued_requests.sum_prefill_tokens,
                max_num_batched_tokens=max_tokens,
            )
            if est is not None:
                est_ms = est * 1000
                estimates.append(est_ms)
                logger.info(
                    f"Prefill engine {wid}:dp{dp}: estimated TTFT {est_ms:.2f}ms "
                    f"(queued={fpm.queued_requests.sum_prefill_tokens}, "
                    f"avg_isl={self._prefill_regression.avg_isl:.1f})"
                )

        if estimates:
            self._diag_estimated_ttft_ms = max(estimates)

        return self._scale_decision(
            estimates, self._config.ttft, num_workers, "prefill TTFT"
        )

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

        estimates: list[float] = []
        for (wid, dp), fpm in fpm_stats.items():
            est = self._decode_regression.estimate_next_itl(
                scheduled_decode_kv=fpm.scheduled_requests.sum_decode_kv_tokens,
                queued_decode_kv=fpm.queued_requests.sum_decode_kv_tokens,
            )
            if est is not None:
                est_ms = est * 1000
                estimates.append(est_ms)
                logger.info(
                    f"Decode engine {wid}:dp{dp}: estimated ITL {est_ms:.2f}ms "
                    f"(sched_kv={fpm.scheduled_requests.sum_decode_kv_tokens}, "
                    f"queued_kv={fpm.queued_requests.sum_decode_kv_tokens})"
                )

        if estimates:
            self._diag_estimated_itl_ms = max(estimates)

        return self._scale_decision(
            estimates, self._config.itl, num_workers, "decode ITL"
        )

    def _agg_prefill_scaling(
        self,
        fpm_stats: dict[tuple[str, int], ForwardPassMetrics],
        num_workers: int,
        max_tokens: int,
    ) -> Optional[int]:
        estimates: list[float] = []
        for fpm in fpm_stats.values():
            est = self._agg_regression.estimate_next_ttft(
                queued_prefill_tokens=fpm.queued_requests.sum_prefill_tokens,
                max_num_batched_tokens=max_tokens,
                current_decode_kv=fpm.scheduled_requests.sum_decode_kv_tokens,
            )
            if est is not None:
                estimates.append(est * 1000)

        if estimates:
            self._diag_estimated_ttft_ms = max(estimates)

        return self._scale_decision(
            estimates, self._config.ttft, num_workers, "agg TTFT"
        )

    def _agg_decode_scaling(
        self,
        fpm_stats: dict[tuple[str, int], ForwardPassMetrics],
        num_workers: int,
    ) -> Optional[int]:
        estimates: list[float] = []
        for fpm in fpm_stats.values():
            est = self._agg_regression.estimate_next_itl(
                scheduled_decode_kv=fpm.scheduled_requests.sum_decode_kv_tokens,
                queued_decode_kv=fpm.queued_requests.sum_decode_kv_tokens,
            )
            if est is not None:
                estimates.append(est * 1000)

        if estimates:
            self._diag_estimated_itl_ms = max(estimates)

        return self._scale_decision(estimates, self._config.itl, num_workers, "agg ITL")

    def _scale_decision(
        self, estimates: list[float], sla: float, num_workers: int, label: str
    ) -> Optional[int]:
        if not estimates:
            self._diag_load_reason = "insufficient_data"
            return None

        sensitivity = self._config.load_scaling_down_sensitivity / 100.0
        logger.info(
            f"Load-based {label}: workers={num_workers}, sla={sla:.1f}ms, "
            f"estimates={[f'{t:.1f}' for t in estimates]}"
        )

        if all(t > sla for t in estimates):
            logger.info(
                f"Load-based {label}: ALL above SLA, scaling up to {num_workers + 1}"
            )
            return num_workers + 1

        if num_workers > 1:
            threshold = sla * sensitivity
            if all(t < threshold for t in estimates):
                desired = max(num_workers - 1, self._config.min_endpoint)
                logger.info(
                    f"Load-based {label}: ALL below threshold ({threshold:.1f}ms), -> {desired}"
                )
                return desired

        self._diag_load_reason = "no_change"
        return None
