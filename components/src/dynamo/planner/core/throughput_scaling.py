# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="attr-defined"

"""Throughput-based scaling logic (Prometheus traffic-driven, predictive).

Mixin consumed by ``PlannerScalingState``.  All methods access state via
``self._config``, ``self._capabilities``, and perf models.
"""

from __future__ import annotations

import logging
import math
from typing import Literal, Optional

from dynamo.planner.config.planner_config import resolve_min_endpoint
from dynamo.planner.core.types import ScalingDecision

logger = logging.getLogger(__name__)


class ThroughputScalingMixin:
    """Traffic-driven throughput-based scaling decisions."""

    # Scratch fields owned by PlannerScalingState, declared here for mypy
    _diag_predicted_num_req: Optional[float]
    _diag_predicted_isl: Optional[float]
    _diag_predicted_osl: Optional[float]
    _diag_predicted_kv_hit_rate: Optional[float]
    _diag_engine_rps_prefill: Optional[float]
    _diag_engine_rps_decode: Optional[float]
    _diag_throughput_reason: Optional[str]
    _diag_throughput_reason_prefill: Optional[str]
    _diag_throughput_reason_decode: Optional[str]

    def _cap_throughput_replicas(
        self, desired: int, current: int, component: str
    ) -> int:
        """Bound one throughput observation's replica change for a component."""
        limit = self._config.max_throughput_scaling_replicas
        bounded = min(max(desired, max(0, current - limit)), current + limit)
        if bounded != desired:
            logger.warning(
                "Throughput target capped for %s: raw=%s, current=%s, "
                "max_delta=%s, bounded=%s",
                component,
                desired,
                current,
                limit,
                bounded,
            )
        return bounded

    def _throughput_single(
        self,
        demand_rps: float,
        isl: float,
        osl: float,
        component: Literal["prefill", "decode"],
        kv_hit_rate: Optional[float] = None,
    ) -> Optional[ScalingDecision]:
        desired = (
            self._compute_prefill_replicas(demand_rps, isl, osl, kv_hit_rate)
            if component == "prefill"
            else self._compute_decode_replicas(demand_rps, isl, osl)
        )
        if desired is None:
            return None
        current = self._num_p_workers if component == "prefill" else self._num_d_workers
        desired = self._cap_throughput_replicas(desired, current, component)
        # Endpoint recovery is a hard invariant, not an ordinary throughput
        # movement, so it may exceed the per-observation delta cap.
        desired = max(desired, resolve_min_endpoint(self._config, component))
        desired, _ceiling_reason = self._fit_single_throughput_ceiling(
            desired,
            component,
        )

        if self._config.enable_load_scaling:
            if component == "prefill":
                self._throughput_lower_bound_p = desired
            else:
                self._throughput_lower_bound_d = desired
            logger.info(
                "Throughput lower bound set to %s for %s",
                desired,
                component,
            )
            self._diag_throughput_reason = (
                "gpu_budget_guard_hold"
                if _ceiling_reason == "gpu_budget_guard_hold"
                else "set_lower_bound"
            )
            return None

        desired, _budget_reason = self._apply_single_scaling_budget(
            desired,
            component,
        )
        if desired == current:
            self._diag_throughput_reason = (
                _budget_reason or _ceiling_reason or "no_change"
            )
            return None
        self._diag_throughput_reason = "scale"
        return (
            ScalingDecision(num_prefill=desired)
            if component == "prefill"
            else ScalingDecision(num_decode=desired)
        )

    def _throughput_disagg(
        self,
        demand_rps: float,
        isl: float,
        osl: float,
        kv_hit_rate: Optional[float] = None,
    ) -> Optional[ScalingDecision]:
        num_p = self._compute_prefill_replicas(demand_rps, isl, osl, kv_hit_rate)
        num_d = self._compute_decode_replicas(demand_rps, isl, osl)
        # _compute_* sets _diag_throughput_reason = "model_not_ready" when
        # the perf model cannot estimate yet. If one side is not ready, the other
        # side's computation was still valid but its decision is blocked,
        # so we label it "partner_not_ready" to keep per-component
        # diagnostics consistent with the aggregate reason.
        if num_p is None or num_d is None:
            self._diag_throughput_reason_prefill = (
                "model_not_ready" if num_p is None else "partner_not_ready"
            )
            self._diag_throughput_reason_decode = (
                "model_not_ready" if num_d is None else "partner_not_ready"
            )
            return None

        num_p = self._cap_throughput_replicas(num_p, self._num_p_workers, "prefill")
        num_d = self._cap_throughput_replicas(num_d, self._num_d_workers, "decode")
        # A runtime endpoint increase must recover immediately even when the
        # gap is larger than the throughput delta cap.
        num_p = max(num_p, resolve_min_endpoint(self._config, "prefill"))
        num_d = max(num_d, resolve_min_endpoint(self._config, "decode"))
        bounded_p, bounded_d = num_p, num_d
        num_p, num_d = self._fit_disagg_throughput_ceiling(num_p, num_d)
        budget_held = (num_p, num_d) == (self._num_p_workers, self._num_d_workers) and (
            bounded_p,
            bounded_d,
        ) != (self._num_p_workers, self._num_d_workers)

        reason = "set_lower_bound" if self._config.enable_load_scaling else "scale"
        self._diag_throughput_reason_prefill = reason
        self._diag_throughput_reason_decode = reason

        if self._config.enable_load_scaling:
            self._throughput_lower_bound_p = num_p
            self._throughput_lower_bound_d = num_d
            logger.info(f"Throughput lower bounds set: prefill={num_p}, decode={num_d}")
            if budget_held:
                self._diag_throughput_reason = "gpu_budget_guard_hold"
                self._diag_throughput_reason_prefill = "gpu_budget_guard_hold"
                self._diag_throughput_reason_decode = "gpu_budget_guard_hold"
            else:
                self._diag_throughput_reason = "set_lower_bound"
            return None

        num_p, num_d, budget_reason = self._apply_disagg_scaling_budget(
            num_p, num_d, source="throughput"
        )
        if num_p == self._num_p_workers and num_d == self._num_d_workers:
            hold_reason = budget_reason or (
                "gpu_budget_guard_hold" if budget_held else "no_change"
            )
            self._diag_throughput_reason = hold_reason
            self._diag_throughput_reason_prefill = hold_reason
            self._diag_throughput_reason_decode = hold_reason
            return None

        self._diag_throughput_reason = "scale"
        return ScalingDecision(num_prefill=num_p, num_decode=num_d)

    def _throughput_agg(
        self,
        demand_rps: float,
        isl: float,
        osl: float,
        kv_hit_rate: Optional[float] = None,
    ) -> Optional[ScalingDecision]:
        d_caps = self._capabilities.decode
        max_tokens = d_caps.max_num_batched_tokens if d_caps else None
        if not max_tokens or max_tokens <= 0:
            logger.warning(
                "max_num_batched_tokens not available, skipping agg throughput"
            )
            self._diag_throughput_reason = "model_not_ready"
            return None

        capacity = self._agg_regression.find_engine_capacity_rps(
            isl=isl,
            osl=osl,
            ttft_sla_ms=self._config.ttft_ms,
            itl_sla_ms=self._config.itl_ms,
            kv_hit_rate=kv_hit_rate,
            accept_length=self._current_decode_accept_length(),
        )
        engine_rps = capacity.rps if capacity is not None else 0.0
        if engine_rps <= 0:
            logger.warning("Agg perf model not ready, skipping throughput scaling")
            self._diag_throughput_reason = "model_not_ready"
            return None
        actual_ttft = capacity.ttft_ms or 0.0
        actual_itl = capacity.itl_ms or 0.0
        if (
            not capacity.eligible
            or actual_ttft > self._config.ttft_ms
            or actual_itl > self._config.itl_ms
        ):
            logger.warning(
                f"Agg SLA not fully met: TTFT={actual_ttft:.1f}ms, ITL={actual_itl:.1f}ms"
            )

        self._diag_engine_rps_prefill = engine_rps
        self._diag_engine_rps_decode = engine_rps

        desired = max(
            math.ceil(demand_rps / engine_rps),
            resolve_min_endpoint(self._config, "decode"),
        )
        logger.info(
            f"Agg: {demand_rps:.2f} rps / {engine_rps:.2f} engine_rps = {desired} replicas"
        )
        desired = self._cap_throughput_replicas(
            desired, self._num_d_workers, "aggregated"
        )
        desired = max(desired, resolve_min_endpoint(self._config, "decode"))
        desired, _ceiling_reason = self._fit_single_throughput_ceiling(
            desired,
            "decode",
        )

        if self._config.enable_load_scaling:
            self._throughput_lower_bound_d = desired
            logger.info("Agg throughput lower bound set to %s", desired)
            self._diag_throughput_reason = (
                "gpu_budget_guard_hold"
                if _ceiling_reason == "gpu_budget_guard_hold"
                else "set_lower_bound"
            )
            return None

        desired, _budget_reason = self._apply_single_scaling_budget(
            desired,
            "decode",
        )
        if desired == self._num_d_workers:
            self._diag_throughput_reason = (
                _budget_reason or _ceiling_reason or "no_change"
            )
            return None
        self._diag_throughput_reason = "scale"
        return ScalingDecision(num_decode=desired)

    def _compute_prefill_replicas(
        self,
        demand_rps: float,
        isl: float,
        osl: float,
        kv_hit_rate: Optional[float] = None,
    ) -> Optional[int]:
        capacity = self._prefill_regression.find_engine_capacity_rps(
            isl=isl,
            osl=osl,
            ttft_sla_ms=self._config.ttft_ms,
            kv_hit_rate=kv_hit_rate,
        )
        engine_rps = capacity.rps if capacity is not None else 0.0
        if engine_rps <= 0:
            logger.warning("Prefill perf model not ready, skipping throughput scaling")
            self._diag_throughput_reason = "model_not_ready"
            return None
        ttft_ms = capacity.ttft_ms or 0.0
        if not capacity.eligible or ttft_ms > self._config.ttft_ms:
            logger.warning(
                f"Prefill TTFT SLA not met: {ttft_ms:.1f}ms > {self._config.ttft_ms:.1f}ms"
            )

        self._diag_engine_rps_prefill = engine_rps

        result = max(
            math.ceil(demand_rps / engine_rps),
            resolve_min_endpoint(self._config, "prefill"),
        )
        logger.info(
            f"Prefill: {demand_rps:.2f} rps / {engine_rps:.2f} = {result}, "
            f"est_ttft={ttft_ms:.1f}ms, isl_raw={isl:.1f}, "
            f"kv_hit_rate={kv_hit_rate or 0.0:.3f}"
        )
        return result

    def _compute_decode_replicas(
        self, demand_rps: float, isl: float, osl: float
    ) -> Optional[int]:
        accept_length = self._current_decode_accept_length()
        capacity = self._decode_regression.find_engine_capacity_rps(
            isl=isl,
            osl=osl,
            itl_sla_ms=self._config.itl_ms,
            accept_length=accept_length,
        )
        engine_rps = capacity.rps if capacity is not None else 0.0
        if engine_rps <= 0:
            logger.warning("Decode perf model not ready, skipping throughput scaling")
            self._diag_throughput_reason = "model_not_ready"
            return None
        itl_ms = capacity.itl_ms or 0.0
        if not capacity.eligible or itl_ms > self._config.itl_ms:
            logger.warning(
                f"Decode ITL SLA not met: {itl_ms:.1f}ms > {self._config.itl_ms:.1f}ms"
            )

        self._diag_engine_rps_decode = engine_rps

        result = max(
            math.ceil(demand_rps / engine_rps),
            resolve_min_endpoint(self._config, "decode"),
        )
        logger.info(
            f"Decode: {demand_rps:.2f} rps / {engine_rps:.2f} = {result}, "
            f"est_itl={itl_ms:.1f}ms, accept_length={accept_length:.2f}"
        )
        return result
