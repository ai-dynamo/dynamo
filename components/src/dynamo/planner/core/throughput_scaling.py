# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="attr-defined"

"""Throughput-based scaling logic (Prometheus traffic-driven, predictive).

Mixin consumed by ``PlannerStateMachine``.  All methods access state
via ``self._config``, ``self._capabilities``, and regression models.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from dynamo.planner.core.types import ScalingDecision, TrafficObservation

logger = logging.getLogger(__name__)


class ThroughputScalingMixin:
    """Traffic-driven throughput-based scaling decisions."""

    # Scratch fields owned by PlannerStateMachine, declared here for mypy
    _diag_predicted_num_req: Optional[float]
    _diag_predicted_isl: Optional[float]
    _diag_predicted_osl: Optional[float]
    _diag_engine_rps_prefill: Optional[float]
    _diag_engine_rps_decode: Optional[float]
    _diag_throughput_reason: Optional[str]

    def _advance_throughput(
        self, traffic: TrafficObservation
    ) -> Optional[ScalingDecision]:
        if not self._config.enable_throughput_scaling:
            self._diag_throughput_reason = "disabled"
            return None

        next_num_req, next_isl, next_osl = self._predict_load()
        if next_num_req is None or next_isl is None or next_osl is None:
            return None

        if traffic.duration_s <= 0:
            logger.warning("Traffic observation has non-positive duration, skipping")
            self._diag_throughput_reason = "no_traffic_data"
            return None
        demand_rps = next_num_req / traffic.duration_s
        mode = self._config.mode

        if mode == "agg":
            return self._throughput_agg(demand_rps, next_isl, next_osl)
        if mode == "disagg":
            return self._throughput_disagg(demand_rps, next_isl, next_osl)
        return self._throughput_single(demand_rps, next_isl, next_osl, mode)

    def _predict_load(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        try:
            nr = self._num_req_predictor.predict_next()
            isl = self._isl_predictor.predict_next()
            osl = self._osl_predictor.predict_next()
            logger.info(
                f"Predicted load: num_req={nr:.2f}, isl={isl:.2f}, osl={osl:.2f}"
            )
            self._diag_predicted_num_req = nr
            self._diag_predicted_isl = isl
            self._diag_predicted_osl = osl
            return nr, isl, osl
        except Exception as e:
            logger.error(f"Failed to predict load: {e}")
            self._diag_throughput_reason = "predict_failed"
            return None, None, None

    def _throughput_single(
        self, demand_rps: float, isl: float, osl: float, component: str
    ) -> Optional[ScalingDecision]:
        desired = (
            self._compute_prefill_replicas(demand_rps, isl, osl)
            if component == "prefill"
            else self._compute_decode_replicas(demand_rps, isl, osl)
        )
        if desired is None:
            return None

        if self._config.enable_load_scaling:
            if component == "prefill":
                self._throughput_lower_bound_p = desired
            else:
                self._throughput_lower_bound_d = desired
            logger.info(f"Throughput lower bound set to {desired} for {component}")
            self._diag_throughput_reason = "set_lower_bound"
            return None

        desired = self._apply_single_budget(desired, component)
        self._diag_throughput_reason = "scale"
        return (
            ScalingDecision(num_prefill=desired)
            if component == "prefill"
            else ScalingDecision(num_decode=desired)
        )

    def _throughput_disagg(
        self, demand_rps: float, isl: float, osl: float
    ) -> Optional[ScalingDecision]:
        num_p = self._compute_prefill_replicas(demand_rps, isl, osl)
        num_d = self._compute_decode_replicas(demand_rps, isl, osl)
        if num_p is None or num_d is None:
            return None

        if self._config.enable_load_scaling:
            self._throughput_lower_bound_p = num_p
            self._throughput_lower_bound_d = num_d
            logger.info(f"Throughput lower bounds set: prefill={num_p}, decode={num_d}")
            self._diag_throughput_reason = "set_lower_bound"
            return None

        num_p, num_d = self._apply_global_budget(num_p, num_d)
        self._diag_throughput_reason = "scale"
        return ScalingDecision(num_prefill=num_p, num_decode=num_d)

    def _throughput_agg(
        self, demand_rps: float, isl: float, osl: float
    ) -> Optional[ScalingDecision]:
        d_caps = self._capabilities.decode
        max_tokens = d_caps.max_num_batched_tokens if d_caps else None
        if not max_tokens or max_tokens <= 0:
            logger.warning(
                "max_num_batched_tokens not available, skipping agg throughput"
            )
            self._diag_throughput_reason = "model_not_ready"
            return None

        (
            engine_rps,
            actual_ttft,
            actual_itl,
        ) = self._agg_regression.find_best_engine_agg_rps(
            isl=isl,
            osl=osl,
            max_num_batched_tokens=max_tokens,
            ttft_sla=self._config.ttft,
            itl_sla=self._config.itl,
            max_kv_tokens=d_caps.max_kv_tokens if d_caps else None,
            max_num_seqs=d_caps.max_num_seqs if d_caps else None,
        )
        if engine_rps <= 0:
            logger.warning("Agg perf model not ready, skipping throughput scaling")
            self._diag_throughput_reason = "model_not_ready"
            return None
        if actual_ttft > self._config.ttft or actual_itl > self._config.itl:
            logger.warning(
                f"Agg SLA not fully met: TTFT={actual_ttft:.1f}ms, ITL={actual_itl:.1f}ms"
            )

        self._diag_engine_rps_prefill = engine_rps
        self._diag_engine_rps_decode = engine_rps

        desired = max(math.ceil(demand_rps / engine_rps), self._config.min_endpoint)
        logger.info(
            f"Agg: {demand_rps:.2f} rps / {engine_rps:.2f} engine_rps = {desired} replicas"
        )

        if self._config.enable_load_scaling:
            self._throughput_lower_bound_d = desired
            logger.info(f"Agg throughput lower bound set to {desired}")
            self._diag_throughput_reason = "set_lower_bound"
            return None

        desired = self._apply_single_budget(desired, "decode")
        self._diag_throughput_reason = "scale"
        return ScalingDecision(num_decode=desired)

    def _compute_prefill_replicas(
        self, demand_rps: float, isl: float, osl: float
    ) -> Optional[int]:
        p_caps = self._capabilities.prefill
        engine_rps, ttft_ms = self._prefill_regression.find_best_engine_prefill_rps(
            ttft_sla=self._config.ttft,
            isl=isl,
            max_num_batched_tokens=p_caps.max_num_batched_tokens if p_caps else None,
        )
        if engine_rps <= 0:
            logger.warning("Prefill perf model not ready, skipping throughput scaling")
            self._diag_throughput_reason = "model_not_ready"
            return None
        if ttft_ms > self._config.ttft:
            logger.warning(
                f"Prefill TTFT SLA not met: {ttft_ms:.1f}ms > {self._config.ttft:.1f}ms"
            )

        self._diag_engine_rps_prefill = engine_rps

        result = max(math.ceil(demand_rps / engine_rps), self._config.min_endpoint)
        logger.info(
            f"Prefill: {demand_rps:.2f} rps / {engine_rps:.2f} = {result}, est_ttft={ttft_ms:.1f}ms"
        )
        return result

    def _compute_decode_replicas(
        self, demand_rps: float, isl: float, osl: float
    ) -> Optional[int]:
        d_caps = self._capabilities.decode
        engine_rps, itl_ms = self._decode_regression.find_best_engine_decode_rps(
            itl=self._config.itl,
            context_length=isl + osl / 2,
            osl=osl,
            max_kv_tokens=d_caps.max_kv_tokens if d_caps else None,
            max_num_seqs=d_caps.max_num_seqs if d_caps else None,
        )
        if engine_rps <= 0:
            logger.warning("Decode perf model not ready, skipping throughput scaling")
            self._diag_throughput_reason = "model_not_ready"
            return None
        if itl_ms > self._config.itl:
            logger.warning(
                f"Decode ITL SLA not met: {itl_ms:.1f}ms > {self._config.itl:.1f}ms"
            )

        self._diag_engine_rps_decode = engine_rps

        result = max(math.ceil(demand_rps / engine_rps), self._config.min_endpoint)
        logger.info(
            f"Decode: {demand_rps:.2f} rps / {engine_rps:.2f} = {result}, est_itl={itl_ms:.1f}ms"
        )
        return result
