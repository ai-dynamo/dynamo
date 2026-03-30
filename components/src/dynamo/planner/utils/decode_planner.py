# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from typing import Optional

from dynamo.planner import SubComponentType
from dynamo.planner.utils.planner_core import BasePlanner
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class DecodePlanner(BasePlanner):
    component_type = SubComponentType.DECODE

    def load_plan_adjustment(self) -> Optional[int]:
        """Load-based scaling decision for decode using FPM data.

        For each engine, estimates next decode ITL:
        - Uses scheduled + queued decode KV tokens + avg decode length
        - Predicts wall time via regression

        Scale up if ALL engines' estimated ITL > SLA.
        Scale down if ALL engines' estimated ITL < SLA * sensitivity.
        """
        if not self.itl_regression.has_sufficient_data():
            logger.info(
                f"ITL regression: insufficient data ({self.itl_regression.num_observations}"
                f"/{self.itl_regression.min_observations}), skipping load-based scaling"
            )
            return None

        fpm_stats = self._get_fpm_stats()
        if not fpm_stats:
            return None

        num_workers = self.shared_state.num_d_workers
        if num_workers == 0:
            return None

        estimated_itls: list[float] = []
        for (wid, dp), fpm in fpm_stats.items():
            scheduled_kv = fpm.scheduled_requests.sum_decode_kv_tokens
            queued_kv = fpm.queued_requests.sum_decode_kv_tokens
            est = self.itl_regression.estimate_next_itl(
                scheduled_decode_kv=scheduled_kv,
                queued_decode_kv=queued_kv,
            )
            if est is None:
                continue
            est_ms = est * 1000
            estimated_itls.append(est_ms)
            logger.info(
                f"Decode engine {wid}:dp{dp}: estimated ITL {est_ms:.2f}ms "
                f"(sched_kv={scheduled_kv}, queued_kv={queued_kv}, "
                f"avg_decode_len={self.itl_regression.avg_decode_length:.1f})"
            )

        return self._load_based_scaling_decision_from_estimates(
            estimates=estimated_itls,
            sla=self.config.itl,
            num_workers=num_workers,
            label="decode ITL",
        )

    def _update_correction_factor(self) -> bool:
        if self.shared_state.num_d_workers == 0:
            logger.warning(
                "No decode workers found for correction factor, skipping correction update"
            )
            return True
        assert self.last_metrics.num_req is not None
        assert self.last_metrics.request_duration is not None
        assert self.last_metrics.isl is not None
        assert self.last_metrics.osl is not None
        assert self.last_metrics.itl is not None
        expect_itl = self.decode_interpolator.interpolate_itl(
            concurrency=self.last_metrics.num_req
            / self.shared_state.num_d_workers
            * self.last_metrics.request_duration
            / self.config.throughput_adjustment_interval,
            context_length=self.last_metrics.isl + self.last_metrics.osl / 2,
        )
        self.d_correction_factor = self.last_metrics.itl / expect_itl
        logger.info(f"Correction factor (decode ITL): {self.d_correction_factor:.3f}")
        if self.prometheus_port != 0 and self.prometheus_metrics is not None:
            self.prometheus_metrics.d_correction_factor.set(self.d_correction_factor)
        return True

    def _compute_replica_requirements(
        self, next_num_req: float, next_isl: float, next_osl: float
    ) -> int:
        if self.d_correction_factor <= 0:
            logger.warning(
                f"d_correction_factor is {self.d_correction_factor}, using default value of 1.0"
            )
            corrected_itl = self.config.itl
        else:
            corrected_itl = self.config.itl / self.d_correction_factor
        (
            pred_decode_thpt_per_gpu,
            _,
            _,
        ) = self.decode_interpolator.find_best_throughput_per_gpu(
            itl=corrected_itl, context_length=next_isl + next_osl / 2
        )
        if pred_decode_thpt_per_gpu <= 0:
            logger.warning(
                f"pred_decode_thpt_per_gpu is {pred_decode_thpt_per_gpu} "
                "(no throughput satisfies ITL target), falling back to min_endpoint"
            )
            return self.config.min_endpoint
        assert self.config.decode_engine_num_gpu is not None
        pred_decode_throughput = (
            next_num_req * next_osl / self.config.throughput_adjustment_interval
        )
        next_num_d = math.ceil(
            pred_decode_throughput
            / pred_decode_thpt_per_gpu
            / self.config.decode_engine_num_gpu
        )
        next_num_d = max(next_num_d, self.config.min_endpoint)
        logger.info(
            f"Decode calculation: {pred_decode_throughput:.2f}(d_thpt) / "
            f"{pred_decode_thpt_per_gpu * self.config.decode_engine_num_gpu:.2f}(d_engine_cap) = "
            f"{next_num_d}(num_d)"
        )
        return next_num_d

    def update_predicted_replicas_metric(self, desired_replicas: int) -> None:
        if self.prometheus_port != 0 and self.prometheus_metrics is not None:
            self.prometheus_metrics.predicted_num_d.set(desired_replicas)
