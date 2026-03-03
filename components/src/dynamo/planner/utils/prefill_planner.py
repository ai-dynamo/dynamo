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


class PrefillPlanner(BasePlanner):
    component_type = SubComponentType.PREFILL

    def load_plan_adjustment(self) -> Optional[int]:
        """Load-based scaling decision for prefill. Returns desired_replicas or None."""
        if not self.ttft_regression.has_sufficient_data():
            logger.info(
                f"TTFT regression: insufficient data ({self.ttft_regression.num_observations}"
                f"/{self.ttft_regression.min_observations}), skipping load-based scaling"
            )
            return None

        x_sla = self.ttft_regression.predict_x_from_sla(self.config.ttft)
        if x_sla is None:
            return None

        if not self.cached_load_metrics.recent:
            return None

        recent = self.cached_load_metrics.recent
        cluster_averaged = self.cached_load_metrics.cluster_averaged

        # Averaged ISL across all workers in the past adjustment interval
        avg_isl = cluster_averaged.get("last_isl", 0.0)
        target_active_tokens = x_sla - avg_isl

        if target_active_tokens <= 0:
            logger.warning(
                f"TTFT SLA unachievable at current ISL: x_sla={x_sla:.1f}, "
                f"avg_isl={avg_isl:.1f}, skipping load-based prefill scaling"
            )
            return None

        num_workers = self.shared_state.num_p_workers
        if num_workers == 0:
            return None

        logger.info(
            f"Load-based prefill: x_sla={x_sla:.1f}, avg_isl={avg_isl:.1f}, "
            f"target_active_tokens={target_active_tokens:.1f}, workers={num_workers}, "
            f"slope={self.ttft_regression.slope:.6f}, intercept={self.ttft_regression.intercept:.3f}"
        )

        # Scale up: ALL workers above target (use recent metrics)
        all_above = all(
            m.get("active_prefill_tokens", 0.0) > target_active_tokens
            for m in recent.values()
        )
        if all_above:
            logger.info(
                f"Load-based prefill: ALL workers above target ({target_active_tokens:.1f}), "
                f"scaling up to {num_workers + 1}"
            )
            return num_workers + 1

        # Scale down: ALL workers below boundary (use recent metrics)
        if num_workers > 1:
            sensitivity = self.config.load_scaling_down_sensitivity / 100.0
            boundary = (
                target_active_tokens * (num_workers - 1) / num_workers * sensitivity
            )
            all_below = all(
                m.get("active_prefill_tokens", 0.0) < boundary for m in recent.values()
            )
            if all_below:
                logger.info(
                    f"Load-based prefill: ALL workers below boundary ({boundary:.1f}), "
                    f"scaling down to {num_workers - 1}"
                )
                return num_workers - 1

        return None

    def _update_correction_factor(self) -> bool:
        expect_ttft = self.prefill_interpolator.interpolate_ttft(self.last_metrics.isl)
        self.p_correction_factor = self.last_metrics.ttft / expect_ttft
        logger.info(f"Correction factor (prefill TTFT): {self.p_correction_factor:.3f}")
        if self.prometheus_port != 0 and self.prometheus_metrics is not None:
            self.prometheus_metrics.p_correction_factor.set(self.p_correction_factor)
        return True

    def _compute_replica_requirements(
        self, next_num_req: float, next_isl: float, next_osl: float
    ) -> int:
        pred_prefill_throughput = (
            next_num_req
            * next_isl
            / self.config.throughput_adjustment_interval
            * min(1, self.p_correction_factor)
        )
        p_thpt_per_gpu = self.prefill_interpolator.interpolate_thpt_per_gpu(next_isl)
        if p_thpt_per_gpu <= 0:
            logger.warning(
                f"p_thpt_per_gpu is {p_thpt_per_gpu} "
                "(no throughput satisfies TTFT target), falling back to min_endpoint"
            )
            return self.config.min_endpoint
        next_num_p = math.ceil(
            pred_prefill_throughput
            / p_thpt_per_gpu
            / self.config.prefill_engine_num_gpu
        )
        next_num_p = max(next_num_p, self.config.min_endpoint)
        logger.info(
            f"Prefill calculation: {pred_prefill_throughput:.2f}(p_thpt) / "
            f"{p_thpt_per_gpu * self.config.prefill_engine_num_gpu:.2f}(p_engine_cap) = "
            f"{next_num_p}(num_p)"
        )
        return next_num_p

    def update_predicted_replicas_metric(self, desired_replicas: int) -> None:
        if self.prometheus_port != 0 and self.prometheus_metrics is not None:
            self.prometheus_metrics.predicted_num_p.set(desired_replicas)
