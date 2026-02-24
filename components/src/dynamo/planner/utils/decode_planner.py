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
        """Load-based scaling decision for decode. Returns desired_replicas or None."""
        if not self.itl_regression.has_sufficient_data():
            logger.info(
                f"ITL regression: insufficient data ({self.itl_regression.num_observations}"
                f"/{self.itl_regression.min_observations}), skipping load-based scaling"
            )
            return None

        x_sla = self.itl_regression.predict_x_from_sla(self.config.itl)
        if x_sla is None:
            return None

        if x_sla <= 0:
            logger.warning(
                f"ITL SLA unachievable: x_sla={x_sla:.1f}, "
                "skipping load-based decode scaling"
            )
            return None

        if not self.cached_load_metrics.recent:
            return None

        recent = self.cached_load_metrics.recent

        num_workers = self.shared_state.num_d_workers
        if num_workers == 0:
            return None

        logger.info(
            f"Load-based decode: x_sla={x_sla:.1f}, workers={num_workers}, "
            f"slope={self.itl_regression.slope:.6f}, intercept={self.itl_regression.intercept:.3f}"
        )

        # Scale up: ALL workers above target (use recent metrics)
        all_above = all(
            m.get("active_decode_blocks", 0.0) > x_sla for m in recent.values()
        )
        if all_above:
            logger.info(
                f"Load-based decode: ALL workers above target ({x_sla:.1f}), "
                f"scaling up to {num_workers + 1}"
            )
            return num_workers + 1

        # Scale down: ALL workers below boundary (use recent metrics)
        if num_workers > 1:
            sensitivity = self.config.load_scaling_down_sensitivity / 100.0
            boundary = x_sla * (num_workers - 1) / num_workers * sensitivity
            all_below = all(
                m.get("active_decode_blocks", 0.0) < boundary for m in recent.values()
            )
            if all_below:
                logger.info(
                    f"Load-based decode: ALL workers below boundary ({boundary:.1f}), "
                    f"scaling down to {num_workers - 1}"
                )
                return num_workers - 1

        return None

    def _update_correction_factor(self) -> bool:
        if self.shared_state.num_d_workers == 0:
            logger.warning(
                "No decode workers found for correction factor, skipping correction update"
            )
            return True
        expect_itl = self.decode_interpolator.interpolate_itl(
            concurrency=self.last_metrics.num_req  # type: ignore
            / self.shared_state.num_d_workers
            * self.last_metrics.request_duration  # type: ignore
            / self.config.throughput_adjustment_interval,
            context_length=self.last_metrics.isl + self.last_metrics.osl / 2,  # type: ignore
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
