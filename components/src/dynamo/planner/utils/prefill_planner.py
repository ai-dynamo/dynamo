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
        """Load-based scaling decision for prefill using FPM data.

        For each engine, simulates prefill scheduling to estimate next TTFT:
        - Uses queued prefill tokens + avg ISL as total tokens to process
        - Chunks into max_num_batched_tokens-sized iterations
        - Sums regression-predicted wall time per chunk

        Scale up if ALL engines' estimated TTFT > SLA.
        Scale down if ALL engines' estimated TTFT < SLA * sensitivity.
        """
        if not self.ttft_regression.has_sufficient_data():
            logger.info(
                f"TTFT regression: insufficient data ({self.ttft_regression.num_observations}"
                f"/{self.ttft_regression.min_observations}), skipping load-based scaling"
            )
            return None

        fpm_stats = self._get_fpm_stats()
        if not fpm_stats:
            return None

        num_workers = self.shared_state.num_p_workers
        if num_workers == 0:
            return None

        max_num_batched_tokens = getattr(
            self.prefill_worker_info, "max_num_batched_tokens", None
        )
        if not max_num_batched_tokens or max_num_batched_tokens <= 0:
            logger.warning(
                "max_num_batched_tokens not available from WorkerInfo, "
                "skipping prefill load-based scaling"
            )
            return None

        estimated_ttfts: list[float] = []
        for (wid, dp), fpm in fpm_stats.items():
            queued_prefill = fpm.queued_requests.sum_prefill_tokens
            est = self.ttft_regression.estimate_next_ttft(
                queued_prefill_tokens=queued_prefill,
                max_num_batched_tokens=max_num_batched_tokens,
            )
            if est is None:
                continue
            est_ms = est * 1000
            estimated_ttfts.append(est_ms)
            logger.info(
                f"Prefill engine {wid}:dp{dp}: estimated TTFT {est_ms:.2f}ms "
                f"(queued_prefill={queued_prefill}, avg_isl={self.ttft_regression.avg_isl:.1f})"
            )

        return self._load_based_scaling_decision_from_estimates(
            estimates=estimated_ttfts,
            sla=self.config.ttft,
            num_workers=num_workers,
            label="prefill TTFT",
        )

    def _update_correction_factor(self) -> bool:
        assert self.last_metrics.isl is not None and self.last_metrics.ttft is not None
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
        assert self.config.prefill_engine_num_gpu is not None
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
