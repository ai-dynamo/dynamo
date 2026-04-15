# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefill engine performance model.

Regression:  wall_time = f(sum_prefill_tokens)
"""

import logging
import math
from typing import Optional

import numpy as np

from dynamo.common.forward_pass_metrics import ForwardPassMetrics
from dynamo.planner.core.perf_model.base import _BaseRegressionModel, _MovingAverage

logger = logging.getLogger(__name__)


class PrefillRegressionModel(_BaseRegressionModel):
    """Predict per-iteration wall time from scheduled prefill tokens.

    Simulation:  estimate TTFT by chunking queued_prefill_tokens + avg_isl
                 into max_num_batched_tokens-sized iterations and summing
                 the predicted wall time for each.
    """

    def __init__(
        self,
        max_num_fpm_samples: int,
        min_observations: int = 5,
        bucket_count: int = 16,
    ):
        super().__init__(
            max_num_fpm_samples, min_observations, ndim=1, bucket_count=bucket_count
        )
        self._avg_isl = _MovingAverage(max_num_fpm_samples)
        self._avg_num_prefill = _MovingAverage(max_num_fpm_samples)

    def _extract_x(self, fpm: ForwardPassMetrics) -> float:
        return float(fpm.scheduled_requests.sum_prefill_tokens)

    def _update_moving_averages(self, fpm: ForwardPassMetrics) -> None:
        sched = fpm.scheduled_requests
        if sched.num_prefill_requests > 0:
            self._avg_isl.add(sched.sum_prefill_tokens / sched.num_prefill_requests)
        self._avg_num_prefill.add(float(sched.num_prefill_requests))

    @property
    def avg_isl(self) -> float:
        return self._avg_isl.value

    def _predict_wall_time(self, prefill_tokens: float) -> float:
        return max(1e-6, float(self._model.predict(np.array([[prefill_tokens]]))[0]))

    def estimate_next_ttft(
        self,
        queued_prefill_tokens: int,
        max_num_batched_tokens: int,
    ) -> Optional[float]:
        """Simulate prefill scheduling to estimate TTFT for the next request.

        Returns estimated TTFT in seconds, or None if the model is not ready.
        """
        if not self._ensure_fitted() or max_num_batched_tokens <= 0:
            return None

        total_tokens = queued_prefill_tokens + self._avg_isl.value
        if total_tokens <= 0:
            return 0.0

        num_iterations = math.ceil(total_tokens / max_num_batched_tokens)
        total_time = 0.0
        remaining = total_tokens
        for _ in range(num_iterations):
            chunk = min(remaining, max_num_batched_tokens)
            total_time += self._predict_wall_time(chunk)
            remaining -= chunk
        return total_time

    def find_best_engine_prefill_rps(
        self, ttft_sla: float, isl: float
    ) -> tuple[float, float]:
        """Find prefill engine request rate under a TTFT target.

        Predicts wall_time for a single prefill at the given ISL.
        If the predicted TTFT exceeds the SLA, logs a warning but
        still returns the best achievable rate so the caller can
        scale based on load matching.

        Returns:
            (engine_rps, actual_ttft_ms) -- 0 rps signals an error
            (model not fitted or invalid input); positive rps is
            the best achievable rate with the predicted TTFT.
        """
        if not self._ensure_fitted() or isl <= 0:
            return (0.0, 0.0)
        wt = self._predict_wall_time(isl)
        actual_ttft_ms = wt * 1000.0
        engine_rps = 1.0 / wt
        if actual_ttft_ms > ttft_sla:
            logger.warning(
                f"TTFT SLA unreachable: predicted {actual_ttft_ms:.1f}ms "
                f"> target {ttft_sla:.1f}ms at ISL={isl:.0f}"
            )
        return (engine_rps, actual_ttft_ms)
