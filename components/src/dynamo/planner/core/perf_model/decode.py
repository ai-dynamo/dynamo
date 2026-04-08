# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode engine performance model.

Regression:  wall_time = f(num_decode_requests, sum_decode_kv_tokens)
"""

import logging
from typing import Optional

import numpy as np

from dynamo.common.forward_pass_metrics import ForwardPassMetrics
from dynamo.planner.core.perf_model.base import _BaseRegressionModel, _MovingAverage

logger = logging.getLogger(__name__)


class DecodeRegressionModel(_BaseRegressionModel):
    """Predict per-iteration wall time from decode batch composition."""

    def __init__(
        self,
        max_num_fpm_samples: int,
        min_observations: int = 5,
        bucket_count: int = 16,
    ):
        super().__init__(
            max_num_fpm_samples, min_observations, ndim=2, bucket_count=bucket_count
        )
        self._avg_decode_len = _MovingAverage(max_num_fpm_samples)
        self._avg_num_decode = _MovingAverage(max_num_fpm_samples)
        self._max_observed_kv: float = 0.0

    def _extract_x(self, fpm: ForwardPassMetrics) -> list[float]:
        sched = fpm.scheduled_requests
        return [float(sched.num_decode_requests), float(sched.sum_decode_kv_tokens)]

    def _update_moving_averages(self, fpm: ForwardPassMetrics) -> None:
        sched = fpm.scheduled_requests
        if sched.num_decode_requests > 0:
            self._avg_decode_len.add(
                sched.sum_decode_kv_tokens / sched.num_decode_requests
            )
        self._avg_num_decode.add(float(sched.num_decode_requests))
        if sched.sum_decode_kv_tokens > self._max_observed_kv:
            self._max_observed_kv = float(sched.sum_decode_kv_tokens)

    @property
    def avg_decode_length(self) -> float:
        return self._avg_decode_len.value

    def _predict_2d(self, num_requests: float, kv_tokens: float) -> float:
        return max(
            1e-6, float(self._model.predict(np.array([[num_requests, kv_tokens]]))[0])
        )

    def estimate_next_itl(
        self,
        scheduled_decode_kv: int,
        queued_decode_kv: int,
    ) -> Optional[float]:
        """Estimate the next decode iteration time in seconds."""
        if not self._ensure_fitted():
            return None
        total_kv = scheduled_decode_kv + queued_decode_kv + self._avg_decode_len.value
        num_req = self._avg_num_decode.value + 1
        return self._predict_2d(num_req, total_kv)

    def find_best_engine_decode_rps(
        self, itl: float, context_length: float, osl: float
    ) -> tuple[float, float]:
        """Find the maximum decode engine request rate within an ITL target.

        Binary searches over batch_size at the given context_length for the
        maximum batch_size where predicted wall_time * 1000 <= itl.  If even
        batch_size=1 violates the target, warns but returns the best
        achievable rate at batch_size=1 so the caller can still scale.

        Request rate is derived via Little's law:
        ``engine_rps = best_batch_size / (osl * wall_time_per_iter)``.

        Returns:
            (engine_rps, actual_itl_ms) -- 0 rps signals an error
            (model not fitted or invalid input); positive rps is
            the best achievable rate with the predicted ITL.
        """
        if not self._ensure_fitted() or context_length <= 0 or osl <= 0 or itl <= 0:
            return (0.0, 0.0)

        max_batch = (
            max(1, int(self._max_observed_kv / context_length))
            if self._max_observed_kv > 0
            else 256
        )
        lo, hi = 1, max_batch
        best_bs, best_wt = 1, self._predict_2d(1, context_length)

        if best_wt * 1000.0 > itl:
            logger.warning(
                f"ITL SLA unreachable: predicted {best_wt * 1000.0:.1f}ms "
                f"> target {itl:.1f}ms at batch_size=1, ctx_len={context_length:.0f}"
            )
            return (best_bs / (osl * best_wt), best_wt * 1000.0)

        while lo <= hi:
            mid = (lo + hi) // 2
            kv = mid * context_length
            wt = self._predict_2d(mid, kv)
            if wt * 1000.0 <= itl:
                best_bs, best_wt = mid, wt
                lo = mid + 1
            else:
                hi = mid - 1

        engine_rps = best_bs / (osl * best_wt)
        return (engine_rps, best_wt * 1000.0)
