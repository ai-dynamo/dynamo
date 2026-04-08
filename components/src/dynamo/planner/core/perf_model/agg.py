# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregated (chunked prefill + decode) engine performance model.

Regression:  wall_time = f(sum_prefill_tokens, sum_decode_kv_tokens)
"""

import logging
import math
from typing import Optional

import numpy as np

from dynamo.common.forward_pass_metrics import ForwardPassMetrics
from dynamo.planner.core.perf_model.base import _BaseRegressionModel, _MovingAverage

logger = logging.getLogger(__name__)


class AggRegressionModel(_BaseRegressionModel):
    """2D regression for aggregated (chunked prefill + decode) engines."""

    def __init__(
        self,
        max_num_fpm_samples: int,
        min_observations: int = 5,
        bucket_count: int = 16,
    ):
        super().__init__(
            max_num_fpm_samples, min_observations, ndim=2, bucket_count=bucket_count
        )
        self._avg_isl = _MovingAverage(max_num_fpm_samples)
        self._avg_decode_len = _MovingAverage(max_num_fpm_samples)
        self._avg_prefill_tokens = _MovingAverage(max_num_fpm_samples)
        self._avg_num_prefill = _MovingAverage(max_num_fpm_samples)
        self._avg_num_decode = _MovingAverage(max_num_fpm_samples)

    def _extract_x(self, fpm: ForwardPassMetrics) -> list[float]:
        sched = fpm.scheduled_requests
        return [float(sched.sum_prefill_tokens), float(sched.sum_decode_kv_tokens)]

    def _update_moving_averages(self, fpm: ForwardPassMetrics) -> None:
        sched = fpm.scheduled_requests
        if sched.num_prefill_requests > 0:
            self._avg_isl.add(sched.sum_prefill_tokens / sched.num_prefill_requests)
        if sched.num_decode_requests > 0:
            self._avg_decode_len.add(
                sched.sum_decode_kv_tokens / sched.num_decode_requests
            )
        self._avg_prefill_tokens.add(float(sched.sum_prefill_tokens))
        self._avg_num_prefill.add(float(sched.num_prefill_requests))
        self._avg_num_decode.add(float(sched.num_decode_requests))

    @property
    def avg_isl(self) -> float:
        return self._avg_isl.value

    @property
    def avg_decode_length(self) -> float:
        return self._avg_decode_len.value

    @property
    def avg_prefill_tokens(self) -> float:
        return self._avg_prefill_tokens.value

    def _predict_2d(self, prefill_tokens: float, decode_kv_tokens: float) -> float:
        return max(
            1e-6,
            float(
                self._model.predict(np.array([[prefill_tokens, decode_kv_tokens]]))[0]
            ),
        )

    def estimate_next_ttft(
        self,
        queued_prefill_tokens: int,
        max_num_batched_tokens: int,
        current_decode_kv: int,
    ) -> Optional[float]:
        """Simulate prefill scheduling with piggybacked decode.

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
            total_time += self._predict_2d(chunk, float(current_decode_kv))
            remaining -= chunk
        return total_time

    def estimate_next_itl(
        self,
        scheduled_decode_kv: int,
        queued_decode_kv: int,
    ) -> Optional[float]:
        """Estimate decode iteration time with piggybacked prefill.

        Returns estimated ITL in seconds, or None if the model is not ready.
        """
        if not self._ensure_fitted():
            return None
        total_kv = scheduled_decode_kv + queued_decode_kv + self._avg_decode_len.value
        return self._predict_2d(self._avg_prefill_tokens.value, total_kv)

    def find_best_engine_agg_rps(
        self,
        isl: float,
        osl: float,
        max_num_batched_tokens: int,
        ttft_sla: float,
        itl_sla: float,
    ) -> tuple[float, float, float]:
        """Find the maximum agg engine request rate under both SLA targets.

        Sweeps over batch_size to find the largest decode concurrency
        where both ITL and TTFT remain within their targets.  Warns if
        even batch_size=1 violates either SLA.

        Request rate is derived via Little's law:
        ``engine_rps = best_batch_size / (osl * wall_time_per_iter)``.

        Args:
            isl: average input sequence length (tokens).
            osl: average output sequence length (tokens).
            max_num_batched_tokens: per-iteration token budget.
            ttft_sla: TTFT target in milliseconds.
            itl_sla: ITL target in milliseconds.

        Returns:
            (engine_rps, actual_ttft_ms, actual_itl_ms) -- 0 rps
            signals an error (model not fitted or invalid input);
            positive rps is the best achievable rate with the
            predicted TTFT/ITL.  If SLAs are violated, a warning
            is logged but the rate is still returned.
        """
        if (
            not self._ensure_fitted()
            or isl <= 0
            or osl <= 0
            or max_num_batched_tokens <= 0
        ):
            return (0.0, 0.0, 0.0)

        avg_ctx = isl + osl / 2.0
        max_bs = max(1, int(max_num_batched_tokens / max(1, avg_ctx))) * 2

        best_rps = 0.0
        best_ttft_ms = 0.0
        best_itl_ms = 0.0

        for bs in range(1, max_bs + 1):
            decode_kv = bs * avg_ctx
            prefill_per_iter = min(bs * isl / max(1.0, osl), max_num_batched_tokens)
            wt = self._predict_2d(prefill_per_iter, decode_kv)
            itl_ms = wt * 1000.0

            est_ttft = self.estimate_next_ttft(
                queued_prefill_tokens=int(prefill_per_iter),
                max_num_batched_tokens=max_num_batched_tokens,
                current_decode_kv=int(decode_kv),
            )
            ttft_ms = est_ttft * 1000.0 if est_ttft is not None else 0.0

            if itl_ms > itl_sla or ttft_ms > ttft_sla:
                if bs == 1:
                    logger.warning(
                        f"Agg SLA unreachable at batch_size=1: "
                        f"TTFT={ttft_ms:.1f}ms (target {ttft_sla:.1f}ms), "
                        f"ITL={itl_ms:.1f}ms (target {itl_sla:.1f}ms)"
                    )
                    best_rps = 1.0 / (osl * wt)
                    best_ttft_ms = ttft_ms
                    best_itl_ms = itl_ms
                break

            best_rps = bs / (osl * wt)
            best_ttft_ms = ttft_ms
            best_itl_ms = itl_ms

        return (best_rps, best_ttft_ms, best_itl_ms)
