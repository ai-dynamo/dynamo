# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diffusion-LLM performance model (LLaDA-class).

The Auto-regressive (AR) regression models (``AggRegressionModel``,
``DecodeRegressionModel``) fit ``wall_time = f(sum_prefill_tokens,
sum_decode_kv_tokens)`` because each AR scheduler step generates one
token per in-flight sequence. The per-step compute and the per-token
time are 1:1.

Diffusion LLMs (LLaDA 2.0 with the ``LowConfidence`` algorithm) commit
output in blocks (default ``page_size=32`` tokens per block) and run
``K`` denoise forward passes per block before committing. So each
request's end-to-end latency is approximately::

    e2e_time ≈ ceil(OSL/page_size) × K × step_time(per_step_tokens)

where ``per_step_tokens = per_worker_concurrency × (ISL + OSL/2)`` is
the average compute per scheduler step. The deterministic block count
(``ceil(OSL/page_size)``) is the dominant variance source — the AR
models conflate this into ``sum_decode_kv_tokens`` and consequently
mis-fit the scaling regime for diffusion workloads.

Empirically (47 aiperf runs on LLaDA 2.0 mini on RTX PRO 6000), a
two-feature linear model::

    avg_latency = α * num_blocks + β * per_step_tokens + γ

achieves mean prediction error ~7% on a 2-worker Dynamo fleet, vs the
AR baseline at 50%+ error.

This module wraps that fit so the planner can use a DLLM-specific
cost branch when the backend reports ``backend_kind='dllm'``.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from dynamo.common.forward_pass_metrics import ForwardPassMetrics
from dynamo.planner.core.perf_model.base import _BaseRegressionModel, _MovingAverage

logger = logging.getLogger(__name__)


class DllmRegressionModel(_BaseRegressionModel):
    """Cost model for diffusion-LM engines (LLaDA-class).

    Features::

        x[0] = num_blocks       = ceil(OSL / page_size)
        x[1] = per_step_tokens  = batch_size * (ISL + OSL/2)

    Target::

        y    = avg_request_latency (seconds)

    Both coefficients are required to be non-negative — more blocks or
    more compute per step monotonically increase wall time.
    """

    def __init__(
        self,
        max_num_fpm_samples: int,
        min_observations: int = 5,
        bucket_count: int = 16,
        page_size: int = 32,
    ):
        super().__init__(
            max_num_fpm_samples,
            min_observations,
            ndim=2,
            bucket_count=bucket_count,
        )
        self._page_size = page_size
        self._avg_isl = _MovingAverage(max_num_fpm_samples)
        self._avg_decode_len = _MovingAverage(max_num_fpm_samples)
        self._avg_num_decode = _MovingAverage(max_num_fpm_samples)
        self._avg_osl_hint = _MovingAverage(max_num_fpm_samples)

    def _features(self, isl: float, osl: float, batch_size: float) -> list[float]:
        num_blocks = math.ceil(osl / self._page_size) if osl > 0 else 0
        per_step_tokens = batch_size * (isl + osl / 2.0)
        return [float(num_blocks), float(per_step_tokens)]

    def _extract_x(self, fpm: ForwardPassMetrics) -> list[float]:
        sched = fpm.scheduled_requests
        # For a diffusion engine, we map AR FPM fields to our features.
        # The engine reports per-step counters; we approximate:
        #   batch_size ≈ num_decode_requests (number of sequences in-flight)
        #   isl       ≈ sum_prefill_tokens / max(1, num_prefill_requests)
        #   osl       ≈ avg_decode_length (moving average is rolled into FPM)
        isl = sched.sum_prefill_tokens / max(1, sched.num_prefill_requests)
        osl = (
            sched.sum_decode_kv_tokens / max(1, sched.num_decode_requests)
            if sched.num_decode_requests > 0
            else self._avg_decode_len.value
        )
        batch = float(sched.num_decode_requests or 1)
        return self._features(isl, osl, batch)

    def _update_moving_averages(self, fpm: ForwardPassMetrics) -> None:
        sched = fpm.scheduled_requests
        if sched.num_prefill_requests > 0:
            self._avg_isl.add(
                sched.sum_prefill_tokens / sched.num_prefill_requests
            )
        if sched.num_decode_requests > 0:
            self._avg_decode_len.add(
                sched.sum_decode_kv_tokens / sched.num_decode_requests
            )
        self._avg_num_decode.add(float(sched.num_decode_requests))

    @property
    def avg_isl(self) -> float:
        return self._avg_isl.value

    @property
    def avg_decode_length(self) -> float:
        return self._avg_decode_len.value

    @property
    def page_size(self) -> int:
        return self._page_size

    def predict_latency(
        self, isl: float, osl: float, batch_size: float
    ) -> Optional[float]:
        """Predict end-to-end request latency in seconds.

        Returns ``None`` if the model has not been fitted yet.
        """
        if not self._ensure_fitted():
            return None
        x = self._features(isl, osl, batch_size)
        return max(1e-6, float(self._model.predict(np.array([x]))[0]))

    def find_best_engine_dllm_rps(
        self,
        isl: float,
        osl: float,
        e2e_sla: float,
        max_num_seqs: Optional[int] = None,
    ) -> tuple[float, float]:
        """Find the maximum engine request rate within an end-to-end SLA.

        For LLaDA-class engines, ``e2e_sla`` is the binding latency target
        (instead of separate TTFT and ITL). The diffusion loop dominates
        latency, so prefill-vs-decode separation has limited meaning.

        Args:
            isl: target input sequence length.
            osl: target output sequence length.
            e2e_sla: per-request latency target in milliseconds.
            max_num_seqs: engine concurrency cap (``max_running_requests``).

        Returns:
            (engine_rps, achieved_latency_ms). 0 rps if model not ready.
        """
        if not self._ensure_fitted() or isl <= 0 or osl <= 0 or e2e_sla <= 0:
            return (0.0, 0.0)

        cap = max_num_seqs if max_num_seqs and max_num_seqs > 0 else 16
        best_bs = 1
        best_lat = self.predict_latency(isl, osl, 1.0) or 0.0
        if best_lat * 1000.0 > e2e_sla:
            logger.warning(
                "DLLM e2e SLA unreachable at batch_size=1: predicted "
                f"{best_lat * 1000.0:.1f}ms > target {e2e_sla:.1f}ms "
                f"(ISL={isl:.0f}, OSL={osl:.0f})"
            )
            return (1.0 / best_lat, best_lat * 1000.0)

        lo, hi = 1, cap
        while lo <= hi:
            mid = (lo + hi) // 2
            lat = self.predict_latency(isl, osl, float(mid))
            if lat is None:
                break
            if lat * 1000.0 <= e2e_sla:
                best_bs, best_lat = mid, lat
                lo = mid + 1
            else:
                hi = mid - 1

        engine_rps = best_bs / max(best_lat, 1e-6)
        return (engine_rps, best_lat * 1000.0)

    # Allow either feature to be slightly negative under numerical noise.
    # The per_step_tokens feature dominates; blocks coefficient should
    # always be positive when more output is asked for.
    _relaxable_feature_indices = frozenset({0})


def fit_from_observations(
    observations: list[tuple[int, int, float, float, float]],
    page_size: int = 32,
) -> tuple[float, float, float]:
    """Helper for offline analysis: fit (α, β, γ) on a list of
    ``(isl, osl, conc_per_worker, avg_lat_s)`` tuples.

    Returns ``(alpha_per_block, beta_per_token, gamma_intercept)``.

    Useful for pre-deployment profiling: run a small sweep, fit the
    LLaDA model offline, then load the coefficients into the planner
    (``load_benchmark_fpms`` equivalent for the DLLM path).
    """
    X = []
    y = []
    for isl, osl, pwc, _rps, lat in observations:
        num_blocks = math.ceil(osl / page_size)
        per_step_tokens = pwc * (isl + osl / 2.0)
        X.append([float(num_blocks), float(per_step_tokens)])
        y.append(lat)
    reg = LinearRegression()
    reg.fit(np.array(X), np.array(y))
    return (
        float(reg.coef_[0]),
        float(reg.coef_[1]),
        float(reg.intercept_),
    )
