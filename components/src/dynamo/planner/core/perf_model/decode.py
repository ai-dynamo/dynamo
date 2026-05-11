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
    """Predict per-iteration wall time from decode batch composition.

    Features: ``[num_decode_requests, sum_decode_kv_tokens]``.  The
    ``sum_decode_kv_tokens`` feature dominates wall time via attention
    compute, while ``num_decode_requests`` has a weaker secondary effect
    from linear-layer work.  Under multicollinearity (both features scale
    with batch size), the ``num_decode_requests`` coefficient can flip
    sign under noisy fits; we accept the small negative value since
    ``sum_decode_kv_tokens`` keeps the overall prediction monotone.
    """

    # num_decode_requests (index 0) is relaxable; sum_decode_kv_tokens (index 1)
    # must remain non-negative.
    _relaxable_feature_indices = frozenset({0})

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

    @property
    def intercept_seconds(self) -> Optional[float]:
        """Regression intercept (no-load wall_time per iteration, in seconds).

        Returns ``None`` if the model has not been fitted yet. The intercept
        represents fixed per-iter overhead (kernel launches, framework
        bookkeeping) that does not scale with batch composition. It is the
        physical lower bound on ITL -- the wall_time prediction approaches
        this value as ``num_req -> 0`` and ``kv -> 0``.

        Used by the rate-bound consolidation predictor: at steady state the
        post-survival ITL diverges as the system approaches the rate-bound
        capacity ``ITL = N * intercept``, beyond which one fewer worker
        cannot sustain the offered request rate.
        """
        if not self._is_fitted:
            return None
        return float(self._model.intercept_)

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

    def estimate_post_consolidation_itl(
        self,
        itl_curr: float,
        num_workers: int,
    ) -> Optional[float]:
        """Estimate steady-state ITL on the survivor after scaling N -> N-1.

        Closed-form Little's-Law solution for the post-consolidation ITL.
        Derivation:

        The regression decomposes ITL into a fixed cost (intercept) plus a
        load-dependent variable cost::

            itl_curr = intercept + V_curr

        where ``V_curr = c_req * num_req_curr + c_kv * kv_curr`` is the
        variable portion (load * regression slopes).

        At steady state, Little's Law ties per-worker concurrency to the
        product of arrival rate and time-in-system. Time-in-system for a
        decode token is ~ITL, so the variable load scales with both:

          1. ``N/(N-1)``: arrival rate per worker after losing a worker
             (cluster offered load is invariant; per-worker share grows).
          2. ``itl_post / itl_curr``: longer ITL means each request lingers
             longer in the batch, further inflating concurrency / kv.

        Hence the variable cost on the survivor at the new steady state is::

            V_post = (itl_post / itl_curr) * (N / (N-1)) * V_curr
                   = (itl_post / itl_curr) * (N / (N-1)) * (itl_curr - intercept)

        and ``itl_post = intercept + V_post``. Substituting and solving the
        linear fixed point in ``itl_post``::

            itl_post = (N-1) * intercept * itl_curr / (N * intercept - itl_curr)

        Equivalently::

            itl_post = itl_curr * (1 + (itl_curr - intercept) /
                                   (N * intercept - itl_curr))

        **Saturation**: the denominator ``N * intercept - itl_curr`` goes to
        zero as ``itl_curr -> N * intercept``. Past this point one fewer
        worker physically cannot sustain the offered request rate (the
        survivor would need to process tokens faster than its intercept
        permits). Returns ``+inf`` in that regime so callers refuse
        scale-down.

        Args:
            itl_curr: Current per-worker ITL in seconds.
            num_workers: Current worker count (must be >= 2 to consolidate).

        Returns:
            Predicted post-consolidation ITL in seconds; ``+inf`` if the
            system is rate-bound (infeasible to lose a worker);
            ``None`` if the regression is not fitted or the intercept is
            non-positive (a noisy fit we can't trust for this projection).
        """
        if not self._ensure_fitted() or num_workers < 2:
            return None
        intercept = self.intercept_seconds
        if intercept is None or intercept <= 0:
            # A non-positive intercept means the fit was dominated by noise
            # or extrapolation past training data -- the rate-bound formula
            # would divide by something meaningless. Caller should fall back
            # to a direct ``estimate_next_itl`` at scaled inputs instead.
            return None
        if itl_curr <= intercept:
            # Below the fixed cost floor (regression noise) -- treat survivor
            # as also at the floor; nothing to amplify.
            return intercept
        denom = num_workers * intercept - itl_curr
        if denom <= 0:
            # Rate-bound: even with infinite cache, one fewer worker cannot
            # keep up with the offered request rate.
            return float("inf")
        return (num_workers - 1) * intercept * itl_curr / denom

    def find_best_engine_decode_rps(
        self,
        itl: float,
        context_length: float,
        osl: float,
        max_kv_tokens: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
    ) -> tuple[float, float]:
        """Find the maximum decode engine request rate within an ITL target.

        Binary searches over batch_size at the given context_length for the
        maximum batch_size where predicted wall_time * 1000 <= itl.  If even
        batch_size=1 violates the target, warns but returns the best
        achievable rate at batch_size=1 so the caller can still scale.

        Request rate is derived via Little's law:
        ``engine_rps = best_batch_size / (osl * wall_time_per_iter)``.

        The upper bound of the sweep is the smallest of:
          - ``max_kv_tokens / context_length`` -- KV cache capacity
          - ``max_num_seqs`` -- engine concurrency limit
        Falls back to ``_max_observed_kv / context_length`` (or 256) if
        neither capability is provided.

        Returns:
            (engine_rps, actual_itl_ms) -- 0 rps signals an error
            (model not fitted or invalid input); positive rps is
            the best achievable rate with the predicted ITL.
        """
        if not self._ensure_fitted() or context_length <= 0 or osl <= 0 or itl <= 0:
            return (0.0, 0.0)

        if max_kv_tokens and max_kv_tokens > 0:
            kv_cap = max(1, int(max_kv_tokens / context_length))
        elif self._max_observed_kv > 0:
            kv_cap = max(1, int(self._max_observed_kv / context_length))
        else:
            kv_cap = 256
        seq_cap = max_num_seqs if max_num_seqs and max_num_seqs > 0 else kv_cap
        max_batch = max(1, min(kv_cap, seq_cap))
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
