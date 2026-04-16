# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base regression infrastructure and utilities for FPM-based performance models.

Provides ``_BaseRegressionModel`` (bucketed observation storage with
linear regression) and ``_MovingAverage``, shared by the prefill,
decode, and agg perf model subclasses.
"""

import logging
import math
from collections import defaultdict, deque
from typing import Union

import numpy as np
from sklearn.linear_model import LinearRegression

from dynamo.common.forward_pass_metrics import ForwardPassMetrics

logger = logging.getLogger(__name__)


class _MovingAverage:
    """Fixed-window moving average that skips leading zeros.

    Initial zero values (pre-traffic idle period) are ignored until the
    first non-zero value arrives, matching the throughput planner's
    load predictor behavior.
    """

    __slots__ = ("_window", "_sum", "_seen_nonzero")

    def __init__(self, window_size: int):
        self._window: deque[float] = deque(maxlen=window_size)
        self._sum: float = 0.0
        self._seen_nonzero: bool = False

    def add(self, value: float) -> None:
        if value == 0.0 and not self._seen_nonzero:
            return
        if value != 0.0:
            self._seen_nonzero = True
        if len(self._window) == self._window.maxlen:
            self._sum -= self._window[0]
        self._window.append(value)
        self._sum += value

    @property
    def value(self) -> float:
        if not self._window:
            return 0.0
        return self._sum / len(self._window)

    def __len__(self) -> int:
        return len(self._window)


# ---------------------------------------------------------------------------
# Bucketed FPM sample retirement.
#
# FPM observations span diverse engine load conditions (from
# pre-deployment benchmarks through live traffic).  A simple FIFO
# window would let sustained traffic at one operating point push
# out data for other conditions, degrading the regression's coverage
# of the full performance surface.
#
# Instead, each input axis is divided into equal-width buckets.
# Observations are assigned to a bucket based on their input features.
# When total samples exceed max_num_fpm_samples, the oldest sample in
# the bucket with the most entries is retired.  This keeps the sample
# distribution roughly uniform across the operating range.
#
# fpm_sample_bucket_size controls the total number of buckets:
#   - 1D models: fpm_sample_bucket_size buckets along the single axis
#   - 2D models: sqrt(fpm_sample_bucket_size) buckets per axis
#     (e.g., 16 -> 4x4 grid)
# The config requires fpm_sample_bucket_size to be a perfect square
# so the 2D decomposition is always clean.
# ---------------------------------------------------------------------------


class _BaseRegressionModel:
    """Shared regression infrastructure for FPM-based models."""

    def __init__(
        self,
        max_num_fpm_samples: int,
        min_observations: int = 5,
        ndim: int = 1,
        bucket_count: int = 16,
    ):
        self.max_num_fpm_samples = max_num_fpm_samples
        self.min_observations = min_observations
        self._ndim = ndim

        if ndim == 1:
            self._buckets_per_axis = bucket_count
        else:
            self._buckets_per_axis = math.isqrt(bucket_count)

        self._buckets: dict[
            tuple[int, ...], deque[tuple[Union[float, list[float]], float]]
        ] = defaultdict(deque)
        self._total_observations = 0

        self._axis_min: list[float] = [float("inf")] * ndim
        self._axis_max: list[float] = [float("-inf")] * ndim

        self._model = LinearRegression()
        self._is_fitted = False

    def _extract_x(self, fpm: ForwardPassMetrics) -> Union[float, list[float]]:
        raise NotImplementedError

    def _update_moving_averages(self, fpm: ForwardPassMetrics) -> None:
        raise NotImplementedError

    def _to_vals(self, x: Union[float, list[float]]) -> list[float]:
        if isinstance(x, list):
            return x
        return [x]

    def _bucket_key(self, x: Union[float, list[float]]) -> tuple[int, ...]:
        """Compute the bucket index for an observation's input features."""
        vals = self._to_vals(x)
        key = []
        for i, v in enumerate(vals):
            lo, hi = self._axis_min[i], self._axis_max[i]
            if hi <= lo:
                key.append(0)
            else:
                idx = int((v - lo) / (hi - lo) * self._buckets_per_axis)
                key.append(max(0, min(idx, self._buckets_per_axis - 1)))
        return tuple(key)

    def _update_axis_bounds(self, x: Union[float, list[float]]) -> bool:
        """Update min/max per axis. Returns True if bounds changed."""
        vals = self._to_vals(x)
        changed = False
        for i, v in enumerate(vals):
            if v < self._axis_min[i]:
                self._axis_min[i] = v
                changed = True
            if v > self._axis_max[i]:
                self._axis_max[i] = v
                changed = True
        return changed

    def _rebuild_buckets(self) -> None:
        """Re-index all observations into buckets using current axis bounds."""
        all_obs = self._gather_observations()
        self._buckets.clear()
        for x, wt in all_obs:
            key = self._bucket_key(x)
            self._buckets[key].append((x, wt))

    def add_observation(self, fpm: ForwardPassMetrics) -> None:
        self._update_moving_averages(fpm)
        if fpm.wall_time == 0.0:
            return
        x = self._extract_x(fpm)
        bounds_changed = self._update_axis_bounds(x)

        if bounds_changed and self._total_observations > 0:
            self._rebuild_buckets()

        key = self._bucket_key(x)
        self._buckets[key].append((x, fpm.wall_time))
        self._total_observations += 1

        if self._total_observations > self.max_num_fpm_samples:
            fattest_key = max(self._buckets, key=lambda k: len(self._buckets[k]))
            self._buckets[fattest_key].popleft()
            self._total_observations -= 1
            if not self._buckets[fattest_key]:
                del self._buckets[fattest_key]

        self._is_fitted = False

    def load_benchmark_fpms(self, fpms: list[ForwardPassMetrics]) -> None:
        """Bootstrap regression from pre-deployment benchmark FPMs."""
        for fpm in fpms:
            self.add_observation(fpm)

    def _gather_observations(self) -> list[tuple[Union[float, list[float]], float]]:
        return [obs for bucket in self._buckets.values() for obs in bucket]

    # Feature indices whose coefficients can be clipped to 0 if negative
    # rather than rejecting the whole fit.  Used when a feature has weak
    # signal and can flip sign under multicollinearity without violating
    # physical monotonicity (e.g. num_decode_requests vs sum_decode_kv_tokens:
    # total KV dominates, so a small negative batch-size coefficient is
    # numerical noise rather than "more requests → less work").  Subclasses
    # override via ``_clippable_feature_indices``.
    _clippable_feature_indices: frozenset[int] = frozenset()

    def _fit(self) -> bool:
        observations = self._gather_observations()
        if len(observations) < self.min_observations:
            return False
        X = np.array([o[0] for o in observations])
        if self._ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array([o[1] for o in observations])
        self._model.fit(X, y)

        # Negative coefficients mean "more load → less compute time", which
        # is physically impossible.  Reject the fit so callers see the model
        # as not ready rather than making inverted scaling decisions.
        # Exception: features in ``_clippable_feature_indices`` are allowed
        # to go slightly negative under noisy fits; clip them to 0 so the
        # model stays usable.
        neg_mask = self._model.coef_ < 0
        if np.any(neg_mask):
            non_clippable_negs = [
                i
                for i in range(len(self._model.coef_))
                if neg_mask[i] and i not in self._clippable_feature_indices
            ]
            if non_clippable_negs:
                logger.warning(
                    f"Regression produced negative coefficients {self._model.coef_.tolist()}, "
                    "model rejected — scaling will be skipped until more data arrives"
                )
                self._is_fitted = False
                return False
            # Clip relaxable negative coefficients to 0
            self._model.coef_ = np.where(neg_mask, 0.0, self._model.coef_)
            logger.debug(
                f"Clipped negative coefficients at indices "
                f"{[i for i in range(len(neg_mask)) if neg_mask[i]]} to 0"
            )

        self._is_fitted = True
        return True

    def _ensure_fitted(self) -> bool:
        return self._is_fitted or self._fit()

    def has_sufficient_data(self) -> bool:
        return self._total_observations >= self.min_observations

    @property
    def num_observations(self) -> int:
        return self._total_observations
