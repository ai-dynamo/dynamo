# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FPM-driven regression models for load-based scaling.

Each model takes ForwardPassMetrics observations and estimates per-engine
TTFT or ITL by simulating the scheduler's chunked prefill / decode
iteration pipeline.

- PrefillRegressionModel: 1D regression (sum_prefill_tokens -> wall_time)
- DecodeRegressionModel:  1D regression (sum_decode_kv_tokens -> wall_time)
- AggRegressionModel:     2D regression (sum_prefill_tokens, sum_decode_kv_tokens -> wall_time)
"""

import logging
import math
from collections import deque
from typing import Optional, Union

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


class _BaseRegressionModel:
    """Shared regression infrastructure for FPM-based models."""

    def __init__(self, window_size: int, min_observations: int = 5, ndim: int = 1):
        self.window_size = window_size
        self.min_observations = min_observations
        self._ndim = ndim
        self._observations: deque[tuple[Union[float, list[float]], float]] = deque(
            maxlen=window_size
        )
        self._model = LinearRegression()
        self._is_fitted = False

    def _extract_x(self, fpm: ForwardPassMetrics) -> Union[float, list[float]]:
        """Return the regression input(s) from an FPM snapshot."""
        raise NotImplementedError

    def _update_moving_averages(self, fpm: ForwardPassMetrics) -> None:
        """Update moving averages (called for every FPM, including idle)."""
        raise NotImplementedError

    def add_observation(self, fpm: ForwardPassMetrics) -> None:
        # Always update moving averages so idle state is reflected.
        self._update_moving_averages(fpm)
        if fpm.wall_time == 0.0:
            return
        self._observations.append((self._extract_x(fpm), fpm.wall_time))
        self._is_fitted = False

    def _fit(self) -> bool:
        if len(self._observations) < self.min_observations:
            return False
        X = np.array([o[0] for o in self._observations])
        if self._ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array([o[1] for o in self._observations])
        self._model.fit(X, y)
        self._is_fitted = True
        return True

    def _ensure_fitted(self) -> bool:
        return self._is_fitted or self._fit()

    def has_sufficient_data(self) -> bool:
        return len(self._observations) >= self.min_observations

    @property
    def num_observations(self) -> int:
        return len(self._observations)


class PrefillRegressionModel(_BaseRegressionModel):
    """Predict per-iteration wall time from scheduled prefill tokens.

    Regression:  wall_time = f(sum_prefill_tokens)
    Simulation:  estimate TTFT by chunking queued_prefill_tokens + avg_isl
                 into max_num_batched_tokens-sized iterations and summing
                 the predicted wall time for each.
    """

    def __init__(self, window_size: int, min_observations: int = 5):
        super().__init__(window_size, min_observations, ndim=1)
        self._avg_isl = _MovingAverage(window_size)
        self._avg_num_prefill = _MovingAverage(window_size)

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

    def estimate_next_ttft(
        self,
        queued_prefill_tokens: int,
        max_num_batched_tokens: int,
    ) -> Optional[float]:
        """Simulate prefill scheduling to estimate TTFT for the next request.

        The scheduler processes prefill tokens in chunks of
        max_num_batched_tokens per iteration.  We sum the regression-predicted
        wall time for each chunk to approximate TTFT.

        Args:
            queued_prefill_tokens: tokens already queued ahead of the next request.
            max_num_batched_tokens: per-iteration token budget (from WorkerInfo/MDC).

        Returns:
            Estimated TTFT in seconds, or None if the model is not ready.
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
            pred = self._model.predict(np.array([[chunk]]))[0]
            total_time += max(0.0, float(pred))
            remaining -= chunk
        return total_time


class DecodeRegressionModel(_BaseRegressionModel):
    """Predict per-iteration wall time from scheduled decode KV tokens.

    Regression:  wall_time = f(sum_decode_kv_tokens)
    Estimation:  predict ITL for the next decode step accounting for
                 queued (preempted) decode load and one additional request.
    """

    def __init__(self, window_size: int, min_observations: int = 5):
        super().__init__(window_size, min_observations, ndim=1)
        self._avg_decode_len = _MovingAverage(window_size)
        self._avg_num_decode = _MovingAverage(window_size)

    def _extract_x(self, fpm: ForwardPassMetrics) -> float:
        return float(fpm.scheduled_requests.sum_decode_kv_tokens)

    def _update_moving_averages(self, fpm: ForwardPassMetrics) -> None:
        sched = fpm.scheduled_requests
        if sched.num_decode_requests > 0:
            self._avg_decode_len.add(
                sched.sum_decode_kv_tokens / sched.num_decode_requests
            )
        self._avg_num_decode.add(float(sched.num_decode_requests))

    @property
    def avg_decode_length(self) -> float:
        return self._avg_decode_len.value

    def estimate_next_itl(
        self,
        scheduled_decode_kv: int,
        queued_decode_kv: int,
    ) -> Optional[float]:
        """Estimate the next decode iteration time.

        Predicts wall time for the total decode KV load: currently scheduled +
        queued (preempted) + one additional request worth of decode context.

        Args:
            scheduled_decode_kv: sum_decode_kv_tokens from the latest FPM.
            queued_decode_kv: sum_decode_kv_tokens from the queued metrics.

        Returns:
            Estimated ITL in seconds, or None if the model is not ready.
        """
        if not self._ensure_fitted():
            return None
        total_kv = scheduled_decode_kv + queued_decode_kv + self._avg_decode_len.value
        return max(0.0, float(self._model.predict(np.array([[total_kv]]))[0]))


class AggRegressionModel(_BaseRegressionModel):
    """2D regression for aggregated (chunked prefill + decode) engines.

    Regression:  wall_time = f(sum_prefill_tokens, sum_decode_kv_tokens)
    Estimation:  estimate TTFT by simulating prefill chunking while assuming
                 steady-state decode load; estimate ITL by predicting decode
                 iteration time while assuming average piggybacked prefill load.
    """

    def __init__(self, window_size: int, min_observations: int = 5):
        super().__init__(window_size, min_observations, ndim=2)
        self._avg_isl = _MovingAverage(window_size)
        self._avg_decode_len = _MovingAverage(window_size)
        self._avg_prefill_tokens = _MovingAverage(window_size)
        self._avg_num_prefill = _MovingAverage(window_size)
        self._avg_num_decode = _MovingAverage(window_size)

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
        return float(
            self._model.predict(np.array([[prefill_tokens, decode_kv_tokens]]))[0]
        )

    def estimate_next_ttft(
        self,
        queued_prefill_tokens: int,
        max_num_batched_tokens: int,
        current_decode_kv: int,
    ) -> Optional[float]:
        """Simulate prefill scheduling with piggybacked decode.

        Same chunking simulation as PrefillRegressionModel, but each
        iteration also carries the current decode KV load (steady state).

        Args:
            queued_prefill_tokens: prefill tokens queued ahead of the next request.
            max_num_batched_tokens: per-iteration token budget (from MDC).
            current_decode_kv: scheduled decode KV tokens from the latest FPM
                (assumed steady during prefill).

        Returns:
            Estimated TTFT in seconds, or None if the model is not ready.
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
            total_time += max(0.0, self._predict_2d(chunk, float(current_decode_kv)))
            remaining -= chunk
        return total_time

    def estimate_next_itl(
        self,
        scheduled_decode_kv: int,
        queued_decode_kv: int,
    ) -> Optional[float]:
        """Estimate decode iteration time with piggybacked prefill.

        Uses the moving average of scheduled prefill tokens as the
        piggybacked prefill load in the next iteration.

        Args:
            scheduled_decode_kv: sum_decode_kv_tokens from the latest FPM.
            queued_decode_kv: sum_decode_kv_tokens from the queued metrics.

        Returns:
            Estimated ITL in seconds, or None if the model is not ready.
        """
        if not self._ensure_fitted():
            return None
        total_kv = scheduled_decode_kv + queued_decode_kv + self._avg_decode_len.value
        return max(0.0, self._predict_2d(self._avg_prefill_tokens.value, total_kv))
