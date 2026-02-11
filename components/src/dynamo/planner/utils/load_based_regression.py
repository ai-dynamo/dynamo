# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections import deque
from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class LoadBasedRegressionModel:
    """Sliding window linear regression for load-based scaling.

    Maintains a fixed-size window of (X, y) observations and provides:
    - Forward prediction: y = mx + b (given X, predict latency)
    - Reverse prediction: X = (y - b) / m (given target SLA, find max load)

    Used to map:
    - Prefill: (active_prefill_tokens + ISL) -> TTFT
    - Decode: active_decode_blocks -> ITL
    """

    def __init__(self, window_size: int, min_observations: int = 5):
        self.window_size = window_size
        self.min_observations = min_observations
        self._observations: deque = deque(maxlen=window_size)
        self._model = LinearRegression()
        self._is_fitted = False

    def add_observation(self, x: float, y: float) -> None:
        """Add an (X, y) observation to the sliding window."""
        self._observations.append((x, y))
        self._is_fitted = False

    def fit(self) -> bool:
        """Fit the linear regression model on current observations.

        Returns:
            True if fitting succeeded, False if insufficient data.
        """
        if len(self._observations) < self.min_observations:
            return False
        X = np.array([obs[0] for obs in self._observations]).reshape(-1, 1)
        y = np.array([obs[1] for obs in self._observations])
        self._model.fit(X, y)
        self._is_fitted = True
        return True

    def predict_x_from_sla(self, target_y: float) -> Optional[float]:
        """Reverse prediction: given a target latency (SLA), find the max load.

        Solves: x = (y - b) / m

        Safety guards:
        - Returns None if insufficient data (cold start)
        - Falls back to observation-based heuristic if slope <= 0
        - Clamps result to non-negative

        Args:
            target_y: Target latency SLA value (e.g., TTFT in ms, ITL in ms)

        Returns:
            Maximum load value that satisfies the SLA, or None if insufficient data.
        """
        if not self._is_fitted and not self.fit():
            return None

        coef = float(self._model.coef_[0])
        intercept = float(self._model.intercept_)

        if coef <= 0:
            logger.warning(
                f"Regression slope is non-positive ({coef:.6f}), "
                "falling back to observation-based heuristic"
            )
            return self._fallback_x_from_observations(target_y)

        x_sla = (target_y - intercept) / coef
        return max(0.0, x_sla)

    def _fallback_x_from_observations(self, target_y: float) -> float:
        """Fallback when regression slope is non-positive.

        Returns the minimum x among observations where y < target_y.
        If all observations have y >= target_y, returns the smallest x overall.
        """
        below = [(x, y) for x, y in self._observations if y < target_y]
        if below:
            result = min(x for x, _ in below)
        else:
            result = min(x for x, _ in self._observations)
        logger.info(
            f"Fallback x from observations: {result:.1f} "
            f"(points below SLA: {len(below)}/{len(self._observations)})"
        )
        return max(0.0, result)

    def has_sufficient_data(self) -> bool:
        """Check if enough observations have been collected (cold start guard)."""
        return len(self._observations) >= self.min_observations

    @property
    def num_observations(self) -> int:
        return len(self._observations)

    @property
    def slope(self) -> Optional[float]:
        """Return the current regression slope, or None if not fitted."""
        if not self._is_fitted and not self.fit():
            return None
        return float(self._model.coef_[0])

    @property
    def intercept(self) -> Optional[float]:
        """Return the current regression intercept, or None if not fitted."""
        if not self._is_fitted and not self.fit():
            return None
        return float(self._model.intercept_)
