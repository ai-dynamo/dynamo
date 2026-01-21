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
import math
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import pandas as pd
import pmdarima
from prophet import Prophet

logger = logging.getLogger("cmdstanpy")
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

# Suppress sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class BasePredictor(ABC):
    """Base class for all load predictors"""

    def __init__(self, minimum_data_points=5):
        self.minimum_data_points = minimum_data_points
        self.data_buffer = []
        # Even if we preload historical data, we still want to ignore the initial
        # post-deployment idle period (a run of zeros) until we see the first
        # non-zero datapoint from live traffic.
        self._seen_nonzero_since_idle_reset = False

    def reset_idle_skip(self):
        """Reset idle-period skipping state (e.g., after warmup, before live)."""
        self._seen_nonzero_since_idle_reset = False

    def add_data_point(self, value):
        """Add new data point to the buffer"""
        if math.isnan(value):
            value = 0

        if value == 0 and not self._seen_nonzero_since_idle_reset:
            # Skip the beginning idle period (leading zeros) even if data_buffer
            # is pre-warmed with historical data.
            return

        if value != 0:
            self._seen_nonzero_since_idle_reset = True

        self.data_buffer.append(value)

    def get_last_value(self):
        """Get the last value from the buffer"""
        if not self.data_buffer:
            return 0
        return self.data_buffer[-1]

    @abstractmethod
    def predict_next(self):
        """Predict the next value"""
        pass


class ConstantPredictor(BasePredictor):
    """
    Assume load is constant and predict the next load to be the same as most recent load
    """

    def __init__(self, **kwargs):
        super().__init__(minimum_data_points=1)

    def predict_next(self):
        return self.get_last_value()


# Auto ARIMA model from pmdarima
class ARIMAPredictor(BasePredictor):
    def __init__(self, window_size=100, minimum_data_points=5):
        super().__init__(minimum_data_points=minimum_data_points)
        self.window_size = window_size  # How many past points to use
        self.model = None
        # Pending points to incrementally update the fitted model with.
        # This avoids re-running auto_arima() on every step.
        self._pending_updates: list[float] = []

    def add_data_point(self, value):
        prev_len = len(self.data_buffer)
        super().add_data_point(value)
        if len(self.data_buffer) > prev_len:
            # Only queue updates if the value wasn't skipped by BasePredictor.
            self._pending_updates.append(float(self.data_buffer[-1]))
        # Keep only the last window_size points
        if len(self.data_buffer) > self.window_size:
            self.data_buffer = self.data_buffer[-self.window_size :]

    def predict_next(self):
        """Predict the next value(s)"""
        if len(self.data_buffer) < self.minimum_data_points:
            return self.get_last_value()

        # Check if all values are the same (constant data)
        # pmdarima will predict 0 for constant data, we need to correct its prediction
        if len(set(self.data_buffer)) == 1:
            return self.data_buffer[0]  # Return the constant value

        try:
            # Fit auto ARIMA model once, then only do incremental updates.
            if self.model is None:
                self.model = pmdarima.auto_arima(
                    self.data_buffer,
                    suppress_warnings=True,
                    error_action="ignore",
                )
            else:
                # Incrementally update model with any new observations since last predict.
                if self._pending_updates:
                    self.model.update(self._pending_updates)

            # Clear pending updates: model is now up-to-date through the latest observed point.
            self._pending_updates = []

            # Make prediction
            forecast = self.model.predict(n_periods=1)
            return forecast[0]
        except Exception as e:
            # Log the specific error for debugging
            logger.warning(f"ARIMA prediction failed: {e}, using last value")
            self._pending_updates = []
            return self.get_last_value()


# Time-series forecasting model from Meta
class ProphetPredictor(BasePredictor):
    def __init__(self, window_size=100, step_size=3600, minimum_data_points=5):
        super().__init__(minimum_data_points=minimum_data_points)
        self.window_size = window_size
        self.curr_step = 0
        self.step_size = step_size
        self.start_date = datetime(2024, 1, 1)  # Base date for generating timestamps
        self.data_buffer = []  # Override to store dicts instead of values
        self._seen_nonzero_since_idle_reset = False

    def add_data_point(self, value):
        """Add new data point to the buffer"""
        # Use proper datetime for Prophet
        timestamp = self.start_date + timedelta(seconds=self.curr_step)
        value = 0 if math.isnan(value) else value

        if value == 0 and not self._seen_nonzero_since_idle_reset:
            # skip the beginning idle period (leading zeros), even if pre-warmed
            return

        if value != 0:
            self._seen_nonzero_since_idle_reset = True

        self.data_buffer.append({"ds": timestamp, "y": value})
        self.curr_step += 1

        # Keep only the last window_size points
        if len(self.data_buffer) > self.window_size:
            self.data_buffer = self.data_buffer[-self.window_size :]

    def get_last_value(self):
        """Get the last value from the buffer"""
        if not self.data_buffer:
            return 0
        return self.data_buffer[-1]["y"]

    def predict_next(self):
        """Predict the next value"""
        if len(self.data_buffer) < self.minimum_data_points:
            return self.get_last_value()

        # Convert to DataFrame
        df = pd.DataFrame(self.data_buffer)

        # Initialize and fit Prophet model
        model = Prophet()

        # Fit the model
        model.fit(df)

        # Create future dataframe for next timestamp
        next_timestamp = self.start_date + timedelta(seconds=self.curr_step)
        future_df = pd.DataFrame({"ds": [next_timestamp]})

        # Make prediction
        forecast = model.predict(future_df)
        return forecast["yhat"].iloc[0]


LOAD_PREDICTORS = {
    "constant": ConstantPredictor,
    "arima": ARIMAPredictor,
    "prophet": ProphetPredictor,
}
