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
from enum import Enum

import pandas as pd
import pmdarima
from prophet import Prophet

from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*force_all_finite.*",
)


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
    class Mode(str, Enum):
        RAW = "raw"
        LOG1P = "log1p"

    def __init__(self, window_size=100, minimum_data_points=5):
        super().__init__(minimum_data_points=minimum_data_points)
        self.window_size = window_size  # How many past points to use
        self.model = None
        # Keep raw values so we can fit in raw space first, then fallback to log1p space.
        self._raw_buffer: list[float] = []
        # Pending raw points to incrementally update the fitted model with.
        self._pending_raw_updates: list[float] = []
        # Current modeling space
        self._mode: ARIMAPredictor.Mode = ARIMAPredictor.Mode.RAW

    def get_last_value(self):
        """Return last value in original scale."""
        if self._raw_buffer:
            return float(self._raw_buffer[-1])
        if not self.data_buffer:
            return 0
        return float(self.data_buffer[-1])

    def add_data_point(self, value):
        prev_len = len(self.data_buffer)
        # Use raw value for idle skipping in BasePredictor. We may transform later.
        super().add_data_point(value)
        if len(self.data_buffer) > prev_len:
            raw = max(0.0, float(self.data_buffer[-1]))
            self._raw_buffer.append(raw)
            self._pending_raw_updates.append(raw)
            # If we are in log1p mode, keep data_buffer in model space.
            if self._mode == ARIMAPredictor.Mode.LOG1P:
                self.data_buffer[-1] = math.log1p(raw)
        # Keep only the last window_size points
        if len(self.data_buffer) > self.window_size:
            self.data_buffer = self.data_buffer[-self.window_size :]
        if len(self._raw_buffer) > self.window_size:
            self._raw_buffer = self._raw_buffer[-self.window_size :]

    def predict_next(self):
        """Predict the next value(s)"""
        if len(self._raw_buffer) < self.minimum_data_points:
            return self.get_last_value()

        # Check if all values are the same (constant data)
        # pmdarima will predict 0 for constant data, we need to correct its prediction
        if len(set(self._raw_buffer)) == 1:
            return float(self._raw_buffer[0])

        try:
            # Fit auto ARIMA model once, then only do incremental updates.
            if self.model is None:
                # Always try raw space first
                self._mode = ARIMAPredictor.Mode.RAW
                self.model = pmdarima.auto_arima(
                    self.data_buffer,
                    suppress_warnings=True,
                    error_action="ignore",
                )
                order = getattr(self.model, "order", None)
                seasonal_order = getattr(self.model, "seasonal_order", None)
                aic = None
                try:
                    aic = float(self.model.aic())  # type: ignore[attr-defined]
                except Exception:
                    aic = None
                logger.info(
                    f"ARIMA selected order={order} seasonal_order={seasonal_order} aic={aic}"
                )

                # If raw collapses to (0,d,0), fallback to log1p(y)
                try:
                    if order is not None and len(order) == 3:
                        p, _, q = order
                        if p == 0 and q == 0:
                            # Build log buffer/model in locals and only swap on success
                            log_buffer = [math.log1p(v) for v in self._raw_buffer]
                            log_model = pmdarima.auto_arima(
                                log_buffer,
                                suppress_warnings=True,
                                error_action="ignore",
                            )

                            # Swap mode + model + buffer atomically
                            self._mode = ARIMAPredictor.Mode.LOG1P
                            self.data_buffer = log_buffer
                            self.model = log_model

                            order2 = getattr(self.model, "order", None)
                            seasonal_order2 = getattr(
                                self.model, "seasonal_order", None
                            )
                            aic2 = None
                            try:
                                aic2 = float(self.model.aic())  # type: ignore[attr-defined]
                            except Exception:
                                aic2 = None
                            logger.info(
                                f"Detect ARIMA model collapses to (0,d,0), fallback to log1p(y) to better handle spiky time series."
                                f"ARIMA (fallback log1p) selected order={order2} seasonal_order={seasonal_order2} aic={aic2}"
                            )
                except Exception:
                    # If fallback fails, keep raw.
                    self._mode = ARIMAPredictor.Mode.RAW

                # Model is fit on all history; clear pending updates.
                self._pending_raw_updates = []
            else:
                # Incrementally update model with any new observations since last predict.
                if self._pending_raw_updates:
                    upd = (
                        [math.log1p(v) for v in self._pending_raw_updates]
                        if self._mode == ARIMAPredictor.Mode.LOG1P
                        else self._pending_raw_updates
                    )
                    self.model.update(upd)

            # Clear pending updates: model is now up-to-date through the latest observed point.
            self._pending_raw_updates = []

            # Make prediction
            forecast = float(self.model.predict(n_periods=1)[0])
            if self._mode == ARIMAPredictor.Mode.LOG1P:
                return max(0.0, math.expm1(forecast))
            return max(0.0, forecast)
        except Exception as e:
            # Log the specific error for debugging
            logger.warning(f"ARIMA prediction failed: {e}, using last value")
            self._pending_raw_updates = []
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
