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
from argparse import Namespace
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
import pmdarima
from filterpy.kalman import KalmanFilter
from prophet import Prophet

from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*force_all_finite.*",
)

# Silence very chatty Prophet/cmdstanpy logs (we keep planner logs at INFO).
for _name in (
    "prophet",
    "prophet.forecaster",
    "prophet.models",
    "cmdstanpy",
    "cmdstanpy.model",
):
    _l = logging.getLogger(_name)
    _l.addHandler(logging.NullHandler())
    _l.propagate = False
    _l.setLevel(logging.WARNING)


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

    def __init__(self, _args: Namespace):
        super().__init__(minimum_data_points=1)

    def predict_next(self):
        return self.get_last_value()


# Auto ARIMA model from pmdarima
class ARIMAPredictor(BasePredictor):
    class Mode(str, Enum):
        RAW = "raw"
        LOG1P = "log1p"

    def __init__(self, args: Namespace):
        super().__init__(minimum_data_points=5)
        self.model = None
        # Keep raw values so we can fit in raw space first, then fallback to log1p space.
        self._raw_buffer: list[float] = []
        # Pending raw points to incrementally update the fitted model with.
        self._pending_raw_updates: list[float] = []
        # Shared log1p knob across predictors. Back-compat: `--arima-mode=log1p`.
        use_log1p = bool(getattr(args, "load_predictor_log1p", False))
        self._requested_mode = (
            ARIMAPredictor.Mode.LOG1P if use_log1p else ARIMAPredictor.Mode.RAW
        )
        self._mode: ARIMAPredictor.Mode = self._requested_mode

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
            # Keep `data_buffer` in the model space.
            if self._mode == ARIMAPredictor.Mode.LOG1P:
                self.data_buffer[-1] = math.log1p(raw)

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
                # First fit: honor requested mode
                self._mode = self._requested_mode
                if self._mode == ARIMAPredictor.Mode.LOG1P:
                    # Ensure model buffer is in log-space
                    self.data_buffer = [math.log1p(v) for v in self._raw_buffer]
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

                # If user requested raw and it collapses to (0,d,0), fallback to log1p(y)
                try:
                    if order is not None and len(order) == 3:
                        p, _, q = order
                        if (
                            p == 0
                            and q == 0
                            and self._requested_mode == ARIMAPredictor.Mode.RAW
                        ):
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
    def __init__(self, args: Namespace):
        super().__init__(minimum_data_points=5)
        self._use_log1p = bool(getattr(args, "load_predictor_log1p", False))
        # Window size is only used by Prophet (to bound refit cost).
        self.window_size = getattr(
            args,
            "prophet_window_size",
            getattr(args, "load_prediction_window_size", 50),
        )
        self.curr_step = 0
        # Use adjustment_interval as step size (seconds per observation)
        self.step_size = getattr(args, "adjustment_interval", 3600)
        self.start_date = datetime(2024, 1, 1)  # Base date for generating timestamps
        self.data_buffer = []  # Override to store dicts instead of values
        self._seen_nonzero_since_idle_reset = False

    def add_data_point(self, value):
        """Add new data point to the buffer"""
        # Use proper datetime for Prophet
        timestamp = self.start_date + timedelta(seconds=self.curr_step * self.step_size)
        value = 0 if math.isnan(value) else value

        if value == 0 and not self._seen_nonzero_since_idle_reset:
            # skip the beginning idle period (leading zeros), even if pre-warmed
            return

        if value != 0:
            self._seen_nonzero_since_idle_reset = True

        if self._use_log1p:
            value = math.log1p(max(0.0, value))
        self.data_buffer.append({"ds": timestamp, "y": value})
        self.curr_step += 1

        # Keep only the last window_size points
        if len(self.data_buffer) > self.window_size:
            self.data_buffer = self.data_buffer[-self.window_size :]

    def get_last_value(self):
        """Get the last value from the buffer"""
        if not self.data_buffer:
            return 0
        y = float(self.data_buffer[-1]["y"])
        return max(0.0, math.expm1(y)) if self._use_log1p else y

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
        yhat = float(forecast["yhat"].iloc[0])
        return max(0.0, math.expm1(yhat)) if self._use_log1p else yhat


class KalmanPredictor(BasePredictor):
    """
    Simple 1D Kalman predictor for online "observe 1 -> predict 1".

    Uses a local linear trend model:
      x_t = x_{t-1} + v_{t-1} + w
      v_t = v_{t-1} + u

    This tends to be a better match than ARIMA for low-latency smoothing + short-horizon
    forecasting in bursty systems.
    """

    def __init__(self, args: Namespace):
        super().__init__(minimum_data_points=getattr(args, "kalman_min_points", 5))
        # Shared log1p knob across predictors. Back-compat: `--kalman-log1p`.
        self._use_log1p = bool(getattr(args, "load_predictor_log1p", False)) or bool(
            getattr(args, "kalman_log1p", False)
        )
        q_level = getattr(args, "kalman_q_level", 1.0)
        q_trend = getattr(args, "kalman_q_trend", 0.1)
        r = getattr(args, "kalman_r", 10.0)
        self._kf = KalmanFilter(dim_x=2, dim_z=1)
        # State: [level, trend]
        self._kf.x = np.array([[0.0], [0.0]], dtype=float)
        self._kf.F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
        self._kf.H = np.array([[1.0, 0.0]], dtype=float)
        self._kf.P *= 1000.0
        self._kf.R = np.array([[max(1e-9, float(r))]], dtype=float)
        self._kf.Q = np.array(
            [
                [max(1e-12, float(q_level)), 0.0],
                [0.0, max(1e-12, float(q_trend))],
            ],
            dtype=float,
        )
        self._initialized = False
        # Gate repeated predict_next() calls: cache the one-step forecast so we
        # don't advance the filter multiple times per interval.
        self._has_cached_pred = False
        self._cached_pred: float = 0.0

    def add_data_point(self, value):
        prev_len = len(self.data_buffer)
        super().add_data_point(value)
        if len(self.data_buffer) == prev_len:
            return
        z_raw = float(self.data_buffer[-1])
        z = math.log1p(max(0.0, z_raw)) if self._use_log1p else z_raw
        # immediately update the filter with new data point
        if not self._initialized:
            self._kf.x = np.array([[z], [0.0]], dtype=float)
            self._initialized = True
        else:
            # If we already predicted this step, don't predict again.
            if not self._has_cached_pred:
                self._kf.predict()
            self._kf.update(np.array([[z]], dtype=float))
        # Consumed this step; clear cached forecast for next interval.
        self._has_cached_pred = False

    def predict_next(self):
        if not self._initialized:
            return self.get_last_value()
        if self._has_cached_pred:
            return (
                max(0.0, math.expm1(self._cached_pred))
                if self._use_log1p
                else self._cached_pred
            )
        # one-step ahead prediction: predict then return predicted level
        self._kf.predict()
        self._cached_pred = float(self._kf.x[0][0])
        self._has_cached_pred = True
        return (
            max(0.0, math.expm1(self._cached_pred))
            if self._use_log1p
            else self._cached_pred
        )


LOAD_PREDICTORS = {
    "constant": ConstantPredictor,
    "arima": ARIMAPredictor,
    "kalman": KalmanPredictor,
    "prophet": ProphetPredictor,
}
