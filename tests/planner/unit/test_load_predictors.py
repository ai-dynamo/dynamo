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

"""Unit tests for load predictor classes in dynamo.planner.utils.load_predictor."""

import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from dynamo.planner.utils.load_predictor import (
    ConstantPredictor,
    KalmanPredictor,
    ProphetPredictor,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    load_predictor_log1p: bool = False,
    step_size: int = 60,
    prophet_window_size: int = 50,
    kalman_q_level: float = 1e-4,
    kalman_q_trend: float = 1e-5,
    kalman_r: float = 0.1,
    kalman_min_points: int = 3,
):
    """Create a minimal mock PlannerConfig for predictor instantiation."""
    cfg = MagicMock()
    cfg.load_predictor_log1p = load_predictor_log1p
    cfg.throughput_adjustment_interval = step_size
    cfg.prophet_window_size = prophet_window_size
    cfg.kalman_q_level = kalman_q_level
    cfg.kalman_q_trend = kalman_q_trend
    cfg.kalman_r = kalman_r
    cfg.kalman_min_points = kalman_min_points
    return cfg


# ---------------------------------------------------------------------------
# ConstantPredictor tests
# ---------------------------------------------------------------------------


class TestConstantPredictor:
    """Tests for ConstantPredictor behaviour."""

    def test_returns_zero_with_no_data(self):
        """predict_next() returns 0 when no data points have been added."""
        predictor = ConstantPredictor(_make_config())
        assert predictor.predict_next() == 0

    def test_returns_last_added_value(self):
        """predict_next() always returns the most recently added value."""
        predictor = ConstantPredictor(_make_config())
        for v in [10.0, 20.0, 30.0]:
            predictor.add_data_point(v)
        assert predictor.predict_next() == 30.0

    def test_skips_leading_zeros(self):
        """Leading zeros (idle period) are excluded from the buffer."""
        predictor = ConstantPredictor(_make_config())
        predictor.add_data_point(0)
        predictor.add_data_point(0)
        assert predictor.predict_next() == 0
        assert len(predictor.data_buffer) == 0

    def test_does_not_skip_zeros_after_nonzero(self):
        """Once a nonzero value has been seen, zeros are retained."""
        predictor = ConstantPredictor(_make_config())
        predictor.add_data_point(5.0)
        predictor.add_data_point(0)
        assert predictor.predict_next() == 0
        assert len(predictor.data_buffer) == 2

    def test_nan_treated_as_zero(self):
        """NaN values are coerced to 0 before being stored."""
        predictor = ConstantPredictor(_make_config())
        predictor.add_data_point(10.0)
        predictor.add_data_point(float("nan"))
        assert predictor.predict_next() == 0


# ---------------------------------------------------------------------------
# ProphetPredictor timestamp bug regression tests
# ---------------------------------------------------------------------------


class TestProphetPredictorTimestamp:
    """Regression tests for the Prophet predictor next-step timestamp bug.

    Bug: predict_next() used
        ``timedelta(seconds=self.curr_step)``
    instead of
        ``timedelta(seconds=self.curr_step * self.step_size)``

    When step_size > 1 (which is always the case in production, default=180s)
    the predicted timestamp fell far below the last training sample, causing
    Prophet to extrapolate into the past rather than one step into the future.
    """

    def _make_prophet(self, step_size=180, log1p=False):
        """Instantiate a ProphetPredictor with a given step_size."""
        return ProphetPredictor(
            _make_config(step_size=step_size, load_predictor_log1p=log1p)
        )

    def _mock_forecast_df(self, yhat: float):
        """Return a minimal forecast DataFrame like Prophet would produce."""
        return pd.DataFrame(
            {"yhat": [yhat], "yhat_lower": [yhat], "yhat_upper": [yhat]}
        )

    # ------------------------------------------------------------------
    # Core timestamp correctness test (the actual bug)
    # ------------------------------------------------------------------

    def test_next_timestamp_uses_step_size(self):
        """The timestamp passed to Prophet.predict() must equal
        start_date + curr_step * step_size, NOT start_date + curr_step.

        With step_size=180, after 6 data points curr_step==6, so the
        expected next timestamp is start_date + 1080 seconds, not
        start_date + 6 seconds.
        """
        step_size = 180
        predictor = self._make_prophet(step_size=step_size)

        for v in [10.0, 12.0, 14.0, 13.0, 15.0, 11.0]:
            predictor.add_data_point(v)

        assert predictor.curr_step == 6

        expected_next_ts = predictor.start_date + timedelta(
            seconds=predictor.curr_step * step_size
        )
        buggy_next_ts = predictor.start_date + timedelta(seconds=predictor.curr_step)

        assert (
            expected_next_ts != buggy_next_ts
        ), "Sanity check: the two timestamps must differ"

        captured_future_df: list[pd.DataFrame] = []

        def fake_predict(df):
            captured_future_df.append(df.copy())
            return self._mock_forecast_df(10.0)

        mock_model = MagicMock()
        mock_model.predict.side_effect = fake_predict

        with patch(
            "dynamo.planner.utils.load_predictor.Prophet",
            return_value=mock_model,
        ):
            predictor.predict_next()

        assert len(captured_future_df) == 1
        actual_ts = captured_future_df[0]["ds"].iloc[0]

        assert actual_ts == expected_next_ts, (
            f"predict_next() passed wrong timestamp to Prophet.\n"
            f"  Expected (correct): {expected_next_ts}\n"
            f"  Got:                {actual_ts}\n"
            f"  Buggy value would be: {buggy_next_ts}"
        )

    def test_next_timestamp_consistent_with_add_data_point(self):
        """The timestamp used for the next prediction must be exactly one
        step_size ahead of the last data-point timestamp in the buffer."""
        step_size = 60
        predictor = self._make_prophet(step_size=step_size)

        for v in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            predictor.add_data_point(v)

        last_data_ts: datetime = predictor.data_buffer[-1]["ds"]
        expected_next_ts = last_data_ts + timedelta(seconds=step_size)

        captured: list[pd.DataFrame] = []

        def fake_predict(df):
            captured.append(df.copy())
            return self._mock_forecast_df(5.0)

        mock_model = MagicMock()
        mock_model.predict.side_effect = fake_predict

        with patch(
            "dynamo.planner.utils.load_predictor.Prophet",
            return_value=mock_model,
        ):
            predictor.predict_next()

        actual_ts = captured[0]["ds"].iloc[0]
        assert actual_ts == expected_next_ts, (
            f"Next-step timestamp must be one step ahead of the last training point.\n"
            f"  Last training ts: {last_data_ts}\n"
            f"  Expected next ts: {expected_next_ts}\n"
            f"  Actual next ts:   {actual_ts}"
        )

    def test_next_timestamp_step_size_1_unchanged(self):
        """Edge-case: step_size=1 was accidentally correct before the fix.
        Verify the fix does not regress this case."""
        step_size = 1
        predictor = self._make_prophet(step_size=step_size)

        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
            predictor.add_data_point(v)

        expected_next_ts = predictor.start_date + timedelta(
            seconds=predictor.curr_step * step_size
        )

        captured: list[pd.DataFrame] = []

        def fake_predict(df):
            captured.append(df.copy())
            return self._mock_forecast_df(3.0)

        mock_model = MagicMock()
        mock_model.predict.side_effect = fake_predict

        with patch(
            "dynamo.planner.utils.load_predictor.Prophet",
            return_value=mock_model,
        ):
            predictor.predict_next()

        actual_ts = captured[0]["ds"].iloc[0]
        assert actual_ts == expected_next_ts

    # ------------------------------------------------------------------
    # Functional / correctness tests
    # ------------------------------------------------------------------

    def test_returns_last_value_when_insufficient_data(self):
        """predict_next() falls back to last value when buffer < 5 points."""
        predictor = self._make_prophet()
        for v in [10.0, 20.0, 30.0]:
            predictor.add_data_point(v)
        assert predictor.predict_next() == 30.0

    def test_returns_zero_when_empty(self):
        """predict_next() returns 0 when no data points exist."""
        predictor = self._make_prophet()
        assert predictor.predict_next() == 0

    def test_predict_next_returns_non_negative_raw_mode(self):
        """In raw mode, predict_next() must never return a negative value."""
        predictor = self._make_prophet(log1p=False)

        for v in [10.0, 12.0, 11.0, 13.0, 10.0, 9.0]:
            predictor.add_data_point(v)

        mock_model = MagicMock()
        mock_model.predict.return_value = self._mock_forecast_df(-5.0)

        with patch(
            "dynamo.planner.utils.load_predictor.Prophet",
            return_value=mock_model,
        ):
            result = predictor.predict_next()

        assert result >= 0.0, f"predict_next() returned negative value {result}"

    def test_predict_next_returns_non_negative_log1p_mode(self):
        """In log1p mode, predict_next() applies expm1 and clamps to >= 0."""
        predictor = self._make_prophet(log1p=True)

        for v in [10.0, 12.0, 11.0, 13.0, 10.0, 9.0]:
            predictor.add_data_point(v)

        mock_model = MagicMock()
        mock_model.predict.return_value = self._mock_forecast_df(-100.0)

        with patch(
            "dynamo.planner.utils.load_predictor.Prophet",
            return_value=mock_model,
        ):
            result = predictor.predict_next()

        assert result >= 0.0, f"predict_next() returned negative value {result}"

    def test_log1p_mode_transforms_data_correctly(self):
        """In log1p mode, values stored in buffer are log1p-transformed,
        and get_last_value() returns the original scale."""
        predictor = self._make_prophet(log1p=True)
        raw_value = 99.0
        predictor.add_data_point(raw_value)

        buffered_y = predictor.data_buffer[-1]["y"]
        assert abs(buffered_y - math.log1p(raw_value)) < 1e-9

        recovered = predictor.get_last_value()
        assert abs(recovered - raw_value) < 1e-6

    def test_window_size_limits_buffer(self):
        """Buffer never exceeds prophet_window_size entries."""
        window = 5
        predictor = ProphetPredictor(
            _make_config(step_size=60, prophet_window_size=window)
        )
        for v in range(1, 20):
            predictor.add_data_point(float(v))

        assert len(predictor.data_buffer) <= window

    def test_curr_step_advances_on_each_non_skipped_point(self):
        """curr_step increments for every data point not idle-skipped."""
        predictor = self._make_prophet()
        predictor.add_data_point(0)
        predictor.add_data_point(0)
        assert predictor.curr_step == 0

        predictor.add_data_point(5.0)
        assert predictor.curr_step == 1

        predictor.add_data_point(6.0)
        assert predictor.curr_step == 2

    def test_skips_leading_zeros_idle_period(self):
        """Leading zeros (idle period) are excluded from data_buffer."""
        predictor = self._make_prophet()
        predictor.add_data_point(0)
        predictor.add_data_point(0)
        assert len(predictor.data_buffer) == 0

        predictor.add_data_point(10.0)
        assert len(predictor.data_buffer) == 1

    def test_does_not_skip_zero_after_nonzero(self):
        """Once a nonzero has been seen, zeros are retained in the buffer."""
        predictor = self._make_prophet()
        predictor.add_data_point(10.0)
        predictor.add_data_point(0)
        assert len(predictor.data_buffer) == 2


# ---------------------------------------------------------------------------
# ProphetPredictor: parameterized step_size regression
# ---------------------------------------------------------------------------


class TestProphetPredictorMultipleStepSizes:
    """Parameterized check that the predict_next() timestamp is always
    exactly one step ahead of the last training timestamp."""

    @pytest.mark.parametrize("step_size", [30, 60, 120, 180, 300])
    def test_next_timestamp_is_one_step_ahead(self, step_size):
        """Verify correct timestamp for step_size={step_size}."""
        predictor = ProphetPredictor(_make_config(step_size=step_size))

        for v in [5.0, 10.0, 8.0, 12.0, 9.0, 11.0]:
            predictor.add_data_point(v)

        last_ts: datetime = predictor.data_buffer[-1]["ds"]
        expected_next_ts = last_ts + timedelta(seconds=step_size)

        captured: list[pd.DataFrame] = []

        def fake_predict(df):
            captured.append(df.copy())
            return pd.DataFrame(
                {"yhat": [5.0], "yhat_lower": [5.0], "yhat_upper": [5.0]}
            )

        mock_model = MagicMock()
        mock_model.predict.side_effect = fake_predict

        with patch(
            "dynamo.planner.utils.load_predictor.Prophet",
            return_value=mock_model,
        ):
            predictor.predict_next()

        actual_ts = captured[0]["ds"].iloc[0]
        assert (
            actual_ts == expected_next_ts
        ), f"step_size={step_size}: expected {expected_next_ts}, got {actual_ts}"


# ---------------------------------------------------------------------------
# KalmanPredictor sanity tests
# ---------------------------------------------------------------------------


class TestKalmanPredictor:
    """Basic sanity checks for KalmanPredictor."""

    def test_returns_zero_before_first_observation(self):
        """predict_next() returns 0 when uninitialised."""
        predictor = KalmanPredictor(_make_config())
        assert predictor.predict_next() == 0

    def test_predict_returns_non_negative(self):
        """Predictions are non-negative after feeding positive data."""
        predictor = KalmanPredictor(_make_config())
        for v in [10.0, 15.0, 20.0, 15.0, 10.0]:
            predictor.add_data_point(v)
        result = predictor.predict_next()
        assert result >= 0.0

    def test_caches_prediction_between_observations(self):
        """Calling predict_next() twice without new data returns the same value."""
        predictor = KalmanPredictor(_make_config())
        for v in [10.0, 12.0, 11.0]:
            predictor.add_data_point(v)
        first = predictor.predict_next()
        second = predictor.predict_next()
        assert first == second

    def test_log1p_mode_returns_non_negative(self):
        """Log1p mode still returns non-negative predictions."""
        predictor = KalmanPredictor(_make_config(load_predictor_log1p=True))
        for v in [10.0, 15.0, 20.0, 15.0, 10.0]:
            predictor.add_data_point(v)
        result = predictor.predict_next()
        assert result >= 0.0
