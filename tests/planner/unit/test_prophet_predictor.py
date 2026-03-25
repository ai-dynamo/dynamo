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

"""Unit tests for ProphetPredictor — focusing on timestamp alignment between
add_data_point() and predict_next().

Regression test for the bug where predict_next() computed:
    next_timestamp = start_date + timedelta(seconds=curr_step)
instead of:
    next_timestamp = start_date + timedelta(seconds=curr_step * step_size)

With the default step_size=180 this meant the forecast target was 180x too
early — Prophet was asked to extrapolate into the past, producing garbage.
"""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from dynamo.planner.utils.load_predictor import ProphetPredictor

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _make_config(step_size: int = 180, window_size: int = 100, log1p: bool = False):
    """Build a minimal PlannerConfig mock with the fields ProphetPredictor uses."""
    cfg = MagicMock()
    cfg.throughput_adjustment_interval = step_size
    cfg.prophet_window_size = window_size
    cfg.load_predictor_log1p = log1p
    return cfg


class TestProphetPredictorTimestampAlignment:
    """Verify that the timestamp used in predict_next() aligns with the ones
    stored by add_data_point() for both step_size=1 and step_size=180."""

    def _collect_stored_timestamps(self, predictor, n_points: int, value: float = 10.0):
        """Feed n_points into the predictor and return the list of stored ds values."""
        for _ in range(n_points):
            predictor.add_data_point(value)
        return [entry["ds"] for entry in predictor.data_buffer]

    @pytest.mark.parametrize("step_size", [1, 180])
    def test_forecast_target_is_one_step_ahead(self, step_size):
        """predict_next() must pass start_date + curr_step * step_size to Prophet.predict().

        This test patches Prophet.predict() to intercept the actual future_df argument,
        so it will fail if the * self.step_size multiplier is missing in predict_next().
        """
        cfg = _make_config(step_size=step_size)
        predictor = ProphetPredictor(cfg)

        n_points = 6  # > minimum_data_points (5)
        self._collect_stored_timestamps(predictor, n_points)

        # After n_points add_data_point() calls, curr_step == n_points.
        # The forecast target should be start_date + curr_step * step_size.
        expected_next_ts = predictor.start_date + timedelta(
            seconds=predictor.curr_step * step_size
        )

        with patch("dynamo.planner.utils.load_predictor.Prophet") as MockProphet:
            mock_model = MagicMock()
            MockProphet.return_value = mock_model
            mock_model.predict.return_value = pd.DataFrame({"yhat": [10.0]})

            predictor.predict_next()

            call_args = mock_model.predict.call_args
            assert call_args is not None, "Prophet.predict() was never called"
            future_df = call_args[0][0]

        actual_ts = future_df["ds"].iloc[0]
        assert actual_ts == expected_next_ts, (
            f"step_size={step_size}: predict_next() passed {actual_ts} "
            f"to Prophet.predict(), expected {expected_next_ts}. "
            "The * self.step_size multiplier may be missing in predict_next()."
        )

    @pytest.mark.parametrize("step_size", [1, 180])
    def test_forecast_target_not_in_the_past(self, step_size):
        """The forecast timestamp must be strictly after all stored timestamps."""
        cfg = _make_config(step_size=step_size)
        predictor = ProphetPredictor(cfg)

        n_points = 6
        self._collect_stored_timestamps(predictor, n_points)

        # Replicate exactly what predict_next() computes for the next_timestamp.
        next_timestamp = predictor.start_date + timedelta(
            seconds=predictor.curr_step * predictor.step_size
        )

        max_stored_ts = max(entry["ds"] for entry in predictor.data_buffer)
        assert next_timestamp > max_stored_ts, (
            f"step_size={step_size}: forecast timestamp {next_timestamp} is not "
            f"after the last stored timestamp {max_stored_ts}. "
            "Prophet would be asked to retrodict, not forecast."
        )

    def test_predict_next_returns_nonnegative(self):
        """End-to-end: predict_next() should return a non-negative value."""
        cfg = _make_config(step_size=180)
        predictor = ProphetPredictor(cfg)

        for v in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            predictor.add_data_point(v)

        result = predictor.predict_next()
        assert isinstance(result, float)
        assert result >= 0.0, f"predict_next() returned negative value: {result}"

    def test_predict_next_returns_last_value_below_minimum_data_points(self):
        """With fewer than minimum_data_points, predict_next() falls back to last value."""
        cfg = _make_config(step_size=180)
        predictor = ProphetPredictor(cfg)

        for v in [3.0, 4.0, 5.0]:  # Only 3 points, minimum is 5
            predictor.add_data_point(v)

        result = predictor.predict_next()
        assert result == predictor.get_last_value(), (
            "Below minimum_data_points, predict_next() should return the last value."
        )
