# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the planner's nonnegative moving average."""

from __future__ import annotations

import pytest

from dynamo.planner.core.perf_model.moving_average import NonnegativeMovingAverage

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _feed(max_len: int, values: list[float]) -> NonnegativeMovingAverage:
    avg = NonnegativeMovingAverage(max_len)
    for value in values:
        avg.add_after_first_nonzero(value)
    return avg


def test_empty_history_is_zero() -> None:
    assert NonnegativeMovingAverage(4).value == 0.0


def test_leading_zero_and_negative_samples_are_ignored() -> None:
    assert _feed(4, [0.0, -3.0, 0.0, -1.0]).value == 0.0


def test_first_positive_sample_starts_window() -> None:
    # Leading 0.0 and -2.0 would pull the mean down if they were counted.
    assert _feed(4, [0.0, -2.0, 6.0]).value == 6.0
    assert _feed(4, [0.0, -2.0, 6.0, 2.0]).value == 4.0


def test_later_zero_and_negative_samples_count_as_zero() -> None:
    assert _feed(4, [8.0, 0.0]).value == 4.0
    # -4.0 is clamped to 0.0: mean of [6, 0, 0].
    assert _feed(4, [6.0, 0.0, -4.0]).value == 2.0


def test_window_evicts_oldest_sample() -> None:
    assert _feed(2, [2.0, 4.0]).value == 3.0
    # 2.0 is evicted: mean of [4, 8].
    assert _feed(2, [2.0, 4.0, 8.0]).value == 6.0
    # 4.0 is evicted, and the clamped -1.0 counts as 0.0: mean of [8, 0].
    assert _feed(2, [2.0, 4.0, 8.0, -1.0]).value == 4.0


@pytest.mark.parametrize("max_len", [0, 1])
def test_window_of_zero_or_one_keeps_latest_sample(max_len: int) -> None:
    assert _feed(max_len, [3.0]).value == 3.0
    assert _feed(max_len, [3.0, 5.0]).value == 5.0
    assert _feed(max_len, [3.0, 5.0, -2.0]).value == 0.0
