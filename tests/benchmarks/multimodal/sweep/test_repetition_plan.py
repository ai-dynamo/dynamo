# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from benchmarks.multimodal.sweep.repetition_plan import balanced_config_orders

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _labels(orders: list[list[dict[str, Any]]]) -> list[list[str]]:
    return [[config["label"] for config in order] for order in orders]


def test_two_arm_four_repetition_order_is_balanced() -> None:
    configs = [{"label": "a"}, {"label": "b"}]

    orders = balanced_config_orders(configs, repetitions=4)

    assert _labels(orders) == [["a", "b"], ["b", "a"], ["b", "a"], ["a", "b"]]


def test_three_arm_orders_cover_rotations_reverses_and_wrap() -> None:
    configs = [{"label": "a"}, {"label": "b"}, {"label": "c"}]

    orders = _labels(balanced_config_orders(configs, repetitions=8))

    expected_cycle = [
        ["a", "b", "c"],
        ["b", "c", "a"],
        ["c", "a", "b"],
        ["c", "b", "a"],
        ["a", "c", "b"],
        ["b", "a", "c"],
    ]
    assert orders == expected_cycle + expected_cycle[:2]


@pytest.mark.parametrize(
    ("configs", "repetitions", "error"),
    [
        ([], 1, "At least one"),
        ([{"label": "a"}], 0, "positive"),
        ([{"label": "a"}, {"label": "a"}], 1, "unique"),
    ],
)
def test_invalid_repetition_plan_is_rejected(
    configs: list[dict[str, Any]], repetitions: int, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        balanced_config_orders(configs, repetitions)
