# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only allocation integration tests using synthetic lifecycle events."""

from __future__ import annotations

import pytest
from autoscaling_arena.measurement import (
    AllocationConfig,
    MeasurementWindow,
    window_gpu_hours,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _operation(at_s, ordinal, *, pool="agg", active=(), starting=(), draining=()):
    return {
        "at_ms": at_s * 1000.0,
        "operation_ordinal": ordinal,
        "pool": pool,
        "state_after_batch": {
            "active": list(active),
            "starting": list(starting),
            "draining": list(draining),
        },
    }


def test_dynamic_window_includes_starting_and_draining_until_removed():
    # One worker until t=2; two provisioned until t=7; one thereafter.
    operations = [
        _operation(2, 0, active=(0,), starting=(1,)),
        _operation(4, 1, active=(0, 1)),
        _operation(5, 2, active=(1,), draining=(0,)),
        _operation(7, 3, active=(1,)),
    ]
    actual = window_gpu_hours(
        operations,
        AllocationConfig({"agg": 1}, {"agg": 4}),
        MeasurementWindow(1, 8),
        replay_duration_s=10,
    )
    assert actual == pytest.approx((1 + 2 * 5 + 1) * 4 / 3600)


def test_window_integrates_separate_pools_with_different_gpu_costs():
    operations = [
        _operation(1, 0, pool="prefill", active=(0, 1)),
        _operation(2, 1, pool="decode", active=(0, 1, 2)),
        _operation(4, 2, pool="prefill", active=(1,)),
    ]
    actual = window_gpu_hours(
        operations,
        AllocationConfig({"prefill": 1, "decode": 2}, {"prefill": 2, "decode": 4}),
        MeasurementWindow(2, 6),
        replay_duration_s=8,
    )
    # P: 2 workers * 2s + 1 worker * 2s; D: 3 workers * 4s.
    assert actual == pytest.approx((6 * 2 + 12 * 4) / 3600)


def test_static_window_preserves_both_idle_edges():
    actual = window_gpu_hours(
        [],
        AllocationConfig({"agg": 2}, {"agg": 4}, allow_idle_tail=True),
        MeasurementWindow(0, 10),
        replay_duration_s=8,
    )
    assert actual == pytest.approx(80 / 3600)


def test_dynamic_window_cannot_extrapolate_past_replay():
    with pytest.raises(ValueError, match="unobserved idle tail"):
        window_gpu_hours(
            [],
            AllocationConfig({"agg": 1}, {"agg": 4}),
            MeasurementWindow(0, 10),
            replay_duration_s=8,
        )


def test_idle_tail_is_not_permitted_for_changing_allocation():
    with pytest.raises(ValueError, match="fixed worker allocation"):
        window_gpu_hours(
            [_operation(2, 0, active=(0, 1))],
            AllocationConfig({"agg": 1}, {"agg": 4}, allow_idle_tail=True),
            MeasurementWindow(0, 10),
            replay_duration_s=8,
        )


@pytest.mark.parametrize("start,end", [(-1, 1), (2, 2), (2, 1), (0, float("inf"))])
def test_invalid_window_is_rejected(start, end):
    with pytest.raises(ValueError, match="measurement window"):
        MeasurementWindow(start, end)


def test_unknown_lifecycle_pool_is_rejected():
    with pytest.raises(ValueError, match="unconfigured pool"):
        window_gpu_hours(
            [_operation(1, 0, pool="decode", active=(0,))],
            AllocationConfig({"agg": 1}, {"agg": 4}),
            MeasurementWindow(0, 2),
            replay_duration_s=2,
        )


def test_boundary_events_use_post_event_state_without_double_counting():
    actual = window_gpu_hours(
        [
            _operation(1, 0, active=(0, 1)),
            _operation(1, 1, active=(0, 1, 2)),
            _operation(3, 2, active=()),
        ],
        AllocationConfig({"agg": 1}, {"agg": 2}),
        MeasurementWindow(1, 3),
        replay_duration_s=4,
    )
    assert actual == pytest.approx(3 * 2 * 2 / 3600)
