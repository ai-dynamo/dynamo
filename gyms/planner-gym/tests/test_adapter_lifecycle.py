# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle-count regressions for Arena rival autoscalers."""

from __future__ import annotations

import asyncio
import importlib
from functools import partial

import pytest

planner_types = pytest.importorskip(
    "dynamo.planner.core.types",
    reason="adapter lifecycle tests require the optional Dynamo runtime",
)

adapters = importlib.import_module("autoscaling_arena.adapters")
aggregation = importlib.import_module("autoscaling_arena.adapters.aggregation")
KedaAutoscaler = adapters.KedaAutoscaler
ReactiveAutoscaler = adapters.ReactiveAutoscaler
StaticAutoscaler = adapters.StaticAutoscaler
current_replica_target = aggregation.current_replica_target
ScheduledTick = planner_types.ScheduledTick
TickInput = planner_types.TickInput
WorkerCounts = planner_types.WorkerCounts


def _cold_start_tick() -> tuple[ScheduledTick, TickInput]:
    counts = WorkerCounts(
        ready_num_prefill=1,
        ready_num_decode=1,
        expected_num_prefill=1,
        expected_num_decode=4,
        decode_scaling_in_progress=True,
    )
    tick = ScheduledTick(
        at_s=15.0,
        need_worker_states=True,
        need_worker_fpm=True,
    )
    return tick, TickInput(now_s=15.0, worker_counts=counts)


def test_current_target_prefers_non_draining_expected_count() -> None:
    _, tick_input = _cold_start_tick()

    assert (
        current_replica_target(
            tick_input.worker_counts,
            role="decode",
            minimum=1,
        )
        == 4
    )


@pytest.mark.parametrize(
    "autoscaler_factory",
    [ReactiveAutoscaler, KedaAutoscaler],
)
def test_hold_tick_does_not_cancel_starting_decode_workers(autoscaler_factory) -> None:
    autoscaler = autoscaler_factory(mode="disagg")
    tick, tick_input = _cold_start_tick()

    effects = asyncio.run(autoscaler.tick(tick, tick_input))

    assert effects.scale_to is not None
    assert effects.scale_to.num_decode == 4


@pytest.mark.parametrize(
    "autoscaler_factory",
    [
        partial(StaticAutoscaler, num_prefill=1, num_decode=1),
        ReactiveAutoscaler,
        KedaAutoscaler,
    ],
)
def test_rivals_expose_noop_regression_bootstrap(autoscaler_factory) -> None:
    autoscaler = autoscaler_factory()
    assert autoscaler.supports_ais_bootstrap is False
    assert (
        autoscaler.install_regressions_from_fpms(
            prefill_fpms=[object()],
            decode_fpms=[object()],
            agg_fpms=[object()],
        )
        is None
    )
