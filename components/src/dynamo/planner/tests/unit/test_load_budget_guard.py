# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for disaggregated load scaling at a GPU budget floor."""

from types import MethodType
from typing import Optional

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.state_machine import PlannerScalingState
from dynamo.planner.core.types import (
    EngineCapabilities,
    FpmObservations,
    WorkerCapabilities,
    WorkerCounts,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _state(
    current_p: int,
    current_d: int,
    *,
    min_gpus: int = 64,
    max_gpus: int = 64,
    p_gpu: int = 1,
    d_gpu: int = 1,
) -> PlannerScalingState:
    config = PlannerConfig.model_construct(
        mode="disagg",
        optimization_target="throughput",
        enable_load_scaling=True,
        enable_throughput_scaling=False,
        min_gpu_budget=min_gpus,
        max_gpu_budget=max_gpus,
        min_endpoint=1,
        prefill_min_endpoint=None,
        decode_min_endpoint=None,
    )
    state = PlannerScalingState(
        config,
        WorkerCapabilities(
            prefill=EngineCapabilities(gpu_cost_per_replica=p_gpu),
            decode=EngineCapabilities(gpu_cost_per_replica=d_gpu),
        ),
    )
    state.observe_worker_counts(
        WorkerCounts(
            ready_num_prefill=current_p,
            ready_num_decode=current_d,
            expected_num_prefill=current_p,
            expected_num_decode=current_d,
        )
    )
    return state


def _decision(
    state: PlannerScalingState,
    proposed_p: Optional[int],
    proposed_d: Optional[int],
):
    state.begin_tick()

    def _prefill(_self, _stats, _workers):
        return proposed_p

    def _decode(_self, _stats, _workers):
        return proposed_d

    state._prefill_easy_decision = MethodType(_prefill, state)
    state._decode_easy_decision = MethodType(_decode, state)
    observations = FpmObservations(
        prefill={(f"p-{index}", 0): object() for index in range(state._num_p_workers)},
        decode={(f"d-{index}", 0): object() for index in range(state._num_d_workers)},
    )
    return state.advance_load(observations)


def test_fixed_budget_scale_down_hold_returns_hold():
    state = _state(28, 36)

    assert _decision(state, 27, None) is None
    diagnostics = state.diagnostics()
    assert diagnostics.load_decision_reason == "gpu_budget_guard_hold"


def test_fixed_budget_both_down_cannot_become_scale_up():
    state = _state(27, 36)

    assert _decision(state, 26, 35) is None
    diagnostics = state.diagnostics()
    assert diagnostics.load_decision_reason == "gpu_budget_guard_hold"


def test_fixed_budget_rebalance_waits_for_decode_scale_up_intent():
    state = _state(27, 37)

    assert _decision(state, 26, None) is None
    assert state.diagnostics().load_decision_reason == "gpu_budget_guard_hold"

    decision = _decision(state, 26, 38)
    assert decision is not None
    assert (decision.num_prefill, decision.num_decode) == (26, 38)


def test_load_loop_does_not_invent_donor_for_scale_up_at_full_budget():
    state = _state(28, 36)

    assert _decision(state, None, 37) is None
    assert state.diagnostics().load_decision_reason == "gpu_budget_guard_hold"


def test_load_loop_still_scales_down_above_floor():
    state = _state(35, 35, max_gpus=80)

    decision = _decision(state, 34, None)
    assert decision is not None
    assert (decision.num_prefill, decision.num_decode) == (34, 35)


def test_single_load_loop_holds_scale_down_at_fixed_budget():
    state = _state(1, 64)
    state._config.mode = "decode"
    state._config.enable_throughput_scaling = False

    decision = _decision(state, None, 63)

    assert decision is not None
    assert decision.num_decode == 64
    assert state.diagnostics().load_decision_reason == "gpu_budget_guard_hold"


def test_agg_load_loop_holds_scale_down_at_fixed_budget():
    state = _state(1, 64)
    state._config.mode = "agg"
    state._config.enable_throughput_scaling = False

    def _agg(_self, _stats, _workers):
        return 63

    state._agg_easy_decision = MethodType(_agg, state)
    decision = _decision(state, None, 63)

    assert decision is not None
    assert decision.num_decode == 64
    assert state.diagnostics().load_decision_reason == "gpu_budget_guard_hold"
