# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace
from typing import Optional

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.state_machine import PlannerScalingState
from dynamo.planner.core.throughput_scaling import ThroughputScalingMixin
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


class _PrefillRegression:
    def find_engine_capacity_rps(self, **kwargs):
        return SimpleNamespace(rps=1.0, ttft_ms=1002.0, eligible=False)


class _ThroughputScalingHarness(ThroughputScalingMixin):
    def __init__(self):
        self._config = SimpleNamespace(
            ttft_ms=200.0,
            min_endpoint=1,
            prefill_min_endpoint=None,
            decode_min_endpoint=None,
            max_throughput_scaling_replicas=8,
        )
        self._prefill_regression = _PrefillRegression()
        self._diag_throughput_reason = None
        self._diag_engine_rps_prefill = None


def test_unreachable_prefill_ttft_does_not_create_replica_floor():
    scaling = _ThroughputScalingHarness()

    replicas = scaling._compute_prefill_replicas(
        demand_rps=0.01,
        isl=1000,
        osl=150,
    )

    assert replicas == 1


def test_prefill_throughput_uses_component_minimum_override():
    scaling = _ThroughputScalingHarness()
    scaling._config.prefill_min_endpoint = 3

    replicas = scaling._compute_prefill_replicas(
        demand_rps=0.01,
        isl=1000,
        osl=150,
    )

    assert replicas == 3


def _disagg_state(
    current_p: int,
    current_d: int,
    raw_p: int,
    raw_d: int,
    *,
    enable_load_scaling: bool,
    cap: int = 8,
    min_gpus: int = 64,
    max_gpus: int = 64,
    p_gpu: int = 1,
    d_gpu: int = 1,
    prefill_min_endpoint: Optional[int] = None,
    decode_min_endpoint: Optional[int] = None,
) -> PlannerScalingState:
    config = PlannerConfig.model_construct(
        mode="disagg",
        optimization_target="throughput",
        enable_load_scaling=enable_load_scaling,
        enable_throughput_scaling=True,
        max_throughput_scaling_replicas=cap,
        min_gpu_budget=min_gpus,
        max_gpu_budget=max_gpus,
        min_endpoint=1,
        prefill_min_endpoint=prefill_min_endpoint,
        decode_min_endpoint=decode_min_endpoint,
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

    def _prefill(_self, _demand, _isl, _osl, _kv_hit_rate=None):
        return raw_p

    def _decode(_self, _demand, _isl, _osl):
        return raw_d

    state._compute_prefill_replicas = MethodType(_prefill, state)
    state._compute_decode_replicas = MethodType(_decode, state)
    return state


def _load_decision(
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


def test_disagg_throughput_caps_persisted_lower_bounds_once_per_observation():
    state = _disagg_state(24, 40, 8, 180, enable_load_scaling=True)

    assert state._throughput_disagg(1.0, 1.0, 1.0) is None
    assert (state._throughput_lower_bound_p, state._throughput_lower_bound_d) == (
        16,
        48,
    )

    # The second builtin consumer evaluates the same throughput observation.
    # Repeating it must not ratchet either bound by another eight replicas.
    assert state._throughput_disagg(1.0, 1.0, 1.0) is None
    assert (state._throughput_lower_bound_p, state._throughput_lower_bound_d) == (
        16,
        48,
    )


def test_disagg_throughput_uses_configured_replica_cap():
    state = _disagg_state(
        24,
        40,
        8,
        180,
        enable_load_scaling=True,
        cap=3,
    )

    assert state._throughput_disagg(1.0, 1.0, 1.0) is None
    assert (state._throughput_lower_bound_p, state._throughput_lower_bound_d) == (
        21,
        43,
    )


def test_disagg_throughput_holds_two_scale_ups_at_gpu_ceiling():
    state = _disagg_state(24, 40, 32, 48, enable_load_scaling=False)

    assert state._throughput_disagg(1.0, 1.0, 1.0) is None
    assert state.diagnostics().throughput_decision_reason == "gpu_budget_guard_hold"


def test_disagg_throughput_natural_noop_is_not_reported_as_budget_hold():
    state = _disagg_state(24, 40, 24, 40, enable_load_scaling=False)

    assert state._throughput_disagg(1.0, 1.0, 1.0) is None
    assert state.diagnostics().throughput_decision_reason == "no_change"


def test_mixed_scaling_does_not_persist_unfunded_scale_ups_at_gpu_ceiling():
    state = _disagg_state(24, 40, 32, 48, enable_load_scaling=True)

    assert state._throughput_disagg(1.0, 1.0, 1.0) is None
    assert (state._throughput_lower_bound_p, state._throughput_lower_bound_d) == (
        24,
        40,
    )
    assert state.diagnostics().throughput_decision_reason == "gpu_budget_guard_hold"


def test_single_mixed_scaling_reports_unfunded_scale_up_at_gpu_ceiling():
    state = _disagg_state(1, 64, 1, 180, enable_load_scaling=True)

    assert state._throughput_single(1.0, 1.0, 1.0, "decode") is None
    assert state._throughput_lower_bound_d == 64
    assert state.diagnostics().throughput_decision_reason == "gpu_budget_guard_hold"


def test_disagg_throughput_holds_scale_down_without_scale_up_at_gpu_floor():
    state = _disagg_state(28, 36, 27, 36, enable_load_scaling=False)

    assert state._throughput_disagg(1.0, 1.0, 1.0) is None
    assert state.diagnostics().throughput_decision_reason == "gpu_budget_guard_hold"


def test_single_throughput_does_not_oscillate_at_tolerance_edge():
    state = _disagg_state(1, 63, 1, 62, enable_load_scaling=False)

    assert state._throughput_single(1.0, 1.0, 1.0, "decode") is None
    assert state.diagnostics().throughput_decision_reason == "gpu_budget_guard_hold"


def test_single_throughput_natural_noop_is_not_reported_as_budget_hold():
    state = _disagg_state(1, 64, 1, 64, enable_load_scaling=False)

    assert state._throughput_single(1.0, 1.0, 1.0, "decode") is None
    assert state.diagnostics().throughput_decision_reason == "no_change"


def test_single_throughput_floor_is_not_inflated_by_minimum_gpu_budget():
    state = _disagg_state(1, 64, 1, 1, enable_load_scaling=True)

    assert state._throughput_single(1.0, 1.0, 1.0, "decode") is None
    assert state._throughput_lower_bound_d == 56


def test_disagg_throughput_uses_only_available_headroom():
    state = _disagg_state(
        24,
        36,
        40,
        36,
        enable_load_scaling=False,
        min_gpus=-1,
    )

    decision = state._throughput_disagg(1.0, 1.0, 1.0)

    assert decision is not None
    assert (decision.num_prefill, decision.num_decode) == (28, 36)


def test_disagg_throughput_caps_scale_down_when_minimum_is_disabled():
    state = _disagg_state(
        20,
        20,
        1,
        20,
        enable_load_scaling=False,
        min_gpus=-1,
    )

    decision = state._throughput_disagg(1.0, 1.0, 1.0)

    assert decision is not None
    assert (decision.num_prefill, decision.num_decode) == (12, 20)


def test_endpoint_and_budget_recovery_take_precedence_over_throughput_cap():
    state = _disagg_state(
        21,
        34,
        30,
        34,
        enable_load_scaling=False,
        prefill_min_endpoint=30,
        decode_min_endpoint=34,
    )

    decision = state._throughput_disagg(1.0, 1.0, 1.0)

    assert decision is not None
    assert (decision.num_prefill, decision.num_decode) == (30, 34)


def test_disagg_endpoint_recovery_overrides_cap_without_minimum_gpu_budget():
    state = _disagg_state(
        1,
        1,
        30,
        1,
        enable_load_scaling=False,
        min_gpus=-1,
        prefill_min_endpoint=30,
        decode_min_endpoint=1,
    )

    decision = state._throughput_disagg(1.0, 1.0, 1.0)

    assert decision is not None
    assert (decision.num_prefill, decision.num_decode) == (30, 1)


def test_endpoint_recovery_can_select_budget_donor():
    state = _disagg_state(
        24,
        40,
        8,
        180,
        enable_load_scaling=False,
        prefill_min_endpoint=30,
        decode_min_endpoint=8,
    )

    decision = state._throughput_disagg(1.0, 1.0, 1.0)

    assert decision is not None
    assert (decision.num_prefill, decision.num_decode) == (30, 34)


def test_mixed_endpoint_recovery_can_select_budget_donor():
    state = _disagg_state(
        24,
        40,
        8,
        180,
        enable_load_scaling=True,
        prefill_min_endpoint=30,
        decode_min_endpoint=8,
    )
    assert state._throughput_disagg(1.0, 1.0, 1.0) is None
    assert (state._throughput_lower_bound_p, state._throughput_lower_bound_d) == (
        30,
        34,
    )

    decision = _load_decision(state, 23, 39)

    assert decision is not None
    assert (decision.num_prefill, decision.num_decode) == (30, 34)
    assert state.diagnostics().load_decision_reason == "gpu_budget_reconcile"


def test_load_only_endpoint_recovery_can_select_budget_donor():
    state = _disagg_state(
        24,
        40,
        8,
        180,
        enable_load_scaling=True,
        prefill_min_endpoint=30,
        decode_min_endpoint=8,
    )
    state._config.enable_throughput_scaling = False

    decision = _load_decision(state, 23, 39)

    assert decision is not None
    assert (decision.num_prefill, decision.num_decode) == (30, 34)
    assert state.diagnostics().load_decision_reason == "gpu_budget_reconcile"


def test_single_endpoint_recovery_overrides_cap_without_minimum_gpu_budget():
    state = _disagg_state(
        1,
        1,
        30,
        1,
        enable_load_scaling=False,
        min_gpus=-1,
        prefill_min_endpoint=30,
    )

    decision = state._throughput_single(1.0, 1.0, 1.0, "prefill")

    assert decision is not None
    assert decision.num_prefill == 30


def test_agg_throughput_uses_replica_cap():
    state = _disagg_state(
        1,
        10,
        1,
        1,
        enable_load_scaling=False,
        min_gpus=-1,
        max_gpus=100,
    )
    state._capabilities = WorkerCapabilities(
        decode=EngineCapabilities(
            gpu_cost_per_replica=1,
            max_num_batched_tokens=4096,
        )
    )
    state._agg_regression = SimpleNamespace(
        find_engine_capacity_rps=lambda **_kwargs: SimpleNamespace(
            rps=1.0,
            ttft_ms=1.0,
            itl_ms=1.0,
            eligible=True,
        )
    )

    decision = state._throughput_agg(100.0, 1.0, 1.0)

    assert decision is not None
    assert decision.num_decode == 18


def test_mixed_scaling_funds_only_atomic_part_of_large_throughput_floor():
    state = _disagg_state(24, 40, 8, 180, enable_load_scaling=True)
    assert state._throughput_disagg(1.0, 1.0, 1.0) is None

    decision = _load_decision(state, 23, 39)

    assert decision is not None
    assert (decision.num_prefill, decision.num_decode) == (23, 41)


def test_capped_throughput_floor_preserves_floor_guard_without_oscillation():
    state = _disagg_state(27, 36, 8, 8, enable_load_scaling=True)
    assert state._throughput_disagg(1.0, 1.0, 1.0) is None
    assert (state._throughput_lower_bound_p, state._throughput_lower_bound_d) == (
        19,
        28,
    )

    assert _load_decision(state, 26, 35) is None
    assert state.diagnostics().load_decision_reason == "gpu_budget_guard_hold"


@pytest.mark.parametrize("gpu_cost_per_replica", [4, 5])
def test_engine_rps_recommendation_is_independent_of_sidecar_cost(
    gpu_cost_per_replica: int,
):
    scaling = _ThroughputScalingHarness()
    scaling._capabilities = WorkerCapabilities(
        prefill=EngineCapabilities(
            num_gpu=4,
            gpu_cost_per_replica=gpu_cost_per_replica,
        )
    )

    replicas = scaling._compute_prefill_replicas(
        demand_rps=2.1,
        isl=1000,
        osl=150,
    )

    assert replicas == 3
