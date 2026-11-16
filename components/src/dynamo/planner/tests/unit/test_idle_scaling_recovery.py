# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    ScheduledRequestMetrics,
)
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.state_machine import PlannerScalingState
from dynamo.planner.core.types import (
    EngineCapabilities,
    FpmObservations,
    TrafficObservation,
    WorkerCapabilities,
    WorkerCounts,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.planner,
]


def _config(mode, target, throughput, speculative_nextn=0):
    return PlannerConfig(
        mode=mode,
        optimization_target=target,
        enable_load_scaling=not throughput,
        enable_throughput_scaling=throughput,
        prefill_scale_up_queue_tokens=1024,
        prefill_scale_down_queue_tokens=0,
        decode_scale_up_kv_rate=90,
        decode_scale_down_kv_rate=10,
        min_endpoint=1,
        max_gpu_budget=16,
        speculative_nextn=speculative_nextn,
    )


def _caps():
    return WorkerCapabilities(
        prefill=EngineCapabilities(
            num_gpu=1, context_length=4096, max_num_batched_tokens=1024
        ),
        decode=EngineCapabilities(
            num_gpu=1, max_kv_tokens=65536, max_num_batched_tokens=1024
        ),
    )


def _inventory(prefill, decode):
    return WorkerCounts(
        ready_num_prefill=prefill,
        expected_num_prefill=prefill,
        ready_num_decode=decode,
        expected_num_decode=decode,
    )


def _idle_stats(count):
    return {
        (str(index), 0): ForwardPassMetrics(worker_id=str(index))
        for index in range(count)
    }


@pytest.fixture
def cold_model(monkeypatch):
    model = MagicMock()
    model.has_sufficient_data.return_value = False
    model.find_engine_capacity_rps.return_value = None
    monkeypatch.setattr(
        "dynamo.planner.core.state_machine.PlannerEnginePerfModel", lambda **_: model
    )
    return model


def _drive_idle_pool_after_restart(config):
    mode = config.mode
    state = PlannerScalingState(config, _caps())
    prefill = 4 if mode in ("prefill", "disagg") else 0
    decode = 4 if mode in ("decode", "disagg", "agg") else 0
    for tick in range(5):
        if tick == 1:
            # Restart reconstructs state from acknowledged deployment counts.
            state = PlannerScalingState(config, _caps())
        state.observe_worker_counts(_inventory(prefill, decode))
        state.begin_tick()
        if config.enable_throughput_scaling:
            decision = state.advance_throughput_from_prediction(
                TrafficObservation(duration_s=60, num_req=0, isl=0, osl=0),
                predicted_num_req=0,
                predicted_isl=0,
                predicted_osl=0,
                predicted_kv_hit_rate=None,
            )
        else:
            decision = state.advance_load(
                FpmObservations(
                    prefill=_idle_stats(prefill), decode=_idle_stats(decode)
                )
            )
        if decision is not None:
            if decision.num_prefill is not None:
                prefill = decision.num_prefill
            if decision.num_decode is not None:
                decode = decision.num_decode
    return prefill, decode


@pytest.mark.parametrize("mode", ["prefill", "decode", "disagg", "agg"])
@pytest.mark.parametrize(
    ("target", "throughput", "nextn"),
    [("sla", True, 0), ("sla", False, 3)],
)
def test_idle_pool_reaches_minimum_after_restart(
    mode, target, throughput, nextn, cold_model
):
    prefill, decode = _drive_idle_pool_after_restart(
        _config(mode, target, throughput, nextn)
    )
    assert prefill == (1 if mode in ("prefill", "disagg") else 0)
    assert decode == (1 if mode in ("decode", "disagg", "agg") else 0)


@pytest.mark.parametrize("mode", ["prefill", "decode", "disagg", "agg"])
@pytest.mark.parametrize("throughput", [False, True])
def test_model_independent_idle_scaling_retains_a_worker_with_zero_floor(
    mode, throughput, cold_model
):
    config = _config(mode, "sla", throughput).model_copy(update={"min_endpoint": 0})
    prefill, decode = _drive_idle_pool_after_restart(config)
    assert prefill == (1 if mode in ("prefill", "disagg") else 0)
    assert decode == (1 if mode in ("decode", "disagg", "agg") else 0)


def test_missing_fpm_and_busy_workers_do_not_take_idle_shortcut(cold_model):
    state = PlannerScalingState(_config("decode", "sla", False), _caps())
    state.observe_worker_counts(_inventory(0, 2))
    assert state.advance_load(FpmObservations()) is None
    busy = _idle_stats(2)
    busy[("0", 0)] = ForwardPassMetrics(
        worker_id="0",
        scheduled_requests=ScheduledRequestMetrics(
            num_decode_requests=1, sum_decode_kv_tokens=512
        ),
    )
    assert state.advance_load(FpmObservations(decode=busy)) is None


def test_idle_rollout_waits_for_authoritative_inventory(cold_model):
    state = PlannerScalingState(_config("decode", "sla", False), _caps())
    counts = _inventory(0, 3)
    counts.expected_num_decode = 4
    counts.decode_scaling_in_progress = True
    state.observe_worker_counts(counts)
    assert state.advance_load(FpmObservations(decode=_idle_stats(3))) is None
    state.observe_worker_counts(_inventory(0, 4))
    decision = state.advance_load(FpmObservations(decode=_idle_stats(4)))
    assert decision.num_decode == 3


def test_unknown_prediction_does_not_shrink_a_cold_pool(cold_model):
    state = PlannerScalingState(_config("decode", "sla", True), _caps())
    state.observe_worker_counts(_inventory(0, 4))
    assert (
        state.advance_throughput_from_prediction(
            TrafficObservation(duration_s=60, num_req=0, isl=0, osl=0),
            predicted_num_req=None,
            predicted_isl=None,
            predicted_osl=None,
            predicted_kv_hit_rate=None,
        )
        is None
    )
