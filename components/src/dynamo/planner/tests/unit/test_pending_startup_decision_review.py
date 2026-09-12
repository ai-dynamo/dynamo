# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup cancellation across real plugin cadence, budget, and merge stages."""

from unittest.mock import Mock

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
    TickInput,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter
from dynamo.planner.plugins.types import (
    ComponentTarget,
    ConstrainStageResponse,
    OverrideResult,
    OverrideType,
    ProposeStageResponse,
    ReconcileStageResponse,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _config(**kwargs):
    return PlannerConfig(
        mode="disagg",
        optimization_target="load",
        served_model_name="test",
        prefill_scale_up_queue_tokens=1000,
        prefill_scale_down_queue_tokens=100,
        decode_scale_up_kv_rate=80,
        decode_scale_down_kv_rate=20,
        min_gpu_budget=-1,
        max_gpu_budget=-1,
        **kwargs,
    )


def _caps():
    return WorkerCapabilities(
        prefill=EngineCapabilities(
            num_gpu=1, max_num_batched_tokens=1024, max_kv_tokens=1000
        ),
        decode=EngineCapabilities(
            num_gpu=1, max_num_batched_tokens=1024, max_kv_tokens=1000
        ),
    )


def _counts():
    return WorkerCounts(
        ready_num_prefill=1,
        ready_num_decode=1,
        expected_num_prefill=1,
        expected_num_decode=2,
        prefill_scaling_in_progress=True,
        decode_scaling_in_progress=True,
        pending_num_decode=1,
    )


def _obs(ready=1, kv=0):
    return FpmObservations(
        decode={
            (f"d{i}", 0): ForwardPassMetrics(
                worker_id=f"d{i}",
                scheduled_requests=ScheduledRequestMetrics(sum_decode_kv_tokens=kv),
            )
            for i in range(ready)
        },
    )


@pytest.mark.asyncio
async def test_idle_plugin_ticks_preserve_startup_observation_window():
    config = _config(
        load_adjustment_interval_seconds=30, scheduling={"scale_interval_seconds": 5}
    )
    adapter = OrchestratorEngineAdapter(config, _caps(), clock=VirtualClock())
    tick = adapter.initial_tick(0)
    for _ in range(12):
        effects = await adapter.tick(
            tick,
            TickInput(
                now_s=tick.at_s, worker_counts=_counts(), fpm_observations=_obs()
            ),
        )
        if tick.at_s < 60:
            assert effects.scale_to is None
        else:
            assert effects.scale_to is not None
            assert effects.scale_to.num_decode == 1
        tick = effects.next_tick


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role,floor,expected_at_60",
    [("prefill", 1, 1), ("decode", 1, 1), ("decode", 2, None)],
)
async def test_interleaved_plugin_floor_only_resets_conflicting_candidate(
    role, floor, expected_at_60
):
    config = _config(
        load_adjustment_interval_seconds=30, scheduling={"scale_interval_seconds": 5}
    )
    adapter = OrchestratorEngineAdapter(config, _caps(), clock=VirtualClock())

    class Floor:
        async def Propose(self, req):
            return ProposeStageResponse(
                override=OverrideResult(
                    targets=[
                        ComponentTarget(
                            sub_component_type=role,
                            replicas=floor,
                            type=OverrideType.AT_LEAST,
                        )
                    ]
                )
            )

    adapter._orchestrator.register_internal("other_floor", "propose", 20, Floor())
    tick = adapter.initial_tick(0)
    for _ in range(12):
        effects = await adapter.tick(
            tick,
            TickInput(
                now_s=tick.at_s, worker_counts=_counts(), fpm_observations=_obs()
            ),
        )
        if tick.at_s < 60:
            assert effects.scale_to is None
        tick = effects.next_tick
    assert (effects.scale_to.num_decode if effects.scale_to else None) == expected_at_60


@pytest.mark.asyncio
@pytest.mark.parametrize("veto", ["hold", "missing_fpm"])
async def test_fresh_evaluated_hold_or_missing_fpm_restarts_window(veto):
    config = _config(
        load_adjustment_interval_seconds=30, scheduling={"scale_interval_seconds": 5}
    )
    adapter = OrchestratorEngineAdapter(config, _caps(), clock=VirtualClock())
    tick = adapter.initial_tick(0)
    for _ in range(24):
        obs = _obs()
        if tick.at_s == 60:
            obs = _obs(kv=500) if veto == "hold" else FpmObservations()
        effects = await adapter.tick(
            tick,
            TickInput(now_s=tick.at_s, worker_counts=_counts(), fpm_observations=obs),
        )
        if tick.at_s < 120:
            assert effects.scale_to is None
        else:
            assert effects.scale_to is not None
            assert effects.scale_to.num_decode == 1
        tick = effects.next_tick


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "floor,ceiling,expected", [(5, -1, 2), (6, -1, None), (-1, 4, None)]
)
async def test_load_budget_preserves_pending_peer(floor, ceiling, expected):
    config = _config()
    config.min_gpu_budget, config.max_gpu_budget = floor, ceiling
    counts = _counts()
    counts.ready_num_decode = counts.expected_num_decode = 3
    counts.expected_num_prefill = 3
    counts.pending_num_prefill, counts.pending_num_decode = 2, 0
    adapter = OrchestratorEngineAdapter(config, _caps(), clock=VirtualClock())
    tick = adapter.initial_tick(0)
    for _ in range(2):
        effects = await adapter.tick(
            tick,
            TickInput(
                now_s=tick.at_s, worker_counts=counts, fpm_observations=_obs(ready=3)
            ),
        )
        tick = effects.next_tick
    assert (effects.scale_to.num_decode if effects.scale_to else None) == expected
    if effects.scale_to:
        assert effects.scale_to.num_prefill is None


@pytest.mark.parametrize("requested_prefill", [3, 4])
def test_throughput_floor_counts_preserved_pending_peer(requested_prefill):
    config = _config()
    config.enable_load_scaling = False
    config.enable_throughput_scaling = True
    config.min_gpu_budget = 5
    state = PlannerScalingState(config, _caps())
    counts = _counts()
    counts.ready_num_decode = counts.expected_num_decode = 3
    counts.expected_num_prefill = 3
    counts.pending_num_prefill, counts.pending_num_decode = 2, 0
    state.observe_worker_counts(counts)
    state._compute_prefill_replicas = Mock(return_value=requested_prefill)
    state._compute_decode_replicas = Mock(return_value=1)
    decision = state._throughput_disagg(1, 100, 100)
    assert decision is not None
    # The floor limits D to 2 while the up-requested P keeps desired=3.
    assert decision.num_prefill is None
    assert decision.num_decode == 2


def _target(replicas, kind=OverrideType.SET):
    return OverrideResult(
        targets=[
            ComponentTarget(sub_component_type="decode", replicas=replicas, type=kind)
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["capped_up", "floor_only", "reconcile_down"])
async def test_external_pipeline_preserves_reduction_provenance(policy):
    config = _config()
    config.enable_load_scaling = False
    adapter = OrchestratorEngineAdapter(config, _caps(), clock=VirtualClock())

    class Proposer:
        async def Propose(self, req):
            return ProposeStageResponse(
                override=_target(1, OverrideType.AT_LEAST)
                if policy == "floor_only"
                else _target(3)
            )

    class Constraint:
        async def Constrain(self, req):
            return ConstrainStageResponse(override=_target(1, OverrideType.AT_MOST))

    class Reconciler:
        async def Reconcile(self, req):
            return ReconcileStageResponse(override=_target(1))

    adapter._orchestrator.register_internal("external", "propose", 0, Proposer())
    if policy == "capped_up":
        adapter._orchestrator.register_internal("ceiling", "constrain", 0, Constraint())
    elif policy == "reconcile_down":
        adapter._orchestrator.register_internal(
            "down_policy", "reconcile", 0, Reconciler()
        )
    tick = adapter.initial_tick(0)
    for index in range(3):
        effects = await adapter.tick(
            tick, TickInput(now_s=tick.at_s, worker_counts=_counts())
        )
        if policy == "reconcile_down" and index > 0:
            assert effects.scale_to is not None
            assert effects.scale_to.num_decode == 1
        else:
            assert effects.scale_to is None
        tick = effects.next_tick
