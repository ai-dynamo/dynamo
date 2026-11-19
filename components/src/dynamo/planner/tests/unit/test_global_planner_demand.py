# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retained global demand follows valid unchanged Local Planner recommendations."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)
from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.connectors.global_planner import GlobalPlannerConnector
from dynamo.planner.connectors.protocol import ScaleResponse, ScaleStatus
from dynamo.planner.core.perf_model.ais_adapter import PlannerEngineCapacity
from dynamo.planner.core.state_machine import PlannerScalingState
from dynamo.planner.core.types import (
    EngineCapabilities,
    FpmObservations,
    ScheduledTick,
    TickInput,
    TrafficObservation,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.merge.types import ComponentKey
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter
from dynamo.planner.plugins.orchestrator.pipeline import PipelineOutcome
from dynamo.planner.plugins.types import ComponentTarget, ScalingProposal

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# "managed" is a Global Planner environment with managed allocation enabled;
# plain "global-planner" is the legacy arbitration mode.
_ENVIRONMENTS = ["managed", "global-planner", "kubernetes", "virtual"]


def _config(mode="decode", environment="managed", **kwargs):
    managed = environment == "managed"
    return PlannerConfig(
        mode=mode,
        environment="global-planner" if managed else environment,
        global_planner_managed_allocation=managed,
        global_planner_namespace="global",
        served_model_name="test",
        optimization_target="load",
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
        prefill=EngineCapabilities(num_gpu=1, max_num_batched_tokens=1000),
        decode=EngineCapabilities(
            num_gpu=1, max_num_batched_tokens=1000, max_kv_tokens=1000
        ),
    )


def _counts(mode, **kwargs):
    return WorkerCounts(
        ready_num_prefill=2 if mode in ("prefill", "disagg") else None,
        expected_num_prefill=2 if mode in ("prefill", "disagg") else None,
        ready_num_decode=2 if mode in ("decode", "agg", "disagg") else None,
        expected_num_decode=2 if mode in ("decode", "agg", "disagg") else None,
        **kwargs,
    )


def _fpm(mode, *, high=False):
    return FpmObservations(
        prefill={
            (f"p{index}", 0): ForwardPassMetrics(
                worker_id=f"p{index}",
                queued_requests=QueuedRequestMetrics(
                    sum_prefill_tokens=1200 if high else 500
                ),
            )
            for index in range(2)
        }
        if mode in ("prefill", "disagg")
        else None,
        decode={
            (f"d{index}", 0): ForwardPassMetrics(
                worker_id=f"d{index}",
                scheduled_requests=ScheduledRequestMetrics(
                    sum_decode_kv_tokens=900 if high else 500
                ),
            )
            for index in range(2)
        }
        if mode in ("decode", "agg", "disagg")
        else None,
    )


async def _load_tick(adapter, mode, *, now=30.0, high=False, counts=None, fpm=None):
    return await adapter.tick(
        ScheduledTick(
            at_s=now,
            at_monotonic_s=now,
            run_load_scaling=True,
            run_throughput_scaling=False,
        ),
        TickInput(
            now_s=now,
            worker_counts=counts or _counts(mode),
            fpm_observations=fpm if fpm is not None else _fpm(mode, high=high),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["prefill", "decode", "agg", "disagg"])
@pytest.mark.parametrize("environment", _ENVIRONMENTS)
async def test_load_pipeline_publishes_valid_unchanged_demand_only_when_managed(
    mode, environment
):
    adapter = OrchestratorEngineAdapter(
        _config(mode, environment), _caps(), clock=VirtualClock()
    )

    effects = await _load_tick(adapter, mode)

    if environment != "managed":
        assert effects.scale_to is None
    else:
        assert effects.scale_to is not None
        assert effects.scale_to.num_prefill == (
            2 if mode in ("prefill", "disagg") else None
        )
        assert effects.scale_to.num_decode == (
            2 if mode in ("decode", "agg", "disagg") else None
        )


@pytest.mark.asyncio
async def test_healthy_hold_cancels_previously_retained_growth(monkeypatch):
    monkeypatch.setenv("DYN_PARENT_DGD_K8S_NAME", "batch")
    monkeypatch.setenv("POD_NAMESPACE", "workers")
    connector = GlobalPlannerConnector(MagicMock(), "batch-runtime", "global")
    connector.remote_client = AsyncMock()
    connector.remote_client.send_scale_request.return_value = ScaleResponse(
        status=ScaleStatus.SCALING,
        message="Demand retained",
        current_replicas={"decode": 2},
    )
    adapter = OrchestratorEngineAdapter(_config(), _caps(), clock=VirtualClock())

    growing = await _load_tick(adapter, "decode", high=True)
    still_growing = await _load_tick(adapter, "decode", now=90.0, high=True)
    hold = await _load_tick(adapter, "decode", now=150.0)
    for effects in (growing, still_growing, hold):
        assert effects.scale_to is not None
        await connector.set_component_replicas(
            [
                TargetReplica(
                    sub_component_type=SubComponentType.DECODE,
                    component_name="Decode",
                    desired_replicas=effects.scale_to.num_decode,
                )
            ]
        )

    submitted = [
        call.args[0].target_replicas[0].desired_replicas
        for call in connector.remote_client.send_scale_request.call_args_list
    ]
    assert submitted == [3, 3, 2]
    assert connector.last_desired_replicas[0].desired_replicas == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["prefill", "decode", "agg", "disagg"])
async def test_valid_sla_consolidation_hold_replaces_previous_growth(mode, monkeypatch):
    config = _config(mode)
    config.optimization_target = "sla"
    config.ttft_ms = 50
    config.itl_ms = 50
    regression = MagicMock()
    regression.has_valid_engine_limits.return_value = True
    regression.has_sufficient_data.return_value = True
    regression.avg_isl = 100
    regression.query_groups.side_effect = lambda stats: [
        (worker_id, [fpm]) for (worker_id, _), fpm in stats.items()
    ]
    regression.estimate_scheduled_decode_itl.return_value = 0.08
    regression.estimate_queued_prefill_time.return_value = 0.08
    monkeypatch.setattr(
        "dynamo.planner.core.state_machine.PlannerEnginePerfModel",
        lambda *args, **kwargs: regression,
    )
    adapter = OrchestratorEngineAdapter(config, _caps(), clock=VirtualClock())

    growth = await _load_tick(adapter, mode)
    regression.estimate_scheduled_decode_itl.return_value = 0.04
    regression.estimate_queued_prefill_time.side_effect = lambda *args, **kwargs: (
        0.01
        if kwargs.get("queue_scale") == 0
        else 0.08
        if kwargs.get("queue_scale", 1) > 1
        else 0.04
    )
    hold = await _load_tick(adapter, mode, now=90.0)

    assert growth.scale_to is not None
    assert hold.scale_to is not None
    if mode in ("prefill", "disagg"):
        assert growth.scale_to.num_prefill == 3
        assert hold.scale_to.num_prefill == 2
    if mode in ("decode", "agg", "disagg"):
        assert growth.scale_to.num_decode == 3
        assert hold.scale_to.num_decode == 2
    assert hold.diagnostics.load_decision_reason == "scale_down_refused_consolidation"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["prefill", "decode", "agg", "disagg"])
@pytest.mark.parametrize(
    "invalid", ["missing_fpm", "scaling", "startup", "missing_metadata"]
)
async def test_invalid_load_observation_does_not_replace_retained_demand(mode, invalid):
    caps = _caps()
    counts = _counts(mode)
    observations = _fpm(mode)
    config = _config(mode)
    if invalid == "missing_fpm":
        observations = FpmObservations()
    elif invalid == "scaling":
        counts.prefill_scaling_in_progress = True
        counts.decode_scaling_in_progress = True
    elif invalid == "startup":
        counts.prefill_scaling_in_progress = True
        counts.decode_scaling_in_progress = True
        if mode in ("prefill", "disagg"):
            counts.pending_num_prefill = 1
        if mode in ("decode", "agg", "disagg"):
            counts.pending_num_decode = 1
    else:
        caps.decode.max_kv_tokens = None
        caps.prefill.context_length = None
        config.optimization_target = "throughput"
    adapter = OrchestratorEngineAdapter(config, caps, clock=VirtualClock())

    effects = await _load_tick(adapter, mode, counts=counts, fpm=observations)

    assert effects.scale_to is None


@pytest.mark.asyncio
async def test_disagg_valid_hold_does_not_manufacture_demand_for_unobserved_peer():
    adapter = OrchestratorEngineAdapter(
        _config("disagg"), _caps(), clock=VirtualClock()
    )
    observations = _fpm("disagg")
    observations.prefill = None

    effects = await _load_tick(adapter, "disagg", fpm=observations)

    assert effects.scale_to is not None
    assert effects.scale_to.num_prefill is None
    assert effects.scale_to.num_decode == 2


@pytest.mark.parametrize("mode", ["prefill", "decode", "agg", "disagg"])
@pytest.mark.parametrize("environment", _ENVIRONMENTS)
@pytest.mark.parametrize("model_ready", [True, False])
def test_throughput_publishes_unchanged_valid_model_result(
    mode, environment, model_ready, monkeypatch
):
    config = _config(mode, environment)
    config.optimization_target = "sla"
    config.enable_load_scaling = False
    config.enable_throughput_scaling = True
    regression = MagicMock()
    regression.has_valid_engine_limits.return_value = True
    regression.find_engine_capacity_rps.return_value = (
        PlannerEngineCapacity(rps=1.0) if model_ready else None
    )
    # Exercise demand projection with controlled capacity estimates; initializing
    # an external performance model is outside this contract test.
    monkeypatch.setattr(
        "dynamo.planner.core.state_machine.PlannerEnginePerfModel",
        lambda *args, **kwargs: regression,
    )
    state = PlannerScalingState(config, _caps())
    state.observe_worker_counts(_counts(mode))

    decision = state.advance_throughput_from_prediction(
        TrafficObservation(duration_s=10, num_req=20, isl=100, osl=100),
        predicted_num_req=20,
        predicted_isl=100,
        predicted_osl=100,
        predicted_kv_hit_rate=0,
    )

    if environment == "managed" and model_ready:
        assert decision is not None
        assert decision.num_prefill == (2 if mode in ("prefill", "disagg") else None)
        assert decision.num_decode == (
            2 if mode in ("decode", "agg", "disagg") else None
        )
    else:
        assert decision is None


@pytest.mark.parametrize(
    "action", ["skip_short_circuit", "skip_no_targets", "skip_tick_timeout"]
)
def test_skipped_pipeline_never_publishes_demand(action):
    adapter = OrchestratorEngineAdapter(_config(), _caps(), clock=VirtualClock())
    outcome = PipelineOutcome(
        execute_action=action,
        final_proposal=ScalingProposal(
            targets=[ComponentTarget(sub_component_type="decode", replicas=2)]
        ),
        proposed_components=frozenset({ComponentKey(sub_component_type="decode")}),
    )
    assert adapter._project_scale_to(outcome, _counts("decode")) is None


def test_merged_baseline_and_rollout_echo_do_not_cancel_demand():
    adapter = OrchestratorEngineAdapter(_config(), _caps(), clock=VirtualClock())
    outcome = PipelineOutcome(
        execute_action="apply",
        final_proposal=ScalingProposal(
            targets=[ComponentTarget(sub_component_type="decode", replicas=2)]
        ),
    )
    assert adapter._project_scale_to(outcome, _counts("decode")) is None

    outcome.proposed_components = frozenset({ComponentKey(sub_component_type="decode")})
    counts = _counts("decode", decode_scaling_in_progress=True)
    counts.expected_num_decode = 3
    assert adapter._project_scale_to(outcome, counts) is None


def test_managed_allocation_requires_global_planner_environment():
    with pytest.raises(ValueError, match="requires environment='global-planner'"):
        PlannerConfig(
            environment="kubernetes",
            global_planner_managed_allocation=True,
            served_model_name="test",
        )
