# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cancellation of startup capacity must not race a drain or manufacture demand."""

from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from unittest.mock import Mock

import pytest
from kubernetes import client

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    ScheduledRequestMetrics,
)
from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.connectors.clients.kubernetes_api import KubernetesAPI
from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.core.state_machine import PlannerScalingState
from dynamo.planner.core.types import (
    EngineCapabilities,
    FpmObservations,
    TickInput,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.environment.base import PlannerEnvironmentImpl
from dynamo.planner.plugins.builtins.observe import EnvironmentObservePlugin
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


def _config(mode="disagg", **kwargs):
    return PlannerConfig(
        mode=mode,
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


def _counts(ready=1, pending=1):
    return WorkerCounts(
        ready_num_prefill=1,
        ready_num_decode=ready,
        prefill_scaling_in_progress=True,
        decode_scaling_in_progress=True,
        pending_num_decode=pending,
    )


def _obs(ready=1, kv=0):
    return FpmObservations(
        prefill={("p", 0): ForwardPassMetrics(worker_id="p")},
        decode={
            (f"d{i}", 0): ForwardPassMetrics(
                worker_id=f"d{i}",
                scheduled_requests=ScheduledRequestMetrics(sum_decode_kv_tokens=kv),
            )
            for i in range(ready)
        },
    )


def _outcome(p=None, d=None, explicit=("prefill", "decode")):
    return PipelineOutcome(
        execute_action="apply",
        final_proposal=ScalingProposal(
            targets=[
                ComponentTarget(sub_component_type=role, replicas=value)
                for role, value in (("prefill", p), ("decode", d))
                if value is not None
            ]
        ),
        proposed_components=frozenset(
            ComponentKey(sub_component_type=role) for role in explicit
        ),
    )


@pytest.mark.parametrize("mode", ["disagg", "decode", "agg"])
@pytest.mark.parametrize("ready", [1, 3])
def test_load_can_cancel_startup_and_drain_serving_workers(mode, ready):
    state = PlannerScalingState(_config(mode), _caps())
    state.observe_worker_counts(_counts(ready))
    decision = state.advance_load(_obs(ready))
    assert decision is not None
    assert decision.num_decode == max(1, ready - 1)
    assert decision.num_prefill is None


@pytest.mark.parametrize("kv", [500, 950])
def test_hold_or_scale_up_signal_does_not_cancel_startup(kv):
    state = PlannerScalingState(_config(), _caps())
    state.observe_worker_counts(_counts())
    assert state.advance_load(_obs(kv=kv)) is None


def test_unknown_scaling_or_drain_stays_blocked():
    state = PlannerScalingState(_config(), _caps())
    state.observe_worker_counts(_counts(ready=3, pending=0))
    assert state.advance_load(_obs(ready=3)) is None


def test_missing_fpm_does_not_cancel_pending_peer():
    state = PlannerScalingState(_config(), _caps())
    state.observe_worker_counts(_counts())
    obs = _obs()
    obs.decode = None
    assert state.advance_load(obs) is None


def test_startup_projection_requires_sustained_explicit_down_signal():
    config = _config()
    clock = VirtualClock()
    adapter = OrchestratorEngineAdapter(config, _caps(), clock=clock)
    counts = _counts()
    down = _outcome(d=1)
    assert adapter._project_scale_to(down, counts) is None
    # Repeated fast plugin ticks cannot accelerate the stabilization interval.
    for _ in range(5):
        assert adapter._project_scale_to(down, counts) is None
    clock.advance(config.load_adjustment_interval_seconds)
    # A renewed up signal resets stabilization; it must not become cancellation.
    assert adapter._project_scale_to(_outcome(d=3), counts) is None
    assert adapter._project_scale_to(down, counts) is None
    clock.advance(config.load_adjustment_interval_seconds)
    decision = adapter._project_scale_to(down, counts)
    assert decision is not None and decision.num_decode == 1
    assert decision.num_prefill is None
    # An omitted role's baseline echo is never a cancellation request.
    assert adapter._project_scale_to(_outcome(p=1, d=1, explicit=()), counts) is None


@pytest.mark.parametrize(
    "floor,ceiling,target,allowed",
    [
        (3, -1, 1, False),  # Ready-equal cancellation would cross GPU floor.
        (-1, 2, 1, True),
        (-1, 2, 5, False),  # Up proposal must not be clamped into a down proposal.
    ],
)
def test_startup_gpu_bounds(floor, ceiling, target, allowed):
    config = _config()
    config.min_gpu_budget, config.max_gpu_budget = floor, ceiling
    clock = VirtualClock()
    adapter = OrchestratorEngineAdapter(config, _caps(), clock=clock)
    counts, outcome = _counts(), _outcome(d=target)
    assert adapter._project_scale_to(outcome, counts) is None
    clock.advance(config.load_adjustment_interval_seconds)
    assert (adapter._project_scale_to(outcome, counts) is not None) is allowed


def _deployment():
    return {
        "metadata": {"name": "qwen", "generation": 2},
        "spec": {
            "components": [
                {"name": "p", "type": "prefill", "replicas": 1},
                {"name": "d", "type": "decode", "replicas": 2},
            ]
        },
        "status": {
            "observedGeneration": 2,
            "conditions": [{"type": "Ready", "status": "False"}],
            "components": {
                "p": {"replicas": 1, "updatedReplicas": 1, "readyReplicas": 1},
                "d": {"replicas": 2, "updatedReplicas": 2, "readyReplicas": 1},
            },
        },
    }


def _pods():
    return [
        client.V1Pod(
            metadata=client.V1ObjectMeta(
                name=name, labels={"nvidia.com/dynamo-component": component}
            ),
            status=client.V1PodStatus(phase=phase),
        )
        for name, component, phase in [
            ("p0", "p", "Running"),
            ("d0", "d", "Running"),
            ("d1", "d", "Pending"),
        ]
    ]


def _connector(deployment, pods):
    api = KubernetesAPI.__new__(KubernetesAPI)
    api.get_graph_deployment = Mock(side_effect=lambda _: deepcopy(deployment))
    api.list_pods_for_graph = Mock(return_value=pods)
    api.update_graph_replicas = Mock()
    api.get_service_replica_target = Mock(
        side_effect=lambda _, name: connector._startup_scale_down_targets[name]
    )
    connector = KubernetesConnector.__new__(KubernetesConnector)
    connector.graph_deployment_name = "qwen"
    connector.kube_api = api
    connector.raise_not_ready = False
    connector._startup_scale_down_lock = Lock()
    connector._startup_scale_down_targets = {}
    connector._startup_read_warnings = set()
    return connector


@pytest.mark.parametrize(
    "unsafe",
    [
        "drain",
        "scale_down",
        "rollout",
        "failed_rollout",
        "stale",
        "old_template",
        "failed_pod",
    ],
)
def test_inventory_refuses_to_label_unsafe_states_as_startup(unsafe):
    deployment, pods = _deployment(), _pods()
    if unsafe == "drain":
        pods[0].metadata.deletion_timestamp = datetime.now(timezone.utc)
    elif unsafe == "scale_down":
        deployment["spec"]["components"][1]["replicas"] = 1
    elif unsafe in ("rollout", "failed_rollout"):
        deployment["status"]["rollingUpdate"] = {
            "phase": "Failed" if unsafe == "failed_rollout" else "InProgress"
        }
    elif unsafe == "stale":
        deployment["metadata"]["generation"] += 1
    elif unsafe == "old_template":
        deployment["status"]["components"]["d"]["updatedReplicas"] = 1
    else:
        pods[2].status.phase = "Failed"
    connector = _connector(deployment, pods)
    inventory = connector._get_worker_inventory_sync("p", "d")
    assert not inventory.startup_in_progress


@pytest.mark.asyncio
async def test_connector_cancels_startup_and_holds_until_drain_finishes():
    deployment, pods = _deployment(), _pods()
    connector = _connector(deployment, pods)
    inventory = await connector.get_worker_inventory("p", "d")
    assert inventory.pending_num_decode == 1
    target = [
        TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=1)
    ]
    await connector.set_component_replicas(target, blocking=False)
    connector.kube_api.update_graph_replicas.assert_called_once_with("qwen", "d", 1)
    # Async DGDSA application has not reached the DGD yet: no repeat writes.
    await connector.set_component_replicas(target, blocking=False)
    assert not (await connector.get_worker_inventory("p", "d")).startup_in_progress
    connector.kube_api.update_graph_replicas.assert_called_once()
    deployment["spec"]["components"][1]["replicas"] = 1
    deployment["status"]["components"]["d"] = {
        "replicas": 1,
        "updatedReplicas": 1,
        "readyReplicas": 1,
    }
    deployment["status"]["conditions"][0]["status"] = "True"
    # Even if Ready/counts have converged, a deleting Pod still blocks reversal.
    pods[2].metadata.deletion_timestamp = datetime.now(timezone.utc)
    up = [TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=2)]
    await connector.set_component_replicas(up, blocking=False)
    connector.kube_api.update_graph_replicas.assert_called_once()
    pods.pop()
    await connector.set_component_replicas(up, blocking=False)
    assert connector.kube_api.update_graph_replicas.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [-1, 3])
async def test_unready_connector_rejects_non_reductions_and_negative_targets(target):
    connector = _connector(_deployment(), _pods())
    await connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE, desired_replicas=target
            )
        ],
        blocking=False,
    )
    connector.kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_builtin_pipeline_cancels_pending_decode_after_observation_window():
    config = _config()
    clock = VirtualClock()
    adapter = OrchestratorEngineAdapter(config, _caps(), clock=clock)
    tick = adapter.initial_tick(start_s=0.0)
    for index in range(2):
        assert tick is not None
        effects = await adapter.tick(
            tick,
            TickInput(
                now_s=tick.at_s,
                worker_counts=_counts(),
                fpm_observations=_obs(),
            ),
        )
        if index == 0:
            assert effects.scale_to is None
        else:
            assert effects.scale_to is not None
            assert effects.scale_to.num_decode == 1
            assert effects.scale_to.num_prefill is None
        tick = effects.next_tick


@pytest.mark.parametrize("mode", ["decode", "agg"])
def test_startup_easy_consolidation_does_not_immediately_require_scale_up(mode):
    config = _config(mode)
    config.optimization_target = "throughput"
    state = PlannerScalingState(config, _caps())
    state.observe_worker_counts(_counts(ready=2))
    # Each serving worker is under the 60% down threshold, but merging them
    # would use 110% KV capacity. Only cancel the pending third worker.
    decision = state.advance_load(_obs(ready=2, kv=550))
    assert decision is not None and decision.num_decode == 2
    # Once the pending worker is gone, unchanged load must continue to hold.
    state.observe_worker_counts(WorkerCounts(ready_num_decode=2, expected_num_decode=2))
    for _ in range(5):
        decision = state.advance_load(_obs(ready=2, kv=550))
        assert decision is None or decision.num_decode == 2


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_sla_one_ready_worker_can_cancel_startup_without_division_by_zero(role):
    config = PlannerConfig(
        mode=role,
        optimization_target="sla",
        enable_throughput_scaling=False,
        enable_load_scaling=True,
        served_model_name="test",
        min_gpu_budget=-1,
        max_gpu_budget=-1,
    )
    state = PlannerScalingState(config, _caps())
    counts = _counts()
    if role == "prefill":
        counts.pending_num_prefill, counts.pending_num_decode = 1, 0
    state.observe_worker_counts(counts)
    regression = Mock()
    regression.has_sufficient_data.return_value = True
    regression.avg_isl = 100
    obs = _obs()
    regression.query_groups.return_value = [
        ("worker", list((obs.prefill if role == "prefill" else obs.decode).values()))
    ]
    regression.estimate_queued_prefill_time.side_effect = (
        lambda *args, **kwargs: 0.005 if kwargs.get("queue_scale") == 0 else 0.01
    )
    regression.estimate_scheduled_decode_itl.return_value = 0.001
    if role == "prefill":
        state._prefill_regression = regression
    else:
        state._decode_regression = regression
    decision = state.advance_load(obs)
    assert decision is not None
    assert (decision.num_prefill if role == "prefill" else decision.num_decode) == 1


def test_pending_peer_capacity_is_preserved_and_charged_to_budget():
    config = _config()
    config.min_gpu_budget = config.max_gpu_budget = 4
    clock = VirtualClock()
    adapter = OrchestratorEngineAdapter(config, _caps(), clock=clock)
    counts = _counts()
    counts.pending_num_prefill = 2
    outcome = _outcome(p=1, d=1, explicit=("decode",))
    # p keeps desired=3, d cancels to 1: total=4, although only 2 are Ready.
    assert adapter._project_scale_to(outcome, counts) is None
    clock.advance(config.load_adjustment_interval_seconds)
    decision = adapter._project_scale_to(outcome, counts)
    assert (
        decision is not None
        and decision.num_prefill is None
        and decision.num_decode == 1
    )


@pytest.mark.parametrize("minimum,ready,expected", [(2, 3, 2), (2, 1, None)])
def test_startup_projection_preserves_component_minimum(minimum, ready, expected):
    config = _config(decode_min_endpoint=minimum)
    clock = VirtualClock()
    adapter = OrchestratorEngineAdapter(config, _caps(), clock=clock)
    counts, outcome = _counts(ready), _outcome(d=0)
    adapter._project_scale_to(outcome, counts)
    clock.advance(config.load_adjustment_interval_seconds)
    decision = adapter._project_scale_to(outcome, counts)
    assert (decision.num_decode if decision else None) == expected


@pytest.mark.asyncio
async def test_connector_rechecks_for_drain_between_observation_and_write():
    deployment, pods = _deployment(), _pods()
    connector = _connector(deployment, pods)
    assert (await connector.get_worker_inventory("p", "d")).startup_in_progress
    pods[0].metadata.deletion_timestamp = datetime.now(timezone.utc)
    await connector.set_component_replicas(
        [TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=1)],
        blocking=False,
    )
    connector.kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_environment_passes_startup_inventory_to_observer_and_clears_it_on_drain():
    deployment, pods = _deployment(), _pods()
    connector = _connector(deployment, pods)
    environment = PlannerEnvironmentImpl(
        config=_config(),
        controller=connector,
        require_prefill=True,
        require_decode=True,
    )
    environment.deployment_state().prefill.info = Mock(k8s_name="p")
    environment.deployment_state().decode.info = Mock(k8s_name="d")
    observer = EnvironmentObservePlugin(
        environment, require_prefill=True, require_decode=True
    )
    await environment._refresh_replica_counts()
    assert observer._collect_worker_counts().pending_num_decode == 1
    pods[0].metadata.deletion_timestamp = datetime.now(timezone.utc)
    await environment._refresh_replica_counts()
    assert not observer._collect_worker_counts().startup_in_progress


@pytest.mark.parametrize("mode", ["disagg", "decode", "agg"])
def test_budget_clamped_up_proposal_cannot_cancel_startup(mode):
    config = _config(mode)
    config.max_gpu_budget = 2 if mode == "disagg" else 1
    state = PlannerScalingState(config, _caps())
    state.observe_worker_counts(_counts())
    assert state.advance_load(_obs(kv=950)) is None


@pytest.mark.parametrize("requested,expected", [(1, 1), (4, None)])
def test_throughput_cancels_startup_only_for_actual_down_intent(requested, expected):
    config = _config()
    config.enable_load_scaling = False
    config.enable_throughput_scaling = True
    config.max_gpu_budget = 2
    state = PlannerScalingState(config, _caps())
    state.observe_worker_counts(_counts())
    state._compute_prefill_replicas = Mock(return_value=1)
    state._compute_decode_replicas = Mock(return_value=requested)
    decision = state._throughput_disagg(1.0, 100.0, 100.0)
    assert (decision.num_decode if decision else None) == expected


def test_worker_becoming_ready_restarts_startup_stabilization():
    config = _config()
    clock = VirtualClock()
    adapter = OrchestratorEngineAdapter(config, _caps(), clock=clock)
    counts, outcome = _counts(ready=2, pending=2), _outcome(d=1)
    adapter._project_scale_to(outcome, counts)
    clock.advance(config.load_adjustment_interval_seconds)
    counts.ready_num_decode, counts.pending_num_decode = 3, 1
    assert adapter._project_scale_to(outcome, counts) is None
    clock.advance(config.load_adjustment_interval_seconds)
    assert adapter._project_scale_to(outcome, counts).num_decode == 1


@pytest.mark.parametrize("power_limit,allowed", [(500, False), (800, True)])
def test_startup_power_budget_includes_pending_peers(power_limit, allowed):
    config = _config()
    config.enable_power_awareness = True
    config.total_gpu_power_limit = power_limit
    caps = _caps()
    caps.prefill.power_watts_per_replica = caps.decode.power_watts_per_replica = 200
    counts = _counts()
    counts.pending_num_prefill = 2
    clock = VirtualClock()
    adapter = OrchestratorEngineAdapter(config, caps, clock=clock)
    outcome = _outcome(d=1, explicit=("decode",))
    adapter._project_scale_to(outcome, counts)
    clock.advance(config.load_adjustment_interval_seconds)
    assert (adapter._project_scale_to(outcome, counts) is not None) is allowed
