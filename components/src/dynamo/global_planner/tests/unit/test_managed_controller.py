# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfers exercise actual target/observation skew, including uncertain writes."""

import asyncio
from dataclasses import replace

import pytest

from dynamo.global_planner import __main__ as gp_main
from dynamo.global_planner.allocation_policy import AllocationPolicy
from dynamo.global_planner.config import GlobalPlannerConfig
from dynamo.global_planner.managed_capacity import (
    ManagedCapacityError,
    ManagedPoolSnapshot,
)
from dynamo.global_planner.managed_config import ManagedAllocationConfig
from dynamo.global_planner.managed_controller import ManagedAllocationController
from dynamo.global_planner.managed_handler import ManagedRequestHandler
from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.protocol import (
    AllocationStatusRequest,
    ScaleRequest,
    ScaleStatus,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.fixture
def policy():
    return AllocationPolicy.model_validate(
        {
            "min_total_gpus": 16,
            "max_total_gpus": 16,
            "pools": [
                {
                    "pool_id": name,
                    "participant_id": f"prod/{name}",
                    "component_name": "worker",
                    "role": "decode",
                    "gpu_per_replica": 4,
                    "min_gpus": 4,
                    "max_gpus": 12,
                    "priority": priority,
                }
                for name, priority in (("interactive", 10), ("batch", 0))
            ],
        }
    )


class Capacity:
    def __init__(self):
        self.snapshots = {
            name: ManagedPoolSnapshot(name, name + "-uid", 2, 2, 2, 0, True)
            for name in ("interactive", "batch")
        }
        self.writes = []
        self.fences = []
        self.error = None
        self.write_error = False

    async def acquire(self, writer_id):
        self.fences.append(writer_id)

    async def observe(self):
        if self.error:
            raise self.error
        return dict(self.snapshots)

    async def scale(self, pool_id, replicas, *, expected_identity, expected_target):
        snap = self.snapshots[pool_id]
        assert (
            snap.identity == expected_identity
            and snap.target_replicas == expected_target
        )
        self.writes.append((pool_id, replicas))
        self.snapshots[pool_id] = replace(snap, target_replicas=replicas, settled=False)
        if self.write_error:
            raise ManagedCapacityError("write may have landed")

    def settle(self, pool_id):
        snap = self.snapshots[pool_id]
        self.snapshots[pool_id] = replace(
            snap,
            observed_replicas=snap.target_replicas,
            ready_replicas=snap.target_replicas,
            terminating_replicas=0,
            settled=True,
        )


@pytest.fixture
def capacity():
    return Capacity()


@pytest.fixture
def controller(policy, capacity):
    return ManagedAllocationController(policy, capacity)


@pytest.mark.asyncio
async def test_reclaim_waits_for_physical_release_and_requester_settlement(
    controller, capacity
):
    controller.submit("interactive", 3, "i1")
    await controller.reconcile_once()
    assert capacity.writes == [("batch", 1)]
    status = controller.status("interactive").allocation
    assert (
        status.desired_replicas,
        status.granted_replicas,
        status.observed_replicas,
    ) == (3, 3, 2)
    assert status.phase == "waiting_release" and status.transfer_id
    capacity.snapshots["batch"] = replace(
        capacity.snapshots["batch"], terminating_replicas=1, ready_replicas=1
    )
    await controller.reconcile_once()
    assert capacity.writes == [("batch", 1)]
    capacity.settle("batch")
    await controller.reconcile_once()
    assert capacity.writes == [("batch", 1), ("interactive", 3)]
    assert controller.active is not None
    await controller.reconcile_once()
    assert controller.active is not None
    capacity.settle("interactive")
    await controller.reconcile_once()
    assert controller.active is None
    assert controller.status("interactive").status == ScaleStatus.SUCCESS
    assert controller.granted == {"interactive": 3, "batch": 1}


@pytest.mark.asyncio
async def test_fixed_floor_returns_volunteered_capacity_to_retained_demand(
    controller, capacity
):
    controller.submit("batch", 2, "b1")
    controller.submit("interactive", 3, "i1")
    await controller.reconcile_once()
    capacity.settle("batch")
    await controller.reconcile_once()
    capacity.settle("interactive")
    await controller.reconcile_once()
    # Interactive's request alone must retain warm capacity at the floor;
    # Batch's still-pending request can take its voluntarily released worker.
    controller.submit("interactive", 2, "i2")
    await controller.reconcile_once()
    assert capacity.writes[-1] == ("interactive", 2)
    assert controller.granted == {"interactive": 2, "batch": 2}
    capacity.settle("interactive")
    await controller.reconcile_once()
    assert capacity.writes[-1] == ("batch", 2)
    capacity.settle("batch")
    await controller.reconcile_once()
    assert controller.active is None


@pytest.mark.asyncio
async def test_new_demand_coalesces_without_rewriting_active_plan(controller, capacity):
    controller.submit("interactive", 3, "i1")
    await controller.reconcile_once()
    transfer_id = controller.active.transfer_id
    controller.submit("interactive", 2, "i2")
    controller.submit("batch", 2, "b1")
    assert controller.active.transfer_id == transfer_id
    assert controller.granted["interactive"] == 3
    capacity.settle("batch")
    await controller.reconcile_once()
    assert capacity.writes[-1] == ("interactive", 3)
    capacity.settle("interactive")
    await controller.reconcile_once()
    await controller.reconcile_once()
    assert capacity.writes[-1] == ("interactive", 2)
    assert controller.active.transfer_id != transfer_id


@pytest.mark.asyncio
async def test_cancelled_pending_demand_is_not_regrown(controller, capacity):
    controller.submit("interactive", 3, "i1")
    controller.submit("batch", 2, "b1")
    await controller.reconcile_once()
    capacity.settle("batch")
    await controller.reconcile_once()
    capacity.settle("interactive")
    await controller.reconcile_once()
    controller.submit("batch", 1, "b2")
    controller.submit("interactive", 2, "i2")
    await controller.reconcile_once()
    assert controller.active is None
    assert controller.granted == {"interactive": 3, "batch": 1}
    assert controller.status("interactive").allocation.reason == "global_minimum"


def test_duplicates_do_not_restore_old_demand(controller):
    controller.submit("batch", 3, "first")
    controller.submit("batch", 1, "second")
    controller.submit("batch", 3, "first")
    assert controller.desired["batch"] == 1
    assert controller.request_ids["batch"] == "second"
    with pytest.raises(ValueError, match="different demand"):
        controller.submit("batch", 2, "first")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [ManagedCapacityError("unavailable"), TimeoutError("unknown write")]
)
async def test_unknown_observation_freezes_and_preserves_reservations(
    controller, capacity, failure
):
    controller.submit("interactive", 3, "i1")
    await controller.reconcile_once()
    reservation = dict(controller.granted)
    capacity.error = failure
    await controller.reconcile_once()
    assert controller.phase == "recovery_required"
    assert controller.granted == reservation
    assert controller.active is not None
    response = controller.status("interactive")
    assert response.status == ScaleStatus.ERROR
    assert response.allocation.observed_replicas is None
    capacity.error = None
    capacity.settle("batch")
    await controller.reconcile_once()
    assert capacity.writes == [("batch", 1)]


@pytest.mark.asyncio
async def test_uncertain_write_never_releases_reservation_or_retries(
    controller, capacity
):
    capacity.write_error = True
    controller.submit("interactive", 3, "i1")
    await controller.reconcile_once()
    assert controller.phase == "recovery_required"
    assert controller.granted["interactive"] == 3
    assert capacity.snapshots["batch"].target_replicas == 1
    capacity.write_error = False
    capacity.settle("batch")
    await controller.reconcile_once()
    assert capacity.writes == [("batch", 1)]


@pytest.mark.asyncio
async def test_timeout_retains_charges(policy, capacity):
    now = [0.0]
    controller = ManagedAllocationController(
        policy, capacity, clock=lambda: now[0], operation_timeout_seconds=10
    )
    controller.submit("interactive", 3, "i1")
    await controller.reconcile_once()
    now[0] = 11.0
    await controller.reconcile_once()
    assert controller.phase == "recovery_required"
    assert capacity.writes == [("batch", 1)]


@pytest.mark.asyncio
async def test_restart_reconstructs_only_settled_valid_band(
    controller, capacity, policy
):
    controller.submit("interactive", 3, "i1")
    await controller.reconcile_once()
    restarted = ManagedAllocationController(policy, capacity)
    await restarted.reconcile_once()
    assert restarted.phase == "recovering"
    assert restarted.granted == {}
    assert len(set(capacity.fences)) == 2
    capacity.settle("batch")
    await restarted.reconcile_once()
    assert restarted.phase == "recovery_required"
    assert "shared bounds" in restarted.reason
    assert capacity.writes == [("batch", 1)]


@pytest.mark.asyncio
async def test_restart_accepts_complete_settled_allocation(
    controller, capacity, policy
):
    await controller.reconcile_once()
    restarted = ManagedAllocationController(policy, capacity)
    await restarted.reconcile_once()
    assert restarted.phase == "ready"
    assert restarted.granted == {"interactive": 2, "batch": 2}
    assert restarted.desired == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"identity": "replacement"},
        {"target_replicas": 1},
        {"observed_replicas": 3},
    ],
)
async def test_external_drift_stops_mutations(controller, capacity, change):
    await controller.reconcile_once()
    capacity.snapshots["batch"] = replace(capacity.snapshots["batch"], **change)
    controller.submit("interactive", 3, "i1")
    await controller.reconcile_once()
    assert controller.phase == "recovery_required"
    assert capacity.writes == []


@pytest.mark.asyncio
async def test_worker_restart_outside_transfer_defers_planning(controller, capacity):
    await controller.reconcile_once()
    capacity.snapshots["batch"] = replace(
        capacity.snapshots["batch"], ready_replicas=1, settled=False
    )
    controller.submit("interactive", 3, "i1")
    await controller.reconcile_once()
    assert controller.phase == "ready" and controller.active is None
    assert controller.status("interactive").allocation.phase == "pending"
    assert capacity.writes == []

    capacity.settle("batch")
    await controller.reconcile_once()
    assert capacity.writes == [("batch", 1)]


@pytest.mark.asyncio
async def test_incomplete_observation_is_not_zero_capacity(controller, capacity):
    del capacity.snapshots["batch"]
    controller.submit("interactive", 3, "i1")
    await controller.reconcile_once()
    assert controller.phase == "recovery_required"
    assert capacity.writes == []


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_requests_and_status_remain_responsive_during_backend_wait(
    controller, capacity
):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_observe():
        entered.set()
        await release.wait()
        return dict(capacity.snapshots)

    capacity.observe = blocked_observe
    task = asyncio.create_task(controller.reconcile_once())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        response = controller.submit("interactive", 3, "i1")
        assert response.allocation.desired_replicas == 3
        assert controller.status("batch").allocation.phase == "recovering"
    finally:
        release.set()
        await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
async def test_equal_priority_does_not_force_reclaim(policy, capacity):
    data = policy.model_dump()
    data["pools"][0]["priority"] = 0
    controller = ManagedAllocationController(
        AllocationPolicy.model_validate(data), capacity
    )
    controller.submit("interactive", 3, "i1")
    await controller.reconcile_once()
    assert controller.active is None
    assert capacity.writes == []
    assert controller.status("interactive").allocation.phase == "pending"


@pytest.mark.asyncio
async def test_managed_handler_binds_caller_target_and_role(controller, policy):
    config = ManagedAllocationConfig(
        policy=policy, caller_namespaces={"interactive": "chat", "batch": "jobs"}
    )
    handler = ManagedRequestHandler(config, controller)
    request = ScaleRequest(
        caller_namespace="chat",
        graph_deployment_name="interactive",
        k8s_namespace="prod",
        target_replicas=[
            TargetReplica(
                sub_component_type=SubComponentType.DECODE, desired_replicas=3
            )
        ],
        request_id="i1",
    )
    response = [r async for r in handler.scale_request.__wrapped__(handler, request)][0]
    assert response["allocation"]["desired_replicas"] == 3
    request.caller_namespace = "jobs"
    response = [r async for r in handler.scale_request.__wrapped__(handler, request)][0]
    assert response["status"] == "error"
    assert controller.request_ids == {"interactive": "i1"}
    status_request = AllocationStatusRequest(
        caller_namespace="chat",
        graph_deployment_name="interactive",
        k8s_namespace="prod",
    )
    response = [
        r async for r in handler.allocation_status.__wrapped__(handler, status_request)
    ][0]
    assert response["allocation"]["pool_id"] == "interactive"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [("request_id", None), ("target_replicas", []), ("k8s_namespace", "other")],
)
async def test_invalid_requests_do_not_mutate_demand(controller, policy, field, value):
    config = ManagedAllocationConfig(
        policy=policy, caller_namespaces={"interactive": "chat", "batch": "jobs"}
    )
    handler = ManagedRequestHandler(config, controller)
    request = ScaleRequest(
        caller_namespace="chat",
        graph_deployment_name="interactive",
        k8s_namespace="prod",
        target_replicas=[
            TargetReplica(
                sub_component_type=SubComponentType.DECODE, desired_replicas=3
            )
        ],
        request_id="i1",
    )
    setattr(request, field, value)
    response = [r async for r in handler.scale_request.__wrapped__(handler, request)][0]
    assert response["status"] == "error"
    assert controller.desired == {}


@pytest.mark.parametrize(
    "legacy",
    [
        {"no_operation": True},
        {"max_total_gpus": 16},
        {"managed_namespaces": ["chat"]},
        {"priority": {"default": 10}},
        {"gpu_cost": {"pools": [{"selector": "prod/*", "gpu_per_replica": 4}]}},
    ],
)
def test_managed_configuration_rejects_ambiguous_legacy_settings(policy, legacy):
    config = {
        "managed_allocation": {
            "policy": policy.model_dump(),
            "caller_namespaces": {"interactive": "chat", "batch": "jobs"},
        },
        **legacy,
    }
    with pytest.raises(ValueError, match="cannot be combined"):
        GlobalPlannerConfig.model_validate(config)


@pytest.mark.parametrize(
    "callers",
    [
        {"interactive": "chat"},
        {"interactive": "chat", "batch": ""},
        {"interactive": "chat", "batch": " jobs"},
    ],
)
def test_caller_bindings_are_complete_and_exact(policy, callers):
    with pytest.raises(ValueError):
        ManagedAllocationConfig(policy=policy, caller_namespaces=callers)


def test_positive_live_minimum_required(policy):
    data = policy.model_dump()
    data["pools"][0]["min_gpus"] = 0
    with pytest.raises(ValueError, match="positive minimum"):
        ManagedAllocationConfig(
            policy=AllocationPolicy.model_validate(data),
            caller_namespaces={"interactive": "chat", "batch": "jobs"},
        )


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_managed_entrypoint_serves_status_while_controller_runs(
    policy, capacity, monkeypatch
):
    registered = asyncio.Event()
    endpoints = {}
    controller = ManagedAllocationController(
        policy, capacity, poll_interval_seconds=0.001
    )

    class Endpoint:
        async def serve_endpoint(self, callback, **kwargs):
            endpoints[self.name] = callback
            if len(endpoints) == 3:
                registered.set()
            await asyncio.Event().wait()

    class Runtime:
        def endpoint(self, name):
            endpoint = Endpoint()
            endpoint.name = name
            return endpoint

    monkeypatch.setenv("DYN_NAMESPACE", "global-infra")
    monkeypatch.setenv("POD_NAMESPACE", "prod")
    monkeypatch.setattr(gp_main, "ManagedKubernetesCapacity", lambda *args: capacity)
    monkeypatch.setattr(
        gp_main, "ManagedAllocationController", lambda *args, **kwargs: controller
    )
    config = GlobalPlannerConfig(
        managed_allocation=ManagedAllocationConfig(
            policy=policy, caller_namespaces={"interactive": "chat", "batch": "jobs"}
        )
    )
    task = asyncio.create_task(gp_main.main.__wrapped__(Runtime(), config))
    try:
        await asyncio.wait_for(registered.wait(), 2)
        assert set(endpoints) == {
            f"global-infra.GlobalPlanner.{name}"
            for name in ("scale_request", "allocation_status", "health")
        }
        assert controller.phase == "ready"
        assert len(capacity.fences) == 1
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert controller.phase == "recovery_required"
