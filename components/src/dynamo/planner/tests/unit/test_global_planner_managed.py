# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed allocation wire compatibility and Local Planner demand tracking."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from pydantic import ValidationError

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.clients.remote_client import RemotePlannerClient
from dynamo.planner.connectors.global_planner import GlobalPlannerConnector
from dynamo.planner.connectors.protocol import (
    AllocationStatusRequest,
    ManagedAllocationStatus,
    ScaleRequest,
    ScaleResponse,
    ScaleStatus,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


async def _stream(*responses):
    for response in responses:
        yield response


def _target(replicas: int) -> TargetReplica:
    return TargetReplica(
        sub_component_type=SubComponentType.DECODE,
        component_name="Decode",
        desired_replicas=replicas,
    )


def _allocation(**changes) -> ManagedAllocationStatus:
    return ManagedAllocationStatus(
        **{
            "pool_id": "batch",
            "controller_id": "controller-1",
            "phase": "draining",
            "reason": "waiting_for_release",
            "desired_replicas": 3,
            "granted_replicas": 1,
            "observed_replicas": 2,
            "ready_replicas": 1,
            "terminating_replicas": 1,
            "transfer_id": "transfer-1",
            **changes,
        }
    )


def _response(
    status: ScaleStatus = ScaleStatus.SCALING, **allocation_changes
) -> ScaleResponse:
    return ScaleResponse(
        status=status,
        message="Waiting for donor release",
        current_replicas={"decode": 1},
        allocation=_allocation(**allocation_changes),
    )


@pytest.fixture
def connector(monkeypatch):
    monkeypatch.setenv("DYN_PARENT_DGD_K8S_NAME", "batch-dgd")
    monkeypatch.setenv("POD_NAMESPACE", "workers")
    instance = GlobalPlannerConnector(MagicMock(), "batch-runtime", "global")
    instance.remote_client = AsyncMock(spec=RemotePlannerClient)
    return instance


def test_legacy_wire_messages_remain_valid():
    request = ScaleRequest(
        caller_namespace="batch-runtime",
        graph_deployment_name="batch-dgd",
        k8s_namespace="workers",
        target_replicas=[_target(2)],
    )
    response = ScaleResponse.model_validate(
        {"status": "success", "message": "done", "current_replicas": {"decode": 2}}
    )

    assert request.request_id is None
    assert response.allocation is None


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_managed_observation_rejects_invalid_replica_counts(count):
    with pytest.raises(ValidationError):
        _allocation(observed_replicas=count)


@pytest.mark.asyncio
async def test_status_endpoint_is_separate_and_does_not_submit_demand():
    runtime = MagicMock()
    scale_client = AsyncMock()
    status_client = AsyncMock()
    clients = {
        "global.GlobalPlanner.scale_request": scale_client,
        "global.GlobalPlanner.allocation_status": status_client,
    }
    runtime.endpoint.side_effect = lambda name: MagicMock(
        client=AsyncMock(return_value=clients[name])
    )
    scale_client.generate.side_effect = lambda _: _stream(_response().model_dump())
    status_client.generate.side_effect = lambda _: _stream(
        _response(phase="settled", observed_replicas=1).model_dump()
    )
    client = RemotePlannerClient(runtime, "global", "GlobalPlanner")
    identity = AllocationStatusRequest(
        caller_namespace="batch-runtime",
        graph_deployment_name="batch-dgd",
        k8s_namespace="workers",
    )
    demand = ScaleRequest(
        **identity.model_dump(), target_replicas=[_target(3)], request_id="demand-1"
    )

    await client.send_scale_request(demand)
    first = await client.get_allocation_status(identity)
    second = await client.get_allocation_status(identity)

    assert first.allocation.phase == second.allocation.phase == "settled"
    assert runtime.endpoint.call_count == 2
    scale_client.generate.assert_awaited_once()
    assert status_client.generate.await_count == 2
    for call in status_client.generate.call_args_list:
        assert json.loads(call.args[0]) == identity.model_dump()
    assert (
        json.loads(scale_client.generate.call_args.args[0])["request_id"] == "demand-1"
    )


@pytest.mark.asyncio
async def test_connection_retry_preserves_demand_id_and_submits_once():
    runtime = MagicMock()
    transport = AsyncMock()
    endpoint = runtime.endpoint.return_value
    endpoint.client = AsyncMock(side_effect=[RuntimeError("unavailable"), transport])
    transport.generate.side_effect = lambda _: _stream(_response().model_dump())
    client = RemotePlannerClient(runtime, "global", "GlobalPlanner", max_retries=2)
    request = ScaleRequest(
        caller_namespace="batch-runtime",
        graph_deployment_name="batch-dgd",
        k8s_namespace="workers",
        target_replicas=[_target(3)],
        request_id="same-demand",
    )

    with patch(
        "dynamo.planner.connectors.clients.remote_client.asyncio.sleep", new=AsyncMock()
    ):
        await client.send_scale_request(request)

    assert endpoint.client.await_count == 2
    transport.generate.assert_awaited_once()
    assert json.loads(transport.generate.call_args.args[0]) == request.model_dump(
        mode="json"
    )


@pytest.mark.asyncio
async def test_status_transport_requires_a_response():
    runtime = MagicMock()
    transport = AsyncMock()
    runtime.endpoint.return_value.client = AsyncMock(return_value=transport)
    transport.generate.side_effect = lambda _: _stream()
    client = RemotePlannerClient(runtime, "global", "GlobalPlanner")
    identity = AllocationStatusRequest(
        caller_namespace="batch-runtime",
        graph_deployment_name="batch-dgd",
        k8s_namespace="workers",
    )

    with pytest.raises(RuntimeError, match="No response"):
        await client.get_allocation_status(identity)


@pytest.mark.asyncio
async def test_pending_demand_is_separate_from_grant_and_does_not_log_success(
    connector,
):
    response = _response()
    connector.remote_client.send_scale_request.return_value = response
    target = _target(3)

    with patch("dynamo.planner.connectors.global_planner.logger") as logger:
        await connector.set_component_replicas([target], blocking=False)

    request = connector.remote_client.send_scale_request.call_args.args[0]
    assert UUID(request.request_id).version == 4
    assert request.target_replicas[0].desired_replicas == 3
    assert connector.last_managed_allocation.granted_replicas == 1
    assert connector.last_desired_replicas[0].desired_replicas == 3
    target.desired_replicas = 99
    assert connector.last_desired_replicas[0].desired_replicas == 3
    logger.warning.assert_not_called()
    logger.error.assert_not_called()
    assert any("pending" in call.args[0] for call in logger.info.call_args_list)
    assert all("successful" not in call.args[0] for call in logger.info.call_args_list)


@pytest.mark.asyncio
async def test_new_decisions_get_new_ids_without_clamping_to_grant(connector):
    connector.remote_client.send_scale_request.return_value = _response()

    await connector.set_component_replicas([_target(3)])
    await connector.set_component_replicas([_target(4)])

    first, second = [
        call.args[0]
        for call in connector.remote_client.send_scale_request.call_args_list
    ]
    assert first.request_id != second.request_id
    assert [
        first.target_replicas[0].desired_replicas,
        second.target_replicas[0].desired_replicas,
    ] == [3, 4]
    assert connector.last_desired_replicas[0].desired_replicas == 4
    assert connector.last_managed_allocation.granted_replicas == 1


@pytest.mark.asyncio
async def test_rejection_retains_demand_and_grant_without_raising(connector):
    connector.remote_client.send_scale_request.return_value = _response(
        ScaleStatus.REJECTED, phase="pending", reason="insufficient_capacity"
    )

    await connector.set_component_replicas([_target(3)])

    assert connector.last_desired_replicas[0].desired_replicas == 3
    assert connector.last_managed_allocation.granted_replicas == 1
    assert connector.last_managed_allocation.reason == "insufficient_capacity"


@pytest.mark.asyncio
async def test_worker_observation_refreshes_status_without_substituting_grant(
    connector,
):
    connector.last_managed_allocation = _allocation(granted_replicas=2)
    connector.managed_allocation = True
    connector.last_desired_replicas = [_target(3)]
    connector.remote_client.get_allocation_status.return_value = _response()
    local = MagicMock()
    local.get_actual_worker_counts = AsyncMock(return_value=(0, 1, False))
    connector._local_k8s_connector = local
    connector._local_k8s_init_attempted = True

    counts = await connector.get_actual_worker_counts(decode_component_name="Decode")

    assert counts == (0, 1, False)
    assert connector.last_managed_allocation.granted_replicas == 1
    assert connector.last_desired_replicas[0].desired_replicas == 3
    connector.remote_client.send_scale_request.assert_not_awaited()
    identity = connector.remote_client.get_allocation_status.call_args.args[0]
    assert identity == AllocationStatusRequest(
        caller_namespace="batch-runtime",
        graph_deployment_name="batch-dgd",
        k8s_namespace="workers",
    )
    local.get_actual_worker_counts.assert_awaited_once_with(
        prefill_component_name=None, decode_component_name="Decode"
    )


@pytest.mark.asyncio
async def test_legacy_worker_observation_never_polls_managed_endpoint(connector):
    local = MagicMock()
    local.get_actual_worker_counts = AsyncMock(return_value=(0, 2, True))
    connector._local_k8s_connector = local
    connector._local_k8s_init_attempted = True

    assert await connector.get_actual_worker_counts() == (0, 2, True)
    connector.remote_client.get_allocation_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_status_error_is_not_reported_as_current_capacity(connector):
    connector.remote_client.get_allocation_status.return_value = _response(
        ScaleStatus.ERROR, phase="validation_error"
    )

    with pytest.raises(RuntimeError, match="allocation status failed"):
        await connector.get_allocation_status()

    assert connector.last_managed_allocation.phase == "validation_error"


@pytest.mark.asyncio
async def test_recovery_hold_preserves_observation_and_later_demand(connector):
    frozen = _response(ScaleStatus.ERROR, phase="recovery_required")
    connector.last_managed_allocation = frozen.allocation
    connector.remote_client.get_allocation_status.return_value = frozen
    connector.remote_client.send_scale_request.return_value = frozen
    local = MagicMock()
    local.get_actual_worker_counts = AsyncMock(return_value=(0, 2, True))
    connector._local_k8s_connector = local
    connector._local_k8s_init_attempted = True

    await connector.set_component_replicas([_target(3)])
    assert await connector.get_actual_worker_counts() == (0, 2, True)
    assert connector.last_managed_allocation.phase == "recovery_required"

    recovered = _response(controller_id="controller-2", phase="pending")
    connector.remote_client.send_scale_request.return_value = recovered
    await connector.set_component_replicas([_target(2)])

    assert connector.remote_client.send_scale_request.await_count == 2
    assert connector.last_managed_allocation.controller_id == "controller-2"
    assert connector.last_desired_replicas[0].desired_replicas == 2


@pytest.mark.asyncio
async def test_status_can_return_legacy_response(connector):
    connector.remote_client.get_allocation_status.return_value = ScaleResponse(
        status=ScaleStatus.SUCCESS, message="legacy", current_replicas={"decode": 2}
    )

    assert await connector.get_allocation_status() is None
    assert connector.last_managed_allocation is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [RuntimeError("discovery unavailable"), TimeoutError()]
)
async def test_status_outage_keeps_observing_and_retries_after_recovery(
    connector, failure
):
    connector.managed_allocation = True
    connector.remote_client.send_scale_request.return_value = _response()
    await connector.set_component_replicas([_target(3)])
    connector.remote_client.get_allocation_status.side_effect = [
        failure,
        _response(controller_id="controller-2", phase="settled"),
    ]
    local = MagicMock()
    local.get_actual_worker_counts = AsyncMock(return_value=(0, 2, True))
    connector._local_k8s_connector = local
    connector._local_k8s_init_attempted = True

    assert await connector.get_actual_worker_counts() == (0, 2, True)
    assert connector.last_managed_allocation is None
    assert await connector.get_actual_worker_counts() == (0, 2, True)
    assert connector.last_managed_allocation.controller_id == "controller-2"
    assert connector.remote_client.get_allocation_status.await_count == 2


@pytest.mark.asyncio
async def test_malformed_status_is_not_treated_as_a_transport_outage(connector):
    connector.remote_client.get_allocation_status.side_effect = ValueError("malformed")

    with pytest.raises(ValueError, match="malformed"):
        await connector.get_allocation_status()


@pytest.mark.asyncio
async def test_unknown_mutation_outcome_remains_an_error(connector):
    connector.remote_client.send_scale_request.side_effect = RuntimeError("transport")

    with pytest.raises(RuntimeError, match="transport"):
        await connector.set_component_replicas([_target(3)])


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_hanging_status_stream_times_out_and_allows_worker_observation(connector):
    cancelled = asyncio.Event()

    async def stalled_stream():
        try:
            await asyncio.Event().wait()
            yield _response().model_dump()
        finally:
            cancelled.set()

    transport = AsyncMock()
    transport.generate.return_value = stalled_stream()
    remote = RemotePlannerClient(
        MagicMock(), "global", "GlobalPlanner", connection_timeout=0.01
    )
    remote._status_client = transport
    connector.remote_client = remote
    connector.last_managed_allocation = _allocation()
    connector.managed_allocation = True
    local = MagicMock()
    local.get_actual_worker_counts = AsyncMock(return_value=(0, 2, True))
    connector._local_k8s_connector = local
    connector._local_k8s_init_attempted = True

    counts = await asyncio.wait_for(connector.get_actual_worker_counts(), timeout=1)

    assert counts == (0, 2, True)
    assert cancelled.is_set()
    assert connector.last_managed_allocation is None
