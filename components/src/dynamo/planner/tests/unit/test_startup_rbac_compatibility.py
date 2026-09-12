# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Older service accounts keep ordinary scaling without granting startup reads."""

from unittest.mock import AsyncMock, Mock

import pytest
from kubernetes.client import ApiException

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.environment.base import PlannerEnvironmentImpl
from dynamo.planner.errors import DynamoGraphDeploymentNotReadyError
from dynamo.planner.monitoring.worker_info import WorkerInfo
from dynamo.planner.tests.unit.test_pending_startup_scaling import (
    _config,
    _connector,
    _deployment,
    _pods,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _settle(deployment, replicas=3):
    deployment["spec"]["components"][1]["replicas"] = replicas
    deployment["status"]["conditions"][0]["status"] = "True"
    deployment["status"]["components"]["d"] = {
        "replicas": replicas,
        "updatedReplicas": replicas,
        "readyReplicas": replicas,
    }


def _env(connector, *, power=False):
    config = _config()
    config.enable_power_awareness = power
    environment = PlannerEnvironmentImpl(
        config=config,
        controller=connector,
        require_prefill=True,
        require_decode=True,
    )
    environment.deployment_state().prefill.info = WorkerInfo(k8s_name="p")
    environment.deployment_state().decode.info = WorkerInfo(k8s_name="d")
    return environment


def _target(replicas):
    return [
        TargetReplica(
            sub_component_type=SubComponentType.DECODE, desired_replicas=replicas
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [2, 4])
async def test_old_rbac_preserves_replica_refresh_and_ready_scaling(target):
    deployment = _deployment()
    _settle(deployment)
    connector = _connector(deployment, _pods())
    connector.kube_api.list_pods_for_graph.side_effect = ApiException(status=403)
    environment = _env(connector)

    await environment._refresh_replica_counts()
    replicas = environment.deployment_state().decode.replicas
    assert (
        replicas.active,
        replicas.expected,
        replicas.scaling,
        replicas.pending_startup,
    ) == (3, 3, False, 0)
    await connector.set_component_replicas(_target(target), blocking=False)
    connector.kube_api.update_graph_replicas.assert_called_once_with(
        "qwen", "d", target
    )


@pytest.mark.asyncio
async def test_pod_permission_changes_enable_and_disable_startup_inventory():
    connector = _connector(_deployment(), _pods())
    environment = _env(connector)
    connector.kube_api.list_pods_for_graph.side_effect = ApiException(status=403)
    await environment._refresh_replica_counts()
    replicas = environment.deployment_state().decode.replicas
    assert replicas.active == 1 and replicas.scaling and replicas.pending_startup == 0

    connector.kube_api.list_pods_for_graph.side_effect = None
    await environment._refresh_replica_counts()
    assert replicas.pending_startup == 1

    connector.kube_api.list_pods_for_graph.side_effect = ApiException(status=403)
    await environment._refresh_replica_counts()
    assert replicas.pending_startup == 0
    await connector.set_component_replicas(_target(1), blocking=False)
    connector.kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_not_ready", [False, True])
async def test_old_rbac_keeps_unready_rejection(raise_not_ready):
    connector = _connector(_deployment(), _pods())
    connector.raise_not_ready = raise_not_ready
    connector.kube_api.list_pods_for_graph.side_effect = ApiException(status=403)
    if raise_not_ready:
        with pytest.raises(DynamoGraphDeploymentNotReadyError):
            await connector.set_component_replicas(_target(1), blocking=False)
    else:
        await connector.set_component_replicas(_target(1), blocking=False)
    connector.kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_pod_read_denial_does_not_release_accepted_downscale():
    deployment = _deployment()
    _settle(deployment, replicas=1)
    connector = _connector(deployment, _pods())
    connector._startup_scale_down_targets = {"d": 1}
    connector.kube_api.list_pods_for_graph.side_effect = ApiException(status=403)
    assert await connector.get_worker_inventory("p", "d") is None
    await connector.set_component_replicas(_target(2), blocking=False)
    assert connector._startup_scale_down_targets == {"d": 1}
    connector.kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_power_awareness_does_not_fall_back_to_unverified_counts():
    connector = _connector(_deployment(), _pods())
    connector.kube_api.list_pods_for_graph.side_effect = ApiException(status=403)
    connector.get_actual_worker_counts = AsyncMock()
    connector.get_power_aware_worker_counts = AsyncMock(
        wraps=connector.get_power_aware_worker_counts
    )
    with pytest.raises(ApiException) as error:
        await _env(connector, power=True)._refresh_replica_counts()
    assert error.value.status == 403
    connector.get_power_aware_worker_counts.assert_awaited_once()
    connector.get_actual_worker_counts.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 500])
async def test_non_permission_errors_still_propagate(status):
    connector = _connector(_deployment(), _pods())
    connector.kube_api.list_pods_for_graph.side_effect = ApiException(status=status)
    connector.get_actual_worker_counts = AsyncMock()
    with pytest.raises(ApiException) as error:
        await _env(connector)._refresh_replica_counts()
    assert error.value.status == status
    connector.get_actual_worker_counts.assert_not_awaited()


@pytest.mark.asyncio
async def test_forbidden_dgd_read_is_not_treated_as_optional_pod_access():
    connector = _connector(_deployment(), _pods())
    connector.kube_api.get_graph_deployment.side_effect = ApiException(status=403)
    connector.get_actual_worker_counts = AsyncMock()
    with pytest.raises(ApiException):
        await _env(connector)._refresh_replica_counts()
    connector.get_actual_worker_counts.assert_not_awaited()


@pytest.mark.asyncio
async def test_scale_get_denial_holds_until_the_accepted_target_is_observed():
    deployment = _deployment()
    connector = _connector(deployment, _pods())
    await connector.set_component_replicas(_target(1), blocking=False)
    connector.kube_api.get_service_replica_target = Mock(
        side_effect=ApiException(status=403)
    )
    inventory = await connector.get_worker_inventory("p", "d")
    assert inventory is not None and inventory.decode_scaling_in_progress
    assert not inventory.startup_in_progress
    assert connector._startup_scale_down_targets == {"d": 1}

    _settle(deployment, replicas=1)
    inventory = await connector.get_worker_inventory("p", "d")
    assert inventory is not None and not inventory.decode_scaling_in_progress
    assert connector._startup_scale_down_targets == {}
    connector.kube_api.get_service_replica_target.assert_called_once()


@pytest.mark.asyncio
async def test_forbidden_replica_write_is_not_silently_ignored():
    deployment = _deployment()
    _settle(deployment)
    connector = _connector(deployment, _pods())
    connector.kube_api.list_pods_for_graph.side_effect = ApiException(status=403)
    connector.kube_api.update_graph_replicas.side_effect = ApiException(status=403)
    with pytest.raises(ApiException) as error:
        await connector.set_component_replicas(_target(2), blocking=False)
    assert error.value.status == 403
    assert connector._startup_scale_down_targets == {}


@pytest.mark.asyncio
async def test_forbidden_pod_read_does_not_admit_startup_via_stale_ready():
    deployment = _deployment()
    deployment["status"]["conditions"][0]["status"] = "True"
    connector = _connector(deployment, _pods())
    connector.kube_api.list_pods_for_graph.side_effect = ApiException(status=403)
    await connector.set_component_replicas(_target(1), blocking=False)
    connector.kube_api.update_graph_replicas.assert_not_called()
