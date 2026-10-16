# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for KubernetesCapacityManager — the K8s observe/scale backend.

Pool state is read from the v1beta1 DGD component model (``spec.components[]``).
"""

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes import client

from dynamo.global_planner.capacity_manager import PoolSpec
from dynamo.global_planner.kubernetes_capacity_manager import KubernetesCapacityManager
from dynamo.planner import SubComponentType, TargetReplica
from dynamo.planner.connectors.base import ReplicaUpdateResult
from dynamo.planner.connectors.clients.kubernetes_api import KubernetesAPI
from dynamo.planner.connectors.kubernetes import (
    ReplicaUpdateAttempt,
    ReplicaUpdateFailure,
)
from dynamo.planner.errors import (
    DynamoGraphDeploymentNotFoundError,
    GPUShapeUnavailableError,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _component(name, replicas, gpu=1, ctype=None, node_count=None):
    main_container = {"name": "main"}
    if gpu is not None:
        main_container["resources"] = {"limits": {"nvidia.com/gpu": gpu}}
    component = {
        "name": name,
        "replicas": replicas,
        "podTemplate": {"spec": {"containers": [main_container]}},
    }
    if ctype is not None:
        component["type"] = ctype
    if node_count is not None:
        component["multinode"] = {"nodeCount": node_count}
    return component


def _with_operator_gpu_status(components):
    statuses = {}
    for component in components:
        main = component["podTemplate"]["spec"]["containers"][0]
        gpu = main.get("resources", {}).get("limits", {}).get("nvidia.com/gpu")
        replicas = component.get("replicas", 1)
        status = {
            "replicas": replicas,
            "updatedReplicas": replicas,
            "readyReplicas": replicas,
            "availableReplicas": replicas,
        }
        if gpu is not None:
            total_gpu = int(gpu) * component.get("multinode", {}).get("nodeCount", 1)
            status.update({"gpusPerEngine": total_gpu, "gpusPerReplica": total_gpu})
        statuses[component["name"]] = status
    return {
        "metadata": {
            "generation": 1,
            "uid": "dgd-uid",
            "resourceVersion": "dgd-rv-1",
        },
        "spec": {"components": components},
        "status": {"observedGeneration": 1, "components": statuses},
    }


def _dgd_spec(
    prefill_replicas,
    decode_replicas,
    prefill_gpu=1,
    decode_gpu=1,
    prefill_node_count=None,
):
    """A v1beta1 DGD spec with typed prefill and decode components."""
    return _with_operator_gpu_status(
        [
            _component(
                "prefill-svc",
                prefill_replicas,
                gpu=prefill_gpu,
                ctype="prefill",
                node_count=prefill_node_count,
            ),
            _component("decode-svc", decode_replicas, gpu=decode_gpu, ctype="decode"),
        ]
    )


def _worker_dgd_spec(replicas, gpu=1, name="worker-svc", ctype="worker"):
    """A v1beta1 DGD spec with a single generic worker component."""
    return _with_operator_gpu_status([_component(name, replicas, gpu=gpu, ctype=ctype)])


def _listed_dgd(spec, name):
    item = deepcopy(spec)
    item.setdefault("metadata", {})["name"] = name
    return item


def _install_connector(
    cm,
    key,
    spec,
    parent_dgd_name="my-dgd",
    *,
    seed_identity=True,
):
    spec.setdefault("metadata", {}).setdefault("name", parent_dgd_name)
    connector = MagicMock()
    connector.parent_dgd_name = parent_dgd_name
    connector.set_component_replicas = AsyncMock(return_value=None)
    connector.set_component_replicas_with_result = AsyncMock(
        return_value=ReplicaUpdateResult(accepted=True, applied=True)
    )
    connector.ensure_replica_writer_fence = AsyncMock()
    connector.wait_for_deployment_ready = AsyncMock()
    connector.wait_for_replica_targets = AsyncMock()
    connector.kube_api = MagicMock()
    connector.kube_api.current_namespace = key.split("/", 1)[0]
    connector.kube_api.get_graph_deployment = MagicMock(return_value=spec)
    connector.kube_api.fetch_authoritative_replica_targets = MagicMock(
        side_effect=lambda deployment, **_kwargs: deployment
    )
    connector.kube_api.list_pods_for_graph = MagicMock(return_value=[])
    connector.kube_api.list_nonterminal_pods_for_graph = MagicMock(return_value=[])
    connector.kube_api.list_graph_pods = MagicMock(return_value=[])
    connector.kube_api.list_scaling_adapters = MagicMock(return_value=[])
    connector.kube_api.list_scaling_adapter_metadata = MagicMock(
        side_effect=lambda: deepcopy(
            connector.kube_api.list_scaling_adapters.return_value
        )
    )
    connector.kube_api.partition_pods_by_graph = MagicMock(
        side_effect=KubernetesAPI.partition_pods_by_graph
    )
    connector.kube_api.replica_targets_stable = MagicMock(
        side_effect=lambda deployment, targets: KubernetesAPI.replica_targets_stable(
            connector.kube_api,
            deployment,
            targets,
        )
    )
    connector.kube_api.adapter_targets_projected = MagicMock(
        side_effect=KubernetesAPI.adapter_targets_projected
    )
    connector.kube_api.is_spec_generation_observed = MagicMock(
        side_effect=KubernetesAPI.is_spec_generation_observed
    )
    connector.kube_api.is_rolling_update_blocking_settlement = MagicMock(
        side_effect=KubernetesAPI.is_rolling_update_blocking_settlement
    )
    connector.kube_api.exclude_checkpoint_capture_pods = MagicMock(
        side_effect=KubernetesAPI.exclude_checkpoint_capture_pods
    )
    connector.kube_api.has_terminating_pods = MagicMock(
        side_effect=KubernetesAPI.has_terminating_pods
    )
    connector.kube_api.pod_graph_uid_mismatches = MagicMock(
        side_effect=KubernetesAPI.pod_graph_uid_mismatches
    )
    connector.kube_api.pod_component_mismatches = MagicMock(
        side_effect=KubernetesAPI.pod_component_mismatches
    )
    connector.kube_api.pcsg_pods_within_desired_replicas = MagicMock(
        side_effect=lambda deployment, pods: KubernetesAPI.pcsg_pods_within_desired_replicas(
            connector.kube_api,
            deployment,
            pods,
        )
    )
    connector.kube_api.deployment_pods_within_desired_replicas = MagicMock(
        side_effect=lambda deployment, pods: KubernetesAPI.deployment_pods_within_desired_replicas(
            connector.kube_api,
            deployment,
            pods,
        )
    )
    connector._test_deployment = spec

    def list_deployments():
        deployments = []
        for participant, candidate in cm.connectors.items():
            if participant.split("/", 1)[0] != connector.kube_api.current_namespace:
                continue
            candidate_spec = getattr(candidate, "_test_deployment", None)
            if not isinstance(candidate_spec, dict):
                continue
            item = deepcopy(candidate_spec)
            item.setdefault("metadata", {})["name"] = candidate.parent_dgd_name
            deployments.append(item)
        return deployments

    connector.kube_api.list_graph_deployments = MagicMock(side_effect=list_deployments)
    connector.kube_api.list_graph_deployment_metadata = MagicMock(
        side_effect=list_deployments
    )
    cm.connectors[key] = connector
    if seed_identity:
        uid = spec.get("metadata", {}).get("uid")
        if isinstance(uid, str) and uid:
            cm._observed_uids[key] = uid
    if cm._inventory_api is None:
        cm._inventory_api = connector.kube_api
    return connector


def test_refresh_participant_records_first_identity_in_one_request():
    cm = KubernetesCapacityManager("default")
    deployment = _dgd_spec(1, 1)
    connector = _install_connector(
        cm,
        "default/model",
        deployment,
        parent_dgd_name="model",
        seed_identity=False,
    )

    cm.refresh_participant("default/model")

    assert cm._observed_uids["default/model"] == "dgd-uid"
    connector.kube_api.get_graph_deployment.assert_called_once_with("model")


def test_refresh_participant_accepts_replacement_in_unbounded_mode():
    cm = KubernetesCapacityManager("default")
    deployment = _dgd_spec(1, 1)
    connector = _install_connector(
        cm, "default/model", deployment, parent_dgd_name="model"
    )
    replacement = deepcopy(deployment)
    replacement["metadata"]["uid"] = "replacement-uid"
    connector.kube_api.get_graph_deployment.return_value = replacement
    cm._last_observed_pools["default/model"] = {"prefill": MagicMock()}
    cm._replica_uids["default/model"] = "dgd-uid"
    cm._replica_settlements["default/model"] = MagicMock()
    cm._missing_participants.add("default/model")

    cm.refresh_participant("default/model")

    assert cm._observed_uids["default/model"] == "replacement-uid"
    assert "default/model" not in cm._last_observed_pools
    assert "default/model" not in cm._replica_uids
    assert "default/model" not in cm._replica_settlements
    assert cm.participant_exists("default/model")


# ---------------------------------------------------------------------------- #
# Deployment-name derivation (K8s operator convention)                         #
# ---------------------------------------------------------------------------- #


def test_managed_deployment_names_explicit():
    cm = KubernetesCapacityManager("my-ns")
    assert cm._managed_deployment_names({"my-ns-model-a", "my-ns-model-b"}) == {
        "model-a",
        "model-b",
    }


def test_managed_deployment_names_implicit():
    cm = KubernetesCapacityManager("my-ns")
    assert cm._managed_deployment_names(None) is None


def test_managed_deployment_names_mismatched_prefix():
    cm = KubernetesCapacityManager("my-ns")
    # Only the caller matching the cluster prefix contributes a deployment name.
    assert cm._managed_deployment_names({"other-ns-model-a", "my-ns-model-b"}) == {
        "model-b"
    }


# ---------------------------------------------------------------------------- #
# Discovery                                                                    #
# ---------------------------------------------------------------------------- #


def test_discover_explicit_mode():
    cm = KubernetesCapacityManager("default")
    with (
        patch(
            "dynamo.global_planner.kubernetes_capacity_manager.KubernetesAPI"
        ) as mock_kube_cls,
        patch(
            "dynamo.global_planner.kubernetes_capacity_manager.KubernetesConnector"
        ) as mock_connector_cls,
    ):
        connector = MagicMock()
        connector.kube_api.get_graph_deployment.return_value = {
            "metadata": {"name": "model-a", "uid": "uid-a"}
        }
        mock_connector_cls.return_value = connector

        # Managed callers are namespaces; the deployment name is derived.
        discovered = cm.discover({"default-model-a"})

        assert discovered == ["model-a"]
        assert "default/model-a" in cm.connectors
        assert "default/model-b" not in cm.connectors
        mock_kube_cls.assert_not_called()
        connector.kube_api.get_graph_deployment.assert_called_once_with("model-a")
        assert not cm._observe_all_deployments
        mock_connector_cls.assert_called_once_with(
            dynamo_namespace="discovered",
            k8s_namespace="default",
            parent_dgd_name="model-a",
            raise_not_ready=True,
            retry_conflicts=False,
        )


def test_discover_explicit_mode_tracks_absent_allowlisted_deployment():
    cm = KubernetesCapacityManager("default")
    with patch(
        "dynamo.global_planner.kubernetes_capacity_manager.KubernetesConnector"
    ) as mock_connector_cls:
        connector = MagicMock()
        connector.kube_api.get_graph_deployment.side_effect = (
            DynamoGraphDeploymentNotFoundError("model-a", "default")
        )
        mock_connector_cls.return_value = connector

        assert cm.discover({"default-model-a"}) == []

    assert cm.connectors == {"default/model-a": connector}
    assert cm._missing_participants == {"default/model-a"}
    assert not cm.participant_exists("default/model-a")


def test_discover_explicit_mode_retains_unknown_deployment_fail_closed():
    cm = KubernetesCapacityManager("default")
    with patch(
        "dynamo.global_planner.kubernetes_capacity_manager.KubernetesConnector"
    ) as mock_connector_cls:
        connector = MagicMock()
        connector.kube_api.get_graph_deployment.side_effect = RuntimeError("timeout")
        mock_connector_cls.return_value = connector

        assert cm.discover({"default-model-a"}) == ["model-a"]

    assert cm.connectors == {"default/model-a": connector}


def test_discover_implicit_mode():
    cm = KubernetesCapacityManager("default")
    with (
        patch(
            "dynamo.global_planner.kubernetes_capacity_manager.KubernetesAPI"
        ) as mock_kube_cls,
        patch(
            "dynamo.global_planner.kubernetes_capacity_manager.KubernetesConnector"
        ) as mock_connector_cls,
    ):
        mock_kube = MagicMock()
        mock_kube_cls.return_value = mock_kube
        mock_kube.list_graph_deployments.return_value = [
            {"metadata": {"name": "model-a", "uid": "uid-a"}},
            {"metadata": {"name": "model-b", "uid": "uid-b"}},
        ]
        mock_connector_cls.return_value = MagicMock()

        discovered = cm.discover(None)

        assert set(discovered) == {"model-a", "model-b"}
        assert "default/model-a" in cm.connectors
        assert "default/model-b" in cm.connectors
        assert cm.participant_exists("default/model-a")
        assert cm.participant_exists("default/model-b")
        assert mock_connector_cls.call_count == 2
        assert cm._observe_all_deployments


def test_discover_tolerates_api_failure():
    cm = KubernetesCapacityManager("default")
    with patch(
        "dynamo.global_planner.kubernetes_capacity_manager.KubernetesAPI",
        side_effect=RuntimeError("no cluster"),
    ):
        # Best-effort: must not raise, returns nothing discovered.
        assert cm.discover(None) == []
    assert cm.connectors == {}


# ---------------------------------------------------------------------------- #
# Registration / scale                                                        #
# ---------------------------------------------------------------------------- #


def test_ensure_participant_idempotent():
    cm = KubernetesCapacityManager("default")
    with patch(
        "dynamo.global_planner.kubernetes_capacity_manager.KubernetesConnector"
    ) as mock_connector_cls:
        mock_connector_cls.return_value = MagicMock()
        cm.ensure_participant(
            "default/my-dgd",
            caller_name="app-ns",
            namespace="default",
            deployment_name="my-dgd",
        )
        cm.ensure_participant(
            "default/my-dgd",
            caller_name="app-ns",
            namespace="default",
            deployment_name="my-dgd",
        )
        assert mock_connector_cls.call_count == 1
        assert cm.participant_exists("default/my-dgd")
        assert not cm.participant_exists("default/other")


@pytest.mark.asyncio
async def test_scale_with_result_calls_connector():
    cm = KubernetesCapacityManager("default", writer_id="global-planner-writer")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))

    async def scale_with_identity(_targets, *, blocking, retry_state):
        assert blocking
        retry_state.dgd_uid = "dgd-uid"
        retry_state.authorities[("prefill", None)] = (
            "prefill-svc",
            False,
            None,
        )
        return ReplicaUpdateResult(accepted=True, applied=True)

    connector.set_component_replicas_with_result.side_effect = scale_with_identity
    targets = [
        TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=3)
    ]
    result = await cm.scale_with_result("default/my-dgd", targets, blocking=True)

    assert result.accepted
    assert result.applied
    connector.ensure_replica_writer_fence.assert_awaited_once_with(
        "global-planner-writer"
    )
    connector.set_component_replicas_with_result.assert_awaited_once()
    assert connector.set_component_replicas_with_result.await_args.args == (targets,)
    assert connector.set_component_replicas_with_result.await_args.kwargs["blocking"]
    assert cm._replica_uids["default/my-dgd"] == "dgd-uid"


@pytest.mark.asyncio
async def test_scale_preserves_accepted_noop_result():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))

    async def accept_noop(_targets, *, blocking, retry_state):
        retry_state.dgd_uid = "dgd-uid"
        retry_state.authorities[("prefill", None)] = (
            "prefill-svc",
            False,
            None,
        )
        return ReplicaUpdateResult(accepted=True, applied=False)

    connector.set_component_replicas_with_result.side_effect = accept_noop

    result = await cm.scale_with_result(
        "default/my-dgd",
        [
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL,
                desired_replicas=1,
            )
        ],
        blocking=True,
    )

    assert result.accepted
    assert not result.applied


@pytest.mark.asyncio
async def test_scale_preserves_public_none_return():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))

    async def accept(_targets, *, blocking, retry_state):
        retry_state.dgd_uid = "dgd-uid"
        retry_state.authorities[("prefill", None)] = (
            "prefill-svc",
            False,
            None,
        )
        return ReplicaUpdateResult(accepted=True, applied=True)

    connector.set_component_replicas_with_result.side_effect = accept
    result = await cm.scale(
        "default/my-dgd",
        [
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL,
                desired_replicas=3,
            )
        ],
        blocking=True,
    )

    assert result is None
    connector.set_component_replicas_with_result.assert_awaited_once()
    connector.wait_for_replica_targets.assert_awaited_once_with(
        expected_dgd_uid="dgd-uid",
        expected_targets={"prefill-svc": 3},
        expected_authorities={"prefill-svc": (False, None)},
    )


@pytest.mark.asyncio
async def test_compatibility_scale_joins_committed_write_after_cancellation():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))
    started = asyncio.Event()
    release = asyncio.Event()

    async def accept_after_release(_targets, *, blocking, retry_state):
        retry_state.dgd_uid = "dgd-uid"
        retry_state.authorities[("prefill", None)] = (
            "prefill-svc",
            False,
            None,
        )
        started.set()
        await release.wait()
        return ReplicaUpdateResult(accepted=True, applied=True)

    connector.set_component_replicas_with_result.side_effect = accept_after_release
    task = asyncio.create_task(
        cm.scale(
            "default/my-dgd",
            [
                TargetReplica(
                    sub_component_type=SubComponentType.PREFILL,
                    desired_replicas=2,
                )
            ],
            blocking=True,
        )
    )
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)

    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    connector.wait_for_replica_targets.assert_awaited_once()
    assert "default/my-dgd" not in cm._replica_settlements


@pytest.mark.asyncio
async def test_compatibility_scale_cancellation_during_reconciliation_stops_new_write():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))

    async def accept(_targets, *, blocking, retry_state):
        retry_state.dgd_uid = "dgd-uid"
        retry_state.authorities[("prefill", None)] = (
            "prefill-svc",
            False,
            None,
        )
        return ReplicaUpdateResult(accepted=True, applied=True)

    connector.set_component_replicas_with_result.side_effect = accept
    first_target = TargetReplica(
        sub_component_type=SubComponentType.PREFILL,
        desired_replicas=2,
    )
    await cm.scale_with_result("default/my-dgd", [first_target], blocking=False)

    settling = asyncio.Event()
    release = asyncio.Event()

    async def wait_for_first_target(**_kwargs):
        settling.set()
        await release.wait()

    connector.wait_for_replica_targets.side_effect = wait_for_first_target
    second_target = first_target.model_copy(update={"desired_replicas": 3})
    task = asyncio.create_task(
        cm.scale("default/my-dgd", [second_target], blocking=False)
    )
    await settling.wait()
    task.cancel()
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert connector.set_component_replicas_with_result.await_count == 1
    assert cm._replica_settlements["default/my-dgd"].protected


@pytest.mark.asyncio
async def test_cancelled_scale_with_result_protects_recorded_settlement():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))
    started = asyncio.Event()
    release = asyncio.Event()

    async def accept_after_release(_targets, *, blocking, retry_state):
        retry_state.dgd_uid = "dgd-uid"
        retry_state.authorities[("prefill", None)] = (
            "prefill-svc",
            False,
            None,
        )
        started.set()
        await release.wait()
        return ReplicaUpdateResult(accepted=True, applied=True)

    connector.set_component_replicas_with_result.side_effect = accept_after_release
    target = TargetReplica(
        sub_component_type=SubComponentType.PREFILL,
        desired_replicas=2,
    )
    task = asyncio.create_task(
        cm.scale_with_result("default/my-dgd", [target], blocking=False)
    )
    await started.wait()
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cm._replica_settlements["default/my-dgd"].protected
    with pytest.raises(RuntimeError, match="awaiting settlement"):
        await cm.scale_with_result("default/my-dgd", [target], blocking=False)


@pytest.mark.asyncio
async def test_run_with_conflict_retry_restarts_complete_operation():
    cm = KubernetesCapacityManager("default")
    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise client.ApiException(status=409)
        return "applied"

    assert await cm.run_with_conflict_retry(operation) == "applied"
    assert attempts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("resource", ["DGD", "DGDSA"])
@pytest.mark.parametrize(
    "mutation",
    ["resource-version", "uid", "appearance", "disappearance"],
)
async def test_run_with_conflict_retry_restarts_snapshot_drift(resource, mutation):
    cm = KubernetesCapacityManager("default")
    stable = {
        "model": {
            "metadata": {"name": "model", "uid": "object-uid", "resourceVersion": "1"}
        }
    }
    before = deepcopy(stable)
    changed = deepcopy(stable)
    if mutation == "resource-version":
        changed["model"]["metadata"]["resourceVersion"] = "2"
    elif mutation == "uid":
        changed["model"]["metadata"]["uid"] = "replacement-uid"
    elif mutation == "appearance":
        before = {}
    else:
        changed = {}
    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        cm._require_inventory_unchanged(
            before,
            changed if attempts == 1 else before,
            {"model"},
            resource,
        )
        return "stable"

    assert await cm.run_with_conflict_retry(operation) == "stable"
    assert attempts == 2


@pytest.mark.asyncio
async def test_run_with_conflict_retry_reuses_replica_identity_state():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))
    attempts = 0

    async def conflict_then_accept(_targets, *, blocking, retry_state):
        nonlocal attempts
        attempts += 1
        retry_state.dgd_uid = "dgd-uid"
        retry_state.authorities[("prefill", None)] = (
            "prefill-svc",
            False,
            None,
        )
        if attempts == 1:
            raise client.ApiException(status=409)
        return ReplicaUpdateResult(accepted=True, applied=True)

    connector.set_component_replicas_with_result.side_effect = conflict_then_accept
    targets = [
        TargetReplica(
            sub_component_type=SubComponentType.PREFILL,
            desired_replicas=2,
        )
    ]

    async def operation():
        return await cm.scale_with_result("default/my-dgd", targets, blocking=False)

    result = await cm.run_with_conflict_retry(operation)

    assert result.applied
    first_state = connector.set_component_replicas_with_result.await_args_list[
        0
    ].kwargs["retry_state"]
    second_state = connector.set_component_replicas_with_result.await_args_list[
        1
    ].kwargs["retry_state"]
    assert first_state is second_state


@pytest.mark.asyncio
async def test_wait_for_ready_uses_accepted_replica_transaction():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))

    async def accept(_targets, *, blocking, retry_state):
        retry_state.dgd_uid = "dgd-uid"
        retry_state.authorities[("prefill", None)] = (
            "prefill-svc",
            False,
            None,
        )
        return ReplicaUpdateResult(accepted=True, applied=True)

    connector.set_component_replicas_with_result.side_effect = accept
    await cm.scale_with_result(
        "default/my-dgd",
        [
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL,
                desired_replicas=2,
            )
        ],
        blocking=False,
    )

    await cm.wait_for_ready("default/my-dgd")

    connector.wait_for_replica_targets.assert_awaited_once_with(
        expected_dgd_uid="dgd-uid",
        expected_targets={"prefill-svc": 2},
        expected_authorities={"prefill-svc": (False, None)},
    )


@pytest.mark.asyncio
async def test_cancelled_wait_for_ready_protects_settlement():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))

    async def accept(_targets, *, blocking, retry_state):
        retry_state.dgd_uid = "dgd-uid"
        retry_state.authorities[("prefill", None)] = (
            "prefill-svc",
            False,
            None,
        )
        return ReplicaUpdateResult(accepted=True, applied=True)

    waiting = asyncio.Event()

    async def wait_for_target(**_kwargs):
        waiting.set()
        await asyncio.Event().wait()

    connector.set_component_replicas_with_result.side_effect = accept
    connector.wait_for_replica_targets.side_effect = wait_for_target
    await cm.scale_with_result(
        "default/my-dgd",
        [
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL,
                desired_replicas=2,
            )
        ],
        blocking=False,
    )
    task = asyncio.create_task(cm.wait_for_ready("default/my-dgd"))
    await waiting.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cm._replica_settlements["default/my-dgd"].protected


@pytest.mark.asyncio
async def test_possibly_committed_write_settles_before_failure_is_reported():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))
    target = TargetReplica(
        sub_component_type=SubComponentType.PREFILL,
        component_name="prefill-svc",
        desired_replicas=0,
    )
    attempt = ReplicaUpdateAttempt(
        target=target,
        component_name="prefill-svc",
        previous_replicas=1,
        dgd_uid="dgd-uid",
        is_adapter=False,
        authority_uid=None,
    )
    failure = ReplicaUpdateFailure(RuntimeError("ambiguous"), [], [attempt])

    async def fail_after_dispatch(_targets, *, blocking, retry_state):
        retry_state.dgd_uid = "dgd-uid"
        raise failure

    connector.set_component_replicas_with_result.side_effect = fail_after_dispatch
    connector.resolve_uncertain_replica_attempts.return_value = (attempt,)

    result = await cm.scale_with_result("default/my-dgd", [target], blocking=False)
    with pytest.raises(ReplicaUpdateFailure) as raised:
        await cm.wait_for_ready("default/my-dgd")

    assert result == ReplicaUpdateResult(
        accepted=True, applied=True, requires_settlement=True
    )
    assert raised.value is failure
    connector.wait_for_replica_targets.assert_awaited_once_with(
        expected_dgd_uid="dgd-uid",
        expected_targets={"prefill-svc": 0},
        expected_authorities={"prefill-svc": (False, None)},
    )
    assert "default/my-dgd" not in cm._replica_settlements


@pytest.mark.asyncio
async def test_compatibility_scale_settles_possibly_committed_write():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))
    target = TargetReplica(
        sub_component_type=SubComponentType.PREFILL,
        component_name="prefill-svc",
        desired_replicas=0,
    )
    attempt = ReplicaUpdateAttempt(
        target=target,
        component_name="prefill-svc",
        previous_replicas=1,
        dgd_uid="dgd-uid",
        is_adapter=False,
        authority_uid=None,
    )
    failure = ReplicaUpdateFailure(RuntimeError("ambiguous"), [], [attempt])

    async def fail_after_dispatch(_targets, *, blocking, retry_state):
        retry_state.dgd_uid = "dgd-uid"
        raise failure

    connector.set_component_replicas_with_result.side_effect = fail_after_dispatch
    connector.resolve_uncertain_replica_attempts.return_value = (attempt,)

    with pytest.raises(ReplicaUpdateFailure) as raised:
        await cm.scale("default/my-dgd", [target], blocking=False)

    assert raised.value is failure
    connector.wait_for_replica_targets.assert_awaited_once_with(
        expected_dgd_uid="dgd-uid",
        expected_targets={"prefill-svc": 0},
        expected_authorities={"prefill-svc": (False, None)},
    )
    assert "default/my-dgd" not in cm._replica_settlements


@pytest.mark.asyncio
async def test_unresolved_replica_write_blocks_complete_observation():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))
    target = TargetReplica(
        sub_component_type=SubComponentType.PREFILL,
        component_name="prefill-svc",
        desired_replicas=0,
    )
    attempt = ReplicaUpdateAttempt(
        target=target,
        component_name="prefill-svc",
        previous_replicas=1,
        dgd_uid="dgd-uid",
        is_adapter=False,
        authority_uid=None,
    )
    failure = ReplicaUpdateFailure(RuntimeError("ambiguous"), [], [attempt])

    async def fail_after_dispatch(_targets, *, blocking, retry_state):
        retry_state.dgd_uid = "dgd-uid"
        raise failure

    connector.set_component_replicas_with_result.side_effect = fail_after_dispatch
    connector.resolve_uncertain_replica_attempts.side_effect = RuntimeError(
        "readback failed"
    )
    await cm.scale_with_result("default/my-dgd", [target], blocking=False)

    with pytest.raises(RuntimeError, match="readback failed"):
        await cm.wait_for_ready("default/my-dgd")
    with pytest.raises(RuntimeError, match="awaiting authoritative settlement"):
        cm.observe(require_complete=True)

    assert cm._replica_settlements["default/my-dgd"].protected

    fenced_attempt = ReplicaUpdateAttempt(
        target=target.model_copy(update={"desired_replicas": 1}),
        component_name="prefill-svc",
        previous_replicas=1,
        dgd_uid="dgd-uid",
        is_adapter=False,
        authority_uid=None,
    )
    connector.resolve_uncertain_replica_attempts.side_effect = None
    connector.resolve_uncertain_replica_attempts.return_value = (fenced_attempt,)
    await cm.reconcile_pending_settlements("default/my-dgd")

    connector.wait_for_replica_targets.assert_awaited_once_with(
        expected_dgd_uid="dgd-uid",
        expected_targets={"prefill-svc": 1},
        expected_authorities={"prefill-svc": (False, None)},
    )
    assert "default/my-dgd" not in cm._replica_settlements
    assert cm.observe(require_complete=True)["default/my-dgd"]


# ---------------------------------------------------------------------------- #
# Observe / current_replicas (v1beta1 component reading)                       #
# ---------------------------------------------------------------------------- #


def test_observe_parses_typed_components():
    cm = KubernetesCapacityManager("default")
    _install_connector(
        cm,
        "default/my-dgd",
        _dgd_spec(prefill_replicas=2, decode_replicas=3, decode_gpu=2),
    )
    pools = cm.observe()["default/my-dgd"]
    assert pools["prefill"] == PoolSpec(
        sub_type="prefill",
        current_replicas=2,
        gpu_per_replica=1,
        component_name="prefill-svc",
    )
    assert pools["decode"] == PoolSpec(
        sub_type="decode",
        current_replicas=3,
        gpu_per_replica=2,
        component_name="decode-svc",
    )


@pytest.mark.parametrize("implicit_inventory", [False, True])
def test_observe_uses_authoritative_adapter_target(implicit_inventory):
    cm = KubernetesCapacityManager("default")
    deployment = _dgd_spec(1, 1)
    deployment["spec"]["components"][0]["scalingAdapter"] = {}
    connector = _install_connector(
        cm, "default/model", deployment, parent_dgd_name="model"
    )
    authoritative = _listed_dgd(deployment, "model")
    authoritative["spec"]["components"][0]["replicas"] = 3
    connector.kube_api.fetch_authoritative_replica_targets.return_value = authoritative

    if implicit_inventory:
        cm._observe_all_deployments = True
        cm._inventory_api = connector.kube_api
        connector.kube_api.list_graph_deployments.return_value = [
            _listed_dgd(deployment, "model")
        ]

    pools = cm.observe()["default/model"]

    assert pools["prefill"].current_replicas == 3
    connector.kube_api.fetch_authoritative_replica_targets.assert_called_once()
    observed = connector.kube_api.fetch_authoritative_replica_targets.call_args.args[0]
    assert observed["spec"]["components"][0]["replicas"] == 1


def test_complete_implicit_observation_batches_authorities_and_fails_closed():
    cm = KubernetesCapacityManager("default")
    cm._observe_all_deployments = True
    deployment = _listed_dgd(_dgd_spec(1, 1), "model")
    deployment["spec"]["components"][0]["scalingAdapter"] = {}
    prefill_status = deployment["status"]["components"]["prefill-svc"]
    prefill_status.update(
        {
            "replicas": 2,
            "updatedReplicas": 2,
            "readyReplicas": 2,
            "availableReplicas": 2,
        }
    )
    connector = _install_connector(
        cm,
        "default/model",
        deployment,
        parent_dgd_name="model",
    )
    cm._inventory_api = connector.kube_api
    connector.kube_api.list_graph_deployments.return_value = [deployment]
    connector.kube_api.list_scaling_adapters.return_value = [
        {
            "metadata": {
                "name": "model-prefill-svc",
                "uid": "adapter-uid",
                "resourceVersion": "adapter-rv-1",
            }
        }
    ]
    terminating = MagicMock()
    terminating.metadata.labels = {
        "nvidia.com/dynamo-graph-deployment-name": "model",
        "nvidia.com/dynamo-graph-deployment-uid": "dgd-uid",
        "nvidia.com/dynamo-component": "prefill-svc",
    }
    terminating.metadata.owner_references = []
    terminating.metadata.deletion_timestamp = object()
    terminating.status = None
    connector.kube_api.list_graph_pods.return_value = [terminating]

    with pytest.raises(RuntimeError, match="still converging"):
        cm.observe(require_complete=True)

    prefill_status.update(
        {
            "replicas": 1,
            "updatedReplicas": 1,
            "readyReplicas": 1,
            "availableReplicas": 1,
        }
    )
    with pytest.raises(RuntimeError, match="still draining"):
        cm.observe(require_complete=True)

    connector.kube_api.list_graph_pods.return_value = []
    snapshot = cm.observe(require_complete=True)

    assert snapshot["default/model"]["prefill"].current_replicas == 1
    assert connector.kube_api.list_graph_deployments.call_count == 3
    assert connector.kube_api.list_graph_deployment_metadata.call_count == 3
    assert connector.kube_api.list_scaling_adapters.call_count == 3
    assert connector.kube_api.list_scaling_adapter_metadata.call_count == 3
    assert connector.kube_api.list_graph_pods.call_count == 3
    connector.kube_api.get_graph_deployment.assert_not_called()


def test_complete_implicit_observation_rejects_duplicate_adapter_inventory():
    cm = KubernetesCapacityManager("default")
    cm._observe_all_deployments = True
    deployment = _listed_dgd(_dgd_spec(1, 1), "model")
    deployment["spec"]["components"][0]["scalingAdapter"] = {}
    connector = _install_connector(
        cm,
        "default/model",
        deployment,
        parent_dgd_name="model",
    )
    cm._inventory_api = connector.kube_api
    connector.kube_api.list_graph_deployments.return_value = [deployment]
    connector.kube_api.list_scaling_adapters.return_value = [
        {"metadata": {"name": "model-prefill-svc"}},
        {"metadata": {"name": "model-prefill-svc"}},
    ]

    with pytest.raises(RuntimeError, match="duplicate DGDSA"):
        cm.observe(require_complete=True)


def test_complete_implicit_observation_rejects_pods_from_replaced_dgd():
    cm = KubernetesCapacityManager("default")
    cm._observe_all_deployments = True
    deployment = _listed_dgd(_dgd_spec(1, 1), "model")
    connector = _install_connector(
        cm,
        "default/model",
        deployment,
        parent_dgd_name="model",
    )
    cm._inventory_api = connector.kube_api
    connector.kube_api.list_graph_deployments.return_value = [deployment]
    pod = {
        "metadata": {
            "name": "prefill-old",
            "labels": {
                "nvidia.com/dynamo-graph-deployment-name": "model",
                "nvidia.com/dynamo-graph-deployment-uid": "old-dgd-uid",
                "nvidia.com/dynamo-component": "prefill-svc",
            },
        }
    }
    connector.kube_api.list_graph_pods.return_value = [pod]

    with pytest.raises(RuntimeError, match="still draining"):
        cm.observe(require_complete=True)

    pod["metadata"]["labels"]["nvidia.com/dynamo-graph-deployment-uid"] = "dgd-uid"
    snapshot = cm.observe(require_complete=True)

    assert snapshot["default/model"]["prefill"].current_replicas == 1


def test_complete_implicit_observation_rejects_removed_component_pods():
    cm = KubernetesCapacityManager("default")
    cm._observe_all_deployments = True
    deployment = _listed_dgd(_dgd_spec(1, 1), "model")
    connector = _install_connector(
        cm,
        "default/model",
        deployment,
        parent_dgd_name="model",
    )
    cm._inventory_api = connector.kube_api
    connector.kube_api.list_graph_deployments.return_value = [deployment]
    connector.kube_api.list_graph_pods.return_value = [
        {
            "metadata": {
                "name": "removed-worker-0",
                "labels": {
                    "nvidia.com/dynamo-graph-deployment-name": "model",
                    "nvidia.com/dynamo-graph-deployment-uid": "dgd-uid",
                    "nvidia.com/dynamo-component": "removed-worker",
                },
            }
        }
    ]

    with pytest.raises(RuntimeError, match="still draining"):
        cm.observe(require_complete=True)


def test_complete_implicit_observation_rejects_deployment_surge_pods():
    cm = KubernetesCapacityManager("default")
    cm._observe_all_deployments = True
    deployment = _listed_dgd(_dgd_spec(0, 1), "model")
    deployment["status"]["components"]["decode-svc"]["componentKind"] = "Deployment"
    deployment["spec"]["components"][1]["podTemplate"]["metadata"] = {
        "annotations": {
            "nvidia.com/deployment-rolling-update-max-surge": "0",
        }
    }
    connector = _install_connector(
        cm,
        "default/model",
        deployment,
        parent_dgd_name="model",
    )
    cm._inventory_api = connector.kube_api
    connector.kube_api.list_graph_deployments.return_value = [deployment]
    connector.kube_api.list_graph_pods.return_value = [
        {
            "metadata": {
                "name": f"decode-{index}",
                "labels": {
                    "nvidia.com/dynamo-graph-deployment-name": "model",
                    "nvidia.com/dynamo-graph-deployment-uid": "dgd-uid",
                    "nvidia.com/dynamo-component": "decode-svc",
                },
            }
        }
        for index in range(2)
    ]

    with pytest.raises(RuntimeError, match="still draining"):
        cm.observe(require_complete=True)


def test_complete_observation_rejects_deployment_that_can_surge():
    cm = KubernetesCapacityManager("default")
    deployment = _dgd_spec(0, 1)
    deployment["status"]["components"]["decode-svc"]["componentKind"] = "Deployment"
    _install_connector(cm, "default/my-dgd", deployment)

    with pytest.raises(RuntimeError, match="maxSurge=0"):
        cm.observe(require_complete=True)


def test_complete_implicit_observation_rejects_dgd_change_during_pod_inventory():
    cm = KubernetesCapacityManager("default")
    cm._observe_all_deployments = True
    deployment = _listed_dgd(_dgd_spec(0, 0), "model")
    replacement = _listed_dgd(_dgd_spec(8, 0), "model")
    replacement["metadata"].update(
        {"uid": "replacement-uid", "resourceVersion": "dgd-rv-2"}
    )
    connector = _install_connector(
        cm,
        "default/model",
        deployment,
        parent_dgd_name="model",
    )
    cm._inventory_api = connector.kube_api
    connector.kube_api.list_graph_deployments.side_effect = [
        [deployment],
    ]
    connector.kube_api.list_graph_deployment_metadata.return_value = [replacement]
    connector.kube_api.list_graph_deployment_metadata.side_effect = None

    with pytest.raises(client.ApiException) as error:
        cm.observe(require_complete=True)
    assert error.value.status == 409


def test_observe_multinode_scales_gpu_count():
    cm = KubernetesCapacityManager("default")
    _install_connector(
        cm, "default/my-dgd", _dgd_spec(1, 1, prefill_gpu=2, prefill_node_count=2)
    )
    # gpu_per_replica = main-container GPUs (2) × nodeCount (2) = 4.
    assert cm.observe()["default/my-dgd"]["prefill"].gpu_per_replica == 4


def test_observe_dra_worker_uses_operator_resolved_gpu_count():
    cm = KubernetesCapacityManager("default")
    deployment = {
        "metadata": {"generation": 2},
        "spec": {
            "components": [
                _component(
                    "decode-svc", replicas=1, gpu=None, ctype="decode", node_count=2
                )
            ]
        },
        "status": {
            "observedGeneration": 2,
            "components": {
                "decode-svc": {
                    "replicas": 1,
                    "updatedReplicas": 1,
                    "readyReplicas": 1,
                    "availableReplicas": 1,
                    "gpusPerEngine": 4,
                    "gpusPerReplica": 5,
                }
            },
        },
    }
    _install_connector(cm, "default/my-dgd", deployment)

    pools = cm.observe(require_complete=True)["default/my-dgd"]

    assert pools["decode"].gpu_per_replica == 5


def test_observe_generic_worker_keyed_by_name_then_role_hint():
    cm = KubernetesCapacityManager("default")
    _install_connector(cm, "default/w", _worker_dgd_spec(replicas=2, gpu=2))

    # No hint yet: the worker still counts, keyed by its component name.
    pools = cm.observe()["default/w"]
    assert "worker-svc" in pools and pools["worker-svc"].gpu_per_replica == 2

    # After its Planner sends a role hint, the worker maps to that pool.
    cm.remember_roles(
        "default/w",
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker-svc",
                desired_replicas=2,
            )
        ],
    )
    pools = cm.observe()["default/w"]
    assert "worker-svc" not in pools
    assert pools["decode"].component_name == "worker-svc"


def test_observe_untyped_worker_with_gpu_counts_toward_budget():
    cm = KubernetesCapacityManager("default")
    _install_connector(cm, "default/w", _worker_dgd_spec(replicas=2, gpu=2, ctype=None))
    pools = cm.observe()["default/w"]
    assert pools["worker-svc"].gpu_per_replica == 2


def test_observe_skips_explicit_zero_gpu_dra_component():
    cm = KubernetesCapacityManager("default")
    component = _component("frontend", replicas=1, gpu=None, ctype=None)
    component["podTemplate"]["spec"] = {
        "resourceClaims": [
            {"name": "rdma", "resourceClaimTemplateName": "frontend-rdma"}
        ],
        "containers": [{"name": "main", "resources": {"claims": [{"name": "rdma"}]}}],
    }
    deployment = {
        "metadata": {"generation": 2},
        "spec": {"components": [component]},
        "status": {
            "observedGeneration": 2,
            "components": {
                "frontend": {
                    "replicas": 1,
                    "updatedReplicas": 1,
                    "readyReplicas": 1,
                    "availableReplicas": 1,
                    "gpusPerEngine": 0,
                    "gpusPerReplica": 0,
                }
            },
        },
    }
    _install_connector(cm, "default/frontend", deployment)

    assert cm.observe(require_complete=True)["default/frontend"] == {}


def test_observe_untyped_dra_worker_missing_shape_fails_closed():
    cm = KubernetesCapacityManager("default")
    component = _component("worker", replicas=1, gpu=None, ctype=None)
    component["podTemplate"]["spec"] = {
        "resourceClaims": [{"name": "gpu", "resourceClaimTemplateName": "worker-gpu"}],
        "containers": [{"name": "main", "resources": {"claims": [{"name": "gpu"}]}}],
    }
    deployment = {
        "metadata": {"generation": 2},
        "spec": {"components": [component]},
        "status": {
            "observedGeneration": 2,
            "components": {
                "worker": {
                    "replicas": 1,
                    "updatedReplicas": 1,
                    "readyReplicas": 1,
                    "availableReplicas": 1,
                }
            },
        },
    }
    _install_connector(cm, "default/worker", deployment)

    with pytest.raises(RuntimeError, match="GPU shape") as exc_info:
        cm.observe(require_complete=True)
    assert isinstance(exc_info.value.__cause__, GPUShapeUnavailableError)


def test_observe_tolerates_read_failure():
    cm = KubernetesCapacityManager("default")
    good = _install_connector(
        cm, "default/good", _dgd_spec(1, 1), parent_dgd_name="good"
    )
    _install_connector(
        cm,
        "default/bad",
        {
            "metadata": {"generation": 1},
            "spec": {"components": [{}]},
            "status": {"observedGeneration": 1, "components": {}},
        },
        parent_dgd_name="bad",
    )

    snapshot = cm.observe()
    assert snapshot["default/good"]["prefill"].current_replicas == 1
    assert snapshot["default/bad"] == {}  # tolerated, empty
    good.kube_api.get_graph_deployment.assert_called_once_with("good")
    good.kube_api.list_graph_deployments.assert_not_called()


def test_observe_require_complete_raises_on_read_failure():
    cm = KubernetesCapacityManager("default")
    _install_connector(
        cm,
        "default/bad",
        {
            "metadata": {"generation": 1},
            "spec": {"components": [{}]},
            "status": {"observedGeneration": 1, "components": {}},
        },
        parent_dgd_name="bad",
    )
    with pytest.raises(RuntimeError):
        cm.observe(require_complete=True)


def test_observe_never_seen_managed_deployment_is_zero_capacity_probe():
    cm = KubernetesCapacityManager("default")
    missing = _install_connector(
        cm,
        "default/missing",
        _dgd_spec(1, 1),
        parent_dgd_name="missing",
        seed_identity=False,
    )
    missing.kube_api.get_graph_deployment.side_effect = (
        DynamoGraphDeploymentNotFoundError("missing", "default")
    )

    assert cm.observe(require_complete=True) == {"default/missing": {}}
    assert not cm.participant_exists("default/missing")
    missing.kube_api.list_graph_pods.assert_called_once_with({"missing"})
    missing.kube_api.list_nonterminal_pods_for_graph.assert_not_called()

    missing = _install_connector(
        cm,
        "default/missing",
        _dgd_spec(1, 1),
        parent_dgd_name="missing",
        seed_identity=False,
    )
    missing.kube_api.get_graph_deployment.side_effect = (
        DynamoGraphDeploymentNotFoundError("missing", "default")
    )
    assert cm.observe() == {"default/missing": {}}
    assert not cm.participant_exists("default/missing")
    missing.kube_api.list_nonterminal_pods_for_graph.assert_not_called()


def test_observe_never_seen_missing_deployment_rejects_residual_pods():
    cm = KubernetesCapacityManager("default")
    missing = _install_connector(
        cm,
        "default/missing",
        _dgd_spec(1, 1),
        parent_dgd_name="missing",
        seed_identity=False,
    )
    missing.kube_api.get_graph_deployment.side_effect = (
        DynamoGraphDeploymentNotFoundError("missing", "default")
    )
    missing.kube_api.list_graph_pods.return_value = [
        {
            "metadata": {
                "labels": {
                    "nvidia.com/dynamo-graph-deployment-name": "missing",
                }
            }
        }
    ]

    with pytest.raises(RuntimeError, match="still has 1 nonterminal Pods"):
        cm.observe(require_complete=True)

    assert not cm.participant_exists("default/missing")


def test_observe_deleted_deployment_fails_closed_while_old_pods_remain():
    cm = KubernetesCapacityManager("default")
    deployment = _dgd_spec(2, 1)
    connector = _install_connector(
        cm, "default/model", deployment, parent_dgd_name="model"
    )
    connector.kube_api.get_graph_deployment.side_effect = [
        deployment,
        deployment,
        DynamoGraphDeploymentNotFoundError("model", "default"),
        DynamoGraphDeploymentNotFoundError("model", "default"),
    ]
    connector.kube_api.list_graph_pods.side_effect = [
        [],
        [
            {
                "metadata": {
                    "name": "prefill-old",
                    "labels": {
                        "nvidia.com/dynamo-graph-deployment-name": "model",
                        "nvidia.com/dynamo-graph-deployment-uid": "dgd-uid",
                        "nvidia.com/dynamo-component": "prefill-svc",
                    },
                }
            }
        ],
    ]

    initial = cm.observe(require_complete=True)
    with pytest.raises(RuntimeError, match="capacity drain cannot be proven"):
        cm.observe(require_complete=True)

    assert initial["default/model"]["prefill"].current_replicas == 2
    assert cm._observed_uids["default/model"] == "dgd-uid"
    assert cm._last_observed_pools["default/model"]["prefill"].current_replicas == 2
    assert not cm.participant_exists("default/model")


def test_observe_deleted_deployment_stays_reserved_after_pods_drain():
    cm = KubernetesCapacityManager("default")
    deployment = _dgd_spec(2, 1)
    connector = _install_connector(
        cm, "default/model", deployment, parent_dgd_name="model"
    )
    connector.kube_api.get_graph_deployment.side_effect = [
        deployment,
        deployment,
        DynamoGraphDeploymentNotFoundError("model", "default"),
        DynamoGraphDeploymentNotFoundError("model", "default"),
    ]

    cm.observe(require_complete=True)
    cm._replica_uids["default/model"] = "dgd-uid"

    with pytest.raises(RuntimeError, match="capacity drain cannot be proven"):
        cm.observe(require_complete=True)

    assert cm._observed_uids["default/model"] == "dgd-uid"
    assert cm._last_observed_pools["default/model"]["prefill"].current_replicas == 2
    assert cm._replica_uids["default/model"] == "dgd-uid"
    assert not cm.participant_exists("default/model")


def test_observe_same_name_replacement_fails_closed_and_preserves_uid_bound_state():
    cm = KubernetesCapacityManager("default")
    deployment = _dgd_spec(2, 1)
    replacement = deepcopy(deployment)
    replacement["metadata"]["uid"] = "replacement-uid"
    connector = _install_connector(
        cm, "default/model", deployment, parent_dgd_name="model"
    )
    connector.kube_api.get_graph_deployment.side_effect = [
        deployment,
        deployment,
        replacement,
        replacement,
    ]
    connector.kube_api.list_graph_pods.side_effect = [
        [],
        [
            {
                "metadata": {
                    "name": "prefill-old",
                    "labels": {
                        "nvidia.com/dynamo-graph-deployment-name": "model",
                        "nvidia.com/dynamo-graph-deployment-uid": "dgd-uid",
                        "nvidia.com/dynamo-component": "prefill-svc",
                    },
                }
            }
        ],
    ]

    cm.observe(require_complete=True)
    cm._replica_uids["default/model"] = "dgd-uid"

    with pytest.raises(RuntimeError, match="still draining"):
        cm.observe(require_complete=True)

    assert cm._observed_uids["default/model"] == "dgd-uid"
    assert cm._replica_uids["default/model"] == "dgd-uid"
    assert cm.participant_exists("default/model")


def test_observe_same_name_replacement_stays_reserved_after_old_pods_drain():
    cm = KubernetesCapacityManager("default")
    deployment = _dgd_spec(2, 1)
    replacement = deepcopy(deployment)
    replacement["metadata"]["uid"] = "replacement-uid"
    connector = _install_connector(
        cm, "default/model", deployment, parent_dgd_name="model"
    )
    connector.kube_api.get_graph_deployment.side_effect = [
        deployment,
        deployment,
        replacement,
        replacement,
    ]

    cm.observe(require_complete=True)
    cm._replica_uids["default/model"] = "dgd-uid"

    with pytest.raises(RuntimeError, match="capacity drain cannot be proven"):
        cm.observe(require_complete=True)

    assert cm._observed_uids["default/model"] == "dgd-uid"
    assert cm._last_observed_pools["default/model"]["prefill"].current_replicas == 2
    assert cm._replica_uids["default/model"] == "dgd-uid"
    assert cm.participant_exists("default/model")


def test_observe_explicit_mode_discovers_previously_missing_deployment():
    cm = KubernetesCapacityManager("default")
    cm._configured_deployment_names = {"model"}
    deployment = _dgd_spec(1, 1)
    old = _install_connector(cm, "default/model", deployment, parent_dgd_name="model")
    old.kube_api.get_graph_deployment.side_effect = [
        DynamoGraphDeploymentNotFoundError("model", "default"),
        DynamoGraphDeploymentNotFoundError("model", "default"),
        deployment,
        deployment,
    ]
    cm._observed_uids.pop("default/model")

    assert cm.observe(require_complete=True) == {"default/model": {}}
    assert "default/model" in cm.connectors
    assert not cm.participant_exists("default/model")
    assert cm._missing_participants == {"default/model"}

    snapshot = cm.observe(require_complete=True)

    assert snapshot["default/model"]["prefill"].current_replicas == 1
    assert cm.connectors["default/model"] is old
    assert cm._missing_participants == set()
    assert cm.participant_exists("default/model")


def test_observe_explicit_mode_rechecks_missing_names_with_scoped_gets():
    cm = KubernetesCapacityManager("default")
    cm._configured_deployment_names = {"first", "second"}
    first = _install_connector(
        cm,
        "default/first",
        _dgd_spec(1, 1),
        parent_dgd_name="first",
        seed_identity=False,
    )
    second = _install_connector(
        cm,
        "default/second",
        _dgd_spec(1, 1),
        parent_dgd_name="second",
        seed_identity=False,
    )
    first.kube_api.get_graph_deployment.side_effect = (
        DynamoGraphDeploymentNotFoundError("first", "default")
    )
    second.kube_api.get_graph_deployment.side_effect = (
        DynamoGraphDeploymentNotFoundError("second", "default")
    )

    assert cm.observe(require_complete=True) == {
        "default/first": {},
        "default/second": {},
    }
    assert cm.observe(require_complete=True) == {
        "default/first": {},
        "default/second": {},
    }

    assert first.kube_api.get_graph_deployment.call_count == 4
    assert second.kube_api.get_graph_deployment.call_count == 4
    first.kube_api.list_graph_deployments.assert_not_called()


def test_observe_explicit_mode_does_not_require_list_permission():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(
        cm, "default/model", _dgd_spec(1, 1), parent_dgd_name="model"
    )
    connector.kube_api.list_graph_deployments.side_effect = PermissionError(
        "list forbidden"
    )

    snapshot = cm.observe(require_complete=True)

    assert snapshot["default/model"]["prefill"].current_replicas == 1
    assert connector.kube_api.get_graph_deployment.call_count == 2
    connector.kube_api.list_graph_deployments.assert_not_called()


def test_observe_explicit_mode_batches_pod_inventory():
    cm = KubernetesCapacityManager("default")
    first = _install_connector(
        cm, "default/first", _dgd_spec(1, 1), parent_dgd_name="first"
    )
    second = _install_connector(
        cm, "default/second", _dgd_spec(2, 1), parent_dgd_name="second"
    )

    snapshot = cm.observe(require_complete=True)

    assert snapshot["default/first"]["prefill"].current_replicas == 1
    assert snapshot["default/second"]["prefill"].current_replicas == 2
    first.kube_api.list_graph_pods.assert_called_once_with({"first", "second"})
    second.kube_api.list_graph_pods.assert_not_called()
    first.kube_api.list_nonterminal_pods_for_graph.assert_not_called()
    second.kube_api.list_nonterminal_pods_for_graph.assert_not_called()
    assert first.kube_api.get_graph_deployment.call_count == 2
    assert second.kube_api.get_graph_deployment.call_count == 2


def test_observe_explicit_mode_rejects_adapter_change_during_pod_inventory():
    cm = KubernetesCapacityManager("default")
    deployment = _dgd_spec(1, 1)
    deployment["spec"]["components"][0]["scalingAdapter"] = {}
    connector = _install_connector(
        cm, "default/model", deployment, parent_dgd_name="model"
    )
    adapter = {
        "metadata": {
            "name": "model-prefill-svc",
            "uid": "adapter-uid",
            "resourceVersion": "adapter-rv-1",
        }
    }
    replacement = deepcopy(adapter)
    replacement["metadata"].update(
        {"uid": "replacement-uid", "resourceVersion": "adapter-rv-2"}
    )
    connector.kube_api.get_service_scaling_adapter.side_effect = [
        adapter,
        replacement,
    ]

    with pytest.raises(RuntimeError, match="DGDSA.*identity changed"):
        cm.observe(require_complete=True)


def test_observe_explicit_mode_with_only_remote_deployments_uses_targeted_reads():
    cm = KubernetesCapacityManager("default")
    remote = _install_connector(
        cm, "tenant/model", _dgd_spec(1, 1), parent_dgd_name="model"
    )

    snapshot = cm.observe(require_complete=True)

    assert snapshot["tenant/model"]["prefill"].current_replicas == 1
    assert remote.kube_api.get_graph_deployment.call_count == 2
    remote.kube_api.list_nonterminal_pods_for_graph.assert_called_once_with("model")
    remote.kube_api.list_graph_pods.assert_not_called()


def test_observe_implicit_mode_refreshes_complete_inventory():
    cm = KubernetesCapacityManager("default")
    cm._observe_all_deployments = True
    first = _install_connector(
        cm, "default/first", _dgd_spec(1, 1), parent_dgd_name="first"
    )
    second = _install_connector(
        cm, "default/second", _dgd_spec(2, 2), parent_dgd_name="second"
    )
    cm._inventory_api = first.kube_api
    first_scaled = _dgd_spec(3, 1)
    initial_inventory = [
        _listed_dgd(_dgd_spec(1, 1), "first"),
        _listed_dgd(_dgd_spec(2, 2), "second"),
    ]
    scaled_inventory = [
        _listed_dgd(first_scaled, "first"),
        _listed_dgd(_dgd_spec(2, 2), "second"),
    ]
    first.kube_api.list_graph_deployments.side_effect = [
        initial_inventory,
        scaled_inventory,
    ]
    first.kube_api.list_graph_deployment_metadata.side_effect = [
        deepcopy(initial_inventory),
        deepcopy(scaled_inventory),
    ]

    snapshot = cm.observe(require_complete=True)
    refreshed_snapshot = cm.observe(require_complete=True)

    assert snapshot["default/first"]["prefill"].current_replicas == 1
    assert snapshot["default/second"]["prefill"].current_replicas == 2
    assert refreshed_snapshot["default/first"]["prefill"].current_replicas == 3
    assert first.kube_api.list_graph_deployments.call_count == 2
    assert first.kube_api.list_graph_deployment_metadata.call_count == 2
    second.kube_api.list_graph_deployments.assert_not_called()
    first.kube_api.get_graph_deployment.assert_not_called()
    second.kube_api.get_graph_deployment.assert_not_called()


def test_observe_implicit_mode_preserves_cross_namespace_identity():
    cm = KubernetesCapacityManager("default")
    cm._observe_all_deployments = True
    local = _install_connector(
        cm, "default/model", _dgd_spec(1, 1), parent_dgd_name="model"
    )
    remote = _install_connector(
        cm, "tenant/model", _dgd_spec(3, 2), parent_dgd_name="model"
    )
    cm._inventory_api = local.kube_api

    snapshot = cm.observe(require_complete=True)

    assert snapshot["default/model"]["prefill"].current_replicas == 1
    assert snapshot["tenant/model"]["prefill"].current_replicas == 3
    local.kube_api.list_graph_deployments.assert_called_once_with()
    local.kube_api.list_graph_deployment_metadata.assert_called_once_with()
    local.kube_api.get_graph_deployment.assert_not_called()
    assert remote.kube_api.get_graph_deployment.call_count == 2


def test_observe_implicit_mode_discards_missing_never_observed_local():
    cm = KubernetesCapacityManager("default")
    cm._observe_all_deployments = True
    connector = _install_connector(
        cm,
        "default/model",
        _dgd_spec(1, 1),
        parent_dgd_name="model",
        seed_identity=False,
    )
    connector.kube_api.list_graph_deployments.side_effect = lambda: []
    connector.kube_api.list_graph_deployment_metadata.side_effect = lambda: []
    cm._inventory_api = connector.kube_api

    assert cm.observe(require_complete=True) == {}

    assert not cm.participant_exists("default/model")
    assert "default/model" not in cm.connectors


def test_observe_implicit_mode_rejects_residual_pods_without_dgds():
    cm = KubernetesCapacityManager("default")
    cm._observe_all_deployments = True
    connector = _install_connector(
        cm,
        "default/model",
        _dgd_spec(1, 1),
        parent_dgd_name="model",
        seed_identity=False,
    )
    connector.kube_api.list_graph_deployments.side_effect = None
    connector.kube_api.list_graph_deployments.return_value = []
    connector.kube_api.list_graph_deployment_metadata.side_effect = None
    connector.kube_api.list_graph_deployment_metadata.return_value = []
    connector.kube_api.list_graph_pods.return_value = [
        {
            "metadata": {
                "labels": {
                    "nvidia.com/dynamo-graph-deployment-name": "model",
                }
            }
        }
    ]
    cm._inventory_api = connector.kube_api

    with pytest.raises(RuntimeError, match="Pods reference DGDs absent"):
        cm.observe(require_complete=True)

    assert cm.connectors["default/model"] is connector


def test_observe_implicit_mode_keeps_drained_missing_observed_local_reserved():
    cm = KubernetesCapacityManager("default")
    cm._observe_all_deployments = True
    connector = _install_connector(
        cm, "default/model", _dgd_spec(1, 1), parent_dgd_name="model"
    )
    connector.kube_api.list_graph_deployments.side_effect = None
    connector.kube_api.list_graph_deployments.return_value = []
    connector.kube_api.list_graph_deployment_metadata.side_effect = None
    connector.kube_api.list_graph_deployment_metadata.return_value = []
    cm._inventory_api = connector.kube_api

    with pytest.raises(RuntimeError, match="capacity drain cannot be proven"):
        cm.observe(require_complete=True)

    assert cm.connectors["default/model"] is connector
    assert cm._observed_uids["default/model"] == "dgd-uid"
    assert not cm.participant_exists("default/model")


def test_observe_implicit_mode_rejects_malformed_dgd_metadata():
    cm = KubernetesCapacityManager("default")
    cm._observe_all_deployments = True
    connector = _install_connector(
        cm, "default/model", _dgd_spec(1, 1), parent_dgd_name="model"
    )
    connector.kube_api.list_graph_deployments.side_effect = lambda: [{"metadata": {}}]
    cm._inventory_api = connector.kube_api

    with pytest.raises(RuntimeError, match="no usable name"):
        cm.observe(require_complete=True)

    assert cm.connectors["default/model"] is connector


def test_observe_implicit_mode_marks_missing_never_observed_remote_deployments():
    cm = KubernetesCapacityManager("default")
    cm._observe_all_deployments = True
    local = _install_connector(
        cm, "default/local", _dgd_spec(1, 1), parent_dgd_name="local"
    )
    first = _install_connector(
        cm,
        "tenant/first",
        _dgd_spec(1, 1),
        parent_dgd_name="first",
        seed_identity=False,
    )
    second = _install_connector(
        cm,
        "tenant/second",
        _dgd_spec(1, 1),
        parent_dgd_name="second",
        seed_identity=False,
    )
    cm._inventory_api = local.kube_api
    first.kube_api.get_graph_deployment.side_effect = (
        DynamoGraphDeploymentNotFoundError("first", "tenant")
    )
    second.kube_api.get_graph_deployment.side_effect = (
        DynamoGraphDeploymentNotFoundError("second", "tenant")
    )

    assert cm.observe(require_complete=True) == {
        "tenant/first": {},
        "tenant/second": {},
    }

    assert first.kube_api.get_graph_deployment.call_count == 2
    assert second.kube_api.get_graph_deployment.call_count == 2
    first.kube_api.list_nonterminal_pods_for_graph.assert_called_once_with("first")
    second.kube_api.list_nonterminal_pods_for_graph.assert_called_once_with("second")
    assert not cm.participant_exists("tenant/first")
    assert not cm.participant_exists("tenant/second")


def test_observe_implicit_mode_discovers_previously_missing_deployment():
    cm = KubernetesCapacityManager("default")
    cm._observe_all_deployments = True
    deployment = _dgd_spec(1, 1)
    old = _install_connector(
        cm,
        "default/model",
        deployment,
        parent_dgd_name="model",
        seed_identity=False,
    )
    discovered = deepcopy(deployment)
    discovered.setdefault("metadata", {})["name"] = "model"
    old.kube_api.list_graph_deployments.side_effect = [
        [],
        [discovered],
    ]
    old.kube_api.list_graph_deployment_metadata.side_effect = [
        [deepcopy(discovered)],
    ]
    cm._inventory_api = old.kube_api

    assert cm.observe() == {}
    assert not cm.participant_exists("default/model")

    with patch(
        "dynamo.global_planner.kubernetes_capacity_manager.KubernetesConnector"
    ) as mock_connector_cls:
        new = MagicMock()
        new.parent_dgd_name = "model"
        new.kube_api = old.kube_api
        mock_connector_cls.return_value = new

        snapshot = cm.observe(require_complete=True)

    assert snapshot["default/model"]["prefill"].current_replicas == 1
    assert cm.connectors["default/model"] is new


def test_current_replicas():
    cm = KubernetesCapacityManager("default")
    _install_connector(
        cm, "default/my-dgd", _dgd_spec(prefill_replicas=2, decode_replicas=5)
    )
    assert cm.current_replicas("default/my-dgd") == {"prefill": 2, "decode": 5}


def test_captured_current_replicas_uses_authoritative_adapter_target():
    cm = KubernetesCapacityManager("default")
    deployment = _dgd_spec(prefill_replicas=1, decode_replicas=2)
    deployment["spec"]["components"][0]["scalingAdapter"] = {}
    connector = _install_connector(cm, "default/my-dgd", deployment)
    authoritative = deepcopy(deployment)
    authoritative["spec"]["components"][0]["replicas"] = 3
    connector.kube_api.fetch_authoritative_replica_targets.return_value = authoritative

    readback = cm.capture_current_replicas("default/my-dgd")

    assert readback() == {"prefill": 3, "decode": 2}
    connector.kube_api.fetch_authoritative_replica_targets.assert_called_once_with(
        deployment
    )


def test_capture_current_replicas_freezes_connector_and_role_hints():
    cm = KubernetesCapacityManager("default")
    old = _install_connector(
        cm,
        "default/my-dgd",
        _worker_dgd_spec(2, name="worker"),
    )
    cm._component_roles["default/my-dgd"] = {"worker": "prefill"}
    readback = cm.capture_current_replicas("default/my-dgd")

    cm.connectors["default/my-dgd"] = MagicMock()
    cm._component_roles["default/my-dgd"]["worker"] = "decode"

    assert readback() == {"prefill": 2}
    old.kube_api.get_graph_deployment.assert_called_once_with("my-dgd")


def test_capture_current_replicas_rejects_recreated_dgd():
    cm = KubernetesCapacityManager("default")
    deployment = _dgd_spec(prefill_replicas=2, decode_replicas=5)
    deployment.setdefault("metadata", {})["uid"] = "original"
    connector = _install_connector(cm, "default/my-dgd", deployment)
    cm._replica_uids["default/my-dgd"] = "original"
    readback = cm.capture_current_replicas("default/my-dgd")
    recreated = deepcopy(deployment)
    recreated["metadata"]["uid"] = "replacement"
    connector.kube_api.get_graph_deployment.return_value = recreated

    with pytest.raises(RuntimeError, match="identity changed"):
        readback()
