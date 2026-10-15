# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for KubernetesCapacityManager — the K8s observe/scale backend.

Pool state is read from the v1beta1 DGD component model (``spec.components[]``).
"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes import client

from dynamo.global_planner.capacity_manager import PoolSpec
from dynamo.global_planner.kubernetes_capacity_manager import KubernetesCapacityManager
from dynamo.planner import SubComponentType, TargetReplica
from dynamo.planner.connectors.base import ReplicaUpdateResult
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
        status = {}
        if gpu is not None:
            total_gpu = int(gpu) * component.get("multinode", {}).get("nodeCount", 1)
            status = {"gpusPerEngine": total_gpu, "gpusPerReplica": total_gpu}
        statuses[component["name"]] = status
    return {
        "metadata": {"generation": 1},
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


def _pod(phase):
    return SimpleNamespace(status=SimpleNamespace(phase=phase))


def _install_connector(cm, key, spec, parent_dgd_name="my-dgd"):
    connector = MagicMock()
    connector.parent_dgd_name = parent_dgd_name
    connector.set_component_replicas = AsyncMock(
        return_value=ReplicaUpdateResult(accepted=True, applied=True)
    )
    connector.wait_for_deployment_ready = AsyncMock()
    connector.kube_api = MagicMock()
    connector.kube_api.current_namespace = key.split("/", 1)[0]
    connector.kube_api.get_graph_deployment = MagicMock(return_value=spec)
    connector.kube_api.list_pods_for_graph = MagicMock(return_value=[])
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
    cm.connectors[key] = connector
    if cm._inventory_api is None:
        cm._inventory_api = connector.kube_api
    return connector


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
        connector.kube_api.list_pods_for_graph.return_value = []
        mock_connector_cls.return_value = connector

        assert cm.discover({"default-model-a"}) == []

    assert cm.connectors == {"default/model-a": connector}
    assert cm._missing_participants == {"default/model-a"}
    assert not cm.participant_exists("default/model-a")


def test_discover_explicit_mode_retains_missing_deployment_with_live_pods():
    cm = KubernetesCapacityManager("default")
    with patch(
        "dynamo.global_planner.kubernetes_capacity_manager.KubernetesConnector"
    ) as mock_connector_cls:
        connector = MagicMock()
        connector.parent_dgd_name = "model-a"
        connector.kube_api.get_graph_deployment.side_effect = (
            DynamoGraphDeploymentNotFoundError("model-a", "default")
        )
        connector.kube_api.list_pods_for_graph.return_value = [_pod("Running")]
        mock_connector_cls.return_value = connector

        assert cm.discover({"default-model-a"}) == ["model-a"]

    assert cm.connectors == {"default/model-a": connector}


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
            {"metadata": {"name": "model-a"}},
            {"metadata": {"name": "model-b"}},
        ]
        mock_connector_cls.return_value = MagicMock()

        discovered = cm.discover(None)

        assert set(discovered) == {"model-a", "model-b"}
        assert "default/model-a" in cm.connectors
        assert "default/model-b" in cm.connectors
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
async def test_scale_calls_connector():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))
    targets = [
        TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=3)
    ]
    result = await cm.scale("default/my-dgd", targets, blocking=True)

    assert result.accepted
    assert result.applied
    connector.set_component_replicas.assert_awaited_once_with(targets, blocking=True)


@pytest.mark.asyncio
async def test_scale_preserves_accepted_noop_result():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))
    connector.set_component_replicas.return_value = ReplicaUpdateResult(
        accepted=True, applied=False
    )

    result = await cm.scale(
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
async def test_scale_treats_legacy_none_result_as_applied():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))
    connector.set_component_replicas.return_value = None

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

    assert result.accepted
    assert result.applied


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_result", [True, False])
async def test_scale_treats_legacy_bool_result_conservatively(legacy_result):
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))
    connector.set_component_replicas.return_value = legacy_result

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

    assert result.accepted is legacy_result
    assert result.applied is legacy_result


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
async def test_run_with_conflict_retry_reuses_replica_identity_state():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))
    connector.set_component_replicas.side_effect = [
        client.ApiException(status=409),
        ReplicaUpdateResult(accepted=True, applied=True),
    ]
    targets = [
        TargetReplica(
            sub_component_type=SubComponentType.PREFILL,
            desired_replicas=2,
        )
    ]

    async def operation():
        return await cm.scale("default/my-dgd", targets, blocking=False)

    result = await cm.run_with_conflict_retry(operation)

    assert result.applied
    first_state = connector.set_component_replicas.await_args_list[0].kwargs[
        "retry_state"
    ]
    second_state = connector.set_component_replicas.await_args_list[1].kwargs[
        "retry_state"
    ]
    assert first_state is second_state


@pytest.mark.asyncio
async def test_wait_for_ready_calls_connector():
    cm = KubernetesCapacityManager("default")
    connector = _install_connector(cm, "default/my-dgd", _dgd_spec(1, 1))

    await cm.wait_for_ready("default/my-dgd")

    connector.wait_for_deployment_ready.assert_awaited_once_with()


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
            "components": {"decode-svc": {"gpusPerEngine": 4, "gpusPerReplica": 5}},
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
            "components": {"frontend": {"gpusPerEngine": 0, "gpusPerReplica": 0}},
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
        "status": {"observedGeneration": 2, "components": {"worker": {}}},
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


def test_observe_managed_deployment_not_found_evicts_without_failing():
    cm = KubernetesCapacityManager("default")
    missing = _install_connector(
        cm, "default/missing", _dgd_spec(1, 1), parent_dgd_name="missing"
    )
    missing.kube_api.get_graph_deployment.side_effect = (
        DynamoGraphDeploymentNotFoundError("missing", "default")
    )

    assert cm.observe(require_complete=True) == {}
    assert not cm.participant_exists("default/missing")

    missing = _install_connector(
        cm, "default/missing", _dgd_spec(1, 1), parent_dgd_name="missing"
    )
    missing.kube_api.get_graph_deployment.side_effect = (
        DynamoGraphDeploymentNotFoundError("missing", "default")
    )
    assert cm.observe() == {}
    assert not cm.participant_exists("default/missing")


def test_observe_missing_deployment_waits_for_labeled_pods_to_drain():
    cm = KubernetesCapacityManager("default")
    missing = _install_connector(
        cm, "default/missing", _dgd_spec(1, 1), parent_dgd_name="missing"
    )
    missing.kube_api.get_graph_deployment.side_effect = (
        DynamoGraphDeploymentNotFoundError("missing", "default")
    )
    missing.kube_api.list_pods_for_graph.side_effect = [[_pod("Running")], []]

    with pytest.raises(RuntimeError, match="nonterminal labeled Pods remain"):
        cm.observe(require_complete=True)
    assert cm.participant_exists("default/missing")

    assert cm.observe(require_complete=True) == {}
    assert not cm.participant_exists("default/missing")


def test_observe_missing_deployment_ignores_terminal_labeled_pods():
    cm = KubernetesCapacityManager("default")
    missing = _install_connector(
        cm, "default/missing", _dgd_spec(1, 1), parent_dgd_name="missing"
    )
    missing.kube_api.get_graph_deployment.side_effect = (
        DynamoGraphDeploymentNotFoundError("missing", "default")
    )
    missing.kube_api.list_pods_for_graph.return_value = [
        _pod("Succeeded"),
        _pod("Failed"),
    ]

    assert cm.observe(require_complete=True) == {}
    assert not cm.participant_exists("default/missing")


def test_observe_missing_deployment_fails_closed_when_drain_read_fails():
    cm = KubernetesCapacityManager("default")
    missing = _install_connector(
        cm, "default/missing", _dgd_spec(1, 1), parent_dgd_name="missing"
    )
    missing.kube_api.get_graph_deployment.side_effect = (
        DynamoGraphDeploymentNotFoundError("missing", "default")
    )
    missing.kube_api.list_pods_for_graph.side_effect = RuntimeError("timeout")

    with pytest.raises(RuntimeError, match="Pod drain could not be verified"):
        cm.observe(require_complete=True)
    assert cm.participant_exists("default/missing")


def test_observe_explicit_mode_rediscovers_recreated_deployment():
    cm = KubernetesCapacityManager("default")
    cm._configured_deployment_names = {"model"}
    deployment = _dgd_spec(1, 1)
    old = _install_connector(cm, "default/model", deployment, parent_dgd_name="model")
    old.kube_api.get_graph_deployment.side_effect = [
        DynamoGraphDeploymentNotFoundError("model", "default"),
        deployment,
    ]

    assert cm.observe(require_complete=True) == {}
    assert "default/model" in cm.connectors
    assert not cm.participant_exists("default/model")
    assert cm._missing_participants == {"default/model"}

    snapshot = cm.observe(require_complete=True)

    assert snapshot["default/model"]["prefill"].current_replicas == 1
    assert cm.connectors["default/model"] is old
    assert cm._missing_participants == set()
    assert cm.participant_exists("default/model")
    old.kube_api.list_pods_for_graph.assert_called_once_with("model")


def test_observe_explicit_mode_does_not_repeat_drain_for_absent_configured_names():
    cm = KubernetesCapacityManager("default")
    cm._configured_deployment_names = {"first", "second"}
    first = _install_connector(
        cm, "default/first", _dgd_spec(1, 1), parent_dgd_name="first"
    )
    second = _install_connector(
        cm, "default/second", _dgd_spec(1, 1), parent_dgd_name="second"
    )
    first.kube_api.get_graph_deployment.side_effect = (
        DynamoGraphDeploymentNotFoundError("first", "default")
    )
    second.kube_api.get_graph_deployment.side_effect = (
        DynamoGraphDeploymentNotFoundError("second", "default")
    )

    assert cm.observe(require_complete=True) == {}
    assert cm.observe(require_complete=True) == {}

    assert first.kube_api.get_graph_deployment.call_count == 2
    assert second.kube_api.get_graph_deployment.call_count == 2
    first.kube_api.list_pods_for_graph.assert_called_once_with("first")
    second.kube_api.list_pods_for_graph.assert_called_once_with("second")
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
    connector.kube_api.get_graph_deployment.assert_called_once_with("model")
    connector.kube_api.list_graph_deployments.assert_not_called()


def test_observe_implicit_mode_lists_once_per_namespace():
    cm = KubernetesCapacityManager("default")
    cm._observe_all_deployments = True
    first = _install_connector(
        cm, "default/first", _dgd_spec(1, 1), parent_dgd_name="first"
    )
    second = _install_connector(
        cm, "default/second", _dgd_spec(2, 2), parent_dgd_name="second"
    )
    cm._inventory_api = first.kube_api

    snapshot = cm.observe(require_complete=True)

    assert snapshot["default/first"]["prefill"].current_replicas == 1
    assert snapshot["default/second"]["prefill"].current_replicas == 2
    first.kube_api.list_graph_deployments.assert_called_once_with()
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
    local.kube_api.get_graph_deployment.assert_not_called()
    remote.kube_api.get_graph_deployment.assert_called_once_with("model")


def test_observe_implicit_mode_rediscovers_recreated_deployment():
    cm = KubernetesCapacityManager("default")
    cm._observe_all_deployments = True
    deployment = _dgd_spec(1, 1)
    old = _install_connector(cm, "default/model", deployment, parent_dgd_name="model")
    recreated = deepcopy(deployment)
    recreated.setdefault("metadata", {})["name"] = "model"
    old.kube_api.list_graph_deployments.side_effect = [[], [recreated]]
    cm._inventory_api = old.kube_api

    assert cm.observe() == {}
    assert not cm.participant_exists("default/model")

    with patch(
        "dynamo.global_planner.kubernetes_capacity_manager.KubernetesConnector"
    ) as mock_connector_cls:
        new = MagicMock()
        new.parent_dgd_name = "model"
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
