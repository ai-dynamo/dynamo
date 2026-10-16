# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shlex
from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, call, patch

import pytest
from kubernetes import client

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.base import PlannerConnector
from dynamo.planner.connectors.clients.kubernetes_api import KubernetesAPI
from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.errors import (
    DeploymentModelNameMismatchError,
    DeploymentValidationError,
    DuplicateSubComponentError,
    DynamoGraphDeploymentNotFoundError,
    DynamoGraphDeploymentNotReadyError,
    EmptyTargetReplicasError,
    GPUShapeUnavailableError,
    ModelNameNotFoundError,
    PlannerError,
    PowerAnnotationMissingError,
    SubComponentNotFoundError,
)
from dynamo.planner.monitoring.dgd_services import (
    Service,
    get_component_from_type_or_name,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.fixture
def mock_kube_api():
    mock_api = Mock()
    mock_api.get_graph_deployment = Mock()
    mock_api.fetch_authoritative_replica_targets = Mock(
        side_effect=lambda deployment: deployment
    )
    mock_api.update_graph_replicas = AsyncMock()
    mock_api.update_dgd_replicas_directly = Mock()
    mock_api.get_service_scaling_adapter = Mock()
    mock_api.get_service_replica_target = Mock()
    mock_api.scaling_adapter_name = Mock(side_effect=KubernetesAPI.scaling_adapter_name)
    mock_api.scaling_adapter_identity_rejection = Mock(
        side_effect=lambda deployment, component_name, adapter, **kwargs: (
            KubernetesAPI.scaling_adapter_identity_rejection(
                mock_api,
                deployment,
                component_name,
                adapter,
                **kwargs,
            )
        )
    )
    mock_api.get_scaling_adapter_desired_replicas = Mock(
        side_effect=KubernetesAPI.get_scaling_adapter_desired_replicas
    )
    mock_api.update_scaling_adapter_replicas = Mock()
    mock_api.patch_scaling_adapter_writer_fence = Mock()
    mock_api.wait_for_graph_deployment_ready = AsyncMock()
    mock_api.is_deployment_ready = Mock()
    mock_api.pending_startup_replicas = Mock(return_value={})
    mock_api.is_spec_generation_observed = Mock(return_value=False)
    mock_api.non_planner_components_stable = Mock(return_value=(True, []))
    mock_api.pcsg_pods_within_desired_replicas = Mock(return_value=True)
    # Default: no terminating pods; tests that want to simulate terminating pods
    # override this per-test.
    mock_api.has_terminating_pods = Mock(return_value=False)
    mock_api.list_pods_for_graph = Mock(return_value=[])
    mock_api.exclude_checkpoint_capture_pods = Mock(
        side_effect=KubernetesAPI.exclude_checkpoint_capture_pods
    )
    mock_api.partition_pods_by_component = Mock(return_value={})
    # Default: no blocking rollout; tests that want InProgress/Pending override.
    mock_api.is_rolling_update_blocking_settlement = Mock(return_value=(False, ""))
    return mock_api


@pytest.fixture
def mock_kube_api_class(mock_kube_api):
    mock_class = Mock()
    mock_class.return_value = mock_kube_api
    return mock_class


@pytest.fixture
def kubernetes_connector(mock_kube_api_class, monkeypatch):
    # Patch the KubernetesAPI class before instantiating the connector
    monkeypatch.setattr(
        "dynamo.planner.connectors.kubernetes.KubernetesAPI", mock_kube_api_class
    )
    with patch.dict(os.environ, {"DYN_PARENT_DGD_K8S_NAME": "test-graph"}):
        connector = KubernetesConnector("test-dynamo-namespace")
        return connector


def _main_container(args=None, gpu=None):
    container = {"name": "main"}
    if args is not None:
        container["args"] = args
    if gpu is not None:
        container["resources"] = {"limits": {"nvidia.com/gpu": str(gpu)}}
    return container


def _component(name, component_type=None, replicas=None, args=None, gpu=None):
    component = {"name": name}
    if component_type is not None:
        component["type"] = component_type
    if replicas is not None:
        component["replicas"] = replicas
    if args is not None or gpu is not None:
        component["podTemplate"] = {
            "spec": {"containers": [_main_container(args=args, gpu=gpu)]}
        }
    return component


def _adapter_component(name, component_type=None, replicas=None):
    component = _component(name, component_type, replicas=replicas)
    # v1beta1 uses key presence as the opt-in marker; the canonical value is
    # intentionally the otherwise-falsy empty object.
    component["scalingAdapter"] = {}
    return component


def _deployment(*components):
    component_statuses = {}
    for component in components:
        status = {}
        container = next(
            (
                container
                for container in component.get("podTemplate", {})
                .get("spec", {})
                .get("containers", [])
                if container.get("name") == "main"
            ),
            {},
        )
        args = []
        for arg in container.get("args", []):
            args.extend(shlex.split(arg))
        for flag in ("--served-model-name", "--model-name", "--model"):
            if flag in args and len(args) > args.index(flag) + 1:
                status["servedModelName"] = args[args.index(flag) + 1]
                break
        if "--endpoint" in args and len(args) > args.index("--endpoint") + 1:
            endpoint = args[args.index("--endpoint") + 1].removeprefix("dyn://")
            parts = endpoint.split(".")
            if len(parts) == 3:
                status["runtimeComponentName"] = parts[1]

        resources = container.get("resources", {})
        gpu = resources.get("limits", {}).get(
            "nvidia.com/gpu",
            resources.get("requests", {}).get("nvidia.com/gpu"),
        )
        if gpu is not None:
            per_engine = int(gpu) * int(
                component.get("multinode", {}).get("nodeCount", 1)
            )
            status["gpusPerEngine"] = per_engine
            status["gpusPerReplica"] = per_engine
        component_statuses[component["name"]] = status

    return {
        "metadata": {
            "name": "test-graph",
            "uid": "test-graph-uid",
            "generation": 1,
        },
        "spec": {"components": list(components)},
        "status": {"observedGeneration": 1, "components": component_statuses},
    }


def _unready_recovery_deployment(*components):
    component_statuses = {
        component["name"]: {
            "replicas": component.get("replicas", 0),
            "updatedReplicas": component.get("replicas", 0),
            "readyReplicas": component.get("replicas", 0),
            "availableReplicas": component.get("replicas", 0),
        }
        for component in components
    }
    return {
        "metadata": {
            "name": "test-graph",
            "uid": "test-graph-uid",
            "generation": 7,
        },
        "spec": {"components": list(components)},
        "status": {
            "observedGeneration": 7,
            "conditions": [{"type": "Ready", "status": "False"}],
            "components": component_statuses,
        },
    }


def _scaling_adapter(component_name, replicas=0):
    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeploymentScalingAdapter",
        "metadata": {
            "name": f"test-graph-{component_name.lower()}",
            "uid": f"adapter-{component_name.lower()}-uid",
            "resourceVersion": "101",
            "labels": {
                "nvidia.com/dynamo-graph-deployment-name": "test-graph",
                "nvidia.com/dynamo-component": component_name,
            },
            "ownerReferences": [
                {
                    "apiVersion": "nvidia.com/v1beta1",
                    "kind": "DynamoGraphDeployment",
                    "name": "test-graph",
                    "uid": "test-graph-uid",
                    "controller": True,
                }
            ],
        },
        "spec": {
            "replicas": replicas,
            "dgdRef": {
                "name": "test-graph",
                "componentName": component_name,
            },
        },
    }


def _model_card_cr(name, worker_type="decode"):
    return {
        "metadata": {"name": name},
        "spec": {
            "data": {
                "model_cards": {
                    "model": {
                        "type": "Model",
                        "card_json": {"worker_type": worker_type},
                    }
                }
            }
        },
    }


def _deployment_with_worker_status(
    component_kind, runtime_namespace=None, annotations=None
):
    worker_status = {"componentKind": component_kind}
    if runtime_namespace is not None:
        worker_status["runtimeNamespace"] = runtime_namespace
    return {
        "metadata": {"annotations": annotations or {}},
        "spec": {"components": [_component("worker", "worker")]},
        "status": {"components": {"worker": worker_status}},
    }


def test_kubernetes_connector_no_env_var():
    with patch("dynamo.planner.connectors.kubernetes.KubernetesAPI"):
        with pytest.raises(DeploymentValidationError) as exc_info:
            KubernetesConnector("test-dynamo-namespace")

    exception = exc_info.value
    assert set(exception.errors) == {
        "DYN_PARENT_DGD_K8S_NAME environment variable is not set"
    }


def test_get_worker_runtime_namespace_uses_status_runtime_namespace(
    kubernetes_connector, mock_kube_api
):
    mock_kube_api.get_graph_deployment.return_value = _deployment_with_worker_status(
        "PodClique",
        runtime_namespace="runtime-from-status",
        annotations={"nvidia.com/current-worker-hash": "abc123"},
    )

    namespace = kubernetes_connector.get_worker_runtime_namespace("base-ns")

    assert namespace == "runtime-from-status"
    mock_kube_api.get_graph_deployment.assert_called_with("test-graph")


def test_get_worker_runtime_namespace_falls_back_to_deployment_hash(
    kubernetes_connector, mock_kube_api
):
    mock_kube_api.get_graph_deployment.return_value = _deployment_with_worker_status(
        "Deployment",
        annotations={"nvidia.com/current-worker-hash": "abc123"},
    )

    namespace = kubernetes_connector.get_worker_runtime_namespace("base-ns")

    assert namespace == "base-ns-abc123"


def test_get_worker_runtime_namespace_falls_back_to_worker_name_when_type_missing(
    kubernetes_connector, mock_kube_api
):
    mock_kube_api.get_graph_deployment.return_value = {
        "metadata": {"annotations": {"nvidia.com/current-worker-hash": "abc123"}},
        "spec": {"components": [_component("worker")]},
        "status": {"components": {"worker": {"componentKind": "Deployment"}}},
    }

    namespace = kubernetes_connector.get_worker_runtime_namespace("base-ns")

    assert namespace == "base-ns-abc123"


def test_get_worker_runtime_namespace_explicit_type_overrides_worker_name_fallback(
    kubernetes_connector, mock_kube_api
):
    mock_kube_api.get_graph_deployment.return_value = {
        "metadata": {"annotations": {"nvidia.com/current-worker-hash": "abc123"}},
        "spec": {
            "components": [
                _component("worker", "frontend"),
                _component("serving", "worker"),
            ]
        },
        "status": {
            "components": {
                "worker": {
                    "componentKind": "Deployment",
                    "runtimeNamespace": "base-ns",
                },
                "serving": {"componentKind": "Deployment"},
            }
        },
    }

    namespace = kubernetes_connector.get_worker_runtime_namespace("base-ns")

    assert namespace == "base-ns-abc123"


def test_get_worker_runtime_namespace_falls_back_to_v2_hash(
    kubernetes_connector, mock_kube_api
):
    mock_kube_api.get_graph_deployment.return_value = _deployment_with_worker_status(
        "Deployment",
        annotations={"nvidia.com/current-worker-hash-v2": "v2abc"},
    )

    namespace = kubernetes_connector.get_worker_runtime_namespace("base-ns")

    assert namespace == "base-ns-v2abc"


def test_get_worker_runtime_namespace_uses_legacy_v1_before_v2(
    kubernetes_connector, mock_kube_api
):
    mock_kube_api.get_graph_deployment.return_value = _deployment_with_worker_status(
        "Deployment",
        annotations={
            "nvidia.com/current-worker-hash": "legacy",
            "nvidia.com/current-worker-hash-v2": "v2abc",
        },
    )

    namespace = kubernetes_connector.get_worker_runtime_namespace("base-ns")

    assert namespace == "base-ns-legacy"


def test_get_worker_runtime_namespace_falls_back_to_base_for_grove(
    kubernetes_connector, mock_kube_api
):
    mock_kube_api.get_graph_deployment.return_value = _deployment_with_worker_status(
        "PodCliqueScalingGroup",
        annotations={"nvidia.com/current-worker-hash": "abc123"},
    )

    namespace = kubernetes_connector.get_worker_runtime_namespace("base-ns")

    assert namespace == "base-ns"


def test_get_worker_runtime_namespace_falls_back_to_leader_worker_set_hash(
    kubernetes_connector, mock_kube_api
):
    mock_kube_api.get_graph_deployment.return_value = _deployment_with_worker_status(
        "LeaderWorkerSet",
        annotations={"nvidia.com/current-worker-hash": "abc123"},
    )

    namespace = kubernetes_connector.get_worker_runtime_namespace("base-ns")

    assert namespace == "base-ns-abc123"


def test_get_worker_runtime_namespace_without_hash(kubernetes_connector, mock_kube_api):
    mock_kube_api.get_graph_deployment.return_value = _deployment_with_worker_status(
        "Deployment"
    )

    namespace = kubernetes_connector.get_worker_runtime_namespace("base-ns")

    assert namespace == "base-ns"


def test_get_worker_runtime_namespace_legacy_hash(kubernetes_connector, mock_kube_api):
    mock_kube_api.get_graph_deployment.return_value = _deployment_with_worker_status(
        "Deployment",
        annotations={"nvidia.com/current-worker-hash": "legacy"},
    )

    namespace = kubernetes_connector.get_worker_runtime_namespace("base-ns")

    assert namespace == "base-ns-legacy"


def test_get_worker_runtime_namespace_missing_status_with_hash_is_indeterminate(
    kubernetes_connector, mock_kube_api
):
    mock_kube_api.get_graph_deployment.return_value = {
        "metadata": {
            "annotations": {"nvidia.com/current-worker-hash": "abc123"},
        },
        "spec": {"components": [_component("worker", "worker")]},
    }

    with pytest.raises(PlannerError, match="runtime namespace is indeterminate"):
        kubernetes_connector.get_worker_runtime_namespace("base-ns")


def test_get_worker_runtime_namespace_missing_status_without_hash_uses_base(
    kubernetes_connector, mock_kube_api
):
    mock_kube_api.get_graph_deployment.return_value = {
        "metadata": {"annotations": {}},
        "spec": {"components": [_component("worker", "worker")]},
    }

    namespace = kubernetes_connector.get_worker_runtime_namespace("base-ns")

    assert namespace == "base-ns"


def test_get_service_name_from_sub_component_type(kubernetes_connector):
    deployment = _deployment(
        _component("test-component-prefill", "prefill", replicas=2),
        _component("test-component-decode", "decode", replicas=3),
    )

    service = get_component_from_type_or_name(deployment, SubComponentType.PREFILL)
    assert service.name == "test-component-prefill"
    assert service.number_replicas() == 2

    # should still work if the component_name is provided
    service = get_component_from_type_or_name(
        deployment, SubComponentType.PREFILL, "test-component-prefill"
    )
    assert service.name == "test-component-prefill"
    assert service.number_replicas() == 2

    # should respect component type first
    service = get_component_from_type_or_name(
        deployment, SubComponentType.DECODE, "test-component-prefill"
    )
    assert service.name == "test-component-decode"
    assert service.number_replicas() == 3


def test_get_service_name_from_v1beta_component_type(kubernetes_connector):
    deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "components": [
                {
                    "name": "prefill",
                    "replicas": 2,
                    "type": "prefill",
                },
                {
                    "name": "decode",
                    "replicas": 3,
                    "type": "decode",
                },
            ]
        },
    }

    service = get_component_from_type_or_name(deployment, SubComponentType.PREFILL)
    assert service.name == "prefill"
    assert service.number_replicas() == 2

    service = get_component_from_type_or_name(deployment, SubComponentType.DECODE)
    assert service.name == "decode"
    assert service.number_replicas() == 3


def test_get_service_name_from_v1beta_worker_type_by_name(kubernetes_connector):
    deployment = _deployment(_component("worker", "worker", replicas=2))

    service = get_component_from_type_or_name(
        deployment, SubComponentType.PREFILL, "worker"
    )

    assert service.name == "worker"
    assert service.number_replicas() == 2


def test_get_service_name_from_unique_v1beta_worker_type_for_decode(
    kubernetes_connector,
):
    deployment = _deployment(_component("arbitrary-name", "worker", replicas=2))

    service = get_component_from_type_or_name(deployment, SubComponentType.DECODE)

    assert service.name == "arbitrary-name"
    assert service.number_replicas() == 2


def test_get_service_name_from_multiple_v1beta_workers_by_name(
    kubernetes_connector,
):
    deployment = _deployment(
        _component("prefill-name", "worker"),
        _component("decode-name", "worker"),
    )

    with pytest.raises(SubComponentNotFoundError):
        get_component_from_type_or_name(deployment, SubComponentType.DECODE)

    service = get_component_from_type_or_name(
        deployment, SubComponentType.DECODE, "decode-name"
    )
    assert service.name == "decode-name"


@pytest.mark.asyncio
async def test_validate_deployment_agg_worker_by_type(
    kubernetes_connector, mock_kube_api
):
    mock_kube_api.get_graph_deployment.return_value = _deployment(
        _component("Frontend", "frontend", replicas=1),
        _component(
            "arbitrary-name",
            "worker",
            replicas=2,
            args=["--model", "Qwen/Qwen3-8B"],
        ),
    )

    await kubernetes_connector.validate_deployment(
        decode_component_name="stale-default-name",
        require_prefill=False,
        require_decode=True,
    )


def test_get_service_name_from_sub_component_type_not_found(kubernetes_connector):
    deployment = _deployment(_component("test-component-decode", "decode", replicas=3))
    with pytest.raises(SubComponentNotFoundError) as exc_info:
        get_component_from_type_or_name(deployment, SubComponentType.PREFILL)

    with pytest.raises(SubComponentNotFoundError) as exc_info:
        get_component_from_type_or_name(
            deployment, SubComponentType.PREFILL, "test-component-decode"
        )

    exception = exc_info.value
    assert exception.sub_component_type == SubComponentType.PREFILL.value


def test_get_service_name_from_sub_component_type_duplicate(kubernetes_connector):
    deployment = _deployment(
        _component("test-component-prefill", "prefill", replicas=2),
        _component("test-component-prefill-2", "prefill", replicas=3),
    )

    with pytest.raises(DuplicateSubComponentError) as exc_info:
        # even though "test-component-prefill" is provided, duplicate component
        # types should result in an error
        get_component_from_type_or_name(
            deployment, SubComponentType.PREFILL, "test-component-prefill"
        )

    exception = exc_info.value
    assert exception.sub_component_type == SubComponentType.PREFILL.value
    assert set(exception.service_names) == {
        "test-component-prefill",
        "test-component-prefill-2",
    }


def test_get_service_name_from_sub_component_type_or_name(kubernetes_connector):
    deployment = _deployment(
        _component("test-component-prefill", replicas=2),
        _component("test-component-decode", replicas=3),
    )

    service = get_component_from_type_or_name(
        deployment, SubComponentType.PREFILL, "test-component-prefill"
    )
    assert service.name == "test-component-prefill"
    assert service.number_replicas() == 2


@pytest.mark.asyncio
async def test_add_component_increases_replicas(kubernetes_connector, mock_kube_api):
    # Arrange
    sub_component_type = SubComponentType.PREFILL
    component_name = "test-component"
    mock_deployment = _deployment(
        _component(component_name, sub_component_type.value, replicas=1)
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.update_graph_replicas.return_value = None
    mock_kube_api.wait_for_graph_deployment_ready.return_value = None

    # Act
    await kubernetes_connector.add_component(sub_component_type)

    # Assert
    mock_kube_api.get_graph_deployment.assert_called_once()
    mock_kube_api.update_dgd_replicas_directly.assert_called_once_with(
        "test-graph", component_name, 2
    )
    mock_kube_api.wait_for_graph_deployment_ready.assert_called_once_with("test-graph")


@pytest.mark.asyncio
async def test_add_component_with_no_replicas_specified(
    kubernetes_connector, mock_kube_api
):
    # Arrange
    sub_component_type = SubComponentType.PREFILL
    component_name = "test-component"
    mock_deployment = _deployment(_component(component_name, sub_component_type.value))
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    # Act
    await kubernetes_connector.add_component(sub_component_type)

    # Assert
    mock_kube_api.update_dgd_replicas_directly.assert_called_once_with(
        "test-graph", component_name, 1
    )
    mock_kube_api.wait_for_graph_deployment_ready.assert_called_once_with("test-graph")


@pytest.mark.asyncio
async def test_add_component_uses_declared_adapter_desired_not_stale_dgd_seed(
    kubernetes_connector, mock_kube_api
):
    deployment = _deployment(_adapter_component("worker", "decode", replicas=1))
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.get_service_scaling_adapter.return_value = _scaling_adapter(
        "worker", replicas=0
    )

    await kubernetes_connector.add_component(SubComponentType.DECODE, blocking=False)

    mock_kube_api.update_scaling_adapter_replicas.assert_called_once_with(
        "test-graph", "worker", 1, resource_version="101"
    )
    mock_kube_api.update_dgd_replicas_directly.assert_not_called()


@pytest.mark.asyncio
async def test_scale_rejects_lost_batch_writer_fence(
    kubernetes_connector, mock_kube_api
):
    deployment = _deployment(_adapter_component("worker", "decode", replicas=0))
    adapter = _scaling_adapter("worker", replicas=0)
    adapter["metadata"]["annotations"] = {
        "dynamo.nvidia.com/planner-writer-fence": "replacement-writer"
    }
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = True
    mock_kube_api.get_service_scaling_adapter.return_value = adapter
    kubernetes_connector._batch_writer_id = "this-writer"

    with pytest.raises(ValueError, match="owned by another writer"):
        await kubernetes_connector.set_component_replicas(
            [
                TargetReplica(
                    sub_component_type=SubComponentType.DECODE,
                    component_name="worker",
                    desired_replicas=1,
                )
            ],
            blocking=False,
        )

    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_add_component_deployment_not_found(kubernetes_connector, mock_kube_api):
    # Arrange
    component_name = "test-component"
    mock_kube_api.get_graph_deployment.side_effect = DynamoGraphDeploymentNotFoundError(
        "test-graph", "default"
    )

    # Act & Assert
    with pytest.raises(DynamoGraphDeploymentNotFoundError):
        await kubernetes_connector.add_component(component_name)


@pytest.mark.asyncio
async def test_add_component_component_not_found(kubernetes_connector, mock_kube_api):
    # Arrange
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {"components": [_component("test-component", "decode")]},
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    # Act
    with pytest.raises(SubComponentNotFoundError) as exc_info:
        await kubernetes_connector.add_component(SubComponentType.PREFILL)

        mock_kube_api.update_graph_replicas.assert_not_called()
        mock_kube_api.wait_for_graph_deployment_ready.assert_not_called()

    exception = exc_info.value
    assert exception.sub_component_type == "prefill"


@pytest.mark.asyncio
async def test_remove_component_decreases_replicas(kubernetes_connector, mock_kube_api):
    # Arrange
    component_name = "test-component"
    sub_component_type = SubComponentType.PREFILL
    mock_deployment = _deployment(
        _component("test-component", sub_component_type.value, replicas=2)
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    # Act
    await kubernetes_connector.remove_component(sub_component_type)

    # Assert
    mock_kube_api.update_dgd_replicas_directly.assert_called_once_with(
        "test-graph", component_name, 1
    )
    mock_kube_api.wait_for_graph_deployment_ready.assert_called_once_with("test-graph")


@pytest.mark.asyncio
async def test_remove_component_with_zero_replicas(kubernetes_connector, mock_kube_api):
    # Arrange
    component_name = "test-component"
    sub_component_type = SubComponentType.PREFILL
    mock_deployment = _deployment(
        _component(component_name, sub_component_type.value, replicas=0)
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    # Act
    await kubernetes_connector.remove_component(sub_component_type)

    # Assert
    mock_kube_api.update_graph_replicas.assert_not_called()
    mock_kube_api.wait_for_graph_deployment_ready.assert_not_called()


@pytest.mark.asyncio
async def test_remove_component_component_not_found(
    kubernetes_connector, mock_kube_api
):
    # Arrange
    component_name = "test-component"
    sub_component_type = SubComponentType.PREFILL
    mock_deployment = _deployment(
        _component(component_name, sub_component_type.value, replicas=0)
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    # Act
    with pytest.raises(SubComponentNotFoundError) as exc_info:
        await kubernetes_connector.remove_component(SubComponentType.DECODE)

        # Assert
        mock_kube_api.update_graph_replicas.assert_not_called()
        mock_kube_api.wait_for_graph_deployment_ready.assert_not_called()

    exception = exc_info.value
    assert exception.sub_component_type == "decode"


@pytest.mark.asyncio
async def test_set_component_replicas(kubernetes_connector, mock_kube_api):
    # Arrange
    target_replicas = [
        TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=3),
        TargetReplica(
            sub_component_type=SubComponentType.DECODE,
            component_name="component2",
            desired_replicas=2,
        ),
    ]
    mock_deployment = _deployment(
        _component("component1", "prefill", replicas=1),
        _component("component2", replicas=1),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.is_deployment_ready.return_value = True
    mock_kube_api.wait_for_graph_deployment_ready.return_value = None

    # Act
    await kubernetes_connector.set_component_replicas(target_replicas)

    # Assert
    mock_kube_api.get_graph_deployment.assert_called_once()
    mock_kube_api.is_deployment_ready.assert_called_once_with(mock_deployment)
    # Should be called twice, once for each component
    expected_calls = [
        call("test-graph", "component1", 3),  # prefill component with 3 replicas
        call("test-graph", "component2", 2),  # decode component with 2 replicas
    ]
    mock_kube_api.update_dgd_replicas_directly.assert_has_calls(
        expected_calls, any_order=True
    )
    mock_kube_api.wait_for_graph_deployment_ready.assert_called_once_with("test-graph")


@pytest.mark.asyncio
async def test_set_component_replicas_component_not_found(
    kubernetes_connector, mock_kube_api
):
    # Arrange
    target_replicas = [
        TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=3),
        TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=2),
    ]
    mock_deployment = _deployment(
        _component("component1", "prefill", replicas=1),
        _component("component2", replicas=1),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.is_deployment_ready.return_value = True
    mock_kube_api.update_graph_replicas.return_value = None
    mock_kube_api.wait_for_graph_deployment_ready.return_value = None

    # Act
    with pytest.raises(SubComponentNotFoundError) as exc_info:
        await kubernetes_connector.set_component_replicas(target_replicas)

    exception = exc_info.value
    assert exception.sub_component_type == SubComponentType.DECODE.value


@pytest.mark.asyncio
async def test_set_component_replicas_undeclared_adapter_ignores_stray_dgdsa(
    kubernetes_connector, mock_kube_api
):
    deployment = _deployment(_component("worker", "decode", replicas=0))
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = True

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                desired_replicas=1,
            )
        ],
        blocking=False,
    )

    mock_kube_api.get_service_scaling_adapter.assert_not_called()
    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()
    mock_kube_api.update_graph_replicas.assert_not_called()
    mock_kube_api.update_dgd_replicas_directly.assert_called_once_with(
        "test-graph", "worker", 1
    )


@pytest.mark.asyncio
async def test_batch_writer_fence_retries_conflict_then_refreshes_authority(
    kubernetes_connector, mock_kube_api
):
    deployment = _unready_recovery_deployment(
        _adapter_component("worker", "worker", replicas=1),
    )
    initial_adapter = _scaling_adapter("worker", replicas=0)
    newer_adapter = deepcopy(initial_adapter)
    newer_adapter["metadata"]["resourceVersion"] = "102"
    fenced_adapter = deepcopy(newer_adapter)
    fenced_adapter["metadata"]["resourceVersion"] = "103"
    fenced_adapter["metadata"]["annotations"] = {
        "dynamo.nvidia.com/planner-writer-fence": "writer-new"
    }
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_spec_generation_observed.return_value = True
    mock_kube_api.get_service_scaling_adapter.side_effect = [
        initial_adapter,
        newer_adapter,
        fenced_adapter,
    ]
    mock_kube_api.patch_scaling_adapter_writer_fence.side_effect = [
        client.ApiException(status=409),
        None,
    ]
    mock_kube_api.get_service_replica_status.return_value = (0, True)

    await kubernetes_connector.acquire_batch_writer_fence("writer-new")

    assert mock_kube_api.patch_scaling_adapter_writer_fence.call_args_list == [
        call("test-graph", "worker", "writer-new", adapter=initial_adapter),
        call("test-graph", "worker", "writer-new", adapter=newer_adapter),
    ]
    mock_kube_api.fetch_authoritative_replica_targets.assert_called_once_with(
        deployment
    )


@pytest.mark.asyncio
async def test_set_component_replicas_component_already_at_desired_replicas(
    kubernetes_connector, mock_kube_api
):
    # Arrange
    target_replicas = [
        TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=3),
        TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=2),
    ]
    mock_deployment = _deployment(
        _component("component1", "prefill", replicas=1),
        _component("component2", "decode", replicas=2),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.is_deployment_ready.return_value = True
    mock_kube_api.update_graph_replicas.return_value = None
    mock_kube_api.wait_for_graph_deployment_ready.return_value = None

    # Act
    await kubernetes_connector.set_component_replicas(target_replicas)

    # Assert
    mock_kube_api.get_graph_deployment.assert_called_once()
    mock_kube_api.is_deployment_ready.assert_called_once_with(mock_deployment)

    # Should be called once, for the prefill component (decode component is already at desired replicas)
    mock_kube_api.update_dgd_replicas_directly.assert_called_once_with(
        "test-graph", "component1", 3
    )
    mock_kube_api.wait_for_graph_deployment_ready.assert_called_once_with("test-graph")


@pytest.mark.asyncio
async def test_set_component_replicas_deployment_not_found(
    kubernetes_connector, mock_kube_api
):
    # Arrange
    target_replicas = [
        TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=3)
    ]
    mock_kube_api.get_graph_deployment.side_effect = DynamoGraphDeploymentNotFoundError(
        "test-graph", "default"
    )

    # Act & Assert
    with pytest.raises(DynamoGraphDeploymentNotFoundError):
        await kubernetes_connector.set_component_replicas(target_replicas)


@pytest.mark.asyncio
async def test_set_component_replicas_empty_target_replicas(
    kubernetes_connector, mock_kube_api
):
    # Arrange
    target_replicas: list[TargetReplica] = []

    # Act & Assert
    with pytest.raises(EmptyTargetReplicasError):
        await kubernetes_connector.set_component_replicas(target_replicas)


@pytest.mark.asyncio
async def test_set_component_replicas_deployment_not_ready_skips_by_default(
    kubernetes_connector, mock_kube_api
):
    """Keep local Kubernetes planners on the legacy skip-tick path."""
    # Arrange
    target_replicas = [
        TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=3),
        TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=2),
    ]
    mock_deployment = _deployment(
        _component("component1", "prefill", replicas=1),
        _component("component2", "decode", replicas=2),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.is_deployment_ready.return_value = False

    # Act
    await kubernetes_connector.set_component_replicas(target_replicas)

    # Assert
    mock_kube_api.get_graph_deployment.assert_called_once()
    mock_kube_api.is_deployment_ready.assert_called_once_with(mock_deployment)
    mock_kube_api.update_graph_replicas.assert_not_called()
    mock_kube_api.wait_for_graph_deployment_ready.assert_not_called()


@pytest.mark.asyncio
async def test_set_component_replicas_deployment_not_ready_can_raise_for_global_planner(
    mock_kube_api_class, mock_kube_api, monkeypatch
):
    """Let GlobalPlanner opt in to retryable not-ready rejection."""
    # Arrange
    monkeypatch.setattr(
        "dynamo.planner.connectors.kubernetes.KubernetesAPI", mock_kube_api_class
    )
    with patch.dict(os.environ, {"DYN_PARENT_DGD_K8S_NAME": "test-graph"}):
        connector = KubernetesConnector("test-dynamo-namespace", raise_not_ready=True)
    target_replicas = [
        TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=3),
        TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=2),
    ]
    mock_deployment = _deployment(
        _component("component1", "prefill", replicas=1),
        _component("component2", "decode", replicas=2),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.is_deployment_ready.return_value = False

    # Act & Assert
    with pytest.raises(DynamoGraphDeploymentNotReadyError):
        await connector.set_component_replicas(target_replicas)

    mock_kube_api.get_graph_deployment.assert_called_once()
    mock_kube_api.is_deployment_ready.assert_called_once_with(mock_deployment)
    mock_kube_api.update_graph_replicas.assert_not_called()
    mock_kube_api.wait_for_graph_deployment_ready.assert_not_called()


@pytest.mark.asyncio
async def test_set_component_replicas_unready_allows_strict_dgdsa_zero_to_one(
    kubernetes_connector, mock_kube_api
):
    deployment = _unready_recovery_deployment(
        _adapter_component("worker", "decode", replicas=0)
    )
    adapter = _scaling_adapter("worker", replicas=0)
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True
    mock_kube_api.get_service_scaling_adapter.return_value = adapter

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker",
                desired_replicas=1,
            )
        ]
    )

    mock_kube_api.update_scaling_adapter_replicas.assert_called_once_with(
        "test-graph", "worker", 1, resource_version="101"
    )
    mock_kube_api.update_graph_replicas.assert_not_called()
    mock_kube_api.wait_for_graph_deployment_ready.assert_awaited_once_with("test-graph")
    assert mock_kube_api.non_planner_components_stable.call_count == 2
    assert kubernetes_connector._startup_scale_down_targets == {}


@pytest.mark.asyncio
async def test_set_component_replicas_unready_requires_pod_snapshot(
    kubernetes_connector, mock_kube_api
):
    deployment = _unready_recovery_deployment(
        _adapter_component("worker", "decode", replicas=0)
    )
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True
    mock_kube_api.get_service_scaling_adapter.return_value = _scaling_adapter(
        "worker", replicas=0
    )
    mock_kube_api.list_pods_for_graph.side_effect = client.ApiException(status=403)

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker",
                desired_replicas=1,
            )
        ]
    )

    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()
    mock_kube_api.wait_for_graph_deployment_ready.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("check_stage", ["initial", "refreshed"])
@pytest.mark.parametrize("hazard", ["nonterminal", "terminating", "pcsg-range"])
async def test_set_component_replicas_unready_rechecks_pod_lifecycle_before_write(
    kubernetes_connector, mock_kube_api, check_stage, hazard
):
    deployment = _unready_recovery_deployment(
        _adapter_component("worker", "decode", replicas=0)
    )
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True
    mock_kube_api.get_service_scaling_adapter.return_value = _scaling_adapter(
        "worker", replicas=0
    )
    mock_kube_api.partition_pods_by_component.side_effect = (
        KubernetesAPI.partition_pods_by_component
    )

    unsafe_pods = []
    if hazard != "pcsg-range":
        unsafe_pods = [
            client.V1Pod(
                metadata=client.V1ObjectMeta(
                    name="stale-worker-0",
                    labels={"nvidia.com/dynamo-component": "worker"},
                    owner_references=[],
                    deletion_timestamp=(
                        datetime.now(timezone.utc) if hazard == "terminating" else None
                    ),
                ),
                status=client.V1PodStatus(
                    phase="Succeeded" if hazard == "terminating" else "Pending"
                ),
            )
        ]

    if check_stage == "initial":
        mock_kube_api.list_pods_for_graph.return_value = unsafe_pods
        if hazard == "pcsg-range":
            mock_kube_api.pcsg_pods_within_desired_replicas.return_value = False
    else:
        mock_kube_api.list_pods_for_graph.side_effect = [[], unsafe_pods]
        if hazard == "pcsg-range":
            mock_kube_api.pcsg_pods_within_desired_replicas.side_effect = [True, False]

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker",
                desired_replicas=1,
            )
        ]
    )

    expected_snapshots = 1 if check_stage == "initial" else 2
    assert mock_kube_api.list_pods_for_graph.call_count == expected_snapshots
    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()
    mock_kube_api.wait_for_graph_deployment_ready.assert_not_called()


@pytest.mark.asyncio
async def test_startup_downscale_uses_strict_dgdsa_and_records_latch(
    kubernetes_connector, mock_kube_api
):
    deployment = _unready_recovery_deployment(
        _adapter_component("worker", "decode", replicas=2)
    )
    deployment["status"]["components"]["worker"] = {
        "replicas": 2,
        "updatedReplicas": 2,
        "readyReplicas": 1,
        "availableReplicas": 1,
    }
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True
    mock_kube_api.pending_startup_replicas.return_value = {"worker": 1}
    mock_kube_api.get_service_replica_status.return_value = (1, False)
    mock_kube_api.get_service_scaling_adapter.return_value = _scaling_adapter(
        "worker", replicas=2
    )
    mock_kube_api.get_service_replica_target.return_value = 2

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker",
                desired_replicas=1,
            )
        ],
        blocking=False,
    )

    mock_kube_api.update_scaling_adapter_replicas.assert_called_once_with(
        "test-graph", "worker", 1, resource_version="101"
    )
    assert kubernetes_connector._startup_scale_down_targets == {"worker": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("target_state", ["replicas", "ready", "missing"])
async def test_set_component_replicas_unready_requires_settled_zero_target(
    kubernetes_connector, mock_kube_api, target_state
):
    deployment = _unready_recovery_deployment(
        _adapter_component("worker", "decode", replicas=0)
    )
    if target_state == "replicas":
        deployment["status"]["components"]["worker"]["replicas"] = 1
    elif target_state == "ready":
        deployment["status"]["components"]["worker"]["readyReplicas"] = 1
    else:
        del deployment["status"]["components"]["worker"]
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True
    mock_kube_api.get_service_scaling_adapter.return_value = _scaling_adapter(
        "worker", replicas=0
    )

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker",
                desired_replicas=1,
            )
        ]
    )

    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()
    mock_kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_set_component_replicas_unready_accepts_available_only_zero_signal(
    kubernetes_connector, mock_kube_api
):
    deployment = _unready_recovery_deployment(
        _adapter_component("worker", "decode", replicas=0)
    )
    del deployment["status"]["components"]["worker"]["readyReplicas"]
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True
    mock_kube_api.get_service_scaling_adapter.return_value = _scaling_adapter(
        "worker", replicas=0
    )

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker",
                desired_replicas=1,
            )
        ]
    )

    mock_kube_api.update_scaling_adapter_replicas.assert_called_once_with(
        "test-graph", "worker", 1, resource_version="101"
    )


@pytest.mark.asyncio
async def test_set_component_replicas_unready_rechecks_peer_stability_before_write(
    kubernetes_connector, mock_kube_api
):
    deployment = _unready_recovery_deployment(
        _adapter_component("worker", "decode", replicas=0),
        _component("peer", "prefill", replicas=1),
    )
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True
    mock_kube_api.get_service_scaling_adapter.return_value = _scaling_adapter(
        "worker", replicas=0
    )
    mock_kube_api.non_planner_components_stable.side_effect = [
        (True, []),
        (False, ["peer"]),
    ]

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker",
                desired_replicas=1,
            )
        ]
    )

    assert mock_kube_api.non_planner_components_stable.call_count == 2
    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()
    mock_kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("peer_ready, expect_write", [(2, True), (1, False)])
async def test_set_component_replicas_unready_uses_peer_dgdsa_desired_state(
    kubernetes_connector, mock_kube_api, peer_ready, expect_write
):
    deployment = _unready_recovery_deployment(
        _adapter_component("worker", "decode", replicas=0),
        # The DGD seed remains zero while the peer DGDSA authoritatively wants
        # two replicas.
        _adapter_component("peer", "prefill", replicas=0),
    )
    deployment["status"]["components"]["peer"] = {
        "replicas": peer_ready,
        "updatedReplicas": peer_ready,
        "readyReplicas": peer_ready,
        "availableReplicas": peer_ready,
    }
    target_adapter = _scaling_adapter("worker", replicas=0)
    peer_adapter = _scaling_adapter("peer", replicas=2)
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True
    mock_kube_api.get_service_scaling_adapter.side_effect = [
        target_adapter,
        peer_adapter,
        target_adapter,
        peer_adapter,
    ]

    def evaluate_settlement(snapshot):
        unstable = []
        for component in snapshot["spec"]["components"]:
            if component.get("type") == "planner":
                continue
            name = component["name"]
            desired = component.get("replicas", 1)
            status = snapshot["status"]["components"][name]
            if not (
                desired == status["updatedReplicas"] == status["availableReplicas"]
            ):
                unstable.append(name)
        return not unstable, unstable

    mock_kube_api.non_planner_components_stable.side_effect = evaluate_settlement

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker",
                desired_replicas=1,
            )
        ]
    )

    if expect_write:
        mock_kube_api.update_scaling_adapter_replicas.assert_called_once_with(
            "test-graph", "worker", 1, resource_version="101"
        )
    else:
        mock_kube_api.update_scaling_adapter_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_set_component_replicas_ready_uses_authoritative_dgdsa_desired_state(
    kubernetes_connector, mock_kube_api
):
    # DGD propagation is stale at one, while the declared adapter still owns a
    # desired value of zero. Targeting one must patch the DGDSA, not skip.
    deployment = _deployment(_adapter_component("worker", "decode", replicas=1))
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = True
    mock_kube_api.get_service_scaling_adapter.return_value = _scaling_adapter(
        "worker", replicas=0
    )

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker",
                desired_replicas=1,
            )
        ]
    )

    mock_kube_api.update_scaling_adapter_replicas.assert_called_once_with(
        "test-graph", "worker", 1, resource_version="101"
    )
    mock_kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_set_component_replicas_ready_rejects_adapter_owned_by_replaced_dgd(
    kubernetes_connector, mock_kube_api
):
    deployment = _deployment(_adapter_component("worker", "decode", replicas=0))
    adapter = _scaling_adapter("worker", replicas=0)
    adapter["metadata"]["ownerReferences"][0]["uid"] = "replacement-uid"
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = True
    mock_kube_api.get_service_scaling_adapter.return_value = adapter

    with pytest.raises(ValueError, match="not controller-owned"):
        await kubernetes_connector.set_component_replicas(
            [
                TargetReplica(
                    sub_component_type=SubComponentType.DECODE,
                    component_name="worker",
                    desired_replicas=1,
                )
            ]
        )

    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()
    mock_kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_set_component_replicas_ready_rejects_missing_dgd_uid(
    kubernetes_connector, mock_kube_api
):
    deployment = _deployment(_adapter_component("worker", "decode", replicas=0))
    del deployment["metadata"]["uid"]
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = True
    mock_kube_api.get_service_scaling_adapter.return_value = _scaling_adapter(
        "worker", replicas=0
    )

    with pytest.raises(ValueError, match="metadata.uid is unavailable"):
        await kubernetes_connector.set_component_replicas(
            [
                TargetReplica(
                    sub_component_type=SubComponentType.DECODE,
                    component_name="worker",
                    desired_replicas=1,
                )
            ]
        )

    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_set_component_replicas_ready_propagates_adapter_conflict(
    kubernetes_connector, mock_kube_api
):
    deployment = _deployment(_adapter_component("worker", "decode", replicas=0))
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = True
    mock_kube_api.get_service_scaling_adapter.return_value = _scaling_adapter(
        "worker", replicas=0
    )
    mock_kube_api.update_scaling_adapter_replicas.side_effect = client.ApiException(
        status=409
    )

    with pytest.raises(client.ApiException) as exc_info:
        await kubernetes_connector.set_component_replicas(
            [
                TargetReplica(
                    sub_component_type=SubComponentType.DECODE,
                    component_name="worker",
                    desired_replicas=1,
                )
            ]
        )

    assert exc_info.value.status == 409
    mock_kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_set_component_replicas_ready_declared_adapter_404_never_falls_back(
    kubernetes_connector, mock_kube_api
):
    deployment = _deployment(_adapter_component("worker", "decode", replicas=0))
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = True
    mock_kube_api.get_service_scaling_adapter.side_effect = client.ApiException(
        status=404
    )

    with pytest.raises(client.ApiException) as exc_info:
        await kubernetes_connector.set_component_replicas(
            [
                TargetReplica(
                    sub_component_type=SubComponentType.DECODE,
                    component_name="worker",
                    desired_replicas=1,
                )
            ]
        )

    assert exc_info.value.status == 404
    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()
    mock_kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_set_component_replicas_unready_nonadapter_is_blocked(
    kubernetes_connector, mock_kube_api
):
    deployment = _unready_recovery_deployment(
        _component("worker", "decode", replicas=0)
    )
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker",
                desired_replicas=1,
            )
        ]
    )

    mock_kube_api.get_service_scaling_adapter.assert_not_called()
    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()
    mock_kube_api.update_graph_replicas.assert_not_called()
    mock_kube_api.wait_for_graph_deployment_ready.assert_not_called()


@pytest.mark.asyncio
async def test_set_component_replicas_unready_missing_adapter_is_blocked_atomically(
    kubernetes_connector, mock_kube_api
):
    deployment = _unready_recovery_deployment(
        _adapter_component("worker", "decode", replicas=0)
    )
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True
    mock_kube_api.get_service_scaling_adapter.side_effect = client.ApiException(
        status=404
    )

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker",
                desired_replicas=1,
            )
        ]
    )

    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()
    mock_kube_api.update_graph_replicas.assert_not_called()
    mock_kube_api.wait_for_graph_deployment_ready.assert_not_called()


@pytest.mark.asyncio
async def test_set_component_replicas_unready_missing_adapter_uses_raise_contract(
    mock_kube_api_class, mock_kube_api, monkeypatch
):
    monkeypatch.setattr(
        "dynamo.planner.connectors.kubernetes.KubernetesAPI", mock_kube_api_class
    )
    with patch.dict(os.environ, {"DYN_PARENT_DGD_K8S_NAME": "test-graph"}):
        connector = KubernetesConnector("test-dynamo-namespace", raise_not_ready=True)
    deployment = _unready_recovery_deployment(
        _adapter_component("worker", "decode", replicas=0)
    )
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True
    mock_kube_api.get_service_scaling_adapter.side_effect = client.ApiException(
        status=404
    )

    with pytest.raises(DynamoGraphDeploymentNotReadyError):
        await connector.set_component_replicas(
            [
                TargetReplica(
                    sub_component_type=SubComponentType.DECODE,
                    component_name="worker",
                    desired_replicas=1,
                )
            ]
        )

    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()
    mock_kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current", "target"),
    [(1, 0), (1, 2)],
    ids=["downscale", "nonzero-scale-up"],
)
async def test_set_component_replicas_unready_blocks_nonbootstrap_mutations(
    kubernetes_connector, mock_kube_api, current, target
):
    deployment = _unready_recovery_deployment(
        _adapter_component("worker", "decode", replicas=current)
    )
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True
    mock_kube_api.get_service_scaling_adapter.return_value = _scaling_adapter(
        "worker", replicas=current
    )

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker",
                desired_replicas=target,
            )
        ]
    )

    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()
    mock_kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_set_component_replicas_unready_rejects_parent_replacement_before_write(
    kubernetes_connector, mock_kube_api
):
    initial_deployment = _unready_recovery_deployment(
        _adapter_component("worker", "decode", replicas=0)
    )
    replacement_deployment = _unready_recovery_deployment(
        _adapter_component("worker", "decode", replicas=0)
    )
    replacement_deployment["metadata"]["uid"] = "replacement-uid"
    mock_kube_api.get_graph_deployment.side_effect = [
        initial_deployment,
        replacement_deployment,
    ]
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True
    mock_kube_api.get_service_scaling_adapter.return_value = _scaling_adapter(
        "worker", replicas=0
    )

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker",
                desired_replicas=1,
            )
        ]
    )

    assert mock_kube_api.get_graph_deployment.call_count == 2
    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()
    mock_kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_set_component_replicas_unready_allows_adapter_same_target_noop(
    kubernetes_connector, mock_kube_api
):
    deployment = _unready_recovery_deployment(
        _adapter_component("worker", "decode", replicas=1)
    )
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True
    mock_kube_api.get_service_scaling_adapter.return_value = _scaling_adapter(
        "worker", replicas=1
    )

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker",
                desired_replicas=1,
            )
        ]
    )

    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()
    mock_kube_api.update_graph_replicas.assert_not_called()
    mock_kube_api.wait_for_graph_deployment_ready.assert_not_called()


@pytest.mark.asyncio
async def test_set_component_replicas_unready_generation_lag_blocks_before_adapter_read(
    kubernetes_connector, mock_kube_api
):
    deployment = _unready_recovery_deployment(
        _adapter_component("worker", "decode", replicas=0)
    )
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = False

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker",
                desired_replicas=1,
            )
        ]
    )

    mock_kube_api.get_service_scaling_adapter.assert_not_called()
    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()
    mock_kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("unsafe_state", ["dgd-deleting", "rolling", "failed"])
async def test_set_component_replicas_unready_rejects_unsafe_dgd_snapshot(
    kubernetes_connector, mock_kube_api, unsafe_state
):
    deployment = _unready_recovery_deployment(
        _adapter_component("worker", "decode", replicas=0)
    )
    if unsafe_state == "dgd-deleting":
        deployment["metadata"]["deletionTimestamp"] = "2026-08-28T21:00:00Z"
    elif unsafe_state == "rolling":
        deployment["status"]["rollingUpdate"] = {"phase": "InProgress"}
    else:
        deployment["status"]["state"] = "FAILED"
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker",
                desired_replicas=1,
            )
        ]
    )

    mock_kube_api.get_service_scaling_adapter.assert_not_called()
    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsafe_adapter",
    [
        "deleting",
        "missing-resource-version",
        "wrong-owner-uid",
        "missing-owner-api-version",
        "wrong-dgd-ref",
        "wrong-component-ref",
        "wrong-label",
        "malformed-owner-references",
        "malformed-dgd-ref",
    ],
)
async def test_set_component_replicas_unready_rejects_untrusted_adapter(
    kubernetes_connector, mock_kube_api, unsafe_adapter
):
    deployment = _unready_recovery_deployment(
        _adapter_component("worker", "decode", replicas=0)
    )
    adapter = _scaling_adapter("worker", replicas=0)
    if unsafe_adapter == "deleting":
        adapter["metadata"]["deletionTimestamp"] = "2026-08-28T21:00:00Z"
    elif unsafe_adapter == "missing-resource-version":
        del adapter["metadata"]["resourceVersion"]
    elif unsafe_adapter == "wrong-owner-uid":
        adapter["metadata"]["ownerReferences"][0]["uid"] = "other-uid"
    elif unsafe_adapter == "missing-owner-api-version":
        del adapter["metadata"]["ownerReferences"][0]["apiVersion"]
    elif unsafe_adapter == "wrong-dgd-ref":
        adapter["spec"]["dgdRef"]["name"] = "other-graph"
    elif unsafe_adapter == "wrong-component-ref":
        adapter["spec"]["dgdRef"]["componentName"] = "other-worker"
    elif unsafe_adapter == "wrong-label":
        adapter["metadata"]["labels"]["nvidia.com/dynamo-component"] = "other-worker"
    elif unsafe_adapter == "malformed-owner-references":
        adapter["metadata"]["ownerReferences"] = {"controller": True}
    else:
        adapter["spec"]["dgdRef"] = ["test-graph", "worker"]
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True
    mock_kube_api.get_service_scaling_adapter.return_value = adapter

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="worker",
                desired_replicas=1,
            )
        ]
    )

    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()
    mock_kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_set_component_replicas_unready_preflights_all_before_first_write(
    kubernetes_connector, mock_kube_api
):
    deployment = _unready_recovery_deployment(
        _adapter_component("prefill-worker", "prefill", replicas=0),
        _adapter_component("decode-worker", "decode", replicas=0),
    )
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True
    mock_kube_api.get_service_scaling_adapter.side_effect = [
        _scaling_adapter("prefill-worker", replicas=0),
        client.ApiException(status=404),
    ]

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL,
                component_name="prefill-worker",
                desired_replicas=1,
            ),
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="decode-worker",
                desired_replicas=1,
            ),
        ]
    )

    assert mock_kube_api.get_service_scaling_adapter.call_count == 2
    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()
    mock_kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_set_component_replicas_unready_rejects_multiple_bootstrap_writes(
    kubernetes_connector, mock_kube_api
):
    deployment = _unready_recovery_deployment(
        _adapter_component("prefill-worker", "prefill", replicas=0),
        _adapter_component("decode-worker", "decode", replicas=0),
    )
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.is_deployment_ready.return_value = False
    mock_kube_api.is_spec_generation_observed.return_value = True
    mock_kube_api.get_service_scaling_adapter.side_effect = [
        _scaling_adapter("prefill-worker", replicas=0),
        _scaling_adapter("decode-worker", replicas=0),
    ]

    await kubernetes_connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL,
                component_name="prefill-worker",
                desired_replicas=1,
            ),
            TargetReplica(
                sub_component_type=SubComponentType.DECODE,
                component_name="decode-worker",
                desired_replicas=1,
            ),
        ]
    )

    mock_kube_api.update_scaling_adapter_replicas.assert_not_called()
    mock_kube_api.update_graph_replicas.assert_not_called()


@pytest.mark.asyncio
async def test_validate_deployment_true(kubernetes_connector, mock_kube_api):
    # Arrange
    mock_deployment = _deployment(
        _component(
            "component1",
            "prefill",
            replicas=1,
            args=["--served-model-name", "prefill-model"],
        ),
        _component("component2", "decode", replicas=2),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    # Act
    await kubernetes_connector.validate_deployment(decode_component_name="component2")


@pytest.mark.asyncio
async def test_validate_deployment_uses_names_for_unannotated_legacy_components(
    kubernetes_connector, mock_kube_api
):
    mock_kube_api.get_graph_deployment.return_value = _deployment(
        _component(
            "prefill",
            replicas=1,
            args=["--served-model-name", "test-model"],
        ),
        _component(
            "decode",
            replicas=1,
            args=["--served-model-name", "test-model"],
        ),
    )

    await kubernetes_connector.validate_deployment(
        prefill_component_name="prefill",
        decode_component_name="decode",
    )


@pytest.mark.asyncio
async def test_validate_deployment_fail(kubernetes_connector, mock_kube_api):
    # Arrange
    mock_deployment = _deployment(
        _component("component1", "prefill", replicas=1),
        _component("component2", "prefill", replicas=2),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    # Act
    with pytest.raises(DeploymentValidationError) as exc_info:
        await kubernetes_connector.validate_deployment()

    exception = exc_info.value
    assert set(exception.errors) == {
        str(DuplicateSubComponentError("prefill", ["component1", "component2"])),
        str(SubComponentNotFoundError("decode")),
    }


def test_get_model_name_both_none_raises_error(kubernetes_connector, mock_kube_api):
    # Arrange
    mock_deployment = _deployment(
        _component("component1", "prefill", replicas=1),
        _component("component2", "decode", replicas=2),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    with pytest.raises(ModelNameNotFoundError):
        kubernetes_connector.get_model_name()


def test_get_model_name_prefill_none_decode_valid_returns_decode(
    kubernetes_connector, mock_kube_api
):
    # Arrange
    mock_deployment = _deployment(
        _component("component1", "prefill", replicas=1),
        _component(
            "component2",
            "decode",
            replicas=2,
            args=["--served-model-name", "test-model"],
        ),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    # Act
    result = kubernetes_connector.get_model_name()

    # Assert
    assert result == "test-model"


def test_get_model_name_mismatch_raises_error(kubernetes_connector, mock_kube_api):
    mock_deployment = _deployment(
        _component(
            "component1",
            "prefill",
            replicas=1,
            args=["--served-model-name", "prefill-model"],
        ),
        _component(
            "component2",
            "decode",
            replicas=2,
            args=["--served-model-name", "decode-model"],
        ),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    # Act & Assert
    with pytest.raises(DeploymentModelNameMismatchError) as exc_info:
        kubernetes_connector.get_model_name()

    exception = exc_info.value
    assert exception.prefill_model_name == "prefill-model"
    assert exception.decode_model_name == "decode-model"


def test_get_model_name_agree_returns_model_name(kubernetes_connector, mock_kube_api):
    # Arrange
    mock_deployment = _deployment(
        _component(
            "component1",
            "prefill",
            replicas=1,
            args=["--served-model-name", "agreed-model"],
        ),
        _component(
            "component2",
            "decode",
            replicas=2,
            args=["--served-model-name", "agreed-model"],
        ),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    # Act
    result = kubernetes_connector.get_model_name()

    # Assert
    assert result == "agreed-model"


def test_protocol_positional_flags_match_kubernetes_connector(
    kubernetes_connector, mock_kube_api
):
    """Protocol-style positional flags must not bind to a deployment argument."""
    mock_kube_api.get_graph_deployment.return_value = _deployment(
        _component(
            "decode-worker",
            "decode",
            replicas=1,
            args=["--served-model-name", "decode-model"],
            gpu=4,
        )
    )
    connector: PlannerConnector = kubernetes_connector

    assert connector.get_model_name(False, True) == "decode-model"
    assert connector.get_gpu_counts(False, True) == (0, 4)


# Planner component facts come exclusively from current operator status.
def test_service_reads_current_component_status_without_inspecting_roles():
    service = Service(
        name="prefill",
        service={
            "roles": [
                {
                    "name": "leader",
                    "podTemplate": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "args": ["--model", "wrong-leader-model"],
                                    "resources": {"limits": {"nvidia.com/gpu": "99"}},
                                }
                            ]
                        }
                    },
                },
                {
                    "name": "worker",
                    "podTemplate": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "args": ["--model", "wrong-worker-model"],
                                }
                            ]
                        }
                    },
                },
            ]
        },
    )
    deployment = {
        "metadata": {"generation": 2},
        "status": {
            "observedGeneration": 2,
            "components": {
                "prefill": {
                    "servedModelName": "Qwen/Qwen3-8B",
                    "runtimeComponentName": "custom-prefill",
                    "gpuPowerLimitWatts": 300,
                    "gpusPerEngine": 2,
                    "gpusPerReplica": 4,
                }
            },
        },
    }

    assert service.get_model_name(deployment) == "Qwen/Qwen3-8B"
    assert service.get_runtime_component_name(deployment) == "custom-prefill"
    assert service.get_gpu_power_limit_watts(deployment) == 300
    assert service.get_gpu_shape(deployment).gpus_per_engine == 2
    assert service.get_gpu_shape(deployment).gpus_per_replica == 4


def test_service_ignores_stale_component_status():
    service = Service(name="decode", service={"name": "decode"})
    deployment = {
        "metadata": {"generation": 3},
        "status": {
            "observedGeneration": 2,
            "components": {
                "decode": {
                    "servedModelName": "stale-model",
                    "runtimeComponentName": "stale-decode",
                    "gpuPowerLimitWatts": 300,
                }
            },
        },
    }

    assert service.get_model_name(deployment) is None
    assert service.get_runtime_component_name(deployment) is None
    with pytest.raises(PowerAnnotationMissingError):
        service.get_gpu_power_limit_watts(deployment)


# Tests for KubernetesConnector.get_gpu_counts()
def test_get_gpu_counts_both_services(kubernetes_connector, mock_kube_api):
    """Test get_gpu_counts returns correct counts for both prefill and decode"""
    mock_deployment = _deployment(
        _component("prefill-worker", "prefill", replicas=1, gpu=2),
        _component("decode-worker", "decode", replicas=1, gpu=4),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    prefill_gpu, decode_gpu = kubernetes_connector.get_gpu_counts()

    assert prefill_gpu == 2
    assert decode_gpu == 4


def test_get_gpu_counts_prefill_only(kubernetes_connector, mock_kube_api):
    """Test get_gpu_counts with require_decode=False"""
    mock_deployment = _deployment(
        _component("prefill-worker", "prefill", replicas=1, gpu=2)
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    prefill_gpu, decode_gpu = kubernetes_connector.get_gpu_counts(
        require_prefill=True, require_decode=False
    )

    assert prefill_gpu == 2
    assert decode_gpu == 0


def test_get_gpu_counts_decode_only(kubernetes_connector, mock_kube_api):
    """Test get_gpu_counts with require_prefill=False"""
    mock_deployment = _deployment(
        _component("decode-worker", "decode", replicas=1, gpu=4)
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    prefill_gpu, decode_gpu = kubernetes_connector.get_gpu_counts(
        require_prefill=False, require_decode=True
    )

    assert prefill_gpu == 0
    assert decode_gpu == 4


def test_get_gpu_shapes_from_operator_resolved_dra_status(
    kubernetes_connector, mock_kube_api
):
    """DRA-backed workers use the current operator-projected GPU shape."""
    mock_deployment = _deployment(_component("decode-worker", "decode", replicas=1))
    mock_deployment["metadata"]["generation"] = 2
    mock_deployment["status"] = {
        "observedGeneration": 2,
        "components": {"decode-worker": {"gpusPerEngine": 2, "gpusPerReplica": 3}},
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    assert kubernetes_connector.get_gpu_counts(
        require_prefill=False, require_decode=True
    ) == (0, 2)
    _, decode_shape = kubernetes_connector.get_gpu_shapes(
        require_prefill=False, require_decode=True
    )
    assert decode_shape.gpus_per_engine == 2
    assert decode_shape.gpus_per_replica == 3


@pytest.mark.parametrize("replica_cost", [4, 5])
def test_get_gpu_shapes_separates_engine_width_from_sidecar_cost(
    kubernetes_connector, mock_kube_api, replica_cost
):
    mock_deployment = _deployment(_component("decode-worker", "decode", replicas=1))
    mock_deployment["metadata"]["generation"] = 2
    mock_deployment["status"] = {
        "observedGeneration": 2,
        "components": {
            "decode-worker": {
                "gpusPerEngine": 4,
                "gpusPerReplica": replica_cost,
            }
        },
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    _, decode_shape = kubernetes_connector.get_gpu_shapes(
        require_prefill=False, require_decode=True
    )

    assert decode_shape.gpus_per_engine == 4
    assert decode_shape.gpus_per_replica == replica_cost


@pytest.mark.parametrize("sidecar_gpu", [0, 1])
def test_missing_shape_does_not_inspect_sidecar_spec(
    kubernetes_connector, mock_kube_api, sidecar_gpu
):
    component = _component("decode-worker", "decode", replicas=1, gpu=4)
    component["podTemplate"]["spec"]["containers"].append(
        {
            "name": "sidecar",
            "resources": {"limits": {"nvidia.com/gpu": str(sidecar_gpu)}},
        }
    )
    deployment = _deployment(component)
    deployment["status"] = {"state": "failed", "components": {"decode-worker": {}}}
    mock_kube_api.get_graph_deployment.return_value = deployment

    with pytest.raises(GPUShapeUnavailableError, match="not current"):
        kubernetes_connector.get_gpu_shapes(require_prefill=False, require_decode=True)


def test_missing_shape_for_pure_dra_worker_fails_closed(
    kubernetes_connector, mock_kube_api
):
    component = _component("decode-worker", "decode", replicas=1)
    component["podTemplate"] = {
        "spec": {
            "resourceClaims": [
                {"name": "gpu", "resourceClaimTemplateName": "gpu-template"}
            ],
            "containers": [
                {"name": "main", "resources": {"claims": [{"name": "gpu"}]}}
            ],
        }
    }
    deployment = _deployment(component)
    deployment["status"] = {"state": "failed", "components": {"decode-worker": {}}}
    mock_kube_api.get_graph_deployment.return_value = deployment

    with pytest.raises(GPUShapeUnavailableError, match="not current"):
        kubernetes_connector.get_gpu_shapes(require_prefill=False, require_decode=True)


@pytest.mark.parametrize("sidecar_gpu", [0, 1])
def test_missing_shape_does_not_inspect_native_sidecar_spec(
    kubernetes_connector, mock_kube_api, sidecar_gpu
):
    component = _component("decode-worker", "decode", replicas=1, gpu=4)
    component["podTemplate"]["spec"]["initContainers"] = [
        {
            "name": "native-sidecar",
            "restartPolicy": "Always",
            "resources": {"limits": {"nvidia.com/gpu": str(sidecar_gpu)}},
        }
    ]
    deployment = _deployment(component)
    deployment["status"] = {"state": "failed", "components": {"decode-worker": {}}}
    mock_kube_api.get_graph_deployment.return_value = deployment

    with pytest.raises(GPUShapeUnavailableError, match="not current"):
        kubernetes_connector.get_gpu_shapes(require_prefill=False, require_decode=True)


@pytest.mark.parametrize("init_gpu", [0, 8])
def test_missing_shape_does_not_inspect_one_shot_init_spec(
    kubernetes_connector, mock_kube_api, init_gpu
):
    component = _component("decode-worker", "decode", replicas=1, gpu=4)
    component["podTemplate"]["spec"]["initContainers"] = [
        {
            "name": "one-shot-init",
            "resources": {"limits": {"nvidia.com/gpu": str(init_gpu)}},
        }
    ]
    deployment = _deployment(component)
    deployment["status"] = {"state": "failed", "components": {"decode-worker": {}}}
    mock_kube_api.get_graph_deployment.return_value = deployment

    with pytest.raises(GPUShapeUnavailableError, match="not current"):
        kubernetes_connector.get_gpu_shapes(require_prefill=False, require_decode=True)


def test_worker_with_zero_physical_shape_uses_logical_gpu_fallback(
    kubernetes_connector, mock_kube_api
):
    deployment = _deployment(_component("decode-worker", "decode", replicas=1))
    deployment["metadata"]["generation"] = 2
    deployment["status"] = {
        "observedGeneration": 2,
        "components": {"decode-worker": {"gpusPerEngine": 0, "gpusPerReplica": 0}},
    }
    mock_kube_api.get_graph_deployment.return_value = deployment

    assert kubernetes_connector.get_gpu_shapes(
        require_prefill=False, require_decode=True
    ) == (None, None)
    with pytest.raises(
        DeploymentValidationError,
        match="Decode mocker requires a configured logical GPU",
    ):
        kubernetes_connector.get_gpu_counts(
            require_prefill=False, require_decode=True, deployment=deployment
        )


@pytest.mark.parametrize("observed_generation", [1, 3])
def test_get_gpu_counts_rejects_noncurrent_dra_status(
    kubernetes_connector, mock_kube_api, observed_generation
):
    """Only the current generation's resolved count can authorize GPU budget."""
    mock_deployment = _deployment(_component("decode-worker", "decode", replicas=1))
    mock_deployment["metadata"]["generation"] = 2
    mock_deployment["status"] = {
        "observedGeneration": observed_generation,
        "components": {"decode-worker": {"gpusPerEngine": 2, "gpusPerReplica": 2}},
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    with pytest.raises(
        GPUShapeUnavailableError, match="Resolved GPU shape.*not current"
    ):
        kubernetes_connector.get_gpu_counts(require_prefill=False, require_decode=True)


def test_get_gpu_counts_missing_gpu_raises_error(kubernetes_connector, mock_kube_api):
    """A missing operator-projected GPU shape fails closed."""
    mock_deployment = _deployment(
        _component("prefill-worker", "prefill", replicas=1),
        _component("decode-worker", "decode", replicas=1, gpu=4),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    with pytest.raises(GPUShapeUnavailableError) as exc_info:
        kubernetes_connector.get_gpu_counts()

    assert "prefill-worker" in str(exc_info.value)
    assert "gpusPerEngine/gpusPerReplica" in str(exc_info.value)


def test_get_gpu_counts_service_not_found_raises_error(
    kubernetes_connector, mock_kube_api
):
    """Test get_gpu_counts raises DeploymentValidationError when service not found"""
    mock_deployment = _deployment(
        _component("prefill-worker", "prefill", replicas=1, gpu=2)
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    with pytest.raises(DeploymentValidationError) as exc_info:
        kubernetes_connector.get_gpu_counts()

    assert "decode GPU shape" in str(exc_info.value)


# Tests for get_actual_worker_counts


@pytest.mark.asyncio
async def test_get_actual_worker_counts_stable(kubernetes_connector, mock_kube_api):
    """Test get_actual_worker_counts when both services are stable"""
    mock_deployment = _deployment(
        _component("prefill-component"),
        _component("decode-component"),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.get_service_replica_status.side_effect = [(2, True), (4, True)]

    (
        prefill_count,
        decode_count,
        is_stable,
    ) = await kubernetes_connector.get_actual_worker_counts(
        prefill_component_name="prefill-component",
        decode_component_name="decode-component",
    )

    assert prefill_count == 2
    assert decode_count == 4
    assert is_stable is True


@pytest.mark.asyncio
async def test_worker_inventory_uses_authoritative_dgdsa_target(
    kubernetes_connector, mock_kube_api
):
    deployment = _deployment(
        _adapter_component("worker", "decode", replicas=1),
    )
    deployment["status"]["components"]["worker"].update(
        {
            "replicas": 0,
            "updatedReplicas": 0,
            "readyReplicas": 0,
            "availableReplicas": 0,
        }
    )
    authoritative = deepcopy(deployment)
    authoritative["spec"]["components"][0]["replicas"] = 0
    mock_kube_api.get_graph_deployment.return_value = deployment
    mock_kube_api.fetch_authoritative_replica_targets.return_value = authoritative
    mock_kube_api.get_service_replica_status.return_value = (0, True)
    mock_kube_api.is_spec_generation_observed.return_value = True

    inventory = await kubernetes_connector.get_worker_inventory(
        prefill_component_name=None,
        decode_component_name="worker",
    )

    assert inventory is not None
    assert inventory.ready_num_decode == 0
    assert inventory.expected_num_decode == 0
    assert inventory.decode_scaling_in_progress is False
    mock_kube_api.fetch_authoritative_replica_targets.assert_called_once_with(
        deployment
    )


@pytest.mark.asyncio
async def test_get_actual_worker_counts_prefill_rollout_in_progress(
    kubernetes_connector, mock_kube_api
):
    """Test get_actual_worker_counts when prefill has rollout in progress"""
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "components": [
                _component("prefill-component"),
                _component("decode-component"),
            ]
        },
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.get_service_replica_status.side_effect = [(2, False), (4, True)]

    (
        prefill_count,
        decode_count,
        is_stable,
    ) = await kubernetes_connector.get_actual_worker_counts(
        prefill_component_name="prefill-component",
        decode_component_name="decode-component",
    )

    assert prefill_count == 2
    assert decode_count == 4
    assert is_stable is False


@pytest.mark.asyncio
async def test_get_actual_worker_counts_prefill_only(
    kubernetes_connector, mock_kube_api
):
    """Test get_actual_worker_counts with only prefill component"""
    mock_deployment = _deployment(
        _component("prefill-component", "prefill", replicas=2)
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.get_service_replica_status.return_value = (2, True)

    (
        prefill_count,
        decode_count,
        is_stable,
    ) = await kubernetes_connector.get_actual_worker_counts(
        prefill_component_name="prefill-component",
        decode_component_name=None,
    )

    assert prefill_count == 2
    assert decode_count == 0
    assert is_stable is True


@pytest.mark.asyncio
async def test_get_actual_worker_counts_decode_only(
    kubernetes_connector, mock_kube_api
):
    """Test get_actual_worker_counts with only decode component"""
    mock_deployment = _deployment(_component("decode-component", "decode", replicas=4))
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.get_service_replica_status.return_value = (4, True)

    (
        prefill_count,
        decode_count,
        is_stable,
    ) = await kubernetes_connector.get_actual_worker_counts(
        prefill_component_name=None,
        decode_component_name="decode-component",
    )

    assert prefill_count == 0
    assert decode_count == 4
    assert is_stable is True


@pytest.mark.asyncio
async def test_get_actual_worker_counts_no_components(
    kubernetes_connector, mock_kube_api
):
    """Test get_actual_worker_counts with no components specified"""
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {"components": []},
        "status": {"components": {}},
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    (
        prefill_count,
        decode_count,
        is_stable,
    ) = await kubernetes_connector.get_actual_worker_counts(
        prefill_component_name=None,
        decode_component_name=None,
    )

    assert prefill_count == 0
    assert decode_count == 0
    assert is_stable is True


@pytest.mark.asyncio
async def test_get_actual_worker_counts_no_pod_list_when_power_disabled(
    kubernetes_connector, mock_kube_api
):
    """The ordinary connector path remains Pod-list free."""
    mock_deployment = _deployment(
        _component("prefill-component"),
        _component("decode-component"),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.get_service_replica_status.side_effect = [(2, True), (4, True)]

    await kubernetes_connector.get_actual_worker_counts(
        prefill_component_name="prefill-component",
        decode_component_name="decode-component",
    )

    mock_kube_api.list_pods_for_graph.assert_not_called()
    mock_kube_api.has_terminating_pods.assert_not_called()


@pytest.mark.asyncio
async def test_get_power_aware_worker_counts_uses_one_partitioned_pod_snapshot(
    kubernetes_connector, mock_kube_api
):
    """The power-aware path lists once and checks locally partitioned Pods."""
    mock_deployment = _deployment(
        _component("prefill-component"),
        _component("decode-component"),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.get_service_replica_status.side_effect = [(2, True), (4, True)]
    prefill_pods = [object()]
    decode_pods = [object()]
    all_pods = [*prefill_pods, *decode_pods]
    mock_kube_api.list_pods_for_graph.return_value = all_pods
    mock_kube_api.partition_pods_by_component.return_value = {
        "prefill-component": prefill_pods,
        "decode-component": decode_pods,
    }
    mock_kube_api.has_terminating_pods.return_value = False

    (
        prefill_count,
        decode_count,
        is_stable,
    ) = await kubernetes_connector.get_power_aware_worker_counts(
        prefill_component_name="prefill-component",
        decode_component_name="decode-component",
    )

    mock_kube_api.list_pods_for_graph.assert_called_once_with("test-graph")
    mock_kube_api.partition_pods_by_component.assert_called_once_with(all_pods)
    assert mock_kube_api.has_terminating_pods.call_args_list == [
        call(prefill_pods),
        call(decode_pods),
    ]
    assert mock_kube_api.has_terminating_pods.call_count == 2
    assert is_stable is True
    assert prefill_count == 2
    assert decode_count == 4


@pytest.mark.asyncio
async def test_get_power_aware_worker_counts_inprogress_rollout_is_unstable(
    kubernetes_connector, mock_kube_api
):
    """InProgress rollout with replica-stable counts must be unstable when power is on.

    Startup settlement already blocks on Pending/InProgress via
    is_rolling_update_blocking_settlement. The runtime power snapshot
    must apply the same gate so
    a scale-up is not admitted while old and new pod generations overlap.
    """
    mock_deployment = _deployment(
        _component("prefill-component"),
        _component("decode-component"),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    # Replica counts look stable to per-service checks.
    mock_kube_api.get_service_replica_status.side_effect = [(2, True), (4, True)]
    mock_kube_api.has_terminating_pods.return_value = False
    # Deployment-level rollingUpdate is InProgress.
    mock_kube_api.is_rolling_update_blocking_settlement.return_value = (
        True,
        "rollingUpdate.phase=InProgress",
    )

    _, _, is_stable = await kubernetes_connector.get_power_aware_worker_counts(
        prefill_component_name="prefill-component",
        decode_component_name="decode-component",
    )

    assert is_stable is False


@pytest.mark.asyncio
async def test_get_power_aware_worker_counts_failed_rollout_is_unstable(
    kubernetes_connector, mock_kube_api
):
    """Failed rollout must be treated as unstable by the power snapshot.

    is_rolling_update_blocking_settlement only covers Pending/InProgress; Failed
    was intentionally excluded there because startup raises immediately. At
    runtime there is no raise, so the power-aware method must be fail-closed
    and return is_stable=False so power-aware ticks do not admit scale-ups
    during a terminal (Failed) rollout state.
    """
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "components": [
                _component("prefill-component"),
                _component("decode-component"),
            ]
        },
        "status": {
            "rollingUpdate": {"phase": "Failed", "message": "pod CrashLoopBackOff"}
        },
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.get_service_replica_status.side_effect = [(2, True), (4, True)]
    mock_kube_api.has_terminating_pods.return_value = False
    # is_rolling_update_blocking_settlement does not cover Failed; simulate that.
    mock_kube_api.is_rolling_update_blocking_settlement.return_value = (False, "")

    _, _, is_stable = await kubernetes_connector.get_power_aware_worker_counts(
        prefill_component_name="prefill-component",
        decode_component_name="decode-component",
    )

    assert is_stable is False


@pytest.mark.asyncio
async def test_get_actual_worker_counts_inprogress_rollout_is_stable_when_power_off(
    kubernetes_connector, mock_kube_api
):
    """InProgress rollout does not affect the ordinary count path.

    Power-disabled planners do not have pods/list RBAC and must not call the
    rolling-update helper. The legacy replica-count path stays unchanged.
    """
    mock_deployment = _deployment(
        _component("prefill-component"),
        _component("decode-component"),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.get_service_replica_status.side_effect = [(2, True), (4, True)]

    _, _, is_stable = await kubernetes_connector.get_actual_worker_counts(
        prefill_component_name="prefill-component",
        decode_component_name="decode-component",
    )

    assert is_stable is True
    mock_kube_api.is_rolling_update_blocking_settlement.assert_not_called()


# Tests for _resolve_dgd_service / get_worker_info component-filter.
#
# Regression: the filter that compares an MDC entry's ``component`` field
# against ``expected_component`` must use the lowercase backend-default
# name (what the Rust runtime writes to MDC), NOT the DGD ``spec.services``
# dict key. The DGD key is typically PascalCase (``prefill``)
# while MDC carries the Endpoint name (``prefill`` / ``backend``);
# returning the DGD component name for the filter would cause every real-world MDC
# entry to be skipped, leaving WorkerInfo without ``context_length`` and
# silently breaking easy-mode load scaling.


@pytest.mark.parametrize("pod_suffix", ["", "-f4k85"])
def test_extract_mdc_entries_uses_truncated_grove_component_name(
    kubernetes_connector, mock_kube_api, pod_suffix
):
    dgd_name = "live-verify-accept-len-win-df9e"
    component_name = "live-verify-accept-len--473a-0-decode"
    cr_name = f"{component_name}{pod_suffix}"
    deployment = _deployment(_component("decode", "decode", replicas=1))
    deployment["metadata"]["name"] = dgd_name
    deployment["status"] = {
        "components": {
            "decode": {"componentNames": [component_name]},
        }
    }
    kubernetes_connector.graph_deployment_name = dgd_name
    mock_kube_api.get_graph_deployment.return_value = deployment
    kubernetes_connector._list_worker_metadata_crs = Mock(
        return_value=[_model_card_cr(cr_name)]
    )

    assert not cr_name.startswith(f"{dgd_name}-")
    entries = kubernetes_connector._extract_mdc_entries()

    assert len(entries) == 1
    assert entries[0].card_json["worker_type"] == "decode"


def test_extract_mdc_entries_uses_dgd_prefix_with_partial_component_names(
    kubernetes_connector, mock_kube_api
):
    deployment = _deployment(_component("decode", "decode", replicas=1))
    deployment["status"] = {
        "components": {
            "Frontend": {"componentNames": ["test-graph-0-frontend"]},
        }
    }
    mock_kube_api.get_graph_deployment.return_value = deployment
    kubernetes_connector._list_worker_metadata_crs = Mock(
        return_value=[_model_card_cr("test-graph-0-decode-f4k85")]
    )

    entries = kubernetes_connector._extract_mdc_entries()

    assert len(entries) == 1
    assert entries[0].card_json["worker_type"] == "decode"


def test_resolve_dgd_service_prefill_uses_backend_default_for_filter(
    kubernetes_connector, mock_kube_api
):
    """vLLM prefill: filter name = "prefill" (MDC side), not DGD component name."""
    mock_deployment = _deployment(_component("custom-prefill", "prefill", replicas=1))
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    dgd_service_name, expected_component = kubernetes_connector._resolve_dgd_service(
        SubComponentType.PREFILL, backend="vllm"
    )

    # k8s operations (e.g. replica patch) still target the DGD component name.
    assert dgd_service_name == "custom-prefill"
    # The filter side must match what the Rust runtime writes to MDC.
    assert expected_component == "prefill"


def test_resolve_dgd_service_uses_projected_endpoint_override(
    kubernetes_connector, mock_kube_api
):
    mock_deployment = _deployment(
        _component(
            "prefill",
            component_type="prefill",
            replicas=1,
            args=[
                "--endpoint",
                "my-ns.my-custom-prefill.generate",
                "--model",
                "Qwen/Qwen3-8B",
            ],
        )
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    dgd_service_name, expected_component = kubernetes_connector._resolve_dgd_service(
        SubComponentType.PREFILL, backend="vllm"
    )

    assert dgd_service_name == "prefill"
    assert expected_component == "my-custom-prefill"


def test_resolve_dgd_service_decode_uses_backend_default_for_filter(
    kubernetes_connector, mock_kube_api
):
    """vLLM decode: MDC carries "backend", NOT "decode"; filter must match that."""
    mock_deployment = _deployment(
        _component("decode", component_type="decode", replicas=1)
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    dgd_service_name, expected_component = kubernetes_connector._resolve_dgd_service(
        SubComponentType.DECODE, backend="vllm"
    )

    assert dgd_service_name == "decode"
    # Critically, vLLM's decode-worker component name is "backend" (from
    # VllmComponentName.decode_worker_component_name). Using
    # SubComponentType.DECODE.value ("decode") here would break decode
    # filtering on every backend.
    assert expected_component == "backend"


def test_resolve_dgd_service_trtllm_decode_uses_backend_name(
    kubernetes_connector, mock_kube_api
):
    """TRT-LLM decode: MDC carries "backend" (matches vLLM/SGLang); filter must match."""
    mock_deployment = _deployment(
        _component("TRTLLMDecodeWorker", "decode", replicas=1)
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    _, expected_component = kubernetes_connector._resolve_dgd_service(
        SubComponentType.DECODE, backend="trtllm"
    )

    assert expected_component == "backend"


def test_resolve_dgd_service_missing_dgd_still_returns_backend_default(
    kubernetes_connector, mock_kube_api
):
    """When DGD lookup fails, still return the backend default for filtering."""
    mock_kube_api.get_graph_deployment.side_effect = DynamoGraphDeploymentNotFoundError(
        "test-graph", "default"
    )

    dgd_service_name, expected_component = kubernetes_connector._resolve_dgd_service(
        SubComponentType.PREFILL, backend="vllm"
    )

    assert dgd_service_name is None
    assert expected_component == "prefill"


def test_resolve_dgd_service_respects_user_endpoint_override(
    kubernetes_connector, mock_kube_api
):
    """If the DGD passes --endpoint ns.comp.ep, the MDC filter must use 'comp'."""
    mock_deployment = _deployment(
        _component(
            "prefill",
            component_type="prefill",
            replicas=1,
            args=[
                "--endpoint",
                "my-ns.my-custom-prefill.generate",
                "--model",
                "Qwen/Qwen3-8B",
            ],
        )
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    dgd_service_name, expected_component = kubernetes_connector._resolve_dgd_service(
        SubComponentType.PREFILL, backend="vllm"
    )

    # k8s operations still target the DGD services key.
    assert dgd_service_name == "prefill"
    # Filter must match what the worker will actually write to MDC, which
    # comes from the user's --endpoint override, not the backend default.
    assert expected_component == "my-custom-prefill"


def test_resolve_dgd_service_endpoint_override_with_dyn_prefix(
    kubernetes_connector, mock_kube_api
):
    """parse_endpoint accepts 'dyn://' prefix; the extracted component must strip it."""
    mock_deployment = _deployment(
        _component(
            "decode",
            component_type="decode",
            replicas=1,
            args=[
                "--endpoint",
                "dyn://ns.user-decode.generate",
            ],
        )
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    _, expected_component = kubernetes_connector._resolve_dgd_service(
        SubComponentType.DECODE, backend="vllm"
    )

    assert expected_component == "user-decode"


def test_resolve_dgd_service_malformed_endpoint_falls_back_to_default(
    kubernetes_connector, mock_kube_api
):
    """Malformed --endpoint (wrong number of parts) falls back to backend default."""
    mock_deployment = _deployment(
        _component(
            "prefill",
            component_type="prefill",
            replicas=1,
            args=["--endpoint", "only-two.parts"],
        )
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    _, expected_component = kubernetes_connector._resolve_dgd_service(
        SubComponentType.PREFILL, backend="vllm"
    )

    assert expected_component == "prefill"


@pytest.mark.asyncio
async def test_wait_for_deployment_ready_does_not_require_backing(kubernetes_connector):
    """Production power-off path must keep the legacy readiness contract."""
    with patch.object(
        kubernetes_connector.kube_api,
        "wait_for_graph_deployment_ready",
        new_callable=AsyncMock,
    ) as wait:
        await kubernetes_connector.wait_for_deployment_ready(include_planner=False)
    wait.assert_awaited_once_with(
        kubernetes_connector.graph_deployment_name,
        include_planner=False,
        require_backing_settled=False,
    )


@pytest.mark.asyncio
async def test_wait_for_settled_graph_deployment_requires_backing(kubernetes_connector):
    """Power settlement path must opt into generation + backing gates."""
    with patch.object(
        kubernetes_connector.kube_api,
        "wait_for_graph_deployment_ready",
        new_callable=AsyncMock,
        return_value={"metadata": {"name": "dgd"}},
    ) as wait:
        got = await kubernetes_connector.wait_for_settled_graph_deployment(
            include_planner=False,
            require_prefill=False,
            require_decode=True,
            decode_component_name="CustomDecode",
        )
    assert got == {"metadata": {"name": "dgd"}}
    wait.assert_awaited_once_with(
        kubernetes_connector.graph_deployment_name,
        include_planner=False,
        require_backing_settled=True,
        require_prefill=False,
        require_decode=True,
        prefill_component_name=None,
        decode_component_name="CustomDecode",
    )
