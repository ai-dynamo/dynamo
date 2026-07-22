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
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.base import PlannerConnector
from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.errors import (
    DeploymentModelNameMismatchError,
    DeploymentValidationError,
    DuplicateSubComponentError,
    DynamoGraphDeploymentNotFoundError,
    DynamoGraphDeploymentNotReadyError,
    EmptyTargetReplicasError,
    ModelNameNotFoundError,
    PlannerError,
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
    mock_api.update_graph_replicas = AsyncMock()
    mock_api.wait_for_graph_deployment_ready = AsyncMock()
    mock_api.is_deployment_ready = Mock()
    mock_api.custom_api.list_namespaced_custom_object.return_value = {"items": []}
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


def _deployment(*components):
    return {
        "metadata": {"name": "test-graph"},
        "spec": {"components": list(components)},
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
                    "name": "VllmPrefillWorker",
                    "replicas": 2,
                    "type": "prefill",
                },
                {
                    "name": "VllmDecodeWorker",
                    "replicas": 3,
                    "type": "decode",
                },
            ]
        },
    }

    service = get_component_from_type_or_name(deployment, SubComponentType.PREFILL)
    assert service.name == "VllmPrefillWorker"
    assert service.number_replicas() == 2

    service = get_component_from_type_or_name(deployment, SubComponentType.DECODE)
    assert service.name == "VllmDecodeWorker"
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
    deployment = _deployment(_component("VllmDecodeWorker", "worker", replicas=2))

    service = get_component_from_type_or_name(
        deployment, SubComponentType.DECODE, "VllmWorker"
    )

    assert service.name == "VllmDecodeWorker"
    assert service.number_replicas() == 2


def test_get_service_name_from_unique_worker_does_not_satisfy_prefill(
    kubernetes_connector,
):
    deployment = _deployment(_component("VllmDecodeWorker", "worker", replicas=2))

    with pytest.raises(SubComponentNotFoundError):
        get_component_from_type_or_name(deployment, SubComponentType.PREFILL)


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
    mock_kube_api.update_graph_replicas.assert_called_once_with(
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
    mock_kube_api.update_graph_replicas.assert_called_once_with(
        "test-graph", component_name, 1
    )
    mock_kube_api.wait_for_graph_deployment_ready.assert_called_once_with("test-graph")


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
    mock_kube_api.update_graph_replicas.assert_called_once_with(
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
    mock_kube_api.update_graph_replicas.assert_has_calls(expected_calls, any_order=True)
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
    mock_kube_api.update_graph_replicas.assert_called_once_with(
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
            "VllmPrefillWorker",
            replicas=1,
            args=["--served-model-name", "test-model"],
        ),
        _component(
            "VllmDecodeWorker",
            replicas=1,
            args=["--served-model-name", "test-model"],
        ),
    )

    await kubernetes_connector.validate_deployment(
        prefill_component_name="VllmPrefillWorker",
        decode_component_name="VllmDecodeWorker",
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


def test_get_model_name_forwards_component_name_overrides(
    kubernetes_connector, mock_kube_api
):
    """Explicit component-name overrides are forwarded to the deployment lookup.

    For aggregated or multi-generic-worker DGDs, callers pass explicit names to
    disambiguate which component to read the model name from.  Dropping them
    causes a silent fallback to role-based lookup that returns the wrong result.
    """
    mock_kube_api.get_graph_deployment.return_value = _deployment(
        _component(
            "custom-prefill",
            "worker",
            replicas=1,
            args=["--served-model-name", "my-model"],
        ),
        _component(
            "custom-decode",
            "worker",
            replicas=1,
            args=["--served-model-name", "my-model"],
        ),
    )

    result = kubernetes_connector.get_model_name(
        require_prefill=True,
        require_decode=True,
        prefill_component_name="custom-prefill",
        decode_component_name="custom-decode",
    )

    assert result == "my-model"


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


# Tests for Service.get_gpu_count()
def test_service_get_gpu_count_valid():
    """Test that get_gpu_count returns GPU count from main container limits."""
    service = Service(
        name="test-service",
        service=_component("test-service", replicas=1, gpu=4),
    )
    assert service.get_gpu_count() == 4


def test_service_get_gpu_count_from_requests_fallback():
    """Test that get_gpu_count falls back to main container requests."""
    service = Service(
        name="test-service",
        service={
            "replicas": 1,
            "podTemplate": {
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                            "resources": {"requests": {"nvidia.com/gpu": "2"}},
                        }
                    ]
                }
            },
        },
    )
    assert service.get_gpu_count() == 2


def test_service_get_gpu_count_limits_preferred_over_requests():
    """Test that limits are preferred over requests when both are present."""
    service = Service(
        name="test-service",
        service={
            "replicas": 1,
            "podTemplate": {
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                            "resources": {
                                "limits": {"nvidia.com/gpu": "4"},
                                "requests": {"nvidia.com/gpu": "2"},
                            },
                        }
                    ]
                }
            },
        },
    )
    assert service.get_gpu_count() == 4


def test_service_get_gpu_count_integer_value():
    """Test that get_gpu_count works with integer GPU values"""
    service = Service(
        name="test-service",
        service={
            "replicas": 1,
            "podTemplate": {
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                            "resources": {"limits": {"nvidia.com/gpu": 2}},
                        }
                    ]
                }
            },
        },
    )
    assert service.get_gpu_count() == 2


def test_service_get_gpu_count_missing_raises_error():
    """Test that get_gpu_count raises ValueError when GPU count is missing"""
    service = Service(
        name="test-service",
        service={"replicas": 1},
    )
    with pytest.raises(ValueError) as exc_info:
        service.get_gpu_count()
    assert "No GPU count specified" in str(exc_info.value)
    assert "test-service" in str(exc_info.value)


def test_service_get_gpu_count_invalid_raises_error():
    """Test that get_gpu_count raises ValueError for invalid GPU count"""
    service = Service(
        name="test-service",
        service={
            "replicas": 1,
            "podTemplate": {
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                            "resources": {"limits": {"nvidia.com/gpu": "invalid"}},
                        }
                    ]
                }
            },
        },
    )
    with pytest.raises(ValueError) as exc_info:
        service.get_gpu_count()
    assert "Invalid GPU count" in str(exc_info.value)


def test_service_reads_v1beta_pod_template_main_container():
    service = Service(
        name="VllmPrefillWorker",
        service={
            "podTemplate": {
                "spec": {
                    "containers": [
                        {
                            "name": "sidecar",
                            "args": ["--ignored"],
                        },
                        {
                            "name": "main",
                            "args": [
                                "--endpoint",
                                "ns.custom-prefill.generate",
                                "--model",
                                "Qwen/Qwen3-8B",
                            ],
                            "resources": {
                                "limits": {
                                    "nvidia.com/gpu": "2",
                                }
                            },
                        },
                    ]
                }
            }
        },
    )

    assert service.get_model_name() == "Qwen/Qwen3-8B"
    assert service.get_component_name_from_endpoint_arg() == "custom-prefill"
    assert service.get_gpu_count() == 2


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


def test_get_gpu_counts_missing_gpu_raises_error(kubernetes_connector, mock_kube_api):
    """Test get_gpu_counts raises DeploymentValidationError when GPU count missing"""
    mock_deployment = _deployment(
        _component("prefill-worker", "prefill", replicas=1),
        _component("decode-worker", "decode", replicas=1, gpu=4),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    with pytest.raises(DeploymentValidationError) as exc_info:
        kubernetes_connector.get_gpu_counts()

    assert "prefill GPU count" in str(exc_info.value)


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

    assert "decode GPU count" in str(exc_info.value)


def test_get_gpu_counts_multinode_per_pod_not_replica_total(
    kubernetes_connector, mock_kube_api
):
    """get_gpu_counts stores per-pod GPU request for budget/perf-model paths."""
    multinode_decode = {
        "name": "decode-worker",
        "type": "decode",
        "replicas": 1,
        "multinode": {"nodeCount": 4},
        "podTemplate": {
            "spec": {
                "containers": [
                    {
                        "name": "main",
                        "resources": {"limits": {"nvidia.com/gpu": "8"}},
                    }
                ]
            }
        },
    }
    mock_deployment = _deployment(
        _component("prefill-worker", "prefill", replicas=1, gpu=2),
        multinode_decode,
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    prefill_gpu, decode_gpu = kubernetes_connector.get_gpu_counts()

    assert prefill_gpu == 2
    assert decode_gpu == 8


def test_replica_gpu_counts_for_power_projection_multinode(
    kubernetes_connector, mock_kube_api
):
    """Power projection multiplies per-pod GPUs by multinode.nodeCount."""
    multinode_decode = {
        "name": "decode-worker",
        "type": "decode",
        "replicas": 1,
        "multinode": {"nodeCount": 4},
        "podTemplate": {
            "spec": {
                "containers": [
                    {
                        "name": "main",
                        "resources": {"limits": {"nvidia.com/gpu": "8"}},
                    }
                ]
            }
        },
    }
    mock_deployment = _deployment(
        _component("prefill-worker", "prefill", replicas=1, gpu=2),
        multinode_decode,
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    (
        prefill_gpu,
        decode_gpu,
    ) = kubernetes_connector.get_replica_gpu_counts_for_power_projection()

    assert prefill_gpu == 2
    assert decode_gpu == 32


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


# Tests for _resolve_dgd_service / get_worker_info component-filter.
#
# Regression: the filter that compares an MDC entry's ``component`` field
# against ``expected_component`` must use the lowercase backend-default
# name (what the Rust runtime writes to MDC), NOT the DGD component name.
# The DGD component name is typically PascalCase (``VllmPrefillWorker``)
# while MDC carries the Endpoint name (``prefill`` / ``backend``);
# returning the DGD component name for the filter would cause every real-world MDC
# entry to be skipped, leaving WorkerInfo without ``context_length`` and
# silently breaking easy-mode load scaling.


def test_resolve_dgd_service_prefill_uses_backend_default_for_filter(
    kubernetes_connector, mock_kube_api
):
    """vLLM prefill: filter name = "prefill" (MDC side), not DGD component name."""
    mock_deployment = _deployment(
        _component("VllmPrefillWorker", "prefill", replicas=1)
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    dgd_service_name, expected_component = kubernetes_connector._resolve_dgd_service(
        SubComponentType.PREFILL, backend="vllm"
    )

    # k8s operations (e.g. replica patch) still target the PascalCase DGD component.
    assert dgd_service_name == "VllmPrefillWorker"
    # The filter side must match what the Rust runtime writes to MDC.
    assert expected_component == "prefill"


def test_resolve_dgd_service_v1beta_endpoint_override(
    kubernetes_connector, mock_kube_api
):
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "components": [
                {
                    "name": "VllmPrefillWorker",
                    "replicas": 1,
                    "type": "prefill",
                    "podTemplate": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "args": [
                                        "--endpoint",
                                        "my-ns.my-custom-prefill.generate",
                                        "--model",
                                        "Qwen/Qwen3-8B",
                                    ],
                                }
                            ]
                        }
                    },
                },
            ]
        },
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    dgd_service_name, expected_component = kubernetes_connector._resolve_dgd_service(
        SubComponentType.PREFILL, backend="vllm"
    )

    assert dgd_service_name == "VllmPrefillWorker"
    assert expected_component == "my-custom-prefill"


def test_resolve_dgd_service_decode_uses_backend_default_for_filter(
    kubernetes_connector, mock_kube_api
):
    """vLLM decode: MDC carries "backend", NOT "decode"; filter must match that."""
    mock_deployment = _deployment(_component("VllmDecodeWorker", "decode", replicas=1))
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    dgd_service_name, expected_component = kubernetes_connector._resolve_dgd_service(
        SubComponentType.DECODE, backend="vllm"
    )

    assert dgd_service_name == "VllmDecodeWorker"
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
            "VllmPrefillWorker",
            "prefill",
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

    # k8s operations still target the DGD component name.
    assert dgd_service_name == "VllmPrefillWorker"
    # Filter must match what the worker will actually write to MDC, which
    # comes from the user's --endpoint override, not the backend default.
    assert expected_component == "my-custom-prefill"


def test_resolve_dgd_service_endpoint_override_with_dyn_prefix(
    kubernetes_connector, mock_kube_api
):
    """parse_endpoint accepts 'dyn://' prefix; the extracted component must strip it."""
    mock_deployment = _deployment(
        _component(
            "VllmDecodeWorker",
            "decode",
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
            "VllmPrefillWorker",
            "prefill",
            replicas=1,
            args=["--endpoint", "only-two.parts"],
        )
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    _, expected_component = kubernetes_connector._resolve_dgd_service(
        SubComponentType.PREFILL, backend="vllm"
    )

    assert expected_component == "prefill"


def test_service_get_component_name_from_endpoint_arg_present():
    service = Service(
        name="VllmPrefillWorker",
        service=_component(
            "VllmPrefillWorker",
            args=[
                "--endpoint",
                "ns.custom-comp.generate",
                "--other",
                "flag",
            ],
        ),
    )
    assert service.get_component_name_from_endpoint_arg() == "custom-comp"


def test_service_get_component_name_from_endpoint_arg_absent():
    service = Service(
        name="VllmPrefillWorker",
        service=_component(
            "VllmPrefillWorker",
            args=["--model", "Qwen/Qwen3-8B"],
        ),
    )
    assert service.get_component_name_from_endpoint_arg() is None


def test_service_get_component_name_from_endpoint_arg_missing_value():
    """--endpoint with no following arg should return None, not raise IndexError."""
    service = Service(
        name="VllmPrefillWorker",
        service=_component("VllmPrefillWorker", args=["--endpoint"]),
    )
    assert service.get_component_name_from_endpoint_arg() is None


# ---------------------------------------------------------------------------
# resolve_frontend_http_port — read the frontend HTTP port from the live pod
# ---------------------------------------------------------------------------


def _make_pod_with_ports(port_specs):
    """Build a duck-typed V1Pod with one container whose ports == port_specs.

    Each ``port_specs`` element is either ``None`` (no ports list at all) or
    a list of ``(name, container_port)`` tuples.  We keep the fixture
    explicit so tests don't accidentally pass through MagicMock auto-spec
    fluff and silently match on something other than what they meant to.
    """
    pod = Mock()
    container = Mock()
    if port_specs is None:
        container.ports = None
    else:
        ports = []
        for name, container_port in port_specs:
            p = Mock()
            p.name = name
            p.container_port = container_port
            ports.append(p)
        container.ports = ports
    pod.spec.containers = [container]
    return pod


class TestResolveFrontendHttpPort:
    """Pin :meth:`KubernetesConnector.resolve_frontend_http_port` behavior.

    The planner reads the frontend's actual HTTP port from
    V1Pod.spec.containers[].ports[name=http] and only falls back to the legacy
    config field when the named port is absent.
    """

    def test_returns_named_http_port_when_present(self):
        """Operator-emitted pod with ``http`` named port: resolver wins."""
        pod = _make_pod_with_ports([("http", 8000)])
        assert (
            KubernetesConnector.resolve_frontend_http_port(pod, fallback=9999) == 8000
        )

    def test_falls_back_when_no_named_http_port(self):
        """Legacy/hand-rolled pod with named ports that aren't ``http``."""
        pod = _make_pod_with_ports([("metrics", 9090), ("admin", 19090)])
        assert (
            KubernetesConnector.resolve_frontend_http_port(pod, fallback=8000) == 8000
        )

    def test_falls_back_when_ports_list_is_none(self):
        """Pod spec missing ``ports`` entirely (the kubernetes client returns
        ``None`` rather than ``[]`` when the field is unset)."""
        pod = _make_pod_with_ports(None)
        assert (
            KubernetesConnector.resolve_frontend_http_port(pod, fallback=8000) == 8000
        )

    def test_falls_back_when_ports_list_is_empty(self):
        pod = _make_pod_with_ports([])
        assert (
            KubernetesConnector.resolve_frontend_http_port(pod, fallback=8000) == 8000
        )

    def test_picks_http_among_multiple_named_ports(self):
        """Multi-port containers must filter by name rather than position —
        the operator may add extra named ports (e.g. ``debug``) without
        breaking discovery."""
        pod = _make_pod_with_ports([("metrics", 9090), ("http", 8001), ("debug", 6060)])
        assert (
            KubernetesConnector.resolve_frontend_http_port(pod, fallback=8000) == 8001
        )

    def test_honors_operator_port_override(self):
        """If the DGD overrides the frontend container port, the resolver
        should follow the live spec rather than mirror the config field."""
        pod = _make_pod_with_ports([("http", 8443)])
        assert (
            KubernetesConnector.resolve_frontend_http_port(pod, fallback=8000) == 8443
        )

    def test_string_container_port_is_coerced_to_int(self):
        """Defensive: the kubernetes client typically yields ``int`` for
        ``container_port`` but YAML hand-rolls can sneak in strings — the
        resolver must coerce, not crash."""
        pod = _make_pod_with_ports([("http", "8000")])
        assert (
            KubernetesConnector.resolve_frontend_http_port(pod, fallback=9999) == 8000
        )

    def test_falls_back_on_malformed_pod_without_raising(self):
        """A malformed pod object (e.g. ``spec is None``) must not bring
        down the admission-control loop."""
        pod = Mock()
        pod.spec = None
        assert (
            KubernetesConnector.resolve_frontend_http_port(pod, fallback=8000) == 8000
        )

    def test_falls_back_when_container_ports_attribute_missing(self):
        """Pod object whose container has no ``ports`` attribute at all
        (some test stubs forget to set it)."""
        pod = Mock()
        container = Mock(spec=[])  # no attributes at all
        pod.spec.containers = [container]
        assert (
            KubernetesConnector.resolve_frontend_http_port(pod, fallback=8000) == 8000
        )

    def test_first_container_with_http_wins_in_multi_container_pod(self):
        """If a pod has multiple containers (sidecars) and only one has the
        ``http`` port, the resolver should still find it."""
        pod = Mock()
        sidecar = Mock()
        sidecar_port = Mock()
        sidecar_port.name = "metrics"
        sidecar_port.container_port = 9090
        sidecar.ports = [sidecar_port]

        main = Mock()
        main_port = Mock()
        main_port.name = "http"
        main_port.container_port = 8000
        main.ports = [main_port]

        pod.spec.containers = [sidecar, main]
        assert (
            KubernetesConnector.resolve_frontend_http_port(pod, fallback=9999) == 8000
        )


# ---------------------------------------------------------------------------
# Aggregated (mode=agg) component resolution
# ---------------------------------------------------------------------------


def _agg_deployment(model_name="test-model", gpu=4, worker_name="VllmDecodeWorker"):
    """DGD with a single generic type:worker component — the agg topology."""
    return _deployment(
        _component(
            worker_name,
            "worker",
            replicas=1,
            args=["--served-model-name", model_name],
            gpu=gpu,
        )
    )


@pytest.mark.asyncio
async def test_validate_deployment_agg_mode(kubernetes_connector, mock_kube_api):
    """validate_deployment must resolve the actual generic worker name in agg mode."""
    mock_kube_api.get_graph_deployment.return_value = _agg_deployment()

    # Must not raise — both the component-existence check and model-name lookup
    # must resolve the unique generic type:worker component from the DGD.
    await kubernetes_connector.validate_deployment(
        prefill_component_name=None,
        decode_component_name="VllmWorker",
        require_prefill=False,
        require_decode=True,
    )


@pytest.mark.asyncio
async def test_validate_deployment_agg_mode_explicit_decode_worker_name(
    kubernetes_connector, mock_kube_api
):
    """validate_deployment must also work when the hint is the real agg name."""
    mock_kube_api.get_graph_deployment.return_value = _agg_deployment()

    await kubernetes_connector.validate_deployment(
        prefill_component_name=None,
        decode_component_name="VllmDecodeWorker",
        require_prefill=False,
        require_decode=True,
    )


def test_get_gpu_counts_agg_mode(kubernetes_connector, mock_kube_api):
    """get_gpu_counts must return the GPU count from the agg worker, not fall back to CLI."""
    mock_kube_api.get_graph_deployment.return_value = _agg_deployment(gpu=4)

    prefill_gpu, decode_gpu = kubernetes_connector.get_gpu_counts(
        require_prefill=False,
        require_decode=True,
        decode_component_name="VllmWorker",
    )

    assert prefill_gpu == 0
    assert decode_gpu == 4


def test_get_worker_info_agg_mode_sets_k8s_name(kubernetes_connector, mock_kube_api):
    """get_worker_info must set k8s_name to the actual generic worker name."""
    mock_kube_api.get_graph_deployment.return_value = _agg_deployment()

    info = kubernetes_connector.get_worker_info(
        sub_component_type=SubComponentType.DECODE,
        backend="vllm",
        component_name="VllmWorker",
    )

    assert info.k8s_name == "VllmDecodeWorker"


# Static fallback path (connector=None) — covers build_worker_info_from_defaults
def test_resolve_worker_info_static_fallback_respects_decode_component_name():
    """resolve_worker_info with connector=None must honour decode_component_name."""
    from dynamo.planner.monitoring.worker_info import resolve_worker_info

    _, decode_info = resolve_worker_info(
        backend="vllm",
        require_prefill=False,
        require_decode=True,
        connector=None,
        config_model_name="some-model",
        decode_component_name="VllmWorker",
    )

    assert decode_info.k8s_name == "VllmWorker"


def test_initialize_gpu_counts_agg_mode_reads_from_dgd(
    kubernetes_connector, mock_kube_api
):
    """_initialize_gpu_counts must set decode_engine_num_gpu from DGD in agg mode
    without requiring a CLI flag."""
    from dynamo.planner.core.budget import _initialize_gpu_counts

    mock_kube_api.get_graph_deployment.return_value = _agg_deployment(gpu=4)

    config = Mock()
    config.prefill_engine_num_gpu = None
    config.decode_engine_num_gpu = None

    _initialize_gpu_counts(
        config,
        kubernetes_connector,
        require_prefill=False,
        require_decode=True,
        decode_component_name="VllmWorker",
    )

    assert config.decode_engine_num_gpu == 4


def test_get_worker_info_agg_mode_mdc_present_dgd_down(
    kubernetes_connector, mock_kube_api
):
    """k8s_name must remain 'VllmWorker' when MDC is populated but DGD is temporarily down.

    Pins the `dgd_service_name or component_name` guard in get_worker_info: when
    _resolve_dgd_service returns None (DGD fetch fails), the explicit component_name
    parameter must be used as the k8s_name_override so the agg worker is still
    identified correctly in subsequent scaling calls.
    """
    mdc_cr = {
        "metadata": {"name": "test-graph-0-VllmWorker-abc123"},
        "spec": {
            "data": {
                "model_cards": {
                    "model": {
                        "type": "Model",
                        "component": "backend",
                        "endpoint": "generate",
                        "card_json": {"display_name": "test-model"},
                    }
                }
            }
        },
    }
    mock_kube_api.custom_api.list_namespaced_custom_object.return_value = {
        "items": [mdc_cr]
    }
    # Simulate DGD temporarily unavailable — _resolve_dgd_service must fall back
    # to component_name so k8s_name is preserved.
    mock_kube_api.get_graph_deployment.side_effect = DynamoGraphDeploymentNotFoundError(
        "test-graph", "default"
    )

    info = kubernetes_connector.get_worker_info(
        sub_component_type=SubComponentType.DECODE,
        backend="vllm",
        component_name="VllmWorker",
    )

    assert info.k8s_name == "VllmWorker"
