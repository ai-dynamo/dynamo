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
    GPUShapeUnavailableError,
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
    # Default: no terminating pods; tests that want to simulate terminating pods
    # override this per-test.
    mock_api.has_terminating_pods = Mock(return_value=False)
    mock_api.list_pods_for_graph = Mock(return_value=[])
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


def _deployment(*components):
    return {
        "metadata": {"name": "test-graph"},
        "spec": {"components": list(components)},
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
    """An invalid scalar GPU count fails before legacy shape fallback."""
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
def test_missing_shape_falls_back_only_for_zero_gpu_sidecar(
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

    if sidecar_gpu == 0:
        assert kubernetes_connector.get_gpu_counts(
            require_prefill=False, require_decode=True
        ) == (0, 4)
    else:
        with pytest.raises(GPUShapeUnavailableError, match="auxiliary-GPU"):
            kubernetes_connector.get_gpu_shapes(
                require_prefill=False, require_decode=True
            )


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

    with pytest.raises(GPUShapeUnavailableError, match="DRA"):
        kubernetes_connector.get_gpu_shapes(require_prefill=False, require_decode=True)


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
    """Test get_gpu_counts raises DeploymentValidationError when GPU count missing"""
    mock_deployment = _deployment(
        _component("prefill-worker", "prefill", replicas=1),
        _component("decode-worker", "decode", replicas=1, gpu=4),
    )
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    with pytest.raises(DeploymentValidationError) as exc_info:
        kubernetes_connector.get_gpu_counts()

    assert "prefill GPU shape" in str(exc_info.value)


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
# name (what the Rust runtime writes to MDC), NOT the DGD component name.
# The DGD component name is typically PascalCase (``VllmPrefillWorker``)
# while MDC carries the Endpoint name (``prefill`` / ``backend``);
# returning the DGD component name for the filter would cause every real-world MDC
# entry to be skipped, leaving WorkerInfo without ``context_length`` and
# silently breaking easy-mode load scaling.


@pytest.mark.parametrize("pod_suffix", ["", "-f4k85"])
def test_extract_mdc_entries_uses_truncated_grove_component_name(
    kubernetes_connector, mock_kube_api, pod_suffix
):
    dgd_name = "live-verify-accept-len-win-df9e"
    component_name = "live-verify-accept-len--473a-0-vllmdecodeworker"
    cr_name = f"{component_name}{pod_suffix}"
    deployment = _deployment(_component("VllmDecodeWorker", "decode", replicas=1))
    deployment["metadata"]["name"] = dgd_name
    deployment["status"] = {
        "components": {
            "VllmDecodeWorker": {"componentNames": [component_name]},
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
    deployment = _deployment(_component("VllmDecodeWorker", "decode", replicas=1))
    deployment["status"] = {
        "components": {
            "Frontend": {"componentNames": ["test-graph-0-frontend"]},
        }
    }
    mock_kube_api.get_graph_deployment.return_value = deployment
    kubernetes_connector._list_worker_metadata_crs = Mock(
        return_value=[_model_card_cr("test-graph-0-vllmdecodeworker-f4k85")]
    )

    entries = kubernetes_connector._extract_mdc_entries()

    assert len(entries) == 1
    assert entries[0].card_json["worker_type"] == "decode"


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
