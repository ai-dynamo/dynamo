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

from dynamo.planner.defaults import (
    Service,
    SubComponentType,
    get_service_from_sub_component_type_or_name,
)
from dynamo.planner.kubernetes_connector import KubernetesConnector, TargetReplica
from dynamo.planner.utils.exceptions import (
    DeploymentModelNameMismatchError,
    DeploymentValidationError,
    DuplicateSubComponentError,
    DynamoGraphDeploymentNotFoundError,
    EmptyTargetReplicasError,
    ModelNameNotFoundError,
    SubComponentNotFoundError,
)


@pytest.fixture
def mock_kube_api():
    mock_api = Mock()
    mock_api.get_graph_deployment = Mock()
    mock_api.update_graph_replicas = AsyncMock()
    mock_api.wait_for_graph_deployment_ready = AsyncMock()
    mock_api.is_deployment_ready = Mock()
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
        "dynamo.planner.kubernetes_connector.KubernetesAPI", mock_kube_api_class
    )
    with patch.dict(os.environ, {"DYN_PARENT_DGD_K8S_NAME": "test-graph"}):
        connector = KubernetesConnector("test-dynamo-namespace")
        return connector


def test_kubernetes_connector_no_env_var():
    with pytest.raises(DeploymentValidationError) as exc_info:
        KubernetesConnector("test-dynamo-namespace")

    exception = exc_info.value
    assert set(exception.errors) == {
        "DYN_PARENT_DGD_K8S_NAME environment variable is not set"
    }


def test_get_service_name_from_sub_component_type(kubernetes_connector):
    deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "test-component-prefill": {
                    "replicas": 2,
                    "subComponentType": "prefill",
                },
                "test-component-decode": {"replicas": 3, "subComponentType": "decode"},
            }
        },
    }

    service = get_service_from_sub_component_type_or_name(
        deployment, SubComponentType.PREFILL
    )
    assert service.name == "test-component-prefill"
    assert service.number_replicas() == 2

    # should still work if the component_name is provided
    service = get_service_from_sub_component_type_or_name(
        deployment, SubComponentType.PREFILL, "test-component-prefill"
    )
    assert service.name == "test-component-prefill"
    assert service.number_replicas() == 2

    # should respect subComponentType first
    service = get_service_from_sub_component_type_or_name(
        deployment, SubComponentType.DECODE, "test-component-prefill"
    )
    assert service.name == "test-component-decode"
    assert service.number_replicas() == 3


def test_get_service_name_from_sub_component_type_not_found(kubernetes_connector):
    deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "test-component-decode": {"replicas": 3, "subComponentType": "decode"},
            }
        },
    }
    with pytest.raises(SubComponentNotFoundError) as exc_info:
        get_service_from_sub_component_type_or_name(
            deployment, SubComponentType.PREFILL
        )

    with pytest.raises(SubComponentNotFoundError) as exc_info:
        get_service_from_sub_component_type_or_name(
            deployment, SubComponentType.PREFILL, "test-component-decode"
        )

    exception = exc_info.value
    assert exception.sub_component_type == SubComponentType.PREFILL.value


def test_get_service_name_from_sub_component_type_duplicate(kubernetes_connector):
    deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "test-component-prefill": {
                    "replicas": 2,
                    "subComponentType": "prefill",
                },
                "test-component-prefill-2": {
                    "replicas": 3,
                    "subComponentType": "prefill",
                },
            }
        },
    }

    with pytest.raises(DuplicateSubComponentError) as exc_info:
        # even though "test-component-prefill" is provided, subComponentType duplicates should result in an error
        get_service_from_sub_component_type_or_name(
            deployment, SubComponentType.PREFILL, "test-component-prefill"
        )

    exception = exc_info.value
    assert exception.sub_component_type == SubComponentType.PREFILL.value
    assert set(exception.service_names) == {
        "test-component-prefill",
        "test-component-prefill-2",
    }


def test_get_service_name_from_sub_component_type_or_name(kubernetes_connector):
    deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "test-component-prefill": {"replicas": 2},
                "test-component-decode": {"replicas": 3},
            }
        },
    }

    service = get_service_from_sub_component_type_or_name(
        deployment, SubComponentType.PREFILL, "test-component-prefill"
    )
    assert service.name == "test-component-prefill"
    assert service.number_replicas() == 2


@pytest.mark.asyncio
async def test_add_component_increases_replicas(kubernetes_connector, mock_kube_api):
    # Arrange
    sub_component_type = SubComponentType.PREFILL
    component_name = "test-component"
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                component_name: {
                    "replicas": 1,
                    "subComponentType": sub_component_type.value,
                }
            }
        },
    }
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
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {component_name: {"subComponentType": sub_component_type.value}}
        },
    }
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
        "spec": {"services": {"test-component": {"subComponentType": "decode"}}},
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
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "test-component": {
                    "replicas": 2,
                    "subComponentType": sub_component_type.value,
                }
            }
        },
    }
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
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                component_name: {
                    "replicas": 0,
                    "subComponentType": sub_component_type.value,
                }
            }
        },
    }
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
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                component_name: {
                    "replicas": 0,
                    "subComponentType": sub_component_type.value,
                }
            }
        },
    }
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
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "component1": {"replicas": 1, "subComponentType": "prefill"},
                "component2": {"replicas": 1},
            }
        },
    }
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
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "component1": {"replicas": 1, "subComponentType": "prefill"},
                "component2": {"replicas": 1},
            }
        },
    }
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
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "component1": {"replicas": 1, "subComponentType": "prefill"},
                "component2": {"replicas": 2, "subComponentType": "decode"},
            }
        },
    }
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


async def test_set_component_replicas_deployment_not_ready(
    kubernetes_connector, mock_kube_api
):
    # Arrange
    target_replicas = [
        TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=3),
        TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=2),
    ]
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "component1": {"replicas": 1, "subComponentType": "prefill"},
                "component2": {"replicas": 2, "subComponentType": "decode"},
            }
        },
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.is_deployment_ready.return_value = False

    # Act & Assert
    await kubernetes_connector.set_component_replicas(target_replicas)

    mock_kube_api.get_graph_deployment.assert_called_once()
    mock_kube_api.is_deployment_ready.assert_called_once_with(mock_deployment)
    mock_kube_api.update_graph_replicas.assert_not_called()
    mock_kube_api.wait_for_graph_deployment_ready.assert_not_called()


@pytest.mark.asyncio
async def test_validate_deployment_true(kubernetes_connector, mock_kube_api):
    # Arrange
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "component1": {
                    "replicas": 1,
                    "subComponentType": "prefill",
                    "extraPodSpec": {
                        "mainContainer": {
                            "args": ["--served-model-name", "prefill-model"]
                        }
                    },
                },
                "component2": {"replicas": 2, "subComponentType": "decode"},
            }
        },
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    # Act
    await kubernetes_connector.validate_deployment(decode_component_name="component2")


@pytest.mark.asyncio
async def test_validate_deployment_fail(kubernetes_connector, mock_kube_api):
    # Arrange
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "component1": {"replicas": 1, "subComponentType": "prefill"},
                "component2": {"replicas": 2, "subComponentType": "prefill"},
            }
        },
    }
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
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "component1": {"replicas": 1, "subComponentType": "prefill"},
                "component2": {"replicas": 2, "subComponentType": "decode"},
            }
        },
    }

    with pytest.raises(ModelNameNotFoundError):
        kubernetes_connector.get_model_name(mock_deployment)


def test_get_model_name_prefill_none_decode_valid_returns_decode(kubernetes_connector):
    # Arrange
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "component1": {"replicas": 1, "subComponentType": "prefill"},
                "component2": {
                    "replicas": 2,
                    "subComponentType": "decode",
                    "extraPodSpec": {
                        "mainContainer": {"args": ["--served-model-name", "test-model"]}
                    },
                },
            }
        },
    }
    # Act
    result = kubernetes_connector.get_model_name(mock_deployment)

    # Assert
    assert result == "test-model"


def test_get_model_name_mismatch_raises_error(kubernetes_connector, mock_kube_api):
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "component1": {
                    "replicas": 1,
                    "subComponentType": "prefill",
                    "extraPodSpec": {
                        "mainContainer": {
                            "args": ["--served-model-name", "prefill-model"]
                        }
                    },
                },
                "component2": {
                    "replicas": 2,
                    "subComponentType": "decode",
                    "extraPodSpec": {
                        "mainContainer": {
                            "args": ["--served-model-name", "decode-model"]
                        }
                    },
                },
            }
        },
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    # Act & Assert
    with pytest.raises(DeploymentModelNameMismatchError) as exc_info:
        kubernetes_connector.get_model_name(mock_deployment)

    exception = exc_info.value
    assert exception.prefill_model_name == "prefill-model"
    assert exception.decode_model_name == "decode-model"


def test_get_model_name_agree_returns_model_name(kubernetes_connector, mock_kube_api):
    # Arrange
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "component1": {
                    "replicas": 1,
                    "subComponentType": "prefill",
                    "extraPodSpec": {
                        "mainContainer": {
                            "args": ["--served-model-name", "agreed-model"]
                        }
                    },
                },
                "component2": {
                    "replicas": 2,
                    "subComponentType": "decode",
                    "extraPodSpec": {
                        "mainContainer": {
                            "args": ["--served-model-name", "agreed-model"]
                        }
                    },
                },
            }
        },
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    # Act
    result = kubernetes_connector.get_model_name(mock_deployment)

    # Assert
    assert result == "agreed-model"


# Tests for Service.get_gpu_count()
def test_service_get_gpu_count_valid():
    """Test that get_gpu_count returns correct GPU count from resources.limits.gpu"""
    service = Service(
        name="test-service",
        service={
            "replicas": 1,
            "resources": {"limits": {"gpu": "4"}},
        },
    )
    assert service.get_gpu_count() == 4


def test_service_get_gpu_count_from_requests_fallback():
    """Test that get_gpu_count falls back to requests.gpu when limits.gpu is missing"""
    service = Service(
        name="test-service",
        service={
            "replicas": 1,
            "resources": {"requests": {"gpu": "2"}},
        },
    )
    assert service.get_gpu_count() == 2


def test_service_get_gpu_count_limits_preferred_over_requests():
    """Test that limits.gpu is preferred over requests.gpu when both are present"""
    service = Service(
        name="test-service",
        service={
            "replicas": 1,
            "resources": {
                "limits": {"gpu": "4"},
                "requests": {"gpu": "2"},
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
            "resources": {"limits": {"gpu": 2}},
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
            "resources": {"limits": {"gpu": "invalid"}},
        },
    )
    with pytest.raises(ValueError) as exc_info:
        service.get_gpu_count()
    assert "Invalid GPU count" in str(exc_info.value)


# Tests for KubernetesConnector.get_gpu_counts()
def test_get_gpu_counts_both_services(kubernetes_connector, mock_kube_api):
    """Test get_gpu_counts returns correct counts for both prefill and decode"""
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "prefill-worker": {
                    "replicas": 1,
                    "subComponentType": "prefill",
                    "resources": {"limits": {"gpu": "2"}},
                },
                "decode-worker": {
                    "replicas": 1,
                    "subComponentType": "decode",
                    "resources": {"limits": {"gpu": "4"}},
                },
            }
        },
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    prefill_gpu, decode_gpu = kubernetes_connector.get_gpu_counts()

    assert prefill_gpu == 2
    assert decode_gpu == 4


def test_get_gpu_counts_prefill_only(kubernetes_connector, mock_kube_api):
    """Test get_gpu_counts with require_decode=False"""
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "prefill-worker": {
                    "replicas": 1,
                    "subComponentType": "prefill",
                    "resources": {"limits": {"gpu": "2"}},
                },
            }
        },
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    prefill_gpu, decode_gpu = kubernetes_connector.get_gpu_counts(
        require_prefill=True, require_decode=False
    )

    assert prefill_gpu == 2
    assert decode_gpu == 0


def test_get_gpu_counts_decode_only(kubernetes_connector, mock_kube_api):
    """Test get_gpu_counts with require_prefill=False"""
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "decode-worker": {
                    "replicas": 1,
                    "subComponentType": "decode",
                    "resources": {"limits": {"gpu": "4"}},
                },
            }
        },
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    prefill_gpu, decode_gpu = kubernetes_connector.get_gpu_counts(
        require_prefill=False, require_decode=True
    )

    assert prefill_gpu == 0
    assert decode_gpu == 4


def test_get_gpu_counts_missing_gpu_raises_error(kubernetes_connector, mock_kube_api):
    """Test get_gpu_counts raises DeploymentValidationError when GPU count missing"""
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "prefill-worker": {
                    "replicas": 1,
                    "subComponentType": "prefill",
                    # No resources.limits.gpu
                },
                "decode-worker": {
                    "replicas": 1,
                    "subComponentType": "decode",
                    "resources": {"limits": {"gpu": "4"}},
                },
            }
        },
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    with pytest.raises(DeploymentValidationError) as exc_info:
        kubernetes_connector.get_gpu_counts()

    assert "prefill GPU count" in str(exc_info.value)


def test_get_gpu_counts_service_not_found_raises_error(
    kubernetes_connector, mock_kube_api
):
    """Test get_gpu_counts raises DeploymentValidationError when service not found"""
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "prefill-worker": {
                    "replicas": 1,
                    "subComponentType": "prefill",
                    "resources": {"limits": {"gpu": "2"}},
                },
                # No decode service
            }
        },
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    with pytest.raises(DeploymentValidationError) as exc_info:
        kubernetes_connector.get_gpu_counts()

    assert "decode GPU count" in str(exc_info.value)


# Tests for get_actual_worker_counts


def test_get_actual_worker_counts_stable(kubernetes_connector, mock_kube_api):
    """Test get_actual_worker_counts when both services are stable"""
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "prefill-component": {},
                "decode-component": {},
            }
        },
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.get_service_replica_status.side_effect = [(2, True), (4, True)]

    (
        prefill_count,
        decode_count,
        is_stable,
    ) = kubernetes_connector.get_actual_worker_counts(
        prefill_component_name="prefill-component",
        decode_component_name="decode-component",
    )

    assert prefill_count == 2
    assert decode_count == 4
    assert is_stable is True


def test_get_actual_worker_counts_prefill_rollout_in_progress(
    kubernetes_connector, mock_kube_api
):
    """Test get_actual_worker_counts when prefill has rollout in progress"""
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "prefill-component": {},
                "decode-component": {},
            }
        },
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.get_service_replica_status.side_effect = [(2, False), (4, True)]

    (
        prefill_count,
        decode_count,
        is_stable,
    ) = kubernetes_connector.get_actual_worker_counts(
        prefill_component_name="prefill-component",
        decode_component_name="decode-component",
    )

    assert prefill_count == 2
    assert decode_count == 4
    assert is_stable is False


def test_get_actual_worker_counts_prefill_only(kubernetes_connector, mock_kube_api):
    """Test get_actual_worker_counts with only prefill component"""
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "prefill-component": {
                    "replicas": 2,
                    "subComponentType": "prefill",
                },
            }
        },
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.get_service_replica_status.return_value = (2, True)

    (
        prefill_count,
        decode_count,
        is_stable,
    ) = kubernetes_connector.get_actual_worker_counts(
        prefill_component_name="prefill-component",
        decode_component_name=None,
    )

    assert prefill_count == 2
    assert decode_count == 0
    assert is_stable is True


def test_get_actual_worker_counts_decode_only(kubernetes_connector, mock_kube_api):
    """Test get_actual_worker_counts with only decode component"""
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {
            "services": {
                "decode-component": {
                    "replicas": 4,
                    "subComponentType": "decode",
                },
            }
        },
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment
    mock_kube_api.get_service_replica_status.return_value = (4, True)

    (
        prefill_count,
        decode_count,
        is_stable,
    ) = kubernetes_connector.get_actual_worker_counts(
        prefill_component_name=None,
        decode_component_name="decode-component",
    )

    assert prefill_count == 0
    assert decode_count == 4
    assert is_stable is True


def test_get_actual_worker_counts_no_components(kubernetes_connector, mock_kube_api):
    """Test get_actual_worker_counts with no components specified"""
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {"services": {}},
        "status": {"services": {}},
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    (
        prefill_count,
        decode_count,
        is_stable,
    ) = kubernetes_connector.get_actual_worker_counts(
        prefill_component_name=None,
        decode_component_name=None,
    )

    assert prefill_count == 0
    assert decode_count == 0
    assert is_stable is True
