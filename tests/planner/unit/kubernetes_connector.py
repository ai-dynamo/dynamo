# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
