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

from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.planner.kubernetes_connector import KubernetesConnector


@pytest.fixture
def mock_kube_api():
    mock_api = Mock()
    mock_api.get_graph_deployment = AsyncMock()
    mock_api.update_graph_replicas = AsyncMock()
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
    connector = KubernetesConnector()
    # Set the namespace attribute that's being accessed in the error
    connector.namespace = "default"
    return connector


@pytest.mark.asyncio
async def test_add_component_increases_replicas(kubernetes_connector, mock_kube_api):
    # Arrange
    component_name = "test-component"
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {"services": {"test-component": {"replicas": 1}}},
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    # Act
    await kubernetes_connector.add_component(component_name)

    # Assert
    mock_kube_api.get_graph_deployment.assert_called_once_with(component_name)
    mock_kube_api.update_graph_replicas.assert_called_once_with(
        "test-graph", component_name, 2
    )


@pytest.mark.asyncio
async def test_add_component_with_no_replicas_specified(
    kubernetes_connector, mock_kube_api
):
    # Arrange
    component_name = "test-component"
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {"services": {"test-component": {}}},
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    # Act
    await kubernetes_connector.add_component(component_name)

    # Assert
    mock_kube_api.update_graph_replicas.assert_called_once_with(
        "test-graph", component_name, 2
    )


@pytest.mark.asyncio
async def test_add_component_deployment_not_found(kubernetes_connector, mock_kube_api):
    # Arrange
    component_name = "test-component"
    mock_kube_api.get_graph_deployment.return_value = None

    # Act & Assert
    with pytest.raises(
        ValueError, match=f"Graph not found for component {component_name}"
    ):
        await kubernetes_connector.add_component(component_name)


@pytest.mark.asyncio
async def test_remove_component_decreases_replicas(kubernetes_connector, mock_kube_api):
    # Arrange
    component_name = "test-component"
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {"services": {"test-component": {"replicas": 2}}},
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    # Act
    await kubernetes_connector.remove_component(component_name)

    # Assert
    mock_kube_api.update_graph_replicas.assert_called_once_with(
        "test-graph", component_name, 1
    )


@pytest.mark.asyncio
async def test_remove_component_with_zero_replicas(kubernetes_connector, mock_kube_api):
    # Arrange
    component_name = "test-component"
    mock_deployment = {
        "metadata": {"name": "test-graph"},
        "spec": {"services": {"test-component": {"replicas": 0}}},
    }
    mock_kube_api.get_graph_deployment.return_value = mock_deployment

    # Act
    await kubernetes_connector.remove_component(component_name)

    # Assert
    mock_kube_api.update_graph_replicas.assert_not_called()
