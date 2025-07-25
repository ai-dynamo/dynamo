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

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from kubernetes import client

from dynamo.planner.kube import KubernetesAPI


@pytest.fixture
def mock_config():
    with patch("dynamo.planner.kube.config") as mock:
        mock.load_incluster_config = MagicMock()
        yield mock


@pytest.fixture
def mock_custom_api():
    with patch("dynamo.planner.kube.client.CustomObjectsApi") as mock:
        yield mock.return_value


@pytest.fixture
def k8s_api(mock_custom_api, mock_config):
    return KubernetesAPI()


@pytest.fixture
def k8s_api_with_namespace(mock_custom_api, mock_config):
    return KubernetesAPI(k8s_namespace="test-namespace")


def test_kubernetes_api_init_with_namespace(
    mock_custom_api: Any, mock_config: Any
) -> None:
    """Test KubernetesAPI initialization with custom namespace"""
    api = KubernetesAPI(k8s_namespace="custom-namespace")
    assert api.current_namespace == "custom-namespace"


def test_kubernetes_api_init_without_namespace(
    mock_custom_api: Any, mock_config: Any
) -> None:
    """Test KubernetesAPI initialization without custom namespace"""
    api = KubernetesAPI()
    # Should use the default namespace logic
    assert api.current_namespace == "default"


def test_get_graph_deployment_from_name(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test _get_graph_deployment method with name"""
    mock_deployment = {"metadata": {"name": "test-deployment"}}
    mock_custom_api.get_namespaced_custom_object.return_value = mock_deployment

    result = k8s_api._get_graph_deployment(graph_deployment_name="test-deployment")

    assert result == mock_deployment
    mock_custom_api.get_namespaced_custom_object.assert_called_once_with(
        group="nvidia.com",
        version="v1alpha1",
        namespace=k8s_api.current_namespace,
        plural="dynamographdeployments",
        name="test-deployment",
        label_selector=None,
    )


def test_get_graph_deployment_from_label_selector(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test _get_graph_deployment method with label selector"""
    mock_deployment = {"items": [{"metadata": {"name": "test-deployment"}}]}
    mock_custom_api.get_namespaced_custom_object.return_value = mock_deployment

    result = k8s_api._get_graph_deployment(label_selector="test-label-selector")

    assert result == mock_deployment
    mock_custom_api.get_namespaced_custom_object.assert_called_once_with(
        group="nvidia.com",
        version="v1alpha1",
        namespace=k8s_api.current_namespace,
        plural="dynamographdeployments",
        name=None,
        label_selector="test-label-selector",
    )


@pytest.mark.asyncio
async def test_get_graph_deployment_success(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test get_graph_deployment method with successful flow"""
    # Mock the component deployment response
    mock_component_deployments = {
        "items": [
            {
                "metadata": {
                    "name": "test-component-deployment",
                    "ownerReferences": [
                        {
                            "apiVersion": "nvidia.com/v1alpha1",
                            "kind": "DynamoGraphDeployment",
                            "name": "test-graph-deployment",
                        }
                    ],
                }
            }
        ]
    }

    # Mock the graph deployment response
    mock_graph_deployment = {
        "metadata": {"name": "test-graph-deployment"},
        "spec": {"services": {"test-component": {"replicas": 1}}},
    }

    # Set up the mock to return different responses for different calls
    mock_custom_api.get_namespaced_custom_object.side_effect = [
        mock_component_deployments,  # First call for component deployments
        mock_graph_deployment,  # Second call for graph deployment
    ]

    result = await k8s_api.get_graph_deployment("test-component", "test-namespace")

    assert result == mock_graph_deployment
    assert mock_custom_api.get_namespaced_custom_object.call_count == 2

    # Check first call (for component deployments)
    first_call = mock_custom_api.get_namespaced_custom_object.call_args_list[0]
    assert (
        first_call[1]["label_selector"]
        == "nvidia.com/dynamo-component=test-component,nvidia.com/dynamo-namespace=test-namespace"
    )

    # Check second call (for graph deployment)
    second_call = mock_custom_api.get_namespaced_custom_object.call_args_list[1]
    assert second_call[1]["name"] == "test-graph-deployment"


@pytest.mark.asyncio
async def test_get_graph_deployment_no_component_deployments(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test get_graph_deployment when no component deployments are found"""
    mock_component_deployments: Dict[str, Any] = {"items": []}
    mock_custom_api.get_namespaced_custom_object.return_value = (
        mock_component_deployments
    )

    result = await k8s_api.get_graph_deployment("test-component", "test-namespace")

    assert result is None
    mock_custom_api.get_namespaced_custom_object.assert_called_once()


@pytest.mark.asyncio
async def test_get_graph_deployment_multiple_component_deployments(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test get_graph_deployment when multiple component deployments are found"""
    mock_component_deployments = {
        "items": [
            {"metadata": {"name": "deployment1"}},
            {"metadata": {"name": "deployment2"}},
        ]
    }
    mock_custom_api.get_namespaced_custom_object.return_value = (
        mock_component_deployments
    )

    with pytest.raises(ValueError) as exc_info:
        await k8s_api.get_graph_deployment("test-component", "test-namespace")

    assert "Multiple component deployments found" in str(exc_info.value)
    assert "Expected exactly one deployment" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_graph_deployment_no_owner_references(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test get_graph_deployment when component deployment has no owner references"""
    mock_component_deployments = {
        "items": [
            {
                "metadata": {
                    "name": "test-component-deployment",
                    # No ownerReferences
                }
            }
        ]
    }
    mock_custom_api.get_namespaced_custom_object.return_value = (
        mock_component_deployments
    )

    result = await k8s_api.get_graph_deployment("test-component", "test-namespace")

    assert result is None


@pytest.mark.asyncio
async def test_get_graph_deployment_no_graph_deployment_owner_ref(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test get_graph_deployment when component has owner references but none are DynamoGraphDeployment"""
    mock_component_deployments = {
        "items": [
            {
                "metadata": {
                    "name": "test-component-deployment",
                    "ownerReferences": [
                        {
                            "apiVersion": "apps/v1",
                            "kind": "Deployment",
                            "name": "some-other-deployment",
                        }
                    ],
                }
            }
        ]
    }
    mock_custom_api.get_namespaced_custom_object.return_value = (
        mock_component_deployments
    )

    result = await k8s_api.get_graph_deployment("test-component", "test-namespace")

    assert result is None


@pytest.mark.asyncio
async def test_get_graph_deployment_owner_ref_no_name(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test get_graph_deployment when owner reference has no name"""
    mock_component_deployments = {
        "items": [
            {
                "metadata": {
                    "name": "test-component-deployment",
                    "ownerReferences": [
                        {
                            "apiVersion": "nvidia.com/v1alpha1",
                            "kind": "DynamoGraphDeployment",
                            # No name field
                        }
                    ],
                }
            }
        ]
    }
    mock_custom_api.get_namespaced_custom_object.return_value = (
        mock_component_deployments
    )

    result = await k8s_api.get_graph_deployment("test-component", "test-namespace")

    assert result is None


@pytest.mark.asyncio
async def test_get_graph_deployment_api_exception_404(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test get_graph_deployment when API returns 404"""
    mock_custom_api.get_namespaced_custom_object.side_effect = client.ApiException(
        status=404
    )

    result = await k8s_api.get_graph_deployment("test-component", "test-namespace")

    assert result is None


@pytest.mark.asyncio
async def test_get_graph_deployment_api_exception_other(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test get_graph_deployment when API returns non-404 exception"""
    mock_custom_api.get_namespaced_custom_object.side_effect = client.ApiException(
        status=500
    )

    with pytest.raises(client.ApiException) as exc_info:
        await k8s_api.get_graph_deployment("test-component", "test-namespace")

    assert exc_info.value.status == 500


@pytest.mark.asyncio
async def test_is_deployment_ready_true(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test is_deployment_ready method when deployment is ready"""
    mock_deployment: Dict[str, Any] = {
        "status": {
            "conditions": [
                {"type": "Ready", "status": "True", "message": "Deployment is ready"}
            ]
        }
    }
    mock_custom_api.get_namespaced_custom_object.return_value = mock_deployment

    result = await k8s_api.is_deployment_ready("test-deployment")

    assert result is True


@pytest.mark.asyncio
async def test_is_deployment_ready_false(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test is_deployment_ready method when deployment is not ready"""
    mock_deployment: Dict[str, Any] = {
        "status": {
            "conditions": [
                {
                    "type": "Ready",
                    "status": "False",
                    "message": "Deployment is not ready",
                }
            ]
        }
    }
    mock_custom_api.get_namespaced_custom_object.return_value = mock_deployment

    result = await k8s_api.is_deployment_ready("test-deployment")

    assert result is False


@pytest.mark.asyncio
async def test_is_deployment_ready_not_found(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test is_deployment_ready method when deployment is not found"""
    mock_custom_api.get_namespaced_custom_object.return_value = None

    with pytest.raises(ValueError) as exc_info:
        await k8s_api.is_deployment_ready("test-deployment")

    assert "not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_wait_for_graph_deployment_ready_success(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test wait_for_graph_deployment_ready when deployment becomes ready"""
    # Mock the _get_graph_deployment response
    mock_deployment: Dict[str, Any] = {
        "status": {
            "conditions": [
                {"type": "Ready", "status": "True", "message": "Deployment is ready"}
            ]
        }
    }

    # Mock the method on the instance
    with patch.object(k8s_api, "_get_graph_deployment", return_value=mock_deployment):
        # Test with minimal attempts and delay for faster testing
        await k8s_api.wait_for_graph_deployment_ready(
            "test-deployment", max_attempts=2, delay_seconds=0.1
        )


@pytest.mark.asyncio
async def test_wait_for_graph_deployment_ready_timeout(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test wait_for_graph_deployment_ready when deployment times out"""
    # Mock the _get_graph_deployment response with not ready status
    mock_deployment: Dict[str, Any] = {
        "status": {
            "conditions": [
                {
                    "type": "Ready",
                    "status": "False",
                    "message": "Deployment is not ready",
                }
            ]
        }
    }

    # Mock the method on the instance
    with patch.object(k8s_api, "_get_graph_deployment", return_value=mock_deployment):
        # Test with minimal attempts and delay for faster testing
        with pytest.raises(TimeoutError) as exc_info:
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment", max_attempts=2, delay_seconds=0.1
            )

        assert "is not ready after" in str(exc_info.value)


@pytest.mark.asyncio
async def test_wait_for_graph_deployment_not_found(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test wait_for_graph_deployment_ready when deployment is not found"""
    # Mock the _get_graph_deployment response to return None
    with patch.object(k8s_api, "_get_graph_deployment", return_value=None):
        # Test with minimal attempts and delay for faster testing
        with pytest.raises(ValueError) as exc_info:
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment", max_attempts=2, delay_seconds=0.1
            )

        assert "not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_wait_for_graph_deployment_no_conditions(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test wait_for_graph_deployment_ready when deployment has no conditions"""
    # Mock the _get_graph_deployment response with no conditions
    mock_deployment: Dict[str, Any] = {"status": {}}

    with patch.object(k8s_api, "_get_graph_deployment", return_value=mock_deployment):
        # Test with minimal attempts and delay for faster testing
        with pytest.raises(TimeoutError) as exc_info:
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment", max_attempts=2, delay_seconds=0.1
            )

        assert "is not ready after" in str(exc_info.value)


@pytest.mark.asyncio
async def test_wait_for_graph_deployment_ready_on_second_attempt(
    k8s_api: KubernetesAPI, mock_custom_api: Any
) -> None:
    """Test wait_for_graph_deployment_ready when deployment becomes ready on second attempt"""
    # Mock the _get_graph_deployment response to return not ready first, then ready
    mock_deployment_not_ready: Dict[str, Any] = {
        "status": {
            "conditions": [
                {
                    "type": "Ready",
                    "status": "False",
                    "message": "Deployment is not ready",
                }
            ]
        }
    }
    mock_deployment_ready: Dict[str, Any] = {
        "status": {
            "conditions": [
                {"type": "Ready", "status": "True", "message": "Deployment is ready"}
            ]
        }
    }

    with patch.object(
        k8s_api,
        "_get_graph_deployment",
        side_effect=[mock_deployment_not_ready, mock_deployment_ready],
    ):
        # Test with minimal attempts and delay for faster testing
        await k8s_api.wait_for_graph_deployment_ready(
            "test-deployment", max_attempts=2, delay_seconds=0.1
        )
