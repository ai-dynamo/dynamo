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

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from kubernetes import client

from dynamo.planner.kube import KubernetesAPI
from dynamo.planner.utils.exceptions import DynamoGraphDeploymentNotFoundError


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


def test_kubernetes_api_init_with_namespace(mock_custom_api, mock_config):
    """Test KubernetesAPI initialization with custom namespace"""
    api = KubernetesAPI(k8s_namespace="custom-namespace")
    assert api.current_namespace == "custom-namespace"


def test_kubernetes_api_init_without_namespace(mock_custom_api, mock_config):
    """Test KubernetesAPI initialization without custom namespace"""
    api = KubernetesAPI()
    # Should use the default namespace logic
    assert api.current_namespace == "default"


def test_get_graph_deployment_from_name(k8s_api, mock_custom_api):
    """Test _get_graph_deployment_from_name method"""
    mock_deployment = {"metadata": {"name": "test-deployment"}}
    mock_custom_api.get_namespaced_custom_object.return_value = mock_deployment

    result = k8s_api._get_graph_deployment_from_name("test-deployment")

    assert result == mock_deployment
    mock_custom_api.get_namespaced_custom_object.assert_called_once_with(
        group="nvidia.com",
        version="v1alpha1",
        namespace=k8s_api.current_namespace,
        plural="dynamographdeployments",
        name="test-deployment",
    )


def test_update_service_replicas_uses_dgdsa_scale(k8s_api, mock_custom_api):
    """Test that update_service_replicas uses DGDSA Scale API when available"""
    mock_custom_api.patch_namespaced_custom_object_scale.return_value = None

    k8s_api.update_service_replicas("test-deployment", "Frontend", 3)

    # Should use Scale subresource with lowercase adapter name
    mock_custom_api.patch_namespaced_custom_object_scale.assert_called_once_with(
        group="nvidia.com",
        version="v1alpha1",
        namespace=k8s_api.current_namespace,
        plural="dynamographdeploymentscalingadapters",
        name="test-deployment-frontend",  # lowercase service name
        body={"spec": {"replicas": 3}},
    )
    # Should NOT fall back to DGD patch
    mock_custom_api.patch_namespaced_custom_object.assert_not_called()


def test_update_service_replicas_fallback_to_dgd(k8s_api, mock_custom_api):
    """Test that update_service_replicas falls back to DGD when DGDSA not found"""
    # DGDSA doesn't exist (404)
    mock_custom_api.patch_namespaced_custom_object_scale.side_effect = (
        client.ApiException(status=404)
    )
    mock_custom_api.patch_namespaced_custom_object.return_value = None

    k8s_api.update_service_replicas("test-deployment", "test-component", 1)

    # Should have tried DGDSA first
    mock_custom_api.patch_namespaced_custom_object_scale.assert_called_once()

    # Should fall back to DGD patch
    mock_custom_api.patch_namespaced_custom_object.assert_called_once_with(
        group="nvidia.com",
        version="v1alpha1",
        namespace=k8s_api.current_namespace,
        plural="dynamographdeployments",
        name="test-deployment",
        body={"spec": {"services": {"test-component": {"replicas": 1}}}},
    )


def test_update_service_replicas_propagates_other_errors(k8s_api, mock_custom_api):
    """Test that update_service_replicas propagates non-404 errors"""
    mock_custom_api.patch_namespaced_custom_object_scale.side_effect = (
        client.ApiException(status=500, reason="Internal Server Error")
    )

    with pytest.raises(client.ApiException) as exc_info:
        k8s_api.update_service_replicas("test-deployment", "test-component", 1)

    assert exc_info.value.status == 500
    # Should NOT fall back to DGD
    mock_custom_api.patch_namespaced_custom_object.assert_not_called()


def test_update_graph_replicas_calls_update_service_replicas(k8s_api, mock_custom_api):
    """Test that deprecated update_graph_replicas calls update_service_replicas"""
    mock_custom_api.patch_namespaced_custom_object_scale.return_value = None

    # Use the deprecated method
    k8s_api.update_graph_replicas("test-deployment", "test-component", 1)

    # Should delegate to update_service_replicas which uses Scale API
    mock_custom_api.patch_namespaced_custom_object_scale.assert_called_once_with(
        group="nvidia.com",
        version="v1alpha1",
        namespace=k8s_api.current_namespace,
        plural="dynamographdeploymentscalingadapters",
        name="test-deployment-test-component",
        body={"spec": {"replicas": 1}},
    )


def test_update_dgd_replicas_directly(k8s_api, mock_custom_api):
    """Test the internal _update_dgd_replicas method"""
    mock_custom_api.patch_namespaced_custom_object.return_value = None

    k8s_api._update_dgd_replicas("test-deployment", "test-component", 1)

    mock_custom_api.patch_namespaced_custom_object.assert_called_once_with(
        group="nvidia.com",
        version="v1alpha1",
        namespace=k8s_api.current_namespace,
        plural="dynamographdeployments",
        name="test-deployment",
        body={"spec": {"services": {"test-component": {"replicas": 1}}}},
    )


@pytest.mark.asyncio
async def test_is_deployment_ready_true(k8s_api, mock_custom_api):
    """Test is_deployment_ready method when deployment is ready"""
    # Mock the _get_graph_deployment_from_name response
    mock_deployment: Dict[str, Any] = {
        "status": {
            "conditions": [
                {"type": "Ready", "status": "True", "message": "Deployment is ready"}
            ]
        }
    }

    result = k8s_api.is_deployment_ready(mock_deployment)
    assert result is True


@pytest.mark.asyncio
async def test_is_deployment_ready_false(k8s_api, mock_custom_api):
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
    result = k8s_api.is_deployment_ready(mock_deployment)
    assert result is False


@pytest.mark.asyncio
async def test_wait_for_graph_deployment_ready_success(k8s_api, mock_custom_api):
    """Test wait_for_graph_deployment_ready when deployment becomes ready"""
    # Mock the _get_graph_deployment_from_name response
    mock_deployment: Dict[str, Any] = {
        "status": {
            "conditions": [
                {"type": "Ready", "status": "True", "message": "Deployment is ready"}
            ]
        }
    }

    # Mock the method on the instance
    with patch.object(k8s_api, "get_graph_deployment", return_value=mock_deployment):
        # Test with minimal attempts and delay for faster testing
        await k8s_api.wait_for_graph_deployment_ready(
            "test-deployment", max_attempts=2, delay_seconds=0.1
        )


@pytest.mark.asyncio
async def test_wait_for_graph_deployment_ready_timeout(k8s_api, mock_custom_api):
    """Test wait_for_graph_deployment_ready when deployment times out"""
    # Mock the _get_graph_deployment_from_name response with not ready status
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
    with patch.object(k8s_api, "get_graph_deployment", return_value=mock_deployment):
        # Test with minimal attempts and delay for faster testing
        with pytest.raises(TimeoutError) as exc_info:
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment", max_attempts=2, delay_seconds=0.1
            )

        assert "is not ready after" in str(exc_info.value)


@pytest.mark.asyncio
async def test_wait_for_graph_deployment_not_found(k8s_api, mock_custom_api):
    """Test wait_for_graph_deployment_ready when deployment is not found"""

    mock_custom_api.get_namespaced_custom_object.side_effect = client.ApiException(
        status=404
    )

    # Test with minimal attempts and delay for faster testing
    with pytest.raises(DynamoGraphDeploymentNotFoundError) as exc_info:
        await k8s_api.wait_for_graph_deployment_ready(
            "test-deployment", max_attempts=2, delay_seconds=0.1
        )

    # Validate the exception fields
    exception = exc_info.value
    assert exception.deployment_name == "test-deployment"
    assert exception.namespace == "default"


@pytest.mark.asyncio
async def test_wait_for_graph_deployment_no_conditions(k8s_api, mock_custom_api):
    """Test wait_for_graph_deployment_ready when deployment has no conditions"""
    # Mock the _get_graph_deployment_from_name response with no conditions
    mock_deployment: Dict[str, Any] = {"status": {}}

    with patch.object(k8s_api, "get_graph_deployment", return_value=mock_deployment):
        # Test with minimal attempts and delay for faster testing
        with pytest.raises(TimeoutError) as exc_info:
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment", max_attempts=2, delay_seconds=0.1
            )

        assert "is not ready after" in str(exc_info.value)


@pytest.mark.asyncio
async def test_wait_for_graph_deployment_ready_on_second_attempt(
    k8s_api, mock_custom_api
):
    """Test wait_for_graph_deployment_ready when deployment becomes ready on second attempt"""
    # Mock the _get_graph_deployment_from_name response to return not ready first, then ready
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
        "_get_graph_deployment_from_name",
        side_effect=[mock_deployment_not_ready, mock_deployment_ready],
    ):
        # Test with minimal attempts and delay for faster testing
        await k8s_api.wait_for_graph_deployment_ready(
            "test-deployment", max_attempts=2, delay_seconds=0.1
        )


@pytest.mark.asyncio
async def test_get_graph_deployment(k8s_api, mock_custom_api):
    """Test get_graph_deployment"""
    mock_deployment = {"metadata": {"name": "parent-dgd"}}

    with patch.object(
        k8s_api, "_get_graph_deployment_from_name", return_value=mock_deployment
    ) as mock_get:
        result = await k8s_api.get_graph_deployment("parent-dgd")

        assert result == mock_deployment
        mock_get.assert_called_once_with("parent-dgd")


@pytest.mark.asyncio
async def test_get_graph_deployment_not_found(k8s_api, mock_custom_api):
    """Test get_graph_deployment when deployment is not found"""
    k8s_api.custom_api.get_namespaced_custom_object.side_effect = client.ApiException(
        status=404
    )
    with pytest.raises(DynamoGraphDeploymentNotFoundError) as exc_info:
        await k8s_api.get_graph_deployment("parent-dgd")

    exception = exc_info.value
    assert exception.deployment_name == "parent-dgd"
    assert exception.namespace == "default"
