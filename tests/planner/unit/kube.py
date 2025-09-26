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
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

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

    # Mock the method on the instance
    with patch.object(
        k8s_api, "_get_graph_deployment_from_name", return_value=mock_deployment
    ):
        result = await k8s_api.is_deployment_ready("test-deployment")
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

    # Mock the method on the instance
    with patch.object(
        k8s_api, "_get_graph_deployment_from_name", return_value=mock_deployment
    ):
        result = await k8s_api.is_deployment_ready("test-deployment")
        assert result is False


@pytest.mark.asyncio
async def test_is_deployment_ready_not_found(k8s_api, mock_custom_api):
    """Test is_deployment_ready method when deployment is not found"""
    # Mock the method on the instance
    with patch.object(k8s_api, "_get_graph_deployment_from_name", return_value=None):
        with pytest.raises(ValueError) as exc_info:
            await k8s_api.is_deployment_ready("test-deployment")

        assert "not found" in str(exc_info.value)


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
    with patch.object(
        k8s_api, "_get_graph_deployment_from_name", return_value=mock_deployment
    ):
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
    with patch.object(
        k8s_api, "_get_graph_deployment_from_name", return_value=mock_deployment
    ):
        # Test with minimal attempts and delay for faster testing
        with pytest.raises(TimeoutError) as exc_info:
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment", max_attempts=2, delay_seconds=0.1
            )

        assert "is not ready after" in str(exc_info.value)


@pytest.mark.asyncio
async def test_wait_for_graph_deployment_not_found(k8s_api, mock_custom_api):
    """Test wait_for_graph_deployment_ready when deployment is not found"""
    # Mock the _get_graph_deployment_from_name response to return None
    with patch.object(k8s_api, "_get_graph_deployment_from_name", return_value=None):
        # Test with minimal attempts and delay for faster testing
        with pytest.raises(ValueError) as exc_info:
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment", max_attempts=2, delay_seconds=0.1
            )

        assert "not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_wait_for_graph_deployment_no_conditions(k8s_api, mock_custom_api):
    """Test wait_for_graph_deployment_ready when deployment has no conditions"""
    # Mock the _get_graph_deployment_from_name response with no conditions
    mock_deployment: Dict[str, Any] = {"status": {}}

    with patch.object(
        k8s_api, "_get_graph_deployment_from_name", return_value=mock_deployment
    ):
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
async def test_get_parent_graph_deployment_with_env_var(k8s_api, mock_custom_api):
    """Test get_parent_graph_deployment with environment variable set"""
    mock_deployment = {"metadata": {"name": "parent-dgd"}}

    with patch.dict(os.environ, {"DYN_PARENT_DGD_K8S_NAME": "parent-dgd"}):
        with patch.object(
            k8s_api, "_get_graph_deployment_from_name", return_value=mock_deployment
        ) as mock_get:
            result = await k8s_api.get_parent_graph_deployment()

            assert result == mock_deployment
            mock_get.assert_called_once_with("parent-dgd")


@pytest.mark.asyncio
async def test_get_parent_graph_deployment_without_env_var(k8s_api, mock_custom_api):
    """Test get_parent_graph_deployment without environment variable"""
    with patch.dict(os.environ, {}, clear=True):
        result = await k8s_api.get_parent_graph_deployment()
        assert result is None


@pytest.mark.asyncio
async def test_get_graph_deployment_delegates_to_parent(k8s_api, mock_custom_api):
    """Test get_graph_deployment delegates to get_parent_graph_deployment"""
    mock_deployment = {"metadata": {"name": "parent-dgd"}}

    with patch.object(
        k8s_api, "get_parent_graph_deployment", return_value=mock_deployment
    ) as mock_parent:
        result = await k8s_api.get_graph_deployment()

        assert result == mock_deployment
        mock_parent.assert_called_once()
