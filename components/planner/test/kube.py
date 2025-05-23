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
from unittest.mock import AsyncMock, MagicMock, patch

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


@pytest.mark.asyncio
async def test_wait_for_graph_deployment_ready_success(k8s_api, mock_custom_api):
    # Mock the get_graph_deployment response
    mock_deployment: Dict[str, Any] = {
        "status": {
            "conditions": [
                {"type": "Ready", "status": "True", "message": "Deployment is ready"}
            ]
        }
    }
    k8s_api.get_graph_deployment = AsyncMock(return_value=mock_deployment)

    # Test with minimal attempts and delay for faster testing
    await k8s_api.wait_for_graph_deployment_ready(
        "test-deployment", max_attempts=2, delay_seconds=0.1
    )

    # Verify get_graph_deployment was called
    k8s_api.get_graph_deployment.assert_called_once_with(
        "test-deployment", k8s_api.current_namespace
    )


@pytest.mark.asyncio
async def test_wait_for_graph_deployment_ready_timeout(k8s_api, mock_custom_api):
    # Mock the get_graph_deployment response with not ready status
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
    k8s_api.get_graph_deployment = AsyncMock(return_value=mock_deployment)

    # Test with minimal attempts and delay for faster testing
    with pytest.raises(TimeoutError) as exc_info:
        await k8s_api.wait_for_graph_deployment_ready(
            "test-deployment", max_attempts=2, delay_seconds=0.1
        )

    assert "is not ready after" in str(exc_info.value)
    assert k8s_api.get_graph_deployment.call_count == 2


@pytest.mark.asyncio
async def test_wait_for_graph_deployment_not_found(k8s_api, mock_custom_api):
    # Mock the get_graph_deployment response to return None
    k8s_api.get_graph_deployment = AsyncMock(return_value=None)

    # Test with minimal attempts and delay for faster testing
    with pytest.raises(ValueError) as exc_info:
        await k8s_api.wait_for_graph_deployment_ready(
            "test-deployment", max_attempts=2, delay_seconds=0.1
        )

    assert "not found" in str(exc_info.value)
    assert k8s_api.get_graph_deployment.call_count == 1


@pytest.mark.asyncio
async def test_wait_for_graph_deployment_no_conditions(k8s_api, mock_custom_api):
    # Mock the get_graph_deployment response with no conditions
    mock_deployment: Dict[str, Any] = {"status": {}}
    k8s_api.get_graph_deployment = AsyncMock(return_value=mock_deployment)

    # Test with minimal attempts and delay for faster testing
    with pytest.raises(TimeoutError) as exc_info:
        await k8s_api.wait_for_graph_deployment_ready(
            "test-deployment", max_attempts=2, delay_seconds=0.1
        )

    assert "is not ready after" in str(exc_info.value)
    assert k8s_api.get_graph_deployment.call_count == 2


@pytest.mark.asyncio
async def test_wait_for_graph_deployment_ready_on_second_attempt(
    k8s_api, mock_custom_api
):
    # Mock the get_graph_deployment response to return not ready first, then ready
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
    k8s_api.get_graph_deployment = AsyncMock(
        side_effect=[mock_deployment_not_ready, mock_deployment_ready]
    )

    # Test with minimal attempts and delay for faster testing
    await k8s_api.wait_for_graph_deployment_ready(
        "test-deployment", max_attempts=2, delay_seconds=0.1
    )

    assert k8s_api.get_graph_deployment.call_count == 2
