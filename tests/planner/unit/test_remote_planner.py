# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for remote planner client.

Tests RemotePlannerClient for delegating scale requests to GlobalPlanner.
Protocol serialization/deserialization is tested implicitly through actual usage.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.planner import SubComponentType
from dynamo.planner.remote_planner_client import RemotePlannerClient
from dynamo.planner.scale_protocol import ScaleRequest, TargetReplicaRequest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.fixture
def mock_runtime():
    """Create a mock DistributedRuntime."""
    runtime = MagicMock()
    namespace_mock = MagicMock()
    component_mock = MagicMock()
    endpoint_mock = MagicMock()
    client_mock = AsyncMock()

    runtime.namespace.return_value = namespace_mock
    namespace_mock.component.return_value = component_mock
    component_mock.endpoint.return_value = endpoint_mock
    endpoint_mock.client = AsyncMock(return_value=client_mock)
    client_mock.wait_for_instances = AsyncMock()

    # Mock round_robin to return a response
    async def mock_round_robin(request_json):
        yield {
            "status": "success",
            "message": "Scaled successfully",
            "current_replicas": {"prefill": 3, "decode": 5},
        }

    client_mock.round_robin = AsyncMock(side_effect=mock_round_robin)

    return runtime, client_mock


@pytest.mark.asyncio
async def test_send_scale_request_success(mock_runtime):
    """Test successful scale request (exercises protocol, client, and serialization)."""
    runtime, mock_client = mock_runtime
    client = RemotePlannerClient(runtime, "central-ns", "Planner")

    request = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplicaRequest(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=3
            ),
            TargetReplicaRequest(
                sub_component_type=SubComponentType.DECODE, desired_replicas=5
            ),
        ],
        blocking=False,
    )

    response = await client.send_scale_request(request)

    assert response.status == "success"
    assert response.message == "Scaled successfully"
    assert response.current_replicas["prefill"] == 3
    assert response.current_replicas["decode"] == 5
    # Verify lazy init happened
    assert client._client is not None
    runtime.namespace.assert_called_once_with("central-ns")


@pytest.mark.asyncio
async def test_send_scale_request_error():
    """Test scale request error handling."""
    runtime = MagicMock()
    namespace_mock = MagicMock()
    component_mock = MagicMock()
    endpoint_mock = MagicMock()
    client_mock = AsyncMock()

    runtime.namespace.return_value = namespace_mock
    namespace_mock.component.return_value = component_mock
    component_mock.endpoint.return_value = endpoint_mock
    endpoint_mock.client = AsyncMock(return_value=client_mock)
    client_mock.wait_for_instances = AsyncMock()

    # Mock round_robin to return error response
    async def mock_round_robin_error(request_json):
        yield {
            "status": "error",
            "message": "Namespace not authorized",
            "current_replicas": {},
        }

    client_mock.round_robin = AsyncMock(side_effect=mock_round_robin_error)

    client = RemotePlannerClient(runtime, "central-ns", "Planner")

    request = ScaleRequest(
        caller_namespace="unauthorized-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplicaRequest(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=1
            )
        ],
    )

    response = await client.send_scale_request(request)

    assert response.status == "error"
    assert "not authorized" in response.message


@pytest.mark.asyncio
async def test_send_scale_request_no_response():
    """Test scale request when no response is received."""
    runtime = MagicMock()
    namespace_mock = MagicMock()
    component_mock = MagicMock()
    endpoint_mock = MagicMock()
    client_mock = AsyncMock()

    runtime.namespace.return_value = namespace_mock
    namespace_mock.component.return_value = component_mock
    component_mock.endpoint.return_value = endpoint_mock
    endpoint_mock.client = AsyncMock(return_value=client_mock)
    client_mock.wait_for_instances = AsyncMock()

    # Mock round_robin to return nothing
    async def mock_round_robin_empty(request_json):
        return
        yield  # Make it a generator but never yield anything

    client_mock.round_robin = AsyncMock(side_effect=mock_round_robin_empty)

    client = RemotePlannerClient(runtime, "central-ns", "Planner")

    request = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplicaRequest(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=1
            )
        ],
    )

    with pytest.raises(RuntimeError, match="No response from centralized planner"):
        await client.send_scale_request(request)


@pytest.mark.asyncio
async def test_multiple_requests_reuse_client(mock_runtime):
    """Test that multiple requests reuse the same client instance."""
    runtime, mock_client = mock_runtime
    client = RemotePlannerClient(runtime, "central-ns", "Planner")

    request1 = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplicaRequest(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=2
            )
        ],
    )

    request2 = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplicaRequest(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=4
            )
        ],
    )

    # Send first request
    await client.send_scale_request(request1)
    first_client = client._client

    # Send second request
    await client.send_scale_request(request2)
    second_client = client._client

    # Should be the same client instance
    assert first_client is second_client
