# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for remote planner protocol and client.

This module tests:
1. Scale protocol data structures (ScaleRequest, ScaleResponse, TargetReplicaRequest)
2. RemotePlannerClient for delegating scale requests to GlobalPlanner
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.planner import SubComponentType
from dynamo.planner.remote_planner_client import RemotePlannerClient
from dynamo.planner.scale_protocol import (
    ScaleRequest,
    ScaleResponse,
    TargetReplicaRequest,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ==============================================================================
# Protocol Data Structure Tests
# ==============================================================================


def test_target_replica_request_creation():
    """Test creating TargetReplicaRequest objects."""
    request = TargetReplicaRequest(
        sub_component_type=SubComponentType.PREFILL, desired_replicas=5
    )
    assert request.sub_component_type == SubComponentType.PREFILL
    assert request.desired_replicas == 5
    assert request.component_name is None


def test_target_replica_request_with_component_name():
    """Test TargetReplicaRequest with component name."""
    request = TargetReplicaRequest(
        sub_component_type=SubComponentType.DECODE,
        component_name="decode-worker",
        desired_replicas=8,
    )
    assert request.sub_component_type == SubComponentType.DECODE
    assert request.component_name == "decode-worker"
    assert request.desired_replicas == 8


def test_scale_request_serialization():
    """Test ScaleRequest serialization to JSON."""
    target_replicas = [
        TargetReplicaRequest(
            sub_component_type=SubComponentType.PREFILL, desired_replicas=3
        ),
        TargetReplicaRequest(
            sub_component_type=SubComponentType.DECODE, desired_replicas=5
        ),
    ]

    request = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=target_replicas,
        blocking=False,
        timestamp=time.time(),
    )

    # Serialize to JSON
    json_str = request.model_dump_json()
    parsed = json.loads(json_str)

    assert parsed["caller_namespace"] == "app-ns"
    assert parsed["graph_deployment_name"] == "my-dgd"
    assert parsed["k8s_namespace"] == "default"
    assert len(parsed["target_replicas"]) == 2
    assert parsed["target_replicas"][0]["desired_replicas"] == 3
    assert parsed["target_replicas"][1]["desired_replicas"] == 5
    assert parsed["blocking"] is False
    assert "timestamp" in parsed


def test_scale_request_deserialization():
    """Test ScaleRequest deserialization from JSON."""
    json_data = {
        "caller_namespace": "test-ns",
        "graph_deployment_name": "test-dgd",
        "k8s_namespace": "kube-system",
        "target_replicas": [
            {"sub_component_type": "prefill", "desired_replicas": 2},
            {
                "sub_component_type": "decode",
                "component_name": "decode-svc",
                "desired_replicas": 4,
            },
        ],
        "blocking": True,
    }

    request = ScaleRequest(**json_data)

    assert request.caller_namespace == "test-ns"
    assert request.graph_deployment_name == "test-dgd"
    assert request.k8s_namespace == "kube-system"
    assert len(request.target_replicas) == 2
    assert request.target_replicas[0].sub_component_type == SubComponentType.PREFILL
    assert request.target_replicas[0].desired_replicas == 2
    assert request.target_replicas[1].sub_component_type == SubComponentType.DECODE
    assert request.target_replicas[1].component_name == "decode-svc"
    assert request.target_replicas[1].desired_replicas == 4
    assert request.blocking is True


def test_scale_request_with_predicted_load():
    """Test ScaleRequest with predicted load context."""
    target_replicas = [
        TargetReplicaRequest(
            sub_component_type=SubComponentType.PREFILL, desired_replicas=10
        )
    ]

    request = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=target_replicas,
        predicted_load={"num_requests": 100, "isl": 512, "osl": 128},
    )

    assert request.predicted_load is not None
    assert request.predicted_load["num_requests"] == 100
    assert request.predicted_load["isl"] == 512
    assert request.predicted_load["osl"] == 128


def test_scale_request_defaults():
    """Test ScaleRequest default values."""
    target_replicas = [
        TargetReplicaRequest(
            sub_component_type=SubComponentType.PREFILL, desired_replicas=1
        )
    ]

    request = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=target_replicas,
    )

    # Check defaults
    assert request.blocking is False
    assert request.timestamp is None
    assert request.predicted_load is None


def test_scale_response_creation():
    """Test creating ScaleResponse objects."""
    response = ScaleResponse(
        status="success",
        message="Scaled successfully",
        current_replicas={"prefill": 3, "decode": 5},
    )

    assert response.status == "success"
    assert response.message == "Scaled successfully"
    assert response.current_replicas["prefill"] == 3
    assert response.current_replicas["decode"] == 5


def test_scale_response_error():
    """Test ScaleResponse for error cases."""
    response = ScaleResponse(
        status="error", message="Namespace not authorized", current_replicas={}
    )

    assert response.status == "error"
    assert "not authorized" in response.message
    assert response.current_replicas == {}


def test_scale_response_serialization():
    """Test ScaleResponse serialization."""
    response = ScaleResponse(
        status="success",
        message="Test message",
        current_replicas={"prefill": 2, "decode": 4},
    )

    json_str = response.model_dump_json()
    parsed = json.loads(json_str)

    assert parsed["status"] == "success"
    assert parsed["message"] == "Test message"
    assert parsed["current_replicas"]["prefill"] == 2
    assert parsed["current_replicas"]["decode"] == 4


def test_scale_response_deserialization():
    """Test ScaleResponse deserialization."""
    json_data = {
        "status": "scaling",
        "message": "Scaling in progress",
        "current_replicas": {"prefill": 5, "decode": 10},
    }

    response = ScaleResponse(**json_data)

    assert response.status == "scaling"
    assert response.message == "Scaling in progress"
    assert response.current_replicas["prefill"] == 5
    assert response.current_replicas["decode"] == 10


# ==============================================================================
# RemotePlannerClient Tests
# ==============================================================================


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
async def test_remote_client_initialization():
    """Test RemotePlannerClient initialization."""
    mock_runtime = MagicMock()
    client = RemotePlannerClient(mock_runtime, "central-ns", "Planner")

    assert client.central_namespace == "central-ns"
    assert client.central_component == "Planner"
    assert client._client is None


@pytest.mark.asyncio
async def test_remote_client_lazy_initialization(mock_runtime):
    """Test RemotePlannerClient lazy initializes endpoint client."""
    runtime, mock_client = mock_runtime
    client = RemotePlannerClient(runtime, "central-ns", "Planner")

    # Client should be None initially
    assert client._client is None

    # Create a scale request
    request = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplicaRequest(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=3
            )
        ],
    )

    # Send request (should trigger lazy init)
    await client.send_scale_request(request)

    # Verify client was initialized
    assert client._client is not None
    runtime.namespace.assert_called_once_with("central-ns")


@pytest.mark.asyncio
async def test_send_scale_request_success(mock_runtime):
    """Test successful scale request."""
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
