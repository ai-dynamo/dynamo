# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for remote planner components.

Tests RemotePlannerClient (low-level) and GlobalPlannerConnector (high-level)
for delegating scale requests to GlobalPlanner.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.planner import SubComponentType, TargetReplica
from dynamo.planner.global_planner_connector import GlobalPlannerConnector
from dynamo.planner.remote_planner_client import RemotePlannerClient
from dynamo.planner.scale_protocol import ScaleRequest, ScaleResponse, ScaleStatus
from dynamo.planner.utils.exceptions import EmptyTargetReplicasError

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
    endpoint_mock = MagicMock()
    client_mock = AsyncMock()

    runtime.endpoint.return_value = endpoint_mock
    endpoint_mock.client = AsyncMock(return_value=client_mock)
    client_mock.wait_for_instances = AsyncMock()

    # Mock scale_request to return a response
    client_mock.scale_request = AsyncMock(
        return_value={
            "status": "success",
            "message": "Scaled successfully",
            "current_replicas": {"prefill": 3, "decode": 5},
        }
    )

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
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=3
            ),
            TargetReplica(
                sub_component_type=SubComponentType.DECODE, desired_replicas=5
            ),
        ],
        blocking=False,
    )

    response = await client.send_scale_request(request)

    assert response.status == ScaleStatus.SUCCESS
    assert response.message == "Scaled successfully"
    assert response.current_replicas["prefill"] == 3
    assert response.current_replicas["decode"] == 5
    # Verify lazy init happened
    assert client._client is not None
    runtime.endpoint.assert_called_once_with("central-ns.Planner.scale_request")


@pytest.mark.asyncio
async def test_send_scale_request_error():
    """Test scale request error handling."""
    runtime = MagicMock()
    endpoint_mock = MagicMock()
    client_mock = AsyncMock()

    runtime.endpoint.return_value = endpoint_mock
    endpoint_mock.client = AsyncMock(return_value=client_mock)
    client_mock.wait_for_instances = AsyncMock()

    # Mock scale_request to return error response
    client_mock.scale_request = AsyncMock(
        return_value={
            "status": "error",
            "message": "Namespace not authorized",
            "current_replicas": {},
        }
    )

    client = RemotePlannerClient(runtime, "central-ns", "Planner")

    request = ScaleRequest(
        caller_namespace="unauthorized-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=1
            )
        ],
    )

    response = await client.send_scale_request(request)

    assert response.status == ScaleStatus.ERROR
    assert "not authorized" in response.message


@pytest.mark.asyncio
async def test_send_scale_request_no_response():
    """Test scale request when no response is received."""
    runtime = MagicMock()
    endpoint_mock = MagicMock()
    client_mock = AsyncMock()

    runtime.endpoint.return_value = endpoint_mock
    endpoint_mock.client = AsyncMock(return_value=client_mock)
    client_mock.wait_for_instances = AsyncMock()

    # Mock scale_request to return None
    client_mock.scale_request = AsyncMock(return_value=None)

    client = RemotePlannerClient(runtime, "central-ns", "Planner")

    request = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplica(
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
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=2
            )
        ],
    )

    request2 = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplica(
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


# ============================================================================
# GlobalPlannerConnector Tests
# ============================================================================


@pytest.fixture
def connector_runtime():
    """Mock runtime for GlobalPlannerConnector"""
    return MagicMock()


@pytest.fixture
def connector(connector_runtime):
    """Create GlobalPlannerConnector instance"""
    return GlobalPlannerConnector(
        runtime=connector_runtime,
        dynamo_namespace="test-ns",
        global_planner_namespace="global-ns",
        model_name="test-model",
    )


@pytest.mark.asyncio
async def test_connector_initialization(connector, connector_runtime):
    """Test GlobalPlannerConnector initialization and async_init"""
    assert connector.dynamo_namespace == "test-ns"
    assert connector.global_planner_namespace == "global-ns"
    assert connector.remote_client is None

    with patch(
        "dynamo.planner.global_planner_connector.RemotePlannerClient"
    ) as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        await connector._async_init()
        mock_client_class.assert_called_once_with(
            connector_runtime, "global-ns", "GlobalPlanner"
        )
        assert connector.remote_client == mock_client


@pytest.mark.asyncio
async def test_connector_set_replicas_success(connector):
    """Test GlobalPlannerConnector scaling with enum conversion and predicted load"""
    target_replicas = [
        TargetReplica(
            sub_component_type=SubComponentType.PREFILL,
            component_name="prefill-svc",
            desired_replicas=3,
        ),
        TargetReplica(
            sub_component_type=SubComponentType.DECODE,
            component_name="decode-svc",
            desired_replicas=5,
        ),
    ]

    with patch.dict(
        os.environ, {"DYN_PARENT_DGD_K8S_NAME": "dgd", "POD_NAMESPACE": "ns"}
    ):
        mock_response = ScaleResponse(
            status=ScaleStatus.SUCCESS,
            message="OK",
            current_replicas={"prefill": 3, "decode": 5},
        )
        mock_client = AsyncMock()
        mock_client.send_scale_request = AsyncMock(return_value=mock_response)
        connector.remote_client = mock_client
        connector.set_predicted_load(100.0, 512.0, 256.0)

        await connector.set_component_replicas(target_replicas, blocking=False)

        # Verify request structure and enum to string conversion
        request = mock_client.send_scale_request.call_args[0][0]
        assert request.caller_namespace == "test-ns"
        assert request.blocking is False
        assert request.predicted_load["num_requests"] == 100.0
        assert len(request.target_replicas) == 2
        assert request.target_replicas[0].sub_component_type == "prefill"
        assert isinstance(request.target_replicas[0].sub_component_type, str)


@pytest.mark.asyncio
async def test_connector_error_handling(connector):
    """Test GlobalPlannerConnector error handling"""
    # Empty list
    with pytest.raises(EmptyTargetReplicasError):
        await connector.set_component_replicas([])

    # Uninitialized
    target = [
        TargetReplica(
            sub_component_type=SubComponentType.PREFILL,
            component_name="p",
            desired_replicas=1,
        )
    ]
    with pytest.raises(RuntimeError, match="not initialized"):
        await connector.set_component_replicas(target)

    # Error response
    with patch.dict(os.environ, {"DYN_PARENT_DGD_K8S_NAME": "d", "POD_NAMESPACE": "n"}):
        mock_response = ScaleResponse(
            status=ScaleStatus.ERROR, message="Failed", current_replicas={}
        )
        mock_client = AsyncMock()
        mock_client.send_scale_request = AsyncMock(return_value=mock_response)
        connector.remote_client = mock_client
        with pytest.raises(RuntimeError, match="GlobalPlanner scaling failed"):
            await connector.set_component_replicas(target)


@pytest.mark.asyncio
async def test_connector_unsupported_and_noop_operations(connector):
    """Test unsupported and no-op operations"""
    # Unsupported
    with pytest.raises(NotImplementedError, match="batch operations"):
        await connector.add_component(SubComponentType.PREFILL)
    with pytest.raises(NotImplementedError, match="batch operations"):
        await connector.remove_component(SubComponentType.DECODE)

    # No-op operations
    await connector.validate_deployment(
        prefill_component_name="p", decode_component_name="d"
    )
    await connector.wait_for_deployment_ready()


def test_connector_model_name_and_predicted_load(connector_runtime):
    """Test GlobalPlannerConnector model name and predicted load tracking"""
    # With model name
    c1 = GlobalPlannerConnector(connector_runtime, "ns", "gns", "GP", model_name="test")
    assert c1.get_model_name() == "test"

    # Without model name
    c2 = GlobalPlannerConnector(connector_runtime, "ns", "gns", "GP", model_name=None)
    assert c2.get_model_name() == "managed-remotely"

    # Predicted load
    c1.set_predicted_load(42.0, 256.0, 128.0)
    assert c1.last_predicted_load == {"num_requests": 42.0, "isl": 256.0, "osl": 128.0}
