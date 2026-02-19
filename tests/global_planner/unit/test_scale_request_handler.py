# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ScaleRequestHandler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.global_planner.scale_handler import ScaleRequestHandler
from dynamo.planner import SubComponentType, TargetReplica
from dynamo.planner.scale_protocol import ScaleRequest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
    pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20"),
]


@pytest.fixture
def mock_runtime():
    """Create a mock DistributedRuntime."""
    return MagicMock()


@pytest.mark.asyncio
async def test_handler_authorization_success(mock_runtime):
    """Test handler authorizes requests from managed namespaces."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime, managed_namespaces=["app-ns"], k8s_namespace="default"
    )

    request = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=3
            )
        ],
    )

    # Mock KubernetesConnector
    with patch(
        "dynamo.global_planner.scale_handler.KubernetesConnector"
    ) as mock_connector_cls:
        mock_connector = AsyncMock()
        mock_connector_cls.return_value = mock_connector
        mock_connector._async_init = AsyncMock()
        mock_connector.set_component_replicas = AsyncMock()
        mock_connector.kube_api = MagicMock()
        mock_connector.kube_api.get_graph_deployment = MagicMock(
            return_value={
                "spec": {
                    "services": {
                        "prefill-svc": {"subComponentType": "prefill", "replicas": 3},
                        "decode-svc": {"subComponentType": "decode", "replicas": 5},
                    }
                }
            }
        )

        # Process request (pass as dict to match endpoint behavior)
        results = []
        async for response in handler.scale_request(request.model_dump()):
            results.append(response)

        assert len(results) == 1
        response = results[0]
        assert response["status"] == "success"
        assert "Scaled" in response["message"]
        assert response["current_replicas"]["prefill"] == 3
        assert response["current_replicas"]["decode"] == 5


@pytest.mark.asyncio
async def test_handler_authorization_failure(mock_runtime):
    """Test handler rejects requests from unauthorized namespaces."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime,
        managed_namespaces=["authorized-ns"],
        k8s_namespace="default",
    )

    request = ScaleRequest(
        caller_namespace="unauthorized-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=3
            )
        ],
    )

    # Process request
    results = []
    async for response in handler.scale_request(request.model_dump()):
        results.append(response)

    assert len(results) == 1
    response = results[0]
    assert response["status"] == "error"
    assert "not authorized" in response["message"]
    assert response["current_replicas"] == {}


@pytest.mark.asyncio
async def test_handler_multiple_dgds(mock_runtime):
    """Test handler creates separate connectors for different DGDs (and caches them)."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime, managed_namespaces=["app-ns"], k8s_namespace="default"
    )

    request1 = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="dgd-1",
        k8s_namespace="default",
        target_replicas=[
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=2
            )
        ],
    )

    request2 = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="dgd-2",  # Different DGD
        k8s_namespace="default",
        target_replicas=[
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=4
            )
        ],
    )

    with patch(
        "dynamo.global_planner.scale_handler.KubernetesConnector"
    ) as mock_connector_cls:
        mock_connector = AsyncMock()
        mock_connector_cls.return_value = mock_connector
        mock_connector._async_init = AsyncMock()
        mock_connector.set_component_replicas = AsyncMock()
        mock_connector.kube_api = MagicMock()
        mock_connector.kube_api.get_graph_deployment = MagicMock(
            return_value={"spec": {"services": {}}}
        )

        # Process both requests
        async for _ in handler.scale_request(request1.model_dump()):
            pass
        async for _ in handler.scale_request(request2.model_dump()):
            pass

        # Verify two connectors were created
        assert "default/dgd-1" in handler.connectors
        assert "default/dgd-2" in handler.connectors
        assert mock_connector_cls.call_count == 2


@pytest.mark.asyncio
async def test_handler_error_handling(mock_runtime):
    """Test handler error handling during scaling."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime, managed_namespaces=["app-ns"], k8s_namespace="default"
    )

    request = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=3
            )
        ],
    )

    with patch(
        "dynamo.global_planner.scale_handler.KubernetesConnector"
    ) as mock_connector_cls:
        mock_connector = AsyncMock()
        mock_connector_cls.return_value = mock_connector
        mock_connector._async_init = AsyncMock()
        # Simulate error during scaling
        mock_connector.set_component_replicas = AsyncMock(
            side_effect=Exception("Scaling failed")
        )

        # Process request (pass as dict to match endpoint behavior)
        results = []
        async for response in handler.scale_request(request.model_dump()):
            results.append(response)

        assert len(results) == 1
        response = results[0]
        assert response["status"] == "error"
        assert "Scaling failed" in response["message"]


@pytest.mark.asyncio
async def test_handler_blocking_mode(mock_runtime):
    """Test handler respects blocking mode."""
    handler = ScaleRequestHandler(
        runtime=mock_runtime, managed_namespaces=["app-ns"], k8s_namespace="default"
    )

    request = ScaleRequest(
        caller_namespace="app-ns",
        graph_deployment_name="my-dgd",
        k8s_namespace="default",
        target_replicas=[
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=3
            )
        ],
        blocking=True,  # Request blocking mode
    )

    with patch(
        "dynamo.global_planner.scale_handler.KubernetesConnector"
    ) as mock_connector_cls:
        mock_connector = AsyncMock()
        mock_connector_cls.return_value = mock_connector
        mock_connector._async_init = AsyncMock()
        mock_connector.set_component_replicas = AsyncMock()
        mock_connector.kube_api = MagicMock()
        mock_connector.kube_api.get_graph_deployment = MagicMock(
            return_value={"spec": {"services": {}}}
        )

        # Process request (pass as dict to match endpoint behavior)
        async for _ in handler.scale_request(request.model_dump()):
            pass

        # Verify blocking=True was passed to connector
        mock_connector.set_component_replicas.assert_called_once()
        call_args = mock_connector.set_component_replicas.call_args
        assert call_args[1]["blocking"] is True
