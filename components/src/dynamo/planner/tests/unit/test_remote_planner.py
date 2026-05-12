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
from dynamo.planner.connectors.global_planner import GlobalPlannerConnector
from dynamo.planner.connectors.protocol import ScaleRequest, ScaleResponse, ScaleStatus
from dynamo.planner.connectors.remote_client import RemotePlannerClient
from dynamo.planner.errors import DeploymentValidationError, EmptyTargetReplicasError
from dynamo.planner.monitoring.worker_info import WorkerInfo


async def _async_responses(*items):
    """Async generator helper: yields each item in sequence, simulating a stream."""
    for item in items:
        yield item


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

    # Mock generate to return a single-item async stream with the response dict
    response_data = {
        "status": "success",
        "message": "Scaled successfully",
        "current_replicas": {"prefill": 3, "decode": 5},
    }
    client_mock.generate = AsyncMock(
        side_effect=lambda _: _async_responses(response_data)
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

    # Mock generate to return a single-item async stream with the error response dict
    client_mock.generate = AsyncMock(
        side_effect=lambda _: _async_responses(
            {
                "status": "error",
                "message": "Namespace not authorized",
                "current_replicas": {},
            }
        )
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

    # Mock generate to return an empty async stream (no items → no response)
    client_mock.generate = AsyncMock(side_effect=lambda _: _async_responses())

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
        "dynamo.planner.connectors.global_planner.RemotePlannerClient"
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
async def test_connector_set_replicas_rejected(connector):
    """REJECTED is a budget-gate outcome — must not raise, must log a warning."""
    target_replicas = [
        TargetReplica(
            sub_component_type=SubComponentType.PREFILL,
            component_name="prefill-svc",
            desired_replicas=2,
        )
    ]

    with patch.dict(
        os.environ, {"DYN_PARENT_DGD_K8S_NAME": "dgd", "POD_NAMESPACE": "ns"}
    ):
        mock_response = ScaleResponse(
            status=ScaleStatus.REJECTED,
            message="budget exceeded",
            current_replicas={},
        )
        mock_client = AsyncMock()
        mock_client.send_scale_request = AsyncMock(return_value=mock_response)
        connector.remote_client = mock_client

        with patch("dynamo.planner.connectors.global_planner.logger") as mock_logger:
            # Must not raise — REJECTED is a legitimate business outcome.
            await connector.set_component_replicas(target_replicas, blocking=False)

            # Warning logged, not error.
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "rejected" in warning_msg.lower()
            assert "budget exceeded" in warning_msg
            mock_logger.error.assert_not_called()

        # Client was still called — execution continued normally.
        mock_client.send_scale_request.assert_awaited_once()


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

    # validate_deployment remains a local no-op (GlobalPlanner validates).
    await connector.validate_deployment(
        prefill_component_name="p", decode_component_name="d"
    )

    # wait_for_deployment_ready with no local k8s connector available must
    # degrade to a no-op rather than raise, so out-of-cluster callers still
    # work.
    connector._local_k8s_connector = None
    connector._local_k8s_init_attempted = True
    await connector.wait_for_deployment_ready()


@pytest.mark.asyncio
@pytest.mark.parametrize("include_planner", [False, True])
async def test_connector_wait_for_deployment_ready_delegates_to_local_k8s(
    connector_runtime, include_planner
):
    """wait_for_deployment_ready must wait for the pool's own workers to be
    ready before returning. Without this, _async_init returns within
    milliseconds and get_worker_info races with MDC registration, caching a
    fallback WorkerInfo (context_length/max_kv_tokens unset) for the pod's
    lifetime and silently disabling load scaling.

    Parametrized to guard against a refactor that hardcodes ``include_planner``
    or drops the kwarg when forwarding.
    """
    c = GlobalPlannerConnector(connector_runtime, "ns", "gns", "GP", model_name="test")
    fake_local = MagicMock()
    fake_local.wait_for_deployment_ready = AsyncMock()
    c._local_k8s_connector = fake_local
    c._local_k8s_init_attempted = True

    await c.wait_for_deployment_ready(include_planner=include_planner)
    fake_local.wait_for_deployment_ready.assert_awaited_once_with(
        include_planner=include_planner
    )


@pytest.mark.asyncio
async def test_connector_wait_for_deployment_ready_propagates_exceptions(
    connector_runtime,
):
    """A TimeoutError (or similar) from the pool-local wait must propagate so
    _async_init surfaces a broken pool DGD rather than silently swallowing the
    failure and falling back into the original bug (fallback WorkerInfo cached
    forever). Mirrors the standalone environment=kubernetes behavior.
    """
    c = GlobalPlannerConnector(connector_runtime, "ns", "gns", "GP", model_name="test")
    fake_local = MagicMock()
    fake_local.wait_for_deployment_ready = AsyncMock(
        side_effect=TimeoutError("workers not ready")
    )
    c._local_k8s_connector = fake_local
    c._local_k8s_init_attempted = True

    with pytest.raises(TimeoutError, match="workers not ready"):
        await c.wait_for_deployment_ready(include_planner=False)


def test_connector_model_name_and_predicted_load(connector_runtime):
    """Test GlobalPlannerConnector model name and predicted load tracking.

    The ``model_name=None`` branch forces KubernetesConnector init to raise
    so the connector falls back to the "managed-remotely" placeholder
    deterministically, independent of the caller's kube config.
    """
    # With model name — local connector is never consulted.
    c1 = GlobalPlannerConnector(connector_runtime, "ns", "gns", "GP", model_name="test")
    assert c1.get_model_name() == "test"

    # Without model name — force the local connector init to fail so we
    # exercise the fallback deterministically.
    with patch(
        "dynamo.planner.connectors.global_planner.KubernetesConnector",
        side_effect=DeploymentValidationError(["forced test failure"]),
    ):
        c2 = GlobalPlannerConnector(
            connector_runtime, "ns", "gns", "GP", model_name=None
        )
        assert c2.get_model_name() == "managed-remotely"

    # Predicted load
    c1.set_predicted_load(42.0, 256.0, 128.0)
    assert c1.last_predicted_load == {"num_requests": 42.0, "isl": 256.0, "osl": 128.0}


def test_connector_get_worker_info_delegates_to_local_k8s(connector_runtime):
    """get_worker_info should delegate to a pool-local KubernetesConnector
    so that MDC-populated capabilities (context_length, max_kv_tokens, ...)
    reach load-scaling under environment=global-planner.
    """
    c = GlobalPlannerConnector(connector_runtime, "ns", "gns", "GP", model_name="test")
    mdc_info = WorkerInfo(
        k8s_name="VllmPrefillWorker",
        component_name="prefill",
        endpoint="generate",
        context_length=32768,
        total_kv_blocks=1000,
        kv_cache_block_size=16,
    )
    fake_local = MagicMock()
    fake_local.get_worker_info = MagicMock(return_value=mdc_info)
    fake_local.get_model_name = MagicMock(return_value="Qwen/Qwen3-8B")
    c._local_k8s_connector = fake_local
    c._local_k8s_init_attempted = True

    info = c.get_worker_info(SubComponentType.PREFILL, backend="vllm")
    assert info.context_length == 32768
    assert info.max_kv_tokens == 16000
    fake_local.get_worker_info.assert_called_once_with(SubComponentType.PREFILL, "vllm")

    # get_model_name should prefer the init-time value and not call the
    # local connector when one was provided.
    assert c.get_model_name() == "test"
    fake_local.get_model_name.assert_not_called()


def test_connector_get_worker_info_falls_back_on_local_init_failure(connector_runtime):
    """If the pool-local KubernetesConnector can't be created (e.g. outside
    a cluster), get_worker_info should fall back to hard-coded defaults
    rather than raising. Forces the init failure explicitly so the test
    doesn't depend on the caller's kube config.
    """
    with patch(
        "dynamo.planner.connectors.global_planner.KubernetesConnector",
        side_effect=DeploymentValidationError(["forced test failure"]),
    ):
        c = GlobalPlannerConnector(
            connector_runtime, "ns", "gns", "GP", model_name="test"
        )
        info = c.get_worker_info(SubComponentType.PREFILL, backend="vllm")

    # Defaults populate component identifiers but leave capability fields
    # unset — callers use this to detect "no MDC" without crashing.
    assert info.context_length is None
    assert info.max_kv_tokens is None
    assert info.component_name is not None
