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
from unittest.mock import MagicMock, patch, sentinel

import pytest
from kubernetes import client

from dynamo.planner.connectors.clients.kubernetes_api import KubernetesAPI
from dynamo.planner.errors import (
    DuplicateSubComponentError,
    DynamoGraphDeploymentNotFoundError,
    PowerAnnotationMissingError,
    RolloutFailedError,
    SubComponentNotFoundError,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.fixture
def mock_config():
    with patch("dynamo.planner.connectors.clients.kubernetes_api.config") as mock:
        mock.load_incluster_config = MagicMock()
        yield mock


@pytest.fixture
def mock_custom_api():
    with patch(
        "dynamo.planner.connectors.clients.kubernetes_api.client.CustomObjectsApi"
    ) as mock:
        yield mock.return_value


@pytest.fixture
def mock_namespace():
    with patch(
        "dynamo.planner.connectors.clients.kubernetes_api.get_current_k8s_namespace",
        return_value="default",
    ) as mock:
        yield mock


@pytest.fixture
def mock_core_api():
    with patch(
        "dynamo.planner.connectors.clients.kubernetes_api.client.CoreV1Api"
    ) as mock:
        yield mock.return_value


@pytest.fixture
def k8s_api(mock_custom_api, mock_core_api, mock_config, mock_namespace):
    return KubernetesAPI()


@pytest.fixture
def k8s_api_with_namespace(mock_custom_api, mock_config):
    return KubernetesAPI(k8s_namespace="test-namespace")


def test_kubernetes_api_init_with_namespace(mock_custom_api, mock_config):
    """Test KubernetesAPI initialization with custom namespace"""
    api = KubernetesAPI(k8s_namespace="custom-namespace")
    assert api.current_namespace == "custom-namespace"


def test_kubernetes_api_init_without_namespace(
    mock_custom_api, mock_config, mock_namespace
):
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
        version="v1beta1",
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
        version="v1beta1",
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
    mock_custom_api.get_namespaced_custom_object.return_value = {
        "metadata": {"name": "test-deployment"},
        "spec": {
            "components": [
                {"name": "test-component", "type": "decode", "replicas": 0},
                {"name": "other-component", "type": "prefill", "replicas": 2},
            ]
        },
    }
    mock_custom_api.patch_namespaced_custom_object.return_value = None

    k8s_api.update_service_replicas("test-deployment", "test-component", 1)

    # Should have tried DGDSA first
    mock_custom_api.patch_namespaced_custom_object_scale.assert_called_once()

    # Should fall back to a narrow DGD JSON Patch.
    mock_custom_api.patch_namespaced_custom_object.assert_not_called()
    mock_custom_api.api_client.call_api.assert_called_once_with(
        "/apis/{group}/{version}/namespaces/{namespace}/{plural}/{name}",
        "PATCH",
        {
            "group": "nvidia.com",
            "version": "v1beta1",
            "namespace": k8s_api.current_namespace,
            "plural": "dynamographdeployments",
            "name": "test-deployment",
        },
        [],
        {
            "Accept": "application/json",
            "Content-Type": "application/json-patch+json",
        },
        body=[
            {
                "op": "test",
                "path": "/spec/components/0/name",
                "value": "test-component",
            },
            {
                "op": "add",
                "path": "/spec/components/0/replicas",
                "value": 1,
            },
        ],
        response_type="object",
        auth_settings=["BearerToken"],
        _return_http_data_only=True,
        collection_formats={},
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
        version="v1beta1",
        namespace=k8s_api.current_namespace,
        plural="dynamographdeploymentscalingadapters",
        name="test-deployment-test-component",
        body={"spec": {"replicas": 1}},
    )


def test_update_dgd_replicas_directly(k8s_api, mock_custom_api):
    """Test the internal _update_dgd_replicas method"""
    mock_custom_api.get_namespaced_custom_object.return_value = {
        "metadata": {"name": "test-deployment"},
        "spec": {
            "components": [
                {"name": "test-component", "type": "prefill", "replicas": 0},
            ]
        },
    }
    mock_custom_api.patch_namespaced_custom_object.return_value = None

    k8s_api._update_dgd_replicas("test-deployment", "test-component", 1)

    mock_custom_api.patch_namespaced_custom_object.assert_not_called()
    mock_custom_api.api_client.call_api.assert_called_once_with(
        "/apis/{group}/{version}/namespaces/{namespace}/{plural}/{name}",
        "PATCH",
        {
            "group": "nvidia.com",
            "version": "v1beta1",
            "namespace": k8s_api.current_namespace,
            "plural": "dynamographdeployments",
            "name": "test-deployment",
        },
        [],
        {
            "Accept": "application/json",
            "Content-Type": "application/json-patch+json",
        },
        body=[
            {
                "op": "test",
                "path": "/spec/components/0/name",
                "value": "test-component",
            },
            {
                "op": "add",
                "path": "/spec/components/0/replicas",
                "value": 1,
            },
        ],
        response_type="object",
        auth_settings=["BearerToken"],
        _return_http_data_only=True,
        collection_formats={},
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
        settled = await k8s_api.wait_for_graph_deployment_ready(
            "test-deployment", max_attempts=2, delay_seconds=0.1
        )
        assert settled is mock_deployment_ready


def _stable_worker_dgd(
    *,
    generation: int,
    observed_generation: int,
    decode_watts: str = "400",
    dgd_name: str = "test-deployment",
) -> Dict[str, Any]:
    """Production-shaped DGD: stable replica counts, explicit generation lag."""
    return {
        "metadata": {"name": dgd_name, "generation": generation},
        "spec": {
            "components": [
                {
                    "name": "decode",
                    "type": "decode",
                    "replicas": 2,
                    "podTemplate": {
                        "metadata": {
                            "annotations": {
                                "dynamo.nvidia.com/gpu-power-limit": decode_watts
                            }
                        },
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "resources": {"limits": {"nvidia.com/gpu": "1"}},
                                }
                            ]
                        },
                    },
                },
                {"name": "Planner", "type": "planner", "replicas": 1},
            ]
        },
        "status": {
            "observedGeneration": observed_generation,
            "components": {
                "decode": {
                    "readyReplicas": 2,
                    "updatedReplicas": 2,
                    "availableReplicas": 2,
                },
                "Planner": {
                    "readyReplicas": 0,
                    "updatedReplicas": 0,
                    "availableReplicas": 0,
                },
            },
        },
    }


def _make_pod(
    name: str,
    *,
    phase: str = "Running",
    annotation: "str | None" = "300",
    deletion_timestamp=None,
    component: str = "decode",
) -> MagicMock:
    """Build a mock Pod object for pod-annotation settlement tests."""
    pod = MagicMock()
    pod.metadata.name = name
    pod.metadata.annotations = (
        {"dynamo.nvidia.com/gpu-power-limit": annotation}
        if annotation is not None
        else {}
    )
    pod.metadata.labels = {"nvidia.com/dynamo-component": component}
    pod.metadata.deletion_timestamp = deletion_timestamp
    pod.status.phase = phase
    return pod


def _mock_pod_list(mock_core_api, pods: list, *, component: str = "decode"):
    """Wire the single DGD-scoped Pod LIST and label its mock Pods."""
    for pod in pods:
        pod.metadata.labels = {"nvidia.com/dynamo-component": component}
    result = MagicMock()
    result.items = pods
    mock_core_api.list_namespaced_pod.return_value = result


def test_is_spec_generation_observed_requires_catch_up(k8s_api):
    assert (
        k8s_api.is_spec_generation_observed(
            _stable_worker_dgd(generation=2, observed_generation=1)
        )
        is False
    )
    assert (
        k8s_api.is_spec_generation_observed(
            _stable_worker_dgd(generation=2, observed_generation=2)
        )
        is True
    )
    assert k8s_api.is_spec_generation_observed({"status": {}}) is False


@pytest.mark.asyncio
async def test_wait_exclude_planner_rejects_unobserved_generation(
    k8s_api, mock_custom_api
):
    """Annotation-only gen bump: counts look stable, but observedGeneration lags.

    Planner must not treat this snapshot as settled — otherwise a restart can
    cache the gen-2 lower cap while Pods still enforce gen-1.
    """
    lagging = _stable_worker_dgd(
        generation=2, observed_generation=1, decode_watts="300"
    )
    with patch.object(k8s_api, "get_graph_deployment", return_value=lagging):
        with pytest.raises(TimeoutError):
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment",
                include_planner=False,
                require_backing_settled=True,
                require_prefill=False,
                require_decode=True,
                max_attempts=2,
                delay_seconds=0.01,
            )


@pytest.mark.asyncio
async def test_wait_exclude_planner_returns_observed_stable_snapshot(
    k8s_api, mock_core_api
):
    """Gen-2 lower cap is adopted only after observedGeneration catches up."""
    lagging = _stable_worker_dgd(
        generation=2, observed_generation=1, decode_watts="300"
    )
    settled = _stable_worker_dgd(
        generation=2, observed_generation=2, decode_watts="300"
    )
    _mock_pod_list(mock_core_api, [_make_pod("pod-0", annotation="300")])
    with patch.object(k8s_api, "get_graph_deployment", side_effect=[lagging, settled]):
        got = await k8s_api.wait_for_graph_deployment_ready(
            "test-deployment",
            include_planner=False,
            require_backing_settled=True,
            require_prefill=False,
            require_decode=True,
            max_attempts=3,
            delay_seconds=0.01,
        )
    assert got is settled
    assert got["status"]["observedGeneration"] == 2
    assert (
        got["spec"]["components"][0]["podTemplate"]["metadata"]["annotations"][
            "dynamo.nvidia.com/gpu-power-limit"
        ]
        == "300"
    )


# ---------------------------------------------------------------------------
# worker_pods_settled unit tests
# ---------------------------------------------------------------------------


def test_worker_pods_settled_all_match(k8s_api, mock_core_api):
    """All running pods carry the expected annotation → settled."""
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    _mock_pod_list(
        mock_core_api,
        [_make_pod("pod-0", annotation="300"), _make_pod("pod-1", annotation="300")],
    )
    settled, pending = k8s_api.worker_pods_settled(dgd, {"decode": "300"})
    assert settled is True
    assert pending == []


def test_worker_pods_settled_differing_prefill_decode_caps(k8s_api, mock_core_api):
    """Prefill 350 W/GPU and decode 300 W/GPU are independent — both must match."""
    dgd = {
        "metadata": {"name": "test-deployment"},
        "spec": {
            "components": [
                {
                    "name": "prefill",
                    "type": "prefill",
                    "replicas": 2,
                    "podTemplate": {
                        "metadata": {
                            "annotations": {"dynamo.nvidia.com/gpu-power-limit": "350"}
                        },
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "resources": {"limits": {"nvidia.com/gpu": "2"}},
                                }
                            ]
                        },
                    },
                },
                {
                    "name": "decode",
                    "type": "decode",
                    "replicas": 2,
                    "podTemplate": {
                        "metadata": {
                            "annotations": {"dynamo.nvidia.com/gpu-power-limit": "300"}
                        },
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "resources": {"limits": {"nvidia.com/gpu": "4"}},
                                }
                            ]
                        },
                    },
                },
            ]
        },
        "status": {
            "observedGeneration": 2,
            "components": {
                "prefill": {
                    "readyReplicas": 2,
                    "updatedReplicas": 2,
                    "availableReplicas": 2,
                },
                "decode": {
                    "readyReplicas": 2,
                    "updatedReplicas": 2,
                    "availableReplicas": 2,
                },
            },
        },
    }

    result = MagicMock()
    result.items = [
        _make_pod("p-0", annotation="350", component="prefill"),
        _make_pod("p-1", annotation="350", component="prefill"),
        _make_pod("d-0", annotation="300", component="decode"),
        _make_pod("d-1", annotation="300", component="decode"),
    ]
    mock_core_api.list_namespaced_pod.return_value = result
    settled, pending = k8s_api.worker_pods_settled(
        dgd, {"prefill": "350", "decode": "300"}
    )
    assert settled is True
    assert pending == []
    mock_core_api.list_namespaced_pod.assert_called_once()


def test_worker_pods_settled_multi_gpu_replica_compares_per_gpu_annotation(
    k8s_api, mock_core_api
):
    """4-GPU replica at 300 W/GPU: pod annotation is "300", not "1200"."""
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    _mock_pod_list(mock_core_api, [_make_pod("pod-0", annotation="300")])
    settled, pending = k8s_api.worker_pods_settled(dgd, {"decode": "300"})
    assert settled is True
    assert pending == []


def test_worker_pods_settled_terminating_pod_stale_annotation_blocks(
    k8s_api, mock_core_api
):
    """A terminating (DeletionTimestamp set) pod still in Running phase is non-terminal.

    It still consumes GPU power and must block pod-annotation settlement until
    it disappears.
    """
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    terminating = _make_pod(
        "pod-old",
        phase="Running",
        annotation="200",  # stale old cap
        deletion_timestamp=sentinel.some_timestamp,
    )
    _mock_pod_list(mock_core_api, [terminating])
    settled, pending = k8s_api.worker_pods_settled(dgd, {"decode": "300"})
    assert settled is False
    assert any("pod-old" in msg and "terminating" in msg for msg in pending)


def test_worker_pods_settled_succeeded_pod_ignored(k8s_api, mock_core_api):
    """A Succeeded pod is terminal and must not block settlement."""
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    _mock_pod_list(
        mock_core_api,
        [
            _make_pod("pod-ok", phase="Running", annotation="300"),
            _make_pod(
                "pod-done", phase="Succeeded", annotation="200"
            ),  # stale but terminal
        ],
    )
    settled, pending = k8s_api.worker_pods_settled(dgd, {"decode": "300"})
    assert settled is True
    assert pending == []


def test_worker_pods_settled_zero_replicas_no_pods_is_settled(k8s_api, mock_core_api):
    """Scale-to-zero with no pods: nothing enforcing a stale cap → settled."""
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    dgd["spec"]["components"][0]["replicas"] = 0
    _mock_pod_list(mock_core_api, [])
    settled, pending = k8s_api.worker_pods_settled(dgd, {"decode": "300"})
    assert settled is True
    assert pending == []


def test_worker_pods_settled_nonzero_replicas_no_pods_is_pending(
    k8s_api, mock_core_api
):
    """Nonzero desired replicas but no pods yet: settlement must wait."""
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    _mock_pod_list(mock_core_api, [])
    settled, pending = k8s_api.worker_pods_settled(dgd, {"decode": "300"})
    assert settled is False
    assert any("no non-terminal pods" in msg for msg in pending)


def test_worker_pods_settled_missing_annotation_on_running_pod_blocks(
    k8s_api, mock_core_api
):
    """A running pod with no annotation must block settlement."""
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    _mock_pod_list(mock_core_api, [_make_pod("pod-0", annotation=None)])
    settled, pending = k8s_api.worker_pods_settled(dgd, {"decode": "300"})
    assert settled is False
    assert any("pod-0" in msg for msg in pending)


def test_worker_pods_settled_inprogress_rollout_blocks_without_listing_pods(
    k8s_api, mock_core_api
):
    """InProgress DGD rollingUpdate phase blocks settlement before listing pods."""
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    dgd["status"]["rollingUpdate"] = {"phase": "InProgress"}
    settled, pending = k8s_api.worker_pods_settled(dgd, {"decode": "300"})
    assert settled is False
    assert pending == ["rollingUpdate.phase=InProgress"]
    mock_core_api.list_namespaced_pod.assert_not_called()


def test_worker_pods_settled_exact_string_comparison_whitespace(k8s_api, mock_core_api):
    """Exact string comparison: DGD '300' must match pod '300' but not ' 300 '."""
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")

    # Matching: both "300"
    _mock_pod_list(mock_core_api, [_make_pod("pod-0", annotation="300")])
    settled, _ = k8s_api.worker_pods_settled(dgd, {"decode": "300"})
    assert settled is True

    # Mismatch: DGD "300", pod " 300 " (unexpected whitespace on pod side)
    _mock_pod_list(mock_core_api, [_make_pod("pod-0", annotation=" 300 ")])
    settled, pending = k8s_api.worker_pods_settled(dgd, {"decode": "300"})
    assert settled is False
    assert any("300" in msg for msg in pending)


@pytest.mark.asyncio
async def test_wait_exclude_planner_rejects_pods_with_stale_annotation(
    k8s_api, mock_core_api
):
    """DGD observed but pods still carry the old cap → settlement must block."""
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    _mock_pod_list(
        mock_core_api,
        [_make_pod("pod-0", annotation="400")],  # old cap
    )
    with patch.object(k8s_api, "get_graph_deployment", return_value=dgd):
        with pytest.raises(TimeoutError):
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment",
                include_planner=False,
                require_backing_settled=True,
                require_prefill=False,
                require_decode=True,
                max_attempts=2,
                delay_seconds=0.01,
            )


@pytest.mark.asyncio
async def test_wait_exclude_planner_rejects_inprogress_rollout(k8s_api, mock_core_api):
    """End-to-end: InProgress DGD rollingUpdate phase blocks settlement."""
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    dgd["status"]["rollingUpdate"] = {"phase": "InProgress"}
    _mock_pod_list(mock_core_api, [_make_pod("pod-0", annotation="300")])
    with patch.object(k8s_api, "get_graph_deployment", return_value=dgd):
        with pytest.raises(TimeoutError):
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment",
                include_planner=False,
                require_backing_settled=True,
                require_prefill=False,
                require_decode=True,
                max_attempts=2,
                delay_seconds=0.01,
            )


@pytest.mark.asyncio
async def test_wait_missing_dgd_annotation_raises_immediately(k8s_api, mock_core_api):
    """A missing DGD power annotation is a config error, not rollout lag.

    It must raise PowerAnnotationMissingError immediately rather than timing
    out after 30 minutes — the pod list must not even be consulted.
    """
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    # Remove the annotation to simulate a misconfigured DGD.
    del dgd["spec"]["components"][0]["podTemplate"]["metadata"]["annotations"][
        "dynamo.nvidia.com/gpu-power-limit"
    ]
    with patch.object(k8s_api, "get_graph_deployment", return_value=dgd):
        with pytest.raises(PowerAnnotationMissingError):
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment",
                include_planner=False,
                require_backing_settled=True,
                require_prefill=False,
                require_decode=True,
                max_attempts=5,
                delay_seconds=0.01,
            )
    mock_core_api.list_namespaced_pod.assert_not_called()


@pytest.mark.asyncio
async def test_wait_settlement_propagates_missing_role_immediately(
    k8s_api, mock_core_api
):
    """SubComponentNotFoundError after generation is observed must propagate, not retry.

    Once observedGeneration >= generation the DGD spec is stable; a missing
    power-relevant role is a configuration error, not a rollout race. The
    component is stable in status so non_planner_components_stable passes;
    only role resolution fails.
    """
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    # Change type to something the power resolver cannot match as a decode role.
    # The component name and status entry are preserved so stability check passes.
    dgd["spec"]["components"][0]["type"] = "frontend"

    with patch.object(k8s_api, "get_graph_deployment", return_value=dgd):
        with pytest.raises(SubComponentNotFoundError):
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment",
                include_planner=False,
                require_backing_settled=True,
                require_prefill=False,
                require_decode=True,
                max_attempts=5,
                delay_seconds=0.01,
            )


@pytest.mark.asyncio
async def test_wait_settlement_propagates_duplicate_role_immediately(
    k8s_api, mock_core_api
):
    """DuplicateSubComponentError is always a config error and must not be retried."""
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    # Add a second stable decode component to trigger DuplicateSubComponentError.
    dgd["spec"]["components"].append(
        {"name": "decode2", "type": "decode", "replicas": 2}
    )
    dgd["status"]["components"]["decode2"] = {
        "readyReplicas": 2,
        "updatedReplicas": 2,
        "availableReplicas": 2,
    }

    with patch.object(k8s_api, "get_graph_deployment", return_value=dgd):
        with pytest.raises(DuplicateSubComponentError):
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment",
                include_planner=False,
                require_backing_settled=True,
                require_prefill=False,
                require_decode=True,
                max_attempts=5,
                delay_seconds=0.01,
            )


@pytest.mark.asyncio
async def test_wait_exclude_planner_settles_untyped_named_worker(
    k8s_api, mock_core_api
):
    """Explicit-name power worker without ``type`` must still be settlement-gated.

    The power resolver matches untyped components by name
    (``_can_use_explicit_component_name``). An old cap on pods must block
    settlement regardless of how the component was resolved.
    """
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    dgd["spec"]["components"][0].pop("type", None)

    # Old cap on pods → must block.
    _mock_pod_list(mock_core_api, [_make_pod("pod-0", annotation="200")])
    with patch.object(k8s_api, "get_graph_deployment", return_value=dgd):
        with pytest.raises(TimeoutError):
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment",
                include_planner=False,
                require_backing_settled=True,
                require_prefill=False,
                require_decode=True,
                decode_component_name="decode",
                max_attempts=2,
                delay_seconds=0.01,
            )


@pytest.mark.asyncio
async def test_wait_legacy_exclude_planner_does_not_list_pods(k8s_api, mock_core_api):
    """Power-off / legacy wait must not touch Pods.

    ``include_planner=False`` without ``require_backing_settled`` restores the
    pre-power readiness contract: replica-count stability only. Generation lag
    and pod-annotation settlement are power-settlement concerns.
    """
    # Replica-stable, but observedGeneration lags — legacy wait must succeed.
    lagging = _stable_worker_dgd(
        generation=2, observed_generation=1, decode_watts="300"
    )
    with patch.object(k8s_api, "get_graph_deployment", return_value=lagging):
        got = await k8s_api.wait_for_graph_deployment_ready(
            "test-deployment",
            include_planner=False,
            require_backing_settled=False,
            max_attempts=2,
            delay_seconds=0.01,
        )
    assert got is lagging
    mock_core_api.list_namespaced_pod.assert_not_called()


def test_get_graph_deployment(k8s_api, mock_custom_api):
    """Test get_graph_deployment"""
    mock_deployment = {"metadata": {"name": "parent-dgd"}}

    with patch.object(
        k8s_api, "_get_graph_deployment_from_name", return_value=mock_deployment
    ) as mock_get:
        result = k8s_api.get_graph_deployment("parent-dgd")

        assert result == mock_deployment
        mock_get.assert_called_once_with("parent-dgd")


def test_get_graph_deployment_not_found(k8s_api, mock_custom_api):
    """Test get_graph_deployment when deployment is not found"""
    k8s_api.custom_api.get_namespaced_custom_object.side_effect = client.ApiException(
        status=404
    )
    with pytest.raises(DynamoGraphDeploymentNotFoundError) as exc_info:
        k8s_api.get_graph_deployment("parent-dgd")

    exception = exc_info.value
    assert exception.deployment_name == "parent-dgd"
    assert exception.namespace == "default"


# Tests for get_service_replica_status


def test_get_service_replica_status_stable_with_available_replicas(
    k8s_api, mock_custom_api
):
    """Test stable case with availableReplicas present (takes precedence over readyReplicas)"""
    deployment: Dict[str, Any] = {
        "spec": {"components": [{"name": "prefill-worker", "replicas": 2}]},
        "status": {
            "components": {
                "prefill-worker": {
                    "availableReplicas": 2,
                    "readyReplicas": 2,
                    "updatedReplicas": 2,
                }
            }
        },
    }

    count, is_stable = k8s_api.get_service_replica_status(deployment, "prefill-worker")

    assert count == 2
    assert is_stable is True


def test_get_service_replica_status_v1beta_components(k8s_api, mock_custom_api):
    """Test stable case using v1beta1 spec.components/status.components."""
    deployment: Dict[str, Any] = {
        "spec": {
            "components": [
                {
                    "name": "prefill-worker",
                    "type": "prefill",
                    "replicas": 2,
                }
            ]
        },
        "status": {
            "components": {
                "prefill-worker": {
                    "availableReplicas": 2,
                    "readyReplicas": 2,
                    "updatedReplicas": 2,
                }
            }
        },
    }

    count, is_stable = k8s_api.get_service_replica_status(deployment, "prefill-worker")

    assert count == 2
    assert is_stable is True


def test_get_service_replica_status_stable_with_ready_replicas_fallback(
    k8s_api, mock_custom_api
):
    """Test stable case falling back to readyReplicas when availableReplicas is not present"""
    deployment: Dict[str, Any] = {
        "spec": {"components": [{"name": "decode-worker", "replicas": 4}]},
        "status": {
            "components": {
                "decode-worker": {
                    "readyReplicas": 4,
                    "updatedReplicas": 4,
                }
            }
        },
    }

    count, is_stable = k8s_api.get_service_replica_status(deployment, "decode-worker")

    assert count == 4
    assert is_stable is True


def test_get_service_replica_status_scale_up_in_progress(k8s_api, mock_custom_api):
    """Test scale-up in progress: desired=4, updated=2, ready=2"""
    deployment: Dict[str, Any] = {
        "spec": {"components": [{"name": "prefill-worker", "replicas": 4}]},
        "status": {
            "components": {
                "prefill-worker": {
                    "availableReplicas": 2,
                    "readyReplicas": 2,
                    "updatedReplicas": 2,
                }
            }
        },
    }

    count, is_stable = k8s_api.get_service_replica_status(deployment, "prefill-worker")

    assert count == 2
    assert is_stable is False


def test_get_service_replica_status_scale_down_in_progress(k8s_api, mock_custom_api):
    """Test scale-down in progress: desired=2, updated=4, ready=4"""
    deployment: Dict[str, Any] = {
        "spec": {"components": [{"name": "decode-worker", "replicas": 2}]},
        "status": {
            "components": {
                "decode-worker": {
                    "availableReplicas": 4,
                    "readyReplicas": 4,
                    "updatedReplicas": 4,
                }
            }
        },
    }

    count, is_stable = k8s_api.get_service_replica_status(deployment, "decode-worker")

    assert count == 4
    assert is_stable is False


def test_get_service_replica_status_rollout_in_progress(k8s_api, mock_custom_api):
    """Test rollout in progress: desired=4, updated=2, ready=4 (old replicas still running)"""
    deployment: Dict[str, Any] = {
        "spec": {"components": [{"name": "prefill-worker", "replicas": 4}]},
        "status": {
            "components": {
                "prefill-worker": {
                    "availableReplicas": 4,
                    "readyReplicas": 4,
                    "updatedReplicas": 2,
                }
            }
        },
    }

    count, is_stable = k8s_api.get_service_replica_status(deployment, "prefill-worker")

    assert count == 4
    assert is_stable is False


def test_get_service_replica_status_missing_status_fields(k8s_api, mock_custom_api):
    """Test handling when status fields are missing"""
    deployment: Dict[str, Any] = {
        "spec": {"components": [{"name": "prefill-worker", "replicas": 2}]},
        "status": {"components": {}},
    }

    count, is_stable = k8s_api.get_service_replica_status(deployment, "prefill-worker")

    # Should default to 0 for missing fields
    assert count == 0
    # desired=2, updated=0, count=0 -> not stable
    assert is_stable is False


def test_get_service_replica_status_empty_deployment(k8s_api, mock_custom_api):
    """Test handling when deployment has no spec or status"""
    deployment: Dict[str, Any] = {}

    count, is_stable = k8s_api.get_service_replica_status(deployment, "prefill-worker")

    # All values default to 0, which makes it "stable" (0 == 0 == 0)
    assert count == 0
    assert is_stable is True


def test_get_service_replica_status_available_replicas_zero(k8s_api, mock_custom_api):
    """Test when availableReplicas is explicitly 0 (should use 0, not fall back to ready)"""
    deployment: Dict[str, Any] = {
        "spec": {"components": [{"name": "prefill-worker", "replicas": 0}]},
        "status": {
            "components": {
                "prefill-worker": {
                    "availableReplicas": 0,
                    "readyReplicas": 2,  # Should be ignored
                    "updatedReplicas": 0,
                }
            }
        },
    }

    count, is_stable = k8s_api.get_service_replica_status(deployment, "prefill-worker")

    # availableReplicas=0 should be used (not readyReplicas)
    assert count == 0
    assert is_stable is True


def test_has_terminating_pods_true_when_running_pod_has_deletion_timestamp(
    k8s_api, mock_core_api
):
    """has_terminating_pods returns True when a Running pod has a deletionTimestamp.

    Deployment status excludes terminating pods. During a decode scale-down
    (4→1) the Deployment can report desired=updated=available=1 while 3 old
    decode pods linger with a deletionTimestamp. get_actual_worker_counts calls
    has_terminating_pods to detect this blind spot and return is_stable=False,
    preventing the planner from admitting the opposing scale-up before old pods
    are fully gone.  get_service_replica_status stays pod-free intentionally so
    non-power paths do not acquire the pods/list RBAC surface.
    """
    terminating = _make_pod(
        "pod-old",
        phase="Running",
        deletion_timestamp=sentinel.ts,
    )

    assert k8s_api.has_terminating_pods([terminating]) is True
    mock_core_api.list_namespaced_pod.assert_not_called()


def test_has_terminating_pods_false_when_no_deletion_timestamp(k8s_api, mock_core_api):
    """has_terminating_pods returns False when all pods are cleanly Running."""
    running = _make_pod("pod-0", phase="Running")

    assert k8s_api.has_terminating_pods([running]) is False
    mock_core_api.list_namespaced_pod.assert_not_called()


def test_list_and_partition_pods_uses_one_dgd_scoped_request(k8s_api, mock_core_api):
    prefill = _make_pod("prefill-0", component="prefill")
    decode = _make_pod("decode-0", component="decode")
    result = MagicMock()
    result.items = [prefill, decode]
    mock_core_api.list_namespaced_pod.return_value = result

    pods = k8s_api.list_pods_for_graph("my-dgd")
    by_component = k8s_api.partition_pods_by_component(pods)

    mock_core_api.list_namespaced_pod.assert_called_once_with(
        namespace="default",
        label_selector="nvidia.com/dynamo-graph-deployment-name=my-dgd",
    )
    assert by_component == {
        "prefill": [prefill],
        "decode": [decode],
    }


def test_worker_pods_settled_terminating_pod_correct_annotation_blocks(
    k8s_api, mock_core_api
):
    """A terminating pod carrying the *expected* annotation must still block settlement.

    P1 regression: before the fix, a terminating pod whose annotation matched
    the DGD snapshot was silently accepted, letting the planner admit a scale-up
    while the old pod was still Running and consuming power.
    """
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    terminating = _make_pod(
        "pod-old",
        phase="Running",
        annotation="300",  # matches expected — was incorrectly accepted before fix
        deletion_timestamp=sentinel.ts,
    )
    _mock_pod_list(mock_core_api, [terminating])
    settled, pending = k8s_api.worker_pods_settled(dgd, {"decode": "300"})
    assert settled is False
    assert any("pod-old" in msg and "terminating" in msg for msg in pending)


@pytest.mark.asyncio
async def test_wait_failed_rollout_raises_immediately(k8s_api, mock_core_api):
    """A Failed rollingUpdate phase must raise RolloutFailedError immediately.

    Failed is a terminal operator state (endTime is set). Retrying until the
    generic 30-minute timeout wastes time and hides the root cause.
    """
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    dgd["status"]["rollingUpdate"] = {
        "phase": "Failed",
        "message": "pod CrashLoopBackOff",
    }
    with patch.object(k8s_api, "get_graph_deployment", return_value=dgd):
        with pytest.raises(RolloutFailedError) as exc_info:
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment",
                include_planner=False,
                require_backing_settled=True,
                require_prefill=False,
                require_decode=True,
                max_attempts=10,
                delay_seconds=0.01,
            )
    assert "test-deployment" in str(exc_info.value)
    assert "Failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_wait_failed_rollout_raises_immediately_while_replicas_unstable(
    k8s_api, mock_core_api
):
    """Failed + unstable replicas must raise RolloutFailedError on attempt 1.

    The stability gate (non_planner_components_stable) must not run before the
    Failed check on the require_backing_settled path. If it does, the loop
    continues indefinitely and the caller gets a generic TimeoutError instead.
    """
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    # Make replicas unstable (desired=2, ready=1) so the old ordering would
    # skip the Failed check via the stability-gate continue.
    dgd["status"]["components"]["decode"]["readyReplicas"] = 1
    dgd["status"]["rollingUpdate"] = {
        "phase": "Failed",
        "message": "pod CrashLoopBackOff",
    }
    with patch.object(k8s_api, "get_graph_deployment", return_value=dgd):
        with pytest.raises(RolloutFailedError) as exc_info:
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment",
                include_planner=False,
                require_backing_settled=True,
                require_prefill=False,
                require_decode=True,
                max_attempts=3,
                delay_seconds=0.01,
            )
    assert "test-deployment" in str(exc_info.value)
    assert "CrashLoopBackOff" in str(exc_info.value)


@pytest.mark.asyncio
async def test_wait_legacy_failed_rollout_does_not_raise(k8s_api, mock_core_api):
    """Power-disabled (require_backing_settled=False) must not raise on a Failed rollout.

    The legacy replica-stability contract predates the rolling-update field; a
    power-disabled planner whose replica counts are stable must start
    successfully even when rollingUpdate.phase is Failed.
    """
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    dgd["status"]["rollingUpdate"] = {
        "phase": "Failed",
        "message": "pod CrashLoopBackOff",
    }
    with patch.object(k8s_api, "get_graph_deployment", return_value=dgd):
        got = await k8s_api.wait_for_graph_deployment_ready(
            "test-deployment",
            include_planner=False,
            require_backing_settled=False,
            max_attempts=2,
            delay_seconds=0.01,
        )
    assert got is dgd
    mock_core_api.list_namespaced_pod.assert_not_called()


@pytest.mark.asyncio
async def test_wait_settlement_verbatim_annotation_propagation(k8s_api, mock_core_api):
    """DGD annotation ' 350 ' propagates verbatim to Pods; settlement uses raw string.

    The operator copies the raw DGD podTemplate annotation onto each Pod without
    normalization, so expected_power stores the raw string ' 350 ' and the
    settlement comparison succeeds only when the Pod carries that same raw value.

    ' 350 ' (DGD) + ' 350 ' (Pod) → settles (operator verbatim copy).
    ' 350 ' (DGD) + '350'   (Pod) → does NOT settle (canonical form differs).
    """
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts=" 350 ")

    # Verbatim match: pod carries the raw DGD annotation string.
    _mock_pod_list(mock_core_api, [_make_pod("pod-0", annotation=" 350 ")])
    with patch.object(k8s_api, "get_graph_deployment", return_value=dgd):
        got = await k8s_api.wait_for_graph_deployment_ready(
            "test-deployment",
            include_planner=False,
            require_backing_settled=True,
            require_prefill=False,
            require_decode=True,
            max_attempts=3,
            delay_seconds=0.01,
        )
    assert got is dgd

    # Canonical form "350" does not match the raw DGD annotation " 350 ".
    _mock_pod_list(mock_core_api, [_make_pod("pod-0", annotation="350")])
    with patch.object(k8s_api, "get_graph_deployment", return_value=dgd):
        with pytest.raises(TimeoutError):
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment",
                include_planner=False,
                require_backing_settled=True,
                require_prefill=False,
                require_decode=True,
                max_attempts=3,
                delay_seconds=0.01,
            )
