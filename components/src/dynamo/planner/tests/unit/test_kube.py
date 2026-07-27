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

from dynamo.planner.connectors.clients.kubernetes_api import KubernetesAPI
from dynamo.planner.errors import DynamoGraphDeploymentNotFoundError

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
def k8s_api(mock_custom_api, mock_config, mock_namespace):
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
                    "name": "VllmDecodeWorker",
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
                "VllmDecodeWorker": {
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


def _ready_dcd(*, generation: int = 1, observed_generation: int = 1) -> Dict[str, Any]:
    return {
        "metadata": {"generation": generation},
        "status": {
            "observedGeneration": observed_generation,
            "conditions": [{"type": "Available", "status": "True"}],
        },
    }


def _mock_dcd_lookup(
    mock_custom_api, dcd: Dict[str, Any], dgd_name: str = "test-deployment"
):
    """Wire CustomObjectsApi to return ``dcd`` for the decode worker DCD name."""

    def _lookup(*args, **kwargs):
        plural = kwargs.get("plural")
        name = kwargs.get("name")
        if (
            plural == "dynamocomponentdeployments"
            and name == f"{dgd_name}-vllmdecodeworker"
        ):
            return dcd
        raise client.ApiException(status=404)

    mock_custom_api.get_namespaced_custom_object.side_effect = _lookup


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
    k8s_api, mock_custom_api
):
    """Gen-2 lower cap is adopted only after observedGeneration catches up."""
    lagging = _stable_worker_dgd(
        generation=2, observed_generation=1, decode_watts="300"
    )
    settled = _stable_worker_dgd(
        generation=2, observed_generation=2, decode_watts="300"
    )
    _mock_dcd_lookup(mock_custom_api, _ready_dcd(generation=2, observed_generation=2))
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


@pytest.mark.asyncio
async def test_wait_exclude_planner_rejects_dgd_observed_while_dcd_lags(
    k8s_api, mock_custom_api
):
    """DGD observedGeneration can advance before the worker DCD rolls Pods."""
    dgd_observed = _stable_worker_dgd(
        generation=2, observed_generation=2, decode_watts="300"
    )
    lagging_dcd = _ready_dcd(generation=2, observed_generation=1)
    lagging_dcd["status"]["conditions"] = []
    _mock_dcd_lookup(mock_custom_api, lagging_dcd)
    with patch.object(k8s_api, "get_graph_deployment", return_value=dgd_observed):
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


def test_is_dcd_ready_requires_generation_and_available(k8s_api):
    assert k8s_api.is_dcd_ready(_ready_dcd(generation=2, observed_generation=2))
    assert not k8s_api.is_dcd_ready(_ready_dcd(generation=2, observed_generation=1))
    missing_available = _ready_dcd(generation=2, observed_generation=2)
    missing_available["status"]["conditions"] = []
    assert not k8s_api.is_dcd_ready(missing_available)


def test_worker_backing_resources_settled_rejects_lagging_dcd(k8s_api, mock_custom_api):
    dgd = _stable_worker_dgd(generation=2, observed_generation=2)
    _mock_dcd_lookup(
        mock_custom_api,
        _ready_dcd(generation=2, observed_generation=1),
    )
    settled, pending = k8s_api.worker_backing_resources_settled(
        dgd, ["VllmDecodeWorker"]
    )
    assert settled is False
    assert pending == [
        "VllmDecodeWorker: DCD test-deployment-vllmdecodeworker not ready "
        "(generation=2, observedGeneration=1)"
    ]


def test_worker_backing_resources_settled_rejects_lagging_pod_clique(
    k8s_api, mock_custom_api
):
    """Grove path: DGD counters can look stable while PodClique gen still lags."""
    dgd = _stable_worker_dgd(generation=2, observed_generation=2)
    dgd["status"]["components"]["VllmDecodeWorker"] = {
        "componentKind": "PodClique",
        "componentNames": ["test-deployment-vllmdecodeworker"],
        "readyReplicas": 2,
        "updatedReplicas": 2,
        "availableReplicas": 2,
    }

    def _lookup(*args, **kwargs):
        if (
            kwargs.get("plural") == "podcliques"
            and kwargs.get("name") == "test-deployment-vllmdecodeworker"
        ):
            return {
                "metadata": {"generation": 2},
                "spec": {"replicas": 2},
                "status": {
                    "observedGeneration": 1,
                    "replicas": 2,
                    "updatedReplicas": 2,
                    "readyReplicas": 2,
                },
            }
        raise client.ApiException(status=404)

    mock_custom_api.get_namespaced_custom_object.side_effect = _lookup
    settled, pending = k8s_api.worker_backing_resources_settled(
        dgd, ["VllmDecodeWorker"]
    )
    assert settled is False
    assert pending == [
        "VllmDecodeWorker: PodClique test-deployment-vllmdecodeworker "
        "not generation-ready"
    ]


def test_worker_backing_rejects_inprogress_rollout_with_ready_old_dcd(
    k8s_api, mock_custom_api
):
    """InProgress rollout: current-hash still names the ready old DCD.

    DGD observedGeneration and replica counters can look settled while the
    annotation points at the old revision and the new DCD is missing. The
    wait must not treat the old DCD as proof the desired revision rolled.

    Deployment componentNames use the real workload shape (``…-deployment``);
    settlement never treats those strings as DCD names.
    """
    old_dcd = "test-deployment-vllmdecodeworker-oldhash1"
    new_dcd = "test-deployment-vllmdecodeworker-newhash2"
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    dgd["metadata"]["annotations"] = {
        "nvidia.com/current-worker-hash-v2": "oldhash1",
    }
    dgd["status"]["rollingUpdate"] = {"phase": "InProgress"}
    dgd["status"]["components"]["VllmDecodeWorker"] = {
        "componentKind": "Deployment",
        "componentNames": [f"{new_dcd}-deployment", f"{old_dcd}-deployment"],
        "readyReplicas": 2,
        "updatedReplicas": 2,
        "availableReplicas": 2,
    }

    def _lookup(*args, **kwargs):
        if kwargs.get("plural") != "dynamocomponentdeployments":
            raise client.ApiException(status=404)
        if kwargs.get("name") == old_dcd:
            return _ready_dcd(generation=1, observed_generation=1)
        raise client.ApiException(status=404)

    mock_custom_api.get_namespaced_custom_object.side_effect = _lookup
    settled, pending = k8s_api.worker_backing_resources_settled(
        dgd, ["VllmDecodeWorker"]
    )
    assert settled is False
    assert pending == ["rollingUpdate.phase=InProgress"]


def test_worker_backing_rejects_lagging_hash_derived_dcd_despite_workload_names(
    k8s_api, mock_custom_api
):
    """Deployment componentNames are workload names; settlement uses hash-derived DCD.

    Real operator status puts ``…-deployment`` into componentNames. A lagging
    hash-derived DCD must still block settlement even when those workload
    names are present (and even if they would 404 as DCD lookups).
    """
    old_dcd = "test-deployment-vllmdecodeworker-oldhash1"
    new_dcd = "test-deployment-vllmdecodeworker-newhash2"
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    dgd["metadata"]["annotations"] = {
        "nvidia.com/current-worker-hash-v2": "oldhash1",
    }
    dgd["status"]["components"]["VllmDecodeWorker"] = {
        "componentKind": "Deployment",
        "componentNames": [f"{new_dcd}-deployment", f"{old_dcd}-deployment"],
        "readyReplicas": 2,
        "updatedReplicas": 2,
        "availableReplicas": 2,
    }

    def _lookup(*args, **kwargs):
        if kwargs.get("plural") != "dynamocomponentdeployments":
            raise client.ApiException(status=404)
        if kwargs.get("name") == old_dcd:
            return _ready_dcd(generation=2, observed_generation=1)
        raise client.ApiException(status=404)

    mock_custom_api.get_namespaced_custom_object.side_effect = _lookup
    settled, pending = k8s_api.worker_backing_resources_settled(
        dgd, ["VllmDecodeWorker"]
    )
    assert settled is False
    assert pending == [
        f"VllmDecodeWorker: DCD {old_dcd} not ready "
        "(generation=2, observedGeneration=1)"
    ]


def test_worker_backing_ignores_deployment_workload_component_names(
    k8s_api, mock_custom_api
):
    """Regression: Deployment componentNames must not be used as DCD names.

    Operator status for a DCD-backed worker looks like
    ``componentNames: [<dcd>-deployment]`` while the DCD itself is ``<dcd>``.
    Settlement must GET the derived DCD, never the workload name.
    """
    dcd_name = "test-deployment-vllmdecodeworker"
    workload_name = f"{dcd_name}-deployment"
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    dgd["status"]["components"]["VllmDecodeWorker"] = {
        "componentKind": "Deployment",
        "componentNames": [workload_name],
        "readyReplicas": 2,
        "updatedReplicas": 2,
        "availableReplicas": 2,
    }

    queried: list[str] = []

    def _lookup(*args, **kwargs):
        if kwargs.get("plural") != "dynamocomponentdeployments":
            raise client.ApiException(status=404)
        name = kwargs.get("name")
        queried.append(name)
        if name == dcd_name:
            return _ready_dcd(generation=2, observed_generation=2)
        raise client.ApiException(status=404)

    mock_custom_api.get_namespaced_custom_object.side_effect = _lookup
    settled, pending = k8s_api.worker_backing_resources_settled(
        dgd, ["VllmDecodeWorker"]
    )
    assert settled is True
    assert pending == []
    assert dcd_name in queried
    assert workload_name not in queried


@pytest.mark.asyncio
async def test_wait_exclude_planner_rejects_inprogress_rollout_ready_old_dcd(
    k8s_api, mock_custom_api
):
    """End-to-end wait: InProgress + ready old DCD must not settle."""
    old_dcd = "test-deployment-vllmdecodeworker-oldhash1"
    new_dcd = "test-deployment-vllmdecodeworker-newhash2"
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    dgd["metadata"]["annotations"] = {
        "nvidia.com/current-worker-hash-v2": "oldhash1",
    }
    dgd["status"]["rollingUpdate"] = {"phase": "InProgress"}
    dgd["status"]["components"]["VllmDecodeWorker"] = {
        "componentKind": "Deployment",
        "componentNames": [f"{new_dcd}-deployment", f"{old_dcd}-deployment"],
        "readyReplicas": 2,
        "updatedReplicas": 2,
        "availableReplicas": 2,
    }

    def _lookup(*args, **kwargs):
        if (
            kwargs.get("plural") == "dynamocomponentdeployments"
            and kwargs.get("name") == old_dcd
        ):
            return _ready_dcd(generation=1, observed_generation=1)
        raise client.ApiException(status=404)

    mock_custom_api.get_namespaced_custom_object.side_effect = _lookup
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


def test_worker_backing_settles_untyped_named_worker(k8s_api, mock_custom_api):
    """Explicit-name power workers without ``type`` must still be settlement-gated.

    The power resolver matches untyped components by name
    (``_can_use_explicit_component_name``). A type-only backing filter would
    skip them and allow caching a lower gen-N cap while the DCD still
    enforces gen-N-1.
    """
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    # Drop type so only the explicit name resolves the role.
    dgd["spec"]["components"][0].pop("type", None)
    _mock_dcd_lookup(
        mock_custom_api,
        _ready_dcd(generation=2, observed_generation=1),
    )
    settled, pending = k8s_api.worker_backing_resources_settled(
        dgd, ["VllmDecodeWorker"]
    )
    assert settled is False
    assert pending == [
        "VllmDecodeWorker: DCD test-deployment-vllmdecodeworker not ready "
        "(generation=2, observedGeneration=1)"
    ]


@pytest.mark.asyncio
async def test_wait_settled_rejects_untyped_named_worker_with_lagging_dcd(
    k8s_api, mock_custom_api
):
    """End-to-end settled wait: untyped named decode + lagging DCD must not pass."""
    dgd = _stable_worker_dgd(generation=2, observed_generation=2, decode_watts="300")
    dgd["spec"]["components"][0].pop("type", None)
    lagging_dcd = _ready_dcd(generation=2, observed_generation=1)
    lagging_dcd["status"]["conditions"] = []
    _mock_dcd_lookup(mock_custom_api, lagging_dcd)
    with patch.object(k8s_api, "get_graph_deployment", return_value=dgd):
        with pytest.raises(TimeoutError):
            await k8s_api.wait_for_graph_deployment_ready(
                "test-deployment",
                include_planner=False,
                require_backing_settled=True,
                require_prefill=False,
                require_decode=True,
                decode_component_name="VllmDecodeWorker",
                max_attempts=2,
                delay_seconds=0.01,
            )


@pytest.mark.asyncio
async def test_wait_legacy_exclude_planner_does_not_query_backing(
    k8s_api, mock_custom_api
):
    """Power-off / legacy wait must not touch DCD or Grove CRs.

    ``include_planner=False`` without ``require_backing_settled`` restores the
    pre-power readiness contract: replica-count stability only. Generation lag
    and backing CR readiness are power-settlement concerns.
    """
    # Replica-stable, but observedGeneration lags and no DCD is registered —
    # legacy wait must still succeed without issuing a backing GET.
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
    mock_custom_api.get_namespaced_custom_object.assert_not_called()


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
