# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for the fail-fast behavior added for #9213.

When a candidate deployment's worker pods enter ``CrashLoopBackOff``,
``DynamoDeploymentClient.wait_for_deployment_ready`` must raise
``DeploymentFailedError`` immediately rather than waiting out the full
``timeout`` — otherwise the thorough-mode profiler burns up to 30 min
of wall-clock per failing candidate.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

# Skip the whole module if the deploy.utils runtime deps aren't available
# in this environment. The test doesn't actually exercise these — it just
# needs the import of `dynamo_deployment` to succeed.
pytest.importorskip("aiofiles")
pytest.importorskip("kubernetes_asyncio")
pytest.importorskip("httpx")

from kubernetes_asyncio import client as k8s_client  # noqa: E402

from deploy.utils.dynamo_deployment import (  # noqa: E402
    DeploymentFailedError,
    DynamoDeploymentClient,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.timeout(5)]


async def test_wait_for_deployment_ready_raises_deployment_failed_on_crashloop(
    monkeypatch,
):
    client = DynamoDeploymentClient(namespace="ns", deployment_name="dgd-test")
    client.deployment_name = "dgd-test"
    client._original_components = ["PrefillWorker"]
    client.components = ["prefillworker"]

    # DGD CR exists but isn't Ready yet.
    client.custom_api = MagicMock()
    client.custom_api.get_namespaced_custom_object = AsyncMock(
        return_value={"status": {"state": "deploying", "conditions": []}}
    )
    # Simulate a crash on the very first poll.
    client._detect_terminal_pod_failure = AsyncMock(  # type: ignore[method-assign]
        return_value="pod p0 container worker in CrashLoopBackOff"
    )

    # Any retry is a regression; never enter the real polling delay.
    monkeypatch.setattr(
        "deploy.utils.dynamo_deployment.asyncio.sleep",
        AsyncMock(side_effect=AssertionError("unexpected readiness retry")),
    )

    with pytest.raises(DeploymentFailedError) as excinfo:
        # Pass a generous timeout so a regression (timeout instead of
        # raise) would be obvious.
        await client.wait_for_deployment_ready(timeout=600)

    assert "CrashLoopBackOff" in str(excinfo.value)
    # Confirm we didn't run out the timeout — there should have been at
    # most one DGD status check before the raise.
    assert client.custom_api.get_namespaced_custom_object.await_count == 1


def _mocked_client() -> DynamoDeploymentClient:
    client = DynamoDeploymentClient(namespace="ns", deployment_name="dgd-test")
    client._init_kubernetes = AsyncMock()  # type: ignore[method-assign]
    client.custom_api = MagicMock()
    client.custom_api.create_namespaced_custom_object = AsyncMock()
    client.custom_api.delete_namespaced_custom_object = AsyncMock()
    return client


async def test_create_deployment_reads_v1beta1_components():
    """A v1beta1 candidate must deploy without a KeyError.

    ``materialize_dgd`` has produced ``spec.components`` — a list of objects
    each carrying a ``name`` — since DGD generation moved to v1beta1, while
    this client still read the v1alpha1 ``spec.services`` mapping.
    """
    client = _mocked_client()

    await client.create_deployment(
        {
            "apiVersion": "nvidia.com/v1beta1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": "candidate", "namespace": "ns"},
            "spec": {
                "components": [
                    {"name": "Frontend"},
                    {"name": "VllmPrefillWorker"},
                ]
            },
        }
    )

    # Original case drives the nvidia.com/dynamo-component label selector;
    # the lowercase projection names the per-component log directory.
    assert client._original_components == ["Frontend", "VllmPrefillWorker"]
    assert client.components == ["frontend", "vllmprefillworker"]

    create_kwargs = client.custom_api.create_namespaced_custom_object.await_args.kwargs
    assert create_kwargs["version"] == "v1beta1"

    await client.delete_deployment()
    delete_kwargs = client.custom_api.delete_namespaced_custom_object.await_args.kwargs
    assert delete_kwargs["version"] == "v1beta1"


async def test_create_deployment_owner_reference_targets_v1beta1_dgdr(monkeypatch):
    """The owner reference must name a DGDR version the API server serves.

    Profiling DGDs are garbage-collected through this reference when the
    owning DynamoGraphDeploymentRequest is deleted. The DGDR CRD stores
    v1beta1, so a reference declaring v1alpha1 would leak every profiling
    deployment once v1alpha1 stops being served.
    """
    monkeypatch.setenv("DGDR_NAME", "dgdr-test")
    monkeypatch.setenv("DGDR_NAMESPACE", "ns")
    monkeypatch.setenv("DGDR_UID", "8f0b0f4e-0000-4000-8000-000000000000")

    client = _mocked_client()

    await client.create_deployment(
        {
            "apiVersion": "nvidia.com/v1beta1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": "candidate", "namespace": "ns"},
            "spec": {"components": [{"name": "Frontend"}]},
        }
    )

    create_kwargs = client.custom_api.create_namespaced_custom_object.await_args.kwargs
    owner_references = create_kwargs["body"]["metadata"]["ownerReferences"]
    assert owner_references[0]["apiVersion"] == "nvidia.com/v1beta1"
    assert owner_references[0]["kind"] == "DynamoGraphDeploymentRequest"
    assert owner_references[0]["name"] == "dgdr-test"


@pytest.mark.unit
@pytest.mark.gpu_0
@pytest.mark.parametrize(
    ("deployment_name", "service_name", "owner_kind"),
    [
        (
            "70b-llama-production",
            "svc-70b-llama-production-abcd-frontend",
            "DynamoComponentDeployment",
        ),
        (
            "70b-llama-production",
            "70b-llama-production-abcd-frontend",
            "DynamoGraphDeployment",
        ),
        ("7" + "a" * 49, "svc-7" + "a" * 53 + "-beef", "DynamoComponentDeployment"),
        ("graph.with.dots", "graph-with-dots-abcd-frontend", "DynamoGraphDeployment"),
    ],
)
async def test_ready_deployment_uses_actual_owned_frontend_service(
    monkeypatch, deployment_name, service_name, owner_kind
):
    """Both connection methods must use the rendered or retained Service name."""
    monkeypatch.setattr("deploy.utils.dynamo_deployment.uuid.uuid4", lambda: "abcd")
    client = DynamoDeploymentClient(namespace="ns", deployment_name=deployment_name)
    monkeypatch.setattr(
        "deploy.utils.dynamo_deployment.asyncio.sleep",
        AsyncMock(side_effect=AssertionError("unexpected readiness retry")),
    )
    graph_uid = "graph-uid"
    component_uid = "component-uid"
    component_name = f"{client.deployment_name}-frontend"
    service = k8s_client.V1Service(
        metadata=k8s_client.V1ObjectMeta(
            name=service_name,
            # Grove Services need not carry the DCD pathway's graph label.
            labels={"nvidia.com/dynamo-graph-deployment-name": client.deployment_name}
            if owner_kind == "DynamoComponentDeployment"
            else None,
            owner_references=[
                k8s_client.V1OwnerReference(
                    api_version="nvidia.com/v1beta1",
                    kind=owner_kind,
                    name=component_name
                    if owner_kind == "DynamoComponentDeployment"
                    else client.deployment_name,
                    uid=component_uid
                    if owner_kind == "DynamoComponentDeployment"
                    else graph_uid,
                    controller=True,
                )
            ],
        ),
        spec=k8s_client.V1ServiceSpec(
            selector={"nvidia.com/dynamo-component-type": "frontend"}
        ),
    )
    client.core_api = MagicMock()
    unowned_service = k8s_client.V1Service(
        metadata=k8s_client.V1ObjectMeta(name="unowned-frontend"),
        spec=k8s_client.V1ServiceSpec(
            selector={"nvidia.com/dynamo-component-type": "frontend"}
        ),
    )
    worker_service = k8s_client.V1Service(
        metadata=k8s_client.V1ObjectMeta(
            name="worker", owner_references=service.metadata.owner_references
        ),
        spec=k8s_client.V1ServiceSpec(
            selector={"nvidia.com/dynamo-component-type": "worker"}
        ),
    )

    async def list_services(namespace, label_selector=None):
        """Apply Kubernetes equality selectors to the rendered Service labels."""
        items = [
            unowned_service,
            worker_service,
            _owned_frontend_service(
                "other-graph-frontend", owner_uid="other-graph-uid"
            ),
            service,
        ]
        if label_selector:
            labels = dict(term.split("=", 1) for term in label_selector.split(","))
            items = [
                item
                for item in items
                if all(
                    (item.metadata.labels or {}).get(key) == value
                    for key, value in labels.items()
                )
            ]
        return k8s_client.V1ServiceList(items=items)

    client.core_api.list_namespaced_service = AsyncMock(side_effect=list_services)
    client.custom_api = MagicMock()

    async def read_resource(**kwargs):
        """Return a ready graph or the component that owns its Service."""
        if kwargs["plural"] == "dynamographdeployments":
            return {
                "metadata": {"uid": graph_uid},
                "status": {
                    "state": "successful",
                    "conditions": [{"type": "Ready", "status": "True"}],
                },
            }
        assert kwargs["plural"] == "dynamocomponentdeployments"
        assert kwargs["name"] == component_name
        return {
            "metadata": {
                "uid": component_uid,
                "ownerReferences": [
                    {
                        "apiVersion": "nvidia.com/v1beta1",
                        "kind": "DynamoGraphDeployment",
                        "uid": graph_uid,
                        "controller": True,
                    }
                ],
            }
        }

    client.custom_api.get_namespaced_custom_object = AsyncMock(
        side_effect=read_resource
    )
    assert await client.wait_for_deployment_ready(timeout=1) is True
    assert (
        client.get_service_url()
        == f"http://{service_name}.ns.svc.cluster.local:{client.frontend_port}"
    )
    client.core_api.list_namespaced_service.assert_awaited_once_with(
        namespace="ns",
    )

    popen = MagicMock()
    monkeypatch.setattr("deploy.utils.dynamo_deployment.subprocess.Popen", popen)
    monkeypatch.setattr("deploy.utils.dynamo_deployment.time.sleep", lambda _: None)
    client.port_forward_frontend(local_port=client.frontend_port + 1, quiet=True)
    assert popen.call_args.args[0][2] == f"svc/{service_name}"


@pytest.mark.unit
@pytest.mark.gpu_0
async def test_ready_deployment_preserves_explicit_service_name(monkeypatch):
    """An explicit endpoint must not require Service discovery permissions."""
    monkeypatch.setattr(
        "deploy.utils.dynamo_deployment.asyncio.sleep",
        AsyncMock(side_effect=AssertionError("unexpected readiness retry")),
    )
    client = DynamoDeploymentClient(namespace="ns", service_name="custom-endpoint")
    client.custom_api = MagicMock()
    client.custom_api.get_namespaced_custom_object = AsyncMock(
        return_value={
            "status": {
                "state": "successful",
                "conditions": [{"type": "Ready", "status": "True"}],
            }
        }
    )
    client._resolve_frontend_service = AsyncMock()  # type: ignore[method-assign]

    assert await client.wait_for_deployment_ready(timeout=1) is True
    client._resolve_frontend_service.assert_not_awaited()
    assert client.service_name == "custom-endpoint"


def _owned_frontend_service(
    name, owner_kind="DynamoGraphDeployment", owner_uid="graph-uid"
):
    """Build a frontend Service with the operator's controlling owner reference."""
    return k8s_client.V1Service(
        metadata=k8s_client.V1ObjectMeta(
            name=name,
            owner_references=[
                k8s_client.V1OwnerReference(
                    api_version="nvidia.com/v1beta1",
                    kind=owner_kind,
                    name="component"
                    if owner_kind == "DynamoComponentDeployment"
                    else "graph",
                    uid=owner_uid,
                    controller=True,
                )
            ],
        ),
        spec=k8s_client.V1ServiceSpec(
            selector={"nvidia.com/dynamo-component-type": "frontend"}
        ),
    )


def _ready_client(monkeypatch, services):
    """Return a ready graph client whose tests fail immediately on any retry."""
    monkeypatch.setattr(
        "deploy.utils.dynamo_deployment.asyncio.sleep",
        AsyncMock(side_effect=AssertionError("unexpected readiness retry")),
    )
    client = DynamoDeploymentClient(namespace="ns")
    client.core_api = MagicMock()
    client.core_api.list_namespaced_service = AsyncMock(
        return_value=k8s_client.V1ServiceList(items=services)
    )
    client.custom_api = MagicMock()
    client.custom_api.get_namespaced_custom_object = AsyncMock(
        return_value={
            "metadata": {"uid": "graph-uid"},
            "status": {
                "state": "successful",
                "conditions": [{"type": "Ready", "status": "True"}],
            },
        }
    )
    return client


@pytest.mark.unit
@pytest.mark.gpu_0
@pytest.mark.parametrize("service_count", [0, 2])
async def test_ready_graph_without_unique_frontend_reports_connection_error(
    monkeypatch, service_count
):
    """Endpoint choice must not prevent a worker-only or multi-frontend graph being ready."""
    client = _ready_client(
        monkeypatch,
        [
            _owned_frontend_service(f"frontend-{index}")
            for index in range(service_count)
        ],
    )

    assert await client.wait_for_deployment_ready(timeout=1) is True
    error = (
        "no owned frontend Service"
        if service_count == 0
        else "multiple owned frontend Services"
    )
    with pytest.raises(RuntimeError, match=error):
        client.get_service_url()
    with pytest.raises(RuntimeError, match=error):
        client.port_forward_frontend()
    assert client.service_name is None


@pytest.mark.unit
@pytest.mark.gpu_0
@pytest.mark.parametrize("status_code", [401, 403])
@pytest.mark.parametrize("failed_resource", ["service-list", "component-get"])
async def test_ready_deployment_surfaces_discovery_permission_errors(
    monkeypatch, status_code, failed_resource
):
    """Service list and DCD reads must expose denied access without retrying."""
    client = _ready_client(
        monkeypatch,
        [
            _owned_frontend_service(
                "frontend", "DynamoComponentDeployment", "component-uid"
            )
        ],
    )
    failure = k8s_client.rest.ApiException(status=status_code, reason="access denied")
    if failed_resource == "service-list":
        client.core_api.list_namespaced_service.side_effect = failure
    else:
        ready_graph = client.custom_api.get_namespaced_custom_object.return_value
        client.custom_api.get_namespaced_custom_object.side_effect = [
            ready_graph,
            failure,
        ]

    with pytest.raises(k8s_client.rest.ApiException) as excinfo:
        await client.wait_for_deployment_ready(timeout=1)
    assert excinfo.value.status == status_code
    client.core_api.list_namespaced_service.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.gpu_0
async def test_frontend_discovery_skips_deleted_component_owner(monkeypatch):
    """A stale Service must not hide a live frontend later in the namespace list."""
    client = _ready_client(
        monkeypatch,
        [
            _owned_frontend_service(
                "stale-frontend", "DynamoComponentDeployment", "deleted-uid"
            ),
            _owned_frontend_service("live-frontend"),
        ],
    )
    ready_graph = client.custom_api.get_namespaced_custom_object.return_value
    client.custom_api.get_namespaced_custom_object.side_effect = [
        ready_graph,
        k8s_client.rest.ApiException(status=404, reason="component no longer exists"),
    ]

    assert await client.wait_for_deployment_ready(timeout=1) is True
    assert client.service_name == "live-frontend"
    assert client.custom_api.get_namespaced_custom_object.await_count == 2


@pytest.mark.unit
@pytest.mark.gpu_0
@pytest.mark.parametrize("terminating_resource", ["service", "component"])
async def test_frontend_discovery_skips_terminating_resources(
    monkeypatch, terminating_resource
):
    """Deleting endpoints must not make the live frontend look ambiguous."""
    stale = _owned_frontend_service(
        "stale-frontend", "DynamoComponentDeployment", "component-uid"
    )
    if terminating_resource == "service":
        stale.metadata.deletion_timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    client = _ready_client(
        monkeypatch, [stale, _owned_frontend_service("live-frontend")]
    )
    ready_graph = client.custom_api.get_namespaced_custom_object.return_value
    client.custom_api.get_namespaced_custom_object.side_effect = [
        ready_graph,
        {
            "metadata": {
                "uid": "component-uid",
                "deletionTimestamp": "2026-01-01T00:00:00Z",
                "ownerReferences": [
                    {
                        "apiVersion": "nvidia.com/v1beta1",
                        "kind": "DynamoGraphDeployment",
                        "uid": "graph-uid",
                        "controller": True,
                    }
                ],
            }
        },
    ]

    assert await client.wait_for_deployment_ready(timeout=1) is True
    assert client.service_name == "live-frontend"
    assert client.custom_api.get_namespaced_custom_object.await_count == (
        1 if terminating_resource == "service" else 2
    )
