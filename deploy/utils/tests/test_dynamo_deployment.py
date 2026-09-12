# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for the fail-fast behavior added for #9213.

When a candidate deployment's worker pods enter ``CrashLoopBackOff``,
``DynamoDeploymentClient.wait_for_deployment_ready`` must raise
``DeploymentFailedError`` immediately rather than waiting out the full
``timeout`` — otherwise the thorough-mode profiler burns up to 30 min
of wall-clock per failing candidate.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

# Skip the whole module if the deploy.utils runtime deps aren't available
# in this environment. The test doesn't actually exercise these — it just
# needs the import of `dynamo_deployment` to succeed.
pytest.importorskip("aiofiles")
pytest.importorskip("kubernetes_asyncio")
pytest.importorskip("httpx")

from deploy.utils.dynamo_deployment import (  # noqa: E402
    DeploymentFailedError,
    DynamoDeploymentClient,
)

pytestmark = pytest.mark.pre_merge


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

    # Avoid sleeping in the test.
    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr("deploy.utils.dynamo_deployment.asyncio.sleep", _no_sleep)

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
