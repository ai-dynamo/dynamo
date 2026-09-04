# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for schema-aware DynamoGraphDeployment helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_logging_config_reads_existing_v1beta1_env(tmp_path) -> None:
    """Recognize JSONL logging already declared in a v1beta1 manifest."""
    manifest = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "logging-test"},
        "spec": {
            "components": [],
            "env": [{"name": "DYN_LOGGING_JSONL", "value": "1"}],
        },
    }
    manifest_path = tmp_path / "deploy.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))

    deployment_spec = DeploymentSpec(str(manifest_path))

    assert deployment_spec.get_logging_config()["jsonl_enabled"] is True


def test_multi_document_manifest_selects_the_graph_deployment(tmp_path) -> None:
    """Recipe manifests bundle the DGD with ConfigMaps and friends.

    Most files under ``recipes/`` are multi-document; loading them with
    ``yaml.safe_load`` raises ComposerError, which previously made the majority
    of the recipe corpus unusable with DeploymentSpec.
    """
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "engine-config"},
        "data": {"engine.yaml": "tensor_parallel_size: 1\n"},
    }
    graph_deployment = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "multi-doc-test"},
        "spec": {"components": []},
    }
    manifest_path = tmp_path / "deploy.yaml"
    manifest_path.write_text(yaml.safe_dump_all([config_map, graph_deployment]))

    deployment_spec = DeploymentSpec(str(manifest_path))

    assert deployment_spec.name == "multi-doc-test"
    assert deployment_spec.schema == "v1beta1"


def test_manifest_without_a_graph_deployment_is_rejected(tmp_path) -> None:
    """A manifest carrying no DGD must fail loudly, not silently pick a Service."""
    manifest_path = tmp_path / "deploy.yaml"
    manifest_path.write_text(
        yaml.safe_dump_all(
            [
                {"apiVersion": "v1", "kind": "Service", "metadata": {"name": "svc"}},
                {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "cm"}},
            ]
        )
    )

    with pytest.raises(ValueError, match="no DynamoGraphDeployment"):
        DeploymentSpec(str(manifest_path))


def _multi_document_manifest(tmp_path):
    """A recipe shaped like the 97 that bundle prerequisites.

    The DGD references the ConfigMap by name, so applying the DGD alone leaves
    the worker in CreateContainerConfigError.
    """
    documents = [
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "engine-config"},
            "data": {"prefill.yaml": "model_path: Qwen/Qwen3-0.6B"},
        },
        {
            "apiVersion": "resource.nvidia.com/v1beta1",
            "kind": "ComputeDomain",
            "metadata": {"name": "test-compute-domain"},
            "spec": {"numNodes": 2},
        },
        {
            "apiVersion": "nvidia.com/v1beta1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": "recipe-under-test"},
            "spec": {
                "components": [
                    {
                        "name": "Worker",
                        "podTemplate": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "main",
                                        "command": ["/bin/bash", "-lc"],
                                        "args": ["exec python3 -m dynamo.vllm"],
                                        "volumeMounts": [
                                            {"name": "cfg", "mountPath": "/etc/engine"}
                                        ],
                                    }
                                ],
                                "volumes": [
                                    {
                                        "name": "cfg",
                                        "configMap": {"name": "engine-config"},
                                    }
                                ],
                            }
                        },
                    }
                ]
            },
        },
    ]
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump_all(documents))
    return path


def test_companion_documents_are_retained_not_discarded(tmp_path) -> None:
    """97 of 178 recipes bundle a resource their own DGD names.

    Selecting the DGD and dropping the rest makes the manifest loadable but
    incomplete: the operator creates the deployment and its workers then wait
    for a ConfigMap nothing ever created.
    """
    spec = DeploymentSpec(str(_multi_document_manifest(tmp_path)))

    assert spec.name == "recipe-under-test"
    assert [d["kind"] for d in spec.companions] == ["ConfigMap", "ComputeDomain"]
    # The DGD itself is never among them.
    assert all(d["kind"] != "DynamoGraphDeployment" for d in spec.companions)


def test_companions_keep_their_file_order(tmp_path) -> None:
    """A ComputeDomain must exist before the workers that claim it."""
    spec = DeploymentSpec(str(_multi_document_manifest(tmp_path)))
    names = [d["metadata"]["name"] for d in spec.companions]
    assert names == ["engine-config", "test-compute-domain"]


def test_a_single_document_manifest_has_no_companions(tmp_path) -> None:
    manifest = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "solo"},
        "spec": {"components": []},
    }
    path = tmp_path / "solo.yaml"
    path.write_text(yaml.safe_dump(manifest))

    assert DeploymentSpec(str(path)).companions == []


async def test_companions_are_applied_before_the_deployment(tmp_path) -> None:
    """Order is the whole point: the DGD references them by name."""
    spec = DeploymentSpec(str(_multi_document_manifest(tmp_path)))
    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=spec,
        namespace="default",
    )

    applied: list = []
    deployment._kubectl = lambda verb, docs, *extra: applied.append(
        (verb, [d["kind"] for d in docs])
    )
    deployment._custom_api = SimpleNamespace(
        create_namespaced_custom_object=AsyncMock(return_value=None)
    )

    await deployment._create_deployment()

    assert applied == [("apply", ["ConfigMap", "ComputeDomain"])]
    deployment._custom_api.create_namespaced_custom_object.assert_awaited_once()


async def test_a_failed_companion_apply_stops_the_deployment(tmp_path) -> None:
    """Creating the DGD anyway would leave a worker stuck on a missing resource,
    which reads as a deployment timeout rather than a manifest problem."""
    spec = DeploymentSpec(str(_multi_document_manifest(tmp_path)))
    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=spec,
        namespace="default",
    )
    deployment._kubectl = lambda *a, **k: SimpleNamespace(
        returncode=1, stderr="forbidden", stdout=""
    )
    deployment._custom_api = SimpleNamespace(
        create_namespaced_custom_object=AsyncMock(return_value=None)
    )

    with pytest.raises(RuntimeError, match="companion resources"):
        await deployment._create_deployment()
    deployment._custom_api.create_namespaced_custom_object.assert_not_awaited()


async def test_companions_are_removed_after_the_deployment(tmp_path) -> None:
    """Deleted after the DGD, so nothing is still mounting them."""
    spec = DeploymentSpec(str(_multi_document_manifest(tmp_path)))
    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=spec,
        namespace="default",
    )
    calls: list = []
    deployment._kubectl = lambda verb, docs, *extra: calls.append((verb, extra))
    deployment._deployment_name = "recipe-under-test"
    deployment._custom_api = SimpleNamespace(
        delete_namespaced_custom_object=AsyncMock(return_value=None)
    )

    await deployment._delete_deployment()

    assert calls == [("delete", ("--ignore-not-found=true", "--wait=false"))]


async def test_teardown_survives_a_failed_companion_delete(tmp_path) -> None:
    """Teardown must not mask the failure the test was reporting."""
    spec = DeploymentSpec(str(_multi_document_manifest(tmp_path)))
    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=spec,
        namespace="default",
    )

    def boom(*a, **k):
        raise OSError("kubectl not found")

    deployment._kubectl = boom
    deployment._deployment_name = "recipe-under-test"
    deployment._custom_api = SimpleNamespace(
        delete_namespaced_custom_object=AsyncMock(return_value=None)
    )

    await deployment._delete_deployment()  # must not raise


def test_a_placeholder_namespace_is_retargeted(tmp_path) -> None:
    """Four companions in the corpus declare `namespace: <your-namespace>`.

    That is neither a valid namespace name nor the one under test, and kubectl
    rejects the whole apply with a namespace mismatch rather than just that
    document.
    """
    documents = [
        {
            "apiVersion": "resource.nvidia.com/v1beta1",
            "kind": "ComputeDomain",
            "metadata": {"name": "cd", "namespace": "<your-namespace>"},
        },
        {
            "apiVersion": "nvidia.com/v1beta1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": "d"},
            "spec": {"components": []},
        },
    ]
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump_all(documents))

    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=DeploymentSpec(str(path)),
        namespace="under-test",
    )
    companion = deployment.deployment_spec.companions[0]

    assert deployment._retarget(companion)["metadata"]["namespace"] == "under-test"
    # The loaded spec is not mutated; -n and the document stay consistent.
    assert companion["metadata"]["namespace"] == "<your-namespace>"


def test_a_companion_without_a_namespace_is_left_alone(tmp_path) -> None:
    """Cluster-scoped kinds must not be given a namespace they cannot have."""
    deployment = ManagedDeployment(
        log_dir=str(tmp_path),
        deployment_spec=DeploymentSpec(str(_multi_document_manifest(tmp_path))),
        namespace="under-test",
    )
    configmap = deployment.deployment_spec.companions[0]

    assert "namespace" not in deployment._retarget(configmap)["metadata"]
