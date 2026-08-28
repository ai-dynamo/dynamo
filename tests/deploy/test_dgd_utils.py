# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for schema-aware DynamoGraphDeployment helpers."""

import pytest
import yaml

from tests.deploy.dgd_utils import DeploymentSpec

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
