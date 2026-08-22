# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for schema-aware DynamoGraphDeployment helpers."""

import pytest
import yaml

from tests.deploy.dgd_utils import DeploymentSpec

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    ("api_version", "spec_body", "env_key"),
    [
        ("nvidia.com/v1alpha1", {"services": {}}, "envs"),
        ("nvidia.com/v1beta1", {"components": []}, "env"),
    ],
)
def test_logging_config_uses_schema_specific_global_env(
    tmp_path, api_version: str, spec_body: dict, env_key: str
) -> None:
    """Set and read logging variables using the active DGD schema."""
    manifest = {
        "apiVersion": api_version,
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "logging-test"},
        "spec": spec_body,
    }
    manifest_path = tmp_path / "deploy.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))

    deployment_spec = DeploymentSpec(str(manifest_path))
    deployment_spec.set_logging(enable_jsonl=True, log_level="debug")

    assert deployment_spec.get_logging_config() == {
        "jsonl_enabled": True,
        "log_level": "debug",
    }
    assert env_key in deployment_spec._deployment_spec["spec"]
    unexpected_key = "env" if env_key == "envs" else "envs"
    assert unexpected_key not in deployment_spec._deployment_spec["spec"]


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
