# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for schema-aware DynamoGraphDeployment helpers."""

import pytest
import yaml

from tests.deploy.dgd_utils import (
    SCHEMA_V1ALPHA1,
    SCHEMA_V1BETA1,
    DeploymentSpec,
    ServiceSpec,
)

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


def test_model_resolves_shell_variable_from_v1alpha1_container_env() -> None:
    service = ServiceSpec(
        "Worker",
        {
            "extraPodSpec": {
                "mainContainer": {
                    "args": ["--model-path", "${MODEL_PATH}"],
                    "env": [{"name": "MODEL_PATH", "value": "Qwen/Qwen3-32B-FP8"}],
                }
            }
        },
        schema=SCHEMA_V1ALPHA1,
    )

    assert service.model == "Qwen/Qwen3-32B-FP8"


def test_model_resolves_shell_variable_from_v1beta1_container_env() -> None:
    service = ServiceSpec(
        "Worker",
        {
            "podTemplate": {
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                            "args": ["--model", "$MODEL_ID"],
                            "env": [
                                {
                                    "name": "MODEL_ID",
                                    "value": "Qwen/Qwen3-32B-FP8",
                                }
                            ],
                        }
                    ]
                }
            }
        },
        schema=SCHEMA_V1BETA1,
    )

    assert service.model == "Qwen/Qwen3-32B-FP8"
