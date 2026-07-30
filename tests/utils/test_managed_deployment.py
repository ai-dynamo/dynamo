# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.utils.managed_deployment import DeploymentSpec

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
]


def _deployment_spec(tmp_path: Path, raw_spec: dict[str, Any]) -> DeploymentSpec:
    manifest = tmp_path / "deployment.yaml"
    manifest.write_text(yaml.safe_dump(raw_spec))
    return DeploymentSpec(str(manifest))


def test_mount_model_cache_pvc_uses_v1beta1_native_fields(
    tmp_path: Path,
) -> None:
    deployment = _deployment_spec(
        tmp_path,
        {
            "apiVersion": "nvidia.com/v1beta1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": "test"},
            "spec": {
                "env": [{"name": "EXISTING", "value": "value"}],
                "components": [
                    {
                        "name": "Frontend",
                        "podTemplate": {
                            "spec": {
                                "containers": [{"name": "main", "image": "frontend"}]
                            }
                        },
                    },
                    {
                        "name": "Worker",
                        "experimental": {
                            "checkpoint": {
                                "targetContainerName": "snapshot-me",
                            }
                        },
                        "podTemplate": {
                            "spec": {
                                "containers": [
                                    {"name": "main", "image": "worker"},
                                    {
                                        "name": "snapshot-me",
                                        "image": "snapshot-worker",
                                        "env": [
                                            {
                                                "name": "EXISTING_TARGET",
                                                "value": "value",
                                            }
                                        ],
                                    },
                                    {"name": "unrelated-sidecar", "image": "sidecar"},
                                ]
                            }
                        },
                    },
                ],
            },
        },
    )

    deployment.mount_model_cache_pvc("model-cache", "/models")
    expected_after_first_mount = copy.deepcopy(deployment.spec())
    deployment.mount_model_cache_pvc("model-cache", "/models")

    raw_spec = deployment.spec()
    assert raw_spec == expected_after_first_mount
    assert raw_spec["spec"]["env"] == [
        {"name": "EXISTING", "value": "value"},
        {"name": "HF_HOME", "value": "/models"},
    ]
    assert "pvcs" not in raw_spec["spec"]
    assert "envs" not in raw_spec["spec"]

    for component in raw_spec["spec"]["components"]:
        assert "volumeMounts" not in component
        pod_spec = component["podTemplate"]["spec"]
        assert pod_spec["volumes"] == [
            {
                "name": "model-cache",
                "persistentVolumeClaim": {"claimName": "model-cache"},
            }
        ]
        main = next(
            container
            for container in pod_spec["containers"]
            if container["name"] == "main"
        )
        assert main["volumeMounts"] == [{"name": "model-cache", "mountPath": "/models"}]
        assert "env" not in main

        if component["name"] == "Worker":
            checkpoint_target = next(
                container
                for container in pod_spec["containers"]
                if container["name"] == "snapshot-me"
            )
            assert checkpoint_target["env"] == [
                {"name": "EXISTING_TARGET", "value": "value"},
                {"name": "HF_HOME", "value": "/models"},
            ]
            assert checkpoint_target["volumeMounts"] == [
                {"name": "model-cache", "mountPath": "/models"}
            ]

            unrelated_sidecar = next(
                container
                for container in pod_spec["containers"]
                if container["name"] == "unrelated-sidecar"
            )
            assert "env" not in unrelated_sidecar
            assert "volumeMounts" not in unrelated_sidecar


def test_mount_model_cache_pvc_rejects_missing_checkpoint_target(
    tmp_path: Path,
) -> None:
    deployment = _deployment_spec(
        tmp_path,
        {
            "apiVersion": "nvidia.com/v1beta1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": "test"},
            "spec": {
                "components": [
                    {
                        "name": "Worker",
                        "experimental": {
                            "checkpoint": {
                                "targetContainerName": "missing",
                            }
                        },
                        "podTemplate": {
                            "spec": {
                                "containers": [{"name": "main", "image": "worker"}]
                            }
                        },
                    }
                ]
            },
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            "v1beta1 component 'Worker' checkpoint targetContainerName 'missing' "
            "does not name a podTemplate container"
        ),
    ):
        deployment.mount_model_cache_pvc("model-cache", "/models")

    assert deployment.spec()["spec"] == {
        "components": [
            {
                "name": "Worker",
                "experimental": {
                    "checkpoint": {
                        "targetContainerName": "missing",
                    }
                },
                "podTemplate": {
                    "spec": {
                        "containers": [{"name": "main", "image": "worker"}],
                    }
                },
            }
        ]
    }


def test_mount_model_cache_pvc_preserves_v1alpha1_fields(
    tmp_path: Path,
) -> None:
    deployment = _deployment_spec(
        tmp_path,
        {
            "apiVersion": "nvidia.com/v1alpha1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": "test"},
            "spec": {
                "services": {
                    "Frontend": {"componentType": "frontend"},
                    "Worker": {"componentType": "worker"},
                }
            },
        },
    )

    deployment.mount_model_cache_pvc("model-cache", "/models")

    assert deployment.spec() == {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "test"},
        "spec": {
            "services": {
                "Frontend": {
                    "componentType": "frontend",
                    "volumeMounts": [{"name": "model-cache", "mountPoint": "/models"}],
                },
                "Worker": {
                    "componentType": "worker",
                    "volumeMounts": [{"name": "model-cache", "mountPoint": "/models"}],
                },
            },
            "pvcs": [{"name": "model-cache", "create": False}],
            "envs": [{"name": "HF_HOME", "value": "/models"}],
        },
    }
