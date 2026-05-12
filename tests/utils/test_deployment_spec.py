# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml

from tests.utils.managed_deployment import DeploymentSpec

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _write_yaml(tmp_path, manifest: dict) -> str:
    path = tmp_path / "deployment.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return str(path)


def test_deployment_spec_mutates_v1alpha1_services(tmp_path):
    path = _write_yaml(
        tmp_path,
        {
            "apiVersion": "nvidia.com/v1alpha1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": "alpha"},
            "spec": {
                "services": {
                    "Frontend": {
                        "componentType": "frontend",
                        "extraPodSpec": {"mainContainer": {"image": "frontend:v1"}},
                        "replicas": 1,
                    },
                    "VllmDecodeWorker": {
                        "componentType": "worker",
                        "extraPodSpec": {
                            "mainContainer": {
                                "args": [
                                    "--model",
                                    "Qwen/Qwen3-0.6B",
                                    "--tensor-parallel-size",
                                    "1",
                                ],
                                "image": "worker:v1",
                            }
                        },
                        "resources": {"limits": {"gpu": "1"}},
                        "replicas": 1,
                    },
                }
            },
        },
    )

    deployment = DeploymentSpec(path)

    deployment["Frontend"].replicas = 2
    deployment["VllmDecodeWorker"].tensor_parallel_size = 4
    deployment.set_service_env_var("VllmDecodeWorker", "TEST_ENV", "true")
    deployment.add_arg_to_service("VllmDecodeWorker", "--max-model-len", "1024")

    spec = deployment.spec()["spec"]["services"]
    assert spec["Frontend"]["replicas"] == 2
    assert spec["VllmDecodeWorker"]["resources"]["limits"]["gpu"] == "4"
    assert spec["VllmDecodeWorker"]["envs"] == [{"name": "TEST_ENV", "value": "true"}]
    assert (
        "--max-model-len"
        in spec["VllmDecodeWorker"]["extraPodSpec"]["mainContainer"]["args"]
    )


def test_deployment_spec_mutates_v1beta1_components(tmp_path):
    path = _write_yaml(
        tmp_path,
        {
            "apiVersion": "nvidia.com/v1beta1",
            "kind": "DynamoGraphDeployment",
            "metadata": {"name": "beta"},
            "spec": {
                "components": [
                    {
                        "name": "Frontend",
                        "podTemplate": {
                            "spec": {
                                "containers": [{"name": "main", "image": "frontend:v1"}]
                            }
                        },
                        "replicas": 1,
                        "type": "frontend",
                    },
                    {
                        "frontendSidecar": "sidecar-frontend",
                        "name": "VllmDecodeWorker",
                        "podTemplate": {
                            "spec": {
                                "containers": [
                                    {
                                        "args": [
                                            "--model",
                                            "Qwen/Qwen3-0.6B",
                                            "--tensor-parallel-size",
                                            "1",
                                        ],
                                        "image": "worker:v1",
                                        "name": "main",
                                        "resources": {
                                            "limits": {"nvidia.com/gpu": "1"},
                                            "requests": {"nvidia.com/gpu": "1"},
                                        },
                                    },
                                    {
                                        "image": "sidecar:v1",
                                        "name": "sidecar-frontend",
                                    },
                                ]
                            }
                        },
                        "replicas": 1,
                        "type": "decode",
                    },
                ]
            },
        },
    )

    deployment = DeploymentSpec(path)

    deployment["Frontend"].replicas = 2
    deployment["VllmDecodeWorker"].tensor_parallel_size = 4
    deployment.set_service_env_var("VllmDecodeWorker", "TEST_ENV", "true")
    deployment.add_arg_to_service("VllmDecodeWorker", "--max-model-len", "1024")
    deployment.set_frontend_sidecar_image("sidecar:v2", service_name="VllmDecodeWorker")

    components = {
        component["name"]: component
        for component in deployment.spec()["spec"]["components"]
    }
    worker_containers = components["VllmDecodeWorker"]["podTemplate"]["spec"][
        "containers"
    ]
    main = worker_containers[0]
    sidecar = worker_containers[1]

    assert deployment.api_version == "v1beta1"
    assert components["Frontend"]["replicas"] == 2
    assert main["resources"]["limits"]["nvidia.com/gpu"] == "4"
    assert main["resources"]["requests"]["nvidia.com/gpu"] == "4"
    assert main["env"] == [{"name": "TEST_ENV", "value": "true"}]
    assert "--max-model-len" in main["args"]
    assert sidecar["image"] == "sidecar:v2"
