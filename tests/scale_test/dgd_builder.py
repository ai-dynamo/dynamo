# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

from tests.utils.managed_deployment import DeploymentSpec


def _get_default_template_path() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "templates", "mocker_deployment.yaml")


class ScaleTestDGDBuilder:
    """Builder for scale test DGD specifications."""

    def __init__(
        self,
        deployment_id: int,
        base_template: Optional[str] = None,
        name_prefix: str = "scale-test",
    ):
        self.deployment_id = deployment_id
        self.name_prefix = name_prefix

        if base_template is None:
            base_template = _get_default_template_path()

        self._spec = DeploymentSpec(base_template)
        deployment_name = f"{name_prefix}-{deployment_id}"
        self._spec.name = deployment_name
        self._set_dynamo_namespace(deployment_name)

    @property
    def deployment_name(self) -> str:
        return self._spec.name

    def _set_dynamo_namespace(self, namespace: str) -> None:
        spec_dict = self._spec.spec()
        for service_name in spec_dict["spec"]["services"]:
            spec_dict["spec"]["services"][service_name]["dynamoNamespace"] = namespace

    def set_deployment_name(self, name: str) -> "ScaleTestDGDBuilder":
        self._spec.name = name
        self._set_dynamo_namespace(name)
        return self

    def set_kubernetes_namespace(self, namespace: str) -> "ScaleTestDGDBuilder":
        self._spec.namespace = namespace
        return self

    def set_model(self, model_path: str) -> "ScaleTestDGDBuilder":
        self._update_mocker_arg("--model-path", model_path)
        return self

    def set_speedup_ratio(self, ratio: float) -> "ScaleTestDGDBuilder":
        self._update_mocker_arg("--speedup-ratio", str(ratio))
        return self

    def set_num_workers(self, num_workers: int) -> "ScaleTestDGDBuilder":
        self._update_mocker_arg("--num-workers", str(num_workers))
        return self

    def set_image(self, image: str) -> "ScaleTestDGDBuilder":
        self._spec.set_image(image)
        return self

    def set_frontend_replicas(self, replicas: int) -> "ScaleTestDGDBuilder":
        self._spec.set_service_replicas("Frontend", replicas)
        return self

    def set_worker_replicas(self, replicas: int) -> "ScaleTestDGDBuilder":
        self._spec.set_service_replicas("MockerWorker", replicas)
        return self

    def set_log_level(self, level: str) -> "ScaleTestDGDBuilder":
        self._spec.set_logging(enable_jsonl=False, log_level=level)
        return self

    def set_router_mode(self, mode: str) -> "ScaleTestDGDBuilder":
        self._update_frontend_arg("--router-mode", mode)
        return self

    def set_image_pull_secrets(self, secret_names: list[str]) -> "ScaleTestDGDBuilder":
        self._spec.set_image_pull_secrets(secret_names)
        return self

    def _update_mocker_arg(self, arg_name: str, arg_value: str) -> None:
        self._spec.add_arg_to_service("MockerWorker", arg_name, arg_value)

    def _update_frontend_arg(self, arg_name: str, arg_value: str) -> None:
        self._spec.add_arg_to_service("Frontend", arg_name, arg_value)

    def build(self) -> DeploymentSpec:
        return self._spec

    def build_dict(self) -> dict:
        return self._spec.spec()


def create_scale_test_specs(
    num_deployments: int,
    kubernetes_namespace: str = "default",
    model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    speedup_ratio: float = 10.0,
    image: str = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1",
    name_prefix: str = "scale-test",
    base_template: Optional[str] = None,
) -> list[DeploymentSpec]:
    specs = []
    for i in range(1, num_deployments + 1):
        builder = ScaleTestDGDBuilder(
            deployment_id=i,
            base_template=base_template,
            name_prefix=name_prefix,
        )
        spec = (
            builder.set_kubernetes_namespace(kubernetes_namespace)
            .set_model(model_path)
            .set_speedup_ratio(speedup_ratio)
            .set_image(image)
            .build()
        )
        specs.append(spec)
    return specs
