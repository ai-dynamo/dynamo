# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DGD Builder utility for scale testing.

Provides a builder pattern for programmatically creating DynamoGraphDeployment
specifications from a base template.
"""

import os
from typing import Optional

from tests.utils.managed_deployment import DeploymentSpec


def _get_default_template_path() -> str:
    """Get the default template path relative to this file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "templates", "mocker_deployment.yaml")


class ScaleTestDGDBuilder:
    """
    Builder for scale test DGD specifications.

    Provides a fluent interface for creating DynamoGraphDeployment specs
    from a base template with customizable parameters.

    Example:
        builder = ScaleTestDGDBuilder(deployment_id=1)
        spec = (
            builder
            .set_kubernetes_namespace("my-namespace")
            .set_model("meta-llama/Llama-3-8b")
            .set_speedup_ratio(5.0)
            .set_image("my-registry/dynamo:latest")
            .build()
        )
    """

    def __init__(
        self,
        deployment_id: int,
        base_template: Optional[str] = None,
        name_prefix: str = "scale-test",
    ):
        """
        Initialize the builder with a deployment ID and base template.

        Args:
            deployment_id: Unique identifier for this deployment (1, 2, 3, ...)
            base_template: Path to the base YAML template. If None, uses default template.
            name_prefix: Prefix for deployment name (default: "scale-test")
        """
        self.deployment_id = deployment_id
        self.name_prefix = name_prefix

        if base_template is None:
            base_template = _get_default_template_path()

        self._spec = DeploymentSpec(base_template)

        # Set default deployment name based on ID
        deployment_name = f"{name_prefix}-{deployment_id}"
        self._spec.name = deployment_name

        # Set dynamo namespace to match deployment name
        self._set_dynamo_namespace(deployment_name)

    @property
    def deployment_name(self) -> str:
        """Get the deployment name."""
        return self._spec.name

    def _set_dynamo_namespace(self, namespace: str) -> None:
        """
        Set the dynamoNamespace for all services.

        This is the Dynamo-level namespace used for service discovery,
        separate from the Kubernetes namespace.
        """
        spec_dict = self._spec.spec()
        for service_name in spec_dict["spec"]["services"]:
            spec_dict["spec"]["services"][service_name]["dynamoNamespace"] = namespace

    def set_deployment_name(self, name: str) -> "ScaleTestDGDBuilder":
        """
        Override the deployment name.

        Args:
            name: Custom deployment name
        """
        self._spec.name = name
        self._set_dynamo_namespace(name)
        return self

    def set_kubernetes_namespace(self, namespace: str) -> "ScaleTestDGDBuilder":
        """
        Set the Kubernetes namespace for the deployment.

        Args:
            namespace: Kubernetes namespace (e.g., "default", "dynamo-test")
        """
        self._spec.namespace = namespace
        return self

    def set_model(self, model_path: str) -> "ScaleTestDGDBuilder":
        """
        Set the model path for the mocker worker.

        Args:
            model_path: HuggingFace model path (e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        """
        # Update the --model-path argument in MockerWorker
        self._update_mocker_arg("--model-path", model_path)
        return self

    def set_speedup_ratio(self, ratio: float) -> "ScaleTestDGDBuilder":
        """
        Set the speedup ratio for the mocker.

        Args:
            ratio: Speedup multiplier (higher = faster simulation)
        """
        self._update_mocker_arg("--speedup-ratio", str(ratio))
        return self

    def set_num_workers(self, num_workers: int) -> "ScaleTestDGDBuilder":
        """
        Set the number of workers per mocker instance.

        Args:
            num_workers: Number of workers
        """
        self._update_mocker_arg("--num-workers", str(num_workers))
        return self

    def set_image(self, image: str) -> "ScaleTestDGDBuilder":
        """
        Set the container image for all services.

        Args:
            image: Container image (e.g., "nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest")
        """
        self._spec.set_image(image)
        return self

    def set_frontend_replicas(self, replicas: int) -> "ScaleTestDGDBuilder":
        """
        Set the number of Frontend replicas.

        Args:
            replicas: Number of replicas
        """
        self._spec.set_service_replicas("Frontend", replicas)
        return self

    def set_worker_replicas(self, replicas: int) -> "ScaleTestDGDBuilder":
        """
        Set the number of MockerWorker replicas.

        Args:
            replicas: Number of replicas
        """
        self._spec.set_service_replicas("MockerWorker", replicas)
        return self

    def set_log_level(self, level: str) -> "ScaleTestDGDBuilder":
        """
        Set the log level for the deployment.

        Args:
            level: Log level (e.g., "debug", "info", "warn", "error")
        """
        self._spec.set_logging(enable_jsonl=False, log_level=level)
        return self

    def set_router_mode(self, mode: str) -> "ScaleTestDGDBuilder":
        """
        Set the router mode for the frontend.

        Args:
            mode: Router mode (e.g., "round-robin", "kv")
        """
        self._update_frontend_arg("--router-mode", mode)
        return self

    def set_image_pull_secrets(self, secret_names: list[str]) -> "ScaleTestDGDBuilder":
        """
        Set imagePullSecrets for all services in the deployment.

        Args:
            secret_names: List of Kubernetes secret names for pulling images
        """
        self._spec.set_image_pull_secrets(secret_names)
        return self

    def _update_mocker_arg(self, arg_name: str, arg_value: str) -> None:
        """Update an argument in the MockerWorker service."""
        self._spec.add_arg_to_service("MockerWorker", arg_name, arg_value)

    def _update_frontend_arg(self, arg_name: str, arg_value: str) -> None:
        """Update an argument in the Frontend service."""
        self._spec.add_arg_to_service("Frontend", arg_name, arg_value)

    def build(self) -> DeploymentSpec:
        """
        Build and return the DeploymentSpec.

        Returns:
            DeploymentSpec object ready for use with ManagedDeployment
        """
        return self._spec

    def build_dict(self) -> dict:
        """
        Build and return the raw deployment spec dictionary.

        Returns:
            Dictionary containing the DGD spec
        """
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
    """
    Create multiple DGD specs for scale testing.

    Args:
        num_deployments: Number of DGD deployments to create
        kubernetes_namespace: Kubernetes namespace for all deployments
        model_path: Model path for the mocker
        speedup_ratio: Speedup ratio for the mocker
        image: Container image for all services
        name_prefix: Prefix for deployment names
        base_template: Path to base YAML template

    Returns:
        List of DeploymentSpec objects
    """
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
