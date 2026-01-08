# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration for scale test infrastructure.

Provides dataclasses for configuring the scale test deployment settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def _get_default_template_path() -> str:
    """Get the default template path relative to this file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "templates", "mocker_deployment.yaml")


@dataclass
class ScaleTestConfig:
    """
    Configuration for scale test infrastructure.

    Attributes:
        kubernetes_namespace: Kubernetes namespace for DGD deployments
        image: Container image for all services
        template_path: Path to the DGD YAML template
        model_path: HuggingFace model path for the mocker
        speedup_ratio: Mocker speedup multiplier
        name_prefix: Prefix for deployment names (e.g., "scale-test")
        deployment_timeout: Timeout in seconds for deployment to become ready
        cleanup_on_exit: Whether to delete DGDs on exit
    """

    kubernetes_namespace: str = "default"
    image: str = "nvcr.io/nvidia/ai-dynamo/dynamo-base:latest"
    template_path: Optional[str] = None
    model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    speedup_ratio: float = 10.0
    name_prefix: str = "scale-test"
    deployment_timeout: int = 600
    cleanup_on_exit: bool = True

    def __post_init__(self):
        if self.template_path is None:
            self.template_path = _get_default_template_path()


@dataclass
class LoadTestConfig:
    """
    Configuration for load testing.

    Attributes:
        duration_sec: Duration of the load test in seconds
        qps: Queries per second to generate
        model: Model name to use in requests (should match mocker's model)
    """

    duration_sec: int = 60
    qps: float = 1.0
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@dataclass
class ScaleManagerConfig:
    """
    Combined configuration for the scale manager.

    Attributes:
        num_deployments: Number of DGD deployments to create
        scale_test: Scale test infrastructure configuration
        load_test: Load test configuration
    """

    num_deployments: int = 5
    scale_test: ScaleTestConfig = field(default_factory=ScaleTestConfig)
    load_test: LoadTestConfig = field(default_factory=LoadTestConfig)

    @classmethod
    def from_args(
        cls,
        num_deployments: int,
        namespace: str = "default",
        image: str = "nvcr.io/nvidia/ai-dynamo/dynamo-base:latest",
        model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        speedup_ratio: float = 10.0,
        duration: int = 60,
        qps: float = 1.0,
        timeout: int = 600,
    ) -> "ScaleManagerConfig":
        """
        Create a ScaleManagerConfig from CLI arguments.

        Args:
            num_deployments: Number of DGD deployments
            namespace: Kubernetes namespace
            image: Container image
            model_path: Model path for mocker
            speedup_ratio: Mocker speedup ratio
            duration: Load test duration
            qps: Queries per second
            timeout: Deployment timeout

        Returns:
            ScaleManagerConfig instance
        """
        return cls(
            num_deployments=num_deployments,
            scale_test=ScaleTestConfig(
                kubernetes_namespace=namespace,
                image=image,
                model_path=model_path,
                speedup_ratio=speedup_ratio,
                deployment_timeout=timeout,
            ),
            load_test=LoadTestConfig(
                duration_sec=duration,
                qps=qps,
                model=model_path,
            ),
        )
