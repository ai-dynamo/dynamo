# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class Resources:
    """Resources for a service."""

    cpu: str | None = None  # example: "3", "300m"
    memory: str | None = None  # example: "10Gi", "1024Mi"
    gpu: str | None = None  # example: "4"

    def __post_init__(self):
        # Validate and normalize CPU format (e.g., "3", "300m")
        if self.cpu is not None:
            self.cpu = self.cpu.strip()
            if not (
                self.cpu.isdigit() or (self.cpu[:-1].isdigit() and self.cpu[-1] == "m")
            ):
                raise ValueError(
                    f"Invalid CPU format: {self.cpu}. Expected format like '3' or '300m'"
                )

        # Validate and normalize memory format (e.g., "10Gi", "1024Mi")
        if self.memory is not None:
            self.memory = self.memory.strip()
            valid_suffixes = [
                "Ki",
                "Mi",
                "Gi",
                "Ti",
                "Pi",
                "Ei",
                "K",
                "M",
                "G",
                "T",
                "P",
                "E",
            ]
            if not any(
                self.memory.endswith(suffix) and self.memory[: -len(suffix)].isdigit()
                for suffix in valid_suffixes
            ):
                if not self.memory.isdigit():
                    raise ValueError(
                        f"Invalid memory format: {self.memory}. Expected format like '10Gi' or '1024Mi'"
                    )

        # Validate and normalize GPU format (should be a number)
        if self.gpu is not None:
            self.gpu = self.gpu.strip()
            if not self.gpu.isdigit():
                raise ValueError(
                    f"Invalid GPU format: {self.gpu}. Expected a number like '1' or '4'"
                )


class DeploymentStatus(str, Enum):
    """Status of a dynamo deployment."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RUNNING = "running"
    FAILED = "failed"
    TERMINATED = "terminate"
    SCALED_TO_ZERO = "scaled_to_zero"


@dataclass
class ScalingPolicy:
    policy: str
    parameters: dict[str, t.Union[int, float, str]] = field(default_factory=dict)


@dataclass
class Service:
    """A single component."""

    name: str
    namespace: str
    class_name: str
    id: str | None = None
    cmd: list[str] = field(default_factory=list)
    resources: Resources | None = None
    environment: dict[str, str] = field(default_factory=dict)
    secrets: list[str] = field(default_factory=list)
    scaling: ScalingPolicy = field(default_factory=lambda: ScalingPolicy(policy="none"))


@dataclass
class Deployment:
    """Graph deployment."""

    name: str
    namespace: str
    services: list[Service] = field(default_factory=list)


class DeploymentManager(ABC):
    """Interface for managing dynamo graph deployments."""

    @abstractmethod
    def create_deployment(self, deployment: Deployment) -> str:
        """Create new deployment.

        Args:
            deployment: Deployment configuration

        Returns:
            The ID of the created deployment
        """
        pass

    @abstractmethod
    def update_deployment(self, deployment_id: str, deployment: Deployment) -> None:
        """Update an existing deployment.

        Args:
            deployment_id: The ID of the deployment to update
            deployment: New deployment configuration
        """
        pass

    @abstractmethod
    def get_deployment(self, deployment_id: str) -> dict[str, t.Any]:
        """Get deployment details.

        Args:
            deployment_id: The ID of the deployment

        Returns:
            Dictionary containing deployment details
        """
        pass

    @abstractmethod
    def list_deployments(self) -> list[dict[str, t.Any]]:
        """List all deployments.

        Returns:
            List of dictionaries containing deployment id and details
        """
        pass

    @abstractmethod
    def delete_deployment(self, deployment_id: str) -> None:
        """Delete a deployment.

        Args:
            deployment_id: The ID of the deployment to delete
        """
        pass

    @abstractmethod
    def get_status(self, deployment_id: str) -> DeploymentStatus:
        """Get the current status of a deployment.

        Args:
            deployment_id: The ID of the deployment

        Returns:
            The current status of the deployment
        """
        pass

    @abstractmethod
    def wait_until_ready(self, deployment_id: str, timeout: int = 3600) -> bool:
        """Wait until a deployment is ready.

        Args:
            deployment_id: The ID of the deployment
            timeout: Maximum time to wait in seconds

        Returns:
            True if deployment became ready, False if timed out
        """
        pass

    @abstractmethod
    def get_endpoint_urls(self, deployment_id: str) -> list[str]:
        """Get the list of endpoint urls attached to a deployment.

        Args:
            deployment_id: The ID of the deployment

        Returns:
            List of deployment's endpoint urls
        """
        pass
