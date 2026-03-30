# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Custom exceptions for the dynamo planner module."""

from typing import List

__all__ = [
    "PlannerError",
    "DynamoGraphDeploymentNotFoundError",
    "ComponentError",
    "ModelNameNotFoundError",
    "DeploymentModelNameMismatchError",
    "UserProvidedModelNameMismatchError",
    "BackendFrameworkNotFoundError",
    "BackendFrameworkInvalidError",
    "SubComponentNotFoundError",
    "DuplicateSubComponentError",
    "DeploymentValidationError",
    "EmptyTargetReplicasError",
]


class PlannerError(Exception):
    """Base exception for all planner-related errors."""


class DynamoGraphDeploymentNotFoundError(PlannerError):
    """Raised when Parent DynamoGraphDeployment cannot be found."""

    def __init__(self, deployment_name: str, namespace: str):
        self.deployment_name = deployment_name
        self.namespace = namespace

        message = (
            "Parent DynamoGraphDeployment not found "
            f"(name: '{deployment_name}' in namespace '{namespace}')"
        )

        super().__init__(message)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(deployment_name={self.deployment_name!r}, "
            f"namespace={self.namespace!r})"
        )


class ComponentError(PlannerError):
    """Base class for subComponent configuration issues."""


class ModelNameNotFoundError(PlannerError):
    """Raised when the model name is not found in the deployment."""

    def __init__(self):
        super().__init__("Model name not found in DynamoGraphDeployment")


class DeploymentModelNameMismatchError(PlannerError):
    """Raised when the model name is not the same in the deployment."""

    def __init__(self, prefill_model_name: str, decode_model_name: str):
        self.prefill_model_name = prefill_model_name
        self.decode_model_name = decode_model_name

        message = (
            "Model name mismatch in DynamoGraphDeployment: "
            f"prefill model name {prefill_model_name} != "
            f"decode model name {decode_model_name}"
        )
        self.message = message
        super().__init__(self.message)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"prefill_model_name={self.prefill_model_name!r}, "
            f"decode_model_name={self.decode_model_name!r})"
        )


class UserProvidedModelNameMismatchError(PlannerError):
    """Raised when the model name does not match the user supplied value."""

    def __init__(self, model_name: str, user_provided_model_name: str):
        self.model_name = model_name
        self.user_provided_model_name = user_provided_model_name

        message = (
            f"Model name {model_name} does not match expected model name "
            f"{user_provided_model_name}"
        )
        self.message = message
        super().__init__(self.message)


class BackendFrameworkNotFoundError(PlannerError):
    """Raised when the backend framework is not supported."""

    def __init__(self):
        super().__init__("Backend framework not found on DynamoGraphDeployment")


class BackendFrameworkInvalidError(PlannerError):
    """Raised when the backend framework does not exist."""

    def __init__(self, backend_framework: str):
        self.backend_framework = backend_framework

        super().__init__(f"Backend framework {backend_framework} is invalid")


class SubComponentNotFoundError(ComponentError):
    """Raised when a required subComponentType is not found in the deployment."""

    def __init__(self, sub_component_type: str):
        self.sub_component_type = sub_component_type

        message = (
            "DynamoGraphDeployment must contain a service with "
            f"subComponentType '{sub_component_type}'"
        )

        super().__init__(message)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sub_component_type={self.sub_component_type!r})"
        )


class DuplicateSubComponentError(ComponentError):
    """Raised when multiple services have the same subComponentType."""

    def __init__(self, sub_component_type: str, service_names: List[str]):
        self.sub_component_type = sub_component_type
        self.service_names = service_names

        message = (
            "DynamoGraphDeployment must contain only one service with "
            f"subComponentType '{sub_component_type}', but found multiple: "
            f"{', '.join(sorted(service_names))}"
        )

        super().__init__(message)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sub_component_type={self.sub_component_type!r}, "
            f"service_names={self.service_names!r})"
        )


class DeploymentValidationError(PlannerError):
    """Raised when deployment validation fails for multiple components."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Service verification failed: {'; '.join(errors)}")


class EmptyTargetReplicasError(PlannerError):
    """Raised when target_replicas is empty or invalid."""

    def __init__(self):
        super().__init__("target_replicas cannot be empty")
