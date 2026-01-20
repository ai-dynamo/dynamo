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

"""Custom exceptions for the dynamo planner module.

This module defines a hierarchy of custom exceptions that provide more specific
error handling than generic ValueError exceptions. Each exception includes
contextual information to help with debugging and error handling.
"""

from typing import List


class PlannerError(Exception):
    """Base exception for all planner-related errors.

    This serves as the root exception class for all custom exceptions
    in the planner module, allowing for broad exception catching when needed.
    """

    pass


class DynamoGraphDeploymentNotFoundError(PlannerError):
    """Raised when Parent DynamoGraphDeployment cannot be found.

    This typically occurs when:
    - The DYN_PARENT_DGD_K8S_NAME environment variable is not set
    - The referenced DynamoGraphDeployment doesn't exist in the namespace
    """

    def __init__(self, deployment_name: str, namespace: str):
        self.deployment_name = deployment_name
        self.namespace = namespace

        message = f"Parent DynamoGraphDeployment not found (name: '{deployment_name}' in namespace '{namespace}')"

        super().__init__(message)


class ComponentError(PlannerError):
    """Base class for subComponent configuration issues.

    This serves as a parent class for all exceptions related to
    subComponentType configuration problems in DynamoGraphDeployments.
    """

    pass


class ModelNameNotFoundError(PlannerError):
    """Raised when the model name is not found in the deployment"""

    def __init__(self):
        message = "Model name not found in DynamoGraphDeployment"
        super().__init__(message)


class DeploymentModelNameMismatchError(PlannerError):
    """Raised when the model name is not the same in the deployment"""

    def __init__(self, prefill_model_name: str, decode_model_name: str):
        self.prefill_model_name = prefill_model_name
        self.decode_model_name = decode_model_name

        message = f"Model name mismatch in DynamoGraphDeployment: prefill model name {prefill_model_name} != decode model name {decode_model_name}"
        self.message = message
        super().__init__(self.message)


class UserProvidedModelNameMismatchError(PlannerError):
    """Raised when the model name is not the same as the user provided model name"""

    def __init__(self, model_name: str, user_provided_model_name: str):
        self.model_name = model_name
        self.user_provided_model_name = user_provided_model_name

        message = f"Model name {model_name} does not match expected model name {user_provided_model_name}"
        self.message = message
        super().__init__(self.message)


class BackendFrameworkNotFoundError(PlannerError):
    """Raised when the backend framework is not supported.

    This occurs when the DynamoGraphDeployment contains an unsupported backend framework.
    """

    def __init__(self):
        message = "Backend framework not found on DynamoGraphDeployment"
        super().__init__(message)


class BackendFrameworkInvalidError(PlannerError):
    """Raised when the backend framework does not exist.

    This occurs when the DynamoGraphDeployment contains an unsupported backend framework.
    """

    def __init__(self, backend_framework: str):
        self.backend_framework = backend_framework

        message = f"Backend framework {backend_framework} is invalid"
        super().__init__(message)


class SubComponentNotFoundError(ComponentError):
    """Raised when a required subComponentType is not found in the deployment.

    This occurs when the DynamoGraphDeployment doesn't contain any service
    with the requested subComponentType (e.g., 'prefill', 'decode').
    """

    def __init__(self, sub_component_type: str):
        self.sub_component_type = sub_component_type

        message = f"DynamoGraphDeployment must contain a service with subComponentType '{sub_component_type}'"

        super().__init__(message)


class DuplicateSubComponentError(ComponentError):
    """Raised when multiple services have the same subComponentType.

    This occurs when the DynamoGraphDeployment contains more than one service
    with the same subComponentType, which violates the expected uniqueness constraint.
    """

    def __init__(self, sub_component_type: str, service_names: List[str]):
        self.sub_component_type = sub_component_type
        self.service_names = service_names

        message = (
            f"DynamoGraphDeployment must contain only one service with "
            f"subComponentType '{sub_component_type}', but found multiple: "
            f"{', '.join(sorted(service_names))}"
        )

        super().__init__(message)


class DeploymentValidationError(PlannerError):
    """Raised when deployment validation fails for multiple components.

    This is used to aggregate multiple validation errors into a single exception,
    providing a comprehensive view of all validation issues.
    """

    def __init__(self, errors: List[str]):
        self.errors = errors
        message = f"Service verification failed: {'; '.join(errors)}"
        super().__init__(message)


class EmptyTargetReplicasError(PlannerError):
    """Raised when target_replicas is empty or invalid.

    This occurs when attempting to set component replicas with an empty
    or invalid target_replicas dictionary.
    """

    def __init__(
        self,
    ):
        message = "target_replicas cannot be empty"
        super().__init__(message)
