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

import logging
import shlex
from typing import Optional

from pydantic import BaseModel

from dynamo.common.utils.runtime import parse_endpoint
from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.errors import DuplicateSubComponentError, SubComponentNotFoundError
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def break_arguments(args: list[str] | None) -> list[str]:
    ans: list[str] = []
    if args is None:
        return ans
    if isinstance(args, str):
        # Use shlex.split to properly handle quoted arguments and JSON values
        ans = shlex.split(args)
    else:
        for arg in args:
            if arg is not None:
                # Use shlex.split to properly handle quoted arguments
                ans.extend(shlex.split(arg))
    return ans


class Service(BaseModel):
    name: str
    service: dict

    def number_replicas(self) -> int:
        return self.service.get("replicas", 0)

    def get_model_name(self) -> Optional[str]:
        args = (
            self.service.get("extraPodSpec", {})
            .get("mainContainer", {})
            .get("args", [])
        )

        args = break_arguments(args)
        if (
            "--served-model-name" in args
            and len(args) > args.index("--served-model-name") + 1
        ):
            return args[args.index("--served-model-name") + 1]
        if (
            "--model-name" in args and len(args) > args.index("--model-name") + 1
        ):  # mocker use --model-name
            return args[args.index("--model-name") + 1]
        if "--model" in args and len(args) > args.index("--model") + 1:
            return args[args.index("--model") + 1]

        return None

    def get_component_name_from_endpoint_arg(self) -> Optional[str]:
        """Return the component name from ``--endpoint`` in the container args.

        Worker backends (vLLM, SGLang, TRT-LLM) accept
        ``--endpoint <namespace>.<component>.<endpoint_name>`` (optionally
        prefixed with ``dyn://``) which overrides the default component
        name written to the MDC ``component`` field. When the user sets
        this, the Planner's MDC filter must match the user's value, not
        the backend default. Returns ``None`` if ``--endpoint`` is not
        present or malformed.
        """
        args = (
            self.service.get("extraPodSpec", {})
            .get("mainContainer", {})
            .get("args", [])
        )
        args = break_arguments(args)
        if "--endpoint" not in args:
            return None
        idx = args.index("--endpoint")
        if len(args) <= idx + 1:
            return None
        try:
            _, component, _ = parse_endpoint(args[idx + 1])
            return component
        except ValueError:
            return None

    def get_gpu_count(self) -> int:
        """Get the GPU count from the service's resource specification.

        Resolution order:
        1. ``resources.limits.gpu`` or ``resources.requests.gpu`` (NVIDIA device-plugin style)
        2. ``--tensor-parallel-size`` / ``--tp`` from ``extraPodSpec.mainContainer.args``
           (Intel DRA deployments omit ``resources.gpu`` and use ``resourceClaims`` instead;
           TP size equals the number of devices per worker)

        Returns:
            The number of GPUs/XPUs configured for this service

        Raises:
            ValueError: If GPU count cannot be determined
        """
        resources = self.service.get("resources", {})
        limits = resources.get("limits", {})
        requests = resources.get("requests", {})

        # 1. Prefer limits, fall back to requests (NVIDIA device-plugin style).
        gpu_str = limits.get("gpu") or requests.get("gpu")
        if gpu_str is not None:
            try:
                return int(gpu_str)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid GPU count '{gpu_str}' for service '{self.name}'. "
                    f"GPU count must be an integer."
                )

        # 2. DRA fallback: Intel XPU deployments use resourceClaims and do not set
        # resources.gpu.  Infer device count from --tensor-parallel-size / --tp in
        # the container args (TP size == number of devices per worker).
        raw_args = (
            self.service.get("extraPodSpec", {})
            .get("mainContainer", {})
            .get("args", [])
        )
        args = break_arguments(raw_args)
        for i, arg in enumerate(args):
            if arg in ("--tensor-parallel-size", "--tp") and i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Invalid --tensor-parallel-size value '{args[i + 1]}' "
                        f"for service '{self.name}'."
                    )

        raise ValueError(
            f"No GPU/XPU count specified for service '{self.name}'. "
            f"Set resources.limits.gpu, resources.requests.gpu, or "
            f"--tensor-parallel-size in extraPodSpec.mainContainer.args."
        )


# TODO: still supporting framework component names for backwards compatibility
# Should be deprecated in favor of service subComponentType
def get_service_from_sub_component_type_or_name(
    deployment: dict,
    sub_component_type: SubComponentType,
    component_name: Optional[str] = None,
) -> Service:
    """
    Get the current replicas for a component in a graph deployment

    Returns: Service object

    Raises:
        SubComponentNotFoundError: If no service with the specified subComponentType is found
        DuplicateSubComponentError: If multiple services with the same subComponentType are found
    """
    services = deployment.get("spec", {}).get("services", {})

    # Collect all available subComponentTypes for better error messages
    available_types = []
    matching_services = []

    for curr_name, curr_service in services.items():
        service_sub_type = curr_service.get("subComponentType", "")
        if service_sub_type:
            available_types.append(service_sub_type)

        if service_sub_type == sub_component_type.value:
            matching_services.append((curr_name, curr_service))

    # Check for duplicates
    if len(matching_services) > 1:
        service_names = [name for name, _ in matching_services]
        raise DuplicateSubComponentError(sub_component_type.value, service_names)

    # If no service found with subCompontType and fallback component_name is not provided or not found,
    # or if the fallback component has a non-empty subComponentType, raise error
    if not matching_services and (
        not component_name
        or component_name not in services
        or services[component_name].get("subComponentType", "") != ""
    ):
        raise SubComponentNotFoundError(sub_component_type.value)
    # If fallback component_name is provided and exists within services, add to matching_services
    elif not matching_services and component_name in services:
        matching_services.append((component_name, services[component_name]))

    name, service = matching_services[0]
    return Service(name=name, service=service)
