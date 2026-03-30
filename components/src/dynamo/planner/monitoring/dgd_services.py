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

import shlex
from typing import Optional

from pydantic import BaseModel

from dynamo.planner.errors import DuplicateSubComponentError, SubComponentNotFoundError
from dynamo.planner.types import SubComponentType


def break_arguments(args: list[str] | None) -> list[str]:
    ans: list[str] = []
    if args is None:
        return ans
    if isinstance(args, str):
        ans = shlex.split(args)
    else:
        for arg in args:
            if arg is not None:
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
        if "--model-name" in args and len(args) > args.index("--model-name") + 1:
            return args[args.index("--model-name") + 1]
        if "--model" in args and len(args) > args.index("--model") + 1:
            return args[args.index("--model") + 1]

        return None

    def get_gpu_count(self) -> int:
        resources = self.service.get("resources", {})
        limits = resources.get("limits", {})
        requests = resources.get("requests", {})

        gpu_str = limits.get("gpu") or requests.get("gpu")

        if gpu_str is None:
            raise ValueError(
                f"No GPU count specified for service '{self.name}'. "
                "Please set resources.limits.gpu or resources.requests.gpu in the DGD."
            )

        try:
            return int(gpu_str)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid GPU count '{gpu_str}' for service '{self.name}'. "
                "GPU count must be an integer."
            ) from exc


def get_service_from_sub_component_type_or_name(
    deployment: dict,
    sub_component_type: SubComponentType,
    component_name: Optional[str] = None,
) -> Service:
    """Get the current replicas for a component in a graph deployment."""
    services = deployment.get("spec", {}).get("services", {})

    matching_services = []

    for curr_name, curr_service in services.items():
        service_sub_type = curr_service.get("subComponentType", "")
        if service_sub_type == sub_component_type.value:
            matching_services.append((curr_name, curr_service))

    if len(matching_services) > 1:
        service_names = [name for name, _ in matching_services]
        raise DuplicateSubComponentError(sub_component_type.value, service_names)

    if not matching_services and (
        not component_name
        or component_name not in services
        or services[component_name].get("subComponentType", "") != ""
    ):
        raise SubComponentNotFoundError(sub_component_type.value)
    elif not matching_services and component_name in services:
        matching_services.append((component_name, services[component_name]))

    name, service = matching_services[0]
    return Service(name=name, service=service)
