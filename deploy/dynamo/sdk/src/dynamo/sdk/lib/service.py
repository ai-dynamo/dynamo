#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union

from _bentoml_sdk import Service, ServiceConfig
from _bentoml_sdk.images import Image
from _bentoml_sdk.service.config import validate

from dynamo.sdk.lib.decorators import DynamoEndpoint

T = TypeVar("T", bound=object)


class RuntimeLinkedServices:
    """
    A class to track the linked services in the runtime.
    """

    def __init__(self) -> None:
        self.edges: Dict[DynamoService, Set[DynamoService]] = defaultdict(set)

    def add(self, edge: Tuple[DynamoService, DynamoService]):
        src, dest = edge
        self.edges[src].add(dest.inner)
        # track the dest node as well so we can cleanup later
        self.edges[dest]

    def remove_unused_edges(self):
        # this method is idempotent
        if not self.edges:
            return
        # remove edges that are not in the current service
        for u, vertices in self.edges.items():
            u.remove_unused_edges(used_edges=vertices)


LinkedServices = RuntimeLinkedServices()


@dataclass
class DynamoConfig:
    """Configuration for Dynamo components"""

    enabled: bool = False
    name: str | None = None
    namespace: str | None = None


class DynamoService(Service[T]):
    """A custom service class that extends BentoML's base Service with Dynamo capabilities"""

    def __init__(
        self,
        config: ServiceConfig,
        inner: type[T],
        image: Optional[Image] = None,
        envs: Optional[list[dict[str, Any]]] = None,
        dynamo_config: Optional[DynamoConfig] = None,
    ):
        service_name = inner.__name__
        service_args = self._get_service_args(service_name)

        if service_args:
            # Validate and merge service args with existing config
            validated_args = validate(service_args)
            config.update(validated_args)
            self._remove_service_args(service_name)

        super().__init__(config=config, inner=inner, image=image, envs=envs or [])

        # Initialize Dynamo configuration
        self._dynamo_config = (
            dynamo_config
            if dynamo_config
            else DynamoConfig(name=inner.__name__, namespace="default")
        )
        if self._dynamo_config.name is None:
            self._dynamo_config.name = inner.__name__

        # Add dynamo configuration to the service config
        # this allows for the config to be part of the service in bento.yaml
        self.config["dynamo"] = asdict(self._dynamo_config)

        # Register Dynamo endpoints
        self._dynamo_endpoints: Dict[str, DynamoEndpoint] = {}
        for field in dir(inner):
            value = getattr(inner, field)
            if isinstance(value, DynamoEndpoint):
                self._dynamo_endpoints[value.name] = value

        self._linked_services: List[DynamoService] = []  # Track linked services

    def _get_service_args(self, service_name: str) -> Optional[dict]:
        """Get ServiceArgs from environment config if specified"""
        config_str = os.environ.get("DYNAMO_SERVICE_CONFIG")
        if config_str:
            config = json.loads(config_str)
            service_config = config.get(service_name, {})
            return service_config.get("ServiceArgs")
        return None

    def is_dynamo_component(self) -> bool:
        """Check if this service is configured as a Dynamo component"""
        return self._dynamo_config.enabled

    def dynamo_address(self) -> Tuple[Optional[str], Optional[str]]:
        """Get the Dynamo address for this component in namespace/name format"""
        if not self.is_dynamo_component():
            raise ValueError("Service is not configured as a Dynamo component")

        # Check if we have a runner map with Dynamo address
        runner_map = os.environ.get("BENTOML_RUNNER_MAP")
        if runner_map:
            try:
                runners = json.loads(runner_map)
                if self.name in runners:
                    address = runners[self.name]
                    if address.startswith("dynamo://"):
                        # Parse dynamo://namespace/name into (namespace, name)
                        _, path = address.split("://", 1)
                        namespace, name = path.split("/", 1)
                        print(
                            f"Resolved Dynamo address from runner map: {namespace}/{name}"
                        )
                        return (namespace, name)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"Failed to parse BENTOML_RUNNER_MAP: {str(e)}") from e

        print(
            f"Using default Dynamo address: {self._dynamo_config.namespace}/{self._dynamo_config.name}"
        )
        return (self._dynamo_config.namespace, self._dynamo_config.name)

    def get_dynamo_endpoints(self) -> Dict[str, DynamoEndpoint]:
        """Get all registered Dynamo endpoints"""
        return self._dynamo_endpoints

    def get_dynamo_endpoint(self, name: str) -> DynamoEndpoint:
        """Get a specific Dynamo endpoint by name"""
        if name not in self._dynamo_endpoints:
            raise ValueError(f"No Dynamo endpoint found with name: {name}")
        return self._dynamo_endpoints[name]

    def list_dynamo_endpoints(self) -> List[str]:
        """List names of all registered Dynamo endpoints"""
        return list(self._dynamo_endpoints.keys())

    def remove_unused_edges(self, used_edges: Set[DynamoService]):
        """Remove a dependancy from the current service based on the key"""
        current_deps = dict(self.dependencies)
        for dep_key, dep_value in current_deps.items():
            if dep_value.on.inner not in used_edges:
                del self.dependencies[dep_key]

    def link(self, next_service: DynamoService):
        """Link this service to another service, creating a pipeline."""
        self._linked_services.append(next_service)
        LinkedServices.add((self, next_service))
        return next_service

    def _remove_service_args(self, service_name: str):
        """Remove ServiceArgs from the environment config after using them, preserving envs"""
        config_str = os.environ.get("DYNAMO_SERVICE_CONFIG")
        if config_str:
            config = json.loads(config_str)
            if service_name in config and "ServiceArgs" in config[service_name]:
                # Save envs to separate env var before removing ServiceArgs
                service_args = config[service_name]["ServiceArgs"]
                if "envs" in service_args:
                    service_envs = os.environ.get("DYNAMO_SERVICE_ENVS", "{}")
                    envs_config = json.loads(service_envs)
                    if service_name not in envs_config:
                        envs_config[service_name] = {}
                    envs_config[service_name]["ServiceArgs"] = {
                        "envs": service_args["envs"]
                    }
                    os.environ["DYNAMO_SERVICE_ENVS"] = json.dumps(envs_config)

                # Remove ServiceArgs from main config
                del config[service_name]["ServiceArgs"]
                os.environ["DYNAMO_SERVICE_CONFIG"] = json.dumps(config)


def service(
    inner: Optional[type[T]] = None,
    /,
    *,
    image: Optional[Image] = None,
    envs: Optional[list[dict[str, Any]]] = None,
    dynamo: Optional[Union[Dict[str, Any], DynamoConfig]] = None,
    **kwargs: Any,
) -> Any:
    """Enhanced service decorator that supports Dynamo configuration

    Args:
        dynamo: Dynamo configuration, either as a DynamoConfig object or dict with keys:
            - enabled: bool (default True)
            - name: str (default: class name)
            - namespace: str (default: "default")
        **kwargs: Existing BentoML service configuration
    """
    config = kwargs

    # Parse dict into DynamoConfig object
    dynamo_config: Optional[DynamoConfig] = None
    if dynamo is not None:
        if isinstance(dynamo, dict):
            dynamo_config = DynamoConfig(**dynamo)
        else:
            dynamo_config = dynamo

    def decorator(inner: type[T]) -> DynamoService[T]:
        if isinstance(inner, Service):
            raise TypeError("service() decorator can only be applied once")
        return DynamoService(
            config=config,
            inner=inner,
            image=image,
            envs=envs or [],
            dynamo_config=dynamo_config,
        )

    return decorator(inner) if inner is not None else decorator
