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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

from _bentoml_sdk import Service, ServiceConfig
from _bentoml_sdk.images import Image

from dynamo.sdk.lib.decorators import DynamoEndpoint

T = TypeVar("T", bound=object)


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
        super().__init__(config=config, inner=inner, image=image, envs=envs or [])

        # Initialize Dynamo configuration
        self._dynamo_config = (
            dynamo_config
            if dynamo_config
            else DynamoConfig(name=inner.__name__, namespace="default")
        )
        if self._dynamo_config.name is None:
            self._dynamo_config.name = inner.__name__

        # Register Dynamo endpoints
        self._dynamo_endpoints: Dict[str, DynamoEndpoint] = {}
        for field in dir(inner):
            value = getattr(inner, field)
            if isinstance(value, DynamoEndpoint):
                self._dynamo_endpoints[value.name] = value

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
    
    @classmethod
    def link(cls, next_service):
        """
        Link this service to another service, creating a pipeline.
        Returns a Graph object for further chaining.
        """
        from dynamo.sdk.lib.dependency import Graph
        print(f"DEBUG: Importing Graph for service {cls.name}")
        return Graph(cls, next_service)

    # todo: add another function to bind an instance of the inner to the self within these methods


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
