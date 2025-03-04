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
from compoundai.sdk.decorators import NovaEndpoint

T = TypeVar("T", bound=object)


@dataclass
class NovaConfig:
    """Configuration for Nova components"""

    enabled: bool = False
    name: str | None = None
    namespace: str | None = None


class CompoundService(Service[T]):
    """A custom service class that extends BentoML's base Service with Nova capabilities"""

    def __init__(
        self,
        config: ServiceConfig,
        inner: type[T],
        image: Optional[Image] = None,
        envs: Optional[list[dict[str, Any]]] = None,
        nova_config: Optional[NovaConfig] = None,
    ):
        super().__init__(config=config, inner=inner, image=image, envs=envs or [])

        # Initialize Nova configuration
        self._nova_config = (
            nova_config
            if nova_config
            else NovaConfig(name=inner.__name__, namespace="default")
        )
        if self._nova_config.name is None:
            self._nova_config.name = inner.__name__

        # Register Nova endpoints
        self._nova_endpoints: Dict[str, NovaEndpoint] = {}
        for field in dir(inner):
            value = getattr(inner, field)
            if isinstance(value, NovaEndpoint):
                self._nova_endpoints[value.name] = value

    def is_nova_component(self) -> bool:
        """Check if this service is configured as a Nova component"""
        return self._nova_config.enabled

    def nova_address(self) -> Tuple[str, str]:
        """Get the Nova address for this component in namespace/name format"""
        if not self.is_nova_component():
            raise ValueError("Service is not configured as a Nova component")

        # Check if we have a runner map with Nova address
        runner_map = os.environ.get("BENTOML_RUNNER_MAP")
        if runner_map:
            try:
                runners = json.loads(runner_map)
                if self.name in runners:
                    address = runners[self.name]
                    if address.startswith("nova://"):
                        # Parse nova://namespace/name into (namespace, name)
                        _, path = address.split("://", 1)
                        namespace, name = path.split("/", 1)
                        print(
                            f"Resolved Nova address from runner map: {namespace}/{name}"
                        )
                        return (namespace, name)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"Failed to parse BENTOML_RUNNER_MAP: {str(e)}") from e

        # Ensure namespace and name are not None
        namespace = self._nova_config.namespace or "default"
        name = self._nova_config.name or self.inner.__name__

        print(f"Using default Nova address: {namespace}/{name}")
        return (namespace, name)

    def get_nova_endpoints(self) -> Dict[str, NovaEndpoint]:
        """Get all registered Nova endpoints"""
        return self._nova_endpoints

    def get_nova_endpoint(self, name: str) -> NovaEndpoint:
        """Get a specific Nova endpoint by name"""
        if name not in self._nova_endpoints:
            raise ValueError(f"No Nova endpoint found with name: {name}")
        return self._nova_endpoints[name]

    def list_nova_endpoints(self) -> List[str]:
        """List names of all registered Nova endpoints"""
        return list(self._nova_endpoints.keys())

    # todo: add another function to bind an instance of the inner to the self within these methods


def service(
    inner: Optional[type[T]] = None,
    /,
    *,
    image: Optional[Image] = None,
    envs: Optional[list[dict[str, Any]]] = None,
    nova: Optional[Union[Dict[str, Any], NovaConfig]] = None,
    **kwargs: Any,
) -> Any:
    """Enhanced service decorator that supports Nova configuration

    Args:
        nova: Nova configuration, either as a NovaConfig object or dict with keys:
            - enabled: bool (default True)
            - name: str (default: class name)
            - namespace: str (default: "default")
        **kwargs: Existing BentoML service configuration
    """
    config = kwargs

    # Parse dict into NovaConfig object
    nova_config: Optional[NovaConfig] = None
    if nova is not None:
        if isinstance(nova, dict):
            nova_config = NovaConfig(**nova)
        else:
            nova_config = nova

    def decorator(inner: type[T]) -> CompoundService[T]:
        if isinstance(inner, Service):
            raise TypeError("service() decorator can only be applied once")
        return CompoundService(
            config=config,
            inner=inner,
            image=image,
            envs=envs or [],
            nova_config=nova_config,
        )

    return decorator(inner) if inner is not None else decorator
