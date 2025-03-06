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
from compoundai.sdk.decorators import DynemoEndpoint

T = TypeVar("T", bound=object)


@dataclass
class DynemoConfig:
    """Configuration for Dynemo components"""

    enabled: bool = False
    name: str | None = None
    namespace: str | None = None


class CompoundService(Service[T]):
    """A custom service class that extends BentoML's base Service with Dynemo capabilities"""

    def __init__(
        self,
        config: ServiceConfig,
        inner: type[T],
        image: Optional[Image] = None,
        envs: Optional[list[dict[str, Any]]] = None,
        dynemo_config: Optional[DynemoConfig] = None,
    ):
        super().__init__(config=config, inner=inner, image=image, envs=envs or [])

        # Initialize Dynemo configuration
        self._dynemo_config = (
            dynemo_config
            if dynemo_config
            else DynemoConfig(name=inner.__name__, namespace="default")
        )
        if self._dynemo_config.name is None:
            self._dynemo_config.name = inner.__name__

        # Register Dynemo endpoints
        self._dynemo_endpoints: Dict[str, DynemoEndpoint] = {}
        for field in dir(inner):
            value = getattr(inner, field)
            if isinstance(value, DynemoEndpoint):
                self._dynemo_endpoints[value.name] = value

    def is_dynemo_component(self) -> bool:
        """Check if this service is configured as a Dynemo component"""
        return self._dynemo_config.enabled

    def dynemo_address(self) -> Tuple[str, str]:
        """Get the Dynemo address for this component in namespace/name format"""
        if not self.is_dynemo_component():
            raise ValueError("Service is not configured as a Dynemo component")

        # Check if we have a runner map with Dynemo address
        runner_map = os.environ.get("BENTOML_RUNNER_MAP")
        if runner_map:
            try:
                runners = json.loads(runner_map)
                if self.name in runners:
                    address = runners[self.name]
                    if address.startswith("dynemo://"):
                        # Parse dynemo://namespace/name into (namespace, name)
                        _, path = address.split("://", 1)
                        namespace, name = path.split("/", 1)
                        print(
                            f"Resolved Dynemo address from runner map: {namespace}/{name}"
                        )
                        return (namespace, name)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"Failed to parse BENTOML_RUNNER_MAP: {str(e)}") from e

        print(
            f"Using default Dynemo address: {self._dynemo_config.namespace}/{self._dynemo_config.name}"
        )
        return (self._dynemo_config.namespace, self._dynemo_config.name)

    def get_dynemo_endpoints(self) -> Dict[str, DynemoEndpoint]:
        """Get all registered Dynemo endpoints"""
        return self._dynemo_endpoints

    def get_dynemo_endpoint(self, name: str) -> DynemoEndpoint:
        """Get a specific Dynemo endpoint by name"""
        if name not in self._dynemo_endpoints:
            raise ValueError(f"No Dynemo endpoint found with name: {name}")
        return self._dynemo_endpoints[name]

    def list_dynemo_endpoints(self) -> List[str]:
        """List names of all registered Dynemo endpoints"""
        return list(self._dynemo_endpoints.keys())

    # todo: add another function to bind an instance of the inner to the self within these methods


def service(
    inner: Optional[type[T]] = None,
    /,
    *,
    image: Optional[Image] = None,
    envs: Optional[list[dict[str, Any]]] = None,
    dynemo: Optional[Union[Dict[str, Any], DynemoConfig]] = None,
    **kwargs: Any,
) -> Any:
    """Enhanced service decorator that supports Dynemo configuration

    Args:
        dynemo: Dynemo configuration, either as a DynemoConfig object or dict with keys:
            - enabled: bool (default True)
            - name: str (default: class name)
            - namespace: str (default: "default")
        **kwargs: Existing BentoML service configuration
    """
    config = kwargs

    # Parse dict into DynemoConfig object
    dynemo_config: Optional[DynemoConfig] = None
    if dynemo is not None:
        if isinstance(dynemo, dict):
            dynemo_config = DynemoConfig(**dynemo)
        else:
            dynemo_config = dynemo

    def decorator(inner: type[T]) -> CompoundService[T]:
        if isinstance(inner, Service):
            raise TypeError("service() decorator can only be applied once")
        return CompoundService(
            config=config,
            inner=inner,
            image=image,
            envs=envs or [],
            dynemo_config=dynemo_config,
        )

    return decorator(inner) if inner is not None else decorator
