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
#  Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, Type, TypeVar

from fastapi import FastAPI

from dynamo.sdk.core.protocol.deployment import Env

T = TypeVar("T", bound=object)


class DynamoTransport(Enum):
    """Transport types supported by Dynamo services"""

    DEFAULT = auto()
    HTTP = auto()


class ServiceConfig(Dict[str, Any]):
    """Base service configuration that can be extended by adapters"""

    pass


class DynamoEndpointInterface(ABC):
    """Generic interface for service endpoints"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this endpoint"""
        pass

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the endpoint implementation"""
        pass

    @property
    @abstractmethod
    def transports(self) -> List[DynamoTransport]:
        """Get the transport type of this endpoint"""
        return [DynamoTransport.DEFAULT]


class ServiceInterface(Generic[T], ABC):
    """Generic interface for service implementations"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the service name"""
        pass

    @property
    @abstractmethod
    def config(self) -> ServiceConfig:
        """Get the service configuration"""
        pass

    @property
    def dependencies(self) -> Dict[str, "DependencyInterface"]:
        """Get the service dependencies"""
        return {}

    @property
    @abstractmethod
    def envs(self) -> List[Env]:
        """Get the service's environment variables"""
        return []

    @property
    @abstractmethod
    def inner(self) -> Type[T]:
        """Get the inner service implementation class"""
        pass

    @abstractmethod
    def get_endpoints(self) -> Dict[str, DynamoEndpointInterface]:
        """Get all registered endpoints"""
        pass

    @abstractmethod
    def get_endpoint(self, name: str) -> DynamoEndpointInterface:
        """Get a specific endpoint by name"""
        pass

    @abstractmethod
    def list_endpoints(self) -> List[str]:
        """List names of all registered endpoints"""
        pass

    @abstractmethod
    def link(self, next_service: "ServiceInterface") -> "ServiceInterface":
        """Link this service to another service, creating a pipeline"""
        pass

    @abstractmethod
    def remove_unused_edges(self, used_edges: Set["ServiceInterface"]) -> None:
        """Remove unused dependencies"""
        pass

    @abstractmethod
    def inject_config(self) -> None:
        """Inject configuration from environment into service configs"""
        pass

    @abstractmethod
    def get_service_configs(self) -> Dict[str, ServiceConfig]:
        """Get all services"""
        return {}

    @property
    def service_configs(self) -> List[ServiceConfig]:
        """Get all service configs"""
        return []

    def all_services(self) -> Dict[str, "ServiceInterface"]:
        """Get all services"""
        return {self.name: self}

    def get_dynamo_endpoints(self) -> Dict[str, DynamoEndpointInterface]:
        """Get all Dynamo endpoints"""
        endpoints = {}
        for field in dir(self.inner):
            value = getattr(self.inner, field)
            if isinstance(value, DynamoEndpointInterface):
                endpoints[value.name] = value
        return endpoints

    def __call__(self) -> T:
        return self.inner()

    def find_dependent_by_name(self, service_name: str) -> "ServiceInterface":
        """Find a dependent service by name"""
        raise NotImplementedError()

    def dynamo_address(self) -> tuple[str, str]:
        raise NotImplementedError()


@dataclass
class LeaseConfig:
    """Configuration for custom dynamo leases"""

    ttl: int = 1  # seconds


class ComponentType(str, Enum):
    """Types of Dynamo components"""

    PLANNER = "planner"


@dataclass
class DynamoConfig:
    """Configuration for Dynamo components"""

    enabled: bool = True
    name: str | None = None
    namespace: str | None = None
    custom_lease: LeaseConfig | None = None
    component_type: ComponentType | None = (
        None  # Indicates if this is a meta/system component
    )


class DeploymentTarget(ABC):
    """Interface for service provider implementations"""

    @abstractmethod
    def create_service(
        self,
        service_cls: Type[T],
        config: ServiceConfig,
        dynamo_config: Optional[DynamoConfig] = None,
        app: Optional[FastAPI] = None,
        **kwargs,
    ) -> ServiceInterface[T]:
        """Create a service instance"""
        pass

    @abstractmethod
    def create_dependency(
        self, on: Optional[ServiceInterface[T]] = None, **kwargs
    ) -> "DependencyInterface[T]":
        """Create a dependency on a service"""
        pass


class DependencyInterface(Generic[T], ABC):
    """Generic interface for service dependencies"""

    @property
    @abstractmethod
    def on(self) -> Optional[ServiceInterface[T]]:
        """Get the service this dependency is on"""
        pass

    @abstractmethod
    def get(self, *args: Any, **kwargs: Any) -> Any:
        """Get the dependency client"""
        pass

    @abstractmethod
    async def get_endpoint(self, name: str) -> Any:
        """Get a specific endpoint from the service"""
        pass

    def __get__(
        self: "DependencyInterface[T]", instance: Any, owner: Any
    ) -> "DependencyInterface[T]" | T:
        raise NotImplementedError()


class RuntimeLinkedServices:
    """
    A class to track the linked services in the runtime.
    """

    def __init__(self) -> None:
        self.edges: Dict[ServiceInterface, Set[ServiceInterface]] = defaultdict(set)

    def add(self, edge: Tuple[ServiceInterface, ServiceInterface]):
        src, dest = edge
        self.edges[src].add(dest)
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
