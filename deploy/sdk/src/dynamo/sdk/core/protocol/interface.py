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

import abc
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum, auto
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, Type, TypeVar

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from .deployment import Env

T = TypeVar("T", bound=object)


class AbstractService(abc.ABC):
    """Base class for Dynamo service interfaces."""

    pass


class LeaseConfig(BaseModel):
    """Configuration for custom dynamo leases"""

    ttl: int = 1  # seconds


class ComponentType:
    """Types of Dynamo components"""

    PLANNER = "planner"


class DynamoConfig(BaseModel):
    """Configuration for Dynamo components"""

    enabled: bool = True
    name: str | None = None
    namespace: str | None = None
    custom_lease: LeaseConfig | None = None
    component_type: str | None = None  # Indicates if this is a meta/system component


class DynamoTransport(Enum):
    """Transport types supported by Dynamo services"""

    DEFAULT = auto()
    HTTP = auto()


class ResourceConfig(BaseModel):
    """Configuration for Dynamo resources"""

    # auto convert gpu and cpu values to string from int
    model_config = ConfigDict(coerce_numbers_to_str=True)

    cpu: str = Field(default="1")
    memory: str = Field(default="500Mi")
    gpu: str = Field(default="0")


class ServiceConfig(BaseModel):
    """Base service configuration that can be extended by adapters"""

    dynamo: DynamoConfig
    resources: ResourceConfig = ResourceConfig()
    workers: int = 1
    image: str | None = None
    envs: List[Env] | None = None
    labels: Dict[str, str] | None = None


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

    def link(self, next_service: "ServiceInterface") -> "ServiceInterface":
        """Link this service to another service, creating a pipeline.

        This method allows linking (injecting) a concrete service implementation by checking if there is a dependency that next_service implements/inherits from.

        Args:
            next_service: The concrete service implementation to link

        Returns:
            The next_service that was linked to this service

        Raises:
            ValueError: If no matching interface is found or if multiple matches are found
        """
        if not isinstance(next_service, ServiceInterface):
            raise ValueError(f"link must be passed a Service, got {type(next_service)}")

        # Get all the deps of the service
        inner_deps = [
            (dep.on.inner, dep_key, dep)
            for dep_key, dep in self.dependencies.items()
            if dep.on is not None
        ]  # type: ignore

        # Get the inner class of the passed in service
        curr_inner = next_service.inner

        # Find deps that next_service implements/inherits from
        matching_deps = []
        for dep_inner, dep_key, original_dep in inner_deps:
            if issubclass(curr_inner, dep_inner):
                matching_deps.append((dep_inner, dep_key, original_dep))

        if not matching_deps:
            raise ValueError(
                f"{curr_inner.__name__} does not fulfill any dependencies required by {self.name}"
            )

        if len(matching_deps) > 1:
            dep_names = [dep_key for _, _, dep_key in matching_deps]
            raise ValueError(
                f"{curr_inner.__name__} fulfills multiple dependencies required by {self.name}: {dep_names}"
            )

        # Get the matching interface, dep_key, and original dependency
        _, _, matching_dep = matching_deps[0]

        # Let's hot swap the on of the existing dependency with the new service
        matching_dep.on = next_service

        # Record the link
        LinkedServices.add((self, next_service))
        return next_service

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

    def is_servable(self) -> bool:
        """Check if this service is ready to be served.

        A service is servable if:
        1. It is not a subclass of AbstractService (concrete service)
        2. If it is a subclass of AbstractService, all abstract methods are implemented
           with @dynamo_endpoint decorators
        """
        # If not a AbstractService, it's servable by default
        if not issubclass(self.inner, AbstractService):
            return True

        # For AbstractService, check implementations
        abstract_endpoints = _get_abstract_dynamo_endpoints(self.inner)
        if (
            not abstract_endpoints
        ):  # No abstract endpoints to implement, so it's servable
            return True
        return all(
            _check_dynamo_endpoint_implemented(self.inner, name)
            for name in abstract_endpoints
        )  # type: ignore[return-value]


def _get_abstract_dynamo_endpoints(cls: type) -> Set[str]:
    """Get all abstract endpoint names from the class's MRO."""
    return {
        name
        for base in cls.mro()
        for name, val in base.__dict__.items()
        if getattr(val, "__is_abstract_dynamo__", False)
    }


def _check_dynamo_endpoint_implemented(cls: type, name: str) -> bool:
    """Check if an endpoint is properly implemented."""
    impl = getattr(cls, name, None)
    # Ensure the implementation is a callable DynamoEndpointInterface
    return (
        impl is not None
        and callable(impl)
        and isinstance(impl, DynamoEndpointInterface)
    )


def validate_dynamo_interfaces(cls: type) -> None:
    """
    Validate that *cls* fully implements every @abstract_endpoint
    declared in its ancestors and that each implementation is
    decorated with @dynamo_endpoint.
    """
    required = _get_abstract_dynamo_endpoints(cls)

    missing: List[str] = []
    undecorated: List[str] = []
    not_callable: List[Tuple[str, str]] = []

    for name in required:
        impl = getattr(cls, name, None)
        if impl is None:
            missing.append(name)
            continue

        if not callable(impl):
            not_callable.append((name, type(impl).__name__))
            continue

        if not isinstance(impl, DynamoEndpointInterface):
            undecorated.append(name)

    problems = []
    if missing:
        problems.append(f"missing implementation(s): {', '.join(missing)}")
    if undecorated:
        problems.append(
            f"method(s) not decorated with @endpoint: {', '.join(undecorated)}"
        )
    if not_callable:
        problems.append(
            ", ".join(f"{n} must be callable, got {kind}" for n, kind in not_callable)
        )

    if problems:
        raise TypeError(
            f"{cls.__name__} violates Dynamo interface — " + "; ".join(problems)
        )


class DeploymentTarget(ABC):
    """Interface for service provider implementations"""

    @abstractmethod
    def create_service(
        self,
        service_cls: Type[T],
        config: ServiceConfig,
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

    @on.setter
    @abstractmethod
    def on(self, value: Optional[ServiceInterface[T]]) -> None:
        """Set the service this dependency is on"""
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
