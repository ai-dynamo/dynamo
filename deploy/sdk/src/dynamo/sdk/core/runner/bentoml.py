#  SPDX-FileCopyrightText: Copyright (c) 2020 Atalaya Tech. Inc
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

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Type, TypeVar

from _bentoml_sdk import Service as BentoService
from _bentoml_sdk.service.dependency import Dependency as BentoDependency
from fastapi import FastAPI

from dynamo.sdk.core.decorators.endpoint import DynamoClient, DynamoEndpoint
from dynamo.sdk.core.protocol.interface import (
    DependencyInterface,
    DeploymentTarget,
    DynamoConfig,
    DynamoEndpointInterface,
    DynamoTransport,
    LinkedServices,
    ServiceConfig,
    ServiceInterface,
)
from dynamo.sdk.core.runner.common import ServiceMixin

T = TypeVar("T", bound=object)


class BentoEndpoint(DynamoEndpoint):
    """BentoML-specific endpoint implementation"""

    def __init__(
        self,
        bentoml_endpoint: Any,
        name: Optional[str] = None,
        transports: Optional[List[DynamoTransport]] = None,
    ):
        self.bentoml_endpoint = bentoml_endpoint
        self._name = name or bentoml_endpoint.name
        self._transports = transports or bentoml_endpoint.transports

    @property
    def name(self) -> str:
        return self._name

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await self.bentoml_endpoint(*args, **kwargs)

    @property
    def transports(self) -> List[DynamoTransport]:
        return self._transports


class BentoServiceAdapter(ServiceMixin, ServiceInterface[T]):
    """BentoML adapter implementing the ServiceInterface"""

    def __init__(
        self,
        service_cls: Type[T],
        config: ServiceConfig,
        dynamo_config: Optional[DynamoConfig] = None,
        app: Optional[FastAPI] = None,
        **kwargs,
    ):
        name = service_cls.__name__
        self._dynamo_config = dynamo_config or DynamoConfig(
            name=name, namespace="default"
        )
        image = kwargs.get("image")
        envs = kwargs.get("envs", [])
        # attributes from decorators
        for attr in ["workers", "resources"]:
            if attr in kwargs:
                config[attr] = kwargs[attr]
        self.image = image
        # Get service args from environment if available
        service_args = self._get_service_args(name)
        if service_args:
            # Update config with service args
            for key, value in service_args.items():
                if key not in config:
                    config[key] = value

            # Extract and apply specific args if needed
            if "resources" in service_args:
                config["resources"] = service_args["resources"]
            if "workers" in service_args:
                config["workers"] = service_args["workers"]
            if "envs" in service_args and envs:
                envs.extend(service_args["envs"])
            elif "envs" in service_args:
                envs = service_args["envs"]
        # Initialize BentoML service
        self._bentoml_service = BentoService(
            config=config,
            inner=service_cls,
            image=image,
            envs=envs or [],
        )

        self._endpoints: Dict[str, BentoEndpoint] = {}
        if not app:
            self.app = FastAPI(title=name)
        else:
            self.app = app
        self._dependencies: Dict[str, "DependencyInterface"] = {}
        self._bentoml_service.config["dynamo"] = asdict(self._dynamo_config)
        self._api_endpoints: list[str] = []
        # Map BentoML endpoints to our generic interface
        for field_name in dir(service_cls):
            field = getattr(service_cls, field_name)
            if isinstance(field, DynamoEndpoint):
                self._endpoints[field.name] = BentoEndpoint(
                    field, field.name, field.transports
                )
                if DynamoTransport.HTTP in field.transports:
                    # Ensure endpoint path starts with '/'
                    path = (
                        field.name if field.name.startswith("/") else f"/{field.name}"
                    )
                    self._api_endpoints.append(path)
            if isinstance(field, DependencyInterface):
                self._dependencies[field_name] = field
        # If any API endpoints exist, mark service as HTTP-exposed and list endpoints
        if self._api_endpoints:
            self._bentoml_service.config["http_exposed"] = True
            self._bentoml_service.config["api_endpoints"] = self._api_endpoints.copy()

    @property
    def dependencies(self) -> dict[str, "DependencyInterface"]:
        return self._dependencies

    @property
    def name(self) -> str:
        return self._bentoml_service.name

    @property
    def config(self) -> ServiceConfig:
        return ServiceConfig(self._bentoml_service.config)

    @property
    def inner(self) -> Type[T]:
        return self._bentoml_service.inner

    def get_endpoints(self) -> Dict[str, DynamoEndpointInterface]:
        return self._endpoints

    def get_endpoint(self, name: str) -> DynamoEndpointInterface:
        if name not in self._endpoints:
            raise ValueError(f"No endpoint found with name: {name}")
        return self._endpoints[name]

    def list_endpoints(self) -> List[str]:
        return list(self._endpoints.keys())

    def link(self, next_service: "ServiceInterface") -> "ServiceInterface":
        # Check if the next service is a BentoML service adapter
        LinkedServices.add((self, next_service))
        return next_service

    def remove_unused_edges(self, used_edges: Set["ServiceInterface"]) -> None:
        current_deps = dict(self._dependencies)
        for dep_key, dep_value in current_deps.items():
            if dep_value.on not in used_edges:
                del self._dependencies[dep_key]

    # Add methods to expose underlying BentoML service when needed
    def get_bentoml_service(self) -> BentoService:
        return self._bentoml_service

    def __call__(self) -> T:
        instance = self.inner()
        return instance

    def find_dependent_by_name(self, name: str) -> "ServiceInterface":
        """Find dynamo service by name"""
        return self.all_services()[name]

    def dynamo_address(self) -> tuple[str, str]:
        return (self._dynamo_config.namespace, self._dynamo_config.name)

    def all_services(self) -> dict[str, "ServiceInterface"]:
        """Get a map of the service and all recursive dependencies"""
        services: dict[str, "ServiceInterface"] = {self.name: self}
        for dep in self.dependencies.values():
            services.update(dep.on.all_services())
        return services


class BentoMLDependency(DependencyInterface[T]):
    """BentoML adapter implementing the DependencyInterface"""

    def __init__(
        self,
        bentoml_dependency: BentoDependency,
        on_service: Optional[BentoServiceAdapter[T]] = None,
    ):
        self._bentoml_dependency = bentoml_dependency
        self._on_service = on_service
        self._dynamo_client = None
        self._runtime = None

    @property
    def on(self) -> Optional[ServiceInterface[T]]:
        return self._on_service

    def get(self, *args: Any, **kwargs: Any) -> Any:
        return self._bentoml_dependency.get(*args, **kwargs)

    def set_runtime(self, runtime: Any) -> None:
        """Set the Dynamo runtime for this dependency"""
        self._runtime = runtime
        if self._dynamo_client:
            self._dynamo_client._runtime = runtime

    async def get_endpoint(self, name: str) -> Any:
        # Implementation depends on what BentoML provides
        # This is a simplified version
        client = self.get()
        if hasattr(client, name):
            return getattr(client, name)
        raise ValueError(f"No endpoint found with name: {name}")

    # Add method to expose underlying BentoML dependency when needed
    def get_bentoml_dependency(self) -> BentoDependency:
        return self._bentoml_dependency

    def __get__(
        self: "DependencyInterface[T]", instance: Any, owner: Any
    ) -> "DependencyInterface[T]" | T | Any:
        if instance is None:
            return self
        if self._dynamo_client is None:
            self._dynamo_client = DynamoClient(self.on)
            if self._runtime:
                self._dynamo_client._runtime = self._runtime
        return self._dynamo_client


class BentoDeploymentTarget(DeploymentTarget):
    """Kubernetes implementation of the DeploymentTarget"""

    def create_service(
        self,
        service_cls: Type[T],
        config: ServiceConfig,
        dynamo_config: Optional[DynamoConfig] = None,
        app: Optional[FastAPI] = None,
        **kwargs,
    ) -> ServiceInterface[T]:
        """Create a BentoServiceAdapter with the given parameters"""
        return BentoServiceAdapter(
            service_cls=service_cls,
            config=config,
            dynamo_config=dynamo_config,
            app=app,
            **kwargs,
        )

    def create_dependency(
        self, on: Optional[ServiceInterface[T]] = None, **kwargs
    ) -> DependencyInterface[T]:
        url = kwargs.get("url")
        deployment = kwargs.get("deployment")
        cluster = kwargs.get("cluster")

        # Get the underlying BentoML service if available
        bentoml_service = None
        if on is not None and isinstance(on, BentoServiceAdapter):
            # this is underlying bentoml service
            bentoml_service = on.get_bentoml_service()

        # Create underlying BentoML dependency
        bentoml_dependency = BentoDependency(
            bentoml_service, url=url, deployment=deployment, cluster=cluster
        )

        # Wrap in our adapter
        return BentoMLDependency(bentoml_dependency, on)
