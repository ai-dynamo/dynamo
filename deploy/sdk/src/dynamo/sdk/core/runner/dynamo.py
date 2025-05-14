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

import contextlib
import logging
import os
import shlex
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Type, TypeVar

import psutil
from circus.arbiter import Arbiter
from circus.sockets import CircusSocket
from circus.watcher import Watcher
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

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=object)

MAX_AF_UNIX_PATH_LENGTH = 103


class LocalEndpoint(DynamoEndpoint):
    """Circus-specific endpoint implementation"""

    def __init__(
        self, name: str, service: "LocalService", transports: List[DynamoTransport]
    ):
        self._name = name
        self._service = service
        self._transports = transports

    @property
    def name(self) -> str:
        return self._name


class LocalService(ServiceMixin, ServiceInterface[T]):
    """Circus implementation of the ServiceInterface"""

    def __init__(
        self,
        inner_cls: Type[T],
        config: ServiceConfig,
        dynamo_config: Optional[DynamoConfig] = None,
        watcher: Optional[Watcher] = None,
        socket: Optional[CircusSocket] = None,
        app: Optional[FastAPI] = None,
    ):
        self._inner_cls = inner_cls
        self._config = config
        name = inner_cls.__name__
        self._dynamo_config = dynamo_config or DynamoConfig(
            name=name, namespace="default"
        )
        # Add the dynamo config to the service config
        self._config["dynamo"] = asdict(self._dynamo_config)
        self._watcher = watcher
        self._socket = socket
        self.app = app or FastAPI(title=name)
        self._dependencies: Dict[str, "DependencyInterface"] = {}
        self._endpoints: Dict[str, LocalEndpoint] = {}
        for field_name in dir(inner_cls):
            field = getattr(inner_cls, field_name)
            if isinstance(field, DynamoEndpoint):
                self._endpoints[field.name] = LocalEndpoint(
                    field.name, self, field.transports
                )
            if isinstance(field, DependencyInterface):
                self._dependencies[field_name] = field

    def find_dependent_by_name(self, name: str) -> "ServiceInterface":
        return self.all_services()[name]

    def all_services(self) -> dict[str, "ServiceInterface"]:
        """Get a map of the service and all recursive dependencies"""
        services: dict[str, "ServiceInterface"] = {self.name: self}
        for dependency in self.dependencies.values():
            services.update(dependency.on.all_services())
        return services

    @property
    def name(self) -> str:
        return self._inner_cls.__name__

    @property
    def config(self) -> ServiceConfig:
        return self._config

    @property
    def inner(self) -> Type[T]:
        return self._inner_cls

    def get_endpoints(self) -> Dict[str, DynamoEndpointInterface]:
        return self._endpoints

    def get_endpoint(self, name: str) -> DynamoEndpointInterface:
        if name not in self._endpoints:
            raise ValueError(f"No endpoint found with name: {name}")
        return self._endpoints[name]

    def list_endpoints(self) -> List[str]:
        return list(self._endpoints.keys())

    def link(self, next_service: "ServiceInterface") -> "ServiceInterface":
        LinkedServices.add((self, next_service))
        return next_service

    def remove_unused_edges(self, used_edges: Set["ServiceInterface"]) -> None:
        current_deps = dict(self._dependencies)
        for dep_key, dep_value in current_deps.items():
            if dep_value.on not in used_edges:
                del self._dependencies[dep_key]

    def dynamo_address(self) -> tuple[str, str]:
        return (self._dynamo_config.namespace, self._dynamo_config.name)

    @property
    def dependencies(self) -> dict[str, "DependencyInterface"]:
        return self._dependencies

    @property
    def endpoints(self) -> dict[str, "LocalEndpoint"]:
        return self._endpoints


class LocalDependency(DependencyInterface[T]):
    """Circus implementation of the DependencyInterface"""

    def __init__(
        self,
        on_service: Optional[LocalService[T]] = None,
    ):
        self._on_service = on_service
        self._dynamo_client = None
        self._runtime = None

    @property
    def on(self) -> Optional[ServiceInterface[T]]:
        return self._on_service

    def get(self, *args: Any, **kwargs: Any) -> Any:
        # Return a client that can communicate with the service
        # through the circus socket
        if not self._on_service:
            raise ValueError("No service specified for this dependency")
        return self._on_service

    async def get_endpoint(self, name: str) -> Any:
        # Get a specific endpoint from the service
        if not self._on_service:
            raise ValueError("No service specified for this dependency")
        return await self._on_service.get_endpoint(name)

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

    def set_runtime(self, runtime: Any) -> None:
        """Set the Dynamo runtime for this dependency"""
        self._runtime = runtime
        if self._dynamo_client:
            self._dynamo_client._runtime = runtime


class LocalDeploymentTarget(DeploymentTarget):
    """Circus implementation of the DeploymentTarget"""

    def __init__(self):
        self._arbiter = None
        self._watchers = []
        self._sockets = {}

    def create_service(
        self,
        service_cls: Type[T],
        config: ServiceConfig,
        dynamo_config: Optional[DynamoConfig] = None,
        **kwargs,
    ) -> ServiceInterface[T]:
        # Get parameters needed for creating a circus watcher
        cmd = kwargs.get("cmd", sys.executable)
        args = kwargs.get("args", [])
        env_vars = kwargs.get("env_vars", {})

        # Create a socket for this service
        socket_path = os.path.join(
            os.environ.get("DYN_CIRCUS_SOCKET_DIR", "/tmp/circus"),
            f"{service_cls.__name__}.sock",
        )

        # Ensure the socket path isn't too long
        if len(socket_path) >= MAX_AF_UNIX_PATH_LENGTH:
            raise ValueError(
                f"Socket path '{socket_path}' exceeds maximum length of {MAX_AF_UNIX_PATH_LENGTH}"
            )

        # Create the socket
        socket = CircusSocket(name=service_cls.__name__, path=socket_path)
        self._sockets[service_cls.__name__] = socket

        # Create a watcher for the service
        watcher = Watcher(
            name=service_cls.__name__,
            cmd=shlex.quote(cmd) if psutil.POSIX else cmd,
            args=args,
            copy_env=True,
            env=env_vars,
            stop_children=True,
            use_sockets=True,
            graceful_timeout=86400,
            respawn=True,
        )
        self._watchers.append(watcher)

        # Create and return the service interface
        return LocalService(
            inner_cls=service_cls,
            config=config,
            dynamo_config=dynamo_config,
            watcher=watcher,
            socket=socket,
        )

    def create_dependency(
        self, on: Optional[ServiceInterface[T]] = None, **kwargs
    ) -> DependencyInterface[T]:
        # Ensure the dependency is on a LocalService
        if on is not None and not isinstance(on, LocalService):
            raise TypeError("LocalDependency can only depend on LocalService")

        # Create and return the dependency interface
        return LocalDependency(on)

    def start_arbiter(self, threaded: bool = False, **kwargs: Any) -> Arbiter:
        """Start the circus arbiter with all configured watchers and sockets"""
        if self._arbiter is not None:
            logger.warning("Arbiter already started")
            return self._arbiter

        # Configure arbiter
        endpoint_port = int(os.environ.get("DYN_CIRCUS_ENDPOINT_PORT", "41234"))
        pubsub_port = int(os.environ.get("DYN_CIRCUS_PUBSUB_PORT", "52345"))

        # Create arbiter with all sockets and watchers
        arbiter = Arbiter(
            watchers=self._watchers,
            sockets=[socket for socket in self._sockets.values()],
            endpoint=f"tcp://127.0.0.1:{endpoint_port}",
            pubsub_endpoint=f"tcp://127.0.0.1:{pubsub_port}",
            check_delay=kwargs.pop("check_delay", 10),
            **kwargs,
        )

        # Start arbiter
        arbiter.start()
        self._arbiter = arbiter
        return arbiter

    def stop_arbiter(self) -> None:
        """Stop the circus arbiter and all managed processes"""
        if self._arbiter is None:
            logger.warning("No arbiter to stop")
            return

        self._arbiter.stop()
        self._arbiter = None

    @contextlib.contextmanager
    def run_services(self, **kwargs: Any):
        """Context manager to run all services and clean up when done"""
        try:
            arbiter = self.start_arbiter(**kwargs)
            yield arbiter
        finally:
            self.stop_arbiter()
