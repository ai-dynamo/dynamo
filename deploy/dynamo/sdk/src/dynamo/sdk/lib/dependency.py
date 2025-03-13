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

import asyncio
from typing import Any, Dict, Optional, TypeVar, List, Type

from _bentoml_sdk.service import Service
from _bentoml_sdk.service.dependency import Dependency

from dynamo.sdk.lib.service import DynamoService

T = TypeVar("T")


class DynamoClient:
    """Client for calling Dynamo endpoints with streaming support"""

    def __init__(self, service: DynamoService[Any]):
        self._service = service
        self._endpoints = service.get_dynamo_endpoints()
        self._dynamo_clients: Dict[str, Any] = {}
        self._runtime = None

    def __getattr__(self, name: str) -> Any:
        if name not in self._endpoints:
            raise AttributeError(
                f"No Dynamo endpoint '{name}' found on service '{self._service.name}'. "
                f"Available endpoints: {list(self._endpoints.keys())}"
            )

        # For streaming endpoints, create/cache the stream function
        if name not in self._dynamo_clients:
            namespace, component_name = self._service.dynamo_address()

            # Create async generator function that uses Queue for streaming
            async def get_stream(*args, **kwargs):
                queue: asyncio.Queue = asyncio.Queue()

                if self._runtime is not None:
                    # Use existing runtime if available
                    async def stream_worker():
                        try:
                            client = (
                                await self._runtime.namespace(namespace)
                                .component(component_name)
                                .endpoint(name)
                                .client()
                            )

                            # TODO: Potentially model dump for a user here so they can pass around Pydantic models
                            stream = await client.generate(*args, **kwargs)
                            async for item in stream:
                                data = item.data()
                                await queue.put(data)
                            await queue.put(None)
                        except Exception:
                            await queue.put(None)
                            raise

                else:
                    # Create dynamo worker if no runtime
                    from dynamo.runtime import DistributedRuntime, dynamo_worker

                    @dynamo_worker()
                    async def stream_worker(runtime: DistributedRuntime):
                        try:
                            # Store runtime for future use
                            self._runtime = runtime

                            client = (
                                await runtime.namespace(namespace)
                                .component(component_name)
                                .endpoint(name)
                                .client()
                            )

                            stream = await client.generate(*args, **kwargs)
                            async for item in stream:
                                data = item.data()
                                await queue.put(data)
                            await queue.put(None)
                        except Exception:
                            await queue.put(None)
                            raise

                # Start worker task with error handling
                worker_task = asyncio.create_task(stream_worker())

                try:
                    # Yield items from queue until None received
                    while True:
                        item = await queue.get()
                        if item is None:
                            break
                        yield item
                finally:
                    try:
                        await worker_task
                    except Exception:
                        raise

            self._dynamo_clients[name] = get_stream

        return self._dynamo_clients[name]


class DynamoDependency(Dependency[T]):
    """Enhanced dependency that supports Dynamo endpoints"""

    def __init__(
        self,
        on: Service[T] | None = None,
        url: str | None = None,
        deployment: str | None = None,
        cluster: str | None = None,
    ):
        super().__init__(on, url=url, deployment=deployment, cluster=cluster)
        self._dynamo_client: Optional[DynamoClient] = None
        self._runtime = None

    # offers an escape hatch to get the endpoint directly
    async def get_endpoint(self, name: str) -> Any:
        """
        usage:
        dep = depends(Worker)

        ...
        await dep.get_endpoint("generate") # equivalent to the following
        router_client = (
            await runtime.namespace("dynamo-init")
            .component("router")
            .endpoint("generate")
            .client()
        )

        """
        # TODO: Read the runtime from the tdist since it is not stored in global
        if self._runtime is None:
            print(
                "Get Endpoint: Runtime not set for DynamoDependency. Cannot get endpoint."
            )
            raise ValueError("Runtime not set for DynamoDependency")

        address = self.on.dynamo_address()
        comp_ns, comp_name = address
        print("Get Endpoint: Dynamo ADDRESS: ", address)
        return (
            await self._runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint(name)
            .client()
        )

    def set_runtime(self, runtime: Any) -> None:
        """Set the Dynamo runtime for this dependency"""
        self._runtime = runtime
        if self._dynamo_client:
            self._dynamo_client._runtime = runtime

    def get(self, *args: Any, **kwargs: Any) -> T | Any:
        # If this is a Dynamo-enabled service, return the Dynamo client
        if isinstance(self.on, DynamoService) and self.on.is_dynamo_component():
            if self._dynamo_client is None:
                self._dynamo_client = DynamoClient(self.on)
                if self._runtime:
                    self._dynamo_client._runtime = self._runtime
            return self._dynamo_client

        # Otherwise fall back to normal BentoML dependency resolution
        return super().get(*args, **kwargs)


def depends(
    on: Service[T] | None = None,
    *,
    url: str | None = None,
    deployment: str | None = None,
    cluster: str | None = None,
) -> DynamoDependency[T]:
    """Create a dependency that's Dynamo-aware.

    If the dependency is on a Dynamo-enabled service, this will return a client
    that can call Dynamo endpoints. Otherwise behaves like normal BentoML dependency.

    Args:
        on: The service to depend on
        url: URL for remote service
        deployment: Deployment name
        cluster: Cluster name

    Raises:
        AttributeError: When trying to call a non-existent Dynamo endpoint
    """
    if on is not None and not isinstance(on, Service):
        raise TypeError("depends() expects a class decorated with @service()")
    return DynamoDependency(on, url=url, deployment=deployment, cluster=cluster)


def create_pipeline(*services):
    """
    Create a pipeline by linking services together.
    
    Args:
        *services: A sequence of service classes to link in order
        
    Returns:
        The first service in the chain for further configuration
        
    Example:
        pipeline = create_pipeline(Frontend, Backend, Middle)
    """
    if len(services) < 2:
        raise ValueError("At least two services are required to create a pipeline")
    
    # Reset dependencies for all services in the chain
    for service in services:
        service.dependencies = {}
    
    # Create the dependency chain
    for i in range(len(services) - 1):
        current = services[i]
        next_service = services[i + 1]
        
        # Use the lowercase name of the target service as the dependency key
        dep_name = next_service.name.lower()
        current.dependencies[dep_name] = DynamoDependency(next_service)
    
    return services[0]  # Return the head of the pipeline

class Graph:
    """
    A graph representing linked services in a pipeline.
    """
    
    def __init__(self, first_service: Type, second_service: Type):
        print(f"DEBUG: Creating new Graph with {first_service.name} -> {second_service.name}")
        self.services: List[Type] = [first_service, second_service]
        self.links: Dict[Type, Type] = {first_service: second_service}
        self.head = first_service
        self.current_service = second_service
        print(f"DEBUG: Initial links: {first_service.name} -> {second_service.name}")
    
    def link(self, next_service: Type) -> 'Graph':
        """
        Link the current service to the next service.
        Returns self for method chaining.
        """
        print(f"DEBUG: Adding link from {self.current_service.name} to {next_service.name}")
        if next_service not in self.services:
            self.services.append(next_service)
            
        self.links[self.current_service] = next_service
        self.current_service = next_service
        print(f"DEBUG: Current services in graph: {[s.name for s in self.services]}")
        print(f"DEBUG: Current links: {[(from_svc.name, to_svc.name) for from_svc, to_svc in self.links.items()]}")
        return self
    
    def apply(self) -> Type:
        """
        Apply the graph configuration by setting up dependencies.
        Returns the first service in the pipeline.
        """
        print(f"DEBUG: Applying graph with {len(self.services)} services")
        print(f"DEBUG: Services: {[s.name for s in self.services]}")
        print(f"DEBUG: Links: {[(from_svc.name, to_svc.name) for from_svc, to_svc in self.links.items()]}")
        
        print("DEBUG: Before clearing dependencies:")
        for service in self.services:
            print(f"DEBUG: {service.name} dependencies: {service.dependencies}")
            
        # Clear all existing dependencies
        for service in self.services:
            print(f"DEBUG: Clearing dependencies for {service.name}")
            service.dependencies = {}
            
        # Set up the new dependencies based on links
        for service, next_service in self.links.items():
            dep_name = next_service.name.lower()
            print(f"DEBUG: Setting dependency {service.name} -> {dep_name}")
            service.dependencies[dep_name] = DynamoDependency(next_service)
            
        print("DEBUG: After applying graph:")
        for service in self.services:
            print(f"DEBUG: {service.name} dependencies: {service.dependencies}")
            
        return self.head  # Return the first service

# We need this function to catch when it's being called
def link_service_method(cls, next_service):
    """
    Link this service to another service, creating a pipeline.
    """
    print(f"DEBUG: link_service_method called for {cls.name} -> {next_service.name}")
    return Graph(cls, next_service)

# Function to check and add the link method
def add_link_method(service_class):
    """
    Check if the class already has a link method, and add ours if not.
    """
    if hasattr(service_class, 'link'):
        print(f"DEBUG: {service_class.name} already has a link method: {service_class.link}")
    else:
        print(f"DEBUG: Adding link method to {service_class.name}")
        service_class.link = classmethod(link_service_method)
        print(f"DEBUG: Added link method: {service_class.link}")
    return service_class