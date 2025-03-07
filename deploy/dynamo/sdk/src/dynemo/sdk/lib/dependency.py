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
from typing import Any, Dict, Optional, TypeVar

from _bentoml_sdk.service import Service
from _bentoml_sdk.service.dependency import Dependency
from dynemo.sdk.lib.service import CompoundService

T = TypeVar("T")


class DynemoClient:
    """Client for calling Dynemo endpoints with streaming support"""

    def __init__(self, service: CompoundService[Any]):
        self._service = service
        self._endpoints = service.get_dynemo_endpoints()
        self._dynemo_clients: Dict[str, Any] = {}
        self._runtime = None

    def __getattr__(self, name: str) -> Any:
        if name not in self._endpoints:
            raise AttributeError(
                f"No Dynemo endpoint '{name}' found on service '{self._service.name}'. "
                f"Available endpoints: {list(self._endpoints.keys())}"
            )

        # For streaming endpoints, create/cache the stream function
        if name not in self._dynemo_clients:
            namespace, component_name = self._service.dynemo_address()

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
                                print(f"Item data: {data}")
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
                                print(f"Item data: {data}")
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

            self._dynemo_clients[name] = get_stream

        return self._dynemo_clients[name]


class DynemoDependency(Dependency[T]):
    """Enhanced dependency that supports Dynemo endpoints"""

    def __init__(
        self,
        on: Service[T] | None = None,
        url: str | None = None,
        deployment: str | None = None,
        cluster: str | None = None,
    ):
        super().__init__(on, url=url, deployment=deployment, cluster=cluster)
        self._dynemo_client: Optional[DynemoClient] = None
        self._runtime = None

    # offers an escape hatch to get the endpoint directly
    async def get_endpoint(self, name: str) -> Any:
        """
        usage:
        dep = depends(Worker)

        ...
        await dep.get_endpoint("generate") # equivalent to the following
        router_client = (
            await runtime.namespace("dynemo-init")
            .component("router")
            .endpoint("generate")
            .client()
        )

        """
        # TODO: Read the runtime from the tdist since it is not stored in global
        if self._runtime is None:
            print(
                "Get Endpoint: Runtime not set for DynemoDependency. Cannot get endpoint."
            )
            raise ValueError("Runtime not set for DynemoDependency")

        address = self.on.dynemo_address()
        comp_ns, comp_name = address
        print("Get Endpoint: Dynemo ADDRESS: ", address)
        return (
            await self._runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint(name)
            .client()
        )

    def set_runtime(self, runtime: Any) -> None:
        """Set the Dynemo runtime for this dependency"""
        self._runtime = runtime
        if self._dynemo_client:
            self._dynemo_client._runtime = runtime

    def get(self, *args: Any, **kwargs: Any) -> T | Any:
        # If this is a Dynemo-enabled service, return the Dynemo client
        if isinstance(self.on, CompoundService) and self.on.is_dynemo_component():
            if self._dynemo_client is None:
                self._dynemo_client = DynemoClient(self.on)
                if self._runtime:
                    self._dynemo_client._runtime = self._runtime
            return self._dynemo_client

        # Otherwise fall back to normal BentoML dependency resolution
        return super().get(*args, **kwargs)


def depends(
    on: Service[T] | None = None,
    *,
    url: str | None = None,
    deployment: str | None = None,
    cluster: str | None = None,
) -> DynemoDependency[T]:
    """Create a dependency that's Dynemo-aware.

    If the dependency is on a Dynemo-enabled service, this will return a client
    that can call Dynemo endpoints. Otherwise behaves like normal BentoML dependency.

    Args:
        on: The service to depend on
        url: URL for remote service
        deployment: Deployment name
        cluster: Cluster name

    Raises:
        AttributeError: When trying to call a non-existent Dynemo endpoint
    """
    if on is not None and not isinstance(on, Service):
        raise TypeError("depends() expects a class decorated with @service()")
    return DynemoDependency(on, url=url, deployment=deployment, cluster=cluster)
