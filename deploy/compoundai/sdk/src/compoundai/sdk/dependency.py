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
from compoundai.sdk.service import CompoundService

T = TypeVar("T")


class NovaClient:
    """Client for calling Nova endpoints with streaming support"""

    def __init__(self, service: CompoundService[Any]):
        self._service = service
        self._endpoints = service.get_nova_endpoints()
        self._nova_clients: Dict[str, Any] = {}
        self._runtime = None

    def __getattr__(self, name: str) -> Any:
        if name not in self._endpoints:
            raise AttributeError(
                f"No Nova endpoint '{name}' found on service '{self._service.name}'. "
                f"Available endpoints: {list(self._endpoints.keys())}"
            )

        # For streaming endpoints, create/cache the stream function
        if name not in self._nova_clients:
            namespace, component_name = self._service.nova_address()

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
                    # Create nova worker if no runtime
                    from triton_distributed_rs import DistributedRuntime, triton_worker

                    @triton_worker()
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

            self._nova_clients[name] = get_stream

        return self._nova_clients[name]


class NovaDependency(Dependency[T]):
    """Enhanced dependency that supports Nova endpoints"""

    def __init__(
        self,
        on: Service[T] | None = None,
        url: str | None = None,
        deployment: str | None = None,
        cluster: str | None = None,
    ):
        super().__init__(on, url=url, deployment=deployment, cluster=cluster)
        self._nova_client: Optional[NovaClient] = None
        self._runtime = None

    def set_runtime(self, runtime: Any) -> None:
        """Set the Nova runtime for this dependency"""
        self._runtime = runtime
        if self._nova_client:
            self._nova_client._runtime = runtime

    def get(self, *args: Any, **kwargs: Any) -> T | Any:
        # If this is a Nova-enabled service, return the Nova client
        if isinstance(self.on, CompoundService) and self.on.is_nova_component():
            if self._nova_client is None:
                self._nova_client = NovaClient(self.on)
                if self._runtime:
                    self._nova_client._runtime = self._runtime
            return self._nova_client

        # Otherwise fall back to normal BentoML dependency resolution
        return super().get(*args, **kwargs)


def depends(
    on: Service[T] | None = None,
    *,
    url: str | None = None,
    deployment: str | None = None,
    cluster: str | None = None,
) -> NovaDependency[T]:
    """Create a dependency that's Nova-aware.

    If the dependency is on a Nova-enabled service, this will return a client
    that can call Nova endpoints. Otherwise behaves like normal BentoML dependency.

    Args:
        on: The service to depend on
        url: URL for remote service
        deployment: Deployment name
        cluster: Cluster name

    Raises:
        AttributeError: When trying to call a non-existent Nova endpoint
    """
    if on is not None and not isinstance(on, Service):
        raise TypeError("depends() expects a class decorated with @service()")
    return NovaDependency(on, url=url, deployment=deployment, cluster=cluster)
