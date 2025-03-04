# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import AsyncGenerator, AsyncIterator, Callable, Dict, List, Optional

class JsonLike:
    """
    Any PyObject which can be serialized to JSON
    """

    ...

RequestHandler = Callable[[JsonLike], AsyncGenerator[JsonLike, None]]

class DistributedRuntime:
    """
    The runtime object for a distributed NOVA applications
    """

    ...

    def namespace(self, name: str, path: str) -> Namespace:
        """
        Create a `Namespace` object
        """
        ...

    def etcd_client(self) -> EtcdClient:
        """
        Get the `EtcdClient` object
        """
        ...

class EtcdClient:
    """
    Etcd is used for discovery in the DistributedRuntime
    """
    async def kv_create_or_validate(self, key: str, value: bytes, lease_id: Optional[int] = None) -> None:
        """
        Atomically create a key if it does not exist, or validate the values are identical if the key exists.
        """
        ...

    async def kv_put(self, key: str, value: bytes, lease_id: Optional[int] = None) -> None:
        """
        Put a key-value pair into etcd
        """
        ...

    async def kv_get_prefix(self, prefix: str) -> List[Dict[str, JsonLike]]:
        """
        Get all keys with a given prefix
        """
        ...

class Namespace:
    """
    A namespace is a collection of components
    """

    ...

    def component(self, name: str) -> Component:
        """
        Create a `Component` object
        """
        ...

class Component:
    """
    A component is a collection of endpoints
    """

    ...

    def create_service(self) -> None:
        """
        Create a service
        """
        ...

    def endpoint(self, name: str) -> Endpoint:
        """
        Create an endpoint
        """
        ...

    def event_subject(self, name: str) -> str:
        """
        Create an event subject
        """
        ...

class Endpoint:
    """
    An Endpoint is a single API endpoint
    """

    ...

    async def serve_endpoint(self, handler: RequestHandler) -> None:
        """
        Serve an endpoint discoverable by all connected clients at
        `{{ namespace }}/components/{{ component_name }}/endpoints/{{ endpoint_name }}`
        """
        ...

    async def client(self) -> Client:
        """
        Create a `Client` capable of calling served instances of this endpoint
        """
        ...

    async def lease_id(self) -> int:
        """
        Return primary lease id. Currently, cannot set a different lease id.
        """
        ...

class Client:
    """
    A client capable of calling served instances of an endpoint
    """

    ...

    async def random(self, request: JsonLike) -> AsyncIterator[JsonLike]:
        """
        Pick a random instance of the endpoint and issue the request
        """
        ...

    async def round_robin(self, request: JsonLike) -> AsyncIterator[JsonLike]:
        """
        Pick the next instance of the endpoint in a round-robin fashion
        """
        ...

    async def direct(self, request: JsonLike, instance: str) -> AsyncIterator[JsonLike]:
        """
        Pick a specific instance of the endpoint
        """
        ...

class KvRouter:
    """
    A router will determine which worker should handle a given request.
    """

    ...

    def __init__(self, drt: DistributedRuntime, component: Component) -> None:
        """
        Create a `KvRouter` object that is associated with the `component`
        """

    def schedule(self, token_ids: List[int], lora_id: int) -> int:
        """
        Return the worker id that should handle the given token ids,
        exception will be raised if there is no worker available.
        """
        ...

class KvMetricsPublisher:
    """
    A metrics publisher will provide KV metrics to the router.
    """

    ...

    def __init__(self) -> None:
        """
        Create a `KvMetricsPublisher` object
        """

    def create_service(self, component: Component) -> None:
        """
        Similar to Component.create_service, but only service created through
        this method will interact with KV router of the same component.
        """

    def publish(self, request_active_slots: int,
        request_total_slots: int,
        kv_active_blocks: int,
        kv_total_blocks: int) -> None:
        """
        Update the KV metrics being reported.
        """
        ...
