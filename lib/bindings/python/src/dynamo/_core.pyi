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

from typing import AsyncGenerator, AsyncIterator, Callable, Dict, List, Optional, Union

def log_message(level: str, message: str, module: str, file: str, line: int) -> None:
    """
    Log a message from Python with file and line info
    """
    ...

class JsonLike:
    """
    Any PyObject which can be serialized to JSON
    """

    ...

RequestHandler = Callable[[JsonLike], AsyncGenerator[JsonLike, None]]

class DistributedRuntime:
    """
    The runtime object for dynamo applications
    """

    ...

    def namespace(self, name: str, path: str) -> Namespace:
        """
        Create a `Namespace` object
        """
        ...

    def etcd_client(self) -> Optional[EtcdClient]:
        """
        Get the `EtcdClient` object. Not available for static workers.
        """
        ...

    def shutdown(self) -> None:
        """
        Shutdown the runtime by triggering the cancellation token
        """
        ...
class EtcdClient:
    """
    Etcd is used for discovery in the DistributedRuntime
    """

    def primary_lease_id(self) -> int:
        """
        return the primary lease id.
        """
        ...

    async def kv_create(
        self, key: str, value: bytes, lease_id: Optional[int] = None
    ) -> None:
        """
        Atomically create a key in etcd, fail if the key already exists.
        """
        ...

    async def kv_create_or_validate(
        self, key: str, value: bytes, lease_id: Optional[int] = None
    ) -> None:
        """
        Atomically create a key if it does not exist, or validate the values are identical if the key exists.
        """
        ...

    async def kv_put(
        self, key: str, value: bytes, lease_id: Optional[int] = None
    ) -> None:
        """
        Put a key-value pair into etcd
        """
        ...

    async def kv_get_prefix(self, prefix: str) -> List[Dict[str, JsonLike]]:
        """
        Get all keys with a given prefix
        """
        ...

    async def revoke_lease(self, lease_id: int) -> None:
        """
        Revoke a lease
        """
        ...

class EtcdKvCache:
    """
    A cache for key-value pairs stored in etcd.
    """

    @staticmethod
    async def new(
        etcd_client: EtcdClient,
        prefix: str,
        initial_values: Dict[str, Union[str, bytes]]
    ) -> "EtcdKvCache":
        """
        Create a new EtcdKvCache instance.

        Args:
            etcd_client: The etcd client to use for operations
            prefix: The prefix to use for all keys in this cache.
                EtcdKvCache will continuously watch the changes of the keys under this prefix.
            initial_values: Initial key-value pairs to populate the cache with
                NOTE: if the key already exists, it won't be updated

        Returns:
            A new EtcdKvCache instance
        """
        ...

    async def get(self, key: str) -> Optional[bytes]:
        """
        Get a value from the cache.

        Args:
            key: The key to retrieve

        Returns:
            The value as bytes if found, None otherwise

        NOTE: this get is cheap because internally there is a cache that holds the latest kv pairs.
        To prevent race condition, there is a lock when reading/writing the internal cache.
        """
        ...

    async def get_all(self) -> Dict[str, bytes]:
        """
        Get all key-value pairs from the cache.

        Returns:
            A dictionary of all key-value pairs, with keys stripped of the prefix
            (i.e., in the same format as in `initial_values`.keys())
        """
        ...

    async def put(
        self,
        key: str,
        value: bytes,
        lease_id: Optional[int] = None
    ) -> None:
        """
        Put a key-value pair into the cache and etcd.

        Args:
            key: The key to store
            value: The value to store
            lease_id: Optional lease ID to associate with this key-value pair
        """
        ...

    async def delete(self, key: str) -> None:
        """
        Delete a key-value pair from the cache and etcd.
        """
        ...

    async def clear_all(self) -> None:
        """
        Delete all key-value pairs from the cache and etcd.
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

class DisaggregatedRouter:
    """
    A router that determines whether to perform prefill locally or remotely based on
    sequence length thresholds.
    """

    def __init__(
        self,
        drt: DistributedRuntime,
        model_name: str,
        default_max_local_prefill_length: int,
    ) -> None:
        """
        Create a `DisaggregatedRouter` object.

        Args:
            drt: The distributed runtime instance
            model_name: Name of the model
            default_max_local_prefill_length: Default maximum sequence length that can be processed locally
        """
        ...

    def prefill_remote(self, prefill_length: int, prefix_hit_length: int) -> bool:
        """
        Determine if prefill should be performed remotely based on sequence lengths.

        Args:
            prefill_length: Total length of the sequence to prefill
            prefix_hit_length: Length of the prefix that was already processed

        Returns:
            True if prefill should be performed remotely, False otherwise
        """
        ...

    def update_value(self, max_local_prefill_length: int) -> None:
        """
        Update the maximum local prefill length threshold.

        Args:
            max_local_prefill_length: New maximum sequence length that can be processed locally
        """
        ...

    def get_model_name(self) -> str:
        """
        Get the name of the model associated with this router.

        Returns:
            The model name as a string
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

    def publish(
        self,
        request_active_slots: int,
        request_total_slots: int,
        kv_active_blocks: int,
        kv_total_blocks: int,
    ) -> None:
        """
        Update the KV metrics being reported.
        """
        ...

class ModelDeploymentCard:
    """
    A model deployment card is a collection of model information
    """

    ...

class OAIChatPreprocessor:
    """
    A preprocessor for OpenAI chat completions
    """

    ...

    async def start(self) -> None:
        """
        Start the preprocessor
        """
        ...

class Backend:
    """
    LLM Backend engine manages resources and concurrency for executing inference
    requests in LLM engines (trtllm, vllm, sglang etc)
    """

    ...

    async def start(self, handler: RequestHandler) -> None:
        """
        Start the backend engine and requests to the downstream LLM engine
        """
        ...

class OverlapScores:
    """
    A collection of prefix matching scores of workers for a given token ids.
    'scores' is a map of worker id to the score which is the number of matching blocks.
    """

    ...

class KvIndexer:
    """
    A KV Indexer that tracks KV Events emitted by workers. Events include add_block and remove_block.
    """

    ...

    def __init__(self, component: Component, block_size: int) -> None:
        """
        Create a `KvIndexer` object
        """

    def find_matches_for_request(
        self, token_ids: List[int], lora_id: int
    ) -> OverlapScores:
        """
        Return the overlapping scores of workers for the given token ids.
        """
        ...

    def block_size(self) -> int:
        """
        Return the block size of the KV Indexer.
        """
        ...

class KvRecorder:
    """
    A recorder for KV Router events.
    """

    ...

    def __init__(
        self,
        component: Component,
        output_path: Optional[str] = None,
        max_lines_per_file: Optional[int] = None,
        max_count: Optional[int] = None,
        max_time: Optional[float] = None,
    ) -> None:
        """
        Create a new KvRecorder instance.

        Args:
            component: The component to associate with this recorder
            output_path: Path to the JSONL file to write events to
            max_lines_per_file: Maximum number of lines per file before rotating to a new file
            max_count: Maximum number of events to record before shutting down
            max_time: Maximum duration in seconds to record before shutting down
        """
        ...

    def event_count(self) -> int:
        """
        Get the count of recorded events.

        Returns:
            The number of events recorded
        """
        ...

    def elapsed_time(self) -> float:
        """
        Get the elapsed time since the recorder was started.

        Returns:
            The elapsed time in seconds as a float
        """
        ...

    def replay_events(
        self,
        indexer: KvIndexer,
        timed: bool = False,
        max_count: Optional[int] = None,
        max_time: Optional[float] = None,
    ) -> int:
        """
        Populate an indexer with the recorded events.

        Args:
            indexer: The KvIndexer to populate with events
            timed: If true, events will be sent according to their recorded timestamps.
                If false, events will be sent without any delay in between.
            max_count: Maximum number of events to send before stopping
            max_time: Maximum duration in seconds to send events before stopping

        Returns:
            The number of events sent to the indexer
        """
        ...

    def shutdown(self) -> None:
        """
        Shutdown the recorder.
        """
        ...

class AggregatedMetrics:
    """
    A collection of metrics of the endpoints
    """

    ...

class KvMetricsAggregator:
    """
    A metrics aggregator will collect KV metrics of the endpoints.
    """

    ...

    def __init__(self, component: Component) -> None:
        """
        Create a `KvMetricsAggregator` object
        """

    def get_metrics(self) -> AggregatedMetrics:
        """
        Return the aggregated metrics of the endpoints.
        """
        ...

class KvEventPublisher:
    """
    A KV event publisher will publish KV events corresponding to the component.
    """

    ...

    def __init__(
        self, component: Component, worker_id: int, kv_block_size: int
    ) -> None:
        """
        Create a `KvEventPublisher` object
        """

    def publish_stored(
        self,
        event_id,
        int,
        token_ids: List[int],
        num_block_tokens: List[int],
        block_hashes: List[int],
        lora_id: int,
        parent_hash: Optional[int] = None,
    ) -> None:
        """
        Publish a KV stored event.
        """
        ...

    def publish_removed(self, event_id, int, block_hashes: List[int]) -> None:
        """
        Publish a KV removed event.
        """
        ...

class HttpService:
    """
    A HTTP service for dynamo applications.
    It is a OpenAI compatible http ingress into the Dynamo Distributed Runtime.
    """

    ...

class HttpError:
    """
    An error that occurred in the HTTP service
    """

    ...

class HttpAsyncEngine:
    """
    An async engine for a distributed Dynamo http service. This is an extension of the
    python based AsyncEngine that handles HttpError exceptions from Python and
    converts them to the Rust version of HttpError
    """

    ...

class ModelType:
    """What type of request this model needs: Chat, Component or Backend (pre-processed)"""
    ...

async def register_llm(model_type: ModelType, endpoint: Endpoint, model_path: str, model_name: Optional[str]) -> None:
    """Attach the model at path to the given endpoint, and advertise it as model_type"""
    ...

class NatsQueue:
    """
    A queue implementation using NATS JetStream for task distribution
    """

    def __init__(self, stream_name: str, nats_server: str, dequeue_timeout: float) -> None:
        """
        Create a new NatsQueue instance.

        Args:
            stream_name: Name of the NATS JetStream stream
            nats_server: URL of the NATS server
            dequeue_timeout: Default timeout in seconds for dequeue operations
        """
        ...

    async def connect(self) -> None:
        """
        Connect to the NATS server
        """
        ...

    async def ensure_connection(self) -> None:
        """
        Ensure connection to the NATS server, connecting if not already connected
        """
        ...

    async def close(self) -> None:
        """
        Close the connection to the NATS server
        """
        ...

    async def enqueue_task(self, task_data: bytes) -> None:
        """
        Enqueue a task to the NATS JetStream

        Args:
            task_data: The task data as bytes
        """
        ...

    async def dequeue_task(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        Dequeue a task from the NATS JetStream

        Args:
            timeout: Optional timeout in seconds for this specific dequeue operation.
                    If None, uses the default timeout specified during initialization.

        Returns:
            The task data as bytes if available, None if no task is available
        """
        ...

    async def get_queue_size(self) -> int:
        """
        Get the current size of the queue

        Returns:
            The number of messages in the queue
        """
        ...

