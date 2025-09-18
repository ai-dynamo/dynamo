# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

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

    def namespace(self, name: str) -> Namespace:
        """
        Create a `Namespace` object
        """
        ...

    def do_not_use_etcd_client(self) -> Optional[EtcdClient]:
        """
        Get the `EtcdClient` object. Not available for static workers.
        This will be removed soon, do not use it.
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

    async def serve_endpoint(self, handler: RequestHandler, graceful_shutdown: bool = True, metrics_labels: Optional[List[Tuple[str, str]]] = None, health_check_payload: Optional[Dict[str, Any]] = None) -> None:
        """
        Serve an endpoint discoverable by all connected clients at
        `{{ namespace }}/components/{{ component_name }}/endpoints/{{ endpoint_name }}`

        Args:
            handler: The request handler function
            graceful_shutdown: Whether to wait for inflight requests to complete during shutdown (default: True)
            metrics_labels: Optional list of metrics labels to add to the metrics
            health_check_payload: Optional dict containing the health check request payload
                                  that will be used to verify endpoint health
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

    def instance_ids(self) -> List[int]:
        """
        Get list of current instance IDs.

        Returns:
            A list of currently available instance IDs
        """
        ...

    async def wait_for_instances(self) -> List[int]:
        """
        Wait for instances to be available for work and return their IDs.

        Returns:
            A list of instance IDs that are available for work
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

def compute_block_hash_for_seq_py(tokens: List[int], kv_block_size: int) -> List[int]:
    """
    Compute block hashes for a sequence of tokens

    Args:
        tokens: List of token IDs
        kv_block_size: Size of each KV cache block

    Returns:
        List of block hashes as integers
    """

    ...

class WorkerStats:
    """
    Worker stats.
    """

    ...

    def __init__(
        self,
        request_active_slots: int,
        request_total_slots: int,
        num_requests_waiting: int,
        data_parallel_rank: Optional[int] = None,
    ) -> None:
        """
        Create a `WorkerStats` object.
        """
        ...

class KvStats:
    """
    KV stats.
    """

    ...

    def __init__(
        self,
        kv_active_blocks: int,
        kv_total_blocks: int,
        gpu_cache_usage_perc: float,
        gpu_prefix_cache_hit_rate: float,
    ) -> None:
        """
        Create a `KvStats` object.
        """
        ...

class SpecDecodeStats:
    """
    Speculative decoding stats.
    """

    ...

    def __init__(
        self,
        num_spec_tokens: int,
        num_drafts: int,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_accepted_tokens_per_pos: List[int],
    ) -> None:
        """
        Create a `SpecDecodeStats` object when running with speculative decoding.
        """
        ...

class ForwardPassMetrics:
    """
    A collection of metrics for a forward pass.
    Includes worker stats, KV stats, and speculative decoding stats.
    """

    ...

    def __init__(
        self,
        worker_stats: WorkerStats,
        kv_stats: KvStats,
        spec_decode_stats: Optional[SpecDecodeStats] = None,
    ) -> None:
        """
        Create a `ForwardPassMetrics` object
        """
        ...

class WorkerMetricsPublisher:
    """
    A metrics publisher will provide metrics to the router.
    """

    ...

    def __init__(self) -> None:
        """
        Create a `WorkerMetricsPublisher` object
        """

    def create_endpoint(self, component: Component, metrics_labels: Optional[List[Tuple[str, str]]] = None) -> None:
        """
        Similar to Component.create_service, but only service created through
        this method will interact with KV router of the same component.
        """

    def publish(
        self,
        metrics: ForwardPassMetrics
    ) -> None:
        """
        Update the metrics being reported.
        """
        ...

class ModelDeploymentCard:
    """
    A model deployment card is a collection of model information
    """

    ...

class ModelRuntimeConfig:
    """
    A model runtime configuration is a collection of runtime information
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

    @property
    def scores(self) -> Dict[int, int]:
        """
        Map of worker_id to the score which is the number of matching blocks.

        Returns:
            Dictionary mapping worker IDs to their overlap scores
        """
        ...

    @property
    def frequencies(self) -> List[int]:
        """
        List of frequencies that the blocks have been accessed.
        Entries with value 0 are omitted.

        Returns:
            List of access frequencies for each block
        """
        ...

class RadixTree:
    """
    A RadixTree that tracks KV cache blocks and can find prefix matches for sequences.

    NOTE: This class is not thread-safe and should only be used from a single thread in Python.
    """

    def __init__(self, expiration_duration_secs: Optional[float] = None) -> None:
        """
        Create a new RadixTree instance.

        Args:
            expiration_duration_secs: Optional expiration duration in seconds for cached blocks.
                                    If None, blocks never expire.
        """
        ...

    def find_matches(
        self, sequence: List[int], early_exit: bool = False
    ) -> OverlapScores:
        """
        Find prefix matches for the given sequence of block hashes.

        Args:
            sequence: List of block hashes to find matches for
            early_exit: If True, stop searching after finding the first match

        Returns:
            OverlapScores containing worker matching scores and frequencies
        """
        ...

    def apply_event(self, worker_id: int, kv_cache_event_bytes: bytes) -> None:
        """
        Apply a KV cache event to update the RadixTree state.

        Args:
            worker_id: ID of the worker that generated the event
            kv_cache_event_bytes: Serialized KV cache event as bytes

        Raises:
            ValueError: If the event bytes cannot be deserialized
        """
        ...

    def remove_worker(self, worker_id: int) -> None:
        """
        Remove all blocks associated with a specific worker.

        Args:
            worker_id: ID of the worker to remove
        """
        ...

    def clear_all_blocks(self, worker_id: int) -> None:
        """
        Clear all blocks for a specific worker.

        Args:
            worker_id: ID of the worker whose blocks should be cleared
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

    def find_matches(self, sequence: List[int]) -> OverlapScores:
        """
        Find prefix matches for the given sequence of block hashes.

        Args:
            sequence: List of block hashes to find matches for

        Returns:
            OverlapScores containing worker matching scores and frequencies
        """
        ...

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

class ApproxKvIndexer:
    """
    A KV Indexer that doesn't use KV cache events. It instead relies solely on the input tokens.
    """

    def __init__(self, component: Component, kv_block_size: int, ttl_secs: float) -> None:
        """
        Create a `ApproxKvIndexer` object
        """
        ...

    def find_matches_for_request(self, token_ids: List[int], lora_id: int) -> OverlapScores:
        """
        Return the overlapping scores of workers for the given token ids.
        """
        ...

    def block_size(self) -> int:
        """
        Return the block size of the ApproxKvIndexer.
        """
        ...

    def process_routing_decision_for_request(self, tokens: List[int], lora_id: int, worker_id: int) -> None:
        """
        Notify the indexer that a token sequence has been sent to a specific worker.
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

class ZmqKvEventPublisherConfig:
    def __init__(
        self,
        worker_id: int,
        kv_block_size: int,
        zmq_endpoint: str = "tcp://127.0.0.1:5557",
        zmq_topic: str = ""
    ) -> None:
        """
        Configuration for the ZmqKvEventPublisher.

        :param worker_id: The worker ID.
        :param kv_block_size: The block size for the key-value store.
        :param zmq_endpoint: The ZeroMQ endpoint. Defaults to "tcp://127.0.0.1:5557".
        :param zmq_topic: The ZeroMQ topic to subscribe to. Defaults to an empty string.
        """
        ...

class ZmqKvEventPublisher:
    def __init__(self, component: Component, config: ZmqKvEventPublisherConfig) -> None:
        """
        Initializes a new ZmqKvEventPublisher instance.

        :param component: The component to be used.
        :param config: Configuration for the event publisher.
        """
        ...

    def shutdown(self) -> None:
        """
        Shuts down the event publisher, stopping any background tasks.
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

class ModelInput:
    """What type of request this model needs: Text or Tokens"""
    ...

class ModelType:
    """What type of request this model needs: Chat, Completions or Embedding"""
    ...

class RouterMode:
    """Router mode for load balancing requests across workers"""
    ...

class RouterConfig:
    """How to route the request"""
    ...

class KvRouterConfig:
    """Values for KV router"""
    ...

async def register_llm(
    model_input: ModelInput,
    model_type: ModelType,
    endpoint: Endpoint,
    model_path: str,
    model_name: Optional[str] = None,
    context_length: Optional[int] = None,
    kv_cache_block_size: Optional[int] = None,
    migration_limit: int = 0,
    router_mode: Optional[RouterMode] = None,
    user_data: Optional[Dict[str, Any]] = None,
    custom_template_path: Optional[str] = None,
) -> None:
    """Attach the model at path to the given endpoint, and advertise it as model_type"""
    ...

class EngineConfig:
    """Holds internal configuration for a Dynamo engine."""
    ...

async def make_engine(args: EntrypointArgs) -> EngineConfig:
    """Make an engine matching the args"""
    ...

async def run_input(runtime: DistributedRuntime, input: str, engine_config: EngineConfig) -> None:
    """Start an engine, connect it to an input, and run until stopped."""
    ...

class Layer:
    """
    A KV cache block layer
    """

    ...

    def __dlpack__(self, stream: Optional[Any] = None, max_version: Optional[Any] = None, dl_device: Optional[Any] = None, copy: Optional[bool] = None) -> Any:
        """
        Get a dlpack capsule of the layer
        """
        ...

    def __dlpack_device__(self) -> Any:
        """
        Get the dlpack device of the layer
        """
        ...

class Block:
    """
    A KV cache block
    """

    ...

    def __len__(self) -> int:
        """
        Get the number of layers in the list
        """
        ...

    def __getitem__(self, index: int) -> Layer:
        """
        Get a layer by index
        """
        ...

    def __iter__(self) -> 'Block':
        """
        Get an iterator over the layers
        """
        ...

    def __next__(self) -> Block:
        """
        Get the next layer in the iterator
        """
        ...

    def to_list(self) -> List[Layer]:
        """
        Get a list of layers
        """
        ...

    def __dlpack__(self, stream: Optional[Any] = None, max_version: Optional[Any] = None, dl_device: Optional[Any] = None, copy: Optional[bool] = None) -> Any:
        """
        Get a dlpack capsule of the block
        Exception raised if the block is not contiguous
        """
        ...

    def __dlpack_device__(self) -> Any:
        """
        Get the dlpack device of the block
        """
        ...

class BlockList:
    """
    A list of KV cache blocks
    """

    ...

    def __len__(self) -> int:
        """
        Get the number of blocks in the list
        """
        ...

    def __getitem__(self, index: int) -> Block:
        """
        Get a block by index
        """
        ...

    def __iter__(self) -> 'BlockList':
        """
        Get an iterator over the blocks
        """
        ...

    def __next__(self) -> Block:
        """
        Get the next block in the iterator
        """
        ...

    def to_list(self) -> List[Block]:
        """
        Get a list of blocks
        """
        ...

class BlockManager:
    """
    A KV cache block manager
    """

    def __init__(
        self,
        worker_id: int,
        num_layer: int,
        page_size: int,
        inner_dim: int,
        dtype: Optional[str] = None,
        host_num_blocks: Optional[int] = None,
        device_num_blocks: Optional[int] = None,
        device_id: int = 0
    ) -> None:
        """
        Create a `BlockManager` object

        Parameters:
        -----------
        worker_id: int
            The worker ID for this block manager
        num_layer: int
            Number of layers in the model
        page_size: int
            Page size for blocks
        inner_dim: int
            Inner dimension size
        dtype: Optional[str]
            Data type (e.g., 'fp16', 'bf16', 'fp32'), defaults to 'fp16' if None
        host_num_blocks: Optional[int]
            Number of host blocks to allocate, None means no host blocks
        device_num_blocks: Optional[int]
            Number of device blocks to allocate, None means no device blocks
        device_id: int
            CUDA device ID, defaults to 0
        """
        ...

    def allocate_host_blocks_blocking(self, count: int) -> BlockList:
        """
        Allocate a list of host blocks (blocking call)

        Parameters:
        -----------
        count: int
            Number of blocks to allocate

        Returns:
        --------
        BlockList
            List of allocated blocks
        """
        ...

    async def allocate_host_blocks(self, count: int) -> BlockList:
        """
        Allocate a list of host blocks

        Parameters:
        -----------
        count: int
            Number of blocks to allocate

        Returns:
        --------
        BlockList
            List of allocated blocks
        """
        ...

    def allocate_device_blocks_blocking(self, count: int) -> BlockList:
        """
        Allocate a list of device blocks (blocking call)

        Parameters:
        -----------
        count: int
            Number of blocks to allocate

        Returns:
        --------
        BlockList
            List of allocated blocks
        """
        ...

    async def allocate_device_blocks(self, count: int) -> BlockList:
        """
        Allocate a list of device blocks

        Parameters:
        -----------
        count: int
            Number of blocks to allocate

        Returns:
        --------
        BlockList
            List of allocated blocks
        """
        ...

class KvbmCacheManager:
    """
    A KV cache manager for VLLM
    """

    def __init__(self, block_manager: BlockManager) -> None:
        ...


class KvbmRequest:
    """
    A request for KV cache
    """

    def __init__(self, request_id: int, tokens: List[int], block_size: int) -> None:
        ...

class ZmqKvEventListener:
    """
    A ZMQ-based key-value cache event listener that operates independently
    of the dynamo runtime or event plane infrastructure.
    """

    def __init__(
        self, zmq_endpoint: str, zmq_topic: str, kv_block_size: int
    ) -> None:
        """
        Create a new ZmqKvEventListener instance.

        Args:
            zmq_endpoint: ZeroMQ endpoint to connect to (e.g., "tcp://127.0.0.1:5557")
            zmq_topic: ZeroMQ topic to subscribe to
            kv_block_size: Size of KV cache blocks
        """
        ...

    async def get_events(self) -> List[str]:
        """
        Get all available KV cache events from the ZMQ listener.

        Returns:
            List of JSON-serialized KV cache events as strings

        Raises:
            ValueError: If events cannot be serialized to JSON
        """
        ...

class KvPushRouter:
    """
    A KV-aware push router that performs intelligent routing based on KV cache overlap.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        block_size: int,
        kv_router_config: KvRouterConfig,
    ) -> None:
        """
        Create a new KvPushRouter instance.

        Args:
            endpoint: The endpoint to connect to for routing requests
            block_size: The KV cache block size
            kv_router_config: Configuration for the KV router
        """
        ...

    async def generate(
        self,
        token_ids: List[int],
        model: str,
        stop_conditions: Optional[JsonLike] = None,
        sampling_options: Optional[JsonLike] = None,
        output_options: Optional[JsonLike] = None,
        router_config_override: Optional[JsonLike] = None,
        worker_id: Optional[int] = None,
    ) -> AsyncIterator[JsonLike]:
        """
        Generate text using the KV-aware router.

        Args:
            token_ids: Input token IDs
            model: Model name to use for generation
            stop_conditions: Optional stop conditions for generation
            sampling_options: Optional sampling configuration
            output_options: Optional output configuration
            router_config_override: Optional router configuration override
            worker_id: Optional worker ID to route to directly. If set, the request
                      will be sent to this specific worker and router states will be
                      updated accordingly.

        Returns:
            An async iterator yielding generation responses

        Note:
            - If worker_id is set, the request bypasses KV matching and routes directly
              to the specified worker while still updating router states.
            - This is different from query_instance_id which doesn't route the request.
        """
        ...

    async def best_worker_id(
        self,
        context_id: str,
        token_ids: List[int],
        router_config_override: Optional[JsonLike] = None,
    ) -> Tuple[int, int]:
        """
        Find the best matching worker for the given tokens without updating states.

        Args:
            context_id: String identifier for the request
            token_ids: List of token IDs to find matches for
            router_config_override: Optional router configuration override

        Returns:
            A tuple of (worker_id, overlap_blocks) where:
                - worker_id: The ID of the best matching worker
                - overlap_blocks: The number of overlapping blocks found
        """
        ...

    async def get_potential_loads(
        self,
        token_ids: List[int],
    ) -> List[Dict[str, int]]:
        """
        Get potential prefill and decode loads for all workers.

        Args:
            token_ids: List of token IDs to evaluate

        Returns:
            A list of dictionaries, each containing:
                - worker_id: The worker ID
                - potential_prefill_tokens: Number of tokens that would need prefill
                - potential_decode_blocks: Number of blocks currently in decode phase
        """
        ...

    async def dump_events(self) -> str:
        """
        Dump all events from the KV router's indexer.

        Returns:
            A JSON string containing all indexer events
        """
        ...

class EntrypointArgs:
    """
    Settings to connect an input to a worker and run them.
    Use by `dynamo run`.
    """

    ...
