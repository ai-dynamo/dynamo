# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from ._prometheus_names import prometheus_names

# Import from specialized modules
from .prometheus_metrics import RuntimeMetrics as PyRuntimeMetrics

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

    def __new__(
        cls,
        event_loop: Any,
        store_kv: str,
        request_plane: str,
        enable_nats: Optional[bool] = None,
    ) -> "DistributedRuntime":
        """
        Create a new DistributedRuntime.

        Args:
            event_loop: The asyncio event loop
            store_kv: Key-value store backend ("etcd", "file", or "mem")
            request_plane: Request plane transport ("tcp", "http", or "nats")
            enable_nats: Whether to enable NATS for KV events. Defaults to True.
                        If request_plane is "nats", NATS is always enabled.
                        Pass False to disable NATS initialization (e.g., for approximate routing).
        """
        ...

    def namespace(self, name: str) -> Namespace:
        """
        Create a `Namespace` object
        """
        ...

    def shutdown(self) -> None:
        """
        Shutdown the runtime by triggering the cancellation token
        """
        ...

    def child_token(self) -> CancellationToken:
        """
        Get a child cancellation token that can be passed to async tasks
        """
        ...

    def register_engine_route(
        self,
        route_name: str,
        callback: Callable[[dict], Awaitable[dict]],
    ) -> None:
        """
        Register an async callback for /engine/{route_name} on the system status server.

        Args:
            route_name: The route path (e.g., "start_profile" creates /engine/start_profile)
            callback: Async function with signature: async def(body: dict) -> dict

        Example:
            async def start_profile(body: dict) -> dict:
                await engine.start_profile(**body)
                return {"status": "ok", "message": "Profiling started"}

            runtime.register_engine_route("start_profile", start_profile)

        The callback receives the JSON request body as a dict and should return
        a dict that will be serialized as the JSON response.

        For GET requests or empty bodies, an empty dict {} is passed.
        """
        ...

class CancellationToken:
    def cancel(self) -> None:
        """
        Cancel the token and all its children
        """
        ...

    async def cancelled(self) -> None:
        """
        Await until the token is cancelled
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

    def connection_id(self) -> int:
        """
        Opaque unique ID for this worker. May change over worker lifetime.
        """
        ...

    @property
    def metrics(self) -> PyRuntimeMetrics:
        """
        Get a PyRuntimeMetrics helper for registering Prometheus metrics callbacks.

        Returns:
            A PyRuntimeMetrics object for callback registration
        """
        ...

    async def unregister_endpoint_instance(self) -> None:
        """
        Unregister this endpoint instance from discovery.

        This removes the endpoint from the instances bucket, preventing the router
        from sending requests to this worker. Use this when a worker is sleeping
        and should not receive any requests.
        """
        ...

    async def register_endpoint_instance(self) -> None:
        """
        Re-register this endpoint instance to discovery.

        This adds the endpoint back to the instances bucket, allowing the router
        to send requests to this worker again. Use this when a worker wakes up
        and should start receiving requests.
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


def compute_block_hash_for_seq_py(
    tokens: List[int],
    kv_block_size: int,
    block_mm_infos: Optional[List[Optional[Dict[str, Any]]]] = None
) -> List[int]:
    """
    Compute block hashes for a sequence of tokens, optionally including multimodal metadata.

    When block_mm_infos is provided, the mm_hashes are included in the hash computation
    to ensure that blocks with identical tokens but different multimodal objects produce
    different hashes.

    Args:
        tokens: List of token IDs
        kv_block_size: Size of each block in tokens
        block_mm_infos: Optional per-block multimodal metadata. Each element corresponds to a block
                       and should be None or a dict with structure:
                       {
                           "mm_objects": [
                               {
                                   "mm_hash": int,  # Hash of the MM object
                               }
                           ]
                       }

    Returns:
        List of block hashes (one per block)

    Example:
        >>> tokens = [1, 2, 3, 4] * 8  # 32 tokens = 1 block
        >>> mm_info = {
        ...     "mm_objects": [{
        ...         "mm_hash": 0xDEADBEEF,
        ...     }]
        ... }
        >>> hashes = compute_block_hash_for_seq_py(tokens, 32, [mm_info])
    """

    ...

class Context:
    """
    Context wrapper around AsyncEngineContext for Python bindings.
    Provides tracing and cancellation capabilities for request handling.
    """

    def __init__(self, id: Optional[str] = None) -> None:
        """
        Create a new Context instance.

        Args:
            id: Optional request ID. If None, a default ID will be generated.
        """
        ...

    def is_stopped(self) -> bool:
        """
        Check if the context has been stopped (synchronous).

        Returns:
            True if the context is stopped, False otherwise.
        """
        ...

    def is_killed(self) -> bool:
        """
        Check if the context has been killed (synchronous).

        Returns:
            True if the context is killed, False otherwise.
        """
        ...

    def stop_generating(self) -> None:
        """
        Issue a stop generating signal to the context.
        """
        ...

    def id(self) -> Optional[str]:
        """
        Get the context ID.

        Returns:
            The context identifier string, or None if not set.
        """
        ...

    async def async_killed_or_stopped(self) -> bool:
        """
        Asynchronously wait until the context is killed or stopped.

        Returns:
            True when the context is killed or stopped.
        """
        ...

    @property
    def trace_id(self) -> Optional[str]:
        """
        Get the distributed trace ID if available.

        Returns:
            The trace ID string, or None if no trace context.
        """
        ...

    @property
    def span_id(self) -> Optional[str]:
        """
        Get the distributed span ID if available.

        Returns:
            The span ID string, or None if no trace context.
        """
        ...

    @property
    def parent_span_id(self) -> Optional[str]:
        """
        Get the parent span ID if available.

        Returns:
            The parent span ID string, or None if no trace context.
        """
        ...

class WorkerMetricsPublisher:
    """
    A metrics publisher will provide metrics to the router for load monitoring.
    """

    ...

    def __init__(self) -> None:
        """
        Create a `WorkerMetricsPublisher` object
        """

    async def create_endpoint(self, component: Component) -> None:
        """
        Create the NATS endpoint for metrics publishing. Must be awaited.

        Only service created through this method will interact with KV router of the same component.

        Args:
            component: The component to create the endpoint for
        """

    def publish(
        self,
        dp_rank: Optional[int],
        active_decode_blocks: int,
    ) -> None:
        """
        Publish worker metrics for load monitoring.

        Args:
            dp_rank: Data parallel rank of the worker (None defaults to 0)
            active_decode_blocks: Number of active KV cache blocks
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

    total_kv_blocks: int | None
    max_num_seqs: int | None
    max_num_batched_tokens: int | None
    tool_call_parser: str | None
    reasoning_parser: str | None
    enable_local_indexer: bool
    runtime_data: dict[str, Any]
    tensor_model_config: Any | None

    def __init__(self) -> None: ...

    def set_engine_specific(self, key: str, value: Any) -> None:
        """Set an engine-specific runtime configuration value"""
        ...

    def get_engine_specific(self, key: str) -> Any | None:
        """Get an engine-specific runtime configuration value"""
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

    Thread-safe: operations route to a dedicated background thread and long calls
    release the Python GIL.
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

    def dump_tree_as_events(self) -> List[str]:
        """
        Dump the current RadixTree state as a list of JSON-serialized KV cache events.

        Returns:
            List of JSON-serialized KV cache events as strings
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
    An approximate KV Indexer that doesn't receive KV cache events from workers.
    Instead, it relies on routing decisions with TTL-based expiration and pruning
    to estimate which blocks are cached on which workers.

    This is useful when:
    - Backend engines don't emit KV events
    - You want to reduce event processing overhead
    - Lower routing accuracy is acceptable
    """

    ...

    def __init__(
        self,
        component: Component,
        kv_block_size: int,
        router_ttl_secs: float = 120.0,
        router_max_tree_size: int = 1048576,
        router_prune_target_ratio: float = 0.8,
    ) -> None:
        """
        Create an `ApproxKvIndexer` object

        Args:
            component: The component to associate with this indexer
            kv_block_size: The KV cache block size
            router_ttl_secs: TTL for blocks in seconds (default: 120.0)
            router_max_tree_size: Maximum tree size before pruning (default: 1048576, which is 2^20)
            router_prune_target_ratio: Target size ratio after pruning (default: 0.8)
        """
        ...

    def find_matches_for_request(
        self, token_ids: List[int]
    ) -> OverlapScores:
        """
        Return the overlapping scores of workers for the given token ids.

        Args:
            token_ids: List of token IDs to find matches for

        Returns:
            OverlapScores containing worker matching scores and frequencies
        """
        ...

    def block_size(self) -> int:
        """
        Return the block size of the ApproxKvIndexer.

        Returns:
            The KV cache block size
        """
        ...

    async def process_routing_decision_for_request(
        self, tokens: List[int], worker_id: int, dp_rank: int = 0
    ) -> None:
        """
        Notify the indexer that a token sequence has been routed to a specific worker.

        This updates the indexer's internal state to track which blocks are likely
        cached on which workers based on routing decisions.

        Args:
            tokens: List of token IDs that were routed
            worker_id: The worker ID the request was routed to
            dp_rank: The data parallel rank (default: 0)
        """
        ...


class KvEventPublisher:
    """
    A KV event publisher will publish KV events corresponding to the component.
    """

    ...

    def __init__(
        self,
        component: Component,
        worker_id: int = 0,
        kv_block_size: int = 0,
        dp_rank: int = 0,
        enable_local_indexer: bool = False,
        zmq_config: Optional[ZmqKvEventPublisherConfig] = None,
    ) -> None:
        """
        Create a `KvEventPublisher` object.

        When zmq_config is provided, the publisher subscribes to a ZMQ socket for
        incoming engine events (e.g. from SGLang/vLLM) and relays them to NATS.
        The zmq_config fields override kv_block_size, dp_rank, and enable_local_indexer.

        When zmq_config is None, events are pushed manually via publish_stored/publish_removed.

        Args:
            component: The component to publish events for
            worker_id: The worker ID (unused, inferred from component)
            kv_block_size: The KV block size (must be > 0; ignored if zmq_config is set)
            dp_rank: The data parallel rank (defaults to 0; ignored if zmq_config is set)
            enable_local_indexer: Enable worker-local KV indexer (ignored if zmq_config is set)
            zmq_config: Optional ZMQ configuration for relay mode
        """

    def publish_stored(
        self,
        token_ids: List[int],
        num_block_tokens: List[int],
        block_hashes: List[int],
        lora_id: int,
        parent_hash: Optional[int] = None,
    ) -> None:
        """
        Publish a KV stored event.

        Event IDs are managed internally by the publisher using a monotonic counter.

        Args:
            token_ids: List of token IDs
            num_block_tokens: Number of tokens per block
            block_hashes: List of block hashes (signed 64-bit integers)
            lora_id: The LoRA ID
            parent_hash: Optional parent hash (signed 64-bit integer)
        """
        ...

    def publish_removed(self, block_hashes: List[int]) -> None:
        """
        Publish a KV removed event.

        Event IDs are managed internally by the publisher using a monotonic counter.

        Args:
            block_hashes: List of block hashes to remove (signed 64-bit integers)
        """
        ...

    def shutdown(self) -> None:
        """
        Shuts down the event publisher, stopping any background tasks.
        """
        ...

class ZmqKvEventPublisherConfig:
    def __init__(
        self,
        worker_id: int,
        kv_block_size: int,
        zmq_endpoint: str = "tcp://127.0.0.1:5557",
        zmq_topic: str = "",
        enable_local_indexer: bool = False,
        dp_rank: int = 0
    ) -> None:
        """
        ZMQ configuration for KvEventPublisher relay mode.

        :param worker_id: The worker ID.
        :param kv_block_size: The block size for the key-value store.
        :param zmq_endpoint: The ZeroMQ endpoint. Defaults to "tcp://127.0.0.1:5557".
        :param zmq_topic: The ZeroMQ topic to subscribe to. Defaults to an empty string.
        :param enable_local_indexer: Whether to enable the worker-local KV indexer. Defaults to False.
        :param dp_rank: The data parallel rank for this publisher. Defaults to 0.
        """
        ...

class HttpService:
    """
    A HTTP service for dynamo applications.
    It is a OpenAI compatible http ingress into the Dynamo Distributed Runtime.
    """

    ...

class PythonAsyncEngine:
    """
    Bridge a Python async generator onto Dynamo's AsyncEngine interface.
    """

    def __init__(self, generator: Any, event_loop: Any) -> None:
        """Wrap a Python generator and event loop for use with Dynamo services."""
        ...



class HttpAsyncEngine:
    """
    An async engine for a distributed Dynamo http service. This is an extension of the
    python based AsyncEngine that handles HttpError exceptions from Python and
    converts them to the Rust version of HttpError
    """

    ...

class KserveGrpcService:
    """
    A gRPC service implementing the KServe protocol for dynamo applications.
    Provides model management for completions, chat completions, and tensor-based models.
    """

    def __init__(self, port: Optional[int] = None, host: Optional[str] = None) -> None:
        """
        Create a new KServe gRPC service.

        Args:
            port: Optional port number to bind the service to
            host: Optional host address to bind the service to
        """
        ...

    def add_completions_model(
        self,
        model: str,
        checksum: str,
        engine: PythonAsyncEngine,
    ) -> None:
        """
        Register a completions model with the service.

        Args:
            model: The model name
            checksum: The model checksum
            engine: The async engine to handle requests
        """
        ...

    def add_chat_completions_model(
        self,
        model: str,
        checksum: str,
        engine: PythonAsyncEngine,
    ) -> None:
        """
        Register a chat completions model with the service.

        Args:
            model: The model name
            checksum: The model checksum
            engine: The async engine to handle requests
        """
        ...

    def add_tensor_model(
        self,
        model: str,
        checksum: str,
        engine: PythonAsyncEngine,
        runtime_config: Optional[ModelRuntimeConfig],
    ) -> None:
        """
        Register a tensor-based model with the service.

        Args:
            model: The model name
            checksum: The model checksum
            engine: The async engine to handle requests
        """
        ...

    def remove_completions_model(self, model: str) -> None:
        """
        Remove a completions model from the service.

        Args:
            model: The model name to remove
        """
        ...

    def remove_chat_completions_model(self, model: str) -> None:
        """
        Remove a chat completions model from the service.

        Args:
            model: The model name to remove
        """
        ...

    def remove_tensor_model(self, model: str) -> None:
        """
        Remove a tensor model from the service.

        Args:
            model: The model name to remove
        """
        ...

    def list_chat_completions_models(self) -> List[str]:
        """
        List all registered chat completions models.

        Returns:
            List of model names
        """
        ...

    def list_completions_models(self) -> List[str]:
        """
        List all registered completions models.

        Returns:
            List of model names
        """
        ...

    def list_tensor_models(self) -> List[str]:
        """
        List all registered tensor models.

        Returns:
            List of model names
        """
        ...

    async def run(self, token: CancellationToken) -> None:
        """
        Run the KServe gRPC service.

        Args:
            token: Cancellation token to stop the service
        """
        ...

class ModelInput:
    """What type of request this model needs: Text, Tokens or Tensor"""
    ...

class ModelType:
    """What type of request this model needs: Chat, Completions, Embedding, Tensor, Images or Prefill"""
    Chat: ModelType
    Completions: ModelType
    Embedding: ModelType
    TensorBased: ModelType
    Prefill: ModelType
    Images: ModelType
    ...

class RouterMode:
    """Router mode for load balancing requests across workers"""
    ...

class RouterConfig:
    """How to route the request"""
    ...

class KvRouterConfig:
    """Values for KV router"""

    def __init__(
        self,
        overlap_score_weight: float = 1.0,
        router_temperature: float = 0.0,
        use_kv_events: bool = True,
        router_replica_sync: bool = False,
        router_track_active_blocks: bool = True,
        router_track_output_blocks: bool = False,
        router_assume_kv_reuse: bool = True,
        router_snapshot_threshold: Optional[int] = 1000000,
        router_reset_states: bool = False,
        router_ttl_secs: float = 120.0,
        router_max_tree_size: int = 1048576,
        router_prune_target_ratio: float = 0.8,
    ) -> None:
        """
        Create a KV router configuration.

        Args:
            overlap_score_weight: Weight for overlap score in worker selection (default: 1.0)
            router_temperature: Temperature for worker sampling via softmax (default: 0.0)
            use_kv_events: Whether to use KV events from workers (default: True)
            router_replica_sync: Enable replica synchronization (default: False)
            router_track_active_blocks: Track active blocks for load balancing (default: True)
            router_track_output_blocks: Track output blocks during generation (default: False).
                When enabled, the router adds placeholder blocks as tokens are generated
                and applies fractional decay based on progress toward expected_output_tokens.
            router_assume_kv_reuse: Assume KV cache reuse when tracking active blocks (default: True).
                When True, computes actual block hashes. When False, generates random hashes.
            router_snapshot_threshold: Number of messages before snapshot (default: 1000000)
            router_reset_states: Reset router state on startup (default: False)
            router_ttl_secs: TTL for blocks in seconds when not using KV events (default: 120.0)
            router_max_tree_size: Maximum tree size before pruning (default: 1048576, which is 2^20)
            router_prune_target_ratio: Target size ratio after pruning (default: 0.8)
        """
        ...

async def register_llm(
    model_input: ModelInput,
    model_type: ModelType,
    endpoint: Endpoint,
    model_path: str,
    model_name: Optional[str] = None,
    context_length: Optional[int] = None,
    kv_cache_block_size: Optional[int] = None,
    router_mode: Optional[RouterMode] = None,
    runtime_config: Optional[ModelRuntimeConfig] = None,
    user_data: Optional[Dict[str, Any]] = None,
    custom_template_path: Optional[str] = None,
    lora_name: Optional[str] = None,
    base_model_path: Optional[str] = None,
) -> None:
    """
    Attach the model at path to the given endpoint, and advertise it as model_type.
    LoRA Registration:
        The `lora_name` and `base_model_path` parameters must be provided together or not at all.
        Providing only one of these parameters will raise a ValueError.
        - `lora_name`: The served model name for the LoRA model
        - `base_model_path`: Path to the base model that the LoRA extends

    For TensorBased models (using ModelInput.Tensor), HuggingFace downloads are skipped
    and a minimal model card is registered directly. Use model_path as the display name
    for these models.
    """
    ...

async def unregister_llm(
    endpoint: Endpoint,
    lora_name: Optional[str] = None,
) -> None:
    """
    Unregister a model from the discovery system.

    If lora_name is provided, unregisters a LoRA adapter instead of a base model.
    """
    ...

def lora_name_to_id(lora_name: str) -> int:
    """Generate a deterministic integer ID from a LoRA name using blake3 hash."""
    ...

async def fetch_llm(remote_name: str, ignore_weights: bool = False) -> str:
    """
    Download a model from Hugging Face, returning it's local path.
    If `ignore_weights` is True, only fetches tokenizer and config files.
    Example: `model_path = await fetch_llm("Qwen/Qwen3-0.6B")`
    """
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
        dp_rank: Optional[int] = None,
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
            dp_rank: Optional data parallel rank to route to. If set along with worker_id,
                    the request will be routed to the specific (worker_id, dp_rank) pair.
                    If only dp_rank is set, the router will select the best worker but
                    force routing to the specified dp_rank.

        Returns:
            An async iterator yielding generation responses

        Note:
            - If worker_id is set, the request bypasses KV matching and routes directly
              to the specified worker while still updating router states.
            - dp_rank allows targeting a specific data parallel replica when workers have
              multiple replicas (data_parallel_size > 1).
            - This is different from query_instance_id which doesn't route the request.
        """
        ...

    async def best_worker(
        self,
        token_ids: List[int],
        router_config_override: Optional[JsonLike] = None,
        request_id: Optional[str] = None,
    ) -> Tuple[int, int, int]:
        """
        Find the best matching worker for the given tokens.

        Args:
            token_ids: List of token IDs to find matches for
            router_config_override: Optional router configuration override
            request_id: Optional request ID. If provided, router states will be updated
                       to track this request (active blocks, lifecycle events). If not
                       provided, this is a query-only operation that doesn't affect state.

        Returns:
            A tuple of (worker_id, dp_rank, overlap_blocks) where:
                - worker_id: The ID of the best matching worker
                - dp_rank: The data parallel rank of the selected worker
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
                - dp_rank: The data parallel rank
                - potential_prefill_tokens: Number of tokens that would need prefill
                - potential_decode_blocks: Number of blocks currently in decode phase

        Note:
            Each (worker_id, dp_rank) pair is returned as a separate entry.
            If you need aggregated loads per worker_id, sum the values manually.
        """
        ...

    async def dump_events(self) -> str:
        """
        Dump all events from the KV router's indexer.

        Returns:
            A JSON string containing all indexer events
        """
        ...

    async def mark_prefill_complete(self, request_id: str) -> None:
        """
        Mark prefill as completed for a request.

        This signals that the request has finished its prefill phase and is now
        in the decode phase. Used to update router state for accurate load tracking.

        Args:
            request_id: The ID of the request that completed prefill

        Note:
            This is typically called automatically by the router when using the
            `generate()` method. Only call this manually if you're using
            `best_worker()` with `request_id` for custom routing.
        """
        ...

    async def free(self, request_id: str) -> None:
        """
        Free a request by its ID, signaling the router to release resources.

        This should be called when a request completes to update the router's
        tracking of active blocks and ensure accurate load balancing.

        Args:
            request_id: The ID of the request to free

        Note:
            This is typically called automatically by the router when using the
            `generate()` method. Only call this manually if you're using
            `best_worker()` with `request_id` for custom routing.
        """
        ...

class EntrypointArgs:
    """
    Settings to connect an input to a worker and run them.
    Use by `dynamo run`.
    """

    ...

class PlannerDecision:
    """A request from planner to client to perform a scaling action.
    Fields: num_prefill_workers, num_decode_workers, decision_id.
            -1 in any of those fields mean not set, usually because planner hasn't decided anything yet.
    Call VirtualConnectorClient.complete(event) when action is completed.
    """
    ...

class VirtualConnectorCoordinator:
    """Internal planner virtual connector component"""

    def __init__(self, runtime: DistributedRuntime, dynamo_namespace: str, check_interval_secs: int, max_wait_time_secs: int, max_retries: int) -> None:
        ...

    async def async_init(self) -> None:
        """Call this before using the object"""
        ...

    def read_state(self) -> PlannerDecision:
        """Get the current values. Most for test / debug."""
        ...

    async def update_scaling_decision(self, num_prefill: Optional[int] = None, num_decode: Optional[int] = None) -> None:
        ...

    async def wait_for_scaling_completion(self) -> None:
        ...

class VirtualConnectorClient:
    """How a client discovers planner requests and marks them complete"""

    def __init__(self, runtime: DistributedRuntime, dynamo_namespace: str) -> None:
        ...

    async def get(self) -> PlannerDecision:
        ...

    async def complete(self, decision: PlannerDecision) -> None:
        ...

    async def wait(self) -> None:
        """Blocks until there is a new decision to fetch using 'get'"""
        ...

__all__ = [
    "Client",
    "Component",
    "Context",
    "KserveGrpcService",
    "ModelDeploymentCard",
    "PythonAsyncEngine",
    "prometheus_names",
]
