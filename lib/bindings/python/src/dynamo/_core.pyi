# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
)

from typing_extensions import NotRequired

# Import from specialized modules
from .prometheus_metrics import RuntimeMetrics as PyRuntimeMetrics

def log_message(level: str, message: str, module: str, file: str, line: int) -> None:
    """
    Log a message from Python with file and line info
    """
    ...

def get_tool_parser_names() -> list[str]:
    """Get list of available tool parser names."""
    ...

def get_reasoning_parser_names() -> list[str]:
    """Get list of available reasoning parser names."""
    ...

def run_kv_indexer(args: List[str]) -> None:
    """Run the KV indexer with the given arguments."""
    ...

def run_slot_tracker(args: List[str]) -> None:
    """Run the KV router slot tracker with the given arguments."""
    ...

def run_select_service(args: List[str]) -> None:
    """Run the Dynamo selection service with the given arguments."""
    ...

# Any Python object that can be serialized to JSON (dict, list, str, int, etc.)
JsonLike = Any

RequestHandler = Callable[..., AsyncIterator[JsonLike]]

class DistributedRuntime:
    """
    The runtime object for dynamo applications
    """

    def __new__(
        cls,
        event_loop: Any,
        discovery_backend: str,
        request_plane: str,
        enable_nats: Optional[bool] = None,
        *,
        event_plane: Optional[str] = None,
    ) -> "DistributedRuntime":
        """
        Create a new DistributedRuntime.

        Args:
            event_loop: The asyncio event loop
            discovery_backend: Discovery backend ("kubernetes", "etcd", "file", or "mem")
            request_plane: Request plane transport ("tcp" or "nats")
            enable_nats: Deprecated; NATS enablement is inferred from runtime config
            event_plane: Event plane transport ("nats" or "zmq")
        """
        ...

    def endpoint(self, path: str) -> Endpoint:
        """
        Get an endpoint directly by path.

        Args:
            path: Endpoint path in format 'namespace.component.endpoint'
                  or 'dyn://namespace.component.endpoint'

        Returns:
            Endpoint: The requested endpoint

        Raises:
            ValueError: If path format is invalid (not 3 parts separated by dots)
            Exception: If namespace or component creation fails

        Example:
            endpoint = runtime.endpoint("demo.backend.generate")
            endpoint = runtime.endpoint("dyn://demo.backend.generate")
        """
        ...

    def shutdown(self) -> None:
        """
        Shutdown the runtime by triggering the cancellation token
        """
        ...

    def set_health_status(self, ready: bool) -> None:
        """
        Explicitly set the system-level health status (Ready / NotReady).
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
            route_name: The route path (e.g., "control/start_profile" creates /engine/control/start_profile)
            callback: Async function with signature: async def(body: dict) -> dict

        Example:
            async def start_profile(body: dict) -> dict:
                await engine.start_profile(**body)
                return {"status": "ok", "message": "Profiling started"}

            runtime.register_engine_route("control/start_profile", start_profile)

        The callback receives the JSON request body as a dict and should return
        a dict that will be serialized as the JSON response.

        For GET requests or empty bodies, an empty dict {} is passed.
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

    async def serve_bidirectional_endpoint(
        self,
        handler: Callable[..., AsyncIterator[JsonLike]],
        graceful_shutdown: bool = True,
        metrics_labels: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """
        Serve a bidirectional (streaming-input, streaming-output) endpoint.

        The handler is an async generator function — `async def
        generate(request_stream)` or `async def generate(request_stream,
        context)` — so calling it returns an async iterator of response frames
        directly (it is not awaited). `request_stream` is a
        `PyAsyncRequestStream` yielding inbound frames as JSON-like Python
        objects; the generator yields response frames as JSON-like Python
        objects.

        Request-stream end (when `__anext__` raises `StopAsyncIteration`)
        is not a cancellation signal: the caller has merely stopped sending
        input. The engine must keep yielding response chunks until it
        chooses to return or observes `context.is_stopped()`.

        Args:
            handler: The async generator factory described above
            graceful_shutdown: Whether to wait for inflight requests to complete during shutdown (default: True)
            metrics_labels: Optional list of metrics labels to add to the metrics
        """
        ...

    async def client(self, router_mode: Optional[RouterMode] = None) -> Client:
        """
        Create a `Client` capable of calling served instances of this endpoint.

        By default this uses round-robin routing when `router_mode` is not provided.
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

class PyAsyncRequestStream:
    """
    Python-visible inbound iterator handed to bidirectional engine
    handlers as the first positional argument. Yields request frames as
    JSON-like Python objects.

    Request-stream end is not a cancellation signal: when this iterator
    raises `StopAsyncIteration`, the caller has merely stopped sending
    input. The engine should keep yielding response chunks until it
    chooses to return or observes `context.is_stopped()`.
    """

    def __aiter__(self) -> "PyAsyncRequestStream": ...
    async def __anext__(self) -> JsonLike: ...

class TransportType:
    """
    A read-only view of an instance's transport, wrapping the runtime
    ``TransportType``. ``kind`` is the transport variant ("tcp" / "nats_tcp")
    and ``address`` is its (transport-specific) address. The address format is
    not a stable parse target.
    """

    @property
    def kind(self) -> str: ...
    @property
    def address(self) -> str: ...

class Instance:
    """
    A read-only view of a single registered instance of an endpoint, wrapping a
    snapshot of the runtime ``Instance``. ``str(instance)`` yields
    ``"namespace/component/endpoint/instance_id"``.
    """

    @property
    def instance_id(self) -> int: ...
    @property
    def namespace(self) -> str: ...
    @property
    def component(self) -> str: ...
    @property
    def endpoint(self) -> str: ...
    @property
    def transport(self) -> TransportType: ...
    @property
    def device_type(self) -> Optional[str]:
        """Device type, e.g. "cpu" or "cuda", or None if unspecified."""
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

    def instances(self) -> List[Instance]:
        """
        Get a snapshot of the current instances with full transport details.

        Like ``instance_ids()``, the result is a snapshot of the watched
        instance set; pair with ``wait_for_instances()`` to block until
        instances exist.

        Returns:
            A list of ``Instance`` for the currently available instances,
            across all transports (TCP, NATS, ...).
        """
        ...

    async def wait_for_instances(self) -> List[int]:
        """
        Wait for instances to be available for work and return their IDs.

        Returns:
            A list of instance IDs that are available for work
        """
        ...

    async def wait_for_instance_by_runtime_data(
            self,
            key: str,
            value: str,
            timeout_s: float | None = None,
        ) -> int:
        """
        Wait for exactly one instance whose MDC runtime_data contains the given string value.
        """
        ...

    async def random(
            self,
            request: JsonLike,
            annotated: bool | None = True,
            context: Context | None = None,
        ) -> AsyncIterator[JsonLike]:
        """
        Pick a random instance of the endpoint and issue the request
        """
        ...

    async def round_robin(
            self,
            request: JsonLike,
            annotated: bool | None = True,
            context: Context | None = None,
        ) -> AsyncIterator[JsonLike]:
        """
        Pick the next instance of the endpoint in a round-robin fashion
        """
        ...

    async def direct(
            self,
            request: JsonLike,
            instance_id: int,
            annotated: bool | None = True,
            context: Context | None = None,
        ) -> AsyncIterator[JsonLike]:
        """
        Pick a specific instance of the endpoint
        """
        ...

    async def generate(
            self,
            request: JsonLike,
            annotated: bool | None = True,
            context: Context | None = None,
        ) -> AsyncIterator[JsonLike]:
        """
        Generate a response from the endpoint
        """
        ...


class ModelCardInstanceId:
    """
    Unique identifier for a worker instance: namespace, component, endpoint and instance_id.
    The instance_id is not currently exposed in the Python bindings.
    """
    def triple(self) -> Tuple[str, str, str]:
        """
        Triple of namespace, component and endpoint this worker is serving.
        """
        ...


def compute_block_hash_for_seq(
    tokens: List[int],
    kv_block_size: int,
    block_mm_infos: Optional[List[Optional[Dict[str, Any]]]] = None,
    lora_name: Optional[str] = None,
    is_eagle: Optional[bool] = None,
    cache_namespace: Optional[str] = None,
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
        lora_name: Optional LoRA adapter name for adapter-aware block hashing.
        is_eagle: Optional Eagle mode flag. When true, hashes use overlapping
                  `kv_block_size + 1` token windows with `kv_block_size` stride.

    Returns:
        List of block hashes (one per block)

    Example:
        >>> tokens = [1, 2, 3, 4] * 8  # 32 tokens = 1 block
        >>> mm_info = {
        ...     "mm_objects": [{
        ...         "mm_hash": 0xDEADBEEF,
        ...     }]
        ... }
        >>> hashes = compute_block_hash_for_seq(tokens, 32, [mm_info])
    """

    ...

class ContextMetadata:
    """
    Live mutable view over propagated context metadata.
    """
    def __getitem__(self, key: str) -> str: ...
    def __setitem__(self, key: str, value: str) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: str) -> bool: ...
    def __iter__(self) -> Any: ...
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]: ...
    def pop(self, key: str, default: Optional[str] = None) -> Optional[str]: ...
    def keys(self) -> List[str]: ...
    def values(self) -> List[str]: ...
    def items(self) -> List[Tuple[str, str]]: ...
    def clear(self) -> None: ...
    def copy(self) -> Dict[str, str]: ...

class Context:
    """
    Context wrapper around AsyncEngineContext for Python bindings.
    Provides tracing and cancellation capabilities for request handling.
    """

    def __init__(
        self,
        id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Create a new Context instance.

        Args:
            id: Optional request ID. If None, a default ID will be generated.
            metadata: Optional propagated metadata map.
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

    def id(self) -> str:
        """
        Get the context ID.

        Returns:
            The context identifier string.
        """
        ...

    def detached(self, id: str) -> "Context":
        """
        Create a context with a fresh cancellation controller and request ID
        while preserving trace parentage and a metadata snapshot.
        """
        ...

    def async_killed_or_stopped(self) -> asyncio.Future[bool]:
        """
        Asynchronously wait until the context is killed or stopped.

        Returns:
            True when the context is killed or stopped.
        """
        ...

    def notify_first_token(self) -> None:
        """Fire the first-token signal so the framework can release any
        deferred ``engine.abort()``. Idempotent; no-op on non-decode
        requests. Engines normally don't need this — the framework
        auto-fires on the first non-empty chunk in the response stream."""
        ...

    @property
    def metadata(self) -> ContextMetadata:
        """
        Get the live propagated context metadata mapping.
        """
        ...

    @metadata.setter
    def metadata(self, metadata: Dict[str, str]) -> None: ...

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

    def trace_headers(self) -> Optional[dict[str, str]]:
        """
        Build W3C trace headers for propagating to downstream inference engines.

        Returns:
            ``{"traceparent": "00-<trace_id>-<span_id>-<flags>"}`` when this
            request carries trace context, ``None`` otherwise. Also emits ``tracestate``,
            ``x-request-id``, ``request-id`` when upstream propagated them.
            Forward unchanged to the inference engine's ``trace_headers`` kwarg.
        """
        ...

    def current_span(self) -> "SpanProxy":
        """
        Handle on the framework's ``engine.generate`` span. Use it to
        ``set_attribute`` / ``add_event`` / ``set_status`` on the parent
        span. Returns a silent no-op proxy when no parent was plumbed in
        (test contexts) or the OTel bridge isn't installed.

        Engines normally reach this through
        ``dynamo.common.backend.telemetry.current_span(context)``.
        """
        ...

    def start_span(
        self, name: str, attrs: Optional[dict[str, Any]] = None
    ) -> "SpanProxy":
        """
        Open a child span under ``engine.generate`` with a dynamic name.
        The returned ``SpanProxy`` is a context manager — the span ends on
        ``__exit__`` / ``close()`` / drop.

        Engines normally reach this through
        ``dynamo.common.backend.telemetry.start_span(context, name)``.
        """
        ...

class SpanProxy:
    """
    Unified span handle returned by ``Context.current_span()`` (the
    framework auto-span) and ``Context.start_span()`` (child spans).
    Mirrors the OTel ``Span`` API: ``set_attribute`` / ``add_event`` /
    ``set_status``. Usable as a Python context manager (closes on
    ``__exit__``). All methods are silent no-ops when the underlying
    span is absent.
    """

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span. Any key is accepted; OTel imposes
        no pre-declaration constraint."""
        ...

    def add_event(self, name: str, attrs: Optional[dict[str, Any]] = None) -> None:
        """Emit a structured event on the span."""
        ...

    def set_status(self, status: str, description: Optional[str] = None) -> None:
        """Set the span's status. ``status`` is ``"ok"`` or ``"error"``;
        ``description`` is optional context (typically a short error name)."""
        ...

    def close(self) -> None:
        """End the underlying span (child spans only — no-op for the
        auto-span). Idempotent."""
        ...

    def __enter__(self) -> "SpanProxy": ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool: ...

class WorkerMetricsPublisher:
    """
    A metrics publisher will provide metrics to the router for load monitoring.
    """

    ...

    def __init__(self) -> None:
        """
        Create a `WorkerMetricsPublisher` object
        """

    async def create_endpoint(self, endpoint: Endpoint) -> None:
        """
        Initialize event-plane publishing for worker metrics. Must be awaited.

        Extracts component information from the endpoint to set up metrics publishing
        on the endpoint-scoped event subject used for routing decisions.

        Args:
            endpoint: The endpoint to extract component information from for metrics publishing
        """

    def publish(
        self,
        dp_rank: Optional[int] = None,
        active_decode_blocks: int | None = None,
        kv_used_blocks: int | None = None,
    ) -> None:
        """
        Publish worker metrics for load monitoring.

        Args:
            dp_rank: Data parallel rank of the worker (None defaults to 0)
            active_decode_blocks: Optional scheduler-compatible decode-block signal
            kv_used_blocks: Optional authoritative total KV blocks currently in use
        """
        ...

class MultimodalEmbeddingCachePublisher:
    """
    A publisher for multimodal encode-worker cache state.
    """

    ...

    def __init__(self) -> None:
        """
        Create a `MultimodalEmbeddingCachePublisher` object.
        """

    async def create_endpoint(self, endpoint: Endpoint) -> None:
        """
        Initialize event-plane publishing for multimodal cache state.

        Args:
            endpoint: The endpoint to extract component information from.
        """

    def publish_delta(self, added_keys: list[str], removed_keys: list[str]) -> None:
        """
        Publish an incremental cache mutation for this worker.

        Args:
            added_keys: Newly cached embedding keys.
            removed_keys: Cache keys no longer present on the worker.
        """
        ...

class SelectionCacheConfig:
    """
    Bounds for the in-flight selection cache. Each field defaults to the
    service default when omitted.
    """

    def __init__(
        self,
        *,
        ttl_secs: Optional[float] = None,
        max_entries: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ) -> None: ...

class SelectionService:
    """
    In-process handle to a runtime-free Dynamo selection core.
    """

    def __init__(
        self,
        *,
        indexer_threads: int = 4,
        indexer_peers: Optional[list[str]] = None,
        replica_sync_port: Optional[int] = None,
        replica_sync_peers: Optional[list[str]] = None,
        selection_cache: Optional[SelectionCacheConfig] = None,
    ) -> None:
        """Create a selection service. `indexer_threads` sizes the KV indexer pool."""
        ...

    def shutdown(self) -> None:
        """
        Stop the service: cancel KV-event listeners and scheduling so that
        in-flight and queued selections fail fast.

        The KV indexer thread pool is released when the handle is dropped.
        Idempotent, and also runs automatically on drop.
        """
        ...

    async def upsert_worker(self, worker: JsonLike) -> JsonLike:
        """Upsert a worker and subscribe to its live KV events; returns its catalog record."""
        ...

    async def delete_worker(self, worker_id: int) -> JsonLike:
        """Remove a worker and tear down its KV-event listener; returns its catalog record."""
        ...

    def list_workers(
        self, *, model_name: Optional[str] = None, routing_group: Optional[str] = None
    ) -> JsonLike:
        """List catalog records, optionally filtered by model and routing group."""
        ...

    def ready(self) -> JsonLike:
        """Readiness: whether at least one worker is schedulable, plus catalog state."""
        ...

    async def overlap_scores(self, request: JsonLike) -> JsonLike:
        """Per-worker KV-overlap scores for a prompt."""
        ...

    async def select(self, request: JsonLike) -> JsonLike:
        """Select the best worker by KV-overlap + load, without booking."""
        ...

    async def select_and_reserve(self, request: JsonLike) -> JsonLike:
        """Select the best worker and book its load."""
        ...

    async def create_reservation(self, request: JsonLike) -> JsonLike:
        """Book a request's load against a worker, keyed by ``selection_id``.

        Without a ``worker_id``, replays the matching ``select``'s cached
        selection (same model/routing-group), booked under ``selection_id``;
        other request fields are ignored. With a ``worker_id`` and the prompt,
        books explicitly under ``selection_id`` on that worker and discards any
        cached selection for the id. ``selection_id`` is required.
        """
        ...

    async def prefill_complete(self, selection_id: str) -> None:
        """Mark a reservation's prefill complete; its load shifts prefill -> decode."""
        ...

    def add_output_block(
        self, selection_id: str, *, decay_fraction: Optional[float] = None
    ) -> None:
        """Record one decode output block for a reservation, advancing its decode load."""
        ...

    async def free_reservation(self, selection_id: str) -> None:
        """Free a finished reservation, releasing its tracked load."""
        ...

    def loads(
        self, *, model_name: Optional[str] = None, routing_group: Optional[str] = None
    ) -> JsonLike:
        """Current per-model active load (pending counts + per-worker potential loads)."""
        ...

    async def potential_loads(self, request: JsonLike) -> JsonLike:
        """Per-worker potential loads for a prompt, without booking."""
        ...

class ModelDeploymentCard:
    """
    A model deployment card is a collection of model information
    """

    def to_json_str(self) -> str:
        """Serialize the model deployment card to a JSON string."""
        ...

    @staticmethod
    def from_json_str(json: str) -> "ModelDeploymentCard":
        """Deserialize a model deployment card from a JSON string."""
        ...

    def model_type(self) -> ModelType:
        """Return the model type of this deployment card."""
        ...

    def source_path(self) -> str:
        """Return the source path of this deployment card."""
        ...

    def local_dir(self) -> str:
        """Resolved metadata directory (post-`download_config`). Raises
        ValueError if the path contains non-UTF-8 bytes."""
        ...

    def name(self) -> str:
        """Return the model name."""
        ...

    def runtime_config(self) -> Any:
        """Return the runtime configuration as a dict."""
        ...

class ModelRuntimeConfig:
    """
    A model runtime configuration is a collection of runtime information
    """

    context_length: int | None
    total_kv_blocks: int | None
    max_num_seqs: int | None
    max_num_batched_tokens: int | None
    tool_call_parser: str | None
    reasoning_parser: str | None
    tokenizer_backend: str | None
    exclude_tools_when_tool_choice_none: bool
    data_parallel_start_rank: int
    data_parallel_size: int
    enable_local_indexer: bool
    kv_state_endpoint: str | None
    enable_eagle: bool
    taints: Set[str]
    stable_routing_id: str | None
    runtime_data: dict[str, Any]
    topology_domains: dict[str, str]
    kv_transfer_domain: str | None
    kv_transfer_enforcement: str | None
    kv_transfer_preferred_weight: float | None
    bootstrap_host: str | None
    bootstrap_port: int | None

    def __init__(self) -> None: ...

    def set_engine_specific(self, key: str, value: Any) -> None:
        """Set an engine-specific runtime configuration value"""
        ...

    def get_engine_specific(self, key: str) -> Any | None:
        """Get an engine-specific runtime configuration value"""
        ...

    def set_structural_tag_mode(self, mode: str) -> None:
        """Set structural tag mode ("off" or "on")."""
        ...

    def set_structural_tag_scope(self, scope: str) -> None:
        """Set structural tag scope ("auto" or "always")."""
        ...

    def set_structural_tag_schema(self, schema: str) -> None:
        """Set structural tag schema mode ("auto" or "strict")."""
        ...

    def set_disaggregated_endpoint(
            self,
            bootstrap_host: str | None = None,
            bootstrap_port: int | None = None,
        ) -> None:
        """Set the disaggregated endpoint for the model"""
        ...

class RoutingConstraints:
    """
    Request-side routing constraints.

    ``required_taints`` is a hard eligibility filter.
    ``preferred_taints`` maps taint -> signed weight.
    Positive weights prefer matching workers, negative weights avoid them,
    and ``0.0`` is neutral. Matching weights are summed and squashed with
    ``tanh``, so opposite preferences cancel before Dynamo converts the
    bounded bias into a strictly positive score multiplier.
    """
    required_taints: Set[str]
    preferred_taints: Dict[str, float]

    def __init__(
        self,
        required_taints: Optional[Set[str]] = None,
        preferred_taints: Optional[Dict[str, float]] = None,
    ) -> None: ...

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

    def __init__(self) -> None:
        """
        Create a new RadixTree instance.
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

    def __init__(self, endpoint: Endpoint, block_size: int) -> None:
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
        self, token_ids: List[int], lora_name: Optional[str] = None, is_eagle: Optional[bool] = None
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
        endpoint: Endpoint,
        kv_block_size: int,
        router_ttl_secs: float = 120.0,
    ) -> None:
        """
        Create an `ApproxKvIndexer` object

        Args:
            component: The component to associate with this indexer
            kv_block_size: The KV cache block size
            router_ttl_secs: TTL for blocks in seconds (default: 120.0)
        """
        ...

    def find_matches_for_request(
        self, token_ids: List[int], lora_name: Optional[str] = None, is_eagle: Optional[bool] = None
    ) -> OverlapScores:
        """
        Return the overlapping scores of workers for the given token ids.

        Args:
            token_ids: List of token IDs to find matches for
            lora_name: Optional LoRA adapter name for adapter-aware matching

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


class KvStoredEventInput(TypedDict):
    type: Literal["stored"]
    token_ids: List[int]
    num_block_tokens: List[int]
    block_hashes: List[int]
    parent_hash: NotRequired[Optional[int]]
    block_mm_infos: NotRequired[Optional[List[Optional[Dict[str, Any]]]]]
    lora_name: NotRequired[Optional[str]]
    is_eagle: NotRequired[Optional[bool]]
    cache_salt: NotRequired[Optional[str]]


class KvRemovedEventInput(TypedDict):
    type: Literal["removed"]
    block_hashes: List[int]


class KvEventPublisher:
    """
    A KV event publisher will publish KV events corresponding to the component.
    """

    ...

    def __init__(
        self,
        endpoint: Endpoint,
        worker_id: Optional[int] = None,
        kv_block_size: int = 0,
        dp_rank: int = 0,
        enable_local_indexer: bool = False,
        zmq_endpoint: Optional[str] = None,
        zmq_topic: Optional[str] = None,
        batching_timeout_ms: Optional[int] = None,
        image_token_id: Optional[int] = None,
        kv_state_endpoint: Optional[str] = None,
    ) -> None:
        """
        Create a `KvEventPublisher` object.

        When zmq_endpoint is provided, the publisher subscribes to a ZMQ socket for
        incoming engine events (e.g. from SGLang/vLLM) and relays them to NATS.

        When zmq_endpoint is None, events are pushed manually via publish_batch,
        publish_stored, or publish_removed.

        Args:
            endpoint: The endpoint to extract component information from for event publishing
            worker_id: Optional worker ID override. Use None to infer from endpoint.
            kv_block_size: The KV block size (must be > 0)
            dp_rank: The data parallel rank (defaults to 0)
            enable_local_indexer: Enable worker-local KV indexer
            zmq_endpoint: Optional ZMQ endpoint for relay mode (e.g. "tcp://127.0.0.1:5557")
            zmq_topic: ZMQ topic to subscribe to (defaults to "" when zmq_endpoint is set)
            batching_timeout_ms: Cross-list batching timeout in milliseconds. None/0
                flushes at each submitted source-list boundary.
            kv_state_endpoint: KV event ownership endpoint; defaults to endpoint.
        """

    def publish_stored(
        self,
        token_ids: List[int],
        num_block_tokens: List[int],
        block_hashes: List[int],
        parent_hash: Optional[int] = None,
        block_mm_infos: Optional[List[Optional[Dict[str, Any]]]] = None,
        lora_name: Optional[str] = None,
        is_eagle: Optional[bool] = None,
        cache_salt: Optional[str] = None,
    ) -> None:
        """
        Publish a KV stored event.

        Event IDs are managed internally by the publisher using a monotonic counter.

        Args:
            token_ids: List of token IDs
            num_block_tokens: Number of tokens per block
            block_hashes: List of block hashes (signed 64-bit integers)
            parent_hash: Optional parent hash (signed 64-bit integer)
            block_mm_infos: Optional list of multimodal info for each block.
                Each item is either None or a dict with "mm_objects" key containing
                a list of {"mm_hash": int, "offsets": [[start, end], ...]} dicts.
            lora_name: Optional LoRA adapter name for adapter-aware block hashing.
            is_eagle: Optional Eagle mode flag. When true, stored blocks are
                reconstructed using overlapping `kv_block_size + 1` token windows.
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

    def publish_batch(
        self, events: Sequence[KvStoredEventInput | KvRemovedEventInput]
    ) -> None:
        """
        Publish an ordered list of KV events as one processor input.

        The complete list is validated before it is enqueued. Compatible
        events are coalesced while preserving source order and the processor's
        existing block-count limits.
        """
        ...

    def shutdown(self) -> None:
        """
        Shuts down the event publisher, stopping any background tasks.
        """
        ...


class FpmEventRelay:
    """
    Relay that bridges ForwardPassMetrics from a local raw ZMQ PUB socket
    (InstrumentedScheduler in EngineCore child process) to the Dynamo event
    plane with automatic discovery registration.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        zmq_endpoint: str,
    ) -> None:
        """
        Create a relay.

        Args:
            endpoint: Dynamo component endpoint (provides runtime + discovery).
            zmq_endpoint: Local ZMQ PUB address to subscribe to
                (e.g., "tcp://127.0.0.1:20380").
        """
        ...

    def shutdown(self) -> None:
        """Shut down the relay task."""
        ...


class FpmDirectPublisher:
    """
    Direct Forward Pass Metrics publisher used by in-process producers such
    as the TRT-LLM adapter. The underlying Rust publisher owns per-DP-rank
    serialization tasks (each with its own 1s idle heartbeat timer) and a
    single event-plane publisher task. Python callers do not manage
    heartbeat: when ``publish`` is not called for ``IDLE_HEARTBEAT_INTERVAL``
    (1.0s, matching vLLM's ``HEARTBEAT_INTERVAL``), the Rust side emits a
    zeroed snapshot on that rank's channel.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        worker_id: str,
        dp_size: int = 1,
    ) -> None:
        """
        Create a publisher with ``dp_size`` per-DP-rank channels.

        Args:
            endpoint: Dynamo component endpoint (provides runtime + discovery).
            worker_id: Unique worker identifier stamped on every emitted FPM.
            dp_size: Number of DP ranks to allocate channels for. Use ``1``
                when attention DP is disabled.
        """
        ...

    def publish(
        self,
        *,
        dp_rank: int,
        scheduled_num_prefill_requests: int,
        scheduled_sum_prefill_tokens: int,
        scheduled_sum_prefill_kv_tokens: int,
        scheduled_num_decode_requests: int,
        scheduled_sum_decode_kv_tokens: int,
        queued_num_prefill_requests: int,
        queued_sum_prefill_tokens: int,
        queued_num_decode_requests: int,
        queued_sum_decode_kv_tokens: int,
        wall_time_secs: float,
    ) -> None:
        """
        Publish one iteration's FPM snapshot for the given DP rank.

        All parameters are keyword-only on the Python side: adjacent ints
        with similar units (``scheduled_*`` vs ``queued_*``, ``*_prefill_*``
        vs ``*_decode_*``) cannot be distinguished by the type system, so
        a transposition would silently corrupt every published snapshot.

        Variance fields (var_prefill_length, var_decode_kv_tokens,
        var_queued_prefill_length, var_queued_decode_kv_tokens) are defaulted
        to 0.0 per the MVP scope; a follow-up PR can add Welford-based
        variance computation.
        """
        ...

    def shutdown(self) -> None:
        """Shut down the publisher and its per-rank serialization tasks."""
        ...


class FpmEventSubscriber:
    """
    Subscriber for ForwardPassMetrics from the Dynamo event plane.
    Auto-discovers engine publishers via the discovery plane.

    Two mutually exclusive usage modes:

    1. **recv mode** (default): call ``recv()`` to pull individual messages.
    2. **tracking mode**: call ``start_tracking()`` once, then poll
       ``get_recent_stats()`` to retrieve the latest FPM bytes keyed by
       ``(worker_id, dp_rank)``.  Stale entries are cleaned up when
       workers are removed (via discovery watch).
    """

    def __init__(self, endpoint: Endpoint) -> None:
        """
        Create a subscriber that auto-discovers FPM publishers.

        No background tasks are started until ``recv()`` or
        ``start_tracking()`` is called.

        Args:
            endpoint: Dynamo component endpoint (provides runtime + discovery).
        """
        ...

    def recv(self) -> Optional[bytes]:
        """
        Blocking receive of the next message (raw msgspec bytes).
        Releases the GIL while waiting.

        On the first call a background subscriber task is spawned (recv mode).
        Cannot be used after ``start_tracking()``.

        Returns:
            Raw msgspec payload, or None if the stream is closed.
        """
        ...

    def start_tracking(self) -> None:
        """
        Start background tracking of the latest FPM per (worker_id, dp_rank).

        Spawns two background tasks:

        1. Event consumption: subscribes to FPM events, extracts the composite
           key (worker_id, dp_rank) from the msgpack payload, stores latest
           raw bytes in an internal map.
        2. MDC discovery watch: monitors ComponentModels for the target
           component.  When a model is removed, all entries whose
           worker_id matches the removed instance_id are purged.

        After calling this, ``recv()`` will raise RuntimeError.
        """
        ...

    def get_recent_stats(self) -> dict[tuple[str, int], bytes]:
        """
        Return the latest FPM bytes for every tracked (worker_id, dp_rank).

        Cleanup of removed engines is handled by the MDC discovery watch
        task spawned by ``start_tracking()``.

        Raises RuntimeError if ``start_tracking()`` has not been called.

        Returns:
            dict mapping ``(worker_id, dp_rank)`` to raw msgspec bytes.
            Decode each value with ``forward_pass_metrics.decode(data)``.
        """
        ...

    def get_model_cards(self) -> dict[str, str]:
        """
        Snapshot of model deployment cards keyed by worker id.

        The snapshot is filtered against the known-workers set so entries
        for already-removed workers are not returned.  Values are the raw
        ``ModelDeploymentCard`` serialized as a JSON string; callers parse
        whichever fields they need (e.g. ``runtime_config``,
        ``display_name``).

        Raises RuntimeError if ``start_tracking()`` has not been called.

        Returns:
            dict mapping ``worker_id`` to ``card_json`` (JSON string).
        """
        ...

    def shutdown(self) -> None:
        """Shut down the subscriber (all background tasks)."""
        ...


class HttpService:
    """
    A HTTP service for dynamo applications.
    It is a OpenAI compatible http ingress into the Dynamo Distributed Runtime.
    """

    def __init__(self, port: Optional[int] = None) -> None:
        """
        Create a new HTTP service.

        Args:
            port: Optional port number to bind the service to (default: 8080)
        """
        ...

    async def run(self, runtime: DistributedRuntime) -> None:
        """
        Run the HTTP service.

        Args:
            runtime: DistributedRuntime instance for token management
        """
        ...

    def shutdown(self) -> None:
        """
        Shutdown the HTTP service by cancelling its internal token.
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
        *,
        runtime_config: Optional[ModelRuntimeConfig] = None,
        tensor_model_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a tensor-based model with the service.

        Args:
            model: The model name
            checksum: The model checksum
            engine: The async engine to handle requests
            runtime_config: Optional runtime-resolved worker metadata
            tensor_model_config: Optional tensor protocol model metadata
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

    async def run(self, runtime: DistributedRuntime) -> None:
        """
        Run the KServe gRPC service.

        Args:
            runtime: DistributedRuntime instance for token management
        """
        ...

    def shutdown(self) -> None:
        """
        Shutdown the KServe gRPC service by cancelling its internal token.
        """
        ...

class ModelInput:
    """What type of request this model needs: Text, Tokens or Tensor"""
    Text: ModelInput
    Tokens: ModelInput
    Tensor: ModelInput


class ModelType:
    """What type of request this model supports: Chat, Completions, Embedding, Tensor, Images, Videos, Realtime, or Empty (no OpenAI surface)"""
    # No OpenAI surface — used by prefill / encode workers whose role is
    # carried by WorkerType. Symmetric with the other ModelType.Foo members.
    Empty: ModelType
    Chat: ModelType
    Completions: ModelType
    Embedding: ModelType
    TensorBased: ModelType
    # Legacy prefill marker (no OpenAI surface). Dual-emitted by new prefill
    # workers for cross-version compat so an old frontend still detects them;
    # the role is otherwise carried by WorkerType.Prefill. Compat window only.
    Prefill: ModelType
    Images: ModelType
    Audios: ModelType
    Videos: ModelType
    Realtime: ModelType

    def __or__(self, other: ModelType) -> ModelType:
        ...

    def supports_chat(self) -> bool:
        """Return True if this model type supports chat."""
        ...

class RouterMode:
    """Router mode for load balancing requests across workers"""
    RoundRobin: "RouterMode"
    Random: "RouterMode"
    PowerOfTwoChoices: "RouterMode"
    KV: "RouterMode"
    Direct: "RouterMode"
    LeastLoaded: "RouterMode"
    DeviceAwareWeighted: "RouterMode"
    ...

class RouterConfig:
    """How to route the request"""
    router_mode: RouterMode
    kv_router_config: KvRouterConfig

    def __init__(
        self,
        mode: RouterMode,
        config: Optional[KvRouterConfig] = None,
        active_decode_blocks_threshold: Optional[float] = None,
        active_prefill_tokens_threshold: Optional[int] = None,
        active_prefill_tokens_threshold_frac: Optional[float] = None,
        enforce_disagg: bool = False,
        session_affinity_ttl_secs: Optional[int] = None,
    ) -> None:
        """
        Create a RouterConfig.

        Args:
            mode: The router mode (RoundRobin, Random, KV, Direct, LeastLoaded, or DeviceAwareWeighted)
            config: Optional KV router configuration (used when mode is KV)
            active_decode_blocks_threshold: Threshold percentage (0.0-1.0) for decode blocks busy detection
            active_prefill_tokens_threshold: Literal token count threshold for prefill busy detection
            active_prefill_tokens_threshold_frac: Fraction of max_num_batched_tokens for busy detection
            enforce_disagg: Deprecated and ignored. Routing topology and readiness come from registered worker types.
            session_affinity_ttl_secs: Router-local session-affinity idle TTL in seconds.
        """
        ...

class AicPerfConfig:
    def __init__(
        self,
        aic_backend: str,
        aic_system: str,
        aic_model_path: str,
        aic_tp_size: int = 1,
        aic_backend_version: Optional[str] = None,
        aic_moe_tp_size: Optional[int] = None,
        aic_moe_ep_size: Optional[int] = None,
        aic_attention_dp_size: Optional[int] = None,
        aic_nextn: Optional[int] = None,
        aic_nextn_accept_rates: Optional[str] = None,
        aic_gemm_dtype: Optional[str] = None,
        aic_moe_dtype: Optional[str] = None,
        aic_fmha_dtype: Optional[str] = None,
        aic_kv_cache_dtype: Optional[str] = None,
        aic_comm_dtype: Optional[str] = None,
    ) -> None:
        ...

class AicEngineConfig:
    """AIC model/backend identity used by native forward-pass estimates."""

    def __init__(
        self,
        model_name: str,
        backend: str,
        system_name: str = "h200_sxm",
        backend_version: Optional[str] = None,
        tp_size: int = 1,
        pp_size: int = 1,
        moe_tp_size: Optional[int] = None,
        moe_ep_size: Optional[int] = None,
        attention_dp_size: Optional[int] = None,
        model_arch: Optional[str] = None,
        weight_dtype: Optional[str] = None,
        moe_dtype: Optional[str] = None,
        activation_dtype: Optional[str] = None,
        kv_cache_dtype: Optional[str] = None,
        kv_block_size: Optional[int] = None,
        extra: Optional[dict[str, str]] = None,
    ) -> None:
        ...

class EnginePerfLimits:
    """Engine limits used by engine-level helper queries and default correction bounds."""

    max_num_batched_tokens: int
    max_num_seqs: int
    max_kv_tokens: int

    def __init__(
        self,
        max_num_batched_tokens: int = 8192,
        max_num_seqs: int = 512,
        max_kv_tokens: int = 2000000,
    ) -> None:
        ...

class RustEnginePerfOptions:
    """Online tuning options for RustEnginePerfModel."""

    def __init__(
        self,
        max_observations: int = 64,
        min_observations: int = 5,
        bucket_count: int = 16,
        max_num_tokens: int = 8192,
        max_batch_size: int = 512,
        max_kv_tokens: int = 2000000,
    ) -> None:
        ...

class OptimizationTarget:
    Throughput: "OptimizationTarget"
    Latency: "OptimizationTarget"

class EngineCapacityRequest:
    """Request shape and SLA policy for find_engine_capacity_rps."""

    def __init__(
        self,
        isl: int,
        osl: int,
        ttft_sla_ms: Optional[float] = None,
        itl_sla_ms: Optional[float] = None,
        e2e_latency_sla_ms: Optional[float] = None,
        kv_hit_rate: Optional[float] = None,
        optimization_target: OptimizationTarget = OptimizationTarget.Throughput,
    ) -> None:
        ...

class EngineCapacity:
    """Per-engine capacity result."""

    rps: float
    ttft_ms: Optional[float]
    itl_ms: Optional[float]
    e2e_latency_ms: Optional[float]
    eligible: bool

class RustEnginePerfModel:
    """Engine-level performance model backed by AIC forward-pass modeling."""

    @staticmethod
    def best_available(
        *,
        engine_args: Optional["MockEngineArgs"] = None,
        aic_config: Optional[AicEngineConfig] = None,
        worker_type: Optional[str] = None,
        limits: Optional[EnginePerfLimits] = None,
        options: Optional[RustEnginePerfOptions] = None,
        bootstrap_fpms: Optional[Any] = None,
    ) -> "RustEnginePerfModel":
        """Build from all available inputs; explicit AIC config is preferred, then engine args, then regression-only."""
        ...

    @staticmethod
    def from_regression(
        *,
        worker_type: str,
        limits: EnginePerfLimits,
        options: Optional[RustEnginePerfOptions] = None,
        bootstrap_fpms: Optional[Any] = None,
    ) -> "RustEnginePerfModel":
        """Build a regression-only model that learns from observed FPM wall times."""
        ...

    @staticmethod
    def from_native(
        *,
        aic_config: AicEngineConfig,
        worker_type: str,
        limits: EnginePerfLimits,
        options: Optional[RustEnginePerfOptions] = None,
        bootstrap_fpms: Optional[Any] = None,
    ) -> "RustEnginePerfModel":
        """Build a strict native AIC model; unsupported AIC configs raise an error."""
        ...

    def estimate_forward_pass_time(self, metrics_by_rank: Any) -> Optional[float]:
        """Estimate one scheduled forward-pass iteration in seconds from current-version FPMs."""
        ...

    def tune_with_fpms(self, iterations: Any) -> None:
        """Tune with current-version observed FPMs: outer list is iterations, inner list is attention-DP ranks."""
        ...

    def diagnostics(self) -> str:
        """Return AIC diagnostics as a JSON string."""
        ...

    def get_min_correction_factor(self) -> Optional[float]:
        """Return the minimum ready native correction factor, or None if no factor is ready."""
        ...

    def get_max_correction_factor(self) -> Optional[float]:
        """Return the maximum ready native correction factor, or None if no factor is ready."""
        ...

    def get_avg_correction_factor(self) -> Optional[float]:
        """Return the average ready native correction factor, or None if no factor is ready."""
        ...

    def get_queued_prefill_time(self, metrics_by_rank: Any) -> Optional[float]:
        """Estimate queued prefill drain time; adjust queued tokens outside the shim for KV reuse."""
        ...

    def get_scheduled_decode_itl(self, metrics_by_rank: Any) -> Optional[float]:
        """Estimate scheduled decode ITL in seconds; aggregated workers include scheduled or learned average prefill load."""
        ...

    def find_engine_capacity_rps(
        self, request: EngineCapacityRequest
    ) -> Optional[EngineCapacity]:
        """Search sustainable per-engine RPS; inspect eligible to see whether eligible SLA metrics passed."""
        ...

class KvRouterConfig:
    """Values for KV router"""

    def __init__(
        self,
        overlap_score_weight: Optional[float] = None,
        host_cache_hit_weight: float = 0.75,
        disk_cache_hit_weight: float = 0.25,
        router_temperature: float = 0.0,
        use_kv_events: bool = True,
        *,
        router_replica_sync: bool = False,
        router_track_active_blocks: bool = True,
        router_track_output_blocks: bool = False,
        router_assume_kv_reuse: bool = True,
        router_track_prefill_tokens: bool = True,
        router_prefill_load_model: str = "none",
        router_ttl_secs: float = 120.0,
        router_queue_threshold: Optional[float] = None,
        router_event_threads: int = 4,
        router_queue_policy: str = "fcfs",
        use_remote_indexer: bool = False,
        serve_indexer: bool = False,
        shared_cache_multiplier: float = 0.0,
        shared_cache_type: str = "none",
        router_predicted_ttl_secs: Optional[float] = None,
        overlap_score_credit: float = 1.0,
        overlap_score_credit_decay: float = 0.0,
        prefill_load_scale: float = 1.0,
        router_policy_config: Optional[str] = None,
        router_tracking_hash: Literal["public-xxh3-v1", "keyed-xxh3-v1"] = "public-xxh3-v1",
        router_tracking_key_file: Optional[str | os.PathLike[str]] = None,
        router_tracking_key_id: Optional[str] = None,
    ) -> None:
        """
        Create a KV router configuration.

        Args:
            overlap_score_weight: Deprecated positional/keyword alias for prefill_load_scale. When present, it takes precedence over prefill_load_scale; a value of 0 also sets overlap_score_credit to 0.
            overlap_score_credit: Finite, non-negative credit multiplier for device-local prefix overlap (default: 1.0). Values above 1.0 give device overlap extra credit and can make adjusted prefill cost negative.
            prefill_load_scale: Scale for adjusted prompt-side prefill load after cache-hit credits (default: 1.0)
            host_cache_hit_weight: Credit multiplier for host-pinned cache hits (default: 0.75)
            disk_cache_hit_weight: Credit multiplier for disk/external cache hits (default: 0.25)
            router_temperature: Temperature for normalized worker sampling via softmax (default: 0.0)
            use_kv_events: Whether to use KV events from workers (default: True)
            router_replica_sync: Enable replica synchronization (default: False)
            router_track_active_blocks: Track active blocks for load balancing (default: True)
            router_track_output_blocks: Track output blocks during generation (default: False).
                When enabled, the router adds placeholder blocks as tokens are generated
                and applies fractional decay based on progress toward expected output
                sequence length (agent_hints.osl in nvext).
            router_assume_kv_reuse: Assume KV cache reuse when tracking active blocks (default: True).
                When True, computes actual block hashes. When False, generates random hashes.
            router_track_prefill_tokens: Include prompt-side prefill tokens in active load accounting (default: True).
            router_tracking_hash: Tracking identity algorithm, "public-xxh3-v1" or
                "keyed-xxh3-v1" (default: "public-xxh3-v1").
            router_tracking_key_file: File containing exactly 32 raw provider-key bytes.
                Required only for keyed tracking mode.
            router_tracking_key_id: Provider-managed key epoch mixed into keyed scope
                derivation. Required only for keyed tracking mode.
            router_prefill_load_model: Prompt-side prefill load model (default: "none").
                "none" keeps static prompt load accounting.
                "aic" decays the oldest active prefill request using AIC-predicted duration.
            router_ttl_secs: TTL for blocks in seconds when not using KV events (default: 120.0)
            router_queue_threshold: Optional queue threshold fraction for prefill token capacity (default: None).
                Requests are queued if all workers exceed this fraction of max_num_batched_tokens.
                Enables priority scheduling via request priority hints.
                Set a numeric value to enable queueing.
            router_policy_config: Startup-only policy-family and cache-bucket queue
                YAML path. When omitted, router_queue_threshold and
                router_queue_policy define one synthetic policy class.
            router_event_threads: Number of KV indexer worker threads (default: 4).
                When > 1, uses a concurrent radix tree with a thread pool,
                including for approximate routing when KV events are disabled.
            router_queue_policy: Scheduling policy for the router queue (default: "fcfs").
                "fcfs": first-come first-served with priority bumps — optimizes tail TTFT.
                "lcfs": last-come first-served with priority bumps — intentionally worsens tail behavior for policy comparisons.
                "wspt": weighted shortest processing time (Smith's rule) — optimizes average TTFT.
            use_remote_indexer: Query a remote KV indexer served from the worker component (default: False).
            serve_indexer: Serve this router's local indexer from the worker component (default: False).
            shared_cache_multiplier: Credit multiplier for shared cache hits beyond the device prefix (default: 0.0).
            shared_cache_type: External shared KV cache type, "none" or "hicache" (default: "none").
            router_predicted_ttl_secs: Enables predict-on-route when set. This TTL
                applies to entries in the local side indexer and requires
                use_kv_events=True. Set to None to disable. Independent of
                router_ttl_secs, which covers pure approximate mode.
        """
        ...

    @staticmethod
    def from_json(config_json: str) -> "KvRouterConfig":
        ...

    def copy(self) -> "KvRouterConfig": ...

    @property
    def overlap_score_credit(self) -> float: ...

    @overlap_score_credit.setter
    def overlap_score_credit(self, value: float) -> None: ...
    @property
    def overlap_score_credit_decay(self) -> float: ...

    @overlap_score_credit_decay.setter
    def overlap_score_credit_decay(self, value: float) -> None: ...
    @property
    def overlap_score_weight(self) -> float: ...

    @overlap_score_weight.setter
    def overlap_score_weight(self, value: float) -> None: ...
    @property
    def prefill_load_scale(self) -> float: ...
    @prefill_load_scale.setter
    def prefill_load_scale(self, value: float) -> None: ...

    def with_overrides(
        self,
        overlap_score_weight: Optional[float] = None,
        *,
        overlap_score_credit: Optional[float] = None,
        overlap_score_credit_decay: Optional[float] = None,
        prefill_load_scale: Optional[float] = None,
    ) -> "KvRouterConfig": ...

class ReasoningConfig:
    def __init__(
        self,
        start_thinking_token_id: int,
        end_thinking_token_id: int,
        thinking_ratio: float,
    ) -> None:
        ...

class SglangArgs:
    def __init__(
        self,
        schedule_policy: Optional[str] = None,
        page_size: Optional[int] = None,
        max_prefill_tokens: Optional[int] = None,
        chunked_prefill_size: Optional[int] = None,
        clip_max_new_tokens: Optional[int] = None,
        schedule_conservativeness: Optional[float] = None,
    ) -> None:
        ...

class TrtllmArgs:
    def __init__(
        self,
        capacity_scheduler_policy: Optional[str] = None,
    ) -> None:
        ...

class MockEngineArgs:
    def __init__(
        self,
        engine_type: str = "vllm",
        num_gpu_blocks: Optional[int] = None,
        block_size: int = 0,
        max_num_seqs: Optional[int] = 256,
        max_num_batched_tokens: Optional[int] = 8192,
        enable_prefix_caching: bool = True,
        enable_chunked_prefill: bool = True,
        speedup_ratio: float = 1.0,
        decode_speedup_ratio: float = 1.0,
        dp_size: int = 1,
        startup_time: Optional[float] = None,
        worker_type: str = "aggregated",
        planner_profile_data: Optional[str | os.PathLike[str]] = None,
        aic_backend: Optional[str] = None,
        aic_system: Optional[str] = None,
        aic_backend_version: Optional[str] = None,
        aic_tp_size: Optional[int] = None,
        aic_model_path: Optional[str] = None,
        aic_moe_tp_size: Optional[int] = None,
        aic_moe_ep_size: Optional[int] = None,
        aic_attention_dp_size: Optional[int] = None,
        aic_nextn: Optional[int] = None,
        aic_nextn_accept_rates: Optional[str] = None,
        aic_mtp_seed: int = 42,
        aic_gemm_dtype: Optional[str] = None,
        aic_moe_dtype: Optional[str] = None,
        aic_fmha_dtype: Optional[str] = None,
        aic_kv_cache_dtype: Optional[str] = None,
        aic_comm_dtype: Optional[str] = None,
        gpu_memory_utilization: Optional[float] = None,
        mem_fraction_static: Optional[float] = None,
        free_gpu_memory_fraction: Optional[float] = None,
        enable_local_indexer: bool = False,
        bootstrap_port: Optional[int] = None,
        handoff_session_timeout_ms: int = 300000,
        kv_bytes_per_token: Optional[int] = None,
        kv_transfer_bandwidth: Optional[float] = None,
        kv_transfer_timing_mode: str = "full_prompt",
        reasoning: Optional[ReasoningConfig] = None,
        response_replay_trace_path: Optional[str | os.PathLike[str]] = None,
        zmq_kv_events_port: Optional[int] = None,
        zmq_replay_port: Optional[int] = None,
        preemption_mode: str = "lifo",
        router_queue_policy: Optional[str] = None,
        sglang: Optional[SglangArgs] = None,
        trtllm: Optional[TrtllmArgs] = None,
        num_g2_blocks: Optional[int] = None,
        num_g3_blocks: Optional[int] = None,
        offload_batch_size: Optional[int] = None,
        bandwidth_g1_to_g2_gbps: Optional[float] = None,
        bandwidth_g2_to_g1_gbps: Optional[float] = None,
        bandwidth_g2_to_g3_gbps: Optional[float] = None,
        bandwidth_g3_to_g2_gbps: Optional[float] = None,
        enable_g4_storage: bool = False,
        bandwidth_g2_to_g4_gbps: Optional[float] = None,
        bandwidth_g4_to_g2_gbps: Optional[float] = None,
        max_model_len: Optional[int] = None,
        g1_backend: str = "kvbm",
    ) -> None:
        ...

    @staticmethod
    def from_json(config_json: str) -> "MockEngineArgs":
        ...

    def copy(self) -> "MockEngineArgs": ...

    @property
    def block_size(self) -> int: ...

    @property
    def num_gpu_blocks(self) -> int: ...

    @num_gpu_blocks.setter
    def num_gpu_blocks(self, value: int) -> None: ...

    @property
    def max_model_len(self) -> Optional[int]: ...

    @property
    def max_num_seqs(self) -> Optional[int]: ...

    @property
    def max_num_batched_tokens(self) -> Optional[int]: ...

    @property
    def enable_prefix_caching(self) -> bool: ...

    @enable_prefix_caching.setter
    def enable_prefix_caching(self, value: bool) -> None: ...

    @property
    def g1_backend(self) -> str: ...

    @property
    def enable_local_indexer(self) -> bool: ...

    @property
    def dp_size(self) -> int: ...

    @property
    def bootstrap_port(self) -> Optional[int]: ...

    @property
    def handoff_session_timeout_ms(self) -> int: ...

    @property
    def kv_transfer_timing_mode(self) -> str: ...

    @property
    def engine_type(self) -> str: ...

    @property
    def response_replay_trace_path(self) -> Optional[os.PathLike[str]]: ...

    @property
    def num_g2_blocks(self) -> Optional[int]: ...

    @property
    def num_g3_blocks(self) -> Optional[int]: ...

    @property
    def offload_batch_size(self) -> Optional[int]: ...

    @property
    def bandwidth_g1_to_g2_gbps(self) -> Optional[float]: ...

    @property
    def bandwidth_g2_to_g1_gbps(self) -> Optional[float]: ...

    @property
    def bandwidth_g2_to_g3_gbps(self) -> Optional[float]: ...

    @property
    def bandwidth_g3_to_g2_gbps(self) -> Optional[float]: ...

    @property
    def enable_g4_storage(self) -> bool: ...

    @property
    def bandwidth_g2_to_g4_gbps(self) -> Optional[float]: ...

    @property
    def bandwidth_g4_to_g2_gbps(self) -> Optional[float]: ...

    @property
    def aic_backend(self) -> Optional[str]: ...

    @aic_backend.setter
    def aic_backend(self, value: Optional[str]) -> None: ...

    @property
    def aic_system(self) -> Optional[str]: ...

    @aic_system.setter
    def aic_system(self, value: Optional[str]) -> None: ...

    @property
    def aic_backend_version(self) -> Optional[str]: ...

    @aic_backend_version.setter
    def aic_backend_version(self, value: Optional[str]) -> None: ...

    @property
    def aic_tp_size(self) -> Optional[int]: ...

    @aic_tp_size.setter
    def aic_tp_size(self, value: Optional[int]) -> None: ...

    @property
    def aic_model_path(self) -> Optional[str]: ...

    @aic_model_path.setter
    def aic_model_path(self, value: Optional[str]) -> None: ...

    @property
    def aic_moe_tp_size(self) -> Optional[int]: ...

    @aic_moe_tp_size.setter
    def aic_moe_tp_size(self, value: Optional[int]) -> None: ...

    @property
    def aic_moe_ep_size(self) -> Optional[int]: ...

    @aic_moe_ep_size.setter
    def aic_moe_ep_size(self, value: Optional[int]) -> None: ...

    @property
    def aic_attention_dp_size(self) -> Optional[int]: ...

    @aic_attention_dp_size.setter
    def aic_attention_dp_size(self, value: Optional[int]) -> None: ...

    @property
    def aic_gemm_dtype(self) -> Optional[str]: ...

    @aic_gemm_dtype.setter
    def aic_gemm_dtype(self, value: Optional[str]) -> None: ...

    @property
    def aic_moe_dtype(self) -> Optional[str]: ...

    @aic_moe_dtype.setter
    def aic_moe_dtype(self, value: Optional[str]) -> None: ...

    @property
    def aic_fmha_dtype(self) -> Optional[str]: ...

    @aic_fmha_dtype.setter
    def aic_fmha_dtype(self, value: Optional[str]) -> None: ...

    @property
    def aic_kv_cache_dtype(self) -> Optional[str]: ...

    @aic_kv_cache_dtype.setter
    def aic_kv_cache_dtype(self, value: Optional[str]) -> None: ...

    @property
    def aic_comm_dtype(self) -> Optional[str]: ...

    @aic_comm_dtype.setter
    def aic_comm_dtype(self, value: Optional[str]) -> None: ...

    @property
    def aic_nextn(self) -> Optional[int]: ...

    @aic_nextn.setter
    def aic_nextn(self, value: Optional[int]) -> None: ...

    @property
    def aic_nextn_accept_rates(self) -> Optional[str]: ...

    @aic_nextn_accept_rates.setter
    def aic_nextn_accept_rates(self, value: Optional[str]) -> None: ...

    @property
    def aic_mtp_seed(self) -> int: ...

    @aic_mtp_seed.setter
    def aic_mtp_seed(self, value: int) -> None: ...

    @property
    def gpu_memory_utilization(self) -> Optional[float]: ...

    @gpu_memory_utilization.setter
    def gpu_memory_utilization(self, value: Optional[float]) -> None: ...

    @property
    def mem_fraction_static(self) -> Optional[float]: ...

    @mem_fraction_static.setter
    def mem_fraction_static(self, value: Optional[float]) -> None: ...

    @property
    def free_gpu_memory_fraction(self) -> Optional[float]: ...

    @free_gpu_memory_fraction.setter
    def free_gpu_memory_fraction(self, value: Optional[float]) -> None: ...

    @property
    def worker_type(self) -> str: ...

    @worker_type.setter
    def worker_type(self, value: str) -> None: ...

    def is_prefill(self) -> bool: ...

    def is_decode(self) -> bool: ...

    def with_overrides(
        self,
        bootstrap_port: Optional[int] = None,
        zmq_kv_events_port: Optional[int] = None,
        zmq_replay_port: Optional[int] = None,
        kv_bytes_per_token: Optional[int] = None,
        num_gpu_blocks: Optional[int] = None,
        aic_backend: Optional[str] = None,
        aic_system: Optional[str] = None,
        aic_backend_version: Optional[str] = None,
        aic_tp_size: Optional[int] = None,
        aic_model_path: Optional[str] = None,
        aic_moe_tp_size: Optional[int] = None,
        aic_moe_ep_size: Optional[int] = None,
        aic_attention_dp_size: Optional[int] = None,
        aic_nextn: Optional[int] = None,
        aic_nextn_accept_rates: Optional[str] = None,
        aic_mtp_seed: Optional[int] = None,
        aic_gemm_dtype: Optional[str] = None,
        aic_moe_dtype: Optional[str] = None,
        aic_fmha_dtype: Optional[str] = None,
        aic_kv_cache_dtype: Optional[str] = None,
        aic_comm_dtype: Optional[str] = None,
        gpu_memory_utilization: Optional[float] = None,
        mem_fraction_static: Optional[float] = None,
        free_gpu_memory_fraction: Optional[float] = None,
        enable_prefix_caching: Optional[bool] = None,
        worker_type: Optional[str] = None,
    ) -> "MockEngineArgs": ...

class WorkerType:
    """
    Processing stage a worker handles.

    Each worker has exactly one role; values are not combinable. Use the
    `needs` argument on register_model to express dependencies in DNF form
    (a list of alternative AND-sets) — for example, an encode worker that
    needs (Prefill AND Decode) OR a single Aggregated peer is expressed as
    `[[WorkerType.Prefill, WorkerType.Decode], [WorkerType.Aggregated]]`.
    """

    Prefill: "WorkerType"
    Decode: "WorkerType"
    Encode: "WorkerType"
    Aggregated: "WorkerType"

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

async def register_model(
    model_input: ModelInput,
    model_type: ModelType,
    endpoint: Endpoint,
    model_path: str,
    model_name: Optional[str] = None,
    *,
    worker_type: WorkerType,
    kv_cache_block_size: Optional[int] = None,
    router_mode: Optional[RouterMode] = None,
    runtime_config: Optional[ModelRuntimeConfig] = None,
    tensor_model_config: Optional[Dict[str, Any]] = None,
    user_data: Optional[Dict[str, Any]] = None,
    custom_template_path: Optional[str] = None,
    media_decoder: Optional[MediaDecoder] = None,
    media_fetcher: Optional[MediaFetcher] = None,
    lora_name: Optional[str] = None,
    base_model_path: Optional[str] = None,
    needs: Optional[List[List[WorkerType]]] = None,
    self_host_metadata: Optional[bool] = None,
    ignore_weights: bool = False,
    max_gpu_lora_count: Optional[int] = None,
    model_aliases: Optional[List[str]] = None,
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
    for these models. Pass tensor protocol metadata through `tensor_model_config`.

    Model serving readiness:
        `worker_type` and `needs` describe the worker's processing stage and
        peer dependencies. `needs` is a DNF list — each inner list is an
        AND-set, the outer list is OR. `worker_type` is required; backends
        declare it literally at each call site.

    When `ignore_weights` is true, remote HuggingFace model resolution skips
    weight files and downloads only the metadata needed for registration.
    """
    ...

async def unregister_model(
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

def resolve_routing_image_token_id(model_id: str, model_dir: str) -> Optional[int]:
    """Routing-side image-placeholder token id for a model, resolved with the
    same per-family logic the frontend's MM-aware KV routing uses. Returns None
    when the model isn't in the MM-routing registry or its config can't be read.
    Only present when the bindings are built with the ``mm-routing`` feature.
    """
    ...

class LoRADownloader:
    """Unified interface for LoRA downloading and caching (local file:// and S3 s3:// URIs)."""

    def __init__(self, cache_path: Optional[str] = None) -> None: ...
    def download_if_needed(self, lora_uri: str) -> Awaitable[str]: ...
    def get_cache_path(self, cache_key: str) -> str: ...
    def is_cached(self, lora_uri: str) -> bool: ...
    def validate_cached(self, cache_key: str) -> bool: ...

    @staticmethod
    def uri_to_cache_key(uri: str) -> str: ...


class MediaDecoder:
    """Media decoder for image and video preprocessing."""

    def __init__(self) -> None: ...
    def enable_image(self, decoder_options: Dict[str, Any]) -> None: ...


class MediaFetcher:
    """Media fetcher for loading remote image/video URLs."""

    def __init__(self) -> None: ...
    def user_agent(self, user_agent: str) -> None: ...
    def allow_direct_ip(self, allow: bool) -> None: ...
    def allow_direct_port(self, allow: bool) -> None: ...
    def allowed_media_domains(self, domains: List[str]) -> None: ...
    def timeout_ms(self, timeout_ms: int) -> None: ...

async def fetch_model(remote_name: str, ignore_weights: bool = False) -> str:
    """
    Download a model from Hugging Face, returning its local path.
    If `ignore_weights` is True, only fetches tokenizer and config files.
    Example: `model_path = await fetch_model("Qwen/Qwen3-0.6B")`
    """
    ...

# Backward-compatible aliases (deprecated, use new names)
fetch_llm = fetch_model
register_llm = register_model
unregister_llm = unregister_model

class EngineConfig:
    """Holds internal configuration for a Dynamo engine."""
    ...

async def make_engine(distributed_runtime: DistributedRuntime, args: EntrypointArgs) -> EngineConfig:
    """Make an engine matching the args"""
    ...

class FrontendExtensionContext:
    """Read-only, live view of frontend state passed to extension route handlers.

    Handlers receive this and answer from current state. The surface is
    intentionally narrow (typed read-only accessors only); it does not expose
    the internal service state.
    """

    def is_ready(self) -> bool:
        """Whether the HTTP service has finished startup and is ready to serve."""
        ...

    def is_cancelled(self) -> bool:
        """Whether the frontend is shutting down (draining)."""
        ...

    def has_any_ready_model(self) -> bool:
        """Whether at least one model is registered and ready to serve."""
        ...

    def is_model_ready_to_serve(self, model: str) -> bool:
        """Whether the named model is registered and ready to serve."""
        ...

    def model_display_names(self) -> list[str]:
        """Sorted display names of all registered models."""
        ...

    def serving_ready_display_names(self) -> list[str]:
        """Sorted display names of models ready to serve."""
        ...

class FrontendRoute:
    """A trusted extension route served on the Dynamo HTTP frontend.

    Currently restricted to static-path ``GET`` routes. ``handler`` is a
    synchronous callable that receives a ``FrontendExtensionContext`` and
    returns a JSON-serializable body (implies HTTP 200) or a ``FrontendResponse``
    to set the status code. Async handlers and path parameters are rejected at
    construction.
    """

    def __init__(
        self,
        method: str,
        path: str,
        handler: Callable[[FrontendExtensionContext], object],
    ) -> None: ...
    @property
    def method(self) -> str: ...
    @property
    def path(self) -> str: ...

class FrontendResponse:
    """Explicit status-code override returned by a ``FrontendRoute`` handler.

    Return this to set a non-200 status (e.g. ``FrontendResponse(503, body)``);
    return a plain JSON-serializable value for the default 200.
    """

    def __init__(self, status_code: int, body: object) -> None: ...

async def run_input(
    distributed_runtime: DistributedRuntime,
    input: str,
    engine_config: EngineConfig,
    frontend_route_extensions: Optional[Sequence[FrontendRoute]] = None,
) -> None:
    """Start an engine, connect it to an input, and run until stopped.

    ``frontend_route_extensions`` supplies additional HTTP routes to the
    frontend (HTTP input only); see ``FrontendRoute``.
    """
    ...

def run_mocker_trace_replay(
    trace_files: Sequence[str | os.PathLike[str]],
    extra_engine_args: Optional[MockEngineArgs] = None,
    prefill_engine_args: Optional[MockEngineArgs] = None,
    decode_engine_args: Optional[MockEngineArgs] = None,
    router_config: Optional[KvRouterConfig] = None,
    aic_perf_config: Optional[AicPerfConfig] = None,
    num_workers: int = 1,
    num_prefill_workers: int = 1,
    num_decode_workers: int = 1,
    replay_concurrency: Optional[int] = None,
    replay_mode: Literal["offline", "online"] = "offline",
    router_mode: Literal["round_robin", "kv_router"] = "round_robin",
    arrival_speedup_ratio: float = 1.0,
    trace_block_size: Optional[int] = None,
    trace_format: Literal[
        "mooncake",
        "mooncake-delta",
        "mooncake_delta",
        "agentic_mooncake",
        "agentic-mooncake",
        "applied_compute_agentic",
        "dynamo",
    ] = "mooncake",
    trace_shared_prefix_ratio: float = 0.0,
    trace_num_prefix_groups: int = 0,
    report_jsonl_path: Optional[str | os.PathLike[str]] = None,
    max_sim_time_ms: Optional[float] = None,
    model_name: Optional[str] = None,
    sla_ttft_ms: Optional[float] = None,
    sla_itl_ms: Optional[float] = None,
    sla_e2e_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Replay mocker trace files and return the simulation report.

    Supports aggregated or disaggregated engine configurations.

    When ``report_jsonl_path`` is provided (offline disagg replay only), one
    JSON object per request is written to that path. Each line includes
    arrival/admit/token timestamps, input/output lengths, the full per-token
    ITL series, and prefill/decode worker indices.

    ``sla_ttft_ms`` / ``sla_itl_ms`` / ``sla_e2e_ms`` are the goodput SLA bounds
    (offline replay only). When any is set, the report carries ``goodput_*`` keys
    classifying SLA-satisfying requests; with none set, goodput is omitted.
    """
    ...

def run_mocker_synthetic_trace_replay(
    input_tokens: int,
    output_tokens: int,
    request_count: int,
    extra_engine_args: Optional[MockEngineArgs] = None,
    prefill_engine_args: Optional[MockEngineArgs] = None,
    decode_engine_args: Optional[MockEngineArgs] = None,
    router_config: Optional[KvRouterConfig] = None,
    aic_perf_config: Optional[AicPerfConfig] = None,
    num_workers: int = 1,
    num_prefill_workers: int = 1,
    num_decode_workers: int = 1,
    replay_concurrency: Optional[int] = None,
    replay_mode: Literal["offline", "online"] = "offline",
    router_mode: Literal["round_robin", "kv_router"] = "round_robin",
    arrival_speedup_ratio: float = 1.0,
    request_rate: Optional[float] = None,
    arrival_interval_ms: Optional[float] = None,
    arrival_seed: int = 42,
    turns_per_session: int = 1,
    shared_prefix_ratio: float = 0.0,
    num_prefix_groups: int = 0,
    inter_turn_delay_ms: float = 0.0,
    model_name: Optional[str] = None,
    sla_ttft_ms: Optional[float] = None,
    sla_itl_ms: Optional[float] = None,
    sla_e2e_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Replay a synthetic mocker workload without requiring a trace file.

    ``sla_ttft_ms`` / ``sla_itl_ms`` / ``sla_e2e_ms`` are the goodput SLA bounds
    (offline replay only); when any is set the report carries ``goodput_*`` keys
    classifying SLA-satisfying requests.
    """
    ...

class PlannerReplayBridge:
    """Drives an offline replay to completion with a Python planner. The Rust
    simulation owns the drive loop and calls back into ``planner`` once per
    ``PlannerTick`` via ``run(planner)`` (``planner`` exposes
    ``initial_tick_ms() -> float`` and ``on_tick(metrics: dict) -> dict``)."""

    def __init__(
        self,
        trace_file: str | os.PathLike[str],
        extra_engine_args: MockEngineArgs,
        num_workers: int,
        router_mode: str = "round_robin",
        router_config: Optional[KvRouterConfig] = None,
        model_name: Optional[str] = None,
        arrival_speedup_ratio: float = 1.0,
        trace_block_size: int = 512,
        sla_ttft_ms: Optional[float] = None,
        sla_itl_ms: Optional[float] = None,
        sla_e2e_ms: Optional[float] = None,
        replay_concurrency: Optional[int] = None,
    ) -> None: ...

    @staticmethod
    def create_disagg(
        trace_file: str | os.PathLike[str],
        prefill_engine_args: MockEngineArgs,
        decode_engine_args: MockEngineArgs,
        num_prefill_workers: int,
        num_decode_workers: int,
        router_mode: str = "round_robin",
        router_config: Optional[KvRouterConfig] = None,
        model_name: Optional[str] = None,
        arrival_speedup_ratio: float = 1.0,
        trace_block_size: int = 512,
        sla_ttft_ms: Optional[float] = None,
        sla_itl_ms: Optional[float] = None,
        sla_e2e_ms: Optional[float] = None,
        replay_concurrency: Optional[int] = None,
    ) -> "PlannerReplayBridge": ...

    @staticmethod
    def from_synthetic(
        input_tokens: int,
        output_tokens: int,
        request_count: int,
        extra_engine_args: MockEngineArgs,
        num_workers: int,
        router_mode: str = "round_robin",
        router_config: Optional[KvRouterConfig] = None,
        model_name: Optional[str] = None,
        replay_concurrency: Optional[int] = None,
        arrival_speedup_ratio: float = 1.0,
        request_rate: Optional[float] = None,
        arrival_interval_ms: Optional[float] = None,
        arrival_seed: int = 42,
        turns_per_session: int = 1,
        shared_prefix_ratio: float = 0.0,
        num_prefix_groups: int = 0,
        inter_turn_delay_ms: float = 0.0,
        sla_ttft_ms: Optional[float] = None,
        sla_itl_ms: Optional[float] = None,
        sla_e2e_ms: Optional[float] = None,
    ) -> "PlannerReplayBridge": ...

    @staticmethod
    def from_synthetic_disagg(
        input_tokens: int,
        output_tokens: int,
        request_count: int,
        prefill_engine_args: MockEngineArgs,
        decode_engine_args: MockEngineArgs,
        num_prefill_workers: int,
        num_decode_workers: int,
        router_mode: str = "round_robin",
        router_config: Optional[KvRouterConfig] = None,
        model_name: Optional[str] = None,
        replay_concurrency: Optional[int] = None,
        arrival_speedup_ratio: float = 1.0,
        request_rate: Optional[float] = None,
        arrival_interval_ms: Optional[float] = None,
        arrival_seed: int = 42,
        turns_per_session: int = 1,
        shared_prefix_ratio: float = 0.0,
        num_prefix_groups: int = 0,
        inter_turn_delay_ms: float = 0.0,
        sla_ttft_ms: Optional[float] = None,
        sla_itl_ms: Optional[float] = None,
        sla_e2e_ms: Optional[float] = None,
    ) -> "PlannerReplayBridge": ...

    def run(self, planner: Any) -> Dict[str, Any]: ...

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

class KvDcRelay:
    def __init__(
        self,
        endpoint: Endpoint,
        dc_id: str,
        namespace_filter: Optional[str] = None,
        endpoint_prefix: Optional[str] = None,
        publication_threshold: int = 16,
        publication_delay_ms: int = 1,
        recovery_attempt_timeout_ms: int = 30_000,
    ) -> None:
        ...

    async def start(self) -> None:
        ...

    async def health(self) -> Dict[str, Any]:
        ...

    async def flush(self) -> None:
        ...

    async def shutdown(self) -> None:
        ...

class KvRouter:
    """
    A KV-aware router that performs intelligent routing based on KV cache overlap.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        block_size: int,
        kv_router_config: KvRouterConfig,
        aic_perf_config: Optional[AicPerfConfig] = None,
    ) -> None:
        """
        Create a new KvRouter instance.

        Args:
            endpoint: The endpoint to connect to for routing requests
            block_size: The KV cache block size
            kv_router_config: Configuration for the KV router
            aic_perf_config: Optional AIC perf-model config for effective prefill load tracking
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
        extra_args: Optional[JsonLike] = None,
        block_mm_infos: Optional[List[Optional[Dict[str, Any]]]] = None,
        multi_modal_data: Optional[JsonLike] = None,
        mm_routing_info: Optional[JsonLike] = None,
        routing_constraints: Optional[RoutingConstraints] = None,
        response_buffer_size: int = 100,
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
            extra_args: Optional extra request arguments to include in the
                       PreprocessedRequest.
            block_mm_infos: Optional block-level multimodal metadata aligned to
                           request blocks. Backward-compatible shortcut; this is
                           converted to mm_routing_info with routing_token_ids=token_ids.
            multi_modal_data: Optional multimodal payload map to preserve image/video
                             data for downstream model execution.
            mm_routing_info: Optional structured routing-only multimodal payload
                            (e.g., {"routing_token_ids": [...], "block_mm_infos": [...]})
                            used by router selection without changing execution token_ids.
            routing_constraints: Optional request routing constraints used to constrain or prefer tainted workers.
            response_buffer_size: Maximum number of responses buffered by the Python
                                  adapter. Set to 0 for demand-driven direct Python
                                  consumption; negative values are rejected.

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

    async def generate_from_request(
        self,
        request: JsonLike,
        response_buffer_size: int = 100,
    ) -> AsyncIterator[JsonLike]:
        """
        Generate from a preprocessed request dict (PreprocessedRequest format).

        Accepts a full request dict with token_ids, model, stop_conditions, etc.
        Set response_buffer_size to 0 for demand-driven direct Python consumption;
        negative values are rejected.
        Returns an async iterator yielding generation responses.
        """
        ...

    async def best_worker(
        self,
        token_ids: List[int],
        router_config_override: Optional[JsonLike] = None,
        request_id: Optional[str] = None,
        update_indexer: bool = False,
        block_mm_infos: Optional[List[Optional[Dict[str, Any]]]] = None,
        lora_name: Optional[str] = None,
        routing_constraints: Optional[RoutingConstraints] = None,
        strict_priority: int = 0,
        policy_class: Optional[str] = None,
        cache_namespace: Optional[str] = None,
    ) -> Tuple[int, int, int]:
        """
        Find the best matching worker for the given tokens.

        Args:
            token_ids: List of token IDs to find matches for
            router_config_override: Optional router configuration override
            request_id: Optional request ID. If provided, router states will be updated
                       to track this request (active blocks, lifecycle events). If not
                       provided, this is a query-only operation that doesn't affect state.
            update_indexer: Whether to record the selected worker in the router's
                           approximate indexer. This is only meaningful when
                           `use_kv_events=False` and is independent from lifecycle
                           state tracking via `request_id`.
            block_mm_infos: Optional block-level multimodal metadata aligned to request
                           blocks. When provided, this is used in block hash computation
                           to enable MM-aware worker selection.
            cache_namespace: Optional cache namespace used in block hash computation.
            policy_class: Requested policy family, or an exact explicit class.
                          Missing, unknown, and ordinary physical-class names use the
                          configured default family before cache-bucket resolution.

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
        block_mm_infos: Optional[List[Optional[Dict[str, Any]]]] = None,
        lora_name: Optional[str] = None,
        cache_namespace: Optional[str] = None,
    ) -> List[Dict[str, int]]:
        """
        Get potential prefill and decode loads for all workers.

        Args:
            token_ids: List of token IDs to evaluate
            block_mm_infos: Optional block-level multimodal metadata aligned to request
                           blocks. When provided, this is used in hash computation
                           for MM-aware potential-load estimation.
            lora_name: Optional LoRA adapter name used in block hash computation.

        Returns:
            A list of dictionaries, each containing:
                - worker_id: The worker ID
                - dp_rank: The data parallel rank
                - potential_prefill_tokens: Number of tokens that would need prefill
                - potential_decode_blocks: Number of blocks currently in decode phase
                - active_requests: Number of active requests tracked on the worker

        Note:
            Each (worker_id, dp_rank) pair is returned as a separate entry.
            If you need aggregated loads per worker_id, sum the values manually.
        """
        ...

    async def get_overlap_scores(
        self,
        token_ids: List[int],
        router_config_override: Optional[JsonLike] = None,
        block_mm_infos: Optional[List[Optional[Dict[str, Any]]]] = None,
        lora_name: Optional[str] = None,
        include_shared: bool = True,
        cache_namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get per-worker KV overlap by storage tier.

        Args:
            token_ids: List of token IDs to evaluate.
            router_config_override: Optional router configuration override for
                                   score-credit fields.
            block_mm_infos: Optional block-level multimodal metadata aligned to
                           request blocks.
            lora_name: Optional LoRA adapter name for adapter-aware matching.
            include_shared: Whether to query the configured shared cache.

        Returns:
            A dictionary containing block_size, num_blocks, shared_cache, and
            workers. Each worker row is keyed by worker_id and dp_rank and
            reports device, host-pinned, disk, and shared-cache overlap blocks.
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

class EngineType:
    """Engine type for Dynamo workers"""
    Echo: "EngineType"
    Dynamic: "EngineType"
    Mocker: "EngineType"
    ...

class EntrypointArgs:
    """
    Settings to connect an input to a worker and run them.
    Use by `dynamo run`.
    """

    def __init__(
        self,
        engine_type: "EngineType",
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        template_file: Optional[str] = None,
        router_config: Optional[RouterConfig] = None,
        kv_cache_block_size: Optional[int] = None,
        http_host: Optional[str] = None,
        http_port: Optional[int] = None,
        http_metrics_port: Optional[int] = None,
        tls_cert_path: Optional[str] = None,
        tls_key_path: Optional[str] = None,
        extra_engine_args: Optional[str] = None,
        mocker_engine_args: Optional[MockEngineArgs] = None,
        runtime_config: Optional[ModelRuntimeConfig] = None,
        namespace: Optional[str] = None,
        namespace_prefix: Optional[str] = None,
        is_prefill: bool = False,
        is_decode: bool = False,
        migration_limit: int = 0,
        migration_max_seq_len: Optional[int] = None,
        chat_engine_factory: Optional[Callable] = None,
        aic_perf_config: Optional[AicPerfConfig] = None,
        *,
        metrics_prefix: Optional[str] = None,
        enable_anthropic_api: Optional[bool] = None,
        strip_anthropic_preamble: Optional[bool] = None,
        enable_streaming_tool_dispatch: Optional[bool] = None,
        enable_streaming_reasoning_dispatch: Optional[bool] = None,
        tokenizer_backend: Optional[str] = None,
    ) -> None:
        """
        Create EntrypointArgs.

        Args:
            engine_type: The type of engine to use
            model_path: Path to the model directory on disk
            model_name: Model name or dynamo endpoint (e.g. 'dyn://namespace.component.endpoint')
            endpoint_id: Optional endpoint ID
            template_file: Optional path to a prompt template file
            router_config: Optional router configuration
            kv_cache_block_size: Optional KV cache block size
            http_host: HTTP host to bind to
            http_port: HTTP port to bind to
            http_metrics_port: HTTP metrics port (for gRPC service)
            tls_cert_path: TLS certificate path (PEM format)
            tls_key_path: TLS key path (PEM format)
            extra_engine_args: Optional path to mocker engine arguments JSON
            mocker_engine_args: Typed mocker engine arguments
            runtime_config: Optional runtime configuration for discovery registration
            namespace: Dynamo namespace for model discovery scoping
            namespace_prefix: Optional namespace prefix
            is_prefill: Whether this is a prefill worker
            is_decode: Whether this is a decode worker (disaggregated); pairs with a prefill peer for readiness
            migration_limit: Maximum number of request migrations (0=disabled)
            migration_max_seq_len: Optional max sequence length for migration
            chat_engine_factory: Optional Python chat completions engine factory callback
            aic_perf_config: Optional AIC perf-model configuration for default KV routing
            metrics_prefix: Optional Prometheus metrics prefix override
            enable_anthropic_api: Optional Anthropic Messages API override
            strip_anthropic_preamble: Optional Anthropic preamble stripping override
            enable_streaming_tool_dispatch: Optional streaming tool dispatch override
            enable_streaming_reasoning_dispatch: Optional streaming reasoning dispatch override
            tokenizer_backend: Optional tokenizer backend override ("default" or "fastokens")
        """
        ...

class PlannerDecision:
    """A request from planner to client to perform a scaling action.
    Fields: num_prefill_workers, num_decode_workers, decision_id.
            -1 in any of those fields mean not set, usually because planner hasn't decided anything yet.
    Call VirtualConnectorClient.complete(event) when action is completed.
    """
    num_prefill_workers: int
    num_decode_workers: int
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

    async def is_scaling_ready(self) -> bool:
        """Return whether the client acknowledged the current scaling decision."""
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


# =============================================================================
# Dynamo Exception Types
#
# Standardized exceptions for Dynamo error categories. All inherit from
# DynamoException. The Rust error type mapping depends on the context in
# which the exception is raised (e.g., backend context wraps as Backend.<*>).
# =============================================================================

class DynamoException(Exception):
    """Base exception for all Dynamo error types."""

    ...

class RouterQueueLimitExceeded(DynamoException):
    """A policy-class queue cap rejected the request."""

    policy_class: str
    limit_kind: str
    current: int
    limit: int

class Unknown(DynamoException):
    """Uncategorized or unknown error."""

    ...

class InvalidArgument(DynamoException):
    """Invalid input (e.g., prompt exceeds context length)."""

    ...

class CannotConnect(DynamoException):
    """Failed to establish a connection."""

    ...

class Disconnected(DynamoException):
    """An established connection was lost."""

    ...

class ConnectionTimeout(DynamoException):
    """A connection or request timed out."""

    ...

class Cancelled(DynamoException):
    """The request was cancelled."""

    ...

class EngineShutdown(DynamoException):
    """The engine process has shut down or crashed."""

    ...

class StreamIncomplete(DynamoException):
    """The response stream was terminated before completion."""

    ...

class SelectionServiceError(DynamoException):
    """
    Raised by `SelectionService` for selector failures that are not malformed
    input.
    """

    # Stable, machine-readable error category, e.g. "not_ready".
    kind: str
    # HTTP-style status code for the failure, e.g. 503.
    status_code: int

# ---------------------------------------------------------------------------
# `dynamo._core.backend` submodule.
#
# Registered at import time by the pyo3 bindings via `sys.modules`, so it has
# no filesystem layout that mypy can discover. Declaring it as a typed
# namespace class lets `from dynamo._core import backend as _backend` resolve
# and preserves attribute typing for the pyclasses it exposes.
# ---------------------------------------------------------------------------

class backend:
    @staticmethod
    def _run_sglang_sidecar(argv: Optional[List[str]] = None) -> None:
        """Run the native SGLang sidecar with CLI-style arguments."""
        ...

    class DisaggregationMode:
        # Mirrors `dynamo_backend_common::DisaggregationMode`. Engines consult
        # this on the WorkerConfig to switch their per-mode protocol behavior;
        # the Rust Worker reads it for registration (Prefill → ModelType.Empty
        # + WorkerType.Prefill, Decode → disable local indexer).
        Aggregated: "backend.DisaggregationMode"
        Prefill: "backend.DisaggregationMode"
        Decode: "backend.DisaggregationMode"
        Encode: "backend.DisaggregationMode"

    class LlmRegistration:
        def __init__(
            self,
            context_length: Optional[int] = None,
            kv_cache_block_size: Optional[int] = None,
            total_kv_blocks: Optional[int] = None,
            max_num_seqs: Optional[int] = None,
            max_num_batched_tokens: Optional[int] = None,
            data_parallel_size: Optional[int] = None,
            data_parallel_start_rank: Optional[int] = None,
            bootstrap_host: Optional[str] = None,
            bootstrap_port: Optional[int] = None,
        ) -> None: ...
        @property
        def context_length(self) -> Optional[int]: ...
        @property
        def kv_cache_block_size(self) -> Optional[int]: ...
        @property
        def total_kv_blocks(self) -> Optional[int]: ...
        @property
        def max_num_seqs(self) -> Optional[int]: ...
        @property
        def max_num_batched_tokens(self) -> Optional[int]: ...
        @property
        def data_parallel_size(self) -> Optional[int]: ...
        @property
        def data_parallel_start_rank(self) -> Optional[int]: ...
        @property
        def bootstrap_host(self) -> Optional[str]: ...
        @property
        def bootstrap_port(self) -> Optional[int]: ...

    class EngineConfig:
        def __init__(
            self,
            model: str,
            served_model_name: Optional[str] = None,
            runtime_data: Optional[Dict[str, Any]] = None,
            llm: Optional["backend.LlmRegistration"] = None,
        ) -> None: ...
        @property
        def model(self) -> str: ...
        @property
        def served_model_name(self) -> Optional[str]: ...
        @property
        def runtime_data(self) -> Dict[str, Any]: ...
        @property
        def llm(self) -> Optional["backend.LlmRegistration"]: ...

    class RuntimeConfig:
        def __init__(
            self,
            discovery_backend: Optional[str] = None,
            request_plane: Optional[str] = None,
            event_plane: Optional[str] = None,
        ) -> None: ...

    class WorkerConfig:
        def __init__(
            self,
            namespace: str,
            component: str = ...,
            endpoint: str = ...,
            model_name: str = ...,
            served_model_name: Optional[str] = None,
            model_input: ModelInput = ...,
            endpoint_types: str = ...,
            custom_jinja_template: Optional[str] = None,
            tool_call_parser: Optional[str] = None,
            reasoning_parser: Optional[str] = None,
            exclude_tools_when_tool_choice_none: bool = ...,
            enable_local_indexer: bool = ...,
            enable_kv_routing: bool = ...,
            metrics_labels: List[Tuple[str, str]] = ...,
            runtime: Optional["backend.RuntimeConfig"] = None,
            disaggregation_mode: "backend.DisaggregationMode" = ...,
            health_check_payload: Optional[Dict[str, Any]] = None,
            structural_tag_mode: str = ...,
            structural_tag_scope: str = ...,
            structural_tag_schema: str = ...,
            route_to_encoder: bool = ...,
            media_decoder: Optional[MediaDecoder] = None,
            media_fetcher: Optional[MediaFetcher] = None,
            kv_state_endpoint: Optional[str] = None,
        ) -> None: ...

    class Worker:
        def __init__(
            self,
            engine: Any,
            config: "backend.WorkerConfig",
            event_loop: Any,
            raw: bool = False,
        ) -> None: ...
        def run(self) -> Awaitable[None]: ...
