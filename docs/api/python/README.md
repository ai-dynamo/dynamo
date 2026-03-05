---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Python API
max-toc-depth: 2
---

# Python API Reference

This page documents the public Python API for NVIDIA Dynamo. Classes and functions are organized by module. Expand any item to see its full API.

| Name | Description |
| --- | --- |
| [`Core Bindings`](#core-bindings) | Low-level Rust-backed runtime, routing, KV cache, memory, and model management bindings. |
| [`dynamo.runtime`](#dynamo.runtime) | Decorators and utilities for defining Dynamo workers and endpoints. |
| [`dynamo.llm`](#dynamo.llm) | LLM serving pipeline components and engine integration. |
| [`dynamo.frontend`](#dynamo.frontend) | HTTP frontend configuration and OpenAI-compatible API gateway. |
| [`dynamo.common`](#dynamo.common) | Shared configuration, constants, storage, and utility functions. |
| [`dynamo.health_check`](#dynamo.health_check) | Health check payload and environment-based configuration. |
| [`dynamo.logits_processing`](#dynamo.logits_processing) | Custom logits processing for LLM token generation. |
| [`dynamo.planner`](#dynamo.planner) | Scaling connectors and decision types for the Dynamo Planner. |
| [`dynamo.router`](#dynamo.router) | Request routing configuration and argument groups. |
| [`dynamo.mocker`](#dynamo.mocker) | Mock engine for testing without GPU resources. |
| [`dynamo.nixl_connect`](#dynamo.nixl_connect) | RDMA-based data transfer operations via the NIXL library. |

---

## Core Bindings

Low-level Rust-backed runtime, routing, KV cache, memory, and model management bindings.

### Classes

<details>
<summary><strong>`RuntimeMetrics` — Helper class for registering Prometheus metrics callbacks on an Endpoint.</strong></summary>

*Source: [`dynamo/prometheus_metrics.pyi#L12`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/prometheus_metrics.pyi#L12)*

Provides utilities for integrating external metrics (e.g., from vLLM, SGLang, TensorRT-LLM).
<b>Methods</b>

#### `register_prometheus_expfmt_callback(callback: Callable[[], str])`

*Source: [`dynamo/prometheus_metrics.pyi#L19`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/prometheus_metrics.pyi#L19)*

Register a Python callback that returns Prometheus exposition text.
The returned text will be appended to the /metrics endpoint output.

This allows you to integrate external Prometheus metrics (e.g. from vLLM)
directly into the endpoint's metrics output.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `callback` | `Callable[[], str]` | A callable that takes no arguments and returns a string      in Prometheus text exposition format |

</details>

<details>
<summary><strong>`JsonLike` — Any PyObject which can be serialized to JSON</strong></summary>

*Source: [`dynamo/_core.pyi#L27`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L27)*

Examples:
    >>> request: JsonLike = `{"prompt": "Hello", "max_tokens": 128}`
    >>> request_list: JsonLike = [1, 2, 3]
    >>> request_str: JsonLike = "plain text input"

</details>

<details>
<summary><strong>`DistributedRuntime` — The runtime object for dynamo applications</strong></summary>

*Source: [`dynamo/_core.pyi#L40`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L40)*

Examples:
    >>> import asyncio
    >>> loop = asyncio.get_event_loop()
    >>> runtime = DistributedRuntime(loop, "etcd", "nats")
    >>> endpoint = runtime.endpoint("myns.backend.generate")
    >>> client = await endpoint.client()
<b>Constructor</b>

#### `__new__(event_loop: Any, discovery_backend: str, request_plane: str, enable_nats: Optional[bool] = None) -> DistributedRuntime`

*Source: [`dynamo/_core.pyi#L51`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L51)*

Create a new DistributedRuntime.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `event_loop` | `Any` | The asyncio event loop |
| `discovery_backend` | `str` | Discovery backend ("kubernetes", "etcd", "file", or "mem") |
| `request_plane` | `str` | Request plane transport ("tcp", "http", or "nats") |
| `enable_nats` | `Optional[bool]` | Whether to enable NATS for KV events. Defaults to True.         If request_plane is "nats", NATS is always enabled.         Pass False to disable NATS initialization (e.g., for approximate routing). |
<b>Methods</b>

#### `endpoint(path: str) -> Endpoint`

*Source: [`dynamo/_core.pyi#L71`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L71)*

Get an endpoint directly by path.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `path` | `str` | Endpoint path in format 'namespace.component.endpoint'   or 'dyn://namespace.component.endpoint' |

<b>Returns:</b> `Endpoint` -- The requested endpoint

<b>Raises</b>

- `ValueError` -- If path format is invalid (not 3 parts separated by dots)
- `Exception` -- If namespace or component creation fails

<b>Example</b>

endpoint = runtime.endpoint("demo.backend.generate")
endpoint = runtime.endpoint("dyn://demo.backend.generate")
#### `shutdown()`

*Source: [`dynamo/_core.pyi#L92`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L92)*

Shutdown the runtime by triggering the cancellation token
#### `register_engine_route(route_name: str, callback: Callable[[dict], Awaitable[dict]])`

*Source: [`dynamo/_core.pyi#L98`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L98)*

Register an async callback for /engine/`{route_name}` on the system status server.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `route_name` | `str` | The route path (e.g., "start_profile" creates /engine/start_profile) |
| `callback` | `Callable[[dict], Awaitable[dict]]` | Async function with signature: async def(body: dict) -> dict |

<b>Example</b>

async def start_profile(body: dict) -> dict:
    await engine.start_profile(**body)
    return `{"status": "ok", "message": "Profiling started"}`

runtime.register_engine_route("start_profile", start_profile)

The callback receives the JSON request body as a dict and should return
a dict that will be serialized as the JSON response.

For GET requests or empty bodies, an empty dict `{}` is passed.

</details>

<details>
<summary><strong>`Endpoint` — An Endpoint is a single API endpoint</strong></summary>

*Source: [`dynamo/_core.pyi#L125`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L125)*

Examples:
    >>> endpoint = runtime.endpoint("dynamo.backend.generate")
    >>> await endpoint.serve_endpoint(RequestHandler().generate)
    >>> client = await endpoint.client()
    >>> async for resp in await client.round_robin("hello"):
    ...     print(resp)
<b>Attributes</b>

- `metrics`: `PyRuntimeMetrics` -- Get a PyRuntimeMetrics helper for registering Prometheus metrics callbacks.

<b>Methods</b>

#### `serve_endpoint(handler: RequestHandler, graceful_shutdown: bool = True, metrics_labels: Optional[List[Tuple[str, str]]] = None, health_check_payload: Optional[Dict[str, Any]] = None)`

*Source: [`dynamo/_core.pyi#L138`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L138)*

Serve an endpoint discoverable by all connected clients at
`{{ namespace }}/components/{{ component_name }}/endpoints/{{ endpoint_name }}`

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `handler` | `RequestHandler` | The request handler function |
| `graceful_shutdown` | `bool` | Whether to wait for inflight requests to complete during shutdown (default: True) |
| `metrics_labels` | `Optional[List[Tuple[str, str]]]` | Optional list of metrics labels to add to the metrics |
| `health_check_payload` | `Optional[Dict[str, Any]]` | Optional dict containing the health check request payload                   that will be used to verify endpoint health |
#### `client(router_mode: Optional[RouterMode] = None) -> Client`

*Source: [`dynamo/_core.pyi#L152`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L152)*

Create a `Client` capable of calling served instances of this endpoint.

By default this uses round-robin routing when `router_mode` is not provided.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `router_mode` | `Optional[RouterMode]` | None |  |

#### `connection_id() -> int`

*Source: [`dynamo/_core.pyi#L160`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L160)*

Opaque unique ID for this worker. May change over worker lifetime.
#### `unregister_endpoint_instance()`

*Source: [`dynamo/_core.pyi#L176`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L176)*

Unregister this endpoint instance from discovery.

This removes the endpoint from the instances bucket, preventing the router
from sending requests to this worker. Use this when a worker is sleeping
and should not receive any requests.
#### `register_endpoint_instance()`

*Source: [`dynamo/_core.pyi#L186`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L186)*

Re-register this endpoint instance to discovery.

This adds the endpoint back to the instances bucket, allowing the router
to send requests to this worker again. Use this when a worker wakes up
and should start receiving requests.

</details>

<details>
<summary><strong>`Client` — A client capable of calling served instances of an endpoint</strong></summary>

*Source: [`dynamo/_core.pyi#L196`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L196)*

Examples:
    >>> client = await endpoint.client()
    >>> async for chunk in await client.round_robin("hello world"):
    ...     print(chunk.get("data"))
    >>> async for resp in await client.random(`{"prompt": "hi"}`):
    ...     print(resp)
<b>Methods</b>

#### `instance_ids() -> List[int]`

*Source: [`dynamo/_core.pyi#L209`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L209)*

Get list of current instance IDs.

<b>Returns:</b> `List[int]` -- A list of currently available instance IDs
#### `wait_for_instances() -> List[int]`

*Source: [`dynamo/_core.pyi#L218`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L218)*

Wait for instances to be available for work and return their IDs.

<b>Returns:</b> `List[int]` -- A list of instance IDs that are available for work
#### `random(request: JsonLike) -> AsyncIterator[JsonLike]`

*Source: [`dynamo/_core.pyi#L227`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L227)*

Pick a random instance of the endpoint and issue the request

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `request` | `JsonLike` |  |  |

#### `round_robin(request: JsonLike) -> AsyncIterator[JsonLike]`

*Source: [`dynamo/_core.pyi#L233`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L233)*

Pick the next instance of the endpoint in a round-robin fashion

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `request` | `JsonLike` |  |  |

#### `direct(request: JsonLike, instance: str) -> AsyncIterator[JsonLike]`

*Source: [`dynamo/_core.pyi#L239`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L239)*

Pick a specific instance of the endpoint

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `request` | `JsonLike` |  |  |
| `instance` | `str` |  |  |


</details>

<details>
<summary><strong>`ModelCardInstanceId` — Unique identifier for a worker instance: namespace, component, endpoint and instance_id.</strong></summary>

*Source: [`dynamo/_core.pyi#L246`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L246)*

The instance_id is not currently exposed in the Python bindings.
<b>Methods</b>

#### `triple() -> Tuple[str, str, str]`

*Source: [`dynamo/_core.pyi#L251`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L251)*

Triple of namespace, component and endpoint this worker is serving.

</details>

<details>
<summary><strong>`Context` — Context wrapper around AsyncEngineContext for Python bindings.</strong></summary>

*Source: [`dynamo/_core.pyi#L296`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L296)*

Provides tracing and cancellation capabilities for request handling.
Examples:
    >>> from dynamo._core import Context
    >>> context = Context()
    >>> context = Context(id="req-42")
    >>> print(context.id())
    >>> context.stop_generating()
    >>> assert context.is_stopped()
<b>Attributes</b>

- `trace_id`: `Optional[str]` -- Get the distributed trace ID if available.
- `span_id`: `Optional[str]` -- Get the distributed span ID if available.
- `parent_span_id`: `Optional[str]` -- Get the parent span ID if available.

<b>Constructor</b>

#### `__init__(id: Optional[str] = None)`

*Source: [`dynamo/_core.pyi#L309`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L309)*

Create a new Context instance.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `id` | `Optional[str]` | Optional request ID. If None, a default ID will be generated. |
<b>Methods</b>

#### `is_stopped() -> bool`

*Source: [`dynamo/_core.pyi#L318`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L318)*

Check if the context has been stopped (synchronous).

<b>Returns:</b> `bool` -- True if the context is stopped, False otherwise.
#### `is_killed() -> bool`

*Source: [`dynamo/_core.pyi#L327`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L327)*

Check if the context has been killed (synchronous).

<b>Returns:</b> `bool` -- True if the context is killed, False otherwise.
#### `stop_generating()`

*Source: [`dynamo/_core.pyi#L336`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L336)*

Issue a stop generating signal to the context.
#### `id() -> Optional[str]`

*Source: [`dynamo/_core.pyi#L342`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L342)*

Get the context ID.

<b>Returns:</b> `Optional[str]` -- The context identifier string, or None if not set.
#### `async_killed_or_stopped() -> bool`

*Source: [`dynamo/_core.pyi#L351`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L351)*

Asynchronously wait until the context is killed or stopped.

<b>Returns:</b> `bool` -- True when the context is killed or stopped.

</details>

<details>
<summary><strong>`WorkerMetricsPublisher` — A metrics publisher will provide metrics to the router for load monitoring.</strong></summary>

*Source: [`dynamo/_core.pyi#L390`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L390)*

Examples:
    >>> publisher = WorkerMetricsPublisher()
    >>> await publisher.create_endpoint(endpoint)
    >>> publisher.publish(dp_rank=0, active_decode_blocks=128)
<b>Constructor</b>

#### `__init__()`

*Source: [`dynamo/_core.pyi#L401`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L401)*

Create a `WorkerMetricsPublisher` object
<b>Methods</b>

#### `create_endpoint(endpoint: Endpoint)`

*Source: [`dynamo/_core.pyi#L406`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L406)*

Initialize the NATS endpoint for publishing worker metrics. Must be awaited.

Extracts component information from the endpoint to set up metrics publishing
on the correct NATS subject for routing decisions.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `endpoint` | `Endpoint` | The endpoint to extract component information from for metrics publishing |
#### `publish(dp_rank: Optional[int], active_decode_blocks: int)`

*Source: [`dynamo/_core.pyi#L417`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L417)*

Publish worker metrics for load monitoring.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `dp_rank` | `Optional[int]` | Data parallel rank of the worker (None defaults to 0) |
| `active_decode_blocks` | `int` | Number of active KV cache blocks |

</details>

<details>
<summary><strong>`ModelRuntimeConfig` — A model runtime configuration is a collection of runtime information</strong></summary>

*Source: [`dynamo/_core.pyi#L452`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L452)*

Examples:
    >>> from dynamo._core import ModelRuntimeConfig
    >>> config = ModelRuntimeConfig()
    >>> config.total_kv_blocks = 2048
    >>> config.max_num_seqs = 256
    >>> config.max_num_batched_tokens = 8192
    >>> config.enable_local_indexer = True
    >>> config.tool_call_parser = "hermes"
    >>> config.set_engine_specific("tp_size", 4)
<b>Attributes</b>

- `total_kv_blocks`: `int | None`
- `max_num_seqs`: `int | None`
- `max_num_batched_tokens`: `int | None`
- `tool_call_parser`: `str | None`
- `reasoning_parser`: `str | None`
- `enable_local_indexer`: `bool`
- `runtime_data`: `dict[str, Any]`
- `tensor_model_config`: `Any | None`

<b>Constructor</b>

#### `__init__()`

*Source: [`dynamo/_core.pyi#L475`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L475)*

<b>Methods</b>

#### `set_engine_specific(key: str, value: Any)`

*Source: [`dynamo/_core.pyi#L477`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L477)*

Set an engine-specific runtime configuration value

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `key` | `str` |  |  |
| `value` | `Any` |  |  |

#### `get_engine_specific(key: str) -> Any | None`

*Source: [`dynamo/_core.pyi#L481`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L481)*

Get an engine-specific runtime configuration value

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `key` | `str` |  |  |


</details>

<details>
<summary><strong>`OverlapScores` — A collection of prefix matching scores of workers for a given token ids.</strong></summary>

*Source: [`dynamo/_core.pyi#L485`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L485)*

'scores' is a map of worker id to the score which is the number of matching blocks.
Examples:
    >>> from dynamo._core import RadixTree
    >>> tree = RadixTree()
    >>> scores = tree.find_matches([42, 43])
    >>> worker_scores = scores.scores  # Dict[int, int]
    >>> freqs = scores.frequencies  # List[int]
<b>Attributes</b>

- `scores`: `Dict[int, int]` -- Map of worker_id to the score which is the number of matching blocks.
- `frequencies`: `List[int]` -- List of frequencies that the blocks have been accessed.


</details>

<details>
<summary><strong>`RadixTree` — A RadixTree that tracks KV cache blocks and can find prefix matches for sequences.</strong></summary>

*Source: [`dynamo/_core.pyi#L518`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L518)*

Thread-safe: operations route to a dedicated background thread and long calls
release the Python GIL.
Examples:
    >>> import json
    >>> from dynamo._core import RadixTree
    >>> tree = RadixTree()
    >>> tree_ttl = RadixTree(expiration_duration_secs=120.0)
    >>> event = \{"event_id": 1, "data": \{"stored": \{"parent_hash": None,
    ...     "blocks": [`{"block_hash": 42, "tokens_hash": 42}`]\}\}\}
    >>> tree.apply_event(0, json.dumps(event).encode())
    >>> scores = tree.find_matches([42])
    >>> print(scores.scores)
<b>Constructor</b>

#### `__init__(expiration_duration_secs: Optional[float] = None)`

*Source: [`dynamo/_core.pyi#L536`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L536)*

Create a new RadixTree instance.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `expiration_duration_secs` | `Optional[float]` | Optional expiration duration in seconds for cached blocks.                     If None, blocks never expire. |
<b>Methods</b>

#### `find_matches(sequence: List[int], early_exit: bool = False) -> OverlapScores`

*Source: [`dynamo/_core.pyi#L546`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L546)*

Find prefix matches for the given sequence of block hashes.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `sequence` | `List[int]` | List of block hashes to find matches for |
| `early_exit` | `bool` | If True, stop searching after finding the first match |

<b>Returns:</b> `OverlapScores` -- OverlapScores containing worker matching scores and frequencies
#### `apply_event(worker_id: int, kv_cache_event_bytes: bytes)`

*Source: [`dynamo/_core.pyi#L561`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L561)*

Apply a KV cache event to update the RadixTree state.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `worker_id` | `int` | ID of the worker that generated the event |
| `kv_cache_event_bytes` | `bytes` | Serialized KV cache event as bytes |

<b>Raises</b>

- `ValueError` -- If the event bytes cannot be deserialized
#### `remove_worker(worker_id: int)`

*Source: [`dynamo/_core.pyi#L574`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L574)*

Remove all blocks associated with a specific worker.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `worker_id` | `int` | ID of the worker to remove |
#### `clear_all_blocks(worker_id: int)`

*Source: [`dynamo/_core.pyi#L583`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L583)*

Clear all blocks for a specific worker.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `worker_id` | `int` | ID of the worker whose blocks should be cleared |
#### `dump_tree_as_events() -> List[str]`

*Source: [`dynamo/_core.pyi#L592`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L592)*

Dump the current RadixTree state as a list of JSON-serialized KV cache events.

<b>Returns:</b> `List[str]` -- List of JSON-serialized KV cache events as strings

</details>

<details>
<summary><strong>`KvIndexer` — A KV Indexer that tracks KV Events emitted by workers. Events include add_block and remove_block.</strong></summary>

*Source: [`dynamo/_core.pyi#L601`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L601)*

Examples:
    >>> from dynamo._core import KvIndexer
    >>> indexer = KvIndexer(endpoint=endpoint, block_size=64)
    >>> scores = indexer.find_matches_for_request(token_ids=[1, 2, 3])
    >>> print(scores.scores)
    >>> print(indexer.block_size())
<b>Constructor</b>

#### `__init__(endpoint: Endpoint, block_size: int)`

*Source: [`dynamo/_core.pyi#L614`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L614)*

Create a `KvIndexer` object

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `endpoint` | `Endpoint` |  |  |
| `block_size` | `int` |  |  |

<b>Methods</b>

#### `find_matches(sequence: List[int]) -> OverlapScores`

*Source: [`dynamo/_core.pyi#L619`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L619)*

Find prefix matches for the given sequence of block hashes.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `sequence` | `List[int]` | List of block hashes to find matches for |

<b>Returns:</b> `OverlapScores` -- OverlapScores containing worker matching scores and frequencies
#### `find_matches_for_request(token_ids: List[int], lora_name: Optional[str] = None) -> OverlapScores`

*Source: [`dynamo/_core.pyi#L631`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L631)*

Return the overlapping scores of workers for the given token ids.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `token_ids` | `List[int]` |  |  |
| `lora_name` | `Optional[str]` | None |  |

#### `block_size() -> int`

*Source: [`dynamo/_core.pyi#L639`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L639)*

Return the block size of the KV Indexer.

</details>

<details>
<summary><strong>`ApproxKvIndexer` — An approximate KV Indexer that doesn't receive KV cache events from workers.</strong></summary>

*Source: [`dynamo/_core.pyi#L645`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L645)*

Instead, it relies on routing decisions with TTL-based expiration and pruning
to estimate which blocks are cached on which workers.

This is useful when:
- Backend engines don't emit KV events
- You want to reduce event processing overhead
- Lower routing accuracy is acceptable
Examples:
    >>> from dynamo._core import ApproxKvIndexer
    >>> indexer = ApproxKvIndexer(
    ...     endpoint=endpoint, kv_block_size=64,
    ...     router_ttl_secs=120.0, router_max_tree_size=1048576,
    ... )
    >>> scores = indexer.find_matches_for_request(token_ids=[1, 2, 3])
    >>> await indexer.process_routing_decision_for_request([1, 2, 3], worker_id=0)
<b>Constructor</b>

#### `__init__(endpoint: Endpoint, kv_block_size: int, router_ttl_secs: float = 120.0, router_max_tree_size: int = 1048576, router_prune_target_ratio: float = 0.8)`

*Source: [`dynamo/_core.pyi#L667`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L667)*

Create an `ApproxKvIndexer` object

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `component` | `Any` | The component to associate with this indexer |
| `kv_block_size` | `int` | The KV cache block size |
| `router_ttl_secs` | `float` | TTL for blocks in seconds (default: 120.0) |
| `router_max_tree_size` | `int` | Maximum tree size before pruning (default: 1048576, which is 2^20) |
| `router_prune_target_ratio` | `float` | Target size ratio after pruning (default: 0.8) |
<b>Methods</b>

#### `find_matches_for_request(token_ids: List[int], lora_name: Optional[str] = None) -> OverlapScores`

*Source: [`dynamo/_core.pyi#L687`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L687)*

Return the overlapping scores of workers for the given token ids.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `token_ids` | `List[int]` | List of token IDs to find matches for |
| `lora_name` | `Optional[str]` | Optional LoRA adapter name for adapter-aware matching |

<b>Returns:</b> `OverlapScores` -- OverlapScores containing worker matching scores and frequencies
#### `block_size() -> int`

*Source: [`dynamo/_core.pyi#L702`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L702)*

Return the block size of the ApproxKvIndexer.

<b>Returns:</b> `int` -- The KV cache block size
#### `process_routing_decision_for_request(tokens: List[int], worker_id: int, dp_rank: int = 0)`

*Source: [`dynamo/_core.pyi#L711`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L711)*

Notify the indexer that a token sequence has been routed to a specific worker.

This updates the indexer's internal state to track which blocks are likely
cached on which workers based on routing decisions.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `tokens` | `List[int]` | List of token IDs that were routed |
| `worker_id` | `int` | The worker ID the request was routed to |
| `dp_rank` | `int` | The data parallel rank (default: 0) |

</details>

<details>
<summary><strong>`KvEventPublisher` — A KV event publisher will publish KV events corresponding to the component.</strong></summary>

*Source: [`dynamo/_core.pyi#L728`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L728)*

Examples:
    >>> publisher = KvEventPublisher(
    ...     endpoint=endpoint, kv_block_size=64, dp_rank=0,
    ...     zmq_endpoint="tcp://127.0.0.1:5557", zmq_topic="",
    ... )
    >>> publisher.publish_stored(
    ...     token_ids=[1, 2, 3, 4], num_block_tokens=[4],
    ...     block_hashes=[123456],
    ... )
    >>> publisher.publish_removed(block_hashes=[123456])
<b>Constructor</b>

#### `__init__(endpoint: Endpoint, worker_id: int = 0, kv_block_size: int = 0, dp_rank: int = 0, enable_local_indexer: bool = False, zmq_endpoint: Optional[str] = None, zmq_topic: Optional[str] = None)`

*Source: [`dynamo/_core.pyi#L745`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L745)*

Create a `KvEventPublisher` object.

When zmq_endpoint is provided, the publisher subscribes to a ZMQ socket for
incoming engine events (e.g. from SGLang/vLLM) and relays them to NATS.

When zmq_endpoint is None, events are pushed manually via publish_stored/publish_removed.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `endpoint` | `Endpoint` | The endpoint to extract component information from for event publishing |
| `worker_id` | `int` | The worker ID (unused, inferred from endpoint) |
| `kv_block_size` | `int` | The KV block size (must be > 0) |
| `dp_rank` | `int` | The data parallel rank (defaults to 0) |
| `enable_local_indexer` | `bool` | Enable worker-local KV indexer |
| `zmq_endpoint` | `Optional[str]` | Optional ZMQ endpoint for relay mode (e.g. "tcp://127.0.0.1:5557") |
| `zmq_topic` | `Optional[str]` | ZMQ topic to subscribe to (defaults to "" when zmq_endpoint is set) |
<b>Methods</b>

#### `publish_stored(token_ids: List[int], num_block_tokens: List[int], block_hashes: List[int], parent_hash: Optional[int] = None, block_mm_infos: Optional[List[Optional[Dict[str, Any]]]] = None, lora_name: Optional[str] = None)`

*Source: [`dynamo/_core.pyi#L773`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L773)*

Publish a KV stored event.

Event IDs are managed internally by the publisher using a monotonic counter.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `token_ids` | `List[int]` | List of token IDs |
| `num_block_tokens` | `List[int]` | Number of tokens per block |
| `block_hashes` | `List[int]` | List of block hashes (signed 64-bit integers) |
| `parent_hash` | `Optional[int]` | Optional parent hash (signed 64-bit integer) |
| `block_mm_infos` | `Optional[List[Optional[Dict[str, Any]]]]` | Optional list of multimodal info for each block. Each item is either None or a dict with "mm_objects" key containing a list of `{"mm_hash": int, "offsets": [[start, end], ...]}` dicts. |
| `lora_name` | `Optional[str]` | Optional LoRA adapter name for adapter-aware block hashing. |
#### `publish_removed(block_hashes: List[int])`

*Source: [`dynamo/_core.pyi#L799`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L799)*

Publish a KV removed event.

Event IDs are managed internally by the publisher using a monotonic counter.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `block_hashes` | `List[int]` | List of block hashes to remove (signed 64-bit integers) |
#### `shutdown()`

*Source: [`dynamo/_core.pyi#L810`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L810)*

Shuts down the event publisher, stopping any background tasks.

</details>

<details>
<summary><strong>`HttpService` — A HTTP service for dynamo applications.</strong></summary>

*Source: [`dynamo/_core.pyi#L816`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L816)*

It is a OpenAI compatible http ingress into the Dynamo Distributed Runtime.
Examples:
    >>> service = HttpService(port=8000)
    >>> service.add_chat_completions_model("my-model", "checksum", engine)
    >>> await service.run(runtime)
    >>> service.shutdown()
<b>Constructor</b>

#### `__init__(port: Optional[int] = None)`

*Source: [`dynamo/_core.pyi#L827`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L827)*

Create a new HTTP service.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `port` | `Optional[int]` | Optional port number to bind the service to (default: 8080) |
<b>Methods</b>

#### `run(runtime: DistributedRuntime)`

*Source: [`dynamo/_core.pyi#L836`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L836)*

Run the HTTP service.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `runtime` | `DistributedRuntime` | DistributedRuntime instance for token management |
#### `shutdown()`

*Source: [`dynamo/_core.pyi#L845`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L845)*

Shutdown the HTTP service by cancelling its internal token.

</details>

<details>
<summary><strong>`PythonAsyncEngine` — Bridge a Python async generator onto Dynamo's AsyncEngine interface.</strong></summary>

*Source: [`dynamo/_core.pyi#L851`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L851)*

Examples:
    >>> import asyncio
    >>> from dynamo._core import PythonAsyncEngine
    >>> async def my_generator(request):
    ...     yield `{"text": "hello"}`
    >>> loop = asyncio.get_running_loop()
    >>> engine = PythonAsyncEngine(my_generator, loop)
<b>Constructor</b>

#### `__init__(generator: Any, event_loop: Any)`

*Source: [`dynamo/_core.pyi#L863`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L863)*

Wrap a Python generator and event loop for use with Dynamo services.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `generator` | `Any` |  |  |
| `event_loop` | `Any` |  |  |


</details>

<details>
<summary><strong>`HttpAsyncEngine` — An async engine for a distributed Dynamo http service. This is an extension of the</strong></summary>

*Source: [`dynamo/_core.pyi#L869`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L869)*

python based AsyncEngine that handles HttpError exceptions from Python and
converts them to the Rust version of HttpError

</details>

<details>
<summary><strong>`KserveGrpcService` — A gRPC service implementing the KServe protocol for dynamo applications.</strong></summary>

*Source: [`dynamo/_core.pyi#L878`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L878)*

Provides model management for completions, chat completions, and tensor-based models.
Examples:
    >>> from dynamo._core import KserveGrpcService, PythonAsyncEngine
    >>> service = KserveGrpcService(port=8787, host="0.0.0.0")
    >>> engine = PythonAsyncEngine(my_generator, loop)
    >>> service.add_completions_model("my-model", "checksum", engine)
    >>> shutdown_signal = service.run(runtime)
    >>> print(service.list_completions_models())
    >>> await shutdown_signal
<b>Constructor</b>

#### `__init__(port: Optional[int] = None, host: Optional[str] = None)`

*Source: [`dynamo/_core.pyi#L892`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L892)*

Create a new KServe gRPC service.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `port` | `Optional[int]` | Optional port number to bind the service to |
| `host` | `Optional[str]` | Optional host address to bind the service to |
<b>Methods</b>

#### `add_completions_model(model: str, checksum: str, engine: PythonAsyncEngine)`

*Source: [`dynamo/_core.pyi#L902`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L902)*

Register a completions model with the service.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `model` | `str` | The model name |
| `checksum` | `str` | The model checksum |
| `engine` | `PythonAsyncEngine` | The async engine to handle requests |
#### `add_chat_completions_model(model: str, checksum: str, engine: PythonAsyncEngine)`

*Source: [`dynamo/_core.pyi#L918`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L918)*

Register a chat completions model with the service.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `model` | `str` | The model name |
| `checksum` | `str` | The model checksum |
| `engine` | `PythonAsyncEngine` | The async engine to handle requests |
#### `add_tensor_model(model: str, checksum: str, engine: PythonAsyncEngine, runtime_config: Optional[ModelRuntimeConfig])`

*Source: [`dynamo/_core.pyi#L934`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L934)*

Register a tensor-based model with the service.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `model` | `str` | The model name |
| `checksum` | `str` | The model checksum |
| `engine` | `PythonAsyncEngine` | The async engine to handle requests |
#### `remove_completions_model(model: str)`

*Source: [`dynamo/_core.pyi#L951`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L951)*

Remove a completions model from the service.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `model` | `str` | The model name to remove |
#### `remove_chat_completions_model(model: str)`

*Source: [`dynamo/_core.pyi#L960`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L960)*

Remove a chat completions model from the service.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `model` | `str` | The model name to remove |
#### `remove_tensor_model(model: str)`

*Source: [`dynamo/_core.pyi#L969`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L969)*

Remove a tensor model from the service.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `model` | `str` | The model name to remove |
#### `list_chat_completions_models() -> List[str]`

*Source: [`dynamo/_core.pyi#L978`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L978)*

List all registered chat completions models.

<b>Returns:</b> `List[str]` -- List of model names
#### `list_completions_models() -> List[str]`

*Source: [`dynamo/_core.pyi#L987`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L987)*

List all registered completions models.

<b>Returns:</b> `List[str]` -- List of model names
#### `list_tensor_models() -> List[str]`

*Source: [`dynamo/_core.pyi#L996`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L996)*

List all registered tensor models.

<b>Returns:</b> `List[str]` -- List of model names
#### `run(runtime: DistributedRuntime)`

*Source: [`dynamo/_core.pyi#L1005`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1005)*

Run the KServe gRPC service.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `runtime` | `DistributedRuntime` | DistributedRuntime instance for token management |
#### `shutdown()`

*Source: [`dynamo/_core.pyi#L1014`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1014)*

Shutdown the KServe gRPC service by cancelling its internal token.

</details>

<details>
<summary><strong>`ModelInput` — What type of request this model needs: Text, Tokens or Tensor</strong></summary>

*Source: [`dynamo/_core.pyi#L1020`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1020)*

Examples:
    >>> from dynamo._core import ModelInput
    >>> input_type = ModelInput.Tokens
    >>> input_type = ModelInput.Text
    >>> input_type = ModelInput.Tensor
<b>Attributes</b>

- `Text`: `ModelInput`
- `Tokens`: `ModelInput`
- `Tensor`: `ModelInput`


</details>

<details>
<summary><strong>`ModelType` — What type of request this model needs: Chat, Completions, Embedding, Tensor, Images, Videos or Prefill</strong></summary>

*Source: [`dynamo/_core.pyi#L1033`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1033)*

Examples:
    >>> from dynamo._core import ModelType
    >>> model_type = ModelType.Chat | ModelType.Completions
    >>> model_type = ModelType.TensorBased
    >>> model_type = ModelType.Prefill
<b>Attributes</b>

- `Chat`: `ModelType`
- `Completions`: `ModelType`
- `Embedding`: `ModelType`
- `TensorBased`: `ModelType`
- `Prefill`: `ModelType`
- `Images`: `ModelType`
- `Audios`: `ModelType`
- `Videos`: `ModelType`


</details>

<details>
<summary><strong>`RouterMode` — Router mode for load balancing requests across workers</strong></summary>

*Source: [`dynamo/_core.pyi#L1051`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1051)*

Examples:
    >>> from dynamo._core import RouterMode
    >>> mode = RouterMode.RoundRobin
    >>> mode = RouterMode.KV
    >>> mode = RouterMode.Direct
<b>Attributes</b>

- `RoundRobin`: `RouterMode`
- `Random`: `RouterMode`
- `KV`: `RouterMode`
- `Direct`: `RouterMode`


</details>

<details>
<summary><strong>`RouterConfig` — How to route the request</strong></summary>

*Source: [`dynamo/_core.pyi#L1065`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1065)*

Examples:
    >>> from dynamo._core import RouterConfig, RouterMode, KvRouterConfig
    >>> config = RouterConfig(mode=RouterMode.RoundRobin)
    >>> kv_config = KvRouterConfig(overlap_score_weight=0.5)
    >>> config = RouterConfig(
    ...     mode=RouterMode.KV,
    ...     config=kv_config,
    ...     active_decode_blocks_threshold=0.9,
    ...     decode_fallback=True,
    ... )
<b>Constructor</b>

#### `__init__(mode: RouterMode, config: Optional[KvRouterConfig] = None, active_decode_blocks_threshold: Optional[float] = None, active_prefill_tokens_threshold: Optional[int] = None, active_prefill_tokens_threshold_frac: Optional[float] = None, decode_fallback: bool = False)`

*Source: [`dynamo/_core.pyi#L1079`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1079)*

Create a RouterConfig.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `mode` | `RouterMode` | The router mode (RoundRobin, Random, KV, or Direct) |
| `config` | `Optional[KvRouterConfig]` | Optional KV router configuration (used when mode is KV) |
| `active_decode_blocks_threshold` | `Optional[float]` | Threshold percentage (0.0-1.0) for decode blocks busy detection |
| `active_prefill_tokens_threshold` | `Optional[int]` | Literal token count threshold for prefill busy detection |
| `active_prefill_tokens_threshold_frac` | `Optional[float]` | Fraction of max_num_batched_tokens for busy detection |
| `decode_fallback` | `bool` | Allow falling back to decode-only mode when prefill workers are unavailable |

</details>

<details>
<summary><strong>`KvRouterConfig` — Values for KV router</strong></summary>

*Source: [`dynamo/_core.pyi#L1101`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1101)*

Examples:
    >>> config = KvRouterConfig()
    >>> config_custom = KvRouterConfig(
    ...     overlap_score_weight=0.5,
    ...     router_temperature=0.5,
    ...     router_track_active_blocks=True,
    ... )
<b>Constructor</b>

#### `__init__(overlap_score_weight: float = 1.0, router_temperature: float = 0.0, use_kv_events: bool = True, durable_kv_events: bool = False, router_replica_sync: bool = False, router_track_active_blocks: bool = True, router_track_output_blocks: bool = False, router_assume_kv_reuse: bool = True, router_snapshot_threshold: Optional[int] = 1000000, router_reset_states: bool = False, router_ttl_secs: float = 120.0, router_max_tree_size: int = 1048576, router_prune_target_ratio: float = 0.8, router_queue_threshold: Optional[float] = None, router_event_threads: int = 4)`

*Source: [`dynamo/_core.pyi#L1113`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1113)*

Create a KV router configuration.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `overlap_score_weight` | `float` | Weight for overlap score in worker selection (default: 1.0) |
| `router_temperature` | `float` | Temperature for worker sampling via softmax (default: 0.0) |
| `use_kv_events` | `bool` | Whether to use KV events from workers (default: True) |
| `durable_kv_events` | `bool` | **Deprecated.** Enable durable KV events using NATS JetStream (default: False). This option will be removed in a future release. The event-plane subscriber (local_indexer mode) is now the recommended path. |
| `router_replica_sync` | `bool` | Enable replica synchronization (default: False) |
| `router_track_active_blocks` | `bool` | Track active blocks for load balancing (default: True) |
| `router_track_output_blocks` | `bool` | Track output blocks during generation (default: False). When enabled, the router adds placeholder blocks as tokens are generated and applies fractional decay based on progress toward expected output sequence length (agent_hints.osl in nvext). |
| `router_assume_kv_reuse` | `bool` | Assume KV cache reuse when tracking active blocks (default: True). When True, computes actual block hashes. When False, generates random hashes. |
| `router_snapshot_threshold` | `Optional[int]` | Number of messages before snapshot (default: 1000000) |
| `router_reset_states` | `bool` | Reset router state on startup (default: False) |
| `router_ttl_secs` | `float` | TTL for blocks in seconds when not using KV events (default: 120.0) |
| `router_max_tree_size` | `int` | Maximum tree size before pruning (default: 1048576, which is 2^20) |
| `router_prune_target_ratio` | `float` | Target size ratio after pruning (default: 0.8) |
| `router_queue_threshold` | `Optional[float]` | Queue threshold fraction for prefill token capacity (default: None). When set, requests are queued if all workers exceed this fraction of max_num_batched_tokens. Enables priority scheduling via latency_sensitivity hints. If None, queueing is disabled and all requests go directly to the scheduler. |
| `router_event_threads` | `int` | Number of event processing threads (default: 4). When > 1, uses a concurrent radix tree with a thread pool. |

</details>

<details>
<summary><strong>`EngineConfig` — Holds internal configuration for a Dynamo engine.</strong></summary>

*Source: [`dynamo/_core.pyi#L1235`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1235)*

Examples:
    >>> args = EntrypointArgs(engine_type=EngineType.Dynamic, model_path="/models/llama")
    >>> engine_config = await make_engine(runtime, args)
    >>> await run_input(runtime, "http", engine_config)

</details>

<details>
<summary><strong>`Layer` — A KV cache block layer</strong></summary>

*Source: [`dynamo/_core.pyi#L1264`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1264)*


</details>

<details>
<summary><strong>`Block` — A KV cache block</strong></summary>

*Source: [`dynamo/_core.pyi#L1283`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1283)*

Examples:
    >>> block = blocks[0]  # get first block from a BlockList
    >>> num_layers = len(block)
    >>> layer = block[0]
    >>> all_layers = block.to_list()
<b>Methods</b>

#### `to_list() -> List[Layer]`

*Source: [`dynamo/_core.pyi#L1319`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1319)*

Get a list of layers

</details>

<details>
<summary><strong>`BlockList` — A list of KV cache blocks</strong></summary>

*Source: [`dynamo/_core.pyi#L1338`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1338)*

<b>Methods</b>

#### `to_list() -> List[Block]`

*Source: [`dynamo/_core.pyi#L1369`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1369)*

Get a list of blocks

</details>

<details>
<summary><strong>`BlockManager` — A KV cache block manager</strong></summary>

*Source: [`dynamo/_core.pyi#L1375`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1375)*

Examples:
    >>> bm = BlockManager(worker_id=0, num_layer=32, page_size=16, inner_dim=128)
    >>> blocks = await bm.allocate_device_blocks(4)
    >>> layer = blocks[0][0]  # first block, first layer
<b>Constructor</b>

#### `__init__(worker_id: int, num_layer: int, page_size: int, inner_dim: int, dtype: Optional[str] = None, host_num_blocks: Optional[int] = None, device_num_blocks: Optional[int] = None, device_id: int = 0)`

*Source: [`dynamo/_core.pyi#L1384`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1384)*

Create a `BlockManager` object

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `worker_id` | `int` |  |  |
| `num_layer` | `int` |  |  |
| `page_size` | `int` |  |  |
| `inner_dim` | `int` |  |  |
| `dtype` | `Optional[str]` | None |  |
| `host_num_blocks` | `Optional[int]` | None |  |
| `device_num_blocks` | `Optional[int]` | None |  |
| `device_id` | `int` | 0 |  |

<b>Methods</b>

#### `allocate_host_blocks_blocking(count: int) -> BlockList`

*Source: [`dynamo/_core.pyi#L1419`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1419)*

Allocate a list of host blocks (blocking call)

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `count` | `int` |  |  |

#### `allocate_host_blocks(count: int) -> BlockList`

*Source: [`dynamo/_core.pyi#L1435`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1435)*

Allocate a list of host blocks

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `count` | `int` |  |  |

#### `allocate_device_blocks_blocking(count: int) -> BlockList`

*Source: [`dynamo/_core.pyi#L1451`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1451)*

Allocate a list of device blocks (blocking call)

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `count` | `int` |  |  |

#### `allocate_device_blocks(count: int) -> BlockList`

*Source: [`dynamo/_core.pyi#L1467`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1467)*

Allocate a list of device blocks

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `count` | `int` |  |  |


</details>

<details>
<summary><strong>`KvRouter` — A KV-aware router that performs intelligent routing based on KV cache overlap.</strong></summary>

*Source: [`dynamo/_core.pyi#L1491`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1491)*

Examples:
    >>> config = KvRouterConfig()
    >>> kv_router = KvRouter(endpoint=endpoint, block_size=64, kv_router_config=config)
    >>> stream = await kv_router.generate(token_ids=[1, 2, 3], model="my-model")
    >>> async for chunk in stream:
    ...     print(chunk)
    >>> worker_id, dp_rank, overlap = await kv_router.best_worker([1, 2, 3])
<b>Constructor</b>

#### `__init__(endpoint: Endpoint, block_size: int, kv_router_config: KvRouterConfig)`

*Source: [`dynamo/_core.pyi#L1503`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1503)*

Create a new KvRouter instance.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `endpoint` | `Endpoint` | The endpoint to connect to for routing requests |
| `block_size` | `int` | The KV cache block size |
| `kv_router_config` | `KvRouterConfig` | Configuration for the KV router |
<b>Methods</b>

#### `generate(token_ids: List[int], model: str, stop_conditions: Optional[JsonLike] = None, sampling_options: Optional[JsonLike] = None, output_options: Optional[JsonLike] = None, router_config_override: Optional[JsonLike] = None, worker_id: Optional[int] = None, dp_rank: Optional[int] = None, extra_args: Optional[JsonLike] = None, block_mm_infos: Optional[List[Optional[Dict[str, Any]]]] = None, multi_modal_data: Optional[JsonLike] = None, mm_routing_info: Optional[JsonLike] = None) -> AsyncIterator[JsonLike]`

*Source: [`dynamo/_core.pyi#L1519`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1519)*

Generate text using the KV-aware router.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `token_ids` | `List[int]` | Input token IDs |
| `model` | `str` | Model name to use for generation |
| `stop_conditions` | `Optional[JsonLike]` | Optional stop conditions for generation |
| `sampling_options` | `Optional[JsonLike]` | Optional sampling configuration |
| `output_options` | `Optional[JsonLike]` | Optional output configuration |
| `router_config_override` | `Optional[JsonLike]` | Optional router configuration override |
| `worker_id` | `Optional[int]` | Optional worker ID to route to directly. If set, the request       will be sent to this specific worker and router states will be       updated accordingly. |
| `dp_rank` | `Optional[int]` | Optional data parallel rank to route to. If set along with worker_id,     the request will be routed to the specific (worker_id, dp_rank) pair.     If only dp_rank is set, the router will select the best worker but     force routing to the specified dp_rank. |
| `extra_args` | `Optional[JsonLike]` | Optional extra request arguments to include in the        PreprocessedRequest. |
| `block_mm_infos` | `Optional[List[Optional[Dict[str, Any]]]]` | Optional block-level multimodal metadata aligned to            request blocks. Backward-compatible shortcut; this is            converted to mm_routing_info with routing_token_ids=token_ids. |
| `multi_modal_data` | `Optional[JsonLike]` | Optional multimodal payload map to preserve image/video              data for downstream model execution. |
| `mm_routing_info` | `Optional[JsonLike]` | Optional structured routing-only multimodal payload             (e.g., `{"routing_token_ids": [...], "block_mm_infos": [...]}`)             used by router selection without changing execution token_ids. |

<b>Returns:</b> `AsyncIterator[JsonLike]` -- An async iterator yielding generation responses

> [!NOTE]
> - If worker_id is set, the request bypasses KV matching and routes directly
  to the specified worker while still updating router states.
- dp_rank allows targeting a specific data parallel replica when workers have
  multiple replicas (data_parallel_size > 1).
- This is different from query_instance_id which doesn't route the request.
#### `best_worker(token_ids: List[int], router_config_override: Optional[JsonLike] = None, request_id: Optional[str] = None, block_mm_infos: Optional[List[Optional[Dict[str, Any]]]] = None, lora_name: Optional[str] = None) -> Tuple[int, int, int]`

*Source: [`dynamo/_core.pyi#L1574`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1574)*

Find the best matching worker for the given tokens.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `token_ids` | `List[int]` | List of token IDs to find matches for |
| `router_config_override` | `Optional[JsonLike]` | Optional router configuration override |
| `request_id` | `Optional[str]` | Optional request ID. If provided, router states will be updated        to track this request (active blocks, lifecycle events). If not        provided, this is a query-only operation that doesn't affect state. |
| `block_mm_infos` | `Optional[List[Optional[Dict[str, Any]]]]` | Optional block-level multimodal metadata aligned to request            blocks. When provided, this is used in block hash computation            to enable MM-aware worker selection. |

<b>Returns:</b> `Tuple[int, int, int]` -- A tuple of (worker_id, dp_rank, overlap_blocks) where:
- worker_id: The ID of the best matching worker
- dp_rank: The data parallel rank of the selected worker
- overlap_blocks: The number of overlapping blocks found
#### `get_potential_loads(token_ids: List[int], lora_name: Optional[str] = None) -> List[Dict[str, int]]`

*Source: [`dynamo/_core.pyi#L1603`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1603)*

Get potential prefill and decode loads for all workers.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `token_ids` | `List[int]` | List of token IDs to evaluate |

<b>Returns:</b> `List[Dict[str, int]]` -- A list of dictionaries, each containing:
- worker_id: The worker ID
- dp_rank: The data parallel rank
- potential_prefill_tokens: Number of tokens that would need prefill
- potential_decode_blocks: Number of blocks currently in decode phase

> [!NOTE]
> Each (worker_id, dp_rank) pair is returned as a separate entry.
If you need aggregated loads per worker_id, sum the values manually.
#### `dump_events() -> str`

*Source: [`dynamo/_core.pyi#L1627`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1627)*

Dump all events from the KV router's indexer.

<b>Returns:</b> `str` -- A JSON string containing all indexer events
#### `mark_prefill_complete(request_id: str)`

*Source: [`dynamo/_core.pyi#L1636`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1636)*

Mark prefill as completed for a request.

This signals that the request has finished its prefill phase and is now
in the decode phase. Used to update router state for accurate load tracking.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `request_id` | `str` | The ID of the request that completed prefill |

> [!NOTE]
> This is typically called automatically by the router when using the
`generate()` method. Only call this manually if you're using
`best_worker()` with `request_id` for custom routing.
#### `free(request_id: str)`

*Source: [`dynamo/_core.pyi#L1653`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1653)*

Free a request by its ID, signaling the router to release resources.

This should be called when a request completes to update the router's
tracking of active blocks and ensure accurate load balancing.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `request_id` | `str` | The ID of the request to free |

> [!NOTE]
> This is typically called automatically by the router when using the
`generate()` method. Only call this manually if you're using
`best_worker()` with `request_id` for custom routing.

</details>

<details>
<summary><strong>`EngineType` — Engine type for Dynamo workers</strong></summary>

*Source: [`dynamo/_core.pyi#L1670`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1670)*

Examples:
    >>> from dynamo._core import EngineType
    >>> engine = EngineType.Dynamic
    >>> engine = EngineType.Echo
    >>> engine = EngineType.Mocker
<b>Attributes</b>

- `Echo`: `EngineType`
- `Dynamic`: `EngineType`
- `Mocker`: `EngineType`


</details>

<details>
<summary><strong>`EntrypointArgs` — Settings to connect an input to a worker and run them.</strong></summary>

*Source: [`dynamo/_core.pyi#L1683`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1683)*

Use by `dynamo run`.
Examples:
    >>> from dynamo._core import EntrypointArgs, EngineType, RouterConfig, RouterMode
    >>> args = EntrypointArgs(
    ...     engine_type=EngineType.Dynamic,
    ...     model_path="/models/llama-3-8b",
    ...     model_name="dyn://dynamo.backend.generate",
    ...     http_port=8000,
    ...     router_config=RouterConfig(mode=RouterMode.KV),
    ... )
    >>> mocker_args = EntrypointArgs(
    ...     engine_type=EngineType.Mocker,
    ...     model_path="/models/llama-3-8b",
    ...     is_prefill=True,
    ... )
<b>Constructor</b>

#### `__init__(engine_type: EngineType, model_path: Optional[str] = None, model_name: Optional[str] = None, endpoint_id: Optional[str] = None, context_length: Optional[int] = None, template_file: Optional[str] = None, router_config: Optional[RouterConfig] = None, kv_cache_block_size: Optional[int] = None, http_host: Optional[str] = None, http_port: Optional[int] = None, http_metrics_port: Optional[int] = None, tls_cert_path: Optional[str] = None, tls_key_path: Optional[str] = None, extra_engine_args: Optional[str] = None, namespace: Optional[str] = None, is_prefill: bool = False, migration_limit: int = 0, chat_engine_factory: Optional[Callable] = None)`

*Source: [`dynamo/_core.pyi#L1703`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1703)*

Create EntrypointArgs.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `engine_type` | `EngineType` | The type of engine to use |
| `model_path` | `Optional[str]` | Path to the model directory on disk |
| `model_name` | `Optional[str]` | Model name or dynamo endpoint (e.g. 'dyn://namespace.component.endpoint') |
| `endpoint_id` | `Optional[str]` | Optional endpoint ID |
| `context_length` | `Optional[int]` | Optional context length override |
| `template_file` | `Optional[str]` | Optional path to a prompt template file |
| `router_config` | `Optional[RouterConfig]` | Optional router configuration |
| `kv_cache_block_size` | `Optional[int]` | Optional KV cache block size |
| `http_host` | `Optional[str]` | HTTP host to bind to |
| `http_port` | `Optional[int]` | HTTP port to bind to |
| `http_metrics_port` | `Optional[int]` | HTTP metrics port (for gRPC service) |
| `tls_cert_path` | `Optional[str]` | TLS certificate path (PEM format) |
| `tls_key_path` | `Optional[str]` | TLS key path (PEM format) |
| `extra_engine_args` | `Optional[str]` | Path to extra engine arguments file |
| `namespace` | `Optional[str]` | Dynamo namespace for model discovery scoping |
| `is_prefill` | `bool` | Whether this is a prefill worker |
| `migration_limit` | `int` | Maximum number of request migrations (0=disabled) |
| `chat_engine_factory` | `Optional[Callable]` | Optional Python chat completions engine factory callback |

</details>

<details>
<summary><strong>`PlannerDecision` — A request from planner to client to perform a scaling action.</strong></summary>

*Source: [`dynamo/_core.pyi#L1749`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1749)*

Fields: num_prefill_workers, num_decode_workers, decision_id.
        -1 in any of those fields mean not set, usually because planner hasn't decided anything yet.
Call VirtualConnectorClient.complete(event) when action is completed.
Examples:
    >>> client = VirtualConnectorClient(runtime, "my-namespace")
    >>> decision = await client.get()
    >>> print(decision.num_prefill_workers, decision.num_decode_workers)
    >>> await client.complete(decision)

</details>

<details>
<summary><strong>`VirtualConnectorCoordinator` — Internal planner virtual connector component</strong></summary>

*Source: [`dynamo/_core.pyi#L1762`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1762)*

<b>Constructor</b>

#### `__init__(runtime: DistributedRuntime, dynamo_namespace: str, check_interval_secs: int, max_wait_time_secs: int, max_retries: int)`

*Source: [`dynamo/_core.pyi#L1765`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1765)*

<b>Methods</b>

#### `async_init()`

*Source: [`dynamo/_core.pyi#L1768`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1768)*

Call this before using the object
#### `read_state() -> PlannerDecision`

*Source: [`dynamo/_core.pyi#L1772`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1772)*

Get the current values. Most for test / debug.
#### `update_scaling_decision(num_prefill: Optional[int] = None, num_decode: Optional[int] = None)`

*Source: [`dynamo/_core.pyi#L1776`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1776)*

#### `wait_for_scaling_completion()`

*Source: [`dynamo/_core.pyi#L1779`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1779)*


</details>

<details>
<summary><strong>`VirtualConnectorClient` — How a client discovers planner requests and marks them complete</strong></summary>

*Source: [`dynamo/_core.pyi#L1782`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1782)*

<b>Constructor</b>

#### `__init__(runtime: DistributedRuntime, dynamo_namespace: str)`

*Source: [`dynamo/_core.pyi#L1785`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1785)*

<b>Methods</b>

#### `get() -> PlannerDecision`

*Source: [`dynamo/_core.pyi#L1788`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1788)*

#### `complete(decision: PlannerDecision)`

*Source: [`dynamo/_core.pyi#L1791`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1791)*

#### `wait()`

*Source: [`dynamo/_core.pyi#L1794`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1794)*

Blocks until there is a new decision to fetch using 'get'

</details>

### Data Models

<details>
<summary><strong>`ModelDeploymentCard` — A model deployment card is a collection of model information</strong></summary>

*Source: [`dynamo/_core.pyi#L431`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L431)*

Examples:
    >>> from dynamo._core import ModelDeploymentCard
    >>> card = ModelDeploymentCard()
    >>> json_str = card.to_json_str()
    >>> restored = ModelDeploymentCard.from_json_str(json_str)
<b>Methods</b>

#### `to_json_str() -> str`

*Source: [`dynamo/_core.pyi#L441`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L441)*

Serialize the model deployment card to a JSON string.
#### `from_json_str(json: str) -> ModelDeploymentCard`

*Source: [`dynamo/_core.pyi#L445`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L445)*

Deserialize a model deployment card from a JSON string.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `json` | `str` |  |  |


</details>

<details>
<summary><strong>`KvbmRequest` — A request for KV cache</strong></summary>

*Source: [`dynamo/_core.pyi#L1483`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1483)*

<b>Constructor</b>

#### `__init__(request_id: int, tokens: List[int], block_size: int)`

*Source: [`dynamo/_core.pyi#L1488`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1488)*


</details>

### Functions

<details>
<summary><strong>`log_message()` — Log a message from Python with file and line info</strong></summary>

#### `log_message(level: str, message: str, module: str, file: str, line: int)`

*Source: [`dynamo/_core.pyi#L21`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L21)*

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `level` | `str` |  |  |
| `message` | `str` |  |  |
| `module` | `str` |  |  |
| `file` | `str` |  |  |
| `line` | `int` |  |  |


</details>

<details>
<summary><strong>`compute_block_hash_for_seq()` — Compute block hashes for a sequence of tokens, optionally including multimodal metadata.</strong></summary>

#### `compute_block_hash_for_seq(tokens: List[int], kv_block_size: int, block_mm_infos: Optional[List[Optional[Dict[str, Any]]]] = None, lora_name: Optional[str] = None) -> List[int]`

*Source: [`dynamo/_core.pyi#L258`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L258)*

When block_mm_infos is provided, the mm_hashes are included in the hash computation
to ensure that blocks with identical tokens but different multimodal objects produce
different hashes.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `tokens` | `List[int]` | List of token IDs |
| `kv_block_size` | `int` | Size of each block in tokens |
| `block_mm_infos` | `Optional[List[Optional[Dict[str, Any]]]]` | Optional per-block multimodal metadata. Each element corresponds to a block            and should be None or a dict with structure:            \{                "mm_objects": [                    `{                        "mm_hash": int,  # Hash of the MM object                    }`                ]            \} |

<b>Returns:</b> `List[int]` -- List of block hashes (one per block)

Examples:
    >>> tokens = [1, 2, 3, 4] * 8  # 32 tokens = 1 block
    >>> hashes = compute_block_hash_for_seq(tokens, 32)
    >>> mm_info = \{"mm_objects": [`{"mm_hash": 0xDEADBEEF}`]\}
    >>> hashes_mm = compute_block_hash_for_seq(tokens, 32, [mm_info])
    >>> hashes_lora = compute_block_hash_for_seq(tokens, 32, lora_name="my-adapter")

</details>

<details>
<summary><strong>`register_model()` — Attach the model at path to the given endpoint, and advertise it as model_type.</strong></summary>

#### `register_model(model_input: ModelInput, model_type: ModelType, endpoint: Endpoint, model_path: str, model_name: Optional[str] = None, context_length: Optional[int] = None, kv_cache_block_size: Optional[int] = None, router_mode: Optional[RouterMode] = None, runtime_config: Optional[ModelRuntimeConfig] = None, user_data: Optional[Dict[str, Any]] = None, custom_template_path: Optional[str] = None, lora_name: Optional[str] = None, base_model_path: Optional[str] = None)`

*Source: [`dynamo/_core.pyi#L1163`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1163)*

LoRA Registration:
    The `lora_name` and `base_model_path` parameters must be provided together or not at all.
    Providing only one of these parameters will raise a ValueError.
    - `lora_name`: The served model name for the LoRA model
    - `base_model_path`: Path to the base model that the LoRA extends

For TensorBased models (using ModelInput.Tensor), HuggingFace downloads are skipped
and a minimal model card is registered directly. Use model_path as the display name
for these models.
Examples:
    >>> from dynamo._core import ModelInput, ModelType, register_model
    >>> await register_model(
    ...     ModelInput.Tokens,
    ...     ModelType.Chat | ModelType.Completions,
    ...     endpoint,
    ...     "Qwen/Qwen3-0.6B",
    ... )
    >>> await register_model(
    ...     ModelInput.Tensor, ModelType.TensorBased,
    ...     endpoint, "echo",
    ...     runtime_config=runtime_config,
    ... )

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_input` | `ModelInput` |  |  |
| `model_type` | `ModelType` |  |  |
| `endpoint` | `Endpoint` |  |  |
| `model_path` | `str` |  |  |
| `model_name` | `Optional[str]` | None |  |
| `context_length` | `Optional[int]` | None |  |
| `kv_cache_block_size` | `Optional[int]` | None |  |
| `router_mode` | `Optional[RouterMode]` | None |  |
| `runtime_config` | `Optional[ModelRuntimeConfig]` | None |  |
| `user_data` | `Optional[Dict[str, Any]]` | None |  |
| `custom_template_path` | `Optional[str]` | None |  |
| `lora_name` | `Optional[str]` | None |  |
| `base_model_path` | `Optional[str]` | None |  |


</details>

<details>
<summary><strong>`unregister_model()` — Unregister a model from the discovery system.</strong></summary>

#### `unregister_model(endpoint: Endpoint, lora_name: Optional[str] = None)`

*Source: [`dynamo/_core.pyi#L1205`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1205)*

If lora_name is provided, unregisters a LoRA adapter instead of a base model.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `endpoint` | `Endpoint` |  |  |
| `lora_name` | `Optional[str]` | None |  |


</details>

<details>
<summary><strong>`lora_name_to_id()` — Generate a deterministic integer ID from a LoRA name using blake3 hash.</strong></summary>

#### `lora_name_to_id(lora_name: str) -> int`

*Source: [`dynamo/_core.pyi#L1216`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1216)*

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `lora_name` | `str` |  |  |


</details>

<details>
<summary><strong>`fetch_model()` — Download a model from Hugging Face, returning its local path.</strong></summary>

#### `fetch_model(remote_name: str, ignore_weights: bool = False) -> str`

*Source: [`dynamo/_core.pyi#L1220`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1220)*

If `ignore_weights` is True, only fetches tokenizer and config files.
Examples:
    >>> model_path = await fetch_model("Qwen/Qwen3-0.6B")
    >>> tokenizer_path = await fetch_model("meta-llama/Llama-3-8B", ignore_weights=True)

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `remote_name` | `str` |  |  |
| `ignore_weights` | `bool` | False |  |


</details>

<details>
<summary><strong>`make_engine()` — Make an engine matching the args</strong></summary>

#### `make_engine(distributed_runtime: DistributedRuntime, args: EntrypointArgs) -> EngineConfig`

*Source: [`dynamo/_core.pyi#L1245`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1245)*

Examples:
    >>> from dynamo._core import EntrypointArgs, EngineType, make_engine
    >>> args = EntrypointArgs(engine_type=EngineType.Dynamic, model_path="/models/llama")
    >>> engine_config = await make_engine(runtime, args)

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `distributed_runtime` | `DistributedRuntime` |  |  |
| `args` | `EntrypointArgs` |  |  |


</details>

<details>
<summary><strong>`run_input()` — Start an engine, connect it to an input, and run until stopped.</strong></summary>

#### `run_input(runtime: DistributedRuntime, input: str, engine_config: EngineConfig)`

*Source: [`dynamo/_core.pyi#L1254`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/_core.pyi#L1254)*

Examples:
    >>> from dynamo._core import run_input
    >>> await run_input(runtime, "http", engine_config)
    >>> await run_input(runtime, "grpc", engine_config)
    >>> await run_input(runtime, "text", engine_config)

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `runtime` | `DistributedRuntime` |  |  |
| `input` | `str` |  |  |
| `engine_config` | `EngineConfig` |  |  |


</details>

---

## dynamo.runtime

Decorators and utilities for defining Dynamo workers and endpoints.

### Classes

<details>
<summary><strong>`LogHandler` — Custom logging handler that sends log messages to the Rust env_logger</strong></summary>

*Source: [`dynamo/runtime/logging.py#L26`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/runtime/logging.py#L26)*

<b>Bases:</b> `logging.Handler`

<b>Methods</b>

#### `emit(record)`

*Source: [`dynamo/runtime/logging.py#L31`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/runtime/logging.py#L31)*

Emit a log record

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `record` | `Any` |  |  |


</details>

<details>
<summary><strong>`VllmColorFormatter` — Formatter that matches Rust tracing's compact colored output style.</strong></summary>

*Source: [`dynamo/runtime/logging.py#L63`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/runtime/logging.py#L63)*

Used for vLLM logs routed through a StreamHandler (bypassing the Rust
bridge) so that VLLM_LOGGING_LEVEL is respected independently of DYN_LOG
while still producing visually consistent colored output.
<b>Bases:</b> `logging.Formatter`

<b>Methods</b>

#### `format(record)`

*Source: [`dynamo/runtime/logging.py#L81`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/runtime/logging.py#L81)*


</details>

### Functions

<details>
<summary><strong>`dynamo_worker()` — Decorator that creates a DistributedRuntime and passes it to the worker function.</strong></summary>

#### `dynamo_worker(enable_nats: bool = True)`

*Source: [`dynamo/runtime/__init__.py#L19`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/runtime/__init__.py#L19)*

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `enable_nats` | `bool` | Whether to enable NATS for KV events. Defaults to True.         If request_plane is "nats", NATS is always enabled.         Pass False (via --no-kv-events flag) to disable NATS initialization. |

Examples:
    >>> from dynamo.runtime import DistributedRuntime, dynamo_worker
    >>>
    >>> @dynamo_worker()
    ... async def worker(runtime: DistributedRuntime):
    ...     endpoint = runtime.endpoint("dynamo.backend.generate")
    ...     await endpoint.serve_endpoint(handler.generate)
    >>>
    >>> asyncio.run(worker())

</details>

<details>
<summary><strong>`dynamo_endpoint()` — Decorator that validates request/response Pydantic models on an async generator endpoint.</strong></summary>

#### `dynamo_endpoint(request_model: Union[Type[BaseModel], Type[Any]], response_model: Type[BaseModel]) -> Callable`

*Source: [`dynamo/runtime/__init__.py#L70`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/runtime/__init__.py#L70)*

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `request_model` | `Union[Type[BaseModel], Type[Any]]` | Pydantic model class (or Any) for incoming requests. |
| `response_model` | `Type[BaseModel]` | Pydantic model class for streamed response items. |

Examples:
    >>> from pydantic import BaseModel
    >>> from dynamo.runtime import dynamo_endpoint
    >>>
    >>> class Request(BaseModel):
    ...     data: str
    >>> class Response(BaseModel):
    ...     char: str
    >>>
    >>> class RequestHandler:
    ...     @dynamo_endpoint(Request, Response)
    ...     async def generate(self, request):
    ...         for char in request.data:
    ...             yield char

</details>

<details>
<summary><strong>`configure_logger()` — Called once to configure the Python logger to use the LogHandler</strong></summary>

#### `configure_logger(service_name: str | None, worker_id: int | None)`

*Source: [`dynamo/runtime/logging.py#L101`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/runtime/logging.py#L101)*

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `service_name` | `str | None` |  |  |
| `worker_id` | `int | None` |  |  |


</details>

<details>
<summary><strong>`configure_dynamo_logging()` — A single place to configure logging for Dynamo.</strong></summary>

#### `configure_dynamo_logging(service_name: str | None = None, worker_id: int | None = None)`

*Source: [`dynamo/runtime/logging.py#L130`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/runtime/logging.py#L130)*

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `service_name` | `str | None` | None |  |
| `worker_id` | `int | None` | None |  |


</details>

<details>
<summary><strong>`log_level_mapping()` — The DYN_LOG variable is set using &quot;debug&quot; or &quot;trace&quot; or &quot;info.</strong></summary>

#### `log_level_mapping(level: str) -> int`

*Source: [`dynamo/runtime/logging.py#L162`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/runtime/logging.py#L162)*

This function maps those to the appropriate logging level and defaults to INFO
if the variable is not set or a bad value.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `level` | `str` |  |  |


</details>

<details>
<summary><strong>`configure_sglang_logging()` — SGLang allows us to create a custom logging config file</strong></summary>

#### `configure_sglang_logging(dyn_level: int)`

*Source: [`dynamo/runtime/logging.py#L184`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/runtime/logging.py#L184)*

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `dyn_level` | `int` |  |  |


</details>

<details>
<summary><strong>`configure_vllm_logging()` — Configure vLLM logging for the main process and subprocesses.</strong></summary>

#### `configure_vllm_logging(dyn_level: int)`

*Source: [`dynamo/runtime/logging.py#L216`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/runtime/logging.py#L216)*

Main process: replaces vLLM's StreamHandler with a new StreamHandler that
uses VllmColorFormatter and writes directly to stderr.  This bypasses the
Rust LogHandler bridge so that VLLM_LOGGING_LEVEL is respected independently
of DYN_LOG (the Rust bridge filters based on DYN_LOG).

Subprocesses (EngineCore, workers): use vLLM's DEFAULT_LOGGING_CONFIG
(StreamHandler to stderr) since the Rust runtime is not initialized there.
Setting VLLM_CONFIGURE_LOGGING=1 without VLLM_LOGGING_CONFIG_PATH causes
vLLM to use its built-in default config in spawned subprocesses.

The dyn_level param is kept for signature compatibility but does not control
the vLLM logger level. Use VLLM_LOGGING_LEVEL env var instead.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `dyn_level` | `int` |  |  |


</details>

---

## dynamo.llm

LLM serving pipeline components and engine integration.

---

## dynamo.frontend

HTTP frontend configuration and OpenAI-compatible API gateway.

### Configuration

<details>
<summary><strong>`KvRouterArgGroup` — CLI arguments for the 16 KvRouterConfig parameters.</strong></summary>

*Source: [`dynamo/common/configuration/groups/kv_router_args.py#L81`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/configuration/groups/kv_router_args.py#L81)*

# confidence: high
Examples:
    >>> import argparse
    >>> from dynamo.common.configuration.groups.kv_router_args import KvRouterArgGroup
    >>>
    >>> parser = argparse.ArgumentParser()
    >>> KvRouterArgGroup().add_arguments(parser)
    >>> args = parser.parse_args([
    ...     "--router-kv-overlap-score-weight", "0.8",
    ...     "--router-temperature", "0.5",
    ... ])
    >>> args.overlap_score_weight
    0.8
<b>Bases:</b> `ArgGroup`

<b>Methods</b>

#### `add_arguments(parser)`

*Source: [`dynamo/common/configuration/groups/kv_router_args.py#L99`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/configuration/groups/kv_router_args.py#L99)*


</details>

<details>
<summary><strong>`KvRouterConfigBase` — Mixin carrying the 16 KvRouterConfig fields.</strong></summary>

*Source: [`dynamo/common/configuration/groups/kv_router_args.py#L40`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/configuration/groups/kv_router_args.py#L40)*

# confidence: high
Examples:
    >>> import argparse
    >>> from dynamo.common.configuration.groups.kv_router_args import (
    ...     KvRouterArgGroup,
    ...     KvRouterConfigBase,
    ... )
    >>>
    >>> class MyRouterConfig(KvRouterConfigBase):
    ...     endpoint: str
    >>>
    >>> kwargs = my_config.kv_router_kwargs()
    >>> kwargs["overlap_score_weight"]
    1.0
<b>Bases:</b> `ConfigBase`

<b>Attributes</b>

- `overlap_score_weight`: `float`
- `router_temperature`: `float`
- `use_kv_events`: `bool`
- `durable_kv_events`: `bool`
- `router_replica_sync`: `bool`
- `router_track_active_blocks`: `bool`
- `router_track_output_blocks`: `bool`
- `router_assume_kv_reuse`: `bool`
- `router_snapshot_threshold`: `int`
- `router_reset_states`: `bool`
- `router_ttl_secs`: `float`
- `router_max_tree_size`: `int`
- `router_prune_target_ratio`: `float`
- `router_queue_threshold`: `Optional[float]`
- `router_event_threads`: `int`
- `router_enable_cache_control`: `bool`

<b>Methods</b>

#### `kv_router_kwargs() -> dict`

*Source: [`dynamo/common/configuration/groups/kv_router_args.py#L76`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/configuration/groups/kv_router_args.py#L76)*

Return a dict suitable for `0`.

</details>

<details>
<summary><strong>`FrontendArgGroup` — Frontend configuration parameters.</strong></summary>

*Source: [`dynamo/frontend/frontend_args.py#L114`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/frontend_args.py#L114)*

# confidence: high
Examples:
    >>> import argparse
    >>> from dynamo.frontend.frontend_args import FrontendArgGroup
    >>>
    >>> parser = argparse.ArgumentParser()
    >>> group = FrontendArgGroup()
    >>> group.add_arguments(parser)
    >>> args = parser.parse_args([
    ...     "--model-name", "Llama-3.2-1B-Instruct",
    ...     "--router-mode", "kv",
    ...     "--http-port", "8080",
    ... ])
    >>> args.model_name
    'Llama-3.2-1B-Instruct'
<b>Bases:</b> `ArgGroup`

<b>Methods</b>

#### `add_arguments(parser)`

*Source: [`dynamo/frontend/frontend_args.py#L134`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/frontend_args.py#L134)*


</details>

<details>
<summary><strong>`FrontendArgGroup` — Frontend configuration parameters.</strong></summary>

*Source: [`dynamo/frontend/frontend_args.py#L114`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/frontend_args.py#L114)*

# confidence: high
Examples:
    >>> import argparse
    >>> from dynamo.frontend.frontend_args import FrontendArgGroup
    >>>
    >>> parser = argparse.ArgumentParser()
    >>> group = FrontendArgGroup()
    >>> group.add_arguments(parser)
    >>> args = parser.parse_args([
    ...     "--model-name", "Llama-3.2-1B-Instruct",
    ...     "--router-mode", "kv",
    ...     "--http-port", "8080",
    ... ])
    >>> args.model_name
    'Llama-3.2-1B-Instruct'
<b>Bases:</b> `ArgGroup`

<b>Methods</b>

#### `add_arguments(parser)`

*Source: [`dynamo/frontend/frontend_args.py#L134`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/frontend_args.py#L134)*


</details>

### Classes

<details>
<summary><strong>`FrontendConfig` — Configuration for the Dynamo frontend.</strong></summary>

*Source: [`dynamo/frontend/frontend_args.py#L42`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/frontend_args.py#L42)*

# confidence: high — adapted from real call site
Examples:
    >>> import argparse
    >>> from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig
    >>>
    >>> parser = argparse.ArgumentParser()
    >>> FrontendArgGroup().add_arguments(parser)
    >>> args = parser.parse_args([
    ...     "--model-name", "Llama-3.2-1B-Instruct",
    ...     "--http-port", "8080",
    ... ])
    >>> config = FrontendConfig.from_cli_args(args)
    >>> config.model_name
    'Llama-3.2-1B-Instruct'
    >>> config.http_port
    8080
<b>Bases:</b> `KvRouterConfigBase`

<b>Attributes</b>

- `interactive`: `bool`
- `kv_cache_block_size`: `Optional[int]`
- `http_host`: `str`
- `http_port`: `int`
- `tls_cert_path`: `Optional[pathlib.Path]`
- `tls_key_path`: `Optional[pathlib.Path]`
- `router_mode`: `str`
- `namespace`: `Optional[str]`
- `namespace_prefix`: `Optional[str]`
- `decode_fallback`: `bool`
- `migration_limit`: `int`
- `active_decode_blocks_threshold`: `Optional[float]`
- `active_prefill_tokens_threshold`: `Optional[int]`
- `active_prefill_tokens_threshold_frac`: `Optional[float]`
- `model_name`: `Optional[str]`
- `model_path`: `Optional[str]`
- `metrics_prefix`: `Optional[str]`
- `kserve_grpc_server`: `bool`
- `grpc_metrics_port`: `int`
- `dump_config_to`: `Optional[str]`
- `discovery_backend`: `str`
- `request_plane`: `str`
- `event_plane`: `str`
- `chat_processor`: `str`
- `enable_anthropic_api`: `bool`
- `debug_perf`: `bool`
- `preprocess_workers`: `int`

<b>Methods</b>

#### `validate()`

*Source: [`dynamo/frontend/frontend_args.py#L95`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/frontend_args.py#L95)*


</details>

<details>
<summary><strong>`ArgGroup` — Base interface for configuration groups.</strong></summary>

*Source: [`dynamo/common/configuration/arg_group.py#L8`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/configuration/arg_group.py#L8)*

Each ArgGroup represents a domain of configuration parameters with clear ownership.

# confidence: high — adapted from real call site
Examples:
    >>> import argparse
    >>> from dynamo.common.configuration.arg_group import ArgGroup
    >>> from dynamo.common.configuration.utils import add_argument
    >>>
    >>> class MyArgGroup(ArgGroup):
    ...     def add_arguments(self, parser) -> None:
    ...         add_argument(
    ...             parser,
    ...             flag_name="--my-option",
    ...             env_var="DYN_MY_OPTION",
    ...             default="value",
    ...             help="A custom option.",
    ...         )
    >>>
    >>> parser = argparse.ArgumentParser()
    >>> MyArgGroup().add_arguments(parser)
<b>Methods</b>

#### `add_arguments(parser)`

*Source: [`dynamo/common/configuration/arg_group.py#L34`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/configuration/arg_group.py#L34)*

Register CLI arguments owned by this group.

This method must be side-effect free beyond parser mutation.
It must not depend on runtime state or other groups.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `parser` | `Any` | argparse.ArgumentParser or argument group |

</details>

<details>
<summary><strong>`FrontendConfig` — Configuration for the Dynamo frontend.</strong></summary>

*Source: [`dynamo/frontend/frontend_args.py#L42`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/frontend_args.py#L42)*

# confidence: high — adapted from real call site
Examples:
    >>> import argparse
    >>> from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig
    >>>
    >>> parser = argparse.ArgumentParser()
    >>> FrontendArgGroup().add_arguments(parser)
    >>> args = parser.parse_args([
    ...     "--model-name", "Llama-3.2-1B-Instruct",
    ...     "--http-port", "8080",
    ... ])
    >>> config = FrontendConfig.from_cli_args(args)
    >>> config.model_name
    'Llama-3.2-1B-Instruct'
    >>> config.http_port
    8080
<b>Bases:</b> `KvRouterConfigBase`

<b>Attributes</b>

- `interactive`: `bool`
- `kv_cache_block_size`: `Optional[int]`
- `http_host`: `str`
- `http_port`: `int`
- `tls_cert_path`: `Optional[pathlib.Path]`
- `tls_key_path`: `Optional[pathlib.Path]`
- `router_mode`: `str`
- `namespace`: `Optional[str]`
- `namespace_prefix`: `Optional[str]`
- `decode_fallback`: `bool`
- `migration_limit`: `int`
- `active_decode_blocks_threshold`: `Optional[float]`
- `active_prefill_tokens_threshold`: `Optional[int]`
- `active_prefill_tokens_threshold_frac`: `Optional[float]`
- `model_name`: `Optional[str]`
- `model_path`: `Optional[str]`
- `metrics_prefix`: `Optional[str]`
- `kserve_grpc_server`: `bool`
- `grpc_metrics_port`: `int`
- `dump_config_to`: `Optional[str]`
- `discovery_backend`: `str`
- `request_plane`: `str`
- `event_plane`: `str`
- `chat_processor`: `str`
- `enable_anthropic_api`: `bool`
- `debug_perf`: `bool`
- `preprocess_workers`: `int`

<b>Methods</b>

#### `validate()`

*Source: [`dynamo/frontend/frontend_args.py#L95`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/frontend_args.py#L95)*


</details>

<details>
<summary><strong>`FrontendConfig` — Configuration for the Dynamo frontend.</strong></summary>

*Source: [`dynamo/frontend/frontend_args.py#L42`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/frontend_args.py#L42)*

# confidence: high — adapted from real call site
Examples:
    >>> import argparse
    >>> from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig
    >>>
    >>> parser = argparse.ArgumentParser()
    >>> FrontendArgGroup().add_arguments(parser)
    >>> args = parser.parse_args([
    ...     "--model-name", "Llama-3.2-1B-Instruct",
    ...     "--http-port", "8080",
    ... ])
    >>> config = FrontendConfig.from_cli_args(args)
    >>> config.model_name
    'Llama-3.2-1B-Instruct'
    >>> config.http_port
    8080
<b>Bases:</b> `KvRouterConfigBase`

<b>Attributes</b>

- `interactive`: `bool`
- `kv_cache_block_size`: `Optional[int]`
- `http_host`: `str`
- `http_port`: `int`
- `tls_cert_path`: `Optional[pathlib.Path]`
- `tls_key_path`: `Optional[pathlib.Path]`
- `router_mode`: `str`
- `namespace`: `Optional[str]`
- `namespace_prefix`: `Optional[str]`
- `decode_fallback`: `bool`
- `migration_limit`: `int`
- `active_decode_blocks_threshold`: `Optional[float]`
- `active_prefill_tokens_threshold`: `Optional[int]`
- `active_prefill_tokens_threshold_frac`: `Optional[float]`
- `model_name`: `Optional[str]`
- `model_path`: `Optional[str]`
- `metrics_prefix`: `Optional[str]`
- `kserve_grpc_server`: `bool`
- `grpc_metrics_port`: `int`
- `dump_config_to`: `Optional[str]`
- `discovery_backend`: `str`
- `request_plane`: `str`
- `event_plane`: `str`
- `chat_processor`: `str`
- `enable_anthropic_api`: `bool`
- `debug_perf`: `bool`
- `preprocess_workers`: `int`

<b>Methods</b>

#### `validate()`

*Source: [`dynamo/frontend/frontend_args.py#L95`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/frontend_args.py#L95)*


</details>

### Data Models

<details>
<summary><strong>`PreprocessWorkerResult` — Picklable return value from the preprocess worker.</strong></summary>

*Source: [`dynamo/frontend/vllm_processor.py#L96`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/vllm_processor.py#L96)*

<b>Attributes</b>

- `dynamo_preproc`: `dict[str, Any]`
- `tokens`: `list[int]`
- `vllm_preproc`: `EngineCoreRequest`
- `sampling_params`: `SamplingParams`
- `request_for_sampling`: `Any`
- `chat_template_kwargs`: `dict[str, Any]`

<b>Constructor</b>

#### `__init__(dynamo_preproc: dict[str, Any], tokens: list[int], vllm_preproc: EngineCoreRequest, sampling_params: SamplingParams, request_for_sampling: Any, chat_template_kwargs: dict[str, Any])`

*Source: [`dynamo/frontend/vllm_processor.py`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/vllm_processor.py)*


</details>

### Functions

<details>
<summary><strong>`preprocess_chat_request_sync()` — Sync version of preprocess_chat_request for worker processes.</strong></summary>

#### `preprocess_chat_request_sync(request: dict[str, Any] | ChatCompletionRequest, tokenizer: TokenizerLike, renderer, tool_parser_class: type[ToolParser] | None) -> PreprocessResult`

*Source: [`dynamo/frontend/prepost.py#L185`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/prepost.py#L185)*

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `request` | `dict[str, Any] | ChatCompletionRequest` |  |  |
| `tokenizer` | `TokenizerLike` |  |  |
| `renderer` | `Any` |  |  |
| `tool_parser_class` | `type[ToolParser] | None` |  |  |


</details>

<details>
<summary><strong>`register_encoder()` — Decorator to register custom encoders for specific types.</strong></summary>

#### `register_encoder(type_class)`

*Source: [`dynamo/common/config_dump/config_dumper.py#L209`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/config_dump/config_dumper.py#L209)*

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `type_class` | `Any` |  |  |


> [!NOTE]
> @register_encoder(MyClass)
def encode_my_class(obj: MyClass):
    return `{"field": obj.field}`

</details>

<details>
<summary><strong>`add_argument()` — Add a CLI argument with env var default, optional alias and dest, and help message construction.</strong></summary>

#### `add_argument(parser, flag_name: str, env_var: str, default: Any, help: str, obsolete_flag: Optional[str] = None, arg_type: Optional[Union[type, Callable[..., Any]]] = str, kwargs: Any = {})`

*Source: [`dynamo/common/configuration/utils.py#L57`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/configuration/utils.py#L57)*

# confidence: high — adapted from real call site
Examples:
    >>> import argparse
    >>> from dynamo.common.configuration.utils import add_argument
    >>>
    >>> parser = argparse.ArgumentParser()
    >>> add_argument(
    ...     parser,
    ...     flag_name="--http-port",
    ...     env_var="DYN_HTTP_PORT",
    ...     default=8000,
    ...     help="HTTP port for the engine.",
    ...     arg_type=int,
    ... )
    >>> args = parser.parse_args([])
    >>> args.http_port
    8000

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `parser` | `Any` | ArgumentParser or argument group |
| `flag_name` | `str` | Primary flag (must start with '--', e.g., "--foo") |
| `env_var` | `str` | Environment variable name (e.g., "DYN_FOO") |
| `default` | `Any` | Default value |
| `help` | `str` | Help text |
| `alias` | `Any` | Optional alias for the flag (must start with '--') |
| `obsolete_flag` | `Optional[str]` | Optional obsolete/legacy flag (for help msg only, must start with '--') |
| `dest` | `Any` | Optional destination name (defaults to flag_name with dashes replaced by underscores) |
| `choices` | `Any` | Optional list of valid values for the argument. |
| `arg_type` | `Optional[Union[type, Callable[..., Any]]]` | Type for the argument (default: str) |

</details>

<details>
<summary><strong>`add_negatable_bool_argument()` — Add negatable boolean flag (--foo / --no-foo).</strong></summary>

#### `add_negatable_bool_argument(parser, flag_name: str, env_var: str, default: bool, help: str, dest: Optional[str] = None, obsolete_flag: Optional[str] = None)`

*Source: [`dynamo/common/configuration/utils.py#L129`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/configuration/utils.py#L129)*

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `parser` | `Any` | ArgumentParser or argument group |
| `flag_name` | `str` | Primary flag (must start with '--', e.g. "--enable-feature") |
| `env_var` | `str` | Environment variable name (e.g., "DYN_ENABLE_FEATURE") |
| `default` | `bool` | Default value |
| `help` | `str` | Help text |
| `dest` | `Optional[str]` | Optional destination name for the parsed value |
| `obsolete_flag` | `Optional[str]` | Optional obsolete/legacy flag (for help msg only, must start with '--') |

</details>

<details>
<summary><strong>`env_or_default()` — Get value from environment variable or return default.</strong></summary>

#### `env_or_default(env_var: str, default: T, value_type: Optional[Union[type, Callable[..., Any]]] = None) -> T`

*Source: [`dynamo/common/configuration/utils.py#L13`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/configuration/utils.py#L13)*

Performs type conversion based on the default value's type.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `env_var` | `str` | Environment variable name (e.g., "DYN_NAMESPACE") |
| `default` | `T` | Default value if env var not set |
| `value_type` | `Optional[Union[type, Callable[..., Any]]]` | If provided, use this type to convert the env value. If None, the type |

<b>Returns:</b> `T` -- Environment variable value (type-converted) or default

</details>

<details>
<summary><strong>`validate_model_name()` — Validate that model-name is a non-empty string.</strong></summary>

#### `validate_model_name(value: str) -> str`

*Source: [`dynamo/frontend/frontend_args.py#L24`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/frontend_args.py#L24)*

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `value` | `str` |  |  |


</details>

<details>
<summary><strong>`validate_model_path()` — Validate that model-path is a valid directory on disk.</strong></summary>

#### `validate_model_path(value: str) -> str`

*Source: [`dynamo/frontend/frontend_args.py#L33`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/frontend_args.py#L33)*

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `value` | `str` |  |  |


</details>

<details>
<summary><strong>`preprocess_chat_request_sync()` — Sync version of preprocess_chat_request for worker processes.</strong></summary>

#### `preprocess_chat_request_sync(request: dict[str, Any] | ChatCompletionRequest, tokenizer: TokenizerLike, renderer, tool_parser_class: type[ToolParser] | None) -> PreprocessResult`

*Source: [`dynamo/frontend/prepost.py#L185`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/prepost.py#L185)*

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `request` | `dict[str, Any] | ChatCompletionRequest` |  |  |
| `tokenizer` | `TokenizerLike` |  |  |
| `renderer` | `Any` |  |  |
| `tool_parser_class` | `type[ToolParser] | None` |  |  |


</details>

<details>
<summary><strong>`enter_generator()` — Increment active request count. Returns current count.</strong></summary>

#### `enter_generator() -> int`

*Source: [`dynamo/frontend/perf_instrumentation.py#L24`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/perf_instrumentation.py#L24)*

Safe without a lock: only called while the GIL is held (all callers are
in Python code), so the read-modify-write on the global int is atomic
with respect to other Python threads.

</details>

<details>
<summary><strong>`exit_generator()` — Decrement active request count. Returns current count.</strong></summary>

#### `exit_generator() -> int`

*Source: [`dynamo/frontend/perf_instrumentation.py#L39`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/perf_instrumentation.py#L39)*


</details>

<details>
<summary><strong>`dump_config()` — Dump the configuration to a file or stdout.</strong></summary>

#### `dump_config(dump_config_to: Optional[str], config: Any)`

*Source: [`dynamo/common/config_dump/config_dumper.py#L72`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/config_dump/config_dumper.py#L72)*

If dump_config_to is not provided, the config will be logged to stdout at VERBOSE level.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `dump_config_to` | `Optional[str]` | Optional path to dump the config to. If None, logs to stdout. |
| `config` | `Any` | The configuration object to dump (must be JSON-serializable). |

</details>

<details>
<summary><strong>`setup_engine_factory()` — When using vllm pre and post processor, create the EngineFactory that</strong></summary>

#### `setup_engine_factory(runtime: DistributedRuntime, router_config: RouterConfig, config: FrontendConfig, vllm_flags: Namespace)`

*Source: [`dynamo/frontend/main.py#L48`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/main.py#L48)*

creates the engines that run requests.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `runtime` | `DistributedRuntime` |  |  |
| `router_config` | `RouterConfig` |  |  |
| `config` | `FrontendConfig` |  |  |
| `vllm_flags` | `Namespace` |  |  |


</details>

<details>
<summary><strong>`parse_args()` — Parse command-line arguments for the Dynamo frontend.</strong></summary>

#### `parse_args() -> tuple[FrontendConfig, Optional[Namespace]]`

*Source: [`dynamo/frontend/main.py#L63`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/main.py#L63)*

<b>Returns:</b> `tuple[FrontendConfig, Optional[Namespace]]` -- Parsed configuration object.

</details>

<details>
<summary><strong>`async_main()` — Main async entry point for the Dynamo frontend.</strong></summary>

#### `async_main()`

*Source: [`dynamo/frontend/main.py#L115`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/main.py#L115)*

Initializes the distributed runtime, configures routing, and starts
the HTTP server or interactive mode based on command-line arguments.

</details>

<details>
<summary><strong>`graceful_shutdown()` — Handle graceful shutdown of the distributed runtime.</strong></summary>

#### `graceful_shutdown(runtime)`

*Source: [`dynamo/frontend/main.py#L248`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/main.py#L248)*

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `runtime` | `Any` | The DistributedRuntime instance to shut down. |

</details>

<details>
<summary><strong>`main()` — Entry point for the Dynamo frontend CLI.</strong></summary>

#### `main()`

*Source: [`dynamo/frontend/main.py#L257`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/frontend/main.py#L257)*


</details>

---

## dynamo.common

Shared configuration, constants, storage, and utility functions.

### Enums

<details>
<summary><strong>`DisaggregationMode` — Disaggregation mode for LLM workers.</strong></summary>

*Source: [`dynamo/common/constants.py#L9`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/constants.py#L9)*

# confidence: high
Examples:
    >>> from dynamo.common.constants import DisaggregationMode
    >>> DisaggregationMode.PREFILL.value
    'prefill'
    >>> DisaggregationMode("agg") == DisaggregationMode.AGGREGATED
    True
<b>Bases:</b> `Enum`

<b>Attributes</b>

- `AGGREGATED`
- `PREFILL`
- `DECODE`


</details>

<details>
<summary><strong>`EmbeddingTransferMode` — Embedding transfer mode for LLM workers.</strong></summary>

*Source: [`dynamo/common/constants.py#L26`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/constants.py#L26)*

# confidence: high
Examples:
    >>> from dynamo.common.constants import EmbeddingTransferMode
    >>> EmbeddingTransferMode.NIXL_WRITE.value
    'nixl-write'
    >>> EmbeddingTransferMode("local") == EmbeddingTransferMode.LOCAL
    True
<b>Bases:</b> `Enum`

<b>Attributes</b>

- `LOCAL`
- `NIXL_WRITE`
- `NIXL_READ`


</details>

### Configuration

<details>
<summary><strong>`TransferRequest` — Data class for transfer requests containing necessary information for embedding transfer.</strong></summary>

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L49`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L49)*

<b>Bases:</b> `BaseModel`

<b>Attributes</b>

- `embeddings_shape`: `List[int]`
- `embedding_dtype_str`: `str`
- `serialized_request`: `Any`


</details>

<details>
<summary><strong>`NvCreateVideoRequest` — Request for video generation (/v1/videos endpoint).</strong></summary>

*Source: [`dynamo/common/protocols/video_protocol.py#L44`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/protocols/video_protocol.py#L44)*

Matches Rust NvCreateVideoRequest in lib/llm/src/protocols/openai/videos.rs.
<b>Bases:</b> `BaseModel`

<b>Attributes</b>

- `prompt`: `str` -- The text prompt for video generation.
- `model`: `str` -- The model to use for video generation.
- `input_reference`: `Optional[str]` -- Optional image reference that guides generation (for I2V).
- `seconds`: `Optional[int]` -- Clip duration in seconds.
- `size`: `Optional[str]` -- Video size in WxH format (default: '832x480').
- `user`: `Optional[str]` -- Optional user identifier.
- `response_format`: `Optional[str]` -- Response format: 'url' or 'b64_json' (default: 'url').
- `nvext`: `Optional[VideoNvExt]` -- NVIDIA extensions.


</details>

<details>
<summary><strong>`NvVideosResponse` — Response structure for video generation.</strong></summary>

*Source: [`dynamo/common/protocols/video_protocol.py#L90`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/protocols/video_protocol.py#L90)*

Matches Rust NvVideosResponse in lib/llm/src/protocols/openai/videos.rs.
<b>Bases:</b> `BaseModel`

<b>Attributes</b>

- `id`: `str` -- Unique identifier for the response.
- `object`: `str` -- Object type (always 'video').
- `model`: `str` -- Model used for generation.
- `status`: `str` -- Generation status.
- `progress`: `int` -- Progress percentage (0-100).
- `created`: `int` -- Unix timestamp of creation.
- `data`: `list[VideoData]` -- List of generated videos.
- `error`: `Optional[str]` -- Error message if generation failed.
- `inference_time_s`: `Optional[float]` -- Inference time in seconds.


</details>

<details>
<summary><strong>`VideoData` — Video data in response.</strong></summary>

*Source: [`dynamo/common/protocols/video_protocol.py#L77`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/protocols/video_protocol.py#L77)*

Matches Rust VideoData in lib/llm/src/protocols/openai/videos.rs.
<b>Bases:</b> `BaseModel`

<b>Attributes</b>

- `url`: `Optional[str]` -- URL of the generated video (if response_format is 'url').
- `b64_json`: `Optional[str]` -- Base64-encoded video (if response_format is 'b64_json').


</details>

### Classes

<details>
<summary><strong>`ConfigBase` — Base configuration class that allows properties with and without defaults in arbitrary order.</strong></summary>

*Source: [`dynamo/common/configuration/config_base.py#L6`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/configuration/config_base.py#L6)*

# confidence: high — adapted from real call site
Examples:
    >>> import argparse
    >>> from dynamo.common.configuration.config_base import ConfigBase
    >>>
    >>> class MyConfig(ConfigBase):
    ...     model_name: str
    ...     http_port: int = 8000
    >>>
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument("--model-name", default="llama")
    >>> parser.add_argument("--http-port", type=int, default=8000)
    >>> args = parser.parse_args([])
    >>> config = MyConfig.from_cli_args(args)
    >>> config.http_port
    8000
<b>Methods</b>

#### `from_cli_args(args: argparse.Namespace)`

*Source: [`dynamo/common/configuration/config_base.py#L27`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/configuration/config_base.py#L27)*


</details>

<details>
<summary><strong>`MultimodalEmbeddingCacheManager` — LRU cache for encoder embeddings.</strong></summary>

*Source: [`dynamo/common/memory/multimodal_embedding_cache_manager.py#L34`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/memory/multimodal_embedding_cache_manager.py#L34)*

Stores tensors keyed by content hash with automatic eviction
when capacity is exceeded.

> [!NOTE]
> This class is NOT thread-safe. It is designed to run within a single
thread (e.g., an asyncio event loop). All access must be from the same
thread to avoid race conditions. This is intentional to keep the
implementation simple and avoid locking overhead.
<b>Attributes</b>

- `stats`: `dict` -- Get cache statistics.

<b>Constructor</b>

#### `__init__(capacity_bytes: int)`

*Source: [`dynamo/common/memory/multimodal_embedding_cache_manager.py#L48`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/memory/multimodal_embedding_cache_manager.py#L48)*

Initialize the encoder cache.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `capacity_bytes` | `int` | Maximum cache capacity in bytes. |
<b>Methods</b>

#### `get(key: str) -> Optional[CachedEmbedding]`

*Source: [`dynamo/common/memory/multimodal_embedding_cache_manager.py#L85`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/memory/multimodal_embedding_cache_manager.py#L85)*

Get a cached embedding from the cache.

If found, the entry is moved to the end (most recently used).

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `key` | `str` | Cache key (typically content hash). |

<b>Returns:</b> `Optional[CachedEmbedding]` -- The cached embedding, or None if not found.
#### `set(key: str, entry: CachedEmbedding) -> bool`

*Source: [`dynamo/common/memory/multimodal_embedding_cache_manager.py#L106`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/memory/multimodal_embedding_cache_manager.py#L106)*

Store a cached embedding in the cache.

If the key already exists, the old value is replaced.
If adding the entry would exceed capacity, LRU entries are evicted.
If the tensor itself is larger than capacity, it is not stored.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `key` | `str` | Cache key (typically content hash). |
| `entry` | `CachedEmbedding` | CachedEmbedding to cache. |

<b>Returns:</b> `bool` -- True if the entry was stored, False if it was too large.

</details>

<details>
<summary><strong>`AsyncEncoderCache` — Async wrapper with request coalescing over MultimodalEmbeddingCacheManager.</strong></summary>

*Source: [`dynamo/common/multimodal/async_encoder_cache.py#L46`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/async_encoder_cache.py#L46)*

Provides async get_or_compute that deduplicates concurrent requests
for the same key, ensuring only one encoding runs at a time per key.

> [!NOTE]
> This class is NOT thread-safe. It is designed to run within a single
asyncio event loop. All access must be from the same thread.
<b>Attributes</b>

- `stats`: `dict` -- Get cache statistics from underlying cache.

<b>Constructor</b>

#### `__init__(cache: MultimodalEmbeddingCacheManager)`

*Source: [`dynamo/common/multimodal/async_encoder_cache.py#L58`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/async_encoder_cache.py#L58)*

Initialize the async encoder cache.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `cache` | `MultimodalEmbeddingCacheManager` | Underlying MultimodalEmbeddingCacheManager for storage. |
<b>Methods</b>

#### `get(key: str) -> Optional[torch.Tensor]`

*Source: [`dynamo/common/multimodal/async_encoder_cache.py#L68`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/async_encoder_cache.py#L68)*

Synchronous get from underlying cache.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `key` | `str` | Cache key. |

<b>Returns:</b> `Optional[torch.Tensor]` -- Cached tensor or None if not found.
#### `get_or_compute(key: str, compute_fn: Callable[[], Awaitable[torch.Tensor]]) -> torch.Tensor`

*Source: [`dynamo/common/multimodal/async_encoder_cache.py#L80`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/async_encoder_cache.py#L80)*

Get from cache or compute with request coalescing.

If the key is in cache, returns immediately.
If another coroutine is already computing this key, waits for that result.
Otherwise, computes and caches the result.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `key` | `str` | Cache key (typically content hash). |
| `compute_fn` | `Callable[[], Awaitable[torch.Tensor]]` | Async function to compute the tensor if not cached. |

<b>Returns:</b> `torch.Tensor` -- The cached or computed tensor.

<b>Raises</b>

- `Exception` -- Re-raises any exception from compute_fn.

</details>

<details>
<summary><strong>`LocalEmbeddingReceiver` — Receiver that reads embeddings from a local file path provided in the serialized request.</strong></summary>

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L178`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L178)*

<b>Bases:</b> `AbstractEmbeddingReceiver`

<b>Attributes</b>

- `received_tensors`
- `tensor_id_counter`

<b>Constructor</b>

#### `__init__()`

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L183`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L183)*

<b>Methods</b>

#### `receive_embeddings(request: TransferRequest) -> tuple[int, torch.Tensor]`

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L188`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L188)*

Receive precomputed embeddings for a given request ID.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `request` | `TransferRequest` | The TransferRequest object containing information to receive embeddings for. |

<b>Returns:</b>

- `int` -- A tuple containing the tensor ID and the received embeddings as a torch.Tensor.
- `torch.Tensor` -- Caller should invoke release_tensor(tensor_id) when the tensor is no longer needed to free up resources.
#### `release_tensor(tensor_id: int)`

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L209`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L209)*

Indicate that the tensor associated with the ID is no longer in use.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `tensor_id` | `int` | The ID of the tensor to release. |

</details>

<details>
<summary><strong>`LocalEmbeddingSender` — Sender that saves embeddings to a local file and sends the file path as the serialized request.</strong></summary>

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L112`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L112)*

<b>Bases:</b> `AbstractEmbeddingSender`

<b>Attributes</b>

- `sender_id`
- `embedding_counter`

<b>Constructor</b>

#### `__init__()`

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L117`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L117)*

<b>Methods</b>

#### `save_embeddings_to_file(embedding_key: str, embeddings: torch.Tensor) -> str`

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L121`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L121)*

Save the embeddings to a local file and return the file path.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `embedding_key` | `str` | A unique key for the embeddings. |
| `embeddings` | `torch.Tensor` | A torch.Tensor of the embeddings to save. |

Returns:
    The file path where the embeddings are saved.
#### `send_embeddings(embeddings: torch.Tensor, stage_embeddings: bool = False) -> tuple[TransferRequest, asyncio.Future]`

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L144`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L144)*

Send precomputed embeddings for a given request ID.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `embeddings` | `torch.Tensor` | A torch.Tensor of the embeddings to send. |
| `stage_embeddings` | `bool` | A boolean indicating whether the embeddings should be staged for the transfer, |

Returns:
    A tuple containing the TransferRequest object and a future that can be awaited to indicate the send is completed.

</details>

<details>
<summary><strong>`NixlReadEmbeddingReceiver` — Counter part of 'NixlReadEmbeddingSender', see 'NixlReadEmbeddingSender' for details.</strong></summary>

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L835`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L835)*

Initial implementation of another usage of NIXL connect library that persists
connection (agent registration) and descriptors (memory registration) across multiple send operations
to avoid the overhead of repeated connection setup and teardown.
[gluo FIXME] This implementation requires more memory allocation and somewhat rigid, should move away
from connect library so we can have single descriptor and chunk for transfer on demand, similarly to
KV cache transfer. We may worry less on memory fragmentation as the memory can be released for next
transfer as soon as the embedding has passed to the framework (NEED TO VERIFY: framework will copy) and
can simply loop around the large buffer.
<b>Bases:</b> `AbstractEmbeddingReceiver`

<b>Attributes</b>

- `connector`
- `tensor_id_counter`
- `aggregated_op_create_time`
- `aggregated_op_wait_time`
- `warmedup_descriptors`
- `inuse_descriptors`

<b>Constructor</b>

#### `__init__(embedding_hidden_size = 8 * 1024, max_item_mm_token = 1024, max_items = 1024)`

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L848`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L848)*

<b>Methods</b>

#### `receive_embeddings(request: TransferRequest) -> tuple[int, torch.Tensor]`

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L880`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L880)*

Receive precomputed embeddings for a given request ID.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `request` | `TransferRequest` | The TransferRequest object containing information to receive embeddings for. |

<b>Returns:</b>

- `int` -- A tuple containing the tensor ID and the received embeddings as a torch.Tensor.
- `torch.Tensor` -- Caller should invoke release_tensor(tensor_id) when the tensor is no longer needed to free up resources.
#### `release_tensor(tensor_id: int)`

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L935`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L935)*

Indicate that the tensor associated with the ID is no longer in use.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `tensor_id` | `int` | The ID of the tensor to release. |

</details>

<details>
<summary><strong>`NixlReadEmbeddingSender` — Initial implementation of NIXL READ based transfer. This implementation uses</strong></summary>

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L792`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L792)*

a monkey-patched version of 'nixl_connect' wrapper library to persist
connection (agent registration) and descriptors across multiple send operations
to avoid the overhead of repeated connection setup and teardown.
NOTE This implementation or the use of 'nixl_connect' needs to be revisited as
the benchmarking result is unexpectedly slow. Keeping it now for completeness,
i.e. provide NIXL WRITE based and READ based transfer classes.
<b>Bases:</b> `AbstractEmbeddingSender`

<b>Attributes</b>

- `connector`

<b>Constructor</b>

#### `__init__()`

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L803`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L803)*

<b>Methods</b>

#### `send_embeddings(embeddings: torch.Tensor, stage_embeddings: bool = False) -> tuple[TransferRequest, asyncio.Future]`

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L806`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L806)*

Send precomputed embeddings.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `embeddings` | `torch.Tensor` | A torch.Tensor of the embeddings to send. |
| `stage_embeddings` | `bool` | A boolean indicating whether the embeddings should be staged for the transfer, |

Returns:
    A tuple containing the TransferRequest object and a future that can be awaited to indicate the send is completed.

</details>

<details>
<summary><strong>`NixlWriteEmbeddingReceiver` — Counter part of 'NixlWriteEmbeddingSender', see 'NixlWriteEmbeddingSender' for details.</strong></summary>

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L606`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L606)*

The receiver manages a ring buffer for sender to write the embeddings into, and respond
to the sender's transfer request with the buffer information for the WRITE transfer.
<b>Bases:</b> `AbstractEmbeddingReceiver`

<b>Attributes</b>

- `ring_buffer`
- `transfer_tensor`
- `receiver_id`
- `nixl_agent`
- `remote_agents`
- `reg_descs`
- `agent_metadata`
- `id_counter`
- `to_buffer_id`

<b>Constructor</b>

#### `__init__(buffer_size = 2 * 8 * 1024 * 1024 * 256 * 2)`

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L613`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L613)*

<b>Methods</b>

#### `receive_embeddings(request: TransferRequest, receive_timeout = 60) -> tuple[int, torch.Tensor]`

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L636`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L636)*

Receive precomputed embeddings for a given request ID.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `request` | `TransferRequest` | The TransferRequest object containing information to receive embeddings for. |
| `receive_timeout` | `Any` | Maximum time to wait for the transfer to complete before raising a TimeoutError. |

<b>Returns:</b>

- `int` -- A tuple containing the tensor ID and the received embeddings as a torch.Tensor.
- `torch.Tensor` -- Caller should invoke release_tensor(tensor_id) when the tensor is no longer needed to free up resources.
#### `release_tensor(tensor_id: int)`

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L754`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L754)*

Indicate that the tensor associated with the ID is no longer in use.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `tensor_id` | `int` | The ID of the tensor to release. |

</details>

<details>
<summary><strong>`NixlWriteEmbeddingSender` — NIXL WRITE-based implementation of the embedding sender interface.</strong></summary>

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L350`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L350)*

Designed for scenarios where the sender transmits dynamically allocated
tensors. Because these tensors allocation is external to the sender,
NIXL memory registration will perform on each send request. The receiver
will manage a pre-allocated buffer, so its NIXL metadata is consistent once
initialized. In such acenarios, let sender initiate the WRITE operations requires
minimal metadata exchange.

> [!NOTE]
> 1. Record the receiver NIXL metadata, this is done:
    * Implicitly through the first transfer request as fallback if the metadata
      hasn't been recorded.
    * [REMOVED] Explicitly through add_agent() API before calling send_embeddings().
      The receiver provides get_agent_metadata() API to return its NIXL metadata.
      This complicates the implementation and add extra responsiblity on the caller side,
      will revisit the necessity if metadata exchange overhead is significant.
2. The sender prepares the embeddings and produces a TransferRequest
   containing sender contact and tensor metadata (shape, dtype, size, etc).
3. The receiver responds with (optional) receiver contact, target tensor
   metadata (buffer address, device, etc) and done signal through NIXL notification.
4. The sender performs a NIXL WRITE to push the data into the
   receiver's buffer.
<b>Bases:</b> `AbstractEmbeddingSender`

<b>Attributes</b>

- `sender_id`
- `nixl_agent`
- `remote_agents`
- `agent_metadata`
- `agent_metadata_b64`
- `transfer_tracker`
- `registered_descs`
- `id_counter`
- `transfer_queue`
- `transfer_timeout`

<b>Constructor</b>

#### `__init__()`

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L376`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L376)*

<b>Methods</b>

#### `send_embeddings(embeddings: torch.Tensor, stage_embeddings: bool = False) -> tuple[TransferRequest, asyncio.Future]`

*Source: [`dynamo/common/multimodal/embedding_transfer.py#L558`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/multimodal/embedding_transfer.py#L558)*

Send precomputed embeddings.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `embeddings` | `torch.Tensor` | A torch.Tensor of the embeddings to send. |
| `stage_embeddings` | `bool` | A boolean indicating whether the embeddings should be staged for the transfer, |

Returns:
    A tuple containing the TransferRequest object and a future that can be awaited to indicate the send is completed.

</details>

<details>
<summary><strong>`LoRAManager` — Minimal Python wrapper around Rust core with extension points.</strong></summary>

*Source: [`dynamo/common/lora/manager.py#L30`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/lora/manager.py#L30)*

The manager uses the Rust-based LoRADownloader for S3 and local file sources,
and allows registering custom Python sources for other protocols.

# confidence: high
Examples:
    >>> from dynamo.common.lora import LoRAManager
    >>> manager = LoRAManager()
    >>> manager.register_custom_source("hf", my_hf_source)
    >>> result = await manager.download_lora("s3://bucket/adapters/lora-v1")
    >>> result["status"]
    'success'
    >>> manager.is_cached("s3://bucket/adapters/lora-v1")
    True
<b>Constructor</b>

#### `__init__(cache_path: Optional[Path] = None)`

*Source: [`dynamo/common/lora/manager.py#L49`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/lora/manager.py#L49)*

Initialize LoRA manager.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `cache_path` | `Optional[Path]` | Optional custom cache path. If not provided, uses DYN_LORA_PATH env var. |
<b>Methods</b>

#### `register_custom_source(scheme: str, source: LoRASourceProtocol)`

*Source: [`dynamo/common/lora/manager.py#L63`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/lora/manager.py#L63)*

Register a custom Python source for a URI scheme.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `scheme` | `str` | URI scheme without "://" (e.g., "hf" for hf:// URIs) |
| `source` | `LoRASourceProtocol` | LoRA source implementing LoRASourceProtocol |
#### `download_lora(lora_uri: str) -> Dict[str, Any]`

*Source: [`dynamo/common/lora/manager.py#L73`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/lora/manager.py#L73)*

Download LoRA if needed, return local path.

The source is inferred from the URI scheme:
- file:// -> Local filesystem (Rust)
- s3:// -> S3 (Rust)
- Custom schemes -> Registered Python sources

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `lora_uri` | `str` | Source URI (file://, s3://, or custom scheme) |

<b>Returns:</b> `Dict[str, Any]` -- Dictionary with:
- status: "success" or "error"
- local_path: Local path to LoRA (if successful)
- message: Error message (if error)
#### `is_cached(lora_uri: str) -> bool`

*Source: [`dynamo/common/lora/manager.py#L118`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/lora/manager.py#L118)*

Check if LoRA is already cached locally.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `lora_uri` | `str` |  |  |


</details>

<details>
<summary><strong>`LoRASourceProtocol` — Protocol for custom Python LoRA sources.</strong></summary>

*Source: [`dynamo/common/lora/manager.py#L15`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/lora/manager.py#L15)*

Users can implement this to add custom sources.
<b>Bases:</b> `Protocol`

<b>Methods</b>

#### `download(lora_uri: str, dest_path: Path) -> Path`

*Source: [`dynamo/common/lora/manager.py#L21`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/lora/manager.py#L21)*

Download LoRA to dest_path, return actual path

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `lora_uri` | `str` |  |  |
| `dest_path` | `Path` |  |  |

#### `exists(lora_uri: str) -> bool`

*Source: [`dynamo/common/lora/manager.py#L25`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/lora/manager.py#L25)*

Check if LoRA exists in this source

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `lora_uri` | `str` |  |  |


</details>

### Functions

<details>
<summary><strong>`get_fs()` — Initialize fsspec filesystem for the given URL.</strong></summary>

#### `get_fs(fs_url: str) -> DirFileSystem`

*Source: [`dynamo/common/storage.py#L42`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/storage.py#L42)*

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `fs_url` | `str` | The URL of the filesystem to initialize. e.g. s3://bucket, gs://bucket, file:///local/path |

# confidence: high — adapted from real call site
Examples:
    >>> from dynamo.common.storage import get_fs
    >>>
    >>> fs = get_fs("file:///tmp/media")
    >>> fs.fs.protocol
    'file'
    >>> fs = get_fs("s3://my-bucket")
    >>> fs.path
    'my-bucket'

<b>Returns:</b>

- `DirFileSystem` -- The initialized DirFileSystem wrapper for the filesystem.
- `DirFileSystem` -- fs.fs.protocol to get the protocol of the filesystem
- `DirFileSystem` -- fs.path to get the bucket or root path
- `DirFileSystem` -- path to the object in the filesystem - f"`{fs.fs.protocol}`://`{fs.path}`/`{path}`"

</details>

<details>
<summary><strong>`get_media_url()` — Build a public URL for a file stored in the media filesystem.</strong></summary>

#### `get_media_url(fs: DirFileSystem, storage_path: str, base_url: Optional[str] = None) -> str`

*Source: [`dynamo/common/storage.py#L83`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/storage.py#L83)*

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `fs` | `DirFileSystem` | The DirFileSystem returned by `0`. |
| `storage_path` | `str` | Relative path within the filesystem (e.g. "videos/req-id.mp4"). |
| `base_url` | `Optional[str]` | Optional CDN / proxy base URL.  When set, the returned URL is `0`.  When *None*, the URL is constructed from the filesystem's protocol and root path. |

# confidence: high
Examples:
    >>> from dynamo.common.storage import get_fs, get_media_url
    >>>
    >>> fs = get_fs("file:///tmp/media")
    >>> get_media_url(fs, "videos/req-123.mp4")
    'file:///tmp/media/videos/req-123.mp4'
    >>> get_media_url(fs, "img.png", base_url="https://cdn.example.com/media")
    'https://cdn.example.com/media/img.png'

<b>Returns:</b> `str` -- Public URL string for the uploaded file.

</details>

<details>
<summary><strong>`upload_to_fs()` — Upload bytes to the media filesystem and return the public URL.</strong></summary>

#### `upload_to_fs(fs: DirFileSystem, storage_path: str, data: bytes, base_url: Optional[str] = None) -> str`

*Source: [`dynamo/common/storage.py#L117`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/storage.py#L117)*

This is the canonical helper for all backends (vLLM, SGLang, TRT-LLM)
to store generated images/videos and produce a response URL.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `fs` | `DirFileSystem` | The DirFileSystem returned by `0`. |
| `storage_path` | `str` | Relative path within the filesystem (e.g. "images/req-id/file.png"). |
| `data` | `bytes` | Raw bytes to upload. |
| `base_url` | `Optional[str]` | Optional CDN / proxy base URL for URL rewriting. |

# confidence: high — adapted from real call site
Examples:
    >>> from dynamo.common.storage import get_fs, upload_to_fs
    >>>
    >>> fs = get_fs("s3://my-media-bucket")
    >>> image_bytes = b"\x89PNG..."
    >>> url = await upload_to_fs(
    ...     fs, "images/req-123/output.png", image_bytes
    ... )
    >>> url
    's3://my-media-bucket/images/req-123/output.png'

<b>Returns:</b> `str` -- Public URL string for the uploaded file.

</details>

<details>
<summary><strong>`add_config_dump_args()` — Add arguments to the parser to dump the config to a file.</strong></summary>

#### `add_config_dump_args(parser: argparse.ArgumentParser)`

*Source: [`dynamo/common/config_dump/config_dumper.py#L159`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/config_dump/config_dumper.py#L159)*

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `parser` | `argparse.ArgumentParser` | The parser to add the arguments to |

</details>

<details>
<summary><strong>`get_config_dump()` — Collect comprehensive config information about a backend instance.</strong></summary>

#### `get_config_dump(config: Any, extra_info: Optional[Dict[str, Any]] = None) -> str`

*Source: [`dynamo/common/config_dump/config_dumper.py#L108`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/config_dump/config_dumper.py#L108)*

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `config` | `Any` | Any JSON-serializable object containing the backend configuration. |
| `extra_info` | `Optional[Dict[str, Any]]` | Optional dict of additional information to include in the dump. |

<b>Returns:</b> `str` -- JSON string containing comprehensive information.

> [!NOTE]
> Returns error information if collection fails, ensuring some diagnostic data is always available.

</details>

<details>
<summary><strong>`get_environment_vars()` — Get relevant environment variables based on prefixes.</strong></summary>

#### `get_environment_vars(prefixes: Optional[List[str]] = None, include_sensitive: bool = False, additional_vars: Optional[Set[str]] = None) -> Dict[str, str]`

*Source: [`dynamo/common/config_dump/environment.py#L51`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/config_dump/environment.py#L51)*

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `prefixes` | `Optional[List[str]]` | List of environment variable prefixes to capture.       If None, uses DEFAULT_ENV_PREFIXES. |
| `include_sensitive` | `bool` | If False, redacts values of potentially sensitive variables.               Default is False for security. |
| `additional_vars` | `Optional[Set[str]]` | Set of specific variable names to include regardless of prefix. |

<b>Returns:</b>

- `Dict[str, str]` -- Dictionary of environment variable names to values.
- `Dict[str, str]` -- Sensitive values are replaced with "&lt;REDACTED>" unless include_sensitive is True.

<b>Examples</b>

```python
>>> get_environment_vars()  # Uses default prefixes
>>> get_environment_vars(prefixes=["MY_APP_"])  # Custom prefixes only
>>> get_environment_vars(additional_vars={"PATH", "HOME"})  # Include specific vars
```

</details>

<details>
<summary><strong>`get_gpu_info()` — Get GPU information if available.</strong></summary>

#### `get_gpu_info() -> Optional[Dict[str, Any]]`

*Source: [`dynamo/common/config_dump/system_info.py#L98`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/config_dump/system_info.py#L98)*

<b>Returns:</b>

- `Optional[Dict[str, Any]]` -- Dictionary containing GPU details if available, None otherwise.
- `Optional[Dict[str, Any]]` -- Attempts to use nvidia-smi via subprocess with XML output format.

> [!NOTE]
> This is a best-effort function and returns None if GPU info cannot be obtained.

</details>

<details>
<summary><strong>`get_runtime_info()` — Get Python runtime information.</strong></summary>

#### `get_runtime_info() -> Dict[str, Any]`

*Source: [`dynamo/common/config_dump/system_info.py#L60`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/config_dump/system_info.py#L60)*

<b>Returns:</b> `Dict[str, Any]` -- Dictionary containing Python version, executable path, and command-line arguments.

> [!NOTE]
> Gracefully handles errors by returning partial information.

</details>

<details>
<summary><strong>`get_system_info()` — Get comprehensive system information.</strong></summary>

#### `get_system_info() -> Dict[str, Any]`

*Source: [`dynamo/common/config_dump/system_info.py#L13`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/config_dump/system_info.py#L13)*

<b>Returns:</b>

- `Dict[str, Any]` -- Dictionary containing platform, architecture, processor, hostname,
- `Dict[str, Any]` -- and operating system details.

> [!NOTE]
> Gracefully handles errors by returning partial information.

</details>

---

## dynamo.health_check

Health check payload and environment-based configuration.

### Classes

<details>
<summary><strong>`HealthCheckPayload` — Base class for managing health check payloads.</strong></summary>

*Source: [`dynamo/health_check.py#L73`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/health_check.py#L73)*

Each backend should extend this class and set self.default_payload
in their __init__ method.

Environment variable DYN_HEALTH_CHECK_PAYLOAD can override the default.
Examples:
    >>> from dynamo.health_check import HealthCheckPayload
    >>>
    >>> class MyBackendHealthCheck(HealthCheckPayload):
    ...     def __init__(self):
    ...         self.default_payload = `{
    ...             "prompt": "health check",
    ...             "max_tokens": 1,
    ...         }`
    ...         super().__init__()
    >>>
    >>> hc = MyBackendHealthCheck()
    >>> hc.to_dict()
    `{'prompt': 'health check', 'max_tokens': 1}`
<b>Attributes</b>

- `default_payload`: `Dict[str, Any]`

<b>Constructor</b>

#### `__init__()`

*Source: [`dynamo/health_check.py#L99`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/health_check.py#L99)*

Initialize health check payload.

Subclasses should call super().__init__() after setting self.default_payload.
<b>Methods</b>

#### `to_dict() -> Dict[str, Any]`

*Source: [`dynamo/health_check.py#L112`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/health_check.py#L112)*

Get the health check payload as a dictionary.

Returns the environment override if DYN_HEALTH_CHECK_PAYLOAD is set,
otherwise returns the default payload.

</details>

### Functions

<details>
<summary><strong>`load_health_check_from_env()` — Load health check payload from environment variable.</strong></summary>

#### `load_health_check_from_env(env_var: str = 'DYN_HEALTH_CHECK_PAYLOAD') -> Optional[Dict[str, Any]]`

*Source: [`dynamo/health_check.py#L21`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/health_check.py#L21)*

Supports two formats:
1. JSON string: export DYN_HEALTH_CHECK_PAYLOAD='`{"prompt": "test", "max_tokens": 1}`'
2. File path: export DYN_HEALTH_CHECK_PAYLOAD='@/path/to/health_check.json'

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `env_var` | `str` | Name of the environment variable to check (default: DYN_HEALTH_CHECK_PAYLOAD) |

Examples:
    >>> import os
    >>> from dynamo.health_check import load_health_check_from_env
    >>>
    >>> os.environ["DYN_HEALTH_CHECK_PAYLOAD"] = (
    ...     '`{"prompt": "test", "max_tokens": 1}`'
    ... )
    >>> payload = load_health_check_from_env()
    >>> payload
    `{'prompt': 'test', 'max_tokens': 1}`

<b>Returns:</b> `Optional[Dict[str, Any]]` -- Dict containing the health check payload, or None if not set.

</details>

---

## dynamo.logits_processing

Custom logits processing for LLM token generation.

<details>
<summary><strong>`BaseLogitsProcessor` — Protocol for logits processors in Dynamo.</strong></summary>

*Source: [`dynamo/logits_processing/base.py#L16`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/logits_processing/base.py#L16)*

All logits processors must implement this interface to be compatible
with backend adapters (TRT-LLM, vLLM, SGLang).
Examples:
    >>> import torch
    >>> from typing import Sequence
    >>> from dynamo.logits_processing import BaseLogitsProcessor
    >>>
    >>> class TemperatureProcessor:
    ...     def __init__(self, temperature: float = 0.7):
    ...         self.temperature = temperature
    ...     def __call__(
    ...         self, input_ids: Sequence[int], logits: torch.Tensor
    ...     ) -> None:
    ...         logits.div_(self.temperature)
    >>>
    >>> assert isinstance(TemperatureProcessor(), BaseLogitsProcessor)
<b>Bases:</b> `Protocol`


</details>

<details>
<summary><strong>`BaseLogitsProcessor` — Protocol for logits processors in Dynamo.</strong></summary>

*Source: [`dynamo/logits_processing/base.py#L16`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/logits_processing/base.py#L16)*

All logits processors must implement this interface to be compatible
with backend adapters (TRT-LLM, vLLM, SGLang).
Examples:
    >>> import torch
    >>> from typing import Sequence
    >>> from dynamo.logits_processing import BaseLogitsProcessor
    >>>
    >>> class TemperatureProcessor:
    ...     def __init__(self, temperature: float = 0.7):
    ...         self.temperature = temperature
    ...     def __call__(
    ...         self, input_ids: Sequence[int], logits: torch.Tensor
    ...     ) -> None:
    ...         logits.div_(self.temperature)
    >>>
    >>> assert isinstance(TemperatureProcessor(), BaseLogitsProcessor)
<b>Bases:</b> `Protocol`


</details>

<details>
<summary><strong>`HelloWorldLogitsProcessor` — Sample Logits Processor that always outputs a hardcoded</strong></summary>

*Source: [`dynamo/logits_processing/examples/hello_world.py#L14`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/logits_processing/examples/hello_world.py#L14)*

response (`RESPONSE`), no matter the input
<b>Bases:</b> `BaseLogitsProcessor`

<b>Attributes</b>

- `tokenizer`
- `token_ids`
- `eos_id`
- `state`

<b>Constructor</b>

#### `__init__(tokenizer: PreTrainedTokenizerBase)`

*Source: [`dynamo/logits_processing/examples/hello_world.py#L20`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/logits_processing/examples/hello_world.py#L20)*


</details>

<details>
<summary><strong>`TemperatureProcessor` — Example logits processor that applies temperature scaling.</strong></summary>

*Source: [`dynamo/logits_processing/examples/temperature.py#L11`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/logits_processing/examples/temperature.py#L11)*

This is a simple demonstration of how to implement a logits processor
that can be used with any Dynamo backend.
<b>Bases:</b> `BaseLogitsProcessor`

<b>Attributes</b>

- `temperature`

<b>Constructor</b>

#### `__init__(temperature: float = 1.0)`

*Source: [`dynamo/logits_processing/examples/temperature.py#L19`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/logits_processing/examples/temperature.py#L19)*

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `temperature` | `float` | Scaling factor. Higher values make distribution more uniform,         lower values make it more peaked. Must be positive. |

</details>

---

## dynamo.planner

Scaling connectors and decision types for the Dynamo Planner.

### Enums

<details>
<summary><strong>`SubComponentType` — Type of sub-component in a disaggregated deployment.</strong></summary>

*Source: [`dynamo/planner/defaults.py#L163`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/defaults.py#L163)*

# confidence: high
Examples:
    >>> from dynamo.planner.defaults import SubComponentType
    >>> SubComponentType.PREFILL.value
    'prefill'
    >>> SubComponentType("decode") == SubComponentType.DECODE
    True
<b>Bases:</b> `str`, `Enum`

<b>Attributes</b>

- `PREFILL`
- `DECODE`


</details>

<details>
<summary><strong>`SubComponentType` — Type of sub-component in a disaggregated deployment.</strong></summary>

*Source: [`dynamo/planner/defaults.py#L163`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/defaults.py#L163)*

# confidence: high
Examples:
    >>> from dynamo.planner.defaults import SubComponentType
    >>> SubComponentType.PREFILL.value
    'prefill'
    >>> SubComponentType("decode") == SubComponentType.DECODE
    True
<b>Bases:</b> `str`, `Enum`

<b>Attributes</b>

- `PREFILL`
- `DECODE`


</details>

<details>
<summary><strong>`ScaleStatus` — Status values for scaling operations</strong></summary>

*Source: [`dynamo/planner/scale_protocol.py#L14`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/scale_protocol.py#L14)*

<b>Bases:</b> `str`, `Enum`

<b>Attributes</b>

- `SUCCESS`
- `ERROR`
- `SCALING`


</details>

<details>
<summary><strong>`SubComponentType` — Type of sub-component in a disaggregated deployment.</strong></summary>

*Source: [`dynamo/planner/defaults.py#L163`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/defaults.py#L163)*

# confidence: high
Examples:
    >>> from dynamo.planner.defaults import SubComponentType
    >>> SubComponentType.PREFILL.value
    'prefill'
    >>> SubComponentType("decode") == SubComponentType.DECODE
    True
<b>Bases:</b> `str`, `Enum`

<b>Attributes</b>

- `PREFILL`
- `DECODE`


</details>

<details>
<summary><strong>`SubComponentType` — Type of sub-component in a disaggregated deployment.</strong></summary>

*Source: [`dynamo/planner/defaults.py#L163`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/defaults.py#L163)*

# confidence: high
Examples:
    >>> from dynamo.planner.defaults import SubComponentType
    >>> SubComponentType.PREFILL.value
    'prefill'
    >>> SubComponentType("decode") == SubComponentType.DECODE
    True
<b>Bases:</b> `str`, `Enum`

<b>Attributes</b>

- `PREFILL`
- `DECODE`


</details>

<details>
<summary><strong>`SubComponentType` — Type of sub-component in a disaggregated deployment.</strong></summary>

*Source: [`dynamo/planner/defaults.py#L163`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/defaults.py#L163)*

# confidence: high
Examples:
    >>> from dynamo.planner.defaults import SubComponentType
    >>> SubComponentType.PREFILL.value
    'prefill'
    >>> SubComponentType("decode") == SubComponentType.DECODE
    True
<b>Bases:</b> `str`, `Enum`

<b>Attributes</b>

- `PREFILL`
- `DECODE`


</details>

<details>
<summary><strong>`ScaleStatus` — Status values for scaling operations</strong></summary>

*Source: [`dynamo/planner/scale_protocol.py#L14`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/scale_protocol.py#L14)*

<b>Bases:</b> `str`, `Enum`

<b>Attributes</b>

- `SUCCESS`
- `ERROR`
- `SCALING`


</details>

<details>
<summary><strong>`SubComponentType` — Type of sub-component in a disaggregated deployment.</strong></summary>

*Source: [`dynamo/planner/defaults.py#L163`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/defaults.py#L163)*

# confidence: high
Examples:
    >>> from dynamo.planner.defaults import SubComponentType
    >>> SubComponentType.PREFILL.value
    'prefill'
    >>> SubComponentType("decode") == SubComponentType.DECODE
    True
<b>Bases:</b> `str`, `Enum`

<b>Attributes</b>

- `PREFILL`
- `DECODE`


</details>

### Configuration

<details>
<summary><strong>`ScaleRequest` — Request to scale a deployment.</strong></summary>

*Source: [`dynamo/planner/scale_protocol.py#L22`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/scale_protocol.py#L22)*

# confidence: high
Examples:
    >>> from dynamo.planner.scale_protocol import ScaleRequest
    >>> from dynamo.planner.defaults import SubComponentType
    >>> from dynamo.planner.kubernetes_connector import TargetReplica
    >>>
    >>> request = ScaleRequest(
    ...     caller_namespace="dynamo",
    ...     graph_deployment_name="my-dgd",
    ...     k8s_namespace="default",
    ...     target_replicas=[
    ...         TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=2),
    ...         TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=4),
    ...     ],
    ... )
    >>> request.caller_namespace
    'dynamo'
<b>Bases:</b> `BaseModel`

<b>Attributes</b>

- `caller_namespace`: `str`
- `graph_deployment_name`: `str`
- `k8s_namespace`: `str`
- `target_replicas`: `List[TargetReplica]`
- `blocking`: `bool`
- `timestamp`: `Optional[float]`
- `predicted_load`: `Optional[dict]`


</details>

<details>
<summary><strong>`ScaleResponse` — Response from scaling operation</strong></summary>

*Source: [`dynamo/planner/scale_protocol.py#L62`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/scale_protocol.py#L62)*

<b>Bases:</b> `BaseModel`

<b>Attributes</b>

- `status`: `ScaleStatus`
- `message`: `str`
- `current_replicas`: `dict`


</details>

<details>
<summary><strong>`ScaleRequest` — Request to scale a deployment.</strong></summary>

*Source: [`dynamo/planner/scale_protocol.py#L22`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/scale_protocol.py#L22)*

# confidence: high
Examples:
    >>> from dynamo.planner.scale_protocol import ScaleRequest
    >>> from dynamo.planner.defaults import SubComponentType
    >>> from dynamo.planner.kubernetes_connector import TargetReplica
    >>>
    >>> request = ScaleRequest(
    ...     caller_namespace="dynamo",
    ...     graph_deployment_name="my-dgd",
    ...     k8s_namespace="default",
    ...     target_replicas=[
    ...         TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=2),
    ...         TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=4),
    ...     ],
    ... )
    >>> request.caller_namespace
    'dynamo'
<b>Bases:</b> `BaseModel`

<b>Attributes</b>

- `caller_namespace`: `str`
- `graph_deployment_name`: `str`
- `k8s_namespace`: `str`
- `target_replicas`: `List[TargetReplica]`
- `blocking`: `bool`
- `timestamp`: `Optional[float]`
- `predicted_load`: `Optional[dict]`


</details>

<details>
<summary><strong>`ScaleRequest` — Request to scale a deployment.</strong></summary>

*Source: [`dynamo/planner/scale_protocol.py#L22`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/scale_protocol.py#L22)*

# confidence: high
Examples:
    >>> from dynamo.planner.scale_protocol import ScaleRequest
    >>> from dynamo.planner.defaults import SubComponentType
    >>> from dynamo.planner.kubernetes_connector import TargetReplica
    >>>
    >>> request = ScaleRequest(
    ...     caller_namespace="dynamo",
    ...     graph_deployment_name="my-dgd",
    ...     k8s_namespace="default",
    ...     target_replicas=[
    ...         TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=2),
    ...         TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=4),
    ...     ],
    ... )
    >>> request.caller_namespace
    'dynamo'
<b>Bases:</b> `BaseModel`

<b>Attributes</b>

- `caller_namespace`: `str`
- `graph_deployment_name`: `str`
- `k8s_namespace`: `str`
- `target_replicas`: `List[TargetReplica]`
- `blocking`: `bool`
- `timestamp`: `Optional[float]`
- `predicted_load`: `Optional[dict]`


</details>

<details>
<summary><strong>`ScaleResponse` — Response from scaling operation</strong></summary>

*Source: [`dynamo/planner/scale_protocol.py#L62`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/scale_protocol.py#L62)*

<b>Bases:</b> `BaseModel`

<b>Attributes</b>

- `status`: `ScaleStatus`
- `message`: `str`
- `current_replicas`: `dict`


</details>

### Classes

<details>
<summary><strong>`SLAPlannerDefaults` — SLA-based planner defaults for throughput and latency targets.</strong></summary>

*Source: [`dynamo/planner/defaults.py#L51`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/defaults.py#L51)*

Provides default values for SLA objectives (TTFT, ITL), load prediction,
and scaling parameters used by the planner.

# confidence: high
Examples:
    >>> from dynamo.planner.defaults import SLAPlannerDefaults
    >>> SLAPlannerDefaults.ttft
    500.0
    >>> SLAPlannerDefaults.itl
    50.0
    >>> SLAPlannerDefaults.mode
    'disagg'
    >>> SLAPlannerDefaults.max_gpu_budget
    8
<b>Bases:</b> `BasePlannerDefaults`

<b>Attributes</b>

- `metric_pulling_prometheus_endpoint`
- `profile_results_dir`
- `isl`
- `osl`
- `ttft`
- `itl`
- `load_predictor`
- `prophet_window_size`
- `load_predictor_log1p`
- `kalman_q_level`
- `kalman_q_trend`
- `kalman_r`
- `kalman_min_points`
- `no_correction`
- `mode`: `Literal['disagg', 'prefill', 'decode', 'agg']`
- `throughput_metrics_source`
- `enable_throughput_scaling`
- `enable_load_scaling`
- `load_router_metrics_url`: `Optional[str]`
- `load_adjustment_interval`
- `load_learning_window`
- `load_scaling_down_sensitivity`
- `load_metric_samples`
- `load_min_observations`


</details>

<details>
<summary><strong>`GlobalPlannerConnector` — Connector that delegates scaling decisions to a centralized GlobalPlanner.</strong></summary>

*Source: [`dynamo/planner/global_planner_connector.py#L24`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L24)*

This connector wraps RemotePlannerClient and implements the PlannerConnector
interface, allowing planner_core.py to treat global-planner environment mode
consistently with kubernetes and virtual modes.

# confidence: high — adapted from real call site
Examples:
    >>> from dynamo.planner.global_planner_connector import GlobalPlannerConnector
    >>>
    >>> connector = GlobalPlannerConnector(
    ...     runtime=runtime,
    ...     dynamo_namespace="test-ns",
    ...     global_planner_namespace="global-ns",
    ...     model_name="Llama-3.2-1B-Instruct",
    ... )
    >>> await connector._async_init()
    >>> connector.get_model_name()
    'Llama-3.2-1B-Instruct'
<b>Bases:</b> `PlannerConnector`

<b>Attributes</b>

- `runtime`
- `dynamo_namespace`
- `global_planner_namespace`
- `global_planner_component`
- `model_name`
- `remote_client`: `Optional[RemotePlannerClient]`
- `last_predicted_load`: `Optional[dict]`

<b>Constructor</b>

#### `__init__(runtime: DistributedRuntime, dynamo_namespace: str, global_planner_namespace: str, global_planner_component: str = 'GlobalPlanner', model_name: Optional[str] = None)`

*Source: [`dynamo/planner/global_planner_connector.py#L47`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L47)*

Initialize GlobalPlannerConnector.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `runtime` | `DistributedRuntime` | Distributed runtime for communication |
| `dynamo_namespace` | `str` | Local dynamo namespace (caller identification) |
| `global_planner_namespace` | `str` | Namespace where GlobalPlanner is deployed |
| `global_planner_component` | `str` | Component name of GlobalPlanner (default: "GlobalPlanner") |
| `model_name` | `Optional[str]` | Optional model name (will be managed remotely if not provided) |
<b>Methods</b>

#### `set_predicted_load(num_requests: Optional[float], isl: Optional[float], osl: Optional[float])`

*Source: [`dynamo/planner/global_planner_connector.py#L86`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L86)*

Set predicted load for inclusion in next scale request.

This is called by planner_core.py before calling set_component_replicas.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `num_requests` | `Optional[float]` |  |  |
| `isl` | `Optional[float]` |  |  |
| `osl` | `Optional[float]` |  |  |

#### `set_component_replicas(target_replicas: list[TargetReplica], blocking: bool = True)`

*Source: [`dynamo/planner/global_planner_connector.py#L100`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L100)*

Set component replicas by delegating to GlobalPlanner.

Sends a ScaleRequest to the GlobalPlanner with the target replica configurations.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `target_replicas` | `list[TargetReplica]` | List of target replica configurations |
| `blocking` | `bool` | Whether to wait for scaling completion (passed to GlobalPlanner) |

<b>Raises</b>

- `EmptyTargetReplicasError` -- If target_replicas is empty
- `RuntimeError` -- If remote_client is not initialized or response indicates error
#### `add_component(sub_component_type: SubComponentType, blocking: bool = True)`

*Source: [`dynamo/planner/global_planner_connector.py#L170`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L170)*

Add a component (not supported for GlobalPlanner).

GlobalPlanner only supports batch operations via set_component_replicas.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_component_type` | `SubComponentType` |  |  |
| `blocking` | `bool` | True |  |

#### `remove_component(sub_component_type: SubComponentType, blocking: bool = True)`

*Source: [`dynamo/planner/global_planner_connector.py#L183`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L183)*

Remove a component (not supported for GlobalPlanner).

GlobalPlanner only supports batch operations via set_component_replicas.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_component_type` | `SubComponentType` |  |  |
| `blocking` | `bool` | True |  |

#### `validate_deployment(prefill_component_name: Optional[str] = None, decode_component_name: Optional[str] = None, kwargs = {})`

*Source: [`dynamo/planner/global_planner_connector.py#L196`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L196)*

Validate deployment (no-op for GlobalPlanner).

The GlobalPlanner validates the deployment on its side, so local
validation is not needed in delegating mode.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `prefill_component_name` | `Optional[str]` | None |  |
| `decode_component_name` | `Optional[str]` | None |  |
| `kwargs` | `Any` | {} |  |

#### `wait_for_deployment_ready()`

*Source: [`dynamo/planner/global_planner_connector.py#L213`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L213)*

Wait for deployment to be ready (no-op for GlobalPlanner).

The GlobalPlanner manages deployment state, so we don't need to
wait locally in delegating mode.
#### `get_model_name(kwargs = {}) -> str`

*Source: [`dynamo/planner/global_planner_connector.py#L225`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L225)*

Get model name.

Returns the model name if provided during initialization, otherwise
returns a placeholder indicating the model is managed remotely.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `kwargs` | `Any` | {} |  |


</details>

<details>
<summary><strong>`KubernetesConnector` — Connector that manages scaling via the Kubernetes DynamoGraphDeployment API.</strong></summary>

*Source: [`dynamo/planner/kubernetes_connector.py#L48`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L48)*

Implements the PlannerConnector interface for Kubernetes-based deployments,
providing add/remove component operations, deployment validation, and
model name discovery from DGD service specs.

# confidence: high — adapted from real call site
Examples:
    >>> from dynamo.planner.kubernetes_connector import KubernetesConnector
    >>> from dynamo.planner.defaults import SubComponentType
    >>>
    >>> connector = KubernetesConnector(
    ...     dynamo_namespace="dynamo",
    ...     k8s_namespace="default",
    ... )
    >>> await connector.validate_deployment()
    >>> model_name = connector.get_model_name()
    >>> await connector.add_component(SubComponentType.PREFILL)
<b>Bases:</b> `PlannerConnector`

<b>Attributes</b>

- `kube_api`
- `user_provided_model_name`: `Optional[str]`
- `parent_dgd_name`
- `graph_deployment_name`

<b>Constructor</b>

#### `__init__(dynamo_namespace: str, model_name: Optional[str] = None, k8s_namespace: Optional[str] = None, parent_dgd_name: Optional[str] = None)`

*Source: [`dynamo/planner/kubernetes_connector.py#L69`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L69)*

<b>Methods</b>

#### `add_component(sub_component_type: SubComponentType, blocking: bool = True)`

*Source: [`dynamo/planner/kubernetes_connector.py#L98`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L98)*

Add a component by increasing its replica count by 1

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_component_type` | `SubComponentType` |  |  |
| `blocking` | `bool` | True |  |

#### `remove_component(sub_component_type: SubComponentType, blocking: bool = True)`

*Source: [`dynamo/planner/kubernetes_connector.py#L118`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L118)*

Remove a component by decreasing its replica count by 1

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_component_type` | `SubComponentType` |  |  |
| `blocking` | `bool` | True |  |

#### `validate_deployment(prefill_component_name: Optional[str] = None, decode_component_name: Optional[str] = None, require_prefill: bool = True, require_decode: bool = True)`

*Source: [`dynamo/planner/kubernetes_connector.py#L139`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L139)*

Verify that the deployment contains services with subComponentType prefill and decode and the model name exists.
Will fallback to worker service names for backwards compatibility. (TODO: deprecate)

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `prefill_component_name` | `Optional[str]` | None |  |
| `decode_component_name` | `Optional[str]` | None |  |
| `require_prefill` | `bool` | True |  |
| `require_decode` | `bool` | True |  |


<b>Raises</b>

- `DynamoGraphDeploymentNotFoundError` -- If the deployment is not found
- `DeploymentValidationError` -- If the deployment does not contain services with subComponentType prefill and decode
#### `get_model_name(deployment: Optional[dict] = None, require_prefill: bool = True, require_decode: bool = True) -> str`

*Source: [`dynamo/planner/kubernetes_connector.py#L191`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L191)*

Get the model name from the deployment

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `deployment` | `Optional[dict]` | None |  |
| `require_prefill` | `bool` | True |  |
| `require_decode` | `bool` | True |  |

#### `get_gpu_counts(deployment: Optional[dict] = None, require_prefill: bool = True, require_decode: bool = True) -> tuple[int, int]`

*Source: [`dynamo/planner/kubernetes_connector.py#L257`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L257)*

Get the GPU counts for prefill and decode services from the deployment.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `deployment` | `Optional[dict]` | Optional deployment dict, fetched if not provided |
| `require_prefill` | `bool` | Whether to require prefill service |
| `require_decode` | `bool` | Whether to require decode service |

<b>Returns:</b> `tuple[int, int]` -- Tuple of (prefill_gpu_count, decode_gpu_count)

<b>Raises</b>

- `DeploymentValidationError` -- If GPU counts cannot be determined from DGD
#### `get_frontend_metrics_url(port: int = 8000) -> Optional[str]`

*Source: [`dynamo/planner/kubernetes_connector.py#L308`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L308)*

Auto-discover the Frontend service's metrics URL from the DGD.

Iterates spec.services to find the service with componentType "frontend",
then constructs the in-cluster URL using the operator's naming convention:
http://`{dgd_name}`-`{service_key_lowercase}`:`{port}`/metrics

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `port` | `int` | 8000 |  |


<b>Returns:</b> `Optional[str]` -- The metrics URL string, or None if no frontend service is found.
#### `wait_for_deployment_ready()`

*Source: [`dynamo/planner/kubernetes_connector.py#L330`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L330)*

Wait for the deployment to be ready
#### `get_actual_worker_counts(prefill_component_name: Optional[str] = None, decode_component_name: Optional[str] = None) -> tuple[int, int, bool]`

*Source: [`dynamo/planner/kubernetes_connector.py#L336`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L336)*

Get actual ready worker counts for prefill and decode from DGD status.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `prefill_component_name` | `Optional[str]` | None |  |
| `decode_component_name` | `Optional[str]` | None |  |


<b>Returns:</b>

- `int` -- tuple[int, int, bool]: (prefill_count, decode_count, is_stable)
- `int` -- - is_stable: False if any service is in a rollout (scaling should be skipped)
#### `set_component_replicas(target_replicas: list[TargetReplica], blocking: bool = True)`

*Source: [`dynamo/planner/kubernetes_connector.py#L382`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L382)*

Set the replicas for multiple components at once

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `target_replicas` | `list[TargetReplica]` |  |  |
| `blocking` | `bool` | True |  |


</details>

<details>
<summary><strong>`PlannerConnector` — Abstract base class for planner connectors that manage scaling operations.</strong></summary>

*Source: [`dynamo/planner/planner_connector.py#L20`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/planner_connector.py#L20)*

Concrete implementations include KubernetesConnector, VirtualConnector,
and GlobalPlannerConnector.

# confidence: high
Examples:
    >>> from dynamo.planner import KubernetesConnector, VirtualConnector
    >>> from dynamo.planner.planner_connector import PlannerConnector
    >>> issubclass(KubernetesConnector, PlannerConnector)
    True
    >>> issubclass(VirtualConnector, PlannerConnector)
    True
<b>Methods</b>

#### `add_component(component_name)`

*Source: [`dynamo/planner/planner_connector.py#L36`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/planner_connector.py#L36)*

Add a component to the planner

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `component_name` | `Any` |  |  |

#### `remove_component(component_name)`

*Source: [`dynamo/planner/planner_connector.py#L41`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/planner_connector.py#L41)*

Remove a component from the planner

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `component_name` | `Any` |  |  |


</details>

<details>
<summary><strong>`VirtualConnector` — This is a virtual connector for planner to output scaling decisions to non-native environments</strong></summary>

*Source: [`dynamo/planner/virtual_connector.py#L28`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/virtual_connector.py#L28)*

This virtual connector does not actually scale the deployment, instead, it communicates with the non-native environment through dynamo-runtime's VirtualConnectorCoordinator.
The deployment environment needs to use VirtualConnectorClient (in the Rust/Python bindings) to read from the scaling decisions and update report scaling status.

# confidence: medium
Examples:
    >>> from dynamo.planner.virtual_connector import VirtualConnector
    >>> from dynamo.planner.defaults import SubComponentType
    >>>
    >>> connector = VirtualConnector(
    ...     runtime=runtime,
    ...     dynamo_namespace="dynamo",
    ...     model_name="Llama-3.2-1B-Instruct",
    ... )
    >>> await connector._async_init()
    >>> await connector.add_component(SubComponentType.PREFILL)
<b>Bases:</b> `PlannerConnector`

<b>Attributes</b>

- `connector`
- `model_name`
- `dynamo_namespace`

<b>Constructor</b>

#### `__init__(runtime: DistributedRuntime, dynamo_namespace: str, model_name: Optional[str] = None)`

*Source: [`dynamo/planner/virtual_connector.py#L48`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/virtual_connector.py#L48)*

<b>Methods</b>

#### `add_component(sub_component_type: SubComponentType, blocking: bool = True)`

*Source: [`dynamo/planner/virtual_connector.py#L83`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/virtual_connector.py#L83)*

Add a component by increasing its replica count by 1

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_component_type` | `SubComponentType` |  |  |
| `blocking` | `bool` | True |  |

#### `remove_component(sub_component_type: SubComponentType, blocking: bool = True)`

*Source: [`dynamo/planner/virtual_connector.py#L99`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/virtual_connector.py#L99)*

Remove a component by decreasing its replica count by 1

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_component_type` | `SubComponentType` |  |  |
| `blocking` | `bool` | True |  |

#### `set_component_replicas(target_replicas: list[TargetReplica], blocking: bool = True)`

*Source: [`dynamo/planner/virtual_connector.py#L115`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/virtual_connector.py#L115)*

Set the replicas for multiple components at once

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `target_replicas` | `list[TargetReplica]` |  |  |
| `blocking` | `bool` | True |  |

#### `validate_deployment(prefill_component_name: Optional[str] = None, decode_component_name: Optional[str] = None, require_prefill: bool = True, require_decode: bool = True)`

*Source: [`dynamo/planner/virtual_connector.py#L142`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/virtual_connector.py#L142)*

Validate the deployment

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `prefill_component_name` | `Optional[str]` | None |  |
| `decode_component_name` | `Optional[str]` | None |  |
| `require_prefill` | `bool` | True |  |
| `require_decode` | `bool` | True |  |

#### `wait_for_deployment_ready()`

*Source: [`dynamo/planner/virtual_connector.py#L152`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/virtual_connector.py#L152)*

Wait for the deployment to be ready
#### `get_model_name(require_prefill: bool = True, require_decode: bool = True) -> str`

*Source: [`dynamo/planner/virtual_connector.py#L156`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/virtual_connector.py#L156)*

Get the model name from the deployment

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `require_prefill` | `bool` | True |  |
| `require_decode` | `bool` | True |  |


</details>

<details>
<summary><strong>`PlannerConnector` — Abstract base class for planner connectors that manage scaling operations.</strong></summary>

*Source: [`dynamo/planner/planner_connector.py#L20`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/planner_connector.py#L20)*

Concrete implementations include KubernetesConnector, VirtualConnector,
and GlobalPlannerConnector.

# confidence: high
Examples:
    >>> from dynamo.planner import KubernetesConnector, VirtualConnector
    >>> from dynamo.planner.planner_connector import PlannerConnector
    >>> issubclass(KubernetesConnector, PlannerConnector)
    True
    >>> issubclass(VirtualConnector, PlannerConnector)
    True
<b>Methods</b>

#### `add_component(component_name)`

*Source: [`dynamo/planner/planner_connector.py#L36`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/planner_connector.py#L36)*

Add a component to the planner

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `component_name` | `Any` |  |  |

#### `remove_component(component_name)`

*Source: [`dynamo/planner/planner_connector.py#L41`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/planner_connector.py#L41)*

Remove a component from the planner

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `component_name` | `Any` |  |  |


</details>

<details>
<summary><strong>`DeploymentModelNameMismatchError` — Raised when the model name is not the same in the deployment</strong></summary>

*Source: [`dynamo/planner/utils/exceptions.py#L89`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L89)*

<b>Bases:</b> `PlannerError`

<b>Attributes</b>

- `prefill_model_name`
- `decode_model_name`
- `message`

<b>Constructor</b>

#### `__init__(prefill_model_name: str, decode_model_name: str)`

*Source: [`dynamo/planner/utils/exceptions.py#L92`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L92)*


</details>

<details>
<summary><strong>`DeploymentValidationError` — Raised when deployment validation fails for multiple components.</strong></summary>

*Source: [`dynamo/planner/utils/exceptions.py#L183`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L183)*

This is used to aggregate multiple validation errors into a single exception,
providing a comprehensive view of all validation issues.
<b>Bases:</b> `PlannerError`

<b>Attributes</b>

- `errors`

<b>Constructor</b>

#### `__init__(errors: List[str])`

*Source: [`dynamo/planner/utils/exceptions.py#L190`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L190)*


</details>

<details>
<summary><strong>`EmptyTargetReplicasError` — Raised when target_replicas is empty or invalid.</strong></summary>

*Source: [`dynamo/planner/utils/exceptions.py#L196`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L196)*

This occurs when attempting to set component replicas with an empty
or invalid target_replicas dictionary.
<b>Bases:</b> `PlannerError`

<b>Constructor</b>

#### `__init__()`

*Source: [`dynamo/planner/utils/exceptions.py#L203`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L203)*


</details>

<details>
<summary><strong>`ModelNameNotFoundError` — Raised when the model name is not found in the deployment</strong></summary>

*Source: [`dynamo/planner/utils/exceptions.py#L81`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L81)*

<b>Bases:</b> `PlannerError`

<b>Constructor</b>

#### `__init__()`

*Source: [`dynamo/planner/utils/exceptions.py#L84`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L84)*


</details>

<details>
<summary><strong>`PlannerError` — Base exception for all planner-related errors.</strong></summary>

*Source: [`dynamo/planner/utils/exceptions.py#L41`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L41)*

This serves as the root exception class for all custom exceptions
in the planner module, allowing for broad exception catching when needed.
<b>Bases:</b> `Exception`


</details>

<details>
<summary><strong>`UserProvidedModelNameMismatchError` — Raised when the model name is not the same as the user provided model name</strong></summary>

*Source: [`dynamo/planner/utils/exceptions.py#L104`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L104)*

<b>Bases:</b> `PlannerError`

<b>Attributes</b>

- `model_name`
- `user_provided_model_name`
- `message`

<b>Constructor</b>

#### `__init__(model_name: str, user_provided_model_name: str)`

*Source: [`dynamo/planner/utils/exceptions.py#L107`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L107)*


</details>

<details>
<summary><strong>`KubernetesConnector` — Connector that manages scaling via the Kubernetes DynamoGraphDeployment API.</strong></summary>

*Source: [`dynamo/planner/kubernetes_connector.py#L48`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L48)*

Implements the PlannerConnector interface for Kubernetes-based deployments,
providing add/remove component operations, deployment validation, and
model name discovery from DGD service specs.

# confidence: high — adapted from real call site
Examples:
    >>> from dynamo.planner.kubernetes_connector import KubernetesConnector
    >>> from dynamo.planner.defaults import SubComponentType
    >>>
    >>> connector = KubernetesConnector(
    ...     dynamo_namespace="dynamo",
    ...     k8s_namespace="default",
    ... )
    >>> await connector.validate_deployment()
    >>> model_name = connector.get_model_name()
    >>> await connector.add_component(SubComponentType.PREFILL)
<b>Bases:</b> `PlannerConnector`

<b>Attributes</b>

- `kube_api`
- `user_provided_model_name`: `Optional[str]`
- `parent_dgd_name`
- `graph_deployment_name`

<b>Constructor</b>

#### `__init__(dynamo_namespace: str, model_name: Optional[str] = None, k8s_namespace: Optional[str] = None, parent_dgd_name: Optional[str] = None)`

*Source: [`dynamo/planner/kubernetes_connector.py#L69`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L69)*

<b>Methods</b>

#### `add_component(sub_component_type: SubComponentType, blocking: bool = True)`

*Source: [`dynamo/planner/kubernetes_connector.py#L98`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L98)*

Add a component by increasing its replica count by 1

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_component_type` | `SubComponentType` |  |  |
| `blocking` | `bool` | True |  |

#### `remove_component(sub_component_type: SubComponentType, blocking: bool = True)`

*Source: [`dynamo/planner/kubernetes_connector.py#L118`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L118)*

Remove a component by decreasing its replica count by 1

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_component_type` | `SubComponentType` |  |  |
| `blocking` | `bool` | True |  |

#### `validate_deployment(prefill_component_name: Optional[str] = None, decode_component_name: Optional[str] = None, require_prefill: bool = True, require_decode: bool = True)`

*Source: [`dynamo/planner/kubernetes_connector.py#L139`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L139)*

Verify that the deployment contains services with subComponentType prefill and decode and the model name exists.
Will fallback to worker service names for backwards compatibility. (TODO: deprecate)

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `prefill_component_name` | `Optional[str]` | None |  |
| `decode_component_name` | `Optional[str]` | None |  |
| `require_prefill` | `bool` | True |  |
| `require_decode` | `bool` | True |  |


<b>Raises</b>

- `DynamoGraphDeploymentNotFoundError` -- If the deployment is not found
- `DeploymentValidationError` -- If the deployment does not contain services with subComponentType prefill and decode
#### `get_model_name(deployment: Optional[dict] = None, require_prefill: bool = True, require_decode: bool = True) -> str`

*Source: [`dynamo/planner/kubernetes_connector.py#L191`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L191)*

Get the model name from the deployment

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `deployment` | `Optional[dict]` | None |  |
| `require_prefill` | `bool` | True |  |
| `require_decode` | `bool` | True |  |

#### `get_gpu_counts(deployment: Optional[dict] = None, require_prefill: bool = True, require_decode: bool = True) -> tuple[int, int]`

*Source: [`dynamo/planner/kubernetes_connector.py#L257`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L257)*

Get the GPU counts for prefill and decode services from the deployment.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `deployment` | `Optional[dict]` | Optional deployment dict, fetched if not provided |
| `require_prefill` | `bool` | Whether to require prefill service |
| `require_decode` | `bool` | Whether to require decode service |

<b>Returns:</b> `tuple[int, int]` -- Tuple of (prefill_gpu_count, decode_gpu_count)

<b>Raises</b>

- `DeploymentValidationError` -- If GPU counts cannot be determined from DGD
#### `get_frontend_metrics_url(port: int = 8000) -> Optional[str]`

*Source: [`dynamo/planner/kubernetes_connector.py#L308`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L308)*

Auto-discover the Frontend service's metrics URL from the DGD.

Iterates spec.services to find the service with componentType "frontend",
then constructs the in-cluster URL using the operator's naming convention:
http://`{dgd_name}`-`{service_key_lowercase}`:`{port}`/metrics

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `port` | `int` | 8000 |  |


<b>Returns:</b> `Optional[str]` -- The metrics URL string, or None if no frontend service is found.
#### `wait_for_deployment_ready()`

*Source: [`dynamo/planner/kubernetes_connector.py#L330`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L330)*

Wait for the deployment to be ready
#### `get_actual_worker_counts(prefill_component_name: Optional[str] = None, decode_component_name: Optional[str] = None) -> tuple[int, int, bool]`

*Source: [`dynamo/planner/kubernetes_connector.py#L336`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L336)*

Get actual ready worker counts for prefill and decode from DGD status.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `prefill_component_name` | `Optional[str]` | None |  |
| `decode_component_name` | `Optional[str]` | None |  |


<b>Returns:</b>

- `int` -- tuple[int, int, bool]: (prefill_count, decode_count, is_stable)
- `int` -- - is_stable: False if any service is in a rollout (scaling should be skipped)
#### `set_component_replicas(target_replicas: list[TargetReplica], blocking: bool = True)`

*Source: [`dynamo/planner/kubernetes_connector.py#L382`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kubernetes_connector.py#L382)*

Set the replicas for multiple components at once

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `target_replicas` | `list[TargetReplica]` |  |  |
| `blocking` | `bool` | True |  |


</details>

<details>
<summary><strong>`DynamoGraphDeploymentNotFoundError` — Raised when Parent DynamoGraphDeployment cannot be found.</strong></summary>

*Source: [`dynamo/planner/utils/exceptions.py#L51`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L51)*

This typically occurs when:
- The DYN_PARENT_DGD_K8S_NAME environment variable is not set
- The referenced DynamoGraphDeployment doesn't exist in the namespace
<b>Bases:</b> `PlannerError`

<b>Attributes</b>

- `deployment_name`
- `namespace`

<b>Constructor</b>

#### `__init__(deployment_name: str, namespace: str)`

*Source: [`dynamo/planner/utils/exceptions.py#L59`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L59)*


</details>

<details>
<summary><strong>`PlannerConnector` — Abstract base class for planner connectors that manage scaling operations.</strong></summary>

*Source: [`dynamo/planner/planner_connector.py#L20`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/planner_connector.py#L20)*

Concrete implementations include KubernetesConnector, VirtualConnector,
and GlobalPlannerConnector.

# confidence: high
Examples:
    >>> from dynamo.planner import KubernetesConnector, VirtualConnector
    >>> from dynamo.planner.planner_connector import PlannerConnector
    >>> issubclass(KubernetesConnector, PlannerConnector)
    True
    >>> issubclass(VirtualConnector, PlannerConnector)
    True
<b>Methods</b>

#### `add_component(component_name)`

*Source: [`dynamo/planner/planner_connector.py#L36`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/planner_connector.py#L36)*

Add a component to the planner

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `component_name` | `Any` |  |  |

#### `remove_component(component_name)`

*Source: [`dynamo/planner/planner_connector.py#L41`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/planner_connector.py#L41)*

Remove a component from the planner

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `component_name` | `Any` |  |  |


</details>

<details>
<summary><strong>`EmptyTargetReplicasError` — Raised when target_replicas is empty or invalid.</strong></summary>

*Source: [`dynamo/planner/utils/exceptions.py#L196`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L196)*

This occurs when attempting to set component replicas with an empty
or invalid target_replicas dictionary.
<b>Bases:</b> `PlannerError`

<b>Constructor</b>

#### `__init__()`

*Source: [`dynamo/planner/utils/exceptions.py#L203`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L203)*


</details>

<details>
<summary><strong>`VirtualConnector` — This is a virtual connector for planner to output scaling decisions to non-native environments</strong></summary>

*Source: [`dynamo/planner/virtual_connector.py#L28`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/virtual_connector.py#L28)*

This virtual connector does not actually scale the deployment, instead, it communicates with the non-native environment through dynamo-runtime's VirtualConnectorCoordinator.
The deployment environment needs to use VirtualConnectorClient (in the Rust/Python bindings) to read from the scaling decisions and update report scaling status.

# confidence: medium
Examples:
    >>> from dynamo.planner.virtual_connector import VirtualConnector
    >>> from dynamo.planner.defaults import SubComponentType
    >>>
    >>> connector = VirtualConnector(
    ...     runtime=runtime,
    ...     dynamo_namespace="dynamo",
    ...     model_name="Llama-3.2-1B-Instruct",
    ... )
    >>> await connector._async_init()
    >>> await connector.add_component(SubComponentType.PREFILL)
<b>Bases:</b> `PlannerConnector`

<b>Attributes</b>

- `connector`
- `model_name`
- `dynamo_namespace`

<b>Constructor</b>

#### `__init__(runtime: DistributedRuntime, dynamo_namespace: str, model_name: Optional[str] = None)`

*Source: [`dynamo/planner/virtual_connector.py#L48`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/virtual_connector.py#L48)*

<b>Methods</b>

#### `add_component(sub_component_type: SubComponentType, blocking: bool = True)`

*Source: [`dynamo/planner/virtual_connector.py#L83`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/virtual_connector.py#L83)*

Add a component by increasing its replica count by 1

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_component_type` | `SubComponentType` |  |  |
| `blocking` | `bool` | True |  |

#### `remove_component(sub_component_type: SubComponentType, blocking: bool = True)`

*Source: [`dynamo/planner/virtual_connector.py#L99`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/virtual_connector.py#L99)*

Remove a component by decreasing its replica count by 1

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_component_type` | `SubComponentType` |  |  |
| `blocking` | `bool` | True |  |

#### `set_component_replicas(target_replicas: list[TargetReplica], blocking: bool = True)`

*Source: [`dynamo/planner/virtual_connector.py#L115`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/virtual_connector.py#L115)*

Set the replicas for multiple components at once

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `target_replicas` | `list[TargetReplica]` |  |  |
| `blocking` | `bool` | True |  |

#### `validate_deployment(prefill_component_name: Optional[str] = None, decode_component_name: Optional[str] = None, require_prefill: bool = True, require_decode: bool = True)`

*Source: [`dynamo/planner/virtual_connector.py#L142`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/virtual_connector.py#L142)*

Validate the deployment

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `prefill_component_name` | `Optional[str]` | None |  |
| `decode_component_name` | `Optional[str]` | None |  |
| `require_prefill` | `bool` | True |  |
| `require_decode` | `bool` | True |  |

#### `wait_for_deployment_ready()`

*Source: [`dynamo/planner/virtual_connector.py#L152`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/virtual_connector.py#L152)*

Wait for the deployment to be ready
#### `get_model_name(require_prefill: bool = True, require_decode: bool = True) -> str`

*Source: [`dynamo/planner/virtual_connector.py#L156`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/virtual_connector.py#L156)*

Get the model name from the deployment

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `require_prefill` | `bool` | True |  |
| `require_decode` | `bool` | True |  |


</details>

<details>
<summary><strong>`DuplicateSubComponentError` — Raised when multiple services have the same subComponentType.</strong></summary>

*Source: [`dynamo/planner/utils/exceptions.py#L160`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L160)*

This occurs when the DynamoGraphDeployment contains more than one service
with the same subComponentType, which violates the expected uniqueness constraint.
<b>Bases:</b> `ComponentError`

<b>Attributes</b>

- `sub_component_type`
- `service_names`

<b>Constructor</b>

#### `__init__(sub_component_type: str, service_names: List[str])`

*Source: [`dynamo/planner/utils/exceptions.py#L167`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L167)*


</details>

<details>
<summary><strong>`SubComponentNotFoundError` — Raised when a required subComponentType is not found in the deployment.</strong></summary>

*Source: [`dynamo/planner/utils/exceptions.py#L140`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L140)*

This occurs when the DynamoGraphDeployment doesn't contain any service
with the requested subComponentType (e.g., 'prefill', 'decode').
<b>Bases:</b> `ComponentError`

<b>Attributes</b>

- `sub_component_type`

<b>Constructor</b>

#### `__init__(sub_component_type: str)`

*Source: [`dynamo/planner/utils/exceptions.py#L147`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L147)*


</details>

<details>
<summary><strong>`SLAPlannerDefaults` — SLA-based planner defaults for throughput and latency targets.</strong></summary>

*Source: [`dynamo/planner/defaults.py#L51`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/defaults.py#L51)*

Provides default values for SLA objectives (TTFT, ITL), load prediction,
and scaling parameters used by the planner.

# confidence: high
Examples:
    >>> from dynamo.planner.defaults import SLAPlannerDefaults
    >>> SLAPlannerDefaults.ttft
    500.0
    >>> SLAPlannerDefaults.itl
    50.0
    >>> SLAPlannerDefaults.mode
    'disagg'
    >>> SLAPlannerDefaults.max_gpu_budget
    8
<b>Bases:</b> `BasePlannerDefaults`

<b>Attributes</b>

- `metric_pulling_prometheus_endpoint`
- `profile_results_dir`
- `isl`
- `osl`
- `ttft`
- `itl`
- `load_predictor`
- `prophet_window_size`
- `load_predictor_log1p`
- `kalman_q_level`
- `kalman_q_trend`
- `kalman_r`
- `kalman_min_points`
- `no_correction`
- `mode`: `Literal['disagg', 'prefill', 'decode', 'agg']`
- `throughput_metrics_source`
- `enable_throughput_scaling`
- `enable_load_scaling`
- `load_router_metrics_url`: `Optional[str]`
- `load_adjustment_interval`
- `load_learning_window`
- `load_scaling_down_sensitivity`
- `load_metric_samples`
- `load_min_observations`


</details>

<details>
<summary><strong>`PlannerConnector` — Abstract base class for planner connectors that manage scaling operations.</strong></summary>

*Source: [`dynamo/planner/planner_connector.py#L20`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/planner_connector.py#L20)*

Concrete implementations include KubernetesConnector, VirtualConnector,
and GlobalPlannerConnector.

# confidence: high
Examples:
    >>> from dynamo.planner import KubernetesConnector, VirtualConnector
    >>> from dynamo.planner.planner_connector import PlannerConnector
    >>> issubclass(KubernetesConnector, PlannerConnector)
    True
    >>> issubclass(VirtualConnector, PlannerConnector)
    True
<b>Methods</b>

#### `add_component(component_name)`

*Source: [`dynamo/planner/planner_connector.py#L36`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/planner_connector.py#L36)*

Add a component to the planner

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `component_name` | `Any` |  |  |

#### `remove_component(component_name)`

*Source: [`dynamo/planner/planner_connector.py#L41`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/planner_connector.py#L41)*

Remove a component from the planner

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `component_name` | `Any` |  |  |


</details>

<details>
<summary><strong>`RemotePlannerClient` — Client for delegating scaling requests to centralized planner.</strong></summary>

*Source: [`dynamo/planner/remote_planner_client.py#L16`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/remote_planner_client.py#L16)*

# confidence: medium
Examples:
    >>> from dynamo.planner.remote_planner_client import RemotePlannerClient
    >>> client = RemotePlannerClient(
    ...     runtime=runtime,
    ...     central_namespace="global-planner",
    ...     central_component="GlobalPlanner",
    ...     connection_timeout=30.0,
    ... )
    >>> response = await client.send_scale_request(scale_request)
    >>> response.status
    &lt;ScaleStatus.SUCCESS: 'success'>
<b>Attributes</b>

- `runtime`
- `central_namespace`
- `central_component`
- `connection_timeout`
- `max_retries`

<b>Constructor</b>

#### `__init__(runtime: DistributedRuntime, central_namespace: str, central_component: str, connection_timeout: float = 30.0, max_retries: int = 3)`

*Source: [`dynamo/planner/remote_planner_client.py#L33`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/remote_planner_client.py#L33)*

<b>Methods</b>

#### `send_scale_request(request: ScaleRequest) -> ScaleResponse`

*Source: [`dynamo/planner/remote_planner_client.py#L105`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/remote_planner_client.py#L105)*

Send scale request to centralized planner

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `request` | `ScaleRequest` |  |  |


</details>

<details>
<summary><strong>`EmptyTargetReplicasError` — Raised when target_replicas is empty or invalid.</strong></summary>

*Source: [`dynamo/planner/utils/exceptions.py#L196`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L196)*

This occurs when attempting to set component replicas with an empty
or invalid target_replicas dictionary.
<b>Bases:</b> `PlannerError`

<b>Constructor</b>

#### `__init__()`

*Source: [`dynamo/planner/utils/exceptions.py#L203`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/utils/exceptions.py#L203)*


</details>

<details>
<summary><strong>`GlobalPlannerConnector` — Connector that delegates scaling decisions to a centralized GlobalPlanner.</strong></summary>

*Source: [`dynamo/planner/global_planner_connector.py#L24`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L24)*

This connector wraps RemotePlannerClient and implements the PlannerConnector
interface, allowing planner_core.py to treat global-planner environment mode
consistently with kubernetes and virtual modes.

# confidence: high — adapted from real call site
Examples:
    >>> from dynamo.planner.global_planner_connector import GlobalPlannerConnector
    >>>
    >>> connector = GlobalPlannerConnector(
    ...     runtime=runtime,
    ...     dynamo_namespace="test-ns",
    ...     global_planner_namespace="global-ns",
    ...     model_name="Llama-3.2-1B-Instruct",
    ... )
    >>> await connector._async_init()
    >>> connector.get_model_name()
    'Llama-3.2-1B-Instruct'
<b>Bases:</b> `PlannerConnector`

<b>Attributes</b>

- `runtime`
- `dynamo_namespace`
- `global_planner_namespace`
- `global_planner_component`
- `model_name`
- `remote_client`: `Optional[RemotePlannerClient]`
- `last_predicted_load`: `Optional[dict]`

<b>Constructor</b>

#### `__init__(runtime: DistributedRuntime, dynamo_namespace: str, global_planner_namespace: str, global_planner_component: str = 'GlobalPlanner', model_name: Optional[str] = None)`

*Source: [`dynamo/planner/global_planner_connector.py#L47`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L47)*

Initialize GlobalPlannerConnector.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `runtime` | `DistributedRuntime` | Distributed runtime for communication |
| `dynamo_namespace` | `str` | Local dynamo namespace (caller identification) |
| `global_planner_namespace` | `str` | Namespace where GlobalPlanner is deployed |
| `global_planner_component` | `str` | Component name of GlobalPlanner (default: "GlobalPlanner") |
| `model_name` | `Optional[str]` | Optional model name (will be managed remotely if not provided) |
<b>Methods</b>

#### `set_predicted_load(num_requests: Optional[float], isl: Optional[float], osl: Optional[float])`

*Source: [`dynamo/planner/global_planner_connector.py#L86`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L86)*

Set predicted load for inclusion in next scale request.

This is called by planner_core.py before calling set_component_replicas.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `num_requests` | `Optional[float]` |  |  |
| `isl` | `Optional[float]` |  |  |
| `osl` | `Optional[float]` |  |  |

#### `set_component_replicas(target_replicas: list[TargetReplica], blocking: bool = True)`

*Source: [`dynamo/planner/global_planner_connector.py#L100`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L100)*

Set component replicas by delegating to GlobalPlanner.

Sends a ScaleRequest to the GlobalPlanner with the target replica configurations.

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `target_replicas` | `list[TargetReplica]` | List of target replica configurations |
| `blocking` | `bool` | Whether to wait for scaling completion (passed to GlobalPlanner) |

<b>Raises</b>

- `EmptyTargetReplicasError` -- If target_replicas is empty
- `RuntimeError` -- If remote_client is not initialized or response indicates error
#### `add_component(sub_component_type: SubComponentType, blocking: bool = True)`

*Source: [`dynamo/planner/global_planner_connector.py#L170`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L170)*

Add a component (not supported for GlobalPlanner).

GlobalPlanner only supports batch operations via set_component_replicas.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_component_type` | `SubComponentType` |  |  |
| `blocking` | `bool` | True |  |

#### `remove_component(sub_component_type: SubComponentType, blocking: bool = True)`

*Source: [`dynamo/planner/global_planner_connector.py#L183`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L183)*

Remove a component (not supported for GlobalPlanner).

GlobalPlanner only supports batch operations via set_component_replicas.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_component_type` | `SubComponentType` |  |  |
| `blocking` | `bool` | True |  |

#### `validate_deployment(prefill_component_name: Optional[str] = None, decode_component_name: Optional[str] = None, kwargs = {})`

*Source: [`dynamo/planner/global_planner_connector.py#L196`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L196)*

Validate deployment (no-op for GlobalPlanner).

The GlobalPlanner validates the deployment on its side, so local
validation is not needed in delegating mode.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `prefill_component_name` | `Optional[str]` | None |  |
| `decode_component_name` | `Optional[str]` | None |  |
| `kwargs` | `Any` | {} |  |

#### `wait_for_deployment_ready()`

*Source: [`dynamo/planner/global_planner_connector.py#L213`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L213)*

Wait for deployment to be ready (no-op for GlobalPlanner).

The GlobalPlanner manages deployment state, so we don't need to
wait locally in delegating mode.
#### `get_model_name(kwargs = {}) -> str`

*Source: [`dynamo/planner/global_planner_connector.py#L225`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/global_planner_connector.py#L225)*

Get model name.

Returns the model name if provided during initialization, otherwise
returns a placeholder indicating the model is managed remotely.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `kwargs` | `Any` | {} |  |


</details>

<details>
<summary><strong>`PlannerConnector` — Abstract base class for planner connectors that manage scaling operations.</strong></summary>

*Source: [`dynamo/planner/planner_connector.py#L20`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/planner_connector.py#L20)*

Concrete implementations include KubernetesConnector, VirtualConnector,
and GlobalPlannerConnector.

# confidence: high
Examples:
    >>> from dynamo.planner import KubernetesConnector, VirtualConnector
    >>> from dynamo.planner.planner_connector import PlannerConnector
    >>> issubclass(KubernetesConnector, PlannerConnector)
    True
    >>> issubclass(VirtualConnector, PlannerConnector)
    True
<b>Methods</b>

#### `add_component(component_name)`

*Source: [`dynamo/planner/planner_connector.py#L36`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/planner_connector.py#L36)*

Add a component to the planner

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `component_name` | `Any` |  |  |

#### `remove_component(component_name)`

*Source: [`dynamo/planner/planner_connector.py#L41`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/planner_connector.py#L41)*

Remove a component from the planner

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `component_name` | `Any` |  |  |


</details>

<details>
<summary><strong>`RemotePlannerClient` — Client for delegating scaling requests to centralized planner.</strong></summary>

*Source: [`dynamo/planner/remote_planner_client.py#L16`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/remote_planner_client.py#L16)*

# confidence: medium
Examples:
    >>> from dynamo.planner.remote_planner_client import RemotePlannerClient
    >>> client = RemotePlannerClient(
    ...     runtime=runtime,
    ...     central_namespace="global-planner",
    ...     central_component="GlobalPlanner",
    ...     connection_timeout=30.0,
    ... )
    >>> response = await client.send_scale_request(scale_request)
    >>> response.status
    &lt;ScaleStatus.SUCCESS: 'success'>
<b>Attributes</b>

- `runtime`
- `central_namespace`
- `central_component`
- `connection_timeout`
- `max_retries`

<b>Constructor</b>

#### `__init__(runtime: DistributedRuntime, central_namespace: str, central_component: str, connection_timeout: float = 30.0, max_retries: int = 3)`

*Source: [`dynamo/planner/remote_planner_client.py#L33`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/remote_planner_client.py#L33)*

<b>Methods</b>

#### `send_scale_request(request: ScaleRequest) -> ScaleResponse`

*Source: [`dynamo/planner/remote_planner_client.py#L105`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/remote_planner_client.py#L105)*

Send scale request to centralized planner

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `request` | `ScaleRequest` |  |  |


</details>

### Functions

<details>
<summary><strong>`get_service_from_sub_component_type_or_name()` — Get the current replicas for a component in a graph deployment</strong></summary>

#### `get_service_from_sub_component_type_or_name(deployment: dict, sub_component_type: SubComponentType, component_name: Optional[str] = None) -> Service`

*Source: [`dynamo/planner/defaults.py#L261`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/defaults.py#L261)*

Returns: Service object

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `deployment` | `dict` |  |  |
| `sub_component_type` | `SubComponentType` |  |  |
| `component_name` | `Optional[str]` | None |  |


<b>Raises</b>

- `SubComponentNotFoundError` -- If no service with the specified subComponentType is found
- `DuplicateSubComponentError` -- If multiple services with the same subComponentType are found

</details>

<details>
<summary><strong>`get_current_k8s_namespace()` — Get the current namespace if running inside a k8s cluster</strong></summary>

#### `get_current_k8s_namespace() -> str`

*Source: [`dynamo/planner/kube.py#L30`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/kube.py#L30)*


</details>

<details>
<summary><strong>`get_service_from_sub_component_type_or_name()` — Get the current replicas for a component in a graph deployment</strong></summary>

#### `get_service_from_sub_component_type_or_name(deployment: dict, sub_component_type: SubComponentType, component_name: Optional[str] = None) -> Service`

*Source: [`dynamo/planner/defaults.py#L261`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/planner/defaults.py#L261)*

Returns: Service object

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `deployment` | `dict` |  |  |
| `sub_component_type` | `SubComponentType` |  |  |
| `component_name` | `Optional[str]` | None |  |


<b>Raises</b>

- `SubComponentNotFoundError` -- If no service with the specified subComponentType is found
- `DuplicateSubComponentError` -- If multiple services with the same subComponentType are found

</details>

---

## dynamo.router

Request routing configuration and argument groups.

### Configuration

<details>
<summary><strong>`DynamoRouterArgGroup` — CLI argument group for standalone router options.</strong></summary>

*Source: [`dynamo/router/args.py#L53`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/router/args.py#L53)*

# confidence: high — adapted from real call site
Examples:
    >>> import argparse
    >>> from dynamo.router.args import DynamoRouterArgGroup, DynamoRouterConfig
    >>>
    >>> parser = argparse.ArgumentParser()
    >>> DynamoRouterArgGroup().add_arguments(parser)
    >>> args = parser.parse_args([
    ...     "--endpoint", "dynamo.prefill.generate",
    ...     "--router-mode", "kv",
    ... ])
    >>> config = DynamoRouterConfig.from_cli_args(args)
    >>> config.endpoint
    'dynamo.prefill.generate'
<b>Bases:</b> `ArgGroup`

<b>Attributes</b>

- `name`

<b>Methods</b>

#### `add_arguments(parser)`

*Source: [`dynamo/router/args.py#L74`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/router/args.py#L74)*

Add router-owned arguments to parser.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `parser` | `Any` |  |  |


</details>

### Classes

<details>
<summary><strong>`DynamoRouterConfig` — Typed configuration for the standalone KV router (router-owned options only).</strong></summary>

*Source: [`dynamo/router/args.py#L17`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/router/args.py#L17)*

# confidence: high
Examples:
    >>> from dynamo.router.args import parse_args, DynamoRouterConfig
    >>> config = parse_args([
    ...     "--endpoint", "dynamo.prefill.generate",
    ...     "--router-mode", "kv",
    ... ])
    >>> config.endpoint
    'dynamo.prefill.generate'
    >>> config.namespace
    'dynamo'
<b>Bases:</b> `KvRouterConfigBase`

<b>Attributes</b>

- `namespace`: `str`
- `endpoint`: `str`
- `router_block_size`: `int`

<b>Methods</b>

#### `validate()`

*Source: [`dynamo/router/args.py#L37`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/router/args.py#L37)*

Validate config invariants (aligned with Rust KvRouterConfig where applicable).

</details>

### Functions

<details>
<summary><strong>`build_kv_router_config()` — Build KvRouterConfig from DynamoRouterConfig.</strong></summary>

#### `build_kv_router_config(router_config: DynamoRouterConfig) -> KvRouterConfig`

*Source: [`dynamo/router/args.py#L101`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/router/args.py#L101)*

# confidence: high
Examples:
    >>> from dynamo.router.args import parse_args, build_kv_router_config
    >>> router_config = parse_args([
    ...     "--endpoint", "dynamo.prefill.generate",
    ... ])
    >>> kv_config = build_kv_router_config(router_config)

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `router_config` | `DynamoRouterConfig` |  |  |


</details>

---

## dynamo.mocker

Mock engine for testing without GPU resources.

### Data Models

<details>
<summary><strong>`ProfileDataResult` — Result of processing --planner-profile-data argument. Cleans up tmpdir on deletion.</strong></summary>

*Source: [`dynamo/mocker/args.py#L28`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/mocker/args.py#L28)*

<b>Attributes</b>

- `npz_path`

<b>Constructor</b>

#### `__init__(npz_path: Path | None, tmpdir: tempfile.TemporaryDirectory | None)`

*Source: [`dynamo/mocker/args.py#L31`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/mocker/args.py#L31)*


</details>

### Functions

<details>
<summary><strong>`get_worker_namespace()` — Get the Dynamo namespace for a worker.</strong></summary>

#### `get_worker_namespace(namespace: Optional[str] = None) -> str`

*Source: [`dynamo/common/utils/namespace.py#L8`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/common/utils/namespace.py#L8)*

Uses the provided namespace, or falls back to the DYN_NAMESPACE environment
variable (defaulting to "dynamo"). If DYN_NAMESPACE_WORKER_SUFFIX is set,
it is appended as "`{namespace}`-`{suffix}`" to support multiple sets of workers
for the same model.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `namespace` | `Optional[str]` | None |  |


</details>

<details>
<summary><strong>`resolve_planner_profile_data()` — Resolve --planner-profile-data to an NPZ file path.</strong></summary>

#### `resolve_planner_profile_data(planner_profile_data: Path | None) -> ProfileDataResult`

*Source: [`dynamo/mocker/args.py#L46`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/mocker/args.py#L46)*

Handles backward compatibility by accepting either:
1. A mocker-format NPZ file (returned as-is)
2. A profiler-style results directory (converted to mocker-format NPZ)

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `planner_profile_data` | `Path | None` | Path from --planner-profile-data argument. |

<b>Returns:</b> `ProfileDataResult` -- ProfileDataResult with npz_path and optional tmpdir for cleanup.

<b>Raises</b>

- `FileNotFoundError` -- If path doesn't contain valid profile data in any supported format.

</details>

<details>
<summary><strong>`create_temp_engine_args_file()` — Create a temporary JSON file with MockEngineArgs from CLI arguments.</strong></summary>

#### `create_temp_engine_args_file(args) -> Path`

*Source: [`dynamo/mocker/args.py#L93`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/mocker/args.py#L93)*

Returns the path to the temporary file.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `args` | `Any` |  |  |


</details>

<details>
<summary><strong>`validate_worker_type_args()` — Resolve disaggregation mode from --disaggregation-mode or legacy boolean flags.</strong></summary>

#### `validate_worker_type_args(args)`

*Source: [`dynamo/mocker/args.py#L149`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/mocker/args.py#L149)*

Raises ValueError if validation fails.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `args` | `Any` |  |  |


</details>

<details>
<summary><strong>`parse_bootstrap_ports()` — Parse comma-separated bootstrap ports string into list of integers.</strong></summary>

#### `parse_bootstrap_ports(ports_str: str | None) -> list[int]`

*Source: [`dynamo/mocker/args.py#L195`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/mocker/args.py#L195)*

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `ports_str` | `str | None` |  |  |


</details>

<details>
<summary><strong>`create_temp_engine_args_file()` — Create a temporary JSON file with MockEngineArgs from CLI arguments.</strong></summary>

#### `create_temp_engine_args_file(args) -> Path`

*Source: [`dynamo/mocker/args.py#L93`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/mocker/args.py#L93)*

Returns the path to the temporary file.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `args` | `Any` |  |  |


</details>

<details>
<summary><strong>`resolve_planner_profile_data()` — Resolve --planner-profile-data to an NPZ file path.</strong></summary>

#### `resolve_planner_profile_data(planner_profile_data: Path | None) -> ProfileDataResult`

*Source: [`dynamo/mocker/args.py#L46`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/mocker/args.py#L46)*

Handles backward compatibility by accepting either:
1. A mocker-format NPZ file (returned as-is)
2. A profiler-style results directory (converted to mocker-format NPZ)

<b>Parameters</b>

| Parameter | Type | Description |
| --- | --- | --- |
| `planner_profile_data` | `Path | None` | Path from --planner-profile-data argument. |

<b>Returns:</b> `ProfileDataResult` -- ProfileDataResult with npz_path and optional tmpdir for cleanup.

<b>Raises</b>

- `FileNotFoundError` -- If path doesn't contain valid profile data in any supported format.

</details>

<details>
<summary><strong>`prefetch_model()` — Pre-fetch model from HuggingFace to avoid rate limiting with many workers.</strong></summary>

#### `prefetch_model(model_path: str)`

*Source: [`dynamo/mocker/main.py#L41`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/mocker/main.py#L41)*

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_path` | `str` |  |  |


</details>

<details>
<summary><strong>`worker()` — Main worker function that launches mocker instances.</strong></summary>

#### `worker()`

*Source: [`dynamo/mocker/main.py#L59`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/mocker/main.py#L59)*

Each mocker gets its own DistributedRuntime instance for true isolation,
while still sharing the same event loop and tokio runtime.

# confidence: high — adapted from real call site
Examples:
    Launch via CLI (the standard invocation pattern):

    >>> # python -m dynamo.mocker --model-path /data/models/Qwen3-0.6B
    >>>
    >>> import uvloop
    >>> from dynamo.mocker.main import worker
    >>> uvloop.run(worker())

    Launch multiple workers with stagger delay:

    >>> # python -m dynamo.mocker \
    >>> #     --model-path /data/models/Qwen3-0.6B \
    >>> #     --num-workers 4 --stagger-delay 0.1

</details>

<details>
<summary><strong>`compute_stagger_delay()` — Compute the stagger delay based on worker count to give the frontend time to process registrations.</strong></summary>

#### `compute_stagger_delay(num_workers: int, stagger_delay: float) -> float`

*Source: [`dynamo/mocker/main.py#L132`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/mocker/main.py#L132)*

Returns the delay in seconds between worker launches.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `num_workers` | `int` |  |  |
| `stagger_delay` | `float` |  |  |


</details>

<details>
<summary><strong>`launch_workers()` — Launch mocker worker(s) with isolated DistributedRuntime instances.</strong></summary>

#### `launch_workers(args, extra_engine_args_path)`

*Source: [`dynamo/mocker/main.py#L154`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/mocker/main.py#L154)*

Each worker gets its own DistributedRuntime, which means:
- Separate etcd/NATS connections
- Separate Component instances (no shared overhead)
- Independent service registration and stats scraping
- But still sharing the same tokio runtime (efficient)

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `args` | `Any` |  |  |
| `extra_engine_args_path` | `Any` |  |  |


</details>

---

## dynamo.nixl_connect

RDMA-based data transfer operations via the NIXL library.

### Enums

<details>
<summary><strong>`DeviceKind` — Type of memory a descriptor has been allocated to.</strong></summary>

*Source: [`dynamo/nixl_connect/__init__.py#L1242`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1242)*

<b>Bases:</b> `IntEnum`

<b>Attributes</b>

- `UNSPECIFIED`
- `HOST` -- System (CPU) memory.
- `CUDA` -- CUDA addressable device (GPU) memory.


</details>

<details>
<summary><strong>`OperationKind` — Kind of an operation.</strong></summary>

*Source: [`dynamo/nixl_connect/__init__.py#L1268`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1268)*

<b>Bases:</b> `IntEnum`

<b>Attributes</b>

- `UNSPECIFIED`
- `READ`
- `WRITE`


</details>

<details>
<summary><strong>`OperationStatus` — Status of an operation.</strong></summary>

*Source: [`dynamo/nixl_connect/__init__.py#L1286`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1286)*

<b>Bases:</b> `IntEnum`

<b>Attributes</b>

- `UNINITIALIZED` -- The operation has not been initialized yet and is not in a valid state.
- `INITIALIZED` -- The operation has been initialized and is ready to be processed.
- `IN_PROGRESS` -- The operation has been initialized and is in-progress (not completed, errored, or cancelled).
- `COMPLETE` -- The operation has been completed successfully.
- `CANCELLED` -- The operation has been cancelled by the user or system.
- `ERRORED` -- The operation has encountered an error and cannot be completed.


</details>

### Configuration

<details>
<summary><strong>`RdmaMetadata` — Pydantic serialization type for describing the passive side of a transfer.</strong></summary>

*Source: [`dynamo/nixl_connect/__init__.py#L1626`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1626)*

<b>Bases:</b> `BaseModel`

<b>Attributes</b>

- `model_config`
- `descriptors`: `List[SerializedDescriptor]`
- `nixl_metadata`: `str`
- `notification_key`: `str`
- `operation_kind`: `int`

<b>Methods</b>

#### `to_descriptors() -> Descriptor | list[Descriptor]`

*Source: [`dynamo/nixl_connect/__init__.py#L1641`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1641)*

Deserializes the request descriptor into a `dynamo.nixl_connect.Descriptor` or list of `dynamo.nixl_connect.Descriptor` objects.
#### `validate_operation_kind(v: int) -> int`

*Source: [`dynamo/nixl_connect/__init__.py#L1653`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1653)*


</details>

<details>
<summary><strong>`SerializedDescriptor` — Pydantic serialization type for memory descriptors.</strong></summary>

*Source: [`dynamo/nixl_connect/__init__.py#L1752`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1752)*

<b>Bases:</b> `BaseModel`

<b>Attributes</b>

- `model_config`
- `device`: `str`
- `ptr`: `int`
- `size`: `int`

<b>Methods</b>

#### `to_descriptor() -> Descriptor`

*Source: [`dynamo/nixl_connect/__init__.py#L1767`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1767)*

Deserialize the serialized descriptor into a `Descriptor` object.
#### `validate_device(v: str) -> str`

*Source: [`dynamo/nixl_connect/__init__.py#L1773`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1773)*

#### `validate_ptr(v: int) -> int`

*Source: [`dynamo/nixl_connect/__init__.py#L1785`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1785)*

#### `validate_size(v: int) -> int`

*Source: [`dynamo/nixl_connect/__init__.py#L1792`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1792)*


</details>

### Classes

<details>
<summary><strong>`AbstractOperation` — Abstract base class for awaitable NIXL based RDMA operations.</strong></summary>

*Source: [`dynamo/nixl_connect/__init__.py#L65`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L65)*

<b>Attributes</b>

- `connection`: `Connection` -- Gets the local connection associated with this operation.
- `operation_kind`: `OperationKind` -- Gets the kind of operation.

<b>Constructor</b>

#### `__init__(connection: Connection, operation_kind: OperationKind, local_descriptors: Descriptor | list[Descriptor], remote_descriptors: Optional[Descriptor | list[Descriptor]], notification_key: Optional[str])`

*Source: [`dynamo/nixl_connect/__init__.py#L70`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L70)*

<b>Methods</b>

#### `wait_for_completion()`

*Source: [`dynamo/nixl_connect/__init__.py#L209`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L209)*

Blocks the caller asynchronously until the operation has completed.

</details>

<details>
<summary><strong>`ActiveOperation` — Abstract class for active operations that initiates a NIXL based RDMA transfer based `RdmaMetadata`</strong></summary>

*Source: [`dynamo/nixl_connect/__init__.py#L243`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L243)*

provided by the remote worker's corresponding `PassiveOperation`.
<b>Bases:</b> `AbstractOperation`

<b>Attributes</b>

- `remote`: `Remote` -- Gets the remote worker associated with this operation.
- `status`: `OperationStatus` -- Gets the status of the operation.

<b>Constructor</b>

#### `__init__(remote: Remote, operation_kind: OperationKind, local_descriptors: Descriptor | list[Descriptor], remote_descriptors: Descriptor | list[Descriptor], notification_key: str)`

*Source: [`dynamo/nixl_connect/__init__.py#L249`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L249)*

<b>Methods</b>

#### `cancel()`

*Source: [`dynamo/nixl_connect/__init__.py#L465`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L465)*

Cancels the operation.
No affect if the operation has already completed or errored, or has been cancelled.

</details>

<details>
<summary><strong>`Connector` — Core class for managing the connection between workers in a distributed environment.</strong></summary>

*Source: [`dynamo/nixl_connect/__init__.py#L613`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L613)*

Use this class to create readable and writable operations, or read and write data to remote workers.
<b>Attributes</b>

- `hostname`: `str` -- Get the name of the current worker's host.
- `is_cuda_available`: `bool`
- `name`: `str | None` -- Get the name of the worker.

<b>Constructor</b>

#### `__init__(worker_id: Optional[str] = None)`

*Source: [`dynamo/nixl_connect/__init__.py#L619`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L619)*

Creates a new Connector instance.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `worker_id` | `Optional[str]` | None |  |

<b>Methods</b>

#### `begin_read(remote_metadata: RdmaMetadata, local_descriptors: Descriptor | list[Descriptor]) -> ReadOperation`

*Source: [`dynamo/nixl_connect/__init__.py#L696`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L696)*

Creates a read operation for fulfilling a remote readable operation.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `remote_metadata` | `RdmaMetadata` |  |  |
| `local_descriptors` | `Descriptor | list[Descriptor]` |  |  |

#### `begin_write(local_descriptors: Descriptor | list[Descriptor], remote_metadata: RdmaMetadata) -> WriteOperation`

*Source: [`dynamo/nixl_connect/__init__.py#L744`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L744)*

Creates a write operation for transferring data to a remote worker.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `local_descriptors` | `Descriptor | list[Descriptor]` |  |  |
| `remote_metadata` | `RdmaMetadata` |  |  |

#### `create_readable(local_descriptors: Descriptor | list[Descriptor]) -> ReadableOperation`

*Source: [`dynamo/nixl_connect/__init__.py#L782`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L782)*

Creates a readable operation for transferring data from a remote worker.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `local_descriptors` | `Descriptor | list[Descriptor]` |  |  |

#### `create_writable(local_descriptors: Descriptor | list[Descriptor]) -> WritableOperation`

*Source: [`dynamo/nixl_connect/__init__.py#L798`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L798)*

Creates a writable operation for transferring data to a remote worker.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `local_descriptors` | `Descriptor | list[Descriptor]` |  |  |

#### `initialize()`

*Source: [`dynamo/nixl_connect/__init__.py#L814`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L814)*

Deprecated method.

</details>

<details>
<summary><strong>`Descriptor` — Memory descriptor that ensures memory is registered w/ NIXL, used for transferring data between workers.</strong></summary>

*Source: [`dynamo/nixl_connect/__init__.py#L832`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L832)*

<b>Attributes</b>

- `device`: `Device` -- Gets the device the of the descriptor.
- `is_registered`: `bool` -- Gets whether the descriptor is registered with NIXL.
- `ptr`: `int` -- Gets the pointer of the descriptor.
- `size`: `int` -- Gets the size of the descriptor.
- `metadata`: `SerializedDescriptor` -- Serializes the descriptor into a `SerializedDescriptor` object.

<b>Constructor</b>

#### `__init__(data: torch.Tensor | tuple[array_module.ndarray, Device | str] | bytes | tuple[int, int, Device | str, Any])`

*Source: [`dynamo/nixl_connect/__init__.py#L837`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L837)*

Memory descriptor for transferring data between workers.



    When `torch.Tensor` is provided, the attributes of the tensor will be used to create the descriptor.

    When `tuple[ndarray, Device]` is provided, the tuple must contain:
    - `ndarray`: The CuPy or NumPy array to be transferred.
    - `Device`: Either a `dynamo.nixl_connect.Device` or a string representing the device type (e.g., "cuda" or "cpu").

    When `bytes` is provided, the pointer and size derived from the bytes object and memory type will be assumed to be CPU.

    When `tuple[int, int, Device|str, Any]` is provided, the tuple must contain the following elements:
    - `int`: Pointer to the data in memory.
    - `int`: Size of the data in bytes.
    - `Device`: Either a `dynamo.nixl_connect.Device` or a string representing the device type (e.g., "cuda" or "cpu").
    - `Any`: Optional reference to the data (e.g., the original tensor or bytes object).
             This is useful for keeping a reference to the data in memory, but it is not required.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `data` | `torch.Tensor | tuple[array_module.ndarray, Device | str] | bytes | tuple[int, int, Device | str, Any]` |  |  |

<b>Methods</b>

#### `from_serialized(serialized: SerializedDescriptor) -> Descriptor`

*Source: [`dynamo/nixl_connect/__init__.py#L1024`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1024)*

Deserializes a `SerializedDescriptor` into a `Descriptor` object.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `serialized` | `SerializedDescriptor` |  |  |

#### `deregister_with_connector(connection: Connection)`

*Source: [`dynamo/nixl_connect/__init__.py#L1060`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1060)*

Deregisters the memory of the descriptor with NIXL.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `connection` | `Connection` |  |  |

#### `register_with_connector(connection: Connection)`

*Source: [`dynamo/nixl_connect/__init__.py#L1088`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1088)*

Registers the memory of the descriptor with NIXL.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `connection` | `Connection` |  |  |


</details>

<details>
<summary><strong>`Device` — Represents a device in the system.</strong></summary>

*Source: [`dynamo/nixl_connect/__init__.py#L1177`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1177)*

<b>Attributes</b>

- `id`: `int` -- Gets the device ID of the device.
- `kind`: `DeviceKind` -- Gets the memory kind of the device.

<b>Constructor</b>

#### `__init__(metadata: str | tuple[DeviceKind, int])`

*Source: [`dynamo/nixl_connect/__init__.py#L1182`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1182)*


</details>

<details>
<summary><strong>`PassiveOperation` — Abstract class for common functionality of passive operations.</strong></summary>

*Source: [`dynamo/nixl_connect/__init__.py#L1324`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1324)*

<b>Bases:</b> `AbstractOperation`

<b>Attributes</b>

- `status`: `OperationStatus` -- Gets the status of the operation.

<b>Constructor</b>

#### `__init__(connection: Connection, operation_kind: OperationKind, local_descriptors: Descriptor | list[Descriptor])`

*Source: [`dynamo/nixl_connect/__init__.py#L1329`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1329)*

<b>Methods</b>

#### `metadata(hex_encode: bool = False) -> RdmaMetadata`

*Source: [`dynamo/nixl_connect/__init__.py#L1392`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1392)*

Gets the request descriptor for the operation.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `hex_encode` | `bool` | False |  |

#### `wait_for_completion()`

*Source: [`dynamo/nixl_connect/__init__.py#L1474`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1474)*

Blocks the caller asynchronously until the operation has completed.

</details>

<details>
<summary><strong>`ReadOperation` — Operation that initiates an RDMA read operation to transfer data from a remote worker's `ReadableOperation`,</strong></summary>

*Source: [`dynamo/nixl_connect/__init__.py#L1482`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1482)*

as described by `remote_metadata`, to local buffers.
<b>Bases:</b> `ActiveOperation`

<b>Constructor</b>

#### `__init__(connection: Connection, remote_metadata: RdmaMetadata, local_descriptors: Descriptor | list[Descriptor])`

*Source: [`dynamo/nixl_connect/__init__.py#L1488`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1488)*

Creates a new instance of `ReadOperation`, registers `local_descriptors` with NIXL,
and begins an RDMA read operation which will transfer data described by `remote_metadata`
to `local_descriptors`.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `connection` | `Connection` |  |  |
| `remote_metadata` | `RdmaMetadata` |  |  |
| `local_descriptors` | `Descriptor | list[Descriptor]` |  |  |

<b>Methods</b>

#### `cancel()`

*Source: [`dynamo/nixl_connect/__init__.py#L1560`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1560)*

Cancels the operation.
No affect if the operation has already completed or errored, or been cancelled.
#### `results() -> list[Descriptor]`

*Source: [`dynamo/nixl_connect/__init__.py#L1567`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1567)*

Gets the results of the operation.
Returns a single descriptor if only one was requested, or a list of descriptors if multiple were requested.
#### `wait_for_completion()`

*Source: [`dynamo/nixl_connect/__init__.py#L1581`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1581)*

Blocks the caller asynchronously until the operation has completed.

</details>

<details>
<summary><strong>`ReadableOperation` — Operation that can be awaited until a remote worker has completed a `ReadOperation`.</strong></summary>

*Source: [`dynamo/nixl_connect/__init__.py#L1588`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1588)*

<b>Bases:</b> `PassiveOperation`

<b>Constructor</b>

#### `__init__(connection: Connection, local_descriptors: Descriptor | list[Descriptor])`

*Source: [`dynamo/nixl_connect/__init__.py#L1593`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1593)*

<b>Methods</b>

#### `wait_for_completion()`

*Source: [`dynamo/nixl_connect/__init__.py#L1619`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1619)*

Blocks the caller asynchronously until the operation has completed.

</details>

<details>
<summary><strong>`Remote` — Identifies a remote NIXL enabled worker relative to a local NIXL enabled worker.</strong></summary>

*Source: [`dynamo/nixl_connect/__init__.py#L1663`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1663)*

<b>Attributes</b>

- `connection`: `Connection` -- Gets the local connection associated with this remote worker.
- `name`: `str` -- Gets the name of the remote worker.

<b>Constructor</b>

#### `__init__(connection: Connection, nixl_metadata: bytes | str)`

*Source: [`dynamo/nixl_connect/__init__.py#L1668`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1668)*


</details>

<details>
<summary><strong>`WritableOperation` — Operation which can be awaited until written to by a `WriteOperation` from a remote worker.</strong></summary>

*Source: [`dynamo/nixl_connect/__init__.py#L1802`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1802)*

<b>Bases:</b> `PassiveOperation`

<b>Constructor</b>

#### `__init__(connection: Connection, local_descriptors: Descriptor | list[Descriptor])`

*Source: [`dynamo/nixl_connect/__init__.py#L1807`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1807)*

Creates a new instance of `WritableOperation`, registers the operation and descriptors w/ NIXL,
and enables an RDMA write operation to occur.



Raises
TypeError
    When `connection` is not a `dynamo.nixl_connect.Connection`.
TypeError
    When `local_descriptors` is not a `dynamo.nixl_connect.Descriptor` or `list[dynamo.nixl_connect.Descriptor]`.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `connection` | `Connection` |  |  |
| `local_descriptors` | `Descriptor | list[Descriptor]` |  |  |

<b>Methods</b>

#### `wait_for_completion()`

*Source: [`dynamo/nixl_connect/__init__.py#L1850`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1850)*

Blocks the caller asynchronously until the operation has completed.

</details>

<details>
<summary><strong>`WriteOperation` — Awaitable write operation which initiates an RDMA write operation to a remote worker</strong></summary>

*Source: [`dynamo/nixl_connect/__init__.py#L1857`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1857)*

which provided a `RdmaMetadata` object from a `WritableOperation`.
<b>Bases:</b> `ActiveOperation`

<b>Constructor</b>

#### `__init__(connection: Connection, local_descriptors: Descriptor | list[Descriptor], remote_metadata: RdmaMetadata)`

*Source: [`dynamo/nixl_connect/__init__.py#L1863`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1863)*

Creates a new instance of `WriteOperation`, registers `local_descriptors` with NIXL,
and begins an RDMA write operation which will transfer from `local_descriptors` to
remote target(s) described by `remote_metadata`



Raises
TypeError
    When `connector` is not a `dynamo.nixl_connect.Connector`.
TypeError
    When `remote_metadata` is not a `dynamo.nixl_connect.RdmaMetadata`.
ValueError
    When `remote_metadata` is not of kind `WRITE`.
ValueError
    When `remote_metadata.nixl_metadata` is not a non-empty `str`.
TypeError
    When `local_descriptors` is not a `dynamo.nixl_connect.Descriptor` or `list[dynamo.nixl_connect.Descriptor]`.

<b>Parameters</b>

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `connection` | `Connection` |  |  |
| `local_descriptors` | `Descriptor | list[Descriptor]` |  |  |
| `remote_metadata` | `RdmaMetadata` |  |  |

<b>Methods</b>

#### `cancel()`

*Source: [`dynamo/nixl_connect/__init__.py#L1936`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1936)*

Cancels the operation.
No affect if the operation has already completed or errored, or has been cancelled.
#### `wait_for_completion()`

*Source: [`dynamo/nixl_connect/__init__.py#L1943`](https://github.com/ai-dynamo/dynamo/blob/main/dynamo/nixl_connect/__init__.py#L1943)*

Blocks the caller asynchronously until the operation has completed.

</details>

---

*Source packages: `dynamo._core`, `dynamo.runtime`, `dynamo.llm`, `dynamo.frontend`, `dynamo.common`, `dynamo.health_check`, `dynamo.logits_processing`, `dynamo.planner`, `dynamo.router`, `dynamo.mocker`, `dynamo.nixl_connect`*
