---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Request Cancellation
---

This document describes how Dynamo implements request cancellation to cancel in-flight requests between Dynamo workers. Request cancellation allows in-flight requests to terminate early, saving computational resources that would otherwise be spent on responses that are no longer needed.

## How Cancellation Works

### Frontend Cancellation Detection

The frontend monitors each client connection for unexpected disconnects. When a client disconnects before the response is fully delivered, the frontend detects this and initiates cancellation. This covers two scenarios:

1. **Connection closed unexpectedly** — The client disconnects during request processing before response streaming begins.
2. **Stream closed unexpectedly** — The client disconnects while an active SSE stream is delivering response tokens.

In both cases, the frontend cancels the request's `AsyncEngineContext`, which propagates cancellation to any linked child contexts on downstream workers.

### Worker Cancellation Detection

On the worker side, the runtime monitors the TCP connection from the frontend for cancellation signals. The worker detects cancellation in three scenarios:

1. **Control message received** — The frontend explicitly sent a cancellation control message.
2. **TCP connection dropped** — The frontend disconnected without sending a control message (e.g., frontend crash or network failure).

When the worker receives a cancellation signal, it sets the corresponding state on the request's `AsyncEngineContext`. It is then up to the worker's engine implementation to observe the cancellation (e.g., by checking `is_stopped()`) and terminate processing accordingly. For details on implementing cancellation handling in a backend worker, see the [Backend Development Guide](../development/backend-guide.md#request-cancellation).

### Cancellation Propagation

Cancellation propagates through multi-tier request chains via linked `AsyncEngineContext` objects. When a parent context is cancelled, all linked child contexts are automatically cancelled as well. This ensures that when a client cancels a request at the frontend, all associated sub-requests on downstream workers are automatically cancelled, saving computational resources across the entire request pipeline.

## Metrics

Dynamo exposes Prometheus metrics to monitor request cancellations at both the frontend and runtime layers.

### Frontend Metric

| Metric | Type | Description |
|--------|------|-------------|
| `dynamo_frontend_model_cancellation_total` | Counter | Total number of request cancellations detected by the frontend |

#### Labels

| Label | Description | Example Values |
|-------|-------------|----------------|
| `model` | The model name from the request | `Qwen/Qwen3-0.6B` |
| `endpoint` | The API endpoint that received the request | `completions`, `chat_completions`, `embeddings`, `images`, `videos`, `audios`, `responses`, `anthropic_messages`, `tensor` |
| `request_type` | Whether the request was unary or streaming | `unary`, `stream` |

**Endpoint:** Available on the frontend HTTP service at `/metrics`.

### Runtime Metric

| Metric | Type | Description |
|--------|------|-------------|
| `dynamo_component_cancellation_total` | Counter | Total number of requests cancelled by the work handler |

This metric uses Dynamo's auto-injected component labels:

| Label | Description | Example Values |
|-------|-------------|----------------|
| `dynamo_namespace` | The Dynamo namespace | `dynamo` |
| `dynamo_component` | The component that handled the request | `backend`, `prefill`, `decode` |
| `dynamo_endpoint` | The endpoint within the component | `generate` |

The counter uses deduplication logic to ensure that each cancelled request is only counted once, even if both a control message and a socket close are detected for the same request.

Note that this metric records cancellation signals received by the worker, not whether the request was actually aborted at the engine level. It is up to the worker's engine implementation to observe the cancellation (e.g., by checking `is_stopped()`) and terminate processing accordingly.

**Endpoint:** Available on the worker system metrics port at `/metrics` (typically port 9100).

### Example Metrics Output

Frontend metrics (from `/metrics` on the frontend HTTP service):
```text
dynamo_frontend_model_cancellation_total{endpoint="chat_completions",model="Qwen/Qwen3-0.6B",request_type="stream"} 5
dynamo_frontend_model_cancellation_total{endpoint="chat_completions",model="Qwen/Qwen3-0.6B",request_type="unary"} 1
dynamo_frontend_model_cancellation_total{endpoint="completions",model="Qwen/Qwen3-0.6B",request_type="stream"} 2
```

Runtime metrics (from `/metrics` on the worker system port):
```text
dynamo_component_cancellation_total{dynamo_component="backend",dynamo_endpoint="generate",dynamo_namespace="dynamo"} 8
```

## AsyncEngineContext Trait

At the core of Dynamo's request cancellation system is the `AsyncEngineContext` trait. This trait is associated with every request stream and provides lifecycle management for async operations, including stream identification, graceful shutdown capabilities, and immediate termination capabilities.

### Key Methods

#### Identification
- **`id()`**: Returns the unique identifier for the stream. This ID is set by the user for request identification, and the same ID can be used for sub-requests to associate them with the original user request.

#### Status Checking
- **`is_stopped()`**: Returns `true` if graceful cancellation has been requested via `stop_generating()`. This represents a signal to the worker that the request has been cancelled and it should return early.
- **`is_killed()`**: Returns `true` if a hard stop has been issued via `kill()`. This typically indicates that the network connection between client and server has been cut or an immediate termination is required.

#### Async Status Monitoring
- **`stopped()`**: An async method that completes when the context becomes stopped. If already stopped, returns immediately.
- **`killed()`**: An async method that completes when the context becomes killed. If already killed, returns immediately.

#### Cancellation Control
- **`stop_generating()`**: The recommended method for cancelling a request. This informs the engine to stop producing results for the stream gracefully. This method is idempotent and does not invalidate results currently in the stream.
- **`stop()`**: Alias for `stop_generating()`.
- **`kill()`**: Extends `stop_generating()` but also indicates a preference to terminate without draining remaining items in the stream. This is implementation-specific and may not be supported by all engines.

#### Child Request Management
- **`link_child(child: Arc<dyn AsyncEngineContext>)`**: Links a child `AsyncEngineContext` to this context. When `stop_generating()`, `stop()`, or `kill()` is called on the parent context, the same method is automatically called on all linked child contexts in the order they were linked. This is especially useful in disaggregated serving scenarios where a frontend receives cancellation notification and needs to cancel requests to workers, and the worker can then cancel its sub-requests (e.g., remote prefill operations).

### Thread Safety

The `AsyncEngineContext` trait ensures thread-safety with `Send + Sync` bounds, allowing safe concurrent access across multiple threads and async tasks.

## Python Bindings

The `AsyncEngineContext` functionality is exposed to Python through the `Context` class, which provides a largely one-to-one mapping from Rust methods to Python methods.

### Python Context Class

The Python `Context` class wraps the Rust `AsyncEngineContext` and exposes the following methods:

- **`id()`**: Returns the unique identifier for the context
- **`is_stopped()`**: Synchronous method equivalent to the Rust `is_stopped()`
- **`is_killed()`**: Synchronous method equivalent to the Rust `is_killed()`
- **`stop_generating()`**: Issues a stop generating signal, equivalent to the Rust method
- **`async_killed_or_stopped()`**: An async method that completes when the context becomes either killed or stopped, whichever happens first. This combines the functionality of the Rust `killed()` and `stopped()` async methods using `tokio::select!`.

For a working example of request cancellation, see the [cancellation demo](https://github.com/ai-dynamo/dynamo/tree/main/examples/custom_backend/cancellation/README.md).

### Context Usage in Python

The context is available optionally in both incoming and outgoing request scenarios:

#### Incoming Requests
For incoming requests, the generate method may optionally accept a `context` argument after the `request` argument. If the `context` parameter is specified in the method signature, it will receive the context object of the incoming request. Request handlers can:

- Check for cancellation synchronously using `context.is_stopped()` before beginning expensive operations
- Listen for cancellation asynchronously using `await context.async_killed_or_stopped()`

Example:
```python
async def generate(self, request, context):
    for i in range(1000):
        # Check for cancellation before expensive work
        if context.is_stopped():
            raise asyncio.CancelledError

        # Perform work...
        await expensive_computation()
        yield result
```

#### Outgoing Requests
For outgoing requests, Python scripts may optionally provide a context object to outgoing runtime endpoint client router operations (such as `generate`, `round_robin`, `random`, `direct` methods) as a keyword argument. The script can cancel the outgoing request via the provided context object.

This is especially useful when child outgoing requests need to be cancelled when the parent incoming request is cancelled. In such cases, the script can simply pass the incoming context object to the outgoing request, automatically linking the cancellation behavior.

Example:
```python
async def generate(self, request, context):
    # Forward the incoming context to outgoing request
    # If the incoming request is cancelled, the outgoing request will be too
    stream = await self.client.generate(request, context=context)
    async for response in stream:
        yield response
```
