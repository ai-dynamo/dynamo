---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Request Cancellation Architecture"
---

This document describes how Dynamo implements request cancellation to cancel in-flight requests between Dynamo workers. Request cancellation allows in-flight requests to terminate early, saving computational resources that would otherwise be spent on responses that are no longer needed.

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

This design enables seamless cancellation propagation through multi-tier request chains, ensuring that when a client cancels a request, all associated sub-requests are automatically cancelled, saving computational resources across the entire request pipeline.
