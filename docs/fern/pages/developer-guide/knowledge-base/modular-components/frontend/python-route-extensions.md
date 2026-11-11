---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Python Route Extensions
subtitle: Add trusted HTTP routes to the Dynamo Frontend from an installed Python package or importable module.
---

Python route extensions let an external package register additional HTTP routes on the Dynamo
Frontend. The routes use the same port as the OpenAI-compatible API and do not require a custom
binary or source build.

Extensions are opt-in. The Frontend loads only the providers selected with
`--frontend-route-extension` or `DYN_FRONTEND_ROUTE_EXTENSIONS`. A selection can be either:

- A name registered under the `dynamo.frontend.routes` entry-point group, recommended for packaged
  extensions.
- A direct `module:function` path for development or one-off deployments.

Extensions add routes but cannot override built-in routes. A duplicate method and path combination
causes Frontend startup to fail.

> [!NOTE]
> Routes use static paths and one of the `GET`, `POST`, `PUT`, `PATCH`, or `DELETE` methods. Path
> parameters and wildcards are rejected.

## Add a route extension

### Define a route provider

A provider is a callable that returns a `FrontendRoute` or an iterable of routes. Each handler
receives a `FrontendExtensionContext` and returns either:

- A JSON-serializable body, which produces HTTP 200.
- A `FrontendResponse` with an explicit status code and body.

```python
# hello_routes.py
from dynamo.llm import FrontendExtensionContext, FrontendRoute


def _hello(ctx: FrontendExtensionContext):
    return {"message": "hello world!"}


def hello_world_routes():
    return [FrontendRoute("GET", "/hello_world", _hello)]
```

### Register the provider

For an installable package, register the provider under the `dynamo.frontend.routes` entry-point
group. The entry-point name is the value passed to the Frontend.

```toml
# pyproject.toml
[project.entry-points."dynamo.frontend.routes"]
hello-world = "hello_routes:hello_world_routes"
```

Install the package so the entry point is discoverable:

```bash
pip install -e .
```

### Start the Frontend

Select the registered provider by name:

```bash
python -m dynamo.frontend --frontend-route-extension hello-world
```

The equivalent environment-variable form is:

```bash
DYN_FRONTEND_ROUTE_EXTENSIONS="hello-world" python -m dynamo.frontend
```

Call the added route:

```bash
curl localhost:8000/hello_world
```

The response is:

```json
{"message": "hello world!"}
```

## Load an importable provider directly

For development or a one-off deployment, pass an importable `module:function` path. The module must
be importable, for example by being available on `PYTHONPATH`.

```bash
python -m dynamo.frontend \
  --frontend-route-extension hello_routes:hello_world_routes
```

A registered entry-point name takes precedence. The `module:function` fallback applies only when no
registered name matches and the value contains `:`.

## Return a custom status

Return `FrontendResponse` when a handler needs a status other than HTTP 200:

```python
from dynamo.llm import FrontendResponse


def _health_ready(ctx):
    body = {"status": "ready" if ctx.has_any_ready_model() else "not ready"}
    return body if ctx.has_any_ready_model() else FrontendResponse(503, body)
```

`FrontendExtensionContext` provides a read-only view of live Frontend state through these methods:

- `is_ready()`
- `is_cancelled()`
- `has_any_ready_model()`
- `is_model_ready_to_serve(name)`
- `model_display_names()`
- `serving_ready_display_names()`

## Read the request and call workers

A handler can be an `async def`. Async handlers run on the Frontend's asyncio event loop, so they can
await request-plane calls to workers. The context also carries the request and the Frontend's
runtime:

- `body`: the raw request body as `bytes`, empty when the request has none.
- `headers`: the request headers as a new `dict`, with lower-case names. Repeated headers are joined
  with `", "`.
- `runtime`: the Frontend's own `DistributedRuntime`, for clients to worker endpoints. Its async
  methods need the event loop, so call them from an `async def` handler.

The following route takes a JSON body, checks a bearer token, sends the body to every instance of a
worker endpoint, and returns the replies:

```python
import hmac
import json
import os

from dynamo.llm import FrontendExtensionContext, FrontendResponse, FrontendRoute

_BEARER = f"Bearer {os.environ['MY_CONTROL_TOKEN']}".encode()
_client = None


async def _broadcast(ctx: FrontendExtensionContext):
    supplied = ctx.headers.get("authorization", "").encode()
    if not hmac.compare_digest(supplied, _BEARER):
        return FrontendResponse(401, {"error": "invalid bearer"})
    try:
        request = json.loads(ctx.body)
    except ValueError:
        return FrontendResponse(400, {"error": "body must be JSON"})

    global _client
    if _client is None:
        _client = await ctx.runtime.endpoint("dynamo.backend.control").client()

    replies = {}
    for instance_id in await _client.wait_for_instances():
        stream = await _client.direct(request, instance_id)
        async for reply in stream:
            replies[str(instance_id)] = reply.data()
    return {"replies": replies}


def control_routes():
    return [FrontendRoute("POST", "/control/broadcast", _broadcast, timeout_s=120)]
```

`timeout_s` sets the route's handler timeout. It defaults to 30 seconds.

## Handler limits

Synchronous handlers execute in a small, dedicated thread pool, separate from inference
tokenization. That pool also starts each async handler, which then runs on the Frontend's event loop.
Keep handlers fast and non-blocking:

- A handler that exceeds its route's `timeout_s` (30 seconds by default, measured from request
  arrival) returns HTTP 503. An async handler is cancelled at the timeout, and a call the event loop
  picks up after its timeout is dropped without running the handler. A 503 does not mean the handler
  had no effect: it may have been cancelled partway.
- A client that disconnects does not cancel a running handler.
- Each route admits at most 64 calls at once, waiting or running. A request beyond that, or one that
  finds the thread pool's queue full, is rejected with the configured overload status code, HTTP 529
  by default. An async handler holds its slot until it finishes or is cancelled.
- A handler that raises returns HTTP 500. That includes a `TimeoutError` the handler raises itself
  before the route's timeout.
- Request bodies are limited by `DYN_HTTP_BODY_LIMIT_MB`, 45 MB by default, the same limit as the
  inference routes.
- Slow work in a synchronous handler delays other extension routes: it occupies a pool thread, and
  Python code holds the Global Interpreter Lock (GIL). In an async handler, a blocking call stalls
  the Frontend's event loop, which also runs Python chat processors (`--dyn-chat-processor vllm` or
  `sglang`) and shutdown handling. Await the work or move it to a thread.

## Load multiple providers

Repeat `--frontend-route-extension` to load multiple providers:

```bash
python -m dynamo.frontend \
  --frontend-route-extension health-routes \
  --frontend-route-extension metadata-routes
```

For the environment variable, use whitespace-separated values:

```bash
DYN_FRONTEND_ROUTE_EXTENSIONS="health-routes metadata-routes" \
  python -m dynamo.frontend
```

Provider names are de-duplicated. An unknown name fails startup and reports the available registered
extensions.

For the complete flag and handler contract reference, see
[Frontend Configuration Reference](../../../../reference/components/frontend-configuration.mdx#python-route-extensions).
