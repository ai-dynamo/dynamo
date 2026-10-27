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

<Note>
Python route extensions currently support synchronous handlers on static-path `GET` routes only.
Path parameters, wildcards, other HTTP methods, and asynchronous handlers are rejected.
</Note>

## Add a route extension

### Define a route provider

A provider is a callable that returns a `FrontendRoute` or an iterable of routes. Each synchronous
handler receives a `FrontendExtensionContext` and returns either:

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

## Handler limits

Handlers execute in a small, dedicated thread pool, separate from inference tokenization. Keep them
fast and non-blocking:

- A handler that exceeds 30 seconds returns HTTP 503.
- A saturated extension pool rejects new requests with the configured overload status code, HTTP
  529 by default.
- Handler failures return HTTP 500.
- Network, disk, or compute-heavy operations can block other extension routes because Python
  handlers contend for the Global Interpreter Lock (GIL).

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
