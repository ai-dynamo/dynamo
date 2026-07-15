---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Python Route Extensions
---

Python route extensions let an external package register additional HTTP routes on the Dynamo frontend, served on the same port as the OpenAI-compatible API — without a custom binary or a from-source build.

Extensions are **opt-in by name**: the frontend only loads a provider you explicitly select on the command line, and only from providers registered under the `dynamo.frontend_routes` entry-point group. They add routes; they cannot change or override the built-in inference routes (a duplicate method+path is rejected at startup).

## Minimal example

**1. Write a route provider.** A provider is a callable that returns a `FrontendRoute` (or an iterable of them). Each handler is synchronous, receives a `FrontendRouteContext`, and returns a JSON-serializable body (HTTP 200) or a `(status_code, body)` tuple.

```python
# hello_routes.py
from dynamo.llm import FrontendRoute, FrontendRouteContext


def _hello(ctx: FrontendRouteContext):
    return {"message": "hello world!"}


def hello_world_routes():
    return [FrontendRoute("GET", "/hello_world", _hello)]
```

**2. Register it as an entry point** under the `dynamo.frontend_routes` group. The entry-point name (here `hello-world`) is what you pass on the command line.

```toml
# pyproject.toml
[project.entry-points."dynamo.frontend_routes"]
hello-world = "hello_routes:hello_world_routes"
```

**3. Install the package** so the entry point is discoverable:

```bash
pip install -e .
```

**4. Launch the frontend** with the extension selected:

```bash
python -m dynamo.frontend --frontend-route-extension hello-world
# equivalently: DYN_FRONTEND_ROUTE_EXTENSIONS="hello-world" python -m dynamo.frontend
```

**5. Call the route:**

```bash
curl localhost:8000/hello_world
# {"message":"hello world!"}
```

## Handler contract

- **Signature:** `handler(ctx: FrontendRouteContext)` — synchronous. Async handlers are rejected.
- **Return:** a JSON-serializable value (implies `200`), or a `(status_code, body)` tuple to set the status.
- **Live state:** `FrontendRouteContext` exposes the current frontend state so responses reflect models registering/draining at runtime — e.g. `ctx.has_any_ready_model()`, `ctx.serving_ready_display_names()`, `ctx.is_model_ready_to_serve(name)`, `ctx.is_ready()`.

## Notes

- Select multiple extensions by repeating `--frontend-route-extension` (or via a space/comma-separated `DYN_FRONTEND_ROUTE_EXTENSIONS`). Names are de-duplicated.
- Passing an unknown name fails fast and lists the available registered extensions.
- Extensions apply to the HTTP frontend only.
