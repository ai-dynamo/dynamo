---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Python Route Extensions
---

Python route extensions let an external package register additional HTTP routes on the Dynamo frontend, served on the same port as the OpenAI-compatible API — without a custom binary or a from-source build.

Extensions are **opt-in**: the frontend only loads a provider you explicitly select on the command line. A selection is either a **name** registered under the `dynamo.frontend.routes` entry-point group (preferred for packaged plugins) or a direct **`module:function`** path (handy for quick/ad-hoc use). Extensions add routes; they cannot override a built-in route (a duplicate method+path is rejected at startup).

The initial contract is intentionally narrow: **static-path `GET` routes** with a synchronous handler. Path parameters (`/{id}`), wildcards, non-`GET` methods, and `async def` handlers are rejected at construction. This surface can grow later without breaking existing extensions.

## Minimal example

**1. Write a route provider.** A provider is a callable that returns a `FrontendRoute` (or an iterable of them). Each handler is synchronous, receives a `FrontendExtensionContext`, and returns a JSON-serializable body (HTTP 200) or a `FrontendResponse` to set the status code.

```python
# hello_routes.py
from dynamo.llm import FrontendRoute, FrontendExtensionContext


def _hello(ctx: FrontendExtensionContext):
    return {"message": "hello world!"}


def hello_world_routes():
    return [FrontendRoute("GET", "/hello_world", _hello)]
```

**2. Register it as an entry point** under the `dynamo.frontend.routes` group. The entry-point name (here `hello-world`) is what you pass on the command line.

```toml
# pyproject.toml
[project.entry-points."dynamo.frontend.routes"]
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

## Quick / ad-hoc: `module:function`

For development or a one-off deployment where packaging a plugin is overkill, pass a `module:function` path directly instead of a registered name — no `pyproject.toml` or install required, as long as the module is importable (e.g. on `PYTHONPATH`):

```bash
python -m dynamo.frontend --frontend-route-extension hello_routes:hello_world_routes
```

A registered entry-point name always takes precedence; the path fallback only applies when the value is not a registered name and contains `:`.

## Handler contract

- **Route:** `GET` only, static path (no `{param}`/`*wildcard`) — enforced at `FrontendRoute` construction.
- **Signature:** `handler(ctx: FrontendExtensionContext)` — synchronous. `async def` handlers are rejected at construction.
- **Return:** a JSON-serializable value (implies `200`), or `FrontendResponse(status_code, body)` to set the status. Ordinary tuples serialize as JSON arrays — use `FrontendResponse` for status overrides:

  ```python
  from dynamo.llm import FrontendResponse

  def _health_ready(ctx):
      body = {"status": "ready" if ctx.has_any_ready_model() else "not ready"}
      return body if ctx.has_any_ready_model() else FrontendResponse(503, body)
  ```

- **Live state:** `FrontendExtensionContext` exposes the current frontend state so responses reflect models registering/draining at runtime — e.g. `ctx.has_any_ready_model()`, `ctx.serving_ready_display_names()`, `ctx.is_model_ready_to_serve(name)`, `ctx.is_ready()`.
- **Keep handlers fast and non-blocking.** Handlers run on a small, dedicated thread pool (isolated from the inference tokenization pool), so a slow handler degrades only other extension routes, not inference. Still, the pool is small and GIL-serialized: a handler that exceeds ~30s is treated as hung and its request returns `503`, and a saturated pool sheds new extension requests with `503`. Do not block on network/disk or run heavy compute inline.

## Notes

- Select multiple extensions by repeating `--frontend-route-extension` (or via a **whitespace-separated** `DYN_FRONTEND_ROUTE_EXTENSIONS`). Names are de-duplicated.
- Passing an unknown name fails fast and lists the available registered extensions.
- Extensions apply to the HTTP frontend only.
