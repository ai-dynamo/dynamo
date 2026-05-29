---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Writing a Python Unified Backend
---

# Writing a Python Unified Backend

> **New — Dynamo's unified backend.** This guide covers the new
> **unified backend** infrastructure in
> [`dynamo.common.backend`](https://github.com/ai-dynamo/dynamo/tree/main/components/src/dynamo/common/backend):
> a shared `LLMEngine` ABC that vLLM, SGLang, TRT-LLM, and a sample
> engine already implement, and that any custom Python engine can plug
> into the same way. For the Rust version of the same contract see
> [Writing a Rust Unified Backend](rust-backend-guide.md). For the
> older lower-level Python worker path (`register_model` +
> `serve_endpoint`) — still the right choice for features the unified
> backend does not yet cover — see
> [Writing Python Workers](backend-guide.md).
>
> **Beta — actively under development.** The unified backend surface
> is beta quality and may change without backwards compatibility
> between releases. See [Feature gaps](#feature-gaps) below for what
> the unified path covers today versus the existing (non-unified)
> backend paths.

This guide walks through building a Python backend for an inference
engine that plugs into Dynamo's distributed runtime via
`dynamo.common.backend`. A "unified backend" is a Python entry point
that implements the shared `LLMEngine` ABC and lets the framework own
runtime lifecycle (signal handling, model registration, graceful
shutdown, cancellation monitoring) — your code just owns inference.

Your backend lives in its own package and **does not need to be part
of the dynamo repository**. It depends on `ai-dynamo` from PyPI (or
the git source) and imports `dynamo.common.backend`. The steps below
assume you're starting a fresh package in your own repo.

The reference example is the **sample engine** at
[`sample_engine.py`](../../components/src/dynamo/common/backend/sample_engine.py)
— a complete, runnable implementation under 120 lines. Read it
alongside this guide.

**Where to look for what:**

- This guide — step-by-step walkthrough for someone starting a new
  backend from scratch.
- [`LLMEngine` ABC docstrings](../../components/src/dynamo/common/backend/engine.py)
  — authoritative method-by-method contract.
- [Package README](../../components/src/dynamo/common/backend/README.md)
  — in-tree reference: `GenerateRequest` / `GenerateChunk` field
  definitions, per-engine cancellation cookbook (vLLM / SGLang /
  TRT-LLM), full `DynamoException` table, file index, and the
  per-engine feature-gap matrix.

## Feature gaps

The unified backend is in beta and does not yet cover the full feature
set of Dynamo's existing (non-unified) backend paths. The summary
below is the common contract — what every engine on the unified path
gets. Per-engine gaps (vLLM, SGLang, TRT-LLM specifics like LoRA,
diffusion, attention DP scheduling) live in the
[package README](../../components/src/dynamo/common/backend/README.md#feature-gaps).

**Supported today**

- Aggregated token-in-token-out inference
- Disaggregated serving (`agg` / `prefill` / `decode`) with bootstrap
  (SGLang) or internal KV transport (vLLM, TRT-LLM)
- Model registration with discovery and endpoint types
- Request cancellation via `abort()` + `context.is_stopped()`
- `DynamoException` error chain wrapping
- Graceful shutdown with signal handling
- Finish reason normalization (handled by the Rust layer)

**Not yet on the unified path**

| Feature | What's missing |
|---------|----------------|
| Metrics & Prometheus | Engine-level metrics, KV utilization gauges, multiprocess registry |
| KV event publishing | Prefix cache events (BlockStored / Removed) to router via ZMQ or NATS |
| Health check payloads | Per-engine custom health probes (BOS token probe, etc.) |
| Logprobs | Selected + top-k logprob extraction and streaming |
| Guided decoding | JSON schema, regex, grammar, choice constraints |
| OpenTelemetry tracing | Trace headers, request perf metrics, OTEL propagation |
| Engine routes | Profiling, memory release/resume, weight updates (disk/tensor/distributed/IPC) |
| Data-parallel routing | DP rank extraction, DP-aware scheduling |
| Text-in-text-out mode | OpenAI-compatible chat/completion with engine-side tokenization |
| Custom Jinja chat templates | `--custom-jinja-template` for model-specific prompt formatting |
| Snapshot/checkpoint | CRIU-based engine state save/restore |
| Multimodal | Images, video, embeddings, separate encode workers |
| LoRA adapters | Dynamic load/unload, ModelDeploymentCard publishing, per-adapter serialization |

If you need one of these features today, keep that workload on the
existing per-engine entry point (`dynamo.<backend>.main`) until the
unified path catches up.

## What you're building

A backend is two things:

1. **An engine class** that subclasses `LLMEngine` — owns the model,
   accepts preprocessed token requests, streams output chunks.
2. **A `main.py` entry point** — a three-line shim that hands the
   engine class to `run()` from `dynamo.common.backend.run`, which
   drives the lifecycle.

The `dynamo.common.backend` package handles everything else: signal
handling, distributed runtime setup, model registration with
discovery, the serving loop, graceful shutdown, cancellation
monitoring, and error chain wrapping. (The lifecycle state machine
actually lives in Rust; `dynamo.common.backend.Worker` is a thin
Python shim over it.)

```text
from_args  →  start()  →  generate() / abort()  →  drain()  →  cleanup()
   |            |               |                     |           |
parse argv,  start engine,  serve requests       pre-cleanup    release
return       return            (concurrent)       drain          resources
engine       metadata
```

## Prerequisites

- Python 3.11 or newer. `dynamo` uses `typing.Required`, which is 3.11+.
- NATS and etcd reachable for end-to-end runs. The dynamo repo's
  `deploy/docker-compose.yml` brings up both in one command if you
  don't already have them running.
- `uv` or `pip` for installing dependencies.
- Familiarity with `async` Python (`asyncio`, async generators) and
  `argparse`.

## Step 1: Create the package

```text
my-backend/
├── pyproject.toml
└── src/
    └── my_backend/
        ├── __init__.py
        ├── engine.py
        └── main.py
```

Minimal `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-backend"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    # ai-dynamo bundles dynamo.common.backend. Pin to the release whose
    # LLMEngine contract you tested against — the surface is still beta
    # and may change between releases.
    "ai-dynamo>=1.2.0",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23"]

[project.scripts]
my-backend = "my_backend.main:main"
```

For a bleeding-edge dependency on the dynamo source tree, install
the runtime wheel from a clone:

```bash
git clone https://github.com/ai-dynamo/dynamo.git
pip install maturin
cd dynamo/lib/bindings/python && maturin build --release --out /tmp/wheels
pip install /tmp/wheels/*.whl       # ai-dynamo-runtime
pip install /path/to/dynamo         # ai-dynamo (components/ tree)
```

Building the wheel needs a Rust toolchain plus `clang`, `cmake`,
`protobuf-compiler`, and `libssl-dev`.

## Step 2: Subclass `LLMEngine`

In `src/my_backend/engine.py`, declare a class that subclasses
`LLMEngine` and owns whatever state your engine needs. Construction
must be cheap and side-effect-free — heavy work goes in `start()`.

```python
# src/my_backend/engine.py
from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncGenerator

from dynamo._core import Context
from dynamo.common.backend import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
    WorkerConfig,
)


class MyBackend(LLMEngine):
    def __init__(self, model_name: str, max_tokens: int = 16):
        self.model_name = model_name
        self.max_tokens = max_tokens
        # Heavy state (engine handles, schedulers, KV allocators) is
        # left None here and initialized in start().
        self._engine = None
```

`GenerateRequest` and `GenerateChunk` are `TypedDict`s describing the
shared shape — see Step 4 for the fields.

## Step 3: Implement `from_args`

`from_args` is a classmethod factory that parses CLI args and returns
`(engine, WorkerConfig)`. The engine is constructed but **not
started**.

```python
@classmethod
async def from_args(
    cls, argv: list[str] | None = None
) -> tuple[MyBackend, WorkerConfig]:
    parser = argparse.ArgumentParser(prog="my-backend")
    parser.add_argument("--model-name", default="my-model")
    parser.add_argument("--max-tokens", type=int, default=16)
    # Runtime / discovery flags — every unified backend needs these.
    parser.add_argument("--namespace", default="dynamo")
    parser.add_argument("--component", default="backend")
    parser.add_argument("--endpoint", default="generate")
    parser.add_argument("--endpoint-types", default="chat,completions")
    parser.add_argument("--discovery-backend", default="etcd")
    parser.add_argument("--request-plane", default="tcp")
    parser.add_argument("--event-plane", default=None)
    args = parser.parse_args(argv)

    engine = cls(model_name=args.model_name, max_tokens=args.max_tokens)
    worker_config = WorkerConfig(
        namespace=args.namespace,
        component=args.component,
        endpoint=args.endpoint,
        model_name=args.model_name,
        served_model_name=args.model_name,
        endpoint_types=args.endpoint_types,
        discovery_backend=args.discovery_backend,
        request_plane=args.request_plane,
        event_plane=args.event_plane,
    )
    return engine, worker_config
```

`from_args` is `async` to match the ABC; you can `await` from it if
your CLI parsing reads config from a file or hits an API. Most
backends don't need to.

For backends that already have a `DynamoRuntimeConfig`-shaped
config object (e.g. ones derived from vLLM's, SGLang's, or
TRT-LLM's existing config), prefer the
`WorkerConfig.from_runtime_config(runtime_cfg, model_name=...)`
helper — it pulls the shared discovery / request-plane / parser
fields off the config in one line.

## Step 4: Implement `LLMEngine` methods

The ABC has three required methods (`start`, `generate`, `cleanup`)
plus two with default no-op implementations (`abort`, `drain`).

### `start()`

Start the engine and return `EngineConfig` metadata. After this
returns, `generate()` MUST be ready for concurrent calls.

```python
async def start(self, worker_id: int) -> EngineConfig:
    # ... load weights, build scheduler, warm up CUDA, etc.
    # Heavy: may take minutes. Emit logger.info checkpoints so
    # operators see progress (Worker logs around start() but not
    # inside it).
    self._engine = await heavy_init(self.model_name)

    return EngineConfig(
        model=self.model_name,
        served_model_name=self.model_name,
        context_length=8192,
        kv_cache_block_size=16,    # None if no block-structured KV
        total_kv_blocks=1024,
        max_num_seqs=64,
        max_num_batched_tokens=8192,
    )
```

`worker_id` is an opaque per-worker identifier — most engines ignore
it. Backends needing a stable cluster-wide key (e.g. TRT-LLM's
`disagg_machine_id` snowflake) should derive from it instead of
hashing host/pid or asking operators for a CLI override.

Every `EngineConfig` field except `model` is optional. `None` means
"don't advertise"; KV-aware routing falls back to round-robin when KV
fields are unset.

### `generate()`

An async generator that yields `GenerateChunk` dicts for a single
request. Called concurrently for multiple in-flight requests.

**Contract** (chunk shape is defined by the `GenerateChunk` TypedDict
— see
[Request / Response Types](../../components/src/dynamo/common/backend/README.md#request--response-types)
in the package README for the field reference):

- Every chunk carries `token_ids` and `index` (use `0` for single
  choice).
- The final chunk additionally carries `finish_reason` and
  `completion_usage`.
- The framework's cancellation monitor calls `engine.abort(context)`
  when the client disconnects or cancels; your loop should also poll
  `context.is_stopped()` between yields and exit cleanly with a
  `finish_reason="cancelled"` chunk.

```python
async def generate(
    self, request: GenerateRequest, context: Context
) -> AsyncGenerator[GenerateChunk, None]:
    prompt_tokens = list(request.get("token_ids", []))
    prompt_len = len(prompt_tokens)

    stop_conditions = request.get("stop_conditions") or {}
    max_new = stop_conditions.get("max_tokens") or self.max_tokens

    def _usage(completion_tokens: int) -> dict[str, int]:
        return {
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_len + completion_tokens,
        }

    for i in range(max_new):
        if context.is_stopped():
            yield {
                "token_ids": [],
                "index": 0,
                "finish_reason": "cancelled",
                "completion_usage": _usage(i),
            }
            return

        token_id = await self._next_token(prompt_tokens)

        chunk: GenerateChunk = {"token_ids": [token_id], "index": 0}
        if i == max_new - 1:
            chunk["finish_reason"] = "length"
            chunk["completion_usage"] = _usage(max_new)
        yield chunk
```

Finish reason normalization (`"abort"` → `"cancelled"`, etc.) is
handled by the Rust layer — emit whatever your engine uses
natively.

### `abort(context)` — optional

Called by the framework only when the client disconnects or the
request is cancelled. NOT called on silent stream drops. Override to
release engine-side resources (KV slots, scheduler entries, remote
schedulers):

```python
async def abort(self, context: Context) -> None:
    request_id = context.id()
    await self._engine.cancel(request_id)
```

For cleanup that must run on every drop path — including silent
drops — use a `try/finally` or a context manager inside `generate`,
not `abort`. The sample engine doesn't override `abort` because it
has no engine-side state to release; the default is a no-op.

### `drain()` — optional

Runs once before shutdown, after the discovery unregister +
grace-period sleep, while NATS/etcd are still alive. Use it for
backend-side draining that must complete before transport teardown
(e.g. in-flight NIXL KV transfers on prefill workers). Default is
no-op.

### `cleanup()`

Two real requirements, both pinned by the Rust-side conformance kit:

- **Null-safe against partial `start()` failure.** If `start()` raises
  partway through, fields you allocate incrementally may still be
  `None`. `cleanup()` must guard each resource (`if self._engine is
  not None: …`) so the post-failure call doesn't crash on
  half-initialized state.
- **Idempotent.** A second call after a successful first must return
  cleanly without re-entering teardown.

The Rust `Worker` drives both: it calls `cleanup()` after `start()`
returns Ok on shutdown, and the conformance kit (`run_conformance`)
additionally calls `cleanup()` on a never-started engine and twice in a
row, failing your tests with `CleanupWithoutStartFailed` /
`SecondCleanupFailed` if either invariant breaks. The guarded
single-shot pattern below covers both:

```python
async def cleanup(self) -> None:
    if self._engine is not None:
        await self._engine.shutdown()
        self._engine = None
```

## Step 5: Write `main.py`

Three lines.

```python
# src/my_backend/main.py
from dynamo.common.backend.run import run
from .engine import MyBackend


def main() -> None:
    run(MyBackend)


if __name__ == "__main__":
    main()
```

`run` installs signal handlers, builds the distributed runtime,
calls `engine.start(worker_id)` with a runtime-allocated identifier,
registers the model with discovery, serves the endpoint, and runs the
graceful-shutdown orchestrator on SIGTERM/SIGINT.

Pair this with the `[project.scripts]` entry from Step 1's
`pyproject.toml` so `my-backend ...` works as a console command.

## Step 6: Errors and logging

**Errors**: the framework wraps non-`DynamoException` errors raised
from `generate()` (or lifecycle methods) as `Unknown`. For typed
error reporting, raise a `DynamoException` subclass directly from
[`dynamo.llm.exceptions`](../../components/src/dynamo/common/backend/README.md#error-handling)
— it propagates unchanged through the Rust bridge:

```python
from dynamo.llm.exceptions import InvalidArgument

async def generate(self, request, context):
    if not request.get("token_ids"):
        raise InvalidArgument("empty prompt")
    ...
```

The package README has the full table of exception types and which
lifecycle phase raises which one. Engine-init failures should raise
`EngineShutdown` from `start()`. Cleanup shouldn't normally raise —
log and swallow if a subsystem fails.

**Logging**: keep levels consistent across unified backends so
operators see the same surface regardless of which engine they're
running:

- `logger.info` — lifecycle milestones (engine init complete,
  serving started, engine shutdown).
- `logger.debug` — per-request events (request abort, cancellation).
- `logger.warning` — recoverable problems (empty outputs, unexpected
  finish reasons).
- `logger.error` — unrecoverable failures only.

The framework also configures `dynamo.runtime.logging` for you; you
just call `logger = logging.getLogger(__name__)` at the top of your
module and use it.

## Step 7: Test your engine

Install the dev extras (`pytest`, `pytest-asyncio`) declared in Step 1:

```bash
pip install -e ".[dev]"
```

The sample engine has a unit-test
[suite](../../components/src/dynamo/common/backend/tests/test_engine.py)
that you can copy as a starting point. The shape of a useful test:

```python
import pytest

from my_backend import MyBackend


class _StubContext:
    def __init__(self, stopped: bool = False) -> None:
        self._stopped = stopped

    def is_stopped(self) -> bool:
        return self._stopped

    def stop(self) -> None:
        self._stopped = True


@pytest.mark.asyncio
async def test_generate_emits_terminal_chunk():
    engine = MyBackend(model_name="m", max_tokens=3)
    await engine.start(worker_id=0)
    try:
        chunks = [
            chunk
            async for chunk in engine.generate(
                {"token_ids": [1, 2, 3]}, _StubContext()
            )
        ]
        assert chunks[-1]["finish_reason"] in ("stop", "length")
        assert chunks[-1]["completion_usage"]["completion_tokens"] == 3
    finally:
        await engine.cleanup()


@pytest.mark.asyncio
async def test_generate_observes_cancellation():
    engine = MyBackend(model_name="m", max_tokens=1000)
    await engine.start(worker_id=0)
    try:
        ctx = _StubContext()
        collected = []
        async for chunk in engine.generate({"token_ids": [1]}, ctx):
            collected.append(chunk)
            if len(collected) >= 2:
                ctx.stop()
        assert collected[-1]["finish_reason"] == "cancelled"
    finally:
        await engine.cleanup()
```

Cover the happy path, cancellation, and any backend-specific edge
cases (stop tokens, max-tokens cap, empty prompt). Three to five
focused tests is plenty — the framework already pins the lifecycle
state machine and cancellation contract with Rust-side tests in
`lib/backend-common`.

## Step 8: Run it locally

Three moving parts need to come up: NATS + etcd (discovery and the
event/request planes), the Dynamo frontend (HTTP → backend
discovery), and your backend.

```bash
pip install -e .

# Ensure NATS + etcd are reachable (NATS_SERVER, ETCD_ENDPOINTS).
# --model-name must be a valid HuggingFace repo (or local path); the
# framework fetches the tokenizer + chat template from it on startup.
# Pick a small public repo for smoke tests.
my-backend --model-name Qwen/Qwen3-0.6B --namespace dynamo

# In another shell, start the Dynamo frontend:
python -m dynamo.frontend --http-port 8000
```

Then send a request:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
          "model": "Qwen/Qwen3-0.6B",
          "messages": [{"role": "user", "content": "hello"}],
          "max_tokens": 32
        }'
```

A successful response has non-empty `choices[0].message.content`
and a `finish_reason` of `stop` or `length`.
`jq -e '.choices[0].finish_reason'` is a good one-liner for a CI
smoke test.

If your backend looks silent, set `DYN_LOG=info` (or
`DYN_LOG=debug,dynamo=debug` for finer scoping) before launching —
the framework configures `tracing` from `DYN_LOG`.

## Reference: the sample engine

[`sample_engine.py`](../../components/src/dynamo/common/backend/sample_engine.py)
is the canonical minimal reference. Run it as-is:

```bash
python -m dynamo.common.backend.sample_main --model-name test-model
```

It generates rotating token IDs with no ML dependencies, so it's a
useful stand-in for AIPerf / end-to-end pipeline smoke tests. Lift
these patterns:

- `from_args` parses CLI args and returns `(engine, WorkerConfig)`
  with no awaits.
- `start()` returns an `EngineConfig` whose KV fields are
  illustrative but not load-bearing (no real KV cache).
- `generate()` polls `context.is_stopped()` between yields and
  emits a `cancelled` terminal on observation.
- `cleanup()` is a no-op because the engine holds no resources.

## Checklist

Before shipping:

- [ ] `LLMEngine` subclassed; `from_args` returns
      `(engine, WorkerConfig)`.
- [ ] `start()` returns `EngineConfig` with at least a non-empty
      `model`.
- [ ] `generate()` polls `context.is_stopped()` between yields and
      emits a `"cancelled"` terminal on observation.
- [ ] Final chunk has `finish_reason` and `completion_usage`.
- [ ] Typed `DynamoException` subclasses used for error reporting
      where the category matters.
- [ ] `cleanup()` releases all engine resources.
- [ ] Logging levels match the standards in Step 6.

## See also

- [`LLMEngine` ABC](../../components/src/dynamo/common/backend/engine.py)
  — authoritative contract.
- [Package README](../../components/src/dynamo/common/backend/README.md)
  — feature gaps, error model, request/response contract.
- [Sample engine](../../components/src/dynamo/common/backend/sample_engine.py)
  — example user guide.
- [Writing a Rust Unified Backend](rust-backend-guide.md) — the Rust
  counterpart, same contract, lower-level.
