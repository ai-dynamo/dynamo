---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Writing Unified Backends
subtitle: Choose Python or Rust for Dynamo's shared backend contract
---

Dynamo's unified backend path lets custom engines implement the same lifecycle
contract used by the built-in backends. The engine owns inference; Dynamo owns
runtime registration, request serving, cancellation monitoring, signal handling,
drain, and graceful shutdown.

Use this path for new token-in-token-out engines unless you need a feature that
is still outside the unified contract.

## Choose an Implementation Language

| Path | Use when |
|---|---|
| Python unified backend | Your engine or serving library is Python-first, or you want the quickest path to integrate a custom engine. |
| Rust unified backend | You want a native Rust binary, tighter control of runtime dependencies, or no Python worker runtime. |
| Python workers (lower-level) | You need custom request handling or features not yet covered by the unified backend contract. Use [Python Workers](backend-guide.md). |

Both unified implementations follow the same shape:

```text
parse config -> start engine -> stream generated chunks -> abort/drain -> cleanup
```

The framework handles model registration, endpoint serving, cancellation
plumbing, and shutdown behavior around that engine contract.

## What the Unified Contract Covers

Supported today:

- aggregated token-in-token-out inference
- disaggregated serving modes for supported engines
- model registration through Dynamo discovery
- request cancellation
- structured backend errors
- graceful shutdown and drain hooks

Still use the lower-level Python worker path when you need features such as
multimodal requests, LoRA adapter management, logprobs, guided decoding,
engine-specific routes, custom request handling, or features that need direct
control of the request payload.

After you implement the backend, package it into a runtime image with
[Runtime Containers](custom-containers.md). For Kubernetes deployment, place the
custom backend in a `DynamoGraphDeployment` and follow the
[Deployment Overview](../kubernetes/model-deployment-guide.md).

<Tabs>
<Tab title="Python" language="python">

## Python Implementation
> **New — Dynamo's unified backend.** This guide covers the new
> **unified backend** infrastructure in
> [`dynamo.common.backend`](https://github.com/ai-dynamo/dynamo/tree/main/components/src/dynamo/common/backend):
> a shared `LLMEngine` ABC that vLLM, SGLang, TRT-LLM, and a sample
> engine already implement, and that any custom Python engine can plug
> into the same way. For the Rust version of the same contract, use the
> Rust tab on this page. For the older lower-level Python worker path (`register_model` +
> `serve_endpoint`) — still the right choice for features the unified
> backend does not yet cover — see
> [Writing Python Workers](backend-guide.md).
>
> **Beta — actively under development.** The unified backend surface
> is beta quality and may change without backwards compatibility
> between releases. See [Feature gaps](#python-feature-gaps) below for what
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

### Python feature gaps

The unified backend is in beta. The summary below is the common
contract — what every engine on the unified path gets — plus the
gaps that apply to all three engines. Per-engine specifics (vLLM
sleep/wake, SGLang diffusion, TRT-LLM custom logits processors,
etc.) live in the
[package README](../../components/src/dynamo/common/backend/README.md#feature-gaps).

**Supported today**

Lifecycle and runtime:
- Aggregated token-in-token-out inference
- Disaggregated serving (`agg` / `prefill` / `decode`) — KV transfer
  uses NIXL across all three engines; SGLang exchanges a Dynamo-level
  bootstrap address (host/port/room), vLLM and TRT-LLM use an
  engine-internal handshake
- Model registration with discovery and endpoint types
- Request cancellation via `abort()` + `context.is_stopped()`
- Graceful shutdown with signal handling
- `drain()` hook for pre-cleanup work (e.g. in-flight NIXL transfers)
- `DynamoException` error chain wrapping
- Finish reason normalization (handled by the Rust layer)
- Engine control plumbing, with per-backend profiling, quiesce/resume, and supported weight-update controls

Observability:
- Health-check canary via `health_check_payload()` (plus
  `DYN_HEALTH_CHECK_PAYLOAD` / `--health-check-payload` overrides)
- Vendor-prefixed Prometheus bridge (`vllm:` / `sglang:` /
  `trtllm_` / `lmcache:`) via `register_prometheus()`
- Framework-owned lifecycle gauges (`cleanup_time_seconds`,
  `drain_time_seconds`, `model_load_time_seconds`) — always on
- Per-rank `dynamo_component_*` gauges + router `kv_used_blocks`
  signal via `component_metrics_dp_ranks()` +
  `attach_snapshot_publisher()` + `ComponentSnapshot` push
- KV event publishing via `kv_event_sources()` returning
  `ZmqSource` or `PushSource`
- KV-aware routing (DP-rank-aware) via `dp_rank.forced_dp_rank` /
  `validate_global_dp_rank` + `EngineConfig.data_parallel_{size,
  start_rank}`
- OpenTelemetry tracing facade — `telemetry.current_span` /
  `start_span` plus W3C trace header propagation through
  `telemetry.engine_trace_kwargs(context)`

Request handling:
- Guided decoding — wired per-engine on the request side with
  engine-specific coverage. vLLM (`StructuredOutputsParams`) and
  TRT-LLM (`GuidedDecodingParams`) cover JSON schema / regex / grammar
  / choice; SGLang (`_get_guided_decoding_params`) covers JSON schema
  only — regex / grammar / choice are silently dropped today (see the
  SGLang-specific gaps in the package README)
- Structural tag generation via `WorkerConfig.structural_tag_{mode,
  scope, schema}` and `serialize_structural_tag`
- Custom Jinja chat templates via
  `WorkerConfig.custom_jinja_template` (frontend applies; the
  backend advertises through model registration)
- Tool / reasoning parser configuration (`tool_call_parser`,
  `reasoning_parser`, `exclude_tools_when_tool_choice_none`)

**Not yet on the unified path (common to all engines)**

| Feature | What's missing |
|---------|----------------|
| Logprob response wire | Legacy handlers extract logprobs onto response chunks (vLLM `_extract_logprobs`, SGLang `_extract_logprobs` in `decode_handler`, TRT-LLM `_extract_logprobs` in `handler_base`); the unified `generate()` loops do not populate `log_probs` / `top_logprobs` / `cum_log_probs` on `GenerateChunk`. vLLM's `build_sampling_params` still passes `output_options.logprobs` to the engine on the unified path, so the engine computes them, but the values are dropped before they reach the chunk. SGLang and TRT-LLM unified `generate()` do not read `output_options.logprobs` at all. |
| Text-in-text-out mode | Unified hardcodes `ModelInput.Tokens`; no engine-side tokenization or chat templating path |
| Multimodal | Images / video / embeddings, NIXL embedding transfer, separate encode workers, `ENCODE` disaggregation role |
| Diffusion | Image (FLUX), video (Wan2.1), LLM diffusion (DLLM) workers; no diffusion engine, MediaOutput, or media scheduling on the unified path |
| LoRA adapters | Dynamic load / unload / list, ModelDeploymentCard publishing, per-adapter serialization locks, per-request adapter threading on prefill |
| Snapshot / checkpoint | CRIU-based engine state save/restore + identity reload |

If you need one of these features today, keep that workload on the
existing per-engine entry point (`dynamo.<backend>.main`) until the
unified path catches up.

### Python: What you are building

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

### Python prerequisites

- Python 3.11 or newer. `dynamo` uses `typing.Required`, which is 3.11+.
- NATS and etcd reachable for end-to-end runs. The dynamo repo's
  `deploy/docker-compose.yml` brings up both in one command if you
  don't already have them running.
- `uv` or `pip` for installing dependencies.
- Familiarity with `async` Python (`asyncio`, async generators) and
  `argparse`.

### Python Step 1: Create the package

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
pip install 'maturin[patchelf]'
cd dynamo/lib/bindings/python && maturin build --release --out /tmp/wheels
pip install /tmp/wheels/*.whl       # ai-dynamo-runtime
pip install /path/to/dynamo         # ai-dynamo (components/ tree)
```

[Maturin](https://github.com/PyO3/maturin) is the Rust-Python bindings build tool. The `patchelf` extra lets maturin patch native extension library paths during the build.

Building the wheel needs a Rust toolchain plus `clang`, `cmake`,
`protobuf-compiler`, and `libssl-dev`.

### Python Step 2: Subclass `LLMEngine`

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

### Python Step 3: Implement `from_args`

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

### Python Step 4: Implement `LLMEngine` methods

The ABC has three required methods (`start`, `generate`, `cleanup`)
plus two with default no-op implementations (`abort`, `drain`).

#### Python: `start()`

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

#### Python: `generate()`

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

#### Python: `abort(context)` — optional

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

#### Python: `drain()` — optional

Runs once before shutdown, after the discovery unregister +
grace-period sleep, while NATS/etcd are still alive. Use it for
backend-side draining that must complete before transport teardown
(e.g. in-flight NIXL KV transfers on prefill workers). Default is
no-op.

#### Python: `cleanup()`

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

### Python Step 5: Write `main.py`

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

### Python Step 6: Errors and logging

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

### Python Step 7: Test your engine

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

### Python Step 8: Run it locally

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

### Python reference: sample engine

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

### Python checklist

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

### Python see also

- [`LLMEngine` ABC](../../components/src/dynamo/common/backend/engine.py)
  — authoritative contract.
- [Package README](../../components/src/dynamo/common/backend/README.md)
  — feature gaps, error model, request/response contract.
- [Sample engine](../../components/src/dynamo/common/backend/sample_engine.py)
  — example user guide.
- Rust tab on this page — the Rust counterpart, same contract,
  lower-level.

</Tab>
<Tab title="Rust" language="rust">

## Rust Implementation
> **New — Dynamo's unified backend.** This guide covers the new
> **unified backend** infrastructure in
> [`dynamo-backend-common`](https://github.com/ai-dynamo/dynamo/tree/main/lib/backend-common): a shared
> `LLMEngine` contract that vLLM, SGLang, TRT-LLM, and the mocker
> already implement, and that any custom engine can plug into the
> same way.
>
> **Beta — actively under development.** The Rust native backend
> surface is beta quality and may change without backwards
> compatibility between releases. See [Feature gaps](#rust-feature-gaps)
> below for what the unified path covers today versus the existing
> (non-unified) backend paths.

This guide walks through building a Rust unified backend for an
inference engine that plugs into Dynamo's distributed runtime. A
unified backend is a standalone Rust binary that owns its engine and
serves requests via the shared
[`LLMEngine`](https://github.com/ai-dynamo/dynamo/blob/main/lib/backend-common/src/engine.rs) contract in
[`dynamo-backend-common`](https://github.com/ai-dynamo/dynamo/tree/main/lib/backend-common) — no Python
worker runtime required. For the Python version of the same contract
use the Python tab on this page.

Your backend lives in its own crate and **does not need to be part of
the dynamo repository**. It pulls `dynamo-backend-common` in as a normal
git or path dependency. The steps below assume you're starting a fresh
crate in your own repo; an optional note in Step 1 covers the in-tree
variant for contributors landing a backend inside `ai-dynamo/dynamo`.

For a Python engine, use the Python tab on this page — same contract,
lighter setup. The non-unified fallback for feature gaps
(multimodal, LoRA, logprobs, etc.) is Python-only; see
[Writing Python Workers](backend-guide.md) if you need one of those
today.

The reference example is the **mocker backend** at
[`lib/backend-common/examples/mocker`](https://github.com/ai-dynamo/dynamo/tree/main/lib/backend-common/examples/mocker)
— a small, complete, pure-Rust implementation. Read it alongside this
guide.

**Where to look for what:**

- This guide — step-by-step walkthrough for someone starting a new
  backend from scratch.
- [`LLMEngine` trait doc comments](https://github.com/ai-dynamo/dynamo/blob/main/lib/backend-common/src/engine.rs)
  — authoritative method-by-method contract.
- [Crate README](https://github.com/ai-dynamo/dynamo/blob/main/lib/backend-common/README.md) — in-tree
  reference: architecture, file index, disaggregation contract,
  error taxonomy, conformance kit.
- [`backend-common` design notes](https://github.com/ai-dynamo/dynamo/blob/main/lib/backend-common/CLAUDE.md)
  — rationale and invariants.

### Rust feature gaps

The unified backend is in beta. The summary below is the common
contract — what every engine on the unified path gets, whether
written in Rust directly or plugged in from Python via the PyO3
`Worker` shim. Per-engine specifics (vLLM sleep/wake, SGLang
diffusion, TRT-LLM custom logits processors, etc.) live in the
[Python package README](../../components/src/dynamo/common/backend/README.md#feature-gaps).

**Supported today**

Lifecycle and runtime:
- Aggregated token-in-token-out inference
- Disaggregated serving (`Aggregated` / `Prefill` / `Decode`) — KV
  transfer uses NIXL across all production engines; SGLang exchanges
  a Dynamo-level bootstrap address, vLLM and TRT-LLM use an
  engine-internal handshake. The Rust
  [mocker example](https://github.com/ai-dynamo/dynamo/tree/main/lib/backend-common/examples/mocker)
  exercises the same wire format CPU-only
- Model registration with discovery and endpoint types
- Request cancellation via in-stream `ctx.is_stopped()` polling plus
  the framework's out-of-band `abort()` monitor
- `drain()` hook for pre-cleanup work
- Typed `DynamoError` with `ErrorType::Backend(BackendError::X)`
- Graceful shutdown with signal handling and 3-phase
  distributed-runtime teardown
- Debug-build stream validator and the `testing::run_conformance` kit
- Engine control plumbing, with per-backend profiling, pause/resume, and supported weight-update controls

Observability:
- Health-check canary via `LLMEngine::health_check_payload()` plus
  the operator override (`DYN_HEALTH_CHECK_PAYLOAD` /
  `--health-check-payload`)
- Vendor-registry bridge into the runtime's `/metrics` output via
  `LLMEngine::setup_metrics()`, plus framework-owned lifecycle
  gauges (`dynamo_component_{cleanup_time_seconds,
  drain_time_seconds, model_load_time_seconds}`) and per-rank
  `dynamo_component_*` gauges driven by `SnapshotPublisher`
- KV event publishing via `kv_event_sources()` returning
  `KvEventSource::Zmq` or `KvEventSource::Push`
- KV-aware routing (DP-rank-aware) — engines advertise their slice
  via `EngineConfig::data_parallel_size` /
  `data_parallel_start_rank`; read the router-forced rank off
  `request.routing.dp_rank` in `generate()`
- OpenTelemetry tracing — the framework auto-opens an
  `engine.generate` span around every `generate()` call with
  attributes for `model` / `input_tokens` / `disagg_role` / `ttft_ms`
  / `output_tokens` / `finish_reason` / ITL percentiles. Static-name
  spans opened with `tracing::info_span!` inside `generate()` nest
  under it automatically; for dynamic span names use
  `dynamo_backend_common::telemetry::start_span(name)`. For outbound
  calls that need to carry trace context (custom HTTP/TCP
  transports), use
  `dynamo_runtime::logging::inject_trace_headers_into_map`. NATS
  egress is auto-injected — engines do nothing.

Request handling:
- Guided decoding — request shape carries
  `SamplingOptions::guided_decoding` (`GuidedDecodingOptions`);
  engine-side coverage on the existing Python-bridged engines is:
  vLLM and TRT-LLM forward JSON schema / regex / grammar / choice;
  SGLang forwards JSON schema only (regex / grammar / choice are
  silently dropped today). A new Rust engine should forward whichever
  variants its backend supports
- Structural tag generation — `WorkerConfig::structural_tag_{mode,
  scope, schema}` (typed enums)
- Custom Jinja chat templates — `WorkerConfig::custom_jinja_template`
  flows to `LocalModelBuilder::custom_template_path` and the
  frontend applies the template at preprocessing time
- Tool / reasoning parser configuration on `WorkerConfig`
  (`tool_call_parser`, `reasoning_parser`,
  `exclude_tools_when_tool_choice_none`)

**Not yet on the unified path (common to all engines)**

| Feature | What's missing |
|---------|----------------|
| `cum_log_probs` response wire | Completion-side `log_probs` / `top_logprobs` are populated on the unified path for vLLM, SGLang, and TRT-LLM (shared helpers in `components/src/dynamo/common/backend/logprobs.py`). Prompt-side logprobs ride on the final chunk's `LLMEngineOutput.engine_data["prompt_logprobs"]` (consumed by `prompt_logprobs_from_engine_data` in the response builders). `cum_log_probs` is still not emitted. |
| Text-in-text-out mode | `ModelInput::Text` is rejected at startup — `Tokens` only |
| Multimodal | Images / video / embeddings, NIXL embedding transfer, separate encode workers; `ENCODE` disaggregation role |
| Diffusion | Image (FLUX), video (Wan2.1), LLM diffusion (DLLM) workers; no diffusion engine, MediaOutput, or media scheduling on the unified path |
| LoRA adapters | Dynamic load / unload / list, ModelDeploymentCard publishing, per-adapter serialization |
| Snapshot / checkpoint | CRIU-based engine state save/restore + identity reload |

If you need one of these features today, keep that workload on the
existing per-engine entry point until the unified path catches up.

### Rust: What you are building

A backend is two things:

1. **An engine type** that implements the `LLMEngine` trait — owns the
   model, accepts preprocessed token requests, streams output tokens.
2. **A `main.rs` entry point** — a three-line shim that hands the
   engine to `dynamo_backend_common::run`, which drives the lifecycle.

The `dynamo-backend-common` crate handles everything else: signal
handling, model registration with discovery, the serving loop, graceful
shutdown, metrics, cancellation plumbing, and the debug-mode contract
validator.

Engines work directly with `PreprocessedRequest` and `LLMEngineOutput`
— the same types used by Dynamo's preprocessing, routing, and frontend.
No Python-shaped translation layer.

```text
construct  →  start()  →  generate() / abort()  →  drain() →  cleanup()
   |            |               |                      |          |
parse args   start engine,  serve requests       pre-cleanup   release
return       return            (concurrent)       drain        resources
engine       metadata
```

### Rust prerequisites

- Rust 1.85 or newer (the dynamo workspace is edition 2024). The
  toolchain pin in Step 1 locks this in for you; older toolchains will
  fail with `feature edition2024 is required` deep inside the build.
- NATS and etcd reachable for end-to-end runs. The dynamo repo's
  `deploy/docker-compose.yml` brings up both in one command if you
  don't already have them running.
- Familiarity with `async` Rust, `tokio`, and `clap`. The trait uses
  `async_trait`, and the framework expects a `tokio` runtime.

### Rust Step 1: Create the crate

Your backend is a standalone Rust binary crate. It can live in its own
repository — the dynamo repo is **not** required to be your parent
workspace. Pick whatever layout you prefer:

```text
my-backend/
├── Cargo.toml
└── src/
    ├── main.rs
    └── engine.rs        # (or my_engine.rs — whatever you call it)
```

`cargo new --bin my-backend` is the fastest starting point; add
`src/engine.rs` yourself afterwards.

#### Rust: Getting the `dynamo-backend-common` crate

`dynamo-backend-common` lives in the
[ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) repository and
is not on crates.io. Depend on it via git:

```toml
[package]
name = "my-backend"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "my-backend"
path = "src/main.rs"

[dependencies]
# Replace <SHA> with the dynamo commit you want to build against.
# `branch = "main"` works too but moves under you on every rebuild.
dynamo-backend-common = { git = "https://github.com/ai-dynamo/dynamo.git", rev = "<SHA>" }

anyhow = "1"
async-stream = "0.3"
async-trait = "0.1"
clap = { version = "4", features = ["derive", "env"] }
futures = "0.3"
# Must match the version pinned by dynamo-runtime — it relies on
# tokio_unstable runtime metrics that change shape across releases.
tokio = { version = "=1.48.0", features = ["full"] }
tracing = "0.1"

[dev-dependencies]
dynamo-backend-common = { git = "https://github.com/ai-dynamo/dynamo.git", rev = "<SHA>", features = ["testing"] }
```

The `testing` feature pulls in the conformance kit used in Step 7.

Pick a SHA with:

```bash
git ls-remote https://github.com/ai-dynamo/dynamo.git main
```

> **No release tags yet.** `dynamo-backend-common` landed after the last
> tagged release (`v1.1.1`), so `tag = "v1.1.1"` won't resolve the
> crate. Track `main` or pin to a specific SHA until a release tag ships
> that includes the crate.

#### Rust: Two build-time requirements you cannot skip

These are easy to miss and surface as confusing compile errors deep
inside `dynamo-runtime`:

1. **`tokio_unstable` cfg flag.** `dynamo-runtime` uses tokio's
   unstable runtime-metrics API. Create `.cargo/config.toml` in your
   crate root:

   ```toml
   [build]
   rustflags = ["--cfg", "tokio_unstable"]
   ```

   Without it, you'll see errors like `method blocking_queue_depth not
   found on RuntimeMetrics` while compiling `dynamo-runtime`.

2. **Rust toolchain pin.** Match dynamo's toolchain so workspace-edition
   crates compile. Create `rust-toolchain.toml`:

   ```toml
   [toolchain]
   channel = "1.93.1"
   ```

   Older toolchains fail with `feature edition2024 is required`.

> **Tip — local development**: while iterating against an unreleased
> change in `dynamo-backend-common`, point the dep at a local clone:
> `dynamo-backend-common = { path = "/path/to/dynamo/lib/backend-common" }`.
> Switch back to the git dep before publishing your crate.

If you'd rather develop *inside* the dynamo workspace as a new
sub-crate, drop your crate under `dynamo/lib/` and use
`dynamo-backend-common = { workspace = true }` instead. The trait
contract is identical, and the `.cargo/config.toml` plus toolchain
pin in the dynamo repo cover the two requirements above for you.

### Rust Step 2: Define your engine struct

In `src/engine.rs` (or whatever you named it), declare a struct that
owns whatever state your engine needs. Anything you allocate inside
`start()` later must live behind interior mutability so the trait's
`&self` methods can reach it.

```rust
use async_trait::async_trait;
use dynamo_backend_common::engine::GenerateContext;
use dynamo_backend_common::{
    BackendError, CommonArgs, DynamoError, EngineConfig, ErrorType, FinishReason, LLMEngine,
    LLMEngineOutput, LLMEngineOutputExt, LlmRegistration, PreprocessedRequest, WorkerConfig,
    chunk, usage,
};
use futures::stream::BoxStream;
use tokio::sync::RwLock;

pub struct MyBackend {
    model: String,
    inner: RwLock<Option<Inner>>,   // allocated in start()
}

// Replace this with whatever your engine owns — handle, scheduler,
// client, channel sender, etc. Fields go here. Truly stateless
// engines can skip `Inner` and `inner` entirely.
struct Inner {}
```

`async-trait` lets the trait use `async fn` (still required for
object-safety with `Arc<dyn LLMEngine>`); `async-stream`'s `stream!`
macro lets the `generate` body yield items from inside an `async` block.

The mocker example uses `OnceCell` for `Inner`; `RwLock<Option<_>>`
also works — pick whichever fits your shutdown semantics.

### Rust Step 3: Wire up CLI arguments

Every backend's CLI shares a common base (`--namespace`, `--component`,
`--endpoint`, etc.) provided by `CommonArgs`. Flatten that into your
engine's `Args` struct and add your engine-specific flags.

```rust
#[derive(clap::Parser, Debug)]
#[command(
    name = env!("CARGO_BIN_NAME"),
    about = "My Dynamo Rust backend."
)]
struct Args {
    #[command(flatten)]
    common: CommonArgs,

    /// HF repo or local model directory.
    #[arg(value_name = "MODEL")]
    model: String,

    /// Public-facing model name advertised to clients.
    #[arg(long)]
    served_model_name: Option<String>,

    // ... engine-specific flags here.
}
```

Define an inherent `from_args` constructor that parses the args and
returns both the engine and a `WorkerConfig`. **`from_args` is not on
the trait** — it stays inherent so the trait can remain object-safe
(`Arc<dyn LLMEngine>` must work).

The snippet below calls a tiny `invalid_arg` helper that builds a
typed `BackendError::InvalidArgument`. Its full definition lives in
Step 6 — for now, mentally substitute "any function that returns a
`DynamoError` with category `InvalidArgument`."

```rust
impl MyBackend {
    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = match argv {
            Some(a) => <Args as clap::Parser>::try_parse_from(a),
            None => <Args as clap::Parser>::try_parse(),
        }
        .map_err(|e| invalid_arg(e.to_string()))?;

        let engine = Self {
            model: args.model.clone(),
            inner: RwLock::new(None),
        };

        let config = WorkerConfig {
            namespace: args.common.namespace,
            component: args.common.component,
            endpoint: args.common.endpoint,
            endpoint_types: args.common.endpoint_types,
            custom_jinja_template: args.common.custom_jinja_template,
            // Pass `--disaggregation-mode` from `CommonArgs` through to the
            // Worker — without this line the worker silently registers as
            // Aggregated regardless of what the operator passed.
            disaggregation_mode: args.common.disaggregation_mode,
            model_name: args.model,
            served_model_name: args.served_model_name,
            ..Default::default()
        };

        Ok((engine, config))
    }
}
```

`WorkerConfig::default()` sets `model_input` to `ModelInput::Tokens`,
which is the only mode `Worker` currently supports — the framework
validates this at startup. Engines needing raw text or tensor inputs
aren't supported yet.

If your engine branches on the disaggregation role inside `generate`
(prefill vs decode), keep the same `DisaggregationMode` on your engine
struct so the runtime registration (`WorkerConfig`) and the per-request
dispatch stay in lockstep.

### Rust Step 4: Implement the `LLMEngine` trait

The trait has three required methods (`start`, `generate`, `cleanup`)
plus two with default implementations you can override (`abort`, `drain`).

#### Rust: `start()`

Start the engine and return `EngineConfig` metadata. After this
returns, the engine MUST be ready for concurrent `generate()` calls.
Use interior mutability for anything you initialize here.

```rust
async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
    tracing::info!(model = %self.model, "starting my backend");

    // ... start your engine (may take minutes for real backends — emit
    // tracing::info! checkpoints so operators see progress).
    let inner = init_engine(&self.model).await?;
    *self.inner.write().await = Some(inner);

    Ok(EngineConfig {
        model: self.model.clone(),
        served_model_name: Some(self.model.clone()),
        // Token-pipeline metadata goes in the `llm` sub-record (RawEngines
        // leave it None).
        llm: Some(LlmRegistration {
            context_length: Some(8192),
            kv_cache_block_size: Some(64),     // None if no block-structured KV
            total_kv_blocks: Some(16384),
            max_num_seqs: Some(256),
            max_num_batched_tokens: Some(8192),
            ..Default::default()
        }),
        ..Default::default()
    })
}
```

`worker_id` is an opaque per-worker identifier — most engines ignore
it with `_worker_id`. Backends needing a stable cluster-wide key
(e.g. TRT-LLM's `disagg_machine_id` snowflake) should derive from it.

Every `EngineConfig` field except `model` is optional. `None` means
"don't advertise"; KV-aware routing falls back to round-robin when KV
fields are unset. Engines wrapping an external runtime can read these
values from the live engine after it comes up, instead of hard-coding
them. The `..Default::default()` is load-bearing: `EngineConfig`
sometimes grows new fields (e.g. `bootstrap_host`/`bootstrap_port`
for SGLang disagg) and the default keeps existing engines compiling.

#### Rust: `generate()`

Yield a stream of `Result<LLMEngineOutput, DynamoError>` items for a
single request. Called concurrently for multiple in-flight requests.

`ctx: GenerateContext` is a thin wrapper that `Deref`s to
`dyn AsyncEngineContext`, so the cancellation methods (`stopped()`,
`is_stopped()`, `id()`) you'd expect are still there. The wrapper
additionally exposes `notify_first_token()` for decode-mode requests
— most engines can ignore it; the framework auto-fires on the first
non-empty chunk.

**Contract** (the [debug-mode validator](../../lib/backend-common/src/validate.rs)
panics on violations):

- Exactly one **terminal item** must be the last item yielded. A
  terminal is either an `Ok(chunk)` with `finish_reason` set, or an
  `Err(DynamoError)`. No items may be yielded after a terminal.
- Non-terminal chunks use `chunk::token(id)` and leave `finish_reason`
  unset.
- The returned stream is `'static`: clone or move any state from
  `&self` or `request` into the stream body before constructing it.

Terminal chunks come from one of four `LLMEngineOutput` constructors,
optionally chained with the `LLMEngineOutputExt` setters
(`.with_tokens(...)`, `.with_usage(...)`):

- `LLMEngineOutput::stop()` — natural completion (e.g. you reached your
  echo limit, the engine hit a stop string).
- `LLMEngineOutput::length()` — `max_tokens` cap reached.
- `LLMEngineOutput::cancelled()` — you observed `ctx.stopped()`.
- `LLMEngineOutput::error(msg)` — message-only error terminal (loses
  the typed `BackendError` variant — yield `Err(DynamoError)` instead
  when the category matters).

Non-terminal chunks use `chunk::token(id)` (single-token convenience).

A streaming-`generate` template:

```rust
async fn generate(
    &self,
    request: PreprocessedRequest,
    ctx: GenerateContext,
) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
    // Destructure once and move fields into the stream — no extra clones
    // (the stream is 'static and outlives `request`).
    let PreprocessedRequest { token_ids, stop_conditions, .. } = request;
    let prompt_tokens = token_ids.len() as u32;
    let mut output_rx = self.submit_to_engine(token_ids, stop_conditions).await?;

    Ok(Box::pin(async_stream::stream! {
        let mut completion_tokens = 0_u32;
        loop {
            tokio::select! {
                biased;

                // Cancellation: emit FinishReason::Cancelled terminal.
                _ = ctx.stopped() => {
                    yield Ok(LLMEngineOutput::cancelled()
                        .with_usage(usage(prompt_tokens, completion_tokens)));
                    break;
                }

                // Next item from the engine.
                next = output_rx.recv() => {
                    let Some(engine_output) = next else {
                        yield Ok(LLMEngineOutput::error(
                            "engine stream ended without a terminal".into()
                        ));
                        break;
                    };

                    // Translate your engine's per-step output into LLMEngineOutput.
                    // For a terminal step set `finish_reason`; otherwise leave it None.
                    let mut out = LLMEngineOutput {
                        token_ids: engine_output.tokens,
                        finish_reason: engine_output.terminal_reason,
                        ..Default::default()
                    };
                    completion_tokens += out.token_ids.len() as u32;

                    if out.finish_reason.is_some() {
                        out = out.with_usage(usage(prompt_tokens, completion_tokens));
                        yield Ok(out);
                        break;
                    }
                    yield Ok(out);
                }
            }
        }
    }))
}
```

`biased` is load-bearing for the channel-receiving pattern above:

1. When cancellation and a pending token are both ready, yield the
   cancellation, not one more token.
2. During cleanup the stream sees both `ctx.stopped()` and
   `rx.recv() -> None` simultaneously; `biased` picks the clean
   cancellation path instead of erroring on a closed channel. The
   mocker's [stream body](../../lib/backend-common/examples/mocker/src/engine.rs)
   spells this out.

If your engine doesn't have a receiver — e.g. you're computing tokens
inline like a deterministic echo backend — the body collapses to a
plain loop that polls cancellation between yields:

```rust
Ok(Box::pin(async_stream::stream! {
    for (i, token_id) in tokens_to_emit.iter().enumerate() {
        tokio::select! {
            biased;
            _ = ctx.stopped() => {
                yield Ok(LLMEngineOutput::cancelled()
                    .with_usage(usage(prompt_tokens, i as u32)));
                return;
            }
            _ = tokio::time::sleep(delay), if !delay.is_zero() => {}
        }
        if i == tokens_to_emit.len() - 1 {
            yield Ok(LLMEngineOutput::stop()
                .with_tokens(vec![*token_id])
                .with_usage(usage(prompt_tokens, (i + 1) as u32)));
        } else {
            yield Ok(chunk::token(*token_id));
        }
    }
}))
```

No channel-close race to worry about; `biased` is still cheap and
recommended for consistency.

**Cancellation rules**:

- The stream **must** poll `ctx.is_stopped()` (or `await ctx.stopped()`)
  between yields.
- On cancellation, emit a terminal with `FinishReason::Cancelled` — not
  `Length` or `Stop`. The conformance kit treats any other terminal
  after cancellation as ignoring the signal.

**Typed errors vs. string errors**:

```rust
// Typed (preferred): preserves BackendError variant end-to-end.
yield Err(DynamoError::builder()
    .error_type(ErrorType::Backend(BackendError::InvalidArgument))
    .message("bad request")
    .build());

// String: convenient, loses the typed variant.
yield Ok(LLMEngineOutput::error("something went wrong".into()));
```

Use typed errors when the failure category matters to the caller. Use
string errors when it doesn't.

#### Rust: `abort()` and per-request cleanup

`abort` is called by the framework **only** when `ctx.stopped()` or
`ctx.killed()` fires — i.e. an explicit client/operator cancel. It is
NOT called when the stream is silently dropped (TCP reset, consumer
timeout without cancellation).

**For cleanup that must run on any drop path** (releasing a scheduler
slot, freeing a request handle), use RAII inside the `generate` stream
body:

```rust
struct RequestGuard { /* ... */ }
impl Drop for RequestGuard {
    fn drop(&mut self) {
        // Always runs when the stream is dropped, however that happens.
    }
}

Ok(Box::pin(async_stream::stream! {
    let _guard = RequestGuard { /* ... */ };
    // ... your stream body
}))
```

The mocker's `ActiveRequestGuard` is the canonical example.

Use `abort` only for out-of-band notifications (e.g. telling a remote
scheduler to stop computing for this request).

#### Rust: `drain()` and `cleanup()`

- `drain()` runs once before shutdown, after the discovery
  unregister + grace-period sleep, while NATS/etcd are still alive.
  Use it for backend-side draining that must complete before the
  transport layer goes away (e.g. in-flight NIXL KV transfers on
  prefill workers). Default is no-op.
- `cleanup()` is called once on shutdown. Release all engine
  resources. The framework guarantees `cleanup()` runs exactly once if
  `start()` succeeded — even if registration or serve fails afterward.

Make `cleanup()` idempotent and tolerant of being called from a
half-initialized state. Engines like vLLM/TRT-LLM tear down NCCL groups
in `cleanup()` and a second attempt can hang.

### Rust Step 5: Write `main.rs`

Three lines. That's it.

```rust
use std::sync::Arc;

mod engine;

fn main() -> anyhow::Result<()> {
    let (engine, config) = engine::MyBackend::from_args(None)?;
    dynamo_backend_common::run(Arc::new(engine), config)
}
```

`run` installs signal handlers, builds the distributed runtime, calls
`engine.start()`, registers the model with discovery, serves the
endpoint, and runs the full graceful-shutdown orchestrator on
SIGTERM/SIGINT.

### Rust Step 6: Errors and logging

**Errors**: every error returned from `start`, `generate`, `cleanup`,
and `from_args` uses `ErrorType::Backend(BackendError::X)`. From the
frontend's perspective, anything bubbling up through the backend has
"originated from the backend" — engine code vs. framework code is not
distinguished. Top-level `ErrorType::X` variants are reserved for
non-backend paths.

A small helper module per backend keeps the call sites clean:

```rust
pub(crate) fn invalid_arg(msg: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::InvalidArgument))
        .message(msg)
        .build()
}
```

Common nested categories: `InvalidArgument`, `CannotConnect`,
`EngineShutdown`, `StreamIncomplete`, `Cancelled`, `ResponseTimeout`,
`Disconnected`, `ConnectionTimeout`, `Unknown`.

**Logging**: keep levels consistent across Rust backends so operators
see the same surface everywhere.

- `tracing::info!` for lifecycle milestones (engine started, cleanup
  complete). `Worker` already logs "Serving {model} on …" and "Engine
  cleanup complete" — add your own only for events those don't cover.
- `tracing::debug!` for per-request events (cancellation, abort).
- `tracing::warn!` for recoverable problems.
- `tracing::error!` only for unrecoverable failures.

### Rust Step 7: Run the conformance kit

Before merging, prove your engine satisfies the contract. The
conformance kit is one call:

```rust
#[tokio::test]
async fn my_engine_passes_conformance() {
    // `run_conformance` takes a factory closure rather than a built
    // engine — the kit constructs a second pristine engine for its
    // "cleanup-without-start" check.
    dynamo_backend_common::testing::run_conformance(|| {
        MyBackend::new(/* your defaults */).expect("construct")
    })
    .await
    .expect("conformance");
}
```

The kit runs `start`/`generate`/`cleanup` directly against your engine
— no external service is involved. If your engine needs a real GPU,
remote model server, or other heavyweight resource to construct, gate
the test with `#[ignore]` and require an explicit opt-in env var.

What it asserts:

| Check | Failure mode |
|---|---|
| `start()` returns a non-empty `EngineConfig.model` | `EmptyModelInConfig` |
| Single `generate()` ends in a terminal chunk | `NoTerminalChunk` |
| No chunks after the terminal | `ChunkAfterTerminal` |
| Interleaved `generate()` calls all succeed | `ConcurrentGenerateFailed` |
| Mid-stream cancel terminates within 2s | `CancellationNotObserved` |
| Cancelled stream's terminal is `FinishReason::Cancelled` | `CancellationIgnored` |
| `cleanup()` succeeds twice (idempotent) | `SecondCleanupFailed` |
| `cleanup()` on a never-started engine succeeds | `CleanupWithoutStartFailed` |

For tests that don't need a real engine, use `testing::mock_context()`
or `testing::cancelling_context(after)` to drive `generate` manually.

### Rust Step 8: Run it locally

Three moving parts need to come up: NATS + etcd (discovery and the
event/request planes), the Dynamo Python frontend (HTTP → backend
discovery), and your backend.

The fastest path is to copy the **mocker example's
[`docker-compose.yml`](../../lib/backend-common/examples/mocker/docker-compose.yml)
and [`Dockerfile.frontend`](../../lib/backend-common/examples/mocker/Dockerfile.frontend)**,
swap in your image, and run `docker compose up --build`. That brings
up NATS + etcd + the Python frontend (built from the dynamo workspace
at the same SHA as your backend) + your backend, all on one network.

For a non-Docker dev loop:

```bash
cargo build --release

# Ensure NATS + etcd are reachable (NATS_SERVER, ETCD_ENDPOINTS).
./target/release/my-backend Qwen/Qwen3-0.6B \
    --namespace dynamo \
    --component backend \
    --endpoint generate

# In another shell, start the Python frontend from the dynamo repo:
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

A successful response has non-empty `choices[0].message.content` and a
`finish_reason` of `stop` or `length`. `jq -e '.choices[0].finish_reason'`
is a good one-liner for a CI smoke test.

`run` initializes `tracing` from the `DYN_LOG` env var (defaults to
`info`); set `DYN_LOG=debug` or
`DYN_LOG=info,dynamo_backend_common=trace` for more detail. `RUST_LOG`
is not honored — `DYN_LOG` replaces it.

### Rust reference: mocker backend

[`lib/backend-common/examples/mocker`](https://github.com/ai-dynamo/dynamo/tree/main/lib/backend-common/examples/mocker)
is the canonical small-but-complete reference. Lift these patterns:

- Single shared scheduler driving many concurrent streams via a
  fan-out task and per-request `mpsc` channels.
- `ActiveRequestGuard` for RAII cleanup that runs on any stream drop.
- `biased` select with `ctx.stopped()` first, channel second — the
  shutdown-race fix discussed in Step 4.
- `cleanup()` signals every active stream via `ctx.stop_generating()`
  so each yields a clean `Cancelled` terminal instead of an error from
  channel-close.

### Rust checklist

Before shipping:

- [ ] `LLMEngine` implemented; `from_args` is inherent (not on the trait).
- [ ] All errors use `ErrorType::Backend(BackendError::X)`.
- [ ] `generate` polls `ctx.is_stopped()` between yields and emits
      `FinishReason::Cancelled` on cancel.
- [ ] Per-request cleanup uses RAII guards, not just `abort`.
- [ ] `cleanup` is idempotent.
- [ ] Conformance kit runs green: `testing::run_conformance(|| ...)`.
- [ ] Logging levels match the standards in Step 6.

### Rust see also

- [Crate README](../../lib/backend-common/README.md) — in-tree
  reference (architecture, file index, contracts at a glance).
- [`LLMEngine` trait](../../lib/backend-common/src/engine.rs) — authoritative contract.
- [Design notes](../../lib/backend-common/CLAUDE.md) — rationale and
  invariants.
- [`Worker`](../../lib/backend-common/src/worker.rs) — runtime
  lifecycle internals (signal handling, graceful shutdown, model
  registration).
- [Conformance kit](../../lib/backend-common/src/testing.rs) —
  `run_conformance`, `mock_context`, `cancelling_context`.
- [Mocker backend](../backends/mocker_backend/README.md) — example user guide.
- [Python sibling](../../components/src/dynamo/common/backend/README.md)
  — Python ABC layered over this crate.

</Tab>
</Tabs>
