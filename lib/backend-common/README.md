<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Rust Backend (`dynamo-backend-common`)

> **Work in progress.** The unified backend supports aggregated and
> disaggregated (prefill/decode) inference; multimodal, LoRA, logprobs,
> guided decoding, and engine-level metrics are still on the
> non-unified path. The Python `Worker`
> ([`dynamo.common.backend`](../../components/src/dynamo/common/backend/))
> is a thin shim over this crate.

> **Looking for a walkthrough?** Start with
> [Writing a Rust Unified Backend](../../docs/development/rust-backend-guide.md).
> This README is the in-tree reference: trait shape, file layout,
> disaggregation contract, error taxonomy, and the conformance kit.

A two-type abstraction that separates **runtime integration** (common
across all backends) from **engine logic** (vLLM, SGLang, TRT-LLM, your
custom engine, etc.).

## Architecture

```text
LLMEngine (trait)              <-- engine boundary (engine.rs)
    |   - start(worker_id) -> Result<EngineConfig, DynamoError>
    |   - generate(request, ctx) -> Result<BoxStream<...>, DynamoError>
    |   - abort(ctx)                            (optional, default no-op)
    |   - drain() -> Result<(), DynamoError>    (optional, default no-op)
    |   - cleanup() -> Result<(), DynamoError>
    |
    +-- MockerBackend          <-- examples/mocker/src/engine.rs
    +-- <your backend>         <-- a separate crate

Worker (concrete, non-generic)  <-- runtime integration (worker.rs)
    - receives WorkerConfig from the per-backend `from_args`
    - creates DistributedRuntime
    - installs SIGTERM/SIGINT handlers
    - calls engine.start(worker_id), registers model with discovery
    - serves the generate endpoint with cancellation monitoring
    - on shutdown: discovery unregister -> grace period
                   -> engine.drain() -> engine.cleanup()
                   -> 3-phase distributed-runtime teardown

run(engine, config)             <-- src/run.rs
    - Single entry point used by each backend's `main.rs`.
    - Non-generic; holds `Arc<dyn LLMEngine>` so PyO3-wrapped engines
      plug in through the same path.
```

`from_args` is **not** on the trait — each backend exposes an inherent
constructor that returns `(Self, WorkerConfig)`. This keeps the trait
fully object-safe (`Arc<dyn LLMEngine>` must work) and lets `run` stay
non-generic.

`generate` takes `GenerateContext` while `abort` takes
`Arc<dyn AsyncEngineContext>`. `GenerateContext` derefs to
`AsyncEngineContext`, so cancellation calls (`ctx.stopped()`,
`ctx.is_stopped()`, `ctx.id()`) work the same in both. Override
`abort` only if you need engine-side cancel notification — most
backends rely on the default no-op plus in-stream polling.

## Quick Start

### Running the mocker example

```bash
cargo run -p dynamo-mocker-backend --release -- \
    --model-path Qwen/Qwen3-0.6B
```

The mocker is a CPU-only reference engine: it wraps the
`dynamo-mocker` scheduler in the `LLMEngine` trait and emits randomized
token IDs at a configurable rate. Useful as a stand-in for AIPerf /
end-to-end pipeline smoke tests without any ML dependencies.

A one-command docker-compose stack (NATS + etcd + frontend + mocker)
lives at
[`examples/mocker/docker-compose.yml`](examples/mocker/docker-compose.yml).

### Running your own backend

```bash
# In your backend crate (which may live in its own repo):
cargo run --release -- --help
```

See the [walkthrough](../../docs/development/rust-backend-guide.md) for
how to set up the crate (Cargo.toml, `tokio_unstable` cfg flag, toolchain
pin) and write the engine.

## Implementing a New Backend

Implement the `LLMEngine` trait on your engine struct, expose an
inherent `from_args`, and write a three-line `main.rs`:

```rust
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::engine::GenerateContext;
use dynamo_backend_common::{
    DynamoError, EngineConfig, LLMEngine, LLMEngineOutput, PreprocessedRequest,
    WorkerConfig,
};
use futures::stream::BoxStream;

pub struct MyBackend { /* engine state */ }

impl MyBackend {
    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        // parse CLI args, build the engine + WorkerConfig
        todo!()
    }
}

#[async_trait]
impl LLMEngine for MyBackend {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        todo!() // start the engine, return registration metadata
    }

    async fn generate(
        &self,
        _request: PreprocessedRequest,
        _ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        todo!() // yield streaming chunks
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        todo!() // release engine resources
    }
}

fn main() -> anyhow::Result<()> {
    let (engine, config) = MyBackend::from_args(None)?;
    dynamo_backend_common::run(Arc::new(engine), config)
}
```

See [`examples/mocker/src/engine.rs`](examples/mocker/src/engine.rs)
for a complete, runnable reference and the
[walkthrough](../../docs/development/rust-backend-guide.md) for the
step-by-step including Cargo.toml, `tokio_unstable` cfg, and the
conformance kit.

## Disaggregated Serving

The Rust crate supports prefill / decode worker splits. A backend
declares its role via `WorkerConfig.disaggregation_mode` and branches on
it inside `generate`:

```rust
use dynamo_backend_common::{DisaggregationMode, WorkerConfig};

let config = WorkerConfig {
    namespace: "dynamo".into(),
    component: "prefill".into(),
    endpoint: "generate".into(),
    disaggregation_mode: DisaggregationMode::Prefill, // or Decode / Aggregated
    ..Default::default()
};
```

Roles and `Worker` behavior:

| Mode | Role | Worker effects |
| --- | --- | --- |
| `Aggregated` | Self-contained inference (default) | Standard registration; KV indexer enabled |
| `Prefill`    | Run prompt → emit 1 token + KV handoff | Registers as `ModelType::Prefill`; advertises `bootstrap_host`/`port` if set in `EngineConfig` |
| `Decode`     | Resume from a prefill peer's KV | Disables the local indexer (KV is owned by the prefill peer) |

The crate re-exports `PrefillResult` and `BootstrapInfo` from
[`dynamo-llm`'s protocol types](../llm/src/protocols/common/preprocessor.rs);
these decorate `PreprocessedRequest` on decode-bound requests. Prefill
terminals carry their handoff payload via the engine's terminal chunk
(e.g. `disaggregated_params`).

For backends with an internal KV transport (vLLM `NixlConnector`,
TRT-LLM's transceiver), leave `EngineConfig.bootstrap_host`/`port` `None`
— only SGLang uses the Dynamo-level handshake today.

## Request / Response Contract

The trait works with the same `PreprocessedRequest` / `LLMEngineOutput`
types used across preprocessing, routing, and the frontend — no
Python-shaped wrappers.

`generate` returns a `BoxStream<'static, Result<LLMEngineOutput, DynamoError>>`:

- Exactly one **terminal item** must be the last item yielded. A
  terminal is either:
  - `Ok(chunk)` with `finish_reason` set (`stop` / `length` /
    `cancelled` / `error`), or
  - `Err(DynamoError)` carrying a typed mid-stream failure.
- Non-terminal items are `Ok(chunk)` with `finish_reason` unset.
- No items may follow a terminal.

Terminal chunks come from
`LLMEngineOutput::stop()` / `::length()` / `::cancelled()` / `::error(msg)`,
optionally chained with `LLMEngineOutputExt::with_tokens(...)` /
`with_usage(usage(prompt, completion))`. Non-terminal chunks use
`chunk::token(id)`.

In debug builds, the framework wraps the stream in a validator
([`src/validate.rs`](src/validate.rs)) that panics if a chunk is yielded
after a terminal. Loud failures in dev and test, compiled out in
release.

## Cancellation Contract

The framework runs a per-request cancellation monitor that watches
`ctx.stopped()` / `ctx.killed()` and calls `engine.abort(ctx)` when
either fires. Engines also **must** poll `ctx.is_stopped()` (or
`await ctx.stopped()`) between yields and emit a terminal with
`FinishReason::Cancelled` when they observe it — the conformance kit
treats any other terminal after cancellation as ignoring the signal.

For cleanup that must run on **any** drop path (TCP reset, consumer
timeout without cancellation), use RAII inside the `generate` stream
body, not `abort` — `abort` only fires on explicit cancel. The mocker's
`ActiveRequestGuard` is the canonical example.

## Error Handling

Errors returned from `start`, `generate`, `cleanup`, and `from_args`
use `ErrorType::Backend(BackendError::X)` from
[`dynamo-runtime`](../runtime/). Common variants:

| Variant | When |
| --- | --- |
| `InvalidArgument` | Engine or setup rejected the input |
| `CannotConnect` | Can't reach discovery / a dependency |
| `EngineShutdown` | Engine failed to start / crashed |
| `StreamIncomplete` | Stream ended before the engine could finish |
| `Cancelled` | Request was cancelled |
| `ResponseTimeout`, `Disconnected`, `ConnectionTimeout` | Transport failures |
| `Unknown` | Uncategorized |

Mid-stream errors have two equivalent terminal forms:

- **Typed** (preferred): yield `Err(DynamoError)` from the stream.
  Forwarded as `Annotated::error` with the `BackendError` variant
  preserved end-to-end.
- **String**: yield `Ok(LLMEngineOutput::error(msg))`. Convenient for
  pure message-level failures. Loses the typed `BackendError` variant.

A tiny helper per backend keeps call sites clean — see the
[guide's Step 6](../../docs/development/rust-backend-guide.md) for the
`invalid_arg` pattern.

## Conformance Kit

Enable the `testing` cargo feature to pull in the kit:

```toml
[dev-dependencies]
dynamo-backend-common = { workspace = true, features = ["testing"] }
```

```rust
#[tokio::test]
async fn my_engine_passes_conformance() {
    dynamo_backend_common::testing::run_conformance(|| {
        MyBackend::new(/* defaults */).expect("construct")
    })
    .await
    .expect("conformance");
}
```

The kit asserts:

| Check | Failure mode |
| --- | --- |
| `start()` returns a non-empty `EngineConfig.model` | `EmptyModelInConfig` |
| Single `generate()` ends in a terminal chunk | `NoTerminalChunk` |
| No chunks after the terminal | `ChunkAfterTerminal` |
| Interleaved `generate()` calls all succeed | `ConcurrentGenerateFailed` |
| Mid-stream cancel terminates within 2s | `CancellationNotObserved` |
| Cancelled stream's terminal is `FinishReason::Cancelled` | `CancellationIgnored` |
| `cleanup()` succeeds twice (idempotent) | `SecondCleanupFailed` |
| `cleanup()` on a never-started engine succeeds | `CleanupWithoutStartFailed` |

`testing::mock_context()` and `testing::cancelling_context(after)` are
available for hand-written tests.

## File Index

```text
lib/backend-common/
    Cargo.toml
    CLAUDE.md            # Design notes (rationale, invariants, phase plans)
    README.md            # This file
    src/
        lib.rs           # Module wiring + public re-exports
        engine.rs        # LLMEngine trait, EngineConfig, GenerateContext,
                         #   chunk::token, LLMEngineOutputExt setters,
                         #   usage() helper, PreprocessedRequest / Output
                         #   re-exports
        worker.rs        # Worker concrete + WorkerConfig (incl.
                         #   disaggregation_mode); lifecycle state machine;
                         #   signal handling + graceful shutdown orchestrator
        run.rs           # `pub fn run(engine, config) -> anyhow::Result<()>`
        adapter.rs       # EngineAdapter — bridges LLMEngine to AsyncEngine;
                         #   cancellation monitor + debug stream validator
        args.rs          # CommonArgs — shared CLI flags every engine flattens
        disagg.rs        # DisaggregationMode enum + clap value-parser
        error.rs         # Re-exports DynamoError / ErrorType / BackendError
        validate.rs      # Debug-build stream validator (compiled out in release)
        testing.rs       # Conformance kit (`testing` feature)
    examples/
        mocker/          # CPU-only reference backend + docker-compose stack
```

The Python `Worker` shim that drives this crate from `dynamo.*.unified_main`
entry points lives at
[`components/src/dynamo/common/backend/worker.py`](../../components/src/dynamo/common/backend/worker.py).

## See Also

- [Writing a Rust Unified Backend](../../docs/development/rust-backend-guide.md)
  — step-by-step walkthrough.
- [`CLAUDE.md`](CLAUDE.md) — design notes (rationale, invariants,
  Phase 2 PyO3 plans).
- [Mocker example](examples/mocker/) — reference engine + compose stack.
- [Python sibling](../../components/src/dynamo/common/backend/README.md)
  — `dynamo.common.backend`, the Python ABC layered over this crate.
- [DEP #8251](https://github.com/ai-dynamo/dynamo/issues/8251) —
  Backend Interface proposal and ongoing status.
