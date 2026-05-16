---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Writing a Rust Unified Backend
---

# Writing a Rust Unified Backend

> **New — Dynamo's unified backend.** This guide covers the new
> **unified backend** infrastructure in
> [`dynamo-backend-common`](https://github.com/ai-dynamo/dynamo/tree/main/lib/backend-common): a shared
> `LLMEngine` contract that vLLM, SGLang, TRT-LLM, and the mocker
> already implement, and that any custom engine can plug into the
> same way.
>
> **Beta — actively under development.** The Rust native backend
> surface is beta quality and may change without backwards
> compatibility between releases. See [Feature gaps](#feature-gaps)
> below for what the unified path covers today versus the existing
> (non-unified) backend paths.

This guide walks through building a Rust unified backend for an
inference engine that plugs into Dynamo's distributed runtime. A
unified backend is a standalone Rust binary that owns its engine and
serves requests via the shared
[`LLMEngine`](https://github.com/ai-dynamo/dynamo/blob/main/lib/backend-common/src/engine.rs) contract in
[`dynamo-backend-common`](https://github.com/ai-dynamo/dynamo/tree/main/lib/backend-common) — no Python
worker runtime required. For the Python version of the same contract
see [Writing a Python Unified Backend](python-backend-guide.md).

Your backend lives in its own crate and **does not need to be part of
the dynamo repository**. It pulls `dynamo-backend-common` in as a normal
git or path dependency. The steps below assume you're starting a fresh
crate in your own repo; an optional note in Step 1 covers the in-tree
variant for contributors landing a backend inside `ai-dynamo/dynamo`.

For a Python engine, use
[Writing a Python Unified Backend](python-backend-guide.md) — same
contract, lighter setup. The non-unified fallback for feature gaps
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

## Feature gaps

The unified backend is in beta and does not yet cover the full feature
set of Dynamo's existing (non-unified) backend paths. The summary
below is the common contract — what every engine on the unified path
gets, whether written in Rust directly or plugged in from Python via
the PyO3 `Worker` shim. Per-engine gaps (vLLM, SGLang, TRT-LLM
specifics like LoRA, diffusion, attention DP scheduling) live in the
[Python package README](../../components/src/dynamo/common/backend/README.md#feature-gaps).

**Supported today**

- Aggregated token-in-token-out inference
- Disaggregated serving (`Aggregated` / `Prefill` / `Decode`) with
  bootstrap (SGLang) or internal KV transport (vLLM, TRT-LLM); the
  Rust [mocker example](https://github.com/ai-dynamo/dynamo/tree/main/lib/backend-common/examples/mocker)
  exercises the same wire format CPU-only
- Model registration with discovery and endpoint types
- Request cancellation via in-stream `ctx.is_stopped()` polling plus
  the framework's out-of-band `abort()` monitor
- Typed `DynamoError` with `ErrorType::Backend(BackendError::X)`
- Graceful shutdown with signal handling, `drain()` hook, and 3-phase
  distributed-runtime teardown
- Debug-build stream validator and the `testing::run_conformance` kit

**Not yet on the unified path**

| Feature | What's missing |
|---------|----------------|
| Metrics & Prometheus | Engine-level metrics, KV utilization gauges, multiprocess registry |
| KV event publishing | Prefix cache events (BlockStored / Removed) to router via ZMQ or NATS |
| Health check payloads | Per-engine custom health probes |
| Logprobs | Selected + top-k logprob extraction and streaming |
| Guided decoding | JSON schema, regex, grammar, choice constraints |
| OpenTelemetry tracing | Trace headers, request perf metrics, OTEL propagation |
| Engine routes | Profiling, memory release/resume, weight updates (disk/tensor/distributed/IPC) |
| Data-parallel routing | DP rank extraction, DP-aware scheduling |
| Text-in-text-out mode | `ModelInput::Text` is rejected at startup — `Tokens` only |
| Custom Jinja chat templates | Plumbed through `WorkerConfig` but not yet honored end-to-end |
| Snapshot/checkpoint | CRIU-based engine state save/restore |
| Multimodal | Images, video, embeddings, separate encode workers |
| LoRA adapters | Dynamic load/unload, per-adapter serialization |

If you need one of these features today, keep that workload on the
existing per-engine entry point until the unified path catches up.

## What you're building

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

## Prerequisites

- Rust 1.85 or newer (the dynamo workspace is edition 2024). The
  toolchain pin in Step 1 locks this in for you; older toolchains will
  fail with `feature edition2024 is required` deep inside the build.
- NATS and etcd reachable for end-to-end runs. The dynamo repo's
  `deploy/docker-compose.yml` brings up both in one command if you
  don't already have them running.
- Familiarity with `async` Rust, `tokio`, and `clap`. The trait uses
  `async_trait`, and the framework expects a `tokio` runtime.

## Step 1: Create the crate

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

### Getting the `dynamo-backend-common` crate

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

### Two build-time requirements you cannot skip

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

## Step 2: Define your engine struct

In `src/engine.rs` (or whatever you named it), declare a struct that
owns whatever state your engine needs. Anything you allocate inside
`start()` later must live behind interior mutability so the trait's
`&self` methods can reach it.

```rust
use async_trait::async_trait;
use dynamo_backend_common::engine::GenerateContext;
use dynamo_backend_common::{
    BackendError, CommonArgs, DynamoError, EngineConfig, ErrorType, FinishReason, LLMEngine,
    LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest, WorkerConfig, chunk, usage,
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

## Step 3: Wire up CLI arguments

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

## Step 4: Implement the `LLMEngine` trait

The trait has three required methods (`start`, `generate`, `cleanup`)
plus two with default implementations you can override (`abort`, `drain`).

### `start()`

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
        context_length: Some(8192),
        kv_cache_block_size: Some(64),     // None if no block-structured KV
        total_kv_blocks: Some(16384),
        max_num_seqs: Some(256),
        max_num_batched_tokens: Some(8192),
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

### `generate()`

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

### `abort()` and per-request cleanup

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

### `drain()` and `cleanup()`

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

## Step 5: Write `main.rs`

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

## Step 6: Errors and logging

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

## Step 7: Run the conformance kit

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

## Step 8: Run it locally

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

## Reference: the mocker backend

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

## Checklist

Before shipping:

- [ ] `LLMEngine` implemented; `from_args` is inherent (not on the trait).
- [ ] All errors use `ErrorType::Backend(BackendError::X)`.
- [ ] `generate` polls `ctx.is_stopped()` between yields and emits
      `FinishReason::Cancelled` on cancel.
- [ ] Per-request cleanup uses RAII guards, not just `abort`.
- [ ] `cleanup` is idempotent.
- [ ] Conformance kit runs green: `testing::run_conformance(|| ...)`.
- [ ] Logging levels match the standards in Step 6.

## See also

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
