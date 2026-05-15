# Backend Common (Rust)

Shared runtime glue for Rust LLM backends. Two-type abstraction:
`Worker` (runtime lifecycle) and `LLMEngine` (trait for engine-specific
logic). A reference implementation lives at
`lib/backend-common/examples/mocker/`.

Engines work directly with `PreprocessedRequest` and `LLMEngineOutput`
— the same types the rest of the Rust pipeline uses. No separate
Python-shaped request/response wrappers.

## Engine Lifecycle

```
construct  ->  start(worker_id)  ->  generate() / abort()  ->  drain()  ->  cleanup()
    |                |                       |                    |            |
parse args,    start engine,           serve requests       pre-cleanup    shutdown,
return engine  return metadata         (concurrent)         drain          release
```

The trait has five methods. `from_args` is NOT on the trait — each
backend exposes a backend-specific constructor (typically a sync
`from_args(argv) -> Result<(Self, WorkerConfig)>` inherent method).
This keeps the trait fully object-safe without a `where Self: Sized`
opt-out and lets `run.rs` stay non-generic.

- `start(&self, worker_id: u64) -> Result<EngineConfig, DynamoError>` —
  interior mutability over `&self` so `Arc<dyn LLMEngine>` can drive
  the lifecycle. `worker_id` is an opaque runtime-allocated identifier;
  most engines ignore it. Backends needing a stable cluster-wide key
  (e.g. TRT-LLM's `disagg_machine_id` snowflake) derive from it.
- `generate(&self, request, ctx: GenerateContext) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError>`
  — streaming inference. `GenerateContext` derefs to
  `dyn AsyncEngineContext` (`ctx.stopped()`, `ctx.is_stopped()`,
  `ctx.id()` work transparently); it additionally exposes
  `notify_first_token()` for decode-mode requests. Author returns a
  plain stream; the framework wraps it in `Annotated` and plumbs
  cancellation.
- `abort(&self, ctx: Arc<dyn AsyncEngineContext>)` — optional, default
  no-op. Note the type asymmetry: `generate` takes `GenerateContext`,
  `abort` takes the unwrapped `Arc<dyn AsyncEngineContext>`. Called by
  the framework ONLY when `ctx.stopped()` or `ctx.killed()` fires
  during an active request — NOT on silent stream drops (TCP reset,
  consumer-side timeout, etc.). For per-request cleanup that must run
  on ANY drop path (releasing a scheduler slot, freeing an engine
  handle), put the release logic inside the `generate` stream body
  using RAII; use `abort` only for out-of-band notifications (e.g.
  telling a remote scheduler to cancel compute early).
- `drain(&self) -> Result<(), DynamoError>` — optional, default no-op.
  Called once during graceful shutdown after the discovery unregister
  + grace-period sleep, but BEFORE `cleanup`. Use it for backend-side
  draining that must complete while NATS / etcd are still alive — e.g.
  prefill workers polling-until-idle so in-flight NIXL KV transfers
  finish before GPU memory is released (issue #7319). Failures are
  logged and swallowed; shutdown proceeds regardless.
- `cleanup(&self) -> Result<(), DynamoError>` — called exactly once.
  Runs after `start()` returns Ok on shutdown (even if registration /
  serve fails), **and** after `start()` raises — so implementations
  must be null-safe against partial state (inner LLM, sockets,
  background tasks). Must also be idempotent: a second call after a
  successful first returns `Ok(())` without re-entering teardown. The
  conformance kit pins both — `CleanupWithoutStartFailed` and
  `SecondCleanupFailed`.

## Contract for `generate`

The stream item type is `Result<LLMEngineOutput, DynamoError>`.

Exactly one **terminal item** must be the last item yielded. A terminal
item is either:

  * `Ok(chunk)` with `finish_reason = Some(...)`, or
  * `Err(dynamo_err)` carrying a typed mid-stream failure.

Non-terminal items are `Ok(chunk)` with `finish_reason` unset. Terminal
`Ok` chunks may carry tokens (the final tokens of the completion) or be
empty — the contract is that `finish_reason` (or an `Err`) marks the end.

`completion_usage` on a terminal `Ok` chunk is optional (but recommended —
the OpenAI frontend aggregates it when present).

In debug builds, the framework wraps the stream in a validator that
panics on violations — loud failures in dev and test, compiled out in
release.

Rule the validator enforces:

1. No item may be yielded after a terminal item (terminal `Ok` chunk or `Err`).

The validator does **not** enforce "stream must end with a terminal
chunk" — a stream may end early for legitimate reasons (adapter breaks
on cancellation before the engine's final yield). The conformance kit
catches the missing-terminal case with `ConformanceFailure::NoTerminalChunk`,
so run it against your engine to confirm end-of-stream correctness.

Engines **must** poll `ctx.is_stopped()` between yields and, on
cancellation, emit a terminal chunk whose `finish_reason` is
`FinishReason::Cancelled`. The conformance kit enforces this — any
other `finish_reason` after cancellation (`Length`, `Stop`, etc.) is
treated as the engine ignoring the cancel signal. The framework also
runs an out-of-band monitor that calls `engine.abort(ctx)` when either
`stopped()` or `killed()` fires — this is for releasing engine-side
resources (KV slots, scheduler entries) and runs concurrently with the
in-stream cancel check.

## Output construction

Non-terminal chunks use `chunk::token(id)`. Terminal chunks use the
upstream `LLMEngineOutput` constructors, optionally chained with the
`LLMEngineOutputExt` fluent setters:

```rust
use dynamo_backend_common::{chunk, LLMEngineOutput, LLMEngineOutputExt, usage};

// Non-terminal
yield Ok(chunk::token(id));

// Length-terminated with final token(s) and usage stats.
yield Ok(LLMEngineOutput::length()
    .with_tokens(vec![final_id])
    .with_usage(usage(prompt_len, n)));

// Cancellation with partial-usage stats.
yield Ok(LLMEngineOutput::cancelled()
    .with_usage(usage(prompt_len, generated)));

// Typed mid-stream error — preserves BackendError variant downstream.
yield Err(DynamoError::builder()
    .error_type(ErrorType::Backend(BackendError::InvalidArgument))
    .message("bad request")
    .build());

// Also available upstream: LLMEngineOutput::stop(), LLMEngineOutput::error(msg).
```

`completion_usage` on the terminal chunk is optional — the frontend
aggregates it when present. `usage(prompt, completion)` computes
`total_tokens` for you (saturating on overflow).

## Design Constraints

- **ZERO duplication across engine implementations.** Before writing
  logic inside an `LLMEngine` impl, check whether the same logic already
  exists in another engine. If it does, extract into `Worker` or a
  shared utility.

- **Exactly two types.** `Worker` owns runtime lifecycle. `LLMEngine`
  owns inference. No intermediate traits or mixins.

- **Object-safe trait.** `Arc<dyn LLMEngine>` must work. All methods
  take `&self`. Constructors are backend-specific, not on the trait.

- **Non-generic `Worker`, `EngineAdapter`, and `run()`.** All hold
  `Arc<dyn LLMEngine>`. This is load-bearing for the PyO3 path:
  `components/src/dynamo/common/backend/worker.py` is a thin shim
  over `dynamo._core.backend.Worker` (this crate), so Python engines
  plug in through the same `Arc<dyn LLMEngine>` slot via a
  `PyLLMEngine` adapter.

- **Reuse `DynamoError`.** Trait methods return `DynamoError`
  (`dynamo_runtime::error`), the workspace-wide standardized error.
  Engine-originated failures use `ErrorType::Backend(BackendError::X)`
  where `BackendError` is the runtime's nested category enum. No custom
  error types inside backend-common.

## Disaggregated Serving

`DisaggregationMode` (`disagg.rs`) is metadata carried on `WorkerConfig`.
`Aggregated` is the default and keeps existing callers unchanged.
`CommonArgs` exposes the `--disaggregation-mode` flag (env
`DYN_DISAGGREGATION_MODE`) so engines that flatten `CommonArgs` get the
flag automatically.

What the **`Worker`** does with the mode at registration time:

- `Prefill` → register with `ModelType::Prefill` regardless of
  `endpoint_types`, so the frontend's `PrefillRouter` targets it.
- `Decode` → keep `endpoint_types`, but force-disable
  `enable_local_indexer` (decode workers don't host the indexer
  endpoint, so they must not advertise it).
- `Aggregated` → register with the parsed `endpoint_types`.

What an **`LLMEngine`** does with the mode (engine-side dispatch in
`generate` and `drain`): see `examples/mocker` for a worked reference.
The mocker stamps a synthetic `disaggregated_params` payload on the
prefill terminal and rejects decode requests that arrive without
`PrefillResult`. Real engines run an analogous protocol with their
own KV transfer transport.

`drain` is the prefill shutdown hook: poll-until-idle so in-flight
NIXL/KV transfers finish before GPU memory is released. Aggregated and
decode engines leave the default no-op.

`PrefillResult` and `BootstrapInfo` are re-exported from
`dynamo-backend-common` so engines don't need a separate `dynamo-llm`
dep just to read these fields off `PreprocessedRequest`.

## Adding a New Engine

1. Create a new Rust crate depending on `dynamo-backend-common`. Place
   it under `lib/` (e.g. `lib/<backend>-rs/`) following the repo's
   Rust-crate convention. Do **not** place Rust crates under
   `components/src/dynamo/` — that tree is the Python package
   namespace.
2. In `src/<backend>_engine.rs`: `struct YourEngine; impl LLMEngine for YourEngine`.
   Plus an inherent `impl YourEngine { pub fn from_args(argv) -> Result<(Self, WorkerConfig), DynamoError> }`.
3. Implement `start`, `generate`, `cleanup` (required) and `abort` (optional).
4. Create `src/main.rs`:
   ```rust
   use std::sync::Arc;
   mod your_engine;

   fn main() -> anyhow::Result<()> {
       let (engine, config) = your_engine::YourEngine::from_args(None)?;
       dynamo_backend_common::run(Arc::new(engine), config)
   }
   ```
5. Use `engine.rs` from `lib/backend-common/examples/mocker/`
   as a template.
6. Run the conformance test kit (see Testing below) against your engine.

## Error Handling

Engines return `DynamoError` from `start`, `generate`, `cleanup`, and
`from_args`. Build one with the builder pattern:

```rust
use dynamo_backend_common::{BackendError, DynamoError, ErrorType};

return Err(DynamoError::builder()
    .error_type(ErrorType::Backend(BackendError::InvalidArgument))
    .message(format!("bad param: {reason}"))
    .build());
```

**Convention: all errors emitted from backend-common and from engine
implementations use `ErrorType::Backend(BackendError::X)`.** From the
frontend/router's perspective, everything bubbling up through the
backend layer has originated "from the backend" — the internal split
between Worker framework code and engine implementation code is not
visible (or relevant) outside the backend. Top-level `ErrorType::X`
variants are reserved for non-backend code paths (pipeline transport,
frontend parsing, router scheduling).

Common nested categories: `BackendError::InvalidArgument` (engine or
setup rejected the input), `BackendError::CannotConnect` (can't reach
discovery / a dependency), `BackendError::EngineShutdown` (engine
failed to start / crashed), `BackendError::Unknown` (uncategorized),
`BackendError::StreamIncomplete` (stream ended before the engine
could finish), `BackendError::Cancelled`, `BackendError::ResponseTimeout`,
`BackendError::Disconnected`, `BackendError::ConnectionTimeout`.

Mid-stream errors have two equivalent terminal forms:

  * **Typed** (preferred): yield `Err(DynamoError)` from the stream.
    Forwarded as `Annotated::error` with the `ErrorType::Backend(...)`
    variant preserved end-to-end. Use this when the failure category
    matters to the caller (e.g. `BackendError::InvalidArgument` vs.
    `Disconnected`).
  * **String**: yield `Ok(LLMEngineOutput::error(msg))`. Convenient for
    pure message-level failures. Loses the typed `BackendError` variant.

## KV-aware Routing Sources

Engines opt into KV-aware routing by overriding two trait methods on
`LLMEngine`:

- `kv_event_sources() -> Vec<KvEventSource>` — one descriptor per
  data-parallel rank that emits KV cache events.
- `metrics_sources() -> Vec<MetricsSource>` — one descriptor per
  data-parallel rank reporting load (`kv_used_blocks`).

Both default to empty. Returning empty opts the worker out of KV-aware
routing entirely; the router falls back to non-KV scheduling for that
worker. `Worker` calls these once after `start()` succeeds and constructs
the publishers itself — engines never instantiate
`KvEventPublisher`/`WorkerMetricsPublisher`. On shutdown, `Worker` drops
the handles while NATS is still alive.

### `KvEventSource` flavors

Pick based on how the engine's KV event API is shaped:

- `Zmq { endpoint, topic, dp_rank }` — engine already publishes to a
  ZMQ PUB socket. `Worker` subscribes directly.
- `Push { on_ready, dp_rank }` — engine has a blocking poll API.
  `Worker` constructs the publisher, then calls `on_ready(publisher)`
  once during setup. The engine stashes the publisher and drives
  `publish_stored` / `publish_removed` from its own thread. The engine
  **must** stop that thread in `cleanup()`.

The `mocker` example uses `Push` with a no-op `on_ready`; real Rust
engines would spawn a polling thread inside `on_ready`. See
`examples/mocker/src/engine.rs` for the wire-up.

### `MetricsSource`

```rust
MetricsSource {
    snapshot: Arc::new(move || Some(Metrics { kv_used_blocks: ... })),
    dp_rank,
}
```

`Worker` invokes `snapshot()` on a fixed interval. It **must** be a
cheap member-field read — engine-internal calls land in the 10s of ms
and stall the publish loop. The conformance kit enforces a 1 ms ceiling
(`MetricsSnapshotTooSlow`). Return `None` to skip publishing for that
tick (e.g. before the engine has emitted its first scheduler iteration).

### Conformance

The kit asserts that both methods are idempotent across calls (rank set
is stable for the engine's lifetime) and that snapshot fns satisfy the
latency ceiling. See `lib/backend-common/src/testing.rs`.

## Logging

Keep logging standardized across all Rust engines. When adding or
changing a log message in one engine, check whether the same lifecycle
event is logged in the others and update them to match.

Level standards:
- `tracing::info!` for lifecycle milestones (engine started, serving
  begun, cleanup complete). `Worker` already emits "Serving {model} on
  …" and "Engine cleanup complete" — engine code adds its own only
  when those don't already cover the event.
- `tracing::debug!` for per-request events (request abort, cancellation).
- `tracing::warn!` for recoverable problems.
- `tracing::error!` only for unrecoverable failures.

## Testing

Enable the `testing` cargo feature to pull in the conformance kit:

```toml
[dev-dependencies]
dynamo-backend-common = { workspace = true, features = ["testing"] }
```

```rust
use dynamo_backend_common::testing;

#[tokio::test]
async fn my_engine_satisfies_contract() {
    testing::run_conformance(MyEngine::new_for_test)
        .await
        .expect("conformance");
}
```

`run_conformance` takes a factory rather than a built engine — it
constructs one engine for the main lifecycle test and a second
pristine engine for the "cleanup before start" check.

The kit asserts:

- `start()` returns a non-empty `EngineConfig.model`.
- A single `generate()` yields a well-formed stream ending in a
  terminal chunk (`finish_reason` set; `completion_usage` is optional).
- 8 interleaved `generate()` calls all complete successfully
  (catches shared-state bugs under concurrent polling).
- After `stop_generating()` fires mid-stream, the stream terminates
  within a 2s deadline (else `CancellationNotObserved`). If the last
  chunk yielded is not a `FinishReason::Cancelled` terminal — any
  other terminal reason, or no terminal at all — the check raises
  `ConformanceFailure::CancellationIgnored`.
- `cleanup()` succeeds and is idempotent (two calls in a row both Ok).
- `cleanup()` is safe on a never-started engine — mirrors `Worker`'s
  post-start-failure path. Failures here surface as
  `CleanupWithoutStartFailed`.

Also available: `testing::mock_context()` and
`testing::cancelling_context(after)` for hand-written tests.

## Key Files

| File | What it does |
|------|-------------|
| `engine.rs` | `LLMEngine` trait, `EngineConfig`, `GenerateContext`, `chunk::token`, `LLMEngineOutputExt` setters, `usage()` helper. Re-exports `PreprocessedRequest` / `LLMEngineOutput` / `FinishReason` / `PrefillResult` / `BootstrapInfo` / etc. |
| `worker.rs` | `Worker` — runtime lifecycle: create `DistributedRuntime`, register model (with `disaggregation_mode` adjustments), serve endpoint, orchestrate drain + cleanup. `WorkerConfig` lives here. |
| `adapter.rs` | `EngineAdapter` — bridges `LLMEngine` to `AsyncEngine`. Cancellation monitor + debug-build validator wrapping. |
| `run.rs` | `pub fn run(engine, config)` — entry point used by all per-backend `main.rs`. Non-generic. |
| `args.rs` | `CommonArgs` — shared CLI flags (`--namespace`, `--component`, `--disaggregation-mode`, etc.) that every engine's `Args` flattens in. |
| `disagg.rs` | `DisaggregationMode` enum (`Aggregated` / `Prefill` / `Decode`) with `clap::ValueEnum` derive. |
| `error.rs` | Re-exports `DynamoError`, `ErrorType`, `BackendError` from `dynamo-runtime`. No custom error types. |
| `validate.rs` | Debug-build stream validator. Compiled out in release. |
| `testing.rs` | Conformance test kit. Gated behind the `testing` feature. |

## PyO3 binding

Shipped: `dynamo._core.backend.Worker` is a PyO3 binding that hands a
`PyLLMEngine` (Python-implemented) into the same `Arc<dyn LLMEngine>`
slot the Rust path uses. The Python-side
`dynamo.common.backend.Worker` (`components/src/dynamo/common/backend/worker.py`)
is a thin wrapper that drives this. The lifecycle state machine,
signal handling, and graceful-shutdown orchestrator live entirely in
this crate — Python adds no lifecycle logic.
