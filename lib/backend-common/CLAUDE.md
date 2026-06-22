# Backend Common (Rust)

Shared runtime glue for Rust backends. `Worker` (runtime lifecycle)
drives one of two engine traits, selected by request modality:

- `LLMEngine` — token pipeline. `generate` takes `PreprocessedRequest`
  (`token_ids` in) and yields `LLMEngineOutput` (`token_ids` out), the
  same types the rest of the Rust pipeline uses. Served through
  `EngineAdapter`. A reference implementation lives at
  `lib/backend-common/examples/mocker/`.
- `RawEngine` — raw media pipeline (image/video/audio generation).
  `generate` takes the forwarded request as `serde_json::Value` and
  yields response objects as `serde_json::Value` — no tokenizer,
  detokenizer, or KV cache. Served through `RawEngineAdapter`. The
  contract is modality-neutral: a new media modality is a new
  `RawEngine`, not a new framework path.

`Worker` registration honours `WorkerConfig.model_input`: `LLMEngine`
requires `ModelInput::Tokens`; `RawEngine` requires `Text`/`Tensor`.
`endpoint_types` parses `images`/`videos`/`audios` (→ `ModelType`) for
the raw path alongside the LLM `chat`/`completions`/`embedding`/etc.

## Engine Lifecycle

```
construct -> start(worker_id) -> setup_metrics -> generate/abort -> drain -> cleanup
    |               |                  |                |             |        |
parse args,    start engine,    wire Prometheus    serve requests pre-cleanup shutdown,
return engine  return metadata  (optional)         (concurrent)   drain       release
```

The trait has twelve methods. `from_args` is NOT on the trait — each
backend exposes a backend-specific constructor (typically a sync
`from_args(argv) -> Result<(Self, WorkerConfig)>` inherent method).
This keeps the trait fully object-safe without a `where Self: Sized`
opt-out and lets `run.rs` stay non-generic.

- `start(&self, worker_id: u64) -> Result<EngineConfig, DynamoError>` —
  interior mutability over `&self` so `Arc<dyn LLMEngine>` can drive
  the lifecycle. `worker_id` is an opaque runtime-allocated identifier;
  most engines ignore it. Backends needing a stable cluster-wide key
  (e.g. TRT-LLM's `disagg_machine_id` snowflake) derive from it.
- `setup_metrics(&self, ctx: MetricsCtx<'_>) -> Result<MetricsBindings, DynamoError>` —
  optional, default returns empty `MetricsBindings`. Single hook for
  both foreign-registry expfmt bridging (side effect on `ctx.metrics`)
  AND the engine-side `dynamo_component_*` declaration: returns
  `dp_ranks` (which ranks the engine will publish snapshots for) plus
  an optional `on_publisher_ready` closure the framework invokes with
  the constructed `SnapshotPublisher`. See **KV-aware Routing &
  Component Metrics** below for the push contract.

  Framework-owned lifecycle gauges
  (`dynamo_component_{cleanup_time_seconds,drain_time_seconds,model_load_time_seconds}`)
  are emitted by `Worker` independent of this method. The Worker
  constructs `LifecycleGauges` after `engine.start()` succeeds, seeds
  `model_load_time` with the elapsed `start()` time, and observes
  cleanup/drain during shutdown.
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
- `health_check_payload(&self) -> Result<Option<Value>, DynamoError>` —
  optional, default `Ok(None)`. Canary payload the runtime sends through
  `generate` to actively probe an idle endpoint; `None` disables active
  probing. Operator overrides (`DYN_HEALTH_CHECK_PAYLOAD` / `WorkerConfig`)
  take precedence.
- `supported_controls(&self) -> Result<Vec<String>, DynamoError>` —
  optional, default empty. Semantic engine-control keys this engine
  advertises (e.g. `start_profile`, `sleep`, `wake_up`). The Worker maps
  each onto a `/engine/control/{key}` route via `register_engine_controls`.
- `engine_control(&self, control: String, body: Value) -> Result<Value, DynamoError>` —
  optional, default returns a `status:"error"` body. Dispatches one
  advertised control. Returning a `status:"error"` body is HTTP 200 at the
  `/engine/*` layer (it 5xx's only when this *raises*).
- `supported_updates(&self) -> Result<Vec<String>, DynamoError>` —
  optional, default empty. A sibling surface to `supported_controls` for
  ops that mutate engine-managed assets rather than the serving lifecycle
  (e.g. vLLM dynamic LoRA `load_lora` / `unload_lora` / `list_loras`). Kept
  separate so LoRA doesn't inflate the control surface. The Worker maps
  each onto a `/engine/update/{key}` route via `register_engine_updates`
  (no quiesce/resume policy — updates never toggle discovery).
- `engine_update(&self, update: String, body: Value) -> Result<Value, DynamoError>` —
  optional, default returns a `status:"error"` body. Dispatches one
  advertised update; same HTTP-200-on-`status:"error"` semantics as
  `engine_control`.
- `on_endpoint_ready(&self, endpoint: Endpoint) -> Result<(), DynamoError>` —
  optional, default no-op. The Worker hands the engine its serving
  `Endpoint` exactly once, after it exists and **before**
  `register_engine_controls` / `register_engine_updates` (so `/engine/*`
  can't fire before the engine has the endpoint). A failure is **fatal to
  startup**. Engines that publish their own discovery records stash it
  (e.g. vLLM dynamic LoRA via `register_model` / `unregister_model`).
  Mirrors the `on_publisher_ready` handoff idiom.

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

- **One `Worker`, one engine trait per modality.** `Worker` owns
  runtime lifecycle. `LLMEngine` (token) and `RawEngine` (raw media)
  own inference; they share the same lifecycle methods and `Worker`
  drives both via the `EngineKind` forwarders. No intermediate traits
  or mixins between an engine trait and a concrete backend. A new media
  modality is a new `RawEngine`, not a new trait or a new `EngineKind`
  variant.

  **Why the two traits repeat the lifecycle signatures (no shared
  `LifecycleEngine` supertrait):** Python expresses "shared lifecycle,
  divergent `generate`" via inheritance (`BaseEngine`); Rust uses two sibling
  traits + a closed `EngineKind` enum — idiomatic for a fixed variant set. A
  supertrait would dedupe the six lifecycle signatures but split every engine
  `impl` (PyO3 bridge, mocker, conformance kit, test mocks) in two, and the
  lifecycle set is closed/stable so that edit cost is paid ~never. The
  asymmetry (Python inheritance ↔ Rust enum) is intentional. Mapping: Python
  `BaseEngine` ≡ the lifecycle methods both traits declare; `isinstance`
  routing ≡ the `EngineKind` match.

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

- `Prefill` → register with `ModelType::Prefill` (legacy marker bit, no
  OpenAI surface — dual-emitted for cross-version compat) and
  `WorkerType::Prefill` regardless of `endpoint_types`, so the frontend's
  `PrefillRouter` targets it via `worker_type`.
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

## KV-aware Routing & Component Metrics

Two engine-declared surfaces; both are **push-driven** — engines call into
framework-owned publishers from their natural producer thread. No tokio
poll task, no snapshot-fn latency ceiling, no GIL on the framework side.

- `kv_event_sources() -> Vec<KvEventSource>` — KV cache event
  descriptors, one per data-parallel rank. Drives the router's prefix
  cache.
- `setup_metrics(ctx) -> MetricsBindings` — declares the engine's
  `dp_ranks` and (optionally) an `on_publisher_ready` closure that
  receives the constructed `SnapshotPublisher`.

`Worker` calls both once after `start()` succeeds, constructs the
publishers itself, and hands them back via the engine's `on_ready`
closures. Engines never instantiate publishers. On shutdown, `Worker`
drops the handles while NATS is still alive; engines stop their
producer threads in `cleanup()`.

### `KvEventSource` flavors

Pick based on how the engine's KV event API is shaped:

- `Zmq { endpoint, topic, dp_rank }` — engine already publishes to a
  ZMQ PUB socket. `Worker` subscribes directly.
- `Push { on_ready, dp_rank }` — engine has a programmatic event
  surface. `Worker` constructs the publisher, then calls
  `on_ready(publisher)` once during setup. The engine stashes the
  publisher and drives `publish_stored` / `publish_removed` from its
  own thread. The engine **must** stop that thread in `cleanup()`.

### `SnapshotPublisher` (mirrors `KvEventSource::Push`)

`setup_metrics(ctx)` returns:

```rust
pub struct MetricsBindings {
    pub dp_ranks: Vec<u32>,
    pub on_publisher_ready: Option<OnSnapshotPublisherReady>,
}
```

When `dp_ranks` is non-empty, `Worker`:

1. Constructs `ComponentGauges::new(metrics, &dp_ranks)` — the
   constructor seeds each rank's child gauges at zero so empty
   `GaugeVec` families still render in `/metrics` (the prometheus
   text encoder skips families with no children).
2. Constructs one `WorkerMetricsPublisher` per rank (NATS endpoint
   for the router's `kv_used_blocks` load signal).
3. Wraps both in an `Arc<SnapshotPublisher>` and invokes
   `on_publisher_ready(publisher)`.

The engine stashes the `Arc<SnapshotPublisher>` and calls
`publisher.publish(dp_rank, ComponentSnapshot { … })` from its natural
producer thread (engine iteration callback, stat-logger, ZMQ recv,
etc.) on every tick. `publish` is the hot path: one atomic gauge
write per field plus one NATS publish. No allocation, no GIL.

```rust
impl SnapshotPublisher {
    pub fn publish(&self, dp_rank: u32, snap: ComponentSnapshot);
}
```

Single source of truth for both consumers — the `/metrics` scrape
(gauges) and the KV router (NATS `kv_used_blocks` signal) see the
same `ComponentSnapshot`.

`ComponentSnapshot.kv_cache_hit_rate` is tri-state: `None` means "no
data yet" or "no prefix cache" — the gauge child is not written, so
`/metrics` omits it. `0.0` is a legitimate zero-hit measurement and
DOES write.

### PyO3 bridge

`lib/bindings/python/rust/backend.rs` implements `setup_metrics` for
Python engines. It calls Python's `component_metrics_dp_ranks()` to
get the rank list, then — if non-empty — builds an
`on_publisher_ready` closure that calls Python's
`attach_snapshot_publisher(publisher)` under the GIL. Python engines
expose two methods, mirroring `KvEventSource::Push` shape:

- `component_metrics_dp_ranks() -> list[int]`
- `attach_snapshot_publisher(publisher: SnapshotPublisher) -> None`

Per-rank engine gauges always emit (seeded at construction) when
`dp_ranks` is non-empty — operators see baseline gauge lines even
before the first push.

### Conformance

The kit asserts `kv_event_sources()` and `component_metrics_dp_ranks()`
are idempotent across calls (rank sets stable for the engine's
lifetime) and that `on_publisher_ready` runs at most once per
`setup_metrics` invocation. See `lib/backend-common/src/testing.rs`.

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

## Tracing / OpenTelemetry

Rust engines use the `tracing` crate directly for static-name spans —
spans opened inside `generate()` nest under the framework's `handle_payload`
parent automatically (set up by the runtime at
`lib/runtime/src/pipeline/network/ingress/push_endpoint.rs`). When
`OTEL_EXPORT_ENABLED=1` (see `lib/runtime/src/logging.rs`), these spans export
as OTLP via the `tracing-opentelemetry` layer; otherwise they remain local.

For **dynamic** span names — names computed at runtime, which `tracing::info_span!`
can't handle — use `dynamo_backend_common::telemetry::start_span(name)`. It
goes through OTel directly while still inheriting the bridged parent context;
the returned `SpanGuard` closes on drop. Both paths land in the same trace
tree.

Two patterns worth knowing:

- Prefer `.instrument(span)` on futures / streams over
  `let _g = span.entered();`. The `Entered` guard pins the span to the
  current thread; holding it across an `.await` either fails to compile
  or — under `tokio`'s single-threaded scheduler — leaves the span entered
  on whatever task polls you next.
- `tokio::spawn(fut.in_current_span())` — bare `tokio::spawn` does NOT inherit
  the current span, so logs and events from the spawned task lose the
  trace_id.

For outbound calls that need to carry trace context (HTTP / TCP / custom
transports), use `dynamo_runtime::logging::inject_trace_headers_into_map(...)`
or `get_distributed_tracing_context()`. NATS egress is auto-injected by the
runtime; engines do nothing.

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

For raw media engines, `testing::run_raw_conformance(factory)` is the
`RawEngine` analog. Raw responses are opaque JSON (no `finish_reason`,
no token bookkeeping, no `kv_event_sources`), so it pins the
modality-neutral contract only: `start()` returns a non-empty model and
leaves `EngineConfig.llm` `None`; `generate()` yields a non-empty all-`Ok`
stream; 8 interleaved calls complete; cancellation terminates the stream
within the deadline; and `cleanup()` is idempotent + safe before start.

Also available: `testing::mock_context()` and
`testing::cancelling_context(after)` for hand-written tests.

## Key Files

| File | What it does |
|------|-------------|
| `engine.rs` | `LLMEngine` trait, `EngineConfig`, `GenerateContext`, `MetricsBindings`, `OnSnapshotPublisherReady`, `ComponentSnapshot`, `chunk::token`, `LLMEngineOutputExt` setters, `usage()` helper. Re-exports `PreprocessedRequest` / `LLMEngineOutput` / `FinishReason` / `PrefillResult` / `BootstrapInfo` / etc. |
| `metrics.rs` | `EngineMetrics` (capability handle passed to `setup_metrics` — `add_expfmt_callback` for foreign registries + precomputed `auto_labels` for FFI). `LifecycleGauges` (framework-owned `cleanup_time` / `drain_time` / `model_load_time`). `ComponentGauges` (per-rank `total_blocks` / `gpu_cache_usage_percent` / `kv_cache_hit_rate`; seeded at construction). |
| `snapshot_publisher.rs` | `SnapshotPublisher` — single push surface. `publish(dp_rank, ComponentSnapshot)` fans out inline to `ComponentGauges` and per-rank `WorkerMetricsPublisher`. |
| `publisher.rs` | `setup_publishers` — constructs `KvEventPublisher`s + `SnapshotPublisher` from engine bindings; owned by `Worker` until shutdown. |
| `worker.rs` | `Worker` — runtime lifecycle: create `DistributedRuntime`, register model (with `disaggregation_mode` adjustments), serve endpoint, orchestrate drain + cleanup. `WorkerConfig` lives here. |
| `adapter.rs` | `EngineAdapter` — bridges `LLMEngine` to `AsyncEngine` (token telemetry, disagg first-token, debug validator). `RawEngineAdapter` — bridges `RawEngine` to `AsyncEngine` (JSON passthrough, cancellation monitor; no token telemetry/disagg). `JsonProbeAdapter` — JSON health-check wrapper for the LLM path (the raw path is already JSON-shaped). |
| `run.rs` | `pub fn run(engine, config)` (LLM) and `pub fn run_raw(engine, config)` (raw media) — entry points used by per-backend `main.rs`. Non-generic. |
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
