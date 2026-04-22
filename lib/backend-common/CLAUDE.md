# Backend Common (Rust)

Shared runtime glue for Rust LLM backends. Two-type abstraction:
`Worker` (runtime lifecycle) and `LLMEngine` (trait for engine-specific
logic). A reference implementation lives at
`lib/backend-common/examples/rotating_tokens/`.

Engines work directly with `PreprocessedRequest` and `LLMEngineOutput`
— the same types the rest of the Rust pipeline uses. No separate
Python-shaped request/response wrappers.

## Engine Lifecycle

```
construct (e.g. from_args)  ->  start()  ->  generate() / abort()  ->  cleanup()
         |                        |                |                        |
    parse args,            start engine,     serve requests           shutdown,
    return engine          return metadata   (concurrent)             release resources
```

The trait has four methods. `from_args` is NOT on the trait — each
backend exposes a backend-specific constructor (typically a sync
`from_args(argv) -> Result<(Self, WorkerConfig)>` inherent method).
This keeps the trait fully object-safe without a `where Self: Sized`
opt-out and lets `run.rs` stay non-generic.

- `start(&self) -> Result<EngineConfig, BackendError>` — interior mutability
  over `&self` so `Arc<dyn LLMEngine>` can drive the lifecycle.
- `generate(&self, request, ctx) -> Result<BoxStream<'static, LLMEngineOutput>, BackendError>`
  — streaming inference. Author returns a plain stream; the framework wraps
  it in `Annotated` and plumbs cancellation.
- `abort(&self, ctx)` — optional, default no-op. Called by the framework
  ONLY when `ctx.stopped()` or `ctx.killed()` fires during an active
  request — NOT on silent stream drops (TCP reset, consumer-side
  timeout, etc.). For per-request cleanup that must run on ANY drop
  path (releasing a scheduler slot, freeing an engine handle), put the
  release logic inside the `generate` stream body using RAII; use
  `abort` only for out-of-band notifications (e.g. telling a remote
  scheduler to cancel compute early).
- `cleanup(&self) -> Result<(), BackendError>` — called once on shutdown.
  Guaranteed to run if `start()` succeeded, even if later registration or
  serve fails.

## Contract for `generate`

Exactly one **terminal chunk** must be the last item yielded. A terminal
chunk is one with `finish_reason = Some(...)`, and must also set
`completion_usage = Some(...)`. Non-terminal chunks leave both fields
unset. Terminal chunks may carry tokens (the final tokens of the
completion) or be empty — the contract is only about `finish_reason`
and `completion_usage` being set together.

In debug builds, the framework wraps the stream in a validator that
panics on violations — loud failures in dev and test, compiled out in
release.

Rules the validator enforces:

1. A chunk with `finish_reason` must also set `completion_usage`.
2. No chunk may be yielded after a terminal chunk.

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

Build `LLMEngineOutput` values via the `chunk` helpers to cut
boilerplate:

```rust
use dynamo_backend_common::{chunk, usage, FinishReason};

// Non-terminal token chunk
yield chunk::token(id);
// Terminal chunk (finish + usage in one call)
yield chunk::terminal(vec![final_id], FinishReason::Length, usage(prompt_len, n));
// Terminal chunk for cancellation
yield chunk::cancelled(usage(prompt_len, generated));
```

`usage(prompt, completion)` computes `total_tokens` for you.

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
  `Arc<dyn LLMEngine>`. This is load-bearing for phase 2 (PyO3
  bindings): a Python engine will plug in through a `PyLLMEngine`
  adapter that implements the same trait.

- **Typed `BackendError`.** `generate` returns `BackendError`, not
  `anyhow::Error`. Downstream gets predictable HTTP mapping per variant.
  Authors with ad-hoc errors can use `.into()` via the `Other` variant.

## Adding a New Engine

1. Create a new Rust crate depending on `dynamo-backend-common`. Place
   it under `lib/` (e.g. `lib/<backend>-rs/`) following the repo's
   Rust-crate convention. Do **not** place Rust crates under
   `components/src/dynamo/` — that tree is the Python package
   namespace.
2. In `src/<backend>_engine.rs`: `struct YourEngine; impl LLMEngine for YourEngine`.
   Plus an inherent `impl YourEngine { pub fn from_args(argv) -> Result<(Self, WorkerConfig), BackendError> }`.
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
5. Use `engine.rs` from `lib/backend-common/examples/rotating_tokens/`
   as a template.
6. Run the conformance test kit (see Testing below) against your engine.

## Error Handling

`Worker` wraps lifecycle errors in `BackendError` variants. Engines
return `BackendError` directly from `start`, `generate`, and `cleanup`.

Mid-stream, non-fatal errors are signalled via a terminal
`LLMEngineOutput` with `finish_reason = Some(FinishReason::Error(msg))`
plus a `completion_usage`. Construct this by hand; there is no
`chunk::error` helper (the helper set is deliberately small).

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
    let engine = MyEngine::new_for_test();
    testing::run_conformance(engine).await.expect("conformance");
}
```

The kit asserts:

- `start()` returns a non-empty `EngineConfig.model`.
- A single `generate()` yields a well-formed stream ending in a
  terminal chunk (`finish_reason` + `completion_usage` both set).
- 8 interleaved `generate()` calls all complete successfully
  (catches shared-state bugs under concurrent polling).
- After `stop_generating()` fires mid-stream, the stream terminates
  within a 2s deadline (else `CancellationNotObserved`). If the last
  chunk yielded is not a `FinishReason::Cancelled` terminal — any
  other terminal reason, or no terminal at all — the check raises
  `ConformanceFailure::CancellationIgnored`.
- `cleanup()` succeeds and is idempotent (two calls in a row both Ok).

Also available: `testing::mock_context()` and
`testing::cancelling_context(after)` for hand-written tests.

## Key Files

| File | What it does |
|------|-------------|
| `engine.rs` | `LLMEngine` trait, `EngineConfig`, `chunk::*` + `usage()` helpers. Re-exports `PreprocessedRequest` / `LLMEngineOutput` / `FinishReason` / etc. |
| `worker.rs` | `Worker` — runtime lifecycle: create `DistributedRuntime`, register model, serve endpoint, cleanup. `WorkerConfig` lives here. |
| `adapter.rs` | `EngineAdapter` — bridges `LLMEngine` to `AsyncEngine`. Cancellation monitor + debug-build validator wrapping. |
| `run.rs` | `pub fn run(engine, config)` — entry point used by all per-backend `main.rs`. Non-generic. |
| `args.rs` | `CommonArgs` — shared CLI flags (`--namespace`, `--component`, etc.) that every engine's `Args` flattens in. |
| `error.rs` | `BackendError` — typed error enum with HTTP-friendly variants. |
| `validate.rs` | Debug-build stream validator. Compiled out in release. |
| `testing.rs` | Conformance test kit. Gated behind the `testing` feature. |

## Phase 2

PyO3 bindings are planned that let the Python backend runtime become
a thin wrapper over this crate. The trait and data types are designed
to support that without restructuring — do not pre-build phase-2
scaffolding here.
