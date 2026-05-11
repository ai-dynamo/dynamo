# Backend Module

Two-class abstraction: `Worker` (runtime integration) and
`LLMEngine` (ABC for engine-specific logic). See `README.md` for full docs.

## Engine Lifecycle

```
from_args(argv) -> start() -> generate()/abort() -> drain() -> cleanup()
     |                |              |                  |          |
  parse args,    start engine,   serve requests    drain in-flight, shutdown,
  return config  return metadata (concurrent)      then cleanup    release resources
```

1. `from_args(argv)` -- classmethod factory. Parses CLI args, returns
   `(engine, WorkerConfig)`. Engine is NOT started yet.
2. `start()` -- starts the engine, returns `EngineConfig`. After this returns
   `generate()` MUST be ready to accept calls.
3. `generate(request, context)` -- streaming inference, called concurrently.
4. `abort(context)` -- cancel an in-flight request (optional, default no-op).
5. `drain()` -- backend-side drain before cleanup (optional, default no-op).
   Called after the discovery unregister + grace period; use it for in-flight
   NIXL transfers (issue #7319) that must complete while the runtime is alive.
6. `cleanup()` -- called exactly once. Runs after `start()` succeeded
   on shutdown, **and** after `start()` raised — so implementations
   must be null-safe against partial state (inner engine handle,
   sockets, background tasks). All current engines guard each
   resource with `if self.engine is not None`. Must also be
   idempotent: a second call after a successful first is a no-op.
   Not called when `start()` was never invoked (e.g. pre-start
   shutdown); use `__del__` for resources allocated in the
   constructor.

## Design Constraints

- **ZERO duplication across engine implementations.** This is the #1 priority.
  The entire reason this module exists is to eliminate the code duplication
  that grew across vllm, sglang, and trtllm. Before writing any logic inside
  a `LLMEngine` subclass, check whether the same logic already exists in
  another engine. If it does, extract it into `Worker` or a shared
  utility and have all engines call the shared version.
  When adding new features, always ask: "is this engine-specific or common?"
  If two or more engines would need the same code, it is common.

- **Exactly two classes.** `Worker` owns runtime lifecycle.
  `LLMEngine` owns inference. Do not add intermediate base classes or mixins.

- **`from_args()` returns `(engine, WorkerConfig)`.**  The tuple return
  makes the contract statically checkable -- a subclass that forgets to
  build a `WorkerConfig` is a type error, not a runtime `AttributeError`.

- **`generate()` delegates to engine with cancellation monitoring.**
  Cancellation monitoring lives in Rust (`dynamo_backend_common::EngineAdapter`):
  it spawns a per-request task that watches `ctx.stopped()` / `ctx.killed()`
  and calls `engine.abort(context)` on cancellation. The Python ABC's
  `generate()` just yields chunks — the cross-stream cancel logic and
  exception → `BackendError` mapping are handled by the bridge.

- **`start()` returns `EngineConfig`.** The model class needs registration
  metadata (`context_length`, `block_size`, `total_kv_blocks`) but must not
  reach into engine internals. `start()` returns this metadata so the boundary
  stays clean.

- **No hooks.** If behavior needs to be shared across engines, put it in
  `Worker` or a shared utility, not in a hook system.

- **Parallel path.** The existing `main.py` / `worker_factory.py` / `init_llm.py`
  entry points remain untouched. The `unified_main.py` files are a separate
  path. Do not break or modify existing backends when changing this module.

## Request / Response Contract

`GenerateRequest` and `GenerateChunk` (`engine.py`) are `TypedDict`s that
type the `generate()` signature.  `GenerateRequest` has `token_ids`
(required) plus optional `sampling_options`, `stop_conditions`, and
`output_options`.  `GenerateChunk` has `token_ids` and `index` (both
required; use `index=0` for single-choice chunks), plus optional
`finish_reason` and `completion_usage` (both required on the final chunk).
Engines may read
backend-specific request keys, but response chunk keys should be added to
the shared contract before use.

Build the `completion_usage` dict inline. Finish reason normalization
(e.g. `"abort"` → `"cancelled"`) is handled by the Rust layer.

## Adding a New Engine

1. Create `<backend>/llm_engine.py` subclassing `LLMEngine`
2. Implement `from_args()`, `start()`, `generate()`, `cleanup()` (required)
   and `abort()` (optional)
3. `from_args()` must parse args and return `(engine, WorkerConfig)`
4. Create `<backend>/unified_main.py` calling `run(<YourEngine>)`
5. Use `sample_engine.py` as the reference implementation

## Disaggregated Serving

`WorkerConfig.disaggregation_mode` is the single source of truth for the
worker's role. Default `DisaggregationMode.AGGREGATED` keeps existing
callers unchanged.

What the **runtime** does with the mode (Rust `Worker` in `lib/backend-common`):

- `Prefill` → register with `ModelType::Prefill` so the frontend's
  `PrefillRouter` targets this worker, regardless of `endpoint_types`.
- `Decode` → keep `endpoint_types`, but force-disable
  `enable_local_indexer` (decode workers don't host the indexer endpoint).
- `Aggregated` → register with the parsed `endpoint_types`.

What the **engine** does with the mode (consumed in each backend's
`generate()`):

- `Prefill`: cap output to one token, run the engine through its
  prefill-only path, pack the resulting handoff payload (vLLM's
  `kv_transfer_params`, SGLang's bootstrap triple, TRT-LLM's encoded
  `LlmDisaggregatedParams`) into the terminal chunk's
  `disaggregated_params`.
- `Decode`: read `request.prefill_result.disaggregated_params`, fail
  loudly if missing (`require_prefill_result`), feed it into the
  engine's resume-from-KV-transfer call.
- `Aggregated`: existing path, no branching.

`drain()` is the prefill-shutdown hook: prefill engines should poll
their scheduler until in-flight NIXL transfers finish before GPU
memory is released (issue #7319). Aggregated/decode engines leave the
default no-op.

`disagg.py` ships `enforce_prefill_max_tokens`, `extract_prefill_result`,
and `require_prefill_result` — small helpers backends call from inside
`generate()` to avoid reinventing the same patterns. They are utilities,
not abstractions; backends free to inline the logic when their generate
path is shaped differently.

The sample engine (`sample_engine.py`) implements the full dispatch in
pure Python with synthetic handoff payloads — useful as a reference
when wiring a new backend, and as a CPU-only smoke test for the
disaggregated wire format (`examples/backends/sample/launch/disagg.sh`).

## Error Handling

`Worker` wraps lifecycle and generate errors in
`DynamoException` subclasses (`dynamo.llm.exceptions`). The Rust bridge
(`engine.rs`) converts these into typed `DynamoError::Backend(...)` for
proper error chain observability. Engines can raise `DynamoException`
subclasses directly from `generate()` -- these pass through unchanged.
Non-`DynamoException` errors are wrapped as `Unknown`.

## Logging

Keep logging **standardized across all three engines** (vllm, sglang, trtllm).
When adding or changing a log message in one `llm_engine.py`, check
whether the same lifecycle event is logged in the other two and update them
to match. The goal is that operators see the same log shape regardless of
backend, making it easier to triage issues across mixed deployments.

Standardize on:
- `logger.info` for lifecycle milestones: engine init complete, serving
  started, engine shutdown.
- `logger.debug` for per-request events: request abort, cancellation.
- `logger.warning` for recoverable problems: empty outputs, unexpected
  finish reasons.
- `logger.error` only for unrecoverable failures.

## Key Files

| File | What it does |
|------|-------------|
| `engine.py` | `LLMEngine` ABC -- the only interface engines must implement |
| `worker.py` | `Worker` -- thin shim over `dynamo._core.backend.Worker`; lifecycle state machine and signal handling live in Rust (`lib/backend-common`) |
| `run.py` | Common entry point -- `run(engine_cls)` used by all `unified_main.py` files |
| `sample_engine.py` | Reference engine -- use as template and for testing |

The Rust `Worker` (in `lib/backend-common/src/worker.rs`) owns:
  - Lifecycle state machine (Init → Running → Stopped)
  - SIGTERM/SIGINT handling and graceful shutdown orchestration
    (discovery unregister → grace period → drain → cleanup)
  - 3-phase distributed runtime shutdown after engine.cleanup() returns

State-machine and orchestrator invariants are pinned by Rust unit tests
in the same crate. Don't reimplement them on the Python side.
