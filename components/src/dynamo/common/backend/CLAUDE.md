# Backend Module

Two-class abstraction: `Worker` (runtime integration) and
`LLMEngine` (ABC for engine-specific logic). See `README.md` for full docs.

## Engine Lifecycle

```
from_args(argv)  ->  start()  ->  generate() / abort()  ->  cleanup()
     |                  |              |                        |
  parse args,      start engine,    serve requests         shutdown,
  return config    return metadata  (concurrent)           release resources
```

1. `from_args(argv)` -- classmethod factory. Parses CLI args, returns
   `(engine, WorkerConfig)`. Engine is NOT started yet.
2. `start()` -- starts the engine, returns `EngineConfig`. After this returns
   `generate()` MUST be ready to accept calls.
3. `generate(request, context)` -- streaming inference, called concurrently.
4. `abort(context)` -- cancel an in-flight request (optional, default no-op).
5. `cleanup()` -- called once on shutdown.

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
  `Worker.generate()` runs a background task that watches
  `context.async_killed_or_stopped()` and calls `engine.abort(context)` on
  cancellation. It also checks `context.is_stopped()` after each yielded
  chunk. Sampling params, prompt building, and output formatting stay inside
  each engine -- they are deeply engine-specific.

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
| `worker.py` | `Worker` -- runtime lifecycle: create runtime, register model, serve endpoint, cleanup |
| `run.py` | Common entry point -- `run(engine_cls)` used by all `unified_main.py` files |
| `sample_engine.py` | Reference engine -- use as template and for testing |
