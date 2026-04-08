# Backend Module

Two-class abstraction: `DynamoBackend` (runtime integration) and
`DynamoEngine` (ABC for engine-specific logic). See `README.md` for full docs.

## Design Constraints

- **ZERO duplication across engine implementations.** This is the #1 priority.
  The entire reason this module exists is to eliminate the code duplication
  that grew across vllm, sglang, and trtllm. Before writing any logic inside
  a `DynamoEngine` subclass, check whether the same logic already exists in
  another engine. If it does, extract it into `common/engine_utils/` or
  `DynamoBackend` and have all engines call the shared version.
  When adding new features, always ask: "is this engine-specific or common?"
  If two or more engines would need the same code, it is common.

- **Exactly two classes.** `DynamoBackend` owns runtime lifecycle.
  `DynamoEngine` owns inference. Do not add intermediate base classes or mixins.

- **`generate()` delegates to engine with cancellation monitoring.**
  `DynamoBackend.generate()` runs a background task that watches
  `context.async_killed_or_stopped()` and calls `engine.abort(context)` on
  cancellation. It also checks `context.is_stopped()` after each yielded
  chunk. Sampling params, prompt building, and output formatting stay inside
  each engine -- they are deeply engine-specific.

- **`init()` returns `EngineConfig`.** The model class needs registration
  metadata (`context_length`, `block_size`, `total_kv_blocks`) but must not
  reach into engine internals. `init()` returns this metadata so the boundary
  stays clean.

- **No hooks.** If behavior needs to be shared across engines, put it in
  `DynamoBackend` or `common/engine_utils/`, not in a hook system.

- **Parallel path.** The existing `main.py` / `worker_factory.py` / `init_llm.py`
  entry points remain untouched. The `unified_main.py` files are a separate
  path. Do not break or modify existing backends when changing this module.

## Response Contract

Every `DynamoEngine.generate()` must yield dicts with:

- `token_ids: list[int]` -- present on every chunk
- `finish_reason: str` -- present only on the final chunk
- `completion_usage: dict` -- present only on the final chunk

Use `build_completion_usage()` and `normalize_finish_reason()` from
`dynamo.common.engine_utils` instead of building these inline.

## Adding a New Engine

1. Create `<backend>/dynamo_engine.py` subclassing `DynamoEngine`
2. Implement `init()`, `generate()`, `cleanup()` (required) and `abort()` (optional)
3. Create `<backend>/unified_main.py` wiring `BackendConfig` + your engine
4. Reuse the backend's existing `parse_args()`
5. Use `sample_engine.py` as the reference implementation

## Error Handling

`DynamoBackend` wraps lifecycle and generate errors in
`DynamoException` subclasses (`dynamo.llm.exceptions`). The Rust bridge
(`engine.rs`) converts these into typed `DynamoError::Backend(...)` for
proper error chain observability. Engines can raise `DynamoException`
subclasses directly from `generate()` -- these pass through unchanged.
Non-`DynamoException` errors are wrapped as `Unknown`.

## Logging

Keep logging **standardized across all three engines** (vllm, sglang, trtllm).
When adding or changing a log message in one `dynamo_engine.py`, check
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
| `engine.py` | `DynamoEngine` ABC -- the only interface engines must implement |
| `model.py` | `DynamoBackend` -- runtime lifecycle: create runtime, register model, serve endpoint, cleanup |
| `sample_engine.py` | Reference engine -- use as template and for testing |
| `../engine_utils/request.py` | `normalize_request_format()` -- call this at the top of `generate()` if your engine receives both OpenAI and internal protocol formats |
| `../engine_utils/response.py` | `build_completion_usage()`, `normalize_finish_reason()` -- use these to build response dicts |
