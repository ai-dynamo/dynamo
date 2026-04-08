# Backend Module

Two-class abstraction: `DynamoPythonBackendModel` (runtime integration) and
`DynamoEngine` (ABC for engine-specific logic). See `README.md` for full docs.

## Design Constraints

- **ZERO duplication across engine implementations.** This is the #1 priority.
  The entire reason this module exists is to eliminate the code duplication
  that grew across vllm, sglang, and trtllm. Before writing any logic inside
  a `DynamoEngine` subclass, check whether the same logic already exists in
  another engine. If it does, extract it into `common/engine_utils/` or
  `DynamoPythonBackendModel` and have all engines call the shared version.
  When adding new features, always ask: "is this engine-specific or common?"
  If two or more engines would need the same code, it is common.

- **Exactly two classes.** `DynamoPythonBackendModel` owns runtime lifecycle.
  `DynamoEngine` owns inference. Do not add intermediate base classes or mixins.

- **`generate()` is a thin pass-through.** `DynamoPythonBackendModel.generate()`
  delegates directly to `DynamoEngine.generate()`. Common pre/post-processing
  may be added here later, but sampling params, prompt building, and output
  formatting stay inside each engine -- they are deeply engine-specific.

- **`init()` returns `EngineConfig`.** The model class needs registration
  metadata (`context_length`, `block_size`, `total_kv_blocks`) but must not
  reach into engine internals. `init()` returns this metadata so the boundary
  stays clean.

- **No hooks.** If behavior needs to be shared across engines, put it in
  `DynamoPythonBackendModel` or `common/engine_utils/`, not in a hook system.

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
2. Create `<backend>/unified_main.py` wiring `BackendConfig` + your engine
3. Reuse the backend's existing `parse_args()`
4. Use `sample_engine.py` as the reference implementation

## What Does NOT Belong Here

- Engine-specific sampling param classes (keep in each engine module)
- Disaggregated serving, multimodal, LoRA, diffusion (added by engine leads)
- Metrics/Prometheus setup (will be added to `DynamoPythonBackendModel` later)
- Health check payloads, sleep/wake engine routes (engine-specific, added later)

## Key Files

| File | What it does |
|------|-------------|
| `engine.py` | `DynamoEngine` ABC -- the only interface engines must implement |
| `model.py` | `DynamoPythonBackendModel` -- runtime lifecycle: create runtime, register model, serve endpoint, cleanup |
| `sample_engine.py` | Reference engine -- use as template and for testing |
| `../engine_utils/request.py` | `normalize_request_format()` -- call this at the top of `generate()` if your engine receives both OpenAI and internal protocol formats |
| `../engine_utils/response.py` | `build_completion_usage()`, `normalize_finish_reason()` -- use these to build response dicts |
