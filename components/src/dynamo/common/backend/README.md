# Dynamo Python Backend

A two-class abstraction that separates **runtime integration** (common across
all backends) from **engine logic** (vLLM, SGLang, TensorRT-LLM, etc.).

## Architecture

```
DynamoRuntime
    |
    v
DynamoPythonBackendModel          <-- runtime integration (model.py)
    |   - creates DistributedRuntime
    |   - sets up endpoints, signal handlers
    |   - registers model with Rust frontend
    |   - serves generate endpoint
    |
    v
DynamoEngine (ABC)                <-- engine boundary (engine.py)
    |   - init() -> EngineConfig
    |   - generate(request, context) -> AsyncGenerator[dict]
    |   - cleanup()
    |
    +-- VllmDynamoEngine          <-- vllm/dynamo_engine.py
    +-- SglangDynamoEngine        <-- sglang/dynamo_engine.py
    +-- TrtllmDynamoEngine        <-- trtllm/dynamo_engine.py
    +-- SampleDynamoEngine        <-- sample_engine.py
```

## Quick Start

### Running the sample engine

```bash
python -m dynamo.common.backend.sample_main \
    --model-name test-model \
    --namespace dynamo \
    --component sample \
    --endpoint generate
```

This starts a backend that generates rotating token IDs. Point a frontend at
`dynamo.sample.generate` to test the full request flow without any ML
dependencies.

### Running a real engine

```bash
# vLLM
python -m dynamo.vllm.unified_main --model Qwen/Qwen3-0.6B ...

# SGLang
python -m dynamo.sglang.unified_main --model-path Qwen/Qwen3-0.6B ...

# TensorRT-LLM
python -m dynamo.trtllm.unified_main --model Qwen/Qwen3-0.6B ...
```

Each `unified_main.py` reuses the backend's existing `parse_args()` and maps
its config into `BackendConfig` + the engine-specific `DynamoEngine`
implementation.

## Implementing a New Engine

Subclass `DynamoEngine` and implement three methods:

```python
from dynamo.common.backend import DynamoEngine, EngineConfig
from dynamo.common.engine_utils import build_completion_usage

class MyEngine(DynamoEngine):
    async def init(self) -> EngineConfig:
        # Start the engine, return metadata for model registration.
        return EngineConfig(
            model="my-model",
            context_length=4096,
            kv_cache_block_size=16,
        )

    async def generate(self, request, context):
        # Yield streaming response dicts.
        async for result in my_engine.run(request):
            yield {"token_ids": result.token_ids}
        yield {
            "token_ids": result.token_ids,
            "finish_reason": "stop",
            "completion_usage": build_completion_usage(
                prompt_tokens, completion_tokens
            ),
        }

    async def cleanup(self):
        # Shut down the engine.
        pass
```

Then wire it up:

```python
from dynamo.common.backend import DynamoPythonBackendModel, BackendConfig

engine = MyEngine(...)
config = BackendConfig(namespace="dynamo", component="my-backend", ...)
model = DynamoPythonBackendModel(config, engine)
await model.run()  # handles runtime, registration, serving, shutdown
```

See `sample_engine.py` for a complete, runnable reference implementation.

## Response Format

All engines yield dicts conforming to this contract:

```python
# Intermediate streaming chunks:
{"token_ids": [int, ...]}

# Final chunk (must include finish_reason):
{
    "token_ids": [int, ...],
    "finish_reason": "stop" | "length" | "cancelled" | "error",
    "completion_usage": {
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int,
        "prompt_tokens_details": {"cached_tokens": int} | None,  # optional
    },
}
```

Use `build_completion_usage()` and `normalize_finish_reason()` from
`dynamo.common.engine_utils` to build these.

## Common Engine Utilities

`dynamo.common.engine_utils` provides helpers shared across all engine
implementations:

| Function | Module | Purpose |
|----------|--------|---------|
| `normalize_request_format(request)` | `request.py` | Ensures `stop_conditions`/`sampling_options` dicts exist; moves top-level OpenAI fields into them |
| `build_completion_usage(prompt, completion, details)` | `response.py` | Builds the standard `completion_usage` dict |
| `normalize_finish_reason(reason)` | `response.py` | Maps engine-specific finish reasons (e.g. `"abort"`) to Dynamo values (`"cancelled"`) |

## File Index

```
common/backend/
    __init__.py          # Re-exports: DynamoEngine, EngineConfig,
                         #   DynamoPythonBackendModel, BackendConfig
    engine.py            # DynamoEngine ABC + EngineConfig dataclass
    model.py             # DynamoPythonBackendModel + BackendConfig
    sample_engine.py     # SampleDynamoEngine (reference impl)
    sample_main.py       # Entry point for sample engine

common/engine_utils/
    __init__.py          # Re-exports all utilities
    request.py           # normalize_request_format()
    response.py          # build_completion_usage(), normalize_finish_reason()

vllm/dynamo_engine.py    # VllmDynamoEngine
vllm/unified_main.py     # Entry point

sglang/dynamo_engine.py  # SglangDynamoEngine
sglang/unified_main.py   # Entry point

trtllm/dynamo_engine.py  # TrtllmDynamoEngine
trtllm/unified_main.py   # Entry point
```
