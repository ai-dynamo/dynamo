# Dynamo Python Backend

> **Work in progress.** The unified backend currently supports minimal
> aggregated inference only. See [Feature Gaps](#feature-gaps) at the bottom
> for what remains to be implemented.

A two-class abstraction that separates **runtime integration** (common across
all backends) from **engine logic** (vLLM, SGLang, TensorRT-LLM, etc.).

## Architecture

```
LLMEngine (ABC)                <-- engine boundary (engine.py)
    |   - from_args(argv) -> (LLMEngine, WorkerConfig)  (factory)
    |   - start() -> EngineConfig        (start engine, return metadata)
    |   - generate(request, context)    (streaming inference)
    |   - abort(context)                (cancel request, optional)
    |   - cleanup()                     (shutdown)
    |
    +-- VllmLLMEngine          <-- vllm/llm_engine.py
    +-- SglangLLMEngine        <-- sglang/llm_engine.py
    +-- TrtllmLLMEngine        <-- trtllm/llm_engine.py
    +-- SampleLLMEngine        <-- sample_engine.py

Worker                  <-- runtime integration (worker.py)
    - receives WorkerConfig from from_args()
    - creates DistributedRuntime
    - sets up endpoints, signal handlers
    - calls engine.start(), registers model
    - serves generate endpoint with cancellation monitoring
    - calls engine.cleanup() on shutdown
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

Each `unified_main.py` calls `run(MyLLMEngine)` from the common
`run.py` module.

## Implementing a New Engine

Subclass `LLMEngine` and implement the required methods:

```python
from dynamo.common.backend import LLMEngine, EngineConfig, WorkerConfig

class MyEngine(LLMEngine):
    @classmethod
    async def from_args(cls, argv=None):
        # Parse CLI args, construct engine and worker_config.
        engine = cls(...)
        worker_config = WorkerConfig(
            namespace="dynamo", component="my-backend", ...
        )
        return engine, worker_config

    async def start(self) -> EngineConfig:
        # Start the engine, return metadata for model registration.
        # After this returns, generate() MUST be ready to accept calls.
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
            "completion_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def abort(self, context):
        # Cancel an in-flight request (optional, default is no-op).
        await my_engine.cancel(context.id())

    async def cleanup(self):
        # Shut down the engine.
        pass
```

Then create an entry point:

```python
# my_backend/unified_main.py
from dynamo.common.backend.run import run
from my_backend.llm_engine import MyEngine

def main():
    run(MyEngine)
```

See `sample_engine.py` for a complete, runnable reference implementation.

## Request / Response Types

`GenerateRequest` and `GenerateChunk` (defined in `engine.py`) are
`TypedDict`s that document the shared fields across all engines.

```python
class GenerateRequest(TypedDict, total=False):
    token_ids: Required[list[int]]
    sampling_options: dict[str, Any]
    stop_conditions: dict[str, Any]
    output_options: dict[str, Any]

class GenerateChunk(TypedDict, total=False):
    token_ids: Required[list[int]]
    finish_reason: str             # final chunk only
    completion_usage: dict[str, int]  # final chunk only
```

Engines may read additional backend-specific keys from the request dict
and write additional keys into response chunks — `TypedDict` does not
reject extra keys at runtime.

Build the `completion_usage` dict inline. Finish reason normalization
(e.g. `"abort"` → `"cancelled"`) is handled by the Rust layer.

## Request Cancellation

`Worker.generate()` automatically monitors for client
disconnections and request cancellations via `context.async_killed_or_stopped()`.
When triggered, it:

1. Calls `engine.abort(context)` to release engine resources (KV cache,
   scheduler slots, etc.)
2. Breaks out of the generation loop
3. Cleans up the monitoring task

Engine implementations should override `abort(context)` to perform
backend-specific cleanup:

| Engine | Abort method | ID used |
|--------|-------------|---------|
| vLLM | `engine_client.abort(request_id)` | `context.id()` |
| SGLang | `tokenizer_manager.abort_request(rid=...)` | `context.trace_id` |
| TRT-LLM | `generation_result.abort()` | Tracked per-request via `context.id()` |
| Sample | *(no-op, default)* | — |

Engines that don't support cancellation can skip overriding `abort()` —
the default implementation is a no-op. The generation loop will still
break on `context.is_stopped()`.

## Error Handling

`Worker` wraps errors in `DynamoException` subclasses from
`dynamo.llm.exceptions` so the Rust bridge can map them to typed
`DynamoError::Backend(...)` responses with proper error chains.

| Phase | Exception raised | When |
|-------|-----------------|------|
| Runtime creation | `CannotConnect` | etcd/NATS unreachable |
| Engine init | `EngineShutdown` | Engine fails to start (OOM, bad config, etc.) |
| Generate | `Unknown` | Untyped exception from engine `generate()` |
| Generate | *(pass-through)* | Engine raises a `DynamoException` subclass directly |

Engine implementations can raise `DynamoException` subclasses directly from
`generate()` for fine-grained error reporting — these propagate unchanged.
Any non-`DynamoException` errors are wrapped as `Unknown`.

Available exception types (from `dynamo.llm.exceptions`):

```python
from dynamo.llm.exceptions import (
    DynamoException,     # Base class
    Unknown,             # Uncategorized error
    InvalidArgument,     # Bad input (e.g., prompt too long)
    CannotConnect,       # Connection failed
    Disconnected,        # Connection lost
    ConnectionTimeout,   # Timeout
    Cancelled,           # Client cancelled
    EngineShutdown,      # Engine crashed or shutting down
    StreamIncomplete,    # Response stream cut short
)
```

## File Index

```
common/backend/
    __init__.py          # Re-exports: LLMEngine, EngineConfig,
                         #   Worker, WorkerConfig
    engine.py            # LLMEngine ABC + EngineConfig dataclass
    worker.py            # Worker + WorkerConfig
    run.py               # Common entry point: run(engine_cls)
    sample_engine.py     # SampleLLMEngine (reference impl)
    sample_main.py       # Entry point for sample engine

vllm/llm_engine.py       # VllmLLMEngine
vllm/unified_main.py     # Entry point -> run(VllmLLMEngine)

sglang/llm_engine.py     # SglangLLMEngine
sglang/unified_main.py   # Entry point -> run(SglangLLMEngine)

trtllm/llm_engine.py     # TrtllmLLMEngine
trtllm/unified_main.py   # Entry point -> run(TrtllmLLMEngine)
```

## Feature Gaps

The unified path currently supports **minimal aggregated inference** only.
Below is a summary of what the existing (non-unified) backends provide that
the unified path does not yet support.

### What works today

- Basic aggregated token-in-token-out inference (all three engines)
- Model registration with endpoint types
- Request cancellation via `abort()` + `context.is_stopped()` monitoring
- `DynamoException` error chain wrapping
- Graceful shutdown with signal handling
- Finish reason normalization handled by Rust layer

### Common gaps (all engines)

| Feature | Description |
|---------|-------------|
| Disaggregated serving | Prefill/decode worker split, bootstrap coordination, KV transfer |
| Metrics & Prometheus | Engine-level metrics, KV cache utilization gauges, Prometheus multiprocess registry |
| KV event publishing | Prefix cache events (BlockStored/Removed) to router via ZMQ or NATS |
| Health check payloads | Per-engine custom health check payloads (BOS token probe, etc.) |
| Logprobs | Selected token + top-k log probability extraction and streaming |
| Guided decoding / structured outputs | JSON schema, regex, grammar, choice constraints |
| OpenTelemetry tracing | `build_trace_headers()`, request performance metrics, OTEL propagation |
| Engine routes | Profiling (start/stop), memory release/resume, weight update (disk/tensor/distributed/IPC) |
| Data-parallel routing | DP rank extraction from routing hints, DP-aware scheduling |
| Text-in-text-out mode | OpenAI-compatible chat/completion with engine-side tokenization |
| Custom Jinja chat templates | `--custom-jinja-template` for model-specific prompt formatting |
| Snapshot/checkpoint | CRIU-based engine state save/restore, identity reloading |

### vLLM-specific gaps

| Feature | Description |
|---------|-------------|
| LoRA adapters | Dynamic load/unload/list, ModelDeploymentCard publishing, per-LoRA serialization locks |
| Multimodal (images/video) | Image/video loading, embedding caching, NIXL RDMA transfer, Qwen VL mRoPE |
| Separate encode worker | `EncodeWorkerHandler` for multimodal encode-only disaggregation |
| Sleep/wake/quiesce | 3-level engine lifecycle control (weights, buffers, everything) |
| Elastic EP scaling | `scale_elastic_ep` with Ray node management |
| GMS shadow mode | GPU Memory Service integration with failover lock |
| ModelExpress P2P | Distributed model loading via P2P |
| KV block clearing | Prefix cache reset endpoint |

### SGLang-specific gaps

| Feature | Description |
|---------|-------------|
| Embedding inference | `async_encode()` path, OpenAI embedding response format |
| Image diffusion | `DiffGenerator` for text-to-image (FLUX, etc.) with TP/DP |
| Video generation | `DiffGenerator` for text-to-video (Wan2.1, etc.) |
| LLM diffusion (DLLM) | Diffusion language model algorithm support |
| Multimodal encode worker | Front-facing `MMEncoder`, embedding LRU cache, NIXL transfer |
| Multimodal worker | Aggregated/disaggregated multimodal inference with `EmbeddingsProcessor` |
| Deferred signal handling | Capturing SGLang's internal signal registrations for coordinated shutdown |
| Output modalities override | Required for diffusion workers (default `["text"]` -> `["image"]`/`["video"]`) |

### TRT-LLM-specific gaps

| Feature | Description |
|---------|-------------|
| Custom logits processors | `TrtllmDynamoLogitsAdapter` with CUDA stream support |
| Attention DP scheduling | `SchedulingParams` with `attention_dp_rank` and `attention_dp_relax` |
| Video diffusion | Auto-detect pipeline from `model_index.json`, MP4 encoding, MediaOutput |
| Multimodal processing | `MultimodalRequestProcessor`, image URL processing, embedding injection |
| Encode helper (EPD) | Remote encode via `encode_client`, NIXL tensor reading |
| KV cache connector | KVBM connector config, consolidator ZMQ integration |
| Fatal vs per-request errors | Distinguishing `RequestError` (recoverable) from fatal engine errors |

### Recommended migration order

1. **Metrics & health checks** -- needed for production observability
2. **Disaggregated serving** -- largest architectural change, unlocks PD split
3. **KV event publishing** -- required for KV-aware routing
4. **Logprobs + guided decoding** -- most-requested inference features
5. **Multimodal / LoRA / diffusion** -- modality-specific, can be parallelized across leads
