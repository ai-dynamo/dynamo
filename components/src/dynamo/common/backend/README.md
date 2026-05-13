# Dynamo Python Backend

**Supported today:** aggregated inference, disaggregated
(prefill/decode) serving with bootstrap (SGLang) or internal KV
transport (vLLM, TRT-LLM), request cancellation, graceful shutdown,
`drain()` hook, error chain wrapping.

> **Work in progress.** Multimodal, LoRA, logprobs, guided decoding,
> and engine-level metrics are still on the non-unified path. See
> [Feature Gaps](#feature-gaps) for the full matrix.

> **Looking for a walkthrough?** Start with the
> [Writing a Python Unified Backend](../../../../../docs/development/python-backend-guide.md)
> guide. This README is the in-tree reference: file layout, per-engine
> cancellation cookbook, disaggregation contract, error-handling table,
> and the feature-gap matrix.

A two-class abstraction that separates **runtime integration** (common across
all backends) from **engine logic** (vLLM, SGLang, TensorRT-LLM, etc.).

## Architecture

```text
LLMEngine (ABC)                <-- engine boundary (engine.py)
    |   - from_args(argv) -> (LLMEngine, WorkerConfig)  (factory)
    |   - start(worker_id) -> EngineConfig    (start engine, return metadata)
    |   - generate(request, context)         (streaming inference)
    |   - abort(context)                     (cancel request, optional)
    |   - drain()                            (pre-cleanup drain, optional)
    |   - cleanup()                          (shutdown)
    |
    +-- VllmLLMEngine          <-- vllm/llm_engine.py
    +-- SglangLLMEngine        <-- sglang/llm_engine.py
    +-- TrtllmLLMEngine        <-- trtllm/llm_engine.py
    +-- SampleLLMEngine        <-- sample_engine.py

Worker                  <-- runtime integration (worker.py)
    - receives WorkerConfig from from_args()
    - creates DistributedRuntime
    - sets up endpoints, signal handlers
    - calls engine.start(worker_id), registers model
    - serves generate endpoint with cancellation monitoring
    - calls engine.drain() then engine.cleanup() on shutdown
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

    async def start(self, worker_id: int) -> EngineConfig:
        # Start the engine, return metadata for model registration.
        # After this returns, generate() MUST be ready to accept calls.
        # `worker_id` is an opaque per-worker key; most engines ignore it.
        return EngineConfig(
            model="my-model",
            context_length=4096,
            kv_cache_block_size=16,
            # Populate `bootstrap_host` / `bootstrap_port` here on prefill
            # workers that advertise a Dynamo-level handshake address.
        )

    async def generate(self, request, context):
        # Yield streaming response dicts.
        async for result in my_engine.run(request):
            yield {"token_ids": result.token_ids, "index": 0}
        yield {
            "token_ids": result.token_ids,
            "index": 0,
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
    index: Required[int]           # choice index; use 0 for single-choice chunks
    finish_reason: str             # final chunk only
    completion_usage: dict[str, int]  # final chunk only
```

Engines may read additional backend-specific keys from the request dict
and write backend-specific keys into response chunks if the shared contract
is extended here first.

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

## Disaggregated Serving

The unified path supports the canonical PD-disagg roles via a single
`--disaggregation-mode` flag. The mode flows from CLI → `WorkerConfig` →
the Rust `Worker`, which uses it to decide model registration
(`ModelType::Prefill` for prefill workers, the parsed `endpoint_types`
for everyone else) and to disable the local KV indexer on decode
workers. Engines read the same field on their runtime config to switch
per-mode behavior in `generate()`.

```text
+-----------+   --disaggregation-mode prefill    +------------------+
|  CLI args |  ------------------------------->  |  WorkerConfig    |
+-----------+                                    +------------------+
                                                          |
                                                          v
                                          ModelType::Prefill registration
                                          (Rust Worker)

                                                          |
                                                          v
                                          generate(): build context_only
                                          handoff payload → terminal carries
                                          disaggregated_params (engine-specific)
```

Each backend's protocol is different:

| Backend | Prefill | Decode |
|---------|---------|--------|
| **vLLM** | Sets `kv_transfer_params.do_remote_decode=True`, caps `max_tokens=1`, packs the connector's transfer handle into the response. | Pulls `kv_transfer_params` from `request.prefill_result` and feeds it back through `sampling_params.extra_args` so the `NixlConnector` imports KV. |
| **SGLang** | Yields `{bootstrap_host, bootstrap_port, bootstrap_room}` as the first chunk, then drains the engine stream silently. Warmup happens in `start()`. | Reads bootstrap info from `request.prefill_result`, passes it to `engine.async_generate` so SGLang's NIXL transport pulls KV. |
| **TRT-LLM** | Builds `LlmDisaggregatedParams(request_type="context_only")`, generates one token, packs the encoded handoff into the response. `drain()` polls the scheduler until idle so in-flight NIXL transfers finish before GPU memory is freed (issue #7319). | Decodes `request.prefill_result.disaggregated_params`, flips `request_type` to `generation_only`, generates normally. |

### Smoke testing without GPUs

The sample backend implements the full disagg dispatch in pure Python
with synthetic handoff payloads — no real KV transfer, but the wire
format is exercised end-to-end. This makes it a fast CI smoke test for
the unified path:

```bash
examples/backends/sample/launch/disagg.sh
```

Spawns the frontend plus a sample prefill worker and a sample decode
worker; the frontend's `PrefillRouter` forwards the synthetic
`disaggregated_params` from prefill to decode.

### Switching production backends to the unified path

Each backend's `disagg.sh` accepts `--unified` to swap in the unified
entry point. With it, the launch script exercises the same disagg flow
through `dynamo.<backend>.unified_main` instead of the legacy
`dynamo.<backend>` dispatch:

```bash
examples/backends/vllm/launch/disagg.sh --unified
examples/backends/sglang/launch/disagg.sh --unified
examples/backends/trtllm/launch/disagg.sh --unified
```

### Helpers

`dynamo.common.backend.disagg` ships small utilities engines can call
directly: `enforce_prefill_max_tokens(request)`,
`extract_prefill_result(request)`, and
`require_prefill_result(request, mode)`. These are optional — engines
are free to inline the logic when their generate path is shaped
differently.

## File Index

```text
common/backend/
    __init__.py          # Re-exports: LLMEngine, EngineConfig,
                         #   GenerateChunk, GenerateRequest,
                         #   Worker, WorkerConfig
    engine.py            # LLMEngine ABC + EngineConfig dataclass +
                         #   GenerateRequest/GenerateChunk TypedDicts
    worker.py            # Worker + WorkerConfig (incl. disaggregation_mode)
    disagg.py            # Disagg request helpers (prefill clamp,
                         #   prefill_result extraction)
    run.py               # Common entry point: run(engine_cls)
    sample_engine.py     # SampleLLMEngine (reference impl)
    sample_main.py       # Entry point for sample engine
    tests/               # test_backend_bindings, test_disagg_helpers,
                         #   test_sample_engine
    CLAUDE.md            # Design notes (rationale, invariants)

vllm/llm_engine.py       # VllmLLMEngine (agg + disagg)
vllm/unified_main.py     # Entry point -> run(VllmLLMEngine)

sglang/llm_engine.py     # SglangLLMEngine (agg + disagg, bootstrap handshake)
sglang/unified_main.py   # Entry point -> run(SglangLLMEngine)

trtllm/llm_engine.py     # TrtllmLLMEngine (agg + disagg)
trtllm/unified_main.py   # Entry point -> run(TrtllmLLMEngine)
```

## Feature Gaps

Below is a summary of what the existing (non-unified) backends provide
that the unified path does not yet support.

### What works today

- Basic aggregated token-in-token-out inference (all three engines)
- Model registration with endpoint types
- Request cancellation via `abort()` + `context.is_stopped()` monitoring
- `DynamoException` error chain wrapping
- Graceful shutdown with signal handling
- Finish reason normalization handled by Rust layer
- **Disaggregated serving** (`agg`/`prefill`/`decode`) — see
  [Disaggregated Serving](#disaggregated-serving) below

### Common gaps (all engines)

| Feature | Description |
|---------|-------------|
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
| LoRA adapters | Dynamic load/unload/list, ModelDeploymentCard publishing, per-LoRA serialization locks. Also: unified prefill does not currently thread per-request LoRA adapters into the engine call. |
| Multimodal (images/video) | Image/video loading, embedding caching, NIXL RDMA transfer, Qwen VL mRoPE. Also: unified prefill does not pack `embedding_params` into the response's `disaggregated_params` — disagg multimodal flows still need the legacy path. |
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
