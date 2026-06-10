# Dynamo Python Backend

**Supported today:** aggregated and disaggregated (prefill/decode)
inference, metrics + Prometheus bridging, KV event publishing,
KV-aware (DP-rank) routing, health-check canaries, OpenTelemetry
tracing, and request-side guided decoding / structural tag.

> **Work in progress.** Multimodal, diffusion (image/video/DLLM), LoRA,
> engine routes (pause/resume, profiling, weight updates),
> text-in-text-out, and snapshot/CRIU are still on the non-unified
> path. See [Feature Gaps](#feature-gaps) for the per-engine matrix.

> **Looking for a walkthrough?** Start with the
> [Writing Unified Backends](../../../../../docs/development/unified-backends.md)
> guide and choose the Python tab. This README is the in-tree reference:
> file layout, per-engine cancellation cookbook, disaggregation contract,
> error-handling table, and the feature-gap matrix.

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
(`ModelType::empty()` + `WorkerType::Prefill` for prefill workers, the
parsed `endpoint_types` for everyone else) and to disable the local KV
indexer on decode workers. Engines read the same field on their runtime
config to switch per-mode behavior in `generate()`.

```text
+-----------+   --disaggregation-mode prefill    +------------------+
|  CLI args |  ------------------------------->  |  WorkerConfig    |
+-----------+                                    +------------------+
                                                          |
                                                          v
                                          WorkerType::Prefill registration
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

## Metrics

Two surfaces:

1. **`dynamo_component_*` gauges + router-input signal** — engines declare
   their DP rank shape via `component_metrics_dp_ranks()`. The framework
   constructs a Rust-owned `SnapshotPublisher` and hands it back through
   `attach_snapshot_publisher(publisher)`. Engine code calls
   `publisher.publish(dp_rank, ComponentSnapshot(...))` from its natural
   push surface (stat-logger / ZMQ recv / poll thread) — event-driven,
   no framework poll loop, no GIL on the gauge write path.
2. **Vendor-prefixed metrics** (`vllm:`, `sglang:`, `trtllm_`,
   `lmcache:`) — engines bridge their own
   `prometheus_client.CollectorRegistry` into the runtime's combined
   `/metrics` output via `register_prometheus(metrics)` using
   `register_global_registry` (or `register_engine_registry` for a
   private registry).

`ComponentSnapshot.kv_cache_hit_rate` is tri-state: `None` means "no data
yet" or "no prefix cache" (gauge skipped); `0.0` is a legitimate
zero-hit measurement.

## Telemetry

> **Requires `DYN_LOGGING_JSONL=1` + `OTEL_EXPORT_ENABLED=1`** for engine
> telemetry to record anything. In any other configuration the calls
> silently no-op; one process-level `WARN` fires on first such call so the
> misconfiguration is visible at default log levels. Trace propagation
> (`context.trace_headers()`) and the auto-recorded `engine.generate`
> attributes are NOT subject to this gate — they work regardless.

The framework opens an `engine.generate` span around every `generate()` call
(see the Rust backend-common README for the full attribute table). Engine
code reaches the recording surface through the
`dynamo.common.backend.telemetry` facade, which mirrors the OpenTelemetry
`Span` API — no Dynamo-specific vocabulary:

```python
from dynamo.common.backend import telemetry

async def generate(self, request, context):
    # Trace headers for the downstream inference engine (W3C traceparent).
    trace_headers = context.trace_headers()
    ...

    # Handle on the framework's engine.generate span. Use it to add
    # attributes, events, or set status. Any attribute key is accepted.
    span = telemetry.current_span(context)
    span.set_attribute("kv_cache_hit_blocks", 8)
    span.add_event("nixl_transfer_complete", {"bytes": 1048576})

    # Open a child span with a dynamic name (real OTel span — renders as
    # a distinct node in Tempo / Jaeger flame charts).
    with telemetry.start_span(context, "tokenize", batch_size=8) as s:
        tokens = self.tokenizer.encode(prompt)
        s.add_event("encoder_warmup_complete")
        s.set_attribute("token_count", len(tokens))

    # On error paths, mark the auto-span as failed (Tempo/Jaeger render
    # this natively).
    if failed:
        span.set_status("error", "kv_transfer_timeout")
```

Two entry points, one `SpanProxy` returned by both:

- `telemetry.current_span(context)` — handle on the auto-span. Not a context
  manager (the framework owns lifecycle). Use freely.
- `telemetry.start_span(context, name, **attrs)` — opens a child span; use
  with `with` so the span ends on exit.

`SpanProxy` methods: `set_attribute(key, value)`, `add_event(name, attrs)`,
`set_status(status, description)`, `close()`.

**Bridge dependency.** The recording surface needs the
`tracing-opentelemetry` layer installed, which today happens only when
`DYN_LOGGING_JSONL=1` AND `OTEL_EXPORT_ENABLED=1`. Without the bridge:

- `current_span(...)` returns a no-op `SpanProxy` (all method calls silent).
- `start_span(...)` returns a no-op `SpanProxy`.
- A `tracing::warn!` fires once per process the first time any of these
  hit the missing-bridge path, so operators can discover the missing
  configuration. Subsequent no-ops in the same process are silent.

Trace propagation (`context.trace_headers()`) and the `Context` cancellation
/ identity surface do NOT depend on the bridge — those work regardless of
mode.

Performance note: attribute values are rendered via Python `repr()` for
non-primitive types. Don't pass large objects per-token inside hot loops;
record summary attributes instead.

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
    logprobs.py          # Shared logprob helpers
                         #   (vLLM/TRT-LLM extractor, SGLang variant,
                         #   option parsing, SGLang gate)
    metrics.py           # Prometheus helpers (gather_with_labels,
                         #   ensure_prometheus_multiproc_dir, registration)
    publisher.py         # ComponentSnapshot dataclass (push payload)
    run.py               # Common entry point: run(engine_cls)
    sample_engine.py     # SampleLLMEngine (reference impl)
    sample_main.py       # Entry point for sample engine
    tests/               # test_backend_bindings, test_disagg_helpers,
                         #   test_logprobs, test_sample_engine
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

Lifecycle and runtime:
- Aggregated token-in-token-out inference (all three engines)
- Model registration with endpoint types
- Request cancellation via `abort()` + `context.is_stopped()` monitoring
- Graceful shutdown with signal handling
- `drain()` hook for pre-cleanup work
- `DynamoException` error chain wrapping
- Finish reason normalization handled by the Rust layer
- Engine control plumbing, with per-backend profiling, pause/resume, and supported weight-update controls
- **Disaggregated serving** (`agg`/`prefill`/`decode`) — KV transfer
  uses NIXL across all three engines; SGLang exchanges a Dynamo-level
  bootstrap address, vLLM and TRT-LLM use an engine-internal handshake.
  See [Disaggregated Serving](#disaggregated-serving) below.
- **Logprobs** — selected-token + top-k logprob extraction and
  streaming, sourced from `dynamo.common.backend.logprobs` and used by
  both unified engines and the legacy handlers (which now delegate
  here). vLLM/TRT-LLM share an extractor; SGLang has a cumulative-array
  variant. The sample engine and Rust mocker emit synthetic logprobs
  when `output_options.logprobs` is set.

Observability:
- **Health-check canary** — `health_check_payload()` + operator
  override via `DYN_HEALTH_CHECK_PAYLOAD` /
  `--health-check-payload`
- **Metrics & Prometheus** — vendor-prefixed bridge (`vllm:`,
  `sglang:`, `trtllm_`, `lmcache:`); framework-owned lifecycle
  gauges (`cleanup_time_seconds`, `drain_time_seconds`,
  `model_load_time_seconds`); per-rank `dynamo_component_*` gauges
  (`total_blocks`, `gpu_cache_usage_percent`, `kv_cache_hit_rate`)
  + router `kv_used_blocks` signal via `SnapshotPublisher`. See
  [Metrics](#metrics) below.
- **KV event publishing** — `kv_event_sources()` returning
  `ZmqSource` or `PushSource` (prefix cache `BlockStored`/`Removed`
  events to the router via NATS).
- **KV-aware routing** — `dp_rank.forced_dp_rank` /
  `validate_global_dp_rank` plus `EngineConfig.data_parallel_size`
  / `data_parallel_start_rank` for DP-aware scheduling.
- **OpenTelemetry tracing** — `telemetry.current_span` /
  `start_span` plus W3C trace-header propagation to the underlying
  inference engine via `telemetry.engine_trace_kwargs(context)`.
  See [Telemetry](#telemetry) below.

Request handling:
- **Guided decoding / structured outputs** — wired per-engine on the
  request side, with engine-specific coverage:
  - vLLM (`build_sampling_params` → `StructuredOutputsParams`):
    JSON schema, regex, grammar, choice.
  - TRT-LLM (`GuidedDecodingParams`): JSON schema, regex, grammar,
    choice, `json_object`.
  - SGLang (`_get_guided_decoding_params`): JSON schema only;
    regex / grammar / choice are silently dropped (see SGLang gaps
    below).
- **Structural tag generation** — `WorkerConfig.structural_tag_{mode,
  scope, schema}` + `serialize_structural_tag` helper
- **Custom Jinja chat templates** — `WorkerConfig.custom_jinja_template`
  flows to `LocalModelBuilder.custom_template_path` (frontend
  applies the template; the backend just registers the path)
- **Tool / reasoning parsers** — `WorkerConfig.tool_call_parser`,
  `reasoning_parser`, `exclude_tools_when_tool_choice_none`

### Common gaps (all engines)

| Feature | Description |
|---------|-------------|
| Text-in-text-out mode | OpenAI-compatible chat/completion with engine-side tokenization. Unified hardcodes `ModelInput.Tokens`. |
| Multimodal | Images / video / embeddings, NIXL embedding transfer, encode workers. `worker.py:_to_rust_disaggregation_mode` rejects the `ENCODE` role. |
| Diffusion | Image (FLUX), video (Wan2.1), LLM diffusion (DLLM) workers; no diffusion engine, MediaOutput, or media scheduling on the unified path. |
| LoRA adapters | Dynamic load / unload / list, ModelDeploymentCard publishing, per-adapter serialization locks, per-request adapter threading on prefill. |
| Snapshot / checkpoint | CRIU-based engine state save/restore + identity reload. |

### vLLM-specific gaps

| Feature | Description |
|---------|-------------|
| Sleep/wake | 3-level vLLM engine lifecycle control (`VllmEnginePauseController`) with shutdown-delay tags |
| Elastic EP scaling | `scale_elastic_ep` endpoint with Ray node management |
| GMS shadow mode | GPU Memory Service integration with failover lock (`--gms-shadow-mode`, `configure_gms_lock_mode`) |
| ModelExpress P2P | Distributed model loading via P2P (`--model-express-url`, `register_modelexpress_loaders`, `mx-source` / `mx-target` load formats) |
| KV block clearing | Prefix cache reset endpoint |
| `VllmEngineMonitor` | Background `EngineDeadError` detection task |
| Instrumented scheduler + FPM relay | Per-forward-pass `ForwardPassMetrics` ZMQ telemetry |
| `KvConnectorProtocol` abstraction | Legacy abstracts NIXL pull / Mooncake push; unified uses vLLM's internal connector only |
| `--headless` multi-node mode | Secondary-node TP/PP worker mode (`run_dynamo_headless`); unified requires every node to run the backend |
| `--benchmark-mode` family | The `--benchmark-*` flag family (mode, prefill/decode granularities, warmup, output path, timeout) injects into `vllm_config.additional_config` |
| "Omni" alternative entry point | `dynamo.vllm.omni.*` parallel mode for alternative tensor workflows |
| Multimodal (vLLM) | NIXL embedding transfer (`EmbeddingTransferMode`, `--embedding-transfer-mode`), embedding LRU cache (`--multimodal-embedding-cache-capacity-gb`), Qwen VL mRoPE, `EncodeWorkerHandler`, `--route-to-encoder` |
| LoRA (vLLM) | Three endpoints (`load_lora`, `unload_lora`, `list_loras`); also: unified prefill doesn't thread per-request LoRA adapters into the engine call |

### SGLang-specific gaps

| Feature | Description |
|---------|-------------|
| Embedding inference | `async_encode()` path, OpenAI embedding response format, `EmbeddingWorkerHandler` |
| Image diffusion | `DiffGenerator` for text-to-image (FLUX, etc.) with TP/DP; `ImageDiffusionWorkerHandler` |
| Video generation | `DiffGenerator` for text-to-video (Wan2.1, etc.); `VideoGenerationWorkerHandler` |
| LLM diffusion (DLLM) | Diffusion language model algorithm support (`--dllm-algorithm`, `DiffusionWorkerHandler`) |
| Multimodal encode worker | Front-facing `MMEncoder`, embedding LRU cache, NIXL transfer (`MultimodalEncodeWorkerHandler`) |
| Multimodal worker | Aggregated and disaggregated-prefill multimodal inference with `EmbeddingsProcessor` |
| Deferred signal handling | `install_graceful_shutdown` captures SGLang's internal `loop.add_signal_handler` registrations for coordinated teardown |
| Snapshot pause | Legacy `prepare_snapshot_engine` wires `SGLangEnginePauseController` to the shared `EngineSnapshotController` (CRIU + identity reload); unified path doesn't invoke it |
| Image/video health-check payloads | `ImageDiffusionHealthCheckPayload`, `VideoGenerationHealthCheckPayload` |
| `register_model_with_readiness_gate` + image/video fast paths | `register.py` skips HF `config.json` download for `ModelType.Images` / `ModelType.Videos` |
| Output modalities override | Required for diffusion workers (default `["text"]` -> `["image"]` / `["video"]`) |
| `protocol.py` Pydantic models | `EmbeddingRequest`, `DisaggPreprocessedRequest`, multimodal content types |
| `--disagg-config` YAML override | `--disagg-config` / `--disagg-config-key` for YAML-based disagg config |
| `--enable-rl` | RL support via `call_tokenizer_manager` route |
| Guided-decoding constraint coverage | `_get_guided_decoding_params` forwards only `json` (and `structural_tag`); `regex` / `grammar` / `choice` are silently dropped on the unified path even though SGLang's engine accepts them |

### TRT-LLM-specific gaps

| Feature | Description |
|---------|-------------|
| Multimodal processing | `MultimodalRequestProcessor` with image URL fetching (`load_tensor_from_path_or_url`, httpx) and embedding injection |
| Image / video diffusion | `DiffusionEngine`, auto-detect pipeline from `model_index.json`, MP4 encoding, `MediaOutput`, full `DiffusionConfig` flag family |
| Encode helper (EPD) | Remote encode via `encode_client`, NIXL tensor reading; full `_encode_and_pack_disaggregated_params` flow |
| KV cache connector | `args.py` accepts `none` or `kvbm` (`VALID_TRTLLM_CONNECTORS`), but unified `TrtllmLLMEngine.from_args()` never calls `build_kv_connector_config()` or `get_consolidator_endpoints()` — `kvbm` is accepted at the CLI but not actually wired |
| Per-role request handlers | Legacy `PrefillHandler` / `DecodeHandler` / `EncodeHandler` / `AggregatedHandler`; unified collapses into one `generate()` |
| Fatal vs per-request errors | Legacy distinguishes recoverable `RequestError` (`finish_reason == "error"` branch) from fatal engine errors; unified treats them identically |
| Backend selection | Legacy `Backend` enum supports `PYTORCH` and `AUTODEPLOY`; unified hardcodes `Backend.PYTORCH` |
| Modality routing | Legacy `init_worker` dispatches `TEXT` / `MULTIMODAL` / `IMAGE_DIFFUSION` / `VIDEO_DIFFUSION` via `--modality`; unified is LLM-only |

> **Attention-DP scheduling is on the unified path.** `from_args()`
> sets `enable_attention_dp` into the engine args, and
> `SchedulingParams(attention_dp_rank=..., attention_dp_relax=False)`
> is constructed in the generate flow.

### Recommended migration order

For users picking what to land next on the unified path:

1. **Text-in-text-out** (`ModelInput.Text`) — common ask; needs
   engine-side tokenization + chat templating path.
2. **LoRA dynamic load/unload + MDC publishing** — production-visible
   feature with concrete API surface (three endpoints on vLLM
   `handlers.py`).
3. **Engine routes / lifecycle endpoints** — sleep/wake, profile
   start/stop, weight updates, KV block clearing, prefix cache
   reset. Visible in operator workflows.
4. **Snapshot / CRIU** — production checkpoint support.
5. **Multimodal / diffusion / video / DLLM** — biggest functional
   gap, but largest scope. Best parallelized across modality leads.
