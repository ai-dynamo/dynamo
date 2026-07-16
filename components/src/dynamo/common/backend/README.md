# Dynamo Python Backend

**Supported today:** aggregated and disaggregated (prefill/decode/encode)
inference, the shared multimodal request and encoder-handoff contract,
metrics + Prometheus bridging, KV event publishing, KV-aware (DP-rank)
routing, health-check canaries, OpenTelemetry tracing, and request-side
guided decoding / structural tag.

> **Work in progress.** Multimodal support is backend-specific: vLLM supports
> aggregated and prefill/decode image and video inference, while separate
> encode workers and SGLang / TRT-LLM multimodal execution remain on their
> non-unified paths. Diffusion (image/video/DLLM),
> LoRA (SGLang / TRT-LLM — vLLM is supported),
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
    |   - is_quiescent() -> Optional[bool]   (prefill drain early-exit, optional)
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
    - drains prefill workers (polls engine.is_quiescent()) then calls engine.cleanup() on shutdown
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
from dynamo.common.backend import LLMEngine, EngineConfig, LlmRegistration, WorkerConfig

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
            # Token-pipeline metadata goes in the `llm` sub-record. Populate
            # `bootstrap_host` / `bootstrap_port` here on prefill workers that
            # advertise a Dynamo-level handshake address.
            llm=LlmRegistration(context_length=4096, kv_cache_block_size=16),
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
The sample engine includes synthetic multimodal handling for aggregated and
Encode/Prefill/Decode deployments. CPU-only direct worker-handoff smokes live in
`examples/backends/sample/launch/multimodal_agg.sh` and
`examples/backends/sample/launch/multimodal_disagg.sh`. These smokes exercise
distinct worker processes and TCP request transport; they intentionally bypass
the frontend and do not claim frontend routing coverage.

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
| **TRT-LLM** | Builds `LlmDisaggregatedParams(request_type="context_only")`, generates one token, packs the encoded handoff into the response. Inherits the default `is_quiescent` (None), so the prefill drain waits the full budget for transfers to finish. | Decodes `request.prefill_result.disaggregated_params`, flips `request_type` to `generation_only`, generates normally. |

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

Backend shape:

- **vLLM** pushes snapshots from its stat logger for each local DP rank and
  bridges `vllm:` plus multiprocess-only `lmcache:` metrics from the global
  registry.
- **SGLang** pushes scheduler snapshots from the metrics leader node to avoid
  double-counting DP ranks and bridges `sglang:` metrics from a private
  multiprocess registry when `--enable-metrics` is set.
- **TRT-LLM** pushes snapshots from the stats poll thread for each attention-DP
  rank and bridges the global `trtllm_` registry.

`WorkerConfig.enable_kv_routing=False` skips snapshot publisher construction,
but the Prometheus bridge still runs. Use it when the worker should expose
vendor metrics without feeding KV-aware routing signals.

## KV Event Publishing

On the unified path, `Worker` owns `KvEventPublisher` construction. Engines
declare sources with `kv_event_sources()`; they do not instantiate
`KvEventPublisher` directly.

Use `ZmqSource` when the engine already emits Dynamo-compatible KV events on a
ZMQ socket:

```python
from dynamo.common.backend.publisher import ZmqSource

async def kv_event_sources(self):
    return [
        ZmqSource(endpoint="tcp://127.0.0.1:5557", dp_rank=0),
    ]
```

Use `PushSource` when the engine needs a live publisher and drives
`publish_stored()` / `publish_removed()` from its own thread:

```python
from dynamo.common.backend.publisher import PushSource

def _on_kv_publisher_ready(self, publisher):
    self._kv_publisher = publisher
    self._start_kv_event_thread()

async def kv_event_sources(self):
    return [PushSource(on_ready=self._on_kv_publisher_ready, dp_rank=0)]
```

Return one source per DP rank owned by this worker, and keep that rank ownership
stable for the engine lifetime. `EngineConfig.llm.kv_cache_block_size` must be
set or `Worker` skips KV event publishers; snapshot publishers still work
without a block size.

For `PushSource`, cleanup is the engine's responsibility. Stop event threads in
`cleanup()`, prevent new publishes once cleanup begins, and let any in-flight
publish loop observe the shutdown signal before resources are released.

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
- `is_quiescent()` prefill-drain early-exit hook
- `DynamoException` error chain wrapping
- Finish reason normalization handled by the Rust layer
- Engine control plumbing, with per-backend profiling, pause/resume, and supported weight-update controls
- **Dynamic LoRA (vLLM)** — load / unload / list adapters at runtime,
  with ModelDeploymentCard publishing for frontend discovery,
  per-adapter serialization locks, and per-request routing. Gated on
  `--enable-lora` **and** `DYN_LORA_ENABLED=true`; SGLang / TRT-LLM
  advertise no LoRA updates yet. Because LoRA ops mutate engine-managed
  adapters rather than the serving lifecycle, they ride the generic
  **engine-update** mechanism (a sibling of engine controls, kept separate
  so the control surface isn't inflated):
  - **Canonical API** (unified backend): `POST /engine/update/load_lora`
    `{lora_name, source:{uri}}`, `POST /engine/update/unload_lora`
    `{lora_name}`, `POST /engine/update/list_loras` `{}` (uniform `POST` +
    JSON body). Engine updates return **HTTP 200** with a
    `{"status": "error", ...}` body on semantic failure (5xx only when the
    handler raises).
  - **`/v1/loras` compatibility alias** — for the unified backend, the
    legacy surface forwards to the engine updates (`POST /v1/loras` →
    `load_lora`, `GET /v1/loras` → `list_loras`,
    `DELETE /v1/loras/{name}` → `unload_lora`), preserving legacy HTTP
    semantics (a `status:"error"` response maps to **HTTP 500** on
    load/unload). When LoRA is unsupported, these return an explicit
    "LoRA management not available" error rather than failing opaquely.
  - **Legacy (non-unified) vLLM** continues to serve `/v1/loras`
    unchanged.
  - Loaded adapters appear in `GET /v1/models`; inference selects an
    adapter by sending `"model": "<lora_name>"`.
- **Sleep/wake (vLLM)** — `sleep` / `wake_up` controls via
  `VllmEnginePauseController` (discovery unregister before sleep,
  re-register after wake; `worker.rs` `engine_control_policy`)
- **KV block clearing (vLLM)** — `POST /engine/control/clear_kv_blocks`
  on the unified worker's system port,
  with an empty JSON object (`{}`). The control resets both the prefix
  cache and connector cache in aggregated, prefill, and decode modes. It
  returns `{"status":"success","message":"KV cache cleared"}` on
  success and HTTP 200 with `status:"error"` on semantic failure. The
  control runs directly without pausing generation or draining requests;
  if blocks are still in use, retry after the active requests finish.
- **Elastic EP scaling (vLLM)** — `scale_elastic_ep` control at parity
  with the legacy handler: `new_data_parallel_size` validation, a
  single-flight lock (concurrent scales rejected, not queued), and the
  `ray.util.state.list_nodes` → GCS shim for ray `--minimal`. Served at
  `/engine/control/scale_elastic_ep` on the system port (the unified
  Worker namespaces controls under `/engine/control/<name>`, matching the
  legacy backend). Requires the Ray DP backend
  (`--data-parallel-backend ray`, `nnodes == 1`) **and `ray` installed**
  (the vLLM runtime image does not ship it). The single head-node backend
  drives `add_dp_placement_groups` to place DP-worker Ray actors across
  the Ray cluster, so multi-node is a Ray-cluster-membership concern
  (operator-managed `ray start`), not a per-node backend concern.
  Locally GPU-validated on H200 GPUs with vLLM 0.24.0: scale-**up** (2→4)
  and scale-**down** (4→2) return `status:ok`, and serving continues after
  each transition. The integration test remains skipped in CI because each
  unquantized Qwen3-30B-A3B replica needs about 57 GiB for weights at TP=1,
  while CI's four-GPU runner has only 24 GiB per GPU; the test requires at
  least 80 GiB per GPU for weights and runtime headroom.
- **Headless multi-node (vLLM)** — `--headless` secondary nodes run
  vLLM workers only (multi-node TP/PP with `--data-parallel-backend mp`),
  bypassing DistributedRuntime; `unified_main` routes them to
  `run_dynamo_headless` before the Worker/engine path. Distinct from
  elastic EP, which uses the Ray backend above.
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
- **Multimodal (vLLM)** — image and video inference in aggregated and
  prefill/decode deployments, frontend-rendered `mm_kwargs` transfer over
  shared memory or NIXL, stable frontend hash forwarding, CPU embedding cache,
  and Qwen-VL decode metadata reconstruction. Separate encode workers are not
  supported by the unified vLLM entry point.

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
  request side with JSON schema, regex, grammar, and choice coverage:
  - vLLM (`build_sampling_params` → `StructuredOutputsParams`):
    JSON schema, regex, grammar, choice.
  - TRT-LLM (`GuidedDecodingParams`): JSON schema, regex, grammar,
    choice, `json_object`.
  - SGLang (`_get_guided_decoding_params`): JSON schema, regex,
    grammar through `ebnf`, and choice through an escaped regex.
- **Structural tag generation** — `WorkerConfig.structural_tag_{mode,
  scope, schema}` + `serialize_structural_tag` helper
- **Custom Jinja chat templates** — `WorkerConfig.custom_jinja_template`
  flows to `LocalModelBuilder.custom_template_path` (frontend
  applies the template; the backend just registers the path)
- **Tool / reasoning parsers** — `WorkerConfig.tool_call_parser`,
  `reasoning_parser`, `exclude_tools_when_tool_choice_none`

### Remaining feature gaps

| Feature | Description |
|---------|-------------|
| Text-in-text-out mode | OpenAI-compatible chat/completion with engine-side tokenization. Unified hardcodes `ModelInput.Tokens`. |
| Multimodal parity | The shared request and encoder-handoff contract are available. vLLM supports aggregated and prefill/decode image and video inference; SGLang / TRT-LLM execution and separate encode workers remain separate work. |
| Diffusion | Image (FLUX), video (Wan2.1), LLM diffusion (DLLM) workers; no diffusion engine, MediaOutput, or media scheduling on the unified path. |
| LoRA adapters (SGLang / TRT-LLM) | Dynamic load / unload / list, ModelDeploymentCard publishing, per-adapter serialization locks, per-request adapter threading. **vLLM is supported on the unified path** — see [What works today](#what-works-today); SGLang and TRT-LLM advertise no LoRA updates yet. |
| Snapshot / checkpoint | CRIU-based engine state save/restore + identity reload. |

### vLLM-specific gaps

| Feature | Description |
|---------|-------------|
| GMS shadow mode | GPU Memory Service integration with failover lock (`--gms-shadow-mode`, `configure_gms_lock_mode`) |
| ModelExpress P2P | Distributed model loading via P2P (`--model-express-url`, `register_modelexpress_loaders`, `mx-source` / `mx-target` load formats) |
| `VllmEngineMonitor` | Background `EngineDeadError` detection task |
| Instrumented scheduler + FPM relay | Per-forward-pass `ForwardPassMetrics` ZMQ telemetry |
| `KvConnectorProtocol` abstraction | Legacy abstracts NIXL pull / Mooncake push; unified uses vLLM's internal connector only |
| `--benchmark-mode` family | The `--benchmark-*` flag family (mode, prefill/decode granularities, warmup, output path, timeout) injects into `vllm_config.additional_config` |
| "Omni" alternative entry point | `dynamo.vllm.omni.*` parallel mode for alternative tensor workflows |
| Separate multimodal encode worker | The unified entry point rejects `--disaggregation-mode encode` and `--route-to-encoder`. Encoder-managed embedding transfer remains on the legacy worker path. P/D video requests also reload raw media on decode because the handoff carries image metadata only. |

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
2. **LoRA dynamic load/unload + MDC publishing** — **done for vLLM**
   (engine updates `/engine/update/load_lora|unload_lora|list_loras` + a
   `/v1/loras` compatibility alias; see [What works today](#what-works-today)).
   Remaining: SGLang and TRT-LLM, which advertise no LoRA updates yet.
3. **Engine routes / lifecycle endpoints** — weight updates. (Profiling,
   sleep/wake, KV block clearing, elastic-EP scaling, and headless
   multi-node already landed.) Visible in operator workflows.
4. **Snapshot / CRIU** — production checkpoint support.
5. **Multimodal / diffusion / video / DLLM** — biggest functional
   gap, but largest scope. Best parallelized across modality leads.
