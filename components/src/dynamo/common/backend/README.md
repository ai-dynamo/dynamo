# Dynamo Python Backend

**Supported today:** aggregated and disaggregated (prefill/decode/encode)
inference, the shared multimodal request and encoder-handoff contract,
metrics + Prometheus bridging, KV event publishing, KV-aware (DP-rank)
routing, health-check canaries, OpenTelemetry tracing, and request-side
guided decoding / structural tag.

> **Looking for a walkthrough?** Start with the
> [Writing Unified Backends](../../../../../docs/development/unified-backends.md)
> guide and choose the Python tab. This README is the in-tree reference:
> file layout, cancellation contract, disaggregation contract, and the
> error-handling table.

A two-class abstraction that separates **runtime integration** (common across
all backends) from **engine logic**.

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
# my_backend/my_backend_main.py
from dynamo.common.backend.run import run
from my_backend.my_backend_engine import MyEngine

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
backend-specific cleanup.

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
2. **Vendor-prefixed metrics** — engines bridge their own
   `prometheus_client.CollectorRegistry` into the runtime's combined
   `/metrics` output via `register_prometheus(metrics)` using
   `register_global_registry` (or `register_engine_registry` for a
   private registry).

`ComponentSnapshot.kv_cache_hit_rate` is tri-state: `None` means "no data
yet" or "no prefix cache" (gauge skipped); `0.0` is a legitimate
zero-hit measurement.

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
    logprobs.py          # Shared logprob helpers (extractors +
                         #   option parsing)
    metrics.py           # Prometheus helpers (gather_with_labels,
                         #   ensure_prometheus_multiproc_dir, registration)
    publisher.py         # ComponentSnapshot dataclass (push payload)
    run.py               # Common entry point: run(engine_cls)
    sample_engine.py     # SampleLLMEngine (reference impl)
    sample_main.py       # Entry point for sample engine
    tests/               # test_backend_bindings, test_disagg_helpers,
                         #   test_logprobs, test_sample_engine
    CLAUDE.md            # Design notes (rationale, invariants)
```
