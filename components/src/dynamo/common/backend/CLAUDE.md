# Backend Module

`Worker` (runtime integration) drives a `BaseEngine` (ABC for
engine-specific logic). `BaseEngine` owns the modality-agnostic lifecycle
(`from_args`/`start`/`abort`/`drain`/`cleanup` + metrics/health hooks);
the two modality subclasses add only their `generate` contract:
`LLMEngine` (token pipeline: `token_ids` in/out, plus `kv_event_sources`)
and `RawEngine` (raw media pipeline: OpenAI request dict in, response dict
out; `DiffusionEngine` is a domain subclass). See `README.md` for full docs.

## Engine Lifecycle

```
from_args -> start -> register_prometheus -> component_metrics_dp_ranks -> attach_snapshot_publisher -> generate/abort -> is_quiescent -> cleanup
     |          |              |                       |                              |                          |             |        |
  parse args, start engine, vendor registry      declare dp_ranks            engine stashes publisher,   serve requests  drain in-flight,
  return     return        bridge (optional)     for component gauges        pushes ComponentSnapshot    (concurrent)    then cleanup,
  config     metadata                                                        from natural producer                       release resources
```

(On the Rust trait side, `register_prometheus`,
`component_metrics_dp_ranks`, and `attach_snapshot_publisher` collapse
into one `setup_metrics(ctx) -> MetricsBindings` method. The PyO3
bridge calls Python's three methods and assembles the bindings —
Python engine authors keep the split API.)

1. `from_args(argv)` -- classmethod factory. Parses CLI args, returns
   `(engine, WorkerConfig)`. Engine is NOT started yet.
2. `start()` -- starts the engine, returns `EngineConfig`. After this returns
   `generate()` MUST be ready to accept calls.
3. `register_prometheus(metrics)` -- bridge a vendor-prefixed registry
   (`vllm:`, `sglang:`, `trtllm_`, `lmcache:`) into the runtime's
   `/metrics` output. Optional, default no-op. The engine receives a
   slim `EngineMetrics` capability handle. Framework-owned lifecycle
   gauges (`cleanup_time_seconds`, `drain_time_seconds`,
   `model_load_time_seconds`) emit regardless of what this method does.
4. `component_metrics_dp_ranks()` -- declare which data-parallel ranks
   this engine will publish snapshots for. Default empty list opts
   out of per-rank gauges (lifecycle gauges still emit). When
   non-empty, the framework constructs a Rust-owned
   `SnapshotPublisher` (per-rank `dynamo_component_*` `GaugeVec`s
   seeded at zero + per-rank `WorkerMetricsPublisher` for the
   router's `kv_used_blocks` NATS signal).
5. `attach_snapshot_publisher(publisher)` -- framework invokes this
   once with the constructed `SnapshotPublisher`. Engine stashes the
   reference and calls `publisher.publish(dp_rank, ComponentSnapshot(...))`
   from its natural producer thread (engine iteration callback,
   stat-logger, ZMQ recv loop, etc.) on every tick. Push is
   event-driven — no framework poll loop, no snapshot-fn latency
   ceiling. `ComponentSnapshot.kv_cache_hit_rate` is tri-state:
   `None` means "no data yet" or "no prefix cache" (gauge child not
   written); `0.0` is a legitimate zero-hit measurement.
6. `generate(request, context)` -- streaming inference, called concurrently.
7. `abort(context)` -- cancel an in-flight request (optional, default no-op).
8. `is_quiescent()` -- whether in-flight KV transfers are done so GPU memory
   can be released (default `None`). Polled only on **prefill workers**,
   between the grace period and cleanup: `True` exits the drain early, `False`
   and `None` (default) keep polling until the budget expires. Override only if
   the engine can observe transfer completion; aggregated/decode are never
   polled.
9. `cleanup()` -- called exactly once. Runs after `start()` succeeded
   on shutdown, **and** after `start()` raised — so implementations
   must be null-safe against partial state (inner engine handle,
   sockets, background tasks). All current engines guard each
   resource with `if self.engine is not None`. Must also be
   idempotent: a second call after a successful first is a no-op.
   Not called when `start()` was never invoked (e.g. pre-start
   shutdown); use `__del__` for resources allocated in the
   constructor. Engine **must** stop any producer thread it started
   for `publisher.publish(...)` here.

## Design Constraints

- **ZERO duplication across engine implementations.** This is the #1 priority.
  The entire reason this module exists is to eliminate the code duplication
  that grew across vllm, sglang, and trtllm. Before writing any logic inside
  a `LLMEngine` subclass, check whether the same logic already exists in
  another engine. If it does, extract it into `Worker` or a shared
  utility and have all engines call the shared version.
  When adding new features, always ask: "is this engine-specific or common?"
  If two or more engines would need the same code, it is common.

- **One `Worker`, one lifecycle base, one subclass per modality.** `Worker`
  owns runtime lifecycle. `BaseEngine` owns the modality-agnostic lifecycle;
  `LLMEngine` and `RawEngine` subclass it and differ *only* in the `generate`
  contract (token vs. raw media). Do not add per-engine mixins or intermediate
  bases between a modality ABC and its concrete backend (e.g. `SampleLLMEngine`).
  A new media modality is a new `RawEngine` implementation, not a new
  engine trait or lifecycle.

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
  metadata (token-pipeline KV/DP fields live in the `llm=LlmRegistration(...)`
  sub-record; `RawEngine`s leave `llm=None`) but must not reach into engine
  internals. `start()` returns this metadata so the boundary stays clean.

- **No hooks.** If behavior needs to be shared across engines, put it in
  `Worker` or a shared utility, not in a hook system.

- **Parallel path.** The existing `main.py` / `worker_factory.py` / `init_llm.py`
  entry points remain untouched. The `common/backend/` entry points (e.g.
  `sample_main.py`) are a separate path. Do not break or modify existing
  backends when changing this module.

## Request / Response Contract

`GenerateRequest` and `GenerateChunk` (`engine.py`) are `TypedDict`s that
type the `generate()` signature.  `GenerateRequest` has `token_ids`
(required) plus optional `sampling_options`, `stop_conditions`, and
`output_options`.  `GenerateChunk` has `token_ids` and `index` (both
required; use `index=0` for single-choice chunks), plus optional
`finish_reason` and `completion_usage` (both required on the final chunk),
`log_probs` (one float per emitted token), and `top_logprobs` (per-position
list of ranked alternative dicts — see `logprobs.py` for the entry shape).
Engines may read backend-specific request keys, but response chunk keys
should be added to the shared contract before use.

Build the `completion_usage` dict inline. Finish reason normalization
(e.g. `"abort"` → `"cancelled"`) is handled by the Rust layer.

## Adding a New Engine

This is the in-process route. Sidecar-based integration for SGLang, vLLM, and
TRT-LLM is under development.

1. Create `<backend>/<backend>_engine.py` subclassing `LLMEngine`
2. Implement `from_args()`, `start()`, `generate()`, `cleanup()` (required)
   and `abort()` (optional)
3. `from_args()` must parse args and return `(engine, WorkerConfig)`
4. Create `<backend>/<backend>_main.py` calling `run(<YourEngine>)`
5. Use `sample_engine.py` as the reference implementation

## Disaggregated Serving

`WorkerConfig.disaggregation_mode` is the single source of truth for the
worker's role. Default `DisaggregationMode.AGGREGATED` keeps existing
callers unchanged.

What the **runtime** does with the mode (Rust `Worker` in `lib/backend-common`):

- `Prefill` → register with `ModelType.Prefill` (legacy marker bit, no
  OpenAI surface — dual-emitted for cross-version compat) and
  `WorkerType.Prefill` regardless of `endpoint_types`, so the frontend's
  `PrefillRouter` targets this worker via `worker_type`.
- `Decode` → keep `endpoint_types`, but force-disable
  `enable_local_indexer` (decode workers don't host the indexer endpoint).
- `Aggregated` → register with the parsed `endpoint_types`.
- `Encode` → register **surface-less** (`ModelType.empty()`) with
  `WorkerType.Encode` and topology needs `[[Prefill, Decode], [Aggregated]]`,
  so the discovery layer registers the worker for readiness only and hides it
  from `/v1/models`. The Encode role is a multimodal encoder upstream of
  P/D/Agg; backend-specific encoder implementations land separately.

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
- `Encode`: produce the encoder handoff payload on the terminal chunk's
  `encoder_result` (object-only) via `encoder_terminal_chunk`. The native
  vLLM backend is text-only and rejects this role at startup.

`route_to_encoder` (on `WorkerConfig`; CLI `--route-to-encoder` / env
`DYN_ROUTE_TO_ENCODER`) makes an `Aggregated` or `Prefill` worker advertise an
upstream `Encode` peer in its topology `needs`. It is meaningful only for
agg/prefill — setting it on `Decode` or `Encode` is rejected at startup with
`BackendError::InvalidArgument`.

`is_quiescent()` lets a prefill worker exit the drain early once its KV
transfers finish, before GPU memory is released. Engines that can't introspect
leave the default `None` (wait the budget); aggregated/decode are never drained.

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

## Metrics cost notes

`publisher.publish(dp_rank, snapshot)` is the hot path: one atomic
gauge write per field plus one NATS publish, no allocation, no GIL
on the gauge side. Engines call it from their natural producer
thread at the engine's own cadence — there is no framework poll
loop.

Per-backend producer:

- **vLLM**: per-iteration stat-logger calls `publisher.publish(...)`.
- **SGLang**: ZMQ pull loop on the leader node calls `publisher.publish(...)`
  on every received `KvMetrics`.
- **TRT-LLM**: a dedicated `_metrics_poll_loop` thread calls
  `engine.llm.get_stats(timeout=0.2)` and then `publisher.publish(...)`.
  The 200 ms `get_stats` cycle is the cost — driven by
  `enable_iter_perf_stats=True` and `return_perf_metrics=True`, both
  unconditional. If you run TRT-LLM without KV routing AND don't
  scrape `dynamo_component_*`, opt out via
  `--trtllm.enable_iter_perf_stats false` (and `--trtllm.return_perf_metrics false`);
  the gauges will keep their seeded zero values and the poll thread
  will run but find nothing to publish.

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

## Trace propagation

Engines must forward W3C trace headers to their underlying inference engine
so that vLLM / TRT-LLM / SGLang's internal OTel spans (scheduler, forward
pass, KV transfer) nest under the framework's `engine.generate` span. Splat
`telemetry.engine_trace_kwargs(context)` into the inference-engine call:

```python
from dynamo.common.backend import telemetry

# vLLM / TRT-LLM: default kwarg name `trace_headers`, unconditional
gen = self.engine_client.generate(
    prompt, sampling_params, request_id,
    **telemetry.engine_trace_kwargs(context),
)

# SGLang: different kwarg name + gated on --enable-trace
stream = await self.engine.async_generate(
    ...,
    **telemetry.engine_trace_kwargs(
        context,
        kwarg_name="external_trace_header",
        enabled=self.enable_trace,
    ),
)
```

`engine_trace_kwargs` returns an empty dict when no trace context is
available or `enabled=False`, so the engine API kwarg is simply absent —
downstream treats absence the same as `None`. Centralizes the build +
gate logic so adding a new backend is one declaration, not three call
sites that drift out of sync.

| Backend | Method | Kwarg |
|---|---|---|
| vLLM | `engine_client.generate` | `trace_headers` (default) |
| TRT-LLM | `engine.llm.generate_async` | `trace_headers` (default) |
| SGLang | `engine.async_generate` | `external_trace_header`, gated on `enable_trace` |

For the lower-level `Context.trace_headers()` method or the
`telemetry.trace_headers(context)` wrapper, see their docstrings — most
engine code should reach for `engine_trace_kwargs` first. Without this
forwarding, the trace_id reaches the worker but never reaches the
inference engine — the trace tree shows a gap where the engine's internal
spans should be.

## Key Files

| File | What it does |
|------|-------------|
| `engine.py` | `BaseEngine` lifecycle ABC + `LLMEngine` / `RawEngine` modality subclasses (`DiffusionEngine` is a `RawEngine` subclass) -- the interface engines implement. |
| `publisher.py` | `ComponentSnapshot` dataclass (the push payload). The `SnapshotPublisher` itself is a Rust-owned object exposed as `dynamo._core.backend.SnapshotPublisher`. |
| `metrics.py` | Prometheus integration helpers. `register_global_registry` / `register_engine_registry` are engine-facing (vendor-registry bridge inside `register_prometheus`). `ensure_prometheus_multiproc_dir` / `gather_with_labels` remain engine-side utilities. |
| `worker.py` | `Worker` -- thin shim over `dynamo._core.backend.Worker`; lifecycle state machine and signal handling live in Rust (`lib/backend-common`) |
| `run.py` | Common entry point -- `run(engine_cls)` used by each backend's main (e.g. `sample_main.py`) |
| `logprobs.py` | Shared logprob helpers: `parse_logprob_options`, `extract_from_completion_output` (vLLM/TRT-LLM shape), `extract_from_sglang_meta` + `build_sglang_logprob_kwargs` (SGLang cumulative-array shape). The backend request handlers delegate here. |
| `sample_engine.py` | Reference engine -- use as template and for testing |

The Rust `Worker` (in `lib/backend-common/src/worker.rs`) owns:
  - Lifecycle state machine (Init → Running → Stopped)
  - SIGTERM/SIGINT handling and graceful shutdown orchestration
    (discovery unregister → grace period → drain → cleanup)
  - 3-phase distributed runtime shutdown after engine.cleanup() returns

State-machine and orchestrator invariants are pinned by Rust unit tests
in the same crate. Don't reimplement them on the Python side.
