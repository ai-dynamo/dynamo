# Frontend Router Topologies and RequestTracker Boundaries

Before changing frontend routing, timing, agent tracing, or Python router
bindings, identify the deployment topology. Dynamo has multiple router paths
that share `KvRouter` logic but do not share the same metadata boundaries.

## Topologies

1. **Integrated Rust frontend**

   ```
   dynamo.frontend
     -> Rust OpenAIPreprocessor
     -> in-process Rust KvPushRouter / KvRouter
     -> worker
     -> Rust DeltaGenerator
   ```

   `--router-mode kv` uses an in-process Rust router. There is no router RPC
   hop. The Rust `DeltaGenerator` creates an `Arc<RequestTracker>`, attaches it
   to `PreprocessedRequest`, and the router updates the same object.

2. **Python chat processor inside `dynamo.frontend`**

   ```
   dynamo.frontend
     -> Python VllmProcessor or SglangProcessor
     -> PyO3 RoutedEngine
     -> in-process Rust KvPushRouter / KvRouter
     -> worker
     -> Python postprocessor
   ```

   `--dyn-chat-processor vllm|sglang` still uses the embedded Rust router.
   However, the Python processor constructs a dict and `RoutedEngine`
   deserializes it into `PreprocessedRequest`. `PreprocessedRequest.tracker`
   is `#[serde(skip)]`, so an `Arc<RequestTracker>` does not cross this
   Python boundary automatically.

3. **Standalone or custom Python router service**

   ```
   frontend
     -> router service RPC
     -> binding-level KvRouter
     -> worker
   ```

   Examples: `python -m dynamo.router` and
   `python -m dynamo.thunderagent_router`. These own a binding-level
   `KvRouter` in another process. Rust annotations and `Arc<RequestTracker>`
   state do not cross the Rust -> Python -> Rust or RPC boundary automatically.

## Invariants

- Do not confuse router scheduler state with `RequestTracker`.
  KV overlap selection, active-request booking, output-block tracking, and
  cleanup can work while response timing, worker attribution, and agent traces
  are missing.

- Never assume `RequestTracker` crosses a Python or process boundary.
  It is intentionally not serialized. If downstream timing must reach the
  frontend, transport a snapshot explicitly in the response data and merge it
  into the frontend-owned tracker.

- Put new framework metadata under `extra_args["dynamo"]`. Do not add new
  Dynamo metadata to `disaggregated_params` or `engine_data`; those are
  engine-owned payloads. The existing `disaggregated_params.worker_id`
  injection is compatibility behavior, not a pattern to extend.

- Strip internal framework metadata before returning an OpenAI response to
  the client unless it is intentionally exposed through `nvext`.

- When adding timing or tracing fields, audit all three topologies. Test
  aggregated and disaggregated serving separately. A fix for the default Rust
  frontend does not automatically fix `--dyn-chat-processor vllm|sglang`, and
  a fix for standalone binding routers does not automatically fix embedded
  routing.

## Key Files

| File | Role |
|---|---|
| `frontend/main.py` | Selects router mode and optional Python chat processor |
| `frontend/vllm_processor.py` | vLLM-native Python pre/postprocessor |
| `router/__main__.py` | Standalone Python router service |
| `thunderagent_router/__main__.py` | Custom registered router facade |
| `../../../lib/llm/src/entrypoint/input/common.rs` | Builds embedded Rust routing pipelines |
| `../../../lib/llm/src/kv_router/push_router.rs` | Worker selection and router-side tracker updates |
| `../../../lib/llm/src/protocols/common/preprocessor.rs` | `PreprocessedRequest`; tracker is `#[serde(skip)]` |
| `../../../lib/bindings/python/rust/llm/routed_engine.rs` | Python processor -> embedded Rust pipeline bridge |
| `../../../lib/bindings/python/rust/llm/kv.rs` | Binding-level router used by standalone/custom Python routers |
