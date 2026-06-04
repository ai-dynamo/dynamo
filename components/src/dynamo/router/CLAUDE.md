# Frontend and Router Boundaries

Before changing standalone routing or its frontend integration, identify the
deployment topology. Dynamo has multiple router paths that share `KvRouter`
logic but do not share the same serialization, RPC, or process boundaries.

## Frontend/Router Boundary Model

### Topologies

1. **Integrated Rust frontend**

   ```
   dynamo.frontend
     -> Rust OpenAIPreprocessor
     -> in-process Rust KvPushRouter / KvRouter
     -> worker
     -> Rust DeltaGenerator
   ```

   `--router-mode kv` uses an in-process Rust router. There is no router RPC
   hop.

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
   deserializes it into `PreprocessedRequest`.

3. **Standalone or custom Python router service**

   ```
   frontend
     -> router service RPC
     -> binding-level KvRouter
     -> worker
   ```

   Examples: `python -m dynamo.router` and
   `python -m dynamo.thunderagent_router`. These own a binding-level
   `KvRouter` in another process.

### Guidance

- Do not infer metadata propagation from shared `KvRouter` logic. Values that
  cross Python serialization, RPC, or process boundaries need an explicit
  transport path in both the request and response directions.

- Keep framework metadata separate from engine-owned payloads. Put new Dynamo
  metadata under `extra_args["dynamo"]`. Do not add it to
  `disaggregated_params` or `engine_data`; the existing
  `disaggregated_params.worker_id` injection is compatibility behavior, not a
  pattern to extend.

- Strip internal framework metadata before returning an OpenAI response to the
  client unless it is intentionally exposed through `nvext`.

- Audit all three topologies when adding frontend-visible metadata. Test
  aggregated and disaggregated serving separately. A fix for the default Rust
  frontend does not automatically fix `--dyn-chat-processor vllm|sglang`, and
  a fix for standalone binding routers does not automatically fix embedded
  routing.

## RequestTracker Propagation and Timing Metrics

- Do not confuse router scheduler state with `RequestTracker`. KV overlap
  selection, active-request booking, output-block tracking, and cleanup can
  work while response timing, worker attribution, and agent traces are
  missing.

- In the integrated Rust frontend, `DeltaGenerator` creates an
  `Arc<RequestTracker>`, attaches it to `PreprocessedRequest`, and the router
  updates the same object.

- Never assume `RequestTracker` crosses a Python or process boundary.
  `PreprocessedRequest.tracker` is `#[serde(skip)]`, so the Python chat
  processor path does not preserve it when `RoutedEngine` deserializes a dict.
  Standalone and custom routers also cannot carry the `Arc<RequestTracker>`
  across Rust -> Python -> Rust or RPC boundaries automatically.

- If downstream timing must reach the frontend, transport a snapshot
  explicitly in the response data and merge it into the frontend-owned
  tracker. Apply the general metadata placement and response-stripping rules
  above.

## Key Files

| File | Role |
|---|---|
| `../frontend/main.py` | Selects router mode and optional Python chat processor |
| `../frontend/vllm_processor.py` | vLLM-native Python pre/postprocessor |
| `__main__.py` | Standalone Python router service |
| `../thunderagent_router/__main__.py` | Custom registered router facade |
| `../../../../lib/llm/src/entrypoint/input/common.rs` | Builds embedded Rust routing pipelines |
| `../../../../lib/llm/src/kv_router/push_router.rs` | Worker selection and router-side tracker updates |
| `../../../../lib/llm/src/protocols/common/preprocessor.rs` | `PreprocessedRequest`; tracker is `#[serde(skip)]` |
| `../../../../lib/bindings/python/rust/llm/routed_engine.rs` | Python processor -> embedded Rust pipeline bridge |
| `../../../../lib/bindings/python/rust/llm/kv.rs` | Binding-level router used by standalone/custom Python routers |
