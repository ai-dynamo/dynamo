# lib/metrics

`dynamo-metrics` holds what every component needs to emit Prometheus metrics: the metric and label names (`prometheus_names`), the registry and its hierarchy trait (`MetricsRegistry`, `MetricsHierarchy`, `create_metric`), builders for typed families (`prom_typed`), the frontend request metrics (`request`) with the load frames built from them (`load`), and the frontend pipeline instruments (`frontend_perf`). It sits below `dynamo-runtime`, which re-exports the registry, names, typed helpers and `frontend_perf` as `dynamo_runtime::metrics`, by name in `lib/runtime/src/metrics/mod.rs`; `request`, `load` and `environment_names` have no runtime path. The observability area in `.github/codeowners/areas.yaml` owns it, except that the frontend area owns `src/load.rs` and shares `src/request.rs` and `src/frontend_perf.rs`. Read [`.ai/metrics-guidelines.md`](../../.ai/metrics-guidelines.md) before you add or change a metric.

## Rules

- **Adding, renaming or removing a name.** Edit `src/prometheus_names.rs`, then regenerate the Python module with `cargo run -p dynamo-codegen --bin gen-python-prometheus-names`. `committed_python_matches_generated` in `lib/bindings/python/codegen` fails when `lib/bindings/python/src/dynamo/prometheus_names.py` is stale. A rename or removal is a user-visible change; follow the stability rules in the guidelines.
- **Adding a dependency.** The crate depends on `prometheus`, `dynamo-error`, `serde` and small utility crates only. Don't add `dynamo-runtime`, a transport, an HTTP server or `tokio`: crates that do not link the runtime, such as the router crates, must be able to link this crate. `prometheus_names` must not import the rest of the crate, so that it can be split off if a crate ever needs the names without `prometheus`.
- **Adding code that needs a `DistributedRuntime`, a component, an endpoint or a pipeline `Context`.** Not here. Implement `MetricsHierarchy` for the type in its own crate; the runtime does this for `DistributedRuntime`, `Namespace`, `Component` and `Endpoint`. The helpers that carry `RequestMetrics` on a request's pipeline context are in `lib/llm/src/preprocessor/request_metrics.rs`. Tests that build a runtime live in `lib/runtime/src/metrics/tests.rs`.
- **Adding an exporter or a route.** Not here. The OTLP exporter is `lib/runtime/src/metrics/otlp_export.rs`, and the `/metrics` scrape route is `lib/http-server/src/scrape.rs`.
- **Adding an environment variable that configures a metric.** Define the name in `src/environment_names.rs` and list it, with its documentation, in the registry in `lib/runtime/src/config/environment_names.rs`.
- **Setting a label.** `create_metric` adds `dynamo_namespace`, `dynamo_component`, `dynamo_endpoint` and `worker_id` from the hierarchy and rejects a caller that sets one of them. Take label names from `prometheus_names::labels`.
- **Changing a load frame.** `load::FrontendLoadFrame` crosses the wire to the Pool Relay between versions; `frames_decode_across_versions` in `lib/llm/src/frontend_load.rs` pins what a reader must accept.

## Frontend Serving-Load Metrics

The frontend's request metrics (`src/request.rs`) also feed the frame (`src/load.rs`) that `dynamo-llm` publishes for the Pool Relay's `SubscribeServingLoad` (`lib/llm/src/frontend_load.rs`), so `/metrics` and the Relay report the same numbers. Three are specific to that use and are experimental (see the Metrics Catalog):

- `dynamo_frontend_awaiting_first_token_requests{model}` (gauge): admitted requests with no output token yet. Requests producing output are `clamp_min(dynamo_frontend_active_requests - dynamo_frontend_awaiting_first_token_requests, 0)`; the frame derives that value (saturating) rather than exporting a second gauge.
- `dynamo_frontend_inflight_input_tokens{model, state}` (gauge, `state` is `awaiting_first_token` or `generating`): known prompt tokens of in-flight requests. Sum over `state` for the total.
- `dynamo_frontend_input_tokens_total{model}` (counter): prompt tokens of every admitted request, counted at tokenization. Use `rate()`.

Keep them consistent: one request transition updates every family through the request's shared `RequestMetrics` handle. Create a model's series when it registers, never from the publish path, which only reads. Before adding or changing a frontend metric, name its reader, check it is not a query over existing metrics or a sibling that should be a label, write its series count in the pull request, and add it to the Metrics Catalog.

## Test

```bash
cargo test -p dynamo-metrics
cargo test -p dynamo-codegen committed_python_matches_generated
cargo test -p dynamo-llm --lib frontend_load::tests   # load frame wire compatibility
cargo test -p dynamo-runtime --lib --features integration metrics::tests   # needs NATS; CI does not build this feature
```
