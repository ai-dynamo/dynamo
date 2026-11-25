# lib/metrics

`dynamo-metrics` holds what every component needs to emit Prometheus metrics: the metric and label names (`prometheus_names`), the registry and its hierarchy trait (`MetricsRegistry`, `MetricsHierarchy`, `create_metric`), builders for typed families (`prom_typed`), and the frontend pipeline instruments (`frontend_perf`). It sits below `dynamo-runtime`, which re-exports it as `dynamo_runtime::metrics`. The observability area in `.github/codeowners/areas.yaml` owns it. Read [`.ai/metrics-guidelines.md`](../../.ai/metrics-guidelines.md) before you add or change a metric.

## Rules

- **Adding, renaming or removing a name.** Edit `src/prometheus_names.rs`, then regenerate the Python module with `cargo run -p dynamo-codegen --bin gen-python-prometheus-names`. `committed_python_matches_generated` in `lib/bindings/python/codegen` fails when `lib/bindings/python/src/dynamo/prometheus_names.py` is stale. A rename or removal is a user-visible change; follow the stability rules in the guidelines.
- **Adding a dependency.** The crate depends on `prometheus` and small utility crates only. Don't add `dynamo-runtime`, a transport, an HTTP server or `tokio`: the router crates do not link the runtime and must be able to link this crate.
- **Adding code that needs a `DistributedRuntime`, a component or an endpoint.** Not here. Implement `MetricsHierarchy` for the type in its own crate; the runtime does this for `DistributedRuntime`, `Namespace`, `Component` and `Endpoint`. Tests that build a runtime live in `lib/runtime/src/metrics/tests.rs`.
- **Adding an exporter.** Not here. The OTLP exporter is `lib/runtime/src/metrics/otlp_export.rs`, and scrape routes belong to the server that serves them.
- **Setting a label.** `create_metric` adds `dynamo_namespace`, `dynamo_component`, `dynamo_endpoint` and `worker_id` from the hierarchy and rejects a caller that sets one of them. Take label names from `prometheus_names::labels`.

## Test

```bash
cargo test -p dynamo-metrics
cargo test -p dynamo-codegen committed_python_matches_generated
cargo test -p dynamo-runtime --lib --features integration metrics::tests   # needs NATS; CI does not build this feature
```
