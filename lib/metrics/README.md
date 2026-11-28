<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-metrics

`dynamo-metrics` is the Prometheus vocabulary, registry and frontend request metrics Dynamo components share. It depends on `prometheus` and not on `dynamo-runtime` or an HTTP server, so a crate can name, register and record metrics without linking either. `dynamo-runtime` re-exports the registry, names, typed helpers and `frontend_perf` as `dynamo_runtime::metrics`; take the request metrics and load frames from `dynamo_metrics`.

| Module | Contents |
|---|---|
| crate root | `MetricsRegistry`, the `MetricsHierarchy` trait, and `create_metric`, which names a metric from its hierarchy and adds the namespace, component, endpoint and worker labels |
| `prometheus_names` | Metric name prefixes, metric names and label names, as constants. The Python module `dynamo.prometheus_names` is generated from this file |
| `prom_typed` | Builds `prometheus` metric families from typed samples, so engine metrics are not rendered to text and parsed back |
| `request` | The frontend request metrics: `Metrics`, the per-request `RequestMetrics` and `InflightGuard`, the response collector, and the error classification the HTTP and gRPC services label requests with |
| `load` | The frontend load frame built from the request metrics and published to the Pool Relay |
| `frontend_perf` | Frontend pipeline instruments: in-flight requests and duration per stage, tokenize, template and detokenize timings, tokenizer cache counters |
| `environment_names` | Names of the environment variables that configure the request metrics |

What needs more than this crate stays with its owner: the `MetricsHierarchy` implementations, the transport and Tokio instruments and the OTLP exporter in `dynamo-runtime`; the `/metrics` scrape route in `dynamo-http-server`.

Rules for adding or changing a metric are in [`.ai/metrics-guidelines.md`](../../.ai/metrics-guidelines.md).
