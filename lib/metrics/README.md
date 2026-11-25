<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-metrics

`dynamo-metrics` is the Prometheus vocabulary and registry Dynamo components share. It depends on `prometheus` and not on `dynamo-runtime`, so a crate can name and register metrics without linking the runtime. `dynamo-runtime` re-exports it as `dynamo_runtime::metrics`.

| Module | Contents |
|---|---|
| crate root | `MetricsRegistry`, the `MetricsHierarchy` trait, and `create_metric`, which names a metric from its hierarchy and adds the namespace, component, endpoint and worker labels |
| `prometheus_names` | Metric name prefixes, metric names and label names, as constants. The Python module `dynamo.prometheus_names` is generated from this file |
| `prom_typed` | Builds `prometheus` metric families from typed samples, so engine metrics are not rendered to text and parsed back |
| `frontend_perf` | Frontend pipeline instruments: in-flight requests and duration per stage, tokenize, template and detokenize timings, tokenizer cache counters |

The runtime keeps what needs it: the `MetricsHierarchy` implementations for `DistributedRuntime`, `Namespace`, `Component` and `Endpoint`, the transport and Tokio instruments, and the OTLP exporter.

Rules for adding or changing a metric are in [`.ai/metrics-guidelines.md`](../../.ai/metrics-guidelines.md).
