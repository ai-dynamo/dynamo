---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: OTLP Metrics Export
subtitle: Push the same metrics Dynamo serves at /metrics to an OpenTelemetry collector, and how the Prometheus-to-OTLP conversion behaves.
---

Dynamo can push its metrics to an OpenTelemetry collector over OTLP, in addition to serving them at `/metrics`. Both surfaces carry the same metrics, including engine metrics from vLLM, SGLang, and TensorRT-LLM.

Export is **off by default**. With it off, nothing is collected and nothing is sent.

## Enable it

Set `OTEL_METRICS_EXPORTER=otlp`. That alone is sufficient — export starts with the runtime and does not depend on the system status server, so it works whether or not `DYN_SYSTEM_PORT` is enabled.

| Variable | Meaning |
|---|---|
| `OTEL_METRICS_EXPORTER` | Set to `otlp` to enable. Any other value disables it. |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | Collector URL. Falls back to `OTEL_EXPORTER_OTLP_ENDPOINT`. |
| `OTEL_EXPORTER_OTLP_METRICS_PROTOCOL` | Transport. Only `grpc` is implemented; any other value fails at startup rather than exporting over the wrong transport. Falls back to `OTEL_EXPORTER_OTLP_PROTOCOL`. |
| `OTEL_METRIC_EXPORT_INTERVAL` | Export period in milliseconds. Default `60000`. |
| `OTEL_EXPORTER_OTLP_METRICS_HEADERS` | Headers for authenticated collectors, as `key=value` pairs separated by commas. Falls back to `OTEL_EXPORTER_OTLP_HEADERS`. |
| `OTEL_RESOURCE_ATTRIBUTES` | Resource attributes applied to every export, as `key=value` pairs separated by commas. |

These are the standard OpenTelemetry variables, and they mean the same thing for metrics, traces, and logs.

<Info>
  A signal-specific variable **replaces** the generic one rather than merging with it, per the OTLP exporter specification. Setting `OTEL_EXPORTER_OTLP_METRICS_HEADERS` means `OTEL_EXPORTER_OTLP_HEADERS` is not consulted for metrics.
</Info>

Header values are split on the first `=` only, so a bearer token containing base64 padding survives intact. Percent-decoding is not applied.

## Relationship to `/metrics`

Enabling OTLP export does not change `/metrics` in any way. The scrape path renders exactly what it always has: Dynamo's own metrics encoded by the Prometheus text encoder, followed by each engine's exposition text appended verbatim.

The two surfaces are fed independently — the scrape collects when Prometheus scrapes it, the exporter collects on its own interval — so enabling export adds one collection per interval and changes nothing about the scrape.

## Conversion principles

Dynamo is a **converter**, not a producer of OTel-native metrics. It follows the [OpenTelemetry Prometheus and OpenMetrics compatibility specification](https://opentelemetry.io/docs/specs/otel/compatibility/prometheus_and_openmetrics/), specifically the *Prometheus Metric points to OTLP* direction.

That distinction matters because most metric names are not Dynamo's: engine metrics come from vLLM, SGLang, and TensorRT-LLM, and rewriting a third party's names means guessing at their intent.

### Names are not altered

A metric exported over OTLP carries the same name it has at `/metrics`. `dynamo_component_requests_total` stays `dynamo_component_requests_total`; `vllm:request_duration_seconds` stays as it is.

<Warning>
  The compatibility specification describes both directions, and they are **not** symmetric. Suffixes such as `_total` and unit words are *added* when converting OTLP to Prometheus. Coming the other way they are left alone. Rules quoted from the *OTLP Metric points to Prometheus* section do not apply here.
</Warning>

Keeping names identical also means a metric can be correlated across both surfaces during a migration, and that a downstream collector re-exporting to Prometheus does not have to reverse a transformation Dynamo applied.

### Units come from UNIT metadata

`Metric.unit` is populated from the Prometheus UNIT metadata the client declares, translated to its UCUM abbreviation — `seconds` becomes `s`, `bytes` becomes `By`. A unit outside the specification's table is passed through unchanged. Units are never inferred from the metric name.

### Type conversions

| Prometheus | OTLP |
|---|---|
| Counter | Monotonic cumulative Sum |
| Gauge | Gauge |
| Histogram | Histogram, with `+Inf` dropped from the bounds and bucket counts de-cumulated |
| Summary | Summary |
| Info, StateSet | **Non-monotonic Sum** — the value of 1 is a count, so it must add up when a label is aggregated away |
| Unknown / untyped | Gauge |

The original Prometheus type word travels with each metric in `metric.metadata` under the `prometheus.type` key.

### Timestamps and start times

- A `_created` series becomes the parent metric's `start_time_unix_nano`, matched by label set, and is not exported as a metric of its own. This is what lets a backend distinguish a counter reset from a jump.
- A sample carrying its own timestamp is reported as observed at that time. Otherwise the export instant is used.
- Gauges carry no start time.

### Dropped and flagged data

- A histogram or summary with no `_count` sample is dropped: its buckets and total cannot be reconciled, and a quantile computed from it would be meaningless.
- A `NaN` value is reported as no-recorded-value with the value left unset, rather than as a raw `NaN` that some backends reject.
- If a collector responds with `partial_success`, the count of rejected data points is logged. A successful response is not assumed to mean the data was kept.

### Known gaps

- **Exemplars are not converted**, so a backend cannot jump from a histogram bucket to the trace that produced it. This is a limitation of the data model rather than a choice: upstream Prometheus defines `Bucket.exemplar` and `Counter.exemplar`, but the `prometheus` Rust crate vendors a reduced copy of that schema without them, so an exemplar has nowhere to arrive. Nothing is being discarded.
- **`target_info` and `job`/`instance` handling** are scrape-side concerns and do not apply, since Dynamo pushes rather than being scraped by the collector.
