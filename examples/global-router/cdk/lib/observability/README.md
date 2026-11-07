<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Existing-fleet observability

Deploy with the same entrypoint as the other independent layers:

```bash
pnpm exec cdk diff --app 'node --require ts-node/register bin/deploy.ts' -c layer=observability -c configPath=config/<observability-config>.json
pnpm exec cdk deploy --app 'node --require ts-node/register bin/deploy.ts' -c layer=observability -c configPath=config/<observability-config>.json
```

The stack imports a cluster; it owns no VPC, nodes, tables, global workloads or pools. It adds the CloudWatch observability EKS add-on, a single Prometheus/OpenTelemetry collector, log groups retained for seven days, regional dashboards, optional global dashboard, and alarms without notification actions. `resolveConflicts: NONE` deliberately prevents taking over a pre-existing add-on. Existing log groups must be imported if they already exist. The add-on collects fleet container logs and basic Container Insights; Application Signals and GPU exporters are disabled. The collector scrapes only the configured global/pool namespaces every 60 seconds. All images/add-ons are pinned in deployment configuration.

`ObservabilityConfig` is the input contract. `dashboardRegions` identifies the regions/clusters and displayed pool names; only the home observability stack sets `globalDashboard: true`. Each region publishes metrics to its own CloudWatch region. Global panels reference those regional metrics; they do not republish metrics or duplicate token totals.

## Metric semantics

The upstream OTel Prometheus receiver preserves histogram data. A transform extracts each selected histogram's cumulative sum and count, then a stateful processor converts each target's cumulative counters to deltas **before** CloudWatch dimensions aggregate replicas. Original histograms are dropped after conversion. CloudWatch `Sum` therefore means tokens/requests/seconds within the displayed period. Weighted latency means divide summed duration by summed count. No p95/p99 claim is made: histogram buckets are not retained by this bounded-cardinality pipeline.

The initial baseline after collector restart or the appearance of a new metric series is discarded; counts before the first two observations are not recovered. There is one collector replica and a Recreate update strategy to avoid duplicate scraping. This collector configuration is intended for operational monitoring, not lossless billing/accounting. Counter reset handling is delegated to the standard OTel cumulative-to-delta processor.

Global frontend metrics' `region` and `pool` identify the serving destination. `ClusterName` identifies ingress. Pool frontend `ServingRegion` and `Pool` dimensions come from Kubernetes scrape metadata. Token panels use only pool frontends, preventing global/pool double counting. Histogram count tracks cache-usage reporting coverage; an absent cache metric is not zero. Session affinity and KV-cache reuse are distinct.

Do not divide `cached_tokens_sum` by `input_tokens_total` per interval and label the result a cache-hit percentage. Input tokens are counted during preprocessing, while cached tokens are observed on the first response annotation containing cache usage. Requests crossing scrape boundaries therefore contribute to different intervals. The resulting interval ratio can exceed 100% even when cache usage for matched requests is valid. The dashboard shows both increments separately. `input_sequence_length_sum` is observed at first output and is not a universally paired denominator either. A trustworthy live ratio requires cached and prompt usage for the same reporting requests, emitted together (ideally one observation containing both values); divide aggregate sums, not averages of per-request percentages. Missing cache usage must remain distinguishable from a reported zero. Longer windows reduce boundary effects but do not fix this contract, and clamping hides the error.

Counter and histogram rollups omit target identities. Queue, concurrency, outstanding-token and worker-engine gauges additionally retain pod/model and, for worker metrics, worker/engine identity in CloudWatch dimensions. This prevents CloudWatch from averaging independent workers or models together before dashboard totals are computed. These are configured serving-model identities, not arbitrary request inputs; rolling pods/workers creates additional short-lived metric series. Other endpoint/request labels remain out of metric dimensions. Keep region/pool counts bounded and monitor metric inventory as the fleet expands. Container Insights adds its standard per-node/pod metrics and logs; logs expire after seven days. Dashboards and alarms have CloudWatch charges. No SNS, email or pager recipient is invented.

## Triage and limitations

Scrape-health alarms treat missing telemetry as unhealthy; error/throttle alarms tolerate idle periods. Request/affinity error alarms are absolute event thresholds; the mean-latency alarm requires at least ten completed observations per minute. Thresholds are simulation starting points, not production SLOs. Do not interpret a quiet graph as proof of healthy traffic; inspect scrape health alongside request volume.

Discovery/relay diagnosis uses application logs and service running-pod counts. These counts measure Running pods, not application readiness; separate frontend scrape/readiness metrics cover the serving path. There is no invented GDS membership metric or stale-discovery alarm. CPU/GPU inference performance is outside these dashboards. CloudWatch's Logs Insights widgets query regional application logs on refresh.

References: [OTel histogram extraction](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/v0.140.1/processor/transformprocessor/README.md#extract_sum_metric), [reset-aware delta conversion](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/v0.140.1/processor/cumulativetodeltaprocessor/README.md), [CloudWatch EKS add-on](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/install-CloudWatch-Observability-EKS-addon.html).

## Local pipeline verification

After building TypeScript, generate a collector config from `collectorConfig()` and pass its JSON path to `python3 testing/observability/verify-pipeline.py <config.json>`. The test starts a local Prometheus fixture and the pinned upstream collector with stdout export, checks exact histogram sum/count increments, verifies counter-reset handling and server-error classification, then stops the container. It requires Docker Desktop and free local port 18082.

## Queue, routing and worker views

The expanded dashboards discover destination pools and frontend/worker targets using CloudWatch SEARCH expressions. A new DGD in the configured pool namespace is automatically scraped: frontend HTTP ports and decode/prefill/worker system ports only. The current native worker allowlist covers the deployed mocker's vLLM-compatible running/waiting requests, KV occupancy, and preemption families. Other backend naming conventions need collector mapping before these panels apply; no universal backend coverage is claimed.

Queue/active/in-flight totals sum the **Average of each target/model/engine series** within each 60-second period. They never use temporal Sum as a gauge value. Gauge rollups with fewer dimensions remain available for compatibility but cannot produce a multi-target total. Per-engine KV occupancy uses Maximum and remains a fraction (1 = full), not a cache-hit percentage. Sampled totals are approximate under pod churn or misaligned scrapes; absent target data can undercount. Worker scrape health is shown beside load. Brief queue spikes between scrapes are invisible. The frontend awaiting-first-token gauge includes preprocessing, routing and prefill; native worker waiting requests describe the engine scheduler queue.

Routing requests retain destination pool/region and local/remote labels. Affinity store writes retain create/move/refresh/repair reasons. Pool request errors retain error type. Outstanding input tokens retain awaiting_first_token/generating state. These labels have bounded operational meaning. Mean input/output sequence length uses each histogram's own sum/count; neither is used as the denominator of a cache-hit percentage. Tokenizer means are split by operation so tokenize and detokenize timings are not mixed.

No Dynamo runtime changes are required for the exposed metrics in this expansion. Exact paired cache-hit percentages remain a frontend instrumentation gap. p95/p99 remain unavailable in this sums/counts CloudWatch export: retaining explicit buckets for a Prometheus-compatible query destination would be a separate bounded infrastructure change; summing bucket boundaries or asking CloudWatch for percentiles of summed durations would be incorrect.

### Multiple mocker workers per pod

With the tested runtime, Kubernetes discovery uses one identity per pod. Multiple mocker runtimes in the same pod share the pod-derived discovery identity and can also conflict on system HTTP port 9090. The collector cannot recover independent worker telemetry when identities and ports collide. Use one mocker worker per pod and vary actual pod replicas when testing capacity. Supporting multiple independent virtual workers inside one container would require explicit runtime/discovery identity and metrics endpoint changes; no such changes are included here.
