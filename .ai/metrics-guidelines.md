<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Metrics Guidelines

Rules for adding or changing a metric, a log event, or a trace span in Dynamo. They keep
metrics useful, cheap to run, and stable for the people who build dashboards and alerts on
them.

The rules apply to every new or changed metric, log event, and span, in Rust, Python, and Go.
Existing code that breaks a rule is not a precedent: do not copy it, and do not rename an
existing metric as a side effect of another change (see
[Stability and deprecation](#stability-and-deprecation)). Engine metrics that Dynamo passes
through unchanged (`vllm:*`, `sglang:*`, `lmcache:*`, and the TensorRT-LLM `trtllm_*` metrics)
keep their engine's names and are out of scope. A metric that Dynamo computes is a Dynamo
metric, even when an engine prefix filter passes it through.

Each rule has an ID, such as **N2**, so that a review can cite it. An exception to a rule
needs a written reason in the pull request.

## Metric, log, or trace

**C1.** A metric exists only if something reads it continuously (an alert, an SLO, a
dashboard, or another Dynamo component) and its labels have a small, known set of values.
Everything else is a structured log event.

| If | Then |
|----|------|
| You can name the dashboard, alert, or component that reads it, and the labels are few and bounded | Metric |
| You need a request, tenant, session, or user ID to make it useful | Log event |
| The reader is a report or an investigation (billing, audit, post-incident) | Log event |
| You need a live signal and the detail behind it | One small metric for the aggregate, and one log event for the detail |
| You want to see where one request spent its time across components | Trace span (sampled) |

For example, failed requests are counted in
`dynamo_frontend_requests_total{model, endpoint, request_type, status, error_type}`, which
feeds an error-rate alert. The request ID and tenant of each failure go in a log event.
`record_unsupported_residency_event` in `lib/kv-router/src/indexer/metrics.rs` shows the
pair in code: it increments a counter whose labels have a fixed set of values, and it logs
the worker and rank as fields of a rate-limited event.

## Before you add a metric

Answer these questions in the pull request description. If an answer rules the metric out,
use a log event instead.

**A1. Reader.** Name the alert, SLO, dashboard panel, or component that reads the metric.
"It might be useful" is not a reader. In-repo readers include the Grafana dashboards under
`dev/observability/grafana_dashboards/` and `deploy/observability/`, and the planner's
queries in `components/src/dynamo/planner/monitoring/`.

**A2. Not a query.** If PromQL over existing metrics gives the same answer, do not add the
metric. Name the PromQL in the pull request when you rule it out.

**A3. Not an engine metric.** Do not re-export an engine metric under a `dynamo_` name;
Dynamo already passes engine metrics through. Add a Dynamo metric only when it measures
something the engine cannot see (the frontend's or the router's view), or when Dynamo
components need the same signal from every engine under one name. In that case, normalize
it across engines and document the mapping from each engine.

**A4. Not a sibling.** If the metric differs from an existing one by one dimension (a
modality, a direction, a role), add a label instead of a new name. When the existing metric
is stable, adding a label is a breaking change (**S1**): add the labeled metric under a new
name and deprecate the old one.

**A5. Static configuration.** Values that never change while the process runs (context
length, block size) go in one `*_info` metric with value 1 and each parameter as a label,
for example `dynamo_model_info{model, context_length, kv_cache_block_size}`.

**A6. Series count.** Write two numbers in the pull request, each as a formula and as a
number at a stated scale. Both are per process.

- **Label combinations:** multiply the largest number of values that each label you declare
  can reach. Name what drives each number: a fixed set (`status` has 3 values) or the
  deployment's configuration (served models, LoRA adapters). Labels that the runtime adds
  have one value per process; do not count them. Count every value a process can see over its
  life: a series stays exported until code removes it, so models or adapters that come and go
  accumulate.
- **Series:** label combinations times the series per combination: 1 for a counter or gauge,
  and buckets + 3 for a histogram (`+Inf`, `_sum`, and `_count`). Python `prometheus_client`
  also exports a `_created` series, so count 2 per counter combination and buckets + 4 per
  histogram combination. Each series costs memory in Prometheus, and the cost across the fleet
  is this number times the number of scraped processes.

For example, a histogram with 10 buckets and `model` as its only label has `models`
combinations and `models x (10 + 3)` series: 130 per frontend at 10 models. A counter with
`model` and a 3-value `status` has `models x 3` combinations, and as many series in Rust or Go.

If a metric can reach more than 100 label combinations in one process, justify it in the pull
request or use a log event or a trace instead (**L1**). A label whose values grow with the
number of workers is not allowed (**L5**).

**A7. Documentation.** Document the metric in the same pull request (**D1**, **D2**).

Write the answers as one block per metric in the pull request description:

```text
Metric:   dynamo_frontend_prefill_queue_wait_seconds (histogram)
Reader:   "Prefill queue wait p95" panel in the router dashboard; alert at 2 s
Labels:   model (served models; 10 in the reference deployment)
Series:   models combinations; models x (10 buckets + 3) = 130 series per frontend at 10 models
Query?:   No. Existing stage_duration_seconds does not split out queue wait.
Docs:     Added to the Metrics Catalog page
```

## Naming

Apply these rules to the exported name, which is what dashboards and alerts see. The APIs
can add a prefix or a suffix to the name in the code.

**N1.** The exported name is snake_case and starts with `dynamo_<subsystem>_`.

**N2.** The name ends with its unit in base form: `seconds` (not `ms`, `us`, or `ns`), `bytes`
(not `KB` or `GB`), `tokens`, or `ratio` for a value from 0 to 1 (not a percentage). For
example, `request_duration_seconds` and `kv_cache_usage_ratio`.

**N3.** A counter only goes up (requests, tokens, errors), ends in `_total`, and is charted
with `rate()`. A counter with a unit puts the unit before `_total`, as in
`detokenize_seconds_total`.

**N4.** A gauge can go up and down (in-flight requests, cache usage) and does not end in
`_total`.

**N5.** Do not put a label name or a label value in the metric name. Use a label.

**N6.** The help text says what is measured and matches the value. If the code measures
something different from what the name says, fix the name. Changing what a metric measures
means a new name.

```text
# BAD -- unit not in base form, percentage, counter without _total, label value in the name
dynamo_frontend_queue_wait_ms
dynamo_component_cache_usage_percent
dynamo_frontend_rejected_requests
dynamo_frontend_images_per_request

# GOOD
dynamo_frontend_queue_wait_seconds
dynamo_component_cache_usage_ratio
dynamo_frontend_rejected_requests_total
dynamo_frontend_multimodal_items_per_request{modality}
```

## Labels

**L1.** Aim for fewer than 10 values per label. A label that can reach about 100 values
does not belong on a metric; use a log event or a trace.

**L2.** Never use a request, session, user, or tenant ID, a prompt, a URL path, or a raw
client-supplied value as a label. Map a client-supplied value to a known, bounded set
first, as `categorizeError` does in `deploy/operator/internal/observability/metrics.go`.

**L3.** The `model` label comes from the set of served models, never from the raw value in
the request. A request that names a model that is not served uses one fixed value. In the
frontend, resolve the label with `ModelManager::metric_model_for`, which returns the
existing sentinel `unknown_model`, so that all frontend metrics agree. A component with no
existing sentinel uses `unknown`, as the endpoint picker's `set_served_model` does.

**L4.** Summing over a label must give a number that means something. A label that mixes
different quantities, such as `kind` with the values `requests` and `bytes`, should be
separate metrics. Static `*_info` metrics are the exception.

**L5.** Do not declare a label whose values grow with the number of workers, such as
`worker_id` or `dp_rank` on a frontend, router, or planner metric: every such process would
then hold series for the whole fleet. Export a per-worker value from the worker's own
metrics instead, where the `worker_id` that the runtime adds identifies the worker; that
label has one value per process and is not affected by this rule. On a worker's own metrics,
add a `dp_rank` label only when an autoscaler or the router needs per-rank values.

**L6.** Pre-create the expected values of bounded labels (such as `status` and `endpoint`)
at startup, so that `rate()` and `absent()` alerts work from the first scrape. Create the
`model` values when the model registers, because they are not known at startup.

**L7.** Histograms are the main cost. Use about 10 buckets, put a bucket edge at every alert
threshold, and use `model` as the only label that you declare.

## Registration

**R1.** Register a metric at startup in every process that updates it, not on first use.
A labeled family has no output until it has at least one child, so registration alone does
not make it visible: pre-create its children as well (**L6**).

**R2.** Do not add a metric that no production code updates.

**R3.** When you add, rename, or remove a name in `lib/runtime/src/metrics/prometheus_names.rs`,
regenerate its Python mirror in the same pull request with
`cargo run -p dynamo-codegen --bin gen-python-prometheus-names`. Never edit the mirror by hand.

## Patterns to follow

The right-hand column shows the shape to aim for in new metrics. It is not a list of metrics
that exist today.

| Instead of | Do this |
|------------|---------|
| `images_per_request`, `videos_per_request`, `audio_per_request` | One `multimodal_items_per_request{modality}` |
| `model_context_length`, `model_kv_cache_block_size`, `model_max_num_seqs` | One `model_info{model, context_length, kv_cache_block_size, max_num_seqs}` with value 1 |
| `num_prefill_replicas`, `num_decode_replicas` | One `num_replicas{sub_component_type}` |
| `router_overhead_total_ms`, or a "percent" gauge that holds 0 to 1 | `_seconds` and `_ratio` names |
| A counter of events above a threshold, next to a histogram with that bucket edge | `increase(x_count[5m]) - increase(x_bucket{le="T"}[5m])` in PromQL, where `T` is an existing bucket edge as exported |

## Stability and deprecation

Renaming or reshaping a metric breaks dashboards and alerts, often silently: a panel just
shows no data. So every metric has one of two stability levels.

- **Experimental:** marked as experimental in the Metrics Catalog. It can change or
  disappear in any release, with a release note.
- **Stable:** documented in the Metrics Catalog without an experimental mark. A metric
  becomes stable after at least one release as experimental, once a maintainer has
  reviewed its name, unit, type, and labels, and a test checks that the code updates it.

A new metric therefore starts as experimental. A catalog entry with no experimental mark is
stable, including an entry whose name breaks these rules.

**S1.** Do not rename a stable metric, change its type or unit, or add or remove labels. Add
a new metric and deprecate the old one. Adding a new value to an existing label is allowed
when existing queries stay correct (for example, `sum()` over the label still means the
same thing).

**S2.** To deprecate a stable metric, mark it deprecated in the Metrics Catalog and in its
help text, name its replacement or a PromQL recipe, and name the release that removes it.
Move in-repo readers (**A1**) to the replacement, and state the deprecation in the pull
request description so that it can go into the release notes.

**S3.** Keep a deprecated metric for at least two releases after the announcement, then
remove it. Never remove it in a patch release.

**S4.** Remove a metric only if every question it answers can still be answered from what
remains, with the same meaning, measurement point, label granularity, scrape target,
timeliness, and availability. A metric that was never registered or never updated can be
removed without deprecation.

Not promised as stable: engine metric names and labels (vLLM, SGLang, and TensorRT-LLM
metrics pass through under their own names), experimental metrics, per-worker debugging
gauges, and metrics that are not in the Metrics Catalog.

## Documentation

**D1.** The [Metrics Catalog](../docs/fern/pages/reference/observability/metrics-catalog.mdx)
is the source of truth for metric documentation, and a metric that is not in it is
unsupported. In the same pull request, add or update the metric's entry: its type, unit,
labels and their expected values, and stability level. Operator metrics go in
[Operator Metrics](../docs/fern/pages/reference/observability/operator-metrics.mdx) instead.
If no section fits the metric's family, add one. If the Metrics Inventory page
(`metrics-inventory.md` in the same directory) exists, add or update the metric's row there
as well.

- Mark a new metric experimental: start its description with `**Experimental.**`.
- Mark a deprecated metric with `deprecated={true}` on its `ParamField`, and name the
  replacement and the removal release in its description.

**D2.** Update the `AGENTS.md` nearest to the code that owns the metric to say what the
metric is and how it can be used. The "Frontend Serving-Load Metrics" section of
`lib/llm/AGENTS.md` is an example.

## Logs and traces

**T1.** Logs are structured JSON, one event per line. Write each event so that its details
are separate fields when the JSONL console format is on
(`DYN_LOGGING_CONSOLE_FORMAT=jsonl`; the default is readable text).

- Rust: use `tracing` macros with `key = value` fields and a fixed message, for example
  `tracing::warn!(worker_id, dp_rank, "Ignoring KV event")`.
- Python: the log bridge forwards only the formatted message to Rust, so fields passed with
  `extra=` are lost. Put the values in lazy `%s` arguments, as `.ai/python-guidelines.md`
  requires.

**T2.** Request-path log events carry `x_request_id`, `trace_id`, and `span_id`.

- Rust: emit the event inside the request's span; the JSONL formatter takes these IDs from
  the span. Attach the span to spawned tasks with `.instrument(...)` or `in_current_span()`.
- Python: include the request ID with `context.id()`; `context.trace_id` and
  `context.trace_headers()` give the trace context.

**T3.** Use a trace span, which is sampled, to show where one request spent its time across
components. Do not add a metric label to get that view.
