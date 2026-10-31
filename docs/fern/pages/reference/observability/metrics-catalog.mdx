---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Metrics Catalog
subtitle: Every metric Dynamo defines, grouped by the process that exposes it, with type, labels, and meaning.
---

Dynamo exposes metrics in Prometheus exposition format at the `/metrics` HTTP endpoint. This page lists every metric that Dynamo's own code defines; use it to look up a metric name you see in a scrape, dashboard, or alert. The core metrics have a detailed entry with semantics and PromQL examples, and the others are listed in tables. [Metric Labels](metric-labels.mdx) documents the label values. For setup and dashboards, see the [Metrics guide](../../cli/operations/observability.mdx#view-metrics-and-dashboards).

This page does not list:

- Inference engine metrics (`vllm:*`, `sglang:*`, `trtllm_*`) that the engine itself defines. See [Engine Metrics](#engine-metrics).
- Kubernetes operator metrics. See [Operator Metrics](operator-metrics.mdx).
- NIXL transfer metrics, which NIXL serves on its own port. See the [NIXL telemetry documentation](https://github.com/ai-dynamo/nixl/blob/main/docs/telemetry.md).
- Velo metrics (`velo_*`), which the Velo library defines. Dynamo processes expose them when `DYN_RESPONSE_PLANE=velo`. See the [Velo metrics reference](https://github.com/ai-dynamo/velo/blob/main/docs/src/appendix/metrics.md).
- KVBM metrics, including the block-pool metrics in the `kvbm-logical` crate. KVBM was removed in v1.7.0; see the documentation for v1.6 and earlier.

## Metric Types

The type of each metric is a [Prometheus metric type](https://prometheus.io/docs/concepts/metric_types/):

- **counter** — a cumulative value that only increases (or resets to zero on restart), e.g. total requests.
- **gauge** — a single value that can go up or down, e.g. in-flight requests.
- **histogram** — samples observations into configurable buckets and exposes `_bucket`, `_sum`, and `_count` series, e.g. request duration. Labeled histograms/counters/gauges register a metric *family*, not one series. The frontend latency and sequence-length histograms take their bucket ranges from environment variables — see [Histogram buckets](../components/frontend-configuration.mdx#histogram-buckets).

<Info>
  Labeled metrics (`HistogramVec`, `CounterVec`, `GaugeVec`) register a metric *family*, not individual time series. For request-populated metrics, a series for a given label combination appears at `/metrics` only after the first matching request. Other families create or remove series from runtime state; their entries below document those semantics.
</Info>

In the tables on this page:

- **Labels** lists the labels specific to the metric. Labels shared by every metric in a section are listed in the section introduction.
- **Status** is **Current**, **Experimental** (a preview metric whose name or meaning may change), or **Deprecated** (still emitted; the description names the replacement). Detailed entries say **Experimental** in their text and mark deprecated metrics with a badge.

Metrics defined in Python, such as the planner metrics, also expose a `_created` series for each counter and histogram.

## Where each family is exposed

| Family | Emitted by | Scrape endpoint |
|--------|------------|-----------------|
| `dynamo_frontend_*` | HTTP frontend (`python -m dynamo.frontend`) | `/metrics` on `DYN_HTTP_PORT` (default 8000) |
| `dynamo_tokio_*`, `dynamo_request_plane_*`, `dynamo_transport_*` | Frontend | `/metrics` on `DYN_HTTP_PORT` |
| `dynamo_component_*` | Backend workers, standalone router, KV state agent, and the frontend's router and runtime metrics | `/metrics` on `DYN_SYSTEM_PORT` (must be set); `/metrics` on `DYN_HTTP_PORT` for the frontend |
| `dynamo_router_*` | Frontend | `/metrics` on `DYN_HTTP_PORT` |
| `dynamo_work_handler_*`, `dynamo_backend_admission_*`, `dynamo_embedding_*` | Backend workers | `/metrics` on `DYN_SYSTEM_PORT` |
| `dynamo_quic_response_*` | Every process, with `DYN_RESPONSE_PLANE=quic` | `/metrics` on `DYN_SYSTEM_PORT`; `/metrics` on `DYN_HTTP_PORT` for the frontend |
| `dynamo_planner_*` | Planner (`python -m dynamo.planner`) | `/metrics` on the planner's own port; see [Planner Metrics](#planner-metrics) |
| `dynamo_kvindexer_*`, `dynamo_kvrouter_*` | Standalone KV indexer (`python -m dynamo.indexer`) | `/metrics` on `--port` |
| `dynamo_epp_*` | Endpoint picker (Gateway API Inference Extension routing) | `/metrics` on `DYN_EPP_METRICS_PORT` (default 9090) |
| `dynamo_global_frontend_*` | Global Frontend (`global-frontend`), which forwards each request to a pool frontend | `/metrics` on `--port` (default 8000) |
| `dynamo_operator_*` | Kubernetes operator | See [Operator Metrics](operator-metrics.mdx) |

Backend workers expose `dynamo_component_*` only when `DYN_SYSTEM_PORT` is set (disabled by default; the Kubernetes operator typically sets `9090`, local examples use `8081`). See [Environment Variables](environment-variables.mdx#system-and-metrics).

## Frontend Metrics

Emitted by `python -m dynamo.frontend` at `/metrics` on the HTTP port (`--http-port`, default `8000`). The frontend serves its [router](#router-metrics) and [runtime](#every-process) metrics on the same port and does not use `DYN_SYSTEM_PORT`. Frontend metrics carry no automatic labels.

- `--metrics-prefix` (`DYN_METRICS_PREFIX`) replaces the `dynamo_frontend` prefix of the `dynamo_frontend_*` metrics, except `dynamo_frontend_stage_*`, `dynamo_frontend_tokenize_seconds`, `dynamo_frontend_template_seconds`, `dynamo_frontend_detokenize_*`, `dynamo_frontend_tokenizer_cache_*`, `dynamo_frontend_worker_*`, `dynamo_frontend_router_*`, `dynamo_frontend_lora_*`, and `dynamo_frontend_event_loop_*`, which keep their names.
- With `--kserve-grpc-server`, the frontend serves metrics on `DYN_GRPC_METRICS_PORT` (default `8788`) instead. The model configuration, migration, router, and routing overhead metrics are not exposed in that mode.

A detailed entry that does not name its labels carries only a `model` label. For endpoints that canonicalize aliases, that label holds the primary served model name — the first `--served-model-name` value — so a request sent to an alias is counted under the primary. A name the frontend has not registered is counted under `unknown_model`.

<ParamField path="dynamo_frontend_active_requests" type="gauge">
  Requests currently being handled by the frontend, from HTTP handler entry until the response stream completes. The top-level in-flight count with no stage breakdown.
</ParamField>

<ParamField path="dynamo_frontend_stage_requests" type="gauge">
  Requests currently in a given frontend pipeline stage. Labeled by `stage` and `phase` — see [Stage values](metric-labels.mdx#stage-values) and [Phase values](metric-labels.mdx#phase-values).
</ParamField>

<ParamField path="dynamo_frontend_inflight_requests" type="gauge" deprecated={true}>
  Inflight requests. Kept for backward compatibility; prefer `dynamo_frontend_active_requests`, which has identical semantics with a clearer name.
</ParamField>

<ParamField path="dynamo_frontend_queued_requests" type="gauge" deprecated={true}>
  Requests in the HTTP processing queue. Kept for backward compatibility; the "waiting for first token" window is now the sum of `dynamo_frontend_stage_requests` across the `preprocess`, `route`, and `dispatch` stages.
</ParamField>

<ParamField path="dynamo_frontend_awaiting_first_token_requests" type="gauge">
  **Experimental.** Admitted requests that have not produced an output token yet: being tokenized, queued, routed, or prefilled. Requests to endpoints that never stream tokens (embeddings, images, videos, tensors) stay here until they finish. Unit: requests. Labeled by `model`. Requests producing output tokens are `clamp_min(dynamo_frontend_active_requests - dynamo_frontend_awaiting_first_token_requests, 0)`; the clamp covers the two gauges being sampled a moment apart.
</ParamField>

<ParamField path="dynamo_frontend_inflight_input_tokens" type="gauge">
  **Experimental.** Known prompt tokens of in-flight requests. Unit: tokens. Labeled by `model` and `state`: `awaiting_first_token` (the requests in `dynamo_frontend_awaiting_first_token_requests`) or `generating`. Summing over `state` gives all in-flight prompt tokens. A request still being tokenized contributes zero.
</ParamField>

<ParamField path="dynamo_frontend_input_tokens_total" type="counter">
  **Experimental.** Prompt tokens of admitted requests, counted at tokenization, including requests that fail or are cancelled before producing output. Unit: tokens. Labeled by `model`. Unlike the sum of `dynamo_frontend_input_sequence_tokens`, which is observed at the first output token, this counts every admitted request when it is tokenized.
</ParamField>

<ParamField path="dynamo_frontend_disconnected_clients" type="gauge">
  Client disconnects since the frontend started. The value only increases. No labels.
</ParamField>

<ParamField path="dynamo_frontend_input_sequence_tokens" type="histogram">
  Prompt length in tokens per request, recorded at the first output token.
</ParamField>

<ParamField path="dynamo_frontend_cached_tokens" type="histogram">
  Prefix-cache hit tokens per request, as reported by the worker.
</ParamField>

<ParamField path="dynamo_frontend_tokenizer_cache_hits_total" type="counter">
  L1 tokenizer prefix-cache hits across all models. One outcome is recorded per encode operation or batch item. No labels.
</ParamField>

<ParamField path="dynamo_frontend_tokenizer_cache_misses_total" type="counter">
  L1 tokenizer prefix-cache misses across all models. No labels.
</ParamField>

<ParamField path="dynamo_frontend_tokenizer_cache_cached_tokens_total" type="counter">
  Tokens returned from the L1 tokenizer prefix cache. Labeled by `model`.
</ParamField>

<ParamField path="dynamo_frontend_tokenizer_cache_uncached_tokens_total" type="counter">
  Tokens freshly encoded after an L1 tokenizer cache lookup. Labeled by `model`.
</ParamField>

Use this PromQL expression to calculate the five-minute token reuse ratio by model:

```promql
sum by (model) (rate(dynamo_frontend_tokenizer_cache_cached_tokens_total[5m]))
/
(
  sum by (model) (rate(dynamo_frontend_tokenizer_cache_cached_tokens_total[5m]))
  +
  sum by (model) (rate(dynamo_frontend_tokenizer_cache_uncached_tokens_total[5m]))
)
```

The ratio is defined only when the model has an active L1 cache and observed tokens. Partial hits increment both token counters. When the cache is disabled with `DYN_TOKENIZER_CACHE=0`, the per-model series are never created, so they are absent from the scrape rather than reported as zero; detect that state with `absent()`, since a `== 0` comparison never matches a missing series. Only the literal value `0` disables the cache; values such as `false` or `off` leave it enabled. With the cache enabled, the counters remain zero when encoding fails or the tokenizer has no registered special-token boundaries.

<ParamField path="dynamo_frontend_inter_token_latency_seconds" type="histogram">
  Inter-token latency in seconds.
</ParamField>

<ParamField path="dynamo_frontend_output_sequence_tokens" type="histogram">
  Output sequence length in tokens.
</ParamField>

<ParamField path="dynamo_frontend_output_tokens_total" type="counter">
  Total output tokens generated.
</ParamField>

<ParamField path="dynamo_frontend_request_duration_seconds" type="histogram">
  End-to-end LLM request duration in seconds.
</ParamField>

<ParamField path="dynamo_frontend_requests_total" type="counter">
  Requests that finished, by outcome. Labeled by `model`, `endpoint`, `request_type`, `status`, and `error_type`.
</ParamField>

<ParamField path="dynamo_frontend_time_to_first_token_seconds" type="histogram">
  Time to first token in seconds.
</ParamField>

The frontend also records these request and pipeline metrics:

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_requests_started_total` | counter | `model`, `endpoint`, `request_type` | Requests accepted by a frontend handler. | Current |
| `dynamo_frontend_failures_total` | counter | `class`, `reason` | Requests that ended in an error, by error class and reason. Cancellations are not counted. | Current |
| `dynamo_frontend_embedding_latency_seconds` | histogram | `model` | Latency of successful `/v1/embeddings` requests. | Current |
| `dynamo_frontend_stage_duration_seconds` | histogram | `stage` | Time spent in each pipeline stage: `preprocess`, `route`, or `transport_roundtrip`. | Current |
| `dynamo_frontend_tokenize_seconds` | histogram | — | Prompt tokenization time per request. | Current |
| `dynamo_frontend_template_seconds` | histogram | — | Chat template rendering time per request. | Current |
| `dynamo_frontend_tokenizer_latency_ms` | histogram | `operation` | Tokenizer latency in milliseconds, for `tokenize` and `detokenize`. | Current |
| `dynamo_frontend_detokenize_total_us` | counter | — | Total detokenization time in microseconds. | Current |
| `dynamo_frontend_detokenize_token_count` | counter | — | Tokens detokenized. | Current |

### Multimodal Requests

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_images_per_request` | histogram | `model` | `image_url` content parts per request. Text-only requests record 0. | Current |
| `dynamo_frontend_videos_per_request` | histogram | `model` | `video_url` content parts per request. | Current |
| `dynamo_frontend_audio_per_request` | histogram | `model` | `audio_url` content parts per request. | Current |
| `dynamo_frontend_image_tokens_per_request` | histogram | `model` | Image placeholder tokens per request that contains images. | Current |

### Migration

Metrics for [request migration](../../kubernetes/fault-tolerance/request-migration.md), the mechanism that continues in-flight requests on a healthy worker when the original worker fails. See [Request Migration Architecture](../../developer-guide/knowledge-base/concepts/fault-tolerance/request-migration-architecture.md) for the internals.

<ParamField path="dynamo_frontend_model_migration_total" type="counter">
  Total request migrations due to worker unavailability. Labeled by `model` and [`migration_type`](metric-labels.mdx#metric-specific-labels) (`new_request` for an initial connection failure, `ongoing_request` for a mid-stream disconnection).
</ParamField>

<ParamField path="dynamo_frontend_model_migration_duration_seconds" type="histogram">
  Time from detecting a migratable worker failure until the replacement response stream is established, migration ends without recovery, or the request is cancelled. Labeled by `model`, [`migration_type`](metric-labels.mdx#metric-specific-labels), and [`outcome`](metric-labels.mdx#metric-specific-labels). Use the histogram's `_count` series to count completed migration events by outcome. The duration excludes work completed before the failure and generation after the replacement stream is established.
</ParamField>

<ParamField path="dynamo_frontend_model_migration_max_seq_len_exceeded_total" type="counter">
  Total times migration was disabled for a request because its sequence length exceeded `--migration-max-seq-len`. A rising value may indicate the limit needs adjustment. Labeled by `model`.
</ParamField>

### Cancellation and Rejection

Counters for requests that end early: [cancellations](../../developer-guide/knowledge-base/concepts/fault-tolerance/request-cancellation-architecture.md) (the client or frontend aborts an in-flight request) and [rejections](../../kubernetes/fault-tolerance/request-rejection.md) (the Frontend sheds load with HTTP 529 by default when all workers are busy). The worker-side counterparts are under [Component metrics](#component-metrics).

<ParamField path="dynamo_frontend_model_cancellation_total" type="counter">
  Total request cancellations detected by the frontend (client disconnect or stream close). Labeled by `model`, `endpoint` (the API route — `chat_completions`, `completions`, `embeddings`, …), and [`request_type`](metric-labels.mdx#metric-specific-labels) (`unary` or `stream`).
</ParamField>

<ParamField path="dynamo_frontend_model_rejection_total" type="counter">
  Total overload responses surfaced to the client as HTTP 529 by default. Incremented for worker-scoped `WorkerOverloaded` and pool-scoped `ResourceExhausted` errors. Labeled by `model` and `endpoint` (the API route).
</ParamField>

### Per-Worker Load and Timing Gauges

`dynamo_frontend_worker_*` gauges appear once workers register and begin serving. They live on the frontend's local registry (not component-scoped) and do **not** carry `dynamo_namespace` or `dynamo_component` labels. Frontend-only — not available on the standalone router. Labeled by `worker_id` (the worker's discovery ID in decimal), `dp_rank`, and `worker_type`.

Only the KV router writes the three `dynamo_frontend_worker_active_*` gauges, so they have no series in other router modes; the three `dynamo_frontend_worker_last_*` gauges are populated in every mode. At each HTTP scrape, these six gauges include only workers in the current committed router membership that pass hard availability checks, and only their current data-parallel (DP) ranks. The timing gauges also allow `dp_rank="none"` when the rank is unknown, provided the worker remains available. The `worker_type` label describes serving attribution: an Encode worker serving Chat/Completions reports its load and timing samples as `decode`, while the [router worker inventory](#router-worker-inventory) gauges below retain its `encode` topology role. Removed or excluded workers disappear from the output; late request completions or load updates cannot restore their exposed series.

<ParamField path="dynamo_frontend_worker_active_decode_blocks" type="gauge">
  Active KV cache decode blocks per worker.
</ParamField>

<ParamField path="dynamo_frontend_worker_active_prefill_tokens" type="gauge">
  Active prefill tokens queued per worker.
</ParamField>

<ParamField path="dynamo_frontend_worker_active_requests" type="gauge">
  Active requests booked per worker, split by an additional [`request_phase`](metric-labels.mdx#metric-specific-labels) label. A request counts as `prefill` from booking until its worker returns the first output token, then as `decode` until it completes. Sum both phases for the worker's total concurrency. The count is the router's view, including requests booked by peer routers when router replica sync is enabled.
</ParamField>

<ParamField path="dynamo_frontend_worker_last_time_to_first_token_seconds" type="gauge">
  Last observed time to first token per worker, in seconds.
</ParamField>

<ParamField path="dynamo_frontend_worker_last_input_sequence_tokens" type="gauge">
  Last observed input sequence length per worker.
</ParamField>

<ParamField path="dynamo_frontend_worker_last_inter_token_latency_seconds" type="gauge">
  Last observed inter-token latency per worker, in seconds.
</ParamField>

### Router Worker Inventory

These frontend-only metrics describe each Frontend's independent view of normalized base model-card discovery and router admission. They do not report Kubernetes Pod existence, and LoRA overlays do not count as additional workers. Both metrics carry `model`, `target_namespace`, `target_component`, `target_endpoint`, and `worker_type` labels. The target labels identify the discovered serving endpoint.

Admission uses first-wins semantics within each Frontend's model and target group. The first observed checksum cohort remains the incumbent; later incompatible worker IDs are excluded individually and do not remove or replace it. If the incumbent cohort disappears, the next remaining cohort can become its successor after initialization.

<ParamField path="dynamo_frontend_router_workers" type="gauge">
  Number of unique physical worker IDs in a model and target group, labeled by `state`. DP ranks do not increase the count. The `available`, `pending`, and `excluded` states are disjoint; `discovered` is their sum. `available` requires successfully committed worker-set membership and current hard availability; `pending` means discovery has observed the worker but initialization has not completed; `excluded` means discovered but currently unavailable for routing.
</ParamField>

Hard availability uses the committed group's client and, when a prefill or encode hop targets that group, the hop's local availability. A matching hop that is still activating reports no available workers. If no hop targets the group, its own client availability applies. These counts describe individual worker groups; use `dynamo_frontend_model_ready` to check whether the frontend has a complete routable serving topology.

Each currently observed group emits all four counts, including zero values for states with no workers. When discovery withdraws a group after its last worker disappears, its count and per-worker state series disappear on the next scrape. Retired rolling-update namespaces are not retained as permanent zero-count groups. Groups never observed by that frontend also have no series. Do not aggregate these counts across Frontends as worker-global readiness; separate Frontends can choose different incumbents from the same discovered workers.

<ParamField path="dynamo_frontend_router_worker_state" type="gauge">
  One sample with value `1` for each discovered worker, additionally labeled by `router_worker_id`, `state`, and `reason`. The state is `available`, `pending`, or `excluded`. The sample changes with the worker's current state and disappears when discovery removes the ID.
</ParamField>

The bounded `reason` values describe the current admission or availability condition:

| Reason | Meaning |
|--------|---------|
| `none` | Available for routing |
| `initializing` | Waiting for initialization |
| `checksum_mismatch` | The individual worker's Model Deployment Card checksum differs from the incumbent; emitted only for rejected worker IDs |
| `materialization_failed` | Creating the serving configuration failed |
| `commit_blocked` | The serving configuration could not be committed |
| `invalid_config` | The discovered configuration is invalid |
| `unavailable` | Absent from the current availability view; includes local inhibition and other availability changes without identifying their exact cause |

When an incumbent is still initializing and an incompatible newcomer is discovered, counts are `discovered=2`, `available=0`, `pending=1`, and `excluded=1`. The incumbent carries `state="pending",reason="initializing"`; only the newcomer carries `state="excluded",reason="checksum_mismatch"`. Once the incumbent is healthy, the same group reports `discovered=2`, `available=1`, `pending=0`, and `excluded=1`. Rejection of the newcomer never makes the incumbent pending or excluded.

To detect an unexpected absence of workers, select one frontend, model, worker role, and the namespaces belonging to one deployment. The following expression alerts when the frontend is being scraped successfully but no selected group has a positive discovered-worker count. It handles both explicit zero counts and series disappearing after group withdrawal:

```promql
(up{job="<frontend-job>", instance="<frontend-instance>"} == 1)
unless on (job, instance)
(
  sum by (job, instance) (
    dynamo_frontend_router_workers{
      job="<frontend-job>",
      instance="<frontend-instance>",
      model="<model>",
      target_namespace=~"<deployment-namespace-pattern>",
      worker_type="<worker-role>",
      state="discovered"
    }
  ) > 0
)
```

Replace the selectors with your scrape labels and a namespace pattern that includes only the rolling versions of one deployment. Enable the rule only for deployments expected to have workers, and use an alert duration such as `for: 3m` to tolerate startup and brief rollout transitions. To alert when no selected worker is available for routing, use `state="available"` instead of `state="discovered"`. Monitor `up == 0` separately for frontend scrape failures. A raw `sum(...) == 0` expression does not match when all selected group series have disappeared.

Preserve `target_namespace` when diagnosing an individual namespace. Do not aggregate independent deployments or different frontend instances.

Inspect workers still discovered but excluded from routing:

```promql
dynamo_frontend_router_worker_state{state="excluded"} == 1
```

Scope these queries to the target frontend and deployment. Evaluate each Frontend separately because discovery order and first-wins admission can produce different incumbents. A missing series does not mean zero workers; check Prometheus `up` separately for scrape health. These metrics reflect updates after the frontend processes discovery and routing state changes, not at the instant a worker process terminates.

### Model Metrics

All `dynamo_frontend_model_*` gauges carry a `model` label.

<ParamField path="dynamo_frontend_model_ready" type="gauge">
  Whether the frontend can currently route at least one inference request for the model. A value of `1` means at least one complete serving topology with a live worker and serving engine is available; `0` means the model is registered but not currently routable. The frontend evaluates this gauge from its live routing catalog at scrape time, so worker registration and removal are reflected in the next scrape. Unlike request-populated metrics, the series appears when the model registers without requiring an inference request. The series is absent while the model is not registered, and the entire family is absent when no models are registered. Detect this state with `absent(dynamo_frontend_model_ready{model="<model>"})`; a `== 0` comparison does not match an absent series. Aliases get no series of their own: a deployment served under several names emits one series, labeled with its primary name, so query an alias through its primary. `/v1/models` still lists alias names, but only the primary has a readiness series. A LoRA adapter is a distinct model rather than a second name for one, and keeps its own series.
</ParamField>

The remaining model gauges are populated from worker backend registration. When multiple workers register the same model name, only the first instance's configuration is recorded.

<ParamField path="dynamo_frontend_model_total_kv_blocks" type="gauge">
  Total KV blocks available for a worker serving the model.
</ParamField>

<ParamField path="dynamo_frontend_model_max_num_seqs" type="gauge">
  Maximum number of sequences for a worker serving the model.
</ParamField>

<ParamField path="dynamo_frontend_model_max_num_batched_tokens" type="gauge">
  Maximum number of batched tokens for a worker serving the model.
</ParamField>

<ParamField path="dynamo_frontend_model_context_length" type="gauge">
  Maximum context length for a worker serving the model. Sourced from the Model Deployment Card.
</ParamField>

<ParamField path="dynamo_frontend_model_kv_cache_block_size" type="gauge">
  KV cache block size for a worker serving the model. Sourced from the Model Deployment Card.
</ParamField>

<ParamField path="dynamo_frontend_model_migration_limit" type="gauge">
  Request migration limit for a worker serving the model. Sourced from the Model Deployment Card.
</ParamField>

### System One Requests

Recorded for the experimental System One endpoint (`/v1/systemone`), which `DYN_ENABLE_SYSTEMONE_API` enables. These carry `model`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_systemone_questions` | histogram | — | Questions per admitted request. | Experimental |
| `dynamo_frontend_systemone_input_tokens` | histogram | — | Prompt tokens per admitted request, summed over its questions. | Experimental |
| `dynamo_frontend_systemone_candidate_ids` | histogram | — | Candidate token IDs requested per admitted request. | Experimental |
| `dynamo_frontend_systemone_cached_tokens` | histogram | — | Cached prompt tokens per completed question, when the worker reports them. | Experimental |
| `dynamo_frontend_systemone_admitted_branches_total` | counter | — | Questions admitted for dispatch. | Experimental |
| `dynamo_frontend_systemone_phase_seconds` | histogram | `phase` | Time in each phase: `preflight` (validating and tokenizing the request), `first_branch` (running the first question), or `fanout` (running the remaining questions on the same worker). | Experimental |

### LoRA Allocation

Registered on every frontend and populated only when LoRA serving is enabled with `DYN_LORA_ENABLED`. The `endpoint` label is `namespace/component/endpoint`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_lora_active_requests` | gauge | `endpoint`, `lora` | In-flight requests per LoRA adapter. | Current |
| `dynamo_frontend_lora_estimated_load` | gauge | `endpoint`, `lora` | Estimated load per adapter (windowed request count). | Current |
| `dynamo_frontend_lora_raw_arrival_count` | gauge | `endpoint`, `lora` | Windowed arrival count per adapter. | Current |
| `dynamo_frontend_lora_replica_factor` | gauge | `endpoint`, `lora` | Replicas allocated to each adapter. | Current |
| `dynamo_frontend_lora_is_active` | gauge | `endpoint`, `lora` | 1 if the adapter is active, 0 if not. | Current |
| `dynamo_frontend_lora_churn_loads_total` | gauge | `endpoint` | Adapter loads in the latest allocation step. A per-step gauge despite its `_total` suffix; non-zero only with `DYN_LORA_ALLOCATION_ALGORITHM=mcf`. | Current |
| `dynamo_frontend_lora_churn_unloads_total` | gauge | `endpoint` | Adapter unloads in the latest allocation step. A per-step gauge despite its `_total` suffix; non-zero only with `DYN_LORA_ALLOCATION_ALGORITHM=mcf`. | Current |
| `dynamo_frontend_lora_overflow_count` | gauge | `endpoint` | Replicas the allocator could not place in the latest step. Non-zero only with `DYN_LORA_ALLOCATION_ALGORITHM=mcf`. | Current |

### Async Runtime and Request Plane

Health of the frontend's Tokio async runtime and of its request path to workers. Frontend only.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_event_loop_delay_seconds` | histogram | — | Delay of a 10 ms timer on the async runtime. Higher values mean the runtime is overloaded. | Current |
| `dynamo_frontend_event_loop_stall_total` | counter | — | Timer delays longer than 5 ms. | Current |
| `dynamo_tokio_alive_tasks` | gauge | — | Live async tasks. | Current |
| `dynamo_tokio_global_queue_depth` | gauge | — | Tasks in the runtime's global queue. | Current |
| `dynamo_tokio_blocking_threads` | gauge | — | Threads in the blocking pool. | Current |
| `dynamo_tokio_blocking_idle_threads` | gauge | — | Idle threads in the blocking pool. | Current |
| `dynamo_tokio_blocking_queue_depth` | gauge | — | Tasks waiting for a blocking-pool thread. | Current |
| `dynamo_tokio_budget_forced_yield_total` | counter | — | Times a task was forced to yield after using its budget. | Current |
| `dynamo_tokio_queue_overload_warnings_total` | counter | — | Warnings logged for sustained runnable-queue pressure. | Current |
| `dynamo_tokio_worker_busy_ratio` | gauge | `worker` | Approximate busy level of each runtime thread, from 0 to 1000. Above 950 means saturated. | Current |
| `dynamo_tokio_worker_mean_poll_time_ns` | gauge | `worker` | Mean task poll time of each runtime thread, in nanoseconds. | Current |
| `dynamo_tokio_worker_local_queue_depth` | gauge | `worker` | Tasks in each runtime thread's local queue. | Current |
| `dynamo_tokio_worker_park_count_total` | counter | `worker` | Times each runtime thread parked. | Current |
| `dynamo_tokio_worker_steal_count_total` | counter | `worker` | Tasks each runtime thread stole from other threads. | Current |
| `dynamo_tokio_worker_overflow_count_total` | counter | `worker` | Times each runtime thread's local queue overflowed. | Current |
| `dynamo_request_plane_inflight_requests` | gauge | — | Requests sent to workers that have not finished. | Current |
| `dynamo_request_plane_queue_seconds` | histogram | — | Time from dispatch until the request is sent to a worker. | Current |
| `dynamo_request_plane_send_seconds` | histogram | — | Time to send a request to a worker. | Current |
| `dynamo_request_plane_roundtrip_ttft_seconds` | histogram | — | Time from sending a request until its first response arrives. | Current |
| `dynamo_transport_tcp_bytes_sent_total` | counter | — | Request bytes sent over TCP. | Current |
| `dynamo_transport_tcp_bytes_received_total` | counter | — | Acknowledgement bytes received over TCP. Streamed responses are not counted. | Current |
| `dynamo_transport_tcp_errors_total` | counter | — | Failed TCP requests, including send failures and timeouts. | Current |
| `dynamo_transport_tcp_accept_backoff_total` | counter | — | Response-server accept failures caused by descriptor or memory exhaustion. | Current |
| `dynamo_transport_nats_errors_total` | counter | `error_type` | NATS request errors. Only with the NATS request plane. | Current |

## Component Metrics

Emitted by backend workers (`python -m dynamo.vllm`, `python -m dynamo.sglang`, etc.) at `/metrics` on `DYN_SYSTEM_PORT`. Metrics registered on an endpoint carry the hierarchy labels (`dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`) and `worker_id` — see [Runtime-injected labels](metric-labels.mdx#runtime-injected-labels). Metrics created on a component rather than an endpoint, such as `dynamo_backend_admission_*`, have no `dynamo_endpoint`, and `dynamo_work_handler_*` has no labels. Workers also expose their inference engine's own metrics; see [Engine Metrics](#engine-metrics).

The request metrics below are recorded for every endpoint a process serves. vLLM, SGLang, and TensorRT-LLM workers also add a `model` label on their inference endpoints. Internal endpoints, such as the `worker_kv_query_source_*` endpoint the router uses to recover a worker's KV index, have no `model` label.

<ParamField path="dynamo_component_inflight_requests" type="gauge">
  Requests currently being processed by the component.
</ParamField>

<ParamField path="dynamo_component_request_bytes_total" type="counter">
  Total bytes received in requests.
</ParamField>

<ParamField path="dynamo_component_request_duration_seconds" type="histogram">
  Time to handle a request, until its response stream ends.
</ParamField>

<ParamField path="dynamo_component_requests_total" type="counter">
  Total requests processed.
</ParamField>

<ParamField path="dynamo_component_errors_total" type="counter">
  Total errors encountered while handling a request. Labeled by `error_type` — see [Component error types](metric-labels.mdx#component-error-types).
</ParamField>

<ParamField path="dynamo_component_response_bytes_total" type="counter">
  Total bytes sent in responses.
</ParamField>

<ParamField path="dynamo_component_cancellation_total" type="counter">
  Total requests cancelled by the work handler. Carries the hierarchy labels. Records cancellation *signals* received by the worker, not whether the engine actually aborted — deduplicated so a control message plus a socket close for the same request counts once. See [Request Cancellation Architecture](../../developer-guide/knowledge-base/concepts/fault-tolerance/request-cancellation-architecture.md).
</ParamField>

The shared TCP request plane, the default, exposes the following `dynamo_work_handler_*` metrics. They are
registered once per process and are independent of backend admission; see [Backend Admission Control](../../developer-guide/knowledge-base/concepts/fault-tolerance/admission-control-architecture.md#backend).
With the NATS request plane, all except `dynamo_work_handler_time_to_first_response_seconds` stay at 0.

<ParamField path="dynamo_work_handler_pool_active_tasks" type="gauge">
  Current TCP request handlers holding worker-pool permits, from dispatch until downstream handling finishes.
</ParamField>

<ParamField path="dynamo_work_handler_pool_capacity" type="gauge">
  Configured capacity of the TCP worker pool, set by `DYN_TCP_WORKER_POOL_SIZE`.
</ParamField>

<ParamField path="dynamo_work_handler_queue_depth" type="gauge">
  Current requests in the bounded TCP work queue awaiting dispatcher pickup.
</ParamField>

<ParamField path="dynamo_work_handler_queue_capacity" type="gauge">
  Configured capacity of the bounded TCP work queue, set by `DYN_TCP_WORK_QUEUE_SIZE`.
</ParamField>

<ParamField path="dynamo_work_handler_enqueue_rejected_total" type="counter">
  Cumulative requests rejected before TCP worker dispatch because the bounded work queue was full or the dispatcher channel was closed.
</ParamField>

<ParamField path="dynamo_work_handler_permit_wait_seconds" type="histogram">
  Time a queued request spends waiting for a TCP worker-pool permit after dispatcher pickup.
</ParamField>

<ParamField path="dynamo_work_handler_network_transit_seconds" type="histogram">
  Network time from the frontend to the worker, computed from the clocks on both hosts.
</ParamField>

<ParamField path="dynamo_work_handler_time_to_first_response_seconds" type="histogram">
  Time from receiving a request until its response stream opens. This is not the time to the first token.
</ParamField>

The `dynamo_backend_admission_*` metrics track engine occupancy, queueing, and request outcomes.
These metrics count requests only when at least one engine limit is configured. With both engine limits
unlimited, they remain at zero.
See [Backend Admission Control](../../developer-guide/knowledge-base/concepts/fault-tolerance/admission-control-architecture.md#backend)
for gate behavior.

The `dp_rank` label contains the rank ID, such as `"4"`, or `"unspecified"` for unranked requests.
Sum across all `dp_rank` values for worker totals.

<ParamField path="dynamo_backend_admission_engine_request_count" type="gauge">
  Requests admitted to the engine that have not finished. Label: `dp_rank`.
</ParamField>

<ParamField path="dynamo_backend_admission_engine_request_waiting_count" type="gauge">
  Admitted requests still awaiting their first generated token or completion. Label: `dp_rank`.
  Replaces `dynamo_backend_admission_engine_wait_count`.
</ParamField>

<ParamField path="dynamo_backend_admission_request_queue_count" type="gauge">
  Requests waiting outside the engine in gate queues. Label: `dp_rank`.
</ParamField>

<ParamField path="dynamo_backend_admission_request_receive_total" type="counter">
  All requests received by the gate, counted once regardless of outcome. Label: `dp_rank`.
</ParamField>

<ParamField path="dynamo_backend_admission_request_admit_total" type="counter">
  Requests passed into the engine. Labels: `dp_rank` and `source`.
  `source="direct"` means admitted without queueing; `source="queue"` means admitted after queueing.
</ParamField>

<ParamField path="dynamo_backend_admission_rejection_total" type="counter">
  Requests the gate could neither admit nor queue. Label: `dp_rank`.
</ParamField>

<ParamField path="dynamo_backend_admission_cancellation_total" type="counter">
  Requests cancelled before engine admission. Label: `dp_rank`.
</ParamField>

### KV Event Publishing

Workers that publish KV cache events to the router: vLLM, SGLang, TensorRT-LLM with KV events, and the mocker. The `kv_publisher` metrics are created on the component, so they have no `dynamo_endpoint`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_kv_publisher_engines_dropped_events_total` | counter | — | KV events the engine dropped before publishing, detected from event ID gaps. | Current |
| `dynamo_component_kv_publisher_zmq_events_total` | counter | `stage`, `event_type` | KV events received from the engine over ZMQ, and accepted. | Current |
| `dynamo_component_kv_publisher_zmq_filtered_events_total` | counter | `event_type`, `reason` | KV events filtered out before conversion. | Current |
| `dynamo_component_kv_publisher_zmq_conversion_issues_total` | counter | `event_type`, `reason` | KV events dropped because they could not be converted. | Current |
| `dynamo_component_kv_publisher_zmq_suspicious_events_total` | counter | `event_type`, `reason` | Suspicious KV events that were still forwarded. | Current |
| `dynamo_component_worker_prefill_completion_enqueue_failures_total` | counter | `reason` | Prefill-completion events dropped because the local publish queue was full or closed. Aggregated and decode workers only. | Current |

### Engine State

Label sets depend on the worker type:

- `python -m dynamo.vllm`, `dynamo.sglang`, and `dynamo.trtllm` add `model`, `dynamo_component`, and `dp_rank` instead of the automatic labels. vLLM reports these only without `--disable-log-stats`. TensorRT-LLM updates the cache gauges only with `--publish-metrics`.
- [Unified backend](../../developer-guide/advanced-customizations/writing-custom-backends/writing-unified-backends.md) workers use the automatic labels plus `model` and `model_name`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_total_blocks` | gauge | `dp_rank` | Total KV cache blocks on the worker. | Current |
| `dynamo_component_gpu_cache_usage_percent` | gauge | `dp_rank` | GPU KV cache usage, from 0.0 to 1.0. | Current |
| `dynamo_component_kv_cache_hit_rate` | gauge | `dp_rank` | Prefix cache hit rate, from 0.0 to 1.0. Registered, but no in-tree engine reports it yet. | Current |
| `dynamo_component_model_load_time_seconds` | gauge | — | Time to load the model and start the engine. | Current |
| `dynamo_component_drain_time_seconds` | gauge | — | Time spent draining in-flight requests at shutdown. Unified backend workers only. | Current |
| `dynamo_component_cleanup_time_seconds` | gauge | — | Time spent releasing engine resources at shutdown. Unified backend workers only. | Current |

### Multimodal and Embedding Caches

These metrics come from several publishers, so their labels differ:

- The image cache metrics, and the embedding cache metrics a worker records in its own process, carry `model` and `dynamo_component`.
- With the `DynamoMultimodalEmbeddingCacheConnector`, the vLLM scheduler records the embedding cache metrics once per data-parallel rank. They carry `model`, `model_name`, `dp_rank`, `dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`, and `worker_id`.
- The SGLang `dynamo_embedding_*` histograms carry `model`, `model_name`, `dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`, and `worker_id`.

The embedding cache metrics require `--multimodal-embedding-cache-capacity-gb` greater than 0. The image cache metrics are registered on vLLM multimodal workers, and the shared image cache records samples only with `DYN_MM_SHARED_IMAGE_CACHE_ENABLED=1`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_embedding_cache_hits_total` | counter | — | Embedding cache hits. | Current |
| `dynamo_component_embedding_cache_misses_total` | counter | — | Embedding cache misses. | Current |
| `dynamo_component_embedding_cache_evictions_total` | counter | — | Embedding cache evictions. | Current |
| `dynamo_component_embedding_cache_utilization` | gauge | — | Embedding cache memory use, from 0.0 to 1.0. | Current |
| `dynamo_component_embedding_cache_current_bytes` | gauge | — | Embedding cache memory use in bytes. | Current |
| `dynamo_component_embedding_cache_entries` | gauge | — | Entries in the embedding cache. | Current |
| `dynamo_component_image_cache_entries` | gauge | — | Entries in the in-process image cache. | Current |
| `dynamo_component_image_shared_cache_get_duration_seconds` | histogram | `outcome`, `size_bucket` | Shared image cache read latency, by outcome and payload size. | Current |
| `dynamo_component_image_shared_cache_set_duration_seconds` | histogram | `outcome`, `size_bucket` | Shared image cache write latency, by outcome and payload size. | Current |
| `dynamo_embedding_batch_size` | histogram | — | Inputs per `/v1/embeddings` request. SGLang embedding workers (`--embedding-worker`) only. | Current |
| `dynamo_embedding_input_tokens` | histogram | — | Prompt tokens per `/v1/embeddings` request. SGLang embedding workers only. | Current |

### TensorRT-LLM Metrics Computed by Dynamo

TensorRT-LLM workers started with `--publish-metrics` add these to the engine's own `trtllm_*` metrics. They carry `model_name`, `disaggregation_mode`, and `engine_type`, plus the labels Dynamo adds to [engine metrics](#engine-metrics).

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `trtllm_num_aborted_requests_total` | counter | — | Requests aborted or cancelled. | Current |
| `trtllm_request_type_image_total` | counter | — | Requests that contain images. | Current |
| `trtllm_request_type_structured_output_total` | counter | — | Requests that use structured (guided) output. | Current |
| `trtllm_kv_transfer_success_total` | counter | — | Successful KV cache transfers. | Current |
| `trtllm_kv_transfer_latency_seconds` | histogram | — | KV cache transfer time per request. | Current |
| `trtllm_kv_transfer_bytes` | histogram | — | KV cache transfer size per request, in bytes. | Current |
| `trtllm_kv_transfer_speed_gb_s` | histogram | — | KV cache transfer speed per request, in GB/s. | Current |
| `trtllm_kv_event_buffer_capacity` | gauge | — | Configured size of the engine's KV event buffer. | Current |
| `trtllm_kv_event_drain_batch_size` | histogram | — | KV events returned per polling drain. Only when KV events are polled. | Current |
| `trtllm_kv_event_id_gap_events_total` | counter | — | Missing KV event IDs detected by Dynamo. | Current |

### Engine Failover

vLLM workers with `--gms-shadow-mode` (GPU Memory Service shadow failover). These carry `engine_id`, `model`, and `dynamo_component`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_engine_failover_state` | gauge | `state` | 1 for the engine's current failover state: `init`, `standby`, `waking`, or `active`. | Current |
| `dynamo_component_engine_failover_state_entered_timestamp_seconds` | gauge | — | Unix time the engine entered its current state. | Current |
| `dynamo_component_engine_failover_last_state_duration_seconds` | gauge | `state` | Duration of the most recent stay in each state. For `waking`, this is the switch time. | Current |
| `dynamo_component_engine_failover_transitions_total` | counter | `from_state`, `to_state` | Failover state transitions. | Current |
| `dynamo_component_engine_failover_switch_attempts_total` | counter | — | Promotions attempted after winning the failover lock. Kept across restarts. | Current |
| `dynamo_component_engine_failover_switch_success_total` | counter | — | Promotions that completed and began serving. Kept across restarts. | Current |

## Router Metrics

Emitted by every process that routes requests: the frontend, which serves them on its HTTP port, and the standalone router (`python -m dynamo.router`), which serves them on `DYN_SYSTEM_PORT`. The router adds `dynamo_namespace` and `dynamo_component` of the workers it routes to, and `worker_id`, which is the router's own ID in hex. The request metrics also carry `router_id`, the same ID in decimal. Not every metric appears in every deployment — see [Availability by Configuration](#availability-by-configuration).

<ParamField path="dynamo_component_router_worker_registered" type="gauge">
  A sample with value `1` for each worker and DP rank in the KV router scheduler's current registration and hard-availability view; absent in other router modes. Carries `router_worker_id`, `dp_rank`, and `worker_type` labels in addition to the component labels. The sample can appear during scheduler preparation before worker-set materialization commits, so it does not indicate committed model readiness. Removed or unavailable workers disappear from the output, rather than emitting `0`; late request bookkeeping cannot restore their exposed series. Prefer the [Router Worker Inventory](#router-worker-inventory) metrics for committed available-worker counts and exclusion reasons on the frontend.
</ParamField>

### Router Request Metrics

`dynamo_component_router_*` histograms and counters for aggregate request-level statistics. Populated per request in every router mode. The KV-specific metrics (`kv_hit_rate`, the [KV-cache reuse](#kv-cache-reuse-telemetry) counters, `non_max_overlap_selections_total`, and `overlap_blocks_lost`) are recorded only with `--router-mode kv`.

<ParamField path="dynamo_component_router_requests_started_total" type="counter">
  Requests admitted by the router scheduler.
</ParamField>

<ParamField path="dynamo_component_router_requests_total" type="counter">
  Total requests processed by the router.
</ParamField>

<ParamField path="dynamo_component_router_time_to_first_token_seconds" type="histogram">
  Time to first token in seconds.
</ParamField>

<ParamField path="dynamo_component_router_inter_token_latency_seconds" type="histogram">
  Average inter-token latency in seconds. With `--router-mode kv`, recorded only when `--router-track-output-blocks` is set; otherwise the histogram stays empty.
</ParamField>

<ParamField path="dynamo_component_router_input_sequence_tokens" type="histogram">
  Input sequence length per routing attempt, excluding query-only lookups. Labeled by `phase` and `model`.
</ParamField>

<ParamField path="dynamo_component_router_output_sequence_tokens" type="histogram">
  Output sequence length in tokens.
</ParamField>

<ParamField path="dynamo_component_router_kv_hit_rate" type="histogram">
  Predicted KV cache hit rate at routing time (0.0–1.0).
</ParamField>

<ParamField path="dynamo_component_router_non_max_overlap_selections_total" type="counter">
  Admitted prefill scheduler selections routed to a worker with less KV cache overlap than another eligible worker. Pinned requests and equal-overlap ties are excluded. Labeled by `worker_type` (`prefill`).
</ParamField>

<ParamField path="dynamo_component_router_overlap_blocks_lost" type="histogram">
  Difference in effective KV cache overlap between the highest-overlap eligible prefill worker and the selected worker for each non-max-overlap selection. Labeled by `worker_type` (`prefill`).
</ParamField>

<ParamField path="dynamo_component_router_kv_transfer_estimated_latency_seconds" type="histogram">
  Upper-bound estimate of the KV cache transfer time from prefill to decode. Recorded only in disaggregated serving.
</ParamField>

### KV-Cache Reuse Telemetry

The cache reuse funnel shows how much of the incoming prompt could be reused, and how much the worker reports reusing. Each step counts tokens, not requests.

| Step | Meaning | Series |
| --- | --- | --- |
| **F0 — Incoming tokens** | Tokens received by the router. | `dynamo_component_router_input_sequence_tokens_sum` |
| **F2 — Best available match** | Prefix the router estimates is cached on the best-matching eligible worker. | `dynamo_component_router_kv_best_eligible_cached_prefix_tokens_total` |
| **F3 — Selected worker's match** | Prefix the router estimates is cached on the worker it chooses. | `dynamo_component_router_kv_selected_cached_prefix_tokens_total` |
| **F4 — Reported reuse** | Tokens the worker reports reusing. Missing reports count as zero. | `dynamo_component_router_kv_worker_reused_tokens_total` |

F2, F3, and F4 are counters labeled by `router_id`, `model`, and `phase`. They need no enable flag.

Compare token increases over the same time window and matching KV-routing traffic. Use a long enough window: the steps are recorded at different times, and F4 depends on worker reports arriving after routing. Short windows may therefore compare different requests. Divide each step by F0 to express it as a percentage of incoming tokens.

vLLM, SGLang, and TensorRT-LLM workers report F4 on the final response chunk of each aggregated or prefill attempt, as `engine_data.kv_cache_hit` with `prompt_tokens` and `reused_tokens`. `reused_tokens` is the engine's per-request cached-token count, so it includes hits on host and storage tiers and KV connectors, the same tiers F2 and F3 count.

- Disaggregated decode attempts do not report, because each engine counts the KV transferred from the prefill worker as cached; read worker reuse from the `prefill` phase. The exception is a decode attempt that prefills locally under conditional disaggregation, which reports under `phase="decode"`.
- Attempts take their `phase` from the frontend's request tracker. A standalone router (`python -m dynamo.router`) does not set one, so all of its attempts are labeled `aggregated`.
- The router counts a report only when its `prompt_tokens` equals the routed prompt length. Engines count multimodal prompts after placeholder expansion, so a multimodal report counts only when the router also routed on expanded tokens; TensorRT-LLM does not report multimodal requests. F2 and F3 count expanded routing tokens while F0 counts the unexpanded prompt, so multimodal ratios to F0 can exceed 100%.
- With `n > 1`, vLLM and SGLang do not report. Later samples reuse the prompt blocks earlier samples just cached, vLLM keeps the first buffered sample's count when it merges outputs, and SGLang caches the prompt before running the samples.
- Engines recompute the last prompt token, so a full-prefix hit reports at least one token less than F3, and up to one block less for vLLM.
- Preemption, partial-block reuse in TensorRT-LLM, and KV loads that fail under a recompute policy can also make F4 differ from F3 when the router's estimate is accurate.

### Per-Request Routing Overhead

`dynamo_router_overhead_*` histograms (milliseconds) track time spent in each phase of the routing decision. Registered on the frontend port with only a `router_id` label (the frontend's discovery instance ID), without the `dynamo_namespace`, `dynamo_component`, and `worker_id` labels the router adds to its other metrics. Registered on every frontend and recorded only for KV routing decisions, so they stay at zero in non-KV modes. Not created on the standalone router.

<ParamField path="dynamo_router_overhead_block_hashing_ms" type="histogram">
  Time computing block hashes.
</ParamField>

<ParamField path="dynamo_router_overhead_indexer_find_matches_ms" type="histogram">
  Time in indexer `find_matches`.
</ParamField>

<ParamField path="dynamo_router_overhead_seq_hashing_ms" type="histogram">
  Time computing sequence hashes.
</ParamField>

<ParamField path="dynamo_router_overhead_scheduling_ms" type="histogram">
  Time in scheduler worker selection.
</ParamField>

<ParamField path="dynamo_router_overhead_shared_cache_query_ms" type="histogram">
  Time querying the shared KV cache. Recorded only with `--shared-cache-type hicache`.
</ParamField>

<ParamField path="dynamo_router_overhead_total_ms" type="histogram">
  Total routing overhead per request.
</ParamField>

### Router Queue Metrics

These metrics have series once a KV router starts, whether or not queueing is enabled; the queue metrics stay at zero unless queueing is enabled by `--router-queue-threshold` or a threshold in `--router-policy-config`. They carry `model`, `worker_type`, and `policy_class`. With policy-family or cache-bucket configuration, `policy_class` is the resolved physical queue.

<ParamField path="dynamo_frontend_router_queue_pending_requests" type="gauge">
  Requests pending in the router scheduler queue.
</ParamField>

<ParamField path="dynamo_frontend_router_queue_pending_isl_tokens" type="gauge">
  Raw input-sequence tokens pending in the queue.
</ParamField>

<ParamField path="dynamo_frontend_router_queue_pending_cached_tokens" type="gauge">
  Cached-token estimate captured when each request enters the queue.
</ParamField>

<ParamField path="dynamo_frontend_router_queue_backpressure_total" type="counter">
  Queue rejections by configured limit reason. Also labeled by `reason`.
</ParamField>

<ParamField path="dynamo_frontend_router_admission_received_total" type="counter">
  Attempts received by router admission, including retries.
</ParamField>

<ParamField path="dynamo_frontend_router_admission_rejected_total" type="counter">
  Attempts rejected because their deadline passed (`reason="due_time_passed"`). The `predicted_miss` reason is reserved and always 0.
</ParamField>

### KV Indexer Metrics

<ParamField path="dynamo_component_kv_cache_events_applied" type="counter">
  KV cache events applied to the router's radix-tree index. Includes events applied to the device tier and lower tiers such as host-pinned memory and disk. Appears when the worker-selection policy requests `WorkerInputs::CACHE`, the router consumes KV events, and workers publish them. Labeled by `status` and `event_type` — see [Metric-specific labels](metric-labels.mdx#metric-specific-labels).
</ParamField>

<ParamField path="dynamo_component_ckf_mutation_total" type="counter">
  CKF block-level mutation outcomes recorded by the router indexer, labeled by `outcome` — see [Metric-specific labels](metric-labels.mdx#metric-specific-labels). **Registered, always zero.** The series is created and exported, but no code path currently increments it, so it reads 0 in every deployment. Do not build alerts or dashboard panels on it. The standalone indexer exports the same counter as `dynamo_kvrouter_ckf_mutation_total`, with the same caveat.
</ParamField>

The standalone indexer exposes the device-tier-only counter `dynamo_kvrouter_kv_cache_events_applied`; see [Standalone KV Indexer Metrics](#standalone-kv-indexer-metrics).

The router's KV index and event ingestion also record:

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_kv_cache_event_warnings` | counter | `warning_kind` | Suspicious KV cache events seen by the index, such as duplicate stores. | Current |
| `dynamo_component_router_kv_event_source_mismatch_workers` | gauge | `model`, `worker_type`, `target_namespace`, `target_component`, `target_endpoint` | Serving workers whose KV event source is missing, ambiguous, or disabled. | Current |
| `dynamo_component_router_kv_zmq_ingress_sources` | gauge | `state` | KV event sources on the ZMQ event plane, by lifecycle state. | Current |
| `dynamo_component_router_kv_zmq_ingress_batches_total` | counter | — | KV event batches received over ZMQ. | Current |
| `dynamo_component_router_kv_zmq_ingress_lifecycle_total` | counter | `action` | Lifecycle transitions and errors of ZMQ KV event sources. | Current |
| `dynamo_component_router_remote_indexer_query_failures_total` | counter | `router_id` | Failed overlap queries to a remote indexer. Only with `--use-remote-indexer`. | Current |
| `dynamo_component_router_remote_indexer_write_failures_total` | counter | `router_id` | Failed routing-decision writes to a remote indexer. Only with `--use-remote-indexer`. | Current |

### Shared KV Cache

Recorded only with `--shared-cache-type hicache`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_router_shared_cache_hit_rate` | histogram | `router_id` | Fraction of request blocks found in the shared KV cache. | Current |
| `dynamo_component_router_shared_cache_beyond_blocks` | histogram | `router_id` | Shared-cache blocks beyond the selected worker's device overlap. | Current |
| `dynamo_router_shared_cache_errors_total` | counter | `router_id` | Failed shared-cache queries and subscriber errors. Frontend only; carries only `router_id`, like the [routing overhead](#per-request-routing-overhead) metrics. | Current |

### Active Sequence Synchronization

Track the active-request events each KV router applies to its view of worker load. The `active_sequence_ingress` metrics carry `model`, `routing_group`, and `worker_type`. The `active_sequence_zmq_ingress` metrics exist when these events travel over direct ZMQ, the default on the ZMQ event plane.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_router_active_sequence_ingress_queue_depth` | gauge | — | Active-sequence events waiting to be applied. | Current |
| `dynamo_component_router_active_sequence_ingress_events_applied_total` | counter | — | Active-sequence events applied. | Current |
| `dynamo_component_router_active_sequence_ingress_drain_batches_total` | counter | — | Event batches applied. | Current |
| `dynamo_component_router_active_sequence_zmq_ingress_sources` | gauge | — | Tracked ZMQ sources of active-sequence events. | Current |
| `dynamo_component_router_active_sequence_zmq_ingress_sequence_gaps_total` | counter | — | Events missing, inferred from sequence-number gaps. | Current |
| `dynamo_component_router_active_sequence_zmq_ingress_out_of_order_total` | counter | — | Events received out of order. | Current |
| `dynamo_component_router_active_sequence_zmq_ingress_reconnects_total` | counter | — | Source reconnect attempts. | Current |
| `dynamo_component_router_active_sequence_zmq_ingress_replacements_total` | counter | — | Source task replacements. | Current |
| `dynamo_component_router_active_sequence_zmq_ingress_envelope_decode_errors_total` | counter | — | Envelope decode errors. | Current |
| `dynamo_component_router_active_sequence_zmq_ingress_payload_decode_errors_total` | counter | — | Payload decode errors, including oversized batches. | Current |
| `dynamo_component_router_active_sequence_zmq_ingress_identity_errors_total` | counter | — | Frame and envelope identity mismatches. | Current |
| `dynamo_component_router_active_sequence_zmq_ingress_forced_aborts_total` | counter | — | Source tasks aborted after a join timeout. | Current |
| `dynamo_component_router_active_sequence_zmq_ingress_dropped_events_total` | counter | — | Events dropped because the inbound queue was full or closed. | Current |

### Approximate Cache

The approximate cache tracks likely-cached blocks when the router does not consume KV events. The two policy gauges are present on every KV router. The LRU metrics update only with `--router-approximate-cache-policy lru` and KV events disabled.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_router_approximate_cache_configured_policy` | gauge | `policy` | 1 for the configured policy: `ttl`, `lru`, or `disabled`. | Experimental |
| `dynamo_component_router_approximate_cache_effective_policy` | gauge | `policy` | 1 for the policy in effect. | Experimental |
| `dynamo_component_router_approximate_lru_ranks` | gauge | `policy` | Index ranks by retention policy: `lru` or `ttl_fallback`. | Experimental |
| `dynamo_component_router_approximate_lru_blocks` | gauge | `state` | Blocks tracked by the approximate LRU, by state. | Experimental |
| `dynamo_component_router_approximate_lru_leases` | gauge | — | Live request leases. | Experimental |
| `dynamo_component_router_approximate_lru_mutation_queue_depth` | gauge | — | Pending LRU mutation commands. | Experimental |
| `dynamo_component_router_approximate_lru_fallback_activations_total` | counter | — | Ranks that fell back to TTL because their capacity was unavailable. | Experimental |
| `dynamo_component_router_approximate_lru_eviction_batches_total` | counter | — | Mutation batches that evicted blocks. | Experimental |
| `dynamo_component_router_approximate_lru_evicted_blocks_total` | counter | — | Blocks evicted. | Experimental |
| `dynamo_component_router_approximate_lru_output_batches_total` | counter | — | Output-block batches sent to the LRU. | Experimental |
| `dynamo_component_router_approximate_lru_messages_per_request` | histogram | — | LRU queue messages per request, per 1-second interval. | Experimental |
| `dynamo_component_router_approximate_lru_mutation_wait_seconds` | histogram | — | Mean LRU queue wait per command, per 1-second interval. | Experimental |

### Availability by Configuration

Not all router metrics appear in every deployment. This matrix shows which groups are **registered** and **populated** in each configuration.

| Metric group | Frontend + KV (agg) | Frontend + KV (disagg) | Frontend + non-KV | Standalone router |
|---|---|---|---|---|
| `dynamo_component_router_*` | Registered and populated | Registered and populated | Request, latency, and sequence-length metrics populated; KV-specific metrics **always zero** | Populated (on `DYN_SYSTEM_PORT`) |
| `dynamo_router_overhead_*` | Registered and populated | Registered and populated | Registered, **always zero** | **Not created** |
| `dynamo_frontend_router_queue_*` | Registered; populated when a CLI or policy-class queue threshold is set | Registered; populated when a CLI or policy-class queue threshold is set | **Not registered** | **Not created** |
| `dynamo_component_kv_cache_events_applied` | Populated when KV events received | Populated when KV events received | **Not registered** | Populated when KV events received |
| `dynamo_component_ckf_mutation_total` | Registered, **always zero** | Registered, **always zero** | **Not registered** | Registered, **always zero** |
| `dynamo_frontend_worker_*` | Registered and populated | Registered and populated (`worker_type` = `prefill`/`decode`) | `worker_last_*` populated (`worker_type` = `decode`); `worker_active_*` absent | **Not created** |

**Key:**

- **Registered and populated** — the metric appears at `/metrics` with real values.
- **Registered, always zero** — the metric appears at `/metrics` but is never incremented (useful for dashboards that expect it to exist).
- **Not registered / Not created** — the metric does not appear at `/metrics` at all.

## Every Process

Emitted by every Dynamo process: on the HTTP port for the frontend, and on `DYN_SYSTEM_PORT` for other processes.

<ParamField path="dynamo_component_uptime_seconds" type="gauge">
  DistributedRuntime uptime. Updated before each Prometheus scrape on both the frontend and the system-status server. Carries only `worker_id`.
</ParamField>

With `DYN_RESPONSE_PLANE=quic`, every process also records:

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_quic_response_active_connections` | gauge | — | Open QUIC response connections. | Current |
| `dynamo_quic_response_bundle_failures_total` | counter | `side` | QUIC connection bundle failures on the `frontend` or `worker` side. | Current |
| `dynamo_quic_response_server_mailbox_resets_total` | counter | `reason` | Responses reset because the frontend's response mailbox was `full` or `closed`. | Current |

## Planner Metrics

Exposed by `python -m dynamo.planner` at `/metrics` on its own port, set by `metric_reporting_prometheus_port`. It defaults to the `PLANNER_PROMETHEUS_PORT` environment variable, or to `0`, which disables the server, when that variable is unset. The Kubernetes operator sets the port to `9085`. Planner metrics carry no automatic labels. See [Planner Configuration](../components/planner-configuration.mdx).

### Replicas, Traffic, and SLA

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_planner_num_prefill_replicas` | gauge | — | Ready prefill replicas. | Current |
| `dynamo_planner_num_decode_replicas` | gauge | — | Ready decode replicas. | Current |
| `dynamo_planner_predicted_num_prefill_replicas` | gauge | — | Prefill replicas the planner decided on. | Current |
| `dynamo_planner_predicted_num_decode_replicas` | gauge | — | Decode replicas the planner decided on. | Current |
| `dynamo_planner_gpu_hours` | gauge | — | GPU hours used since the planner started. | Current |
| `dynamo_planner_observed_ttft_ms` | gauge | — | Observed TTFT in milliseconds. | Current |
| `dynamo_planner_observed_itl_ms` | gauge | — | Observed ITL in milliseconds. | Current |
| `dynamo_planner_observed_requests_per_second` | gauge | — | Observed request rate. | Current |
| `dynamo_planner_observed_request_duration_seconds` | gauge | — | Observed average request duration. | Current |
| `dynamo_planner_observed_input_sequence_tokens` | gauge | — | Observed average prompt length. | Current |
| `dynamo_planner_observed_output_sequence_tokens` | gauge | — | Observed average output length. | Current |
| `dynamo_planner_predicted_requests_per_second` | gauge | — | Predicted request rate for the next interval. | Current |
| `dynamo_planner_predicted_input_sequence_tokens` | gauge | — | Predicted prompt length for the next interval. | Current |
| `dynamo_planner_predicted_output_sequence_tokens` | gauge | — | Predicted output length for the next interval. | Current |
| `dynamo_planner_sla_target_ttft_ms` | gauge | — | Configured TTFT target in milliseconds. | Current |
| `dynamo_planner_sla_target_itl_ms` | gauge | — | Configured ITL target in milliseconds. | Current |
| `dynamo_planner_estimated_ttft_ms` | gauge | — | Highest TTFT estimated across engines, in milliseconds. | Current |
| `dynamo_planner_estimated_itl_ms` | gauge | — | Highest ITL estimated across engines, in milliseconds. | Current |
| `dynamo_planner_engine_prefill_capacity_requests_per_second` | gauge | — | Request rate one prefill engine can serve within the SLA. | Current |
| `dynamo_planner_engine_decode_capacity_requests_per_second` | gauge | — | Request rate one decode engine can serve within the SLA. | Current |
| `dynamo_planner_engine_queued_prefill_tokens` | gauge | `worker_id`, `dp_rank` | Queued prefill tokens per prefill engine, from forward-pass metrics. | Current |
| `dynamo_planner_engine_queued_decode_kv_tokens` | gauge | `worker_id`, `dp_rank` | Queued decode KV tokens per decode engine, from forward-pass metrics. | Current |
| `dynamo_planner_engine_inflight_decode_kv_tokens` | gauge | `worker_id`, `dp_rank` | Scheduled decode KV tokens per decode engine, from forward-pass metrics. | Current |

### Scaling Decisions

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_planner_load_scaling_decision` | gauge | `dynamo_planner_load_scaling_decision` | Reason for the latest load-based scaling decision. A Prometheus Enum: the label, named after the metric, holds the reason. One series per reason; the current one is 1. | Current |
| `dynamo_planner_throughput_scaling_decision` | gauge | `dynamo_planner_throughput_scaling_decision` | Reason for the latest throughput-based scaling decision. A Prometheus Enum: the label, named after the metric, holds the reason. One series per reason; the current one is 1. | Current |
| `dynamo_planner_signal_unavailable` | gauge | `component`, `signal` | 1 if a required planner input was unavailable at its latest observation. | Current |
| `dynamo_planner_degraded_signals_total` | counter | `component`, `signal` | Times a required input was unavailable for 3 observations in a row. | Current |
| `dynamo_planner_power_budget_total_watts` | gauge | — | Configured total GPU power budget, in watts. Only with power awareness enabled. | Current |
| `dynamo_planner_power_projected_watts` | gauge | — | Projected GPU power draw at the current replica counts, in watts. Only with power awareness enabled. | Current |
| `dynamo_planner_power_budget_utilization` | gauge | — | Projected power divided by the budget. Only with power awareness enabled. | Current |

### Plugin Pipeline

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_planner_plugin_evaluations_total` | counter | `plugin_id`, `stage`, `result` | Plugin evaluations, by outcome. | Current |
| `dynamo_planner_plugin_latency_seconds` | histogram | `plugin_id`, `stage` | Latency of successful plugin calls. | Current |
| `dynamo_planner_plugin_circuit_state` | gauge | `plugin_id` | Circuit breaker state per plugin: 0 closed, 0.5 half open, 1 open. | Current |
| `dynamo_planner_plugin_held_over_total` | counter | `plugin_id`, `stage` | Times a plugin's cached result was reused instead of calling it. | Current |
| `dynamo_planner_plugin_cache_age_seconds` | gauge | `plugin_id` | Age of each plugin's cached result. | Current |
| `dynamo_planner_plugin_override_active` | gauge | `plugin_id`, `stage`, `override_type` | 1 if the plugin contributed an override to the final proposal in the latest tick. | Current |
| `dynamo_planner_reconcile_clamped_total` | counter | `sub_component_type`, `source` | Times the reconcile stage clamped the replica count to a floor or ceiling. | Current |
| `dynamo_planner_constrain_capped_total` | counter | `sub_component_type`, `source` | Times the constrain stage capped the final replica count. | Current |
| `dynamo_planner_reject_short_circuited_total` | counter | `plugin_id` | Times a plugin's rejection stopped the pipeline and skipped execution. | Current |
| `dynamo_planner_tick_skipped_total` | counter | `plugin_id` | Times a plugin was skipped because its execution interval had not elapsed. | Current |
| `dynamo_planner_tick_requires_unsatisfied_total` | counter | `plugin_id`, `missing_field` | Times a plugin was skipped because an input it requires was missing. | Current |
| `dynamo_planner_tick_lag_seconds` | gauge | `plugin_id` | Delay between a plugin's scheduled time and its evaluation. | Current |
| `dynamo_planner_tick_duration_seconds` | histogram | — | Time spent in each planner tick. | Current |
| `dynamo_planner_tick_timeout_total` | counter | — | Ticks aborted for exceeding the tick time limit. | Current |

## Standalone KV Indexer Metrics

`python -m dynamo.indexer` serves its own `/metrics` on its HTTP port (`--port`), with no automatic labels. See [Standalone KV Indexer](../../developer-guide/knowledge-base/modular-components/router/standalone-indexer.md).

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_kvindexer_requests_total` | counter | `endpoint`, `method` | HTTP requests handled, excluding `/metrics`. | Current |
| `dynamo_kvindexer_request_duration_seconds` | histogram | `endpoint` | HTTP request latency. | Current |
| `dynamo_kvindexer_errors_total` | counter | `endpoint`, `status_class` | HTTP error responses, `4xx` or `5xx`. | Current |
| `dynamo_kvindexer_models` | gauge | — | Active model and routing-group indexers. | Current |
| `dynamo_kvindexer_workers` | gauge | — | Registered worker instances. | Current |
| `dynamo_kvindexer_listeners` | gauge | `status` | ZMQ listeners, by status. | Current |
| `dynamo_kvrouter_kv_cache_events_applied` | counter | `event_type`, `status` | KV cache events applied to the index. Device tier only. | Current |
| `dynamo_kvrouter_kv_cache_event_warnings` | counter | `warning_kind` | Suspicious KV cache events seen by the index. | Current |
| `dynamo_kvrouter_ckf_mutation_total` | counter | `outcome` | Block-level index mutation outcomes. Always 0: nothing increments it yet. | Current |

## KV State Agent Metrics

`python -m dynamo.kv_state_agent` serves these on `DYN_SYSTEM_PORT`, with `dynamo_namespace`, `dynamo_component`, and `worker_id`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_kv_state_agent_slots` | gauge | `state` | Agent slots, by state: `total`, `active`, `detached`, or `failed`. | Current |
| `dynamo_component_kv_state_agent_capacity_rejected_total` | counter | — | Attach requests rejected because `--max-slots` was reached. | Current |

## Endpoint Picker Metrics

Emitted by the endpoint picker (EPP) when routing through the Gateway API Inference Extension, at `/metrics` on `DYN_EPP_METRICS_PORT` (default 9090). Set the port to `0` to disable the endpoint.

<ParamField path="dynamo_epp_cached_tokens" type="histogram">
  Prompt tokens the model server served from its KV cache per request, read from `usage.prompt_tokens_details.cached_tokens` in the response body. This is the *observed* cache hit, as opposed to the overlap the KV router estimates at selection time. Buckets match `dynamo_frontend_cached_tokens` so the two can share a dashboard. The `model` label is the model the pool serves, resolved once at startup rather than from the request body.
</ParamField>

Requires the client to request usage accounting on streaming calls (`"stream_options": {"include_usage": true}`); without it the response carries no usage block and nothing is recorded.

## Global Frontend Metrics

Emitted by the Global Frontend at `/metrics` on its serving port. It forwards each request, as received, to a pool frontend, so it sees requests and response bytes but not tokens: inter-token latency and token counts come from the pool frontends' `dynamo_frontend_*` metrics. Per-pool series are labeled by `pool`, the pool's `<cluster>/<namespace>/<DGD name>`, and `region`, the pool's region. No label comes from the request body.

<ParamField path="dynamo_global_frontend_requests_total" type="counter">
  **Experimental.** Requests forwarded to each pool. Labeled by `pool`, `region`, `local` and `status`. `status` is the pool's HTTP status code, counted when its response headers arrive; `no_response` when the pool could not be reached or sent no response; or `cancelled` when the client left before the pool's response headers arrived. A Dynamo pool frontend sends its headers once a worker has the request, so a request the client gives up on while a worker queues it counts under its status, not as `cancelled`. `local` is `true` for a pool in the Global Frontend's `--region`, `false` for one elsewhere, and `unknown` without `--region`; the share of requests kept in the region is the `local="true"` rate over the total.
</ParamField>

<ParamField path="dynamo_global_frontend_rejected_requests_total" type="counter">
  **Experimental.** Requests the Global Frontend answered without forwarding them. Labeled by `error_type`, with the frontend's values: `validation` (the body has no string `model`, or the route is invalid), `not_found` (no pool serves the model) or `unavailable` (no pool can take the request). A body over `DYN_HTTP_BODY_LIMIT_MB` gets 413 before it is read and is not counted.
</ParamField>

<ParamField path="dynamo_global_frontend_time_to_first_byte_seconds" type="histogram">
  **Experimental.** Time from the Global Frontend receiving a request to the first body chunk of its pool's successful (2xx) response. For a streamed request this is the pool's first event, usually the first token, so it approximates the time to first token as the client sees it, including the round trip to a pool in another region. If the pool frontend sends SSE keep-alives before the first token, the first keep-alive counts as the first byte: at the `DYN_HTTP_SSE_KEEP_ALIVE_INTERVAL_MS` interval when set, or after 15 seconds for a request whose output a reasoning parser may hold back. Labeled by `pool` and `region`. Buckets match the `dynamo_frontend_time_to_first_token_seconds` defaults.
</ParamField>

<ParamField path="dynamo_global_frontend_request_duration_seconds" type="histogram">
  **Experimental.** Time from the Global Frontend receiving a request to the end of its pool's successful (2xx) response, or to the client leaving. Labeled by `pool` and `region`. Buckets match the `dynamo_frontend_request_duration_seconds` defaults.
</ParamField>

<ParamField path="dynamo_global_frontend_response_errors_total" type="counter">
  **Experimental.** Successful responses whose body broke off before it ended, for example because the pool frontend restarted mid-stream. The client's response ends early. Labeled by `pool` and `region`.
</ParamField>

<ParamField path="dynamo_global_frontend_inflight_requests" type="gauge">
  **Experimental.** Requests this Global Frontend has sent to each pool and not yet finished. Other replicas count their own. Labeled by `pool` and `region`; read at scrape time, so a pool removed from the configuration loses its series.
</ParamField>

## Mocker Metrics

The mocker (`python -m dynamo.mocker`) simulates an inference engine. It emits imitation engine metrics with simulated values on `DYN_SYSTEM_PORT`, choosing the set by `--engine-type`. They carry the engine's usual labels plus `dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`, `worker_id`, `model`, and `model_name`. These imitate engine metrics, so this table has no Labels, Description, or Status column; each metric means what it means in the engine it imitates.

| Metric | Type | Engine type |
|---|---|---|
| `vllm:num_requests_running` | gauge | `vllm` (default) |
| `vllm:num_requests_waiting` | gauge | `vllm` |
| `vllm:kv_cache_usage_perc` | gauge | `vllm` |
| `vllm:num_preemptions_total` | counter | `vllm` |
| `vllm:time_to_first_token_seconds` | histogram | `vllm` |
| `vllm:inter_token_latency_seconds` | histogram | `vllm` |
| `vllm:request_time_per_output_token_seconds` | histogram | `vllm` |
| `vllm:e2e_request_latency_seconds` | histogram | `vllm` |
| `sglang:num_running_reqs` | gauge | `sglang` |
| `sglang:num_queue_reqs` | gauge | `sglang` |
| `sglang:cache_hit_rate` | gauge | `sglang` |
| `sglang:num_requests_total` | counter | `sglang` |
| `sglang:time_to_first_token_seconds` | histogram | `sglang` |
| `sglang:inter_token_latency_seconds` | histogram | `sglang` |
| `sglang:request_time_per_output_token_seconds` | histogram | `sglang` |
| `sglang:e2e_request_latency_seconds` | histogram | `sglang` |
| `trtllm_num_requests_running` | gauge | `trtllm` |
| `trtllm_num_requests_waiting` | gauge | `trtllm` |
| `trtllm_kv_cache_utilization` | gauge | `trtllm` |
| `trtllm_time_to_first_token_seconds` | histogram | `trtllm` |
| `trtllm_time_per_output_token_seconds` | histogram | `trtllm` |
| `trtllm_e2e_request_latency_seconds` | histogram | `trtllm` |

## Engine Metrics

Backend workers also expose their inference engine's own metrics at `/metrics` on `DYN_SYSTEM_PORT`. The engine defines these metrics, so their names, types, and meanings follow the engine version you run. Dynamo passes them through with the engine's names, types, and values, and adds the `dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`, `worker_id`, `model`, and `model_name` labels.

| Engine | Prefix | Enabled by | Engine reference | Dynamo guide |
|--------|--------|------------|------------------|--------------|
| vLLM | `vllm:`, plus `lmcache:` with LMCache | On by default; `--disable-log-stats` turns it off | [vLLM Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/) | [vLLM Observability](../../developer-guide/knowledge-base/modular-components/backends/vllm/observability.md) |
| SGLang | `sglang:` | `--enable-metrics` | [SGLang Production Metrics](https://docs.sglang.io/docs/references/production_metrics) | [SGLang Observability](../../developer-guide/knowledge-base/modular-components/backends/sglang/observability.md) |
| TensorRT-LLM | `trtllm_` | `--publish-metrics` | [TensorRT-LLM metrics collector](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/metrics/collector.py) | [TensorRT-LLM Observability](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/observability.md) |

Some `trtllm_*` metrics are computed by Dynamo, not by TensorRT-LLM; see [TensorRT-LLM Metrics Computed by Dynamo](#tensorrt-llm-metrics-computed-by-dynamo). To compare equivalent metrics across engines, see [Metrics Comparison](metrics-comparison.md).

## Related

- [Metric Labels](metric-labels.mdx) — the dimensions attached to these metrics.
- [Environment Variables](environment-variables.mdx) — variables that enable and configure metric emission.
- [Operator Metrics](operator-metrics.mdx) — Kubernetes operator metrics catalog.
- [Metrics Comparison](metrics-comparison.md) — engine pass-through metrics (`vllm:*`, `sglang:*`, `trtllm_*`).
- [OTLP Metrics Export](otlp-metrics-export.mdx) — pushing metrics to an OpenTelemetry collector.
