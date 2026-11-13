---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Metrics Catalog
subtitle: Every metric Dynamo defines, grouped by the process that exposes it, with type, labels, and meaning.
---

Dynamo exposes metrics in Prometheus exposition format at the `/metrics` HTTP endpoint. This page lists every metric that Dynamo's own code defines; use it to look up a metric name you see in a scrape, dashboard, or alert. Each section lists its metrics in a table, followed by notes and PromQL examples where needed. [Metric Labels](metric-labels.mdx) documents the label values. For setup and dashboards, see the [Metrics guide](../../cli/operations/observability.mdx#view-metrics-and-dashboards).

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
- **histogram** — samples observations into configurable buckets and exposes `_bucket`, `_sum`, and `_count` series, e.g. request duration. The frontend latency and sequence-length histograms take their bucket ranges from environment variables — see [Histogram buckets](../components/frontend-configuration.mdx#histogram-buckets).

<Info>
  Labeled metrics (`HistogramVec`, `CounterVec`, `GaugeVec`) register a metric *family*, not individual time series. For request-populated metrics, a series for a given label combination appears at `/metrics` only after the first matching request. Other families create or remove series from runtime state; their entries below document those semantics.
</Info>

Every metric table on this page has the same columns:

- **Labels** lists the labels specific to the metric. Labels shared by every metric in a section are listed in the section introduction.
- **Status** is empty for a stable metric, which is renamed or removed only after it has been marked Deprecated. **Experimental** marks a preview metric whose name or meaning may change, and **Deprecated** marks a metric that is still emitted but scheduled for removal; its description names the replacement.

Metrics defined in Python, such as the planner metrics, also expose a `_created` series for each counter and histogram.

## Where each family is exposed

| Family | Emitted by | Scrape endpoint |
|--------|------------|-----------------|
| `dynamo_frontend_*` | HTTP frontend (`python -m dynamo.frontend`) | `/metrics` on `DYN_HTTP_PORT` (default 8000) |
| `dynamo_tokio_*`, `dynamo_request_plane_*`, `dynamo_transport_*` | Frontend | `/metrics` on `DYN_HTTP_PORT` |
| `dynamo_component_*` | Backend workers, standalone router, KV state agent, and the frontend's router and runtime metrics | `/metrics` on `DYN_SYSTEM_PORT` (must be set); `/metrics` on `DYN_HTTP_PORT` for the frontend |
| `dynamo_router_*` | Frontend | `/metrics` on `DYN_HTTP_PORT` |
| `dynamo_work_handler_*`, `dynamo_backend_admission_*`, `dynamo_embedding_*` | Backend workers | `/metrics` on `DYN_SYSTEM_PORT` |
| `trtllm_*` metrics that Dynamo computes | TensorRT-LLM workers | `/metrics` on `DYN_SYSTEM_PORT` |
| Imitation `vllm:*`, `sglang:*`, and `trtllm_*` metrics | Mocker (`python -m dynamo.mocker`) | `/metrics` on `DYN_SYSTEM_PORT` |
| `dynamo_quic_response_*` | Every process, with `--response-plane quic` or `DYN_RESPONSE_PLANE=quic` | `/metrics` on `DYN_SYSTEM_PORT`; `/metrics` on `DYN_HTTP_PORT` for the frontend |
| `dynamo_planner_*` | Planner (`python -m dynamo.planner`) | `/metrics` on the planner's own port; see [Planner Metrics](#planner-metrics) |
| `dynamo_kvindexer_*`, `dynamo_kvrouter_*` | Standalone KV indexer (`python -m dynamo.indexer`) | `/metrics` on `--port` |
| `dynamo_epp_*` | Endpoint picker (Gateway API Inference Extension routing) | `/metrics` on `DYN_EPP_METRICS_PORT` (default 9090) |
| `dynamo_global_frontend_*` | Global Frontend (`global-frontend`), which forwards each request to a pool frontend | `/metrics` on `--port` (default 8000) |
| `dynamo_power_agent_*` | Power agent DaemonSet (`deploy/power-agent`) | `/metrics` on `--prometheus-port` (default `0`, off; the Helm chart sets `9100`) |
| `dynamo_operator_*` | Kubernetes operator | See [Operator Metrics](operator-metrics.mdx) |

Backend workers expose `dynamo_component_*` only when `DYN_SYSTEM_PORT` is set (disabled by default; the Kubernetes operator typically sets `9090`, local examples use `8081`). See [Environment Variables](environment-variables.mdx#system-and-metrics).

## Frontend Metrics

Emitted by `python -m dynamo.frontend` at `/metrics` on the HTTP port (`--http-port`, default `8000`). The frontend serves its [router](#router-metrics) and [runtime](#every-process) metrics on the same port and does not use `DYN_SYSTEM_PORT`. Frontend metrics carry no automatic labels.

- `--metrics-prefix` (`DYN_METRICS_PREFIX`) replaces the `dynamo_frontend` prefix of the `dynamo_frontend_*` metrics, except `dynamo_frontend_stage_*`, `dynamo_frontend_tokenize_seconds`, `dynamo_frontend_template_seconds`, `dynamo_frontend_detokenize_*`, `dynamo_frontend_tokenizer_cache_*`, `dynamo_frontend_worker_*`, `dynamo_frontend_router_*`, `dynamo_frontend_lora_*`, `dynamo_frontend_event_loop_*`, and `dynamo_frontend_metrics_start_time_seconds`, which keep their names.
- With `--kserve-grpc-server`, the frontend serves metrics on `DYN_GRPC_METRICS_PORT` (default `8788`) instead. The model configuration and migration metrics, the `dynamo_router_*` metrics, and the metrics in the runtime registry (`dynamo_component_*`, including the router request metrics, and `dynamo_quic_response_*`) are not exposed in that mode; the `dynamo_frontend_router_*` metrics are.

For endpoints that canonicalize aliases, the `model` label holds the primary served model name — the first `--served-model-name` value — so a request sent to an alias is counted under the primary. A name the frontend has not registered is counted under `unknown_model`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_metrics_start_time_seconds` | gauge | — | Unix timestamp in seconds when the HTTP frontend first initializes its metrics in this process. One series per frontend scrape target; all HTTP registries in the process expose the same fixed value. Planner uses it alongside request counters to detect a new exporter lifetime and reset its rate baseline. This is not the OS process start time or a model readiness signal. | Experimental |
| `dynamo_frontend_active_requests` | gauge | `model` | Requests currently being handled by the frontend, from when the handler accepts the request, after body parsing, readiness checks, and validation, until the response stream completes. The top-level in-flight count with no stage breakdown. | |
| `dynamo_frontend_stage_requests` | gauge | `stage`, `phase` | Requests currently in a given frontend pipeline stage. See [Stage values](metric-labels.mdx#stage-values) and [Phase values](metric-labels.mdx#phase-values). | |
| `dynamo_frontend_inflight_requests` | gauge | `model` | In-flight requests, kept for backward compatibility. Use `dynamo_frontend_active_requests`, which has the same value. | Deprecated |
| `dynamo_frontend_queued_requests` | gauge | `model` | Requests in the HTTP processing queue: counted from when the request passes validation until its first response token, or until it ends without one. Unlike `dynamo_frontend_awaiting_first_token_requests`, it does not count requests during validation or System One requests. | |
| `dynamo_frontend_awaiting_first_token_requests` | gauge | `model` | Admitted requests that have not produced an output token yet: being tokenized, queued, routed, or prefilled. Requests to endpoints that never stream tokens (embeddings, images, videos, tensors) stay here until they finish. | Experimental |
| `dynamo_frontend_inflight_input_tokens` | gauge | `model`, `state` | Known prompt tokens of in-flight requests. `state` is `awaiting_first_token` (the requests in `dynamo_frontend_awaiting_first_token_requests`) or `generating`; sum over `state` for all in-flight prompt tokens. A request still being tokenized contributes zero. | Experimental |
| `dynamo_frontend_input_tokens_total` | counter | `model` | Prompt tokens of admitted requests, counted at tokenization, including requests that fail or are cancelled before producing output. | Experimental |
| `dynamo_frontend_requests_started_total` | counter | `model`, `endpoint`, `request_type` | Requests accepted by a frontend handler. | |
| `dynamo_frontend_requests_total` | counter | `model`, `endpoint`, `request_type`, `status`, `error_type` | Requests that finished, by outcome. | |
| `dynamo_frontend_failures_total` | counter | `class`, `reason` | Requests that ended in an error, by error class and reason. Cancellations are not counted. | |
| `dynamo_frontend_disconnected_clients` | gauge | — | Client disconnects since the frontend started. The value only increases. Use `dynamo_frontend_model_cancellation_total`. The old value is `sum(dynamo_frontend_model_cancellation_total)`. | Deprecated |
| `dynamo_frontend_time_to_first_token_seconds` | histogram | `model` | Time to first token in seconds. | |
| `dynamo_frontend_inter_token_latency_seconds` | histogram | `model` | Inter-token latency in seconds. | |
| `dynamo_frontend_request_duration_seconds` | histogram | `model` | End-to-end LLM request duration in seconds. | |
| `dynamo_frontend_embedding_latency_seconds` | histogram | `model` | Latency of successful `/v1/embeddings` requests. | |
| `dynamo_frontend_input_sequence_tokens` | histogram | `model` | Prompt length in tokens per request, recorded at the first output token. | |
| `dynamo_frontend_output_sequence_tokens` | histogram | `model` | Output sequence length in tokens. | |
| `dynamo_frontend_output_tokens_total` | counter | `model` | Total output tokens generated. | |
| `dynamo_frontend_cached_tokens` | histogram | `model` | Prefix-cache hit tokens per request, as reported by the worker. | |
| `dynamo_frontend_stage_duration_seconds` | histogram | `stage` | Time spent in each pipeline stage: `preprocess`, `route`, or `transport_roundtrip`. The `preprocess` stage is not recorded for `/v1/completions` or embeddings requests. | |
| `dynamo_frontend_tokenize_seconds` | histogram | — | Prompt tokenization time per request. Not recorded for `/v1/completions` or embeddings requests. | |
| `dynamo_frontend_template_seconds` | histogram | — | Chat template rendering time per request. | |
| `dynamo_frontend_tokenizer_latency_ms` | histogram | `operation` | Tokenizer latency in milliseconds, one observation per request: `tokenize` is the prompt's tokenization time, and `detokenize` is the average detokenization time per output token. | |
| `dynamo_frontend_detokenize_total_us` | counter | — | Total detokenization time in microseconds. | |
| `dynamo_frontend_detokenize_token_count` | counter | — | Tokens detokenized. | |
| `dynamo_frontend_tokenizer_cache_hits_total` | counter | — | L1 tokenizer prefix-cache hits across all models. One outcome is recorded per encode operation or batch item. A prompt encoded in segments, as some model renderers produce, records one outcome: a hit when any of its segments of at least 128 bytes is cached, and none when it has no such segment. | |
| `dynamo_frontend_tokenizer_cache_misses_total` | counter | — | L1 tokenizer prefix-cache misses across all models. | |
| `dynamo_frontend_tokenizer_cache_cached_tokens_total` | counter | `model` | Tokens returned from the L1 tokenizer prefix cache. | |
| `dynamo_frontend_tokenizer_cache_uncached_tokens_total` | counter | `model` | Tokens freshly encoded after an L1 tokenizer cache lookup. | |

`dynamo_frontend_awaiting_first_token_requests`, `dynamo_frontend_inflight_input_tokens`, and `dynamo_frontend_input_tokens_total` have a zero-valued series for each registered model, and for `unknown_model`, before the first request. With `--dyn-chat-processor vllm` or `sglang`, chat requests skip Dynamo's preprocessor, so they record no `preprocess` stage and no template, tokenization, detokenization, or tokenizer-cache metrics.

Requests producing output tokens are `clamp_min(dynamo_frontend_active_requests - dynamo_frontend_awaiting_first_token_requests, 0)`; the clamp covers the two gauges being sampled a moment apart. Unlike the sum of `dynamo_frontend_input_sequence_tokens`, which is observed at the first output token, `dynamo_frontend_input_tokens_total` counts every admitted request when it is tokenized.

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

The ratio is defined only when the model has an active L1 cache and observed tokens. Partial hits increment both token counters. When the cache is disabled with `DYN_TOKENIZER_CACHE=0`, the per-model series are never created, so they are absent from the scrape rather than reported as zero; detect that state with `absent()`, since a `== 0` comparison never matches a missing series. Only the literal value `0` disables the cache; values such as `false` or `off` leave it enabled. With the cache enabled, the counters remain zero when encoding fails, or when the tokenizer has no registered special tokens or has special tokens that can overlap, which turns the cache off.

### Multimodal Requests

These histograms carry a `model` label.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_images_per_request` | histogram | — | `image_url` content parts per request. Text-only requests record 0. | |
| `dynamo_frontend_videos_per_request` | histogram | — | `video_url` content parts per request. | |
| `dynamo_frontend_audio_per_request` | histogram | — | `audio_url` content parts per request. | |
| `dynamo_frontend_image_tokens_per_request` | histogram | — | Image placeholder tokens per request that contains images. Recorded only when every image resolves and no processor override is set. | |

### Migration

Metrics for [request migration](../../kubernetes/fault-tolerance/request-migration.md), the mechanism that continues in-flight requests on a healthy worker when the original worker fails. See [Request Migration Architecture](../../developer-guide/knowledge-base/concepts/fault-tolerance/request-migration-architecture.md) for the internals. These carry a `model` label.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_model_migration_total` | counter | [`migration_type`](metric-labels.mdx#metric-specific-labels) | Migratable worker failures, one per failed attempt: `new_request` when a response stream cannot be established, `ongoing_request` when an established stream fails. The last failure is counted even when no retry remains, including with the default `--migration-limit 0`, so this is not the number of migrations performed. | |
| `dynamo_frontend_model_migration_duration_seconds` | histogram | [`migration_type`](metric-labels.mdx#metric-specific-labels), [`outcome`](metric-labels.mdx#metric-specific-labels) | Time from detecting a migratable worker failure until the replacement response stream is established, migration ends without recovery, or the request is cancelled. | |
| `dynamo_frontend_model_migration_max_seq_len_exceeded_total` | counter | — | Times migration was disabled for a request because its sequence length exceeded `--migration-max-seq-len`. A rising value may indicate the limit needs adjustment. | |

Use the `_count` series of `dynamo_frontend_model_migration_duration_seconds` to count completed migration events by outcome. The duration excludes work completed before the failure and generation after the replacement stream is established.

### Cancellation and Rejection

Counters for requests that end early: [cancellations](../../developer-guide/knowledge-base/concepts/fault-tolerance/request-cancellation-architecture.md) (the client or frontend aborts an in-flight request) and [rejections](../../kubernetes/fault-tolerance/request-rejection.md) (the Frontend sheds load with HTTP 529 by default when all workers are busy). Both carry `model` and `endpoint` labels. The worker-side counterparts are under [Component metrics](#component-metrics).

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_model_cancellation_total` | counter | [`request_type`](metric-labels.mdx#metric-specific-labels) | Request cancellations detected by the frontend: client disconnect or stream close. `endpoint` is the API route (`chat_completions`, `completions`, `embeddings`, …); `request_type` is `unary` or `stream`. | |
| `dynamo_frontend_model_rejection_total` | counter | — | Overload responses surfaced to the client as HTTP 529 by default, for worker-scoped `WorkerOverloaded` and pool-scoped `ResourceExhausted` errors. Router queue rejections also return HTTP 529 but are not counted here; see `dynamo_frontend_router_admission_rejected_total`. | |

### Per-Worker Load and Timing Gauges

`dynamo_frontend_worker_*` gauges appear once workers register and begin serving. They live on the frontend's local registry (not component-scoped) and do **not** carry `dynamo_namespace` or `dynamo_component` labels. Frontend-only — not available on the standalone router. Labeled by `worker_id` (the worker's discovery ID in decimal), `dp_rank`, and `worker_type`.

Only the KV router writes the three `dynamo_frontend_worker_active_*` gauges, so they have no series in other router modes; the three `dynamo_frontend_worker_last_*` gauges are populated in every mode. At each HTTP scrape, these six gauges include only workers in the current committed router membership that pass hard availability checks, and only their current data-parallel (DP) ranks. The timing gauges also allow `dp_rank="none"` when the rank is unknown, provided the worker remains available. The `worker_type` label describes serving attribution: an Encode worker serving Chat/Completions reports its load and timing samples as `decode`, while the [router worker inventory](#router-worker-inventory) gauges below retain its `encode` topology role. Removed or excluded workers disappear from the output; late request completions or load updates cannot restore their exposed series.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_worker_active_decode_blocks` | gauge | — | Active KV cache decode blocks per worker. | |
| `dynamo_frontend_worker_active_prefill_tokens` | gauge | — | Active prefill tokens queued per worker. | |
| `dynamo_frontend_worker_active_requests` | gauge | [`request_phase`](metric-labels.mdx#metric-specific-labels) | Active requests booked per worker. A request counts as `prefill` from booking until its worker returns the first output token, then as `decode` until it completes; sum both phases for the worker's total concurrency. The count is the router's view, including requests booked by peer routers when router replica sync is enabled. | Experimental |
| `dynamo_frontend_worker_last_time_to_first_token_seconds` | gauge | — | Last observed time to first token per worker, in seconds. | |
| `dynamo_frontend_worker_last_input_sequence_tokens` | gauge | — | Last observed input sequence length per worker. | |
| `dynamo_frontend_worker_last_inter_token_latency_seconds` | gauge | — | Last observed inter-token latency per worker, in seconds. | |

### Router Worker Inventory

These frontend-only metrics describe each Frontend's independent view of normalized base model-card discovery and router admission. They do not report Kubernetes Pod existence, and LoRA overlays do not count as additional workers. Both metrics carry `model`, `target_namespace`, `target_component`, `target_endpoint`, `worker_type`, and `state` labels. The target labels identify the discovered serving endpoint.

Admission uses first-wins semantics within each Frontend's model and target group. The first observed checksum cohort remains the incumbent; later incompatible worker IDs are excluded individually and do not remove or replace it. If the incumbent cohort disappears, the next remaining cohort can become its successor after initialization.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_router_workers` | gauge | — | Number of unique physical worker IDs in a model and target group, by state. DP ranks do not increase the count. | Experimental |
| `dynamo_frontend_router_worker_state` | gauge | `router_worker_id`, `reason` | One sample with value `1` for each discovered worker, labeled with its current state and reason. The sample changes with the worker's state and disappears when discovery removes the ID. | Experimental |

The `available`, `pending`, and `excluded` states are disjoint; `discovered`, which only `dynamo_frontend_router_workers` reports, is their sum. `available` requires successfully committed worker-set membership and current hard availability; `pending` means discovery has observed the worker but initialization has not completed; `excluded` means discovered but currently unavailable for routing.

Hard availability uses the committed group's client and, when a prefill or encode hop targets that group, the hop's local availability. A matching hop that is still activating reports no available workers. If no hop targets the group, its own client availability applies. These counts describe individual worker groups; use `dynamo_frontend_model_ready` to check whether the frontend has a complete routable serving topology.

Each currently observed group emits all four counts, including zero values for states with no workers. When discovery withdraws a group after its last worker disappears, its count and per-worker state series disappear on the next scrape. Retired rolling-update namespaces are not retained as permanent zero-count groups. Groups never observed by that frontend also have no series. Do not aggregate these counts across Frontends as worker-global readiness; separate Frontends can choose different incumbents from the same discovered workers.

The bounded `reason` values describe the current admission or availability condition:

| Reason | Meaning |
|--------|---------|
| `none` | Available for routing |
| `initializing` | Waiting for initialization |
| `checksum_mismatch` | The individual worker's Model Deployment Card checksum differs from the incumbent; emitted only for rejected worker IDs |
| `materialization_failed` | Creating the serving configuration failed |
| `commit_blocked` | The serving configuration could not be committed |
| `invalid_config` | The worker's data-parallel rank range is invalid: `data_parallel_size` is 0 or above the supported maximum, or the range overflows |
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

All `dynamo_frontend_model_*` gauges carry a `model` label. The gauges other than `dynamo_frontend_model_ready` are populated from worker backend registration. Each time a worker group for the model is committed or rebuilt, its configuration overwrites these values, so with several groups, such as prefill and decode, they show the most recently committed one. A worker that joins an existing group does not change them. `dynamo_frontend_model_total_kv_blocks`, `dynamo_frontend_model_max_num_seqs`, and `dynamo_frontend_model_max_num_batched_tokens` change only when the committed configuration reports a value: a group that omits one keeps the previous value, and the series is absent until some group reports it.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_model_ready` | gauge | — | Whether the frontend can currently route at least one inference request for the model. `1` means at least one complete serving topology with a live worker and serving engine is available; `0` means the model is registered but not currently routable. | |
| `dynamo_frontend_model_total_kv_blocks` | gauge | — | Total KV blocks available for a worker serving the model. | |
| `dynamo_frontend_model_max_num_seqs` | gauge | — | Maximum number of sequences for a worker serving the model. | |
| `dynamo_frontend_model_max_num_batched_tokens` | gauge | — | Maximum number of batched tokens for a worker serving the model. | |
| `dynamo_frontend_model_context_length` | gauge | — | Maximum context length for a worker serving the model, from the Model Deployment Card. | |
| `dynamo_frontend_model_kv_cache_block_size` | gauge | — | KV cache block size for a worker serving the model, from the Model Deployment Card. | |
| `dynamo_frontend_model_migration_limit` | gauge | — | Migration limit in the worker's Model Deployment Card. Dynamo's workers always set it to `0`; the limit the frontend applies is its own `--migration-limit`, which no metric reports. | |

The frontend evaluates `dynamo_frontend_model_ready` from its live routing catalog at scrape time, so worker registration and removal are reflected in the next scrape. Unlike request-populated metrics, the series appears when the model registers, without requiring an inference request. The series is absent while the model is not registered, and the entire family is absent when no models are registered. Detect this state with `absent(dynamo_frontend_model_ready{model="<model>"})`; a `== 0` comparison does not match an absent series. Aliases get no series of their own: a deployment served under several names emits one series, labeled with its primary name, so query an alias through its primary. `/v1/models` still lists alias names, but only the primary has a readiness series. A LoRA adapter is a distinct model rather than a second name for one, and keeps its own series.

### System One Requests

Recorded for the experimental System One (`/v1/systemone`) and Decisions (`/v1/decisions`) endpoints, which `--enable-systemone-api` (`DYN_ENABLE_SYSTEMONE_API`) enables. These carry `model`. A request is recorded only after its preflight succeeds, and each phase only when it completes without an error.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_systemone_questions` | histogram | — | Questions per request. | Experimental |
| `dynamo_frontend_systemone_input_tokens` | histogram | — | Prompt tokens per request, summed over its questions. | Experimental |
| `dynamo_frontend_systemone_candidate_ids` | histogram | — | Candidate token IDs per request. | Experimental |
| `dynamo_frontend_systemone_cached_tokens` | histogram | — | Cached prompt tokens per completed question, when the worker reports them. | Experimental |
| `dynamo_frontend_systemone_admitted_branches_total` | counter | — | Questions admitted for dispatch. Use `dynamo_frontend_systemone_questions_sum`, which has the same value. | Deprecated |
| `dynamo_frontend_systemone_phase_seconds` | histogram | `phase` | Time in each phase: `preflight` (validating and tokenizing the request), `first_branch` (running the first question), or `fanout` (running the remaining questions on the same worker). | Experimental |

### LoRA Allocation

Registered on every frontend and populated only when LoRA serving is enabled with `DYN_LORA_ENABLED` and the allocation controller is not turned off with `DYN_LORA_ALLOCATION_ENABLED=false`. These carry an `endpoint` label, `namespace/component/endpoint`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_lora_active_requests` | gauge | `lora` | In-flight requests per LoRA adapter. | |
| `dynamo_frontend_lora_estimated_load` | gauge | `lora` | Estimated load per adapter: the windowed arrival count smoothed by the `DYN_LORA_ALLOCATION_PREDICTOR_TYPE` predictor (`ema` by default; `none` uses the raw count), rounded. | |
| `dynamo_frontend_lora_raw_arrival_count` | gauge | `lora` | Windowed arrival count per adapter. | |
| `dynamo_frontend_lora_replica_factor` | gauge | `lora` | Replicas allocated to each adapter. | |
| `dynamo_frontend_lora_is_active` | gauge | `lora` | 1 if the adapter is active, 0 if not. | |
| `dynamo_frontend_lora_churn_loads_total` | gauge | — | Adapter loads in the latest allocation step. A per-step gauge despite its `_total` suffix; non-zero only with `DYN_LORA_ALLOCATION_ALGORITHM=mcf`. | |
| `dynamo_frontend_lora_churn_unloads_total` | gauge | — | Adapter unloads in the latest allocation step. A per-step gauge despite its `_total` suffix; non-zero only with `DYN_LORA_ALLOCATION_ALGORITHM=mcf`. | |
| `dynamo_frontend_lora_overflow_count` | gauge | — | Replicas the allocator could not place in the latest step. Non-zero only with `DYN_LORA_ALLOCATION_ALGORITHM=mcf`. | |

### Async Runtime and Request Plane

Health of the frontend's Tokio async runtime and of its request path to workers. Frontend only.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_event_loop_delay_seconds` | histogram | — | Delay of a 10 ms timer on the async runtime. Higher values mean the runtime is overloaded. | |
| `dynamo_frontend_event_loop_stall_total` | counter | — | Timer delays longer than 5 ms. Use `dynamo_frontend_event_loop_delay_seconds`. The old value is `dynamo_frontend_event_loop_delay_seconds_count - ignoring(le) dynamo_frontend_event_loop_delay_seconds_bucket{le="0.005"}`. | Deprecated |
| `dynamo_tokio_alive_tasks` | gauge | — | Live async tasks. | |
| `dynamo_tokio_global_queue_depth` | gauge | — | Tasks in the runtime's global queue. | |
| `dynamo_tokio_blocking_threads` | gauge | — | Threads in the blocking pool. | |
| `dynamo_tokio_blocking_idle_threads` | gauge | — | Idle threads in the blocking pool. | |
| `dynamo_tokio_blocking_queue_depth` | gauge | — | Tasks waiting for a blocking-pool thread. | |
| `dynamo_tokio_budget_forced_yield_total` | counter | — | Times a task was forced to yield after using its budget. | |
| `dynamo_tokio_queue_overload_warnings_total` | counter | — | Warnings logged for sustained runnable-queue pressure. | |
| `dynamo_tokio_worker_busy_ratio` | gauge | `worker` | Mean task poll time of each runtime thread, rescaled to 0–1000 (1 ms or more reads 1000); it does not measure busy time. Use `dynamo_tokio_worker_mean_poll_time_ns`. The old value is `floor(clamp_max(dynamo_tokio_worker_mean_poll_time_ns / 1e6, 1) * 1000)`. | Deprecated |
| `dynamo_tokio_worker_mean_poll_time_ns` | gauge | `worker` | Mean task poll time of each runtime thread, in nanoseconds. | |
| `dynamo_tokio_worker_local_queue_depth` | gauge | `worker` | Tasks in each runtime thread's local queue. | |
| `dynamo_tokio_worker_park_count_total` | counter | `worker` | Times each runtime thread parked. | |
| `dynamo_tokio_worker_steal_count_total` | counter | `worker` | Tasks each runtime thread stole from other threads. | |
| `dynamo_tokio_worker_overflow_count_total` | counter | `worker` | Times each runtime thread's local queue overflowed. | |
| `dynamo_request_plane_inflight_requests` | gauge | — | Requests dispatched to workers that have not finished, counted from the start of dispatch. | |
| `dynamo_request_plane_queue_seconds` | histogram | — | Time from dispatch until the request is ready to send: registering its response stream and encoding it. The send is in `dynamo_request_plane_send_seconds`. | |
| `dynamo_request_plane_send_seconds` | histogram | — | Time to send a request to a worker and receive its acknowledgement. | |
| `dynamo_request_plane_roundtrip_ttft_seconds` | histogram | — | Time from sending a request until its first response arrives. | |
| `dynamo_transport_tcp_bytes_sent_total` | counter | — | Request bytes sent over TCP. | |
| `dynamo_transport_tcp_bytes_received_total` | counter | — | Acknowledgement bytes received over TCP. Streamed responses are not counted. | |
| `dynamo_transport_tcp_errors_total` | counter | — | Failed TCP requests, including send failures and timeouts. | |
| `dynamo_transport_tcp_accept_backoff_total` | counter | — | Response-server accept failures caused by descriptor or memory exhaustion. | |
| `dynamo_transport_nats_errors_total` | counter | `error_type` | NATS request errors. `error_type` is `max_payload_exceeded` (rejected before sending) or `request_failed`. Only with the NATS request plane. | |

## Component Metrics

Emitted by backend workers (`python -m dynamo.vllm`, `python -m dynamo.sglang`, etc.) at `/metrics` on `DYN_SYSTEM_PORT`. Metrics registered on an endpoint carry the hierarchy labels (`dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`) and `worker_id` — see [Runtime-injected labels](metric-labels.mdx#runtime-injected-labels). Metrics created on a component rather than an endpoint, such as `dynamo_backend_admission_*`, have no `dynamo_endpoint`, and `dynamo_work_handler_*` has no labels. Workers also expose their inference engine's own metrics; see [Engine Metrics](#engine-metrics).

The request metrics below are recorded for every endpoint a process serves. vLLM workers also add `model` and `model_name` on their inference endpoints, except embedding, classify, and multimodal encode workers, which add only `model`. SGLang workers add `model`, except multimodal encode workers, which add `model` and `model_name`, and image and video diffusion workers, which add neither. TensorRT-LLM workers add `model` and `model_name` only with `--publish-kv-events`, `--publish-metrics`, or forward-pass metrics (`--fpm-trace` or `DYN_FORWARDPASS_METRIC_PORT`). Internal endpoints, such as the `worker_kv_query_source_*` endpoint the router uses to recover a worker's KV index, have no `model` label.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_inflight_requests` | gauge | — | Requests currently being processed by the component. | |
| `dynamo_component_request_bytes_total` | counter | — | Total bytes received in requests. | |
| `dynamo_component_request_duration_seconds` | histogram | — | Time to handle a request, until its response stream ends. | |
| `dynamo_component_requests_total` | counter | — | Total requests processed. | |
| `dynamo_component_errors_total` | counter | `error_type` | Errors encountered while handling a request. See [Component error types](metric-labels.mdx#component-error-types). | |
| `dynamo_component_response_bytes_total` | counter | — | Total bytes sent in responses. | |
| `dynamo_component_cancellation_total` | counter | — | Requests cancelled by the work handler. Counts cancellation signals received by the worker, not whether the engine actually aborted, deduplicated so a control message plus a socket close for the same request counts once. See [Request Cancellation Architecture](../../developer-guide/knowledge-base/concepts/fault-tolerance/request-cancellation-architecture.md). | |

The shared TCP request plane, the default, exposes the following `dynamo_work_handler_*` metrics. They are
registered once per process and are independent of backend admission; see [Backend Admission Control](../../developer-guide/knowledge-base/concepts/fault-tolerance/admission-control-architecture.md#backend).
With the NATS request plane, all except `dynamo_work_handler_time_to_first_response_seconds` stay at 0.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_work_handler_pool_active_tasks` | gauge | — | TCP request handlers currently holding worker-pool permits, from dispatch until downstream handling finishes. | |
| `dynamo_work_handler_pool_capacity` | gauge | — | Configured capacity of the TCP worker pool, set by `DYN_TCP_WORKER_POOL_SIZE`. | |
| `dynamo_work_handler_queue_depth` | gauge | — | Requests currently in the bounded TCP work queue awaiting dispatcher pickup. | |
| `dynamo_work_handler_queue_capacity` | gauge | — | Configured capacity of the bounded TCP work queue, set by `DYN_TCP_WORK_QUEUE_SIZE`. | |
| `dynamo_work_handler_enqueue_rejected_total` | counter | — | Requests rejected before TCP worker dispatch because the bounded work queue was full or the dispatcher channel was closed. | |
| `dynamo_work_handler_permit_wait_seconds` | histogram | — | Time a queued request spends waiting for a TCP worker-pool permit after dispatcher pickup. | |
| `dynamo_work_handler_network_transit_seconds` | histogram | — | Time from the frontend sending a request until a TCP worker-pool task starts handling it, computed from the clocks on both hosts. For a request that waits in the work queue, it includes the queue time and the wait for a pool permit. | |
| `dynamo_work_handler_time_to_first_response_seconds` | histogram | — | Time from receiving a request until its response stream opens. This is not the time to the first token. | |

The `dynamo_backend_admission_*` metrics track engine occupancy, queueing, and request outcomes.
These metrics count requests only when at least one engine limit is configured. With both engine limits
unlimited, they remain at zero.
See [Backend Admission Control](../../developer-guide/knowledge-base/concepts/fault-tolerance/admission-control-architecture.md#backend)
for gate behavior.

These carry a `dp_rank` label: the rank ID, such as `"4"`, or `"unspecified"` for unranked requests.
Sum across all `dp_rank` values for worker totals.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_backend_admission_engine_request_count` | gauge | — | Requests admitted to the engine that have not finished. | Experimental |
| `dynamo_backend_admission_engine_request_waiting_count` | gauge | — | Admitted requests still awaiting their first generated token or completion. | Experimental |
| `dynamo_backend_admission_request_queue_count` | gauge | — | Requests waiting outside the engine in gate queues. | Experimental |
| `dynamo_backend_admission_request_receive_total` | counter | — | All requests received by the gate, counted once regardless of outcome. | Experimental |
| `dynamo_backend_admission_request_admit_total` | counter | `source` | Requests passed into the engine. `source="direct"` means admitted without queueing; `source="queue"` means admitted after queueing. | Experimental |
| `dynamo_backend_admission_rejection_total` | counter | — | Requests the gate could neither admit nor queue. | Experimental |
| `dynamo_backend_admission_cancellation_total` | counter | — | Requests cancelled before engine admission. | Experimental |

### KV Event Publishing

Workers that publish KV cache events to the router: vLLM and SGLang with `--kv-events-config`, TensorRT-LLM with `--publish-kv-events`, unified backend workers that declare KV event sources, and the mocker. The `kv_publisher` metrics are created on the component, so they have no `dynamo_endpoint`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_kv_publisher_engines_dropped_events_total` | counter | — | Gaps detected in KV event IDs before batching. On the ZMQ path, where the publisher assigns the IDs, gaps in the engine's own sequence numbers are logged, not counted. | |
| `dynamo_component_kv_publisher_zmq_events_total` | counter | `stage`, `event_type` | KV events received from the engine over ZMQ, and accepted. | |
| `dynamo_component_kv_publisher_zmq_filtered_events_total` | counter | `event_type`, `reason` | KV events filtered out before conversion. | |
| `dynamo_component_kv_publisher_zmq_conversion_issues_total` | counter | `event_type`, `reason` | KV events dropped because they could not be converted. | |
| `dynamo_component_kv_publisher_zmq_suspicious_events_total` | counter | `event_type`, `reason` | Suspicious KV events that were still forwarded. | |
| `dynamo_component_worker_prefill_completion_enqueue_failures_total` | counter | `reason` | Prefill-completion events dropped because the local publish queue was full or closed. vLLM, SGLang, and TensorRT-LLM aggregated and decode workers only, whether or not they publish KV events. | |

### Engine State

Label sets depend on the worker type:

- `python -m dynamo.vllm`, `dynamo.sglang`, and `dynamo.trtllm` add `model` and `dynamo_component` instead of the automatic labels. TensorRT-LLM updates the cache gauges only with `--publish-metrics` or forward-pass metrics (`--fpm-trace` or `DYN_FORWARDPASS_METRIC_PORT`).
- [Unified backend](../../developer-guide/advanced-customizations/writing-custom-backends/writing-unified-backends.md) workers use the automatic labels plus `model` and `model_name`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_total_blocks` | gauge | `dp_rank` | Total KV cache blocks on the worker. | |
| `dynamo_component_gpu_cache_usage_percent` | gauge | `dp_rank` | GPU KV cache usage, from 0.0 to 1.0. | |
| `dynamo_component_kv_cache_hit_rate` | gauge | `dp_rank` | Prefix cache hit rate, from 0.0 to 1.0. Registered, but no in-tree engine reports it yet. | |
| `dynamo_component_model_load_time_seconds` | gauge | — | Time to load the model and start the engine. | |
| `dynamo_component_drain_time_seconds` | gauge | — | Time a prefill worker waits at shutdown, after the grace period, before releasing the engine: until the engine reports its KV transfers done, or for `DYN_PREFILL_DRAIN_TIMEOUT_S` (default 30 seconds, capped at the graceful-shutdown timeout minus 5 seconds). No in-tree engine reports that yet, so prefill workers wait the full time. About 0 on aggregated and decode workers. Unified backend workers only. | |
| `dynamo_component_cleanup_time_seconds` | gauge | — | Time spent releasing engine resources at shutdown. Unified backend workers only. | |

### Multimodal and Embedding Caches

These metrics come from several publishers, so their labels differ:

- The image cache metrics, and the embedding cache metrics a worker records in its own process, carry `model` and `dynamo_component`.
- With the `DynamoMultimodalEmbeddingCacheConnector`, the vLLM scheduler records the embedding cache metrics once per data-parallel rank. They carry `model`, `model_name`, `dp_rank`, `dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`, and `worker_id`.
- The SGLang `dynamo_embedding_*` histograms carry `model`, `model_name`, `dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`, and `worker_id`.

The embedding cache metrics require `--multimodal-embedding-cache-capacity-gb` greater than 0; TensorRT-LLM workers also need `--publish-kv-events`, `--publish-metrics`, or forward-pass metrics. vLLM encode workers keep an embedding cache but do not export these metrics. The image cache metrics are registered on vLLM multimodal workers, and the shared image cache records samples only with `DYN_MM_SHARED_IMAGE_CACHE_ENABLED=1`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_embedding_cache_hits_total` | counter | — | Embedding cache hits. | |
| `dynamo_component_embedding_cache_misses_total` | counter | — | Embedding cache misses. | |
| `dynamo_component_embedding_cache_evictions_total` | counter | — | Embedding cache evictions. | |
| `dynamo_component_embedding_cache_utilization` | gauge | — | Embedding cache memory use, from 0.0 to 1.0. | |
| `dynamo_component_embedding_cache_current_bytes` | gauge | — | Embedding cache memory use in bytes. | |
| `dynamo_component_embedding_cache_entries` | gauge | — | Entries in the embedding cache. | |
| `dynamo_component_image_cache_entries` | gauge | — | Entries in the in-process image cache. | |
| `dynamo_component_image_shared_cache_get_duration_seconds` | histogram | `outcome`, `size_bucket` | Shared image cache read latency, by outcome and payload size. | |
| `dynamo_component_image_shared_cache_set_duration_seconds` | histogram | `outcome`, `size_bucket` | Shared image cache write latency, by outcome and payload size. | |
| `dynamo_embedding_batch_size` | histogram | — | Inputs per `/v1/embeddings` request. SGLang embedding workers (`--embedding-worker`) only. | |
| `dynamo_embedding_input_tokens` | histogram | — | Prompt tokens per `/v1/embeddings` request. SGLang embedding workers only. | |

### TensorRT-LLM Metrics Computed by Dynamo

TensorRT-LLM workers started with `--publish-metrics` add these to the engine's own `trtllm_*` metrics; the `trtllm_kv_transfer_*` metrics also need `--publish-per-request-metrics`. They carry `model_name`, `disaggregation_mode`, and `engine_type`, plus the labels Dynamo adds to [engine metrics](#engine-metrics).

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `trtllm_num_aborted_requests_total` | counter | — | Requests aborted or cancelled. | |
| `trtllm_request_type_image_total` | counter | — | Requests that contain images. | |
| `trtllm_request_type_structured_output_total` | counter | — | Requests that use structured (guided) output. | |
| `trtllm_kv_transfer_success_total` | counter | — | Successful KV cache transfers. Use `trtllm_kv_transfer_bytes_count`, which has the same value. | Deprecated |
| `trtllm_kv_transfer_latency_seconds` | histogram | — | KV cache transfer time per request. | |
| `trtllm_kv_transfer_bytes` | histogram | — | KV cache transfer size per request, in bytes. | |
| `trtllm_kv_transfer_speed_gb_s` | histogram | — | KV cache transfer speed per request, in GB/s. | |
| `trtllm_kv_event_buffer_capacity` | gauge | — | Configured size of the engine's KV event buffer when KV events are polled; 0 otherwise. | |
| `trtllm_kv_event_drain_batch_size` | histogram | — | KV events returned per non-empty polling drain. Only when KV events are polled. | |
| `trtllm_kv_event_id_gap_events_total` | counter | — | Missing KV event IDs detected by Dynamo. Only when KV events are polled. | |

### Engine Failover

vLLM workers with `--gms-shadow-mode` (GPU Memory Service shadow failover). These carry `engine_id`, `model`, and `dynamo_component`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_engine_failover_state` | gauge | `state` | 1 for the engine's current failover state: `init`, `standby`, `waking`, or `active`. | |
| `dynamo_component_engine_failover_state_entered_timestamp_seconds` | gauge | — | Unix time the engine entered its current state. | |
| `dynamo_component_engine_failover_last_state_duration_seconds` | gauge | `state` | Duration of the most recent stay in each state. For `waking`, this is the switch time. | |
| `dynamo_component_engine_failover_transitions_total` | counter | `from_state`, `to_state` | Failover state transitions. | |
| `dynamo_component_engine_failover_switch_attempts_total` | counter | — | Promotions attempted after a standby took the failover lock from a previous holder; the first promotion at startup is not counted. Kept across restarts. | |
| `dynamo_component_engine_failover_switch_success_total` | counter | — | Of those promotions, the ones that completed and began serving. Kept across restarts. | |

## Router Metrics

Emitted by every process that routes requests: the frontend, which serves them on its HTTP port, and the standalone router (`python -m dynamo.router`), which serves them on `DYN_SYSTEM_PORT`. The router adds `dynamo_namespace` and `dynamo_component` of the first worker component the process routes to (one value per process, even when it routes to both prefill and decode workers), and `worker_id`, which is the router's own ID in hex. The request metrics also carry `router_id`, the same ID in decimal. Not every metric appears in every deployment — see [Availability by Configuration](#availability-by-configuration).

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_router_worker_registered` | gauge | `router_worker_id`, `dp_rank`, `worker_type` | A sample with value `1` for each worker and DP rank in the KV router scheduler's current registration and hard-availability view; absent in other router modes. Removed or unavailable workers disappear from the output rather than emitting `0`. | |

The `dynamo_component_router_worker_registered` sample can appear during scheduler preparation, before worker-set materialization commits, so it does not indicate committed model readiness. Late request bookkeeping cannot restore a removed worker's series. Prefer the [Router Worker Inventory](#router-worker-inventory) metrics for committed available-worker counts and exclusion reasons on the frontend.

### Router Request Metrics

`dynamo_component_router_*` histograms and counters for aggregate request-level statistics. Populated per routing attempt in every router mode. In disaggregated serving, the prefill and decode hops record into the same series, so the request counters count a request once per hop, and the time-to-first-token and output-length histograms also get a prefill-hop sample when the prefill worker returns a token, as vLLM's does. The KV-specific metrics (`kv_hit_rate`, the [KV-cache reuse](#kv-cache-reuse-telemetry) counters, `non_max_overlap_selections_total`, and `overlap_blocks_lost`) are recorded only with `--router-mode kv`, and `non_max_overlap_selections_total` and `overlap_blocks_lost` only by a router that routes to prefill workers; in aggregated serving they stay at zero.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_router_requests_started_total` | counter | — | Requests admitted by the router scheduler. | |
| `dynamo_component_router_requests_total` | counter | — | Total requests processed by the router. | |
| `dynamo_component_router_time_to_first_token_seconds` | histogram | — | Time to first token in seconds. | |
| `dynamo_component_router_inter_token_latency_seconds` | histogram | — | Average inter-token latency in seconds, recorded once per request when it completes. With `--router-mode kv`, it is recorded only when `--router-track-output-blocks` is set, as the request's running average at each output block, so a request adds one sample per output block; without the flag the histogram stays empty. | |
| `dynamo_component_router_input_sequence_tokens` | histogram | `phase`, `model` | Input sequence length per routing attempt, excluding query-only lookups. | |
| `dynamo_component_router_output_sequence_tokens` | histogram | — | Output sequence length in tokens. | |
| `dynamo_component_router_kv_hit_rate` | histogram | — | Predicted KV cache hit rate at routing time, from 0.0 to 1.0. | |
| `dynamo_component_router_non_max_overlap_selections_total` | counter | `worker_type` | Admitted prefill scheduler selections routed to a worker with less KV cache overlap than another eligible worker, with the exclusions listed for `dynamo_component_router_overlap_blocks_lost`. `worker_type` is `prefill`. Use `dynamo_component_router_overlap_blocks_lost_count`, which has the same value. | Deprecated |
| `dynamo_component_router_overlap_blocks_lost` | histogram | `worker_type` | Difference in effective KV cache overlap between the highest-overlap eligible prefill worker and the selected worker, for each admitted prefill selection routed to a worker with less overlap than another eligible worker. [Explicitly pinned requests](../../developer-guide/knowledge-base/modular-components/router/worker-filtering.md#candidate-eligibility) (direct routing or an explicit worker target) and equal-overlap ties are excluded. Session-affinity selections are compared against the best eligible worker when the selection policy has no worker filters of its own (the default); otherwise they are excluded like pinned requests. `worker_type` is `prefill`. | |
| `dynamo_component_router_kv_transfer_estimated_latency_seconds` | histogram | — | Upper-bound estimate of the KV cache transfer time from prefill to decode. Recorded only in disaggregated serving. | |

### KV-Cache Reuse Telemetry

The cache reuse funnel shows how much of the incoming prompt could be reused, and how much the worker reports reusing. Each step counts tokens, not requests. The first step, F0, is the tokens the router receives, recorded in every router mode: `dynamo_component_router_input_sequence_tokens_sum`. These counters record the other steps. They carry `router_id`, `model`, and `phase`, and are recorded only with `--router-mode kv`; they need no other flag:

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_router_kv_best_eligible_cached_prefix_tokens_total` | counter | — | F2, best available match: prefix tokens the router estimates are cached on the best-matching eligible worker. | |
| `dynamo_component_router_kv_selected_cached_prefix_tokens_total` | counter | — | F3, selected worker's match: prefix tokens the router estimates are cached on the worker it chooses. | |
| `dynamo_component_router_kv_worker_reused_tokens_total` | counter | — | F4, reported reuse: tokens the worker reports reusing. Missing reports count as zero. | |

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

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_router_overhead_block_hashing_ms` | histogram | — | Time computing block hashes. | |
| `dynamo_router_overhead_indexer_find_matches_ms` | histogram | — | Time in indexer `find_matches`. | |
| `dynamo_router_overhead_seq_hashing_ms` | histogram | — | Time computing sequence hashes. | |
| `dynamo_router_overhead_scheduling_ms` | histogram | — | Time in scheduler worker selection, including time waiting in the router queue when queueing is enabled. | |
| `dynamo_router_overhead_shared_cache_query_ms` | histogram | — | Time querying the shared KV cache. Recorded only with `--shared-cache-type hicache`. | |
| `dynamo_router_overhead_total_ms` | histogram | — | Total routing overhead per request, including time waiting in the router queue when queueing is enabled. | |

### Router Queue Metrics

These metrics have series once a KV router starts, whether or not queueing is enabled. The `dynamo_frontend_router_queue_*` metrics stay at zero unless queueing is enabled by `--router-queue-threshold` or a threshold in `--router-policy-config`; the `dynamo_frontend_router_admission_*` counters are recorded either way. They carry `model`, `worker_type`, and `policy_class`, but not the `dynamo_namespace`, `dynamo_component`, and `worker_id` labels of the other router metrics. `policy_class` is the resolved class: a request that names no class is reported as the default class, and a request whose class, after any request-classifier override, is unknown fails before it is queued, so these queue metrics do not count it.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_router_queue_pending_requests` | gauge | — | Requests pending in the router scheduler queue. | |
| `dynamo_frontend_router_queue_pending_isl_tokens` | gauge | — | Raw input-sequence tokens pending in the queue. | |
| `dynamo_frontend_router_queue_pending_cached_tokens` | gauge | — | Cached-token estimate captured when each request enters the queue. | |
| `dynamo_frontend_router_queue_backpressure_total` | counter | `reason` | Queue rejections, by the configured limit that caused them. | |
| `dynamo_frontend_router_admission_received_total` | counter | — | Attempts received by router admission, including retries. | |
| `dynamo_frontend_router_admission_rejected_total` | counter | `reason` | Scheduling attempts rejected before admission, by `reason`: `due_time_passed` or `queue_size_exceeded`. The `predicted_miss` reason is reserved and stays zero. | |

Each rejected attempt is counted once in `dynamo_frontend_router_admission_rejected_total`, labeled `due_time_passed` when the request's explicit deadline, or the budget of an `edf` class with `controlled_delay`, passes before admission, or `queue_size_exceeded` when a class queue request or token limit rejects it. An attempt whose response receiver is already closed when the router checks it, because the client went away, is not counted; a receiver that closes after that check can still be counted. A `queue_size_exceeded` rejection is also counted, by limit, in `dynamo_frontend_router_queue_backpressure_total` when the frontend's selection call returns that queue rejection; a cancellation can drop the call before it returns, so the two counts can differ. They describe the same rejections: do not add them together.

### KV Indexer Metrics

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_kv_cache_events_applied` | counter | `status`, `event_type` | KV cache events applied to the router's radix-tree index, including events for lower tiers such as host-pinned memory and disk. See [Metric-specific labels](metric-labels.mdx#metric-specific-labels). | |
| `dynamo_component_ckf_mutation_total` | counter | `outcome` | CKF block-level mutation outcomes recorded by the router indexer. Always 0: nothing increments it. No replacement; remove it from dashboards and alerts. | Deprecated |
| `dynamo_component_kv_cache_event_warnings` | counter | `warning_kind` | Suspicious KV cache events seen by the index, such as duplicate stores. | |
| `dynamo_component_router_kv_event_source_mismatch_workers` | gauge | `model`, `worker_type`, `target_namespace`, `target_component`, `target_endpoint` | Serving worker DP ranks whose KV event source is missing, ambiguous, or disabled, or that lack an expected recovery target. While the KV event subscription fails, every rank counts. | |
| `dynamo_component_router_kv_zmq_ingress_sources` | gauge | `state` | KV event sources on the ZMQ event plane, by lifecycle state. | |
| `dynamo_component_router_kv_zmq_ingress_batches_total` | counter | — | KV event batches received over ZMQ. | |
| `dynamo_component_router_kv_zmq_ingress_lifecycle_total` | counter | `action` | Lifecycle transitions and errors of ZMQ KV event sources. | |
| `dynamo_component_router_remote_indexer_query_failures_total` | counter | `router_id` | Failed overlap queries to a remote indexer. Only with `--use-remote-indexer`. | |
| `dynamo_component_router_remote_indexer_write_failures_total` | counter | `router_id` | Failed routing-decision writes to a remote indexer. Only with `--use-remote-indexer`. | |

`dynamo_component_kv_cache_events_applied` appears, at zero, once the router builds a local KV index: when the worker-selection policy requests `WorkerInputs::CACHE` or `WorkerInputs::KV_TRANSFER`, and either `--use-remote-indexer` is not set or `--router-predicted-ttl-secs` is set. It counts the KV events workers publish and the router's own predicted stores and expirations, which it makes with KV events disabled or with `--router-predicted-ttl-secs`. With `--router-event-threads 1`, TTL-based predicted events are not counted. The standalone indexer exposes the device-tier-only counter `dynamo_kvrouter_kv_cache_events_applied`; see [Standalone KV Indexer Metrics](#standalone-kv-indexer-metrics).

### Shared KV Cache

Recorded only with `--shared-cache-type hicache`. These carry a `router_id` label.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_router_shared_cache_hit_rate` | histogram | — | Fraction of request blocks found in the shared KV cache. | |
| `dynamo_component_router_shared_cache_beyond_blocks` | histogram | — | Shared-cache blocks beyond the selected worker's device overlap. | |
| `dynamo_router_shared_cache_errors_total` | counter | — | Failed shared-cache queries and subscriber errors. Frontend only; carries only `router_id`, like the [routing overhead](#per-request-routing-overhead) metrics. | |

### Active Sequence Synchronization

Track the active-request events each KV router applies to its view of worker load. The `active_sequence_ingress` metrics carry `model`, `routing_group`, and `worker_type`. The `active_sequence_zmq_ingress` metrics exist when these events travel over direct ZMQ, the default on the ZMQ event plane.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_router_active_sequence_ingress_queue_depth` | gauge | — | Active-sequence events waiting to be applied. | |
| `dynamo_component_router_active_sequence_ingress_events_applied_total` | counter | — | Active-sequence events applied. | |
| `dynamo_component_router_active_sequence_ingress_drain_batches_total` | counter | — | Event batches applied. | |
| `dynamo_component_router_active_sequence_zmq_ingress_sources` | gauge | — | Tracked ZMQ sources of active-sequence events. | |
| `dynamo_component_router_active_sequence_zmq_ingress_sequence_gaps_total` | counter | — | Event batches (envelopes) missing, inferred from sequence-number gaps. | |
| `dynamo_component_router_active_sequence_zmq_ingress_out_of_order_total` | counter | — | Event batches received out of order. | |
| `dynamo_component_router_active_sequence_zmq_ingress_reconnects_total` | counter | — | Source reconnect attempts. | |
| `dynamo_component_router_active_sequence_zmq_ingress_replacements_total` | counter | — | Source task replacements. | |
| `dynamo_component_router_active_sequence_zmq_ingress_envelope_decode_errors_total` | counter | — | Envelope decode errors. | |
| `dynamo_component_router_active_sequence_zmq_ingress_payload_decode_errors_total` | counter | — | Payload decode errors, including oversized batches. | |
| `dynamo_component_router_active_sequence_zmq_ingress_identity_errors_total` | counter | — | Frame and envelope identity mismatches. | |
| `dynamo_component_router_active_sequence_zmq_ingress_forced_aborts_total` | counter | — | Source tasks aborted after a join timeout. | |
| `dynamo_component_router_active_sequence_zmq_ingress_dropped_events_total` | counter | — | Events dropped because the inbound queue was full or closed. | |

### Approximate Cache

The approximate cache tracks likely-cached blocks when the router does not consume KV events. The two policy gauges are present on every KV router. The LRU metrics update only with `--router-approximate-cache-policy lru`, KV events disabled, and neither `--use-remote-indexer` nor `--serve-indexer`; with either of those, the router falls back to TTL.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_router_approximate_cache_configured_policy` | gauge | `policy` | 1 for the configured policy, `ttl` or `lru`; the `disabled` series is always 0. | Experimental |
| `dynamo_component_router_approximate_cache_effective_policy` | gauge | `policy` | 1 for the policy in effect: `disabled` when the router builds no KV index, `lru` with the approximate LRU, and `ttl` otherwise, including when the router consumes KV events. | Experimental |
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
| `dynamo_component_router_*` | Registered and populated; `non_max_overlap_selections_total`, `overlap_blocks_lost`, and `kv_transfer_estimated_latency_seconds` stay at zero | Registered and populated | Request, latency, and sequence-length metrics populated; `kv_hit_rate`, `non_max_overlap_selections_total`, `overlap_blocks_lost`, and the shared-cache histograms **always zero**; other KV-specific metrics **not created** | Populated (on `DYN_SYSTEM_PORT`) |
| `dynamo_router_overhead_*` | Registered and populated | Registered and populated | Registered, **always zero** | **Not created** |
| `dynamo_frontend_router_queue_*` | Registered; populated when a CLI or policy-class queue threshold is set | Registered; populated when a CLI or policy-class queue threshold is set | **Not registered** | **Not created** |
| `dynamo_frontend_router_admission_*` | Registered and populated | Registered and populated | **Not registered** | **Not created** |
| `dynamo_component_kv_cache_events_applied` | Populated by KV events, or by predicted events when KV events are disabled, except TTL predictions with `--router-event-threads 1` | Populated by KV events, or by predicted events when KV events are disabled, except TTL predictions with `--router-event-threads 1` | **Not registered** | Populated by KV events, or by predicted events when KV events are disabled, except TTL predictions with `--router-event-threads 1` |
| `dynamo_frontend_worker_*` | Registered and populated | Registered and populated (`worker_type` = `prefill`/`decode`) | `worker_last_*` populated (`worker_type` = `decode`); `worker_active_*` absent | **Not created** |

**Key:**

- **Registered and populated** — the metric appears at `/metrics` with real values.
- **Registered, always zero** — the metric appears at `/metrics` but is never incremented (useful for dashboards that expect it to exist).
- **Not registered / Not created** — the metric does not appear at `/metrics` at all.

## Every Process

Emitted by every process that runs a Dynamo DistributedRuntime: on the HTTP port for the frontend (not with `--kserve-grpc-server`), and on `DYN_SYSTEM_PORT` for other processes.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_uptime_seconds` | gauge | `worker_id` | DistributedRuntime uptime. Updated before each Prometheus scrape on both the frontend and the system-status server. | |

With the QUIC response plane (`--response-plane quic` or `DYN_RESPONSE_PLANE=quic`), every process also records:

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_quic_response_active_connections` | gauge | — | Open QUIC response connections. | |
| `dynamo_quic_response_bundle_failures_total` | counter | `side` | QUIC connection bundle failures on the `frontend` or `worker` side. | |
| `dynamo_quic_response_server_mailbox_resets_total` | counter | `reason` | Responses reset because the frontend's response mailbox was `full` or `closed`. | |

## Planner Metrics

Exposed by `python -m dynamo.planner` at `/metrics` on its own port, set by `metric_reporting_prometheus_port`. It defaults to the `PLANNER_PROMETHEUS_PORT` environment variable, or to `0`, which disables the server, when that variable is unset. The Kubernetes operator sets the port to `9085`. Planner metrics carry no automatic labels. See [Planner Configuration](../components/planner-configuration.mdx).

### Replicas, Traffic, and SLA

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_planner_num_prefill_replicas` | gauge | — | Ready prefill replicas. | |
| `dynamo_planner_num_decode_replicas` | gauge | — | Ready decode replicas. | |
| `dynamo_planner_predicted_num_prefill_replicas` | gauge | — | Prefill replicas the planner decided on. | |
| `dynamo_planner_predicted_num_decode_replicas` | gauge | — | Decode replicas the planner decided on. | |
| `dynamo_planner_gpu_hours` | gauge | — | GPU hours used since the planner started. | |
| `dynamo_planner_observed_ttft_ms` | gauge | — | Observed TTFT in milliseconds. | |
| `dynamo_planner_observed_itl_ms` | gauge | — | Observed ITL in milliseconds. | |
| `dynamo_planner_observed_requests_per_second` | gauge | — | Observed request rate. | |
| `dynamo_planner_observed_request_duration_seconds` | gauge | — | Observed average request duration. | |
| `dynamo_planner_observed_input_sequence_tokens` | gauge | — | Observed average prompt length. | |
| `dynamo_planner_observed_output_sequence_tokens` | gauge | — | Observed average output length. | |
| `dynamo_planner_predicted_requests_per_second` | gauge | — | Predicted request rate for the next interval. | |
| `dynamo_planner_predicted_input_sequence_tokens` | gauge | — | Predicted prompt length for the next interval. | |
| `dynamo_planner_predicted_output_sequence_tokens` | gauge | — | Predicted output length for the next interval. | |
| `dynamo_planner_sla_target_ttft_ms` | gauge | — | Configured TTFT target in milliseconds. | |
| `dynamo_planner_sla_target_itl_ms` | gauge | — | Configured ITL target in milliseconds. | |
| `dynamo_planner_estimated_ttft_ms` | gauge | — | Highest TTFT estimated across engines, in milliseconds. | |
| `dynamo_planner_estimated_itl_ms` | gauge | — | Highest ITL estimated across engines, in milliseconds. | |
| `dynamo_planner_engine_prefill_capacity_requests_per_second` | gauge | — | Request rate one prefill engine can serve within the SLA. | |
| `dynamo_planner_engine_decode_capacity_requests_per_second` | gauge | — | Request rate one decode engine can serve within the SLA. | |
| `dynamo_planner_engine_queued_prefill_tokens` | gauge | `worker_id`, `dp_rank` | Queued prefill tokens per prefill engine, from forward-pass metrics. | |
| `dynamo_planner_engine_queued_decode_kv_tokens` | gauge | `worker_id`, `dp_rank` | Queued decode KV tokens per decode engine, from forward-pass metrics. | |
| `dynamo_planner_engine_inflight_decode_kv_tokens` | gauge | `worker_id`, `dp_rank` | Scheduled decode KV tokens per decode engine, from forward-pass metrics. | |

### Scaling Decisions

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_planner_load_scaling_decision` | gauge | `dynamo_planner_load_scaling_decision` | Reason for the latest load-based scaling decision. A Prometheus Enum: the label, named after the metric, holds the reason. One series per reason; the current one is 1. | |
| `dynamo_planner_throughput_scaling_decision` | gauge | `dynamo_planner_throughput_scaling_decision` | Reason for the latest throughput-based scaling decision. A Prometheus Enum: the label, named after the metric, holds the reason. One series per reason; the current one is 1. | |
| `dynamo_planner_signal_unavailable` | gauge | `component`, `signal` | 1 if a required planner input was unavailable at its latest observation. | Experimental |
| `dynamo_planner_degraded_signals_total` | counter | `component`, `signal` | Times a required input was unavailable for 3 observations in a row. | Experimental |
| `dynamo_planner_power_budget_total_watts` | gauge | — | Configured total GPU power budget, in watts. Set only with power awareness enabled; 0 otherwise. | |
| `dynamo_planner_power_projected_watts` | gauge | — | Projected GPU power draw at the current replica counts, in watts. Set only with power awareness enabled; 0 otherwise. | |
| `dynamo_planner_power_budget_utilization` | gauge | — | Projected power divided by the budget. Set only with power awareness enabled; 0 otherwise. Use `dynamo_planner_power_projected_watts / dynamo_planner_power_budget_total_watts`, which has the same value once both are set. | Deprecated |

### Plugin Pipeline

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_planner_plugin_evaluations_total` | counter | `plugin_id`, `stage`, `result` | Plugin evaluations, by outcome. | |
| `dynamo_planner_plugin_latency_seconds` | histogram | `plugin_id`, `stage` | Latency of successful plugin calls. In the propose, reconcile, and constrain stages the plugins run concurrently, and each call records the time until the slowest one in its stage finishes. | |
| `dynamo_planner_plugin_circuit_state` | gauge | `plugin_id` | Circuit breaker state per plugin: 0 closed, 0.5 half open, 1 open. | |
| `dynamo_planner_plugin_held_over_total` | counter | `plugin_id`, `stage` | Times a plugin's cached result was reused instead of calling it. Use `dynamo_planner_plugin_evaluations_total{result="held_over"}`, which has the same value. | Deprecated |
| `dynamo_planner_plugin_cache_age_seconds` | gauge | `plugin_id` | Age of each plugin's cached result. | |
| `dynamo_planner_plugin_override_active` | gauge | `plugin_id`, `stage`, `override_type` | 1 for each override type (`SET`, `AT_LEAST`, `AT_MOST`, or `REJECT`) the plugin returned in the stage at its latest evaluation, whether or not the override won the merge. | |
| `dynamo_planner_reconcile_clamped_total` | counter | `sub_component_type`, `source` | Times the reconcile stage clamped the replica count to a floor or ceiling. | |
| `dynamo_planner_constrain_capped_total` | counter | `sub_component_type`, `source` | Times the constrain stage clamped the final replica count to a floor or ceiling. | |
| `dynamo_planner_reject_short_circuited_total` | counter | `plugin_id` | Times a plugin's rejection stopped the pipeline and skipped execution. Use `dynamo_planner_plugin_evaluations_total`. The old value is `sum by (plugin_id) (dynamo_planner_plugin_evaluations_total{result="reject"})`. | Deprecated |
| `dynamo_planner_tick_skipped_total` | counter | `plugin_id` | Times a plugin was skipped because its execution interval had not elapsed. | |
| `dynamo_planner_tick_requires_unsatisfied_total` | counter | `plugin_id`, `missing_field` | Times a plugin was skipped because an input it requires was missing. | |
| `dynamo_planner_tick_lag_seconds` | gauge | `plugin_id` | Delay between a plugin's scheduled time and its evaluation. | |
| `dynamo_planner_tick_duration_seconds` | histogram | — | Time the plugin pipeline takes in each planner tick. | |
| `dynamo_planner_tick_timeout_total` | counter | — | Ticks aborted for exceeding the tick time limit. | |

### Live Retune

Recorded when live retune is configured (`retune` in the planner configuration), and by a planner without `retune` while it cleans up after a removed live retune: resetting the routers it changed and deleting its Leases. See [Planner Live Retune](../../kubernetes/auto-deployment/planner-live-retune.md).

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_planner_retune_decisions_total` | counter | `outcome` | Retune decisions, counted once the state write persisted them. | Experimental |
| `dynamo_planner_retune_failures_total` | counter | `reason` | Failed retune windows and supervisor failures, by reason. | Experimental |
| `dynamo_planner_retune_skipped_total` | counter | `reason` | Due windows the launch gate held back, counted once per window and reason. | Experimental |
| `dynamo_planner_retune_leader` | gauge | — | 1 while this planner holds the retune leadership Lease. | Experimental |
| `dynamo_planner_retune_ready` | gauge | — | 1 while every spec and health premise holds, so windows launch. | Experimental |
| `dynamo_planner_retune_initialized` | gauge | — | 1 once the leader finished initializing from the persisted state. | Experimental |
| `dynamo_planner_retune_rollback_pending` | gauge | — | 1 while routers are being reset to their startup parameters. | Experimental |
| `dynamo_planner_retune_controller_generation` | gauge | — | Generation of the installed retune controller; it grows on each rebuild. | Experimental |
| `dynamo_planner_retune_baseline_score` | gauge | — | Objective score of the current parameters on the latest launched window; NaN when not scored. | Experimental |
| `dynamo_planner_retune_best_score` | gauge | — | Best candidate score on the latest launched window; NaN when there is none. | Experimental |
| `dynamo_planner_retune_sweep_seconds` | gauge | — | Wall-clock duration of the latest sweep; NaN when it did not run. | Experimental |
| `dynamo_planner_retune_missing_candidates` | gauge | — | Candidates of the latest sweep without a score. | Experimental |
| `dynamo_planner_retune_router_converged` | gauge | — | 1 when every frontend router holds the target policy parameters. | Experimental |
| `dynamo_planner_retune_router_patch_errors_total` | counter | `code` | Failed `/router_policy` PATCH or reset calls, by HTTP status or error kind. | Experimental |
| `dynamo_planner_retune_trace_sink_missing` | gauge | `pod` | 1 for a Ready frontend pod without its own trace segment directory. | Experimental |
| `dynamo_planner_retune_trace_dropped_members_total` | counter | — | Runs of trace bytes without a valid gzip member that window cuts skipped, such as corrupt members or a segment's unfinished last write. A run counts once, however many members it held. | Experimental |

## Standalone KV Indexer Metrics

`python -m dynamo.indexer` serves its own `/metrics` on its HTTP port (`--port`), with no automatic labels. See [Standalone KV Indexer](../../developer-guide/knowledge-base/modular-components/router/standalone-indexer.md).

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_kvindexer_requests_total` | counter | `endpoint`, `method` | HTTP requests handled, excluding `/metrics`. | |
| `dynamo_kvindexer_request_duration_seconds` | histogram | `endpoint` | HTTP request latency. | |
| `dynamo_kvindexer_errors_total` | counter | `endpoint`, `status_class` | HTTP error responses, `4xx` or `5xx`. | |
| `dynamo_kvindexer_models` | gauge | — | Active model and routing-group indexers. | |
| `dynamo_kvindexer_workers` | gauge | — | Registered worker instances. | |
| `dynamo_kvindexer_listeners` | gauge | `status` | ZMQ listeners, by status. | |
| `dynamo_kvrouter_kv_cache_events_applied` | counter | `event_type`, `status` | KV cache events applied to the index. Device tier only. | |
| `dynamo_kvrouter_kv_cache_event_warnings` | counter | `warning_kind` | Suspicious KV cache events seen by the index. | |
| `dynamo_kvrouter_ckf_mutation_total` | counter | `outcome` | Block-level index mutation outcomes. Always 0: nothing increments it. No replacement; remove it from dashboards and alerts. | Deprecated |

## KV State Agent Metrics

`python -m dynamo.kv_state_agent` serves these on `DYN_SYSTEM_PORT`, with `dynamo_namespace`, `dynamo_component`, and `worker_id`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_kv_state_agent_slots` | gauge | `state` | Agent slots, by state: `total`, `active`, `detached`, or `failed`. | |
| `dynamo_component_kv_state_agent_capacity_rejected_total` | counter | — | Attach requests rejected because `--max-slots` was reached. | |

## Power Agent Metrics

Emitted by the power agent, a DaemonSet (`deploy/power-agent`) that enforces the per-GPU power limits set by the `dynamo.nvidia.com/gpu-power-limit` Pod annotation. It serves them at `/metrics` on `--prometheus-port` (`PROMETHEUS_PORT`); the default, `0`, turns the endpoint off, and the Helm chart sets `9100`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_power_agent_applied_limit_watts` | gauge | `gpu` | Power limit last applied to, or read back from, each GPU, in watts. `gpu` is the actuator's GPU index; with the DCGM actuator, a GPU that moves to a new index leaves its old series until the agent restarts. | |
| `dynamo_power_agent_multi_pod_gpu_total` | counter | `disposition` | Times a GPU had more than one Pod: `agree` when their caps match, `conflict` when they differ. | |
| `dynamo_power_agent_apply_failures_total` | counter | — | Times a cap could not be applied: the NVML or DCGM write failed, or the DCGM actuator could not verify the GPU's identity before writing. | |
| `dynamo_power_agent_safe_default_applied_total` | counter | — | Times the agent applied its safe-default cap, after a conflict between Pods or an annotation it could not parse. | |
| `dynamo_power_agent_cap_clamped_total` | counter | `direction` | Times a requested cap was clamped to the GPU's supported range: `min` (raised to the minimum) or `max` (lowered to the maximum). | |
| `dynamo_power_agent_k8s_list_failures_total` | counter | — | Times listing the node's Pods failed, so the agent skipped that reconcile cycle and kept the caps already applied. | |

## Endpoint Picker Metrics

Emitted by the endpoint picker (EPP) when routing through the Gateway API Inference Extension, at `/metrics` on `DYN_EPP_METRICS_PORT` (default 9090). Set the port to `0` to disable the endpoint.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_epp_cached_tokens` | histogram | `model` | Prompt tokens the model server served from its KV cache per request, read from `usage.prompt_tokens_details.cached_tokens` in the response body. This is the observed cache hit, as opposed to the overlap the KV router estimates at selection time. | |

Buckets match the default buckets of `dynamo_frontend_cached_tokens`, so the two can share a dashboard. The `model` label is the model the pool serves, resolved once at startup rather than from the request body. Recording needs a usage block in the response. On a streaming call the client must set `"stream_options": {"include_usage": true}`, unless the worker's Frontend sidecar sets `DYN_ENABLE_FORCE_INCLUDE_USAGE`; without a usage block, nothing is recorded.

These metrics show where the EPP spends a request's time and when its async runtime stalls. The settings that control them, and the EPP's diagnostic log lines, are in [EPP Runtime Environment](../components/gateway-api-routing.mdx#epp-runtime-environment).

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_epp_stage_duration_seconds` | histogram | `model`, [`stage`](metric-labels.mdx#epp-stage-values) | Time one routed request spends in each EPP stage. `receive_body`, `pick`, `rewrite`, and `first_response` together split the EPP's part of a request's time to first token. | Experimental |
| `dynamo_epp_event_loop_delay_seconds` | histogram | — | How late a 10 ms sleep on the EPP's async runtime wakes. It runs late only when no worker thread services the runtime's timers, so it shows stalls of the whole runtime, not of one task. | Experimental |
| `dynamo_epp_task_poll_seconds` | histogram | [`task`](metric-labels.mdx#epp-task-values) | Task polls that held an async worker thread at least `DYN_EPP_TASK_METRICS_MIN_US`. The work a poll runs through `block_in_place` does not count, nor does the rest of the poll once another thread has taken its worker over. Recorded only with `DYN_EPP_TASK_METRICS=1`. | Experimental |
| `dynamo_epp_task_schedule_delay_seconds` | histogram | [`task`](metric-labels.mdx#epp-task-values) | Waits of at least `DYN_EPP_TASK_METRICS_MIN_US` between a task being woken and being polled. Long waits of `task="h2_connection"` mean a gateway connection, and every stream on it, stood still. Recorded only with `DYN_EPP_TASK_METRICS=1`. | Experimental |

## Global Frontend Metrics

Emitted by the Global Frontend at `/metrics` on its serving port. It forwards each request, as received, to a pool frontend, so it sees requests and response bytes but not tokens: inter-token latency and token counts come from the pool frontends' `dynamo_frontend_*` metrics. Per-pool series are labeled by `pool`, the pool's `<cluster>/<namespace>/<DGD name>`, and `region`, the region Global Discovery reports for the pool, or the cluster for a pool given with `--pool`. No label comes from the request body. The Global Frontend routes to the `--pool` pools or the pools Global Discovery reports. With `--relay-view-port`, a pool is routable only while a Relay view reports its model ready; with `--pool-relay`, this applies only to the pools given a Relay.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_global_frontend_requests_total` | counter | `pool`, `region`, `local`, `status` | Requests forwarded to each pool. `status` is the pool's HTTP status code, counted when its response headers arrive; `no_response` when the pool could not be reached or sent no response; or `cancelled` when the client left before the pool's response headers arrived. | Experimental |
| `dynamo_global_frontend_rejected_requests_total` | counter | `error_type` | Requests the Global Frontend answered without forwarding them: `validation` (the body has no string `model`, or the route is invalid), `not_found` (no routable pool serves the model), or `unavailable` (no pool can take the request). | Experimental |
| `dynamo_global_frontend_time_to_first_byte_seconds` | histogram | `pool`, `region` | Time from the Global Frontend receiving a request to the first body chunk of its pool's successful (2xx) response. For a streamed request this approximates the time to first token as the client sees it, including the round trip to a pool in another region. | Experimental |
| `dynamo_global_frontend_request_duration_seconds` | histogram | `pool`, `region` | Time from the Global Frontend receiving a request to the end of its pool's successful (2xx) response, or to the client leaving. | Experimental |
| `dynamo_global_frontend_response_errors_total` | counter | `pool`, `region` | Successful responses whose body broke off before it ended, for example because the pool frontend restarted mid-stream. The client's response ends early. | Experimental |
| `dynamo_global_frontend_inflight_requests` | gauge | `pool`, `region` | Requests this Global Frontend has sent to each pool and not yet finished. Other replicas count their own. Read at scrape time for the routable pools, so a pool that stops being routable loses its series. | Experimental |

For a streaming request, a Dynamo pool frontend sends its headers once a worker has the request, so a request the client gives up on while a worker queues it counts under its status in `dynamo_global_frontend_requests_total`, not as `cancelled`. A non-streaming response's headers arrive with its complete body, so a non-streaming request the client abandons before it completes counts as `cancelled`. `local` is `true` for a pool in the Global Frontend's `--region`, `false` for one elsewhere, and `unknown` without `--region`; the share of requests kept in the region is the `local="true"` rate over the total. A body over `DYN_HTTP_BODY_LIMIT_MB` gets 413 before it is read and is not counted as a rejection.

If the pool frontend sends SSE keep-alives before the first token, the first keep-alive counts as the first byte: at the `DYN_HTTP_SSE_KEEP_ALIVE_INTERVAL_MS` interval when set, or after 15 seconds for a request whose output a reasoning parser may hold back. The time-to-first-byte and duration histograms use the `dynamo_frontend_time_to_first_token_seconds` and `dynamo_frontend_request_duration_seconds` default buckets.

With `--router-policy affinity`, the Global Frontend also records how it keeps sessions on their pools. These metrics count requests that carry an `x-dynamo-session-id` header, and they have no `pool` or `region` label. Each scrape adds to the counters what has happened since the previous scrape, so a counter series appears at the first scrape after its first event. The breaker opens after `--affinity-breaker-failures` consecutive `backend` errors or timeouts and stays open for `--affinity-breaker-cooldown-ms`. A request's time in affinity storage is part of its time to first byte and duration.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_global_frontend_affinity_lookups_total` | counter | `outcome` | Pin lookups for session requests: `hit` when the session has a live pin to one of the request's candidate pools, and `miss` when it has no pin, its pin has expired, or its pinned pool is not a candidate. A request whose pin write conflicts is looked up, and counted, a second time. A lookup that fails or runs out of time is not counted. | Experimental |
| `dynamo_global_frontend_affinity_store_errors_total` | counter | `kind` | Session requests routed without affinity, counted once each: `backend` (the store returned an error), `timeout` (the lookup, write, and retry ran past `--affinity-timeout-ms`), `conflict` (the pin write conflicted on both tries), or `breaker_open` (the store was skipped while the breaker was open). The request goes to the best pool chosen so far. | Experimental |
| `dynamo_global_frontend_affinity_store_writes_total` | counter | `reason` | Conditional pin writes sent to the affinity store, whether or not they apply: `create` (the session had no live pin), `move` (it was pinned to another pool), `refresh` (same pool, with half or less of `--session-idle-secs` left), or `repair` (replacing a record that could not be decoded). A retry after a conflict counts again. | Experimental |
| `dynamo_global_frontend_affinity_cache_total` | counter | `outcome` | Pin reads through the in-memory cache in front of DynamoDB: `hit` (a cached entry younger than `--affinity-cache-ttl-ms` answered) or `miss` (the read went to DynamoDB). `invalidation` counts cache entries dropped after a failed read or a conflicting or failed write, and new sessions left uncached because the cache already holds `--affinity-cache-entries` sessions. Only with `--affinity-store dynamodb` and a nonzero `--affinity-cache-ttl-ms`. | Experimental |
| `dynamo_global_frontend_overhead_affinity_store_ms` | histogram | — | Time each session request spent in affinity storage, in milliseconds: its lookups, writes, and retry, cut off at `--affinity-timeout-ms` (default 20). Not recorded while the breaker is open. Uses the buckets of `dynamo_router_overhead_shared_cache_query_ms`. | Experimental |

<ParamField path="dynamo_global_frontend_overhead_selection_ms" type="histogram">
  **Experimental.** Time to select a pool for a request, in milliseconds, with either selection path (`--use-selection-core` or not) and any `--router-policy`. Counts every request that reaches selection, also one that no pool can take. No labels: a process runs one path and one policy, so compare paths across runs. Unit and buckets match the router's `dynamo_router_overhead_*_ms` histograms.
</ParamField>

## Mocker Metrics

The mocker (`python -m dynamo.mocker`) simulates an inference engine. It emits imitation engine metrics with simulated values on `DYN_SYSTEM_PORT`, choosing the set by `--engine-type`. Every mocker metric carries `dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`, `worker_id`, and `model`, plus the engine labels named before each table. Each one means what it means in the engine it imitates. They are Experimental because their names follow the engines' metric names, which Dynamo does not control.

With `--engine-type vllm`, the default, the metrics also carry `model_name` and `engine`, which holds the DP rank:

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `vllm:num_requests_running` | gauge | — | Requests in model execution batches. | Experimental |
| `vllm:num_requests_waiting` | gauge | — | Requests waiting to be processed. | Experimental |
| `vllm:kv_cache_usage_perc` | gauge | — | KV cache usage, where 1 means 100 percent. | Experimental |
| `vllm:num_preemptions_total` | counter | — | Preemptions since the engine started. | Experimental |
| `vllm:time_to_first_token_seconds` | histogram | — | Time to first token in seconds. | Experimental |
| `vllm:inter_token_latency_seconds` | histogram | — | Inter-token latency in seconds. | Experimental |
| `vllm:request_time_per_output_token_seconds` | histogram | — | Time per output token of each request, in seconds. | Experimental |
| `vllm:e2e_request_latency_seconds` | histogram | — | End-to-end request latency in seconds. | Experimental |

With `--engine-type sglang`, the metrics also carry `model_name` and `engine_type` (`unified`, `prefill`, or `decode`). `tp_rank`, `pp_rank`, and `moe_ep_rank` are always `0`, `dp_rank` appears only when `--data-parallel-size` is above 1, and `is_streaming` is always `true`:

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `sglang:num_running_reqs` | gauge | `tp_rank`, `pp_rank`, `moe_ep_rank`, `dp_rank` | Running requests. | Experimental |
| `sglang:num_queue_reqs` | gauge | `tp_rank`, `pp_rank`, `moe_ep_rank`, `dp_rank` | Requests in the waiting queue. | Experimental |
| `sglang:cache_hit_rate` | gauge | `tp_rank`, `pp_rank`, `moe_ep_rank`, `dp_rank` | Prefix cache hit rate. | Experimental |
| `sglang:num_requests_total` | counter | `is_streaming` | Requests processed. | Experimental |
| `sglang:time_to_first_token_seconds` | histogram | `is_streaming` | Time to first token in seconds. Prefill workers do not emit it. | Experimental |
| `sglang:inter_token_latency_seconds` | histogram | — | Inter-token latency in seconds. | Experimental |
| `sglang:request_time_per_output_token_seconds` | histogram | `is_streaming` | Time per output token of each request, in seconds: the time from its first to its last token, divided by its output tokens after the first. Requests with one output token, such as a prefill worker's, are not recorded. | Experimental |
| `sglang:e2e_request_latency_seconds` | histogram | `is_streaming` | End-to-end request latency in seconds. | Experimental |

With `--engine-type trtllm`, the metrics also carry `model_name` and `engine_type`, which is `trtllm`:

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `trtllm_num_requests_running` | gauge | — | Active requests. With data parallelism, holds the most recently reported rank's value. | Experimental |
| `trtllm_num_requests_waiting` | gauge | — | Queued requests. With data parallelism, holds the most recently reported rank's value. | Experimental |
| `trtllm_kv_cache_utilization` | gauge | — | KV cache utilization. With data parallelism, holds the most recently reported rank's value. | Experimental |
| `trtllm_time_to_first_token_seconds` | histogram | — | Time to first token in seconds. | Experimental |
| `trtllm_time_per_output_token_seconds` | histogram | — | Time per output token in seconds. Requests with one output token, such as a prefill worker's, are not recorded. | Experimental |
| `trtllm_e2e_request_latency_seconds` | histogram | — | End-to-end request latency in seconds. | Experimental |

## Engine Metrics

Backend workers also expose their inference engine's own metrics at `/metrics` on `DYN_SYSTEM_PORT`. The engine defines these metrics, so their names, types, and meanings follow the engine version you run. Dynamo passes them through with the engine's names, types, and values, and adds the `dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`, `worker_id`, `model`, and `model_name` labels.

| Engine | Prefix | Enabled by | Engine reference | Dynamo guide |
|--------|--------|------------|------------------|--------------|
| vLLM | `vllm:`, plus `lmcache:` with LMCache | Always on; `python -m dynamo.vllm` ignores `--disable-log-stats` | [vLLM Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/) | [vLLM Observability](../../developer-guide/knowledge-base/modular-components/backends/vllm/observability.md) |
| SGLang | `sglang:` | `--enable-metrics` | [SGLang Production Metrics](https://docs.sglang.io/docs/references/production_metrics) | [SGLang Observability](../../developer-guide/knowledge-base/modular-components/backends/sglang/observability.md) |
| TensorRT-LLM | `trtllm_` | `--publish-metrics`; per-request metrics also need `--publish-per-request-metrics` | [TensorRT-LLM metrics collector](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/metrics/collector.py) | [TensorRT-LLM Observability](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/observability.md) |

Some `trtllm_*` metrics are computed by Dynamo, not by TensorRT-LLM; see [TensorRT-LLM Metrics Computed by Dynamo](#tensorrt-llm-metrics-computed-by-dynamo). To compare equivalent metrics across engines, see [Metrics Comparison](metrics-comparison.md).

## Related

- [Metric Labels](metric-labels.mdx) — the dimensions attached to these metrics.
- [Environment Variables](environment-variables.mdx) — variables that enable and configure metric emission.
- [Operator Metrics](operator-metrics.mdx) — Kubernetes operator metrics catalog.
- [Metrics Comparison](metrics-comparison.md) — engine pass-through metrics (`vllm:*`, `sglang:*`, `trtllm_*`).
- [OTLP Metrics Export](otlp-metrics-export.mdx) — pushing metrics to an OpenTelemetry collector.
