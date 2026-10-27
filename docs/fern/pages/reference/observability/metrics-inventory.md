---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Metrics Inventory
subtitle: Every metric Dynamo defines, grouped by the process that exposes it, with type, labels, and status.
---

This page lists every Prometheus metric that Dynamo's own code defines. Use it to look up a metric name you see in a scrape, dashboard, or alert. For detailed semantics, PromQL examples, and availability rules of the core frontend, worker, and router metrics, see the [Metrics Catalog](metrics-catalog.mdx). For label values, see [Metric Labels](metric-labels.mdx).

This page does not list:

- Inference engine metrics (`vllm:*`, `sglang:*`, `trtllm_*`) that the engine itself defines. See [Engine metrics](metrics-catalog.mdx#engine-metrics).
- Kubernetes operator metrics. See [Operator Metrics](operator-metrics.mdx).
- NIXL transfer metrics, which NIXL serves on its own port. See the [NIXL telemetry documentation](https://github.com/ai-dynamo/nixl/blob/main/docs/telemetry.md).

## How to Read This Page

- **Type** is the Prometheus metric type. A histogram also exposes `_bucket`, `_sum`, and `_count` series.
- **Labels** lists the labels specific to the metric. Labels shared by every metric in a section are listed in the section introduction.
- **Status** is **Current**, **Experimental** (a preview metric whose name or meaning may change), or **Deprecated** (still emitted; the description names the replacement).
- A metric with labels appears in a scrape only after its first series is recorded. Until then the metric is absent, not zero.
- Metrics defined in Python, such as the planner metrics, also expose a `_created` series for each counter and histogram.

## Frontend

Exposed by `python -m dynamo.frontend` at `/metrics` on the HTTP port (`--http-port`, default `8000`). The frontend serves its [router](#router) and [runtime](#every-process) metrics on the same port and does not use `DYN_SYSTEM_PORT`. Metrics in this section carry no automatic labels.

- `--metrics-prefix` (`DYN_METRICS_PREFIX`) replaces the `dynamo_frontend` prefix of the metrics in [Requests and Latency](#requests-and-latency), [Multimodal Requests](#multimodal-requests), [Model State](#model-state), and [Migration, Cancellation, and Rejection](#migration-cancellation-and-rejection). Other names do not change.
- With `--kserve-grpc-server`, the frontend serves metrics on `DYN_GRPC_METRICS_PORT` (default `8788`) instead. The model configuration, migration, router, and routing overhead metrics are not exposed in that mode.

### Requests and Latency

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_requests_started_total` | counter | `model`, `endpoint`, `request_type` | Requests accepted by a frontend handler. | Current |
| `dynamo_frontend_requests_total` | counter | `model`, `endpoint`, `request_type`, `status`, `error_type` | Requests that finished, by outcome. | Current |
| `dynamo_frontend_failures_total` | counter | `class`, `reason` | Requests that ended in an error, by error class and reason. Cancellations are not counted. | Current |
| `dynamo_frontend_active_requests` | gauge | `model` | Requests in progress, from handler entry until the response completes. | Current |
| `dynamo_frontend_inflight_requests` | gauge | `model` | Same value as `dynamo_frontend_active_requests`. Use `dynamo_frontend_active_requests` instead. | Deprecated |
| `dynamo_frontend_queued_requests` | gauge | `model` | Requests that have not produced an output token yet. Use `dynamo_frontend_stage_requests` summed over the `preprocess`, `route`, and `dispatch` stages instead. | Deprecated |
| `dynamo_frontend_awaiting_first_token_requests` | gauge | `model` | Admitted requests that have not produced an output token yet. | Experimental |
| `dynamo_frontend_inflight_input_tokens` | gauge | `model`, `state` | Prompt tokens of in-flight requests, split into `awaiting_first_token` and `generating`. | Experimental |
| `dynamo_frontend_input_tokens_total` | counter | `model` | Prompt tokens of admitted requests, counted at tokenization. | Experimental |
| `dynamo_frontend_disconnected_clients` | gauge | — | Client disconnects since the frontend started. The value only increases. | Current |
| `dynamo_frontend_request_duration_seconds` | histogram | `model` | End-to-end request duration. | Current |
| `dynamo_frontend_time_to_first_token_seconds` | histogram | `model` | Time to first token (TTFT). | Current |
| `dynamo_frontend_inter_token_latency_seconds` | histogram | `model` | Inter-token latency (ITL). | Current |
| `dynamo_frontend_input_sequence_tokens` | histogram | `model` | Prompt length per request, recorded at the first output token. | Current |
| `dynamo_frontend_output_sequence_tokens` | histogram | `model` | Output length per request. | Current |
| `dynamo_frontend_output_tokens_total` | counter | `model` | Output tokens generated. | Current |
| `dynamo_frontend_cached_tokens` | histogram | `model` | Prefix-cache hit tokens per request, as reported by the worker. | Current |
| `dynamo_frontend_embedding_latency_seconds` | histogram | `model` | Latency of successful `/v1/embeddings` requests. | Current |

### Multimodal Requests

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_images_per_request` | histogram | `model` | `image_url` content parts per request. Text-only requests record 0. | Current |
| `dynamo_frontend_videos_per_request` | histogram | `model` | `video_url` content parts per request. | Current |
| `dynamo_frontend_audio_per_request` | histogram | `model` | `audio_url` content parts per request. | Current |
| `dynamo_frontend_image_tokens_per_request` | histogram | `model` | Image placeholder tokens per request that contains images. | Current |

### Model State

The configuration gauges are set from the worker's model card when the frontend commits a worker group for the model.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_model_ready` | gauge | `model` | 1 when the frontend can route at least one request for the model, 0 when it cannot. Computed at scrape time. | Current |
| `dynamo_frontend_model_total_kv_blocks` | gauge | `model` | Total KV cache blocks of a worker serving the model. | Current |
| `dynamo_frontend_model_max_num_seqs` | gauge | `model` | Maximum concurrent sequences of a worker serving the model. | Current |
| `dynamo_frontend_model_max_num_batched_tokens` | gauge | `model` | Maximum batched tokens of a worker serving the model. | Current |
| `dynamo_frontend_model_context_length` | gauge | `model` | Maximum context length in tokens. | Current |
| `dynamo_frontend_model_kv_cache_block_size` | gauge | `model` | KV cache block size in tokens. | Current |
| `dynamo_frontend_model_migration_limit` | gauge | `model` | Migration limit recorded in the worker's model card. | Current |

### Migration, Cancellation, and Rejection

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_model_migration_total` | counter | `model`, `migration_type` | Request migrations caused by worker failures. | Current |
| `dynamo_frontend_model_migration_duration_seconds` | histogram | `model`, `migration_type`, `outcome` | Time from a worker failure until the request recovers, fails, or is cancelled. | Current |
| `dynamo_frontend_model_migration_max_seq_len_exceeded_total` | counter | `model` | Requests not migrated because they exceeded `--migration-max-seq-len`. | Current |
| `dynamo_frontend_model_cancellation_total` | counter | `model`, `endpoint`, `request_type` | Requests cancelled by a client disconnect or a closed stream. | Current |
| `dynamo_frontend_model_rejection_total` | counter | `model`, `endpoint` | Requests rejected because workers were overloaded (HTTP 529 by default). | Current |

### Pipeline and Tokenizer

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_stage_requests` | gauge | `stage`, `phase` | Requests currently in each pipeline stage: `preprocess`, `route`, or `dispatch`. | Current |
| `dynamo_frontend_stage_duration_seconds` | histogram | `stage` | Time spent in each pipeline stage: `preprocess`, `route`, or `transport_roundtrip`. | Current |
| `dynamo_frontend_tokenize_seconds` | histogram | — | Prompt tokenization time per request. | Current |
| `dynamo_frontend_template_seconds` | histogram | — | Chat template rendering time per request. | Current |
| `dynamo_frontend_tokenizer_latency_ms` | histogram | `operation` | Tokenizer latency in milliseconds, for `tokenize` and `detokenize`. | Current |
| `dynamo_frontend_detokenize_total_us` | counter | — | Total detokenization time in microseconds. | Current |
| `dynamo_frontend_detokenize_token_count` | counter | — | Tokens detokenized. | Current |
| `dynamo_frontend_tokenizer_cache_hits_total` | counter | — | Tokenizer prefix-cache hits. | Current |
| `dynamo_frontend_tokenizer_cache_misses_total` | counter | — | Tokenizer prefix-cache misses. | Current |
| `dynamo_frontend_tokenizer_cache_cached_tokens_total` | counter | `model` | Tokens served from the tokenizer prefix cache. No series when `DYN_TOKENIZER_CACHE=0`. | Current |
| `dynamo_frontend_tokenizer_cache_uncached_tokens_total` | counter | `model` | Tokens encoded after a tokenizer prefix-cache lookup. No series when `DYN_TOKENIZER_CACHE=0`. | Current |

### Worker Inventory and Load

The `dynamo_frontend_worker_*` gauges include only workers that are currently available for routing. Their `worker_id` label is the worker's discovery ID in decimal.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_router_workers` | gauge | `model`, `target_namespace`, `target_component`, `target_endpoint`, `worker_type`, `state` | Discovered workers per model and serving endpoint, by router state. See [Router Worker Inventory](metrics-catalog.mdx#router-worker-inventory). | Current |
| `dynamo_frontend_router_worker_state` | gauge | `model`, `target_namespace`, `target_component`, `target_endpoint`, `worker_type`, `router_worker_id`, `state`, `reason` | One sample with value 1 per discovered worker, labeled with its current state and reason. | Current |
| `dynamo_frontend_worker_active_requests` | gauge | `worker_id`, `dp_rank`, `worker_type`, `request_phase` | Requests booked on each worker, in the `prefill` or `decode` phase. KV routing only. | Current |
| `dynamo_frontend_worker_active_prefill_tokens` | gauge | `worker_id`, `dp_rank`, `worker_type` | Prefill tokens queued on each worker. KV routing only. | Current |
| `dynamo_frontend_worker_active_decode_blocks` | gauge | `worker_id`, `dp_rank`, `worker_type` | KV cache blocks in use for decode on each worker. KV routing only. | Current |
| `dynamo_frontend_worker_last_time_to_first_token_seconds` | gauge | `worker_id`, `dp_rank`, `worker_type` | Most recent TTFT observed for each worker. | Current |
| `dynamo_frontend_worker_last_input_sequence_tokens` | gauge | `worker_id`, `dp_rank`, `worker_type` | Most recent prompt length observed for each worker. | Current |
| `dynamo_frontend_worker_last_inter_token_latency_seconds` | gauge | `worker_id`, `dp_rank`, `worker_type` | Most recent ITL observed for each worker. | Current |

### Router Queue and Admission

These metrics carry `model`, `worker_type`, and `policy_class`. Their series appear when a KV router starts. The queue gauges stay at 0 unless queueing is enabled with `--router-queue-threshold` or a limit in `--router-policy-config`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_frontend_router_queue_pending_requests` | gauge | — | Requests waiting in the router scheduler queue. | Current |
| `dynamo_frontend_router_queue_pending_isl_tokens` | gauge | — | Prompt tokens of requests waiting in the queue. | Current |
| `dynamo_frontend_router_queue_pending_cached_tokens` | gauge | — | Estimated cached tokens of requests waiting in the queue. | Current |
| `dynamo_frontend_router_queue_backpressure_total` | counter | `reason` | Requests rejected because a queue limit was reached. | Current |
| `dynamo_frontend_router_admission_received_total` | counter | — | Attempts received by router admission, including retries. | Current |
| `dynamo_frontend_router_admission_rejected_total` | counter | `reason` | Attempts rejected because their deadline passed (`due_time_passed`). The `predicted_miss` reason is reserved and always 0. | Current |

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

## Router

Emitted by every process that routes requests: the frontend, which serves them on its HTTP port, and the standalone router (`python -m dynamo.router`), which serves them on `DYN_SYSTEM_PORT`. The router adds `dynamo_namespace` and `dynamo_component` of the workers it routes to, and `worker_id`, which is the router's own ID in hex. The request metrics also carry `router_id`, the same ID in decimal.

In round-robin and other non-KV modes, the request, latency, and sequence-length metrics are still recorded. The KV-specific metrics stay at zero or have no series.

### Routing Requests

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_router_requests_started_total` | counter | `router_id` | Requests admitted by the router. | Current |
| `dynamo_component_router_requests_total` | counter | `router_id` | Requests that completed through the router. | Current |
| `dynamo_component_router_time_to_first_token_seconds` | histogram | `router_id` | TTFT observed at the router. | Current |
| `dynamo_component_router_inter_token_latency_seconds` | histogram | `router_id` | Average ITL per request observed at the router. In KV mode, recorded only with `--router-track-output-blocks`. | Current |
| `dynamo_component_router_input_sequence_tokens` | histogram | `router_id`, `model`, `phase` | Prompt length per routing attempt. | Current |
| `dynamo_component_router_output_sequence_tokens` | histogram | `router_id` | Output length per request. | Current |
| `dynamo_component_router_kv_hit_rate` | histogram | `router_id` | Predicted KV cache hit rate (0.0–1.0) at routing time. KV routing only. | Current |
| `dynamo_component_router_kv_transfer_estimated_latency_seconds` | histogram | `router_id` | Upper-bound estimate of the KV cache transfer time from prefill to decode. Disaggregated serving only. | Current |
| `dynamo_component_router_non_max_overlap_selections_total` | counter | `router_id`, `worker_type` | Prefill selections that chose a worker with less KV cache overlap than another eligible worker. | Current |
| `dynamo_component_router_overlap_blocks_lost` | histogram | `router_id`, `worker_type` | Overlap blocks given up in each such selection. | Current |
| `dynamo_component_router_worker_registered` | gauge | `router_worker_id`, `dp_rank`, `worker_type` | One sample with value 1 per worker and DP rank the router has registered. KV routing only. | Current |

### KV Cache Reuse

These counters carry `router_id`, `model`, and `phase`. See [KV-cache reuse telemetry](metrics-catalog.mdx#kv-cache-reuse-telemetry) for how to compare them.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_router_kv_best_eligible_cached_prefix_tokens_total` | counter | — | Cached prefix tokens on the best eligible worker at selection time. | Current |
| `dynamo_component_router_kv_selected_cached_prefix_tokens_total` | counter | — | Cached prefix tokens on the selected worker. | Current |
| `dynamo_component_router_kv_worker_reused_tokens_total` | counter | — | Prompt tokens the worker reports reusing from its KV cache. | Current |

### Routing Overhead

Frontend only, in milliseconds, with a `router_id` label. Registered on every frontend and recorded only for KV routing decisions.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_router_overhead_block_hashing_ms` | histogram | — | Time computing block hashes. | Current |
| `dynamo_router_overhead_seq_hashing_ms` | histogram | — | Time computing sequence hashes. | Current |
| `dynamo_router_overhead_indexer_find_matches_ms` | histogram | — | Time finding cache matches in the KV index. | Current |
| `dynamo_router_overhead_scheduling_ms` | histogram | — | Time selecting a worker. | Current |
| `dynamo_router_overhead_shared_cache_query_ms` | histogram | — | Time querying the shared KV cache. | Current |
| `dynamo_router_overhead_total_ms` | histogram | — | Total routing overhead per request. | Current |

### Shared KV Cache

Recorded only with `--shared-cache-type hicache`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_router_shared_cache_hit_rate` | histogram | `router_id` | Fraction of request blocks found in the shared KV cache. | Current |
| `dynamo_component_router_shared_cache_beyond_blocks` | histogram | `router_id` | Shared-cache blocks beyond the selected worker's device overlap. | Current |
| `dynamo_router_shared_cache_errors_total` | counter | `router_id` | Failed shared-cache queries and subscriber errors. Frontend only. | Current |

### KV Index and Event Ingestion

Present when the router keeps a local KV index, which is the default for KV routing.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_kv_cache_events_applied` | counter | `event_type`, `status` | KV cache events applied to the router's index, by event type and result. | Current |
| `dynamo_component_kv_cache_event_warnings` | counter | `warning_kind` | Suspicious KV cache events seen by the index, such as duplicate stores. | Current |
| `dynamo_component_ckf_mutation_total` | counter | `outcome` | Block-level index mutation outcomes. Always 0: nothing increments it yet. | Current |
| `dynamo_component_router_kv_event_source_mismatch_workers` | gauge | `model`, `worker_type`, `target_namespace`, `target_component`, `target_endpoint` | Serving workers whose KV event source is missing, ambiguous, or disabled. | Current |
| `dynamo_component_router_kv_zmq_ingress_sources` | gauge | `state` | KV event sources on the ZMQ event plane, by lifecycle state. | Current |
| `dynamo_component_router_kv_zmq_ingress_batches_total` | counter | — | KV event batches received over ZMQ. | Current |
| `dynamo_component_router_kv_zmq_ingress_lifecycle_total` | counter | `action` | Lifecycle transitions and errors of ZMQ KV event sources. | Current |
| `dynamo_component_router_remote_indexer_query_failures_total` | counter | `router_id` | Failed overlap queries to a remote indexer. Only with `--use-remote-indexer`. | Current |
| `dynamo_component_router_remote_indexer_write_failures_total` | counter | `router_id` | Failed routing-decision writes to a remote indexer. Only with `--use-remote-indexer`. | Current |

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

## Backend Workers

Exposed by each worker at `/metrics` on `DYN_SYSTEM_PORT`, which is off by default; the Kubernetes operator sets it to `9090`. Unless a section says otherwise, the runtime adds `dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`, and `worker_id`; see [Runtime-injected labels](metric-labels.mdx#runtime-injected-labels). Metrics created on a component rather than an endpoint have no `dynamo_endpoint`. Workers also expose their inference engine's own metrics; see [Engine metrics](metrics-catalog.mdx#engine-metrics).

### Request Handling

Recorded for every endpoint a process serves. Backends also add a `model` label.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_requests_total` | counter | — | Requests handled. | Current |
| `dynamo_component_inflight_requests` | gauge | — | Requests being handled. | Current |
| `dynamo_component_request_duration_seconds` | histogram | — | Time to handle a request, until its response stream ends. | Current |
| `dynamo_component_request_bytes_total` | counter | — | Request bytes received. | Current |
| `dynamo_component_response_bytes_total` | counter | — | Response bytes sent. | Current |
| `dynamo_component_errors_total` | counter | `error_type` | Errors while handling requests, by [error type](metric-labels.mdx#component-error-types). | Current |
| `dynamo_component_cancellation_total` | counter | — | Requests cancelled while being handled, counted once per request. | Current |

### Work Queue and Worker Pool

Limits of the TCP request plane, the default. Registered once per process with no labels. With the NATS request plane, all except `dynamo_work_handler_time_to_first_response_seconds` stay at 0.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_work_handler_queue_depth` | gauge | — | Requests waiting in the work queue. | Current |
| `dynamo_work_handler_queue_capacity` | gauge | — | Work queue capacity, set by `DYN_TCP_WORK_QUEUE_SIZE`. | Current |
| `dynamo_work_handler_enqueue_rejected_total` | counter | — | Requests rejected because the work queue was full or closed. | Current |
| `dynamo_work_handler_pool_active_tasks` | gauge | — | Requests holding a worker-pool slot. | Current |
| `dynamo_work_handler_pool_capacity` | gauge | — | Worker-pool size, set by `DYN_TCP_WORKER_POOL_SIZE`. | Current |
| `dynamo_work_handler_permit_wait_seconds` | histogram | — | Time a queued request waited for a worker-pool slot. | Current |
| `dynamo_work_handler_network_transit_seconds` | histogram | — | Network time from the frontend to the worker, computed from the clocks on both hosts. | Current |
| `dynamo_work_handler_time_to_first_response_seconds` | histogram | — | Time from receiving a request until its response stream opens. This is not the time to the first token. | Current |

### Backend Admission

Created on the worker's component, so these carry `dynamo_namespace`, `dynamo_component`, and `worker_id` but no `dynamo_endpoint`. The `dp_rank` label is the rank number, or `unspecified` for requests without a rank. See [Backend Admission Control](../../developer-guide/knowledge-base/concepts/fault-tolerance/admission-control-architecture.md#backend).

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_backend_admission_engine_request_count` | gauge | `dp_rank` | Requests admitted to the engine that have not finished. | Current |
| `dynamo_backend_admission_engine_request_waiting_count` | gauge | `dp_rank` | Admitted requests still waiting for their first token or completion. | Current |
| `dynamo_backend_admission_request_queue_count` | gauge | `dp_rank` | Requests waiting in the admission queue. | Current |
| `dynamo_backend_admission_request_receive_total` | counter | `dp_rank` | Requests received by the admission gate. | Current |
| `dynamo_backend_admission_request_admit_total` | counter | `dp_rank`, `source` | Requests passed to the engine, either `direct` or from the `queue`. | Current |
| `dynamo_backend_admission_rejection_total` | counter | `dp_rank` | Requests rejected because they could not be admitted and the queue was full. | Current |
| `dynamo_backend_admission_cancellation_total` | counter | `dp_rank` | Requests cancelled before admission. | Current |

### Event Publishing

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

These carry `model` and `dynamo_component`. The embedding cache metrics require `--multimodal-embedding-cache-capacity-gb` greater than 0. The image cache metrics are registered on vLLM multimodal workers, and the shared image cache records samples only with `DYN_MM_SHARED_IMAGE_CACHE_ENABLED=1`.

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

TensorRT-LLM workers started with `--publish-metrics` add these to the engine's own `trtllm_*` metrics. They carry `model_name`, `disaggregation_mode`, and `engine_type`, plus the labels Dynamo adds to [engine metrics](metrics-catalog.mdx#engine-metrics).

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

## Every Process

Emitted by every Dynamo process: on the HTTP port for the frontend, and on `DYN_SYSTEM_PORT` for other processes.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_uptime_seconds` | gauge | — | Time since the process's runtime started. Carries only `worker_id`. | Current |
| `dynamo_quic_response_active_connections` | gauge | — | Open QUIC response connections. Only with `DYN_RESPONSE_PLANE=quic`. | Current |
| `dynamo_quic_response_bundle_failures_total` | counter | `side` | QUIC connection bundle failures on the `frontend` or `worker` side. Only with `DYN_RESPONSE_PLANE=quic`. | Current |
| `dynamo_quic_response_server_mailbox_resets_total` | counter | `reason` | Responses reset because the frontend's response mailbox was `full` or `closed`. Only with `DYN_RESPONSE_PLANE=quic`. | Current |

## Planner

Exposed by `python -m dynamo.planner` at `/metrics` on its own port, set by `metric_reporting_prometheus_port` (default `PLANNER_PROMETHEUS_PORT`, else `0`, which disables the server). The Kubernetes operator sets the port to `9085`. Planner metrics carry no automatic labels. See [Planner Configuration](../components/planner-configuration.mdx).

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
| `dynamo_planner_load_scaling_decision` | gauge | `dynamo_planner_load_scaling_decision` | Reason for the latest load-based scaling decision. One series per reason; the current one is 1. | Current |
| `dynamo_planner_throughput_scaling_decision` | gauge | `dynamo_planner_throughput_scaling_decision` | Reason for the latest throughput-based scaling decision. One series per reason; the current one is 1. | Current |
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

## Other Services

### Standalone KV Indexer

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

### KV State Agent

`python -m dynamo.kv_state_agent` serves these on `DYN_SYSTEM_PORT`, with `dynamo_namespace`, `dynamo_component`, and `worker_id`.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_component_kv_state_agent_slots` | gauge | `state` | Agent slots, by state: `total`, `active`, `detached`, or `failed`. | Current |
| `dynamo_component_kv_state_agent_capacity_rejected_total` | counter | — | Attach requests rejected because `--max-slots` was reached. | Current |

### Endpoint Picker

The endpoint picker (EPP) for Gateway API Inference Extension routing serves `/metrics` on `DYN_EPP_METRICS_PORT` (default `9090`; `0` disables it).

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `dynamo_epp_cached_tokens` | histogram | `model` | Prompt tokens the model server served from its KV cache, per request. Appears after the first response that reports cached tokens. | Current |

## Mocker

The mocker (`python -m dynamo.mocker`) simulates an inference engine. It emits imitation engine metrics with simulated values on `DYN_SYSTEM_PORT`, choosing the set by `--engine-type`. They carry the engine's usual labels plus `dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`, `worker_id`, and `model`.

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

## KVBM

> [!WARNING]
> KVBM is deprecated since v1.5.0, and `main` no longer builds it. These metrics apply only to releases that ship KVBM. See [Deprecations](../general/releases/deprecations.mdx).

KVBM serves these from its own server at `/metrics` on `DYN_KVBM_METRICS_PORT` (default `6880`) when `DYN_KVBM_METRICS=1`, with no labels. In the block names, `d2d` means GPU to disk for offload and disk to GPU for onboard.

| Metric | Type | Labels | Description | Status |
|---|---|---|---|---|
| `kvbm_offload_blocks_d2h` | counter | — | Blocks offloaded from GPU to host memory. | Deprecated |
| `kvbm_offload_blocks_h2d` | counter | — | Blocks offloaded from host memory to disk. | Deprecated |
| `kvbm_offload_blocks_d2d` | counter | — | Blocks offloaded from GPU directly to disk. | Deprecated |
| `kvbm_offload_blocks_d2o` | counter | — | Blocks offloaded from GPU to object storage. Always 0. | Deprecated |
| `kvbm_onboard_blocks_h2d` | counter | — | Blocks onboarded from host memory to GPU. | Deprecated |
| `kvbm_onboard_blocks_d2d` | counter | — | Blocks onboarded from disk to GPU. | Deprecated |
| `kvbm_onboard_blocks_o2d` | counter | — | Blocks onboarded from object storage to GPU. Always 0. | Deprecated |
| `kvbm_matched_tokens` | counter | — | Host and disk tokens matched for onboarding. | Deprecated |
| `kvbm_host_cache_hit_rate` | gauge | — | Host cache hit rate over recent requests, from 0.0 to 1.0. | Deprecated |
| `kvbm_disk_cache_hit_rate` | gauge | — | Disk cache hit rate over recent requests, from 0.0 to 1.0. | Deprecated |
| `kvbm_object_cache_hit_rate` | gauge | — | Object storage cache hit rate. Always 0. | Deprecated |
| `kvbm_object_read_failures` | counter | — | Failed object storage reads. Always 0. | Deprecated |
| `kvbm_object_write_failures` | counter | — | Failed object storage writes. Always 0. | Deprecated |

## Related

- [Metrics Catalog](metrics-catalog.mdx): semantics, PromQL examples, and availability rules for the core metrics.
- [Metric Labels](metric-labels.mdx): label meanings and values.
- [Environment Variables](environment-variables.mdx#system-and-metrics): variables that enable metrics endpoints.
- [Metrics Comparison](metrics-comparison.md): equivalent engine metrics across vLLM, SGLang, and TensorRT-LLM.
- [OTLP Metrics Export](otlp-metrics-export.mdx): pushing metrics to an OpenTelemetry collector.
- [Operator Metrics](operator-metrics.mdx): Kubernetes operator metrics.
