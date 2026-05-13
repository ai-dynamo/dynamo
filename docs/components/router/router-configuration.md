---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Configuration and Tuning
subtitle: Router flags, event transport, load tracking, and tuning guidance
---

This page collects the main router flags for frontend-embedded and standalone deployments. For the routing cost model and worker-selection behavior, see [Routing Concepts](router-concepts.md).

## Routing Behavior

- `--router-kv-overlap-score-weight`: Controls the importance of prefix cache overlaps in prefill cost calculations. Higher values improve Time To First Token (TTFT) at the cost of Inter-Token Latency (ITL). When set to 0, the router ignores prefix caches and uses pure load balancing. Defaults to 1.
- `--router-temperature`: Controls worker selection randomness through softmax sampling of router cost logits. A value of 0 (default) ensures deterministic selection of the lowest-cost worker, while higher values introduce more randomness.
- `--router-track-prefill-tokens`: Enables prompt-side load accounting in the worker cost model. This should stay enabled if you want queue thresholds, `active_prefill_tokens`, and AIC prefill load decay to reflect prompt work.
- `--router-prefill-load-model`: Selects the router's prompt-side load model. `none` keeps the existing static prompt load accounting. `aic` predicts one expected prefill duration per admitted request and lazily decays only the oldest active prefill request on each worker.
- `--router-queue-threshold`: Queue threshold fraction for prefill token capacity (default: 4.0). The router holds incoming requests in a priority queue while all workers exceed this fraction of `max_num_batched_tokens`, releasing them when capacity frees up. This defers dispatch rather than rejecting work, so routing decisions use the freshest load metrics at the moment a request is actually sent to a worker. It also enables priority scheduling via `priority` hints in `nvext.agent_hints`. Must be greater than 0. Set to `None` to disable queueing.
- `--router-queue-policy`: Scheduling policy for the router queue (default: `fcfs`).

`fcfs` orders by adjusted arrival time (`priority_jump - arrival_offset`) and optimizes tail TTFT.
`lcfs` orders by adjusted reverse arrival time (`priority_jump + arrival_offset`) and mainly serves controlled comparison experiments.
`wspt` orders by `(1 + priority_jump) / isl_tokens` and optimizes average TTFT.

For `--router-mode device-aware-weighted`, set `DYN_ENCODER_CUDA_TO_CPU_RATIO` to the approximate throughput ratio of one non-CPU worker relative to one CPU worker. The default is `8`.

## KV Event Transport and Persistence

- `--no-router-kv-events`: Disables KV event tracking. By default, the router uses KV events to monitor block creation and deletion from workers. When disabled, the router predicts cache state from routing decisions with TTL-based expiration and pruning.
- `--router-durable-kv-events`: **Deprecated.** Enables JetStream mode for KV event transport. The event-plane subscriber in local indexer mode is now the recommended path.
- `--router-reset-states`: Only applies in JetStream mode (`--router-durable-kv-events`). Resets the router state on startup by clearing both the JetStream event stream and NATS object store, starting from a fresh state.
- `--router-snapshot-threshold`: Only applies in JetStream mode (`--router-durable-kv-events`). Sets the number of messages in JetStream before triggering a snapshot.

## Block Tracking

- `--no-router-track-active-blocks`: Disables tracking of active blocks used for ongoing generation or decode phases. Disable this when routing to workers that only perform prefill.
- `--router-track-output-blocks`: **Experimental.** Enables tracking of output blocks during generation. When enabled, the router adds placeholder blocks as tokens are generated and applies fractional decay based on progress toward the expected output sequence length (`agent_hints.osl` in `nvext`).
- `--no-router-assume-kv-reuse`: When tracking active blocks, disables the assumption of KV cache reuse. This is useful in disaggregated setups where transferred blocks are not actually deduplicated on the decode side.
- `--no-router-track-prefill-tokens`: Disables prompt-side prefill token accounting in the router's active load model. Use this for decode-only routing paths where prompt processing already happened elsewhere.
- `--router-replica-sync`: Disabled by default. Enables NATS-based synchronization of local routing decisions between router replicas.

## KV Indexer / Approx KV Indexer

- `--router-ttl-secs`: Time-to-live in seconds for blocks in the router's local cache predictions. Defaults to 120.0 seconds when `--no-router-kv-events` is used.
- `--router-max-tree-size`: Maximum tree size before pruning is triggered. Defaults to 1048576 (2^20 blocks) when `--no-router-kv-events` is used.
- `--router-prune-target-ratio`: Target size ratio to prune down to when `--router-max-tree-size` is exceeded. Defaults to 0.8 when `--no-router-kv-events` is used.
- `--router-event-threads`: Number of event processing threads for the KV indexer (default: 4). With KV events enabled, values greater than 1 use the concurrent radix tree; approximate mode always uses a single-threaded indexer.

To implement KV event publishing for custom inference engines, see [KV Event Publishing for Custom Engines](../../integrations/kv-events-custom-engines.md).
For details on per-request agent hints (`priority`, `osl`, `speculative_prefill`), see [NVIDIA Request Extensions (`nvext`)](../frontend/nvext.md#agent-hints).

### Session Control and Sticky Routing

When a request carries `nvext.session_control`, the KV router activates two additional components:

- **AgentController**: Sends session lifecycle RPCs (`open_session`, `close_session`) to the worker's `session_control` endpoint. The event-plane client is lazily initialized on the first session request.
- **StickySessionRouter**: Maintains an in-memory `session_id -> worker_id` affinity map with sliding-window TTL. Subsequent requests with the same `session_id` are routed to the pinned worker, bypassing KV overlap scoring.

These activate automatically with `--router-mode kv` -- no additional flags are needed. Requests without `session_control` are unaffected and follow the standard KV-aware routing path. Session control currently requires the SGLang backend with `--enable-streaming-session`. See [SGLang for Agentic Workloads -- Session Control](../../backends/sglang/agents.md#session-control-for-subagent-kv-isolation-experimental) for details.

## Tuning Guidelines

`--router-kv-overlap-score-weight` is the primary knob for balancing prefill efficiency against decode load. Prefill-heavy workloads benefit from a higher weight, which steers requests toward workers with better cache overlap and reduces TTFT. Decode-heavy workloads benefit from a lower weight, which distributes decode load more evenly and reduces ITL. The default of 1.0 is a reasonable starting point. This weight can also be overridden per request via `nvext.agent_hints.kv_overlap_score_weight`.

Use `--no-router-kv-events` when you are not confident that your backend engine emits KV events correctly. In this mode the router falls back to approximate routing, predicting cache state from its own routing decisions with TTL-based expiration and pruning.

Use `--no-router-assume-kv-reuse` in disaggregated setups where the decode worker does not reuse transferred KV cache blocks. Without this flag, the router undercounts decode blocks when duplicates exist, leading to inaccurate load estimates.

Use `--no-router-track-prefill-tokens` when a router is serving decode-only traffic and prompt processing has already completed elsewhere. This keeps decode routing decisions focused on decode-side load instead of briefly charging prompt tokens to the decode worker after handoff.

Use `--router-track-output-blocks` when your workload is output-heavy and you want the router to account for output-side KV cache growth in load balancing. If you also pass `nvext.agent_hints.osl` per request, the router applies fractional decay to output blocks so that requests nearing completion contribute less future load.

`--router-queue-threshold` controls when incoming requests are held in a priority queue. The router waits while all workers exceed the configured fraction of `max_num_batched_tokens`, then releases work as capacity frees up. Set it to `None` to disable queueing entirely.

Use `--router-prefill-load-model aic` when you want prompt-side load tracking to decay the oldest active prefill request using an AIC-predicted duration instead of keeping prompt load static until first token. This requires `--router-track-prefill-tokens` and the shared `--aic-*` config.

Use `--router-queue-policy wspt` when your workload has a mix of short and long requests and you want to minimize average TTFT. Use the default `fcfs` when you want to minimize tail TTFT.

## Prometheus Metrics

The router exposes Prometheus metrics on the frontend's HTTP port (default 8000) at `/metrics`:

- **Router request metrics** (`dynamo_component_router_*`): Registered via the component's metrics hierarchy and exposed on the frontend via the `drt_metrics` bridge. In KV mode they are populated per request; in non-KV modes they are registered with zero values. The standalone router also registers these metrics, available on `DYN_SYSTEM_PORT` when set.
- **Routing overhead metrics** (`dynamo_router_overhead_*`) and **per-worker gauges** (`dynamo_frontend_worker_*`): Registered on the frontend's own Prometheus registry. These are frontend-only and not available on the standalone router.

For the full list of router metrics, see the [Metrics reference](../../observability/metrics.md#router-metrics).
