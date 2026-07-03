---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Configuration and Tuning
subtitle: Router flags, event transport, load tracking, and tuning guidance
---

This page collects the main router flags for frontend-embedded and standalone deployments. For the routing cost model and worker-selection behavior, see [Routing Concepts](router-concepts.md).

## Routing Behavior

- `--router-kv-overlap-score-credit`: Device-local prefix-overlap credit multiplier in the prefill cost calculation, from 0.0 to 1.0. Higher values improve Time To First Token (TTFT) at the cost of Inter-Token Latency (ITL). When set to 0, the router ignores prefix caches and skips creating a local indexer. Defaults to 1.
- `--router-kv-overlap-score-credit-decay`: Decays device-local overlap credit for workers whose active prefill load exceeds the least-loaded eligible worker. `0` disables decay. Defaults to 0.
- `--router-prefill-load-scale`: Scale applied to adjusted prompt-side prefill load after device, lower-tier, and shared-cache credits are subtracted. Defaults to 1.
- `--router-host-cache-hit-weight`: Credit multiplier for host-pinned (CPU offload) prefix overlap, from 0.0 to 1.0. Symmetric to `--router-kv-overlap-score-credit` but applied to the host-pinned tier when a backend exposes CPU offload via a KV connector. Defaults to 0.75.
- `--router-disk-cache-hit-weight`: Credit multiplier for disk/lower-tier (e.g. NVMe-backed) prefix overlap, from 0.0 to 1.0. Defaults to 0.25.
- `--load-aware`: Preset for load-aware KV routing without cache-reuse signals. On the frontend, it implies `--router-mode kv`. It sets `overlap_score_credit=0`, disables KV events, durable KV events, and KV reuse assumptions, enables active-block and prefill-token load tracking, disables remote/shared cache indexers, and preserves `--router-prefill-load-scale`, `--router-host-cache-hit-weight`, and `--router-disk-cache-hit-weight`.
- `--router-temperature`: Controls worker selection randomness through softmax sampling of normalized router cost logits. A value of 0 (default) ensures deterministic selection of the lowest-cost worker, while higher values introduce more randomness.
- `--router-track-prefill-tokens`: Enables prompt-side load accounting in the worker cost model. This should stay enabled if you want queue thresholds, `active_prefill_tokens`, and AIC prefill load decay to reflect prompt work.
- `--router-prefill-load-model`: Selects the router's prompt-side load model. `none` keeps the existing static prompt load accounting. `aic` predicts one expected prefill duration per admitted request and lazily decays only the oldest active prefill request on each worker.
- `--router-queue-threshold`: Queue threshold fraction for prefill token capacity (default: 16.0). The router holds incoming requests in a priority queue while all eligible workers exceed `threshold * max_num_batched_tokens`, releasing them when capacity frees up. This defers dispatch rather than rejecting work, so routing decisions use the freshest load metrics at the moment a request is actually sent to a worker. `nvext.agent_hints.strict_priority` selects an absolute pending-queue tier, while `nvext.agent_hints.priority` adjusts ordering within the configured policy. Must be greater than or equal to 0; use `0.0` for maximum queueing sensitivity. Set to `None` to disable queueing. See the SGLang note under [Tuning Guidelines](#tuning-guidelines) for caveats around how `max_num_batched_tokens` is populated on that backend, and see [Priority Scheduling](priority-scheduling.md) for how router priority differs from backend engine priority.
- `--router-queue-policy`: Scheduling policy for the router queue (default: `fcfs`).
- `--router-policy-config`: Startup-only policy-family and cache-bucket YAML path. When omitted, `--router-queue-threshold` and `--router-queue-policy` retain the single default queue. The equivalent environment variable is `DYN_ROUTER_POLICY_CONFIG`.

For how queue backpressure differs from candidate filtering and busy-threshold overload handling, see [Router Filtering](router-filtering.md).

`fcfs` orders by adjusted arrival time (`priority_jump - arrival_offset`) and optimizes tail TTFT.
`lcfs` orders by adjusted reverse arrival time (`priority_jump + arrival_offset`) and mainly serves controlled comparison experiments.
`wspt` orders by `(1 + priority_jump) / isl_tokens` and optimizes average TTFT.

For all three policies, the complete pending-queue key is
`(strict_priority, policy_key)`. Higher strict tiers always win; the selected
policy orders requests within a tier.

### Policy-Class Queues

YAML profiles define a matrix from client-requested policy family and
router-observed cache bucket to a physical policy-class queue. Clients send the
requested family through `x-dynamo-meta-policy-class`. The router computes
uncached ISL as `raw ISL - best cached tokens across eligible workers`, selects
the highest matching `uncached_isl_buckets` floor, and resolves the
family/bucket pair to one configured class.

An exact header matching a class with neither `policy_family` nor
`cache_bucket` selects that explicit class directly and intentionally bypasses
cache-derived classification. A recognized family selects that family.
Missing, empty, unknown, or ordinary physical-class names use
`default_policy_family`, so a client cannot bypass cache bucketing by naming a
matrix class directly.

Each class owns its FCFS or WSPT heap, busy thresholds, queue limits, quantum,
deficit, and counters. Absolute and fractional busy thresholds use OR
semantics. When neither is specified, the fractional threshold defaults to
`16.0`. A class queues only when every eligible worker is busy for that class,
but a new arrival cannot bypass an existing backlog in the same class.

Queue limits are configured per discovered worker endpoint with
`request_queue_limit_per_worker`, `raw_isl_token_queue_limit_per_worker`, and
`cached_token_queue_limit_per_worker`. The effective class-local limit is the
configured value multiplied by the current number of discovered endpoints.
Limits are checked against current usage before adding the incoming request,
so the request that crosses a limit is accepted and the next queued request is
rejected with HTTP 529 and the effective total. Worker removal does not evict
queued requests; new arrivals reject until usage drains or capacity returns.
DRR charges the uncached-token snapshot captured at enqueue, while raw, cached,
and uncached snapshots remain unchanged for limits, WSPT, counters, and later
dispatch. For the ring cursor, deficit charging, weighted bursts, and bounded
bulk-credit behavior, see
[Deficit Round Robin Queue Scheduling](deficit-round-robin.md).

Every matrix class must identify both `policy_family` and `cache_bucket`; a
class with neither field is explicit, while specifying only one is invalid.
Every configured family must have exactly one physical class for every bucket.
Bucket floors begin at zero and increase strictly.
Class, family, and bucket names use metric-safe identifiers.

Profiles resolve in this order: exact model profile, root profile, then the
synthetic single-class fallback. A model profile completely replaces the root
profile; fields, buckets, families, and classes are not inherited. With no
YAML, the router preserves the existing synthetic `default` queue and does not
compute cache state for classification. See the tested
[sample policy](../../../examples/router/policy-class-queues.yaml).

```bash
python -m dynamo.frontend \
    --router-mode kv \
    --router-policy-config examples/router/policy-class-queues.yaml
```

The previous missing-ISL global admission cap is removed. Cache-derived bucket
selection now resolves directly to an ordinary policy class, and that class
owns the queue threshold, ordering, DRR weight, counters, and limits. There is
no separate first-stage admission queue or global cross-class cap.

This is intentionally not behavior preserving. Class limits are worker-scaled
and class-local rather than global; rejection returns the structured
policy-class HTTP 529 response rather than the previous overload 429 path; and
it does not exclude the entire router instance. The previous flat
`default_policy_class` and `uncached_isl_policy_class_tiers` schema is not
accepted, and ordinary physical classes are no longer direct header
overrides. The sample is a Baseten-oriented continuing-session starting point,
not a compatibility profile.

For `--router-mode device-aware-weighted`, set `DYN_ENCODER_CUDA_TO_CPU_RATIO` to the approximate throughput ratio of one non-CPU worker relative to one CPU worker. The default is `8`.

## Session Affinity

Session affinity is disabled by default. On the frontend, set
`--router-session-affinity-ttl-secs` or `DYN_ROUTER_SESSION_AFFINITY_TTL_SECS` to
a value from `1` through `31536000` to enable it, then send
`X-Dynamo-Session-ID` to keep related requests on one worker. Supplying the header
without the TTL option provides session identity but does not enable router affinity.

The first affinity request creates one immutable binding from the session ID to a
worker and, when available, a data-parallel rank. The binding is scoped to the
existing endpoint and phase, so disaggregated prefill and decode routes remain
separate. Later requests exact-dispatch to that target without transport fallback.
An existing local or shared binding takes precedence over explicit routing headers;
those headers are proposals only while the claim is absent. Direct mode therefore
requires an explicit target for a new binding, but an existing binding supplies the
target for later requests. Query-only requests remain read-only and do not create or
close claims.

With etcd or FileStore on a filesystem shared by all replicas, frontends coordinate
through an immutable distributed claim. The existing-session hot path reads only the
process-local cache. A cache miss reads shared storage first and attempts an atomic
insertion only when the claim is absent. Racing frontends all cache and dispatch to
the stored winner. Storage errors fail the request before scheduler bookkeeping or
dispatch. MemoryStore coordinates only callers sharing the same process and store.
Kubernetes discovery does not provide cross-process affinity and keeps process-local
behavior.

For distributed backends, `--router-session-affinity-ttl-secs` controls only
process-local cache eviction. A cache miss after local eviction reloads the immutable
claim. The claim itself follows the creating frontend's existing etcd lease or
FileStore ownership lifetime; it is not a global idle-session timeout. Delete events
eventually invalidate other frontend caches. Watch lag, disconnect, or restart clears
the entire local affinity cache, and later requests reload claims on demand.

`X-Dynamo-Session-Final: true` marks a terminal request. Dynamo routes that request
normally, then evicts the closing frontend's cache entry and idempotently deletes the
shared claim. Other replicas observe the delete eventually. Close must not race active
requests, and callers must not use that session ID again. The same no-reuse rule
applies after claim expiry. If the bound worker disappears while the claim exists,
exact dispatch fails; start a new session with a new session ID.

Global idle-session TTL, rebinding, dead-worker replacement, compare-and-swap updates,
fencing, generations, broader `WorkerSet` affinity, and backend-tokenized path
expansion are outside this contract. The setting remains independent of
`--router-ttl-secs` and `--router-predicted-ttl-secs`; omit it to disable affinity.

### AIC Prefill Load Model

Use `--router-prefill-load-model aic` when you want prompt-side load tracking to decay the oldest active prefill request using an AIC-predicted duration instead of keeping prompt load static until first token. For the cost-model behavior, see [Prefill Load Modeling](router-concepts.md#prefill-load-modeling).

Enable it on the frontend like this:

```bash
python -m dynamo.frontend \
    --router-mode kv \
    --router-prefill-load-model aic \
    --aic-backend vllm \
    --aic-system h200_sxm \
    --aic-model-path nvidia/Llama-3.1-8B-Instruct-FP8
```

Required when `--router-prefill-load-model=aic` is enabled:

- `--router-mode kv` on the frontend
- `--router-track-prefill-tokens`
- `--aic-backend`
- `--aic-system`
- `--aic-model-path`

Optional AIC knobs:

- `--aic-backend-version`: pinned AIC database version; if omitted, Dynamo uses a backend-specific default
- `--aic-tp-size`: tensor-parallel size for the modeled backend; defaults to `1`
- `--aic-moe-tp-size`: MoE tensor-parallel size for models that require AIC MoE parallelism
- `--aic-moe-ep-size`: MoE expert-parallel size for models that require AIC MoE parallelism
- `--aic-attention-dp-size`: attention data-parallel size for models that require AIC MoE parallelism

For MoE models, these values must satisfy AIC's parallelism constraint:
`aic_tp_size * aic_attention_dp_size == aic_moe_tp_size * aic_moe_ep_size`.
For Kimi-style TP-only MoE runs, use `--aic-moe-tp-size` equal to `--aic-tp-size`,
`--aic-moe-ep-size 1`, and `--aic-attention-dp-size 1`.

## KV Event Transport and Persistence

- `--no-router-kv-events`: Disables KV event tracking. By default, the router consumes KV events to monitor block creation and deletion from workers that publish them. When disabled, the router predicts cache state from routing decisions with TTL-based expiration.
- `--router-durable-kv-events`: **Deprecated.** Enables JetStream mode for KV event transport. The event-plane subscriber in local indexer mode is now the recommended path.
- `--router-reset-states`: Only applies in JetStream mode (`--router-durable-kv-events`). Resets the router state on startup by clearing both the JetStream event stream and NATS object store, starting from a fresh state.
- `--router-snapshot-threshold`: Only applies in JetStream mode (`--router-durable-kv-events`). Sets the number of messages in JetStream before triggering a snapshot.

## Topology-Aware KV Transfer

Topology-aware KV transfer is configured on workers through runtime metadata, not with frontend router flags. In Kubernetes, use `spec.experimental.kvTransferPolicy` on the `DynamoGraphDeployment`; the operator injects the worker environment and topology files. Outside Kubernetes, set `DYN_TOPOLOGY_ENABLED`, `DYN_TOPOLOGY_MOUNT_PATH`, `DYN_KV_TRANSFER_DOMAIN`, `DYN_KV_TRANSFER_ENFORCEMENT`, and `DYN_KV_TRANSFER_PREFERRED_WEIGHT` on workers.

For the full runtime contract and routing behavior, see [Topology-Aware KV Transfer](topology-aware-kv-transfer.md).
For Kubernetes deployment examples, see [Kubernetes Topology-Aware KV Transfer](../../kubernetes/topology-aware-kv-transfer.md).

## Block Tracking

- `--no-router-track-active-blocks`: Disables tracking of active blocks used for ongoing generation or decode phases. Disable this when routing to workers that only perform prefill.
- `--router-track-output-blocks`: **Experimental.** Enables tracking of output blocks during generation. When enabled, the router adds placeholder blocks as tokens are generated and applies fractional decay based on progress toward the expected output sequence length (`agent_hints.osl` in `nvext`). For the cost-model behavior, see [Decode Load Modeling](router-concepts.md#decode-load-modeling).
- `--no-router-assume-kv-reuse`: When tracking active blocks, disables the assumption of KV cache reuse. This is useful in disaggregated setups where transferred blocks are not actually deduplicated on the decode side.
- `--no-router-track-prefill-tokens`: Disables prompt-side prefill token accounting in the router's active load model. Use this for decode-only routing paths where prompt processing already happened elsewhere.
- `--router-replica-sync`: Disabled by default. Enables NATS-based synchronization of local routing decisions between router replicas.

## KV Indexer / Approx KV Indexer

- `--router-ttl-secs`: Time-to-live in seconds for blocks in the router's local cache predictions. Defaults to 120.0 seconds when `--no-router-kv-events` is used.
- `--router-event-threads`: Number of KV indexer worker threads (default: 4). Values greater than 1 use the concurrent radix tree for event-driven routing, approximate routing with `--no-router-kv-events`, and the predict-on-route side indexer.
- `--router-predicted-ttl-secs`: Enables predict-on-route with this TTL in seconds for entries in a local side indexer. Requires KV events; omit to disable. When enabled, the router feeds each routing decision into the side indexer and scores each worker with the larger overlap from the primary indexer and the local side indexer. Independent of `--router-ttl-secs`; kept short so decisions the engine never confirms (cancelled requests, prefill failures) age out quickly.

### When to use `--router-predicted-ttl-secs`

Without this setting, an event-driven router depends entirely on engine KV events to learn which worker now holds which prefix. That works for steady-state traffic, but creates a race when many sibling requests arrive in a single batch — for example, 16 problems × 4 samples each with a shared system prompt, or any parallel-sampling / best-of-N workload. No engine has emitted a "block stored" event yet, so the router scores every sibling with zero overlap and round-robins them across workers. The prefix then gets prefilled on every worker instead of being reused.

Setting `--router-predicted-ttl-secs 5` makes the router record each routing decision into a secondary, short-TTL approximate indexer. When the next sibling is scored, the router queries both indexers and takes the per-worker max overlap, so siblings see the first sibling's prefix immediately and pin to the same worker. The primary event-driven indexer is untouched — engines compute their sequence hashes with salts and cryptographic digests the router cannot reproduce, so inserting router-computed hashes into the primary would key the same physical block under two different hashes and pollute the tree. Running the two trees in parallel sidesteps that entirely; the side tree has a short TTL and its entries simply expire once the primary takes over.

Do not combine this setting with `--no-router-kv-events`, including when the approximate primary is remote: approximate mode already inserts on routing decisions by construction, and running a second approximate side indexer is redundant. With `--use-remote-indexer` and KV events enabled, the side indexer remains local to the consumer router while the remote indexer remains the shared primary view. If a router also serves an indexer for other routers, the side indexer is still local only; it is never served or consumed as the remote primary.

To implement KV event publishing for custom inference engines, see [KV Event Publishing for Custom Engines](../../integrations/kv-events-custom-engines.md).
For details on per-request agent hints (`priority`, `osl`, `speculative_prefill`), see [NVIDIA Request Extensions (`nvext`)](../frontend/nvext.md#agent-hints).

## Tuning Guidelines

`--router-kv-overlap-score-credit` is the primary knob for cache reuse. It credits device-local prefix overlap against the prefill load and must be between 0.0 and 1.0. Higher values steer requests toward workers with better cache overlap and reduce TTFT. Lower values distribute load more evenly and reduce ITL. The default of 1.0 is a reasonable starting point. For direct router APIs and EPP integrations, the same router policy can be overridden per request with `router_config_override.overlap_score_credit`; it is not an `nvext.agent_hints` field.

Use `--router-kv-overlap-score-credit-decay` to reduce that device-local credit when a worker has more active prefill work than the least-loaded eligible worker. This helps prevent busy, cache-rich workers from repeatedly winning while newly autoscaled or lightly loaded workers receive too little traffic. The router normalizes the excess active prefill blocks by the incoming request size and multiplies the configured overlap credit by `1 / (1 + decay * normalized_excess)`. For example, a decay of `1` halves device credit at one request-equivalent of excess prefill load. Host, disk, and shared-cache credits are unchanged. This setting requires prefill-token tracking to have an effect and defaults to `0`.

Use `--load-aware` when you want the KV scheduler's active load model without prefix/cache reuse. This is equivalent to using KV mode with overlap credit set to 0, KV events disabled, KV reuse assumptions disabled, active load tracking enabled, and shared-cache routing disabled. `--router-prefill-load-scale` remains available to tune prompt-side load relative to decode blocks.

Deprecated: `--router-kv-overlap-score-weight`, `--kv-overlap-score-weight`, `DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT`, and `DYN_OVERLAP_SCORE_WEIGHT` are still accepted, but emit deprecation warnings. Nonzero legacy values map to `prefill_load_scale` to preserve existing behavior without changing overlap credit. A legacy value of 0 maps to both `prefill_load_scale=0` and `overlap_score_credit=0`, which preserves the old no-overlap/no-indexer behavior. If a deprecated overlap score weight is still present, it takes precedence over the newer prefill load scale field; a legacy value of 0 also takes precedence over the newer overlap credit field. When migrating to `--router-prefill-load-scale` or `DYN_ROUTER_PREFILL_LOAD_SCALE`, remove the deprecated flag, env var, or JSON field from the deployment config. Use `--router-kv-overlap-score-credit` or `DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT` only when you mean to tune the cache-overlap credit itself.

If an older config used overlap score weight above 1.0 to make the router care more about TTFT, keep the overlap credit at or below 1.0 and move that larger value to `--router-prefill-load-scale` instead. `prefill_load_scale` multiplies the overlap-adjusted prompt-side load, so it still implicitly accounts for device, host, disk, and shared-cache credits.

Use `--router-prefill-load-scale` when prompt-side load should count more or less than decode-side block load after cache-hit credits are applied. The final score is `prefill_load_scale * adjusted_prefill_blocks + decode_blocks`.

Use `--router-host-cache-hit-weight` and `--router-disk-cache-hit-weight` when the backend exposes lower-tier prefix cache via a KV connector (for example, vLLM's `OffloadingConnector` for CPU offload, or a disk-backed tier). These multipliers control how much each lower-tier hit credits against the prefill load, mirroring the role of `--router-kv-overlap-score-credit` for the device tier. A worker holding a full prefix in CPU offload gets `host_cache_hit_weight * matched_blocks` credit against its prefill cost; raising the weight makes the router more willing to route prefix-matched requests to that worker even if a different worker has a partial device-local match.

Use `--no-router-kv-events` when you are not confident that your backend engine emits KV events correctly. In this mode the router falls back to approximate routing, predicting cache state from its own routing decisions with TTL-based expiration.

Use `--router-predicted-ttl-secs 5` when the workload fires bursts of sibling requests with shared prefixes — parallel sampling, best-of-N, agent fan-out. It closes the window between the routing decision and the engine's first "block stored" event so siblings co-locate on the worker the first sibling picked. See the configuration section above for the side-indexer mechanics.

Use `--no-router-assume-kv-reuse` in disaggregated setups where the decode worker does not reuse transferred KV cache blocks. Without this flag, the router undercounts decode blocks when duplicates exist, leading to inaccurate load estimates.

Use `--no-router-track-prefill-tokens` when a router is serving decode-only traffic and prompt processing has already completed elsewhere. This keeps decode routing decisions focused on decode-side load instead of briefly charging prompt tokens to the decode worker after handoff.

Use `--router-track-output-blocks` when your workload is output-heavy and you want the router to account for output-side KV cache growth in load balancing. If you also pass `nvext.agent_hints.osl` per request, the router applies fractional decay to output blocks so that requests nearing completion contribute less future load. See [Decode Load Modeling](router-concepts.md#decode-load-modeling) for the cost-model details.

`--router-queue-threshold` controls when incoming requests are held in a priority queue. The router waits while all eligible workers exceed `threshold * max_num_batched_tokens`, then releases work as capacity frees up. A lower value queues earlier; `0.0` queues as soon as all eligible workers have any active prefill tokens. Priority hints have no router-level effect when requests do not enter this queue.

This threshold delays dispatch. It does not remove workers from the candidate set; for that distinction, see [Router Filtering](router-filtering.md).

Use `DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS` when queued requests may wait long enough for worker cache state to materially change before dispatch. The default is `10` seconds; set it to `0` to disable dequeue-time overlap refresh.

**Note for the SGLang backend.** In Dynamo v1.1.0 and later, the value the SGLang worker publishes for `max_num_batched_tokens` in its Model Deployment Card depends on the server args:

- If `--max-prefill-tokens` is set, MDC's `max_num_batched_tokens` equals that value (the per-step prefill window — the value most users expect).
- If `--max-prefill-tokens` is not set, MDC's `max_num_batched_tokens` falls back to `max_total_num_tokens` from SGLang's `scheduler_info`, which is the **total KV cache pool** in tokens. On large GPUs with high `mem-fraction-static` the pool can be hundreds of thousands of tokens — much larger than `chunked-prefill-size`.

The threshold is applied as `active_tokens > threshold * max_num_batched_tokens`, so this fallback inflates the effective denominator and a threshold like `1.0` may effectively never queue. To get the originally intended "fraction of the per-step prefill window" semantics on SGLang, either set `--max-prefill-tokens` explicitly on the SGLang backend so the MDC value matches the prefill window, or use a much smaller `--router-queue-threshold` (for example `0.1`) to compensate for the inflated denominator.

Use `--router-prefill-load-model aic` when you want prompt-side load tracking to decay the oldest active prefill request using an AIC-predicted duration instead of keeping prompt load static until first token. This requires `--router-track-prefill-tokens` and the shared `--aic-*` config; see [AIC Prefill Load Model](#aic-prefill-load-model) for the full flag set and [Prefill Load Modeling](router-concepts.md#prefill-load-modeling) for the cost-model details.

Use `--router-queue-policy wspt` when your workload has a mix of short and long requests and you want to minimize average TTFT. Use the default `fcfs` when you want to minimize tail TTFT.

## Prometheus Metrics

The router exposes Prometheus metrics on the frontend's HTTP port (default 8000) at `/metrics`:

- **Router request metrics** (`dynamo_component_router_*`): Registered via the component's metrics hierarchy and exposed on the frontend via the `drt_metrics` bridge. In KV mode they are populated per request; in non-KV modes they are registered with zero values. The standalone router also registers these metrics, available on `DYN_SYSTEM_PORT` when set.
- **Routing overhead metrics** (`dynamo_router_overhead_*`) and **per-worker gauges** (`dynamo_frontend_worker_*`): Registered on the frontend's own Prometheus registry. These are frontend-only and not available on the standalone router.

For the full list of router metrics, see the [Metrics reference](../../observability/metrics.md#router-metrics).
