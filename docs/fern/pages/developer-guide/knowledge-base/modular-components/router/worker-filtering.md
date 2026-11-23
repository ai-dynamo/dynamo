---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Router Filtering
subtitle: Candidate eligibility, DP-rank filtering, and queue backpressure boundaries
---

This page describes which workers the KV router is allowed to consider before it scores candidates. Filtering is separate from the cost model: filters decide whether a worker or DP rank is eligible at all, while scoring ranks the eligible candidates by KV overlap and load.

## Candidate Eligibility

The router applies these hard eligibility checks before worker scoring:

- **Allowed worker IDs**: Request routing hints can restrict routing to a specific set of worker IDs. A pinned worker must also be in this set.
- **Pinned worker and DP rank**: Direct routing, phase-specific routing, and session affinity resolve to an exact `worker_id` and optional `dp_rank`. The router validates that the worker exists and that the requested DP rank belongs to that worker.
- **DP-rank bounds**: For unpinned KV routing, each eligible worker expands into the ranks in `[data_parallel_start_rank, data_parallel_start_rank + data_parallel_size)`. Ranks outside that range are never considered.
- **Required taints**: `required_taints` are hard topology constraints. A worker missing a required taint is filtered out.
- **Not-ready workers**: A worker whose card carries the `dynamo.routing/not-ready` taint is never selected automatically. Only a worker ID supplied by the caller reaches it. See [Not-Ready Workers](#not-ready-workers).
- **Busy-threshold overload**: When busy thresholds mark a worker overloaded, the router removes that worker from the scheduling candidate set. A pinned overloaded worker returns `PinnedWorkerOverloaded`; if every otherwise eligible worker is overloaded, the scheduler returns `AllEligibleWorkersOverloaded`.

These checks are centralized in the router's routing eligibility path so selection, pinned validation, and queue admission all use the same worker and DP-rank rules.

## Not-Ready Workers

A worker can publish itself to discovery while staying out of ordinary routing. This lets a check send requests through a frontend to that exact worker before it takes client traffic.

Set `DYN_START_NOT_READY=true` on the worker. It then registers its model with the `dynamo.routing/not-ready` taint. The setting is off by default.

While the taint is present:

- **Automatic selection skips the worker.** This includes requests whose `required_taints` all match, and requests routed by session affinity.
- **A caller-supplied worker ID reaches it.** Use `nvext.backend_instance_id` or `nvext.decode_worker_id` for decode and aggregated workers, `nvext.prefill_worker_id` for prefill workers, or the equivalent `x-dynamo-worker-instance-id` and `x-dynamo-prefill-instance-id` headers. The pin overrides only the not-ready hold: `required_taints`, allowed worker IDs, DP-rank bounds, and overload checks still apply.
- **A pinned request does not fall back.** If the pinned worker is ineligible, routing fails instead of choosing another worker.

If every worker for a model is not ready, unpinned requests fail with no endpoint.

To release the worker, update its taints without `dynamo.routing/not-ready`. The worker's `POST /engine/update/model_taints` route replaces the caller-managed taint set, so send the taints the worker should keep:

```bash
curl -X POST http://<worker-system-address>/engine/update/model_taints \
  -H 'Content-Type: application/json' \
  -d '{"taints": []}'
```

Because the route replaces the whole set, any update that omits `dynamo.routing/not-ready` releases the worker. A caller that manages other taints on a held worker must include it until the worker should serve.

A restarted worker registers a new instance and starts held again.

This applies to the KV router. The round-robin, random, least-loaded, device-aware, and direct router modes do not read worker taints, so a worker started with `DYN_START_NOT_READY=true` receives traffic from frontends in those modes.

## Busy Thresholds

Busy thresholds are worker filters. They are configured separately from the queue threshold:

- `--active-decode-blocks-threshold`
- `--active-prefill-tokens-threshold`
- `--active-prefill-tokens-threshold-frac`

When a threshold is exceeded, worker monitoring reports that worker as overloaded. The scheduler then excludes that worker from normal candidate selection. The thresholds can be updated at runtime through the [`/busy_threshold` admin endpoint](../../../../reference/components/frontend-configuration.mdx#admin).

Busy filtering preserves error classification:

- If a request has no compatible worker after allow-list, DP-rank, and required-taint checks, routing fails with no endpoint.
- If compatible workers exist but all are overloaded, routing fails as overload.
- If an exact pinned worker is overloaded, routing fails as pinned overload instead of rerouting.

## Queue Backpressure

`--router-queue-threshold` is not candidate eligibility. It is admission backpressure.

When queueing is enabled, the router checks active prefill tokens for the request's eligible workers. If all eligible workers are above the configured fraction of `max_num_batched_tokens`, the request waits in the router queue. The request is scored only after capacity frees up, so dispatch uses fresh load and cache state.

Queueing does not permanently remove a worker from the candidate set. It delays the routing decision; busy-threshold overload filtering removes overloaded workers from candidate selection.

When policy classes are configured, the router uses weighted DRR to choose
which dispatchable class runs next. For credit accumulation, blocked-class
behavior, and oversized-request bulk credit, see
[Deficit Round Robin Queue Scheduling](deficit-round-robin.md).

## Scoring Signals

These signals affect candidate scoring, not hard filtering:

- KV cache overlap on device, host-pinned memory, disk, or shared cache
- Active decode blocks, active-request count, and prompt-side prefill load
- `preferred_taints`, which multiply the worker cost when present
- `router_temperature`, which samples among eligible candidates when non-zero
- `overlap_score_credit`, `prefill_load_scale`, `decode_active_request_weight`, and shared-cache weighting

Use required taints or allowed worker IDs when routing must exclude workers. Use preferred taints and cost-model tuning when routing should only bias toward a subset.
