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
- **Busy-threshold overload**: When busy thresholds mark a worker overloaded, the router removes that worker from the scheduling candidate set. A pinned overloaded worker returns `PinnedWorkerOverloaded`; if every otherwise eligible worker is overloaded, the scheduler returns `AllEligibleWorkersOverloaded`.

These checks are centralized in the router's routing eligibility path so selection, pinned validation, and queue admission all use the same worker and DP-rank rules.

## Busy Thresholds

Busy thresholds are worker filters. They are configured separately from the queue threshold:

- `--active-decode-blocks-threshold`
- `--active-prefill-tokens-threshold`
- `--active-prefill-tokens-threshold-frac`

When a threshold is exceeded, worker monitoring reports that worker as overloaded. The scheduler then excludes that worker from normal candidate selection. The thresholds can be updated at runtime through the `/busy_threshold` HTTP endpoint.

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
- Active decode blocks and prompt-side prefill load
- `preferred_taints`, which multiply the worker cost when present
- `router_temperature`, which samples among eligible candidates when non-zero
- `overlap_score_credit`, `prefill_load_scale`, and shared-cache weighting

Use required taints or allowed worker IDs when routing must exclude workers. Use preferred taints and cost-model tuning when routing should only bias toward a subset.
