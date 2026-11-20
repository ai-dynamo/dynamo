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

Busy thresholds are worker filters. Set them for a worker role with `busy_threshold` in that role's entry of the `--router-policy-config` file:

```yaml
worker_selection:
  prefill:
    type: dynamo-default-cost-fn
    busy_threshold:
      active_prefill_tokens_frac: 1.5
  decode:
    type: dynamo-default-cost-fn
    busy_threshold:
      active_decode_blocks: 0.85
```

| Key | Meaning |
|---|---|
| `active_decode_blocks` | Fraction of the worker's KV cache blocks in use, from `0.0` through `1.0`, above which the worker is busy |
| `active_prefill_tokens_frac` | Non-negative fraction of the worker's `max_num_batched_tokens` above which its active prefill tokens make the worker busy |

The `aggregated`, `prefill`, and `decode` roles accept the key on any policy type, including `{type: default, busy_threshold: {...}}`. `encode` rejects it, because the router does not monitor the load of encode workers. A role's entry is taken whole: a limit it omits is not checked, and `busy_threshold: {}` checks nothing. The key with no value (`busy_threshold:` alone, for example after its only limit is commented out) is a startup error. Each role's entry is its own, so a prefill pool does not use the `decode` entry. Only the frontend reads the key; the standalone router and the inference-gateway endpoint picker accept it in a policy file and ignore it.

A role's `busy_threshold` rejects requests: once every eligible worker of the role is past it, the request fails with HTTP 529. The `prefill_busy_threshold_frac` and `decode_busy_threshold_frac` keys of a [policy class](configuration-and-tuning.md#policy-class-queues) in the same file do not reject. They hold the request in the router queue until a worker is free; see [Queue Backpressure](#queue-backpressure).

A role whose entry has no `busy_threshold` keeps the earlier rules for now. A worker set that advertises router settings uses its own threshold flags, and a worker set that advertises nothing uses the frontend's:

- `--active-decode-blocks-threshold`
- `--active-prefill-tokens-threshold`
- `--active-prefill-tokens-threshold-frac`

A prefill pool without the key uses the same advertised or flag thresholds as the decode set it fronts, not the `decode` entry of the policy file. On the frontend, `--active-decode-blocks-threshold` and `--active-prefill-tokens-threshold-frac` and their environment variables are deprecated and scheduled for removal in v1.9; see [Deprecations](../../../../reference/general/releases/deprecations.mdx#v180). The absolute `--active-prefill-tokens-threshold` was deprecated earlier and has no policy-file key. The frontend logs each role's starting thresholds and where they came from (`busy_thresholds_source`) on the `Activating router` line, and for a prefill pool on the `Prefill pool busy thresholds` line.

A worker set in direct routing mode has no busy thresholds, because the caller picks its workers. Its prefill pool has none either: the pool follows the mode of the decode set it fronts, whatever mode the prefill workers advertise.

When a threshold is exceeded, worker monitoring reports that worker as overloaded. The scheduler then excludes that worker from normal candidate selection. The thresholds can be updated at runtime, for a whole model or for one role, through the [`/busy_threshold` admin endpoint](configuration-and-tuning.md#busy-thresholds).

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
