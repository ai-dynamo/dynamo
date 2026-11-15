<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-scheduler

The router's scheduler kernel: it runs the routing stages (index lookup, slot tracking,
session affinity, worker selection, admission) atomically for one request. It owns the
request and response vocabulary, eligibility, the queue actor and `LocalScheduler`, the
overlap and prefill-load analysis, the selection kernel, the validated kernel configuration,
and the tracking hash. It depends on the five stage crates, `dynamo-router-protocol`,
`dynamo-kv-hashing`, and `dynamo-kv-hints`, and on nothing that opens a socket or serves
HTTP; `dynamo-kv-router` composes it with those adapters. Its `protocols` module holds the
kernel's own request, response, and config-like types (`RouterRequest`, `RouterResponse`,
`PotentialLoad`, `RoutingConstraints`, `WorkerConfigLike`, `WorkerSelectionResult`,
`RouterEventSink`) and re-exports nothing; the kernel imports stage and wire types from their
owners.

## Working rules

Invariants, lock order, and actor rules that are not derivable from the code.

- A request ID identifies at most one active scheduler booking. Duplicate adds conflict regardless of the target worker. A serialized migration retry may reuse the ID only after the previous booking has been released.
- Before an unpinned retry, exclude every worker already failed by that migration state machine. Preserve caller allowlists and routing constraints. An affinity-derived pin may be invalidated and rebound; an explicit request pin remains exact.
- A failed stream releases its scheduler booking before the error reaches the retry manager, so a later attempt cannot overlap stale cleanup.
- Cleanup that can outlive an admitted request attempt must carry its internal `AttemptId`. `RequestGuard` fences scheduler mutations with `(request_id, attempt_id)`; the admission lifecycle lease stays request-ID-only until handoff.
- Prepared candidate rows must not select with anything the actor's bookings changed. The caller fills their load inputs from its own projection; the actor re-reads every rank it wrote since and fills those rows again, or projects and fills every row itself, so each admission still observes every earlier booking. Record every actor-side write to a rank's load in `ActorWriteLog` after the write, or a prepared projection misses it.
- `SchedulerQueueActor::admit_one` is the required admission path: compute projected load, select a worker, skip the capacity reservation if the response receiver is closed, then reserve capacity before responding. Failed response delivery must release that capacity. Do not bypass this for normal scheduling.
- Do not remove or weaken `admission_gate` without proving selection and capacity reservation cannot assign more work than the workers can hold.
- Potential-load projection goes through `ActiveSequencesMultiWorker::potential_blocks_and_tokens_at(...)` with `SchedulingRequest::prefill_token_deltas()`; do not scan per-worker `ActiveSequences` from scheduling.
- `SchedulingRequest` helper methods are the single source for effective cached tokens, effective overlap, worker allowance, prefill-token defaults, and request block count; do not duplicate this logic in policies or selectors.
- Weighted shortest processing time (WSPT) uses cache-aware prefill cost: pinned requests use the pinned worker's effective cached tokens, unpinned requests the best allowed worker. Do not silently fall back to raw input sequence length unless tracking is disabled or cache data is absent.
- Pinned-worker and allowed-worker constraints are validated before selection and respected by queue capacity checks, selector candidate iteration, and WSPT priority.
- Prefill load hints are computed at scheduler/request boundaries from selected-worker `cached_tokens`; do not move ISL/cache-token math back into `ActiveSequences`.
- Selectors are side-effect free: no capacity reservation, no queue mutation, no `PromptRegistry` mutation.
- Do not hold the pending-heap lock while selecting, reading worker capacity, responding, or awaiting; the queue heap is only for waiting requests.
- Do not hold `workers_with_configs.borrow()` across `.await`; take a short synchronous snapshot or borrow for selection only.
- Within one strict-priority tier, deadline-bearing entries order earliest-due-first ahead of all non-deadline entries before the FCFS/LCFS/WSPT policy score applies; all deadline math uses `tokio::time::Instant` so paused-clock tests stay coherent.
- Any change to queue ordering, WSPT keys, capacity checks, admission serialization, or selector scoring includes focused tests and before/after routing or queue benchmarks.
- Keep text and external IDs such as request IDs on standard hash collections; use `FxHashMap` / `FxHashSet` for internal numeric hot-path keys only.
