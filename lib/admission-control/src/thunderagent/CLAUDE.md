# ThunderAgent admission strategy

This module implements ThunderAgent as one `PolicyClassAdmissionStrategy`. It controls whether a session's next request may enter the normal ready queue and, when ready, whether it must return to a specific worker. It does not own the queue, select the final worker, or communicate with engines.

## Files

- `capacity.rs` converts worker runtime configuration into per-worker/rank token capacity.
- `config.rs` defines and validates ThunderAgent's tuning parameters.
- `registration.rs` constructs the configured strategy for a policy class.
- `strategy.rs` contains the session state machine, accounting, pause/resume policy, placement, and tests.

## Current policy defaults

| Field | Default | Role |
|---|---:|---|
| `pause_threshold` | `0.95` | Start pressure handling above this fraction of logical worker capacity. |
| `pause_target` | `0.80` | Pause down to, and greedily resume up to, this fraction. There is no separate resume watermark. |
| `resume_timeout_seconds` | `1800` | Bound starvation by releasing a timed-out request to normal worker selection. |
| `session_retention_seconds` | `1800` | Retain quiescent placement and footprint state after the last successful turn. |
| `scheduler_interval_seconds` | `5` | Minimum interval between reconciliation passes. |

## Differences from upstream ThunderAgent

The comparison target is `ThunderAgent-org/ThunderAgent` at `7ddc861027`, primarily `scheduler/router.py` and `backend/state.py`. This implementation retains upstream's global new-program fairness gate, smallest-ACTING-first and deferred-REASONING pause, three resume-priority groups, largest-first packing, and starvation timeout; the intentional differences are:

| Change | Why |
|---|---|
| Require a stable `session_id`; bypass sessionless traffic instead of grouping it under a default program. | Identity is the working-set ownership key, and bypass keeps unrelated non-agent traffic out of ThunderAgent state. |
| Preserve the assigned worker across normal pressure suspension; upstream clears the backend and BFD may resume elsewhere. Structural worker removal and the starvation timeout remain escape paths. | In the matched replay this removed 219 migrations, improved turns by 12.4%, and raised physical cache reuse by 2.26 percentage points. |
| Trigger pressure at 0.95, drain to 0.80, and also use 0.80 as the resume ceiling; upstream pauses only after projected overflow and resumes against remaining capacity. | The proactive high/low pair avoids engine-cache saturation, while earlier pausing and a separate resume ceiling both lost throughput or reuse. |
| Account exact live logical context from `RequestProgress` against device plus native-offload capacity; omit upstream's character estimate, shared-token discount, per-program buffer, ACTING weight, and optional decay. | Exact Dynamo observations remove interacting heuristics; the buffer ablation did not help, and no retained decision required decay or polled residency. |
| Expire quiescent state after a 30-minute inactivity lease instead of requiring upstream's explicit `/programs/release` call. | Immediate terminal cleanup admitted too much live context and hurt reuse; the lease bounded retained state while matching or improving the no-expiry control. |
| Encode only `Running`, `IdleResident`, and `Suspended`, with waiters and rollback owned by native queue admission. | This removes invalid status/lifecycle combinations and gives cancellation and same-session concurrency one authoritative owner without changing the retained pause/resume policy. |

## State model

ThunderAgent keys state by `session_id`. The ID must remain stable across a trajectory's turns; a fresh per-request ID creates unrelated programs and defeats continuation admission and affinity.

Each retained program is in exactly one state:

- `Running { progress, pause_after_completion }`: one request is admitted or running. Its live progress handle supplies the current logical footprint. It cannot be suspended mid-request.
- `IdleResident { footprint, last_activity }`: the previous request completed and its logical footprint remains resident on the assigned worker.
- `Suspended { footprint, since }`: the logical footprint has no capacity reservation. A current request, when present, remains deferred until resume.

`New` is represented by no program entry. `Expired` is represented by removing the entry rather than retaining a tombstone. The worker assignment and completed-step count are common program fields. `RequestState` records the live request-progress and worker-eligibility handles plus the prior program snapshot used for rollback. `SessionRequests` serializes concurrent requests for one session: one current request and an ordered waiter set.

## Request flow

```text
AdmissionRequest
  -> no session_id: Bypass
  -> another request for the session is current: Defer as a session waiter
  -> begin request
       -> create Running state from the request's live progress handle
       -> suspended session: Defer
       -> valid sticky worker available: Ready(Exact)
       -> sticky worker temporarily overloaded: Defer without migration
       -> new session while any session is suspended: Defer for fairness
       -> no eligible worker has usable capacity metadata: Ready(Any), letting the normal router fail open
       -> enough projected capacity: Ready(Exact) on least-used eligible worker
       -> otherwise: Defer
  -> router queue and selector
  -> Dispatched: commit the selected worker
  -> Completed: store returned context size, become IdleResident, apply deferred suspension
  -> Aborted or pre-dispatch failure: restore the prior program snapshot
  -> promote the next request waiting for the same session
```

Worker eligibility is live. A deferred request retains `WorkerEligibility` and takes a fresh snapshot during reconciliation, so worker churn and transient overload do not require copied router state.

## Logical capacity accounting

Capacity is `total_kv_blocks * block_size + native_offloading_capacity_tokens` for each worker/rank. Workers with missing or zero device capacity are excluded from ThunderAgent capacity gating, with a one-time warning. If no worker reports usable metadata, capacity gating is disabled and requests continue through normal router selection.

Only Running or IdleResident programs with an assigned worker contribute usage:

```text
normal usage = program tokens
```

Running programs read their full logical context directly from the retained `RequestProgress` handle. The response path updates that handle with one relaxed atomic operation and sends no per-output actor event. Completion remains authoritative.

This is a logical projection, not live engine or indexer residency.

Completed sessions remain retained for `session_retention_seconds` (30 minutes by default). The clock resets after every successful turn. Once the retention lease expires, reconciliation removes an IdleResident or Suspended session only when it has no current or waiting request. This approximates useful cache residence without requiring clients to emit a terminal signal; a later turn is treated as a new ThunderAgent program and still goes through normal KV-aware worker selection.

## Reconciliation algorithm

Reconciliation runs no more often than `scheduler_interval_seconds` and always uses this order:

1. Expire quiescent retained sessions whose retention lease elapsed.
2. Greedy resume.
3. Forced resume for timed-out sessions.
4. Pause overloaded workers.

Resume-before-pause is intentional source behavior.

### Greedy resume

The usable ceiling is `pause_target * capacity`. Suspended programs are considered in these groups, then by increasing context size:

1. A waiting continuation after at least one completed turn.
2. A first-step program, whether waiting or idle.
3. An idle multi-step program with no waiting request.

Sessions that cannot fit an eligible worker are skipped. The remaining candidates are placed largest-context first onto the eligible worker with the most remaining capacity. Resuming a session with a deferred request emits `MakeReady`; resuming an idle session only changes its program state.

### Forced resume

A current request suspended for `resume_timeout_seconds` bypasses the normal fit test, clears its assignment, and returns to normal worker selection with `WorkerPlacement::Any`. This is the starvation backstop and the only pressure-policy path that relaxes an otherwise structurally valid assignment. A session waiting only because every structurally valid worker is temporarily overloaded remains deferred.

### Pause

When a worker exceeds `pause_threshold * capacity`, ThunderAgent suspends the smallest IdleResident sessions until usage reaches `pause_target * capacity`. Running sessions cannot be interrupted, so if those suspensions are insufficient they are marked and become Suspended only after their current request completes.

## Primitive mapping

| Primitive | Current implementation |
|---|---|
| Session identity | `AdmissionRequest::session_id` keys program and request state. |
| Token accounting | Incoming and completed context sizes update one logical total per session. |
| Block and resume | `Defer` keeps the request in the KV-router admission controller; `MakeReady` releases it. |
| Retention capacity | Static device KV plus native-offload capacity from worker configuration. |
| Pack and fairness | New-session fairness, grouped resume, largest-first placement, smallest-resident suspension, deferred Running suspension, high/low watermarks, and timeout. |
| Sticky placement | `WorkerPlacement::Exact` preserves the selected worker/rank across turns and pressure suspension. Structural worker removal or the starvation timeout can clear it. |
| Session expiry | A quiescent retained session is forgotten after `session_retention_seconds`; a later request starts a new program. |

## Invariants

- Sessionless requests allocate no ThunderAgent state.
- At most one request per session mutates program state at a time.
- Placement never widens the request's router-owned eligibility.
- Temporary overload does not migrate a sticky session before the configured starvation timeout.
- A request is released at most once.
- Abort restores the exact program state that existed before admission.
- Deferred requests remain owned and accounted for by the KV-router queue, not this module.
- Retention expiry never removes a Running session or a session with a current or waiting request.

## Deliberately absent

- Live indexer/cache residency queries.
- Client-driven session-final cleanup; inactivity retention is the lifecycle mechanism.
- In-flight decode preemption or migration.
- Soft KV demotion, prefetch, or retention actions.
- Separate plug-ins for accounting, pressure, victim selection, packing, decay, or placement.

Add these only when an algorithm requiring them is being implemented and measured.

## Validation

Run `cargo test -p dynamo-admission-control`. The tests in `strategy.rs` cover admission, serialization, eligibility changes, pause/resume ordering, timeout, dispatch/abort rollback, and token accounting.
