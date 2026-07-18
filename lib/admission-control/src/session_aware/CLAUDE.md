# Session-Aware Admission Control

This module implements Session-Aware Admission Control as one `PolicyClassAdmissionPolicy`. It controls whether a session's next request may enter the normal ready queue and, when ready, whether it must return to a specific worker. It does not own the queue, select the final worker, or communicate with engines.

## Files

- `capacity.rs` converts worker runtime configuration into per-worker/rank token capacity.
- `config.rs` defines and validates the session-aware admission-control tuning parameters.
- `registration.rs` constructs the configured policy for a policy class.
- `policy.rs` contains the session state machine, accounting, pause/resume policy, placement, and tests.

## Current policy defaults

| Field | Default | Role |
|---|---:|---|
| `pause_threshold` | `0.95` | Start pressure handling above this fraction of logical worker capacity. |
| `pause_target` | `0.80` | Pause down to, and greedily resume up to, this fraction. There is no separate resume watermark. |
| `resume_timeout_seconds` | `1800` | Bound starvation by releasing a timed-out request to normal worker selection. |
| `session_retention_seconds` | `1800` | Retain quiescent placement and footprint state after the last successful turn. |
| `scheduler_interval_seconds` | `5` | Minimum interval between reconciliation passes. |

## Inspiration and comparison

This policy is inspired by the [ThunderAgent paper](https://arxiv.org/abs/2602.13692) and compared with its [reference implementation at commit `7ddc861027`](https://github.com/ThunderAgent-org/ThunderAgent/tree/7ddc861027). Session-Aware Admission Control retains the reference policy's global new-program fairness gate, smallest-ACTING-first and deferred-REASONING pause, three resume-priority groups, largest-first packing, and starvation timeout; the intentional differences are:

| Change | Why |
|---|---|
| Require a stable `session_id`; bypass sessionless traffic instead of grouping it under a default program. | Identity is the working-set ownership key, and bypass keeps unrelated non-agent traffic out of session-aware admission-control state. |
| Dispatch the next turn of an assigned active session without reapplying retained-capacity admission, while still fitting concurrent Running context within device capacity; periodic pressure handling pauses idle victims or marks running victims after completion. | This preserves upstream's active-program continuation semantics without allowing native-offload retention capacity to over-admit live GPU work. Pressure suspension clears affinity so the existing global packer can rebalance resumed work. |
| Trigger pressure at 0.95, drain to 0.80, and also use 0.80 as the resume ceiling; upstream pauses only after projected overflow and resumes against remaining capacity. | The proactive high/low pair avoids engine-cache saturation, while earlier pausing and a separate resume ceiling both lost throughput or reuse. |
| Account exact live logical context from `RequestProgress`; use device capacity to admit Running requests and device plus native-offload capacity for retained sessions. Omit upstream's character estimate, unused shared-token hook, per-program buffer, ACTING weight, and optional decay. | Host HiCache can retain an idle session, but must not be treated as room to enqueue more backend work. Exact Dynamo observations remove interacting heuristics. |
| Consume Dynamo's session-final signal instead of requiring upstream's separate `/programs/release` call, with a 30-minute inactivity lease as fallback. | The terminal request releases accounting immediately and is still forwarded normally; the lease bounds retained state for clients that cannot signal completion. |
| Encode only `Running`, `IdleResident`, and `Suspended`, with waiters and rollback owned by native queue admission. | This removes invalid status/lifecycle combinations and gives cancellation and same-session concurrency one authoritative owner without changing the retained pause/resume policy. |

## State model

Session-Aware Admission Control keys state by `session_id`. The ID must remain stable across a trajectory's turns; a fresh per-request ID creates unrelated programs and defeats continuation admission and affinity.

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
  -> session-final request: forget retained placement/accounting and route the request normally
  -> begin request
       -> create Running state from the request's live progress handle
       -> suspended session: Defer
       -> valid sticky worker available: Ready(Exact)
       -> sticky worker temporarily overloaded: Defer without migration
       -> new session while any session is suspended: Defer for fairness
       -> no eligible worker has usable capacity metadata: Ready(Any), letting the normal router fail open
       -> enough device running capacity and total retention capacity: Ready(Exact) on least-used eligible worker
       -> otherwise: Defer
  -> router queue and selector
  -> Dispatched: commit the selected worker
  -> Completed: store returned context size, become IdleResident, apply deferred suspension
  -> Aborted or pre-dispatch failure: restore the prior program snapshot
  -> promote the next request waiting for the same session
```

Worker eligibility is live. A deferred request retains `WorkerEligibility` and takes a fresh snapshot during reconciliation, so worker churn and transient overload do not require copied router state.

Policy-family and cache-bucket classification, including exact-placement reclassification, are existing KV-router behavior outside this concrete policy. Session-Aware Admission Control neither declares buckets nor changes their defaults.

To opt in selected traffic, configure this policy on an explicit class (a class with neither `policy_family` nor `cache_bucket`):

```yaml
- name: agents
  admission:
    type: session_aware
  quantum: 1
```

Every HTTP request using that explicit class must include `x-dynamo-meta-policy-class: agents`; otherwise it follows the default family and bypasses this policy. To apply Session-Aware admission without per-request policy metadata, configure it on the default policy family's sole cache bucket instead.

Registration rejects `session_aware` on a class in a multi-bucket family. Those buckets are separate physical queues, and cache-state changes can classify later turns of one session into another bucket; until cross-class policy ownership is defined, accepting that shape would silently split the program table.

## Logical capacity accounting

Total retention capacity is `total_kv_blocks * block_size + native_offloading_capacity_tokens` for each worker/rank. A Running request must additionally fit in `total_kv_blocks * block_size` device capacity; this keeps the deferred queue ahead of SGLang even when host HiCache has room. Workers with missing or zero device capacity are excluded from session-aware admission-control capacity gating, with a one-time warning. If no worker reports usable metadata, capacity gating is disabled and requests continue through normal router selection.

Only Running or IdleResident programs with an assigned worker contribute usage:

```text
normal usage = program tokens
```

Running programs read their full logical context directly from the retained `RequestProgress` handle. The response path updates that handle with one relaxed atomic operation and sends no per-output actor event. Completion remains authoritative.

This is a logical projection, not live engine or indexer residency.

When a request carries the session-final signal, admission immediately removes the session's retained placement and accounting state, then forwards that terminal request through normal routing. If another turn for the same session is already running, the final request waits behind it before releasing state. Completion or cancellation of the terminal request cannot recreate the released program.

For clients that cannot emit the signal, completed sessions remain retained for `session_retention_seconds` (30 minutes by default). The clock resets after every successful turn. Once the retention lease expires, reconciliation removes an IdleResident or Suspended session only when it has no current or waiting request. A later turn is treated as a new session-aware program and still goes through normal KV-aware worker selection.

## Reconciliation algorithm

Reconciliation runs no more often than `scheduler_interval_seconds` and always uses this order:

1. Expire quiescent retained sessions whose retention lease elapsed.
2. Greedy resume.
3. Forced resume for timed-out sessions.
4. Pause overloaded workers.

Resume-before-pause is intentional source behavior.

### Greedy resume

The usable ceiling is `pause_target * capacity`. Suspended programs are considered in these groups:

1. A waiting continuation after at least one completed turn.
2. A first-step program, whether waiting or idle.
3. An idle multi-step program with no waiting request.

Within a group, sessions with fewer eligible workers are considered first, then by increasing context size. This preserves the source small-context preference when routing constraints are equal without letting flexible work consume a constrained session's only worker. Candidates reserve actual per-worker retention and device capacity as they are selected; the feasible set is then placed largest-context first onto the eligible worker with the most remaining capacity. If that final repack is incompatible with routing constraints, the feasible selection placement is retained. Resuming a session with a deferred request emits `MakeReady`; resuming an idle session only changes its program state.

### Forced resume

A current request suspended for `resume_timeout_seconds` bypasses the normal fit test and returns to normal worker selection with `WorkerPlacement::Any`. This is the starvation backstop. A session waiting only because every structurally valid worker is temporarily overloaded retains its assignment and remains deferred.

### Pause

When a worker exceeds `pause_threshold * capacity`, Session-Aware Admission Control suspends the smallest IdleResident sessions until usage reaches `pause_target * capacity`. Running sessions cannot be interrupted, so if those suspensions are insufficient they are marked and become Suspended only after their current request completes. Actual pressure suspension clears the assigned worker so greedy resume can globally repack the program.

## Primitive mapping

| Primitive | Current implementation |
|---|---|
| Session identity | `AdmissionRequest::session_id` keys program and request state. |
| Token accounting | Incoming and completed context sizes update one logical total per session. |
| Block and resume | `Defer` keeps the request in the KV-router policy queue; `MakeReady` releases it. |
| Retention capacity | Static device KV plus native-offload capacity from worker configuration. |
| Pack and fairness | New-session fairness, grouped resume, largest-first placement, smallest-resident suspension, deferred Running suspension, high/low watermarks, and timeout. |
| Sticky placement | `WorkerPlacement::Exact` preserves the selected worker/rank across active turns. Pressure suspension clears it so global resume packing can rebalance workers. |
| Session release | A session-final request forgets retained state immediately; inactivity expiry is the fallback for clients without that signal. |

## Invariants

- Sessionless requests allocate no session-aware admission-control state.
- At most one request per session mutates program state at a time.
- Placement never widens the request's router-owned eligibility.
- Transient router overload alone does not migrate a sticky session; an actual pressure suspension may.
- A request is released at most once.
- Abort restores the exact program state that existed before admission, except that a session-final request never recreates released state.
- Deferred requests remain owned and accounted for by the KV-router queue, not this module.
- Retention expiry never removes a Running session or a session with a current or waiting request.

## Deliberately absent

- Live indexer/cache residency queries.
- In-flight decode preemption or migration.
- Soft KV demotion, prefetch, or retention actions.
- Separate plug-ins for accounting, pressure, victim selection, packing, decay, or placement.

Add these only when an algorithm requiring them is being implemented and measured.

## Validation

Run `cargo test -p dynamo-admission-control`. The tests in `policy.rs` cover admission, serialization, eligibility changes, pause/resume ordering, timeout, dispatch/abort rollback, and token accounting.
