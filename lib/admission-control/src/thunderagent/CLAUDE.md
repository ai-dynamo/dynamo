# ThunderAgent admission strategy

This module implements ThunderAgent as one `PolicyClassAdmissionStrategy`. It controls whether a session's next request may enter the normal ready queue and, when ready, whether it must return to a specific worker. It does not own the queue, select the final worker, or communicate with engines.

## Files

- `capacity.rs` converts worker runtime configuration into per-worker/rank token capacity.
- `config.rs` defines and validates ThunderAgent's tuning parameters.
- `registration.rs` constructs the configured strategy for a policy class.
- `strategy.rs` contains the session state machine, accounting, pause/resume policy, placement, and tests.

## State model

ThunderAgent keys state by `session_id`.

Each program has two independent state dimensions:

- `Reasoning`: a request is admitted or running. It cannot be paused mid-request.
- `Acting`: the previous request completed and the agent is between turns.
- `Active`: its logical KV footprint counts against an assigned worker.
- `Paused`: it has no reservation or assigned worker; a returning request remains deferred until resume.

`RequestState` records the admission ID, context size, live worker-eligibility handle, dispatch status, and the prior program snapshot used for rollback. `SessionRequests` serializes concurrent requests for one session: one current request and an ordered waiter set.

## Request flow

```text
AdmissionRequest
  -> no session_id: Bypass
  -> another request for the session is current: Defer as a session waiter
  -> begin request
       -> update logical context and mark Reasoning
       -> paused session: Defer
       -> valid sticky worker available: Ready(Exact)
       -> sticky worker temporarily overloaded: Defer without migration
       -> new session while any session is paused: Defer for fairness
       -> capacity unavailable: Ready(Any), letting the normal router fail open
       -> enough projected capacity: Ready(Exact) on least-used eligible worker
       -> otherwise: Defer
  -> router queue and selector
  -> Dispatched: commit the selected worker
  -> Completed: store returned context size, mark Acting, apply deferred pause
  -> Aborted or pre-dispatch failure: restore the prior program snapshot
  -> promote the next request waiting for the same session
```

Worker eligibility is live. A deferred request retains `WorkerEligibility` and takes a fresh snapshot during reconciliation, so worker churn and transient overload do not require copied router state.

## Logical capacity accounting

Capacity is `total_kv_blocks * block_size + native_offloading_capacity_tokens` for each worker/rank. Missing or zero device capacity disables ThunderAgent capacity gating for that worker set.

Only active programs with an assigned worker contribute usage:

```text
normal usage = program tokens + buffer_per_program
```

Reasoning programs use their full logical context. Acting programs apply `acting_token_weight` during normal pressure and resume calculations. A separate exponentially decayed Acting value is used only when choosing placement after `resume_timeout_seconds`:

```text
decayed tokens = token_total * 2^(-idle_seconds / acting_decay_tau_seconds)
```

This is a logical projection, not live engine or indexer residency.

## Reconciliation algorithm

Reconciliation runs no more often than `scheduler_interval_seconds` and always uses this order:

1. Greedy resume.
2. Forced resume for timed-out sessions.
3. Pause overloaded workers.

Resume-before-pause is intentional source behavior.

### Greedy resume

The usable ceiling is `(pause_threshold - resume_hysteresis) * capacity`. Paused sessions are considered in these groups, then by increasing context size:

1. Continuing Reasoning sessions after their first step.
2. First-step sessions.
3. Acting sessions.

Sessions that cannot fit an eligible worker are skipped. The remaining candidates are placed largest-context first onto the eligible worker with the most remaining capacity. Resuming a session with a deferred request emits `MakeReady`; resuming an idle session only changes its program state.

### Forced resume

A session deferred for `resume_timeout_seconds` bypasses the normal fit test. It is placed on the eligible worker with the greatest `capacity - decayed_usage`. A session waiting only because all structurally valid workers are temporarily overloaded remains deferred.

### Pause

When a worker exceeds `pause_threshold * capacity`, ThunderAgent pauses the smallest Acting sessions until usage reaches `pause_target * capacity`. Reasoning sessions cannot be interrupted, so if Acting pauses are insufficient they are marked and become paused only after their current request completes.

## Primitive mapping

| Primitive | Current implementation |
|---|---|
| Session identity | `AdmissionRequest::session_id` keys program and request state. |
| Token accounting | Incoming and completed context sizes update one logical total per session. |
| Block and resume | `Defer` keeps the request in the KV-router admission controller; `MakeReady` releases it. |
| Retention capacity | Static device KV plus native-offload capacity from worker configuration. |
| Pack and fairness | New-session fairness, grouped resume, largest-first placement, smallest-Acting pause, deferred Reasoning pause, hysteresis, and timeout. |
| Sticky placement | `WorkerPlacement::Exact` preserves the selected worker/rank across turns. |
| Session close | Not implemented; session-final handling is intentionally deferred. |

## Invariants

- Sessionless requests allocate no ThunderAgent state.
- At most one request per session mutates program state at a time.
- Placement never widens the request's router-owned eligibility.
- Temporary overload does not silently migrate a sticky session.
- A request is released at most once.
- Abort restores the exact program state that existed before admission.
- Paused requests remain owned and accounted for by the KV-router queue, not this module.

## Deliberately absent

- Live indexer/cache residency queries.
- Session-final cleanup.
- In-flight decode preemption or migration.
- Soft KV demotion, prefetch, or retention actions.
- Separate plug-ins for accounting, pressure, victim selection, packing, decay, or placement.

Add these only when an algorithm requiring them is being implemented and measured.

## Validation

Run `cargo test -p dynamo-admission-control`. The tests in `strategy.rs` cover admission, serialization, eligibility changes, pause/resume ordering, timeout, dispatch/abort rollback, and token accounting.
