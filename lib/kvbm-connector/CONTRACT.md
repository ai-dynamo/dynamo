# vLLM ↔ KVBM Connector Contract

This document is the **source of truth** for how vLLM's V1 scheduler calls the KVBM connector. The contract is narrow and specialized — vLLM does not expose this surface to user code, and the engine-core loop is single-threaded — so we design and test against the **actual** call pattern, not against an unconstrained API.

Defending against scheduler behavior that vLLM cannot produce is over-engineering. Defending against contract-violating **peer** behavior (peer crash, `Frame::Error`, watchdog timeout, RDMA failure) is in scope and orthogonal to this document.

## Scope

- **Covers:** the scheduler-side ↔ leader-side surface of the connector, i.e. methods on `KVConnectorBase_V1` invoked by `vllm/v1/core/sched/scheduler.py` on the engine-core thread.
- **Out of scope:** worker-side connector methods (`start_load_kv`, `wait_for_layer_load`, `save_kv_layer`, `wait_for_save`, `get_finished` on the worker, `bind_connector_metadata`). Those run in worker processes with their own threading model (typically a single forward-pass thread per worker) and are not the subject of this document.
- **Out of scope:** velo wire-level error semantics, kvbm-hub HTTP API, intra-engine block-tier movement. Each has its own contract document.

## The Threading Model

> **THE foundational invariant.** Every other rule follows from this.

The vLLM V1 engine-core loop (`vllm/v1/engine/core.py:405-433`) runs **one step at a time on one thread**:

```
loop {
    if !scheduler.has_requests(): continue
    scheduler_output = scheduler.schedule()        // calls GNMT, USAA
    future = model_executor.execute_model(...)
    model_output = future.result()
    _process_aborts_queue()                         // calls finish_requests
    engine_core_outputs = scheduler.update_from_output(...)  // calls request_finished
}
```

Aborts arrive on a thread-safe queue but are drained on the engine-core thread between schedule + update_from_output. There is **no concurrent dispatch** to scheduler-side connector methods from different threads.

**Consequences:**

- For any single `request_id`, all scheduler-side connector calls (`get_num_new_matched_tokens`, `update_state_after_alloc`, `request_finished`, `update_connector_output`) are **strictly serialized** in lifecycle order.
- For the same `request_id`, GNMT and `request_finished` are **never concurrent**. Same for USAA and `request_finished`. Same for any pair.
- For different `request_id`s within one engine-core tick, calls are still **sequential** (the scheduler iterates its waiting queue one request at a time).
- The connector implementation does not need locks or atomics to serialize its own state mutations *induced by these methods*. It may need them to coordinate **its own** concurrent activity — see §Concurrency.

## Method Roster (Scheduler-Side)

The methods our connector implements and their vLLM callsites:

| Method | vLLM Callsite | Purpose |
|---|---|---|
| `get_num_new_matched_tokens(request, num_computed_tokens) → (Optional[int], bool)` | `scheduler.py:619` inside `schedule()` | Ask the connector how many additional tokens it can supply beyond local prefix cache. Return `(None, _)` to defer (scheduler re-queues). Return `(N, True)` to declare async load (request goes to WAITING_FOR_REMOTE_KVS). Return `(N, False)` for sync. |
| `update_state_after_alloc(request, blocks, num_external_tokens)` | `scheduler.py:779` inside `schedule()` | Scheduler has allocated G1 blocks; commit the CD plan. |
| `build_connector_meta(scheduler_output) → KVConnectorMetadata` | `scheduler.py:959` once per tick | Serialize per-tick metadata to ship to worker-side connectors. |
| `update_connector_output(KVConnectorOutput)` | `scheduler.py:2115` inside `update_from_output()` | Receive worker-side aggregate output (finished_sending, finished_recving, invalid_block_ids). |
| `request_finished(request, block_ids) → (bool, Optional[KvXferParams])` | `scheduler.py:2032` from `_free_request()` | Called **exactly once** per request, before blocks are freed. Return `True` to defer block-free until the connector signals via `get_finished()`. |
| `request_finished_all_groups(...)` | `scheduler.py:2034` | HMA variant; same contract per request, multiple block-group lists. |
| `take_events() → Iterable[KVCacheEvent]` | scheduler metrics path | Drain per-tick KV cache events for telemetry. |

The KVBM connector's leader (`ConnectorLeader`) implements these methods directly. Disaggregation-specific behavior dispatches into the prefill-side / decode-side leaders (`PrefillDisaggLeader`, `DecodeDisaggLeader`) which in turn drive the `ConditionalDisaggCoordinator`.

## Per-Request Lifecycle

Three canonical traces. Each is a strict ordering on the engine-core thread; no other ordering is possible from vLLM.

### Trace 1: Local-only request (no external KV)

```
schedule() tick T:
    get_num_new_matched_tokens(req, ncm=0)   → (0, False)
    (no USAA — num_external_tokens is 0)
... model runs to completion ...
update_from_output() tick T_n:
    request_finished(req, block_ids)         → (False, None)
    [blocks freed by scheduler]
```

### Trace 2: Async-loaded request (CD-bound, decode side)

```
schedule() tick T_0:
    get_num_new_matched_tokens(req, ncm=0)   → (N, True)
    update_state_after_alloc(req, blocks, N)
    [request transitions to WAITING_FOR_REMOTE_KVS]
... worker-side connector pulls KV via velo ...
... worker emits finished_recving[req] in some KVConnectorOutput ...
update_from_output() tick T_k:
    update_connector_output(KVConnectorOutput { finished_recving: {req}, ... })
    [scheduler caches blocks, request → WAITING]
schedule() tick T_{k+1}:
    [request has ncm > 0 now; GNMT NOT called again — see else branch
     at scheduler.py:653-658]
    update_state_after_alloc(req, blocks, 0)  // possibly
... model runs forward + emits ...
update_from_output() tick T_n:
    request_finished(req, block_ids)         → (False, None) or (True, params)
```

### Trace 3: GNMT-deferred request (connector returns None)

```
schedule() tick T_0:
    get_num_new_matched_tokens(req, ncm=0)   → (None, _)
    [scheduler prepends to skipped_waiting; request NOT scheduled]
schedule() tick T_1:
    get_num_new_matched_tokens(req, ncm=0)   → (None, _)  // still not ready
    [...]
schedule() tick T_k:
    get_num_new_matched_tokens(req, ncm=0)   → (N, true/false)
    update_state_after_alloc(req, blocks, N)
... continues as Trace 1 or 2 ...
```

### Per-request ordering invariants vLLM guarantees

For any request_id, on the engine-core thread:

1. `get_num_new_matched_tokens` may be called **zero or more times** with the same `request` before any `update_state_after_alloc`. Each call is synchronous; the next call (or USAA) does not start until the previous returns.
2. `update_state_after_alloc` is called **at most twice** per request — once when blocks are allocated, and possibly a second time when async-load blocks land (per `base.py:491-495` docstring).
3. `request_finished` is called **exactly once** per request, after model output is processed, before blocks are freed.
4. `request_finished` is **always preceded by either** (a) at least one GNMT call **or** (b) an abort that never reached GNMT (cleanup-on-abort path). Abort-without-prior-GNMT means the request was aborted before scheduling — the connector has no state for it; the call is a no-op idempotent against UntrackedRequest.
5. After `request_finished`, no further scheduler-side method is called for that `request_id`. (`build_connector_meta` may still reference the rid in a metadata slot if it was scheduled in the same tick; this is the scheduler's responsibility to settle.)

## GNMT Idempotence

GNMT can legitimately be called multiple times for the same `request_id`. Two ways this happens:

- **Deferred answer** (Trace 3): connector returns `(None, _)`; scheduler will re-call on a later tick. Subsequent calls may pass a different `num_computed_tokens` if vLLM's local prefix cache advanced.
- **Chunked prefill / preemption**: vLLM may evict and restart a request; the next GNMT pass receives a non-zero `num_computed_tokens`. The connector's reply must respect the new `num_computed_tokens` floor.

What this means for us:

- The connector must be **idempotent under repeated GNMT calls before USAA**. Returning the same `(N, true/false)` for the same `(request_id, num_computed_tokens)` is the safe baseline. Returning a different `N` is permitted if the underlying matchable set changed (e.g., a remote prefill peer came online), but the contract requires that USAA reconcile to the most recent reply.
- The connector does **NOT** need to defend against concurrent GNMT calls for the same rid — they are serialized.
- The connector **does** need to defend against state mutations from its own background tokio tasks (e.g., remote-search results landing) that affect the cached GNMT reply — but that's a connector-internal concern, not a vLLM contract concern. See §Concurrency.

## Concurrency

**vLLM-side:** single-threaded engine-core, as above.

**Connector-side:** the KVBM connector spawns its own tokio runtime and dispatches background work that may continue across engine-core ticks. Sources of concurrency *inside* the connector:

- `run_setup` spawn task (prefill side, started from `ensure_started`): async peer-resolve, factory.attach, drain commits+availability, manage chunked output, sit until lifecycle-watcher cancels or the session terminates cooperatively.
- Lifecycle watcher (spawned inside `run_setup`): polls the session's lifecycle stream; on `Detached`/`Failed`/watchdog-timeout, fires `cleanup_failed_request` and cancels the per-request token.
- `on_request_finished`'s observer-drain spawn task: waits on `observer.has_pending` then calls `session.finalize` and evicts `coord.states`.
- Offload pipeline's register-observer callback (kvbm-engine code): synchronously invokes the `commit_fn` closure when G1→G2 register events land, which routes to `commit_output_blocks` on the connector.
- Decode-side equivalents: `commit_gnmt_remote`, the local G2→G1 onboard kick, the remote pipeline driver.

These tasks run concurrently with each other AND with the next engine-core tick. **All concurrency hardening in this codebase belongs at this layer**, not at the vLLM API surface.

The 5 codex review iterations consolidated in commit `31fe4529245` are all examples of this kind of hardening:

| Iteration | Coordination |
|---|---|
| 1. `cleanup_claimed` CAS | Lifecycle watcher vs `run_setup` spawn-catch — both connector-spawned. |
| 2. Watcher cooperative-vs-failure discriminator | Lifecycle watcher (tokio) vs pre-USAA stash mutation (engine-core thread inside `cleanup_failed_request`). |
| 3. `on_request_finished` unconditional evict | Engine-core thread (`request_finished`) closing the leak window that lifecycle watcher used to cover. |
| 4. Defer `states.remove` + strong Arc across drain wait | Observer drain spawn task vs Arc-drop-induced eviction. |
| 4b. `inflight_dispatches` counter | Observer's `pending.remove` vs `commit_fn` dispatch return. |
| 5. Hold session lock across drained dispatch | `run_setup` attach drain dispatch vs `on_request_finished` finalize. |

All five address connector-internal tokio task races. **None** would be needed if vLLM were the only source of concurrency.

## Failure Surface

Failures the connector must handle, and how vLLM surfaces them:

| Failure | vLLM-visible surface | Connector duty |
|---|---|---|
| Connector deferred reply (GNMT → None) | `(None, _)` from GNMT | Re-evaluable on next tick; no state retained. |
| Async-load failure (recv failed) | Worker emits the request in `KVConnectorOutput.finished_recving` AND its failed blocks in `get_block_ids_with_load_errors()` no later than the same forward pass | `update_connector_output` routes the failure into the per-request state machine; eventual `request_finished` reaps state. |
| Mid-flight peer crash (CD-bound) | Worker may emit `finished_recving` with the rid (signalling vLLM to unblock) | Connector marks the G1 destinations failed via the worker-side `mark_failed_onboarding` callback so vLLM treats them as aborted; eventual `request_finished` releases state. |
| Abort before scheduling | `finish_requests([rid], FINISHED_ABORTED)` from outside the scheduler thread (via abort queue) → `request_finished` on engine-core thread | `request_finished` may arrive for an `UntrackedRequest`; must be idempotent. |
| Abort after scheduling but pre-completion | Same path as above | Same idempotence requirement; in-flight CD setup must tear down via per-request CancellationToken. |

What vLLM does **not** signal directly, and the connector must observe through its own channels (velo session lifecycle, hub heartbeat, watchdog):

- Peer instance crash (process exit).
- velo `Frame::Error` mid-stream.
- velo heartbeat loss → `LifecycleEvent::Detached` / `Failed`.
- Watchdog timeout on a wedged session.

These are detected by the lifecycle watcher and routed through `cleanup_failed_request`. They are NOT contract violations from vLLM — they are peer or transport failures.

## What This Contract Does NOT Promise

- That the connector will receive `request_finished` within any specific time bound after the model emits the final token.
- That `update_connector_output` callbacks arrive at a particular tick relative to `request_finished` for the same rid.
- That the same `num_computed_tokens` value is passed across repeated GNMT calls for the same rid.
- That `build_connector_meta`'s metadata is necessarily consumed by the worker side in the same tick (the executor may pipeline).
- Any behavior across the engine-core process boundary — a crashed engine-core leaves the connector's state with no cleanup hook from vLLM; cleanup is the operator's responsibility.

## Test Discipline (What This Contract Forbids in Tests)

These shapes are **forbidden** in the kvbm-connector test suite — they assert against scenarios vLLM cannot produce:

1. **Concurrent invocation of the same scheduler-side method for the same rid.** No `tokio::join!(gnmt, finish)`, no thread-spawn pairs that race two scheduler-side methods on one rid.
2. **Concurrent invocation of GNMT and `request_finished` for the same rid**, in either order, on multiple threads.
3. **Out-of-order lifecycle**: `request_finished` arriving before any GNMT for a rid that was actually scheduled (untracked-abort is allowed; mid-life arrival is not).
4. **Re-entrant scheduler calls**: `request_finished` calling back into `get_num_new_matched_tokens` synchronously, or any similar shape. vLLM does not do this.

These shapes ARE encouraged in tests:

1. **Repeated GNMT for the same rid with varying `num_computed_tokens`**, asserting idempotent reply.
2. **Engine-core-thread-then-tokio-task**: simulate a GNMT call, drive the runtime to advance the spawned `run_setup`, then call USAA (or `request_finished`) — assert the spawned task coexists cleanly.
3. **Tokio-task-vs-tokio-task races** within the connector: lifecycle watcher firing concurrently with `cleanup_failed_request` from the spawn-catch; observer drain racing `request_finished`'s spawn task. These are the real races.
4. **Peer-failure injection** via `MockSession`'s paired-mode: detach, Frame::Error, watchdog. Asserts the lifecycle watcher fires and `cleanup_failed_request` runs without leaks or double-notifications.
5. **`KVConnectorOutput` injection**: simulate the worker emitting finished_recving / finished_sending / failed-block-ids; assert `update_connector_output` routes them correctly.

## Maintenance

This document tracks the vLLM V1 connector API as of the version pinned by the dynamo workspace (see `lib/bindings/kvbm` Cargo / requirements). When vLLM's scheduler.py changes the call pattern, this file must be updated **before** the connector code is changed to match the new shape. Pull-request reviewers should reject scheduler-side hardening that does not cite the specific behavior in this contract that motivates it.

Last verified against vLLM `vllm/v1/core/sched/scheduler.py` and `vllm/v1/engine/core.py` on 2026-05-26.
