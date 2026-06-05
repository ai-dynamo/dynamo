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

### Cross-lifecycle stale-release race (recompute policy)

Under `kv_load_failure_policy=recompute`, vLLM reuses the same `request_id` when a CD-bound request fails and gets rescheduled (`vllm/v1/core/sched/scheduler.py:1943-1973`: `failed_recving_kv_req_ids` is keyed by rid and the `Request` object is retained, with `num_computed_tokens` truncated and the request requeued). The four `ConditionalDisaggCoordinator::release(rid)` call sites partition by thread of origin:

| # | Call site | Thread |
|---|---|---|
| 1 | `CdRequestStatePayload::Drop` (decode_leader.rs) | engine-core (fires from `process_finished_onboarding_take` via `update_connector_output`) |
| 2 | `decode_usaa` pre-USAA replay spawn (decode_leader.rs) | tokio runtime |
| 3 | `commit_usaa1` outer replay spawn (decode_leader.rs) | tokio runtime |
| 4 | `commit_usaa1` post-insert replay spawn (decode_leader.rs) | tokio runtime |

Sites #2–#4 all spawn a tokio task that awaits `worker_hook.mark_failed_onboarding` (potentially unbounded under back-pressure) and then calls `release_request` + `coordinator.release` by rid name. Under recompute, vLLM can reschedule the same rid while this await is parked; a fresh GNMT + USAA installs a new `Arc<CdRequestState>` and `Arc<CdRequest>` for the same rid. When the parked task eventually wakes and calls release-by-name, it wipes the new lifecycle's freshly-installed state and budget reservation.

**Fix.** Sites #2–#4 capture the per-lifecycle Arcs at spawn time and use `release_request_if_matches(rid, &captured_wrapper)` / `coordinator.release_if_matches(rid, &captured_coord)`. Both methods atomic-remove via `DashMap::remove_if` with `Arc::ptr_eq` against the captured snapshot; a mismatch (subsequent lifecycle replaced the entry) returns `false` without touching state.

Site #1 stays on `release` (by name) — `CdRequestStatePayload::Drop` is engine-core and serialized against `commit_usaa1` per the foundational invariant, so the cross-lifecycle window does not exist for it.

Discriminating test: `cd_decode_e2e::release_if_matches_enforces_arc_identity` — two distinct rids capture two handles, cross-handle release calls no-op, matching-handle release fires.

**Cleanup_failed_request paths are also guarded.** Both `ConditionalDisaggCoordinator::cleanup_failed_request` (`driver.rs`) and the wrapper-side `DecodeDisaggLeader::cleanup_failed_request` (`decode_leader.rs`) have the same shape as the spawn-replays — `mark_failed_onboarding.await` followed by state removal — so they get the same epoch-guard treatment:

- The coordinator already captures its own `state: Arc<CdRequest>` at the top of the method (for the `cleanup_claimed` CAS); the bottom-of-method `self.states.remove(rid)` was upgraded to `self.states.remove_if(rid, |_, v| Arc::ptr_eq(&state, v))`. A recompute reschedule that replaced the entry during the await makes the removal a no-op against the new lifecycle.
- The wrapper captures its own `Arc<CdRequestState>` at the top (this is also where the CAS runs against `cleanup_claimed`) and additionally captures the coordinator's `Arc<CdRequest>` via `state_for_decode` BEFORE the await. The bottom-of-method releases use `self.release_request_if_matches(rid, &captured_wrapper)` + `self.coordinator.release_if_matches(rid, &captured_coord)`. The pre-USAA branch's pending-failure stash also uses the captured Arc rather than a fresh `cd_request_state.get` — so the stash always lands on THIS lifecycle's state, even if a new lifecycle has been inserted (the new lifecycle starts with `pending_failure: None` and its own USAA-time consumer).

**`maybe_complete` eager release is also guarded.** The wrapper's `maybe_complete` runs as an `async fn` from inside the local-kick spawn and `run_remote_pipeline` spawn, with an `await` on `worker_hook.mark_onboarding_complete` before the eager release. Same shape as the cleanup paths above — captures the wrapper's `state` Arc (already captured at the top for the readiness check) and captures `captured_coord` via `state_for_decode` BEFORE the await; bottom-of-method uses `release_request_if_matches` + `coordinator.release_if_matches`. The canonical cleanup is still the `CdRequestStatePayload::Drop` triggered by `process_finished_onboarding_take`; this method just opens the budget early.

The full enumeration: 3 spawn-replays + 2 cleanup_failed_request paths + 1 maybe_complete = 6 async-spawned sites, all identity-checked. By-name `coordinator.release` and `release_request` remain on the engine-core thread only (`CdRequestStatePayload::Drop` via `process_finished_onboarding_take`, `commit_gnmt_remote`'s pre-payload bailout, `request_finished`).

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

## Disagg-Internal Invariants

These invariants govern how the disagg path turns a vLLM GNMT call into a CD dispatch. They are not vLLM-facing — vLLM does not see them — but the connector code MUST hold them, and future refactors that weaken them will silently corrupt the protocol.

### Invariant A — All-or-nothing prefix advertisement

**Statement.** For the prefix window `[0, num_computed_tokens / block_size)`, decode advertises EITHER the full set of prefix hashes on the session OR no prefix at all. Partial advertisement is forbidden.

The full set is constructed from three sources, tried in order:

- **G2-resident prefix blocks** (fast path) — returned by `ConnectorLeader::find_prefix_g2_blocks`. These are immediately made-available on the session at GNMT.
- **G3-resident prefix blocks** (Stage 2 promotion path) — when the G2 query misses, the connector first asks `find_prefix_g3_hashes` whether decode's NVMe tier holds the prefix. If yes, the promotion plan's `source_tier` is `G3` and the actual G3→G2 stage is fired as an uncancellable task at USAA via `promote_g3_to_g2`.
- **G1-only prefix blocks** (Stage 1 promotion path) — when both G2 and G3 miss, the fallback plan's `source_tier` is `G1` and the G1→G2 transfer is fired at USAA via `promote_g1_to_g2`.

In all three cases, the GNMT-time `session.commit` already includes the full planned-prefix hashes so `finish_commits` seals the planned set up front. The promoted G2 blocks (from either tier) are made-available on the session as they land; `session.finish_availability` is deferred until the promotion task completes.

**Enforced by.**

- `ConnectorLeader::find_prefix_g2_blocks` (`lib/kvbm-connector/src/connector/leader/mod.rs`). On any G2 miss the function drops the partial hits (RAII returns them to G2's inactive pool) and returns `Ok(Vec::new())`. Emits `prefix_g2_incomplete_skip`.
- `ConnectorLeaderShim::find_prefix_g3_hashes` (`lib/kvbm-connector/src/connector/leader/p2p/transport.rs`). On any G3 miss (or absent NVMe tier) returns `Ok(Vec::new())`. Emits `prefix_g3_incomplete_skip` on partial-G3 hit so post-mortem triage can distinguish "G3 doesn't have it" from "G3 has it but evicted partway through the scan."
- `DecodeDisaggLeader::commit_gnmt_remote` (`decode_leader.rs`). When `find_prefix_g2_blocks` returns empty AND `num_computed_tokens > 0`, tries `find_prefix_g3_hashes` first; builds `PendingTierPromotion { source_tier, prefix_block_count, prefix_hashes }` with the appropriate tier discriminant. Plumbs the planned hashes to `begin_remote_prefill` and stashes the plan on `CdRequestState`.
- `DecodeDisaggLeader::commit_usaa1` (`decode_leader.rs`). Matches on `plan.source_tier`:
  - `G1`: pairs `block_ids[..prefix_block_count]` with the GNMT-time `prefix_hashes` to build `Vec<ExternalBlock<G1>>` and calls `inner.promote_g1_to_g2(source_blocks)`.
  - `G3`: calls `inner.promote_g3_to_g2(prefix_hashes)` directly — no vLLM block_ids needed (the source blocks live in decode's own G3 manager; the shim re-matches G3 + stages via `kvbm_engine::leader::stage_g3_to_g2`).

  Either arm spawns a task that awaits the resulting future and drives `session.make_available` + `session.finish_availability` on the promoted G2 blocks; the task body downstream is identical for both tiers.
- `ConditionalDisaggCoordinator::begin_remote_prefill` (`coordinator/driver.rs`). Includes `pending_promotion_hashes` in the up-front `session.commit` (positionally first, ahead of `local_match`); calls `session.finish_commits` unconditionally; skips `session.finish_availability` when promotion is pending.

**Consumed by.** Decode's `commit_gnmt_remote` treats an empty `find_prefix_g2_blocks` result as "try G3, then plan G1 promotion" rather than "advertise nothing." Prefill's `ensure_started` pulls the full `[0, DNPT/BS)` hash range regardless of whether decode sourced each block from G2 directly, from G3 via stage, or from G1 via offload; all three look identical on the session.

**Why partial is forbidden.** Publishing `[0..M)` (the leading-contiguous G2-resident portion) to the session tells prefill "decode has prefix `[0..M)` available, not `[M..P)`." But decode's G1 holds the FULL prefix `[0..P)` — the "missing" `[M..P)` is a hole only in G2, not in the conversation state. Advertising the partial set creates an inconsistent view (prefill treats `[M..P)` as cache misses while decode actually has them in G1).

**Promotion failure handling.** Both `promote_g1_to_g2` Err (synchronous enqueue Err, `TransferHandle::wait` Err, or partial register after transfer) and `promote_g3_to_g2` Err (G3 evicted between GNMT and USAA, no parallel worker configured, `stage_g3_to_g2` failure) route the task to `session.close(reason)`. The reason string carries a tier prefix (`"g1->g2 promotion: …"` or `"g3->g2 promotion: …"`). The prefill peer's lifecycle watcher observes `Detached`/`Failed` and runs `cleanup_failed_request` through the standard CD failure path; vLLM is notified via `mark_failed_onboarding`. Decode's G2/G3 cache is unaffected — only the in-flight transfer is lost.

**G3 eviction window.** `find_prefix_g3_hashes` returns hashes (not pinned `ImmutableBlock<G3>` entries). Between the GNMT-time call and the USAA-time `promote_g3_to_g2`, the matched G3 blocks could be evicted under cache pressure. `promote_g3_to_g2` re-matches at USAA and Errs on partial match; that Err routes through the standard failure handling above. The eviction window is microseconds under the V1 scheduler (GNMT and USAA are back-to-back on the engine-core thread).

**Promotion-task lifetime.** The promotion `JoinHandle` lives on `DecodeBits.pending_promotion_task` next to the session. Dropping it does NOT abort the task (tokio's `JoinHandle` detaches on drop). The task survives request teardown — the resulting G2 blocks remain registered in the cache and benefit future requests, regardless of whether the source was G1 or G3.

**Audit events.** Tier-specific planning + enqueue events, tier-agnostic terminal events:
- `prefix_g1_to_g2_promotion_planned` / `prefix_g3_to_g2_promotion_planned` (GNMT)
- `prefix_g1_to_g2_promotion_enqueued` / `prefix_g3_to_g2_promotion_enqueued` (USAA)
- `prefix_g2_promotion_landed` (task completion; same for both tiers)
- `prefix_g2_promotion_failed` (task error; carries `source_tier` field for triage)

### Invariant B — Sequential-left-to-right match terminating at first miss

**Statement.** The match for both prefix and external (local-match) windows is computed by walking the request's canonical PLH chain in absolute-position order (`all_sequence_hashes[i]` for ascending `i`) and stopping at the first hash that is not present in G2. The returned matched window is therefore a contiguous prefix of the requested range. Partial matches that skip a hole in the middle are never produced.

**Enforced by.** `BlockManager::match_blocks` (in kvbm-logical) walks ordered slices and returns hits in input order, terminating at the first miss. `OnboardingState.shards` (`connector/leader/slot.rs:348+`) and `matched_span()` (`slot.rs:190+`) walk shards contiguously, mask-prefix by `num_computed_tokens / block_size`, and stop at the first gap.

**Consumed by.** `commit_gnmt_remote` (decode-side, builds `RemotePrefillRequest` from a CONTIGUOUS local-match window) and `ensure_started` (prefill-side, walks the same PLH chain from `[0, DNPT/BS)`). Both rely on the contiguity to align the `decode_offset_blocks + i` placement contract.

**Why sequential-only.** The placement contract on remote pull is positional: prefill places block i at absolute index `decode_offset_blocks + i`. If decode's matched set had a hole at position k, decode would advertise positions `{0, 1, ..., k-1, k+1, ...}` — but prefill's positional placement assumes contiguity from `decode_offset_blocks`. A hole at k would map prefill's i=k to decode's position k+1, mis-placing every subsequent block.

**Defense-in-depth.** The DNPT digest (`expected_hash_digest` on `RemotePrefillRequest`) covers the FULL `[0, DNPT/BS)` slice and is verified by prefill in `ensure_started` (`driver.rs:1651+`). A hash-chain divergence — including one introduced by a contiguity-violation refactor — fails loud at GNMT-handshake time rather than producing a wrong-block RDMA pull. Pinned by `cd_prefill_dnpt_digest_mismatch_rejected`.

### Multi-delta availability hardening (prefill side)

The session API permits availability to land in multiple `Available` deltas — CONTRACT (`lib/kvbm-engine/src/p2p/session/CONTRACT.md` §2.7/§2.8) specifies that delta order on the wire equals the holder's `make_available` call order, with no guarantee that a single delta covers a contiguous position range. Stage 1 is the first documented holder that splits availability (decode publishes local-match synchronously at GNMT and the promoted prefix later from the task); peers may legitimately do this in other future scenarios too.

**Prefill consumer hardening.** `ConditionalDisaggCoordinator::kick_onboard` (`coordinator/driver.rs`) builds a `SequenceHash → position` map from `bits.expected_hashes` and sorts the registered G2 blocks by absolute position before taking the suffix paired with vLLM's positional G1 destinations. Without this, an arrival-order suffix would mis-pair the G2 sources under any multi-delta arrival — corrupting the K/V cache silently. Pinned by `cd_prefill_kick_onboard_robust_to_split_delta_availability`.

## Maintenance

This document tracks the vLLM V1 connector API as of the version pinned by the dynamo workspace (see `lib/bindings/kvbm` Cargo / requirements). When vLLM's scheduler.py changes the call pattern, this file must be updated **before** the connector code is changed to match the new shape. Pull-request reviewers should reject scheduler-side hardening that does not cite the specific behavior in this contract that motivates it.

Last verified against vLLM `vllm/v1/core/sched/scheduler.py` and `vllm/v1/engine/core.py` on 2026-05-26.
