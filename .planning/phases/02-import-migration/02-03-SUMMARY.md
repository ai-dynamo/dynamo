---
phase: 02-import-migration
plan: 03
subsystem: infra
tags: [rust, imports, kvbm-connector, velo, nova, migration]

# Dependency graph
requires:
  - phase: 02-import-migration
    plan: 02
    provides: crate::v2::* and crate::integrations::* namespaces cleared; tests.rs gated; kvbm_config substituted
provides:
  - connector/worker/nova/ renamed to connector/worker/velo/
  - All dynamo_nova and dynamo_nova_backend imports eliminated from lib/kvbm-connector/src/
  - All runtime.nova/.nova() field/method accesses replaced with runtime.messenger()
  - InstanceLeader::builder().nova() replaced with .messenger()
  - velo::Event replaces dynamo_nova::events::LocalEvent in scheduler
  - velo::EventHandle replaces dynamo_nova::events::EventHandle in state/scheduler
  - cargo check -p kvbm-connector passes with zero errors
  - cargo check --workspace passes with zero errors
affects: [compile-baseline, phase-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "runtime.messenger() replaces runtime.nova() — KvbmRuntime accessor for Arc<Messenger>"
    - "runtime.event_system() replaces runtime.nova.events().local().clone() — Arc<velo::EventManager>"
    - "runtime.tokio() replaces runtime.nova.runtime().clone() — tokio Handle"
    - "Handler::typed_unary_async replaces NovaHandler::typed_unary_async"
    - "Handler::unary_handler_async replaces NovaHandler::unary_handler_async"
    - "velo::Event replaces dynamo_nova::events::LocalEvent"
    - "event_manager.merge_events() for multi-event merges (VeloEvents lacks merge_events; use Messenger::event_manager())"
    - "Event::trigger(self) consumes — store as Event not Arc<Event> when trigger is needed"

key-files:
  created: []
  modified:
    - lib/kvbm-connector/src/connector/worker/velo/client.rs
    - lib/kvbm-connector/src/connector/worker/velo/service.rs
    - lib/kvbm-connector/src/connector/worker/velo/mod.rs
    - lib/kvbm-connector/src/connector/worker/velo/protocol.rs
    - lib/kvbm-connector/src/connector/worker/mod.rs
    - lib/kvbm-connector/src/connector/worker/state.rs
    - lib/kvbm-connector/src/connector/worker/init/pending.rs
    - lib/kvbm-connector/src/connector/leader/init.rs
    - lib/kvbm-connector/src/connector/leader/scheduler.rs
    - lib/kvbm-connector/src/connector/leader/control.rs
    - lib/kvbm-connector/src/connector/leader/onboard.rs

key-decisions:
  - "VeloEvents (from Messenger::events()) lacks merge_events — use Messenger::event_manager() which returns EventManager with merge_events(&self, ...) and awaiter(&self, ...)"
  - "velo::Event::trigger(self) consumes ownership — cannot call via Arc<Event>; store events as plain Event, not Arc<Event>"
  - "execute_local_transfer on InstanceLeader is pub(crate) — use instance_leader.parallel_worker().execute_local_transfer() instead"
  - "execute_local_layerwise_onboard on PhysicalWorker requires #[cfg(feature = nccl)] — gate call in kvbm-connector with same flag"

patterns-established:
  - "Velo API: Handler::typed_unary_async(name, |ctx| async { let v: T = ctx.input; ... })"
  - "Velo API: Handler::unary_handler_async(name, |_ctx| async { Ok(Some(Bytes::from(...))) })"
  - "Event vs EventHandle pattern: new_event() → Event, Event::handle() → EventHandle for serializing across process boundary"

requirements-completed: [VELO-01, VELO-02, VELO-03, VELO-04, VELO-05]

# Metrics
duration: 28min
completed: 2026-03-11
---

# Phase 2 Plan 3: Nova→Velo Transport Sweep Summary

**nova/worker directory renamed to velo/, all dynamo_nova/dynamo_nova_backend types replaced with velo equivalents, runtime.nova/.nova() eliminated; cargo check --workspace passes with zero errors**

## Performance

- **Duration:** ~28 min
- **Started:** 2026-03-11T10:10:00Z
- **Completed:** 2026-03-11T10:38:00Z
- **Tasks:** 3 (Tasks 1+3 inspection/verification; Task 2 implementation committed)
- **Files modified:** 11

## Accomplishments

- Eliminated all `dynamo_nova`, `dynamo_nova_backend` imports — zero active matches in lib/kvbm-connector/src/
- Renamed `connector/worker/nova/` → `connector/worker/velo/` via `git mv`
- Replaced `NovaHandler::typed_unary_async` → `Handler::typed_unary_async` and `NovaHandler::unary_handler_async` → `Handler::unary_handler_async` throughout
- Replaced all `runtime.nova` / `runtime.nova()` → `runtime.messenger()` across 6 files
- Updated `InstanceLeader::builder().nova()` → `.messenger()` in leader/init.rs
- Replaced `dynamo_nova_backend::{PeerInfo, WorkerAddress}` → `velo::{PeerInfo, WorkerAddress}`
- Replaced `dynamo_nova::events::{EventHandle, LocalEvent}` → `velo::{EventHandle, Event}`
- cargo check -p kvbm-connector: zero errors
- cargo check --workspace: zero errors (no regressions)

## Task Commits

1. **Task 1: Pre-pass inspection** - No commit (inspection only)
2. **Task 2: Pass 6 — nova→velo sweep** - `6948648db` (fix)
3. **Task 3: Phase 2 gate verification** - No commit (all gates passed)

## Files Created/Modified

- `lib/kvbm-connector/src/connector/worker/velo/client.rs` — dynamo_nova::Nova → velo::Messenger; TypedUnaryResult → velo::TypedUnaryResult; field renamed nova → messenger
- `lib/kvbm-connector/src/connector/worker/velo/service.rs` — NovaHandler → Handler; Nova → Messenger; nova.register_handler → messenger.register_handler
- `lib/kvbm-connector/src/connector/worker/velo/mod.rs` — no nova imports (only protocol message types and string constants)
- `lib/kvbm-connector/src/connector/worker/velo/protocol.rs` — no nova imports (only serde types)
- `lib/kvbm-connector/src/connector/worker/mod.rs` — mod nova → mod velo; runtime.nova.clone() → runtime.messenger().clone(); velo::service::init; nova.instance_id() → messenger().instance_id(); execute_local_layerwise_onboard gated with #[cfg(feature = "nccl")]
- `lib/kvbm-connector/src/connector/worker/state.rs` — ForwardPassNovaEvent type alias → velo::EventHandle; VeloWorkerService::new(runtime.nova.clone()) → VeloWorkerService::new(runtime.messenger().clone())
- `lib/kvbm-connector/src/connector/worker/init/pending.rs` — runtime.nova.events().local() → runtime.event_system(); runtime.nova.runtime() → runtime.tokio(); runtime.nova.instance_id() → runtime.messenger().instance_id()
- `lib/kvbm-connector/src/connector/leader/init.rs` — dynamo_nova_backend → velo; runtime.nova → runtime.messenger(); InstanceLeader builder .nova() → .messenger()
- `lib/kvbm-connector/src/connector/leader/scheduler.rs` — dynamo_nova::events::EventHandle → velo::EventHandle; dynamo_nova::events::LocalEvent → velo::Event; runtime.nova() → runtime.messenger(); merge/awaiter via event_manager; forward_pass_promise: Event (not Arc<Event>)
- `lib/kvbm-connector/src/connector/leader/control.rs` — nova() → messenger() in discover_and_register_peer call
- `lib/kvbm-connector/src/connector/leader/onboard.rs` — instance_leader.execute_local_transfer (pub(crate)) → parallel_worker().execute_local_transfer (pub via WorkerTransfers trait); g2/g1_block_ids wrapped in Arc::from()

## Decisions Made

- `VeloEvents` (returned by `Messenger::events()`) does not expose `merge_events` — only `EventManager` (from `velo_events`) does. Solution: use `messenger.event_manager()` in the cleanup task to get an `EventManager` with `merge_events(&self)` and `awaiter(&self)`.
- `velo::Event::trigger(self)` takes ownership (not `&self`), so `Arc<Event>` cannot be used when trigger is needed. Changed `forward_pass_promise` from `Arc<velo::Event>` to `velo::Event` — `handle()` still works via `&self` before the event is moved into the closure.
- `InstanceLeader::execute_local_transfer` is `pub(crate)` — inaccessible from `kvbm-connector`. Fixed by routing through `instance_leader.parallel_worker()` which is `pub` and implements `WorkerTransfers` (which has `pub execute_local_transfer`).
- `PhysicalWorker::execute_local_layerwise_onboard` is `#[cfg(feature = "nccl")]` — gated the call in kvbm-connector with the same flag. Non-nccl builds log a warning and skip.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] VeloEvents lacks merge_events — used EventManager instead**
- **Found during:** Task 2 (cargo check after scheduler.rs changes)
- **Issue:** Plan mapped `nova.events().merge_events()` to `messenger.events().merge_events()`, but `VeloEvents` has no `merge_events` method. Only `EventManager` (from `velo_events`) has it.
- **Fix:** Changed `spawn_forward_pass_cleanup_task` to capture `messenger.event_manager()` (returns owned `EventManager`) instead of `messenger.clone()`. Both `merge_events(&self)` and `awaiter(&self)` work on `EventManager`.
- **Files modified:** lib/kvbm-connector/src/connector/leader/scheduler.rs
- **Committed in:** `6948648db`

**2. [Rule 1 - Bug] Event::trigger(self) ownership — removed Arc wrapping**
- **Found during:** Task 2 (cargo check after scheduler.rs changes)
- **Issue:** `dynamo_nova::events::LocalEvent::trigger()` was `&self` (non-consuming); `velo::Event::trigger(self)` consumes the receiver. Wrapping in `Arc<Event>` prevents calling `trigger()`.
- **Fix:** Changed `forward_pass_promise` from `Arc<velo::Event>` to `velo::Event` throughout. `Event::handle(&self)` still works before the event is moved into the async closure. Updated `spawn_forward_pass_cleanup_task` signature accordingly.
- **Files modified:** lib/kvbm-connector/src/connector/leader/scheduler.rs
- **Committed in:** `6948648db`

**3. [Rule 1 - Bug] InstanceLeader::execute_local_transfer is pub(crate)**
- **Found during:** Task 2 (cargo check after leader changes)
- **Issue:** `instance_leader.execute_local_transfer()` was marked `pub(crate)` on `InstanceLeader` — inaccessible from external crate. This was a pre-existing issue deferred from Plan 02.
- **Fix:** Routed through `instance_leader.parallel_worker()` (which is `pub`) and called `execute_local_transfer` on the returned `Arc<dyn ParallelWorkers>` (which implements the public `WorkerTransfers` trait). Block IDs wrapped in `Arc::from()` to match the trait signature.
- **Files modified:** lib/kvbm-connector/src/connector/leader/onboard.rs
- **Committed in:** `6948648db`

**4. [Rule 1 - Bug] execute_local_layerwise_onboard requires nccl feature**
- **Found during:** Task 2 (cargo check after worker/mod.rs changes)
- **Issue:** `PhysicalWorker::execute_local_layerwise_onboard` only exists under `#[cfg(feature = "nccl")]`. kvbm-connector has no nccl feature. This was a pre-existing issue deferred from Plan 02.
- **Fix:** Gated the call with `#[cfg(feature = "nccl")]`; added `#[cfg(not(feature = "nccl"))]` branch that logs a warning and skips the layerwise onboard.
- **Files modified:** lib/kvbm-connector/src/connector/worker/mod.rs
- **Committed in:** `6948648db`

---

**Total deviations:** 4 auto-fixed (all Rule 1 — API differences between dynamo_nova and velo)
**Impact on plan:** All fixes necessary for correct compilation. VeloEvents API differs from dynamo_nova in key ways (no merge_events, consuming trigger). One pub(crate) visibility issue resolved via public trait route.

## Issues Encountered

None beyond the API differences documented in deviations above.

## Next Phase Readiness

- Phase 2 import migration complete: all namespaces cleared
  - crate::v2::*, crate::integrations::* — zero active matches (Plan 02)
  - dynamo_nova, dynamo_nova_backend — zero active matches (Plan 03)
- cargo check --workspace passes with zero errors
- kvbm-connector is compile-clean against current workspace
- Phase 3 (if any) can proceed against a fully compile-clean baseline

---
*Phase: 02-import-migration*
*Completed: 2026-03-11*
