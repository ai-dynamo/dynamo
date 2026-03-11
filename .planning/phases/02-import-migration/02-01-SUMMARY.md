---
phase: 02-import-migration
plan: 01
subsystem: infra
tags: [rust, imports, kvbm-connector, kvbm-logical, kvbm-physical, kvbm-engine, migration]

# Dependency graph
requires:
  - phase: 01-workspace-wiring
    provides: kvbm-connector Cargo.toml wired with kvbm-logical, kvbm-physical, kvbm-engine deps
provides:
  - No crate::logical::* imports remain in lib/kvbm-connector/src/
  - No crate::physical::* imports remain in lib/kvbm-connector/src/
  - No crate::distributed::* imports remain in lib/kvbm-connector/src/
  - NovaWorkerClient renamed to VeloWorkerClient in all leader files
  - NovaWorkerService renamed to VeloWorkerService in all worker files
affects: [02-import-migration-02, compile-baseline]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "kvbm_logical:: for blocks/manager (BlockRegistry, BlockManager, FrequencyTrackingCapacity)"
    - "kvbm_common:: for LogicalLayoutHandle (re-exported by kvbm_logical but lives in kvbm_common)"
    - "kvbm_physical:: for TransferOptions (root re-export), layout::*, transfer::context::TokioRuntime"
    - "kvbm_engine::leader:: for InstanceLeader, FindMatchesOptions, FindMatchesResult, Leader, OnboardingStatus, StagingMode, ReadyResult"
    - "kvbm_engine::worker:: for VeloWorkerClient, VeloWorkerService, LeaderLayoutConfig, WorkerLayoutResponse, DirectWorker, WorkerTransfers, SerializedLayout, Worker"
    - "kvbm_engine::offload:: for TransferHandle, TransferStatus, ExternalBlock, OffloadEngine, PendingTracker, PipelineBuilder, ObjectPipelineBuilder, ObjectPresenceFilter, S3PresenceChecker, create_policy_from_config"
    - "kvbm_engine::object:: for ObjectLockManager, create_lock_manager, create_object_client, ObjectBlockOps"

key-files:
  created: []
  modified:
    - lib/kvbm-connector/src/connector/leader/init.rs
    - lib/kvbm-connector/src/connector/leader/mod.rs
    - lib/kvbm-connector/src/connector/leader/slot.rs
    - lib/kvbm-connector/src/connector/leader/finish.rs
    - lib/kvbm-connector/src/connector/leader/scheduler.rs
    - lib/kvbm-connector/src/connector/leader/onboard.rs
    - lib/kvbm-connector/src/connector/worker/mod.rs
    - lib/kvbm-connector/src/connector/worker/nova/client.rs
    - lib/kvbm-connector/src/connector/worker/nova/service.rs
    - lib/kvbm-connector/src/connector/worker/state.rs
    - lib/kvbm-connector/src/connector/worker/init/pending.rs
    - lib/kvbm-connector/src/vllm/layout.rs

key-decisions:
  - "kvbm_common::LogicalLayoutHandle (not kvbm_logical::) is the correct path — LogicalLayoutHandle lives in kvbm_common; kvbm_logical just re-exports it"
  - "kvbm_physical::transfer::context::TokioRuntime path confirmed via grep before applying"
  - "When crate::{} blocks mixed logical+physical+v2 imports, they are split into separate use statements per namespace pass"
  - "crate::v2::distributed::* maps to kvbm_engine:: same as crate::distributed::*"

patterns-established:
  - "Import migration: fix one namespace per pass, cargo check gate between passes"
  - "Mixed import blocks split by extracting target namespace items to separate use statements"

requirements-completed: [IMP-02, IMP-03, IMP-04]

# Metrics
duration: 25min
completed: 2026-03-11
---

# Phase 2 Plan 1: Import Migration Pass 1-3 Summary

**crate::logical/physical/distributed import namespaces replaced with kvbm_logical, kvbm_physical, and kvbm_engine across 12 files; NovaWorkerClient and NovaWorkerService renamed to VeloWorkerClient/VeloWorkerService**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-03-11T09:28:00Z
- **Completed:** 2026-03-11T09:53:43Z
- **Tasks:** 3
- **Files modified:** 12

## Accomplishments

- Eliminated all `crate::logical::*` imports (3 files), replacing with `kvbm_logical::` and `kvbm_common::` paths
- Eliminated all `crate::physical::*` imports (6 files), replacing with `kvbm_physical::` paths
- Eliminated all `crate::distributed::*` and `crate::v2::distributed::*` imports (10 files), replacing with `kvbm_engine::*` paths
- Renamed `NovaWorkerClient` → `VeloWorkerClient` and `NovaWorkerService` → `VeloWorkerService` in all affected files
- All three passes gated with `cargo check -p kvbm-connector`; remaining errors only from `v2::`, `integrations::`, and `dynamo_nova_backend` namespaces (future passes)

## Task Commits

1. **Task 1: Pass 1 — Replace crate::logical::*** - `9f2c1f7fe` (fix)
2. **Task 2: Pass 2 — Replace crate::physical::*** - `d54097116` (fix)
3. **Task 3: Pass 3 — Replace crate::distributed::*** - `ca8e49519` (fix)

## Files Created/Modified

- `lib/kvbm-connector/src/connector/leader/init.rs` — logical/distributed imports replaced; VeloWorkerClient rename
- `lib/kvbm-connector/src/connector/leader/mod.rs` — distributed imports replaced; VeloWorkerClient rename in WorkerClients struct
- `lib/kvbm-connector/src/connector/leader/slot.rs` — distributed imports replaced (incl. test module)
- `lib/kvbm-connector/src/connector/leader/finish.rs` — distributed offload imports replaced
- `lib/kvbm-connector/src/connector/leader/scheduler.rs` — distributed offload ExternalBlock import replaced
- `lib/kvbm-connector/src/connector/leader/onboard.rs` — logical + physical imports replaced (not in plan's file list but had both namespaces)
- `lib/kvbm-connector/src/connector/worker/mod.rs` — logical, physical, distributed imports replaced
- `lib/kvbm-connector/src/connector/worker/nova/client.rs` — physical, distributed imports replaced
- `lib/kvbm-connector/src/connector/worker/nova/service.rs` — distributed import replaced
- `lib/kvbm-connector/src/connector/worker/state.rs` — physical, distributed imports replaced; VeloWorkerService rename
- `lib/kvbm-connector/src/connector/worker/init/pending.rs` — all three namespaces replaced; split crate::{} block
- `lib/kvbm-connector/src/vllm/layout.rs` — physical import replaced

## Decisions Made

- `kvbm_common::LogicalLayoutHandle` confirmed as the correct path (not `kvbm_logical::`) — LogicalLayoutHandle lives in kvbm_common; kvbm_logical re-exports it
- Mixed `crate::{ namespace1::X, namespace2::Y }` blocks are split into separate `use` statements rather than trying to partially rewrite the block
- `crate::v2::distributed::*` maps identically to `kvbm_engine::*` as `crate::distributed::*` does

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed crate::logical::LogicalLayoutHandle in pending.rs (not in plan's file list for Task 1)**
- **Found during:** Task 1 (Pass 1 verification)
- **Issue:** pending.rs had `logical::LogicalLayoutHandle` inside a mixed `crate::{}` import block; cargo check revealed it
- **Fix:** Split the crate::{} block and replaced with `kvbm_common::LogicalLayoutHandle`
- **Files modified:** lib/kvbm-connector/src/connector/worker/init/pending.rs
- **Verification:** `cargo check -p kvbm-connector 2>&1 | grep "^error" | grep -v "distributed|physical|v2|..."` — no logical errors
- **Committed in:** `9f2c1f7fe` (Task 1 commit)

**2. [Rule 1 - Bug] Fixed crate::logical and crate::physical imports in leader/onboard.rs (not in plan's file list)**
- **Found during:** Task 2 (Pass 2 verification)
- **Issue:** `connector/leader/onboard.rs` had both `crate::logical::LogicalLayoutHandle` and `crate::physical::TransferOptions` in same import; cargo check revealed it
- **Fix:** Replaced both with `kvbm_common::LogicalLayoutHandle` and `kvbm_physical::TransferOptions`
- **Files modified:** lib/kvbm-connector/src/connector/leader/onboard.rs
- **Verification:** `cargo check -p kvbm-connector 2>&1 | grep "^error" | grep "physical"` — no physical errors
- **Committed in:** `d54097116` (Task 2 commit)

**3. [Rule 1 - Bug] Fixed crate::physical::layout::LayoutConfig in worker/state.rs (not in plan's explicit file list for Task 2)**
- **Found during:** Task 2 (Pass 2 verification)
- **Issue:** `connector/worker/state.rs` had `physical::layout::LayoutConfig` inside `crate::{}` block; cargo check revealed it
- **Fix:** Split block, replaced with `kvbm_physical::layout::LayoutConfig`
- **Files modified:** lib/kvbm-connector/src/connector/worker/state.rs
- **Verification:** No physical errors in cargo check
- **Committed in:** `d54097116` (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (all Rule 1 — bugs from incomplete file discovery in plan)
**Impact on plan:** All fixes were necessary to clear the named namespaces per-pass. No scope creep — all fixes are directly in the target namespace for the respective pass.

## Issues Encountered

None — the deviations above were additional files containing the same namespace patterns as planned files. All were handled inline.

## Next Phase Readiness

- Namespaces cleared: `crate::logical::*`, `crate::physical::*`, `crate::distributed::*` — zero matches
- Remaining broken namespaces in kvbm-connector: `crate::v2::*`, `crate::integrations::*`, `dynamo_nova_backend`
- Plan 02 (the more complex passes) can proceed against a cleaner baseline

---
*Phase: 02-import-migration*
*Completed: 2026-03-11*
