---
phase: 04-test-porting
plan: 02
subsystem: testing
tags: [rust, testing, import-migration, velo, kvbm-connector]

# Dependency graph
requires:
  - phase: 04-test-porting
    plan: 01
    provides: testing/mod.rs skeleton with commented-out submodule stubs
provides:
  - lib/kvbm-connector/src/testing/connector.rs with ConnectorTestConfig, TestConnectorInstance, TestConnectorCluster, TestConnectorWorker, MockTensor
  - crate::testing::{ConnectorTestConfig, TestConnectorInstance, TestConnectorCluster, TestConnectorWorker} re-exported
  - cargo check -p kvbm-connector --features testing passing after connector.rs ported
affects: [04-03]

# Tech tracking
tech-stack:
  added: []
  patterns: [nova→velo import migration pattern, with_nova→with_messenger builder API]

key-files:
  created:
    - lib/kvbm-connector/src/testing/connector.rs
  modified:
    - lib/kvbm-connector/src/testing/mod.rs

key-decisions:
  - "super::{managers, nova, physical, token_blocks} replaced with kvbm_engine::testing::{managers, messenger, physical, token_blocks} as individual use items"
  - "leader_nova field kept as Arc<Messenger> (renamed from Arc<Nova>) — same role, different type name"
  - "velo::WorkerAddress imported directly (not via kvbm_engine re-export) to match struct field type"

patterns-established:
  - "nova::create_nova_instance_tcp() → messenger::create_messenger_tcp() from kvbm_engine::testing::messenger"
  - "KvbmRuntime::builder().with_nova() → .with_messenger() API"

requirements-completed: [TEST-02]

# Metrics
duration: 10min
completed: 2026-03-11
---

# Phase 4 Plan 02: Port testing/connector.rs Summary

**testing/connector.rs ported from kvbm-next with all dynamo_nova/dynamo_kvbm_config/crate::v2 imports migrated to workspace paths; pub mod connector activated in testing/mod.rs; cargo check --features testing passes zero errors**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-03-11T18:45:00Z
- **Completed:** 2026-03-11T18:54:45Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `lib/kvbm-connector/src/testing/connector.rs` created (1,756 lines → ported to workspace paths)
- All `dynamo_kvbm_config::`, `dynamo_nova::`, `dynamo_nova_backend::` imports eliminated
- All `crate::v2::*` paths replaced with workspace equivalents (`kvbm_engine`, `kvbm_physical`, `kvbm_common`, `crate::`)
- `Nova` → `Messenger` API migration: `create_nova_instance_tcp()` → `create_messenger_tcp()`, `with_nova()` → `with_messenger()`
- `super::{managers, nova, physical, token_blocks}` replaced with `kvbm_engine::testing::{managers, messenger, physical, token_blocks}` top-level imports
- Inline `crate::v2::distributed::worker::WorkerTransfers`, `crate::v2::logical::LogicalLayoutHandle`, `crate::v2::physical::transfer::TransferOptions` inside async methods migrated to `kvbm_engine::worker::WorkerTransfers`, `kvbm_common::LogicalLayoutHandle`, `kvbm_physical::TransferOptions`
- `testing/mod.rs` updated: `pub mod connector` uncommented, four key types re-exported
- `cargo check -p kvbm-connector --features testing` exits zero — TEST-02 satisfied

## Task Commits

1. **Task 1: Port testing/connector.rs with import migration** — `de21dff6f` (feat)
2. **Task 2: Activate pub mod connector in testing/mod.rs** — `c68fc64e5` (feat)

## Files Created/Modified

- `lib/kvbm-connector/src/testing/connector.rs` — New file: 1,756-line port with all imports migrated
- `lib/kvbm-connector/src/testing/mod.rs` — Uncommented `pub mod connector` and re-exports

## Decisions Made

- `super::{managers, nova, physical, token_blocks}` was a single grouped import in the source. In the workspace these come from `kvbm_engine::testing::*` submodules so each was imported individually at the top level. `nova` module → `messenger` module.
- `leader_nova` field kept as `Arc<Messenger>` (same name `leader_nova` preserved for minimal diff, type is `Arc<Messenger>` not `Arc<Nova>`)
- `velo::WorkerAddress` imported directly since `TestConnectorWorker.worker_address` is a `WorkerAddress` field, not accessed via `kvbm_engine` re-export

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None. First `cargo check` after writing the file passed with zero errors.

## User Setup Required

None.

## Next Phase Readiness

- `crate::testing::connector::ConnectorTestConfig` and three companion types are now accessible
- `worker/tests.rs` (pre-staged from Phase 2) can now compile since `ConnectorTestConfig::new()` and `TestConnectorInstance::builder()` exist
- Plan 03 can now create `src/testing/e2e/` and `src/testing/scheduler/` and uncomment their declarations in `mod.rs`

---
*Phase: 04-test-porting*
*Completed: 2026-03-11*
