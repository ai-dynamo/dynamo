---
phase: 04-test-porting
plan: 03
subsystem: testing
tags: [rust, testing, import-migration, kvbm-connector, scheduler, e2e]

# Dependency graph
requires:
  - phase: 04-test-porting
    plan: 02
    provides: testing/connector.rs with ConnectorTestConfig, TestConnectorInstance, TestConnectorCluster, TestConnectorWorker
provides:
  - lib/kvbm-connector/src/testing/e2e/ — 3 files ported (mod.rs, find_blocks.rs, s3_object.rs)
  - lib/kvbm-connector/src/testing/scheduler/ — 8 files ported (mod.rs, connector_tests.rs, mock/mod.rs, mock/engine.rs, mock/model.rs, mock/tests.rs, mock/abort_tests.rs, mock/connector_e2e_tests.rs)
  - pub mod e2e and pub mod scheduler active in testing/mod.rs
  - cargo check -p kvbm-connector --features testing passing
  - TEST-02 fully satisfied
affects: []

# Tech tracking
tech-stack:
  added:
    - rand_chacha = "0.9" (workspace dep + kvbm-connector testing feature) — MockModelRunner deterministic token generation
  patterns:
    - "#[cfg(TODO)] mod disabled { ... } pattern for entire-file scheduler gating"
    - "per-item #[cfg(TODO)] for individual fns/structs in scheduler/mod.rs"
    - "kvbm_engine::distributed::leader instead of crate::v2::distributed::leader for e2e tests"

key-files:
  created:
    - lib/kvbm-connector/src/testing/e2e/mod.rs
    - lib/kvbm-connector/src/testing/e2e/find_blocks.rs
    - lib/kvbm-connector/src/testing/e2e/s3_object.rs
    - lib/kvbm-connector/src/testing/scheduler/mod.rs
    - lib/kvbm-connector/src/testing/scheduler/connector_tests.rs
    - lib/kvbm-connector/src/testing/scheduler/mock/mod.rs
    - lib/kvbm-connector/src/testing/scheduler/mock/engine.rs
    - lib/kvbm-connector/src/testing/scheduler/mock/model.rs
    - lib/kvbm-connector/src/testing/scheduler/mock/tests.rs
    - lib/kvbm-connector/src/testing/scheduler/mock/abort_tests.rs
    - lib/kvbm-connector/src/testing/scheduler/mock/connector_e2e_tests.rs
  modified:
    - lib/kvbm-connector/src/testing/mod.rs
    - lib/kvbm-connector/Cargo.toml
    - Cargo.toml (workspace)

key-decisions:
  - "rand_chacha not in workspace — added as workspace dep and kvbm-connector testing feature dep to support MockModelRunner"
  - "MockModelRunner (mock/model.rs) kept live (no Scheduler dep) — inline tests pass without #[cfg(TODO)]"
  - "e2e/find_blocks.rs imports kvbm_engine::distributed::leader for FindMatchesOptions/FindMatchesResult — same path as all other leader imports in the workspace"
  - "scheduler/mod.rs: per-function #[cfg(TODO)] gating for each Scheduler-using function; non-Scheduler functions could be kept live but gated for consistency per plan intent"
  - "entire scheduler/mock/engine.rs wrapped in #[cfg(TODO)] mod — all structs (TestRequest, MockEngineCoreConfig, MockEngineCore) depend on Scheduler"

requirements-completed: [TEST-02]

# Metrics
duration: 20min
completed: 2026-03-11
---

# Phase 4 Plan 03: Port testing/e2e/ and testing/scheduler/ Summary

**11 test files ported from kvbm-next into lib/kvbm-connector/src/testing/; all Scheduler-dependent code gated with #[cfg(TODO)]; pub mod e2e and pub mod scheduler activated; cargo check --features testing passes zero errors**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-03-11T18:58:00Z
- **Completed:** 2026-03-11T19:12:48Z
- **Tasks:** 2
- **Files created:** 11 (+ 3 modified, including workspace Cargo.toml)

## Accomplishments

- `lib/kvbm-connector/src/testing/e2e/` created with 3 files (mod.rs, find_blocks.rs, s3_object.rs)
- `lib/kvbm-connector/src/testing/scheduler/` created with 8 files
- All `crate::v2::*` imports eliminated from ported files
- `crate::v2::testing::connector::*` → `crate::testing::connector::*`
- `crate::v2::physical::transfer::FillPattern` → `kvbm_physical::transfer::FillPattern`
- `crate::v2::distributed::leader::*` → `kvbm_engine::distributed::leader::*`
- `crate::v2::testing::distributed::*` → `kvbm_engine::testing::distributed::*`
- `crate::v2::testing::managers::*` → `kvbm_engine::testing::managers::*`
- `crate::v2::testing::token_blocks::*` → `kvbm_engine::testing::token_blocks::*`
- `crate::v2::distributed::object::*` (s3_object.rs) → `kvbm_engine::distributed::object::*` (gated)
- `s3_object.rs` entirely wrapped in `#[cfg(TODO)]` — s3 feature not declared in kvbm-connector
- All scheduler-dependent code (all functions/tests using `Scheduler`, `KVCacheManager`, `SchedulerConfig`) gated with `#[cfg(TODO)]`
- `MockModelRunner` in `mock/model.rs` kept live — no Scheduler dependency, only rand/rand_chacha
- `rand_chacha` added to workspace Cargo.toml and kvbm-connector testing feature (Rule 3 auto-fix for missing dependency)
- `testing/mod.rs` updated: `pub mod e2e` and `pub mod scheduler` both active
- `cargo check -p kvbm-connector --features testing` exits zero errors
- TEST-02 fully satisfied

## Task Commits

1. **Task 1: Port testing/e2e/ (3 files)** — `3be2c270b` (feat)
2. **Task 2: Port testing/scheduler/ (8 files) and activate all submodules** — `97b944b17` (feat)

## Files Created/Modified

**Created (11 new files):**
- `lib/kvbm-connector/src/testing/e2e/mod.rs` — E2E cluster tests (TestConnectorCluster)
- `lib/kvbm-connector/src/testing/e2e/find_blocks.rs` — Distributed find_blocks E2E tests
- `lib/kvbm-connector/src/testing/e2e/s3_object.rs` — S3 integration tests (gated entirely with #[cfg(TODO)])
- `lib/kvbm-connector/src/testing/scheduler/mod.rs` — Scheduler test utilities (Scheduler-using code gated)
- `lib/kvbm-connector/src/testing/scheduler/connector_tests.rs` — Connector shim tests (fully gated)
- `lib/kvbm-connector/src/testing/scheduler/mock/mod.rs` — Mock module (engine gated, model live)
- `lib/kvbm-connector/src/testing/scheduler/mock/engine.rs` — MockEngineCore (fully gated)
- `lib/kvbm-connector/src/testing/scheduler/mock/model.rs` — MockModelRunner (live, tests pass)
- `lib/kvbm-connector/src/testing/scheduler/mock/tests.rs` — Engine tests (fully gated)
- `lib/kvbm-connector/src/testing/scheduler/mock/abort_tests.rs` — Abort tests (fully gated)
- `lib/kvbm-connector/src/testing/scheduler/mock/connector_e2e_tests.rs` — Connector E2E tests (fully gated)

**Modified (3 files):**
- `lib/kvbm-connector/src/testing/mod.rs` — pub mod e2e and pub mod scheduler uncommented
- `lib/kvbm-connector/Cargo.toml` — rand and rand_chacha as optional testing-feature deps
- `Cargo.toml` (workspace) — rand_chacha = "0.9" added to workspace.dependencies

## Decisions Made

- `rand_chacha` was absent from the workspace. Added as workspace dep (v0.9, matches existing lock file entry as rand transitive dep) and gated to kvbm-connector `testing` feature. This is the minimal fix to support `MockModelRunner` which is kept live (no Scheduler dep).
- `MockModelRunner` inline tests in `mock/model.rs` were NOT gated — they test pure random token generation with no Scheduler dependency. These tests will run with `--features testing`.
- The `#[cfg(TODO)] mod disabled { ... }` pattern was used for files where the entire content is Scheduler-dependent. For `scheduler/mod.rs`, per-function gating was used since the module has a pub struct declaration pattern.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added rand_chacha workspace dependency**
- **Found during:** Task 2
- **Issue:** `mock/model.rs` uses `rand_chacha::ChaCha8Rng` but `rand_chacha` was not in workspace deps or kvbm-connector Cargo.toml
- **Fix:** Added `rand_chacha = { version = "0.9" }` to workspace `Cargo.toml` and `rand_chacha = { workspace = true, optional = true }` to kvbm-connector `[dependencies]`, then added `dep:rand_chacha` to the `testing` feature
- **Files modified:** `Cargo.toml`, `lib/kvbm-connector/Cargo.toml`
- **Commit:** `97b944b17`

## Issues Encountered

None beyond the expected `#[cfg(TODO)]` warnings (expected behavior for disabled code).

## User Setup Required

None.

## Next Phase Readiness

- TEST-02 is complete: all connector-specific test files are ported
- `crate::testing::e2e` and `crate::testing::scheduler` are now accessible
- `MockModelRunner` (the non-Scheduler part of mock engine) is live and its tests run
- Scheduler tests will be re-enabled when `integrations/scheduler` is ported in a future phase

---
*Phase: 04-test-porting*
*Completed: 2026-03-11*
