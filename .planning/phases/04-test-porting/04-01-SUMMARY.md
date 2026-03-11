---
phase: 04-test-porting
plan: 01
subsystem: testing
tags: [cargo, figment, tracing-subscriber, rust, feature-flags]

# Dependency graph
requires:
  - phase: 03-compilation-gate
    provides: kvbm-connector compiles with nccl/testing feature passthrough declared
provides:
  - figment optional dep in kvbm-connector [dependencies] gated by testing feature
  - tracing-subscriber dev-dep for integration test helpers
  - lib.rs testing module declaration under cfg(feature = "testing")
  - testing/mod.rs skeleton with commented-out submodule stubs ready for Plans 02/03
affects: [04-02, 04-03]

# Tech tracking
tech-stack:
  added: [figment 0.10 (optional, testing-gated)]
  patterns: [optional-dep via dep:figment in feature list, cfg-gated pub mod for testing infra]

key-files:
  created:
    - lib/kvbm-connector/src/testing/mod.rs
  modified:
    - lib/kvbm-connector/Cargo.toml
    - lib/kvbm-connector/src/lib.rs

key-decisions:
  - "figment placed in [dependencies] (not dev-deps) because ConnectorTestConfig will export Figment as a field for downstream use"
  - "testing/mod.rs uses commented-out submodule stubs (Phase A skeleton) so cargo check --features testing passes before Plans 02/03 create the submodule files"

patterns-established:
  - "Phase A skeleton pattern: declare module file with all sub-mods commented out so cargo check gives structure feedback before content files exist"

requirements-completed: [TEST-01]

# Metrics
duration: 5min
completed: 2026-03-11
---

# Phase 4 Plan 01: Testing Feature Scaffold Summary

**figment optional dep + tracing-subscriber dev-dep wired into kvbm-connector Cargo.toml; testing/mod.rs skeleton created; `cargo check --features testing` passes confirming TEST-01**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-11T18:10:00Z
- **Completed:** 2026-03-11T18:15:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- figment 0.10 added as an optional dep in [dependencies], activated only via `dep:figment` in the testing feature
- tracing-subscriber added to [dev-dependencies] using workspace version
- `#[cfg(feature = "testing")] pub mod testing;` added to lib.rs
- `lib/kvbm-connector/src/testing/mod.rs` created as a Phase A skeleton (empty module with commented-out submodule stubs for Plans 02/03)
- `cargo check -p kvbm-connector --features testing` exits zero — TEST-01 satisfied
- `cargo check -p kvbm-connector` (default, no features) exits zero — no regression

## Task Commits

Each task was committed atomically:

1. **Task 1: Update Cargo.toml and lib.rs for testing feature** - `0b1221989` (chore)
2. **Task 2: Create testing/mod.rs skeleton** - `4277001d9` (feat)

**Plan metadata:** (docs commit pending)

## Files Created/Modified

- `lib/kvbm-connector/Cargo.toml` - Added figment optional dep, dep:figment in testing feature, tracing-subscriber dev-dep
- `lib/kvbm-connector/src/lib.rs` - Added `#[cfg(feature = "testing")] pub mod testing;` at end of file
- `lib/kvbm-connector/src/testing/mod.rs` - New skeleton file with doc comments, commented-out submodule stubs for connector/e2e/scheduler

## Decisions Made

- figment placed in `[dependencies]` (not `[dev-dependencies]`) because it will be a struct field type in `ConnectorTestConfig`, which is exported for downstream crate use — dev-only deps cannot be part of public types
- Phase A skeleton approach: comment out all submodule declarations so the module compiles immediately without requiring the submodule files that Plans 02/03 will create

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Testing feature Cargo infrastructure is in place
- Plan 02 can now create `src/testing/connector.rs` and uncomment `pub mod connector;` in mod.rs
- Plan 03 can then create `src/testing/e2e/` and `src/testing/scheduler/` and uncomment their declarations

---
*Phase: 04-test-porting*
*Completed: 2026-03-11*
