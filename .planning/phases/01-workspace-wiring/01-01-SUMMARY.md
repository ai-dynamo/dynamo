---
phase: 01-workspace-wiring
plan: "01"
subsystem: infra
tags: [cargo, workspace, toml, rust, kvbm-connector]

# Dependency graph
requires: []
provides:
  - "kvbm-connector registered as workspace member in root Cargo.toml"
  - "kvbm-common added to workspace members (pre-existing gap fixed)"
  - "kvbm-connector declared in [workspace.dependencies] for downstream use"
  - "lib/kvbm-connector/Cargo.toml has full dependency graph and testing feature"
affects: [02-import-migration, 03-compilation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "{ workspace = true } inheritance for all kvbm-connector dependencies"
    - "Explicit testing feature chain: kvbm-engine/testing + kvbm-logical/testing + kvbm-physical/testing"

key-files:
  created:
    - "lib/kvbm-connector/Cargo.toml"
  modified:
    - "Cargo.toml"

key-decisions:
  - "kvbm-connector uses version 0.1.0 (not 1.0.0 like sibling kvbm crates) — preserved from existing package declaration"
  - "testing feature declares all three sub-crates explicitly (not transitively) for Cargo resolver 3 correctness"
  - "velo-events and velo-transports excluded from kvbm-connector deps — source scan confirmed zero direct uses"
  - "serde_json, tracing, thiserror, oneshot added beyond plan spec — required to eliminate unresolved external crate errors"

patterns-established:
  - "Workspace dep inheritance: all kvbm-connector deps via { workspace = true } — no inline version pinning"
  - "SPDX header on all Cargo.toml files matches sibling crate pattern"

requirements-completed: [WS-01, WS-02, WS-03]

# Metrics
duration: 12min
completed: "2026-03-11"
---

# Phase 1 Plan 01: Workspace Wiring Summary

**kvbm-connector wired into Cargo workspace as a first-class member with full dep graph, testing feature, and workspace.dependencies entry — enabling Phase 2 import migration**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-11T09:08:02Z
- **Completed:** 2026-03-11T09:20:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Root `Cargo.toml` updated: `lib/kvbm-common` and `lib/kvbm-connector` added to `[workspace.members]`, `kvbm-connector = { path = "lib/kvbm-connector", version = "0.1.0" }` added to `[workspace.dependencies]`
- `lib/kvbm-connector/Cargo.toml` replaced with full declaration: SPDX header, 14 workspace deps, and `[features] testing` chain
- `cargo check -p kvbm-connector` confirms package is workspace-recognized; only Phase 2-scope import path errors remain
- No regressions in existing workspace members (`cargo check --workspace` clean outside kvbm-connector)

## Task Commits

Each task was committed atomically:

1. **Task 1: Register workspace members and declare kvbm-connector in workspace.dependencies** - `e2e9997de` (chore)
2. **Task 2: Populate kvbm-connector/Cargo.toml with full dependency graph and testing feature** - `c51a2a156` (chore)

**Plan metadata:** (docs commit to follow)

## Files Created/Modified

- `Cargo.toml` - Added `lib/kvbm-common` and `lib/kvbm-connector` to `[workspace.members]`; added `kvbm-connector` to `[workspace.dependencies]`
- `lib/kvbm-connector/Cargo.toml` - Full replacement with SPDX header, complete `[dependencies]` (workspace = true), and `[features] testing`

## Decisions Made

- Used `version = "0.1.0"` for the kvbm-connector workspace.dependencies entry — matches the crate's own declared version (all other kvbm crates use 1.0.0, but kvbm-connector explicitly declares 0.1.0)
- `testing` feature explicitly lists all three sub-crates rather than relying on transitivity — safer under Cargo resolver 3
- Excluded `velo-events` and `velo-transports` — source scan showed zero direct namespace usage in kvbm-connector

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added serde_json, tracing, thiserror, oneshot to dependencies**
- **Found during:** Task 2 (verifying cargo check -p kvbm-connector)
- **Issue:** Plan's dependency list omitted four crates (`tracing`, `serde_json`, `thiserror`, `oneshot`) that are directly referenced in kvbm-connector source. Without them, `cargo check` emitted "unresolved module or unlinked crate" errors for properly-named (non-renamed) crates, which the plan explicitly requires to be absent.
- **Fix:** Added `thiserror`, `tracing`, `oneshot`, `serde_json` to `[dependencies]` in `lib/kvbm-connector/Cargo.toml` via `{ workspace = true }` (all four are already declared in root `[workspace.dependencies]`)
- **Files modified:** `lib/kvbm-connector/Cargo.toml`
- **Verification:** Re-ran `cargo check -p kvbm-connector`; the four crates no longer appear in unresolved external crate errors; only renamed crates (`dynamo_nova`, `dynamo_kvbm_config`, `dynamo_nova_backend`) and internal path errors remain — all Phase 2 scope
- **Committed in:** `c51a2a156` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical dependency declaration)
**Impact on plan:** Necessary to satisfy the plan's success criterion of no "unresolved external crate" errors. Four workspace-standard dependencies were missing from the plan's dep list. No scope creep.

## Issues Encountered

None beyond the deviation above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 2 (import migration) can proceed: Cargo recognizes kvbm-connector as a workspace member with a valid dependency graph
- Remaining errors are exclusively Phase 2-scope: renamed crates (`dynamo_nova` -> `velo`, `dynamo_kvbm_config` -> `kvbm-config`) and internal module path rewires (`crate::v2`, `crate::distributed`, etc.)
- No blockers

---
*Phase: 01-workspace-wiring*
*Completed: 2026-03-11*

## Self-Check: PASSED

- FOUND: lib/kvbm-connector/Cargo.toml
- FOUND: Cargo.toml
- FOUND: .planning/phases/01-workspace-wiring/01-01-SUMMARY.md
- FOUND commit: e2e9997de (Task 1)
- FOUND commit: c51a2a156 (Task 2)
