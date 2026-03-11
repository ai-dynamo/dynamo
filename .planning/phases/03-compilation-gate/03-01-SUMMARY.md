---
phase: 03-compilation-gate
plan: 01
subsystem: infra
tags: [rust, cargo, nccl, cuda, features]

# Dependency graph
requires:
  - phase: 02-import-migration
    provides: "all imports resolved, nccl cfg gates in place at worker/mod.rs lines 530 and 552"
provides:
  - "nccl feature declared in kvbm-connector Cargo.toml — eliminates unexpected_cfg warnings"
  - "cargo check -p kvbm-connector passes with zero errors and zero warnings"
  - "cargo check --workspace passes with zero errors — no regressions"
affects:
  - "any future phase enabling nccl feature at build time"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Feature passthrough: connector declares nccl = [\"kvbm-engine/nccl\"] — delegates implementation to engine layer"

key-files:
  created: []
  modified:
    - lib/kvbm-connector/Cargo.toml

key-decisions:
  - "nccl feature is a passthrough to kvbm-engine/nccl — connector gates call sites, engine holds the cudarc dep"
  - "No collectives or testing-nccl features added (YAGNI, minimize diff)"
  - "No #[allow(unexpected_cfgs)] suppressions — feature declared properly in Cargo.toml"

patterns-established:
  - "Feature passthrough pattern: crate using cfg(feature) gates should declare the feature, delegating impl to the dep that owns it"

requirements-completed: [COMP-01, COMP-02, COMP-03]

# Metrics
duration: 3min
completed: 2026-03-11
---

# Phase 3 Plan 01: Compilation Gate Summary

**nccl feature declared in kvbm-connector Cargo.toml as passthrough to kvbm-engine/nccl, eliminating both unexpected_cfg warnings and achieving a clean zero-error, zero-warning cargo check**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-11T17:24:55Z
- **Completed:** 2026-03-11T17:27:30Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added `nccl = ["kvbm-engine/nccl"]` to `[features]` in `lib/kvbm-connector/Cargo.toml`
- Eliminated two `unexpected cfg condition value: "nccl"` warnings triggered by `#[cfg(feature = "nccl")]` at worker/mod.rs lines 530 and 552
- Confirmed `cargo check -p kvbm-connector` finishes with zero errors and zero warnings
- Confirmed `cargo check --workspace` finishes with zero errors — no regressions in any other crate

## Task Commits

Each task was committed atomically:

1. **Task 1: Declare nccl feature in kvbm-connector Cargo.toml** - `bdd7cfbe3` (feat)
2. **Task 2: Workspace regression check** - no code changes; read-only verification

**Plan metadata:** (docs commit below)

## Files Created/Modified

- `lib/kvbm-connector/Cargo.toml` - Added `nccl = ["kvbm-engine/nccl"]` entry to `[features]` section

## Decisions Made

- Feature is a passthrough only: kvbm-connector does not implement NCCL; it only gates a call to `worker.execute_local_layerwise_onboard()`. The real NCCL/cudarc dependency lives in kvbm-engine, so the connector's feature must delegate there.
- No additional features (`collectives`, `testing-nccl`) added per YAGNI principle agreed in prior phase context.
- No warning suppression via `#[allow(unexpected_cfgs)]` — proper feature declaration is the correct fix.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The fix was a single two-line addition. `cargo check --workspace` output showed only pre-existing build-script warnings from `kvbm-kernels` (CUDA kernel compilation) and `dynamo-llm` (CUDA KV build), neither of which are errors or related to this change.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 3 complete: `cargo check -p kvbm-connector` and `cargo check --workspace` both green.
- COMP-01, COMP-02, COMP-03 all satisfied.
- kvbm-connector is now a fully wired, warning-free workspace member ready for runtime testing or further feature work.

---
*Phase: 03-compilation-gate*
*Completed: 2026-03-11*
