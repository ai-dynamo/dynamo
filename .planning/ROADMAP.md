# Roadmap: kvbm-connector Port

## Overview

Port `lib/kvbm-connector` from a stale branch into the current workspace: wire it into Cargo, replace all broken imports (including nova→velo), confirm clean compilation, then port connector-specific tests from `ryan/kvbm-next:lib/kvbm/src/v2/testing` and verify they pass using the `testing` feature infrastructure from kvbm-logical/physical/engine.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Workspace Wiring** - Add kvbm-connector to the Cargo workspace and declare all dependencies (completed 2026-03-11)
- [ ] **Phase 2: Import Migration** - Replace all stale internal imports with correct workspace crate paths and migrate nova to velo
- [ ] **Phase 3: Compilation Gate** - Verify the crate and full workspace compile cleanly
- [ ] **Phase 4: Test Porting** - Port connector tests from ryan/kvbm-next and verify they pass

## Phase Details

### Phase 1: Workspace Wiring
**Goal**: kvbm-connector is a recognized workspace member with its dependency graph declared
**Depends on**: Nothing (first phase)
**Requirements**: WS-01, WS-02, WS-03
**Success Criteria** (what must be TRUE):
  1. `lib/kvbm-connector` appears in the `[workspace.members]` list in root `Cargo.toml`
  2. `kvbm-connector/Cargo.toml` lists all required workspace crate dependencies (`kvbm-logical`, `kvbm-engine`, `kvbm-physical`, `kvbm-common`, `kvbm-config`, `velo`)
  3. `kvbm-connector` is declared in `[workspace.dependencies]` so downstream crates can reference it
**Plans**: 1 plan

Plans:
- [ ] 01-01-PLAN.md — Register workspace members, populate kvbm-connector Cargo.toml dependencies and testing feature

### Phase 2: Import Migration
**Goal**: Every broken import path inside kvbm-connector is replaced with the correct workspace crate path, including the nova-to-velo transport swap
**Depends on**: Phase 1
**Requirements**: IMP-01, IMP-02, IMP-03, IMP-04, IMP-05, VELO-01, VELO-02, VELO-03, VELO-04, VELO-05
**Success Criteria** (what must be TRUE):
  1. No `crate::v2::`, `crate::distributed::`, `crate::logical::`, or `crate::physical::` import paths remain anywhere in `lib/kvbm-connector/src/`
  2. `crate::integrations::*` self-referential imports are resolved to correct local or workspace paths
  3. All files under `src/connector/worker/nova/` reference `velo` transport types instead of `nova` types
  4. No `nova` module import appears anywhere in the codebase
**Plans**: 4 plans

Plans:
- [ ] 02-01-PLAN.md — Pass 1-3: Replace crate::logical::*, crate::physical::*, and crate::distributed::* imports
- [ ] 02-02-PLAN.md — Pass 4-5: Replace crate::v2::* imports and resolve crate::integrations::* self-refs
- [ ] 02-03-PLAN.md — Pass 6: Complete nova→velo sweep (directory rename, type renames, runtime.nova→messenger)
- [ ] 02-04-PLAN.md — Gap closure: rename ForwardPassNovaEvent and remaining Nova identifiers/comments

### Phase 3: Compilation Gate
**Goal**: kvbm-connector compiles cleanly as a workspace member and does not break any other workspace crate
**Depends on**: Phase 2
**Requirements**: COMP-01, COMP-02, COMP-03
**Success Criteria** (what must be TRUE):
  1. `cargo check -p kvbm-connector` exits with zero errors
  2. `cargo check -p kvbm-connector` exits with zero warnings, or all remaining warnings are pre-existing and documented
  3. `cargo check --workspace` exits with zero errors (no regressions in other crates)
**Plans**: 1 plan

Plans:
- [ ] 03-01-PLAN.md — Declare nccl feature in kvbm-connector Cargo.toml and verify clean compilation

### Phase 4: Test Porting
**Goal**: Connector-specific tests from `ryan/kvbm-next` are ported into the workspace, compile under `--features testing`, and pass
**Depends on**: Phase 3
**Requirements**: TEST-01, TEST-02, TEST-03, TEST-04
**Success Criteria** (what must be TRUE):
  1. `kvbm-connector/Cargo.toml` has a `testing` feature that pulls in testing infra from kvbm-logical, kvbm-physical, kvbm-engine
  2. Connector tests from `ryan/kvbm-next:lib/kvbm/src/v2/testing` are present in `lib/kvbm-connector/`
  3. `cargo test -p kvbm-connector --features testing` compiles without errors
  4. `cargo test -p kvbm-connector --features testing` passes (all tests green)
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Workspace Wiring | 1/1 | Complete   | 2026-03-11 |
| 2. Import Migration | 0/4 | Not started | - |
| 3. Compilation Gate | 0/1 | Not started | - |
| 4. Test Porting | 0/TBD | Not started | - |
