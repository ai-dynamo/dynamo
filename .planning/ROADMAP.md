# Roadmap: kvbm-connector Port

## Overview

Port `lib/kvbm-connector` from a stale branch into the current workspace by completing three sequential steps: wire it into the Cargo workspace, replace all broken internal imports with correct workspace crate paths (including the nova-to-velo transport swap), then confirm the crate compiles cleanly without breaking anything else. The milestone is complete when `cargo check -p kvbm-connector` and `cargo check --workspace` both pass.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Workspace Wiring** - Add kvbm-connector to the Cargo workspace and declare all dependencies
- [ ] **Phase 2: Import Migration** - Replace all stale internal imports with correct workspace crate paths and migrate nova to velo
- [ ] **Phase 3: Compilation Gate** - Verify the crate and full workspace compile cleanly

## Phase Details

### Phase 1: Workspace Wiring
**Goal**: kvbm-connector is a recognized workspace member with its dependency graph declared
**Depends on**: Nothing (first phase)
**Requirements**: WS-01, WS-02, WS-03
**Success Criteria** (what must be TRUE):
  1. `lib/kvbm-connector` appears in the `[workspace.members]` list in root `Cargo.toml`
  2. `kvbm-connector/Cargo.toml` lists all required workspace crate dependencies (`kvbm-logical`, `kvbm-engine`, `kvbm-physical`, `kvbm-common`, `kvbm-config`, `velo`)
  3. `kvbm-connector` is declared in `[workspace.dependencies]` so downstream crates can reference it
**Plans**: TBD

### Phase 2: Import Migration
**Goal**: Every broken import path inside kvbm-connector is replaced with the correct workspace crate path, including the nova-to-velo transport swap
**Depends on**: Phase 1
**Requirements**: IMP-01, IMP-02, IMP-03, IMP-04, IMP-05, VELO-01, VELO-02, VELO-03, VELO-04, VELO-05
**Success Criteria** (what must be TRUE):
  1. No `crate::v2::`, `crate::distributed::`, `crate::logical::`, or `crate::physical::` import paths remain anywhere in `lib/kvbm-connector/src/`
  2. `crate::integrations::*` self-referential imports are resolved to correct local or workspace paths
  3. All files under `src/connector/worker/nova/` reference `velo` transport types instead of `nova` types
  4. No `nova` module import appears anywhere in the codebase
**Plans**: TBD

### Phase 3: Compilation Gate
**Goal**: kvbm-connector compiles cleanly as a workspace member and does not break any other workspace crate
**Depends on**: Phase 2
**Requirements**: COMP-01, COMP-02, COMP-03
**Success Criteria** (what must be TRUE):
  1. `cargo check -p kvbm-connector` exits with zero errors
  2. `cargo check -p kvbm-connector` exits with zero warnings, or all remaining warnings are pre-existing and documented
  3. `cargo check --workspace` exits with zero errors (no regressions in other crates)
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Workspace Wiring | 0/TBD | Not started | - |
| 2. Import Migration | 0/TBD | Not started | - |
| 3. Compilation Gate | 0/TBD | Not started | - |
