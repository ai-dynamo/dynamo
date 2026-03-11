# Milestones

## v1.0 kvbm-connector Port (Shipped: 2026-03-11)

**Phases completed:** 4 phases, 10 plans, 22 tasks
**LOC:** 16,403 Rust lines across 44 files
**Timeline:** Single day — 2026-03-11

**Key accomplishments:**
1. Registered kvbm-connector as workspace member with full dep graph (25 deps, `testing` feature chain to kvbm-engine/logical/physical)
2. Migrated all stale `crate::v2::*`, `crate::distributed::*`, `crate::logical::*`, `crate::physical::*` imports to current workspace crate paths
3. Completed nova→velo transport migration: renamed `connector/worker/nova/` → `velo/`, updated all types (`ForwardPassVeloEvent`, `velo_event`), method calls (`runtime.messenger()`)
4. `cargo check -p kvbm-connector` zero errors/warnings; `cargo check --workspace` clean
5. Ported 1,756-line `testing/connector.rs` plus 11 e2e/scheduler test files from `ryan/kvbm-next`
6. `cargo test -p kvbm-connector --features testing` — 132 tests pass, 0 failures

**Tech debt carried forward:**
- 25× `#[cfg(TODO)]` gates in `testing/scheduler/` — deferred until `integrations/scheduler::Scheduler` is ported

**Archive:** `.planning/milestones/v1.0-ROADMAP.md`

---

