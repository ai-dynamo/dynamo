# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — kvbm-connector Port

**Shipped:** 2026-03-11
**Phases:** 4 | **Plans:** 10 | **Sessions:** 1 (single day)

### What Was Built
- kvbm-connector registered as workspace member with full dep graph (25 deps)
- Complete import migration: 5 legacy namespace patterns eliminated, nova→velo transport migration
- `cargo check` green at crate and workspace level (zero errors, zero warnings)
- 1,756-line `testing/connector.rs` + 11 e2e/scheduler test files ported; 132 tests passing

### What Worked
- **Sequential phase dependencies paid off**: each phase set a clean baseline for the next (Cargo wiring → import migration → compilation gate → test porting). No phase had to revisit prior work.
- **Phase A skeleton pattern** for testing/mod.rs (commented-out submodule declarations): let cargo check pass before any test files existed, enabling clean incremental porting.
- **Import migration passes**: breaking the 31-file import migration into 6 named passes (logical, physical, distributed, v2, integrations, nova→velo) prevented overwhelming context and made each pass atomic and verifiable.
- **Gap closure plan (02-04)** caught the `ForwardPassNovaEvent` rename that initial verification missed — the re-verification loop worked as intended.
- **`#[cfg(TODO)]` gating strategy** for Scheduler-dependent code: preserved all test code in a compilable state without requiring the missing `Scheduler` type. Pattern is clear and reversible.

### What Was Inefficient
- **ROADMAP.md checkbox tracking** fell behind — phases 2 and 3 showed "Not started" in the progress table even after completion because the CLI `roadmap update-plan-progress` command didn't update the top-level phase checkboxes correctly. Had to fix manually during milestone completion.
- **`pub mod scheduler;` uncommitted WIP**: the plan 04-04 executor commented it out in the working tree without committing the change (and the committed code had it active). The integration checker caught this, but it introduced an inconsistency between committed and working tree state.
- **SUMMARY.md `one_liner` field not set**: `summary-extract --fields one_liner` returned null for all plans, preventing automatic accomplishment extraction during milestone complete. Plans wrote headline descriptions but not in the expected `one_liner` frontmatter field.

### Patterns Established
- **`#[cfg(TODO)]` gating convention**: use `#[cfg(TODO)] mod disabled { ... }` for entire-file gating, per-item `#[cfg(TODO)]` for selective gating. Clear signal for future enablement.
- **Import migration pass naming**: when porting a large codebase, name each migration pass explicitly (Pass 1-3: logical/physical, Pass 4: v2, Pass 5: integrations, Pass 6: nova→velo) — enables atomic commits per pass and clear grep-verification.
- **Phase A skeleton for testing modules**: create module files with all submodule declarations commented out before any files exist. Avoids cargo check failures during incremental porting.

### Key Lessons
1. **The compiler is the test for import migration work.** `cargo check -p kvbm-connector` after each pass is the definitive verification — no unit test suite needed for correctness during migration.
2. **Commit the correct state, don't leave WIP in working tree.** The uncommitted `testing/mod.rs` change caused confusion across the verification and audit steps. If a file needs to be reverted, revert and commit explicitly.
3. **Gap closure is worth the overhead.** Phase 02-04 added ~3 minutes but closed a real gap (NovaEvent → VeloEvent rename) that the initial verification found. The re-verification loop is fast and reliable.
4. **Preserve the test code even when it can't run.** Gating Scheduler-dependent code with `#[cfg(TODO)]` rather than deleting it keeps the porting work for future activation. The 25 gated items represent real future value.

### Cost Observations
- Model mix: ~100% sonnet (all agents used `sonnet` per config `model_profile: balanced`)
- Sessions: 1 session, single day execution
- Notable: 4-wave sequential execution (each wave depended on prior) — no parallelization opportunity within phase 4, but phases 1-3 were also sequential by design

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | 1 | 4 | First milestone — baseline established |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| v1.0 | 132 | N/A (compile gate) | rand_chacha (workspace) |

### Top Lessons (Verified Across Milestones)

1. Sequential dependency phases with per-phase cargo verification create a reliable compile-forward migration strategy.
2. `#[cfg(TODO)]` gating preserves deferred code without breaking the build.
