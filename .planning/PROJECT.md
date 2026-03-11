# kvbm-connector Port

## What This Is

Port `lib/kvbm-connector` — a framework-integration layer between inference backends (vLLM, etc.) and the KVBM distributed KV cache system — from an old branch into the current workspace. The crate was extracted from a prior monolithic structure but was never updated to use the refactored workspace crates. **v1.0 shipped:** kvbm-connector now compiles cleanly in the current codebase with all imports resolved and 132 tests passing.

## Core Value

kvbm-connector compiles as a workspace member with all imports resolved against current crate structure, with tests verified green.

## Requirements

### Validated

- ✓ KVBM crates exist and export needed types (`kvbm-logical`, `kvbm-engine`, `kvbm-physical`, `kvbm-common`, `kvbm-config`) — existing
- ✓ `velo` crate available from git (replaces old `nova` RPC layer) — existing
- ✓ Codebase map complete — existing
- ✓ `lib/kvbm-connector` added to workspace `Cargo.toml` members list — v1.0
- ✓ `kvbm-connector/Cargo.toml` dependencies wired (25 workspace crate refs) — v1.0
- ✓ All `crate::v2::*` imports replaced with correct workspace crate paths — v1.0
- ✓ All `crate::distributed::*` imports replaced with `kvbm_engine` paths — v1.0
- ✓ All `crate::logical::*` / `crate::physical::*` replaced with workspace equivalents — v1.0
- ✓ `nova` RPC layer (`src/connector/worker/nova/`) migrated to `velo` — v1.0
- ✓ All `crate::integrations::*` self-referential imports resolved — v1.0
- ✓ `cargo check -p kvbm-connector` passes with zero errors and zero warnings — v1.0
- ✓ Connector-specific tests ported from `ryan/kvbm-next:lib/kvbm/src/v2/testing` — v1.0
- ✓ `testing` feature wired to kvbm-logical/physical/engine testing infra — v1.0
- ✓ `cargo test -p kvbm-connector --features testing` passes (132 tests green) — v1.0

### Active

- [ ] Wire vLLM Python component to kvbm-connector (INT-01)
- [ ] Wire TRT-LLM backend to kvbm-connector (INT-02)
- [ ] Wire SGLang backend to kvbm-connector (INT-03)
- [ ] Make kvbm-connector accessible from Python bindings (INT-04)
- [ ] Re-enable `testing/scheduler/` tests once `integrations/scheduler::Scheduler` is ported

### Out of Scope

- Python/vLLM integration wiring — v1.0 was compile-only; now active for v2
- TRT-LLM and SGLang backend hookup — future milestone
- New features or refactoring beyond what's needed to compile

## Context

**Current state (after v1.0):**
- 44 Rust files, 16,403 lines in `lib/kvbm-connector/src/`
- `cargo check -p kvbm-connector` and `cargo check --workspace` both clean
- 132 tests pass under `cargo test -p kvbm-connector --features testing`
- 25× `#[cfg(TODO)]` gates in `testing/scheduler/` — deferred until Scheduler is ported

**Original background:**
- kvbm-connector was in a branch where `lib/kvbm-engine` and related crates were a single monolith under a `v2::` namespace
- That monolith was split into: `kvbm-common`, `kvbm-config`, `kvbm-engine`, `kvbm-kernels`, `kvbm-logical`, `kvbm-physical`
- kvbm-connector's imports were never updated after the split
- The `nova` RPC layer in `src/connector/worker/nova/` was the old velo-transports — replaced by the `velo` git dependency

## Constraints

- **Tech stack**: Rust workspace, Cargo
- **Scope v1**: Changes confined to `lib/kvbm-connector/` and root `Cargo.toml` membership entry
- **Compatibility**: Don't break existing workspace members

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| nova → velo is a rename (not rewrite) | User confirmed: same API, name change + minor updates | ✓ Good — find-and-replace migration worked cleanly |
| Compile-only scope for this milestone | vLLM integration is a separate concern | ✓ Good — clean boundary, v1.0 shipped in one day |
| kvbm-connector uses version 0.1.0 (not 1.0.0) | Consistent with pre-existing Cargo.toml convention | ✓ Good |
| figment in [dependencies] not dev-deps | ConnectorTestConfig exports Figment as public API | ✓ Good |
| Phase A skeleton pattern for testing/mod.rs | cargo check passes before files are created | ✓ Good — reduced iteration |
| rand_chacha added to workspace | MockModelRunner requires it; not previously a dep | ✓ Good — minimal workspace change |
| Scheduler-dependent tests gated with #[cfg(TODO)] | No workspace equivalent for Scheduler yet | — Pending re-enable when Scheduler is ported |
| leader_nova field name kept as Arc<Messenger> | Minimal diff policy — type is correct, name is cosmetic | — Pending cleanup in future phase |

---
*Last updated: 2026-03-11 after v1.0 milestone*
