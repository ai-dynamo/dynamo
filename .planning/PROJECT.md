# kvbm-connector Port

## What This Is

Port `lib/kvbm-connector` — a framework-integration layer between inference backends (vLLM, etc.) and the KVBM distributed KV cache system — from an old branch into the current workspace. The crate was extracted from a prior monolithic structure but was never updated to use the refactored workspace crates. This project makes it compile cleanly in the current codebase.

## Core Value

kvbm-connector compiles as a workspace member with all imports resolved against current crate structure.

## Requirements

### Validated

- ✓ KVBM crates exist and export needed types (`kvbm-logical`, `kvbm-engine`, `kvbm-physical`, `kvbm-common`, `kvbm-config`) — existing
- ✓ `velo` crate available from git (replaces old `nova` RPC layer) — existing
- ✓ Codebase map complete — existing

### Active

- [ ] Add `lib/kvbm-connector` to workspace `Cargo.toml` members list
- [ ] Wire up `kvbm-connector/Cargo.toml` dependencies (currently empty — needs workspace crate refs)
- [ ] Replace `crate::v2::*` imports with correct workspace crate paths (`kvbm_logical`, `kvbm_engine`, `kvbm_common`)
- [ ] Replace `crate::distributed::*` imports with `kvbm_engine` paths
- [ ] Replace `crate::logical::*` / `crate::physical::*` with `kvbm_logical` / `kvbm_physical`
- [ ] Replace `nova` RPC layer (`src/connector/worker/nova/`) with `velo` equivalents
- [ ] Fix `crate::integrations::*` self-referential imports
- [ ] Resolve all remaining compilation errors (`cargo check -p kvbm-connector` passes)

### Out of Scope

- Python/vLLM integration wiring — compile-only milestone
- TRT-LLM and SGLang backend hookup — future milestone
- New features or refactoring beyond what's needed to compile

## Context

- kvbm-connector was in a branch where `lib/kvbm-engine` and related crates were a single monolith under a `v2::` namespace
- That monolith was split into: `kvbm-common`, `kvbm-config`, `kvbm-engine`, `kvbm-kernels`, `kvbm-logical`, `kvbm-physical`
- kvbm-connector's imports were never updated after the split — it still references `crate::v2::*`, `crate::distributed::*`, `crate::logical::*`, `crate::physical::*`
- The `nova` RPC layer in `src/connector/worker/nova/` is the old velo-transports — replaced by the `velo` git dependency with essentially the same API (name change + minor updates)
- 31 Rust files, ~10k lines, all in `lib/kvbm-connector/src/`

## Constraints

- **Tech stack**: Rust workspace, Cargo — no changes to other crates unless necessary
- **Scope**: Changes confined to `lib/kvbm-connector/` and root `Cargo.toml` membership entry
- **Compatibility**: Don't break existing workspace members

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| nova → velo is a rename (not rewrite) | User confirmed: same API, name change + minor updates | — Pending |
| Compile-only scope for this milestone | vLLM integration is a separate concern | — Pending |

---
*Last updated: 2026-03-11 after initialization*
