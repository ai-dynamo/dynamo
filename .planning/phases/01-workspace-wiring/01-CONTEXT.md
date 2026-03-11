# Phase 1: Workspace Wiring - Context

**Gathered:** 2026-03-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Add `lib/kvbm-connector` to the Cargo workspace and declare its full dependency graph. This covers:
1. Adding `lib/kvbm-connector` to `[workspace.members]` in root `Cargo.toml`
2. Populating `kvbm-connector/Cargo.toml` with all required workspace crate dependencies
3. Adding `kvbm-connector` to `[workspace.dependencies]` for downstream use

Code changes to fix broken imports are Phase 2. This phase is Cargo.toml changes only.

</domain>

<decisions>
## Implementation Decisions

### kvbm-common workspace membership
- Add `lib/kvbm-common` to `[workspace.members]` — it's in workspace.dependencies but currently missing from members (existing gap). Fix it alongside kvbm-connector.

### derive_getters placement
- Add `derive_getters` to root `[workspace.dependencies]` — consistent with workspace convention of centralizing all dependency declarations.

### velo crate mapping (for Cargo.toml declarations)
- `dynamo_nova` → depends on `velo` (git crate, re-exports `Nova`, `EventHandle`, `NovaHandler` equivalents)
- `dynamo_nova_backend` → depends on `velo-common` (exports `PeerInfo`, `WorkerAddress`, `InstanceId`)
- Both `velo` and `velo-common` are already declared in `[workspace.dependencies]` pointing to `ryan/velo-messenger` branch

### Claude's Discretion
- Exact `derive_getters` version to pin in workspace.dependencies
- Whether kvbm-connector needs `velo-events` or `velo-transports` beyond `velo` + `velo-common` (determine from imports)
- Feature flags for kvbm-logical `testing` feature (determine if kvbm-connector needs it)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Root `Cargo.toml` workspace.dependencies: `anyhow`, `bytes`, `dashmap`, `derive_builder`, `futures`, `parking_lot`, `serde`, `tokio`, `uuid`, `cudarc`, `axum`, all `kvbm-*` crates, `velo`/`velo-common`/`velo-events`/`velo-transports`, `dynamo-memory`, `dynamo-tokens` — all available for `kvbm-connector/Cargo.toml` via `workspace = true`

### External crate imports in kvbm-connector
From scanning source files, kvbm-connector uses:
- `anyhow`, `bytes`, `cudarc`, `dashmap`, `derive_builder`, `derive_getters`, `futures`, `parking_lot`, `serde`, `tokio`, `uuid`, `axum` — external
- `dynamo_kvbm_config` = `kvbm-config`
- `dynamo_memory` = `dynamo-memory`
- `dynamo_tokens` = `dynamo-tokens`
- `dynamo_nova` = `velo` (to be mapped in Phase 2)
- `dynamo_nova_backend` = `velo-common` (to be mapped in Phase 2)
- `kvbm-logical`, `kvbm-engine`, `kvbm-physical`, `kvbm-common` (accessed via broken `crate::*` paths — Phase 2 will fix imports, Phase 1 just declares deps)

### Confirmed module-to-crate mapping (user-verified)
Inside kvbm-connector source, these internal module paths map to workspace crates:
- `crate::logical::*` → `kvbm_logical` (`lib/kvbm-logical`)
- `crate::physical::*` → `kvbm_physical` (`lib/kvbm-physical`)
- `crate::distributed::*` → `kvbm_engine` (`lib/kvbm-engine`)
- `crate::v2::{logical, physical, distributed}::*` → same mapping above

### Missing from workspace.dependencies
- `derive_getters` — add to root Cargo.toml workspace.dependencies

### Established Patterns
- All local workspace crates use `workspace = true` syntax in their Cargo.toml
- Workspace deps declare version, features at root; crates opt in with `{ workspace = true }`
- git velo crates already declared at workspace level — `{ workspace = true }` in kvbm-connector

### Integration Points
- Root `Cargo.toml` `[workspace.members]` — add `"lib/kvbm-connector"` and `"lib/kvbm-common"`
- Root `Cargo.toml` `[workspace.dependencies]` — add `kvbm-connector` entry and `derive_getters`
- `lib/kvbm-connector/Cargo.toml` — currently has no `[dependencies]` section; add all deps

</code_context>

<specifics>
## Specific Ideas

No specific requirements — pure Cargo.toml configuration.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-workspace-wiring*
*Context gathered: 2026-03-11*
