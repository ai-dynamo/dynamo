# Phase 3: Compilation Gate - Context

**Gathered:** 2026-03-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Confirm that `kvbm-connector` compiles cleanly as a workspace member and does not break any other workspace crate. This means zero `cargo check` errors and zero warnings — or any remaining warnings are documented as pre-existing.

At discuss-phase time, the crate already compiles with 0 errors. The only remaining work is resolving 2 warnings about an undeclared `nccl` feature, then running the full regression check.

</domain>

<decisions>
## Implementation Decisions

### nccl feature
- Add `nccl = ["kvbm-engine/nccl"]` to `kvbm-connector/Cargo.toml` — proper feature wiring that enables the same NCCL capability through the engine layer
- Only add `nccl` — do not mirror `collectives` or `testing-nccl` from kvbm-engine (YAGNI, minimize diff)
- Do NOT update `[workspace.dependencies]` for kvbm-connector — no downstream crates currently use it, no need to advertise the feature

### Verification gates
- Run `cargo check -p kvbm-connector` → must exit with zero warnings after the nccl fix
- Run `cargo check --workspace` → must exit with zero errors (COMP-03 regression check)
- Both checks run in the same plan (not split into separate plans)

### Claude's Discretion
- Whether to add a comment above the `nccl` feature line in Cargo.toml explaining what it gates
- Any formatting/ordering of the new feature entry relative to existing features

</decisions>

<code_context>
## Existing Code Insights

### Current compilation state
- `cargo check -p kvbm-connector` → 0 errors, 2 warnings (both: `unexpected cfg condition value: nccl`)
- `cargo check --workspace` → 0 errors, 2 warnings (same)
- Warnings at: `lib/kvbm-connector/src/connector/worker/mod.rs:530` and `:552`

### nccl usage in kvbm-connector
- `#[cfg(feature = "nccl")]` gates `execute_local_layerwise_onboard` call (intra-pass layer-wise onboard from G2 to G1)
- `#[cfg(not(feature = "nccl"))]` emits a tracing warning when NCCL unavailable
- This is real behavior gating, not dead code — the feature needs to be declared

### nccl in kvbm-engine
- `kvbm-engine/Cargo.toml`: `nccl = ["dep:cudarc"]`
- `collectives = ["dep:nixl-sys", "nccl"]`
- `testing-nccl = ["collectives"]`
- Only `nccl` needs to be wired from connector; collectives/testing-nccl are not referenced in connector source

### Integration Points
- `lib/kvbm-connector/Cargo.toml` `[features]` section — add `nccl = ["kvbm-engine/nccl"]`
- Root `Cargo.toml` `[workspace.dependencies]` — no change needed

</code_context>

<specifics>
## Specific Ideas

No specific requirements — pure Cargo.toml feature declaration + verification run.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 03-compilation-gate*
*Context gathered: 2026-03-11*
