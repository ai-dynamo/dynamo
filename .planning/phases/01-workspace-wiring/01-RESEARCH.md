# Phase 1: Workspace Wiring - Research

**Researched:** 2026-03-11
**Domain:** Cargo workspace configuration (Rust)
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- Add `lib/kvbm-common` to `[workspace.members]` — fix this pre-existing gap alongside kvbm-connector
- Add `derive_getters` to root `[workspace.dependencies]` — consistent with workspace convention
- `dynamo_nova` maps to `velo` (git crate); `dynamo_nova_backend` maps to `velo-common` — both already declared in workspace.dependencies
- `kvbm-connector` MUST declare a `testing` feature that activates testing infra from `kvbm-logical`, `kvbm-physical`, and `kvbm-engine` via their respective `testing` features

### Claude's Discretion

- Exact `derive_getters` version to pin in workspace.dependencies
- Whether kvbm-connector needs `velo-events` or `velo-transports` beyond `velo` + `velo-common` (determine from imports)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| WS-01 | `lib/kvbm-connector` is listed as a member in root `Cargo.toml` | Add `"lib/kvbm-connector"` to `[workspace.members]`; also fix `lib/kvbm-common` (in workspace.dependencies but absent from members) |
| WS-02 | `kvbm-connector/Cargo.toml` declares all required workspace crate dependencies | Full dep list enumerated below from source scan; all are already in `[workspace.dependencies]` except `derive-getters` which must be added |
| WS-03 | `kvbm-connector` is listed in `[workspace.dependencies]` for downstream use | Add `kvbm-connector = { path = "lib/kvbm-connector", version = "0.1.0" }` following the established kvbm pattern |
</phase_requirements>

## Summary

Phase 1 is pure Cargo.toml configuration — no source code changes. The work consists of three mechanical edits across two files: the root `Cargo.toml` and `lib/kvbm-connector/Cargo.toml`.

The root `Cargo.toml` needs two additions to `[workspace.members]` (`"lib/kvbm-connector"` and `"lib/kvbm-common"`, the latter having a pre-existing gap where it is declared in `[workspace.dependencies]` but absent from members) and two additions to `[workspace.dependencies]` (`kvbm-connector` itself for downstream use, and `derive-getters` to promote it from per-crate pinning to workspace-level declaration). The connector's own `Cargo.toml` needs a full `[dependencies]` section populated and a `[features]` section added for the `testing` feature flag.

All required external dependencies are already present in `[workspace.dependencies]`. The `testing` feature follows a well-established pattern in this workspace: `kvbm-engine` defines `testing = ["kvbm-logical/testing", "kvbm-physical/testing"]`, and `kvbm-connector`'s `testing` feature should extend that chain to include `kvbm-engine/testing`.

**Primary recommendation:** Make all edits in a single commit. The three files form an atomic unit — the workspace won't resolve correctly until all three are consistent.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Cargo workspace | Rust built-in | Dependency graph resolution | The only tool for Rust multi-crate repos |
| `workspace = true` syntax | Cargo 1.64+ | Inherit version/features from root | Established pattern in this repo for all kvbm crates |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `derive-getters` | `0.5` | Getter derive macro | Already at version `0.5` in workspace.dependencies; kvbm-connector uses it in `connector/worker/mod.rs` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `{ workspace = true }` for derive-getters | inline `{ version = "0.5" }` | Per-crate pinning (what engine/physical currently do) works but conflicts with the locked decision to centralize it |

**Installation:** No `npm install` equivalent — Cargo resolves automatically after `Cargo.toml` edits.

## Architecture Patterns

### Workspace Members Pattern

```toml
# Root Cargo.toml [workspace]
members = [
    # ... existing members ...
    "lib/kvbm-common",     # Fix pre-existing gap
    "lib/kvbm-connector",  # New addition
]
```

Both must be contiguous physical directories with a `Cargo.toml` — both already exist.

### Workspace Dependencies Pattern (for declaring kvbm-connector)

```toml
# Root Cargo.toml [workspace.dependencies]
# kvbm section — follow exact pattern of sibling crates:
kvbm-connector = { path = "lib/kvbm-connector", version = "0.1.0" }
```

Note: other kvbm crates use `version = "1.0.0"` but kvbm-connector's own `Cargo.toml` declares `version = "0.1.0"` — preserve that value.

### derive-getters Workspace Promotion Pattern

```toml
# Root Cargo.toml [workspace.dependencies]
# Already present — NO CHANGE NEEDED
derive-getters = { version = "0.5" }
```

The root `Cargo.toml` already has `derive-getters = { version = "0.5" }` at line 89. The CONTEXT.md decision to "add" it is already satisfied. The action is to reference it from kvbm-connector via `{ workspace = true }` rather than inline pinning.

### kvbm-connector/Cargo.toml Full Structure

```toml
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

[package]
name = "kvbm-connector"
version = "0.1.0"
edition.workspace = true
description.workspace = true
authors.workspace = true
license.workspace = true
homepage.workspace = true
repository.workspace = true
keywords.workspace = true

[dependencies]
# Internal workspace crates
kvbm-common  = { workspace = true }
kvbm-config  = { workspace = true }
kvbm-engine  = { workspace = true }
kvbm-logical = { workspace = true }
kvbm-physical = { workspace = true }

# Velo (git crates, already declared in workspace.dependencies)
velo        = { workspace = true }
velo-common = { workspace = true }

# Dynamo crates
dynamo-memory = { workspace = true }
dynamo-tokens = { workspace = true }

# External — workspace
anyhow         = { workspace = true }
axum           = { workspace = true }
bytes          = { workspace = true }
cudarc         = { workspace = true }
dashmap        = { workspace = true }
derive_builder = { workspace = true }
derive-getters = { workspace = true }
futures        = { workspace = true }
parking_lot    = { workspace = true }
serde          = { workspace = true }
tokio          = { workspace = true }
uuid           = { workspace = true }

[features]
testing = [
    "kvbm-engine/testing",
    "kvbm-logical/testing",
    "kvbm-physical/testing",
]
```

### Testing Feature Chain Pattern

Established in this repo (`kvbm-engine/Cargo.toml` line 67):

```toml
testing = ["kvbm-logical/testing", "kvbm-physical/testing"]
```

kvbm-connector extends the chain one level up:

```toml
testing = [
    "kvbm-engine/testing",    # pulls in kvbm-logical/testing + kvbm-physical/testing transitively
    "kvbm-logical/testing",   # explicit for clarity
    "kvbm-physical/testing",  # explicit for clarity
]
```

All three `testing` features are empty (`testing = []`) in their respective crates — they gate `cfg(feature = "testing")` blocks and expose test modules.

### Anti-Patterns to Avoid

- **Inline version pinning for derive-getters:** `derive-getters = "0.5"` in connector's Cargo.toml — use `{ workspace = true }` since it is already in workspace.dependencies
- **Version mismatch:** kvbm-connector's version is `0.1.0`, not `1.0.0` — use the crate's declared version when adding to workspace.dependencies
- **Missing kvbm-common in members:** Do not skip the kvbm-common gap fix — it is in workspace.dependencies with a path declaration but not in members, which is an inconsistency Cargo tolerates only because nothing references it via path resolution from a non-member

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dep version management | Per-crate inline version strings | `{ workspace = true }` | All sibling kvbm crates use this pattern; workspace.dependencies already has all needed entries |
| Testing infra activation | Custom cfg flags | `[features] testing = [...]` | Established pattern; downstream invokes with `--features testing` |

**Key insight:** Every external dependency kvbm-connector needs is already in `[workspace.dependencies]`. There is nothing to add except `kvbm-connector` itself (as a workspace dep for downstream) — `derive-getters` is already there.

## Common Pitfalls

### Pitfall 1: Version String for kvbm-connector Entry
**What goes wrong:** Copying `version = "1.0.0"` from sibling kvbm entries for the kvbm-connector workspace.dependencies entry
**Why it happens:** All other kvbm workspace crates use `version = "1.0.0"` via `version.workspace = true`; kvbm-connector explicitly declares `version = "0.1.0"`
**How to avoid:** Read `lib/kvbm-connector/Cargo.toml` first; use the package's own declared version
**Warning signs:** `cargo check` error about version constraint not satisfied

### Pitfall 2: Forgetting kvbm-common in Members
**What goes wrong:** Adding only `kvbm-connector` to members, leaving `kvbm-common` absent
**Why it happens:** The task description focuses on kvbm-connector; kvbm-common's gap is a pre-existing issue
**How to avoid:** Both additions are in the same edit; make them together
**Warning signs:** `cargo check -p kvbm-common` fails with "package not found in workspace"

### Pitfall 3: velo-events / velo-transports Inclusion
**What goes wrong:** Unnecessarily adding `velo-events` or `velo-transports` to kvbm-connector/Cargo.toml
**Why it happens:** They are in workspace.dependencies; tempting to include defensively
**How to avoid:** Source scan shows zero direct use of `velo_events` or `velo_transports` in kvbm-connector source; include only `velo` and `velo-common` (the mapped equivalents of `dynamo_nova` and `dynamo_nova_backend`)
**Warning signs:** Unused dependency warnings after Phase 3 compilation

### Pitfall 4: SPDX Header on kvbm-connector/Cargo.toml
**What goes wrong:** Current `Cargo.toml` has no SPDX header; adding one places it before `[package]`
**Why it happens:** The kvbm-connector Cargo.toml was written without the header; other crates have it
**How to avoid:** Add the header as comment lines above `[package]` — match the pattern from kvbm-logical/Cargo.toml

### Pitfall 5: Workspace Resolver Version
**What goes wrong:** Cargo 3 resolver (in use: `resolver = "3"`) handles feature unification differently from resolver 2
**Why it happens:** Resolver 3 is new (Rust 2024 edition); feature activation for `testing` may behave differently
**How to avoid:** Declare `testing` features explicitly on all three crates rather than relying on transitive propagation alone; the explicit list (`kvbm-engine/testing`, `kvbm-logical/testing`, `kvbm-physical/testing`) is safer

## Code Examples

### Adding Members to Root Cargo.toml

```toml
# Source: direct inspection of /Cargo.toml (lines 5-26)
# Current state:
members = [
    "lib/llm",
    "lib/runtime",
    # ... existing ...
    "lib/kvbm-physical",
    "lib/async-openai",
    # ...
]

# After change — add two entries:
members = [
    "lib/llm",
    "lib/runtime",
    # ... existing ...
    "lib/kvbm-physical",
    "lib/kvbm-common",       # fix pre-existing gap
    "lib/kvbm-connector",    # new
    "lib/async-openai",
    # ...
]
```

### Adding kvbm-connector to workspace.dependencies

```toml
# Source: pattern from existing kvbm section in /Cargo.toml (lines 51-57)
# Add after kvbm-physical:
kvbm-connector = { path = "lib/kvbm-connector", version = "0.1.0" }
```

### Validation Command

```bash
# After all edits — confirm workspace resolves:
cargo check -p kvbm-connector

# Confirm existing members still resolve:
cargo check --workspace
```

Note: `cargo check -p kvbm-connector` will produce import errors (broken `crate::v2::*` paths) because the source code is not yet fixed — that is Phase 2. The workspace membership itself is valid once Cargo can load the `Cargo.toml` without error. The expected Phase 1 outcome is that Cargo resolves the dependency graph, not that the crate compiles.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-crate version pinning | `{ workspace = true }` | Cargo 1.64 (2022) | Centralized dependency management; resolver handles version unification |
| Resolver 2 | Resolver 3 (Rust 2024 edition) | Cargo 1.85+ | Stricter feature unification; explicit feature declarations more important |

**Deprecated/outdated:**
- Inline version strings for deps that exist in workspace.dependencies: still functional but inconsistent with this workspace's convention

## Open Questions

1. **Does `cargo check -p kvbm-connector` fail with a "package not found" or with import errors?**
   - What we know: kvbm-connector has broken `crate::v2::*` imports throughout; adding it to workspace members allows Cargo to find the package
   - What's unclear: Whether Cargo will error on the package not being in workspace before even reaching import resolution, or only on import resolution
   - Recommendation: Treat Phase 1 success as "Cargo can resolve the dependency graph for kvbm-connector" — import errors are expected and are Phase 2 scope

2. **velo-events or velo-transports for kvbm-connector?**
   - What we know: Source scan of `lib/kvbm-connector/src/` shows zero direct uses of `velo_events::` or `velo_transports::` namespaces; the nova/ module uses `dynamo_nova` and `dynamo_nova_backend` which map to `velo` and `velo-common` respectively
   - What's unclear: Whether fixing imports in Phase 2 reveals transitive needs
   - Recommendation: Include only `velo` and `velo-common` in Phase 1; add others in Phase 2 if import resolution requires them

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Cargo built-in test runner (rustc) |
| Config file | none — workspace-level `Cargo.toml` |
| Quick run command | `cargo check -p kvbm-connector 2>&1 \| head -5` |
| Full suite command | `cargo check --workspace` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| WS-01 | `lib/kvbm-connector` in `[workspace.members]` | smoke | `cargo check -p kvbm-connector 2>&1 \| grep -E "error\[E|Finished"` | ✅ (Cargo.toml) |
| WS-02 | All required deps declared in `kvbm-connector/Cargo.toml` | smoke | `cargo check -p kvbm-connector 2>&1 \| grep "unresolved import\|no external crate"` | ✅ (Cargo.toml) |
| WS-03 | `kvbm-connector` in `[workspace.dependencies]` | smoke | `cargo check -p kvbm-connector 2>&1 \| grep -v "error"` | ✅ (Cargo.toml) |

**Important:** `cargo check -p kvbm-connector` will emit import errors (Phase 2 scope). WS-01/02/03 are validated by Cargo being able to LOAD the dependency graph — i.e., no "package not found", no "unresolved external crate" errors. Import resolution errors (`use` path errors) are expected and acceptable at this phase gate.

### Sampling Rate
- **Per task commit:** `cargo check -p kvbm-connector 2>&1 | head -20` (observe only package-level errors, not import errors)
- **Per wave merge:** `cargo check --workspace` (confirm no regressions in existing members)
- **Phase gate:** Workspace resolves `kvbm-connector` as a known member; existing workspace members still check clean

### Wave 0 Gaps
None — no test files need to be created for this phase. Validation is purely structural (Cargo.toml inspection + `cargo check` output).

## Sources

### Primary (HIGH confidence)
- Direct file inspection: `/Cargo.toml` — current workspace members and dependencies
- Direct file inspection: `lib/kvbm-connector/Cargo.toml` — current package declaration, empty dependencies
- Direct file inspection: `lib/kvbm-logical/Cargo.toml` — testing feature pattern (`testing = []`)
- Direct file inspection: `lib/kvbm-engine/Cargo.toml` — testing feature chain pattern (`testing = ["kvbm-logical/testing", "kvbm-physical/testing"]`)
- Direct file inspection: `lib/kvbm-physical/Cargo.toml` — testing feature pattern, derive-getters usage
- Direct source scan: `lib/kvbm-connector/src/` — confirmed velo-events/transports not used directly

### Secondary (MEDIUM confidence)
- Cargo workspace documentation (knowledge, stable since 1.64) — `workspace = true` syntax and resolver behavior

### Tertiary (LOW confidence)
- Resolver 3 feature unification behavior differences — based on Rust 2024 edition release notes (training knowledge, August 2025 cutoff)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — direct file inspection of all relevant Cargo.toml files
- Architecture: HIGH — all patterns observed directly from existing workspace members
- Pitfalls: HIGH for version/member issues (verified); MEDIUM for resolver 3 behavior (training knowledge)

**Research date:** 2026-03-11
**Valid until:** 2026-04-11 (stable Cargo workspace conventions change rarely)
