# Phase 3: Compilation Gate - Research

**Researched:** 2026-03-11
**Domain:** Rust Cargo feature declarations, `cargo check` compilation verification
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Add `nccl = ["kvbm-engine/nccl"]` to `kvbm-connector/Cargo.toml` — proper feature wiring that enables the same NCCL capability through the engine layer
- Only add `nccl` — do not mirror `collectives` or `testing-nccl` from kvbm-engine (YAGNI, minimize diff)
- Do NOT update `[workspace.dependencies]` for kvbm-connector — no downstream crates currently use it, no need to advertise the feature

### Claude's Discretion
- Whether to add a comment above the `nccl` feature line in Cargo.toml explaining what it gates
- Any formatting/ordering of the new feature entry relative to existing features

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| COMP-01 | `cargo check -p kvbm-connector` passes with zero errors | Already satisfied — 0 errors confirmed by live run |
| COMP-02 | `cargo check -p kvbm-connector` passes with zero warnings (or pre-existing and documented) | Blocked by 2 `unexpected_cfgs` warnings; fix is declaring `nccl` feature in Cargo.toml |
| COMP-03 | `cargo check --workspace` passes with zero errors (no regressions) | Confirmed 0 errors currently; must re-verify after the Cargo.toml change |
</phase_requirements>

---

## Summary

Phase 3 has exactly one code change: add a single feature entry to `lib/kvbm-connector/Cargo.toml`. The crate already compiles with 0 errors. The only open issue is 2 Rust compiler warnings (`unexpected_cfg` condition value: `nccl`) because `kvbm-connector/Cargo.toml` uses `#[cfg(feature = "nccl")]` in source but never declares that feature in `[features]`.

The fix is a one-line addition: `nccl = ["kvbm-engine/nccl"]`. This wires the connector's optional NCCL behavior through the engine layer that already declares and implements the feature. After the fix, both `cargo check -p kvbm-connector` and `cargo check --workspace` must exit clean.

**Primary recommendation:** Add `nccl = ["kvbm-engine/nccl"]` to `[features]` in `lib/kvbm-connector/Cargo.toml`, then run both verification commands.

---

## Current State (Verified by Live Run)

Live run of `cargo check -p kvbm-connector` (2026-03-11):

```
warning: unexpected `cfg` condition value: `nccl`
   --> lib/kvbm-connector/src/connector/worker/mod.rs:530:19
    |
530 |             #[cfg(feature = "nccl")]
    |                   ^^^^^^^^^^^^^^^^
    |
    = note: expected values for `feature` are: `testing`
    = help: consider adding `nccl` as a feature in `Cargo.toml`

warning: unexpected `cfg` condition value: `nccl`
   --> lib/kvbm-connector/src/connector/worker/mod.rs:552:23
    |
552 |             #[cfg(not(feature = "nccl"))]
    |                       ^^^^^^^^^^^^^^^^
    |
    = note: expected values for `feature` are: `testing`

warning: `kvbm-connector` (lib) generated 2 warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.62s
```

- Zero errors. Two warnings. Both warnings are the same root cause: undeclared `nccl` feature.
- The compiler itself prints `= help: consider adding 'nccl' as a feature in Cargo.toml` — the fix is unambiguous.

---

## Standard Stack

### Core Tools
| Tool | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| `cargo check` | rustc 1.93.1+ (workspace) | Compile-check without linking | Fast, produces same warning/error output as `cargo build` |
| Cargo `[features]` | Cargo edition 2024 / resolver 3 | Declare optional compile-time capabilities | Native Cargo mechanism for conditional compilation |

No additional libraries are needed. This is a Cargo manifest edit.

---

## Architecture Patterns

### Cargo Feature Declaration Pattern (Connector → Engine passthrough)

**What:** A wrapper crate exposes a feature that simply enables the same feature in a dependency.
**When to use:** Connector wraps an engine that has the real implementation. The connector does not add any feature-gated code of its own — it only gates call sites conditionally.

```toml
# lib/kvbm-connector/Cargo.toml — after the fix
[features]
testing = [
    "kvbm-engine/testing",
    "kvbm-logical/testing",
    "kvbm-physical/testing",
]
nccl = ["kvbm-engine/nccl"]
```

kvbm-engine already declares:
```toml
# lib/kvbm-engine/Cargo.toml (existing, verified)
[features]
nccl = ["dep:cudarc"]
collectives = ["dep:nixl-sys", "nccl"]
testing-nccl = ["collectives"]
```

The connector does NOT declare `collectives` or `testing-nccl` (YAGNI — they are not referenced in connector source).

### Anti-Patterns to Avoid
- **Duplicating `collectives` or `testing-nccl`:** Connector source never references these; adding them broadens the feature surface unnecessarily.
- **Adding `nccl` to `[workspace.dependencies]`:** No downstream crate currently depends on kvbm-connector via workspace; advertising a feature there is premature.
- **Suppressing with `#[allow(unexpected_cfgs)]`:** Would silence the warning without declaring the feature, which would cause `#[cfg(feature = "nccl")]` blocks to never compile even when callers request the feature.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Declaring optional features | Custom cfg attribute schemes | Cargo `[features]` | Native mechanism; resolver 3 handles feature unification correctly |
| Verifying zero warnings | Parsing compiler stdout | `cargo check` exit code + stderr | Cargo exits non-zero only for errors; warnings are surfaced on stderr for human review |

**Key insight:** Cargo's `unexpected_cfg` lint was elevated to deny-by-default in recent Rust editions to catch exactly this pattern (feature used in `#[cfg]` but not declared). The fix is always to declare the feature — not suppress the lint.

---

## Common Pitfalls

### Pitfall 1: `cargo check --workspace` uses default features
**What goes wrong:** After adding `nccl` to the connector's `[features]`, the workspace check still compiles without `nccl` enabled (features are opt-in). This is correct behavior — the warning will be gone because the feature is *declared*, not because it is *enabled*.
**Why it happens:** Misunderstanding that declaring a feature eliminates `unexpected_cfg` warnings regardless of whether the feature is active in the current build.
**How to avoid:** Understand that `#[cfg(feature = "nccl")]` blocks remain dead code in default builds. The warning fires because the feature name is not in the manifest, not because the feature is inactive.
**Warning signs:** If warnings persist after adding the entry, the feature name may be misspelled or placed in the wrong section.

### Pitfall 2: Cargo.toml `[features]` placement
**What goes wrong:** Adding the entry to `[dependencies]` instead of `[features]`, or placing it inside the `testing` array.
**Why it happens:** Quick edits to unfamiliar TOML.
**How to avoid:** `nccl = ["kvbm-engine/nccl"]` is a top-level key inside the `[features]` table, parallel to `testing = [...]`.

### Pitfall 3: Workspace check catches previously hidden errors
**What goes wrong:** `cargo check --workspace` surfaces errors in *other* crates that were not visible when only checking kvbm-connector.
**Why it happens:** Some crates may have independent issues. This is a COMP-03 regression risk (pre-existing, not caused by this change, but newly surfaced).
**How to avoid:** Run `cargo check --workspace` before the change to establish a baseline. If errors appear that aren't caused by the Cargo.toml addition, they are pre-existing and must be documented.

---

## Code Examples

### The Exact Edit (Cargo.toml)

Current `[features]` section in `lib/kvbm-connector/Cargo.toml`:

```toml
[features]
testing = [
    "kvbm-engine/testing",
    "kvbm-logical/testing",
    "kvbm-physical/testing",
]
```

After the fix:

```toml
[features]
testing = [
    "kvbm-engine/testing",
    "kvbm-logical/testing",
    "kvbm-physical/testing",
]
# Gates NCCL-accelerated intra-pass layer-wise onboard (G2→G1 via cudarc).
nccl = ["kvbm-engine/nccl"]
```

The comment is discretionary (per CONTEXT.md). The entry itself is mandatory.

### Verification Commands

```bash
# COMP-01 + COMP-02: zero errors AND zero warnings for kvbm-connector
cargo check -p kvbm-connector

# COMP-03: zero errors across the full workspace
cargo check --workspace
```

Both commands must exit with `Finished` and no warnings/errors attributed to kvbm-connector.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `cfg` lints were warnings-as-you-go | `unexpected_cfg` elevated to warn-by-default | Rust 1.78 (check-cfg stabilized) | Any `#[cfg(feature = "foo")]` without `foo` in `[features]` now warns |

**Deprecated/outdated:**
- Silencing `unexpected_cfg` with `build.rs` rustc-check-cfg: Not needed here. Cargo 1.78+ automatically passes `--check-cfg` flags derived from `[features]`. Just declare the feature.

---

## Open Questions

None. The change is fully specified, verified against live compiler output, and the fix path is unambiguous.

---

## Validation Architecture

`nyquist_validation` is enabled in `.planning/config.json`.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `cargo check` (compiler as validator) |
| Config file | `Cargo.toml` (no separate test config needed) |
| Quick run command | `cargo check -p kvbm-connector` |
| Full suite command | `cargo check --workspace` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| COMP-01 | `cargo check -p kvbm-connector` exits zero errors | smoke | `cargo check -p kvbm-connector` | N/A — compiler |
| COMP-02 | `cargo check -p kvbm-connector` exits zero warnings | smoke | `cargo check -p kvbm-connector` | N/A — compiler |
| COMP-03 | `cargo check --workspace` exits zero errors | smoke | `cargo check --workspace` | N/A — compiler |

### Sampling Rate
- **Per task commit:** `cargo check -p kvbm-connector`
- **Per wave merge:** `cargo check --workspace`
- **Phase gate:** Both commands green before `/gsd:verify-work`

### Wave 0 Gaps
None — no test files needed. All validation is compiler output.

---

## Sources

### Primary (HIGH confidence)
- Live `cargo check -p kvbm-connector` run — confirmed exact warning text, line numbers, and error count
- `lib/kvbm-connector/Cargo.toml` — read directly: current `[features]` section has only `testing`
- `lib/kvbm-engine/Cargo.toml` — read directly: `nccl = ["dep:cudarc"]` confirmed at line 64
- `lib/kvbm-connector/src/connector/worker/mod.rs:530,552` — read directly: `#[cfg(feature = "nccl")]` usage confirmed

### Secondary (MEDIUM confidence)
- Rust reference on check-cfg: https://doc.rust-lang.org/nightly/rustc/check-cfg/cargo-specifics.html (printed by compiler in warning output)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — verified by live compiler run and direct file reads
- Architecture: HIGH — feature wiring pattern is standard Cargo, confirmed against engine's existing `[features]`
- Pitfalls: HIGH — pitfall 1 verified by understanding Cargo feature semantics; pitfalls 2/3 from direct code inspection

**Research date:** 2026-03-11
**Valid until:** Stable (Cargo feature declaration semantics are not fast-moving; valid for months)
