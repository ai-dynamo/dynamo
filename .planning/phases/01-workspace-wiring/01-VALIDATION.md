---
phase: 1
slug: workspace-wiring
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-11
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Cargo built-in test runner (rustc) |
| **Config file** | none — workspace-level `Cargo.toml` |
| **Quick run command** | `cargo check -p kvbm-connector 2>&1 | head -20` |
| **Full suite command** | `cargo check --workspace` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo check -p kvbm-connector 2>&1 | head -20`
- **After every plan wave:** Run `cargo check --workspace`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | WS-01 | smoke | `cargo check -p kvbm-connector 2>&1 \| grep -E "error\[E\|Finished"` | ✅ | ⬜ pending |
| 1-01-02 | 01 | 1 | WS-02 | smoke | `cargo check -p kvbm-connector 2>&1 \| grep "unresolved import\|no external crate"` | ✅ | ⬜ pending |
| 1-01-03 | 01 | 1 | WS-03 | smoke | `cargo check --workspace 2>&1 \| grep -E "error\[E\|Finished"` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Note:** `cargo check -p kvbm-connector` may emit import resolution errors (Phase 2 scope). WS-01/02/03 are validated by Cargo being able to LOAD the dependency graph — no "package not found" or "unresolved external crate" errors. Import resolution errors (`use` path errors) are expected and acceptable at this phase gate.

---

## Wave 0 Requirements

None — no test files need to be created for this phase. Validation is purely structural (Cargo.toml inspection + `cargo check` output).

*Existing infrastructure covers all phase requirements.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
