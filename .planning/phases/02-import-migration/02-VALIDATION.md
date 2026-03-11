---
phase: 2
slug: import-migration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-11
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust built-in) |
| **Config file** | none — feature flags control test inclusion |
| **Quick run command** | `cargo check -p kvbm-connector` |
| **Full suite command** | `cargo check --workspace` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo check -p kvbm-connector`
- **After every plan wave:** Run `cargo check --workspace`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 1 | IMP-03 | compile gate | `cargo check -p kvbm-connector` | ✅ | ⬜ pending |
| 2-01-02 | 01 | 1 | IMP-04 | compile gate | `cargo check -p kvbm-connector` | ✅ | ⬜ pending |
| 2-01-03 | 01 | 1 | IMP-02 | compile gate | `cargo check -p kvbm-connector` | ✅ | ⬜ pending |
| 2-02-01 | 02 | 1 | IMP-01 | compile gate | `cargo check -p kvbm-connector` | ✅ | ⬜ pending |
| 2-02-02 | 02 | 1 | IMP-05 | compile gate | `cargo check -p kvbm-connector` | ✅ | ⬜ pending |
| 2-03-01 | 03 | 2 | VELO-01 | compile gate | `cargo check -p kvbm-connector` | ✅ | ⬜ pending |
| 2-03-02 | 03 | 2 | VELO-02 | compile gate | `cargo check -p kvbm-connector` | ✅ | ⬜ pending |
| 2-03-03 | 03 | 2 | VELO-03 | compile gate | `cargo check -p kvbm-connector` | ✅ | ⬜ pending |
| 2-03-04 | 03 | 2 | VELO-04 | grep verification | `grep -r 'dynamo_nova\|nova_backend' lib/kvbm-connector/src/` | ✅ | ⬜ pending |
| 2-03-05 | 03 | 2 | VELO-05 | compile gate | `cargo check -p kvbm-connector` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

None — this phase has no test files to create. The existing `connector/worker/tests.rs` must be gated (not deleted) with `#[cfg(all(test, feature = "testing"))]` as a precondition gating, and its broken imports must be commented out with a TODO for Phase 4.

*Existing infrastructure covers all phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| No nova module imports remain | VELO-04 | grep verification needed | Run `grep -r 'dynamo_nova\|nova_backend\|use.*nova' lib/kvbm-connector/src/` — expect zero matches |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
