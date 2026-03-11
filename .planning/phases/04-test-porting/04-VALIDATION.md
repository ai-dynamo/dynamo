---
phase: 4
slug: test-porting
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-11
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (built-in Rust test runner) |
| **Config file** | `lib/kvbm-connector/Cargo.toml` |
| **Quick run command** | `cargo check -p kvbm-connector --features testing` |
| **Full suite command** | `cargo test -p kvbm-connector --features testing` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo check -p kvbm-connector --features testing`
- **After every plan wave:** Run `cargo test -p kvbm-connector --features testing --no-run` (compile gate), then `cargo test -p kvbm-connector --features testing` (run gate)
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 4-01-01 | 01 | 0 | TEST-01 | smoke | `cargo check -p kvbm-connector --features testing` | ✅ `kvbm-connector/Cargo.toml` | ⬜ pending |
| 4-01-02 | 01 | 0 | TEST-02 | unit (build artifact) | `cargo build -p kvbm-connector --features testing` | ❌ W0 | ⬜ pending |
| 4-01-03 | 01 | 0 | TEST-03 | compile | `cargo test -p kvbm-connector --features testing --no-run` | ❌ W0 | ⬜ pending |
| 4-01-04 | 01 | 1 | TEST-04 | unit/integration | `cargo test -p kvbm-connector --features testing` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `lib/kvbm-connector/src/testing/mod.rs` — declares submodules, re-exports ConnectorTestConfig, TestConnectorInstance, TestConnectorCluster, TestConnectorWorker
- [ ] `lib/kvbm-connector/src/testing/connector.rs` — ported and import-migrated from `ryan/kvbm-next:lib/kvbm/src/v2/testing/connector.rs`
- [ ] `lib/kvbm-connector/src/testing/e2e/mod.rs` — ported from source
- [ ] `lib/kvbm-connector/src/testing/e2e/find_blocks.rs` — ported from source
- [ ] `lib/kvbm-connector/src/testing/scheduler/mod.rs` — ported; scheduler-dependent tests disabled
- [ ] `lib/kvbm-connector/src/testing/scheduler/connector_tests.rs` — ported with `#[cfg(TODO)]`
- [ ] `lib/kvbm-connector/src/testing/scheduler/mock/mod.rs` — ported
- [ ] `lib/kvbm-connector/src/testing/scheduler/mock/engine.rs` — ported; scheduler-dependent code disabled
- [ ] `lib/kvbm-connector/src/testing/scheduler/mock/model.rs` — ported
- [ ] `lib/kvbm-connector/src/testing/scheduler/mock/tests.rs` — ported with `#[cfg(TODO)]`
- [ ] `lib/kvbm-connector/src/testing/scheduler/mock/abort_tests.rs` — ported with `#[cfg(TODO)]`
- [ ] `lib/kvbm-connector/src/testing/scheduler/mock/connector_e2e_tests.rs` — ported with `#[cfg(TODO)]`
- [ ] `figment` dep added to `kvbm-connector/Cargo.toml` (optional, testing-gated)
- [ ] `tracing-subscriber` added to `[dev-dependencies]` in `kvbm-connector/Cargo.toml`
- [ ] `pub mod testing;` added to `lib/kvbm-connector/src/lib.rs` under `#[cfg(feature = "testing")]`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GPU/NIXL hardware path | TEST-04 | Requires RDMA hardware not available in CI | Run on hardware node with `cargo test -p kvbm-connector --features testing -- --test-output immediate` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
