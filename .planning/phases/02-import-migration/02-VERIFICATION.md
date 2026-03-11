---
phase: 02-import-migration
verified: 2026-03-11T11:00:00Z
status: gaps_found
score: 11/12 must-haves verified
gaps:
  - truth: "All type names, docstrings, and comments that reference Nova updated to Velo"
    status: partial
    reason: "ForwardPassNovaEvent type alias and all its dependent identifiers (forward_pass_nova_event field, nova_event local variables, set/take/clear_forward_pass_nova_event methods) were not renamed to Velo equivalents. Plan 03 Task 2 Step 6 explicitly required this rename."
    artifacts:
      - path: "lib/kvbm-connector/src/connector/worker/state.rs"
        issue: "Type alias `ForwardPassNovaEvent = velo::EventHandle` still uses Nova name (lines 32, 130, 165, 324, 325, 330, 331, 336, 337). Field `forward_pass_nova_event` and methods `set_forward_pass_nova_event`, `take_forward_pass_nova_event`, `clear_forward_pass_nova_event` all retain Nova naming."
      - path: "lib/kvbm-connector/src/connector/worker/mod.rs"
        issue: "Local variable `nova_event` used in trigger task (lines 328, 338, 362, 363, 444, 445, 446, 493). Multiple doc comments and inline comments still say 'Nova' where 'Velo' was required (lines 18, 28, 102, 168, 248, 314, 322, 327, 330, 440, 445, 492, 497)."
    missing:
      - "Rename type alias ForwardPassNovaEvent → ForwardPassVeloEvent in state.rs"
      - "Rename field forward_pass_nova_event → forward_pass_velo_event in WorkerState struct"
      - "Rename methods set_forward_pass_nova_event, take_forward_pass_nova_event, clear_forward_pass_nova_event to Velo equivalents"
      - "Rename local variable nova_event → velo_event in worker/mod.rs trigger task"
      - "Update doc comments and inline comments in worker/mod.rs and state.rs: 'Nova' → 'Velo' where referring to the transport/event system"
      - "Update pending.rs module doc comment (line 14: 'Worker exports Nova peer address')"
      - "Update control.rs doc comment (line 171: 'registers it with Nova for communication')"
      - "Update leader/mod.rs doc comments (lines 201, 277, 306) referring to Nova messages"
---

# Phase 2: Import Migration Verification Report

**Phase Goal:** Every broken import path inside kvbm-connector is replaced with the correct workspace crate path, including the nova-to-velo transport swap
**Verified:** 2026-03-11T11:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | No crate::logical::* import paths remain in lib/kvbm-connector/src/ | VERIFIED | grep returns zero matches |
| 2  | No crate::physical::* import paths remain in lib/kvbm-connector/src/ | VERIFIED | grep returns zero matches |
| 3  | No crate::distributed::* import paths remain in lib/kvbm-connector/src/ | VERIFIED | grep returns zero matches |
| 4  | cargo check -p kvbm-connector passes after each of the three passes (Plan 01) | VERIFIED | Three commits present; cargo check currently passes with zero errors |
| 5  | No crate::v2::* import paths remain active in lib/kvbm-connector/src/ | VERIFIED | Only commented-out line in tests.rs (Phase 4 TODO as required) |
| 6  | No crate::integrations::* import paths remain active in lib/kvbm-connector/src/ | VERIFIED | grep returns zero matches |
| 7  | connector/worker/tests.rs is gated behind #[cfg(all(test, feature = "testing"))] with broken test imports commented out as Phase 4 TODOs | VERIFIED | mod.rs line 658-659 has cfg gate; tests.rs line 18 has commented import with TODO(Phase 4) |
| 8  | cargo check -p kvbm-connector passes after pass 4 and again after pass 5 | VERIFIED | Zero errors; zero dynamo_nova/nova_backend errors; only warnings (expected nccl feature) |
| 9  | Directory src/connector/worker/nova/ is renamed to src/connector/worker/velo/ | VERIFIED | velo/ directory exists with all 4 files; nova/ directory does not exist |
| 10 | No dynamo_nova, dynamo_nova_backend, or nova module imports remain anywhere in lib/kvbm-connector/src/ | VERIFIED | grep returns zero matches for all three patterns |
| 11 | All self.runtime.nova and self.runtime.nova() occurrences replaced with self.runtime.messenger() | VERIFIED | grep for runtime\.nova returns zero active matches; messenger() confirmed in init.rs, state.rs, mod.rs, pending.rs, scheduler.rs, control.rs |
| 12 | All type names, docstrings, and comments that reference Nova updated to Velo | PARTIAL FAIL | ForwardPassNovaEvent type alias and 6 dependent identifiers unrenamed; ~15 "Nova" references in comments/docstrings across state.rs and worker/mod.rs remain |

**Score:** 11/12 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `lib/kvbm-connector/src/connector/leader/init.rs` | crate::logical and crate::distributed imports replaced | VERIFIED | Uses kvbm_logical::, kvbm_engine::, runtime.messenger() throughout |
| `lib/kvbm-connector/src/connector/leader/mod.rs` | crate::distributed imports replaced | VERIFIED | Uses kvbm_engine::worker::VeloWorkerClient |
| `lib/kvbm-connector/src/connector/worker/state.rs` | crate::distributed imports replaced | VERIFIED (import) | Imports fixed; but ForwardPassNovaEvent type alias still holds Nova naming |
| `lib/kvbm-connector/src/connector/worker/init/pending.rs` | crate::physical imports replaced | VERIFIED | Uses kvbm_physical::, runtime.event_system(), runtime.tokio() |
| `lib/kvbm-connector/src/config.rs` | crate::v2 imports replaced with kvbm_common | VERIFIED | CacheLayout and ModelExecutorBackend defined locally; kvbm_config used throughout |
| `lib/kvbm-connector/src/common/block_assignments.rs` | crate::v2 imports replaced with kvbm_common/kvbm_logical | VERIFIED | kvbm_common::{BlockId, SequenceHash}; kvbm_logical::blocks::* |
| `lib/kvbm-connector/src/connector/worker/tests.rs` | test file gated behind cfg(all(test, feature = "testing")) | VERIFIED | Gate present in mod.rs line 659; broken imports commented with TODO |
| `lib/kvbm-connector/src/vllm/config.rs` | crate::v2::integrations::config self-ref resolved to crate::config | VERIFIED | Uses crate::config::{AttentionConfig, IntegrationsConfig, ParallelConfig} |
| `lib/kvbm-connector/src/connector/worker/velo/` | Renamed directory with velo transport implementation | VERIFIED | client.rs, service.rs, protocol.rs, mod.rs all present |
| `lib/kvbm-connector/src/connector/worker/velo/client.rs` | Uses velo::Messenger, VeloWorkerClient instead of dynamo_nova types | VERIFIED | use velo::Messenger present; no dynamo_nova imports |
| `lib/kvbm-connector/src/connector/worker/velo/service.rs` | Uses velo::Handler, velo::Messenger instead of dynamo_nova types | VERIFIED | use velo::{Handler, Messenger}; Handler::typed_unary_async used |
| `lib/kvbm-connector/src/connector/worker/mod.rs` | mod nova renamed to mod velo, pub use updated | VERIFIED | mod velo (line 36); pub use velo::client::ConnectorWorkerClient (line 40) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| connector/leader/init.rs | kvbm_logical::blocks, kvbm_logical::manager | use kvbm_logical::{blocks, manager} | WIRED | Lines 12-13: kvbm_logical::blocks::{BlockDuplicationPolicy, BlockRegistry}, kvbm_logical::manager::* |
| connector/leader/mod.rs | kvbm_engine::leader, kvbm_engine::worker | use kvbm_engine::{leader, worker} | WIRED | Line 12: use kvbm_engine::worker::VeloWorkerClient |
| connector/worker/mod.rs | kvbm_common::LogicalLayoutHandle, kvbm_physical::transfer::TransferOptions | use kvbm_common, kvbm_physical | WIRED | Lines 57-59: kvbm_common::LogicalLayoutHandle, kvbm_physical::TransferOptions |
| connector/worker/velo/client.rs | velo::Messenger | use ::velo::Messenger | WIRED | Line 7: use velo::Messenger |
| connector/worker/state.rs | kvbm_engine::worker::VeloWorkerService | VeloWorkerService::new(self.runtime.messenger().clone(), worker) | WIRED | Line 254: VeloWorkerService::new(self.runtime.messenger().clone(), worker) |
| connector/leader/init.rs | InstanceLeaderBuilder::messenger() | .messenger(self.runtime.messenger().clone()) | WIRED | Line 383: .messenger(self.runtime.messenger().clone()) |
| config.rs | kvbm_common::{CacheLayout, ModelExecutorBackend} | use kvbm_common::{CacheLayout, ModelExecutorBackend} | NOTE | CacheLayout and ModelExecutorBackend defined locally (absent from workspace); no kvbm_common import needed |
| connector/leader/scheduler.rs | crate::common::{CachedRequestData, NewRequestData, SchedulerOutput} | use crate::common::{...} | WIRED | Line 22: pub use crate::common::{CachedRequestData, NewRequestData, SchedulerOutput} |
| vllm/config.rs | crate::config::{AttentionConfig, IntegrationsConfig, ParallelConfig} | use crate::config::{...} | WIRED | Line 12: use crate::config::{AttentionConfig, IntegrationsConfig, ParallelConfig} |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| IMP-01 | 02-02-PLAN | All crate::v2::* imports replaced | SATISFIED | Zero active crate::v2:: matches; commit 49c55572f |
| IMP-02 | 02-01-PLAN | All crate::distributed::* imports replaced | SATISFIED | Zero crate::distributed:: matches; commit ca8e49519 |
| IMP-03 | 02-01-PLAN | All crate::logical::* imports replaced | SATISFIED | Zero crate::logical:: matches; commit 9f2c1f7fe |
| IMP-04 | 02-01-PLAN | All crate::physical::* imports replaced | SATISFIED | Zero crate::physical:: matches; commit d54097116 |
| IMP-05 | 02-02-PLAN | All crate::integrations::* self-refs resolved | SATISFIED | Zero active crate::integrations:: matches; commit 49c55572f |
| VELO-01 | 02-03-PLAN | nova/client.rs updated to use velo transport types | SATISFIED | velo/client.rs uses velo::Messenger; no dynamo_nova |
| VELO-02 | 02-03-PLAN | nova/service.rs updated to use velo types | SATISFIED | velo/service.rs uses velo::{Handler, Messenger} |
| VELO-03 | 02-03-PLAN | nova/protocol.rs updated for velo protocol | SATISFIED | velo/protocol.rs has no nova imports (only serde types) |
| VELO-04 | 02-03-PLAN | All nova module imports updated to velo equivalents | SATISFIED | Zero dynamo_nova/nova_backend/mod nova matches; commit 6948648db |
| VELO-05 | 02-03-PLAN | velo dependency declared in kvbm-connector/Cargo.toml | SATISFIED | Cargo.toml line 24: velo = { workspace = true } |

All 10 phase-2 requirement IDs are satisfied. No orphaned requirements.

### Anti-Patterns Found

| File | Line(s) | Pattern | Severity | Impact |
|------|---------|---------|----------|--------|
| `connector/worker/state.rs` | 32, 130, 165, 324, 325, 330, 331, 336, 337 | Type alias `ForwardPassNovaEvent` and field/method names retain "Nova" naming | Warning | Inconsistency with phase goal of complete nova→velo rename; identifiers still reference removed transport |
| `connector/worker/mod.rs` | 328, 338, 362, 363, 444, 445, 446, 493 | Local variable `nova_event` in trigger task | Warning | Same — partial rename of the ForwardPassNovaEvent chain |
| `connector/worker/mod.rs` | 18, 28, 102, 168, 248, 314, 322, 327, 330, 440, 445, 492, 497 | Doc comments and inline comments say "Nova" (e.g. "trigger Nova event", "Nova forward pass event") | Info | Documentation inconsistency; does not affect compilation |
| `connector/worker/state.rs` | 121, 128, 144, 323, 328, 334 | Doc comments for field/methods say "Nova" | Info | Documentation inconsistency |
| `connector/worker/init/pending.rs` | 14 | Module doc: "Worker exports Nova peer address" | Info | Stale doc comment |
| `connector/leader/control.rs` | 171 | Doc: "registers it with Nova for communication" | Info | Stale doc comment |
| `connector/leader/mod.rs` | 201, 277, 306 | Doc comments refer to "Nova messages" | Info | Stale doc comments |

### Human Verification Required

None — all checks are programmatically verifiable.

### Gaps Summary

One gap blocks the "All type names, docstrings, and comments that reference Nova updated to Velo" must-have truth from Plan 03.

The `ForwardPassNovaEvent` type alias in `state.rs` was supposed to be renamed to `ForwardPassVeloEvent` per Plan 03 Task 2 Step 6. The rename was not performed. This cascades to:

- The struct field `forward_pass_nova_event` in `WorkerState`
- Three public methods: `set_forward_pass_nova_event`, `take_forward_pass_nova_event`, `clear_forward_pass_nova_event`
- The local variable `nova_event` in `worker/mod.rs` (the trigger task at lines 328, 338, 362, 444, 446, 493)

Additionally, approximately 20 doc comments and inline comments across `worker/mod.rs`, `state.rs`, `pending.rs`, `control.rs`, and `leader/mod.rs` still use "Nova" to describe the event/transport system, where "Velo" was required.

The gap is purely cosmetic/naming — the transport swap is functionally complete (`velo::EventHandle` is the correct backing type, and all imports are clean). However, the must-have truth explicitly covers type names and docstrings, so this registers as a partial failure.

The core phase goal ("every broken import path replaced, including the nova-to-velo transport swap") is **functionally achieved**: `cargo check --workspace` passes with zero errors, all broken import namespaces are eliminated, and the velo transport is wired correctly. The gap is in the completeness of the naming sweep, not in import correctness.

---

_Verified: 2026-03-11T11:00:00Z_
_Verifier: Claude (gsd-verifier)_
