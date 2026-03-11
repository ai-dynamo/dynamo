---
phase: 02-import-migration
plan: 02
subsystem: infra
tags: [rust, imports, kvbm-connector, kvbm-common, kvbm-logical, kvbm-engine, kvbm-config, migration]

# Dependency graph
requires:
  - phase: 02-import-migration
    plan: 01
    provides: Passes 1-3 import namespaces cleared (logical/physical/distributed)
provides:
  - No crate::v2::* import paths remain active in lib/kvbm-connector/src/
  - No crate::integrations::* import paths remain active in lib/kvbm-connector/src/
  - connector/worker/tests.rs gated with cfg(all(test, feature = "testing"))
  - crate::v2::testing imports commented out as Phase 4 TODOs
  - kvbm_config replaces dynamo_kvbm_config throughout
  - CacheLayout and ModelExecutorBackend defined locally in config.rs
  - lib.rs re-exports BlockId, SequenceHash, G1/G2/G3/G4, InstanceId, KvbmRuntime
affects: [02-import-migration-03, compile-baseline]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "kvbm_common:: for BlockId, SequenceHash (Pass 4)"
    - "kvbm_logical:: for blocks::{BlockMetadata, ImmutableBlock, MutableBlock}, KvbmSequenceHashProvider (Pass 4)"
    - "kvbm_engine:: for InstanceId, G1/G2/G3/G4 (Pass 4)"
    - "crate::config::{AttentionConfig, IntegrationsConfig, ParallelConfig} replaces crate::v2::integrations::config (Pass 5)"
    - "crate::common::{CachedRequestData, NewRequestData, SchedulerOutput} replaces crate::v2::integrations::common (Pass 5)"
    - "crate::connector::leader::scheduler::KvConnectorMetadata replaces crate::v2::integrations::connector::leader::scheduler (Pass 5)"
    - "kvbm_config:: replaces dynamo_kvbm_config:: (name mismatch fix)"

key-files:
  created: []
  modified:
    - lib/kvbm-connector/src/lib.rs
    - lib/kvbm-connector/src/config.rs
    - lib/kvbm-connector/src/common/block_assignments.rs
    - lib/kvbm-connector/src/common/output.rs
    - lib/kvbm-connector/src/connector/mod.rs
    - lib/kvbm-connector/src/connector/leader/mod.rs
    - lib/kvbm-connector/src/connector/leader/init.rs
    - lib/kvbm-connector/src/connector/leader/slot.rs
    - lib/kvbm-connector/src/connector/leader/scheduler.rs
    - lib/kvbm-connector/src/connector/leader/request.rs
    - lib/kvbm-connector/src/connector/worker/mod.rs
    - lib/kvbm-connector/src/connector/worker/nova/client.rs
    - lib/kvbm-connector/src/connector/worker/init/pending.rs
    - lib/kvbm-connector/src/connector/worker/tests.rs
    - lib/kvbm-connector/src/vllm/config.rs

key-decisions:
  - "CacheLayout and ModelExecutorBackend defined locally in config.rs — absent from entire workspace (not in kvbm_common or any other crate); types inferred from usage context and comments"
  - "dynamo_kvbm_config renamed to kvbm_config — Cargo.toml declares dep as kvbm-config, Rust crate name is kvbm_config"
  - "lib.rs re-exports BlockId, SequenceHash, G1/G2/G3/G4, InstanceId, KvbmRuntime — codebase uses crate:: prefix for these types throughout"
  - "Pass 4 and Pass 5 committed atomically in single commit (all changes interdependent; splitting would leave intermediate broken state)"

patterns-established:
  - "Self-referential integrations:: paths: strip integrations:: prefix — crate IS the old integrations module"
  - "Cargo dep name vs Rust crate name mismatch: kvbm-config (Cargo) → kvbm_config (Rust)"

requirements-completed: [IMP-01, IMP-05]

# Metrics
duration: 8min
completed: 2026-03-11
---

# Phase 2 Plan 2: Import Migration Pass 4-5 Summary

**crate::v2::* and crate::integrations::* namespaces eliminated across 15 files; tests.rs gated for Phase 4; CacheLayout/ModelExecutorBackend defined locally; kvbm_config substituted for dynamo_kvbm_config**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-03-11T09:56:00Z
- **Completed:** 2026-03-11T10:04:00Z
- **Tasks:** 2 (committed together)
- **Files modified:** 15

## Accomplishments

- Eliminated all `crate::v2::*` imports (Pass 4): BlockId/SequenceHash → kvbm_common, logical blocks → kvbm_logical, InstanceId → kvbm_engine, integrations paths → direct crate:: paths
- Eliminated all `crate::integrations::*` imports (Pass 5): all self-referential paths stripped to `crate::` equivalents
- Gated `connector/worker/tests.rs` with `#[cfg(all(test, feature = "testing"))]` — broken `crate::v2::testing::*` imports commented as Phase 4 TODOs
- Defined `CacheLayout` and `ModelExecutorBackend` locally in config.rs (absent from entire workspace)
- Fixed `dynamo_kvbm_config` → `kvbm_config` (Cargo dep name mismatch)
- Added crate-level re-exports in `lib.rs`: `BlockId`, `SequenceHash`, `G1/G2/G3/G4`, `InstanceId`, `KvbmRuntime`
- Fixed `runtime.config` → `runtime.config()` and `runtime.nixl_agent` → `runtime.nixl_agent().cloned()` (private field → accessor methods)
- cargo check -p kvbm-connector: zero v2:: or integrations:: errors; remaining errors exclusively from dynamo_nova/nova namespace (Pass 6 scope)

## Task Commits

1. **Task 1+2: Pass 4+5 — crate::v2::* + crate::integrations::* fixes** - `49c55572f` (fix)

## Files Created/Modified

- `lib/kvbm-connector/src/lib.rs` — added re-exports for BlockId, SequenceHash, G1/G2/G3/G4, InstanceId, KvbmRuntime
- `lib/kvbm-connector/src/config.rs` — replaced crate::v2::{CacheLayout, ModelExecutorBackend} with local definitions; dynamo_kvbm_config → kvbm_config
- `lib/kvbm-connector/src/common/block_assignments.rs` — crate::v2::{BlockId, SequenceHash} → kvbm_common; v2::logical::blocks → kvbm_logical; v2::KvbmSequenceHashProvider → kvbm_logical
- `lib/kvbm-connector/src/common/output.rs` — crate::v2::BlockId → kvbm_common; v2::integrations::connector::leader::scheduler::KvConnectorMetadata → crate::connector::leader::scheduler
- `lib/kvbm-connector/src/connector/mod.rs` — crate::v2::{G1, G2, G3} → kvbm_engine::{G1, G2, G3}
- `lib/kvbm-connector/src/connector/leader/mod.rs` — v2::logical::blocks::ImmutableBlock → kvbm_logical; dynamo_kvbm_config → kvbm_config; runtime.config → runtime.config()
- `lib/kvbm-connector/src/connector/leader/init.rs` — crate::integrations::connector::worker::ConnectorWorkerClient → crate::connector::worker; dynamo_kvbm_config → kvbm_config
- `lib/kvbm-connector/src/connector/leader/slot.rs` — v2::{BlockId, KvbmSequenceHashProvider, SequenceHash} → kvbm_common + kvbm_logical
- `lib/kvbm-connector/src/connector/leader/scheduler.rs` — integrations::connector::leader::slot::RequestSlot → crate::connector::leader::slot; v2::BlockId → kvbm_common; v2::logical::blocks::ImmutableBlock → kvbm_logical; v2::integrations::common → crate::common
- `lib/kvbm-connector/src/connector/leader/request.rs` — v2::integrations::common::Request → crate::common::Request
- `lib/kvbm-connector/src/connector/worker/mod.rs` — v2::integrations::connector::leader::scheduler::KvConnectorMetadata → crate::connector::leader::scheduler; v2::integrations::vllm::layout::determine_kv_layout → crate::vllm::layout; cfg(all(test, feature = "testing")) gate on mod tests
- `lib/kvbm-connector/src/connector/worker/nova/client.rs` — v2::BlockId → kvbm_common; v2::InstanceId → kvbm_engine
- `lib/kvbm-connector/src/connector/worker/init/pending.rs` — v2::physical → kvbm_physical; dynamo_kvbm_config → kvbm_config; runtime.nixl_agent → runtime.nixl_agent().cloned()
- `lib/kvbm-connector/src/connector/worker/tests.rs` — v2::integrations::connector::leader::scheduler → crate::connector::leader::scheduler; v2::testing commented with Phase 4 TODO
- `lib/kvbm-connector/src/vllm/config.rs` — v2::integrations::config → crate::config

## Decisions Made

- `CacheLayout` and `ModelExecutorBackend` are not in any workspace crate — defined locally in config.rs with enum variants inferred from comments (`Ray`, `MultiProcessor`, `Unknown` for backend; `NHD`, `HND`, `Unknown` for layout with `.parse()` method)
- `dynamo_kvbm_config` → `kvbm_config`: Cargo.toml names dep as `kvbm-config`, so Rust crate name is `kvbm_config`, not `dynamo_kvbm_config`
- Pass 4 and Pass 5 committed as single atomic commit because the changes are interdependent (splitting would leave broken intermediate state)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed crate::v2::{G1, G2, G3} in connector/mod.rs (not in plan's file list)**
- **Found during:** Task 1 (Pass 4 verification)
- **Issue:** connector/mod.rs had `pub use crate::v2::{G1, G2, G3}` — not in the plan's file list
- **Fix:** Replaced with `pub use kvbm_engine::{G1, G2, G3}`
- **Files modified:** lib/kvbm-connector/src/connector/mod.rs
- **Committed in:** `49c55572f`

**2. [Rule 1 - Bug] Fixed crate::v2::integrations::common::Request in connector/leader/request.rs (not in plan's file list)**
- **Found during:** Task 1 (Pass 4 verification)
- **Issue:** connector/leader/request.rs had `pub use crate::v2::integrations::common::Request` — not in the plan's file list
- **Fix:** Replaced with `pub use crate::common::Request`
- **Files modified:** lib/kvbm-connector/src/connector/leader/request.rs
- **Committed in:** `49c55572f`

**3. [Rule 2 - Missing critical] Defined CacheLayout and ModelExecutorBackend locally in config.rs**
- **Found during:** Task 1 pre-check
- **Issue:** `CacheLayout` and `ModelExecutorBackend` not found in kvbm_common or any workspace crate — grep across entire lib/ confirmed absence
- **Fix:** Defined both enums locally in config.rs with variants inferred from usage comments
- **Files modified:** lib/kvbm-connector/src/config.rs
- **Committed in:** `49c55572f`

**4. [Rule 1 - Bug] Fixed dynamo_kvbm_config → kvbm_config across 4 files**
- **Found during:** Task 1 (Pass 4 cargo check)
- **Issue:** Code used `dynamo_kvbm_config::*` but Cargo.toml declares `kvbm-config` (Rust name: `kvbm_config`)
- **Fix:** Replaced all `dynamo_kvbm_config` occurrences with `kvbm_config`
- **Files modified:** config.rs, connector/leader/mod.rs, connector/leader/init.rs, connector/worker/init/pending.rs
- **Committed in:** `49c55572f`

**5. [Rule 2 - Missing critical] Added crate-level re-exports to lib.rs**
- **Found during:** Task 1 (Pass 4 cargo check)
- **Issue:** Codebase uses `crate::BlockId`, `crate::InstanceId`, `crate::G1/G2/G3/G4`, `crate::KvbmRuntime` throughout but lib.rs didn't re-export them
- **Fix:** Added `pub use kvbm_common::{BlockId, SequenceHash}` and `pub use kvbm_engine::{G1, G2, G3, G4, InstanceId, KvbmRuntime}` to lib.rs
- **Files modified:** lib/kvbm-connector/src/lib.rs
- **Committed in:** `49c55572f`

**6. [Rule 1 - Bug] Fixed runtime.config → runtime.config() (private field)**
- **Found during:** Task 1 (Pass 4 cargo check)
- **Issue:** Code accessed `self.runtime.config.onboard.mode` directly on private field; KvbmRuntime has pub accessor `config()`
- **Fix:** Changed to `self.runtime.config().onboard.mode`
- **Files modified:** lib/kvbm-connector/src/connector/leader/mod.rs
- **Committed in:** `49c55572f`

**7. [Rule 1 - Bug] Fixed runtime.nixl_agent → runtime.nixl_agent().cloned()**
- **Found during:** Task 1 (Pass 4 cargo check)
- **Issue:** `runtime.nixl_agent` was direct private field access; KvbmRuntime has pub `nixl_agent()` returning `Option<&NixlAgent>`
- **Fix:** Changed to `runtime.nixl_agent().cloned()` (NixlAgent is Clone)
- **Files modified:** lib/kvbm-connector/src/connector/worker/init/pending.rs
- **Committed in:** `49c55572f`

---

**Total deviations:** 7 auto-fixed (all Rule 1/2 — additional file coverage and pre-existing type/API issues)

## Deferred Issues

The following pre-existing API issues were found during cargo check and are out of scope for this plan (Pass 6 scope):

- `execute_local_transfer is private` (connector/leader/onboard.rs:207) — InstanceLeader's method is pub(crate); needs Pass 6 API surface review
- `execute_local_layerwise_onboard not found on DirectWorker` (connector/worker/mod.rs:539) — method name likely changed in nova→velo rename; Pass 6 scope
- `E0282 type annotations needed` — all downstream of unresolved nova types; will resolve after Pass 6

## Next Phase Readiness

- Namespaces cleared: `crate::v2::*`, `crate::integrations::*` — zero active matches
- Remaining broken namespaces in kvbm-connector: `dynamo_nova`, `dynamo_nova_backend` (Pass 6 scope)
- Plan 03 (nova→velo transport swap) can proceed against a cleaner baseline

---
*Phase: 02-import-migration*
*Completed: 2026-03-11*
