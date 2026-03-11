# Phase 4: Test Porting - Research

**Researched:** 2026-03-11
**Domain:** Rust test infrastructure porting — connector-specific test utilities and unit tests
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Scope of files to port**
- Port **all files** from `ryan/kvbm-next:lib/kvbm/src/v2/testing/` (all 25+ files including subdirectories: e2e/, offloading/, scheduler/)
- Also port `lib/kvbm/src/v2/integrations/connector/worker/tests.rs` — it contains inline unit tests for ConnectorWorker and uses connector testing utilities
- Start with full verbatim copy; disable (comment out or `#[ignore]`) tests that don't compile or don't make sense after migration

**DRY — sub-crate testing exports take priority**
- kvbm-logical, kvbm-physical, kvbm-engine each already have `src/testing/` modules exported under their `testing` feature
- When a v2/testing/ file duplicates what a sub-crate testing module already provides (e.g. managers.rs, token_blocks.rs, physical.rs), **use the sub-crate export instead** — do not copy duplicates into kvbm-connector
- Only add to `lib/kvbm-connector/src/testing/` what is genuinely connector-specific (ConnectorTestConfig, TestConnectorInstance, connector scheduler mocks, connector e2e fixtures)

**Test file location**
- Connector-specific testing utilities: `lib/kvbm-connector/src/testing/` module, gated behind `#[cfg(feature = "testing")]`
- worker/tests.rs inline unit tests: `lib/kvbm-connector/src/connector/worker/tests.rs` (mirrors source structure), included via `mod tests` in worker/mod.rs under `#[cfg(all(test, feature = "testing"))]`

**Import migration strategy**
- Port verbatim first, then fix imports iteratively (same approach as Phase 2)
- Apply same import mappings: `crate::v2::integrations::connector::*` → local paths, `crate::v2::testing::*` → sub-crate testing exports or local testing module
- `dynamo_kvbm_config` → `kvbm_config`, `crate::v2::*` → workspace crate paths

**Nova → Velo in tests**
- Apply the same nova→velo mapping established in Phase 2: `dynamo_nova::Nova` → `velo::Messenger`, `dynamo_nova_backend::WorkerAddress` → `velo::WorkerAddress`, `runtime.nova` → `runtime.messenger`
- If a test file uses a nova API with **no known velo equivalent**, stop and report — do not silently disable or guess the mapping

**Testing feature (TEST-01)**
- `testing` feature already declared in `kvbm-connector/Cargo.toml` (added in Phase 1) with `kvbm-engine/testing`, `kvbm-logical/testing`, `kvbm-physical/testing` — TEST-01 is already satisfied; confirm and document

**Disabled tests**
- Tests that don't make sense after migration: wrap in `#[cfg(TODO)]` or `#[ignore]` with a comment explaining why. Do not delete — they may become relevant later.

### Claude's Discretion

None specified.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TEST-01 | `testing` feature enabled in `kvbm-connector/Cargo.toml` pulling in test infra from kvbm-logical, kvbm-physical, kvbm-engine | **Already satisfied** — feature exists with all three sub-crate deps. Confirm with `cargo check -p kvbm-connector --features testing`. |
| TEST-02 | Connector-specific tests ported from `ryan/kvbm-next:lib/kvbm/src/v2/testing` | Port `connector.rs`, `e2e/`, `scheduler/` sub-modules into `lib/kvbm-connector/src/testing/`. Skip files covered by sub-crate exports. |
| TEST-03 | Ported tests compile with `cargo test -p kvbm-connector --features testing` | Import mapping table below. Key gaps: `figment` dep needed, `tracing-subscriber` in dev-deps, scheduler tests need `#[cfg(TODO)]` due to missing `Scheduler` type. |
| TEST-04 | Ported tests pass with `cargo test -p kvbm-connector --features testing` | worker/tests.rs tests can pass once `TestConnectorInstance` is available. Scheduler/mock tests require the `Scheduler` type from `v2::integrations::scheduler` which has no equivalent in the workspace — must be disabled with `#[cfg(TODO)]`. |
</phase_requirements>

## Summary

Phase 4 ports connector-specific test infrastructure from `ryan/kvbm-next:lib/kvbm/src/v2/testing/` into `lib/kvbm-connector/src/testing/`. The sub-crates (kvbm-engine, kvbm-logical, kvbm-physical) already export their own `testing` modules under the `testing` feature. The connector layer only needs what is genuinely connector-specific: `ConnectorTestConfig`, `TestConnectorInstance`, `TestConnectorCluster`, `TestConnectorWorker`, connector e2e tests, and connector scheduler integration tests.

The source branch's `scheduler/` subdirectory tests (`scheduler/mod.rs`, `scheduler/connector_tests.rs`, `scheduler/mock/`) import `crate::v2::integrations::scheduler::{Scheduler, KVCacheManager, SchedulerConfig}` — a `Scheduler` integration module that does **not exist** in the current workspace. These files must be ported verbatim and then the tests disabled with `#[cfg(TODO)]` until the Scheduler module is available.

The `testing` feature in `kvbm-connector/Cargo.toml` (TEST-01) is already declared and correct. The main work is: add `figment` as an optional dep (testing feature), add `tracing-subscriber` to dev-dependencies, create `lib/kvbm-connector/src/testing/` with the connector.rs content, and fix the import paths using the established Phase 2 mapping table.

**Primary recommendation:** Create `src/testing/connector.rs` first (the fixture types that `worker/tests.rs` already references), verify the unit tests pass, then port e2e and scheduler sub-modules with tests disabled where Scheduler is unavailable.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `kvbm_engine::testing::*` | workspace | Managers, distributed, events, messenger, physical, token_blocks | Already exported by kvbm-engine/testing feature — use directly, do not re-copy |
| `kvbm_logical::testing::*` | workspace | Blocks, sequences, pools, config, managers | Already exported by kvbm-logical/testing feature |
| `kvbm_physical::testing::*` | workspace | TestAgent, physical layouts | Already exported by kvbm-physical/testing feature |
| `figment` | 0.10 | Config builder for `ConnectorTestConfig` — builds `Figment` objects that `KvbmConfig::extract_from()` consumes | Same version as kvbm-config; ConnectorTestConfig stores `Figment` directly |
| `tracing-subscriber` | 0.3 | Per-test log filtering in `#[tokio::test]` bodies | Workspace dep, only needed in dev-dependencies |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `velo::Messenger` | workspace | Replaces `dynamo_nova::Nova` in connector test fixture | Every test that instantiates a KvbmRuntime |
| `velo::WorkerAddress` | workspace | Replaces `dynamo_nova_backend::WorkerAddress` | Where peer registration is handled |
| `kvbm_config::{KvbmConfig, NixlConfig}` | workspace | Config extraction from Figment; NIXL backend defaults | Already in kvbm-connector deps |
| `dynamo_memory::nixl::NixlDescriptor` | workspace | Mock tensor descriptor for test workers | Already in kvbm-connector deps |

### Cargo.toml Changes Required

**Add to `[dependencies]` (optional, testing-gated):**
```toml
figment = { version = "0.10", features = ["env", "toml", "json"], optional = true }
```

**Update `[features]`:**
```toml
testing = [
    "kvbm-engine/testing",
    "kvbm-logical/testing",
    "kvbm-physical/testing",
    "dep:figment",
]
```

**Add to `[dev-dependencies]`:**
```toml
tracing-subscriber = { workspace = true }
```

## Architecture Patterns

### Recommended Project Structure

```
lib/kvbm-connector/src/
├── testing/                    # NEW — gated behind #[cfg(feature = "testing")]
│   ├── mod.rs                  # Re-exports: ConnectorTestConfig, TestConnectorInstance, etc.
│   ├── connector.rs            # Port from v2/testing/connector.rs (connector-specific types)
│   ├── e2e/
│   │   ├── mod.rs              # Port from v2/testing/e2e/mod.rs
│   │   └── find_blocks.rs      # Port from v2/testing/e2e/find_blocks.rs
│   └── scheduler/
│       ├── mod.rs              # Port from v2/testing/scheduler/mod.rs
│       ├── connector_tests.rs  # Port from v2/testing/scheduler/connector_tests.rs
│       └── mock/
│           ├── mod.rs
│           ├── engine.rs
│           ├── model.rs
│           ├── tests.rs        # Disable tests needing Scheduler
│           ├── abort_tests.rs  # Disable tests needing Scheduler
│           └── connector_e2e_tests.rs  # Disable tests needing Scheduler
├── connector/worker/
│   ├── mod.rs                  # Already has: #[cfg(all(test, feature = "testing"))] mod tests;
│   └── tests.rs                # Already exists — uncomment imports after testing/ is ported
└── lib.rs                      # Add: #[cfg(feature = "testing")] pub mod testing;
```

### Pattern 1: Source File → Workspace Mapping (DRY Filter)

**What:** Before copying a file, check if a sub-crate already exports the same utilities.

| Source File | Action | Workspace Path |
|------------|--------|---------------|
| `v2/testing/managers.rs` | **Skip** — covered by `kvbm_engine::testing::managers` | Use `kvbm_engine::testing::*` |
| `v2/testing/token_blocks.rs` | **Skip** — covered by `kvbm_engine::testing::token_blocks` | Use `kvbm_engine::testing::*` |
| `v2/testing/physical.rs` | **Skip** — covered by `kvbm_engine::testing::physical` | Use `kvbm_engine::testing::physical` |
| `v2/testing/distributed.rs` | **Skip** — covered by `kvbm_engine::testing::distributed` | Use `kvbm_engine::testing::distributed` |
| `v2/testing/events.rs` | **Skip** — covered by `kvbm_engine::testing::events` | Use `kvbm_engine::testing::events` |
| `v2/testing/nova.rs` | **Port as `testing/messenger.rs`** — skip, use `kvbm_engine::testing::messenger` which already has the velo equivalent (`create_messenger_tcp`, `MessengerPair`) | Use `kvbm_engine::testing::messenger` |
| `v2/testing/offloading/` | **Skip** — covered by `kvbm_engine::testing::offloading` | Use `kvbm_engine::testing::offloading` |
| `v2/testing/connector.rs` | **Port** — connector-specific (ConnectorTestConfig, TestConnectorInstance, TestConnectorCluster, TestConnectorWorker, MockTensor) | `src/testing/connector.rs` |
| `v2/testing/e2e/mod.rs`, `find_blocks.rs` | **Port** — connector-specific e2e | `src/testing/e2e/` |
| `v2/testing/e2e/s3_object.rs` | **Port under s3 feature gate** | `src/testing/e2e/s3_object.rs` |
| `v2/testing/scheduler/` (all files) | **Port verbatim**, disable tests that need `Scheduler` type | `src/testing/scheduler/` |

### Pattern 2: Import Migration for Ported Files

Apply these mappings when adapting the source:

| Source Import | Workspace Import |
|--------------|-----------------|
| `dynamo_kvbm_config::{KvbmConfig, NixlConfig}` | `kvbm_config::{KvbmConfig, NixlConfig}` |
| `dynamo_nova::Nova` | `velo::Messenger` |
| `dynamo_nova_backend::WorkerAddress` | `velo::WorkerAddress` |
| `dynamo_nova::am::Nova` | `velo::Messenger` |
| `crate::v2::BlockId` | `kvbm_common::BlockId` or `crate::BlockId` |
| `crate::v2::distributed::leader::InstanceLeader` | `kvbm_engine::leader::InstanceLeader` |
| `crate::v2::integrations::connector::leader::ConnectorLeader` | `crate::connector::leader::ConnectorLeader` |
| `crate::v2::integrations::connector::worker::{ConnectorWorker, ConnectorWorkerInterface}` | `crate::connector::worker::{ConnectorWorker, ConnectorWorkerInterface}` |
| `crate::v2::logical::SequenceHash` | `kvbm_common::SequenceHash` or `kvbm_engine::SequenceHash` |
| `crate::v2::physical::layout::LayoutConfig` | `kvbm_physical::layout::LayoutConfig` |
| `crate::v2::physical::transfer::{BlockChecksum, FillPattern}` | `kvbm_physical::transfer::{BlockChecksum, FillPattern}` |
| `crate::{InstanceId, KvbmRuntime}` | `kvbm_engine::{InstanceId, KvbmRuntime}` |
| `crate::physical::layout::{BlockDimension, PhysicalLayout}` | `kvbm_physical::layout::{BlockDimension, PhysicalLayout}` |
| `super::{managers, nova, physical, token_blocks}` | `kvbm_engine::testing::*` for the covered ones |
| `crate::v2::testing::connector::{ConnectorTestConfig, TestConnectorInstance, TestConnectorCluster}` | `crate::testing::connector::{ConnectorTestConfig, TestConnectorInstance, TestConnectorCluster}` |
| `crate::v2::testing::{managers, token_blocks}` | `kvbm_engine::testing::{managers, token_blocks}` |
| `crate::v2::integrations::scheduler::{Scheduler, KVCacheManager, SchedulerConfig, ...}` | **No equivalent** — gate tests with `#[cfg(TODO)]` |
| `crate::v2::integrations::common::{Request, SchedulerOutput}` | `crate::common::{Request, SchedulerOutput}` (or `crate::{Request, SchedulerOutput}` via re-export) |
| `crate::v2::G1` | `kvbm_engine::G1` |

### Pattern 3: Disabled Test Annotation

Tests that cannot compile due to missing `Scheduler` type:

```rust
// Source: [original test name]
// TODO: Disabled — depends on integrations::scheduler::Scheduler which has no workspace
// equivalent in this phase. Re-enable when Scheduler is ported.
#[cfg(TODO)]
#[test]
fn test_that_uses_scheduler() { ... }
```

### Anti-Patterns to Avoid

- **Copying files that sub-crates already export:** `managers.rs`, `token_blocks.rs`, `physical.rs`, `distributed.rs`, `events.rs`, `offloading/` — these are all in `kvbm_engine::testing`. Duplicating them creates divergence.
- **Guessing nova→velo API mappings:** If a nova API has no clear velo equivalent, stop and report. Do not silently swap or remove calls.
- **Using `mod tests` without feature gate:** `worker/mod.rs` already has `#[cfg(all(test, feature = "testing"))] mod tests;` — do not change this guard.
- **Adding figment as a dev-dependency:** `ConnectorTestConfig` stores `Figment` as a field and is exported for downstream use, so `figment` must be in `[dependencies]` (optional, testing-gated), not `[dev-dependencies]`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Messenger pair creation for tests | Custom TCP setup | `kvbm_engine::testing::messenger::{create_messenger_tcp, create_messenger_pair_tcp}` | Already migrated from nova pair; verified in kvbm-engine testing |
| Block manager test setup | Custom BlockManager constructor | `kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder}` | Covers all G1/G2/G3/G4 variants |
| Token sequence generation | Custom token helpers | `kvbm_engine::testing::token_blocks::{create_token_sequence, generate_sequence_hashes}` | Deterministic hash-stable sequences |
| Distributed session setup | Custom leader pair | `kvbm_engine::testing::distributed::create_instance_leader_pair` | Handles TCP peering, runtime sharing |
| Physical transfer test agent | Custom NIXL agent | `kvbm_engine::testing::physical::{TestAgent, TestAgentBuilder}` | Handles NIXL init complexity |

**Key insight:** The sub-crate testing modules were created precisely to eliminate duplication across integration points. The `nova.rs` → `kvbm_engine::testing::messenger` migration is already complete and verified.

## Common Pitfalls

### Pitfall 1: Scheduler Type Has No Workspace Equivalent

**What goes wrong:** `crate::v2::integrations::scheduler::{Scheduler, KVCacheManager, SchedulerConfig}` imports fail at compile — this entire module does not exist in any current workspace crate.

**Why it happens:** The `integrations/scheduler/` module was in the monolithic `lib/kvbm` crate but was not ported to the workspace split. `kvbm-connector` has `connector/leader/scheduler.rs` which is a different type (`KvConnectorMetadata`, `ForwardPassBuilder`).

**How to avoid:** Gate all test code that imports from `v2::integrations::scheduler` with `#[cfg(TODO)]`. The affected files are: `testing/scheduler/mod.rs` (its test functions only), `testing/scheduler/connector_tests.rs` (entire file — it's `#[cfg(test)]`), `testing/scheduler/mock/tests.rs`, `testing/scheduler/mock/abort_tests.rs`, `testing/scheduler/mock/connector_e2e_tests.rs`.

**Warning signs:** Compile error mentioning `use of undeclared crate or module 'scheduler'` inside `testing/scheduler/` or `mock/`.

### Pitfall 2: figment Not in kvbm-connector Deps

**What goes wrong:** `src/testing/connector.rs` imports `figment::Figment` and `figment::providers::{Format, Json}` directly. This fails because `figment` is only a dep of `kvbm-config`, not `kvbm-connector`.

**Why it happens:** `ConnectorTestConfig` stores a `Figment` as a field and exposes it via builder methods. It cannot be abstracted away without breaking the dual-API design.

**How to avoid:** Add `figment = { version = "0.10", features = ["env", "toml", "json"], optional = true }` to `[dependencies]` in `kvbm-connector/Cargo.toml`, then gate it in the `testing` feature: `testing = ["dep:figment", ...]`.

**Warning signs:** `error[E0432]: unresolved import 'figment'` when compiling with `--features testing`.

### Pitfall 3: tests.rs References Testing Types That Don't Exist Yet

**What goes wrong:** `connector/worker/tests.rs` already exists with calls to `TestConnectorInstance::builder()` and `ConnectorTestConfig::new()`, but these types are commented out (`// TODO(Phase 4): Restore when testing infra is ported`). The file will still fail to compile if the commented import is not restored and the types are not present.

**Why it happens:** The file was pre-staged in Phase 2/3 with placeholder comments.

**How to avoid:** The very first task of Phase 4 should create `src/testing/connector.rs` with `ConnectorTestConfig` and `TestConnectorInstance`, then restore the `use crate::testing::connector::{ConnectorTestConfig, TestConnectorInstance};` import in `tests.rs`.

**Warning signs:** `error[E0425]: cannot find function 'create_test_instance'` or `error[E0412]: cannot find type 'TestConnectorInstance'` in `connector/worker/tests.rs`.

### Pitfall 4: worker/mod.rs Gate Is Correct — Don't Change It

**What goes wrong:** Attempting to change `#[cfg(all(test, feature = "testing"))] mod tests;` to just `#[cfg(test)] mod tests;` would compile `tests.rs` without the testing feature, causing failures on the `TestConnectorInstance` import.

**Why it happens:** `tests.rs` uses testing infrastructure that is only available under the `testing` feature.

**How to avoid:** Leave `#[cfg(all(test, feature = "testing"))] mod tests;` exactly as is. The test command `cargo test -p kvbm-connector --features testing` correctly enables both `test` and `testing`.

### Pitfall 5: lib.rs Needs `pub mod testing` Added

**What goes wrong:** `src/testing/` is created but not declared in `lib.rs`, so it's never compiled.

**How to avoid:** Add to `lib.rs`:
```rust
#[cfg(feature = "testing")]
pub mod testing;
```

## Code Examples

### `src/testing/mod.rs` — Module Declaration Pattern

```rust
// Source: pattern from kvbm-engine/src/testing/mod.rs
#[cfg(feature = "testing")]
pub mod connector;
pub mod e2e;
pub mod scheduler;

// Re-export commonly used testing utilities
pub use connector::{
    ConnectorTestConfig,
    TestConnectorInstance,
    TestConnectorCluster,
    TestConnectorWorker,
};
// DO NOT re-export managers, token_blocks, physical, events, distributed, offloading
// Use kvbm_engine::testing::* for those
```

### `lib.rs` — Feature-Gated Module Declaration

```rust
// Add after existing pub mod declarations:
#[cfg(feature = "testing")]
pub mod testing;
```

### `kvbm-connector/Cargo.toml` — Dep Changes

```toml
[dependencies]
# ... existing deps ...
figment = { version = "0.10", features = ["env", "toml", "json"], optional = true }

[features]
testing = [
    "kvbm-engine/testing",
    "kvbm-logical/testing",
    "kvbm-physical/testing",
    "dep:figment",
]
nccl = ["kvbm-engine/nccl"]

[dev-dependencies]
tracing-subscriber = { workspace = true }
```

### Disabled Scheduler Test Pattern

```rust
// Source: create_test_scheduler_with_connector in scheduler/mod.rs
// TODO: Disabled — requires v2::integrations::scheduler::Scheduler which has no
// workspace equivalent. Re-enable when integrations/scheduler is ported.
#[cfg(TODO)]
pub fn create_test_scheduler_with_connector(...) -> anyhow::Result<(Scheduler, ...)> {
    ...
}

// Entire test file gated:
// #[cfg(test)]         <- original
// #[cfg(TODO)]         <- replacement (stops compilation of the test module)
```

### Import Restoration in `connector/worker/tests.rs`

```rust
// Remove the comment markers from Phase 3 placeholder:
// BEFORE (current state):
// TODO(Phase 4): Restore when testing infra is ported
// use crate::v2::testing::connector::{ConnectorTestConfig, TestConnectorInstance};

// AFTER (Phase 4):
use crate::testing::connector::{ConnectorTestConfig, TestConnectorInstance};
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `dynamo_nova::Nova` for test instances | `velo::Messenger` via `kvbm_engine::testing::messenger` | Phase 2 migration | `nova.rs` source file is fully replaced by `kvbm_engine::testing::messenger` |
| `dynamo_nova_backend::WorkerAddress` | `velo::WorkerAddress` | Phase 2 migration | Direct import from `velo` |
| `dynamo_kvbm_config::KvbmConfig` | `kvbm_config::KvbmConfig` | Phase 2 migration | Same API, renamed crate |
| Monolithic `v2/testing/` module in single crate | Split across sub-crate `testing/` modules | Workspace split | DRY: sub-crate testing is the source of truth for non-connector utilities |

**Deprecated/outdated:**
- `nova.rs` (source): Replaced entirely by `kvbm_engine::testing::messenger` — do not port.
- `managers.rs`, `token_blocks.rs`, `physical.rs`, `distributed.rs`, `events.rs`, `offloading/` (source): Covered by sub-crate testing modules — do not copy into kvbm-connector.

## Open Questions

1. **`v2::integrations::scheduler::Scheduler` — when will this type be available?**
   - What we know: It is defined in `lib/kvbm/src/v2/integrations/scheduler/` in the source branch. It has no equivalent in the current workspace.
   - What's unclear: Is there a future phase to port this module, or will it be replaced?
   - Recommendation: Disable affected tests with `#[cfg(TODO)]` and leave a clear comment. Do not attempt to implement a stub.

2. **`TestConnectorInstance` with full NIXL/RDMA — does it require GPU hardware to pass?**
   - What we know: `ConnectorTestConfig::new()` configures workers with `NixlConfig::default()` (UCX + POSIX backends). POSIX backend works without GPU.
   - What's unclear: The worker tests in `tests.rs` call `instance.initialize().await` which triggers the full `InstanceLeader` build path. If GPU memory or RDMA is required, tests will fail in CI.
   - Recommendation: Investigate whether `NixlConfig::empty()` (no NIXL backends) allows tests to pass in CPU-only environments. The CONTEXT.md suggests `worker_without_nixl()` exists for this purpose.

3. **`e2e/s3_object.rs` — S3 feature gate**
   - What we know: The source gates this with `#[cfg(all(test, feature = "s3"))]`.
   - What's unclear: Does kvbm-connector need an `s3` feature, or can this be safely disabled with `#[cfg(TODO)]`?
   - Recommendation: Port the file as `#[cfg(TODO)]` unless the `s3` feature is present in kvbm-connector's feature set (it is not currently).

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | cargo test (built-in Rust test runner) |
| Config file | `kvbm-connector/Cargo.toml` (standard) |
| Quick run command | `cargo test -p kvbm-connector --features testing -- --test-output immediate 2>&1 \| tail -20` |
| Full suite command | `cargo test -p kvbm-connector --features testing` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TEST-01 | `testing` feature declared and activates sub-crate testing | smoke | `cargo check -p kvbm-connector --features testing` | ✅ `kvbm-connector/Cargo.toml` |
| TEST-02 | connector.rs ported to `src/testing/connector.rs` | unit (build artifact) | `cargo build -p kvbm-connector --features testing` | ❌ Wave 0: create `src/testing/connector.rs` |
| TEST-03 | All ported tests compile cleanly | compile | `cargo test -p kvbm-connector --features testing --no-run` | ❌ Wave 0: create `src/testing/` tree |
| TEST-04 | All enabled tests pass | unit/integration | `cargo test -p kvbm-connector --features testing` | ❌ Wave 0: create `src/testing/` tree |

### Sampling Rate

- **Per task commit:** `cargo check -p kvbm-connector --features testing`
- **Per wave merge:** `cargo test -p kvbm-connector --features testing --no-run` (compile gate), then `cargo test -p kvbm-connector --features testing` (run gate)
- **Phase gate:** Full suite green (`cargo test -p kvbm-connector --features testing`) before `/gsd:verify-work`

### Wave 0 Gaps

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

## Sources

### Primary (HIGH confidence)

- Direct inspection of `ryan/kvbm-next` branch via `git show` — source testing files, imports, types
- Direct inspection of current workspace files — Cargo.toml, sub-crate testing modules, lib.rs, worker/mod.rs, worker/tests.rs
- `lib/kvbm-engine/src/testing/messenger.rs` — confirms `nova.rs` is fully replaced by the messenger equivalent
- `lib/kvbm-connector/src/connector/worker/mod.rs` — confirms `#[cfg(all(test, feature = "testing"))] mod tests;` already present
- `lib/kvbm-connector/Cargo.toml` — confirms TEST-01 `testing` feature exists with correct sub-crate deps; `figment` absent

### Secondary (MEDIUM confidence)

- `lib/kvbm-config/src/lib.rs` — `KvbmConfig::figment_for_leader()`, `figment_for_worker()`, and `extract_from()` signatures verified; figment version 0.10 confirmed
- `lib/kvbm-engine/src/testing/mod.rs` — confirms which utilities are exported (managers, token_blocks, physical, distributed, events, messenger, offloading)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all Cargo.toml deps and testing feature declarations inspected directly
- Architecture: HIGH — all source files inspected via git show; DRY filter confirmed against sub-crate exports
- Pitfalls: HIGH — missing Scheduler type confirmed by scanning entire workspace for `pub struct Scheduler`; figment dep gap confirmed by inspecting all Cargo.toml files

**Research date:** 2026-03-11
**Valid until:** 2026-04-11 (stable codebase, no fast-moving deps)
