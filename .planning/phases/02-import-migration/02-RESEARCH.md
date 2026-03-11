# Phase 2: Import Migration - Research

**Researched:** 2026-03-11
**Domain:** Rust workspace import surgery — broken `crate::*` paths to workspace crates + nova→velo rename
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- Rename directory `src/connector/worker/nova/` to `src/connector/worker/velo/`
- Apply full rename inside files: update type names, docstrings, comments that reference "Nova" to say "Velo"
- This is a complete nova→velo sweep, not just import surgery
- Fix one import namespace at a time, running `cargo check -p kvbm-connector` between each pass
- Order: simple-to-complex by mapping confidence:
  1. `crate::logical::*` → `kvbm_logical::*`
  2. `crate::physical::*` → `kvbm_physical::*`
  3. `crate::distributed::*` → `kvbm_engine::*` (distributed subtree)
  4. `crate::v2::*` → correct workspace crate per type
  5. `crate::integrations::*` self-refs → `crate::*`
  6. `nova`→`velo` transport swap (directory rename, type renames, docstrings)
- Each namespace fix is committed atomically — one commit per namespace step
- Blocker escalation: if a type cannot be found, hard-stop and document
- Testing imports in `connector/worker/tests.rs` must be gated `#[cfg(test)]` + `#[cfg(feature = "testing")]`

### Claude's Discretion

(None — all decisions locked)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| IMP-01 | All `crate::v2::*` imports replaced with correct workspace crate paths | Mapping table in Architecture Patterns below |
| IMP-02 | All `crate::distributed::*` imports replaced with `kvbm_engine` paths | Mapping table in Architecture Patterns below |
| IMP-03 | All `crate::logical::*` imports replaced with `kvbm_logical::*` | `kvbm_logical::blocks::*`, `kvbm_logical::manager::*` confirmed |
| IMP-04 | All `crate::physical::*` imports replaced with `kvbm_physical::*` | `kvbm_physical::layout::*`, `kvbm_physical::transfer::*` confirmed |
| IMP-05 | All `crate::integrations::*` self-referential imports resolved | Maps to `crate::*` — lib.rs IS the old integrations module |
| VELO-01 | `src/connector/worker/nova/client.rs` updated to use velo transport types | `dynamo_nova::Nova` → `velo::Messenger`, `dynamo_nova::am::*` → `velo::*` |
| VELO-02 | `src/connector/worker/nova/service.rs` updated to use velo types | `dynamo_nova::{Nova, am::NovaHandler}` → `velo::{Messenger, Handler}` |
| VELO-03 | `src/connector/worker/nova/protocol.rs` updated for velo protocol | Only uses `crate::BlockId` — no nova imports; file is clean |
| VELO-04 | All `nova` module imports across codebase updated to velo equivalents | Full inventory below |
| VELO-05 | `velo` dependency declared in `kvbm-connector/Cargo.toml` | Already declared as `velo = { workspace = true }` |
</phase_requirements>

---

## Summary

Phase 2 is pure import surgery: fix every broken `crate::v2::*`, `crate::distributed::*`, `crate::logical::*`, `crate::physical::*`, and `crate::integrations::*` reference in `lib/kvbm-connector/src/`, then rename `src/connector/worker/nova/` to `velo/` and replace all nova transport types with velo equivalents.

The workspace already has the correct destination crates (`kvbm_logical`, `kvbm_physical`, `kvbm_engine`) and the `velo` git dependency declared in `kvbm-connector/Cargo.toml`. `KvbmRuntime` in `kvbm-engine` already uses `messenger: Arc<Messenger>` not `nova: Arc<Nova>` — so "nova" references in `self.runtime.nova.*` also need updating to `self.runtime.messenger().*`.

The most important non-obvious finding: `KvbmRuntime` has no `.nova` field and no `.nova()` method. The connector's heavy use of `self.runtime.nova` (field) and `self.runtime.nova()` (method) must both become `self.runtime.messenger()`. Additionally, `InstanceLeaderBuilder::nova()` must become `InstanceLeaderBuilder::messenger()`.

**Primary recommendation:** Execute the six-pass strategy from CONTEXT.md strictly in order. Do not blend passes. Run `cargo check -p kvbm-connector` between each pass.

---

## Standard Stack

### Core (already in Cargo.toml)

| Crate (Rust name) | Cargo name | Purpose |
|-------------------|-----------|---------|
| `kvbm_logical` | `kvbm-logical` | `blocks::*`, `manager::*`, `registry::*` — logical block management |
| `kvbm_physical` | `kvbm-physical` | `layout::*`, `transfer::*` — physical layout and NIXL |
| `kvbm_engine` | `kvbm-engine` | distributed coordination: `leader::*`, `worker::*`, `offload::*` |
| `velo` | `velo` (git) | Messenger, Handler, EventHandle, InstanceId, PeerInfo, WorkerAddress |
| `velo_common` | `velo-common` | Shared velo types |

**No new dependencies needed for this phase.** `velo` is already declared. `dynamo_nova` and `dynamo_nova_backend` must be removed (not added).

---

## Architecture Patterns

### Complete Broken Import Inventory

Every broken import found by source scan, with confirmed replacement:

#### Pass 1: `crate::logical::*`

| File | Broken Import | Replacement |
|------|--------------|-------------|
| `connector/leader/init.rs:12` | `crate::logical::blocks::{BlockDuplicationPolicy, BlockRegistry}` | `kvbm_logical::blocks::{BlockDuplicationPolicy, BlockRegistry}` |
| `connector/leader/init.rs:13` | `crate::logical::manager::{BlockManager, FrequencyTrackingCapacity}` | `kvbm_logical::manager::{BlockManager, FrequencyTrackingCapacity}` |
| `connector/worker/mod.rs:57` | `crate::logical::LogicalLayoutHandle` | `kvbm_common::LogicalLayoutHandle` (re-exported from `kvbm_engine::worker`) |

**Note on `LogicalLayoutHandle`:** This type lives in `kvbm_common` but is re-exported through `kvbm_engine::worker` and `kvbm_logical` (via `kvbm_common`). Use `kvbm_common::LogicalLayoutHandle` directly, or the existing `kvbm_engine::worker` re-export — whichever the imports in that file already prefer.

#### Pass 2: `crate::physical::*`

| File | Broken Import | Replacement |
|------|--------------|-------------|
| `vllm/layout.rs:8` | `crate::physical::layout::{BlockDimension, LayoutConfig}` | `kvbm_physical::layout::{BlockDimension, LayoutConfig}` |
| `connector/worker/mod.rs:58` | `crate::physical::TransferOptions` | `kvbm_physical::transfer::TransferOptions` (or `kvbm_physical::TransferOptions`) |
| `connector/worker/nova/client.rs:10` | `crate::physical::layout::LayoutConfig` | `kvbm_physical::layout::LayoutConfig` |
| `connector/worker/init/pending.rs:39` | `v2::physical::TransferManager, layout::{BlockDimension, LayoutConfig, PhysicalLayoutBuilder}` | `kvbm_physical::{TransferManager (via manager), layout::{BlockDimension, LayoutConfig, PhysicalLayoutBuilder}}` |

**Note on `pending.rs`:** This file mixes multiple broken namespaces. It imports via `crate::v2::distributed::*` and `crate::v2::physical::*` — both must be fixed together.

#### Pass 3: `crate::distributed::*`

| File | Broken Import | Replacement |
|------|--------------|-------------|
| `connector/leader/mod.rs:9-10` | `crate::distributed::leader::{FindMatchesOptions, FindMatchesResult, Leader, OnboardingStatus, StagingMode}` | `kvbm_engine::leader::{FindMatchesOptions, FindMatchesResult, Leader, OnboardingStatus, StagingMode}` |
| `connector/leader/mod.rs:12` | `crate::distributed::worker::NovaWorkerClient` | `kvbm_engine::worker::VeloWorkerClient` |
| `connector/leader/init.rs:9` | `crate::distributed::leader::InstanceLeader` | `kvbm_engine::leader::InstanceLeader` |
| `connector/leader/init.rs:10` | `crate::distributed::worker::{LeaderLayoutConfig, NovaWorkerClient, Worker}` | `kvbm_engine::worker::{LeaderLayoutConfig, VeloWorkerClient, Worker}` |
| `connector/leader/slot.rs:11` | `crate::distributed::leader::FindMatchesResult` | `kvbm_engine::leader::FindMatchesResult` |
| `connector/leader/slot.rs:12` | `crate::distributed::offload::TransferHandle` | `kvbm_engine::offload::TransferHandle` |
| `connector/leader/finish.rs:4` | `crate::distributed::offload::{TransferHandle, TransferStatus}` | `kvbm_engine::offload::{TransferHandle, TransferStatus}` |
| `connector/worker/nova/service.rs:8` | `crate::distributed::worker::LeaderLayoutConfig` | `kvbm_engine::worker::LeaderLayoutConfig` |
| `connector/worker/state.rs:26-27` | `crate::distributed::worker::{LeaderLayoutConfig, NovaWorkerService, WorkerLayoutResponse}` | `kvbm_engine::worker::{LeaderLayoutConfig, VeloWorkerService, WorkerLayoutResponse}` |
| `connector/leader/scheduler.rs:6` | `crate::distributed::offload::ExternalBlock` | `kvbm_engine::offload::ExternalBlock` |

**Type rename confirmation:** `NovaWorkerClient` → `VeloWorkerClient` (exported from `kvbm_engine::worker`), `NovaWorkerService` → `VeloWorkerService` (exported from `kvbm_engine::worker`). Both confirmed via `kvbm-engine/src/worker/mod.rs`.

#### Pass 4: `crate::v2::*`

| File | Broken Import | Replacement |
|------|--------------|-------------|
| `config.rs:14` | `crate::v2::{CacheLayout, ModelExecutorBackend}` | `kvbm_common::{CacheLayout, ModelExecutorBackend}` — verify in kvbm-common |
| `common/block_assignments.rs:24` | `crate::v2::{BlockId, SequenceHash}` | `kvbm_common::{BlockId, SequenceHash}` |
| `common/block_assignments.rs:363` | `crate::v2::logical::blocks::{BlockMetadata, ImmutableBlock, MutableBlock}` | `kvbm_logical::blocks::{BlockMetadata, ImmutableBlock, MutableBlock}` |
| `common/output.rs:6` | `crate::v2::BlockId` | `kvbm_common::BlockId` |
| `common/output.rs:7` | `crate::v2::integrations::connector::leader::scheduler::KvConnectorMetadata` | `crate::connector::leader::scheduler::KvConnectorMetadata` (IMP-05) |
| `connector/leader/mod.rs:13` | `crate::v2::distributed::leader::InstanceLeader` | `kvbm_engine::leader::InstanceLeader` (already in Pass 3) |
| `connector/leader/mod.rs:14` | `crate::v2::distributed::offload::OffloadEngine` | `kvbm_engine::offload::OffloadEngine` |
| `connector/leader/mod.rs:15` | `crate::v2::distributed::worker::SerializedLayout` | `kvbm_engine::worker::SerializedLayout` |
| `connector/leader/mod.rs:16` | `crate::v2::logical::blocks::ImmutableBlock` | `kvbm_logical::blocks::ImmutableBlock` |
| `connector/leader/slot.rs:13` | `crate::v2::{BlockId, KvbmSequenceHashProvider, SequenceHash}` | `kvbm_common::{BlockId, SequenceHash}`, `kvbm_logical::KvbmSequenceHashProvider` |
| `connector/leader/init.rs:14-20` | `crate::v2::distributed::object::{ObjectLockManager, create_lock_manager, create_object_client}` | `kvbm_engine::object::{ObjectLockManager, create_lock_manager, create_object_client}` |
| `connector/leader/init.rs:17-20` | `crate::v2::distributed::offload::{ObjectPipelineBuilder, ObjectPresenceFilter, OffloadEngine, PendingTracker, PipelineBuilder, S3PresenceChecker, create_policy_from_config}` | `kvbm_engine::offload::{...}` |
| `connector/leader/init.rs:504` | `crate::v2::distributed::object::ObjectBlockOps` | `kvbm_engine::object::ObjectBlockOps` |
| `connector/worker/mod.rs:59` | `crate::v2::distributed::worker::{DirectWorker, WorkerTransfers}` | `kvbm_engine::worker::{DirectWorker, WorkerTransfers}` |
| `connector/worker/mod.rs:60` | `crate::v2::integrations::connector::leader::scheduler::KvConnectorMetadata` | `crate::connector::leader::scheduler::KvConnectorMetadata` (IMP-05) |
| `connector/worker/mod.rs:61` | `crate::v2::integrations::vllm::layout::determine_kv_layout` | `crate::vllm::layout::determine_kv_layout` (IMP-05) |
| `connector/worker/nova/client.rs:11-13` | `crate::v2::{BlockId, InstanceId}`, `crate::v2::distributed::worker::{LeaderLayoutConfig, WorkerLayoutResponse}` | `kvbm_common::{BlockId}`, `kvbm_engine::{InstanceId}`, `kvbm_engine::worker::{LeaderLayoutConfig, WorkerLayoutResponse}` |
| `connector/worker/init/pending.rs:32-40` | `crate::v2::distributed::{object::create_object_client, worker::{DirectWorker, LeaderLayoutConfig, WorkerLayoutResponse}}`, `crate::v2::physical::{TransferManager, layout::{BlockDimension, LayoutConfig, PhysicalLayoutBuilder}}` | See below |
| `connector/leader/scheduler.rs:7-8` | `crate::integrations::connector::leader::slot::RequestSlot`, `crate::v2::BlockId`, `crate::v2::logical::blocks::ImmutableBlock` | IMP-05 + `kvbm_common::BlockId` + `kvbm_logical::blocks::ImmutableBlock` |
| `connector/leader/scheduler.rs:19` | `crate::v2::integrations::common::{CachedRequestData, NewRequestData, SchedulerOutput}` | UNKNOWN — must locate in kvbm-engine or another crate. Flag for investigation. |

**`pending.rs` full v2 replacements:**
- `crate::v2::distributed::object::create_object_client` → `kvbm_engine::object::create_object_client`
- `crate::v2::distributed::worker::{DirectWorker, LeaderLayoutConfig, WorkerLayoutResponse}` → `kvbm_engine::worker::{DirectWorker, LeaderLayoutConfig, WorkerLayoutResponse}`
- `crate::v2::physical::TransferManager` → `kvbm_physical::TransferManager`
- `crate::v2::physical::layout::{BlockDimension, LayoutConfig, PhysicalLayoutBuilder}` → `kvbm_physical::layout::{BlockDimension, LayoutConfig, PhysicalLayoutBuilder}`
- `crate::logical::LogicalLayoutHandle` → `kvbm_common::LogicalLayoutHandle`
- `crate::physical::transfer::context::TokioRuntime` — **must verify** this exists in `kvbm_physical::transfer::context::TokioRuntime`

#### Pass 5: `crate::integrations::*` self-refs

`lib/kvbm-connector/src/lib.rs` **is** the extracted integrations module. All `crate::integrations::X` references become `crate::X`.

| File | Broken Import | Replacement |
|------|--------------|-------------|
| `connector/leader/init.rs:11` | `crate::integrations::connector::worker::ConnectorWorkerClient` | `crate::connector::worker::ConnectorWorkerClient` |
| `connector/worker/mod.rs:60` | `crate::v2::integrations::connector::leader::scheduler::KvConnectorMetadata` | `crate::connector::leader::scheduler::KvConnectorMetadata` |
| `connector/worker/mod.rs:61` | `crate::v2::integrations::vllm::layout::determine_kv_layout` | `crate::vllm::layout::determine_kv_layout` |
| `connector/leader/scheduler.rs:7` | `crate::integrations::connector::leader::slot::RequestSlot` | `crate::connector::leader::slot::RequestSlot` |
| `common/output.rs:7` | `crate::v2::integrations::connector::leader::scheduler::KvConnectorMetadata` | `crate::connector::leader::scheduler::KvConnectorMetadata` |
| `vllm/config.rs:12` | `crate::v2::integrations::config::{AttentionConfig, IntegrationsConfig, ParallelConfig}` | `crate::config::{AttentionConfig, IntegrationsConfig, ParallelConfig}` |
| `connector/leader/scheduler.rs:19` | `crate::v2::integrations::common::{CachedRequestData, NewRequestData, SchedulerOutput}` | `crate::common::{CachedRequestData, NewRequestData, SchedulerOutput}` |

#### Pass 6: nova→velo transport swap

**Directory rename:** `src/connector/worker/nova/` → `src/connector/worker/velo/`

This requires updating `connector/worker/mod.rs` line `mod nova;` to `mod velo;` and `pub use nova::client::ConnectorWorkerClient` to `pub use velo::client::ConnectorWorkerClient`.

**Type/API renames (confirmed from velo crate source):**

| Old (dynamo_nova) | New (velo) | Location |
|-------------------|-----------|----------|
| `dynamo_nova::Nova` | `velo::Messenger` | `use ::velo::Messenger` |
| `dynamo_nova::am::NovaHandler` | `velo::Handler` | `use velo::Handler` (verify exact name) |
| `dynamo_nova::am::TypedUnaryResult<T>` | `velo::TypedUnaryResult<T>` | verify exact velo re-export |
| `dynamo_nova_backend::{PeerInfo, WorkerAddress}` | `velo::{PeerInfo, WorkerAddress}` | confirmed: `kvbm_engine` re-exports `pub use velo::{InstanceId, PeerInfo, WorkerAddress}` |
| `NovaWorkerClient` | `VeloWorkerClient` | `kvbm_engine::worker::VeloWorkerClient` |
| `NovaWorkerService` | `VeloWorkerService` | `kvbm_engine::worker::VeloWorkerService` |

**`self.runtime.nova` → `self.runtime.messenger()`:**

`KvbmRuntime` in `kvbm-engine` has **no** `.nova` field and **no** `.nova()` method. It has `pub fn messenger(&self) -> &Arc<Messenger>`. Every occurrence of `self.runtime.nova` (field access) and `self.runtime.nova()` (method call) in the connector must become `self.runtime.messenger()`.

Occurrences to fix:
- `connector/worker/init/pending.rs:148` — tracing `fields(instance_id = ?runtime.nova.instance_id())` → `runtime.messenger().instance_id()`
- `connector/worker/init/pending.rs:166` — `runtime.nova.events().local().clone()` → `runtime.messenger().event_system()` or `runtime.event_system()`
- `connector/worker/init/pending.rs:168` — `runtime.nova.runtime().clone()` → `runtime.messenger().runtime().clone()` or `runtime.tokio()`
- `connector/worker/init/pending.rs:258` — `runtime.nova.instance_id()` → `runtime.messenger().instance_id()`
- `connector/worker/state.rs:255` — `NovaWorkerService::new(self.runtime.nova.clone(), worker)` → `VeloWorkerService::new(self.runtime.messenger().clone(), worker)`
- `connector/worker/mod.rs:184` — `let nova = runtime.nova.clone()` → `let messenger = runtime.messenger().clone()`
- `connector/worker/mod.rs:334` — `self.runtime.nova().clone()` → `self.runtime.messenger().clone()`
- `connector/worker/mod.rs:343` — `self.runtime.nova().tracker().spawn_on` → `self.runtime.messenger().tracker().spawn_on`
- `connector/worker/mod.rs:375` — tracing field → `self.runtime.messenger().instance_id()`
- `connector/leader/scheduler.rs:166` — `self.runtime.nova().events().new_event()?` → `self.runtime.messenger().events().new_event()?`
- `connector/leader/scheduler.rs:475` — same pattern
- `connector/leader/scheduler.rs:506` — `let nova = self.runtime.nova().clone()` → `let messenger = self.runtime.messenger().clone()`
- `connector/leader/init.rs:48,52` — `self.runtime.nova.clone()` → `self.runtime.messenger().clone()`
- `connector/leader/init.rs:385` — `.nova(self.runtime.nova.clone())` → `.messenger(self.runtime.messenger().clone())`
- `connector/leader/init.rs:497` — `self.runtime.nova.instance_id()` → `self.runtime.messenger().instance_id()`
- `connector/leader/init.rs:618` — tracing field → `self.runtime.messenger().instance_id()`
- `connector/leader/control.rs:233` — `let nova = leader.runtime.nova()` → `let messenger = leader.runtime.messenger()`

**Event system access pattern (confirmed from kvbm-engine):**
- `nova.events().new_event()` → `messenger.events().new_event()`
- `nova.events().local().clone()` → the velo `Messenger` may expose `event_system()` directly on `KvbmRuntime`. Use `runtime.event_system()` which returns `Arc<velo::EventManager>`. Inspect pending.rs context to decide.
- `nova.runtime().clone()` → `runtime.tokio()` (already a method on `KvbmRuntime`)
- `nova.tracker().spawn_on(...)` → `messenger.tracker().spawn_on(...)`

**Handler registration pattern (service.rs):**

The connector's `nova/service.rs` uses `NovaHandler::typed_unary_async(name, closure).build()`. The velo equivalent is confirmed by `kvbm-engine/src/worker/velo/mod.rs` which uses `Handler` from `::velo`. Verify the exact handler builder API name in the velo crate — look for `Handler::typed_unary_async` or `TypedUnaryHandlerBuilder`. The CONTEXT.md says "treat as find-and-replace, don't redesign."

**ForwardPassNovaEvent type alias:**
- `connector/worker/state.rs:33` — `dynamo_nova::events::EventHandle` → `velo::EventHandle`

**dynamo_nova::events::LocalEvent:**
- `connector/leader/scheduler.rs:470,502,503` — `dynamo_nova::events::LocalEvent` → `velo::LocalEvent` (or equivalent velo type — verify)

### Recommended Execution Order (with cargo check gates)

```
Pass 1: crate::logical::*    → cargo check -p kvbm-connector
Pass 2: crate::physical::*   → cargo check -p kvbm-connector
Pass 3: crate::distributed::* → cargo check -p kvbm-connector
Pass 4: crate::v2::*         → cargo check -p kvbm-connector
Pass 5: crate::integrations::* → cargo check -p kvbm-connector
Pass 6: nova→velo (dir rename + types + runtime.nova) → cargo check -p kvbm-connector
```

### Anti-Patterns to Avoid

- **Mixing passes:** Do not fix a v2:: import while doing a logical:: pass. Each pass is committed separately.
- **Silently wrapping:** If a type isn't found in the expected crate, stop and flag — don't guess.
- **Touching logic:** Only imports and type names change. No algorithmic changes.
- **Forgetting worker/mod.rs pub use:** When renaming `mod nova` to `mod velo`, update `pub use nova::client::ConnectorWorkerClient` to `pub use velo::client::ConnectorWorkerClient`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Find workspace type location | Manual search | Verified mapping table in this document | Already done; table is authoritative |
| Nova event bridge | Custom event relay | `velo::EventHandle` + `messenger.events()` | Exact same API, name change only |
| Worker RPC clients | Custom impl | `VeloWorkerClient` / `VeloWorkerService` from `kvbm_engine::worker` | Already exist as drops-ins |

---

## Common Pitfalls

### Pitfall 1: `self.runtime.nova` Has No Equivalent Field
**What goes wrong:** Trying `self.runtime.nova` on `KvbmRuntime` — compile error because the field is named `messenger`.
**Why it happens:** The old code worked against `dynamo_nova::Nova`; `KvbmRuntime` was already ported to `velo::Messenger`.
**How to avoid:** Always use `self.runtime.messenger()` (the accessor method). Never access `.messenger` (the private field) directly.
**Warning signs:** `error[E0609]: no field 'nova' on type 'Arc<KvbmRuntime>'`

### Pitfall 2: InstanceLeader Builder Method is `.messenger()` Not `.nova()`
**What goes wrong:** `leader_builder.nova(...)` in `connector/leader/init.rs:385` causes compile error.
**Why it happens:** `InstanceLeaderBuilder` takes a `messenger: Arc<Messenger>` — the builder method is `.messenger()`.
**How to avoid:** Replace `.nova(self.runtime.nova.clone())` with `.messenger(self.runtime.messenger().clone())` in Pass 6.

### Pitfall 3: `crate::v2::integrations::common` Is Local, Not kvbm-engine
**What goes wrong:** Assuming `CachedRequestData`, `NewRequestData`, `SchedulerOutput` in `scheduler.rs:19` come from `kvbm_engine`.
**Why it happens:** These types live in `crate::common` (the local `common/` module). They re-export from that module.
**How to avoid:** Replace `crate::v2::integrations::common::*` with `crate::common::*`.

### Pitfall 4: Test Imports in tests.rs Are Phase 4 Scope
**What goes wrong:** Attempting to resolve `crate::v2::testing::connector::{ConnectorTestConfig, TestConnectorInstance}` in Pass 4.
**Why it happens:** `kvbm-engine::testing` has no `connector` submodule — `TestConnectorInstance` and `ConnectorTestConfig` do not exist yet.
**How to avoid:** Gate the entire `tests.rs` content with `#[cfg(all(test, feature = "testing"))]` and mark the import block as a TODO for Phase 4. The test file references at lines 14-17 must be removed or gated.

### Pitfall 5: `dynamo_nova::events::LocalEvent` Exact Type Name in velo
**What goes wrong:** Using wrong type name for `LocalEvent` in `scheduler.rs`.
**Why it happens:** velo may expose this as `velo::events::LocalEvent` or directly `velo::LocalEvent`.
**How to avoid:** After Pass 6 begins, check velo's re-exports for the `EventHandle` and `LocalEvent` types used in `scheduler.rs`. The kvbm-engine codebase uses `velo::EventHandle` directly — use the same pattern.

### Pitfall 6: `TokioRuntime` in pending.rs
**What goes wrong:** `crate::physical::transfer::context::TokioRuntime` may not exist at that path.
**Why it happens:** `kvbm_physical` exports `TransferOptions` and `TransferManager` but `TokioRuntime` is an implementation detail.
**How to avoid:** Search `kvbm_physical::transfer::context` for `TokioRuntime` before assuming path. Alternative: use `runtime.tokio()` directly and adapt the `TransferManagerBuilder` call.

---

## Code Examples

### Confirmed Velo Messenger API (from kvbm-engine/src/worker/velo/client.rs)
```rust
// Source: lib/kvbm-engine/src/worker/velo/client.rs
use ::velo::Messenger;

// Events
let event = self.messenger.events().new_event()?;
let awaiter = self.messenger.events().awaiter(event.handle())?;

// Spawning tasks
self.messenger.tracker().spawn_on(async move { ... }, runtime_handle);

// Instance ID
self.messenger.instance_id()

// Runtime handle
self.messenger.runtime()
```

### Confirmed KvbmRuntime API (from kvbm-engine/src/runtime/mod.rs)
```rust
// Source: lib/kvbm-engine/src/runtime/mod.rs
pub fn messenger(&self) -> &Arc<Messenger>       // use this instead of .nova
pub fn tokio(&self) -> Handle                     // use this instead of runtime.nova.runtime()
pub fn event_system(&self) -> Arc<velo::EventManager>  // use this instead of nova.events().local()
```

### Confirmed InstanceLeader Builder (from kvbm-engine/src/leader/instance.rs)
```rust
// Source: lib/kvbm-engine/src/leader/instance.rs
// .nova() does NOT exist — use .messenger()
let leader = InstanceLeader::builder()
    .messenger(runtime.messenger().clone())   // replaces .nova(self.runtime.nova.clone())
    .registry(registry)
    .g2_manager(g2_manager)
    .workers(worker_clients)
    .with_cached_worker_metadata(worker_metadata)
    .build()?;
```

### Confirmed VeloWorkerService Construction (from kvbm-engine/src/worker/mod.rs)
```rust
// Source: lib/kvbm-engine/src/worker/mod.rs
// NovaWorkerService::new → VeloWorkerService::new, same signature
use kvbm_engine::worker::{VeloWorkerService, VeloWorkerClient};
let service = VeloWorkerService::new(runtime.messenger().clone(), worker)?;
```

### kvbm-engine Worker/Leader Exports Summary
```rust
// Source: lib/kvbm-engine/src/lib.rs + worker/mod.rs + leader/mod.rs
pub use velo::{InstanceId, PeerInfo, WorkerAddress};   // re-exported at crate root

// kvbm_engine::worker exports:
pub use velo::{VeloWorkerClient, VeloWorkerService, VeloWorkerServiceBuilder};
pub use protocol::{LeaderLayoutConfig, WorkerLayoutResponse, ConnectRemoteResponse};
pub use physical::{PhysicalWorker, DirectWorker};

// kvbm_engine::leader exports:
pub use instance::InstanceLeader;
pub use types::{FindMatchesOptions, FindMatchesResult, StagingMode};
pub use onboarding::*;   // includes OnboardingStatus

// kvbm_engine::offload exports:
pub use handle::{TransferHandle, TransferId, TransferResult, TransferStatus};
pub use source::{ExternalBlock, SourceBlock, SourceBlocks};
// OffloadEngine, OffloadEngineBuilder, PendingTracker, PipelineBuilder, create_policy_from_config
// ObjectPresenceFilter, S3PresenceChecker, ObjectPipelineBuilder
```

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (Rust built-in) |
| Config file | none — feature flags control test inclusion |
| Quick run command | `cargo check -p kvbm-connector` |
| Full suite command | `cargo check --workspace` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | Notes |
|--------|----------|-----------|-------------------|-------|
| IMP-01 | v2 imports resolve | compile gate | `cargo check -p kvbm-connector` | Pass after each namespace |
| IMP-02 | distributed imports resolve | compile gate | `cargo check -p kvbm-connector` | |
| IMP-03 | logical imports resolve | compile gate | `cargo check -p kvbm-connector` | |
| IMP-04 | physical imports resolve | compile gate | `cargo check -p kvbm-connector` | |
| IMP-05 | integrations self-refs resolve | compile gate | `cargo check -p kvbm-connector` | |
| VELO-01 | client.rs uses velo types | compile gate | `cargo check -p kvbm-connector` | |
| VELO-02 | service.rs uses velo types | compile gate | `cargo check -p kvbm-connector` | |
| VELO-03 | protocol.rs clean | compile gate | `cargo check -p kvbm-connector` | Already clean — no nova imports |
| VELO-04 | no nova imports remain | grep verification | `grep -r 'dynamo_nova\|nova_backend' lib/kvbm-connector/src/` | Run after Pass 6 |
| VELO-05 | velo dep declared | already satisfied | `cargo check -p kvbm-connector` | `velo = { workspace = true }` already in Cargo.toml |

### Sampling Rate
- **Per pass commit:** `cargo check -p kvbm-connector`
- **Phase gate:** `cargo check --workspace` + grep for remaining nova imports

### Wave 0 Gaps
None — this phase has no test files to create. The existing `connector/worker/tests.rs` must be gated (not deleted) with `#[cfg(all(test, feature = "testing"))]` as a precondition gating, and its broken imports must be commented out with a TODO for Phase 4.

---

## Open Questions

1. **`velo::Handler` exact name for handler registration in service.rs**
   - What we know: kvbm-engine uses `Handler::typed_unary_async` pattern (see velo/mod.rs comment about it)
   - What's unclear: Whether the velo crate exports `Handler` directly or as `velo::Handler` vs a sub-path
   - Recommendation: Before Pass 6, check `use ::velo::...` imports in `kvbm-engine/src/worker/velo/service.rs` for the exact handler builder name

2. **`dynamo_nova::events::LocalEvent` velo equivalent**
   - What we know: `kvbm_engine` uses `velo::EventHandle` for event handles
   - What's unclear: Whether `LocalEvent` in `scheduler.rs` lines 470/502/503 maps to `velo::LocalEvent` or a different type
   - Recommendation: Inspect `velo` crate's public API at the start of Pass 6; the type is used as `Arc<dynamo_nova::events::LocalEvent>` — likely `Arc<velo::events::LocalEvent>`

3. **`CacheLayout` and `ModelExecutorBackend` in kvbm-common**
   - What we know: `config.rs` imports `crate::v2::{CacheLayout, ModelExecutorBackend}`
   - What's unclear: These are not visible in `kvbm-common/src/lib.rs` from the quick check — they may live in a sub-module
   - Recommendation: Before Pass 4, run `grep -rn 'CacheLayout\|ModelExecutorBackend' lib/kvbm-common/src/` to confirm the path

4. **`TokioRuntime` in kvbm_physical**
   - What we know: `pending.rs` imports `crate::physical::transfer::context::TokioRuntime`
   - What's unclear: Whether `kvbm_physical::transfer::context::TokioRuntime` exists
   - Recommendation: Before Pass 2, check `lib/kvbm-physical/src/transfer/` for a `context` module

---

## Sources

### Primary (HIGH confidence)
- Direct source scan of `lib/kvbm-connector/src/` — every broken import listed verbatim
- `lib/kvbm-engine/src/runtime/mod.rs` — confirmed `KvbmRuntime` has `messenger()` not `nova()`
- `lib/kvbm-engine/src/worker/mod.rs` — confirmed `VeloWorkerClient`, `VeloWorkerService` exports
- `lib/kvbm-engine/src/worker/velo/mod.rs` — confirmed `::velo::Messenger` API patterns
- `lib/kvbm-engine/src/worker/velo/client.rs` — confirmed `messenger.events()`, `.tracker()`, `.runtime()`
- `lib/kvbm-engine/src/leader/instance.rs` — confirmed `InstanceLeaderBuilder::messenger()`
- `lib/kvbm-engine/src/lib.rs` — confirmed `pub use velo::{InstanceId, PeerInfo, WorkerAddress}`
- `lib/kvbm-engine/src/offload/mod.rs` — confirmed `TransferHandle`, `ExternalBlock`, `OffloadEngine` exports
- `lib/kvbm-logical/src/lib.rs` — confirmed `blocks::*`, `manager::*`, `registry::*` exports
- `lib/kvbm-physical/src/lib.rs` — confirmed `layout::*`, `transfer::*` exports
- `lib/kvbm-connector/Cargo.toml` — confirmed `velo`, `velo-common` already declared; VELO-05 already satisfied

### Secondary (MEDIUM confidence)
- `lib/kvbm-engine/src/testing/mod.rs` — confirmed `TestConnectorInstance`/`ConnectorTestConfig` do NOT exist, supporting Phase 4 deferral for test imports

## Metadata

**Confidence breakdown:**
- Pass 1-3 mappings: HIGH — directly verified against workspace crate lib.rs files
- Pass 4-5 mappings: HIGH — verified for most types; 3 open questions for edge cases
- Pass 6 nova→velo: HIGH — KvbmRuntime API fully verified; handler builder name is MEDIUM pending velo service.rs check
- Test gating recommendation: HIGH — TestConnectorInstance confirmed absent from kvbm-engine

**Research date:** 2026-03-11
**Valid until:** 2026-04-10 (stable workspace, 30 days)
