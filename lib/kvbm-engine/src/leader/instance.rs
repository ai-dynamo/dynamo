// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::SystemTime;

use ::velo::Messenger;
use anyhow::Result;
use dashmap::DashMap;
use tokio::sync::{Mutex, watch};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use std::sync::{Arc, OnceLock};

use kvbm_config::DisaggregationRole;
use kvbm_protocols::control::{
    ControlError, HostInfo, InstanceDescription, LayoutDescription, ModuleId, TierCapacity,
    TierKind, WorkerInfo,
};

use crate::{
    BlockId, G2, G3, InstanceId, SequenceHash,
    object::ObjectBlockOps,
    p2p::{
        RemoteBlockSet,
        session::{SessionFactory, SessionManager},
    },
    worker::RemoteDescriptor,
};
use kvbm_common::LogicalLayoutHandle;
use kvbm_logical::{
    blocks::{BlockRegistry, ImmutableBlock},
    manager::BlockManager,
};
use kvbm_observability::SharedKvbmObservability;
use kvbm_physical::transfer::{TransferCompleteNotification, TransferOptions};

use kvbm_physical::manager::SerializedLayout;

use super::{
    super::worker::Worker,
    super::worker::group::{ParallelWorkers, SpmdParallelWorkers},
    AsyncSessionResult, BlockHolder, FindMatchesOptions, FindMatchesResult, Leader,
    MetadataTransport, OnboardingStatus, ReadyResult, SessionId,
    accessor::{BlockAccessor, PolicyContext},
    consolidator::{ConsolidatorCell, ConsolidatorParams, new_cell, spawn_into_cell},
    discovery::RemoteDiscoveryHandle,
    dispatch::{PullRef, WirePullOptions, plan_pull},
    remote_search,
    parallelism::{ParallelismTemplate, stamp_parallelism_descriptors},
    stage_g3_to_g2,
    velo::{ExportMetadataCallback, VeloLeaderService},
};

/// Primary leader implementation for the distributed KVBM system.
///
/// `InstanceLeader` coordinates block onboarding across local and remote
/// instances. It owns a G2 (host memory) `BlockManager` and an optional G3
/// (disk) `BlockManager`, a set of workers for executing physical transfers,
/// and a parallel worker abstraction for multi-rank RDMA operations.
///
/// Key responsibilities:
/// - **Block matching**: finding which requested sequence hashes are already
///   cached locally (via `BlockAccessor` policies).
/// - **Block matching**: resolving which requested sequence hashes are cached
///   locally across the G2/G3 tiers via `find_matches`.
/// - **Remote connectivity**: exchanging serialized layout metadata with peer
///   instances so workers can perform RDMA transfers.
/// - **Velo RPC**: registering handlers via `VeloLeaderService` so remote
///   leaders can initiate sessions and exchange metadata.
#[derive(Clone)]
pub struct InstanceLeader {
    /// Velo instance for distributed communication.
    messenger: Arc<Messenger>,

    /// Full Velo handle when available. Required for the
    /// `core/register_leader` control handler's
    /// `velo.discover_and_register_peer` path, which fans out to both
    /// the messenger registry and the streaming transport registry.
    /// `messenger.discover_and_register_peer` only handles the
    /// messenger side, so without this the streaming-transport
    /// registry stays empty and `attach_anchor` fails with
    /// "TCP streaming: peer <id> not registered". Optional because
    /// some test/bench paths build a leader with just a messenger.
    velo: Option<Arc<velo::Velo>>,

    /// Block registry for deduplication.
    #[allow(dead_code)]
    pub(crate) registry: BlockRegistry,

    /// G2 (host memory) block manager (wrapped in Arc since BlockManager doesn't implement Clone).
    pub(crate) g2_manager: Arc<BlockManager<G2>>,

    /// Optional G3 (disk) block manager
    pub(crate) g3_manager: Option<Arc<BlockManager<G3>>>,

    /// Workers for executing transfers (at least 1 required).
    /// Multiple workers enable parallel transfers and redundancy.
    workers: Vec<Arc<dyn Worker>>,

    /// Parallel worker abstraction wrapping the workers.
    /// Used for RDMA transfers with proper handle mapping storage.
    parallel_worker: Option<Arc<dyn ParallelWorkers>>,

    /// Cached worker metadata (avoids querying workers repeatedly).
    cached_worker_metadata: Option<Vec<SerializedLayout>>,

    /// Map of session states for holding blocks alive (RAII).
    ///
    /// Populated by the local-staging `AsyncSession` path in `find_matches`
    /// and cleared via [`InstanceLeader::release_session`].
    session_states: Arc<DashMap<SessionId, SessionState>>,

    /// List of remote leader instance IDs (mutable for post-construction configuration).
    remote_leaders: Arc<std::sync::RwLock<Vec<InstanceId>>>,

    /// Client for the `kvbm.leader.export_metadata` RPC, used to import a peer
    /// leader's worker layout metadata before an RDMA pull.
    transport: Arc<MetadataTransport>,

    // ========================================================================
    // G4/Object Storage
    // ========================================================================
    /// Object storage client for G4 search and load operations.
    /// Leader calls has_blocks on S3 directly, coordinates workers for get_blocks.
    object_client: Option<Arc<dyn ObjectBlockOps>>,

    // ========================================================================
    // Cross-parallelism metadata
    // ========================================================================
    /// Parallelism template used to stamp [`ParallelismDescriptor`]s onto
    /// per-worker [`SerializedLayout`] payloads before forwarding to peer
    /// leaders. When `None`, the export callback returns the raw worker
    /// metadata unchanged — preserving pre-AB-1a behaviour for callers
    /// that have not yet configured cross-parallelism.
    parallelism_template: Option<ParallelismTemplate>,

    /// The block-layout mode this leader operates in (Operational or
    /// Universal). Stored at build time from
    /// [`InstanceLeaderBuilder::block_layout_mode`] so [`Self::describe`] can
    /// include the same mode in the `layout_compat` payload that
    /// [`register_with_hub`] emitted at register time — matching by
    /// construction instead of re-deriving from exports.
    block_layout_mode: kvbm_common::BlockLayoutMode,

    // ========================================================================
    // Disagg control plane
    // ========================================================================
    /// Keeps disagg sessions opened by the control plane's `transfer` module
    /// alive until their lifecycle ends.
    session_manager: Arc<SessionManager>,

    /// The disagg `SessionFactory`, injected post-construction via
    /// [`InstanceLeader::set_session_factory`]. Empty until the connector
    /// builds the factory (which itself holds an `Arc<InstanceLeader>`, so it
    /// cannot exist at `InstanceLeader` build time). The control plane's
    /// `transfer` module reads it at RPC-invocation time, by which point a
    /// remote client could only have connected after full init.
    session_factory: Arc<OnceLock<Arc<dyn SessionFactory>>>,

    // ========================================================================
    // Describe state (Phase C)
    // ========================================================================
    /// Disaggregation role this leader plays — `None` for standalone. Set at
    /// construction time from `KvbmConfig::disagg.as_ref().map(|d| d.role)`.
    role: Option<DisaggregationRole>,

    /// Process start time. Captured at construction; surfaced via
    /// [`Self::describe`].
    started_at: SystemTime,

    /// Stringified hub instance id, injected post-construction via
    /// [`Self::set_hub_instance_id`] when the connector successfully
    /// registers with the hub. Empty for standalone leaders or before
    /// hub registration completes.
    hub_instance_id: Arc<OnceLock<String>>,

    /// Opaque JSON of the leader's `KvbmConfig`, injected post-construction
    /// via [`Self::set_config_blob`]. The connector serialises its
    /// `KvbmRuntime::config()` and stores the result here so `describe`
    /// can surface it to the hub UI without `kvbm-protocols` depending on
    /// `kvbm-config`. First-write-wins; subsequent calls are no-ops.
    config_blob: Arc<OnceLock<serde_json::Value>>,

    /// Modules enabled on this leader's control plane. Injected
    /// post-construction by the connector via [`Self::set_modules`] after
    /// `ControlPlaneBuilder::register` returns. The leader cannot fetch
    /// this from the control plane directly because `ControlPlane` is
    /// built after `InstanceLeader` and isn't held inside it. Empty until
    /// set; surfaced via [`Self::describe`].
    modules: Arc<OnceLock<Vec<ModuleId>>>,

    /// Shared Prometheus registry + metric handles for this leader, as
    /// owned by [`crate::KvbmRuntime`]. Optional because some test paths
    /// (e.g. `testing/distributed.rs`) build a bare leader without a
    /// runtime. The `metrics` control module reads this; when `None`,
    /// the module is not registered.
    observability: Option<SharedKvbmObservability>,

    /// When true, host (G2) is bypassed: disk hits are returned directly as
    /// G3 blocks for G3→G1 transfer instead of staging through G2. Driven by
    /// `kvbm_config::CacheConfig::bypass_host_cache()` at init time.
    pub(crate) bypass_host: bool,

    // ========================================================================
    // In-process consolidator (optional)
    // ========================================================================
    /// Live in-process consolidator, owned by this leader.
    ///
    /// Populated post-construction via [`Self::with_consolidator`];  starts
    /// empty.  Backed by `Arc<OnceLock<…>>` so the first call wins and the
    /// guard is kept alive until the last `InstanceLeader` clone is dropped
    /// (at which point the guard's `Drop` fires and shuts the background tasks
    /// down without blocking the caller).
    consolidator: ConsolidatorCell,

    // ========================================================================
    // Remote search (hub-indexer discovery + transfer-plane pull)
    // ========================================================================
    /// Optional remote-block discovery (the hub's KV indexer), injected
    /// post-construction by the connector via [`Self::set_remote_discovery`]
    /// — the indexer client/peer resolver only exist after hub registration,
    /// which happens after the leader is built (same lifetime problem as
    /// [`Self::set_session_factory`]). When set,
    /// [`find_matches_with_options`](Self::find_matches_with_options) with
    /// `search_remote` drives a transfer-plane remote pull for the
    /// locally-uncached prefix; when unset, no remote search runs.
    remote_discovery: Arc<OnceLock<RemoteDiscoveryHandle>>,

    /// Block-count threshold for remote search: a search is issued only when
    /// the number of remaining locally-uncached full blocks **exceeds** this
    /// value. Derived from `RemoteSearch::min_remote_blocks(block_size)` at
    /// build time.
    min_remote_blocks: usize,
}

/// Builder for InstanceLeader.
#[derive(Default)]
pub struct InstanceLeaderBuilder {
    messenger: Option<Arc<Messenger>>,
    velo: Option<Arc<velo::Velo>>,
    registry: Option<BlockRegistry>,
    g2_manager: Option<Arc<BlockManager<G2>>>,
    g3_manager: Option<Arc<BlockManager<G3>>>,
    workers: Vec<Arc<dyn Worker>>,
    remote_leaders: Option<Vec<InstanceId>>,
    cached_worker_metadata: Option<Vec<SerializedLayout>>,
    object_client: Option<Arc<dyn ObjectBlockOps>>,
    parallelism_template: Option<ParallelismTemplate>,
    role: Option<DisaggregationRole>,
    observability: Option<SharedKvbmObservability>,
    bypass_host: bool,
    block_layout_mode: kvbm_common::BlockLayoutMode,
    min_remote_blocks: usize,
}

impl InstanceLeaderBuilder {
    /// Initialize builder with components from KvbmRuntime.
    ///
    /// This extracts Velo from the runtime. Use this when the runtime
    /// has already been constructed and you want the leader to share
    /// the same Velo instance for distributed communication.
    ///
    /// # Example
    /// ```ignore
    /// let runtime = KvbmRuntime::from_env_leader().await?;
    /// let leader = InstanceLeaderBuilder::default()
    ///     .with_runtime(&runtime)
    ///     .g2_manager(g2_manager)
    ///     .build()?;
    /// ```
    pub fn with_runtime(self, runtime: &crate::KvbmRuntime) -> Self {
        let mut b = self
            .messenger(runtime.messenger().clone())
            .observability(runtime.observability().clone());
        if let Some(v) = runtime.velo() {
            b = b.velo(v.clone());
        }
        b
    }

    pub fn messenger(mut self, messenger: Arc<Messenger>) -> Self {
        self.messenger = Some(messenger);
        self
    }

    pub fn velo(mut self, velo: Arc<velo::Velo>) -> Self {
        self.velo = Some(velo);
        self
    }

    pub fn registry(mut self, registry: BlockRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    pub fn with_g2_manager(mut self, manager: Option<BlockManager<G2>>) -> Self {
        self.g2_manager = manager.map(Arc::new);
        self
    }

    pub fn with_g3_manager(mut self, manager: Option<BlockManager<G3>>) -> Self {
        self.g3_manager = manager.map(Arc::new);
        self
    }

    pub fn g2_manager(mut self, manager: Arc<BlockManager<G2>>) -> Self {
        self.g2_manager = Some(manager);
        self
    }

    pub fn g3_manager(mut self, manager: Arc<BlockManager<G3>>) -> Self {
        self.g3_manager = Some(manager);
        self
    }

    /// Add a single worker (convenience method).
    pub fn worker(mut self, worker: Arc<dyn Worker>) -> Self {
        self.workers.push(worker);
        self
    }

    /// Set all workers at once.
    pub fn workers(mut self, workers: Vec<Arc<dyn Worker>>) -> Self {
        self.workers = workers;
        self
    }

    pub fn remote_leaders(mut self, leaders: Vec<InstanceId>) -> Self {
        self.remote_leaders = Some(leaders);
        self
    }

    /// Cache worker metadata upfront to avoid querying workers later.
    ///
    /// This is useful when workers have already exported metadata during initialization
    /// (e.g., in the connector pattern where workers return metadata in their init response).
    pub fn with_cached_worker_metadata(mut self, metadata: Vec<SerializedLayout>) -> Self {
        self.cached_worker_metadata = Some(metadata);
        self
    }

    /// Mark this leader as running in host-bypass mode. When set, disk hits
    /// are returned as G3 blocks for direct G3→G1 onboarding instead of being
    /// staged through G2. Set this when the cache config has
    /// `bypass_host_cache() == true`.
    pub fn bypass_host(mut self, bypass: bool) -> Self {
        self.bypass_host = bypass;
        self
    }

    /// Set the object storage client for G4 search and load operations.
    ///
    /// The leader uses this client to:
    /// - Query S3 for block presence via `has_blocks`
    /// - Coordinate workers to load blocks from S3 via `get_blocks`
    pub fn object_client(mut self, client: Arc<dyn ObjectBlockOps>) -> Self {
        self.object_client = Some(client);
        self
    }

    /// Set the parallelism template used to stamp [`ParallelismDescriptor`]s
    /// onto per-worker metadata exported to peer leaders. When unset, the
    /// export RPC returns raw worker metadata (pre-AB-1a behaviour); the
    /// peer's cross-parallelism dispatcher then falls back to the symmetric
    /// path.
    pub fn parallelism_template(mut self, template: ParallelismTemplate) -> Self {
        self.parallelism_template = Some(template);
        self
    }

    /// Set the block-layout compatibility policy applied at
    /// `connect_remote`. Defaults to
    /// [`kvbm_common::BlockLayoutMode::Operational`] (strict per-worker
    /// equality, the pre-existing behaviour). Sourced from
    /// `KvbmConfig.block_layout` at the builder call site.
    pub fn block_layout_mode(mut self, mode: kvbm_common::BlockLayoutMode) -> Self {
        self.block_layout_mode = mode;
        self
    }

    /// Set the disaggregation role this leader plays. Surfaced via
    /// [`InstanceLeader::describe`] for the hub UI. Defaults to `None`
    /// (standalone — not part of a P/D split).
    pub fn role(mut self, role: DisaggregationRole) -> Self {
        self.role = Some(role);
        self
    }

    /// Set the shared KVBM observability handle (Prometheus registry +
    /// metric collectors) for this leader. Sourced from
    /// [`crate::KvbmRuntime::observability`]; populated by
    /// [`Self::with_runtime`]. The `metrics` control module reads this;
    /// callers that don't need that module can leave it unset.
    pub fn observability(mut self, observability: SharedKvbmObservability) -> Self {
        self.observability = Some(observability);
        self
    }

    /// Set the remote-search block-count threshold. A search is issued only
    /// when the number of remaining locally-uncached full blocks exceeds this.
    pub fn min_remote_blocks(mut self, n: usize) -> Self {
        self.min_remote_blocks = n;
        self
    }

    pub fn build(self) -> Result<InstanceLeader> {
        let messenger = self
            .messenger
            .ok_or_else(|| anyhow::anyhow!("Velo instance required"))?;
        let transport = Arc::new(MetadataTransport::new(messenger.clone()));

        // Create event system for notification aggregation
        let events = Arc::new(messenger.event_manager());

        // Get current tokio runtime handle
        let runtime = tokio::runtime::Handle::current();

        // // Validate at least one worker
        // if self.workers.is_empty() {
        //     anyhow::bail!("At least one worker required");
        // }

        // todo: we will need a common builder pattern for creating "general" parallel workers
        // - we could also use an enum and match as the number of types will be limited

        // Create parallel worker if workers are provided. When a
        // parallelism template is configured (AB-1a step 2), install
        // it on the SPMD layer so connect_remote can run cross-leader
        // compatibility gates (AB-1b). The cached worker metadata is
        // forwarded so the block-layout compat check in connect_remote
        // can compare against this leader's actual per-worker
        // SerializedLayouts.
        let parallel_worker: Option<Arc<dyn ParallelWorkers>> = if !self.workers.is_empty() {
            let mut spmd =
                SpmdParallelWorkers::new(self.workers.to_vec(), events.clone(), runtime.clone())
                    .with_block_layout_mode(self.block_layout_mode);
            if let Some(template) = self.parallelism_template.clone() {
                spmd = spmd.with_local_template(template);
            }
            if let Some(metadata) = self.cached_worker_metadata.clone() {
                spmd = spmd.with_local_metadata(metadata);
            }
            Some(Arc::new(spmd))
        } else {
            None
        };

        Ok(InstanceLeader {
            messenger,
            velo: self.velo,
            registry: self
                .registry
                .ok_or_else(|| anyhow::anyhow!("block registry required"))?,
            g2_manager: self
                .g2_manager
                .ok_or_else(|| anyhow::anyhow!("g2_manager required"))?,
            g3_manager: self.g3_manager,
            workers: self.workers,
            parallel_worker,
            cached_worker_metadata: self.cached_worker_metadata,
            session_states: Arc::new(DashMap::new()),
            remote_leaders: Arc::new(std::sync::RwLock::new(
                self.remote_leaders.unwrap_or_default(),
            )),
            transport,
            object_client: self.object_client,
            parallelism_template: self.parallelism_template,
            block_layout_mode: self.block_layout_mode,
            session_manager: SessionManager::with_default_watchdog(runtime),
            session_factory: Arc::new(OnceLock::new()),
            role: self.role,
            started_at: SystemTime::now(),
            hub_instance_id: Arc::new(OnceLock::new()),
            config_blob: Arc::new(OnceLock::new()),
            modules: Arc::new(OnceLock::new()),
            observability: self.observability,
            bypass_host: self.bypass_host,
            consolidator: new_cell(),
            remote_discovery: Arc::new(OnceLock::new()),
            min_remote_blocks: self.min_remote_blocks,
        })
    }
}

/// Internal session state for holding matched blocks.
#[allow(dead_code)] // Used for RAII block lifetime management
struct SessionState {
    session_id: SessionId,
    matched_g2_blocks: Vec<ImmutableBlock<G2>>,
    matched_g3_blocks: Vec<ImmutableBlock<G3>>,
    status_tx: watch::Sender<OnboardingStatus>,
    /// Cancellation for a background remote-search driver bound to this
    /// session. `None` for sessions without a driver. [`release_session`]
    /// fires it so a cancelled/preempted request tears the driver down
    /// (and the driver dispatches the remote `close_session`).
    cancel: Option<CancellationToken>,
}

/// Result of scanning for blocks across tiers.
///
/// Unlike `FindMatchesResult`, this scans all given hashes without stopping on first miss.
/// Returns blocks found in each tier along with their sorted positions.
pub struct ScanBlocksResult {
    /// Blocks found in G2 (host memory).
    pub g2_blocks: HashMap<SequenceHash, ImmutableBlock<G2>>,

    /// Blocks found in G3 (disk).
    pub g3_blocks: HashMap<SequenceHash, ImmutableBlock<G3>>,

    /// All found blocks sorted by position (lowest to highest).
    /// Each entry indicates which tier (G2/G3) the block was found in.
    pub sorted_matches: Vec<(SequenceHash, LogicalLayoutHandle)>,
}

impl InstanceLeader {
    /// Get a reference to the G2 BlockManager.
    pub fn g2_manager(&self) -> &Arc<BlockManager<G2>> {
        &self.g2_manager
    }

    /// Get a reference to the optional G3 BlockManager.
    pub fn g3_manager(&self) -> Option<&Arc<BlockManager<G3>>> {
        self.g3_manager.as_ref()
    }

    /// Get the block registry.
    pub fn registry(&self) -> &BlockRegistry {
        &self.registry
    }

    /// Get a reference to the Velo instance.
    ///
    /// This provides access to the Velo distributed system for features
    /// like event coordination and cross-instance communication.
    pub fn messenger(&self) -> &Arc<Messenger> {
        &self.messenger
    }

    /// Optional full Velo handle. Use for paths that need both messenger
    /// and streaming-transport peer registration (e.g.
    /// `velo.discover_and_register_peer`). `None` when the leader was
    /// built from a bare messenger (some test/bench paths).
    pub fn velo(&self) -> Option<&Arc<velo::Velo>> {
        self.velo.as_ref()
    }

    /// Get the tokio runtime handle from Velo.
    ///
    /// This handle should be used for spawning background tasks that need to
    /// run on the KVBM runtime's executor (e.g., offload engine pipelines).
    pub fn runtime(&self) -> tokio::runtime::Handle {
        self.messenger.runtime().clone()
    }

    /// Check if a parallel_worker is configured.
    ///
    /// The parallel_worker is required for local transfer operations
    /// (e.g., offloading blocks between tiers).
    pub fn has_parallel_worker(&self) -> bool {
        self.parallel_worker.is_some()
    }

    /// Get the parallel worker for distributed operations.
    ///
    /// The parallel worker fans out operations to all workers and aggregates results.
    /// It implements `ObjectBlockOps` for coordinated object storage uploads.
    pub fn parallel_worker(&self) -> Option<Arc<dyn ParallelWorkers>> {
        self.parallel_worker.clone()
    }

    /// Get the object storage client for G4 operations.
    ///
    /// Returns `Some` if object storage is configured, `None` otherwise.
    /// Reserved for the parked G4 search machinery in [`crate::leader::search`].
    pub fn object_client(&self) -> Option<Arc<dyn ObjectBlockOps>> {
        self.object_client.clone()
    }

    /// Add a remote leader to the search list.
    ///
    /// Tracks peer leader instance IDs for the upcoming async remote-search
    /// path. (Peer-leader search is not currently wired into
    /// `find_matches_with_options`.) Allows adding remote leaders after
    /// construction (e.g., when instance IDs are only known after cluster
    /// setup).
    pub fn add_remote_leader(&self, instance_id: InstanceId) {
        let mut remote_leaders = self.remote_leaders.write().unwrap();
        if !remote_leaders.contains(&instance_id) {
            remote_leaders.push(instance_id);
        }
    }

    /// Set all remote leaders at once.
    pub fn set_remote_leaders(&self, instance_ids: Vec<InstanceId>) {
        let mut remote_leaders = self.remote_leaders.write().unwrap();
        *remote_leaders = instance_ids;
    }

    /// Get the list of remote leader instance IDs.
    pub fn remote_leaders(&self) -> Vec<InstanceId> {
        self.remote_leaders.read().unwrap().clone()
    }

    /// The disagg [`SessionManager`] — keeps control-plane-opened sessions
    /// alive until their lifecycle ends.
    pub fn session_manager(&self) -> &Arc<SessionManager> {
        &self.session_manager
    }

    /// Shared KVBM observability handle (Prometheus registry + collectors).
    ///
    /// `Some` when the leader was built via [`InstanceLeaderBuilder::with_runtime`]
    /// (or [`InstanceLeaderBuilder::observability`] directly). `None` for bare
    /// test leaders that bypass the runtime. The control plane's `metrics`
    /// module reads this; when `None`, the module is not registered.
    pub fn observability(&self) -> Option<&SharedKvbmObservability> {
        self.observability.as_ref()
    }

    /// A clonable handle to the disagg `SessionFactory` cell.
    ///
    /// The cell is populated post-construction via [`set_session_factory`].
    /// The control plane's `transfer` module holds this handle and reads the
    /// factory lazily, at RPC-invocation time.
    ///
    /// [`set_session_factory`]: InstanceLeader::set_session_factory
    pub fn session_factory_cell(&self) -> Arc<OnceLock<Arc<dyn SessionFactory>>> {
        Arc::clone(&self.session_factory)
    }

    /// Inject the disagg `SessionFactory` once it has been built.
    ///
    /// Idempotent: a second call is a no-op (the factory is built once during
    /// connector init). Returns whether this call set the value.
    pub fn set_session_factory(&self, factory: Arc<dyn SessionFactory>) -> bool {
        self.session_factory.set(factory).is_ok()
    }

    /// Inject the remote-block discovery seam (the hub's KV indexer, wrapped by
    /// the connector) once the hub registration that backs it has completed.
    ///
    /// Idempotent: first-write-wins. Returns whether this call set the value.
    /// Enables the transfer-plane remote-search pull in
    /// [`find_matches_with_options`](Self::find_matches_with_options) when
    /// `search_remote` is requested.
    pub fn set_remote_discovery(&self, discovery: RemoteDiscoveryHandle) -> bool {
        self.remote_discovery.set(discovery).is_ok()
    }

    // ========================================================================
    // Transfer control-plane methods
    //
    // Substantive logic lives in `leader/control/modules/transfer.rs`; these
    // methods are the public entry points used by both the velo handler
    // shims and any in-process caller. See the plan at
    // ~/.claude/plans/control-transfer-v1.md for the full design.
    // ========================================================================

    /// HOLDER-SIDE. Open a transfer session populated by a multi-tier
    /// search. v1 implements G2 only; G3/G4 are wired in subsequent
    /// phases via the populator's stage phase.
    ///
    /// `find_mode = Sync` awaits the find phase; the response carries
    /// the matched hashes inline. `find_mode = Async` returns as soon
    /// as the session is opened.
    pub async fn open_transfer_session(
        self: &Arc<Self>,
        req: kvbm_protocols::control::modules::transfer::OpenTransferSessionRequest,
    ) -> Result<
        kvbm_protocols::control::modules::transfer::OpenTransferSessionResponse,
        kvbm_protocols::control::ControlError,
    > {
        crate::leader::control::modules::transfer::open_transfer_session(self, req).await
    }

    /// HOLDER-SIDE. Close a parked transfer session by id. Idempotent.
    pub async fn close_transfer_session(
        self: &Arc<Self>,
        req: kvbm_protocols::control::modules::transfer::CloseTransferSessionRequest,
    ) -> Result<
        kvbm_protocols::control::modules::transfer::CloseTransferSessionResponse,
        kvbm_protocols::control::ControlError,
    > {
        crate::leader::control::modules::transfer::close_transfer_session(self, req).await
    }

    /// PULLER-SIDE. Attach to a transfer session living on
    /// `req.source_instance_id`, drain commits/availability, and pull
    /// the (optionally selected) blocks into this instance's local G2
    /// pool. Long-poll — returns when the pull is complete.
    pub async fn pull_from_session(
        self: &Arc<Self>,
        req: kvbm_protocols::control::modules::transfer::PullFromSessionRequest,
    ) -> Result<
        kvbm_protocols::control::modules::transfer::PullFromSessionResponse,
        kvbm_protocols::control::ControlError,
    > {
        crate::leader::control::modules::transfer::pull_from_session(self, req).await
    }

    // ========================================================================
    // Describe (Phase C)
    // ========================================================================

    /// Inject the leader's `KvbmConfig` serialised as JSON. Surfaced via
    /// [`Self::describe`] under `InstanceDescription::config`.
    ///
    /// First-write-wins (matches [`Self::set_session_factory`] semantics).
    /// Returns `true` if this call stored the value, `false` if the cell
    /// was already populated.
    pub fn set_config_blob(&self, value: serde_json::Value) -> bool {
        self.config_blob.set(value).is_ok()
    }

    /// Inject the hub's instance id post-registration. Surfaced via
    /// [`Self::describe`] under `InstanceDescription::hub_instance_id`.
    /// First-write-wins.
    pub fn set_hub_instance_id(&self, id: InstanceId) -> bool {
        self.hub_instance_id.set(id.to_string()).is_ok()
    }

    /// Inject the list of control-plane modules enabled on this leader.
    /// Called by the connector after `ControlPlaneBuilder::register`
    /// returns. First-write-wins. Surfaced via [`Self::describe`].
    pub fn set_modules(&self, modules: Vec<ModuleId>) -> bool {
        self.modules.set(modules).is_ok()
    }

    // ========================================================================
    // In-process consolidator
    // ========================================================================

    /// Spawn an in-process kvbm-consolidator and attach it to this leader.
    ///
    /// The consolidator subscribes to:
    /// - The `vllm_zmq_endpoint` from `params` for G1 (GPU-side) events, if
    ///   supplied.  Pass `None` to disable ZMQ ingress.
    /// - The `EventsManager` from `params` for G2/G3 KVBM events.
    ///
    /// It publishes deduplicated, kv-router-compatible events (u64 wire
    /// format) on `params.egress_endpoint`.
    ///
    /// The engine owns the resulting [`Consolidator`] for its entire lifetime.
    /// Background tasks are cancelled and joined when the last clone of this
    /// `InstanceLeader` is dropped, without blocking the caller's thread.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `with_consolidator` has already been called on this leader
    ///   (idempotency guard).
    /// - The ZMQ publisher or subscriber socket cannot be bound/connected.
    ///
    /// # Why async?
    ///
    /// `ConsolidatorBuilder::build()` internally awaits the ZMQ socket
    /// bind/connect operations.  There is no sync path available.
    pub async fn with_consolidator(&self, params: ConsolidatorParams) -> anyhow::Result<()> {
        spawn_into_cell(&self.consolidator, params).await
    }

    /// Return a handle for direct event injection into the consolidator.
    ///
    /// Returns `None` if `with_consolidator` has not been called yet or if
    /// the consolidator has already been shut down.
    pub fn consolidator_handle(&self) -> Option<kvbm_consolidator::ConsolidatorHandle> {
        self.consolidator.get()?.handle()
    }

    /// Explicitly shut down the consolidator and await background tasks.
    ///
    /// This is the deterministic counterpart to `ConsolidatorGuard::Drop`.
    /// Callers who need to know shutdown has completed (ZMQ sockets unbound,
    /// background tasks exited) before proceeding should call this and await
    /// its completion. Subsequent calls — and any later `Drop` — are no-ops.
    ///
    /// Returns `true` if a consolidator was running and was shut down;
    /// `false` if no consolidator was ever started, or it was already
    /// shut down by a prior call.
    pub async fn shutdown_consolidator(&self) -> bool {
        let Some(guard) = self.consolidator.get() else {
            return false;
        };
        match guard.take() {
            Some(c) => {
                c.shutdown().await;
                true
            }
            None => false,
        }
    }

    /// Get the disaggregation role this leader plays, if any.
    pub fn role(&self) -> Option<DisaggregationRole> {
        self.role
    }

    /// Build a structured topology snapshot of this leader.
    ///
    /// **Lifecycle:** in steady state the connector pushes this payload to the
    /// hub via `HubClient::push_describe` after `set_config_blob` and after
    /// `set_hub_instance_id`. The hub may also fall back to pulling this
    /// snapshot via the [`DESCRIBE_INSTANCE_HANDLER`] velo handler when its
    /// cache is cold.
    ///
    /// **Pre-stamping behaviour:** if workers have not yet stamped their
    /// layouts, `describe` returns `Ok(InstanceDescription)` with empty
    /// `workers`, empty `tier_capacity`, `block_size: None`, and
    /// `parallelism: None`. The identity / capability / process fields
    /// (`instance_id`, `worker_ids`, `modules`, `role`, `host`, `started_at`)
    /// are always populated. Callers decide whether to wait for stamping
    /// before pushing.
    ///
    /// [`DESCRIBE_INSTANCE_HANDLER`]: kvbm_protocols::control::DESCRIBE_INSTANCE_HANDLER
    pub async fn describe(&self) -> Result<InstanceDescription, ControlError> {
        use super::describe_map::{
            to_disagg_role, to_layout_config_description, to_parallelism_description,
            to_storage_kind_description, to_tier_kind,
        };
        use super::layout_compat::build_layout_compat_payload_with_template;

        let exports: Vec<SerializedLayout> = self
            .assemble_export_metadata()
            .await
            .map_err(|e| ControlError::Internal(format!("assemble_export_metadata: {e:#}")))?;

        let mut workers: Vec<WorkerInfo> = Vec::with_capacity(exports.len());
        for s in &exports {
            let unpacked = s
                .unpack()
                .map_err(|e| ControlError::Internal(format!("unpack SerializedLayout: {e:#}")))?;

            // Honest `None` when the worker carries no stamped descriptor.
            // Never synthesise a `Some(1x1)` placeholder — that would lie
            // about topology for a multi-worker TP leader pre-stamping (or
            // for any leader built without a `ParallelismTemplate`).
            let parallelism = unpacked
                .parallelism
                .as_ref()
                .map(to_parallelism_description);

            let layouts: Vec<LayoutDescription> = unpacked
                .layouts
                .iter()
                .map(|ld| {
                    let cfg = &ld.layout.layout_config;
                    let bytes_per_block = cfg.bytes_per_block();
                    let block_layout = kv_block_layout_name(&ld.layout.layout_type_details);
                    LayoutDescription {
                        tier: to_tier_kind(ld.logical_type),
                        config: to_layout_config_description(cfg),
                        location: to_storage_kind_description(&ld.layout.location),
                        layout_type: layout_type_name(&ld.layout.layout_type_details).to_owned(),
                        block_layout,
                        bytes_per_block,
                        total_bytes: bytes_per_block.saturating_mul(cfg.num_blocks),
                    }
                })
                .collect();

            workers.push(WorkerInfo {
                worker_id: unpacked.worker_address.worker_id,
                nixl_agent_name: unpacked.worker_address.nixl_agent_name.clone(),
                parallelism,
                layouts,
            });
        }

        let block_size = common_page_size(&workers);
        let parallelism = aggregate_parallelism(&workers);
        let tier_capacity = sum_tier_capacity(&workers);

        // Modules: read whatever the connector injected post-`ControlPlaneBuilder::register`.
        // Empty until `set_modules` has fired — which it always has by the time the
        // connector pushes describe; the fallback-pull path may serve an empty list
        // briefly during a cold restart and that's acceptable.
        let modules = self.modules.get().cloned().unwrap_or_default();

        // Build the layout-compat payload using the same sources as the register-time path
        // (`init.rs` connector): cached_worker_metadata[0] + parallelism_template.
        // If either is absent (standalone leader / no P2P) the field stays None and
        // the hub's describe-push validation is skipped (legacy path).
        let layout_compat = self
            .cached_worker_metadata
            .as_ref()
            .and_then(|v| v.first())
            .and_then(|worker| {
                match build_layout_compat_payload_with_template(
                    self.block_layout_mode,
                    worker,
                    self.parallelism_template.as_ref(),
                ) {
                    Ok(p) => Some(p),
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            mode = ?self.block_layout_mode,
                            "describe: failed to build layout_compat payload; \
                             sending None (hub will skip describe-push validation)"
                        );
                        None
                    }
                }
            });

        Ok(InstanceDescription {
            instance_id: self.messenger.instance_id().to_string(),
            worker_ids: workers.iter().map(|w| w.worker_id).collect(),
            hub_instance_id: self.hub_instance_id.get().cloned(),
            block_size,
            parallelism,
            tier_capacity,
            workers,
            modules,
            role: self.role.map(to_disagg_role),
            config: self.config_blob.get().cloned(),
            host: HostInfo {
                hostname: read_hostname(),
                pid: std::process::id(),
            },
            started_at: self.started_at,
            layout_compat,
        })
    }

    /// Scan for all blocks matching any of the given sequence hashes.
    ///
    /// Unlike `find_matches`, this:
    /// - Does NOT stop on first miss
    /// - Returns blocks from both G2 and G3 tiers separately
    /// - Acquires blocks from pools (caller owns until dropped via RAII)
    /// - Returns `sorted_matches` ordered by `SequenceHash::position()`
    ///
    /// # Arguments
    /// * `sequence_hashes` - Hashes to scan for
    /// * `touch` - Whether to update frequency tracking (for MultiLRU eviction policy)
    ///
    /// # Algorithm
    /// 1. Scan G2 manager for candidates
    /// 2. Scan G3 manager for remaining candidates
    /// 3. Build sorted_matches from both, sorted by position (lowest to highest)
    pub fn scan_blocks(&self, sequence_hashes: &[SequenceHash], touch: bool) -> ScanBlocksResult {
        // Step 1: Scan G2 for all candidates
        let g2_blocks = self.g2_manager.scan_matches(sequence_hashes, touch);

        // Step 2: Find remaining hashes not in G2
        let remaining: Vec<SequenceHash> = sequence_hashes
            .iter()
            .filter(|h| !g2_blocks.contains_key(h))
            .copied()
            .collect();

        // Step 3: Scan G3 for remaining (if G3 exists)
        let g3_blocks = if let Some(ref g3_manager) = self.g3_manager {
            if !remaining.is_empty() {
                g3_manager.scan_matches(&remaining, touch)
            } else {
                HashMap::new()
            }
        } else {
            HashMap::new()
        };

        // Step 4: Build sorted_matches from both tiers
        let mut sorted_matches: Vec<(SequenceHash, LogicalLayoutHandle)> =
            Vec::with_capacity(g2_blocks.len() + g3_blocks.len());

        // Add G2 matches
        for hash in g2_blocks.keys() {
            sorted_matches.push((*hash, LogicalLayoutHandle::G2));
        }

        // Add G3 matches
        for hash in g3_blocks.keys() {
            sorted_matches.push((*hash, LogicalLayoutHandle::G3));
        }

        // Sort by SequenceHash position (lowest to highest)
        sorted_matches.sort_by_key(|(hash, _)| hash.position());

        ScanBlocksResult {
            g2_blocks,
            g3_blocks,
            sorted_matches,
        }
    }

    /// Scan blocks using a custom policy that controls iteration and yields results.
    ///
    /// This provides maximum flexibility for implementing custom scanning strategies.
    /// The policy receives access to a `BlockAccessor` for acquiring blocks and a
    /// `PolicyContext` for yielding results incrementally.
    ///
    /// # Arguments
    /// * `hashes` - Sequence hashes to scan
    /// * `touch` - Whether to update frequency tracking on block access
    /// * `policy` - Function that implements the scanning strategy
    ///
    /// # Design
    ///
    /// The accessor does NOT hold locks between calls. Each `.find()` call is
    /// independent. This enables:
    /// - Custom iteration patterns (sorted, BTree scan, binary search, etc.)
    /// - Yielding results incrementally (e.g., contiguous subsequences)
    /// - Future parallel execution (accessor is Send + Sync)
    ///
    /// # Example: Simple linear scan
    /// ```ignore
    /// let blocks = leader.scan_with_policy(&hashes, true, |hashes, ctx| {
    ///     for hash in hashes {
    ///         if let Some(block) = ctx.accessor().find(*hash) {
    ///             ctx.yield_item(block);
    ///         }
    ///     }
    /// });
    /// ```
    ///
    /// # Example: Find contiguous subsequences
    /// ```ignore
    /// let runs: Vec<Vec<TieredBlock>> = leader.scan_with_policy(&hashes, true, |hashes, ctx| {
    ///     let mut run = Vec::new();
    ///     let mut last_pos: Option<u64> = None;
    ///
    ///     for hash in hashes.iter().sorted_by_key(|h| h.position()) {
    ///         if let Some(block) = ctx.accessor().find(*hash) {
    ///             let pos = block.position();
    ///             if last_pos.map_or(true, |p| pos == p + 1) {
    ///                 run.push(block);
    ///             } else {
    ///                 if !run.is_empty() { ctx.yield_item(std::mem::take(&mut run)); }
    ///                 run.push(block);
    ///             }
    ///             last_pos = Some(pos);
    ///         } else if !run.is_empty() {
    ///             ctx.yield_item(std::mem::take(&mut run));
    ///             last_pos = None;
    ///         }
    ///     }
    ///     if !run.is_empty() { ctx.yield_item(run); }
    /// });
    /// ```
    pub fn scan_with_policy<F, T>(&self, hashes: &[SequenceHash], touch: bool, policy: F) -> Vec<T>
    where
        F: FnOnce(&[SequenceHash], &mut PolicyContext<T>),
    {
        let accessor = BlockAccessor::new(self, touch);
        let mut ctx = PolicyContext {
            accessor,
            results: Vec::new(),
        };
        policy(hashes, &mut ctx);
        ctx.results
    }

    pub fn builder() -> InstanceLeaderBuilder {
        InstanceLeaderBuilder::default()
    }

    /// Assemble the per-worker [`SerializedLayout`] vector that the
    /// `kvbm.leader.export_metadata` RPC handler returns to a peer leader.
    ///
    /// Collects from the cache when present, else queries each worker.
    /// If a [`ParallelismTemplate`] is configured, stamps a
    /// [`ParallelismDescriptor`] onto each per-worker payload via
    /// [`stamp_parallelism_descriptors`]; otherwise returns the raw worker
    /// metadata unchanged (pre-AB-1a shape) and the peer's cross-parallelism
    /// dispatcher falls back to the symmetric path.
    pub async fn assemble_export_metadata(&self) -> Result<Vec<SerializedLayout>> {
        let raw = if let Some(cached) = self.cached_worker_metadata.clone() {
            cached
        } else {
            let mut metadata = Vec::with_capacity(self.workers.len());
            for worker in &self.workers {
                metadata.push(worker.export_metadata()?.await?);
            }
            metadata
        };

        match &self.parallelism_template {
            Some(template) => stamp_parallelism_descriptors(template, raw),
            None => Ok(raw),
        }
    }

    /// Register Velo handlers for leader-to-leader communication.
    ///
    /// This must be called after construction to enable distributed onboarding.
    pub fn register_handlers(&self) -> Result<()> {
        // Create export_metadata callback if we have workers or cached metadata.
        // Delegates to assemble_export_metadata so the stamping logic is
        // testable without Velo plumbing.
        let export_metadata_callback: Option<ExportMetadataCallback> =
            if !self.workers.is_empty() || self.cached_worker_metadata.is_some() {
                let leader = self.clone();
                Some(Arc::new(move || {
                    let leader = leader.clone();
                    Box::pin(async move { leader.assemble_export_metadata().await })
                }))
            } else {
                None
            };

        let mut service = VeloLeaderService::new(self.messenger.clone());

        if let Some(callback) = export_metadata_callback {
            service = service.with_export_metadata(callback);
        }

        service.register_handlers()?;

        Ok(())
    }

    /// Build and register the public leader [`ControlPlane`].
    ///
    /// Distinct from [`register_handlers`](Self::register_handlers), which
    /// wires the engine-internal `VeloLeaderService`. The control plane is
    /// public surface, organized as modules:
    /// - `core` (always-on) — `register_leader`.
    /// - `transfer` (always-on) — G2 search → disagg-session creation. Reads
    ///   the `SessionFactory` lazily from the cell populated by
    ///   [`set_session_factory`](Self::set_session_factory).
    /// - `dev` (opt-in) — `reset`. Safe in production.
    ///
    /// `dev` comes from `control.dev` in `KvbmConfig`. Takes `Arc<Self>`
    /// because the `core` / `dev` modules hold an `Arc<InstanceLeader>`. The
    /// returned [`ControlPlane`] carries only introspection metadata; the
    /// handlers live on the messenger and the modules' captured state outlives
    /// the returned handle.
    pub fn register_control_plane(
        self: &Arc<Self>,
        dev: bool,
        metrics: bool,
    ) -> Result<Arc<crate::leader::ControlPlane>> {
        use crate::leader::control::{
            ControlPlane, CoreModule, DevModule, MetricsModule, TransferModule,
        };

        let mut builder =
            ControlPlane::builder(self.messenger.clone(), self.messenger.instance_id())
                .with_module(CoreModule::new(Arc::clone(self)))
                .with_module(TransferModule::new(Arc::clone(self)));

        if dev {
            builder = builder.with_module(DevModule::new(Arc::clone(self)));
        }
        if metrics {
            match self.observability.as_ref() {
                Some(obs) => {
                    builder =
                        builder.with_module(MetricsModule::new(Arc::clone(self), Arc::clone(obs)));
                }
                None => {
                    tracing::warn!(
                        "control plane `metrics` module requested but no observability \
                         handle was provided to InstanceLeaderBuilder; module not registered"
                    );
                }
            }
        }

        builder.register()
    }

    /// Store session state (held blocks and status channel).
    ///
    /// Blocks are kept alive via RAII until the session is removed from storage.
    fn store_session_state(&self, state: SessionState) {
        self.session_states.insert(state.session_id, state);
    }

    /// Release a completed session, dropping any held blocks.
    ///
    /// This is optional - sessions will naturally be cleaned up when the InstanceLeader
    /// is dropped. Call this explicitly if you need to release blocks earlier.
    pub fn release_session(&self, session_id: SessionId) {
        if let Some((_, state)) = self.session_states.remove(&session_id) {
            // Tear down a bound remote-search driver, if any. The driver's
            // cancel branch best-effort closes the remote transfer session.
            if let Some(token) = &state.cancel {
                token.cancel();
            }
        }
    }

    /// Test-only: is `session_id` registered in the session-state map?
    #[cfg(any(test, feature = "testing"))]
    pub fn has_session(&self, session_id: SessionId) -> bool {
        self.session_states.contains_key(&session_id)
    }

    /// Test-only: insert a sentinel session state so a test can verify that
    /// `release_session` removes it. The held block vectors are empty; the map
    /// entry alone is what the test observes.
    #[cfg(any(test, feature = "testing"))]
    pub fn insert_test_session_marker(&self, session_id: SessionId) {
        let (status_tx, _rx) = watch::channel(OnboardingStatus::Searching);
        self.session_states.insert(
            session_id,
            SessionState {
                session_id,
                matched_g2_blocks: Vec::new(),
                matched_g3_blocks: Vec::new(),
                status_tx,
                cancel: None,
            },
        );
    }
    // ========================================================================
    // RDMA Metadata Management
    // These methods handle layout metadata export/import for remote RDMA transfers.
    // ========================================================================

    /// Check if metadata for a remote instance has been loaded.
    ///
    /// Returns true if `import_remote_metadata` has been successfully called
    /// for the given instance.
    pub fn has_remote_metadata(&self, instance: InstanceId) -> bool {
        self.parallel_worker
            .as_ref()
            .map(|pw| pw.has_remote_metadata(instance))
            .unwrap_or(false)
    }

    /// Get the number of workers attached to this leader.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Export metadata from all workers.
    ///
    /// Returns a `Vec<SerializedLayout>` where each element corresponds to a worker
    /// in rank order. This metadata can be sent to remote instances to enable
    /// RDMA transfers.
    ///
    /// # Returns
    /// Vector of serialized layouts, one per worker
    pub async fn export_worker_metadata(&self) -> Result<Vec<SerializedLayout>> {
        // Return cached metadata if available
        if let Some(cached) = &self.cached_worker_metadata {
            return Ok(cached.clone());
        }

        // Otherwise, query workers
        let mut metadata = Vec::with_capacity(self.workers.len());

        for worker in &self.workers {
            let serialized = worker.export_metadata()?.await?;
            metadata.push(serialized);
        }

        Ok(metadata)
    }

    /// Import metadata from a remote instance's workers.
    ///
    /// This imports layout metadata from a remote instance, enabling RDMA transfers
    /// to pull data from that instance. Metadata is imported rank-by-rank:
    /// - local worker 0 imports remote worker 0's metadata
    /// - local worker 1 imports remote worker 1's metadata
    /// - etc.
    ///
    /// # Arguments
    /// * `remote_instance` - The instance ID of the remote leader
    /// * `metadata` - Vector of SerializedLayout from remote workers (one per worker)
    ///
    /// # Errors
    /// Returns an error if:
    /// - No parallel worker configured
    /// - Metadata was already imported for this instance
    /// - Worker count mismatch between local and remote
    /// - Individual worker metadata import fails
    pub async fn import_remote_metadata(
        &self,
        remote_instance: InstanceId,
        metadata: Vec<SerializedLayout>,
    ) -> Result<()> {
        let parallel_worker = self
            .parallel_worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No parallel worker configured"))?;

        // Idempotent: if already imported, skip. The TP=N asymmetric path can
        // fan out multiple concurrent callers (one per local worker rank) into
        // ensure_remote_metadata → import_remote_metadata for the same
        // remote_instance. The second caller sees has_remote_metadata=true
        // here; returning Ok(()) is correct — the metadata is already loaded
        // and the handle mappings are already populated.
        if parallel_worker.has_remote_metadata(remote_instance) {
            return Ok(());
        }

        // Connect to remote - this imports metadata and stores handle mappings
        parallel_worker
            .connect_remote(remote_instance, metadata)?
            .await?;

        Ok(())
    }

    /// Ensure worker transfer metadata for a remote instance has been imported.
    ///
    /// The leader requests `Vec<SerializedLayout>` from the remote leader's
    /// `kvbm.leader.export_metadata` handler and imports it through the
    /// configured parallel worker. Repeated calls are no-ops once the metadata
    /// has been imported.
    pub async fn ensure_remote_metadata(&self, remote_instance: InstanceId) -> Result<()> {
        if self.has_remote_metadata(remote_instance) {
            return Ok(());
        }

        let metadata = self.transport.request_metadata(remote_instance).await?;
        self.import_remote_metadata(remote_instance, metadata).await
    }

    /// Public cross-parallelism RDMA pull entrypoint (AB-4).
    ///
    /// Resolves `refs` into per-local-worker pull plans via
    /// [`plan_pull`], dispatches each plan to its target local worker,
    /// and awaits all per-rank notifications. Source and destination
    /// layouts are hardcoded to [`LogicalLayoutHandle::G2`] (locked
    /// decision #1 — `T = G2` concrete).
    ///
    /// `refs` must be paired (`src_block_id` in the remote leader's
    /// block-id space, `dst_block_id` in the local leader's). The
    /// caller (typically a session) owns hash→block-id resolution.
    ///
    /// Convenience wrapper around [`Self::rdma_pull_with_opts`] with
    /// default [`WirePullOptions`].
    pub async fn rdma_pull(&self, remote_instance: InstanceId, refs: Vec<PullRef>) -> Result<()> {
        self.rdma_pull_with_opts(remote_instance, refs, WirePullOptions::default())
            .await
    }

    /// As [`Self::rdma_pull`] but takes a caller-supplied
    /// [`WirePullOptions`] (NIXL write notification + metric route).
    ///
    /// Steps:
    ///
    /// 1. Empty `refs` → `Ok(())` immediately (no planning, no RPCs).
    /// 2. Lazily import peer metadata if not already cached.
    /// 3. Look up the cached per-rank
    ///    [`ParallelismDescriptor`] set for `remote_instance`. If
    ///    absent (Legacy unstamped peer), fall back to the legacy
    ///    same-rank-zip dispatch on
    ///    [`crate::worker::group::ParallelWorkers::execute_remote_onboard_for_instance`]
    ///    — backwards compatible with peers that haven't upgraded to
    ///    stamping.
    /// 4. (Strict path only.) Read the local [`ParallelismTemplate`].
    ///    Coherence guard: `template.tp_size == parallel_worker.worker_count()`.
    /// 5. [`plan_pull`] → `Vec<(local_rank, WorkerPullPlan)>`.
    /// 6. Dispatch each plan to `parallel_worker.workers()[local_rank]`
    ///    via [`crate::worker::WorkerTransfers::execute_remote_pull_plan`].
    /// 7. Aggregate per-plan notifications, await.
    ///
    /// Locked decision #5: `plan_pull` is the always-on path *for peers
    /// that stamp descriptors*. The symmetric case is the degenerate
    /// output (one shard per local rank, full extents); the worker
    /// handler routes it through the planner-driven
    /// [`kvbm_physical::manager::TransferManager::execute_transfer_selection`]
    /// like any other plan. This is a behaviour change for stamped
    /// symmetric callers vs the legacy direct-onboard path — under
    /// symmetric + identical layouts the planner still emits a single
    /// `CopyPlan::Direct`, but planning overhead lands on the hot path.
    /// Unstamped peers preserve the pre-AB-4 direct-onboard path.
    pub async fn rdma_pull_with_opts(
        &self,
        remote_instance: InstanceId,
        refs: Vec<PullRef>,
        opts: WirePullOptions,
    ) -> Result<()> {
        if refs.is_empty() {
            return Ok(());
        }

        self.ensure_remote_metadata(remote_instance).await?;

        let parallel_worker = self
            .parallel_worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("rdma_pull: no parallel worker configured"))?;

        // Legacy peers (no stamped descriptors) preserve the pre-AB-4
        // direct-onboard path. `connect_remote`'s rank-count gate
        // enforces same-rank symmetry for them, so the per-worker
        // execute_remote_onboard fan-out is correct.
        let Some(descriptors) = parallel_worker.remote_descriptors_for(remote_instance) else {
            return self
                .rdma_pull_legacy_fallback(parallel_worker.as_ref(), remote_instance, refs, opts)
                .await;
        };

        let template = self.parallelism_template.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "rdma_pull: peer {} has stamped descriptors but no local ParallelismTemplate \
                 is configured; cross-parallelism transfers require \
                 InstanceLeaderBuilder::parallelism_template(...)",
                remote_instance
            )
        })?;

        // Coherence guard: same invariant the asymmetric branch of
        // SpmdParallelWorkers enforces. A template that disagrees with
        // the worker count would produce mis-shaped plans.
        if template.tp_size != parallel_worker.worker_count() {
            anyhow::bail!(
                "rdma_pull: local ParallelismTemplate tp_size ({}) disagrees with worker count ({}); \
                 template must describe the local worker grid",
                template.tp_size,
                parallel_worker.worker_count(),
            );
        }

        let plans = plan_pull(
            &template,
            &descriptors,
            remote_instance,
            LogicalLayoutHandle::G2,
            LogicalLayoutHandle::G2,
            &refs,
            &opts,
        )?;

        if plans.is_empty() {
            // plan_pull skips local ranks with no shards. With refs
            // non-empty and PP=1 every rank participates, so this
            // shouldn't normally happen — but guard against future
            // layer-range filtering quietly producing nothing to do.
            return Ok(());
        }

        let workers = parallel_worker.workers();
        let mut notifications = Vec::with_capacity(plans.len());
        for (local_rank, plan) in plans {
            let worker = workers.get(local_rank).ok_or_else(|| {
                anyhow::anyhow!(
                    "rdma_pull: plan_pull produced a plan for local_rank {local_rank} but only \
                     {} workers are registered",
                    workers.len()
                )
            })?;
            notifications.push(worker.execute_remote_pull_plan(plan)?);
        }

        let events = Arc::new(self.messenger.event_manager());
        let aggregated = TransferCompleteNotification::aggregate(
            notifications,
            &events,
            &tokio::runtime::Handle::current(),
        )?;
        aggregated.await?;
        Ok(())
    }

    /// Fallback path for peers that import via the Legacy (unstamped)
    /// `connect_remote` strategy. Pre-AB-4 every cross-leader pull
    /// took this path; AB-4 only switches stamped peers onto the
    /// planner. Refs are unzipped back into parallel `(src_ids,
    /// dst_ids)` vectors and handed to
    /// `ParallelWorkers::execute_remote_onboard_for_instance`, whose
    /// symmetric branch fans the SAME transfer out to every local
    /// worker (`remote_handles` keyed by local `worker_idx`, which
    /// equals remote rank for legacy peers by the rank-count match
    /// gate in `connect_remote`).
    async fn rdma_pull_legacy_fallback(
        &self,
        parallel_worker: &dyn ParallelWorkers,
        remote_instance: InstanceId,
        refs: Vec<PullRef>,
        opts: WirePullOptions,
    ) -> Result<()> {
        let mut src_block_ids: Vec<BlockId> = Vec::with_capacity(refs.len());
        let mut dst_block_ids: Vec<BlockId> = Vec::with_capacity(refs.len());
        for r in refs {
            src_block_ids.push(r.src_block_id);
            dst_block_ids.push(r.dst_block_id);
        }

        // Project WirePullOptions onto the full TransferOptions for the
        // legacy path. The legacy executor honours nixl_write_notification
        // and metric_route; all other TransferOptions fields default.
        let transfer_opts = TransferOptions {
            nixl_write_notification: opts.nixl_write_notification,
            metric_route: opts.metric_route,
            ..Default::default()
        };

        let notification = parallel_worker.execute_remote_onboard_for_instance(
            remote_instance,
            LogicalLayoutHandle::G2,
            src_block_ids,
            LogicalLayoutHandle::G2,
            Arc::from(dst_block_ids),
            transfer_opts,
        )?;
        notification.await?;
        Ok(())
    }

    /// Pull remote block sets into local G2 block IDs (legacy shim).
    ///
    /// AB-4: this is now a thin wrapper around [`Self::rdma_pull_with_opts`]
    /// — every `RemoteBlockSet` is flattened into combined `PullRef`s
    /// (the prior per-block-set loop), then a single rdma_pull call
    /// dispatches the work through the cross-parallelism planner.
    ///
    /// The notification return type is preserved by spawning the
    /// async pull on the current tokio runtime and wrapping its
    /// future via a fresh velo Event. AB-5 swaps callers to
    /// `rdma_pull_with_opts` directly and this shim can be retired.
    pub async fn pull_remote_block_sets(
        &self,
        remote_instance: InstanceId,
        block_sets: &[RemoteBlockSet],
        local_dst_block_ids: &[BlockId],
    ) -> Result<TransferCompleteNotification> {
        // Length / count validation up front so callers see the same
        // hard bail they used to see, before any async work spawns.
        let source_count: usize = block_sets.iter().map(|set| set.blocks.len()).sum();
        if source_count != local_dst_block_ids.len() {
            anyhow::bail!(
                "Block count mismatch: source={}, destination={}",
                source_count,
                local_dst_block_ids.len()
            );
        }

        // Flatten every block_set into a single combined ref vector.
        // Order across sets must match local_dst_block_ids — same
        // offset bookkeeping the pre-shim loop used.
        let mut refs: Vec<PullRef> = Vec::with_capacity(source_count);
        let mut offset = 0usize;
        for block_set in block_sets {
            for block in &block_set.blocks {
                refs.push(PullRef {
                    src_block_id: block.block_id,
                    dst_block_id: local_dst_block_ids[offset],
                });
                offset += 1;
            }
        }

        // Wrap the async rdma_pull as a TransferCompleteNotification
        // so this shim keeps its pre-AB-4 return type. The leader is
        // already inside a tokio runtime (callers `.await` this fn).
        let events = self.messenger.event_manager();
        let event = events.new_event()?;
        let awaiter = events.awaiter(event.handle())?;

        let leader = self.clone();
        tokio::runtime::Handle::current().spawn(async move {
            match leader
                .rdma_pull_with_opts(remote_instance, refs, WirePullOptions::default())
                .await
            {
                Ok(()) => {
                    let _ = event.trigger();
                }
                Err(e) => {
                    let _ = event.poison(e.to_string());
                }
            }
        });

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    // ========================================================================
    // Private Worker Mirror Methods
    // These methods execute operations across all workers and aggregate results.
    // ========================================================================

    /// Execute local transfer across all workers, returning aggregated notification.
    ///
    /// Delegates to the parallel_worker which fans out to all workers and
    /// aggregates their notifications into a single composite notification.
    #[allow(dead_code)]
    pub(crate) fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst_block_ids: Vec<BlockId>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let parallel_worker = self
            .parallel_worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No parallel worker configured"))?;

        parallel_worker.execute_local_transfer(
            src,
            dst,
            Arc::from(src_block_ids),
            Arc::from(dst_block_ids),
            options,
        )
    }

    /// Execute remote onboard across all workers, returning aggregated notification.
    ///
    /// Delegates to the parallel_worker which fans out to all workers and
    /// aggregates their notifications into a single composite notification.
    #[allow(dead_code)]
    pub(crate) fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Vec<BlockId>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let parallel_worker = self
            .parallel_worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No parallel worker configured"))?;

        parallel_worker.execute_remote_onboard(src, dst, Arc::from(dst_block_ids), options)
    }

    /// Execute remote offload across all workers, returning aggregated notification.
    ///
    /// Delegates to the parallel_worker which fans out to all workers and
    /// aggregates their notifications into a single composite notification.
    #[allow(dead_code)]
    pub(crate) fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        dst: RemoteDescriptor,
        src_block_ids: Vec<BlockId>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let parallel_worker = self
            .parallel_worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No parallel worker configured"))?;

        parallel_worker.execute_remote_offload(src, Arc::from(src_block_ids), dst, options)
    }
}

// ---------------------------------------------------------------------------
// Describe helpers (Phase C)
// ---------------------------------------------------------------------------

/// Discriminant name of a `LayoutTypeDetails` variant — snake_case to match
/// the rest of the control-plane JSON.
fn layout_type_name(details: &kvbm_physical::layout::LayoutTypeDetails) -> &'static str {
    use kvbm_physical::layout::LayoutTypeDetails;
    match details {
        LayoutTypeDetails::FullyContiguous(_) => "fully_contiguous",
        LayoutTypeDetails::LayerSeparate(_) => "layer_separate",
    }
}

/// snake_case name of the `KvBlockLayout` discriminant carried by a
/// `LayoutTypeDetails` variant. Both variants carry one; this helper
/// abstracts the unwrap.
fn kv_block_layout_name(details: &kvbm_physical::layout::LayoutTypeDetails) -> String {
    use kvbm_common::KvBlockLayout;
    use kvbm_physical::layout::LayoutTypeDetails;
    let kbl: KvBlockLayout = match details {
        LayoutTypeDetails::FullyContiguous(d) => d.kv_block_layout,
        LayoutTypeDetails::LayerSeparate(d) => d.kv_block_layout,
    };
    match kbl {
        KvBlockLayout::Universal => "universal".to_owned(),
        KvBlockLayout::OperationalHND => "operational_hnd".to_owned(),
        KvBlockLayout::OperationalNHD => "operational_nhd".to_owned(),
        KvBlockLayout::Unknown => "unknown".to_owned(),
        // `Custom` carries an axis ordering; render as a hyphenated tag of
        // the four axis discriminants, e.g. `custom[block-layer-page-head]`.
        // Stable + diagnosable even though the exact layout is dynamic.
        KvBlockLayout::Custom(dims) => {
            let parts: Vec<&'static str> = dims.iter().map(block_dim_short_name).collect();
            format!("custom[{}]", parts.join("-"))
        }
    }
}

fn block_dim_short_name(d: &kvbm_common::BlockDim) -> &'static str {
    use kvbm_common::BlockDim;
    match d {
        BlockDim::Layer => "layer",
        BlockDim::Outer => "outer",
        BlockDim::Page => "page",
        BlockDim::Head => "head",
    }
}

/// Common `page_size` across all (worker, layout) pairs, or `None` if
/// heterogeneous / empty.
fn common_page_size(workers: &[WorkerInfo]) -> Option<usize> {
    let mut seen: Option<usize> = None;
    for w in workers {
        for l in &w.layouts {
            let p = l.config.page_size;
            match seen {
                None => seen = Some(p),
                Some(s) if s == p => {}
                Some(_) => return None,
            }
        }
    }
    seen
}

/// Top-level [`ParallelismDescription`] when every worker has a stamped
/// descriptor AND they agree on `tp_size`/`pp_size`/`shard_axis`/`global_extents`.
///
/// **Returns `None`** if any worker has `parallelism: None` (unstamped /
/// single-rank leader without a template). The aggregate must NEVER lie:
/// synthesising a `Some(1x1)` for a multi-worker leader whose descriptors
/// aren't stamped yet would tell operators "this leader is single-rank" when
/// it isn't. The per-worker `rank` and `layer_ownership` are intentionally
/// projected to rank 0 / the union range to give a "leader-wide view".
fn aggregate_parallelism(
    workers: &[WorkerInfo],
) -> Option<kvbm_protocols::control::ParallelismDescription> {
    use kvbm_protocols::control::{LayerRange, ParallelismDescription};

    // Any unstamped worker → aggregate is unknown. This is the bug fix:
    // pre-fix, an unstamped worker's synthesised 1x1 placeholder would
    // "agree" with other placeholders and the aggregate would lie.
    let first = workers.first()?.parallelism.clone()?;
    let mut layer_start = first.layer_ownership.start;
    let mut layer_end = first.layer_ownership.end;
    for w in workers.iter().skip(1) {
        let p = w.parallelism.as_ref()?;
        if p.tp_size != first.tp_size
            || p.pp_size != first.pp_size
            || p.shard_axis != first.shard_axis
            || p.global_extents != first.global_extents
        {
            return None;
        }
        layer_start = layer_start.min(p.layer_ownership.start);
        layer_end = layer_end.max(p.layer_ownership.end);
    }
    Some(ParallelismDescription {
        rank: 0,
        layer_ownership: LayerRange {
            start: layer_start,
            end: layer_end,
        },
        ..first
    })
}

/// Sum tier capacity across all workers, grouping by [`TierKind`].
fn sum_tier_capacity(workers: &[WorkerInfo]) -> Vec<TierCapacity> {
    use std::collections::HashMap;
    let mut acc: HashMap<TierKind, TierCapacity> = HashMap::new();
    for w in workers {
        for l in &w.layouts {
            let entry = acc.entry(l.tier).or_insert(TierCapacity {
                tier: l.tier,
                num_blocks: 0,
                bytes_per_block: l.bytes_per_block,
                total_bytes: 0,
            });
            entry.num_blocks = entry.num_blocks.saturating_add(l.config.num_blocks);
            entry.total_bytes = entry.total_bytes.saturating_add(l.total_bytes as u64);
        }
    }
    let mut out: Vec<TierCapacity> = acc.into_values().collect();
    // Deterministic order: G1, G2, G3, G4.
    out.sort_by_key(|t| match t.tier {
        TierKind::G1 => 0,
        TierKind::G2 => 1,
        TierKind::G3 => 2,
        TierKind::G4 => 3,
    });
    out
}

/// Read hostname from libc, falling back to the `HOSTNAME` env var and
/// finally `"unknown"`. Avoids a hard `hostname` crate dep — std + env
/// suffice for the level of identity we need in describe.
fn read_hostname() -> String {
    if let Ok(name) = std::env::var("HOSTNAME")
        && !name.is_empty()
    {
        return name;
    }
    // Fall back to `uname` via /proc/sys/kernel/hostname on Linux.
    if let Ok(name) = std::fs::read_to_string("/proc/sys/kernel/hostname") {
        let trimmed = name.trim();
        if !trimmed.is_empty() {
            return trimmed.to_owned();
        }
    }
    "unknown".to_owned()
}

impl Leader for InstanceLeader {
    fn find_matches_with_options(
        &self,
        sequence_hashes: &[SequenceHash],
        options: FindMatchesOptions,
    ) -> Result<FindMatchesResult> {
        // Search G2 (host memory) for matches
        // Uses match_blocks which stops at first miss (implements "first hole" policy).
        // This ensures we only find contiguous blocks from the start of the sequence.

        // todo: add explicit timing tracing here
        // let start_time = Instant::now();
        let matched_g2_blocks = self.g2_manager.match_blocks(sequence_hashes);
        //let g2_search_time = Instant::now().duration_since(start_time);

        // Search G3 (disk) for remaining hashes if G3 is available
        let remaining_hashes: Vec<_> = sequence_hashes
            .iter()
            .filter(|h| !matched_g2_blocks.iter().any(|b| b.sequence_hash() == **h))
            .copied()
            .collect();

        let matched_g3_blocks = if let Some(ref g3_manager) = self.g3_manager {
            // Uses match_blocks on remaining hashes (those not found in G2).
            // Since G2 already applied first-hole policy, G3 search continues from where G2 stopped.
            g3_manager.match_blocks(&remaining_hashes)
        } else {
            Vec::new()
        };

        let has_remote_leaders = !self.remote_leaders.read().unwrap().is_empty();
        let has_object_client = self.object_client.is_some();
        let has_discovery = self.remote_discovery.get().is_some();
        let local_g2_count = matched_g2_blocks.len();
        let local_g3_count = matched_g3_blocks.len();

        // Warm-cache short-circuit: if the synchronous local G2 prefix match
        // already covers every queried hash, there is nothing remote to find or
        // pull. Return `Ready` directly and skip the remote-search driver
        // entirely — this is the path a request to the instance that already
        // holds the blocks takes (it must not "remote search" itself).
        let local_covers_all =
            !sequence_hashes.is_empty() && matched_g2_blocks.len() == sequence_hashes.len();

        // Hub-indexer remote search (the transfer-plane driver) runs only when
        // discovery is injected, the local G2 prefix is incomplete, and there
        // are no local G3 hits to stage. The legacy peer-leader / G4-object
        // search path has been removed; cases it used to handle now degrade to
        // the local match (or local G3→G2 staging). The G4/object-storage search
        // machinery is parked in `leader::search` and is not wired in yet.
        let use_indexer_search = options.search_remote
            && has_discovery
            && matched_g3_blocks.is_empty()
            && !has_object_client
            && !has_remote_leaders
            && !local_covers_all;


        // Host-bypass short-circuit: when G2 is intentionally unconfigured we
        // never want to take the AsyncSession + stage_g3_to_g2 path. Return
        // immediately with both G2 (typically empty) and G3 blocks attached
        // so the caller can issue G3→G1 directly via GDS.
        if self.bypass_host {
            return Ok(FindMatchesResult::Ready(ReadyResult::new_with_g3(
                matched_g2_blocks,
                matched_g3_blocks,
                super::MatchBreakdown {
                    host_blocks: local_g2_count,
                    disk_blocks: local_g3_count,
                    object_blocks: 0,
                },
            )));
        }

        // Ready when the local G2 prefix is complete, or when there is no local
        // G3 to stage and no indexer search to run. Otherwise fall through to an
        // AsyncSession (indexer-driven pull, or local G3→G2 staging).
        if local_covers_all || (matched_g3_blocks.is_empty() && !use_indexer_search) {
            // No session needed - blocks owned directly by ReadyResult (RAII)
            return Ok(FindMatchesResult::Ready(ReadyResult::new(
                matched_g2_blocks,
                super::MatchBreakdown {
                    host_blocks: local_g2_count,
                    disk_blocks: 0,
                    object_blocks: 0,
                },
            )));
        }

        // AsyncSession path: local G3 blocks found — stage G3→G2, then deliver.
        let session_id = SessionId::from(Uuid::new_v4());

        let (status_tx, status_rx) = watch::channel(OnboardingStatus::Searching);
        let all_g2_blocks = Arc::new(Mutex::new(None));
        let match_breakdown = Arc::new(Mutex::new(super::MatchBreakdown {
            host_blocks: local_g2_count,
            disk_blocks: local_g3_count,
            object_blocks: 0,
        }));

        // Cancellation token bound to this session, used only by the
        // hub-indexer remote-search driver so `release_session` can tear it
        // down on request cancellation/preemption.
        let cancel_token = use_indexer_search.then(CancellationToken::new);

        // Store session state to keep the matched blocks alive (RAII) until the
        // caller releases the session. The staging task below re-matches the
        // registry-backed handles independently.
        let state = SessionState {
            session_id,
            matched_g2_blocks,
            matched_g3_blocks,
            status_tx: status_tx.clone(),
            cancel: cancel_token.clone(),
        };
        self.store_session_state(state);

        // Hub-indexer remote search: spawn the transfer-plane driver. It
        // re-matches local G2 (after pulling) to fill `all_g2_blocks` and sets
        // the terminal `OnboardingStatus::Complete`. Any failure/timeout/cancel
        // degrades to the local match so the shard never stays non-terminal.
        if use_indexer_search {
            let discovery = self
                .remote_discovery
                .get()
                .cloned()
                .expect("use_indexer_search implies remote_discovery is set");
            let cancel = cancel_token.expect("use_indexer_search sets a cancel token");
            let driver = remote_search::RemoteSearchDriver {
                leader: Arc::new(self.clone()),
                discovery,
                sequence_hashes: sequence_hashes.to_vec(),
                local_block_count: local_g2_count,
                min_remote_blocks: self.min_remote_blocks,
                status_tx: status_tx.clone(),
                blocks: all_g2_blocks.clone(),
                match_breakdown: match_breakdown.clone(),
                session_id,
                cancel,
            };
            self.runtime().spawn(driver.run());
            return Ok(FindMatchesResult::AsyncSession(AsyncSessionResult::new(
                session_id,
                status_rx,
                all_g2_blocks,
                match_breakdown,
            )));
        }

        // Stage local G3→G2 off-thread, then publish the deliverable G2 blocks.
        // We only report blocks that are actually available in G2: the G2 prefix
        // plus any G3 successfully staged into G2. If staging fails (or there is
        // no parallel worker), we deliver just the G2 prefix rather than claiming
        // undelivered G3 matches.
        let g2_manager = self.g2_manager.clone();
        let g3_manager = self.g3_manager.clone();
        let parallel_worker = self.parallel_worker.clone();
        let sequence_hashes = sequence_hashes.to_vec();
        let all_g2_blocks_task = all_g2_blocks.clone();
        let match_breakdown_task = match_breakdown.clone();
        let status_tx_task = status_tx.clone();

        self.messenger.runtime().spawn(async move {
            // Re-match the local tiers (registry-backed handles, cheap to re-resolve).
            let mut g2 = g2_manager.match_blocks(&sequence_hashes);
            let remaining: Vec<_> = sequence_hashes
                .iter()
                .filter(|h| !g2.iter().any(|b| b.sequence_hash() == **h))
                .copied()
                .collect();

            if let Some(g3_manager) = g3_manager.as_ref() {
                let g3 = g3_manager.match_blocks(&remaining);
                if !g3.is_empty() {
                    match parallel_worker.as_ref() {
                        Some(pw) => {
                            let holder = BlockHolder::new(g3);
                            match stage_g3_to_g2(&holder, &g2_manager, &**pw).await {
                                // Staging registers the new G2 blocks before
                                // returning, so re-match the full sequence: a
                                // staged block can fill a hole between the G2
                                // prefix and a later already-resident G2 run,
                                // extending the contiguous match. Counting only
                                // `prefix + staged` would under-report that tail.
                                Ok(_staged) => g2 = g2_manager.match_blocks(&sequence_hashes),
                                Err(e) => tracing::warn!(
                                    %session_id,
                                    error = %e,
                                    "local G3→G2 staging failed; delivering G2 prefix only"
                                ),
                            }
                        }
                        None => tracing::warn!(
                            %session_id,
                            "no parallel worker configured; cannot stage G3, delivering G2 prefix only"
                        ),
                    }
                }
            }

            // Sort by position so the consumer sees a contiguous run from the start.
            g2.sort_by_key(|b| b.sequence_hash().position());
            let matched_blocks = g2.len();

            if let Ok(mut breakdown) = match_breakdown_task.try_lock() {
                breakdown.host_blocks = matched_blocks;
                breakdown.disk_blocks = 0;
            }
            *all_g2_blocks_task.lock().await = Some(g2);

            status_tx_task
                .send(OnboardingStatus::Complete { matched_blocks })
                .ok();
        });

        Ok(FindMatchesResult::AsyncSession(AsyncSessionResult::new(
            session_id,
            status_rx,
            all_g2_blocks,
            match_breakdown,
        )))
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::G2;
    use crate::leader::types::StagingMode;
    use crate::testing::{managers::TestManagerBuilder, messenger::create_messenger_tcp};
    use kvbm_common::KvDim;
    use kvbm_config::ParallelismMode;
    use kvbm_logical::blocks::BlockRegistry;
    use kvbm_physical::manager::{LogicalLayoutDescriptor, WorkerAddress};

    fn stub_metadata_for(worker_id: u64) -> SerializedLayout {
        SerializedLayout::pack(
            WorkerAddress::new(worker_id, format!("agent-{worker_id}")),
            Vec::new(),
            Vec::<LogicalLayoutDescriptor>::new(),
            None,
        )
        .unwrap()
    }

    async fn leader_with_cached_metadata(
        cached: Vec<SerializedLayout>,
        template: Option<ParallelismTemplate>,
    ) -> Result<InstanceLeader> {
        let messenger = create_messenger_tcp().await?;
        let registry = BlockRegistry::builder().build();
        let g2 = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(2)
                .block_size(4)
                .registry(registry.clone())
                .build(),
        );

        let mut builder = InstanceLeader::builder()
            .messenger(messenger)
            .registry(registry)
            .g2_manager(g2)
            .with_cached_worker_metadata(cached);
        if let Some(t) = template {
            builder = builder.parallelism_template(t);
        }
        builder.build()
    }

    fn template(tp_size: usize) -> ParallelismTemplate {
        ParallelismTemplate {
            tp_size,
            pp_size: 1,
            parallelism_mode: ParallelismMode::TensorParallel,
            shard_axis: KvDim::HeadCount,
            global_extents: vec![(KvDim::HeadCount, 16)],
            num_layers: 12,
            dtype_width_bytes: 2,
        }
    }

    /// Regression: `find_matches` must never report more matched blocks than it
    /// can actually deliver in G2. When local G3 blocks match but cannot be
    /// staged to G2 (here: no parallel worker), only the deliverable G2 prefix
    /// is reported — not `g2 + g3`. (Pre-fix this path reported `g2 + g3` while
    /// leaving the block payload empty.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn find_matches_local_g3_does_not_overreport_without_staging() -> Result<()> {
        use crate::G3;
        use crate::leader::Leader;
        use crate::testing::token_blocks::{create_token_sequence, generate_sequence_hashes};

        let messenger = create_messenger_tcp().await?;
        let registry = BlockRegistry::builder().build();
        let g2 = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(8)
                .block_size(4)
                .registry(registry.clone())
                .build(),
        );
        let g3 = Arc::new(
            TestManagerBuilder::<G3>::new()
                .block_count(8)
                .block_size(4)
                .registry(registry.clone())
                .build(),
        );
        // No workers → no parallel worker → G3 cannot be staged into G2.
        let leader = InstanceLeader::builder()
            .messenger(messenger)
            .registry(registry)
            .g2_manager(g2.clone())
            .g3_manager(g3.clone())
            .build()?;

        // 4-block sequence: first 2 blocks live in G2 (the deliverable prefix),
        // the next 2 in G3 (matched but unstageable here).
        let seq = create_token_sequence(4, 4, 0);
        crate::testing::managers::populate_manager_with_blocks(&g2, &seq.blocks()[..2])?;
        crate::testing::managers::populate_manager_with_blocks(&g3, &seq.blocks()[2..])?;
        let hashes = generate_sequence_hashes(&seq);

        let result = leader.find_matches(&hashes)?;
        let async_session = result
            .as_async()
            .expect("local G3 present (non-bypass) must yield an AsyncSession");
        async_session.wait_for_completion().await?;

        let reported = match async_session.status() {
            OnboardingStatus::Complete { matched_blocks } => matched_blocks,
            other => panic!("expected Complete, got {other:?}"),
        };
        let delivered = async_session.get_blocks_count().unwrap_or(0);

        // The invariant: never claim more than is deliverable in G2.
        assert_eq!(
            reported, delivered,
            "reported matched_blocks ({reported}) must equal delivered G2 blocks ({delivered})"
        );
        // Only the G2 prefix (2) is deliverable; the 2 unstageable G3 blocks are dropped.
        assert_eq!(delivered, 2, "expected only the G2 prefix to be delivered");
        Ok(())
    }

    // TODO(test): hole-fill under-report reproducer for the local G3→G2 staging
    // path. `match_blocks` is a first-hole prefix match, so when G2 holds
    // [h0,h1,h3] (hole at h2) and G3 holds [h2], staging h2 into G2 must make the
    // contiguous match grow to [h0,h1,h2,h3] (4) — not the [h0,h1] prefix + [h2]
    // staged (3). The fix re-matches the full sequence after a successful stage;
    // this test asserts matched_blocks == 4. It needs a parallel worker that can
    // actually stage G3→G2 in-process: the current `TestInstanceLeaderWithWorkers`
    // fixture registers only a G2 layout (no G3), so this requires adding a G3
    // layout registration (or a lightweight `ParallelWorkers` stub returning a
    // pre-completed transfer notification). Deferred to avoid that infra build-out.

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn assemble_export_metadata_stamps_when_template_set() -> Result<()> {
        let cached = vec![stub_metadata_for(0), stub_metadata_for(1)];
        let leader = leader_with_cached_metadata(cached, Some(template(2))).await?;

        let exported = leader.assemble_export_metadata().await?;
        assert_eq!(exported.len(), 2);
        for (i, layout) in exported.iter().enumerate() {
            let unpacked = layout.unpack()?;
            let desc = unpacked
                .parallelism
                .expect("descriptor must be stamped when template is set");
            assert_eq!(desc.rank, i);
            assert_eq!(desc.tp_size, 2);
            assert_eq!(desc.shard_axis, KvDim::HeadCount);
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn assemble_export_metadata_passes_through_when_no_template() -> Result<()> {
        let cached = vec![stub_metadata_for(0), stub_metadata_for(1)];
        let leader = leader_with_cached_metadata(cached, None).await?;

        let exported = leader.assemble_export_metadata().await?;
        assert_eq!(exported.len(), 2);
        for layout in &exported {
            let unpacked = layout.unpack()?;
            assert!(
                unpacked.parallelism.is_none(),
                "no template configured → no descriptor stamped"
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn assemble_export_metadata_errors_on_length_mismatch() -> Result<()> {
        // Template says tp_size = 4 but cache only has 2 entries.
        let cached = vec![stub_metadata_for(0), stub_metadata_for(1)];
        let leader = leader_with_cached_metadata(cached, Some(template(4))).await?;
        let err = leader.assemble_export_metadata().await.unwrap_err();
        assert!(
            err.to_string().contains("tp_size * pp_size"),
            "expected length-mismatch error, got: {err}"
        );
        Ok(())
    }

    // ------------------------------------------------------------------
    // Describe (Phase C)
    // ------------------------------------------------------------------

    /// Pre-stamping snapshot: the leader's pre-worker-stamp state still
    /// produces a valid `InstanceDescription` with identity + process info.
    /// The cached metadata is empty layouts (no `LogicalLayoutDescriptor`s),
    /// so `workers` lands with `WorkerInfo` entries that carry the
    /// `worker_id` and an empty `layouts` vec.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn describe_pre_stamping_populates_identity_fields() -> Result<()> {
        let cached = vec![stub_metadata_for(7), stub_metadata_for(8)];
        let leader = leader_with_cached_metadata(cached, None).await?;

        let d = leader.describe().await.expect("describe ok");
        assert_eq!(d.worker_ids, vec![7, 8]);
        assert_eq!(d.workers.len(), 2);
        // Layouts are empty (no `LogicalLayoutDescriptor`s in stub metadata)
        // — so block_size aggregates to None, tier_capacity is empty.
        assert!(d.block_size.is_none());
        assert!(d.tier_capacity.is_empty());
        // Identity / process / capability fields populated.
        assert_eq!(d.instance_id, leader.messenger().instance_id().to_string());
        assert!(d.hub_instance_id.is_none(), "no set_hub_instance_id call");
        assert!(d.config.is_none(), "no set_config_blob call");
        assert!(d.modules.is_empty(), "no set_modules call");
        assert!(d.role.is_none());
        assert_ne!(d.host.pid, 0);
        Ok(())
    }

    /// **Regression test** — describe MUST NOT synthesise a fake `tp_size=1,
    /// pp_size=1` parallelism when workers haven't stamped descriptors yet
    /// (or when the leader was built without a `ParallelismTemplate`).
    /// Reporting 1x1 for a multi-worker TP leader is a lie about topology.
    ///
    /// Pre-fix: per-worker parallelism was synthesised to `Some(1x1)` and
    /// the aggregate cascaded to `Some(1x1)`. Post-fix: both are `None` and
    /// the wire honestly says "parallelism unknown / not stamped".
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn describe_does_not_invent_1x1_parallelism_when_unstamped() -> Result<()> {
        // Two workers, no template — `stub_metadata_for` packs
        // `parallelism = None`. A pre-fix implementation would lie that
        // the leader is single-rank (1x1) when it actually has two workers.
        let cached = vec![stub_metadata_for(0), stub_metadata_for(1)];
        let leader = leader_with_cached_metadata(cached, None).await?;

        let d = leader.describe().await.expect("describe ok");
        assert_eq!(d.workers.len(), 2);

        // Per-worker parallelism must be None when the descriptor is unstamped.
        for w in &d.workers {
            assert!(
                w.parallelism.is_none(),
                "worker {} reports parallelism without a stamped descriptor: {:?}",
                w.worker_id,
                w.parallelism
            );
        }
        // Aggregate must be None when any worker is unstamped — never a
        // synthetic 1x1 for what is actually a 2-worker leader.
        assert!(
            d.parallelism.is_none(),
            "top-level parallelism synthesised without stamped descriptors: {:?}",
            d.parallelism
        );
        Ok(())
    }

    /// `set_config_blob` / `set_hub_instance_id` / `set_modules` are
    /// idempotent: first-write-wins, subsequent calls return false.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn describe_setters_are_idempotent() -> Result<()> {
        let leader = leader_with_cached_metadata(vec![], None).await?;
        let first = serde_json::json!({"a": 1});
        let second = serde_json::json!({"b": 2});
        assert!(leader.set_config_blob(first.clone()));
        assert!(!leader.set_config_blob(second.clone()));
        let d = leader.describe().await.expect("describe ok");
        assert_eq!(d.config.as_ref(), Some(&first));

        assert!(leader.set_modules(vec![kvbm_protocols::control::ModuleId::Core]));
        assert!(!leader.set_modules(vec![kvbm_protocols::control::ModuleId::Dev]));
        let d2 = leader.describe().await.expect("describe ok");
        assert_eq!(d2.modules, vec![kvbm_protocols::control::ModuleId::Core]);
        Ok(())
    }

    // ------------------------------------------------------------------------
    // Remote-search driver (transfer-plane, hub-indexer discovery)
    // ------------------------------------------------------------------------

    use crate::leader::discovery::{RemoteBlockDiscovery, RemoteCandidates};
    use futures::future::BoxFuture;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock discovery that records call count and returns a canned outcome.
    struct MockDiscovery {
        calls: Arc<AtomicUsize>,
        outcome: Option<RemoteCandidates>,
    }

    impl RemoteBlockDiscovery for MockDiscovery {
        fn discover(
            &self,
            _hashes: Vec<SequenceHash>,
        ) -> BoxFuture<'static, Result<Option<RemoteCandidates>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let outcome = self.outcome.clone();
            Box::pin(async move { Ok(outcome) })
        }
    }

    async fn leader_for_remote_search(min_remote_blocks: usize) -> Result<Arc<InstanceLeader>> {
        let messenger = create_messenger_tcp().await?;
        let registry = BlockRegistry::builder().build();
        let g2 = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(8)
                .block_size(4)
                .registry(registry.clone())
                .build(),
        );
        let leader = InstanceLeader::builder()
            .messenger(messenger)
            .registry(registry)
            .g2_manager(g2)
            .min_remote_blocks(min_remote_blocks)
            .build()?;
        Ok(Arc::new(leader))
    }

    fn some_hashes(n: usize) -> Vec<SequenceHash> {
        // Distinct, deterministic PLHs; none are resident in the fresh G2, so
        // the local prefix match is empty and the whole range is "remote".
        (0..n as u64).map(|i| SequenceHash::new(i, None, i)).collect()
    }

    /// Below the block threshold, the driver must not consult discovery and
    /// must still reach a terminal `Complete` (degrade to the local match).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remote_search_below_threshold_skips_discovery() -> Result<()> {
        let leader = leader_for_remote_search(/* min_remote_blocks */ 10).await?;
        let calls = Arc::new(AtomicUsize::new(0));
        leader.set_remote_discovery(Arc::new(MockDiscovery {
            calls: Arc::clone(&calls),
            outcome: None,
        }));

        // 3 remaining blocks <= threshold(10) → no remote search.
        let hashes = some_hashes(3);
        let result = leader.find_matches_with_options(
            &hashes,
            FindMatchesOptions {
                search_remote: true,
                staging_mode: StagingMode::Full,
            },
        )?;
        let async_session = result.as_async().expect("discovery drives the async path");
        async_session.wait_for_completion().await?;

        assert_eq!(calls.load(Ordering::SeqCst), 0, "discovery must be skipped");
        assert!(
            matches!(async_session.status(), OnboardingStatus::Complete { matched_blocks: 0 }),
            "empty local G2 → Complete{{0}}, got {:?}",
            async_session.status()
        );
        Ok(())
    }

    /// Above threshold with a discovery miss (`Ok(None)`): discovery is
    /// consulted once and the driver degrades to the (empty) local match
    /// rather than wedging non-terminal.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remote_search_discovery_miss_degrades_to_local() -> Result<()> {
        let leader = leader_for_remote_search(/* min_remote_blocks */ 1).await?;
        let calls = Arc::new(AtomicUsize::new(0));
        leader.set_remote_discovery(Arc::new(MockDiscovery {
            calls: Arc::clone(&calls),
            outcome: None,
        }));

        // 5 remaining blocks > threshold(1) → remote search attempted.
        let hashes = some_hashes(5);
        let result = leader.find_matches_with_options(
            &hashes,
            FindMatchesOptions {
                search_remote: true,
                staging_mode: StagingMode::Full,
            },
        )?;
        let async_session = result.as_async().expect("discovery drives the async path");
        async_session.wait_for_completion().await?;

        assert_eq!(calls.load(Ordering::SeqCst), 1, "discovery consulted once");
        assert!(
            matches!(async_session.status(), OnboardingStatus::Complete { matched_blocks: 0 }),
            "discovery miss degrades to local match, got {:?}",
            async_session.status()
        );
        Ok(())
    }

    /// Discovery that parks forever after signalling it started — lets a test
    /// catch the driver mid-flight and cancel it.
    struct BlockingDiscovery {
        started: Arc<tokio::sync::Notify>,
    }

    impl RemoteBlockDiscovery for BlockingDiscovery {
        fn discover(
            &self,
            _hashes: Vec<SequenceHash>,
        ) -> BoxFuture<'static, Result<Option<RemoteCandidates>>> {
            let started = Arc::clone(&self.started);
            Box::pin(async move {
                started.notify_one();
                std::future::pending::<()>().await;
                unreachable!()
            })
        }
    }

    /// The architectural justification for an engine-owned driver: a
    /// cancelled/preempted request (`release_session`) must tear the in-flight
    /// remote search down and drive the session to a terminal state, not wedge
    /// it at `Searching` forever.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn release_session_cancels_inflight_remote_search() -> Result<()> {
        let leader = leader_for_remote_search(/* min_remote_blocks */ 1).await?;
        let started = Arc::new(tokio::sync::Notify::new());
        leader.set_remote_discovery(Arc::new(BlockingDiscovery {
            started: Arc::clone(&started),
        }));

        let result = leader.find_matches_with_options(
            &some_hashes(5),
            FindMatchesOptions {
                search_remote: true,
                staging_mode: StagingMode::Full,
            },
        )?;
        let async_session = result.as_async().expect("discovery drives the async path");
        let session_id = async_session.session_id();

        // Wait until the driver is parked inside discovery, then cancel.
        started.notified().await;
        assert!(matches!(async_session.status(), OnboardingStatus::Searching));

        leader.release_session(session_id);

        // The driver's cancel branch must drive the session to terminal
        // promptly — otherwise this times out.
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            async_session.wait_for_completion(),
        )
        .await
        .expect("cancellation must reach a terminal state before the timeout")?;
        assert!(matches!(
            async_session.status(),
            OnboardingStatus::Complete { .. }
        ));
        Ok(())
    }

    /// Without an injected discovery, `search_remote` does not take the
    /// indexer path: with no remote leaders / object client the result is a
    /// synchronous local `Ready`, not an async session.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn search_remote_without_discovery_is_ready_local() -> Result<()> {
        let leader = leader_for_remote_search(1).await?;
        let hashes = some_hashes(5);
        let result = leader.find_matches_with_options(
            &hashes,
            FindMatchesOptions {
                search_remote: true,
                staging_mode: StagingMode::Full,
            },
        )?;
        assert!(
            result.is_ready(),
            "no discovery + no remote leaders/object → local Ready"
        );
        Ok(())
    }
}
