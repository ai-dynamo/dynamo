// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::SystemTime;

use ::velo::Messenger;
use anyhow::Result;
use dashmap::DashMap;
use tokio::sync::{Mutex, mpsc, watch};
use uuid::Uuid;

use std::sync::{Arc, OnceLock};

use kvbm_config::DisaggregationRole;
use kvbm_protocols::control::{
    ControlError, HostInfo, InstanceDescription, LayoutDescription, ModuleId, TierCapacity,
    TierKind, WorkerInfo,
};

use crate::{
    BlockId, G2, G3, InstanceId, SequenceHash,
    disagg::RemoteBlockSet,
    disagg::session::{SessionFactory, SessionManager},
    object::ObjectBlockOps,
    worker::RemoteDescriptor,
};
use kvbm_common::LogicalLayoutHandle;
use kvbm_logical::{
    blocks::{BlockRegistry, ImmutableBlock},
    manager::BlockManager,
};
use kvbm_physical::transfer::{TransferCompleteNotification, TransferOptions};

use kvbm_physical::manager::{LayoutHandle, SerializedLayout};

use super::{
    super::worker::Worker,
    super::worker::group::{ParallelWorkers, SpmdParallelWorkers},
    AsyncSessionResult,
    FindMatchesOptions,
    FindMatchesResult,
    Leader,
    OnboardingStatus,
    ReadyResult,
    // Legacy SessionHandle for deferred operations
    SessionHandle as LegacySessionHandle,
    SessionId,
    StagingMode,
    accessor::{BlockAccessor, PolicyContext},
    dispatch::{PullRef, WirePullOptions, plan_pull},
    parallelism::{ParallelismTemplate, stamp_parallelism_descriptors},
    session::{
        BlockHolder, ControlRole, ControllableSessionOptions, ControllableSessionResult,
        InitiatorSession, MessageTransport, OnboardMessage, OnboardSessionTx, ResponderSession,
        ServerSession, ServerSessionHandle, ServerSessionOptions, SessionHandle, SessionMessage,
        SessionMessageTx, SessionPhase, create_server_session, session_handle_state_channel,
        session_message_channel,
    },
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
/// - **Session management**: creating, attaching, and driving onboard sessions
///   between endpoint (source) and controller (destination) roles.
/// - **Remote connectivity**: exchanging serialized layout metadata with peer
///   instances so workers can perform RDMA transfers.
/// - **Velo RPC**: registering handlers via `VeloLeaderService` so remote
///   leaders can initiate sessions and exchange metadata.
#[derive(Clone)]
pub struct InstanceLeader {
    /// Velo instance for distributed communication.
    messenger: Arc<Messenger>,

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

    /// Map of active sessions (session_id -> message channel).
    sessions: Arc<DashMap<SessionId, OnboardSessionTx>>,

    /// Cached worker metadata (avoids querying workers repeatedly).
    cached_worker_metadata: Option<Vec<SerializedLayout>>,

    /// Map of session states for holding blocks alive (RAII).
    session_states: Arc<DashMap<SessionId, SessionState>>,

    /// List of remote leader instance IDs (mutable for post-construction configuration).
    remote_leaders: Arc<std::sync::RwLock<Vec<InstanceId>>>,

    /// Message transport for session communication.
    transport: Arc<MessageTransport>,

    // ========================================================================
    // Unified Session Protocol
    // ========================================================================
    /// Map of session message receivers.
    /// Used by SessionHandle/SessionEndpoint/ControllableSession.
    session_sessions: Arc<DashMap<SessionId, SessionMessageTx>>,

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
}

/// Builder for InstanceLeader.
#[derive(Default)]
pub struct InstanceLeaderBuilder {
    messenger: Option<Arc<Messenger>>,
    registry: Option<BlockRegistry>,
    g2_manager: Option<Arc<BlockManager<G2>>>,
    g3_manager: Option<Arc<BlockManager<G3>>>,
    workers: Vec<Arc<dyn Worker>>,
    sessions: Option<Arc<DashMap<SessionId, OnboardSessionTx>>>,
    remote_leaders: Option<Vec<InstanceId>>,
    cached_worker_metadata: Option<Vec<SerializedLayout>>,
    object_client: Option<Arc<dyn ObjectBlockOps>>,
    parallelism_template: Option<ParallelismTemplate>,
    role: Option<DisaggregationRole>,
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
        self.messenger(runtime.messenger().clone())
    }

    pub fn messenger(mut self, messenger: Arc<Messenger>) -> Self {
        self.messenger = Some(messenger);
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

    /// Set the disaggregation role this leader plays. Surfaced via
    /// [`InstanceLeader::describe`] for the hub UI. Defaults to `None`
    /// (standalone — not part of a P/D split).
    pub fn role(mut self, role: DisaggregationRole) -> Self {
        self.role = Some(role);
        self
    }

    pub fn build(self) -> Result<InstanceLeader> {
        let messenger = self
            .messenger
            .ok_or_else(|| anyhow::anyhow!("Velo instance required"))?;
        let transport = Arc::new(MessageTransport::velo(messenger.clone()));

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
        // compatibility gates (AB-1b).
        let parallel_worker: Option<Arc<dyn ParallelWorkers>> = if !self.workers.is_empty() {
            let mut spmd =
                SpmdParallelWorkers::new(self.workers.to_vec(), events.clone(), runtime.clone());
            if let Some(template) = self.parallelism_template.clone() {
                spmd = spmd.with_local_template(template);
            }
            Some(Arc::new(spmd))
        } else {
            None
        };

        Ok(InstanceLeader {
            messenger,
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
            sessions: self.sessions.unwrap_or_else(|| Arc::new(DashMap::new())),
            session_states: Arc::new(DashMap::new()),
            remote_leaders: Arc::new(std::sync::RwLock::new(
                self.remote_leaders.unwrap_or_default(),
            )),
            transport,
            session_sessions: Arc::new(DashMap::new()),
            object_client: self.object_client,
            parallelism_template: self.parallelism_template,
            session_manager: SessionManager::with_default_watchdog(runtime),
            session_factory: Arc::new(OnceLock::new()),
            role: self.role,
            started_at: SystemTime::now(),
            hub_instance_id: Arc::new(OnceLock::new()),
            config_blob: Arc::new(OnceLock::new()),
            modules: Arc::new(OnceLock::new()),
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
    /// The client is used by InitiatorSession for G4 parallel search.
    pub fn object_client(&self) -> Option<Arc<dyn ObjectBlockOps>> {
        self.object_client.clone()
    }

    /// Add a remote leader to the search list.
    ///
    /// Remote leaders are queried during `find_matches_with_options` when
    /// `search_remote == true`. This method allows adding remote leaders
    /// after construction (e.g., when instance IDs are only known after
    /// cluster setup).
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
        let instance_id = self.messenger.instance_id();
        let g2_manager = self.g2_manager.clone();
        let g3_manager = self.g3_manager.clone();
        let parallel_worker = self.parallel_worker.clone();
        let transport = self.transport.clone();
        let sessions = self.sessions.clone();

        let spawn_responder = move |msg: OnboardMessage| -> Result<()> {
            if let OnboardMessage::CreateSession {
                requester,
                session_id,
                sequence_hashes,
            } = msg
            {
                let (tx, rx) = mpsc::channel(100);
                sessions.insert(session_id, tx);

                let session = ResponderSession::new(
                    session_id,
                    instance_id,
                    requester,
                    g2_manager.clone(),
                    g3_manager.clone(),
                    parallel_worker.clone(),
                    transport.clone(),
                );

                tokio::spawn(async move {
                    if let Err(e) = session.run(rx, sequence_hashes).await {
                        tracing::warn!(error = %e, "ResponderSession error");
                    }
                });

                Ok(())
            } else {
                anyhow::bail!("spawn_responder called with non-CreateSession message")
            }
        };

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

        let mut service = VeloLeaderService::new(self.messenger.clone(), self.sessions.clone())
            .with_spawn_responder(spawn_responder)
            .with_session_sessions(self.session_sessions.clone());

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
    /// - `test` (opt-in) — `register_test_blocks`. Usable in production but
    ///   logs a warning when enabled.
    ///
    /// `dev` / `test` come from `control.dev` / `control.test` in
    /// `KvbmConfig`. Takes `Arc<Self>` because the `core` / `dev` modules
    /// hold an `Arc<InstanceLeader>`. The returned [`ControlPlane`] carries
    /// only introspection metadata; the handlers live on the messenger and
    /// the modules' captured state outlives the returned handle.
    pub fn register_control_plane(
        self: &Arc<Self>,
        dev: bool,
        test: bool,
    ) -> Result<Arc<crate::leader::ControlPlane>> {
        use crate::leader::control::{
            ControlPlane, CoreModule, DevModule, TestModule, TransferModule,
        };

        let mut builder =
            ControlPlane::builder(self.messenger.clone(), self.messenger.instance_id())
                .with_module(CoreModule::new(Arc::clone(self)))
                .with_module(TransferModule::new(
                    self.g2_manager.clone(),
                    self.session_factory_cell(),
                    self.session_manager.clone(),
                ));

        if dev {
            builder = builder.with_module(DevModule::new(Arc::clone(self)));
        }
        if test {
            tracing::warn!(
                "control plane `test` module enabled — exposes test-only handlers; \
                 not intended for production"
            );
            builder = builder.with_module(TestModule::new(self.g2_manager.clone()));
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
        self.session_states.remove(&session_id);
        self.sessions.remove(&session_id);
        self.session_sessions.remove(&session_id);
    }

    /// Test-only: is `session_id` registered in any of the three session maps?
    #[cfg(any(test, feature = "testing"))]
    pub fn has_session(&self, session_id: SessionId) -> bool {
        self.sessions.contains_key(&session_id)
            || self.session_states.contains_key(&session_id)
            || self.session_sessions.contains_key(&session_id)
    }

    /// Test-only: insert a sentinel entry into `sessions` so a test can verify
    /// that `release_session` removes it. The channel has capacity 1 and its
    /// receiver is dropped immediately; the map entry alone is what the test
    /// observes.
    #[cfg(any(test, feature = "testing"))]
    pub fn insert_test_session_marker(&self, session_id: SessionId) {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        self.sessions.insert(session_id, tx);
    }

    // ========================================================================
    // Inverted Control Pattern (Prefill-Decode) Methods
    // ========================================================================

    /// Create a controllable session for local blocks.
    ///
    /// This is the "Decode side" of the inverted control pattern:
    /// 1. Search local G2 and G3 for matches
    /// 2. Create a ControllableSession that holds the blocks
    /// 3. Return session_id to be sent to Prefill out-of-band
    ///
    /// By default, G3→G2 staging starts immediately (auto_stage=true).
    pub fn create_controllable_session(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<ControllableSessionResult> {
        self.create_controllable_session_with_options(
            sequence_hashes,
            ControllableSessionOptions::default(),
        )
    }

    /// Create a controllable session with custom options.
    ///
    /// Use this when you need to control auto-staging behavior.
    pub fn create_controllable_session_with_options(
        &self,
        sequence_hashes: &[SequenceHash],
        options: ControllableSessionOptions,
    ) -> Result<ControllableSessionResult> {
        let session_id = SessionId::from(Uuid::new_v4());

        // Local search only
        let matched_g2_blocks = self.g2_manager.match_blocks(sequence_hashes);

        // Find remaining hashes not in G2
        let remaining_hashes: Vec<_> = sequence_hashes
            .iter()
            .filter(|h| !matched_g2_blocks.iter().any(|b| b.sequence_hash() == **h))
            .copied()
            .collect();

        // Search G3 for remaining hashes
        let matched_g3_blocks = if let Some(ref g3_manager) = self.g3_manager {
            g3_manager.match_blocks(&remaining_hashes)
        } else {
            Vec::new()
        };

        let local_g2_count = matched_g2_blocks.len();
        let local_g3_count = matched_g3_blocks.len();

        // Create session channel using unified SessionMessage protocol
        let (tx, rx) = session_message_channel(100);
        self.session_sessions.insert(session_id, tx);

        // Collect G2 layout handles from workers for round-robin block allocation
        let worker_g2_handles: Vec<LayoutHandle> = self
            .parallel_worker
            .as_ref()
            .map(|pw| pw.workers().iter().filter_map(|w| w.g2_handle()).collect())
            .unwrap_or_default();

        let endpoint = super::session::SessionEndpoint::new(
            session_id,
            self.messenger.instance_id(),
            self.transport.clone(),
            rx,
        );

        let (cmd_tx, cmd_rx) = mpsc::channel(16);

        let session = ServerSession::new_with_staging(
            endpoint,
            BlockHolder::new(matched_g2_blocks),
            BlockHolder::new(matched_g3_blocks),
            worker_g2_handles,
            self.g2_manager.clone(),
            self.parallel_worker.clone(),
            cmd_rx,
            ServerSessionOptions {
                auto_stage: options.auto_stage,
            },
        );

        // Keep handle alive to prevent cmd channel from closing
        let _handle = ServerSessionHandle::new(session_id, self.messenger.instance_id(), cmd_tx);

        // Spawn session task
        let session_sessions = self.session_sessions.clone();
        tokio::spawn(async move {
            let _handle = _handle; // move handle into task to keep cmd channel open
            if let Err(e) = session.run().await {
                tracing::warn!(error = %e, "ServerSession error");
            }
            // Clean up when session completes
            session_sessions.remove(&session_id);
        });

        Ok(ControllableSessionResult {
            session_id,
            local_g2_count,
            local_g3_count,
        })
    }

    // ========================================================================
    // Unified Session Protocol
    // ========================================================================

    /// Attach to a remote session.
    /// Returns a `SessionHandle` that uses `SessionMessage` for communication.
    ///
    /// # Arguments
    /// * `remote_instance` - The instance hosting the session
    /// * `session_id` - The session to attach to
    ///
    /// # Example
    /// ```ignore
    /// let handle = leader.attach_session(remote_id, session_id).await?;
    /// let state = handle.wait_for_ready().await?;
    /// handle.trigger_staging().await?;
    /// ```
    pub async fn attach_session(
        &self,
        remote_instance: InstanceId,
        session_id: SessionId,
    ) -> Result<SessionHandle> {
        // Create local channel for receiving state updates
        let (state_tx, state_rx) = session_handle_state_channel();

        // Register handler for this session's messages
        let (msg_tx, msg_rx) = session_message_channel(100);
        self.session_sessions.insert(session_id, msg_tx);

        // Spawn receiver task to update state
        tokio::spawn(Self::run_session_receiver(msg_rx, state_tx));

        // Send attach message using new protocol
        let msg = SessionMessage::Attach {
            peer: self.messenger.instance_id(),
            session_id,
            as_role: ControlRole::Controller,
        };
        self.transport.send_session(remote_instance, msg).await?;

        let mut handle = SessionHandle::new(
            session_id,
            remote_instance,
            self.messenger.instance_id(),
            self.transport.clone(),
            state_rx,
        );

        // Add RDMA support if parallel worker is configured
        if let Some(parallel_worker) = &self.parallel_worker {
            handle = handle.with_rdma_support(parallel_worker.clone());
        }

        Ok(handle)
    }

    // ========================================================================
    // Endpoint Session Creation (Server-Side)
    // ========================================================================

    /// Create an endpoint session that a remote peer can attach to.
    ///
    /// This searches local G2/G3 for blocks matching the given sequence hashes
    /// and creates a session that exposes them for remote RDMA pull.
    ///
    /// Returns `(session_id, handle)` where:
    /// - `session_id` - Send to remote peer for attachment
    /// - `handle` - Use to control the session (send layer notifications, close)
    ///
    /// # Example
    /// ```ignore
    /// // Create session for sequence hashes
    /// let (session_id, handle) = leader.create_endpoint_session(&hashes)?;
    ///
    /// // Send session_id to remote peer out-of-band
    /// // Remote attaches via: remote_leader.attach_session(local_id, session_id)
    ///
    /// // For layerwise transfer, notify when layers are ready
    /// handle.notify_layers_ready(0..1).await?;
    /// ```
    pub fn create_endpoint_session(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<(SessionId, ServerSessionHandle)> {
        let session_id = SessionId::from(uuid::Uuid::new_v4());

        // Local search
        let matched_g2_blocks = self.g2_manager.match_blocks(sequence_hashes);

        // Collect layout handles from workers
        // Note: For single-worker setups, all blocks use the same handle
        // For multi-worker (SPMD), each block gets the handle from its assigned worker
        let worker_g2_handles: Vec<LayoutHandle> = self
            .parallel_worker
            .as_ref()
            .map(|pw| pw.workers().iter().filter_map(|w| w.g2_handle()).collect())
            .unwrap_or_default();

        // Assign layout handle to each matched block
        // For now, use the first worker's handle for all blocks (single-worker assumption)
        // TODO: For SPMD, map blocks to worker handles based on block assignment
        let layout_handle = worker_g2_handles
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("No G2 layout handle available from workers"))?;
        let layout_handles: Vec<LayoutHandle> = vec![layout_handle; matched_g2_blocks.len()];

        // Get sequence hashes from matched blocks
        let matched_hashes: Vec<SequenceHash> = matched_g2_blocks
            .iter()
            .map(|b| b.sequence_hash())
            .collect();

        // Create the session channel
        let (msg_tx, msg_rx) = session_message_channel(100);
        self.session_sessions.insert(session_id, msg_tx);

        // Create BlockHolder from matched blocks
        let block_holder = BlockHolder::new(matched_g2_blocks);

        // Create the session and handle
        let (session, handle) = create_server_session(
            session_id,
            self.messenger.instance_id(),
            block_holder,
            layout_handles,
            matched_hashes,
            self.transport.clone(),
            msg_rx,
        );

        // Spawn the session task
        let session_sessions = self.session_sessions.clone();
        tokio::spawn(async move {
            if let Err(e) = session.run().await {
                tracing::warn!(error = %e, "ServerSession error");
            }
            // Clean up when session completes
            session_sessions.remove(&session_id);
        });

        Ok((session_id, handle))
    }

    /// Create an endpoint session for specific pre-allocated blocks.
    ///
    /// Unlike `create_endpoint_session`, this doesn't search - it uses the
    /// provided blocks directly. Useful when the caller already has blocks
    /// to expose (e.g., after prefill computation).
    ///
    /// # Arguments
    /// * `blocks` - Blocks to expose for RDMA pull
    /// * `sequence_hashes` - Sequence hashes for the blocks (must match block count)
    /// * `layout_handles` - Layout handles for the blocks (must match block count)
    ///
    /// # Example
    /// ```ignore
    /// // After prefill computation, expose blocks for Decode to pull
    /// let (session_id, handle) = leader.create_endpoint_session_for_blocks(
    ///     prefill_blocks,
    ///     &hashes,
    ///     &layout_handles,
    /// )?;
    /// ```
    pub fn create_endpoint_session_for_blocks(
        &self,
        blocks: BlockHolder<G2>,
        sequence_hashes: &[SequenceHash],
        layout_handles: &[LayoutHandle],
    ) -> Result<(SessionId, ServerSessionHandle)> {
        let session_id = SessionId::from(uuid::Uuid::new_v4());

        // Create the session channel
        let (msg_tx, msg_rx) = session_message_channel(100);
        self.session_sessions.insert(session_id, msg_tx);

        // Create the session and handle
        let (session, handle) = create_server_session(
            session_id,
            self.messenger.instance_id(),
            blocks,
            layout_handles.to_vec(),
            sequence_hashes.to_vec(),
            self.transport.clone(),
            msg_rx,
        );

        // Spawn the session task
        let session_sessions = self.session_sessions.clone();
        tokio::spawn(async move {
            if let Err(e) = session.run().await {
                tracing::warn!(error = %e, "ServerSession error");
            }
            // Clean up when session completes
            session_sessions.remove(&session_id);
        });

        Ok((session_id, handle))
    }

    /// Internal: Process incoming SessionMessage for a session.
    async fn run_session_receiver(
        mut rx: mpsc::Receiver<SessionMessage>,
        state_tx: super::session::SessionHandleStateTx,
    ) {
        while let Some(msg) = rx.recv().await {
            match msg {
                SessionMessage::StateResponse { state, .. } => {
                    state_tx.update(state);
                }
                SessionMessage::BlocksStaged {
                    staged_blocks,
                    remaining,
                    layer_range,
                    ..
                } => {
                    state_tx.add_staged_blocks(staged_blocks, remaining, layer_range);
                }
                SessionMessage::Error { message, .. } => {
                    tracing::warn!(%message, "Session error");
                    state_tx.set_failed();
                    break;
                }
                SessionMessage::Close { .. } => {
                    state_tx.set_phase(SessionPhase::Complete);
                    break;
                }
                _ => {
                    // Ignore control commands (sent by controller, not received)
                }
            }
        }
    }

    /// Get the session sessions map (for Velo handler registration).
    #[expect(dead_code)]
    pub(crate) fn session_sessions(&self) -> Arc<DashMap<SessionId, SessionMessageTx>> {
        self.session_sessions.clone()
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
        let mut transfer_opts = TransferOptions::default();
        transfer_opts.nixl_write_notification = opts.nixl_write_notification;
        transfer_opts.metric_route = opts.metric_route;

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
        KvBlockLayout::UniversalTP => "universal_tp".to_owned(),
        KvBlockLayout::UniversalPP => "universal_pp".to_owned(),
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
        // For distributed search, remote instances use scan_matches for broad coverage,
        // then first-hole filtering is applied in InitiatorSession after aggregation.

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

        // Determine if we can return immediately (Ready) or need async session
        // Ready if:
        //   - g3 blocks is empty
        //   - AND NOT (search_remote AND has_remote_leaders)
        //   - AND NOT (search_remote AND has_object_client)
        //
        // AsyncSession (is_ready=false) if:
        //   - g3 is not empty, or
        //   - search_remote is true AND (has_remote_leaders OR has_object_client)
        let has_remote_leaders = !self.remote_leaders.read().unwrap().is_empty();
        let has_object_client = self.object_client.is_some();
        let needs_remote_search =
            options.search_remote && (has_remote_leaders || has_object_client);
        let is_ready = matched_g3_blocks.is_empty() && !needs_remote_search;
        let local_g2_count = matched_g2_blocks.len();
        let local_g3_count = matched_g3_blocks.len();

        if is_ready {
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

        // AsyncSession path: G3 blocks found or remote search enabled
        let session_id = SessionId::from(Uuid::new_v4());

        // AsyncSession: staging locally and/or remote searching
        let (status_tx, status_rx) = watch::channel(OnboardingStatus::Searching);
        let all_g2_blocks = Arc::new(Mutex::new(None));
        let match_breakdown = Arc::new(Mutex::new(super::MatchBreakdown {
            host_blocks: local_g2_count,
            disk_blocks: local_g3_count,
            object_blocks: 0,
        }));

        // Store session state to keep blocks alive
        let state = SessionState {
            session_id,
            matched_g2_blocks,
            matched_g3_blocks,
            status_tx: status_tx.clone(),
        };
        self.store_session_state(state);

        // If no remote search, handle local-only staging
        if !options.search_remote {
            // Local-only staging (Prepare or Full mode)
            // TODO: Implement local G3→G2 staging
            let total_matched = local_g2_count + local_g3_count;
            status_tx
                .send(OnboardingStatus::Complete {
                    matched_blocks: total_matched,
                })
                .ok();

            return Ok(FindMatchesResult::AsyncSession(AsyncSessionResult::new(
                session_id,
                status_rx,
                all_g2_blocks,
                match_breakdown,
                None, // No session handle for local-only staging (yet)
            )));
        }

        // Remote search path
        let (tx, rx) = mpsc::channel(100);
        self.sessions.insert(session_id, tx);

        // Create control channel for Hold/Prepare modes
        let (session_handle, control_rx) = if matches!(
            options.staging_mode,
            StagingMode::Hold | StagingMode::Prepare
        ) {
            let (control_tx, control_rx) = mpsc::channel(10);
            let handle = LegacySessionHandle::new(session_id, options.staging_mode, control_tx);
            (Some(handle), Some(control_rx))
        } else {
            (None, None)
        };

        let session = InitiatorSession::new(
            session_id,
            self.messenger.instance_id(),
            options.staging_mode,
            self.g2_manager.clone(),
            self.g3_manager.clone(),
            self.parallel_worker.clone(),
            self.transport.clone(),
            status_tx.clone(),
            all_g2_blocks.clone(),
            match_breakdown.clone(),
            control_rx.unwrap_or_else(|| {
                let (_, rx) = mpsc::channel(1);
                rx
            }),
            self.object_client.clone(),
        );

        let remote_leaders = self.remote_leaders.read().unwrap().clone();
        let sequence_hashes = sequence_hashes.to_vec();

        let handle = self.messenger.runtime();

        handle.spawn(async move {
            if let Err(e) = session.run(rx, remote_leaders, sequence_hashes).await {
                tracing::warn!(error = %e, "InitiatorSession error");
                // Try to update status to indicate error
                status_tx
                    .send(OnboardingStatus::Complete { matched_blocks: 0 })
                    .ok();
            }
        });

        Ok(FindMatchesResult::AsyncSession(AsyncSessionResult::new(
            session_id,
            status_rx,
            all_g2_blocks,
            match_breakdown,
            session_handle,
        )))
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::G2;
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
        }
    }

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
        assert!(!leader.set_modules(vec![kvbm_protocols::control::ModuleId::Test]));
        let d2 = leader.describe().await.expect("describe ok");
        assert_eq!(d2.modules, vec![kvbm_protocols::control::ModuleId::Core]);
        Ok(())
    }
}
