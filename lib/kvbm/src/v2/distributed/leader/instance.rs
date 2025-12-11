// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use anyhow::Result;
use dashmap::DashMap;
use dynamo_nova::{am::Nova, events::LocalEventSystem};
use tokio::sync::{Mutex, mpsc, watch};
use uuid::Uuid;

use std::sync::Arc;

use crate::{
    logical::{
        blocks::{BlockRegistry, ImmutableBlock},
        manager::BlockManager,
    },
    physical::transfer::{TransferCompleteNotification, TransferOptions},
    v2::{
        BlockId, G2, G3, InstanceId, SequenceHash, distributed::worker::RemoteDescriptor,
        logical::LogicalLayoutHandle,
    },
};

use crate::v2::physical::manager::{LayoutHandle, SerializedLayout};

use super::{
    super::parallelism::{ParallelWorker, ReplicatedWorker},
    super::worker::Worker,
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
    nova::{ExportMetadataCallback, NovaLeaderService},
    session::{
        BlockHolder,
        ControlRole,
        ControllableSession,
        ControllableSessionOptions,
        ControllableSessionResult,
        EndpointSessionHandle,
        InitiatorSession,
        MessageTransport,
        OnboardMessage,
        OnboardSessionTx,
        RemoteSessionHandle,
        RemoteSessionMessage,
        RemoteSessionTx,
        ResponderSession,
        // Import the new unified SessionHandle with an alias to distinguish from legacy
        SessionHandle as UnifiedSessionHandle,
        SessionMessage,
        SessionMessageTx,
        SessionPhase,
        create_endpoint_session,
        remote_session_state_channel,
        session_handle_state_channel,
        session_message_channel,
    },
};

/// Represents a leader instance in the distributed KVBM system.
///
/// The InstanceLeader coordinates block onboarding across local and remote instances,
/// managing G2 (host memory) and optional G3 (disk) block managers.
#[derive(Clone)]
pub struct InstanceLeader {
    /// Nova instance for distributed communication.
    nova: Arc<Nova>,

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
    parallel_worker: Option<Arc<dyn ParallelWorker>>,

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
    // Inverted Control Pattern (Prefill-Decode) Fields
    // ========================================================================
    /// Map of controllable sessions (Decode side).
    /// Used when this instance hosts sessions that can be controlled remotely.
    controllable_sessions: Arc<DashMap<SessionId, RemoteSessionTx>>,

    /// Map of remote session receivers (Prefill side).
    /// Used when this instance controls sessions on remote instances.
    remote_sessions: Arc<DashMap<SessionId, RemoteSessionTx>>,

    // ========================================================================
    // Unified Session Protocol (New)
    // ========================================================================
    /// Map of unified session message receivers.
    /// Used by the new SessionHandle/SessionEndpoint protocol.
    session_sessions: Arc<DashMap<SessionId, SessionMessageTx>>,
}

/// Builder for InstanceLeader.
#[derive(Default)]
pub struct InstanceLeaderBuilder {
    nova: Option<Arc<Nova>>,
    registry: Option<BlockRegistry>,
    g2_manager: Option<Arc<BlockManager<G2>>>,
    g3_manager: Option<Arc<BlockManager<G3>>>,
    workers: Vec<Arc<dyn Worker>>,
    sessions: Option<Arc<DashMap<SessionId, OnboardSessionTx>>>,
    remote_leaders: Option<Vec<InstanceId>>,
    cached_worker_metadata: Option<Vec<SerializedLayout>>,
}

impl InstanceLeaderBuilder {
    /// Initialize builder with components from KvbmRuntime.
    ///
    /// This extracts Nova from the runtime. Use this when the runtime
    /// has already been constructed and you want the leader to share
    /// the same Nova instance for distributed communication.
    ///
    /// # Example
    /// ```ignore
    /// let runtime = KvbmRuntime::from_env_leader().await?;
    /// let leader = InstanceLeaderBuilder::default()
    ///     .with_runtime(&runtime)
    ///     .g2_manager(g2_manager)
    ///     .build()?;
    /// ```
    pub fn with_runtime(self, runtime: &crate::v2::KvbmRuntime) -> Self {
        self.nova(runtime.nova().clone())
    }

    pub fn nova(mut self, nova: Arc<Nova>) -> Self {
        self.nova = Some(nova);
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

    pub fn build(self) -> Result<InstanceLeader> {
        let nova = self
            .nova
            .ok_or_else(|| anyhow::anyhow!("Nova instance required"))?;
        let transport = Arc::new(MessageTransport::nova(nova.clone()));

        // Create event system for notification aggregation
        // worker_id is used for event handle generation - use 0 for the leader
        let events = LocalEventSystem::new(0);

        // Get current tokio runtime handle
        let runtime = tokio::runtime::Handle::current();

        // // Validate at least one worker
        // if self.workers.is_empty() {
        //     anyhow::bail!("At least one worker required");
        // }

        // todo: we will need a common builder pattern for creating "general" parallel workers
        // - we could also use an enum and match as the number of types will be limited

        // Create parallel worker if workers are provided
        let parallel_worker: Option<Arc<dyn ParallelWorker>> = if !self.workers.is_empty() {
            Some(Arc::new(ReplicatedWorker::new(
                self.workers.to_vec(),
                events.clone(),
                runtime.clone(),
            )))
        } else {
            None
        };

        Ok(InstanceLeader {
            nova,
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
            controllable_sessions: Arc::new(DashMap::new()),
            remote_sessions: Arc::new(DashMap::new()),
            session_sessions: Arc::new(DashMap::new()),
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

    /// Get a reference to the Nova instance.
    ///
    /// This provides access to the Nova distributed system for features
    /// like event coordination and cross-instance communication.
    pub fn nova(&self) -> &Arc<Nova> {
        &self.nova
    }

    /// Get the tokio runtime handle from Nova.
    ///
    /// This handle should be used for spawning background tasks that need to
    /// run on the KVBM runtime's executor (e.g., offload engine pipelines).
    pub fn runtime(&self) -> tokio::runtime::Handle {
        self.nova.runtime().clone()
    }

    /// Check if a parallel_worker is configured.
    ///
    /// The parallel_worker is required for local transfer operations
    /// (e.g., offloading blocks between tiers).
    pub fn has_parallel_worker(&self) -> bool {
        self.parallel_worker.is_some()
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

    /// Register Nova handlers for leader-to-leader communication.
    ///
    /// This must be called after construction to enable distributed onboarding.
    pub fn register_handlers(&self) -> Result<()> {
        let instance_id = self.nova.instance_id();
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
                        eprintln!("ResponderSession error: {e}");
                    }
                });

                Ok(())
            } else {
                anyhow::bail!("spawn_responder called with non-CreateSession message")
            }
        };

        // Create export_metadata callback if we have workers or cached metadata
        let export_metadata_callback: Option<ExportMetadataCallback> =
            if !self.workers.is_empty() || self.cached_worker_metadata.is_some() {
                let workers = self.workers.clone();
                let cached_metadata = self.cached_worker_metadata.clone();
                Some(Arc::new(move || {
                    let workers = workers.clone();
                    let cached_metadata = cached_metadata.clone();
                    Box::pin(async move {
                        // Return cached metadata if available
                        if let Some(cached) = cached_metadata {
                            return Ok(cached);
                        }
                        // Otherwise, query workers
                        let mut metadata = Vec::with_capacity(workers.len());
                        for worker in &workers {
                            let serialized = worker.export_metadata()?.await?;
                            metadata.push(serialized);
                        }
                        Ok(metadata)
                    })
                }))
            } else {
                None
            };

        let mut service = NovaLeaderService::new(self.nova.clone(), self.sessions.clone())
            .with_spawn_responder(spawn_responder)
            .with_controllable_sessions(self.controllable_sessions.clone())
            .with_remote_sessions(self.remote_sessions.clone())
            .with_session_sessions(self.session_sessions.clone());

        if let Some(callback) = export_metadata_callback {
            service = service.with_export_metadata(callback);
        }

        service.register_handlers()?;

        Ok(())
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
    ///
    /// This cleans up all session types: legacy onboard sessions, remote sessions,
    /// and unified sessions.
    pub fn release_session(&self, session_id: SessionId) {
        self.session_states.remove(&session_id);
        self.sessions.remove(&session_id);
        self.remote_sessions.remove(&session_id);
        self.session_sessions.remove(&session_id);
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

        // Create session channel
        let (tx, rx) = mpsc::channel(100);
        self.controllable_sessions.insert(session_id, tx);

        // Collect G2 layout handles from workers for round-robin block allocation
        let worker_g2_handles: Vec<LayoutHandle> = self
            .parallel_worker
            .as_ref()
            .map(|pw| pw.workers().iter().filter_map(|w| w.g2_handle()).collect())
            .unwrap_or_default();

        // Create controllable session
        let session = ControllableSession::new(
            session_id,
            self.nova.instance_id(),
            matched_g2_blocks,
            matched_g3_blocks,
            worker_g2_handles,
            self.g2_manager.clone(),
            self.g3_manager.clone(),
            self.parallel_worker.clone(),
            self.transport.clone(),
            rx,
            options,
        );

        // Spawn session task
        let controllable_sessions = self.controllable_sessions.clone();
        tokio::spawn(async move {
            if let Err(e) = session.run().await {
                eprintln!("ControllableSession error: {e}");
            }
            // Clean up when session completes
            controllable_sessions.remove(&session_id);
        });

        Ok(ControllableSessionResult {
            session_id,
            local_g2_count,
            local_g3_count,
        })
    }

    /// Attach to a remote session on another instance (Decode).
    ///
    /// This is the "Prefill side" of the inverted control pattern:
    /// 1. Send AttachSession to Decode
    /// 2. Receive RemoteSessionHandle for controlling the session
    /// 3. Use handle to query state, trigger staging, pull blocks via RDMA
    ///
    /// If workers are configured, the handle will have RDMA support enabled,
    /// allowing use of `pull_blocks_rdma()` methods.
    #[deprecated(note = "Use attach_session instead")]
    pub async fn attach_remote_session(
        &self,
        remote_instance: InstanceId,
        session_id: SessionId,
    ) -> Result<RemoteSessionHandle> {
        // Create local channel for receiving state updates
        let (state_tx, state_rx) = remote_session_state_channel();

        // Register handler for this session's messages
        let (msg_tx, msg_rx) = mpsc::channel(100);
        self.remote_sessions.insert(session_id, msg_tx);

        // Spawn receiver task to update state
        tokio::spawn(Self::run_remote_session_receiver(msg_rx, state_tx));

        // Send attach message
        let msg = RemoteSessionMessage::AttachSession {
            controller: self.nova.instance_id(),
            session_id,
        };
        self.transport
            .send_remote_session(remote_instance, msg)
            .await?;

        let mut handle = RemoteSessionHandle::new(
            session_id,
            remote_instance,
            self.nova.instance_id(),
            self.transport.clone(),
            state_rx,
        );

        // Add RDMA support if parallel worker is configured
        if let Some(parallel_worker) = &self.parallel_worker {
            handle = handle.with_rdma_support(parallel_worker.clone());
        }

        Ok(handle)
    }

    /// Internal: Process incoming messages for a remote session.
    async fn run_remote_session_receiver(
        mut rx: mpsc::Receiver<RemoteSessionMessage>,
        state_tx: super::session::RemoteSessionStateTx,
    ) {
        while let Some(msg) = rx.recv().await {
            match msg {
                RemoteSessionMessage::SessionState {
                    g2_blocks,
                    g3_pending_count,
                    g3_blocks,
                    phase,
                    ..
                } => {
                    state_tx.update_from_session_state(
                        g2_blocks,
                        g3_pending_count,
                        g3_blocks,
                        phase,
                    );
                }
                RemoteSessionMessage::BlocksStaged {
                    staged_blocks,
                    g3_remaining_count,
                    ..
                } => {
                    state_tx.update_from_blocks_staged(staged_blocks, g3_remaining_count);
                }
                RemoteSessionMessage::SessionError { error, .. } => {
                    eprintln!("Remote session error: {}", error);
                    break;
                }
                _ => {
                    // Ignore outbound message types (AttachSession, TriggerStaging, etc.)
                }
            }
        }
    }

    /// Release resources for a remote session handle.
    pub fn release_remote_session(&self, session_id: SessionId) {
        self.remote_sessions.remove(&session_id);
    }

    /// Get the controllable sessions map (for Nova handler registration).
    #[expect(dead_code)]
    pub(crate) fn controllable_sessions(&self) -> Arc<DashMap<SessionId, RemoteSessionTx>> {
        self.controllable_sessions.clone()
    }

    /// Get the remote sessions map (for Nova handler registration).
    #[expect(dead_code)]
    pub(crate) fn remote_sessions(&self) -> Arc<DashMap<SessionId, RemoteSessionTx>> {
        self.remote_sessions.clone()
    }

    // ========================================================================
    // Unified Session Protocol (New)
    // These methods use the new SessionHandle/SessionMessage protocol.
    // ========================================================================

    /// Attach to a remote session using the unified session protocol.
    ///
    /// This is the preferred method for new code, replacing `attach_remote_session`.
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
    ) -> Result<UnifiedSessionHandle> {
        // Create local channel for receiving state updates
        let (state_tx, state_rx) = session_handle_state_channel();

        // Register handler for this session's messages
        let (msg_tx, msg_rx) = session_message_channel(100);
        self.session_sessions.insert(session_id, msg_tx);

        // Spawn receiver task to update state
        tokio::spawn(Self::run_session_receiver(msg_rx, state_tx));

        // Send attach message using new protocol
        let msg = SessionMessage::Attach {
            peer: self.nova.instance_id(),
            session_id,
            as_role: ControlRole::Controller,
        };
        self.transport.send_session(remote_instance, msg).await?;

        let mut handle = UnifiedSessionHandle::new(
            session_id,
            remote_instance,
            self.nova.instance_id(),
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
    ) -> Result<(SessionId, EndpointSessionHandle)> {
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
        let (session, handle) = create_endpoint_session(
            session_id,
            self.nova.instance_id(),
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
                eprintln!("EndpointSession error: {e}");
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
    ) -> Result<(SessionId, EndpointSessionHandle)> {
        let session_id = SessionId::from(uuid::Uuid::new_v4());

        // Create the session channel
        let (msg_tx, msg_rx) = session_message_channel(100);
        self.session_sessions.insert(session_id, msg_tx);

        // Create the session and handle
        let (session, handle) = create_endpoint_session(
            session_id,
            self.nova.instance_id(),
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
                eprintln!("EndpointSession error: {e}");
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
                    eprintln!("Session error: {}", message);
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

    /// Get the session sessions map (for Nova handler registration).
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
    /// Returns a Vec<SerializedLayout> where each element corresponds to a worker
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

        // Check if already loaded
        if parallel_worker.has_remote_metadata(remote_instance) {
            anyhow::bail!("Metadata already imported for instance {}", remote_instance);
        }

        // Connect to remote - this imports metadata and stores handle mappings
        parallel_worker
            .connect_remote(remote_instance, metadata)?
            .await?;

        Ok(())
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
        let matched_g2_blocks = self.g2_manager.match_blocks(sequence_hashes);

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
        //   - there are no remote leaders
        //   - OR there are remote leaders, but search_remote is false
        //
        // is_ready is false if:
        //   - g3 is not empty, or
        //   - remote_search is true and there are remote leaders
        let has_remote_leaders = !self.remote_leaders.read().unwrap().is_empty();
        let is_ready = matched_g3_blocks.is_empty()
            && (!has_remote_leaders || (has_remote_leaders && !options.search_remote));

        if is_ready {
            // No session needed - blocks owned directly by ReadyResult (RAII)
            return Ok(FindMatchesResult::Ready(ReadyResult::new(
                matched_g2_blocks,
            )));
        }

        // AsyncSession path: G3 blocks found or remote search enabled
        let session_id = SessionId::from(Uuid::new_v4());
        let local_g2_count = matched_g2_blocks.len();
        let local_g3_count = matched_g3_blocks.len();

        // AsyncSession: staging locally and/or remote searching
        let (status_tx, status_rx) = watch::channel(OnboardingStatus::Searching);
        let all_g2_blocks = Arc::new(Mutex::new(None));

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
            self.nova.instance_id(),
            options.staging_mode,
            self.g2_manager.clone(),
            self.g3_manager.clone(),
            self.parallel_worker.clone(),
            self.transport.clone(),
            status_tx.clone(),
            all_g2_blocks.clone(),
            control_rx.unwrap_or_else(|| {
                let (_, rx) = mpsc::channel(1);
                rx
            }),
        );

        let remote_leaders = self.remote_leaders.read().unwrap().clone();
        let sequence_hashes = sequence_hashes.to_vec();

        let handle = self.nova.runtime();

        handle.spawn(async move {
            if let Err(e) = session.run(rx, remote_leaders, sequence_hashes).await {
                eprintln!("InitiatorSession error: {e}");
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
            session_handle,
        )))
    }
}
