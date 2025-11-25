// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dashmap::DashMap;
use dynamo_nova::am::Nova;
use tokio::sync::{Mutex, mpsc, watch};
use uuid::Uuid;

use std::sync::Arc;

use crate::{
    logical::{blocks::ImmutableBlock, manager::BlockManager},
    physical::transfer::{TransferCompleteNotification, TransferOptions},
    v2::{
        BlockId, InstanceId, SequenceHash,
        distributed::worker::RemoteDescriptor,
        integrations::{G2, G3},
        logical::LogicalLayoutHandle,
    },
};

use super::{
    super::worker::Worker,
    FindMatchesOptions, FindMatchesResult, Leader, OnboardingStatus, SessionHandle, SessionId,
    StagingMode,
    nova::NovaLeaderService,
    session::{
        InitiatorSession, MessageTransport, NovaTransport, OnboardMessage, OnboardSessionTx,
        ResponderSession,
    },
};

/// Represents a leader instance in the distributed KVBM system.
///
/// The InstanceLeader coordinates block onboarding across local and remote instances,
/// managing G2 (host memory) and optional G3 (disk) block managers.
#[derive(Clone)]
pub struct InstanceLeader {
    /// The instance ID of this leader.
    instance_id: InstanceId,

    /// Runtime handle for spawning async tasks.
    rt: tokio::runtime::Handle,

    /// Nova instance for distributed communication.
    nova: Arc<Nova>,

    /// G2 (host memory) block manager (wrapped in Arc since BlockManager doesn't implement Clone).
    g2_manager: Arc<BlockManager<G2>>,

    /// Optional G3 (disk) block manager
    g3_manager: Option<Arc<BlockManager<G3>>>,

    /// Workers for executing transfers (at least 1 required).
    /// Multiple workers enable parallel transfers and redundancy.
    workers: Vec<Arc<dyn Worker>>,

    /// Map of active sessions (session_id -> message channel).
    sessions: Arc<DashMap<SessionId, OnboardSessionTx>>,

    /// Map of session states for holding blocks alive (RAII).
    session_states: Arc<DashMap<SessionId, SessionState>>,

    /// List of remote leader instance IDs.
    remote_leaders: Vec<InstanceId>,

    /// Message transport for session communication.
    transport: Arc<dyn MessageTransport>,
}

/// Builder for InstanceLeader.
#[derive(Default)]
pub struct InstanceLeaderBuilder {
    instance_id: Option<InstanceId>,
    rt: Option<tokio::runtime::Handle>,
    nova: Option<Arc<Nova>>,
    g2_manager: Option<Arc<BlockManager<G2>>>,
    g3_manager: Option<Arc<BlockManager<G3>>>,
    workers: Vec<Arc<dyn Worker>>,
    sessions: Option<Arc<DashMap<SessionId, OnboardSessionTx>>>,
    remote_leaders: Option<Vec<InstanceId>>,
}

impl InstanceLeaderBuilder {
    pub fn instance_id(mut self, id: InstanceId) -> Self {
        self.instance_id = Some(id);
        self
    }

    pub fn nova(mut self, nova: Arc<Nova>) -> Self {
        self.nova = Some(nova);
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

    pub fn build(self) -> Result<InstanceLeader> {
        let nova = self
            .nova
            .ok_or_else(|| anyhow::anyhow!("Nova instance required"))?;
        let transport = Arc::new(NovaTransport::new(nova.clone()));

        // // Validate at least one worker
        // if self.workers.is_empty() {
        //     anyhow::bail!("At least one worker required");
        // }

        Ok(InstanceLeader {
            instance_id: self
                .instance_id
                .ok_or_else(|| anyhow::anyhow!("instance_id required"))?,
            rt: self.rt.unwrap_or_else(tokio::runtime::Handle::current),
            nova,
            g2_manager: self
                .g2_manager
                .ok_or_else(|| anyhow::anyhow!("g2_manager required"))?,
            g3_manager: self.g3_manager,
            workers: self.workers,
            sessions: self.sessions.unwrap_or_else(|| Arc::new(DashMap::new())),
            session_states: Arc::new(DashMap::new()),
            remote_leaders: self.remote_leaders.unwrap_or_default(),
            transport,
        })
    }
}

/// Internal session state for holding matched blocks.
#[allow(dead_code)] // Used for RAII block lifetime management
struct SessionState {
    session_id: SessionId,
    g2_blocks: Vec<ImmutableBlock<G2>>,
    g3_blocks: Vec<ImmutableBlock<G3>>,
    status_tx: watch::Sender<OnboardingStatus>,
}

impl InstanceLeader {
    pub fn builder() -> InstanceLeaderBuilder {
        InstanceLeaderBuilder::default()
    }

    /// Register Nova handlers for leader-to-leader communication.
    ///
    /// This must be called after construction to enable distributed onboarding.
    pub fn register_handlers(&self) -> Result<()> {
        let instance_id = self.instance_id;
        let g2_manager = self.g2_manager.clone();
        let g3_manager = self.g3_manager.clone();
        // TODO: Pass all workers to session or use aggregated transfer methods
        let worker = self.workers.first().cloned();
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
                    worker.clone(),
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

        NovaLeaderService::new(self.nova.clone(), self.sessions.clone())
            .with_spawn_responder(spawn_responder)
            .register_handlers()?;

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
    pub fn release_session(&self, session_id: SessionId) {
        self.session_states.remove(&session_id);
        self.sessions.remove(&session_id);
    }

    // ========================================================================
    // Private Worker Mirror Methods
    // These methods execute operations across all workers and aggregate results.
    // ========================================================================

    /// Execute local transfer across all workers, returning aggregated notification.
    ///
    /// TODO: Implement proper notification aggregation using LocalEventSystem::merge_events
    /// or create a composite notification that triggers when all workers complete.
    /// Current implementation uses first worker only.
    #[allow(dead_code)]
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst_block_ids: Vec<BlockId>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        // TODO: Fan out to all workers and aggregate notifications
        // For now, use first worker
        // self.workers[0].execute_local_transfer(src, dst, src_block_ids, dst_block_ids, options)
        todo!("implement local transfer")
    }

    /// Execute remote onboard across all workers, returning aggregated notification.
    ///
    /// TODO: Implement proper notification aggregation.
    /// Current implementation uses first worker only.
    #[allow(dead_code)]
    fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Vec<BlockId>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        // TODO: Fan out to all workers and aggregate notifications
        // self.workers[0].execute_remote_onboard(src, dst, dst_block_ids, options)
        todo!("implement remote onboard")
    }

    /// Execute remote offload across all workers, returning aggregated notification.
    ///
    /// TODO: Implement proper notification aggregation.
    /// Current implementation uses first worker only.
    #[allow(dead_code)]
    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        dst: RemoteDescriptor,
        src_block_ids: Vec<BlockId>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        // TODO: Fan out to all workers and aggregate notifications
        // self.workers[0].execute_remote_offload(src, dst, src_block_ids, options)
        todo!("implement remote offload")
    }
}

impl Leader for InstanceLeader {
    fn find_matches_with_options(
        &self,
        sequence_hashes: &[SequenceHash],
        options: FindMatchesOptions,
    ) -> Result<FindMatchesResult> {
        let session_id = SessionId::from(Uuid::new_v4());
        let (status_tx, status_rx) = watch::channel(OnboardingStatus::Searching);
        let all_g2_blocks = Arc::new(Mutex::new(None));

        // Phase 1: Local search only
        if !options.search_remote {
            // Search G2 (host memory) for matches
            let g2_matches = self.g2_manager.match_blocks(sequence_hashes);
            let matched_count = g2_matches.len();

            // Search G3 (disk) for remaining hashes if G3 is available
            let remaining_hashes: Vec<_> = sequence_hashes
                .iter()
                .filter(|h| !g2_matches.iter().any(|b| b.sequence_hash() == **h))
                .copied()
                .collect();

            let g3_matches = if let Some(ref g3_manager) = self.g3_manager {
                g3_manager.match_blocks(&remaining_hashes)
            } else {
                Vec::new()
            };

            let total_matched = matched_count + g3_matches.len();

            // todo: what are the conditions where we might early exit here without the need for a session?

            // Update status to Complete
            status_tx
                .send(OnboardingStatus::Complete {
                    matched: total_matched,
                })
                .ok();

            // Store session state to keep blocks alive
            let state = SessionState {
                session_id,
                g2_blocks: g2_matches,
                g3_blocks: g3_matches,
                status_tx,
            };
            self.store_session_state(state);

            return Ok(FindMatchesResult::new(
                session_id,
                status_rx,
                all_g2_blocks,
                None, // No session handle for local-only search
            ));
        }

        // Phase 2: Remote search
        let (tx, rx) = mpsc::channel(100);
        self.sessions.insert(session_id, tx);

        // Create control channel for Hold/Prepare modes
        let (session_handle, control_rx) = if matches!(
            options.staging_mode,
            StagingMode::Hold | StagingMode::Prepare
        ) {
            let (control_tx, control_rx) = mpsc::channel(10);
            let handle = SessionHandle::new(session_id, options.staging_mode, control_tx);
            (Some(handle), Some(control_rx))
        } else {
            (None, None)
        };

        // TODO: Pass all workers or use aggregated transfer methods
        let worker = self.workers.first().cloned();

        let session = InitiatorSession::new(
            session_id,
            self.instance_id,
            options.staging_mode,
            self.g2_manager.clone(),
            self.g3_manager.clone(),
            worker,
            self.transport.clone(),
            status_tx.clone(),
            all_g2_blocks.clone(),
            control_rx.unwrap_or_else(|| {
                let (_, rx) = mpsc::channel(1);
                rx
            }),
        );

        let remote_leaders = self.remote_leaders.clone();
        let sequence_hashes = sequence_hashes.to_vec();

        tokio::spawn(async move {
            if let Err(e) = session.run(rx, remote_leaders, sequence_hashes).await {
                eprintln!("InitiatorSession error: {e}");
                // Try to update status to indicate error
                status_tx
                    .send(OnboardingStatus::Complete { matched: 0 })
                    .ok();
            }
        });

        Ok(FindMatchesResult::new(
            session_id,
            status_rx,
            all_g2_blocks,
            session_handle,
        ))
    }
}
