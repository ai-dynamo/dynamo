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

use crate::v2::physical::manager::LayoutHandle;

use super::{
    super::worker::Worker,
    FindMatchesOptions, FindMatchesResult, Leader, OnboardingStatus, SessionHandle, SessionId,
    StagingMode,
    nova::NovaLeaderService,
    session::{
        ControllableSession, ControllableSessionOptions, ControllableSessionResult,
        InitiatorSession, MessageTransport, OnboardMessage, OnboardSessionTx, RemoteSessionHandle,
        RemoteSessionMessage, RemoteSessionTx, ResponderSession, remote_session_state_channel,
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
}

/// Builder for InstanceLeader.
#[derive(Default)]
pub struct InstanceLeaderBuilder {
    nova: Option<Arc<Nova>>,
    g2_manager: Option<Arc<BlockManager<G2>>>,
    g3_manager: Option<Arc<BlockManager<G3>>>,
    workers: Vec<Arc<dyn Worker>>,
    sessions: Option<Arc<DashMap<SessionId, OnboardSessionTx>>>,
    remote_leaders: Option<Vec<InstanceId>>,
}

impl InstanceLeaderBuilder {
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
        let transport = Arc::new(MessageTransport::nova(nova.clone()));

        // // Validate at least one worker
        // if self.workers.is_empty() {
        //     anyhow::bail!("At least one worker required");
        // }

        Ok(InstanceLeader {
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
            controllable_sessions: Arc::new(DashMap::new()),
            remote_sessions: Arc::new(DashMap::new()),
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
        let instance_id = self.nova.instance_id();
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
            .with_controllable_sessions(self.controllable_sessions.clone())
            .with_remote_sessions(self.remote_sessions.clone())
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
        let g2_matches = self.g2_manager.match_blocks(sequence_hashes);

        // Find remaining hashes not in G2
        let remaining_hashes: Vec<_> = sequence_hashes
            .iter()
            .filter(|h| !g2_matches.iter().any(|b| b.sequence_hash() == **h))
            .copied()
            .collect();

        // Search G3 for remaining hashes
        let g3_matches = if let Some(ref g3_manager) = self.g3_manager {
            g3_manager.match_blocks(&remaining_hashes)
        } else {
            Vec::new()
        };

        let local_g2_count = g2_matches.len();
        let local_g3_count = g3_matches.len();

        // Create session channel
        let (tx, rx) = mpsc::channel(100);
        self.controllable_sessions.insert(session_id, tx);

        // TODO: Get actual G2 layout handle from manager
        let g2_layout_handle = LayoutHandle::new(0, 0); // Placeholder

        // Create controllable session
        let session = ControllableSession::new(
            session_id,
            self.nova.instance_id(),
            g2_matches,
            g3_matches,
            g2_layout_handle,
            self.g2_manager.clone(),
            self.g3_manager.clone(),
            self.workers.first().cloned(),
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
    /// 3. Use handle to query state, trigger staging, pull blocks
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

        Ok(RemoteSessionHandle::new(
            session_id,
            remote_instance,
            self.nova.instance_id(),
            self.transport.clone(),
            state_rx,
        ))
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
    pub(crate) fn controllable_sessions(&self) -> Arc<DashMap<SessionId, RemoteSessionTx>> {
        self.controllable_sessions.clone()
    }

    /// Get the remote sessions map (for Nova handler registration).
    pub(crate) fn remote_sessions(&self) -> Arc<DashMap<SessionId, RemoteSessionTx>> {
        self.remote_sessions.clone()
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
            self.nova.instance_id(),
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
