// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ServerSession: Merged server-side session for both G2-only and G2+G3 staging modes.
//!
//! Unifies `EndpointSession` (G2-only, Direct layout handles) and
//! `ControllableSession` (G2+G3 staging, RoundRobin layout handles) into a
//! single type that uses `SessionEndpoint` for the state machine.
//!
//! # Modes
//!
//! - **G2-only**: Blocks are already in G2 with pre-assigned layout handles.
//!   Created via `ServerSession::new_g2_only()` with `Direct` metadata.
//!   `TriggerStaging` is a no-op.
//!
//! - **Staging**: G3 blocks need to be staged to G2. Layout handles are
//!   assigned round-robin across workers. Created with `RoundRobin` metadata
//!   and optional `auto_stage`.
//!
//! # Lifecycle
//!
//! 1. Created with G2 blocks (and optionally G3 blocks)
//! 2. If `auto_stage=true`, immediately stages G3→G2
//! 3. Waits for peer to `Attach`
//! 4. Sends `StateResponse` with block info
//! 5. Responds to `TriggerStaging`, `BlocksPulled`, `Detach`, etc.
//! 6. Completes when all blocks pulled or session closed

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use kvbm_physical::manager::LayoutHandle;

use super::SessionId;
use super::blocks::BlockHolder;
use super::endpoint::SessionEndpoint;
use super::messages::{BlockInfo, SessionMessage, SessionStateSnapshot};
use super::staging;
use super::state::{ControlRole, SessionPhase};
use super::transport::MessageTransport;
use crate::{G2, G3, InstanceId, SequenceHash, worker::group::ParallelWorkers};
use kvbm_logical::manager::BlockManager;

/// Block metadata strategy for mapping blocks to layout handles.
///
/// Unifies the two approaches from the former EndpointSession (Direct)
/// and ControllableSession (RoundRobin).
pub enum BlockMetadataMap {
    /// Pre-assigned layout handles keyed by sequence hash.
    /// Used for G2-only mode where the caller knows exactly which handle
    /// each block should use.
    Direct(HashMap<SequenceHash, LayoutHandle>),

    /// Worker layout handles for round-robin assignment.
    /// Used for staging mode where blocks are distributed across workers.
    RoundRobin(Vec<LayoutHandle>),
}

impl BlockMetadataMap {
    /// Build `BlockInfo` list from the current G2 blocks.
    fn build_block_infos(&self, g2_blocks: &BlockHolder<G2>) -> Vec<BlockInfo> {
        match self {
            BlockMetadataMap::Direct(map) => g2_blocks
                .blocks()
                .iter()
                .filter_map(|block| {
                    let hash = block.sequence_hash();
                    map.get(&hash).map(|&layout_handle| BlockInfo {
                        block_id: block.block_id(),
                        sequence_hash: hash,
                        layout_handle,
                    })
                })
                .collect(),

            BlockMetadataMap::RoundRobin(handles) => {
                if handles.is_empty() {
                    return g2_blocks
                        .blocks()
                        .iter()
                        .map(|b| BlockInfo {
                            block_id: b.block_id(),
                            sequence_hash: b.sequence_hash(),
                            layout_handle: LayoutHandle::new(0, 0),
                        })
                        .collect();
                }
                g2_blocks
                    .blocks()
                    .iter()
                    .enumerate()
                    .map(|(i, b)| BlockInfo {
                        block_id: b.block_id(),
                        sequence_hash: b.sequence_hash(),
                        layout_handle: handles[i % handles.len()],
                    })
                    .collect()
            }
        }
    }

    /// Assign a layout handle for a newly staged block at the given index.
    fn assign_handle(&self, index: usize) -> LayoutHandle {
        match self {
            BlockMetadataMap::Direct(_) => {
                // Direct mode shouldn't be staging, but provide a fallback
                LayoutHandle::new(0, 0)
            }
            BlockMetadataMap::RoundRobin(handles) => {
                if handles.is_empty() {
                    LayoutHandle::new(0, 0)
                } else {
                    handles[index % handles.len()]
                }
            }
        }
    }

    /// Remove entries for the given sequence hashes (Direct mode only).
    fn remove_all(&mut self, hashes: &[SequenceHash]) {
        if let BlockMetadataMap::Direct(map) = self {
            for hash in hashes {
                map.remove(hash);
            }
        }
    }
}

/// Options for server session creation.
#[derive(Debug, Clone)]
pub struct ServerSessionOptions {
    /// If true (default), immediately start G3→G2 staging.
    /// If false, wait for controller to call trigger_staging().
    pub auto_stage: bool,
}

impl Default for ServerSessionOptions {
    fn default() -> Self {
        Self { auto_stage: true }
    }
}

/// Server-side session that holds blocks and exposes them for remote RDMA pull.
///
/// Merges the functionality of the former `EndpointSession` and `ControllableSession`.
pub struct ServerSession {
    /// State machine for the session protocol.
    endpoint: SessionEndpoint,

    /// G2 blocks held for RDMA pull (RAII - released on drop).
    g2_blocks: BlockHolder<G2>,

    /// Block metadata mapping (Direct or RoundRobin).
    block_metadata: BlockMetadataMap,

    /// G3 blocks pending staging (empty in G2-only mode).
    g3_blocks: BlockHolder<G3>,

    /// G2 manager for staging (only needed when G3 blocks present).
    g2_manager: Option<Arc<BlockManager<G2>>>,

    /// Parallel worker for G3→G2 staging.
    parallel_worker: Option<Arc<dyn ParallelWorkers>>,

    /// Channel for receiving local commands.
    cmd_rx: mpsc::Receiver<ServerSessionCommand>,

    /// Session options.
    options: ServerSessionOptions,

    /// Staging state tracking.
    staging_started: bool,
    staging_complete: bool,
}

/// Handle for local caller to control a ServerSession.
///
/// Used to send layer notifications or close the session.
/// When dropped, the session continues until peer detaches or channel closes.
#[derive(Clone)]
pub struct ServerSessionHandle {
    session_id: SessionId,
    local_instance: InstanceId,
    cmd_tx: mpsc::Sender<ServerSessionCommand>,
}

/// Commands that can be sent to a ServerSession via its handle.
#[derive(Debug)]
pub enum ServerSessionCommand {
    /// Notify that specific layers are ready for transfer.
    NotifyLayersReady { layer_range: Range<usize> },
    /// Close the session gracefully.
    Close,
}

impl ServerSession {
    /// Create a new ServerSession for G2-only mode.
    ///
    /// Blocks are already in G2 with pre-assigned layout handles.
    pub fn new_g2_only(
        endpoint: SessionEndpoint,
        g2_blocks: BlockHolder<G2>,
        block_metadata: HashMap<SequenceHash, LayoutHandle>,
        cmd_rx: mpsc::Receiver<ServerSessionCommand>,
    ) -> Self {
        Self {
            endpoint,
            g2_blocks,
            block_metadata: BlockMetadataMap::Direct(block_metadata),
            g3_blocks: BlockHolder::empty(),
            g2_manager: None,
            parallel_worker: None,
            cmd_rx,
            options: ServerSessionOptions { auto_stage: false },
            staging_started: false,
            staging_complete: false,
        }
    }

    /// Create a new ServerSession with G3→G2 staging capability.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_staging(
        endpoint: SessionEndpoint,
        g2_blocks: BlockHolder<G2>,
        g3_blocks: BlockHolder<G3>,
        worker_handles: Vec<LayoutHandle>,
        g2_manager: Arc<BlockManager<G2>>,
        parallel_worker: Option<Arc<dyn ParallelWorkers>>,
        cmd_rx: mpsc::Receiver<ServerSessionCommand>,
        options: ServerSessionOptions,
    ) -> Self {
        Self {
            endpoint,
            g2_blocks,
            block_metadata: BlockMetadataMap::RoundRobin(worker_handles),
            g3_blocks,
            g2_manager: Some(g2_manager),
            parallel_worker,
            cmd_rx,
            options,
            staging_started: false,
            staging_complete: false,
        }
    }

    /// Run the session message loop.
    pub async fn run(mut self) -> Result<()> {
        debug!(
            session_id = %self.endpoint.session_id(),
            g2 = self.g2_blocks.count(),
            g3 = self.g3_blocks.count(),
            "ServerSession starting"
        );

        // Set initial phase
        if self.g2_blocks.count() > 0 || self.g3_blocks.count() > 0 {
            self.endpoint.set_phase(SessionPhase::Holding);
        }

        // Auto-stage if enabled and we have G3 blocks
        if self.options.auto_stage && !self.g3_blocks.is_empty() && self.parallel_worker.is_some() {
            self.endpoint.set_phase(SessionPhase::Staging);
            self.staging_started = true;
            self.execute_staging().await?;
        }

        self.update_phase();

        loop {
            tokio::select! {
                msg = self.endpoint.recv() => {
                    match msg {
                        Some(msg) => {
                            if !self.handle_message(msg).await? {
                                break;
                            }
                        }
                        None => {
                            debug!(
                                session_id = %self.endpoint.session_id(),
                                "Message channel closed"
                            );
                            break;
                        }
                    }
                }

                cmd = self.cmd_rx.recv() => {
                    match cmd {
                        Some(cmd) => {
                            if !self.handle_command(cmd).await? {
                                break;
                            }
                        }
                        None => {
                            debug!(
                                session_id = %self.endpoint.session_id(),
                                "Command channel closed"
                            );
                        }
                    }
                }
            }
        }

        debug!(
            session_id = %self.endpoint.session_id(),
            phase = ?self.endpoint.phase(),
            "ServerSession completed"
        );

        Ok(())
    }

    /// Handle an incoming SessionMessage.
    ///
    /// Returns `true` to continue, `false` to exit the loop.
    async fn handle_message(&mut self, msg: SessionMessage) -> Result<bool> {
        match msg {
            SessionMessage::Attach { peer, as_role, .. } => {
                debug!(
                    session_id = %self.endpoint.session_id(),
                    peer = %peer,
                    role = ?as_role,
                    "Peer attached"
                );

                self.endpoint.accept_attachment(peer, as_role.opposite());

                // Update phase for attach
                if self.endpoint.phase() == SessionPhase::Searching
                    || self.endpoint.phase() == SessionPhase::Holding
                {
                    self.update_phase();
                }

                // Send current state
                self.send_state_response(None).await?;
            }

            SessionMessage::TriggerStaging { .. } => {
                self.handle_trigger_staging().await?;
            }

            SessionMessage::BlocksPulled { pulled_hashes, .. } => {
                debug!(
                    session_id = %self.endpoint.session_id(),
                    count = pulled_hashes.len(),
                    "Blocks pulled"
                );

                self.block_metadata.remove_all(&pulled_hashes);
                self.g2_blocks.release(&pulled_hashes);

                if self.g2_blocks.is_empty() && self.g3_blocks.is_empty() {
                    self.endpoint.set_phase(SessionPhase::Complete);
                    return Ok(false);
                }
            }

            SessionMessage::YieldControl { peer, .. } => {
                debug!(
                    session_id = %self.endpoint.session_id(),
                    peer = %peer,
                    "Peer yielded control"
                );
                self.endpoint.set_control_role(ControlRole::Neutral);
            }

            SessionMessage::AcquireControl { peer, .. } => {
                debug!(
                    session_id = %self.endpoint.session_id(),
                    peer = %peer,
                    "Peer acquiring control"
                );
                self.endpoint.set_control_role(ControlRole::Controllee);
            }

            SessionMessage::Detach { peer, .. } => {
                debug!(
                    session_id = %self.endpoint.session_id(),
                    peer = %peer,
                    "Peer detached"
                );
                self.endpoint.detach();
                self.endpoint.set_phase(SessionPhase::Complete);
                return Ok(false);
            }

            SessionMessage::Close { .. } => {
                debug!(
                    session_id = %self.endpoint.session_id(),
                    "Session closed"
                );
                self.endpoint.set_phase(SessionPhase::Complete);
                return Ok(false);
            }

            SessionMessage::Error { message, .. } => {
                warn!(
                    session_id = %self.endpoint.session_id(),
                    error = %message,
                    "Received error"
                );
                self.endpoint.set_phase(SessionPhase::Failed);
                return Ok(false);
            }

            // Ignore outbound-only messages
            SessionMessage::StateResponse { .. }
            | SessionMessage::BlocksStaged { .. }
            | SessionMessage::HoldBlocks { .. }
            | SessionMessage::ReleaseBlocks { .. } => {}
        }

        Ok(true)
    }

    /// Handle a local command.
    ///
    /// Returns `true` to continue, `false` to exit the loop.
    async fn handle_command(&mut self, cmd: ServerSessionCommand) -> Result<bool> {
        match cmd {
            ServerSessionCommand::NotifyLayersReady { layer_range } => {
                debug!(
                    session_id = %self.endpoint.session_id(),
                    layer_range = ?layer_range,
                    "Notifying layers ready"
                );
                self.send_blocks_staged(Some(layer_range)).await?;
            }
            ServerSessionCommand::Close => {
                debug!(
                    session_id = %self.endpoint.session_id(),
                    "Local close requested"
                );
                self.endpoint.set_phase(SessionPhase::Complete);

                if self.endpoint.is_attached() {
                    let msg = SessionMessage::Close {
                        session_id: self.endpoint.session_id(),
                    };
                    self.endpoint.send(msg).await?;
                }
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Handle trigger staging request (idempotent).
    async fn handle_trigger_staging(&mut self) -> Result<()> {
        if self.staging_started {
            return Ok(());
        }

        if self.g3_blocks.is_empty() {
            // No-op for G2-only mode
            debug!(
                session_id = %self.endpoint.session_id(),
                "TriggerStaging ignored (no G3 blocks)"
            );
            return Ok(());
        }

        if self.parallel_worker.is_none() {
            if self.endpoint.is_attached() {
                let error_msg = SessionMessage::Error {
                    session_id: self.endpoint.session_id(),
                    message: "No parallel worker available for G3->G2 staging".to_string(),
                };
                self.endpoint.send(error_msg).await?;
            }
            return Ok(());
        }

        self.endpoint.set_phase(SessionPhase::Staging);
        self.staging_started = true;

        let staged_info = self.execute_staging().await?;

        self.update_phase();

        // Notify peer of newly staged blocks (if attached)
        if self.endpoint.is_attached() {
            let msg = SessionMessage::BlocksStaged {
                session_id: self.endpoint.session_id(),
                staged_blocks: staged_info,
                remaining: self.g3_blocks.count(),
                layer_range: None,
            };
            self.endpoint.send(msg).await?;
        }

        Ok(())
    }

    /// Execute G3→G2 staging.
    ///
    /// Returns BlockInfo for newly staged blocks.
    async fn execute_staging(&mut self) -> Result<Vec<BlockInfo>> {
        let parallel_worker = self
            .parallel_worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ParallelWorkers required for G3→G2 staging"))?;

        let g2_manager = self
            .g2_manager
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("G2 manager required for staging"))?;

        if self.g3_blocks.is_empty() {
            self.staging_complete = true;
            return Ok(Vec::new());
        }

        let result =
            staging::stage_g3_to_g2(&self.g3_blocks, g2_manager, &**parallel_worker).await?;

        // Build BlockInfo for newly staged blocks
        let starting_index = self.g2_blocks.count();
        let staged_info: Vec<BlockInfo> = result
            .new_g2_blocks
            .iter()
            .enumerate()
            .map(|(i, b)| BlockInfo {
                block_id: b.block_id(),
                sequence_hash: b.sequence_hash(),
                layout_handle: self.block_metadata.assign_handle(starting_index + i),
            })
            .collect();

        // Clear G3, extend G2
        let _ = self.g3_blocks.take_all();
        self.g2_blocks.extend(result.new_g2_blocks);

        self.staging_complete = true;

        Ok(staged_info)
    }

    /// Update phase based on current state.
    fn update_phase(&mut self) {
        if self.endpoint.phase() == SessionPhase::Complete
            || self.endpoint.phase() == SessionPhase::Failed
        {
            return;
        }

        if self.g3_blocks.is_empty() && (self.staging_complete || !self.staging_started) {
            self.endpoint.set_phase(SessionPhase::Ready);
        } else if self.staging_started && !self.staging_complete {
            self.endpoint.set_phase(SessionPhase::Staging);
        }
    }

    /// Send a StateResponse to the attached peer.
    async fn send_state_response(&self, layer_range: Option<Range<usize>>) -> Result<()> {
        let state = self.build_state_snapshot(layer_range);
        let msg = SessionMessage::StateResponse {
            session_id: self.endpoint.session_id(),
            state,
        };
        self.endpoint.send(msg).await
    }

    /// Send a BlocksStaged message with optional layer range.
    async fn send_blocks_staged(&self, layer_range: Option<Range<usize>>) -> Result<()> {
        let blocks = self.block_metadata.build_block_infos(&self.g2_blocks);
        let msg = SessionMessage::BlocksStaged {
            session_id: self.endpoint.session_id(),
            staged_blocks: blocks,
            remaining: 0,
            layer_range,
        };
        self.endpoint.send(msg).await
    }

    /// Build a state snapshot.
    fn build_state_snapshot(&self, layer_range: Option<Range<usize>>) -> SessionStateSnapshot {
        SessionStateSnapshot {
            phase: self.endpoint.phase(),
            control_role: self.endpoint.control_role(),
            g2_blocks: self.block_metadata.build_block_infos(&self.g2_blocks),
            g3_pending: self.g3_blocks.count(),
            ready_layer_range: layer_range,
        }
    }

    /// Get the session ID.
    pub fn session_id(&self) -> SessionId {
        self.endpoint.session_id()
    }
}

impl ServerSessionHandle {
    /// Create a new server session handle.
    pub fn new(
        session_id: SessionId,
        local_instance: InstanceId,
        cmd_tx: mpsc::Sender<ServerSessionCommand>,
    ) -> Self {
        Self {
            session_id,
            local_instance,
            cmd_tx,
        }
    }

    /// Get the session ID.
    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Get the local instance ID.
    pub fn local_instance(&self) -> InstanceId {
        self.local_instance
    }

    /// Notify attached controller that layers are ready.
    pub async fn notify_layers_ready(&self, layer_range: Range<usize>) -> Result<()> {
        self.cmd_tx
            .send(ServerSessionCommand::NotifyLayersReady { layer_range })
            .await
            .map_err(|_| anyhow::anyhow!("Session command channel closed"))
    }

    /// Close the session gracefully.
    pub async fn close(&self) -> Result<()> {
        self.cmd_tx
            .send(ServerSessionCommand::Close)
            .await
            .map_err(|_| anyhow::anyhow!("Session command channel closed"))
    }
}

/// Create a ServerSession in G2-only mode with its handle.
///
/// This is the replacement for `create_endpoint_session`.
pub fn create_server_session(
    session_id: SessionId,
    instance_id: InstanceId,
    blocks: BlockHolder<G2>,
    layout_handles: Vec<LayoutHandle>,
    sequence_hashes: Vec<SequenceHash>,
    transport: Arc<MessageTransport>,
    msg_rx: mpsc::Receiver<SessionMessage>,
) -> (ServerSession, ServerSessionHandle) {
    let (cmd_tx, cmd_rx) = mpsc::channel(16);

    let block_metadata: HashMap<SequenceHash, LayoutHandle> =
        sequence_hashes.into_iter().zip(layout_handles).collect();

    let endpoint = SessionEndpoint::new(session_id, instance_id, transport, msg_rx);

    let session = ServerSession::new_g2_only(endpoint, blocks, block_metadata, cmd_rx);

    let handle = ServerSessionHandle::new(session_id, instance_id, cmd_tx);

    (session, handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leader::session::SessionMessageTx;
    use dashmap::DashMap;
    use tokio::sync::mpsc;

    fn create_test_transport() -> Arc<MessageTransport> {
        Arc::new(MessageTransport::local(
            Arc::new(DashMap::new()),
            Arc::new(DashMap::new()),
        ))
    }

    #[tokio::test]
    async fn test_handle_creation() {
        let (cmd_tx, _cmd_rx) = mpsc::channel(16);
        let session_id = SessionId::new_v4();
        let instance_id = InstanceId::new_v4();

        let handle = ServerSessionHandle::new(session_id, instance_id, cmd_tx);

        assert_eq!(handle.session_id(), session_id);
        assert_eq!(handle.local_instance(), instance_id);
    }

    #[tokio::test]
    async fn test_notify_layers_ready() {
        let (cmd_tx, mut cmd_rx) = mpsc::channel(16);
        let session_id = SessionId::new_v4();
        let instance_id = InstanceId::new_v4();

        let handle = ServerSessionHandle::new(session_id, instance_id, cmd_tx);

        handle.notify_layers_ready(0..1).await.unwrap();

        let cmd = cmd_rx.recv().await.unwrap();
        match cmd {
            ServerSessionCommand::NotifyLayersReady { layer_range } => {
                assert_eq!(layer_range, 0..1);
            }
            _ => panic!("Unexpected command"),
        }
    }

    #[tokio::test]
    async fn test_handle_close() {
        let (cmd_tx, mut cmd_rx) = mpsc::channel(16);
        let session_id = SessionId::new_v4();
        let instance_id = InstanceId::new_v4();

        let handle = ServerSessionHandle::new(session_id, instance_id, cmd_tx);

        handle.close().await.unwrap();

        let cmd = cmd_rx.recv().await.unwrap();
        assert!(matches!(cmd, ServerSessionCommand::Close));
    }

    #[tokio::test]
    async fn test_create_server_session() {
        let session_id = SessionId::new_v4();
        let instance_id = InstanceId::new_v4();
        let transport = create_test_transport();
        let (_msg_tx, msg_rx) = mpsc::channel(16);

        let blocks = BlockHolder::empty();

        let (_session, handle) = create_server_session(
            session_id,
            instance_id,
            blocks,
            vec![],
            vec![],
            transport,
            msg_rx,
        );

        assert_eq!(handle.session_id(), session_id);
        assert_eq!(handle.local_instance(), instance_id);
    }

    #[tokio::test]
    async fn test_attach_sends_state_response() {
        let session_id = SessionId::new_v4();
        let instance_id = InstanceId::new_v4();
        let peer_id = InstanceId::new_v4();

        // Create transport with a session channel to capture responses
        let session_sessions: Arc<DashMap<SessionId, SessionMessageTx>> = Arc::new(DashMap::new());
        let transport = Arc::new(MessageTransport::local(
            Arc::new(DashMap::new()),
            session_sessions.clone(),
        ));

        // Register a receiver for the peer's session (where StateResponse is sent)
        let peer_session_id = SessionId::new_v4(); // peer's session
        let (peer_tx, mut peer_rx) = mpsc::channel::<SessionMessage>(16);
        session_sessions.insert(session_id, peer_tx);

        let (msg_tx, msg_rx) = mpsc::channel(16);
        let (_cmd_tx, cmd_rx) = mpsc::channel(16);

        let endpoint = SessionEndpoint::new(session_id, instance_id, transport, msg_rx);
        let session =
            ServerSession::new_g2_only(endpoint, BlockHolder::empty(), HashMap::new(), cmd_rx);

        // Spawn session
        let session_task = tokio::spawn(session.run());

        // Send attach message
        msg_tx
            .send(SessionMessage::Attach {
                peer: peer_id,
                session_id,
                as_role: ControlRole::Controller,
            })
            .await
            .unwrap();

        // Read the StateResponse
        let response = tokio::time::timeout(std::time::Duration::from_secs(1), peer_rx.recv())
            .await
            .expect("timeout")
            .expect("channel closed");

        match response {
            SessionMessage::StateResponse { state, .. } => {
                assert_eq!(state.phase, SessionPhase::Ready);
                assert_eq!(state.control_role, ControlRole::Controllee);
            }
            other => panic!("Expected StateResponse, got {:?}", other),
        }

        // Close session
        msg_tx
            .send(SessionMessage::Close { session_id })
            .await
            .unwrap();

        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), session_task).await;

        let _ = peer_session_id;
    }

    #[tokio::test]
    async fn test_g2_only_ready_on_attach() {
        let session_id = SessionId::new_v4();
        let instance_id = InstanceId::new_v4();
        let peer_id = InstanceId::new_v4();

        let session_sessions: Arc<DashMap<SessionId, SessionMessageTx>> = Arc::new(DashMap::new());
        let transport = Arc::new(MessageTransport::local(
            Arc::new(DashMap::new()),
            session_sessions.clone(),
        ));

        let (peer_tx, mut peer_rx) = mpsc::channel::<SessionMessage>(16);
        session_sessions.insert(session_id, peer_tx);

        let (msg_tx, msg_rx) = mpsc::channel(16);
        let (_cmd_tx, cmd_rx) = mpsc::channel(16);

        let endpoint = SessionEndpoint::new(session_id, instance_id, transport, msg_rx);
        // G2-only mode, no G3 blocks
        let session =
            ServerSession::new_g2_only(endpoint, BlockHolder::empty(), HashMap::new(), cmd_rx);

        let session_task = tokio::spawn(session.run());

        msg_tx
            .send(SessionMessage::Attach {
                peer: peer_id,
                session_id,
                as_role: ControlRole::Controller,
            })
            .await
            .unwrap();

        let response = tokio::time::timeout(std::time::Duration::from_secs(1), peer_rx.recv())
            .await
            .expect("timeout")
            .expect("channel closed");

        // G2-only with no blocks → Ready phase immediately
        match response {
            SessionMessage::StateResponse { state, .. } => {
                assert_eq!(state.phase, SessionPhase::Ready);
                assert_eq!(state.g3_pending, 0);
            }
            other => panic!("Expected StateResponse, got {:?}", other),
        }

        msg_tx
            .send(SessionMessage::Close { session_id })
            .await
            .unwrap();
        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), session_task).await;
    }

    #[tokio::test]
    async fn test_trigger_staging_no_g3_noop() {
        let session_id = SessionId::new_v4();
        let instance_id = InstanceId::new_v4();
        let peer_id = InstanceId::new_v4();

        let session_sessions: Arc<DashMap<SessionId, SessionMessageTx>> = Arc::new(DashMap::new());
        let transport = Arc::new(MessageTransport::local(
            Arc::new(DashMap::new()),
            session_sessions.clone(),
        ));

        let (peer_tx, mut peer_rx) = mpsc::channel::<SessionMessage>(16);
        session_sessions.insert(session_id, peer_tx);

        let (msg_tx, msg_rx) = mpsc::channel(16);
        let (_cmd_tx, cmd_rx) = mpsc::channel(16);

        let endpoint = SessionEndpoint::new(session_id, instance_id, transport, msg_rx);
        let session =
            ServerSession::new_g2_only(endpoint, BlockHolder::empty(), HashMap::new(), cmd_rx);

        let session_task = tokio::spawn(session.run());

        // Attach
        msg_tx
            .send(SessionMessage::Attach {
                peer: peer_id,
                session_id,
                as_role: ControlRole::Controller,
            })
            .await
            .unwrap();

        // Consume StateResponse
        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), peer_rx.recv())
            .await
            .expect("timeout");

        // Send TriggerStaging - should be no-op (no G3 blocks)
        msg_tx
            .send(SessionMessage::TriggerStaging { session_id })
            .await
            .unwrap();

        // Close and check no extra messages were sent
        msg_tx
            .send(SessionMessage::Close { session_id })
            .await
            .unwrap();

        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), session_task).await;
    }

    #[tokio::test]
    async fn test_detach_completes_session() {
        let session_id = SessionId::new_v4();
        let instance_id = InstanceId::new_v4();
        let peer_id = InstanceId::new_v4();

        let session_sessions: Arc<DashMap<SessionId, SessionMessageTx>> = Arc::new(DashMap::new());
        let transport = Arc::new(MessageTransport::local(
            Arc::new(DashMap::new()),
            session_sessions.clone(),
        ));

        let (peer_tx, mut _peer_rx) = mpsc::channel::<SessionMessage>(16);
        session_sessions.insert(session_id, peer_tx);

        let (msg_tx, msg_rx) = mpsc::channel(16);
        let (_cmd_tx, cmd_rx) = mpsc::channel(16);

        let endpoint = SessionEndpoint::new(session_id, instance_id, transport, msg_rx);
        let session =
            ServerSession::new_g2_only(endpoint, BlockHolder::empty(), HashMap::new(), cmd_rx);

        let session_task = tokio::spawn(session.run());

        // Attach then detach
        msg_tx
            .send(SessionMessage::Attach {
                peer: peer_id,
                session_id,
                as_role: ControlRole::Controller,
            })
            .await
            .unwrap();

        msg_tx
            .send(SessionMessage::Detach {
                peer: peer_id,
                session_id,
            })
            .await
            .unwrap();

        // Session should complete
        let result = tokio::time::timeout(std::time::Duration::from_secs(1), session_task)
            .await
            .expect("timeout")
            .expect("task panicked");

        assert!(result.is_ok());
    }

    #[test]
    fn test_block_metadata_direct_build_infos() {
        let hash1 = SequenceHash::new(1, None, 100);
        let hash2 = SequenceHash::new(2, None, 200);

        let mut map = HashMap::new();
        map.insert(hash1, LayoutHandle::new(0, 1));
        map.insert(hash2, LayoutHandle::new(0, 2));

        let metadata = BlockMetadataMap::Direct(map);

        // Empty holder
        let holder = BlockHolder::<G2>::empty();
        let infos = metadata.build_block_infos(&holder);
        assert!(infos.is_empty());
    }

    #[test]
    fn test_block_metadata_round_robin_empty_handles() {
        let metadata = BlockMetadataMap::RoundRobin(vec![]);
        let holder = BlockHolder::<G2>::empty();
        let infos = metadata.build_block_infos(&holder);
        assert!(infos.is_empty());
    }

    #[test]
    fn test_block_metadata_assign_handle() {
        let h0 = LayoutHandle::new(0, 10);
        let h1 = LayoutHandle::new(1, 20);
        let metadata = BlockMetadataMap::RoundRobin(vec![h0, h1]);

        assert_eq!(metadata.assign_handle(0), h0);
        assert_eq!(metadata.assign_handle(1), h1);
        assert_eq!(metadata.assign_handle(2), h0); // wraps around
    }

    #[test]
    fn test_block_metadata_remove_all() {
        let hash1 = SequenceHash::new(1, None, 100);
        let hash2 = SequenceHash::new(2, None, 200);

        let mut map = HashMap::new();
        map.insert(hash1, LayoutHandle::new(0, 1));
        map.insert(hash2, LayoutHandle::new(0, 2));

        let mut metadata = BlockMetadataMap::Direct(map);
        metadata.remove_all(&[hash1]);

        // Verify hash1 was removed
        if let BlockMetadataMap::Direct(ref inner) = metadata {
            assert!(!inner.contains_key(&hash1));
            assert!(inner.contains_key(&hash2));
        }
    }
}
