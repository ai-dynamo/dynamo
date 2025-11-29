// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EndpointSession: Server-side session using the unified SessionMessage protocol.
//!
//! This is a lightweight session runner that exposes blocks for remote RDMA pull.
//! It uses `SessionEndpoint` for the state machine and processes incoming `SessionMessage`.
//!
//! Key features:
//! - Holds blocks in G2 via RAII (`BlockHolder`)
//! - Processes Attach/BlocksPulled/Detach messages
//! - Supports layerwise transfer notifications
//! - Can be owned by various handlers (Nova active message, local API, etc.)
//!
//! # Usage
//!
//! ```ignore
//! // Create an endpoint session for specific blocks
//! let (session_id, handle) = leader.create_endpoint_session_for_blocks(blocks, hashes)?;
//!
//! // Remote peer attaches and pulls blocks
//! // ...
//!
//! // For layerwise transfer, notify when layers are ready
//! handle.notify_layers_ready(0..1).await?;
//! handle.notify_layers_ready(0..2).await?;
//! // ...
//! ```

use std::ops::Range;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use crate::v2::physical::manager::LayoutHandle;

use super::SessionId;
use super::blocks::BlockHolder;
use super::endpoint::SessionEndpoint;
use super::messages::{BlockInfo, SessionMessage, SessionStateSnapshot};
use super::state::{ControlRole, SessionPhase};
use super::transport::MessageTransport;
use crate::v2::{G2, InstanceId, SequenceHash};

/// Server-side session that processes incoming SessionMessage.
///
/// This session:
/// - Holds blocks in G2 for remote RDMA pull
/// - Processes Attach/BlocksPulled/Detach messages
/// - Supports layerwise transfer notifications
///
/// The session runs in a background task and can be controlled via
/// [`EndpointSessionHandle`].
pub struct EndpointSession {
    /// Session endpoint for state machine and messaging.
    endpoint: SessionEndpoint,

    /// Blocks held in G2 (RAII - released on drop).
    g2_blocks: BlockHolder<G2>,

    /// Layout handles for the blocks (one per block).
    layout_handles: Vec<LayoutHandle>,

    /// Sequence hashes for the blocks (one per block).
    sequence_hashes: Vec<SequenceHash>,

    /// Channel for receiving local commands.
    cmd_rx: mpsc::Receiver<EndpointSessionCommand>,
}

/// Handle for local caller to control an EndpointSession.
///
/// Used to send layer notifications without blocking on the message loop.
/// When dropped, the session will be notified to close gracefully.
#[derive(Clone)]
pub struct EndpointSessionHandle {
    session_id: SessionId,
    local_instance: InstanceId,
    cmd_tx: mpsc::Sender<EndpointSessionCommand>,
}

/// Commands that can be sent to an EndpointSession.
#[derive(Debug)]
pub enum EndpointSessionCommand {
    /// Notify that specific layers are ready for transfer.
    NotifyLayersReady { layer_range: Range<usize> },
    /// Close the session gracefully.
    Close,
}

impl EndpointSession {
    /// Create a new endpoint session.
    ///
    /// # Arguments
    /// * `session_id` - Unique session identifier
    /// * `instance_id` - This instance's ID
    /// * `blocks` - G2 blocks to expose for RDMA pull
    /// * `layout_handles` - Layout handles for each block
    /// * `sequence_hashes` - Sequence hashes for each block
    /// * `transport` - Message transport for sending messages
    /// * `msg_rx` - Channel for receiving SessionMessage from remote
    /// * `cmd_rx` - Channel for receiving local commands
    pub fn new(
        session_id: SessionId,
        instance_id: InstanceId,
        blocks: BlockHolder<G2>,
        layout_handles: Vec<LayoutHandle>,
        sequence_hashes: Vec<SequenceHash>,
        transport: Arc<MessageTransport>,
        msg_rx: mpsc::Receiver<SessionMessage>,
        cmd_rx: mpsc::Receiver<EndpointSessionCommand>,
    ) -> Self {
        // Create endpoint in Controllee state (waiting for controller to attach)
        let endpoint = SessionEndpoint::new(session_id, instance_id, transport, msg_rx);

        Self {
            endpoint,
            g2_blocks: blocks,
            layout_handles,
            sequence_hashes,
            cmd_rx,
        }
    }

    /// Run the session message loop.
    ///
    /// This processes incoming messages and local commands until the session
    /// completes or fails.
    pub async fn run(mut self) -> Result<()> {
        debug!(
            session_id = %self.endpoint.session_id(),
            "EndpointSession starting with {} blocks",
            self.g2_blocks.count()
        );

        // Set initial phase based on whether we have blocks
        if self.g2_blocks.count() > 0 {
            self.endpoint.set_phase(SessionPhase::Holding);
        }

        loop {
            tokio::select! {
                // Handle incoming SessionMessage
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

                // Handle local commands
                cmd = self.cmd_rx.recv() => {
                    match cmd {
                        Some(cmd) => {
                            if !self.handle_command(cmd).await? {
                                break;
                            }
                        }
                        None => {
                            // Command channel closed, continue processing messages
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
            "EndpointSession completed"
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

                // Accept attachment as the opposite role
                self.endpoint.accept_attachment(peer, as_role.opposite());
                self.endpoint.set_phase(SessionPhase::Ready);

                // Send initial state
                self.send_state_response().await?;
            }

            SessionMessage::TriggerStaging { .. } => {
                // No-op for EndpointSession - blocks are already in G2
                debug!(
                    session_id = %self.endpoint.session_id(),
                    "TriggerStaging ignored (blocks already staged)"
                );
            }

            SessionMessage::BlocksPulled { pulled_hashes, .. } => {
                debug!(
                    session_id = %self.endpoint.session_id(),
                    count = pulled_hashes.len(),
                    "Blocks pulled"
                );

                // Release the pulled blocks from our holder
                self.g2_blocks.release(&pulled_hashes);

                // If all blocks pulled, session is complete
                if self.g2_blocks.is_empty() {
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

            // Ignore state responses (we're the server)
            SessionMessage::StateResponse { .. } | SessionMessage::BlocksStaged { .. } => {}

            // Ignore hold/release (not supported by EndpointSession)
            SessionMessage::HoldBlocks { .. } | SessionMessage::ReleaseBlocks { .. } => {}
        }

        Ok(true)
    }

    /// Handle a local command.
    ///
    /// Returns `true` to continue, `false` to exit the loop.
    async fn handle_command(&mut self, cmd: EndpointSessionCommand) -> Result<bool> {
        match cmd {
            EndpointSessionCommand::NotifyLayersReady { layer_range } => {
                debug!(
                    session_id = %self.endpoint.session_id(),
                    layer_range = ?layer_range,
                    "Notifying layers ready"
                );
                self.send_blocks_staged_with_layer_range(Some(layer_range))
                    .await?;
            }
            EndpointSessionCommand::Close => {
                debug!(
                    session_id = %self.endpoint.session_id(),
                    "Local close requested"
                );
                self.endpoint.set_phase(SessionPhase::Complete);

                // Notify peer if attached
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

    /// Send a StateResponse to the attached peer.
    async fn send_state_response(&self) -> Result<()> {
        let state = self.build_state_snapshot(None);
        let msg = SessionMessage::StateResponse {
            session_id: self.endpoint.session_id(),
            state,
        };
        self.endpoint.send(msg).await
    }

    /// Send a BlocksStaged message with optional layer range.
    async fn send_blocks_staged_with_layer_range(
        &self,
        layer_range: Option<Range<usize>>,
    ) -> Result<()> {
        let blocks = self.build_block_infos();
        let msg = SessionMessage::BlocksStaged {
            session_id: self.endpoint.session_id(),
            staged_blocks: blocks,
            remaining: 0, // All blocks are staged in EndpointSession
            layer_range,
        };
        self.endpoint.send(msg).await
    }

    /// Build block info list from current state.
    fn build_block_infos(&self) -> Vec<BlockInfo> {
        self.g2_blocks
            .blocks()
            .iter()
            .enumerate()
            .filter_map(|(i, block)| {
                // Only include blocks that have valid layout handles
                self.layout_handles.get(i).map(|&layout_handle| BlockInfo {
                    block_id: block.block_id(),
                    sequence_hash: self
                        .sequence_hashes
                        .get(i)
                        .copied()
                        .unwrap_or_else(|| block.sequence_hash()),
                    layout_handle,
                })
            })
            .collect()
    }

    /// Build a state snapshot.
    fn build_state_snapshot(&self, layer_range: Option<Range<usize>>) -> SessionStateSnapshot {
        SessionStateSnapshot {
            phase: self.endpoint.phase(),
            control_role: self.endpoint.control_role(),
            g2_blocks: self.build_block_infos(),
            g3_pending: 0,
            ready_layer_range: layer_range,
        }
    }
}

impl EndpointSessionHandle {
    /// Create a new endpoint session handle.
    pub fn new(
        session_id: SessionId,
        local_instance: InstanceId,
        cmd_tx: mpsc::Sender<EndpointSessionCommand>,
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
    ///
    /// This sends a `BlocksStaged` message with the specified layer range.
    /// The controller can then pull those layers via RDMA.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Layer 0 is ready
    /// handle.notify_layers_ready(0..1).await?;
    ///
    /// // Layers 0-1 are ready
    /// handle.notify_layers_ready(0..2).await?;
    ///
    /// // All layers ready
    /// handle.notify_layers_ready(0..60).await?;
    /// ```
    pub async fn notify_layers_ready(&self, layer_range: Range<usize>) -> Result<()> {
        self.cmd_tx
            .send(EndpointSessionCommand::NotifyLayersReady { layer_range })
            .await
            .map_err(|_| anyhow::anyhow!("Session command channel closed"))
    }

    /// Close the session gracefully.
    ///
    /// This sends a close command to the session task.
    pub async fn close(&self) -> Result<()> {
        self.cmd_tx
            .send(EndpointSessionCommand::Close)
            .await
            .map_err(|_| anyhow::anyhow!("Session command channel closed"))
    }
}

/// Create an EndpointSession with its handle.
///
/// Returns the session (to be spawned) and a handle for controlling it.
pub fn create_endpoint_session(
    session_id: SessionId,
    instance_id: InstanceId,
    blocks: BlockHolder<G2>,
    layout_handles: Vec<LayoutHandle>,
    sequence_hashes: Vec<SequenceHash>,
    transport: Arc<MessageTransport>,
    msg_rx: mpsc::Receiver<SessionMessage>,
) -> (EndpointSession, EndpointSessionHandle) {
    let (cmd_tx, cmd_rx) = mpsc::channel(16);

    let session = EndpointSession::new(
        session_id,
        instance_id,
        blocks,
        layout_handles,
        sequence_hashes,
        transport,
        msg_rx,
        cmd_rx,
    );

    let handle = EndpointSessionHandle::new(session_id, instance_id, cmd_tx);

    (session, handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashmap::DashMap;

    fn create_test_transport() -> Arc<MessageTransport> {
        Arc::new(MessageTransport::local(
            Arc::new(DashMap::new()),
            Arc::new(DashMap::new()),
            Arc::new(DashMap::new()),
        ))
    }

    #[tokio::test]
    async fn test_endpoint_session_handle_creation() {
        let (cmd_tx, _cmd_rx) = mpsc::channel(16);
        let session_id = SessionId::new_v4();
        let instance_id = InstanceId::new_v4();

        let handle = EndpointSessionHandle::new(session_id, instance_id, cmd_tx);

        assert_eq!(handle.session_id(), session_id);
        assert_eq!(handle.local_instance(), instance_id);
    }

    #[tokio::test]
    async fn test_notify_layers_ready() {
        let (cmd_tx, mut cmd_rx) = mpsc::channel(16);
        let session_id = SessionId::new_v4();
        let instance_id = InstanceId::new_v4();

        let handle = EndpointSessionHandle::new(session_id, instance_id, cmd_tx);

        // Send notification
        handle.notify_layers_ready(0..1).await.unwrap();

        // Verify command received
        let cmd = cmd_rx.recv().await.unwrap();
        match cmd {
            EndpointSessionCommand::NotifyLayersReady { layer_range } => {
                assert_eq!(layer_range, 0..1);
            }
            _ => panic!("Unexpected command"),
        }
    }

    #[tokio::test]
    async fn test_create_endpoint_session() {
        let session_id = SessionId::new_v4();
        let instance_id = InstanceId::new_v4();
        let transport = create_test_transport();
        let (msg_tx, msg_rx) = mpsc::channel(16);

        let blocks = BlockHolder::empty();
        let layout_handles = vec![];
        let sequence_hashes = vec![];

        let (_session, handle) = create_endpoint_session(
            session_id,
            instance_id,
            blocks,
            layout_handles,
            sequence_hashes,
            transport,
            msg_rx,
        );

        assert_eq!(handle.session_id(), session_id);
        assert_eq!(handle.local_instance(), instance_id);
    }
}
