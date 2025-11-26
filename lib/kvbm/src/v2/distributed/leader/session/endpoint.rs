// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SessionEndpoint: Point-to-point session primitive.
//!
//! This is the core building block for unified sessions. It handles:
//! - State machine (control role + attachment state + phase)
//! - Message receive channel for incoming [`SessionMessage`]
//! - State publication via watch channel for observers
//! - Transport for sending messages to peer
//!
//! It does NOT handle:
//! - Block holding (use [`BlockHolder`] for that)
//! - Staging logic (caller invokes staging)
//! - Multi-peer orchestration (that's [`InitiatorSession`]'s job)
//!
//! # Usage
//!
//! ```ignore
//! // Create an endpoint
//! let (tx, rx) = mpsc::channel(32);
//! let endpoint = SessionEndpoint::new(
//!     SessionId::new_v4(),
//!     my_instance_id,
//!     transport,
//!     rx,
//! );
//!
//! // Process messages
//! while let Some(msg) = endpoint.recv().await {
//!     match msg {
//!         SessionMessage::Attach { peer, as_role, .. } => {
//!             endpoint.accept_attachment(peer, as_role);
//!             // ... handle attachment
//!         }
//!         // ... other messages
//!     }
//! }
//! ```

use std::sync::Arc;
use tokio::sync::{mpsc, watch};

use anyhow::Result;

use crate::v2::InstanceId;

use super::{
    SessionId,
    messages::{BlockInfo, SessionMessage, SessionStateSnapshot},
    state::{AttachmentState, ControlRole, SessionPhase},
    transport::MessageTransport,
};

/// A point-to-point session endpoint.
///
/// This is the common building block for all session types. It encapsulates:
/// - Identity (session_id, instance_id)
/// - State machine (control role, attachment, phase)
/// - Communication (message receive, state publication)
///
/// The endpoint starts in `Neutral + Unattached` state by default.
pub struct SessionEndpoint {
    session_id: SessionId,
    instance_id: InstanceId,

    // State
    control_role: ControlRole,
    attachment: AttachmentState,
    phase: SessionPhase,

    // Communication
    transport: Arc<MessageTransport>,
    msg_rx: mpsc::Receiver<SessionMessage>,
    state_tx: watch::Sender<SessionStateSnapshot>,
}

impl SessionEndpoint {
    /// Create a new endpoint in `Neutral + Unattached` state.
    pub fn new(
        session_id: SessionId,
        instance_id: InstanceId,
        transport: Arc<MessageTransport>,
        msg_rx: mpsc::Receiver<SessionMessage>,
    ) -> Self {
        let initial_state = SessionStateSnapshot {
            phase: SessionPhase::default(),
            control_role: ControlRole::default(),
            g2_blocks: Vec::new(),
            g3_pending: 0,
        };
        let (state_tx, _) = watch::channel(initial_state);

        Self {
            session_id,
            instance_id,
            control_role: ControlRole::default(),
            attachment: AttachmentState::default(),
            phase: SessionPhase::default(),
            transport,
            msg_rx,
            state_tx,
        }
    }

    /// Create a new endpoint with pre-attached state.
    ///
    /// Used when creating a session that is already attached to a peer
    /// (e.g., ResponderSession which is pre-attached to the initiator).
    pub fn new_attached(
        session_id: SessionId,
        instance_id: InstanceId,
        peer: InstanceId,
        role: ControlRole,
        phase: SessionPhase,
        transport: Arc<MessageTransport>,
        msg_rx: mpsc::Receiver<SessionMessage>,
    ) -> Self {
        let initial_state = SessionStateSnapshot {
            phase,
            control_role: role,
            g2_blocks: Vec::new(),
            g3_pending: 0,
        };
        let (state_tx, _) = watch::channel(initial_state);

        Self {
            session_id,
            instance_id,
            control_role: role,
            attachment: AttachmentState::Attached { peer },
            phase,
            transport,
            msg_rx,
            state_tx,
        }
    }

    // =========================================================================
    // State Accessors
    // =========================================================================

    /// Get the session ID.
    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Get this endpoint's instance ID.
    pub fn instance_id(&self) -> InstanceId {
        self.instance_id
    }

    /// Get the current control role.
    pub fn control_role(&self) -> ControlRole {
        self.control_role
    }

    /// Check if a peer is attached.
    pub fn is_attached(&self) -> bool {
        self.attachment.is_attached()
    }

    /// Get the attached peer's instance ID.
    pub fn peer(&self) -> Option<InstanceId> {
        self.attachment.peer()
    }

    /// Get the current session phase.
    pub fn phase(&self) -> SessionPhase {
        self.phase
    }

    /// Check if the session is in a terminal state.
    pub fn is_complete(&self) -> bool {
        self.phase.is_terminal()
    }

    // =========================================================================
    // State Transitions
    // =========================================================================

    /// Set the session phase.
    pub fn set_phase(&mut self, phase: SessionPhase) {
        self.phase = phase;
    }

    /// Accept an attachment from a peer.
    ///
    /// Transitions from `Unattached` to `Attached` and sets the control role.
    pub fn accept_attachment(&mut self, peer: InstanceId, role: ControlRole) {
        self.attachment = AttachmentState::Attached { peer };
        self.control_role = role;
    }

    /// Detach from the current peer.
    ///
    /// Returns the detached peer's instance ID if there was one.
    pub fn detach(&mut self) -> Option<InstanceId> {
        let peer = self.attachment.peer();
        self.attachment = AttachmentState::Unattached;
        self.control_role = ControlRole::Neutral;
        peer
    }

    /// Yield control to peer.
    ///
    /// Transitions from `Controller` to `Neutral`.
    /// Returns `Err` if not currently `Controller`.
    pub fn yield_control(&mut self) -> Result<()> {
        if self.control_role != ControlRole::Controller {
            anyhow::bail!("Cannot yield control: not currently Controller");
        }
        self.control_role = ControlRole::Neutral;
        Ok(())
    }

    /// Acquire control from peer.
    ///
    /// Transitions from `Neutral` or `Controllee` to `Controller`.
    /// The peer must be in `Neutral` state for this to succeed.
    pub fn acquire_control(&mut self) -> Result<()> {
        if self.control_role == ControlRole::Controller {
            // Already controller, no-op
            return Ok(());
        }
        self.control_role = ControlRole::Controller;
        Ok(())
    }

    /// Handle a peer yielding control to us.
    ///
    /// Transitions to `Neutral` (peer has yielded, we can now acquire if we want).
    pub fn peer_yielded_control(&mut self) {
        // When peer yields, we stay in our current role or become neutral
        // The peer is now Neutral, so we can acquire if desired
        if self.control_role == ControlRole::Controllee {
            self.control_role = ControlRole::Neutral;
        }
    }

    /// Handle a peer acquiring control.
    ///
    /// Transitions from `Neutral` to `Controllee`.
    pub fn peer_acquired_control(&mut self) -> Result<()> {
        if self.control_role == ControlRole::Controller {
            anyhow::bail!("Cannot transition to Controllee: currently Controller");
        }
        self.control_role = ControlRole::Controllee;
        Ok(())
    }

    // =========================================================================
    // Message Handling
    // =========================================================================

    /// Receive the next message.
    ///
    /// Returns `None` when the channel is closed.
    pub async fn recv(&mut self) -> Option<SessionMessage> {
        self.msg_rx.recv().await
    }

    /// Try to receive a message without blocking.
    pub fn try_recv(&mut self) -> Result<SessionMessage, mpsc::error::TryRecvError> {
        self.msg_rx.try_recv()
    }

    // =========================================================================
    // Outbound Messages
    // =========================================================================

    /// Send an attach message to a peer.
    pub async fn send_attach(&self, peer: InstanceId, as_role: ControlRole) -> Result<()> {
        let msg = SessionMessage::Attach {
            peer: self.instance_id,
            session_id: self.session_id,
            as_role,
        };
        self.send_to(peer, msg).await
    }

    /// Send a detach message to the current peer.
    pub async fn send_detach(&self) -> Result<()> {
        let peer = self
            .peer()
            .ok_or_else(|| anyhow::anyhow!("Cannot detach: not attached"))?;

        let msg = SessionMessage::Detach {
            peer: self.instance_id,
            session_id: self.session_id,
        };
        self.send_to(peer, msg).await
    }

    /// Send yield control message to peer.
    pub async fn send_yield_control(&self) -> Result<()> {
        let peer = self
            .peer()
            .ok_or_else(|| anyhow::anyhow!("Cannot yield: not attached"))?;

        let msg = SessionMessage::YieldControl {
            peer: self.instance_id,
            session_id: self.session_id,
        };
        self.send_to(peer, msg).await
    }

    /// Send acquire control message to peer.
    pub async fn send_acquire_control(&self) -> Result<()> {
        let peer = self
            .peer()
            .ok_or_else(|| anyhow::anyhow!("Cannot acquire: not attached"))?;

        let msg = SessionMessage::AcquireControl {
            peer: self.instance_id,
            session_id: self.session_id,
        };
        self.send_to(peer, msg).await
    }

    /// Send a message to a specific peer.
    pub async fn send_to(&self, peer: InstanceId, msg: SessionMessage) -> Result<()> {
        self.transport.send_session(peer, msg).await
    }

    /// Send a message to the currently attached peer.
    pub async fn send(&self, msg: SessionMessage) -> Result<()> {
        let peer = self
            .peer()
            .ok_or_else(|| anyhow::anyhow!("Cannot send: not attached"))?;
        self.send_to(peer, msg).await
    }

    // =========================================================================
    // State Publication
    // =========================================================================

    /// Publish the current state snapshot.
    ///
    /// This updates all watchers with the new state.
    pub fn publish_state(&self, g2_blocks: Vec<BlockInfo>, g3_pending: usize) {
        let _ = self.state_tx.send(SessionStateSnapshot {
            phase: self.phase,
            control_role: self.control_role,
            g2_blocks,
            g3_pending,
        });
    }

    /// Get a receiver for state updates.
    ///
    /// Returns a watch receiver that will receive state snapshots
    /// whenever they are published.
    pub fn state_rx(&self) -> watch::Receiver<SessionStateSnapshot> {
        self.state_tx.subscribe()
    }

    /// Get the transport for direct access (for legacy interop).
    pub fn transport(&self) -> &Arc<MessageTransport> {
        &self.transport
    }
}

/// Channel type for sending SessionMessages to an endpoint.
pub type SessionMessageTx = mpsc::Sender<SessionMessage>;

/// Create a new session message channel.
pub fn session_message_channel(
    buffer: usize,
) -> (SessionMessageTx, mpsc::Receiver<SessionMessage>) {
    mpsc::channel(buffer)
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

    #[test]
    fn test_endpoint_initial_state() {
        let (_, rx) = mpsc::channel(32);
        let transport = create_test_transport();
        let session_id = SessionId::new_v4();
        let instance_id = InstanceId::new_v4();

        let endpoint = SessionEndpoint::new(session_id, instance_id, transport, rx);

        assert_eq!(endpoint.session_id(), session_id);
        assert_eq!(endpoint.instance_id(), instance_id);
        assert_eq!(endpoint.control_role(), ControlRole::Neutral);
        assert!(!endpoint.is_attached());
        assert!(endpoint.peer().is_none());
        assert_eq!(endpoint.phase(), SessionPhase::Searching);
        assert!(!endpoint.is_complete());
    }

    #[test]
    fn test_endpoint_attachment() {
        let (_, rx) = mpsc::channel(32);
        let transport = create_test_transport();
        let session_id = SessionId::new_v4();
        let instance_id = InstanceId::new_v4();
        let peer_id = InstanceId::new_v4();

        let mut endpoint = SessionEndpoint::new(session_id, instance_id, transport, rx);

        // Accept attachment
        endpoint.accept_attachment(peer_id, ControlRole::Controllee);

        assert!(endpoint.is_attached());
        assert_eq!(endpoint.peer(), Some(peer_id));
        assert_eq!(endpoint.control_role(), ControlRole::Controllee);

        // Detach
        let detached = endpoint.detach();
        assert_eq!(detached, Some(peer_id));
        assert!(!endpoint.is_attached());
        assert_eq!(endpoint.control_role(), ControlRole::Neutral);
    }

    #[test]
    fn test_endpoint_pre_attached() {
        let (_, rx) = mpsc::channel(32);
        let transport = create_test_transport();
        let session_id = SessionId::new_v4();
        let instance_id = InstanceId::new_v4();
        let peer_id = InstanceId::new_v4();

        let endpoint = SessionEndpoint::new_attached(
            session_id,
            instance_id,
            peer_id,
            ControlRole::Controller,
            SessionPhase::Holding,
            transport,
            rx,
        );

        assert!(endpoint.is_attached());
        assert_eq!(endpoint.peer(), Some(peer_id));
        assert_eq!(endpoint.control_role(), ControlRole::Controller);
        assert_eq!(endpoint.phase(), SessionPhase::Holding);
    }

    #[test]
    fn test_control_transitions() {
        let (_, rx) = mpsc::channel(32);
        let transport = create_test_transport();
        let session_id = SessionId::new_v4();
        let instance_id = InstanceId::new_v4();
        let peer_id = InstanceId::new_v4();

        let mut endpoint = SessionEndpoint::new(session_id, instance_id, transport, rx);
        endpoint.accept_attachment(peer_id, ControlRole::Controller);

        // Yield control
        assert!(endpoint.yield_control().is_ok());
        assert_eq!(endpoint.control_role(), ControlRole::Neutral);

        // Can't yield again
        assert!(endpoint.yield_control().is_err());

        // Acquire control
        assert!(endpoint.acquire_control().is_ok());
        assert_eq!(endpoint.control_role(), ControlRole::Controller);
    }

    #[test]
    fn test_phase_transitions() {
        let (_, rx) = mpsc::channel(32);
        let transport = create_test_transport();
        let session_id = SessionId::new_v4();
        let instance_id = InstanceId::new_v4();

        let mut endpoint = SessionEndpoint::new(session_id, instance_id, transport, rx);

        assert_eq!(endpoint.phase(), SessionPhase::Searching);
        assert!(!endpoint.is_complete());

        endpoint.set_phase(SessionPhase::Holding);
        assert_eq!(endpoint.phase(), SessionPhase::Holding);

        endpoint.set_phase(SessionPhase::Complete);
        assert!(endpoint.is_complete());
    }

    #[tokio::test]
    async fn test_state_publication() {
        let (_, rx) = mpsc::channel(32);
        let transport = create_test_transport();
        let session_id = SessionId::new_v4();
        let instance_id = InstanceId::new_v4();

        let endpoint = SessionEndpoint::new(session_id, instance_id, transport, rx);
        let mut state_rx = endpoint.state_rx();

        // Initial state
        let state = state_rx.borrow().clone();
        assert_eq!(state.phase, SessionPhase::Searching);
        assert_eq!(state.g2_blocks.len(), 0);

        // Publish new state
        endpoint.publish_state(vec![], 5);

        // Wait for change
        state_rx.changed().await.unwrap();
        let state = state_rx.borrow().clone();
        assert_eq!(state.g3_pending, 5);
    }
}
