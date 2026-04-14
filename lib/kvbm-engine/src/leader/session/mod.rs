// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Session Module
//!
//! This module provides session management for distributed block transfers.
//!
//! ## Core Building Blocks
//!
//! Composable building blocks for session management:
//!
//! - `BlockHolder<T>`: RAII container for holding blocks during sessions
//! - `SessionEndpoint`: Point-to-point session primitive with state machine
//! - `SessionHandle`: Unified handle for controlling remote sessions
//! - `SessionMessage`: Unified message protocol with bidirectional control
//! - `SessionPhase`, `ControlRole`, `AttachmentState`: State machine types
//!
//! ## Session Implementations
//!
//! - `ServerSession`: Server-side session (merges former EndpointSession + ControllableSession)
//! - `InitiatorSession`: Multi-peer search orchestrator (OnboardMessage)
//! - `ResponderSession`: Responds to search requests (OnboardMessage)

// Core session building blocks
mod blocks;
mod endpoint;
mod handle;
mod server_session;
mod staging;
mod state;

// Session implementations
mod initiator;
mod messages;
mod responder;
pub mod transport;

// =============================================================================
// Core Building Blocks
// =============================================================================

/// RAII container for holding blocks during sessions.
pub use blocks::BlockHolder;

/// Point-to-point session endpoint with state machine.
pub use endpoint::{SessionEndpoint, SessionMessageTx, session_message_channel};

/// Server-side session (unified replacement for EndpointSession + ControllableSession).
pub use server_session::{
    ServerSession, ServerSessionCommand, ServerSessionHandle, ServerSessionOptions,
    create_server_session,
};

// Backwards-compatible aliases for the former EndpointSession types.
pub use server_session::ServerSessionCommand as EndpointSessionCommand;
pub use server_session::ServerSessionHandle as EndpointSessionHandle;

/// Unified handle for controlling remote sessions.
pub use handle::{SessionHandle, SessionHandleStateTx, session_handle_state_channel};

/// State machine types for the unified session model.
pub use state::{AttachmentState, ControlRole, SessionPhase};

/// Unified session message protocol.
pub use messages::{BlockInfo, SessionMessage, SessionStateSnapshot};

// =============================================================================
// Session Implementations
// =============================================================================

/// Session implementations for initiator and responder patterns.
pub use initiator::InitiatorSession;
pub use responder::ResponderSession;

/// Backwards-compatible re-exports (ControllableSessionResult is still used externally).
pub use server_session::ServerSessionOptions as ControllableSessionOptions;

/// Result of creating a controllable/server session.
#[derive(Debug, Clone)]
pub struct ControllableSessionResult {
    /// The unique session ID.
    pub session_id: super::SessionId,
    /// Number of G2 blocks found.
    pub local_g2_count: usize,
    /// Number of G3 blocks found.
    pub local_g3_count: usize,
}

/// Message types for session communication.
pub use messages::{BlockMatch, OnboardMessage};

/// Transport types.
pub use transport::{LocalTransport, MessageTransport, VeloTransport};

use anyhow::Result;
use dashmap::DashMap;
use tokio::sync::mpsc;

pub type SessionId = uuid::Uuid;
pub type OnboardSessionTx = mpsc::Sender<OnboardMessage>;

/// Route an [`OnboardMessage`] to its per-session task channel.
///
/// Looks up the session ID in the `DashMap` registry and forwards the message
/// through the session's mpsc sender. Each session processes messages serially
/// via its channel, so ordering is preserved per-session.
pub async fn dispatch_onboard_message(
    sessions: &DashMap<SessionId, OnboardSessionTx>,
    message: OnboardMessage,
) -> Result<()> {
    let session_id = message.session_id();

    let sender = sessions.get(&session_id).map(|entry| entry.value().clone());
    if let Some(sender) = sender {
        sender
            .send(message)
            .await
            .map_err(|e| anyhow::anyhow!("failed to send to session {session_id}: {e}"))?;
        return Ok(());
    }

    anyhow::bail!("no session task registered for session {session_id}");
}

/// Route a unified [`SessionMessage`] to its session task.
///
/// All message variants are routed through a single `DashMap<SessionId, SessionMessageTx>`
/// registry.
pub async fn dispatch_session_message(
    sessions: &DashMap<SessionId, SessionMessageTx>,
    message: SessionMessage,
) -> Result<()> {
    let session_id = message.session_id();

    let sender = sessions.get(&session_id).map(|entry| entry.value().clone());
    if let Some(sender) = sender {
        sender
            .send(message)
            .await
            .map_err(|e| anyhow::anyhow!("failed to send to session {session_id}: {e}"))?;
        return Ok(());
    }

    anyhow::bail!("no session registered for session {session_id}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dispatch_onboard_message() {
        let sessions: DashMap<SessionId, OnboardSessionTx> = DashMap::new();
        let session_id = SessionId::new_v4();
        let (tx, mut rx) = mpsc::channel(16);
        sessions.insert(session_id, tx);

        let msg = OnboardMessage::CloseSession {
            requester: crate::InstanceId::new_v4(),
            session_id,
        };

        dispatch_onboard_message(&sessions, msg).await.unwrap();

        let received = rx.recv().await.unwrap();
        assert_eq!(received.session_id(), session_id);
    }

    #[tokio::test]
    async fn test_dispatch_session_message() {
        let sessions: DashMap<SessionId, SessionMessageTx> = DashMap::new();
        let session_id = SessionId::new_v4();
        let (tx, mut rx) = mpsc::channel(16);
        sessions.insert(session_id, tx);

        let msg = SessionMessage::Close { session_id };

        dispatch_session_message(&sessions, msg).await.unwrap();

        let received = rx.recv().await.unwrap();
        assert_eq!(received.session_id(), session_id);
    }

    #[tokio::test]
    async fn test_dispatch_missing_onboard_session() {
        let sessions: DashMap<SessionId, OnboardSessionTx> = DashMap::new();
        let session_id = SessionId::new_v4();

        let msg = OnboardMessage::CloseSession {
            requester: crate::InstanceId::new_v4(),
            session_id,
        };

        let result = dispatch_onboard_message(&sessions, msg).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_dispatch_missing_session_message() {
        let sessions: DashMap<SessionId, SessionMessageTx> = DashMap::new();
        let session_id = SessionId::new_v4();

        let msg = SessionMessage::Close { session_id };

        let result = dispatch_session_message(&sessions, msg).await;
        assert!(result.is_err());
    }
}
