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
// Migration notes:
// - OnboardMessage corresponds to SessionMessage
// - RemoteSessionMessage corresponds to SessionMessage
// - RemoteSessionHandle corresponds to SessionHandle
// - RemoteSessionPhase corresponds to SessionPhase
//
// InitiatorSession, ResponderSession, and ControllableSession use
// the older message types but internally use BlockHolder for RAII block management.

// Core session building blocks
mod blocks;
mod endpoint;
mod endpoint_session;
mod handle;
mod state;

// Session implementations
mod controllable;
mod initiator;
mod messages;
// mod onboard; // Old implementation, will be replaced incrementally
mod remote_handle;
mod responder;
pub mod transport;

// =============================================================================
// Core Building Blocks
// =============================================================================

/// RAII container for holding blocks during sessions.
pub use blocks::BlockHolder;

/// Point-to-point session endpoint with state machine.
pub use endpoint::{SessionEndpoint, SessionMessageTx, session_message_channel};

/// Server-side session that processes incoming SessionMessage.
pub use endpoint_session::{
    EndpointSession, EndpointSessionCommand, EndpointSessionHandle, create_endpoint_session,
};

/// Unified handle for controlling remote sessions.
pub use handle::{SessionHandle, SessionHandleStateTx, session_handle_state_channel};

/// State machine types for the unified session model.
pub use state::{AttachmentState, ControlRole, SessionPhase};

/// Unified session message protocol.
pub use messages::{BlockInfo, SessionMessage, SessionStateSnapshot};

// =============================================================================
// Session Implementations
// =============================================================================

/// Session implementations for initiator, responder, and controllable patterns.
pub use controllable::{ControllableSession, ControllableSessionResult};
pub use initiator::InitiatorSession;
pub use responder::ResponderSession;

/// Message types for session communication.
pub use messages::{
    BlockMatch, ControllableSessionOptions, G2BlockInfo, G3BlockInfo, OnboardMessage,
    RemoteSessionMessage, RemoteSessionPhase,
};

/// Remote session handle and state channel.
pub use remote_handle::{
    RemoteSessionHandle, RemoteSessionState, RemoteSessionStateTx, remote_session_state_channel,
};

/// Transport types.
pub use transport::{LocalTransport, MessageTransport, RemoteSessionTx, VeloTransport};

// Re-export from onboard for backward compatibility (will be removed later)
// pub use onboard::OnboardingSession;

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

    if let Some(entry) = sessions.get(&session_id) {
        entry
            .value()
            .send(message)
            .await
            .map_err(|e| anyhow::anyhow!("failed to send to session {session_id}: {e}"))?;
        return Ok(());
    }

    anyhow::bail!("no session task registered for session {session_id}");
}

/// Route a [`RemoteSessionMessage`] to either the Decode-side or Prefill-side session task.
///
/// Message routing by variant:
/// - `AttachSession`, `TriggerStaging`, `BlocksPulled`, `DetachSession` ->
///   `controllable_sessions` (commands sent **to** Decode)
/// - `SessionState`, `BlocksStaged`, `SessionError` ->
///   `remote_sessions` (responses sent **to** Prefill)
pub async fn dispatch_remote_session_message(
    controllable_sessions: &DashMap<SessionId, RemoteSessionTx>,
    remote_sessions: &DashMap<SessionId, RemoteSessionTx>,
    message: RemoteSessionMessage,
) -> Result<()> {
    let session_id = message.session_id();

    // Route based on message type:
    // - AttachSession, TriggerStaging, BlocksPulled, DetachSession → controllable_sessions (Decode)
    // - SessionState, BlocksStaged, SessionError → remote_sessions (Prefill)
    let sessions = match &message {
        RemoteSessionMessage::AttachSession { .. }
        | RemoteSessionMessage::TriggerStaging { .. }
        | RemoteSessionMessage::BlocksPulled { .. }
        | RemoteSessionMessage::DetachSession { .. } => controllable_sessions,
        RemoteSessionMessage::SessionState { .. }
        | RemoteSessionMessage::BlocksStaged { .. }
        | RemoteSessionMessage::SessionError { .. } => remote_sessions,
    };

    if let Some(entry) = sessions.get(&session_id) {
        entry
            .value()
            .send(message)
            .await
            .map_err(|e| anyhow::anyhow!("failed to send to remote session {session_id}: {e}"))?;
        return Ok(());
    }

    anyhow::bail!("no remote session registered for session {session_id}");
}

/// Route a unified [`SessionMessage`] to its session task.
///
/// This is the new unified protocol that replaces both [`dispatch_onboard_message`]
/// and [`dispatch_remote_session_message`]. All message variants are routed through
/// a single `DashMap<SessionId, SessionMessageTx>` registry since the unified
/// protocol no longer splits by direction.
pub async fn dispatch_session_message(
    sessions: &DashMap<SessionId, SessionMessageTx>,
    message: SessionMessage,
) -> Result<()> {
    let session_id = message.session_id();

    if let Some(entry) = sessions.get(&session_id) {
        entry
            .value()
            .send(message)
            .await
            .map_err(|e| anyhow::anyhow!("failed to send to session {session_id}: {e}"))?;
        return Ok(());
    }

    anyhow::bail!("no session registered for session {session_id}");
}
