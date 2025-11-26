// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Session Module
//!
//! This module provides session management for distributed block transfers.
//!
//! ## New Unified Model (Recommended)
//!
//! The new session model uses composable building blocks:
//!
//! - [`BlockHolder<T>`]: RAII container for holding blocks during sessions
//! - [`SessionEndpoint`]: Point-to-point session primitive with state machine
//! - [`SessionHandle`]: Unified handle for controlling remote sessions
//! - [`SessionMessage`]: Unified message protocol with bidirectional control
//! - [`SessionPhase`], [`ControlRole`], [`AttachmentState`]: State machine types
//!
//! ## Legacy Types (Migration Path)
//!
//! The following types are being migrated to the new model:
//!
//! - `OnboardMessage` → Use [`SessionMessage`] for new code
//! - `RemoteSessionMessage` → Use [`SessionMessage`] for new code
//! - `RemoteSessionHandle` → Use [`SessionHandle`] for new code
//! - `RemoteSessionPhase` → Use [`SessionPhase`] for new code
//!
//! Existing session implementations (`InitiatorSession`, `ResponderSession`,
//! `ControllableSession`) still use the legacy types but internally use
//! the new [`BlockHolder`] for RAII block management.

// Core session building blocks (new unified model)
mod blocks;
mod endpoint;
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
// New Unified Session Model (Recommended)
// =============================================================================

/// RAII container for holding blocks during sessions.
pub use blocks::BlockHolder;

/// Point-to-point session endpoint with state machine.
pub use endpoint::{SessionEndpoint, SessionMessageTx, session_message_channel};

/// Unified handle for controlling remote sessions.
pub use handle::{SessionHandle, SessionHandleStateTx, session_handle_state_channel};

/// State machine types for the unified session model.
pub use state::{AttachmentState, ControlRole, SessionPhase};

/// Unified session message protocol.
pub use messages::{BlockInfo, SessionMessage, SessionStateSnapshot};

// =============================================================================
// Legacy Types (For Backward Compatibility)
// =============================================================================
// These types are being incrementally migrated to the new unified model.
// Use the new types above for new code.

/// Legacy session implementations.
pub use controllable::{ControllableSession, ControllableSessionResult};
pub use initiator::InitiatorSession;
pub use responder::ResponderSession;

/// Legacy message types (use [`SessionMessage`] for new code).
pub use messages::{
    BlockMatch, ControllableSessionOptions, G2BlockInfo, G3BlockInfo, OnboardMessage,
    RemoteSessionMessage, RemoteSessionPhase,
};

/// Legacy remote session handle (use [`SessionHandle`] for new code).
pub use remote_handle::{
    RemoteSessionHandle, RemoteSessionState, RemoteSessionStateTx, remote_session_state_channel,
};

/// Transport types.
pub use transport::{LocalTransport, MessageTransport, NovaTransport, RemoteSessionTx};

// Re-export from onboard for backward compatibility (will be removed later)
// pub use onboard::OnboardingSession;

use anyhow::Result;
use dashmap::DashMap;
use tokio::sync::mpsc;

pub type SessionId = uuid::Uuid;
pub type OnboardSessionTx = mpsc::Sender<OnboardMessage>;

/// Dispatch an inbound active message to the per-session task via its channel.
/// Each session's channel serializes message handling for that session.
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

/// Dispatch a remote session message to the appropriate session task.
///
/// Routes messages to either controllable_sessions (Decode side) or remote_sessions (Prefill side)
/// based on the message type.
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

/// Dispatch a unified SessionMessage to the appropriate session task.
///
/// This is the new unified protocol that will replace both dispatch_onboard_message
/// and dispatch_remote_session_message.
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
