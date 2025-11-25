// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod controllable;
mod initiator;
mod messages;
mod onboard; // Old implementation, will be replaced incrementally
mod remote_handle;
mod responder;
pub mod transport;

pub use controllable::{ControllableSession, ControllableSessionResult};
pub use initiator::InitiatorSession;
pub use messages::{
    BlockMatch, ControllableSessionOptions, G2BlockInfo, G3BlockInfo, OnboardMessage,
    RemoteSessionMessage, RemoteSessionPhase,
};
pub use remote_handle::{
    RemoteSessionHandle, RemoteSessionState, RemoteSessionStateTx, remote_session_state_channel,
};
pub use responder::ResponderSession;
pub use transport::{LocalTransport, MessageTransport, NovaTransport, RemoteSessionTx};

// Re-export from onboard for backward compatibility (will be removed later)
pub use onboard::OnboardingSession;

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
