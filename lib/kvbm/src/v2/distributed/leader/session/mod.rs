// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod initiator;
mod messages;
mod onboard; // Old implementation, will be replaced incrementally
mod responder;
pub mod transport;

pub use initiator::InitiatorSession;
pub use messages::{BlockMatch, OnboardMessage};
pub use responder::ResponderSession;
pub use transport::{LocalTransport, MessageTransport, NovaTransport};

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
