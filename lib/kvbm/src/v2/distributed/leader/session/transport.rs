// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use bytes::Bytes;
use dashmap::DashMap;
use dynamo_nova::am::Nova;

use std::sync::Arc;

use crate::v2::InstanceId;

use super::{OnboardSessionTx, SessionId, dispatch_onboard_message, messages::{OnboardMessage, RemoteSessionMessage}};

/// Channel sender for remote session messages.
pub type RemoteSessionTx = tokio::sync::mpsc::Sender<RemoteSessionMessage>;

/// Transport abstraction for sending onboarding messages without boxing futures.
///
/// This enum allows sessions to work with different transport mechanisms:
/// - Nova (distributed): Uses Nova active messages
/// - Local (testing): Direct channel dispatch
pub enum MessageTransport {
    Nova(NovaTransport),
    Local(LocalTransport),
}

impl MessageTransport {
    pub fn nova(nova: Arc<Nova>) -> Self {
        Self::Nova(NovaTransport::new(nova))
    }

    pub fn local(
        sessions: Arc<DashMap<SessionId, OnboardSessionTx>>,
        controllable_sessions: Arc<DashMap<SessionId, RemoteSessionTx>>,
        remote_sessions: Arc<DashMap<SessionId, RemoteSessionTx>>,
    ) -> Self {
        Self::Local(LocalTransport::new(sessions, controllable_sessions, remote_sessions))
    }

    /// Send an OnboardMessage to a target instance.
    pub async fn send(&self, target: InstanceId, message: OnboardMessage) -> Result<()> {
        match self {
            MessageTransport::Nova(transport) => transport.send(target, message).await,
            MessageTransport::Local(transport) => transport.send(target, message).await,
        }
    }

    /// Send a RemoteSessionMessage to a target instance.
    ///
    /// This is used for the inverted control pattern (Prefill-Decode) messages.
    pub async fn send_remote_session(
        &self,
        target: InstanceId,
        message: RemoteSessionMessage,
    ) -> Result<()> {
        match self {
            MessageTransport::Nova(transport) => transport.send_remote_session(target, message).await,
            MessageTransport::Local(transport) => transport.send_remote_session(target, message).await,
        }
    }
}

/// Nova-based transport using active messages (fire-and-forget).
pub struct NovaTransport {
    nova: Arc<Nova>,
}

impl NovaTransport {
    pub fn new(nova: Arc<Nova>) -> Self {
        Self { nova }
    }

    pub async fn send(&self, target: InstanceId, message: OnboardMessage) -> Result<()> {
        eprintln!(
            "[TRANSPORT] Sending {:?} to instance {}",
            std::mem::discriminant(&message),
            target
        );

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        self.nova
            .am_send("kvbm.leader.onboard")?
            .raw_payload(bytes)
            .instance(target)
            .send()
            .await?;

        eprintln!("[TRANSPORT] Successfully sent to {}", target);

        Ok(())
    }

    /// Send a RemoteSessionMessage to a target instance.
    pub async fn send_remote_session(
        &self,
        target: InstanceId,
        message: RemoteSessionMessage,
    ) -> Result<()> {
        eprintln!(
            "[TRANSPORT] Sending RemoteSession {:?} to instance {}",
            std::mem::discriminant(&message),
            target
        );

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        self.nova
            .am_send("kvbm.leader.remote_session")?
            .raw_payload(bytes)
            .instance(target)
            .send()
            .await?;

        eprintln!("[TRANSPORT] Successfully sent remote session msg to {}", target);

        Ok(())
    }
}

/// Local transport for testing or same-instance communication.
///
/// Directly dispatches messages to session channels without network overhead.
pub struct LocalTransport {
    sessions: Arc<DashMap<SessionId, OnboardSessionTx>>,
    controllable_sessions: Arc<DashMap<SessionId, RemoteSessionTx>>,
    remote_sessions: Arc<DashMap<SessionId, RemoteSessionTx>>,
}

impl LocalTransport {
    pub fn new(
        sessions: Arc<DashMap<SessionId, OnboardSessionTx>>,
        controllable_sessions: Arc<DashMap<SessionId, RemoteSessionTx>>,
        remote_sessions: Arc<DashMap<SessionId, RemoteSessionTx>>,
    ) -> Self {
        Self {
            sessions,
            controllable_sessions,
            remote_sessions,
        }
    }

    pub async fn send(&self, _target: InstanceId, message: OnboardMessage) -> Result<()> {
        dispatch_onboard_message(&self.sessions, message).await
    }

    /// Send a RemoteSessionMessage.
    ///
    /// Routes to either controllable_sessions (Decode side) or remote_sessions (Prefill side)
    /// based on the message type.
    pub async fn send_remote_session(
        &self,
        _target: InstanceId,
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
            | RemoteSessionMessage::DetachSession { .. } => &self.controllable_sessions,
            RemoteSessionMessage::SessionState { .. }
            | RemoteSessionMessage::BlocksStaged { .. }
            | RemoteSessionMessage::SessionError { .. } => &self.remote_sessions,
        };

        if let Some(entry) = sessions.get(&session_id) {
            entry
                .value()
                .send(message)
                .await
                .map_err(|e| anyhow::anyhow!("failed to send to session {session_id}: {e}"))?;
            return Ok(());
        }

        anyhow::bail!("no remote session registered for session {session_id}");
    }
}
