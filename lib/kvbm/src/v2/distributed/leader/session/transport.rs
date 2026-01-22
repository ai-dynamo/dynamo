// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use bytes::Bytes;
use dashmap::DashMap;
use dynamo_nova::am::Nova;

use std::sync::Arc;

use crate::v2::InstanceId;
use crate::v2::physical::manager::SerializedLayout;

use super::{
    OnboardSessionTx, SessionId, SessionMessageTx, dispatch_onboard_message,
    messages::{OnboardMessage, RemoteSessionMessage, SessionMessage},
};

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
        Self::Local(LocalTransport::new(
            sessions,
            controllable_sessions,
            remote_sessions,
        ))
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
            MessageTransport::Nova(transport) => {
                transport.send_remote_session(target, message).await
            }
            MessageTransport::Local(transport) => {
                transport.send_remote_session(target, message).await
            }
        }
    }

    /// Request worker metadata from a remote leader for RDMA transfers.
    ///
    /// This makes a synchronous RPC call to the remote leader's export_metadata
    /// handler and returns the Vec<SerializedLayout> from all remote workers.
    ///
    /// # Arguments
    /// * `target` - Instance ID of the remote leader
    ///
    /// # Returns
    /// Vec<SerializedLayout> containing metadata from each remote worker (in rank order)
    pub async fn request_metadata(&self, target: InstanceId) -> Result<Vec<SerializedLayout>> {
        match self {
            MessageTransport::Nova(transport) => transport.request_metadata(target).await,
            MessageTransport::Local(_) => {
                anyhow::bail!("request_metadata not supported for local transport")
            }
        }
    }

    /// Send a SessionMessage to a target instance.
    ///
    /// This is the new unified session message protocol that replaces both
    /// OnboardMessage and RemoteSessionMessage for session communication.
    pub async fn send_session(&self, target: InstanceId, message: SessionMessage) -> Result<()> {
        match self {
            MessageTransport::Nova(transport) => transport.send_session(target, message).await,
            MessageTransport::Local(transport) => transport.send_session(target, message).await,
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
        tracing::debug!(
            msg = message.variant_name(),
            target = %target,
            "Sending message"
        );

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        self.nova
            .am_send("kvbm.leader.onboard")?
            .raw_payload(bytes)
            .instance(target)
            .send()
            .await?;

        tracing::debug!(target = %target, "Successfully sent");

        Ok(())
    }

    /// Send a RemoteSessionMessage to a target instance.
    pub async fn send_remote_session(
        &self,
        target: InstanceId,
        message: RemoteSessionMessage,
    ) -> Result<()> {
        tracing::debug!(
            msg = message.variant_name(),
            target = %target,
            "Sending RemoteSession"
        );

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        self.nova
            .am_send("kvbm.leader.remote_session")?
            .raw_payload(bytes)
            .instance(target)
            .send()
            .await?;

        tracing::debug!(target = %target, "Successfully sent remote session msg");

        Ok(())
    }

    /// Request worker metadata from a remote leader for RDMA transfers.
    ///
    /// Makes a unary RPC call to get Vec<SerializedLayout> from
    /// the remote leader's workers.
    pub async fn request_metadata(&self, target: InstanceId) -> Result<Vec<SerializedLayout>> {
        tracing::debug!(target = %target, "Requesting metadata from instance");

        let response: Bytes = self
            .nova
            .unary("kvbm.leader.export_metadata")?
            .instance(target)
            .send()
            .await?;

        // Deserialize the response
        let metadata: Vec<SerializedLayout> = serde_json::from_slice(&response)?;

        tracing::debug!(
            count = metadata.len(),
            target = %target,
            "Received metadata entries"
        );

        Ok(metadata)
    }

    /// Send a SessionMessage to a target instance.
    ///
    /// Uses the new unified "kvbm.leader.session" handler.
    pub async fn send_session(&self, target: InstanceId, message: SessionMessage) -> Result<()> {
        tracing::debug!(
            msg = message.variant_name(),
            target = %target,
            "Sending Session"
        );

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        self.nova
            .am_send("kvbm.leader.session")?
            .raw_payload(bytes)
            .instance(target)
            .send()
            .await?;

        tracing::debug!(target = %target, "Successfully sent session msg");

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
    /// Unified session message receivers (new protocol).
    session_sessions: Arc<DashMap<SessionId, SessionMessageTx>>,
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
            session_sessions: Arc::new(DashMap::new()),
        }
    }

    /// Create a LocalTransport with support for unified SessionMessage protocol.
    pub fn with_session_sessions(
        sessions: Arc<DashMap<SessionId, OnboardSessionTx>>,
        controllable_sessions: Arc<DashMap<SessionId, RemoteSessionTx>>,
        remote_sessions: Arc<DashMap<SessionId, RemoteSessionTx>>,
        session_sessions: Arc<DashMap<SessionId, SessionMessageTx>>,
    ) -> Self {
        Self {
            sessions,
            controllable_sessions,
            remote_sessions,
            session_sessions,
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

    /// Send a SessionMessage (new unified protocol).
    ///
    /// Routes to session_sessions by session ID.
    pub async fn send_session(&self, _target: InstanceId, message: SessionMessage) -> Result<()> {
        let session_id = message.session_id();

        if let Some(entry) = self.session_sessions.get(&session_id) {
            entry
                .value()
                .send(message)
                .await
                .map_err(|e| anyhow::anyhow!("failed to send to session {session_id}: {e}"))?;
            return Ok(());
        }

        anyhow::bail!("no session registered for session {session_id}");
    }
}
