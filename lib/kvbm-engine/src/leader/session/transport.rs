// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use ::velo::Messenger;
use anyhow::Result;
use bytes::Bytes;
use dashmap::DashMap;

use std::sync::Arc;

use crate::InstanceId;
use kvbm_physical::manager::SerializedLayout;

use super::{
    OnboardSessionTx, SessionId, SessionMessageTx, dispatch_onboard_message,
    messages::{OnboardMessage, SessionMessage},
};

/// Transport abstraction for sending onboarding messages without boxing futures.
///
/// This enum allows sessions to work with different transport mechanisms:
/// - Velo (distributed): Uses Velo active messages
/// - Local (testing): Direct channel dispatch
pub enum MessageTransport {
    Velo(VeloTransport),
    Local(LocalTransport),
}

impl MessageTransport {
    pub fn velo(messenger: Arc<Messenger>) -> Self {
        Self::Velo(VeloTransport::new(messenger))
    }

    pub fn local(
        sessions: Arc<DashMap<SessionId, OnboardSessionTx>>,
        session_sessions: Arc<DashMap<SessionId, SessionMessageTx>>,
    ) -> Self {
        Self::Local(LocalTransport::new(sessions, session_sessions))
    }

    /// Send an OnboardMessage to a target instance.
    pub async fn send(&self, target: InstanceId, message: OnboardMessage) -> Result<()> {
        match self {
            MessageTransport::Velo(transport) => transport.send(target, message).await,
            MessageTransport::Local(transport) => transport.send(target, message).await,
        }
    }

    /// Request worker metadata from a remote leader for RDMA transfers.
    ///
    /// This makes a synchronous RPC call to the remote leader's export_metadata
    /// handler and returns the `Vec<SerializedLayout>` from all remote workers.
    pub async fn request_metadata(&self, target: InstanceId) -> Result<Vec<SerializedLayout>> {
        match self {
            MessageTransport::Velo(transport) => transport.request_metadata(target).await,
            MessageTransport::Local(_) => {
                anyhow::bail!("request_metadata not supported for local transport")
            }
        }
    }

    /// Send a SessionMessage to a target instance.
    ///
    /// This is the unified session message protocol used for all session communication.
    pub async fn send_session(&self, target: InstanceId, message: SessionMessage) -> Result<()> {
        match self {
            MessageTransport::Velo(transport) => transport.send_session(target, message).await,
            MessageTransport::Local(transport) => transport.send_session(target, message).await,
        }
    }
}

/// Velo-based transport using active messages (fire-and-forget).
pub struct VeloTransport {
    messenger: Arc<Messenger>,
}

impl VeloTransport {
    pub fn new(messenger: Arc<Messenger>) -> Self {
        Self { messenger }
    }

    pub async fn send(&self, target: InstanceId, message: OnboardMessage) -> Result<()> {
        tracing::debug!(
            msg = message.variant_name(),
            target = %target,
            "Sending message"
        );

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        self.messenger
            .am_send("kvbm.leader.onboard")?
            .raw_payload(bytes)
            .instance(target)
            .send()
            .await?;

        tracing::debug!(target = %target, "Successfully sent");

        Ok(())
    }

    /// Request worker metadata from a remote leader for RDMA transfers.
    ///
    /// Makes a unary RPC call to get `Vec<SerializedLayout>` from
    /// the remote leader's workers.
    pub async fn request_metadata(&self, target: InstanceId) -> Result<Vec<SerializedLayout>> {
        tracing::debug!(target = %target, "Requesting metadata from instance");

        let response: Bytes = self
            .messenger
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
    /// Uses the unified "kvbm.leader.session" handler.
    pub async fn send_session(&self, target: InstanceId, message: SessionMessage) -> Result<()> {
        tracing::debug!(
            msg = message.variant_name(),
            target = %target,
            "Sending Session"
        );

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        self.messenger
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
    /// Unified session message receivers.
    session_sessions: Arc<DashMap<SessionId, SessionMessageTx>>,
}

impl LocalTransport {
    pub fn new(
        sessions: Arc<DashMap<SessionId, OnboardSessionTx>>,
        session_sessions: Arc<DashMap<SessionId, SessionMessageTx>>,
    ) -> Self {
        Self {
            sessions,
            session_sessions,
        }
    }

    pub async fn send(&self, _target: InstanceId, message: OnboardMessage) -> Result<()> {
        dispatch_onboard_message(&self.sessions, message).await
    }

    /// Send a SessionMessage (unified protocol).
    ///
    /// Routes to session_sessions by session ID.
    pub async fn send_session(&self, _target: InstanceId, message: SessionMessage) -> Result<()> {
        let session_id = message.session_id();

        let sender = self
            .session_sessions
            .get(&session_id)
            .map(|entry| entry.value().clone());
        if let Some(sender) = sender {
            sender
                .send(message)
                .await
                .map_err(|e| anyhow::anyhow!("failed to send to session {session_id}: {e}"))?;
            return Ok(());
        }

        anyhow::bail!("no session registered for session {session_id}");
    }
}
