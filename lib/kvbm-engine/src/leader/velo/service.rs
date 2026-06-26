// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use ::velo::{Handler, Messenger};
use anyhow::Result;
use bytes::Bytes;
use dashmap::DashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::leader::session::{
    OnboardMessage, OnboardSessionTx, SessionId, SessionMessage, SessionMessageTx,
    dispatch_onboard_message, dispatch_session_message,
};
use kvbm_physical::manager::SerializedLayout;

/// Type alias for async export metadata callback.
/// Returns a boxed future that resolves to `Vec<SerializedLayout>`.
pub type ExportMetadataCallback = Arc<
    dyn Fn() -> Pin<Box<dyn Future<Output = Result<Vec<SerializedLayout>>> + Send>> + Send + Sync,
>;

/// Velo leader service for handling distributed onboarding messages.
///
/// This service registers handlers for:
/// 1. OnboardMessage: Standard find_matches flow (initiator → responder)
/// 2. SessionMessage: Unified session protocol
/// 3. Export metadata RPC: Returns worker layout metadata for RDMA
pub struct VeloLeaderService {
    messenger: Arc<Messenger>,
    sessions: Arc<DashMap<SessionId, OnboardSessionTx>>,
    /// Callback to spawn new responder sessions.
    /// Takes the CreateSession message and creates a new responder task.
    spawn_responder: Option<Arc<dyn Fn(OnboardMessage) -> Result<()> + Send + Sync>>,

    // Unified session protocol
    /// Map of unified session receivers.
    session_sessions: Option<Arc<DashMap<SessionId, SessionMessageTx>>>,

    // RDMA metadata export
    /// Callback to export worker metadata for RDMA transfers.
    export_metadata: Option<ExportMetadataCallback>,
}

impl VeloLeaderService {
    pub fn new(
        messenger: Arc<Messenger>,
        sessions: Arc<DashMap<SessionId, OnboardSessionTx>>,
    ) -> Self {
        Self {
            messenger,
            sessions,
            spawn_responder: None,
            session_sessions: None,
            export_metadata: None,
        }
    }

    /// Set the callback for spawning responder sessions.
    pub fn with_spawn_responder<F>(mut self, f: F) -> Self
    where
        F: Fn(OnboardMessage) -> Result<()> + Send + Sync + 'static,
    {
        self.spawn_responder = Some(Arc::new(f));
        self
    }

    /// Set the unified session sessions map.
    pub fn with_session_sessions(
        mut self,
        sessions: Arc<DashMap<SessionId, SessionMessageTx>>,
    ) -> Self {
        self.session_sessions = Some(sessions);
        self
    }

    /// Set the callback for exporting worker metadata (RDMA).
    ///
    /// This callback is invoked when a remote leader requests metadata
    /// to enable RDMA transfers. The callback should return `Vec<SerializedLayout>`
    /// containing metadata from all workers.
    pub fn with_export_metadata(mut self, callback: ExportMetadataCallback) -> Self {
        self.export_metadata = Some(callback);
        self
    }

    /// Register all Velo handlers for leader-to-leader communication.
    pub fn register_handlers(self) -> Result<()> {
        self.register_onboard_handler()?;

        // Register session handler if unified protocol is configured
        if self.session_sessions.is_some() {
            self.register_session_handler()?;
        }

        // Register export_metadata handler if callback is configured
        if self.export_metadata.is_some() {
            self.register_export_metadata_handler()?;
        }

        Ok(())
    }

    /// Register the "kvbm.leader.onboard" handler.
    ///
    /// This handler is intentionally simple and fast:
    /// - Deserializes the message
    /// - If CreateSession and session doesn't exist, spawns responder
    /// - Dispatches to session channel
    /// - Returns immediately (< 1ms)
    fn register_onboard_handler(&self) -> Result<()> {
        let sessions = self.sessions.clone();
        let spawn_responder = self.spawn_responder.clone();

        let handler = Handler::am_handler_async("kvbm.leader.onboard", move |ctx| {
            let sessions = sessions.clone();
            let spawn_responder = spawn_responder.clone();

            async move {
                // Fast path: just deserialize and dispatch
                let message: OnboardMessage = serde_json::from_slice(&ctx.payload)
                    .map_err(|e| anyhow::anyhow!("failed to deserialize OnboardMessage: {e}"))?;

                let session_id = message.session_id();

                tracing::debug!(
                    variant = message.variant_name(),
                    %session_id,
                    "Received onboard message"
                );

                // If this is a CreateSession and no session exists, spawn responder
                if matches!(message, OnboardMessage::CreateSession { .. })
                    && !sessions.contains_key(&session_id)
                {
                    tracing::debug!(%session_id, "Spawning new ResponderSession");
                    if let Some(ref spawner) = spawn_responder {
                        spawner(message.clone()).ok(); // Best-effort spawn
                    }
                }

                // Dispatch to session channel (will create if needed by spawner above)
                tracing::debug!(%session_id, "Dispatching message to session");
                dispatch_onboard_message(&sessions, message).await?;

                Ok(())
            }
        })
        .build();

        self.messenger.register_handler(handler)?;

        Ok(())
    }

    /// Register the "kvbm.leader.session" handler.
    ///
    /// This handler supports the unified session protocol.
    /// Routes SessionMessages to the appropriate session endpoint.
    fn register_session_handler(&self) -> Result<()> {
        let session_sessions = self
            .session_sessions
            .clone()
            .expect("session_sessions required for handler registration");

        let handler = Handler::am_handler_async("kvbm.leader.session", move |ctx| {
            let session_sessions = session_sessions.clone();

            async move {
                let message: SessionMessage = serde_json::from_slice(&ctx.payload)
                    .map_err(|e| anyhow::anyhow!("failed to deserialize SessionMessage: {e}"))?;

                let session_id = message.session_id();

                tracing::debug!(
                    variant = message.variant_name(),
                    %session_id,
                    "Received session message"
                );

                // Dispatch to session endpoint
                dispatch_session_message(&session_sessions, message).await?;

                Ok(())
            }
        })
        .build();

        self.messenger.register_handler(handler)?;

        Ok(())
    }

    /// Register the "kvbm.leader.export_metadata" handler.
    ///
    /// This handler returns `Vec<SerializedLayout>` containing metadata from all workers.
    /// Used by remote leaders to enable RDMA transfers.
    fn register_export_metadata_handler(&self) -> Result<()> {
        let export_metadata = self
            .export_metadata
            .clone()
            .expect("export_metadata callback required for handler registration");

        let handler = Handler::unary_handler_async("kvbm.leader.export_metadata", move |_ctx| {
            let export_metadata = export_metadata.clone();

            async move {
                tracing::debug!("Received export_metadata request");

                // Call the async callback to get metadata from all workers
                let metadata_vec = export_metadata().await?;

                // Serialize the Vec<SerializedLayout> for transport
                let serialized = serde_json::to_vec(&metadata_vec)?;

                tracing::debug!(
                    count = metadata_vec.len(),
                    "Returning worker metadata entries"
                );

                Ok(Some(Bytes::from(serialized)))
            }
        })
        .build();

        self.messenger.register_handler(handler)?;

        Ok(())
    }
}
