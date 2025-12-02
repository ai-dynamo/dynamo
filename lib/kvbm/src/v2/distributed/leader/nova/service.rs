// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use bytes::Bytes;
use dashmap::DashMap;
use dynamo_nova::am::{Nova, NovaHandler};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::v2::distributed::leader::session::{
    OnboardMessage, OnboardSessionTx, RemoteSessionMessage, RemoteSessionTx, SessionId,
    SessionMessage, SessionMessageTx, dispatch_onboard_message, dispatch_remote_session_message,
    dispatch_session_message,
};
use crate::v2::physical::manager::SerializedLayout;

/// Type alias for async export metadata callback.
/// Returns a boxed future that resolves to Vec<SerializedLayout>.
pub type ExportMetadataCallback = Arc<
    dyn Fn() -> Pin<Box<dyn Future<Output = Result<Vec<SerializedLayout>>> + Send>> + Send + Sync,
>;

/// Nova leader service for handling distributed onboarding messages.
///
/// This service registers handlers for:
/// 1. OnboardMessage: Standard find_matches flow (initiator → responder)
/// 2. RemoteSessionMessage: Inverted control pattern (Prefill-Decode)
/// 3. SessionMessage: Unified session protocol (new)
/// 4. Export metadata RPC: Returns worker layout metadata for RDMA
pub struct NovaLeaderService {
    nova: Arc<Nova>,
    sessions: Arc<DashMap<SessionId, OnboardSessionTx>>,
    /// Callback to spawn new responder sessions.
    /// Takes the CreateSession message and creates a new responder task.
    spawn_responder: Option<Arc<dyn Fn(OnboardMessage) -> Result<()> + Send + Sync>>,

    // Inverted control pattern (Prefill-Decode) fields
    /// Map of controllable sessions (Decode side).
    controllable_sessions: Option<Arc<DashMap<SessionId, RemoteSessionTx>>>,
    /// Map of remote session receivers (Prefill side).
    remote_sessions: Option<Arc<DashMap<SessionId, RemoteSessionTx>>>,

    // Unified session protocol (new)
    /// Map of unified session receivers.
    session_sessions: Option<Arc<DashMap<SessionId, SessionMessageTx>>>,

    // RDMA metadata export
    /// Callback to export worker metadata for RDMA transfers.
    export_metadata: Option<ExportMetadataCallback>,
}

impl NovaLeaderService {
    pub fn new(nova: Arc<Nova>, sessions: Arc<DashMap<SessionId, OnboardSessionTx>>) -> Self {
        Self {
            nova,
            sessions,
            spawn_responder: None,
            controllable_sessions: None,
            remote_sessions: None,
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

    /// Set the controllable sessions map (Decode side of inverted pattern).
    pub fn with_controllable_sessions(
        mut self,
        sessions: Arc<DashMap<SessionId, RemoteSessionTx>>,
    ) -> Self {
        self.controllable_sessions = Some(sessions);
        self
    }

    /// Set the remote sessions map (Prefill side of inverted pattern).
    pub fn with_remote_sessions(
        mut self,
        sessions: Arc<DashMap<SessionId, RemoteSessionTx>>,
    ) -> Self {
        self.remote_sessions = Some(sessions);
        self
    }

    /// Set the unified session sessions map (new protocol).
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
    /// to enable RDMA transfers. The callback should return Vec<SerializedLayout>
    /// containing metadata from all workers.
    pub fn with_export_metadata(mut self, callback: ExportMetadataCallback) -> Self {
        self.export_metadata = Some(callback);
        self
    }

    /// Register all Nova handlers for leader-to-leader communication.
    pub fn register_handlers(self) -> Result<()> {
        self.register_onboard_handler()?;

        // Only register remote_session handler if inverted pattern is configured
        if self.controllable_sessions.is_some() || self.remote_sessions.is_some() {
            self.register_remote_session_handler()?;
        }

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

        let handler = NovaHandler::am_handler_async("kvbm.leader.onboard", move |ctx| {
            let sessions = sessions.clone();
            let spawn_responder = spawn_responder.clone();

            async move {
                // Fast path: just deserialize and dispatch
                let message: OnboardMessage = serde_json::from_slice(&ctx.payload)
                    .map_err(|e| anyhow::anyhow!("failed to deserialize OnboardMessage: {e}"))?;

                let session_id = message.session_id();

                eprintln!(
                    "[HANDLER] Received message: {:?} for session {}",
                    std::mem::discriminant(&message),
                    session_id
                );

                // If this is a CreateSession and no session exists, spawn responder
                if matches!(message, OnboardMessage::CreateSession { .. })
                    && !sessions.contains_key(&session_id) {
                        eprintln!("[HANDLER] Spawning new ResponderSession for {}", session_id);
                        if let Some(ref spawner) = spawn_responder {
                            spawner(message.clone()).ok(); // Best-effort spawn
                        }
                    }

                // Dispatch to session channel (will create if needed by spawner above)
                eprintln!("[HANDLER] Dispatching message to session {}", session_id);
                dispatch_onboard_message(&sessions, message).await?;

                Ok(())
            }
        })
        .build();

        self.nova.register_handler(handler)?;

        Ok(())
    }

    /// Register the "kvbm.leader.remote_session" handler.
    ///
    /// This handler supports the inverted control pattern (Prefill-Decode):
    /// - Routes messages to controllable_sessions (Decode side) or remote_sessions (Prefill side)
    fn register_remote_session_handler(&self) -> Result<()> {
        let controllable_sessions = self
            .controllable_sessions
            .clone()
            .unwrap_or_else(|| Arc::new(DashMap::new()));
        let remote_sessions = self
            .remote_sessions
            .clone()
            .unwrap_or_else(|| Arc::new(DashMap::new()));

        let handler = NovaHandler::am_handler_async("kvbm.leader.remote_session", move |ctx| {
            let controllable_sessions = controllable_sessions.clone();
            let remote_sessions = remote_sessions.clone();

            async move {
                let message: RemoteSessionMessage =
                    serde_json::from_slice(&ctx.payload).map_err(|e| {
                        anyhow::anyhow!("failed to deserialize RemoteSessionMessage: {e}")
                    })?;

                let session_id = message.session_id();

                eprintln!(
                    "[HANDLER] Received remote session message: {:?} for session {}",
                    std::mem::discriminant(&message),
                    session_id
                );

                // Dispatch to appropriate session map
                dispatch_remote_session_message(&controllable_sessions, &remote_sessions, message)
                    .await?;

                Ok(())
            }
        })
        .build();

        self.nova.register_handler(handler)?;

        Ok(())
    }

    /// Register the "kvbm.leader.session" handler.
    ///
    /// This handler supports the new unified session protocol.
    /// Routes SessionMessages to the appropriate session endpoint.
    fn register_session_handler(&self) -> Result<()> {
        let session_sessions = self
            .session_sessions
            .clone()
            .expect("session_sessions required for handler registration");

        let handler = NovaHandler::am_handler_async("kvbm.leader.session", move |ctx| {
            let session_sessions = session_sessions.clone();

            async move {
                let message: SessionMessage = serde_json::from_slice(&ctx.payload)
                    .map_err(|e| anyhow::anyhow!("failed to deserialize SessionMessage: {e}"))?;

                let session_id = message.session_id();

                eprintln!(
                    "[HANDLER] Received session message: {:?} for session {}",
                    std::mem::discriminant(&message),
                    session_id
                );

                // Dispatch to session endpoint
                dispatch_session_message(&session_sessions, message).await?;

                Ok(())
            }
        })
        .build();

        self.nova.register_handler(handler)?;

        Ok(())
    }

    /// Register the "kvbm.leader.export_metadata" handler.
    ///
    /// This handler returns Vec<SerializedLayout> containing metadata from all workers.
    /// Used by remote leaders to enable RDMA transfers.
    fn register_export_metadata_handler(&self) -> Result<()> {
        let export_metadata = self
            .export_metadata
            .clone()
            .expect("export_metadata callback required for handler registration");

        let handler =
            NovaHandler::unary_handler_async("kvbm.leader.export_metadata", move |_ctx| {
                let export_metadata = export_metadata.clone();

                async move {
                    eprintln!("[HANDLER] Received export_metadata request");

                    // Call the async callback to get metadata from all workers
                    let metadata_vec = export_metadata().await?;

                    // Serialize the Vec<SerializedLayout> for transport
                    let serialized = serde_json::to_vec(&metadata_vec)?;

                    eprintln!(
                        "[HANDLER] Returning {} worker metadata entries",
                        metadata_vec.len()
                    );

                    Ok(Some(Bytes::from(serialized)))
                }
            })
            .build();

        self.nova.register_handler(handler)?;

        Ok(())
    }
}
