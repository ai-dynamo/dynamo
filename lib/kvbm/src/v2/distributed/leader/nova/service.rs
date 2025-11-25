// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dashmap::DashMap;
use dynamo_nova::am::{Nova, NovaHandler};

use std::sync::Arc;

use crate::v2::distributed::leader::session::{
    OnboardMessage, OnboardSessionTx, SessionId, dispatch_onboard_message,
};

/// Nova leader service for handling distributed onboarding messages.
///
/// This service registers a simple handler that:
/// 1. Deserializes incoming OnboardMessage
/// 2. Dispatches to the appropriate session channel
/// 3. For CreateSession messages, spawns a new responder session if needed
pub struct NovaLeaderService {
    nova: Arc<Nova>,
    sessions: Arc<DashMap<SessionId, OnboardSessionTx>>,
    /// Callback to spawn new responder sessions.
    /// Takes the CreateSession message and creates a new responder task.
    spawn_responder: Option<Arc<dyn Fn(OnboardMessage) -> Result<()> + Send + Sync>>,
}

impl NovaLeaderService {
    pub fn new(nova: Arc<Nova>, sessions: Arc<DashMap<SessionId, OnboardSessionTx>>) -> Self {
        Self {
            nova,
            sessions,
            spawn_responder: None,
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

    /// Register all Nova handlers for leader-to-leader communication.
    pub fn register_handlers(self) -> Result<()> {
        self.register_onboard_handler()?;
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

                eprintln!("[HANDLER] Received message: {:?} for session {}",
                    std::mem::discriminant(&message), session_id);

                // If this is a CreateSession and no session exists, spawn responder
                if matches!(message, OnboardMessage::CreateSession { .. }) {
                    if !sessions.contains_key(&session_id) {
                        eprintln!("[HANDLER] Spawning new ResponderSession for {}", session_id);
                        if let Some(ref spawner) = spawn_responder {
                            spawner(message.clone()).ok(); // Best-effort spawn
                        }
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
}
