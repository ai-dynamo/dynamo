// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session lifecycle controller for backend KV sessions.
//!
//! Manages the close RPC to workers via the event plane. Session affinity
//! (routing the same session to the same worker) is handled separately by
//! [`super::router::StickySessionRouter`]. There is no open RPC: radix-native
//! sessions are implicit on the worker (the first tagged generate creates the KV).
//!
//! The controller:
//! - Lazily initializes a session_control event plane client
//! - Captures a deferred `SessionCloseAction` (close_session) for execution after
//!   generation completes

use std::sync::Arc;
use std::time::Duration;

use dynamo_runtime::{
    component::Component,
    pipeline::{PushRouter, RouterMode, SingleIn},
    protocols::annotated::Annotated,
};
use futures::StreamExt;
use tokio::sync::OnceCell;

/// Untyped event plane client for session_control endpoint.
pub type EventPlaneClient = PushRouter<serde_json::Value, Annotated<serde_json::Value>>;

/// Deferred session close, executed after generation completes.
pub struct SessionCloseAction {
    pub session_id: String,
    pub client: EventPlaneClient,
    pub instance_id: u64,
}

impl SessionCloseAction {
    /// Fire the close_session RPC as a background task.
    pub fn execute(&self, context_id: &str) {
        let client = self.client.clone();
        let instance_id = self.instance_id;
        let session_id = self.session_id.clone();
        let context_id = context_id.to_owned();

        tokio::spawn(async move {
            let request = serde_json::json!({
                "action": "close_session",
                "session_id": session_id,
            });
            send_session_request(
                &client,
                request,
                instance_id,
                &session_id,
                &context_id,
                "close_session",
            )
            .await;
        });
    }
}

/// Session lifecycle controller.
///
/// Owns a lazy event plane client for the `session_control` endpoint.
pub struct SessionLifecycleController {
    /// `None` means we checked and no worker exposes session_control.
    session_control: OnceCell<Option<EventPlaneClient>>,
    component: Component,
}

impl SessionLifecycleController {
    pub fn new(component: Component) -> Self {
        tracing::debug!("SessionLifecycleController initialized");
        SessionLifecycleController {
            session_control: OnceCell::new(),
            component,
        }
    }

    pub fn close_expired_session(self: Arc<Self>, session_id: String, instance_id: u64) {
        tokio::spawn(async move {
            let Some(client) = self.get_session_control_client().await else {
                return;
            };
            tracing::info!(
                worker_id = instance_id,
                session_id = %session_id,
                "Session affinity expired, closing worker session"
            );
            let request = serde_json::json!({
                "action": "close_session",
                "session_id": session_id,
            });
            send_session_request(
                &client,
                request,
                instance_id,
                &session_id,
                "session-affinity-reaper",
                "close_session",
            )
            .await;
        });
    }

    /// Build a deferred close action for RequestGuard::finish().
    pub async fn deferred_close(
        &self,
        session_id: String,
        instance_id: u64,
    ) -> Option<SessionCloseAction> {
        self.get_session_control_client()
            .await
            .map(|client| SessionCloseAction {
                session_id,
                client,
                instance_id,
            })
    }

    async fn get_session_control_client(&self) -> Option<EventPlaneClient> {
        let maybe_client = self
            .session_control
            .get_or_init(|| async {
                let c = match self.component.endpoint("session_control").client().await {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!(
                            "Failed to create session_control client: {e}. \
                             Session control will be ignored for all requests."
                        );
                        return None;
                    }
                };
                // Wait briefly for at least one worker to register its
                // session_control endpoint. If none appear, session control
                // is unavailable (worker not launched with --enable-session-radix-cache).
                match tokio::time::timeout(Duration::from_secs(5), c.wait_for_instances()).await {
                    Ok(Ok(_)) => {}
                    _ => {
                        tracing::warn!(
                            "No session_control endpoint registered. \
                             Session control will be ignored. \
                             To enable, launch the backend with --enable-session-radix-cache."
                        );
                        return None;
                    }
                }
                match EventPlaneClient::from_client_no_fault_detection(c, RouterMode::KV).await {
                    Ok(client) => Some(client),
                    Err(e) => {
                        tracing::warn!(
                            "Failed to create session_control event plane client: {e}. \
                             Session control will be ignored."
                        );
                        None
                    }
                }
            })
            .await;
        maybe_client.clone()
    }
}

/// Send a session lifecycle request to a specific worker and return the first response.
///
/// Used by the fire-and-forget close_session paths.
async fn send_session_request(
    client: &EventPlaneClient,
    request: serde_json::Value,
    instance_id: u64,
    session_id: &str,
    context_id: &str,
    action_label: &str,
) -> Option<Annotated<serde_json::Value>> {
    match client.direct(SingleIn::new(request), instance_id).await {
        Ok(mut stream) => {
            let resp = stream.next().await;
            if let Some(ref r) = resp {
                tracing::info!(
                    request_id = %context_id,
                    worker_id = instance_id,
                    %session_id,
                    ?r,
                    "{action_label} response"
                );
            }
            // Drain remaining stream to avoid "Failed to publish complete final" errors.
            while stream.next().await.is_some() {}
            resp
        }
        Err(e) => {
            tracing::warn!(
                request_id = %context_id,
                worker_id = instance_id,
                %session_id,
                "Failed {action_label}: {e}"
            );
            None
        }
    }
}
