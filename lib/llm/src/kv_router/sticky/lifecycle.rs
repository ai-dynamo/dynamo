// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session lifecycle controller for backend KV sessions.
//!
//! Manages open/close RPCs to workers via the event plane. Session affinity
//! (routing the same session to the same worker) is handled separately by
//! [`super::router::StickySessionRouter`].
//!
//! The controller:
//! - Lazily initializes a session_control event plane client
//! - Fires `open_session` inline (fail-fast if the client can't connect)
//! - Captures a deferred `SessionCloseAction` for execution after generation

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use dynamo_runtime::{
    component::Component,
    pipeline::{PushRouter, RouterMode, SingleIn},
    protocols::annotated::Annotated,
};
use futures::StreamExt;
use tokio::sync::OnceCell;

/// Untyped event plane client for session_control endpoint.
pub type EventPlaneClient = PushRouter<serde_json::Value, Annotated<serde_json::Value>>;

/// Default capacity for session KV slots (characters).
const DEFAULT_SESSION_CAPACITY: u64 = 65_536;
/// Extra worker-side timeout so router affinity expiry closes sessions first.
const SESSION_TIMEOUT_FALLBACK_BUFFER_SECS: u64 = 30;

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

    /// Open a backend session on a selected worker before generation starts.
    pub async fn open_session(
        &self,
        session_id: &str,
        timeout_secs: u64,
        instance_id: u64,
        context_id: &str,
    ) -> Result<bool> {
        let Some(client) = self.get_session_control_client().await else {
            return Ok(false);
        };
        let worker_timeout_secs = timeout_secs.saturating_add(SESSION_TIMEOUT_FALLBACK_BUFFER_SECS);

        // Open session synchronously -- the session must exist on the worker
        // before the first generate request arrives.
        let request = serde_json::json!({
            "action": "open_session",
            "session_id": session_id,
            "timeout": worker_timeout_secs,
            "capacity_of_str_len": DEFAULT_SESSION_CAPACITY,
        });
        let resp = send_session_request(
            &client,
            request,
            instance_id,
            session_id,
            context_id,
            "open_session",
        )
        .await;

        let resp =
            resp.ok_or_else(|| anyhow!("open_session RPC failed for session {session_id}"))?;
        ensure_session_open_succeeded(&resp, session_id)?;

        Ok(true)
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
                // is unavailable (worker not launched with --enable-streaming-session).
                match tokio::time::timeout(Duration::from_secs(5), c.wait_for_instances()).await {
                    Ok(Ok(_)) => {}
                    _ => {
                        tracing::warn!(
                            "No session_control endpoint registered. \
                             Session control will be ignored. \
                             To enable, launch the backend with --enable-streaming-session."
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

fn ensure_session_open_succeeded(
    response: &Annotated<serde_json::Value>,
    session_id: &str,
) -> Result<()> {
    if response.is_error() {
        return Err(anyhow!(
            "open_session returned annotated error for session {session_id}"
        ));
    }

    let body = response.data.as_ref().ok_or_else(|| {
        anyhow!("open_session returned no response body for session {session_id}")
    })?;

    let status = body.get("status").and_then(|value| value.as_str());
    match status {
        Some("ok") => Ok(()),
        Some(other) => {
            let message = body
                .get("message")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown error");
            Err(anyhow!(
                "open_session failed for session {session_id}: status={other}, message={message}"
            ))
        }
        None => Err(anyhow!(
            "open_session returned malformed response for session {session_id}: missing status"
        )),
    }
}

/// Send a session lifecycle request to a specific worker and return the first response.
///
/// Used by both synchronous (open_session) and fire-and-forget (close_session) paths.
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
