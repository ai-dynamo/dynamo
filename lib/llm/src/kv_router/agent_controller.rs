// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session lifecycle controller for subagent KV isolation.
//!
//! Manages open/close RPCs to workers via the event plane. Session affinity
//! (routing the same session to the same worker) is handled separately by
//! [`super::sticky_sessions::StickySessionRouter`].
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

use crate::protocols::openai::nvext::SessionAction;

use super::sticky_sessions::StickySessionRouter;

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
        spawn_session_request(
            self.client.clone(),
            serde_json::json!({
                "action": "close_session",
                "session_id": self.session_id,
            }),
            self.instance_id,
            &self.session_id,
            context_id,
            "close_session",
        );
    }
}

/// Session lifecycle controller.
///
/// Owns a lazy event plane client for the `session_control` endpoint
/// and coordinates with [`StickySessionRouter`] for affinity management.
pub struct AgentController {
    session_control: OnceCell<EventPlaneClient>,
    component: Component,
}

impl AgentController {
    pub fn new(component: Component) -> Self {
        tracing::info!("AgentController initialized (session lifecycle RPCs, lazy client)");
        AgentController {
            session_control: OnceCell::new(),
            component,
        }
    }

    pub fn close_expired_session(self: Arc<Self>, session_id: String, instance_id: u64) {
        tokio::spawn(async move {
            match self.get_session_control_client().await {
                Ok(client) => {
                    tracing::info!(
                        worker_id = instance_id,
                        session_id = %session_id,
                        "Session affinity expired, closing worker session"
                    );
                    spawn_session_request(
                        client,
                        serde_json::json!({
                            "action": "close_session",
                            "session_id": session_id.clone(),
                        }),
                        instance_id,
                        &session_id,
                        "session-affinity-reaper",
                        "close_session",
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        worker_id = instance_id,
                        session_id = %session_id,
                        "Failed to create session_control client for affinity expiry close: {e}"
                    );
                }
            }
        });
    }

    /// Called after worker selection. Fires open_session if needed,
    /// returns a deferred close action for RequestGuard::finish().
    ///
    /// Also manages sticky session bindings: Open inserts affinity,
    /// Close removes it.
    ///
    /// Returns Err if session_control.action == Open but the client
    /// cannot be created (fail-fast: don't silently serve without isolation).
    pub async fn on_routed(
        &self,
        request: &crate::preprocessor::PreprocessedRequest,
        instance_id: u64,
        context_id: &str,
        sticky: Option<&StickySessionRouter>,
    ) -> Result<Option<SessionCloseAction>> {
        let routing = request.routing.as_ref();
        let session_control = routing.and_then(|r| r.session_control.clone());

        let Some(sc) = session_control else {
            return Ok(None);
        };

        let Some(action) = sc.action else {
            // No action -- just session_id for sticky routing (handled by StickySessionRouter)
            return Ok(None);
        };

        match action {
            SessionAction::Open => {
                // Fail fast if we can't connect to the session_control endpoint.
                let client = self.get_session_control_client().await?;
                let worker_timeout_secs = sc
                    .timeout
                    .saturating_add(SESSION_TIMEOUT_FALLBACK_BUFFER_SECS);

                // Open session synchronously -- the session must exist on the
                // worker before the first generate request arrives, otherwise
                // SGLang rejects it with "session does not exist".
                let request = serde_json::json!({
                    "action": "open_session",
                    "session_id": sc.session_id,
                    "timeout": worker_timeout_secs,
                    "capacity_of_str_len": DEFAULT_SESSION_CAPACITY,
                });
                match client.direct(SingleIn::new(request), instance_id).await {
                    Ok(mut stream) => {
                        if let Some(resp) = stream.next().await {
                            ensure_session_open_succeeded(&resp, &sc.session_id)?;
                            tracing::info!(
                                request_id = %context_id,
                                worker_id = instance_id,
                                session_id = %sc.session_id,
                                router_ttl_secs = sc.timeout,
                                worker_timeout_secs,
                                ?resp,
                                "open_session response"
                            );
                        }
                        // Drain remaining stream items
                        while stream.next().await.is_some() {}

                        // Bind affinity only after the worker confirms the
                        // session exists, otherwise retries can get pinned to a
                        // worker that never opened the session.
                        if let Some(sticky) = sticky {
                            sticky.bind(
                                &sc.session_id,
                                instance_id,
                                Duration::from_secs(sc.timeout),
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            request_id = %context_id,
                            worker_id = instance_id,
                            session_id = %sc.session_id,
                            "Failed open_session: {e}"
                        );
                        return Err(e);
                    }
                }

                Ok(None)
            }
            SessionAction::Close => {
                // Remove affinity immediately
                if let Some(sticky) = sticky {
                    sticky.unbind(&sc.session_id);
                }

                // Defer close to after generation completes
                match self.get_session_control_client().await {
                    Ok(client) => Ok(Some(SessionCloseAction {
                        session_id: sc.session_id.clone(),
                        client,
                        instance_id,
                    })),
                    Err(e) => {
                        tracing::warn!(
                            session_id = %sc.session_id,
                            "Failed to create session_control client for close, \
                             worker session will not be released: {e}"
                        );
                        Ok(None)
                    }
                }
            }
        }
    }

    async fn get_session_control_client(&self) -> Result<EventPlaneClient> {
        let client = self
            .session_control
            .get_or_try_init(|| async {
                let c = self.component.endpoint("session_control").client().await?;
                // Wait for at least one worker to register its session_control
                // endpoint before returning. Without this, .direct() fails on
                // the first call because discovery hasn't propagated yet.
                c.wait_for_instances().await?;
                EventPlaneClient::from_client_no_fault_detection(c, RouterMode::KV).await
            })
            .await?;
        Ok(client.clone())
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

/// Fire-and-forget session lifecycle request to a specific worker.
fn spawn_session_request(
    client: EventPlaneClient,
    request: serde_json::Value,
    instance_id: u64,
    session_id: &str,
    context_id: &str,
    action_label: &str,
) {
    let session_id = session_id.to_owned();
    let context_id = context_id.to_owned();
    let action_label = action_label.to_owned();

    tokio::spawn(async move {
        match client.direct(SingleIn::new(request), instance_id).await {
            Ok(mut stream) => {
                if let Some(resp) = stream.next().await {
                    tracing::info!(
                        request_id = %context_id,
                        worker_id = instance_id,
                        %session_id,
                        ?resp,
                        "{action_label} response"
                    );
                }
                while stream.next().await.is_some() {}
            }
            Err(e) => {
                tracing::warn!(
                    request_id = %context_id,
                    worker_id = instance_id,
                    %session_id,
                    "Failed {action_label}: {e}"
                );
            }
        }
    });
}
