// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use async_trait::async_trait;
use dynamo_runtime::{
    component::Component,
    pipeline::{PushRouter, RouterMode, SingleIn},
    protocols::annotated::Annotated,
};
use futures::StreamExt;
use tokio::sync::Mutex as AsyncMutex;

use super::{SESSION_TIMEOUT_FALLBACK_BUFFER, SessionTarget};

const DEFAULT_SESSION_CAPACITY: u64 = 65_536;

type EventPlaneClient = PushRouter<serde_json::Value, Annotated<serde_json::Value>>;

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct LifecycleError {
    message: String,
}

impl LifecycleError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[async_trait]
pub trait SessionLifecycleBackend: Send + Sync {
    async fn open(
        &self,
        session_id: &str,
        timeout: Duration,
        target: SessionTarget,
        context_id: &str,
    ) -> Result<(), LifecycleError>;

    async fn close(
        &self,
        session_id: &str,
        target: SessionTarget,
        context_id: &str,
    ) -> Result<(), LifecycleError>;
}

pub struct EventSessionLifecycle {
    component: Component,
    client: AsyncMutex<Option<EventPlaneClient>>,
}

impl EventSessionLifecycle {
    pub fn new(component: Component) -> Self {
        Self {
            component,
            client: AsyncMutex::new(None),
        }
    }

    async fn client(&self) -> Result<EventPlaneClient, LifecycleError> {
        let mut cached = self.client.lock().await;
        if let Some(client) = cached.as_ref() {
            return Ok(client.clone());
        }

        let client = self
            .component
            .endpoint("session_control")
            .client()
            .await
            .map_err(|error| {
                LifecycleError::new(format!("failed to create session-control client: {error}"))
            })?;
        tokio::time::timeout(Duration::from_secs(5), client.wait_for_instances())
            .await
            .map_err(|_| {
                LifecycleError::new("no session-control endpoint registered within five seconds")
            })?
            .map_err(|error| {
                LifecycleError::new(format!(
                    "failed waiting for a session-control endpoint: {error}"
                ))
            })?;
        let router = EventPlaneClient::from_client_no_fault_detection(client, RouterMode::KV)
            .await
            .map_err(|error| {
                LifecycleError::new(format!("failed to create session-control router: {error}"))
            })?;
        *cached = Some(router.clone());
        Ok(router)
    }

    async fn send(
        &self,
        request: serde_json::Value,
        session_id: &str,
        target: SessionTarget,
        context_id: &str,
        action: &str,
    ) -> Result<Annotated<serde_json::Value>, LifecycleError> {
        let client = self.client().await?;
        let mut stream = client
            .dispatch_exact(SingleIn::new(request), target.worker_id)
            .await
            .map_err(|error| {
                LifecycleError::new(format!(
                    "{action} RPC failed for session {session_id}: {error}"
                ))
            })?;
        let response = stream.next().await.ok_or_else(|| {
            LifecycleError::new(format!(
                "{action} returned no response for session {session_id}"
            ))
        })?;
        while stream.next().await.is_some() {}
        tracing::info!(
            request_id = %context_id,
            worker_id = target.worker_id,
            %session_id,
            ?response,
            "{action} response"
        );
        Ok(response)
    }

    fn response_body<'a>(
        response: &'a Annotated<serde_json::Value>,
        session_id: &str,
        action: &str,
    ) -> Result<&'a serde_json::Value, LifecycleError> {
        if response.is_error() {
            return Err(LifecycleError::new(format!(
                "{action} returned an annotated error for session {session_id}"
            )));
        }
        let body = response.data.as_ref().ok_or_else(|| {
            LifecycleError::new(format!(
                "{action} returned no response body for session {session_id}"
            ))
        })?;
        match body.get("status").and_then(serde_json::Value::as_str) {
            Some("ok") => Ok(body),
            Some(status) => Err(LifecycleError::new(format!(
                "{action} failed for session {session_id}: status={status}, message={}",
                body.get("message")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown error")
            ))),
            None => Err(LifecycleError::new(format!(
                "{action} returned a malformed response for session {session_id}"
            ))),
        }
    }
}

#[async_trait]
impl SessionLifecycleBackend for EventSessionLifecycle {
    async fn open(
        &self,
        session_id: &str,
        timeout: Duration,
        target: SessionTarget,
        context_id: &str,
    ) -> Result<(), LifecycleError> {
        let worker_timeout = timeout.saturating_add(SESSION_TIMEOUT_FALLBACK_BUFFER);
        let request = serde_json::json!({
            "action": "open_session",
            "session_id": session_id,
            "timeout": worker_timeout.as_secs(),
            "capacity_of_str_len": DEFAULT_SESSION_CAPACITY,
        });
        let response = self
            .send(request, session_id, target, context_id, "open_session")
            .await?;
        Self::response_body(&response, session_id, "open_session")?;
        Ok(())
    }

    async fn close(
        &self,
        session_id: &str,
        target: SessionTarget,
        context_id: &str,
    ) -> Result<(), LifecycleError> {
        let request = serde_json::json!({
            "action": "close_session",
            "session_id": session_id,
        });
        let response = self
            .send(request, session_id, target, context_id, "close_session")
            .await?;
        Self::response_body(&response, session_id, "close_session")?;
        Ok(())
    }
}
