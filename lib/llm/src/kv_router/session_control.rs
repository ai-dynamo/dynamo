// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_runtime::{
    component::Component,
    pipeline::{PushRouter, RouterMode, SingleIn},
    protocols::annotated::Annotated,
};
use futures::StreamExt;

/// A PushRouter client typed for session_control requests/responses.
///
/// Both request and response are untyped JSON. The worker's session_control
/// endpoint dispatches open_session / close_session to SGLang's TokenizerManager.
pub type SessionControlClient = PushRouter<serde_json::Value, Annotated<serde_json::Value>>;

/// Create a session_control client from a component.
///
/// Connects to the "session_control" endpoint on the given component and returns
/// a PushRouter client for sending session lifecycle operations to workers.
pub(crate) async fn create_session_control_client(
    component: &Component,
) -> Result<SessionControlClient> {
    let client = component.endpoint("session_control").client().await?;
    SessionControlClient::from_client(client, RouterMode::KV).await
}

/// Fire-and-forget open_session to the worker that will serve this subagent.
///
/// Spawns a detached task that sends the open request and logs the outcome.
pub fn spawn_open_session(
    client: &SessionControlClient,
    session_id: &str,
    instance_id: u64,
    timeout: u64,
    context_id: &str,
) {
    let client = client.clone();
    let session_id = session_id.to_owned();
    let context_id = context_id.to_owned();

    tokio::spawn(async move {
        let request = serde_json::json!({
            "action": "open_session",
            "session_id": session_id,
            "timeout": timeout,
            "capacity_of_str_len": 65536,
        });
        match client.direct(SingleIn::new(request), instance_id).await {
            Ok(mut stream) => {
                if let Some(resp) = stream.next().await {
                    tracing::info!(
                        request_id = %context_id,
                        worker_id = instance_id,
                        %session_id,
                        ?resp,
                        "open_session response"
                    );
                }
                while stream.next().await.is_some() {}
            }
            Err(e) => {
                tracing::warn!(
                    request_id = %context_id,
                    worker_id = instance_id,
                    %session_id,
                    "Failed to open session: {e}"
                );
            }
        }
    });
}

/// Fire-and-forget close_session on the worker holding this subagent's KV.
///
/// Spawns a detached task that sends the close request and logs the outcome.
pub fn spawn_close_session(
    client: &SessionControlClient,
    session_id: &str,
    instance_id: u64,
    context_id: &str,
) {
    let client = client.clone();
    let session_id = session_id.to_owned();
    let context_id = context_id.to_owned();

    tokio::spawn(async move {
        let request = serde_json::json!({
            "action": "close_session",
            "session_id": session_id,
        });
        match client.direct(SingleIn::new(request), instance_id).await {
            Ok(mut stream) => {
                if let Some(resp) = stream.next().await {
                    tracing::info!(
                        request_id = %context_id,
                        worker_id = instance_id,
                        %session_id,
                        ?resp,
                        "close_session response"
                    );
                }
                while stream.next().await.is_some() {}
            }
            Err(e) => {
                tracing::warn!(
                    request_id = %context_id,
                    worker_id = instance_id,
                    %session_id,
                    "Failed to close session: {e}"
                );
            }
        }
    });
}
