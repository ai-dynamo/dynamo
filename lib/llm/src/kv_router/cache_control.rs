// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dynamo_runtime::{
    component::Component,
    pipeline::{PushRouter, RouterMode, SingleIn},
    protocols::annotated::Annotated,
};
use futures::StreamExt;

use crate::protocols::TokenIdType;

/// A PushRouter client typed for cache_control requests/responses.
///
/// Both request and response are untyped JSON. The worker's cache_control
/// endpoint returns {"status": "ok"/"error", ...} but the router treats
/// PIN as fire-and-forget and only logs the response at debug level.
pub type CacheControlClient = PushRouter<serde_json::Value, Annotated<serde_json::Value>>;

/// Create a cache_control client from a component.
///
/// Connects to the "cache_control" endpoint on the given component and returns
/// a PushRouter client for sending cache control operations (pin_prefix,
/// unpin_prefix) to workers.
pub async fn create_cache_control_client(component: &Component) -> Result<CacheControlClient> {
    let client = component.endpoint("cache_control").client().await?;
    CacheControlClient::from_client(client, RouterMode::KV).await
}

/// Fire-and-forget pin_prefix to the worker that served this request.
///
/// Spawns a detached task that sends the pin request and logs the outcome.
/// Does nothing if `client` is `None` (logs a warning) or `pin` is false.
pub fn spawn_pin_prefix(
    client: Option<&CacheControlClient>,
    token_ids: &[TokenIdType],
    instance_id: u64,
    context_id: &str,
) {
    let Some(cc) = client else {
        tracing::warn!(
            request_id = %context_id,
            "pin=true but no cache_control_client configured"
        );
        return;
    };

    let cc = cc.clone();
    let token_ids = token_ids.to_vec();
    let context_id = context_id.to_owned();

    tokio::spawn(async move {
        let pin_request = serde_json::json!({
            "action": "pin_prefix",
            "token_ids": token_ids,
        });
        match cc.direct(SingleIn::new(pin_request), instance_id).await {
            Ok(mut stream) => {
                if let Some(resp) = stream.next().await {
                    tracing::debug!(
                        request_id = %context_id,
                        worker_id = instance_id,
                        ?resp,
                        "pin_prefix response"
                    );
                }
            }
            Err(e) => {
                tracing::warn!(
                    request_id = %context_id,
                    worker_id = instance_id,
                    "Failed to pin prefix: {e}"
                );
            }
        }
    });
}
