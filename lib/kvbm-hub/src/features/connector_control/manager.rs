// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `ConnectorControlManager` — proxies HTTP control requests to a specific
//! connector via velo.

use std::sync::{Arc, OnceLock};

use axum::Router;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header::CONTENT_TYPE};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, put};
use futures::future::BoxFuture;
use kvbm_control_protocol::ControlReply;
use serde_json::Value as JsonValue;
use velo_common::InstanceId;

use crate::features::{FeatureError, FeatureManager, HubContext};
use crate::handlers::{HEARTBEAT_HANDLER, HeartbeatAck, HeartbeatRequest};
use crate::protocol::{self, Feature, FeatureKey};
use crate::registry::PeerRegistry;

use super::{REGISTER_LEADER_HANDLER, RESET_HANDLER};

/// HTTP→velo proxy for the connector-leader control plane.
///
/// State is filled during [`FeatureManager::attach`]:
/// - `velo` — needed to forward unary calls to a target connector.
/// - `registry` — needed to confirm a target instance is currently
///   registered before issuing the velo call.
///
/// When the hub is launched without a velo transport (`velo_port: None`)
/// the proxy routes return `503 Service Unavailable`.
pub struct ConnectorControlManager {
    velo: OnceLock<Arc<velo::Velo>>,
    registry: OnceLock<Arc<dyn PeerRegistry>>,
}

impl Default for ConnectorControlManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ConnectorControlManager {
    pub fn new() -> Self {
        Self {
            velo: OnceLock::new(),
            registry: OnceLock::new(),
        }
    }
}

impl std::fmt::Debug for ConnectorControlManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectorControlManager")
            .field("velo_attached", &self.velo.get().is_some())
            .field("registry_attached", &self.registry.get().is_some())
            .finish()
    }
}

impl FeatureManager for ConnectorControlManager {
    fn key(&self) -> FeatureKey {
        FeatureKey::ConnectorControl
    }

    fn attach<'a>(&'a self, ctx: HubContext) -> BoxFuture<'a, Result<(), FeatureError>> {
        Box::pin(async move {
            let _ = self.registry.set(ctx.registry);
            if let Some(v) = ctx.velo {
                let _ = self.velo.set(v);
            } else {
                tracing::warn!(
                    "ConnectorControlManager: hub has no velo transport — \
                     proxy routes will return 503"
                );
            }
            Ok(())
        })
    }

    fn on_register<'a>(
        &'a self,
        _instance_id: InstanceId,
        _feature: &'a Feature,
    ) -> BoxFuture<'a, Result<(), FeatureError>> {
        // No client-side `Feature::ConnectorControl` variant exists — this
        // manager only contributes routes. The hub's dispatcher will never
        // call this. Return a key-mismatch error if it ever does, which
        // surfaces as a clear bug.
        Box::pin(async move {
            Err(FeatureError::KeyMismatch {
                manager: FeatureKey::ConnectorControl,
                payload: _feature.key(),
            })
        })
    }

    fn on_unregister(&self, _instance_id: InstanceId) {
        // No state per instance — nothing to do.
    }

    fn control_router(self: Arc<Self>) -> Router {
        routes(self)
    }

    fn public_router(self: Arc<Self>) -> Router {
        routes(self)
    }
}

fn routes(manager: Arc<ConnectorControlManager>) -> Router {
    Router::new()
        .route(protocol::paths::CONNECTOR_RESET, put(proxy_reset))
        .route(
            protocol::paths::CONNECTOR_REGISTER_LEADER,
            put(proxy_register_leader),
        )
        .route(protocol::paths::CONNECTOR_HEALTH, get(proxy_health))
        .with_state(manager)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn proxy_reset(
    State(mgr): State<Arc<ConnectorControlManager>>,
    Path(instance_id): Path<InstanceId>,
    body: Bytes,
) -> Response {
    proxy_unary(&mgr, instance_id, RESET_HANDLER, body).await
}

async fn proxy_register_leader(
    State(mgr): State<Arc<ConnectorControlManager>>,
    Path(instance_id): Path<InstanceId>,
    body: Bytes,
) -> Response {
    proxy_unary(&mgr, instance_id, REGISTER_LEADER_HANDLER, body).await
}

async fn proxy_health(
    State(mgr): State<Arc<ConnectorControlManager>>,
    Path(instance_id): Path<InstanceId>,
) -> Response {
    let Some(velo) = mgr.velo.get() else {
        return service_unavailable("hub has no velo transport configured");
    };
    let Some(registry) = mgr.registry.get() else {
        return service_unavailable("registry not attached");
    };
    if !registry.contains(instance_id) {
        return error_response(StatusCode::NOT_FOUND, "instance not registered");
    }
    let req = HeartbeatRequest { seq: 0 };
    let result: Result<HeartbeatAck, anyhow::Error> = async {
        let ack: HeartbeatAck = velo
            .typed_unary(HEARTBEAT_HANDLER)?
            .payload(&req)?
            .instance(instance_id)
            .send()
            .await?;
        Ok(ack)
    }
    .await;

    match result {
        Ok(ack) => json_response(
            StatusCode::OK,
            serde_json::json!({
                "velo_reachable": true,
                "ack_seq": ack.seq,
                "ack_ok": ack.ok,
            }),
        ),
        Err(e) => {
            tracing::warn!(instance = %instance_id, error = %e, "health probe failed");
            json_response(
                StatusCode::BAD_GATEWAY,
                serde_json::json!({
                    "velo_reachable": false,
                    "error": e.to_string(),
                }),
            )
        }
    }
}

async fn proxy_unary(
    mgr: &ConnectorControlManager,
    instance_id: InstanceId,
    handler: &'static str,
    body: Bytes,
) -> Response {
    let Some(velo) = mgr.velo.get() else {
        return service_unavailable("hub has no velo transport configured");
    };
    let Some(registry) = mgr.registry.get() else {
        return service_unavailable("registry not attached");
    };
    if !registry.contains(instance_id) {
        return error_response(StatusCode::NOT_FOUND, "instance not registered");
    }

    let payload = if body.is_empty() {
        // Connector handlers expect a JSON object even for "default"
        // cases. An empty HTTP body becomes `{}` so e.g.
        // `ResetRequest::default()` deserializes correctly.
        Bytes::from_static(b"{}")
    } else {
        body
    };

    let send = velo
        .unary(handler)
        .map(|b| b.raw_payload(payload).instance(instance_id));
    let send = match send {
        Ok(s) => s,
        Err(e) => return error_response(StatusCode::BAD_GATEWAY, &format!("velo build: {e}")),
    };

    let bytes = match send.send().await {
        Ok(b) => b,
        Err(e) => {
            // Velo transport itself failed — connector's HTTP handler
            // never had a chance to run. 502 = upstream unreachable.
            tracing::warn!(instance = %instance_id, %handler, error = %e, "velo proxy failed");
            return error_response(StatusCode::BAD_GATEWAY, &format!("velo proxy: {e}"));
        }
    };

    // The connector wraps every response in `ControlReply<T>`. Inspect
    // the envelope so the proxy returns the same HTTP status code that
    // the connector's local axum handler would have returned for the
    // same logical outcome.
    let reply: ControlReply<JsonValue> = match serde_json::from_slice(&bytes) {
        Ok(r) => r,
        Err(e) => {
            // Response wasn't a `ControlReply` envelope — pass through
            // as 502 with diagnostics. Indicates a connector/protocol
            // mismatch (older connector, or non-control handler hit).
            tracing::warn!(
                instance = %instance_id, %handler, error = %e,
                "velo response did not parse as ControlReply"
            );
            return error_response(
                StatusCode::BAD_GATEWAY,
                &format!("malformed connector reply: {e}"),
            );
        }
    };
    match reply {
        ControlReply::Ok(value) => {
            // Unwrap the envelope so the HTTP body is the inner shape
            // (matches what the connector's axum handler returns).
            let body_bytes = serde_json::to_vec(&value).unwrap_or_else(|_| b"{}".to_vec());
            raw_json_response(StatusCode::OK, Bytes::from(body_bytes))
        }
        ControlReply::Err(err) => {
            let status = StatusCode::from_u16(err.http_status())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            tracing::info!(
                instance = %instance_id, %handler, kind = err.kind(), %status,
                "proxy: connector returned control error"
            );
            json_response(
                status,
                serde_json::json!({
                    "error": err.to_string(),
                    "kind": err.kind(),
                }),
            )
        }
    }
}

fn service_unavailable(msg: &str) -> Response {
    error_response(StatusCode::SERVICE_UNAVAILABLE, msg)
}

fn error_response(status: StatusCode, msg: &str) -> Response {
    json_response(
        status,
        serde_json::json!({
            "error": msg,
        }),
    )
}

fn json_response(status: StatusCode, value: serde_json::Value) -> Response {
    let body = serde_json::to_vec(&value).unwrap_or_else(|_| b"{}".to_vec());
    raw_json_response(status, Bytes::from(body))
}

fn raw_json_response(status: StatusCode, bytes: Bytes) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    (status, headers, bytes).into_response()
}
