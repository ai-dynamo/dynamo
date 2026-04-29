// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local axum control HTTP server for ConnectorLeader.
//!
//! Off by default — gated on [`kvbm_config::ControlConfig::enabled`].
//! When disabled, control operations reach the connector via velo
//! handlers (see `velo_control.rs`) and the hub's HTTP→velo proxy.
//!
//! Each route here is a thin shim around [`super::ConnectorControlApi`];
//! the same trait backs the velo handlers, so the wire shape and
//! semantics are identical across transports.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::put,
};
use kvbm_config::ControlConfig;
use serde::Serialize;
use tokio::net::TcpListener;

use super::ConnectorLeader;
use super::control_api::{
    ConnectorControlApi, ControlError, RegisterLeaderRequest, RegisterLeaderResponse,
    ResetRequest, ResetResponse,
};

/// JSON error envelope returned to HTTP clients.
#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    kind: Option<&'static str>,
}

fn map_error(err: ControlError) -> (StatusCode, Json<ErrorBody>) {
    // Single source of truth for ControlError → HTTP status lives in
    // `kvbm_control_protocol::ControlError::http_status()` so the local
    // axum shim and the hub's HTTP→velo proxy always agree.
    let status = StatusCode::from_u16(err.http_status())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let kind = err.kind();
    (
        status,
        Json(ErrorBody {
            error: err.to_string(),
            kind: Some(kind),
        }),
    )
}

/// `PUT /reset` — body: [`ResetRequest`], response: [`ResetResponse`].
async fn reset_handler(
    State(leader): State<Arc<ConnectorLeader>>,
    Json(req): Json<ResetRequest>,
) -> impl IntoResponse {
    tracing::info!(?req.tiers, "reset request");
    match leader.reset(req).await {
        Ok(resp) => {
            // If any tier failed, surface as 500 with the partial response body.
            let status = if resp.failed.is_empty() {
                StatusCode::OK
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (status, Json(resp)).into_response()
        }
        Err(e) => map_error(e).into_response(),
    }
}

/// `PUT /register_leader` — body: [`RegisterLeaderRequest`],
/// response: [`RegisterLeaderResponse`].
async fn register_leader_handler(
    State(leader): State<Arc<ConnectorLeader>>,
    Json(req): Json<RegisterLeaderRequest>,
) -> impl IntoResponse {
    tracing::info!(instance_id = %req.instance_id, "register_leader request");
    match leader.register_leader(req).await {
        Ok(resp) => (StatusCode::OK, Json(resp)).into_response(),
        Err(e) => map_error(e).into_response(),
    }
}

/// Build the control router.
fn build_router(leader: Arc<ConnectorLeader>) -> Router {
    Router::new()
        .route("/reset", put(reset_handler))
        .route("/register_leader", put(register_leader_handler))
        .with_state(leader)
}

/// Start the local axum control server on `cfg.bind_addr:cfg.port`.
///
/// Returns a shutdown sender. Caller is expected to gate construction on
/// [`ControlConfig::enabled`] and store the sender for lifetime
/// management (typically via `OnceLock` on `ConnectorLeader`).
pub async fn start_control_server(
    leader: Arc<ConnectorLeader>,
    runtime_handle: tokio::runtime::Handle,
    cfg: &ControlConfig,
) -> anyhow::Result<oneshot::Sender<()>> {
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    let router = build_router(leader);
    let addr = format!("{}:{}", cfg.bind_addr, cfg.port);

    let listener = TcpListener::bind(&addr).await?;
    tracing::info!("Control server listening on {}", addr);

    runtime_handle.spawn(async move {
        let server = axum::serve(listener, router);

        tokio::select! {
            result = server => {
                if let Err(e) = result {
                    tracing::error!("Control server error: {}", e);
                }
            }
            _ = async {
                let _ = shutdown_rx.await;
            } => {
                tracing::info!("Control server shutting down");
            }
        }
    });

    Ok(shutdown_tx)
}
