// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP control service for ConnectorLeader.
//!
//! Provides endpoints for runtime control operations such as resetting block managers.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::put,
};
use serde::Serialize;
use tokio::net::TcpListener;

use super::ConnectorLeader;

/// Port for the control server.
const CONTROL_SERVER_PORT: u16 = 9999;

/// Logical type for reset operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalType {
    G2,
    G3,
    All,
}

impl LogicalType {
    /// Parse logical type from lowercase string.
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "g2" => Some(Self::G2),
            "g3" => Some(Self::G3),
            "all" => Some(Self::All),
            _ => None,
        }
    }
}

/// Response for successful operations.
#[derive(Debug, Serialize)]
struct SuccessResponse {
    status: &'static str,
}

/// Response for error operations.
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

/// Handler for PUT /reset/{logical_type}
async fn reset_handler(
    State(leader): State<Arc<ConnectorLeader>>,
    Path(logical_type): Path<String>,
) -> impl IntoResponse {
    tracing::info!(logical_type = %logical_type, "Received reset request");

    let logical_type = match LogicalType::from_str(&logical_type) {
        Some(lt) => lt,
        None => {
            tracing::error!(
                logical_type = %logical_type,
                "Invalid logical type in reset request"
            );
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!(
                        "Invalid logical type '{}'. Valid values: g2, g3, all",
                        logical_type
                    ),
                }),
            )
                .into_response();
        }
    };

    let instance_leader = match leader.instance_leader() {
        Some(il) => il,
        None => {
            tracing::error!("Reset request failed: InstanceLeader not initialized");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: "InstanceLeader not initialized".to_string(),
                }),
            )
                .into_response();
        }
    };

    let mut errors = Vec::new();

    // Reset G2 if requested
    if matches!(logical_type, LogicalType::G2 | LogicalType::All) {
        tracing::info!("Resetting G2 block manager");
        match instance_leader.g2_manager().reset_inactive_pool() {
            Ok(()) => tracing::info!("G2 block manager reset successfully"),
            Err(e) => {
                tracing::error!(error = %e, "G2 reset failed");
                errors.push(format!("G2 reset failed: {}", e));
            }
        }
    }

    // Reset G3 if requested
    if matches!(logical_type, LogicalType::G3 | LogicalType::All) {
        match instance_leader.g3_manager() {
            Some(g3) => {
                tracing::info!("Resetting G3 block manager");
                match g3.reset_inactive_pool() {
                    Ok(()) => tracing::info!("G3 block manager reset successfully"),
                    Err(e) => {
                        tracing::error!(error = %e, "G3 reset failed");
                        errors.push(format!("G3 reset failed: {}", e));
                    }
                }
            }
            None if logical_type == LogicalType::G3 => {
                tracing::error!("G3 reset requested but G3 manager not configured");
                return (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: "G3 manager not configured".to_string(),
                    }),
                )
                    .into_response();
            }
            None => {
                // G3 not configured but reset all was requested - that's fine, skip it
                tracing::info!("G3 manager not configured, skipping G3 reset");
            }
        }
    }

    if errors.is_empty() {
        tracing::info!(logical_type = ?logical_type, "Reset completed successfully");
        (StatusCode::OK, Json(SuccessResponse { status: "ok" })).into_response()
    } else {
        tracing::error!(errors = ?errors, "Reset completed with errors");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: errors.join("; "),
            }),
        )
            .into_response()
    }
}

/// Build the control router.
fn build_router(leader: Arc<ConnectorLeader>) -> Router {
    Router::new()
        .route("/reset/{logical_type}", put(reset_handler))
        .with_state(leader)
}

/// Start the control server.
///
/// This spawns a background task that listens on `0.0.0.0:9999` and serves control endpoints.
/// Returns a shutdown sender that can be used to gracefully stop the server.
pub async fn start_control_server(
    leader: Arc<ConnectorLeader>,
    runtime_handle: tokio::runtime::Handle,
) -> anyhow::Result<oneshot::Sender<()>> {
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    let router = build_router(leader);
    let addr = format!("0.0.0.0:{}", CONTROL_SERVER_PORT);

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
