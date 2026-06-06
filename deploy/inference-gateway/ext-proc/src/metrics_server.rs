// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal Prometheus `/metrics` endpoint for the EPP.
//!
//! The KV-router's per-worker load gauges (`worker_active_decode_blocks`,
//! `worker_active_prefill_tokens`) and the scheduler queue gauges are defined
//! and updated entirely in the shared library (`dynamo_llm::kv_router::metrics`)
//! — the very same statics the standalone frontend serves on its HTTP metrics
//! port (`lib/llm/src/http/service/service_v2.rs`). This module does not define
//! or duplicate any metric: it reuses the library's registration functions and
//! exposes the registry over HTTP, so an EPP replica's per-worker load view —
//! including load synced from peer replicas (ai-dynamo/dynamo#10384) — is
//! scrapeable exactly like the standalone router's.

use std::sync::Arc;

use anyhow::Result;
use axum::{Router, http::StatusCode, response::IntoResponse, routing::get};
use dynamo_llm::kv_router::metrics::{register_router_queue_metrics, register_worker_load_metrics};
use prometheus::{Encoder, Registry, TextEncoder};

/// Encode the registry in Prometheus text exposition format (same encode path
/// as `dynamo_runtime`'s metrics handler).
async fn metrics_handler(registry: Arc<Registry>) -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let mut buffer = Vec::new();
    if let Err(e) = encoder.encode(&registry.gather(), &mut buffer) {
        tracing::error!(error = %e, "failed to encode metrics");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "failed to encode metrics".to_string(),
        );
    }
    match String::from_utf8(buffer) {
        Ok(body) => (StatusCode::OK, body),
        Err(e) => {
            tracing::error!(error = %e, "metrics output was not valid UTF-8");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "invalid metrics encoding".to_string(),
            )
        }
    }
}

/// Bind an HTTP server on `0.0.0.0:port` serving `/metrics` with the KV-router
/// worker-load and scheduler-queue gauges. Reuses the library registration
/// functions so no metric definitions are duplicated here.
pub async fn spawn_metrics_server(port: u16) -> Result<()> {
    let registry = Registry::new();
    register_worker_load_metrics(&registry)?;
    register_router_queue_metrics(&registry)?;
    let registry = Arc::new(registry);

    let app = Router::new().route(
        "/metrics",
        get({
            let registry = Arc::clone(&registry);
            move || metrics_handler(Arc::clone(&registry))
        }),
    );

    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(address = %addr, "EPP metrics server listening on /metrics");
    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            tracing::error!(error = %e, "EPP metrics server exited");
        }
    });
    Ok(())
}
