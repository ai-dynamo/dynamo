// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP metrics server exposing `/metrics` in Prometheus text format.
//!
//! The primary API is [`start_metrics_server`] which accepts a port and registry.
//! For backward compatibility with v1 env vars, [`start_metrics_server_from_env`]
//! reads `DYN_KVBM_METRICS` / `DYN_KVBM_METRICS_PORT`.

use std::sync::Arc;

use axum::Router;
use axum::routing::get;
use prometheus::Registry;
use tracing::{info, warn};

const DEFAULT_PORT: u16 = 6880;

/// Start the KVBM metrics HTTP server on the given port.
///
/// Returns a `JoinHandle` for the server task. The server binds to `0.0.0.0:{port}`
/// and serves Prometheus text format at `/metrics`.
pub fn start_metrics_server(registry: Registry, port: u16) -> tokio::task::JoinHandle<()> {
    let registry = Arc::new(registry);

    let app = Router::new().route(
        "/metrics",
        get({
            let registry = Arc::clone(&registry);
            move || {
                let registry = Arc::clone(&registry);
                async move {
                    let encoder = prometheus::TextEncoder::new();
                    let metric_families = registry.gather();
                    match encoder.encode_to_string(&metric_families) {
                        Ok(body) => (
                            axum::http::StatusCode::OK,
                            [(
                                axum::http::header::CONTENT_TYPE,
                                "text/plain; version=0.0.4; charset=utf-8",
                            )],
                            body,
                        ),
                        Err(e) => (
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            [(
                                axum::http::header::CONTENT_TYPE,
                                "text/plain; charset=utf-8",
                            )],
                            format!("Failed to encode metrics: {e}"),
                        ),
                    }
                }
            }
        }),
    );

    tokio::spawn(async move {
        let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
        info!("KVBM metrics server listening on {addr}");
        let listener = match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => l,
            Err(e) => {
                warn!("Failed to bind KVBM metrics server to {addr}: {e}");
                return;
            }
        };
        if let Err(e) = axum::serve(listener, app).await {
            warn!("KVBM metrics server exited with error: {e}");
        }
    })
}

/// Start the KVBM metrics HTTP server if enabled via environment variables.
///
/// Reads `DYN_KVBM_METRICS` (set to `"1"` or `"true"` to enable) and
/// `DYN_KVBM_METRICS_PORT` (default `6880`). This is the v1-compatible entrypoint.
///
/// Returns `Some(JoinHandle)` if started, `None` if disabled.
pub fn start_metrics_server_from_env(registry: Registry) -> Option<tokio::task::JoinHandle<()>> {
    let enabled = std::env::var("DYN_KVBM_METRICS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if !enabled {
        info!("KVBM metrics server disabled (set DYN_KVBM_METRICS=1 to enable)");
        return None;
    }

    let port = std::env::var("DYN_KVBM_METRICS_PORT")
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(DEFAULT_PORT);

    Some(start_metrics_server(registry, port))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_server_disabled_by_default() {
        temp_env::with_vars_unset(vec!["DYN_KVBM_METRICS", "DYN_KVBM_METRICS_PORT"], || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let result = rt.block_on(async { start_metrics_server_from_env(Registry::new()) });
            assert!(result.is_none());
        });
    }
}
