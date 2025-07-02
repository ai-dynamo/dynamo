// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use axum::{body, http::StatusCode, response::IntoResponse, routing::get, Router};
use prometheus::{
    proto, register_gauge_with_registry, Encoder, Gauge, Opts, Registry, TextEncoder,
};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tracing;

/// Runtime metrics for HTTP server
pub struct RuntimeMetrics {
    uptime_gauge: Gauge,
}

impl RuntimeMetrics {
    pub fn new(metrics_registry: &Arc<Registry>) -> anyhow::Result<Arc<Self>> {
        let uptime_opts = Opts::new(
            "uptime_seconds",
            "Total uptime of the DistributedRuntime in seconds",
        )
        .namespace("dynamo")
        .subsystem("runtime");

        let uptime_gauge = register_gauge_with_registry!(uptime_opts, metrics_registry)?;

        Ok(Arc::new(Self { uptime_gauge }))
    }

    pub fn update_uptime(&self, uptime_seconds: f64) {
        self.uptime_gauge.set(uptime_seconds);
    }
}

/// HTTP server state containing pre-created metrics
pub struct HttpServerState {
    drt: Arc<crate::DistributedRuntime>,
    registry: Arc<Registry>,
    runtime_metrics: Arc<RuntimeMetrics>,
}

impl HttpServerState {
    /// Create new HTTP server state with pre-created metrics
    pub fn new(drt: Arc<crate::DistributedRuntime>) -> anyhow::Result<Self> {
        let registry = Arc::new(Registry::new());

        // Create runtime metrics
        let runtime_metrics = RuntimeMetrics::new(&registry)?;

        Ok(Self {
            drt,
            registry,
            runtime_metrics,
        })
    }
}

/// Start HTTP server with DistributedRuntime support
pub async fn start_http_server(
    host: &str,
    port: u16,
    cancel_token: CancellationToken,
    drt: Arc<crate::DistributedRuntime>,
) -> anyhow::Result<()> {
    // Create HTTP server state with pre-created metrics
    let server_state = Arc::new(HttpServerState::new(drt)?);

    let app = Router::new()
        // .route(
        //     "/health",
        //     get({
        //         let state = Arc::clone(&server_state);
        //         move || health_handler(state)
        //     }),
        // )
        .route(
            "/metrics",
            get({
                let state = Arc::clone(&server_state);
                move || metrics_handler(state)
            }),
        );

    let address = format!("{}:{}", host, port);
    tracing::debug!("Starting HTTP server on: {}", address);

    let listener = match TcpListener::bind(&address).await {
        Ok(listener) => {
            // get the actual address and port, print in debug level
            let actual_address = listener.local_addr()?;
            tracing::debug!("HTTP server bound to: {}", actual_address);
            listener
        }
        Err(e) => {
            tracing::error!("Failed to bind to address {}: {}", address, e);
            return Err(anyhow::anyhow!("Failed to bind to address: {}", e));
        }
    };

    let observer = cancel_token.child_token();
    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(observer.cancelled_owned())
        .await
    {
        tracing::error!("HTTP server error: {}", e);
    }
    Ok(())
}

// /// Health handler
// async fn health_handler(state: Arc<HttpServerState>) -> impl IntoResponse {
//     let uptime = state.drt.uptime();
//     let response = format!("OK\nUptime: {} seconds", uptime.as_secs());
//     (StatusCode::OK, response)
// }

/// Metrics handler with DistributedRuntime uptime
async fn metrics_handler(state: Arc<HttpServerState>) -> impl IntoResponse {
    // Update the uptime gauge with current value
    let uptime_seconds = state.drt.uptime().as_secs_f64();
    state.runtime_metrics.update_uptime(uptime_seconds);

    // Gather metrics from the registry
    let metric_families = state.registry.gather();

    let encoder = TextEncoder::new();
    let mut buffer = Vec::new();

    match encoder.encode(&metric_families, &mut buffer) {
        Ok(()) => match String::from_utf8(buffer) {
            Ok(response) => (StatusCode::OK, response),
            Err(e) => {
                tracing::error!("Failed to encode metrics as UTF-8: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to encode metrics as UTF-8".to_string(),
                )
            }
        },
        Err(e) => {
            tracing::error!("Failed to encode metrics: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to encode metrics".to_string(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_http_server_lifecycle() {
        let cancel_token = CancellationToken::new();
        let cancel_token_for_server = cancel_token.clone();
        let runtime = crate::Runtime::from_current().unwrap();
        // Use from_settings_without_discovery to avoid nested HTTP server startup
        let drt = crate::DistributedRuntime::from_settings_without_discovery(runtime)
            .await
            .unwrap();

        // start HTTP server
        let server_handle = tokio::spawn(async move {
            let _ = start_http_server("127.0.0.1", 0, cancel_token_for_server, Arc::new(drt)).await;
        });

        // wait for a while to let the server start
        sleep(Duration::from_millis(100)).await;

        // cancel token
        cancel_token.cancel();

        // wait for the server to shut down
        let result = tokio::time::timeout(Duration::from_secs(5), server_handle).await;
        assert!(
            result.is_ok(),
            "HTTP server should shut down when cancel token is cancelled"
        );
    }

    // #[tokio::test]
    // async fn test_health_handler() {
    //     let runtime = crate::Runtime::single_threaded().unwrap();
    //     let drt = crate::DistributedRuntime::from_settings(runtime)
    //         .await
    //         .unwrap();

    //     // Wait a bit to ensure uptime is measurable
    //     tokio::time::sleep(Duration::from_millis(10)).await;

    //     let response = health_handler(Arc::new(drt)).await;
    //     let response = response.into_response();
    //     let (parts, body) = response.into_parts();

    //     assert_eq!(parts.status, StatusCode::OK);

    //     // Check that the response contains uptime information
    //     let body_bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    //     let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
    //     assert!(body_str.contains("OK"));
    //     assert!(body_str.contains("Uptime:"));
    // }

    #[tokio::test]
    async fn test_metrics_handler() {
        let runtime = crate::Runtime::from_current().unwrap();
        // Use from_settings_without_discovery to avoid starting HTTP server in test
        let drt = crate::DistributedRuntime::from_settings_without_discovery(runtime)
            .await
            .unwrap();

        // Wait a bit to ensure uptime is measurable
        tokio::time::sleep(Duration::from_millis(10)).await;

        let server_state = HttpServerState::new(Arc::new(drt)).unwrap();
        let response = metrics_handler(Arc::new(server_state)).await;
        let response = response.into_response();
        let (parts, body) = response.into_parts();

        assert_eq!(parts.status, StatusCode::OK);

        // Check that the response contains the uptime metric
        let body_bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
        assert!(body_str.contains("dynamo_runtime_uptime_seconds"));
    }
}
