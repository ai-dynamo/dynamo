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
pub async fn spawn_http_server(
    host: &str,
    port: u16,
    cancel_token: CancellationToken,
    drt: Arc<crate::DistributedRuntime>,
) -> anyhow::Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    tracing::info!(
        "[spawn_http_server] called with host={}, port={}",
        host,
        port
    );
    // Create HTTP server state with pre-created metrics
    let server_state = Arc::new(HttpServerState::new(drt)?);

    let app = Router::new()
        .route(
            "/health",
            get({
                let state = Arc::clone(&server_state);
                move || health_handler(state.clone())
            }),
        )
        .route(
            "/live",
            get({
                let state = Arc::clone(&server_state);
                move || health_handler(state)
            }),
        )
        .route(
            "/metrics",
            get({
                let state = Arc::clone(&server_state);
                move || metrics_handler(state)
            }),
        )
        .fallback(|| async {
            tracing::info!("[fallback handler] called");
            (StatusCode::NOT_FOUND, "Route not found").into_response()
        });

    let address = format!("{}:{}", host, port);
    tracing::info!("[spawn_http_server] binding to: {}", address);

    let listener = match TcpListener::bind(&address).await {
        Ok(listener) => {
            // get the actual address and port, print in debug level
            let actual_address = listener.local_addr()?;
            tracing::info!(
                "[spawn_http_server] HTTP server bound to: {}",
                actual_address
            );
            (listener, actual_address)
        }
        Err(e) => {
            tracing::error!("Failed to bind to address {}: {}", address, e);
            return Err(anyhow::anyhow!("Failed to bind to address: {}", e));
        }
    };
    let (listener, actual_address) = listener;

    let observer = cancel_token.child_token();
    // Spawn the server in the background and return the handle
    let handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(observer.cancelled_owned())
            .await
        {
            tracing::error!("HTTP server error: {}", e);
        }
    });
    Ok((actual_address, handle))
}

/// Health handler
async fn health_handler(state: Arc<HttpServerState>) -> impl IntoResponse {
    tracing::info!("[health_handler] called");
    let uptime = state.drt.uptime();
    let response = format!("OK\nUptime: {} seconds\n", uptime.as_secs());
    (StatusCode::OK, response)
}

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

        // Test basic HTTP server lifecycle without DistributedRuntime
        let app = Router::new().route("/test", get(|| async { (StatusCode::OK, "test") }));

        // start HTTP server
        let server_handle = tokio::spawn(async move {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(cancel_token_for_server.cancelled_owned())
                .await;
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

    #[tokio::test]
    async fn test_runtime_metrics_creation() {
        // Test RuntimeMetrics creation and functionality
        let registry = Arc::new(Registry::new());
        let runtime_metrics = RuntimeMetrics::new(&registry).unwrap();

        // Wait a bit to ensure uptime is measurable
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Test updating uptime
        let uptime_seconds = 123.456;
        runtime_metrics.update_uptime(uptime_seconds);

        // Gather metrics from the registry
        let metric_families = registry.gather();

        let encoder = TextEncoder::new();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();

        let response = String::from_utf8(buffer).unwrap();
        assert!(response.contains("dynamo_runtime_uptime_seconds"));
        assert!(response.contains("123.456"));
    }

    #[tokio::test]
    async fn test_runtime_metrics_namespace() {
        // Test that metrics have correct namespace
        let registry = Arc::new(Registry::new());
        let runtime_metrics = RuntimeMetrics::new(&registry).unwrap();

        runtime_metrics.update_uptime(42.0);

        let metric_families = registry.gather();
        let encoder = TextEncoder::new();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();

        let response = String::from_utf8(buffer).unwrap();
        // Check for the full metric name with namespace and subsystem
        assert!(response.contains("dynamo_runtime_uptime_seconds"));
        assert!(response.contains("Total uptime of the DistributedRuntime in seconds"));
    }

    /*
    #[tokio::test]
    async fn test_spawn_http_server_endpoints() {
        use std::sync::Arc;
        use tokio::time::sleep;
        use tokio_util::sync::CancellationToken;
        // use tokio::io::{AsyncReadExt, AsyncWriteExt};
        // use reqwest for HTTP requests
        let runtime = crate::Runtime::from_settings().unwrap();
        let drt = Arc::new(
            crate::DistributedRuntime::from_settings_without_discovery(runtime)
                .await
                .unwrap(),
        );
        let cancel_token = CancellationToken::new();
        let (addr, server_handle) = spawn_http_server("127.0.0.1", 0, cancel_token.clone(), drt)
            .await
            .unwrap();
        println!("[test] Waiting for server to start...");
        sleep(std::time::Duration::from_millis(1000)).await;
        println!("[test] Server should be up, starting requests...");
        let client = reqwest::Client::new();
        for (path, expect_200, expect_body) in [
            ("/health", true, "OK"),
            ("/live", true, "OK"),
            ("/someRandomPathNotFoundHere", false, "Route not found"),
        ] {
            println!("[test] Sending request to {}", path);
            let url = format!("http://{}{}", addr, path);
            let response = client.get(&url).send().await.unwrap();
            let status = response.status();
            let body = response.text().await.unwrap();
            println!(
                "[test] Response for {}: status={}, body={:?}",
                path, status, body
            );
            if expect_200 {
                assert_eq!(status, 200, "Response: status={}, body={:?}", status, body);
            } else {
                assert_eq!(status, 404, "Response: status={}, body={:?}", status, body);
            }
            assert!(
                body.contains(expect_body),
                "Response: status={}, body={:?}",
                status,
                body
            );
        }
        cancel_token.cancel();
        match server_handle.await {
            Ok(_) => println!("[test] Server shut down normally"),
            Err(e) => {
                if e.is_panic() {
                    println!("[test] Server panicked: {:?}", e);
                } else {
                    println!("[test] Server cancelled: {:?}", e);
                }
            }
        }
    }
    */
}
