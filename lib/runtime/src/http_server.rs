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

use crate::component::Namespace;
use crate::profiling::MetricsRegistry;
use axum::{body, http::StatusCode, response::IntoResponse, routing::get, Router};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Instant;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tracing;

/// HTTP server state containing metrics and uptime tracking
pub struct HttpServerState {
    registry: Arc<dyn MetricsRegistry>,
    namespace: Option<Namespace>,
    start_time: OnceLock<Instant>,
    uptime_gauge: Arc<prometheus::Gauge>,
}

impl HttpServerState {
    /// Create new HTTP server state with the provided metrics registry
    pub fn new(registry: Arc<dyn MetricsRegistry>) -> anyhow::Result<Self> {
        let uptime_gauge = match registry.create_gauge(
            "uptime_seconds",
            "Total uptime of the DistributedRuntime in seconds",
            &[("service", "dynamo"), ("subsystem", "runtime")],
        ) {
            Ok(gauge) => gauge,
            Err(e) => return Err(anyhow::anyhow!("Failed to create uptime gauge: {}", e)),
        };
        Ok(Self {
            namespace: None,
            start_time: OnceLock::new(),
            registry,
            uptime_gauge,
        })
    }

    /// Initialize the start time (can only be called once)
    pub fn initialize_start_time(&self) -> Result<(), &'static str> {
        self.start_time
            .set(Instant::now())
            .map_err(|_| "Start time already initialized")
    }

    pub fn uptime(&self) -> std::time::Duration {
        self.start_time
            .get()
            .expect("Start time not initialized")
            .elapsed()
    }

    /// Get a reference to the metrics registry
    pub fn get_registry(&self) -> &Arc<dyn MetricsRegistry> {
        &self.registry
    }

    /// Update the uptime gauge with current value
    pub fn update_uptime_gauge(&self) {
        let uptime_seconds = self.uptime().as_secs_f64();
        self.uptime_gauge.set(uptime_seconds);
    }

    /// Set the namespace for this HTTP server state
    pub fn set_namespace(&mut self, namespace: Namespace) {
        self.namespace = Some(namespace);
    }
}

/// Start HTTP server with metrics support
pub async fn spawn_http_server(
    host: &str,
    port: u16,
    cancel_token: CancellationToken,
    metrics_registry: Arc<dyn MetricsRegistry>,
) -> anyhow::Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    // Create HTTP server state with the provided metrics registry
    let server_state = Arc::new(HttpServerState::new(metrics_registry)?);

    // // Initialize the start time
    server_state
        .initialize_start_time()
        .map_err(|e| anyhow::anyhow!("Failed to initialize start time: {}", e))?;

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
    let uptime = state.uptime();
    let response = format!("OK\nUptime: {} seconds\n", uptime.as_secs());
    //let health = state.DB_errors <= 0;
    (StatusCode::OK, response)
}

/// Metrics handler with DistributedRuntime uptime
async fn metrics_handler(state: Arc<HttpServerState>) -> impl IntoResponse {
    // Update the uptime gauge with current value
    state.update_uptime_gauge();

    // Get metrics from the registry
    match state.get_registry().encode_prometheus_str() {
        Ok(response) => (StatusCode::OK, response),
        Err(e) => {
            tracing::error!("Failed to get metrics from registry: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to get metrics".to_string(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiling::MetricsRegistry;
    use tokio::time::{sleep, Duration};

    // Test wrapper that implements MetricsRegistry for prometheus::Registry
    #[cfg(test)]
    struct TestMetricsRegistry {
        registry: Arc<prometheus::Registry>,
        prefix: String,
    }

    #[cfg(test)]
    impl TestMetricsRegistry {
        fn new(registry: Arc<prometheus::Registry>) -> Self {
            Self {
                registry,
                prefix: "namespace".to_string(),
            }
        }
    }

    #[cfg(test)]
    impl MetricsRegistry for TestMetricsRegistry {
        fn metrics_prefix(&self) -> String {
            self.prefix.clone()
        }

        fn metrics_hierarchy(&self) -> Vec<String> {
            vec![self.prefix.clone()]
        }

        fn drt(&self) -> &crate::DistributedRuntime {
            panic!("drt() not implemented for TestMetricsRegistry")
        }

        fn registry(&self) -> prometheus::Registry {
            self.registry.as_ref().clone()
        }
    }

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
        let prometheus_registry = Arc::new(prometheus::Registry::new());
        let registry = Arc::new(TestMetricsRegistry::new(prometheus_registry));
        let runtime_metrics = HttpServerState::new(registry).unwrap();

        // Initialize start time
        runtime_metrics.initialize_start_time().unwrap();

        // Wait a bit to ensure uptime is measurable
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Test updating uptime
        let uptime_seconds = 123.456;
        runtime_metrics.uptime_gauge.set(uptime_seconds);

        // Get metrics from the registry
        let response = runtime_metrics.registry.encode_prometheus_str().unwrap();
        println!("Full metrics response:\n{}", response);

        let expected = "\
# HELP namespace__uptime_seconds Total uptime of the DistributedRuntime in seconds
# TYPE namespace__uptime_seconds gauge
namespace__uptime_seconds 123.456
";
        assert_eq!(response, expected);
    }

    #[tokio::test]
    async fn test_runtime_metrics_namespace() {
        // Test that metrics have correct namespace
        let prometheus_registry = Arc::new(prometheus::Registry::new());
        let registry = Arc::new(TestMetricsRegistry::new(prometheus_registry));
        let runtime_metrics = HttpServerState::new(registry).unwrap();

        // Initialize start time
        runtime_metrics.initialize_start_time().unwrap();

        runtime_metrics.uptime_gauge.set(42.0);

        let response = runtime_metrics.registry.encode_prometheus_str().unwrap();
        println!("Full metrics response:\n{}", response);

        let expected = "\
# HELP namespace__uptime_seconds Total uptime of the DistributedRuntime in seconds
# TYPE namespace__uptime_seconds gauge
namespace__uptime_seconds 42
";
        assert_eq!(response, expected);
    }

    #[tokio::test]
    async fn test_start_time_initialization() {
        // Test that start time can only be initialized once
        let prometheus_registry = Arc::new(prometheus::Registry::new());
        let registry = Arc::new(TestMetricsRegistry::new(prometheus_registry));
        let runtime_metrics = HttpServerState::new(registry).unwrap();

        // First initialization should succeed
        assert!(runtime_metrics.initialize_start_time().is_ok());

        // Second initialization should fail
        assert!(runtime_metrics.initialize_start_time().is_err());

        // Uptime should work after initialization
        let _uptime = runtime_metrics.uptime();
        // If we get here, uptime calculation works correctly
    }

    #[tokio::test]
    #[should_panic(expected = "Start time not initialized")]
    async fn test_uptime_without_initialization() {
        // Test that uptime panics if start time is not initialized
        let prometheus_registry = Arc::new(prometheus::Registry::new());
        let registry = Arc::new(TestMetricsRegistry::new(prometheus_registry));
        let runtime_metrics = HttpServerState::new(registry).unwrap();

        // This should panic because start time is not initialized
        let _uptime = runtime_metrics.uptime();
    }

    #[tokio::test]
    async fn test_spawn_http_server_endpoints() {
        use std::sync::Arc;
        use tokio::time::sleep;
        use tokio_util::sync::CancellationToken;
        // use reqwest for HTTP requests
        let cancel_token = CancellationToken::new();
        let prometheus_registry = Arc::new(prometheus::Registry::new());
        let metrics_registry = Arc::new(TestMetricsRegistry::new(prometheus_registry));
        let (addr, server_handle) =
            spawn_http_server("127.0.0.1", 0, cancel_token.clone(), metrics_registry)
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
}
