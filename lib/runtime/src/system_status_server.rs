// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::config::HealthStatus;
use crate::logging::make_request_span;
use crate::metrics::MetricsRegistry;
use crate::metrics::prometheus_names::{nats_client, nats_service};
use crate::traits::DistributedRuntimeProvider;
use axum::{Router, http::StatusCode, response::IntoResponse, routing::get};
use serde_json::json;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use tokio::{net::TcpListener, task::JoinHandle};
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;

/// System status server information containing socket address and handle
#[derive(Debug)]
pub struct SystemStatusServerInfo {
    pub socket_addr: std::net::SocketAddr,
    pub handle: Option<Arc<JoinHandle<()>>>,
}

impl SystemStatusServerInfo {
    pub fn new(socket_addr: std::net::SocketAddr, handle: Option<JoinHandle<()>>) -> Self {
        Self {
            socket_addr,
            handle: handle.map(Arc::new),
        }
    }

    pub fn address(&self) -> String {
        self.socket_addr.to_string()
    }

    pub fn hostname(&self) -> String {
        self.socket_addr.ip().to_string()
    }

    pub fn port(&self) -> u16 {
        self.socket_addr.port()
    }
}

impl Clone for SystemStatusServerInfo {
    fn clone(&self) -> Self {
        Self {
            socket_addr: self.socket_addr,
            handle: self.handle.clone(),
        }
    }
}

/// System status server state containing the distributed runtime reference
pub struct SystemStatusState {
    // global drt registry is for printing out the entire Prometheus format output
    root_drt: Arc<crate::DistributedRuntime>,
}

impl SystemStatusState {
    /// Create new system status server state with the provided distributed runtime
    pub fn new(drt: Arc<crate::DistributedRuntime>) -> anyhow::Result<Self> {
        Ok(Self { root_drt: drt })
    }

    /// Get a reference to the distributed runtime
    pub fn drt(&self) -> &crate::DistributedRuntime {
        &self.root_drt
    }
}

/// Start system status server with metrics support
pub async fn spawn_system_status_server(
    host: &str,
    port: u16,
    cancel_token: CancellationToken,
    drt: Arc<crate::DistributedRuntime>,
) -> anyhow::Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    // Create system status server state with the provided distributed runtime
    let server_state = Arc::new(SystemStatusState::new(drt)?);
    let health_path = server_state
        .drt()
        .system_health
        .lock()
        .unwrap()
        .health_path
        .clone();
    let live_path = server_state
        .drt()
        .system_health
        .lock()
        .unwrap()
        .live_path
        .clone();

    let app = Router::new()
        .route(
            &health_path,
            get({
                let state = Arc::clone(&server_state);
                move || health_handler(state)
            }),
        )
        .route(
            &live_path,
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
        })
        .layer(TraceLayer::new_for_http().make_span_with(make_request_span));

    let address = format!("{}:{}", host, port);
    tracing::info!("[spawn_system_status_server] binding to: {}", address);

    let listener = match TcpListener::bind(&address).await {
        Ok(listener) => {
            // get the actual address and port, print in debug level
            let actual_address = listener.local_addr()?;
            tracing::info!(
                "[spawn_system_status_server] system status server bound to: {}",
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
            tracing::error!("System status server error: {}", e);
        }
    });

    Ok((actual_address, handle))
}

/// Health handler
#[tracing::instrument(skip_all, level = "trace")]
async fn health_handler(state: Arc<SystemStatusState>) -> impl IntoResponse {
    let system_health = state.drt().system_health.lock().unwrap();
    let (healthy, endpoints) = system_health.get_health_status();
    let uptime = Some(system_health.uptime());

    let healthy_string = if healthy { "ready" } else { "notready" };
    let status_code = if healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    let response = json!({
        "status": healthy_string,
        "uptime": uptime,
        "endpoints": endpoints
    });

    tracing::trace!("Response {}", response.to_string());

    (status_code, response.to_string())
}

/// Metrics handler with DistributedRuntime uptime
#[tracing::instrument(skip_all, level = "trace")]
async fn metrics_handler(state: Arc<SystemStatusState>) -> impl IntoResponse {
    // Update the uptime gauge with current value
    state
        .drt()
        .system_health
        .lock()
        .unwrap()
        .update_uptime_gauge();

    // Execute all the callbacks starting at the DistributedRuntime level
    assert!(state.drt().basename() == "");
    let callback_results = state
        .drt()
        .execute_metrics_callbacks(&state.drt().hierarchy());
    for result in callback_results {
        if let Err(e) = result {
            tracing::error!("Error executing metrics callback: {}", e);
        }
    }

    // Get all metrics from DistributedRuntime (top-level)
    match state.drt().prometheus_metrics_fmt() {
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

// Regular tests: cargo test system_status_server --lib
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::Duration;

    // This is a basic test to verify the HTTP server is working before testing other more complicated tests
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

        // server starts immediately, no need to wait

        // cancel token
        cancel_token.cancel();

        // wait for the server to shut down
        let result = tokio::time::timeout(Duration::from_secs(5), server_handle).await;
        assert!(
            result.is_ok(),
            "HTTP server should shut down when cancel token is cancelled"
        );
    }
}

// Integration tests: cargo test system_status_server --lib --features integration
#[cfg(all(test, feature = "integration"))]
mod integration_tests {
    use super::*;
    use crate::distributed::distributed_test_utils::create_test_drt_async;
    use crate::metrics::MetricsRegistry;
    use anyhow::Result;
    use rstest::rstest;
    use std::sync::Arc;
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_uptime_from_system_health() {
        // Test that uptime is available from SystemHealth
        temp_env::async_with_vars([("DYN_SYSTEM_ENABLED", Some("false"))], async {
            let drt = create_test_drt_async().await;

            // Get uptime from SystemHealth
            let uptime = drt.system_health.lock().unwrap().uptime();
            // Uptime should exist (even if close to zero)
            assert!(uptime.as_nanos() > 0 || uptime.is_zero());

            // Sleep briefly and check uptime increases
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            let uptime_after = drt.system_health.lock().unwrap().uptime();
            assert!(uptime_after > uptime);
        })
        .await;
    }

    #[tokio::test]
    async fn test_runtime_metrics_initialization_and_namespace() {
        // Test that metrics have correct namespace
        temp_env::async_with_vars([("DYN_SYSTEM_ENABLED", Some("false"))], async {
            let drt = create_test_drt_async().await;
            // SystemStatusState is already created in distributed.rs when DYN_SYSTEM_ENABLED=false
            // so we don't need to create it again here

            // The uptime_seconds metric should already be registered and available
            let response = drt.prometheus_metrics_fmt().unwrap();
            println!("Full metrics response:\n{}", response);

            // Filter out NATS client metrics for comparison
            let filtered_response: String = response
                .lines()
                .filter(|line| {
                    !line.contains(nats_client::PREFIX) && !line.contains(nats_service::PREFIX)
                })
                .collect::<Vec<_>>()
                .join("\n");

            // Check that uptime_seconds metric is present with correct namespace
            assert!(
                filtered_response.contains("# HELP dynamo_component_uptime_seconds"),
                "Should contain uptime_seconds help text"
            );
            assert!(
                filtered_response.contains("# TYPE dynamo_component_uptime_seconds gauge"),
                "Should contain uptime_seconds type"
            );
            assert!(
                filtered_response.contains("dynamo_component_uptime_seconds"),
                "Should contain uptime_seconds metric with correct namespace"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn test_uptime_gauge_updates() {
        // Test that the uptime gauge is properly updated and increases over time
        temp_env::async_with_vars([("DYN_SYSTEM_ENABLED", Some("false"))], async {
            let drt = create_test_drt_async().await;

            // Get initial uptime
            let initial_uptime = drt.system_health.lock().unwrap().uptime();

            // Update the gauge with initial value
            drt.system_health.lock().unwrap().update_uptime_gauge();

            // Sleep for 100ms
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;

            // Get uptime after sleep
            let uptime_after_sleep = drt.system_health.lock().unwrap().uptime();

            // Update the gauge again
            drt.system_health.lock().unwrap().update_uptime_gauge();

            // Verify uptime increased by at least 100ms
            let elapsed = uptime_after_sleep - initial_uptime;
            assert!(
                elapsed >= std::time::Duration::from_millis(100),
                "Uptime should have increased by at least 100ms after sleep, but only increased by {:?}",
                elapsed
            );
        })
        .await;
    }

    #[tokio::test]
    async fn test_http_requests_fail_when_system_disabled() {
        // Test that system status server is not running when disabled
        temp_env::async_with_vars([("DYN_SYSTEM_ENABLED", Some("false"))], async {
            let drt = create_test_drt_async().await;

            // Verify that system status server info is None when disabled
            let system_info = drt.system_status_server_info();
            assert!(
                system_info.is_none(),
                "System status server should not be running when DYN_SYSTEM_ENABLED=false"
            );

            println!("✓ System status server correctly disabled when DYN_SYSTEM_ENABLED=false");
        })
        .await;
    }

    /// This test verifies the health and liveness endpoints of the system status server.
    /// It checks that the endpoints respond with the correct HTTP status codes and bodies
    /// based on the initial health status and any custom endpoint paths provided via environment variables.
    /// The test is parameterized using multiple #[case] attributes to cover various scenarios,
    /// including different initial health states ("ready" and "notready"), default and custom endpoint paths,
    /// and expected response codes and bodies.
    #[rstest]
    #[case("ready", 200, "ready", None, None, 3)]
    #[case("notready", 503, "notready", None, None, 3)]
    #[case("ready", 200, "ready", Some("/custom/health"), Some("/custom/live"), 5)]
    #[case(
        "notready",
        503,
        "notready",
        Some("/custom/health"),
        Some("/custom/live"),
        5
    )]
    #[tokio::test]
    #[cfg(feature = "integration")]
    async fn test_health_endpoints(
        #[case] starting_health_status: &'static str,
        #[case] expected_status: u16,
        #[case] expected_body: &'static str,
        #[case] custom_health_path: Option<&'static str>,
        #[case] custom_live_path: Option<&'static str>,
        #[case] expected_num_tests: usize,
    ) {
        use std::sync::Arc;
        // use tokio::io::{AsyncReadExt, AsyncWriteExt};
        // use reqwest for HTTP requests

        // Closure call is needed here to satisfy async_with_vars

        crate::logging::init();

        #[allow(clippy::redundant_closure_call)]
        temp_env::async_with_vars(
            [
                ("DYN_SYSTEM_ENABLED", Some("true")),
                ("DYN_SYSTEM_PORT", Some("0")),
                (
                    "DYN_SYSTEM_STARTING_HEALTH_STATUS",
                    Some(starting_health_status),
                ),
                ("DYN_SYSTEM_HEALTH_PATH", custom_health_path),
                ("DYN_SYSTEM_LIVE_PATH", custom_live_path),
            ],
            (async || {
                let drt = Arc::new(create_test_drt_async().await);

                // Get system status server info from DRT (instead of manually spawning)
                let system_info = drt
                    .system_status_server_info()
                    .expect("System status server should be started by DRT");
                let addr = system_info.socket_addr;

                let client = reqwest::Client::new();

                // Prepare test cases
                let mut test_cases = vec![];
                match custom_health_path {
                    None => {
                        // When using default paths, test the default paths
                        test_cases.push(("/health", expected_status, expected_body));
                    }
                    Some(chp) => {
                        // When using custom paths, default paths should not exist
                        test_cases.push(("/health", 404, "Route not found"));
                        test_cases.push((chp, expected_status, expected_body));
                    }
                }
                match custom_live_path {
                    None => {
                        // When using default paths, test the default paths
                        test_cases.push(("/live", expected_status, expected_body));
                    }
                    Some(clp) => {
                        // When using custom paths, default paths should not exist
                        test_cases.push(("/live", 404, "Route not found"));
                        test_cases.push((clp, expected_status, expected_body));
                    }
                }
                test_cases.push(("/someRandomPathNotFoundHere", 404, "Route not found"));
                assert_eq!(test_cases.len(), expected_num_tests);

                for (path, expect_status, expect_body) in test_cases {
                    println!("[test] Sending request to {}", path);
                    let url = format!("http://{}{}", addr, path);
                    let response = client.get(&url).send().await.unwrap();
                    let status = response.status();
                    let body = response.text().await.unwrap();
                    println!(
                        "[test] Response for {}: status={}, body={:?}",
                        path, status, body
                    );
                    assert_eq!(
                        status, expect_status,
                        "Response: status={}, body={:?}",
                        status, body
                    );
                    assert!(
                        body.contains(expect_body),
                        "Response: status={}, body={:?}",
                        status,
                        body
                    );
                }
            })(),
        )
        .await;
    }

    #[tokio::test]
    async fn test_health_endpoint_tracing() -> Result<()> {
        use std::sync::Arc;

        // Closure call is needed here to satisfy async_with_vars

        #[allow(clippy::redundant_closure_call)]
        let _ = temp_env::async_with_vars(
            [
                ("DYN_SYSTEM_ENABLED", Some("true")),
                ("DYN_SYSTEM_PORT", Some("0")),
                ("DYN_SYSTEM_STARTING_HEALTH_STATUS", Some("ready")),
                ("DYN_LOGGING_JSONL", Some("1")),
                ("DYN_LOG", Some("trace")),
            ],
            (async || {
                // TODO Add proper testing for
                // trace id and parent id

                crate::logging::init();

                let drt = Arc::new(create_test_drt_async().await);

                // Get system status server info from DRT (instead of manually spawning)
                let system_info = drt
                    .system_status_server_info()
                    .expect("System status server should be started by DRT");
                let addr = system_info.socket_addr;
                let client = reqwest::Client::new();
                for path in [("/health"), ("/live"), ("/someRandomPathNotFoundHere")] {
                    let traceparent_value =
                        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
                    let tracestate_value = "vendor1=opaqueValue1,vendor2=opaqueValue2";
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        reqwest::header::HeaderName::from_static("traceparent"),
                        reqwest::header::HeaderValue::from_str(traceparent_value)?,
                    );
                    headers.insert(
                        reqwest::header::HeaderName::from_static("tracestate"),
                        reqwest::header::HeaderValue::from_str(tracestate_value)?,
                    );
                    let url = format!("http://{}{}", addr, path);
                    let response = client.get(&url).headers(headers).send().await.unwrap();
                    let status = response.status();
                    let body = response.text().await.unwrap();
                    tracing::info!(body = body, status = status.to_string());
                }

                Ok::<(), anyhow::Error>(())
            })(),
        )
        .await;
        Ok(())
    }

    #[tokio::test]
    async fn test_health_endpoint_with_changing_health_status() {
        // Test health endpoint starts in not ready status, then becomes ready
        // when endpoints are created (DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS=generate)
        const ENDPOINT_NAME: &str = "generate";
        const ENDPOINT_HEALTH_CONFIG: &str = "[\"generate\"]";
        temp_env::async_with_vars(
            [
                ("DYN_SYSTEM_ENABLED", Some("true")),
                ("DYN_SYSTEM_PORT", Some("0")),
                ("DYN_SYSTEM_STARTING_HEALTH_STATUS", Some("notready")),
                ("DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS", Some(ENDPOINT_HEALTH_CONFIG)),
            ],
            async {
                let drt = Arc::new(create_test_drt_async().await);

                // Check if system status server was started
                let system_info_opt = drt.system_status_server_info();

                // Ensure system status server was spawned by DRT
                assert!(
                    system_info_opt.is_some(),
                    "System status server was not spawned by DRT. Expected DRT to spawn server when DYN_SYSTEM_ENABLED=true, but system_status_server_info() returned None. Environment: DYN_SYSTEM_ENABLED={:?}, DYN_SYSTEM_PORT={:?}",
                    std::env::var("DYN_SYSTEM_ENABLED"),
                    std::env::var("DYN_SYSTEM_PORT")
                );

                // Get the system status server info from DRT - this should never fail now due to above check
                let system_info = system_info_opt.unwrap();
                let addr = system_info.socket_addr;

                // Initially check health - should be not ready
                let client = reqwest::Client::new();
                let health_url = format!("http://{}/health", addr);

                let response = client.get(&health_url).send().await.unwrap();
                let status = response.status();
                let body = response.text().await.unwrap();

                // Health should be not ready (503) initially
                assert_eq!(status, 503, "Health should be 503 (not ready) initially, got: {}", status);
                assert!(body.contains("\"status\":\"notready\""), "Health should contain status notready");

                // Now create a namespace, component, and endpoint to make the system healthy
                let namespace = drt.namespace("ns1234").unwrap();
                let component = namespace.component("comp1234").unwrap();

                // Create a simple test handler
                use crate::pipeline::{async_trait, network::Ingress, AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, SingleIn};
                use crate::protocols::annotated::Annotated;

                struct TestHandler;

                #[async_trait]
                impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for TestHandler {
                    async fn generate(&self, input: SingleIn<String>) -> crate::Result<ManyOut<Annotated<String>>> {
                        let (data, ctx) = input.into_parts();
                        let response = Annotated::from_data(format!("You responded: {}", data));
                        Ok(crate::pipeline::ResponseStream::new(
                            Box::pin(crate::stream::iter(vec![response])),
                            ctx.context()
                        ))
                    }
                }

                // Create the ingress and start the endpoint service
                let ingress = Ingress::for_engine(std::sync::Arc::new(TestHandler)).unwrap();

                // Start the service and endpoint
                tokio::spawn(async move {
                    let _ = component
                        .service_builder()
                        .create()
                        .await
                        .unwrap()
                        .endpoint(ENDPOINT_NAME)
                        .endpoint_builder()
                        .handler(ingress)
                        .start()
                        .await;
                });

                // Hit health endpoint 200 times to verify consistency
                let mut success_count = 0;
                let mut failures = Vec::new();

                for i in 1..=200 {
                    let response = client.get(&health_url).send().await.unwrap();
                    let status = response.status();
                    let body = response.text().await.unwrap();

                    if status == 200 && body.contains("\"status\":\"ready\"") {
                        success_count += 1;
                    } else {
                        failures.push((i, status.as_u16(), body.clone()));
                        if failures.len() <= 5 {  // Only log first 5 failures
                            tracing::warn!("Request {}: status={}, body={}", i, status, body);
                        }
                    }
                }

                tracing::info!("Health endpoint test results: {}/200 requests succeeded", success_count);
                if !failures.is_empty() {
                    tracing::warn!("Failed requests: {}", failures.len());
                }

                // Expect at least 150 out of 200 requests to be successful
                assert!(success_count >= 150, "Expected at least 150 out of 200 requests to succeed, but only {} succeeded", success_count);
            },
        )
        .await;
    }

    #[tokio::test]
    async fn test_spawn_system_status_server_endpoints() {
        // use reqwest for HTTP requests
        temp_env::async_with_vars(
            [
                ("DYN_SYSTEM_ENABLED", Some("true")),
                ("DYN_SYSTEM_PORT", Some("0")),
                ("DYN_SYSTEM_STARTING_HEALTH_STATUS", Some("ready")),
            ],
            async {
                let drt = Arc::new(create_test_drt_async().await);

                // Get system status server info from DRT (instead of manually spawning)
                let system_info = drt
                    .system_status_server_info()
                    .expect("System status server should be started by DRT");
                let addr = system_info.socket_addr;
                let client = reqwest::Client::new();
                for (path, expect_200, expect_body) in [
                    ("/health", true, "ready"),
                    ("/live", true, "ready"),
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
                // DRT handles server cleanup automatically
            },
        )
        .await;
    }
}
