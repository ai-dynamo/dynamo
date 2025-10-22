// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP endpoint for receiving requests via Axum/HTTP/2

use super::*;
use crate::SystemHealth;
use crate::config::HealthStatus;
use crate::logging::TraceParent;
use anyhow::Result;
use axum::{
    Router,
    body::Bytes,
    extract::{Path, State as AxumState},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::post,
};
use derive_builder::Builder;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as Http2Builder;
use hyper_util::service::TowerToHyperService;
use std::collections::HashMap;
use std::net::SocketAddr;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use tower::Service;
use tower_http::trace::TraceLayer;
use tracing::Instrument;

/// Default root path for dynamo RPC endpoints
const DEFAULT_RPC_ROOT_PATH: &str = "/v1/rpc";

/// version of crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Shared HTTP server that handles multiple endpoints on a single port
pub struct SharedHttpServer {
    handlers: Arc<tokio::sync::RwLock<HashMap<String, Arc<EndpointHandler>>>>,
    bind_addr: SocketAddr,
    cancellation_token: CancellationToken,
}

/// Handler for a specific endpoint
struct EndpointHandler {
    service_handler: Arc<dyn PushWorkHandler>,
    instance_id: i64,
    namespace: Arc<String>,
    component_name: Arc<String>,
    endpoint_name: Arc<String>,
    system_health: Arc<Mutex<SystemHealth>>,
    inflight: Arc<AtomicU64>,
    notify: Arc<Notify>,
}

impl SharedHttpServer {
    pub fn new(bind_addr: SocketAddr, cancellation_token: CancellationToken) -> Arc<Self> {
        Arc::new(Self {
            handlers: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            bind_addr,
            cancellation_token,
        })
    }

    /// Register an endpoint handler with this server
    pub async fn register_endpoint(
        &self,
        subject: String,
        service_handler: Arc<dyn PushWorkHandler>,
        instance_id: i64,
        namespace: String,
        component_name: String,
        endpoint_name: String,
        system_health: Arc<Mutex<SystemHealth>>,
    ) -> Result<()> {
        let handler = Arc::new(EndpointHandler {
            service_handler,
            instance_id,
            namespace: Arc::new(namespace),
            component_name: Arc::new(component_name),
            endpoint_name: Arc::new(endpoint_name.clone()),
            system_health: system_health.clone(),
            inflight: Arc::new(AtomicU64::new(0)),
            notify: Arc::new(Notify::new()),
        });

        // Set health status
        system_health
            .lock()
            .set_endpoint_health_status(&endpoint_name, HealthStatus::Ready);

        let subject_clone = subject.clone();
        self.handlers.write().await.insert(subject, handler);
        tracing::debug!("Registered endpoint handler for subject: {}", subject_clone);
        Ok(())
    }

    /// Unregister an endpoint handler
    pub async fn unregister_endpoint(&self, subject: &str, endpoint_name: &str) {
        if let Some(handler) = self.handlers.write().await.remove(subject) {
            handler
                .system_health
                .lock()
                .set_endpoint_health_status(endpoint_name, HealthStatus::NotReady);
            tracing::debug!("Unregistered endpoint handler for subject: {}", subject);
        }
    }

    /// Start the shared HTTP server
    pub async fn start(self: Arc<Self>) -> Result<()> {
        let rpc_root_path = std::env::var("DYN_HTTP_RPC_ROOT_PATH")
            .unwrap_or_else(|_| DEFAULT_RPC_ROOT_PATH.to_string());
        let route_pattern = format!("{}/{{*endpoint}}", rpc_root_path);

        let app = Router::new()
            .route(&route_pattern, post(handle_shared_request))
            .layer(TraceLayer::new_for_http())
            .with_state(self.clone());

        tracing::info!(
            "Starting shared HTTP/2 endpoint server on {} at path {}/:endpoint",
            self.bind_addr,
            rpc_root_path
        );

        let listener = tokio::net::TcpListener::bind(&self.bind_addr).await?;
        let cancellation_token = self.cancellation_token.clone();

        loop {
            tokio::select! {
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, _addr)) => {
                            let app_clone = app.clone();
                            let cancel_clone = cancellation_token.clone();

                            tokio::spawn(async move {
                                // Create HTTP/2 connection builder with prior knowledge
                                let http2_builder = Http2Builder::new(TokioExecutor::new());

                                let io = TokioIo::new(stream);
                                let tower_service = app_clone.into_service();

                                // Wrap Tower service for Hyper compatibility
                                let hyper_service = TowerToHyperService::new(tower_service);

                                tokio::select! {
                                    result = http2_builder.serve_connection(io, hyper_service) => {
                                        if let Err(e) = result {
                                            tracing::debug!("HTTP/2 connection error: {}", e);
                                        }
                                    }
                                    _ = cancel_clone.cancelled() => {
                                        tracing::trace!("Connection cancelled");
                                    }
                                }
                            });
                        }
                        Err(e) => {
                            tracing::error!("Failed to accept connection: {}", e);
                        }
                    }
                }
                _ = cancellation_token.cancelled() => {
                    tracing::info!("SharedHttpServer received cancellation signal, shutting down");
                    return Ok(());
                }
            }
        }
    }

    /// Wait for all inflight requests across all endpoints
    pub async fn wait_for_inflight(&self) {
        let handlers = self.handlers.read().await;
        for handler in handlers.values() {
            while handler.inflight.load(Ordering::SeqCst) > 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }
    }
}

/// HTTP handler for the shared server
async fn handle_shared_request(
    AxumState(server): AxumState<Arc<SharedHttpServer>>,
    Path(endpoint_path): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    // Look up the handler for this endpoint
    let handlers = server.handlers.read().await;
    let handler = match handlers.get(&endpoint_path) {
        Some(h) => h.clone(),
        None => {
            tracing::warn!("No handler found for endpoint: {}", endpoint_path);
            return (StatusCode::NOT_FOUND, "Endpoint not found");
        }
    };
    drop(handlers);

    // Increment inflight counter
    handler.inflight.fetch_add(1, Ordering::SeqCst);

    // Extract tracing headers
    let traceparent = TraceParent::from_axum_headers(&headers);

    // Spawn async handler
    let service_handler = handler.service_handler.clone();
    let inflight = handler.inflight.clone();
    let notify = handler.notify.clone();
    let namespace = handler.namespace.clone();
    let component_name = handler.component_name.clone();
    let endpoint_name = handler.endpoint_name.clone();
    let instance_id = handler.instance_id;

    tokio::spawn(async move {
        tracing::trace!(instance_id, "handling new HTTP request");
        let result = service_handler
            .handle_payload(body)
            .instrument(tracing::info_span!(
                "handle_payload",
                component = component_name.as_ref(),
                endpoint = endpoint_name.as_ref(),
                namespace = namespace.as_ref(),
                instance_id = instance_id,
                trace_id = traceparent.trace_id,
                parent_id = traceparent.parent_id,
                x_request_id = traceparent.x_request_id,
                x_dynamo_request_id = traceparent.x_dynamo_request_id,
                tracestate = traceparent.tracestate
            ))
            .await;
        match result {
            Ok(_) => {
                tracing::trace!(instance_id, "request handled successfully");
            }
            Err(e) => {
                tracing::warn!("Failed to handle request: {}", e.to_string());
            }
        }

        // Decrease inflight counter
        inflight.fetch_sub(1, Ordering::SeqCst);
        notify.notify_one();
    });

    // Return 202 Accepted immediately (like NATS ack)
    (StatusCode::ACCEPTED, "")
}

/// Shared state for HTTP endpoint handler
#[derive(Clone)]
struct HttpEndpointState {
    service_handler: Arc<dyn PushWorkHandler>,
    instance_id: i64,
    namespace: Arc<String>,
    component_name: Arc<String>,
    endpoint_name: Arc<String>,
    system_health: Arc<Mutex<SystemHealth>>,
    inflight: Arc<AtomicU64>,
    notify: Arc<Notify>,
}

#[derive(Builder)]
pub struct HttpEndpoint {
    pub service_handler: Arc<dyn PushWorkHandler>,
    pub cancellation_token: CancellationToken,
    #[builder(default = "true")]
    pub graceful_shutdown: bool,
    #[builder(default = "DEFAULT_RPC_ROOT_PATH.to_string()")]
    pub rpc_root_path: String,
}

impl HttpEndpoint {
    pub fn builder() -> HttpEndpointBuilder {
        HttpEndpointBuilder::default()
    }

    /// Get the RPC root path from environment or use default
    fn get_rpc_root_path() -> String {
        std::env::var("DYN_HTTP_RPC_ROOT_PATH")
            .unwrap_or_else(|_| DEFAULT_RPC_ROOT_PATH.to_string())
    }

    pub async fn start(
        self,
        bind_addr: SocketAddr,
        namespace: String,
        component_name: String,
        endpoint_name: String,
        instance_id: i64,
        system_health: Arc<Mutex<SystemHealth>>,
    ) -> Result<()> {
        let inflight = Arc::new(AtomicU64::new(0));
        let notify = Arc::new(Notify::new());

        let state = HttpEndpointState {
            service_handler: self.service_handler.clone(),
            instance_id,
            namespace: Arc::new(namespace),
            component_name: Arc::new(component_name.clone()),
            endpoint_name: Arc::new(endpoint_name.clone()),
            system_health: system_health.clone(),
            inflight: inflight.clone(),
            notify: notify.clone(),
        };

        // Set initial health status
        system_health
            .lock()
            .set_endpoint_health_status(&endpoint_name, HealthStatus::Ready);

        // Build the router
        // Pattern: POST /v1/rpc/*endpoint (catches all paths under /v1/rpc/)
        // Example: POST /v1/rpc/namespace.default.component.backend.endpoint.generate.instance.0
        let rpc_root_path = Self::get_rpc_root_path();
        let route_pattern = format!("{}/{{*endpoint}}", rpc_root_path);

        let app = Router::new()
            .route(&route_pattern, post(handle_request))
            .layer(TraceLayer::new_for_http())
            .with_state(state);

        tracing::info!(
            "Starting HTTP/2 endpoint server on {} at path {}/:endpoint",
            bind_addr,
            rpc_root_path
        );

        // Create the server
        let listener = tokio::net::TcpListener::bind(bind_addr).await?;
        let cancellation_token = self.cancellation_token.clone();

        // Run server with graceful shutdown
        let server_handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    accept_result = listener.accept() => {
                        match accept_result {
                            Ok((stream, _addr)) => {
                                let app_clone = app.clone();
                                let cancel_clone = cancellation_token.clone();

                                tokio::spawn(async move {
                                    // Create HTTP/2 connection builder with prior knowledge
                                    let http2_builder = Http2Builder::new(TokioExecutor::new());

                                    let io = TokioIo::new(stream);
                                    let tower_service = app_clone.into_service();

                                    // Wrap Tower service for Hyper compatibility
                                    let hyper_service = TowerToHyperService::new(tower_service);

                                    tokio::select! {
                                        result = http2_builder.serve_connection(io, hyper_service) => {
                                            if let Err(e) = result {
                                                tracing::debug!("HTTP/2 connection error: {}", e);
                                            }
                                        }
                                        _ = cancel_clone.cancelled() => {
                                            tracing::trace!("Connection cancelled");
                                        }
                                    }
                                });
                            }
                            Err(e) => {
                                tracing::error!("Failed to accept connection: {}", e);
                            }
                        }
                    }
                    _ = cancellation_token.cancelled() => {
                        tracing::info!("HttpEndpoint received cancellation signal, shutting down service");
                        return Ok::<(), anyhow::Error>(());
                    }
                }
            }
        });

        // Wait for server to complete or be cancelled
        server_handle.await??;

        // Mark as not ready
        system_health
            .lock()
            .set_endpoint_health_status(&endpoint_name, HealthStatus::NotReady);

        // Wait for inflight requests if graceful shutdown is enabled
        if self.graceful_shutdown {
            tracing::info!(
                "Waiting for {} inflight requests to complete",
                inflight.load(Ordering::SeqCst)
            );
            while inflight.load(Ordering::SeqCst) > 0 {
                notify.notified().await;
            }
            tracing::info!("All inflight requests completed");
        } else {
            tracing::info!("Skipping graceful shutdown, not waiting for inflight requests");
        }

        Ok(())
    }
}

/// HTTP handler for incoming requests
async fn handle_request(
    AxumState(state): AxumState<HttpEndpointState>,
    Path(_endpoint): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    // Increment inflight counter
    state.inflight.fetch_add(1, Ordering::SeqCst);

    // Extract tracing headers
    let traceparent = TraceParent::from_axum_headers(&headers);

    // Spawn async handler
    let service_handler = state.service_handler.clone();
    let inflight = state.inflight.clone();
    let notify = state.notify.clone();
    let namespace = state.namespace.clone();
    let component_name = state.component_name.clone();
    let endpoint_name = state.endpoint_name.clone();
    let instance_id = state.instance_id;

    tokio::spawn(async move {
        tracing::trace!(instance_id, "handling new HTTP request");
        let result = service_handler
            .handle_payload(body)
            .instrument(tracing::info_span!(
                "handle_payload",
                component = component_name.as_ref(),
                endpoint = endpoint_name.as_ref(),
                namespace = namespace.as_ref(),
                instance_id = instance_id,
                trace_id = traceparent.trace_id,
                parent_id = traceparent.parent_id,
                x_request_id = traceparent.x_request_id,
                x_dynamo_request_id = traceparent.x_dynamo_request_id,
                tracestate = traceparent.tracestate
            ))
            .await;
        match result {
            Ok(_) => {
                tracing::trace!(instance_id, "request handled successfully");
            }
            Err(e) => {
                tracing::warn!("Failed to handle request: {}", e.to_string());
            }
        }

        // Decrease inflight counter
        inflight.fetch_sub(1, Ordering::SeqCst);
        notify.notify_one();
    });

    // Return 202 Accepted immediately (like NATS ack)
    (StatusCode::ACCEPTED, "")
}

/// Extension trait for TraceParent to support Axum headers
impl TraceParent {
    pub fn from_axum_headers(headers: &HeaderMap) -> Self {
        let mut traceparent = TraceParent::default();

        if let Some(value) = headers.get("traceparent") {
            if let Ok(s) = value.to_str() {
                traceparent.trace_id = Some(s.to_string());
            }
        }

        if let Some(value) = headers.get("tracestate") {
            if let Ok(s) = value.to_str() {
                traceparent.tracestate = Some(s.to_string());
            }
        }

        if let Some(value) = headers.get("x-request-id") {
            if let Ok(s) = value.to_str() {
                traceparent.x_request_id = Some(s.to_string());
            }
        }

        if let Some(value) = headers.get("x-dynamo-request-id") {
            if let Ok(s) = value.to_str() {
                traceparent.x_dynamo_request_id = Some(s.to_string());
            }
        }

        traceparent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{ManyOut, PipelineError, SingleIn};
    use bytes::Bytes;

    struct MockHandler;

    #[async_trait::async_trait]
    impl PushWorkHandler for MockHandler {
        async fn handle_payload(&self, _payload: Bytes) -> Result<(), PipelineError> {
            Ok(())
        }

        fn add_metrics(
            &self,
            _endpoint: &crate::component::Endpoint,
            _metrics_labels: Option<&[(&str, &str)]>,
        ) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_http_endpoint_builder() {
        let handler = Arc::new(MockHandler);
        let token = CancellationToken::new();

        let endpoint = HttpEndpoint::builder()
            .service_handler(handler)
            .cancellation_token(token)
            .graceful_shutdown(true)
            .build()
            .unwrap();

        assert!(endpoint.graceful_shutdown);
    }

    #[test]
    fn test_traceparent_from_axum_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("traceparent", "test-trace-id".parse().unwrap());
        headers.insert("tracestate", "test-state".parse().unwrap());
        headers.insert("x-request-id", "req-123".parse().unwrap());
        headers.insert("x-dynamo-request-id", "dyn-456".parse().unwrap());

        let traceparent = TraceParent::from_axum_headers(&headers);
        assert_eq!(traceparent.trace_id, Some("test-trace-id".to_string()));
        assert_eq!(traceparent.tracestate, Some("test-state".to_string()));
        assert_eq!(traceparent.x_request_id, Some("req-123".to_string()));
        assert_eq!(traceparent.x_dynamo_request_id, Some("dyn-456".to_string()));
    }
}
