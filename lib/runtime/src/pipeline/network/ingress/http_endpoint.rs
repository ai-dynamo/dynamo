// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
use dashmap::DashMap;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as Http2Builder;
use hyper_util::service::TowerToHyperService;
use parking_lot::Mutex;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{Notify, RwLock};
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;
use tracing::Instrument;

/// Default root path for dynamo RPC endpoints
const DEFAULT_RPC_ROOT_PATH: &str = "/v1/rpc";

/// version of crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Shared HTTP server that handles multiple endpoints on a single port
pub struct SharedHttpServer {
    handlers: Arc<DashMap<String, Arc<EndpointHandler>>>,
    bind_addr: SocketAddr,
    actual_addr: RwLock<Option<SocketAddr>>,
    cancellation_token: CancellationToken,
}

/// Handler for a specific endpoint
struct EndpointHandler {
    service_handler: Arc<dyn PushWorkHandler>,
    instance_id: u64,
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
            handlers: Arc::new(DashMap::new()),
            bind_addr,
            actual_addr: RwLock::new(None),
            cancellation_token,
        })
    }

    /// Get the actual bound address (after `bind_and_start` resolves).
    pub fn actual_address(&self) -> Option<SocketAddr> {
        self.actual_addr.try_read().ok().and_then(|g| *g)
    }

    /// Register an endpoint handler with this server
    #[allow(clippy::too_many_arguments)]
    pub async fn register_endpoint(
        &self,
        subject: String,
        service_handler: Arc<dyn PushWorkHandler>,
        instance_id: u64,
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

        // Insert handler FIRST to ensure it's ready to receive requests
        let subject_clone = subject.clone();
        self.handlers.insert(subject, handler);

        // THEN set health status to Ready (after handler is registered and ready)
        system_health
            .lock()
            .set_endpoint_health_status(&endpoint_name, HealthStatus::Ready);

        tracing::debug!("Registered endpoint handler for subject: {subject_clone}");
        Ok(())
    }

    /// Unregister an endpoint handler
    pub async fn unregister_endpoint(&self, subject: &str, endpoint_name: &str) {
        if let Some((_, handler)) = self.handlers.remove(subject) {
            handler
                .system_health
                .lock()
                .set_endpoint_health_status(endpoint_name, HealthStatus::NotReady);
            tracing::debug!(
                endpoint_name = %endpoint_name,
                subject = %subject,
                "Unregistered HTTP endpoint handler"
            );

            let inflight_count = handler.inflight.load(Ordering::SeqCst);
            if inflight_count > 0 {
                tracing::info!(
                    endpoint_name = %endpoint_name,
                    inflight_count = inflight_count,
                    "Waiting for inflight HTTP requests to complete"
                );
                while handler.inflight.load(Ordering::SeqCst) > 0 {
                    handler.notify.notified().await;
                }
                tracing::info!(
                    endpoint_name = %endpoint_name,
                    "All inflight HTTP requests completed"
                );
            }
        }
    }

    /// Bind the TCP listener and start the accept loop.
    ///
    /// Returns the actual bound `SocketAddr` (important when binding to port 0).
    pub async fn bind_and_start(self: Arc<Self>) -> Result<SocketAddr> {
        let rpc_root_path = std::env::var("DYN_HTTP_RPC_ROOT_PATH")
            .unwrap_or_else(|_| DEFAULT_RPC_ROOT_PATH.to_string());
        let route_pattern = format!("{}/{{*endpoint}}", rpc_root_path);

        let app = Router::new()
            .route(&route_pattern, post(handle_shared_request))
            .layer(TraceLayer::new_for_http())
            .with_state(self.clone());

        let listener = tokio::net::TcpListener::bind(&self.bind_addr).await?;
        let actual_addr = listener.local_addr()?;

        tracing::info!(
            requested = %self.bind_addr,
            actual = %actual_addr,
            rpc_root = %rpc_root_path,
            "HTTP/2 endpoint server bound"
        );

        // Store the actual address so `address()` returns the real port.
        *self.actual_addr.write().await = Some(actual_addr);

        let cancellation_token = self.cancellation_token.clone();

        // Spawn the accept loop in the background.
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    accept_result = listener.accept() => {
                        match accept_result {
                            Ok((stream, _addr)) => {
                                let app_clone = app.clone();
                                let cancel_clone = cancellation_token.clone();

                                tokio::spawn(async move {
                                    let http2_builder = Http2Builder::new(TokioExecutor::new());

                                    let io = TokioIo::new(stream);
                                    let tower_service = app_clone.into_service();
                                    let hyper_service = TowerToHyperService::new(tower_service);

                                    tokio::select! {
                                        result = http2_builder.serve_connection(io, hyper_service) => {
                                            if let Err(e) = result {
                                                tracing::debug!("HTTP/2 connection error: {e}");
                                            }
                                        }
                                        _ = cancel_clone.cancelled() => {
                                            tracing::trace!("Connection cancelled");
                                        }
                                    }
                                });
                            }
                            Err(e) => {
                                tracing::error!("Failed to accept connection: {e}");
                            }
                        }
                    }
                    _ = cancellation_token.cancelled() => {
                        tracing::info!("SharedHttpServer received cancellation signal, shutting down");
                        return;
                    }
                }
            }
        });

        Ok(actual_addr)
    }

    /// Wait for all inflight requests across all endpoints
    pub async fn wait_for_inflight(&self) {
        for handler in self.handlers.iter() {
            while handler.value().inflight.load(Ordering::SeqCst) > 0 {
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
    // Look up the handler for this endpoint (lock-free read with DashMap)
    let handler = match server.handlers.get(&endpoint_path) {
        Some(h) => h.clone(),
        None => {
            tracing::warn!("No handler found for endpoint: {endpoint_path}");
            return (StatusCode::NOT_FOUND, "Endpoint not found");
        }
    };

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
                request_id = traceparent.request_id,
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

        if let Some(value) = headers.get("traceparent")
            && let Ok(s) = value.to_str()
        {
            traceparent.trace_id = Some(s.to_string());
        }

        if let Some(value) = headers.get("tracestate")
            && let Ok(s) = value.to_str()
        {
            traceparent.tracestate = Some(s.to_string());
        }

        if let Some(value) = headers.get("x-request-id")
            && let Ok(s) = value.to_str()
        {
            traceparent.x_request_id = Some(s.to_string());
        }

        // Read request-id from internal headers, with fallback to deprecated x-dynamo-request-id
        if let Some(s) = headers
            .get("request-id")
            .and_then(|v| v.to_str().ok())
            .filter(|s| uuid::Uuid::parse_str(s).is_ok())
        {
            traceparent.request_id = Some(s.to_string());
        } else if let Some(s) = headers
            .get("x-dynamo-request-id")
            .and_then(|v| v.to_str().ok())
            .filter(|s| uuid::Uuid::parse_str(s).is_ok())
        {
            traceparent.request_id = Some(s.to_string());
        }

        traceparent
    }
}

// Implement RequestPlaneServer trait for SharedHttpServer
#[async_trait::async_trait]
impl super::unified_server::RequestPlaneServer for SharedHttpServer {
    async fn register_endpoint(
        &self,
        endpoint_name: String,
        service_handler: Arc<dyn PushWorkHandler>,
        instance_id: u64,
        namespace: String,
        component_name: String,
        system_health: Arc<Mutex<SystemHealth>>,
    ) -> Result<()> {
        // For HTTP, we use endpoint_name as both the subject (routing key) and endpoint_name
        self.register_endpoint(
            endpoint_name.clone(),
            service_handler,
            instance_id,
            namespace,
            component_name,
            endpoint_name,
            system_health,
        )
        .await
    }

    async fn unregister_endpoint(&self, endpoint_name: &str) -> Result<()> {
        self.unregister_endpoint(endpoint_name, endpoint_name).await;
        Ok(())
    }

    fn address(&self) -> String {
        let addr = self.actual_address().unwrap_or(self.bind_addr);
        format!("http://{}:{}", addr.ip(), addr.port())
    }

    fn transport_name(&self) -> &'static str {
        "http"
    }

    fn is_healthy(&self) -> bool {
        // Server is healthy if it has been created
        // TODO: Add more sophisticated health checks (e.g., check if listener is active)
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_traceparent_from_axum_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("traceparent", "test-trace-id".parse().unwrap());
        headers.insert("tracestate", "test-state".parse().unwrap());
        headers.insert("x-request-id", "req-123".parse().unwrap());
        headers.insert(
            "x-dynamo-request-id",
            "550e8400-e29b-41d4-a716-446655440000".parse().unwrap(),
        );

        let traceparent = TraceParent::from_axum_headers(&headers);
        assert_eq!(traceparent.trace_id, Some("test-trace-id".to_string()));
        assert_eq!(traceparent.tracestate, Some("test-state".to_string()));
        assert_eq!(traceparent.x_request_id, Some("req-123".to_string()));
        assert_eq!(
            traceparent.request_id,
            Some("550e8400-e29b-41d4-a716-446655440000".to_string())
        );
    }

    #[test]
    fn test_shared_http_server_creation() {
        use std::net::{IpAddr, Ipv4Addr};
        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 0);
        let token = CancellationToken::new();

        let server = SharedHttpServer::new(bind_addr, token);
        assert!(server.handlers.is_empty());
    }

    #[tokio::test]
    async fn test_bind_and_start_assigns_os_port() {
        use std::net::{IpAddr, Ipv4Addr};
        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 0);
        let token = CancellationToken::new();

        let server = SharedHttpServer::new(bind_addr, token.clone());
        let actual_addr = server.clone().bind_and_start().await.unwrap();

        // OS should assign a non-zero port
        assert_ne!(actual_addr.port(), 0);

        // actual_address() should return the real bound address
        assert_eq!(server.actual_address(), Some(actual_addr));

        // address() should contain the real port
        let addr_str =
            <SharedHttpServer as super::unified_server::RequestPlaneServer>::address(&*server);
        assert!(addr_str.contains(&actual_addr.port().to_string()));

        token.cancel();
    }

    #[tokio::test]
    async fn test_two_servers_get_different_ports() {
        use std::net::{IpAddr, Ipv4Addr};
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 0);

        let token1 = CancellationToken::new();
        let token2 = CancellationToken::new();

        let server1 = SharedHttpServer::new(addr, token1.clone());
        let server2 = SharedHttpServer::new(addr, token2.clone());

        let actual1 = server1.clone().bind_and_start().await.unwrap();
        let actual2 = server2.clone().bind_and_start().await.unwrap();

        // Two servers binding to port 0 must get different ports
        assert_ne!(actual1.port(), actual2.port());

        token1.cancel();
        token2.cancel();
    }

    #[tokio::test]
    async fn test_bind_and_start_with_explicit_port() {
        use std::net::{IpAddr, Ipv4Addr};

        // First bind to port 0 to get a free port
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let free_port = listener.local_addr().unwrap().port();
        drop(listener); // Release the port

        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), free_port);
        let token = CancellationToken::new();

        let server = SharedHttpServer::new(bind_addr, token.clone());
        let actual_addr = server.clone().bind_and_start().await.unwrap();

        // When binding to an explicit port, actual should match
        assert_eq!(actual_addr.port(), free_port);

        token.cancel();
    }
}
