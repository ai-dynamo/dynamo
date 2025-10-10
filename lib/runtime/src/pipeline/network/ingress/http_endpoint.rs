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
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;
use tracing::Instrument;

/// Default root path for dynamo RPC endpoints
const DEFAULT_RPC_ROOT_PATH: &str = "/v1/dynamo";

/// version of crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

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
            .unwrap()
            .set_endpoint_health_status(&endpoint_name, HealthStatus::Ready);

        // Build the router
        // Pattern: POST /v1/dynamo/*endpoint (catches all paths under /v1/dynamo/)
        // Example: POST /v1/dynamo/namespace.default.component.backend.endpoint.generate.instance.0
        let rpc_root_path = Self::get_rpc_root_path();
        let route_pattern = format!("{}/{{*endpoint}}", rpc_root_path);

        let app = Router::new()
            .route(&route_pattern, post(handle_request))
            .layer(TraceLayer::new_for_http())
            .with_state(state);

        tracing::info!(
            "Starting HTTP endpoint server on {} at path {}/:endpoint",
            bind_addr,
            rpc_root_path
        );

        // Create the server
        let listener = tokio::net::TcpListener::bind(bind_addr).await?;
        let server = axum::serve(listener, app.into_make_service());

        // Run server with graceful shutdown
        tokio::select! {
            result = server => {
                result?;
            }
            _ = self.cancellation_token.cancelled() => {
                tracing::info!("HttpEndpoint received cancellation signal, shutting down service");
            }
        }

        // Mark as not ready
        system_health
            .lock()
            .unwrap()
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
