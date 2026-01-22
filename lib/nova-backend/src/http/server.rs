// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP server using Axum
//!
//! Provides three POST routes and one GET route:
//! - POST /message → forwards to message_stream
//! - POST /response → forwards to response_stream
//! - POST /event → forwards to event_stream
//! - GET /health → health check endpoint (returns 200 OK)

use axum::{
    Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use base64::Engine;
use bytes::Bytes;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error};

use crate::TransportAdapter;

/// HTTP server state shared across handlers
#[derive(Clone)]
struct ServerState {
    channels: TransportAdapter,
}

/// HTTP server for receiving messages
pub struct HttpServer {
    listener: tokio::net::TcpListener,
    channels: TransportAdapter,
    cancel_token: CancellationToken,
}

impl HttpServer {
    /// Create a new HTTP server with the given listener
    pub fn new(
        listener: tokio::net::TcpListener,
        channels: TransportAdapter,
        cancel_token: CancellationToken,
    ) -> Self {
        Self {
            listener,
            channels,
            cancel_token,
        }
    }

    /// Run the HTTP server
    pub async fn run(self) -> anyhow::Result<()> {
        let state = ServerState {
            channels: self.channels,
        };

        // Build router with three message routes and one health route
        let app = Router::new()
            .route("/message", post(handle_message))
            .route("/response", post(handle_response))
            .route("/event", post(handle_event))
            .route("/health", get(handle_health))
            .with_state(state);

        // Run server with graceful shutdown
        axum::serve(self.listener, app)
            .with_graceful_shutdown(async move {
                self.cancel_token.cancelled().await;
                debug!("HTTP server shutting down gracefully");
            })
            .await?;

        Ok(())
    }
}

/// Handle POST /message
async fn handle_message(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    handle_request(state.channels.message_stream.clone(), headers, body).await
}

/// Handle POST /response
async fn handle_response(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    handle_request(state.channels.response_stream.clone(), headers, body).await
}

/// Handle POST /event
async fn handle_event(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    handle_request(state.channels.event_stream.clone(), headers, body).await
}

/// Common request handling logic
async fn handle_request(
    sender: flume::Sender<(Bytes, Bytes)>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Extract X-Transport-Header
    let header_b64 = match headers.get("X-Transport-Header") {
        Some(value) => match value.to_str() {
            Ok(s) => s,
            Err(e) => {
                error!("Invalid X-Transport-Header encoding: {}", e);
                return (StatusCode::BAD_REQUEST, "Invalid header encoding").into_response();
            }
        },
        None => {
            error!("Missing X-Transport-Header");
            return (StatusCode::BAD_REQUEST, "Missing X-Transport-Header").into_response();
        }
    };

    // Decode base64 header
    let header_bytes = match base64::engine::general_purpose::STANDARD.decode(header_b64) {
        Ok(bytes) => Bytes::from(bytes),
        Err(e) => {
            error!("Failed to decode base64 header: {}", e);
            return (StatusCode::BAD_REQUEST, "Invalid base64 header").into_response();
        }
    };

    // Payload is the raw body bytes
    let payload_bytes = body;

    // Forward to appropriate stream
    // Note: For now, passing None as the third element (no ResponseSender)
    // This will need to be updated if we implement the direct reply mechanism
    if let Err(e) = sender.send_async((header_bytes, payload_bytes)).await {
        error!("Failed to forward message to stream: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to process message",
        )
            .into_response();
    }

    // Return 202 Accepted
    (StatusCode::ACCEPTED, "").into_response()
}

/// Handle GET /health
async fn handle_health() -> Response {
    // Simple health check - just return 200 OK
    (StatusCode::OK, "").into_response()
}
