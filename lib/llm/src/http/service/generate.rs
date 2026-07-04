// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP handler for the token-in/token-out `Generate` API
//! (`POST /inference/v1/generate`).
//!
//! This is an experimental endpoint, enabled by default (matching vLLM,
//! which mounts `/inference/v1/generate` for any generate-capable model).
//! It registers the route with a placeholder handler that returns HTTP 501
//! Not Implemented; the real handler (engine dispatch + `LLMEngineOutput`
//! accumulation) lands in a follow-up. Disable via `enable_generate_endpoints`.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    middleware,
    response::{IntoResponse, Response},
    routing::post,
};
use serde::Serialize;

use dynamo_http_server::request::get_body_limit;

use super::openai::smart_json_error_middleware;
use super::{RouteDoc, service_v2};
use crate::protocols::openai::generate::GenerateRequest;

/// vLLM-style nested error body: `{"error": {"message", "type", "code"}}`.
#[derive(Serialize, Debug)]
struct GenerateError {
    error: GenerateErrorBody,
}

#[derive(Serialize, Debug)]
struct GenerateErrorBody {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: u16,
}

/// Create an Axum [`Router`] for the token-in/token-out `Generate` endpoint.
/// If no path is provided, the default path is `/inference/v1/generate`.
pub fn generate_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/inference/v1/generate".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(handler_generate))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc], router)
}

/// Placeholder handler for the `Generate` endpoint.
///
/// Accepts and validates the request body shape, then returns HTTP 501 with a
/// vLLM-style nested-`error` body. Real dispatch is implemented in a follow-up.
async fn handler_generate(
    State(_state): State<Arc<service_v2::State>>,
    Json(_request): Json<GenerateRequest>,
) -> Response {
    let code = StatusCode::NOT_IMPLEMENTED;
    (
        code,
        Json(GenerateError {
            error: GenerateErrorBody {
                message: "The /inference/v1/generate endpoint is not implemented yet".to_string(),
                error_type: "not_implemented".to_string(),
                code: code.as_u16(),
            },
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::StatusCode;
    use super::service_v2::HttpService;
    use tokio_util::sync::CancellationToken;

    /// Spin up an `HttpService` bound to an ephemeral port and return the port
    /// plus the run handle. Mirrors the reqwest-based router tests in
    /// `service_v2`.
    async fn serve(enable_generate: bool) -> (u16, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind ephemeral port");
        let port = listener.local_addr().unwrap().port();
        let service = HttpService::builder()
            .port(port)
            .enable_generate_endpoints(enable_generate)
            .build()
            .unwrap();
        let cancel_token = CancellationToken::new();
        let handle = tokio::spawn(async move {
            service.run_with_listener(cancel_token, listener).await.ok();
        });
        // Give the server a moment to start listening.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        (port, handle)
    }

    #[tokio::test]
    async fn generate_route_returns_501_when_enabled() {
        let (port, handle) = serve(true).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
            .send()
            .await
            .expect("generate request failed");
        assert_eq!(resp.status().as_u16(), StatusCode::NOT_IMPLEMENTED.as_u16());
        handle.abort();
    }

    #[tokio::test]
    async fn generate_route_404_when_disabled() {
        let (port, handle) = serve(false).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
            .send()
            .await
            .expect("generate request failed");
        assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
        handle.abort();
    }
}
