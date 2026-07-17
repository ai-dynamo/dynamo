// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP client for vLLM's chat render endpoint.
//!
//! The client is intentionally limited to protocol handling. Callers decide how
//! a render failure affects request routing.

use std::time::Duration;

use anyhow::Context;
use futures::StreamExt;
use reqwest::header::CONTENT_TYPE;
use reqwest::{Client, StatusCode, Url};
use serde::Deserialize;
use thiserror::Error;

const CHAT_RENDER_PATH: &str = "/v1/chat/completions/render";
const MAX_ERROR_BODY_BYTES: usize = 1024;

/// Default deadline for a vLLM render request.
pub const DEFAULT_VLLM_RENDER_TIMEOUT: Duration = Duration::from_secs(5);

/// A reusable client for vLLM's `/v1/chat/completions/render` endpoint.
#[derive(Clone, Debug)]
pub struct VllmRenderClient {
    client: Client,
    endpoint: Url,
    timeout: Duration,
}

/// Failures returned by [`VllmRenderClient::render_chat`].
#[derive(Debug, Error)]
pub enum VllmRenderError {
    /// The renderer could not be reached or the connection failed.
    #[error("vLLM renderer is unavailable: {source}")]
    Unavailable {
        #[source]
        source: reqwest::Error,
    },
    /// The renderer did not complete the request before the configured deadline.
    #[error("vLLM render request timed out after {timeout:?}: {source}")]
    Timeout {
        timeout: Duration,
        #[source]
        source: reqwest::Error,
    },
    /// The renderer returned an HTTP error response.
    #[error("vLLM renderer returned {status}: {body}")]
    UpstreamStatus { status: StatusCode, body: String },
    /// The renderer returned a successful response that did not match its contract.
    #[error("vLLM renderer returned an invalid response: {source}")]
    InvalidResponse {
        #[source]
        source: serde_json::Error,
    },
}

#[derive(Debug, Deserialize)]
struct VllmRenderResponse {
    token_ids: Vec<u32>,
}

impl VllmRenderClient {
    /// Build a pooled HTTP client from the vLLM renderer's base URL.
    ///
    /// The base URL selects either a local sidecar (for example,
    /// `http://127.0.0.1:8000`) or an external Service. The vLLM-specific chat
    /// render path is appended by this client.
    pub fn new(base_url: &str, timeout: Duration) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !timeout.is_zero(),
            "vLLM render timeout must be greater than zero"
        );

        let mut endpoint = Url::parse(base_url)
            .with_context(|| format!("invalid vLLM renderer base URL {base_url:?}"))?;
        anyhow::ensure!(
            matches!(endpoint.scheme(), "http" | "https") && endpoint.host_str().is_some(),
            "vLLM renderer base URL must be an absolute HTTP(S) URL"
        );
        endpoint.set_path(CHAT_RENDER_PATH);
        endpoint.set_query(None);
        endpoint.set_fragment(None);

        let client = Client::builder()
            .timeout(timeout)
            .build()
            .context("building vLLM renderer HTTP client")?;

        Ok(Self {
            client,
            endpoint,
            timeout,
        })
    }

    /// Forward an OpenAI chat-completions JSON body and return its prompt tokens.
    ///
    /// The body is sent unchanged so vLLM remains responsible for validating
    /// engine-specific request fields and applying the chat template.
    pub async fn render_chat(&self, request_body: &[u8]) -> Result<Vec<u32>, VllmRenderError> {
        let response = self
            .client
            .post(self.endpoint.clone())
            .header(CONTENT_TYPE, "application/json")
            .body(request_body.to_vec())
            .send()
            .await
            .map_err(|source| self.classify_transport_error(source))?;

        let status = response.status();
        if !status.is_success() {
            return Err(VllmRenderError::UpstreamStatus {
                status,
                body: read_error_body(response).await,
            });
        }

        let body = response
            .bytes()
            .await
            .map_err(|source| self.classify_transport_error(source))?;
        let response: VllmRenderResponse = serde_json::from_slice(&body)
            .map_err(|source| VllmRenderError::InvalidResponse { source })?;

        Ok(response.token_ids)
    }

    fn classify_transport_error(&self, source: reqwest::Error) -> VllmRenderError {
        if source.is_timeout() {
            VllmRenderError::Timeout {
                timeout: self.timeout,
                source,
            }
        } else {
            VllmRenderError::Unavailable { source }
        }
    }
}

async fn read_error_body(response: reqwest::Response) -> String {
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();

    while body.len() < MAX_ERROR_BODY_BYTES {
        let Some(chunk) = stream.next().await else {
            break;
        };
        let Ok(chunk) = chunk else {
            break;
        };
        let remaining = MAX_ERROR_BODY_BYTES - body.len();
        body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }

    String::from_utf8_lossy(&body).into_owned()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::body::Bytes;
    use axum::routing::post;
    use axum::{Json, Router};
    use serde_json::json;
    use tokio::net::TcpListener;
    use tokio::sync::mpsc;
    use tokio::task::JoinHandle;

    use super::*;

    async fn spawn_server(router: Router) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        (format!("http://{address}"), task)
    }

    #[tokio::test]
    async fn forwards_body_to_chat_render_endpoint() {
        let (body_tx, mut body_rx) = mpsc::unbounded_channel();
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(move |body: Bytes| {
                let body_tx = body_tx.clone();
                async move {
                    body_tx.send(body).unwrap();
                    Json(json!({
                        "token_ids": [1, 2, 3],
                        "features": {"ignored": true}
                    }))
                }
            }),
        );
        let (base_url, server) = spawn_server(router).await;
        let client = VllmRenderClient::new(&base_url, DEFAULT_VLLM_RENDER_TIMEOUT).unwrap();
        let request =
            br#"{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hello"}]}"#;

        let token_ids = client.render_chat(request).await.unwrap();

        assert_eq!(token_ids, vec![1, 2, 3]);
        assert_eq!(body_rx.recv().await.unwrap().as_ref(), request);
        server.abort();
    }

    #[tokio::test]
    async fn classifies_timeout() {
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(|| async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Json(json!({"token_ids": [1]}))
            }),
        );
        let (base_url, server) = spawn_server(router).await;
        let timeout = Duration::from_millis(10);
        let client = VllmRenderClient::new(&base_url, timeout).unwrap();

        let error = client.render_chat(b"{}").await.unwrap_err();

        assert!(matches!(
            error,
            VllmRenderError::Timeout {
                timeout: actual,
                ..
            } if actual == timeout
        ));
        server.abort();
    }

    #[tokio::test]
    async fn classifies_unavailable_renderer() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let client =
            VllmRenderClient::new(&format!("http://{address}"), DEFAULT_VLLM_RENDER_TIMEOUT)
                .unwrap();

        let error = client.render_chat(b"{}").await.unwrap_err();

        assert!(matches!(error, VllmRenderError::Unavailable { .. }));
    }

    #[tokio::test]
    async fn classifies_upstream_status_and_bounds_body() {
        let response_body = Arc::new("x".repeat(MAX_ERROR_BODY_BYTES * 2));
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(move || {
                let response_body = response_body.clone();
                async move {
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        response_body.as_str().to_owned(),
                    )
                }
            }),
        );
        let (base_url, server) = spawn_server(router).await;
        let client = VllmRenderClient::new(&base_url, DEFAULT_VLLM_RENDER_TIMEOUT).unwrap();

        let error = client.render_chat(b"{}").await.unwrap_err();

        match error {
            VllmRenderError::UpstreamStatus { status, body } => {
                assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
                assert_eq!(body.len(), MAX_ERROR_BODY_BYTES);
            }
            other => panic!("expected upstream status error, got {other:?}"),
        }
        server.abort();
    }

    #[tokio::test]
    async fn classifies_invalid_success_response() {
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(|| async { Json(json!({"prompt": "missing token_ids"})) }),
        );
        let (base_url, server) = spawn_server(router).await;
        let client = VllmRenderClient::new(&base_url, DEFAULT_VLLM_RENDER_TIMEOUT).unwrap();

        let error = client.render_chat(b"{}").await.unwrap_err();

        assert!(matches!(error, VllmRenderError::InvalidResponse { .. }));
        server.abort();
    }

    #[test]
    fn rejects_invalid_client_config() {
        assert!(VllmRenderClient::new("unix:///tmp/vllm.sock", Duration::from_secs(1)).is_err());
        assert!(VllmRenderClient::new("http://127.0.0.1:8000", Duration::ZERO).is_err());
    }
}
