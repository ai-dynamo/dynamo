// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP client for vLLM's chat render endpoint.
//!
//! The client is intentionally limited to protocol handling. Callers decide how
//! a render failure affects request routing.
//!
//! The client sends no `Authorization` header, so it targets unauthenticated
//! in-cluster renderers (a sidecar or an internal Service). Authenticated
//! renderers (vLLM `--api-key` / `VLLM_API_KEY`) are a follow-up.

use std::time::Duration;

use anyhow::Context;
use bytes::Bytes;
use futures::StreamExt;
use reqwest::header::CONTENT_TYPE;
use reqwest::{Client, StatusCode, Url};
use serde::Deserialize;
use thiserror::Error;

const CHAT_RENDER_PATH: &str = "/v1/chat/completions/render";
const MAX_ERROR_BODY_BYTES: usize = 1024;

/// A reusable client for vLLM's `/v1/chat/completions/render` endpoint.
#[derive(Clone, Debug)]
pub struct VllmRenderClient {
    client: Client,
    endpoint: Url,
    timeout: Duration,
    max_response_bytes: usize,
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
    /// The renderer returned a successful response larger than the configured limit.
    #[error("vLLM renderer response is too large: {received} bytes exceeds the {limit}-byte limit")]
    ResponseTooLarge { limit: usize, received: u64 },
}

#[derive(Debug, Deserialize)]
struct VllmRenderResponse {
    token_ids: Vec<u32>,
}

/// Parse and validate a tokenizer service base URL.
pub(crate) fn parse_tokenizer_service_base_url(base_url: &str) -> anyhow::Result<Url> {
    let url = Url::parse(base_url)
        .with_context(|| format!("invalid tokenizer service base URL {base_url:?}"))?;
    anyhow::ensure!(
        matches!(url.scheme(), "http" | "https") && url.host_str().is_some(),
        "tokenizer service base URL must be an absolute HTTP(S) URL"
    );
    Ok(url)
}

impl VllmRenderClient {
    /// Build a pooled HTTP client from the vLLM renderer's base URL.
    ///
    /// The base URL selects either a local sidecar (for example,
    /// `http://127.0.0.1:8000`) or an external Service. The vLLM-specific chat
    /// render path is appended by this client.
    pub fn new(
        base_url: &str,
        timeout: Duration,
        max_response_bytes: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !timeout.is_zero(),
            "vLLM render timeout must be greater than zero"
        );
        anyhow::ensure!(
            max_response_bytes > 0,
            "vLLM render maximum response bytes must be greater than zero"
        );

        let mut endpoint = parse_tokenizer_service_base_url(base_url)?;
        {
            let mut path_segments = endpoint.path_segments_mut().map_err(|_| {
                anyhow::anyhow!("vLLM renderer base URL cannot be used as a base URL")
            })?;
            path_segments.pop_if_empty();
            path_segments.extend(CHAT_RENDER_PATH.trim_start_matches('/').split('/'));
        }
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
            max_response_bytes,
        })
    }

    /// Forward an OpenAI chat-completions JSON body and return its prompt tokens.
    ///
    /// The body is sent unchanged so vLLM remains responsible for validating
    /// engine-specific request fields and applying the chat template.
    pub async fn render_chat(&self, request_body: Bytes) -> Result<Vec<u32>, VllmRenderError> {
        let response = self
            .client
            .post(self.endpoint.clone())
            .header(CONTENT_TYPE, "application/json")
            .body(request_body)
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

        match response.content_length() {
            Some(received) if received > self.max_response_bytes as u64 => {
                return Err(VllmRenderError::ResponseTooLarge {
                    limit: self.max_response_bytes,
                    received,
                });
            }
            _ => {}
        }

        let body = self.read_success_body(response).await?;
        let response: VllmRenderResponse = serde_json::from_slice(&body)
            .map_err(|source| VllmRenderError::InvalidResponse { source })?;

        Ok(response.token_ids)
    }

    async fn read_success_body(
        &self,
        response: reqwest::Response,
    ) -> Result<Vec<u8>, VllmRenderError> {
        let mut body = Vec::new();
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|source| self.classify_transport_error(source))?;
            let received = body.len().saturating_add(chunk.len());
            if received > self.max_response_bytes {
                return Err(VllmRenderError::ResponseTooLarge {
                    limit: self.max_response_bytes,
                    received: received as u64,
                });
            }
            body.extend_from_slice(&chunk);
        }

        Ok(body)
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
    use std::convert::Infallible;
    use std::sync::Arc;

    use axum::body::{Body, Bytes};
    use axum::http::HeaderMap;
    use axum::response::Response;
    use axum::routing::post;
    use axum::{Json, Router};
    use futures::stream;
    use serde_json::json;
    use tokio::net::TcpListener;
    use tokio::sync::mpsc;
    use tokio::task::JoinHandle;

    use super::*;

    const TEST_TIMEOUT: Duration = Duration::from_secs(5);
    const TEST_MAX_RESPONSE_BYTES: usize = 1024;

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
        let (request_tx, mut request_rx) = mpsc::unbounded_channel();
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(move |headers: HeaderMap, body: Bytes| {
                let request_tx = request_tx.clone();
                async move {
                    request_tx.send((headers, body)).unwrap();
                    Json(json!({
                        "token_ids": [1, 2, 3],
                        "features": {"ignored": true}
                    }))
                }
            }),
        );
        let (base_url, server) = spawn_server(router).await;
        let client =
            VllmRenderClient::new(&base_url, TEST_TIMEOUT, TEST_MAX_RESPONSE_BYTES).unwrap();
        let request = Bytes::from_static(
            br#"{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hello"}]}"#,
        );

        let token_ids = client.render_chat(request.clone()).await.unwrap();

        assert_eq!(token_ids, vec![1, 2, 3]);
        let (headers, body) = request_rx.recv().await.unwrap();
        assert_eq!(body, request);
        assert_eq!(headers.get(CONTENT_TYPE).unwrap(), "application/json");
        server.abort();
    }

    #[tokio::test]
    async fn preserves_base_url_path_prefix() {
        const PREFIXED_CHAT_RENDER_PATH: &str = "/gateway/vllm/v1/chat/completions/render";

        let router = Router::new().route(
            PREFIXED_CHAT_RENDER_PATH,
            post(|| async { Json(json!({"token_ids": [4, 5]})) }),
        );
        let (base_url, server) = spawn_server(router).await;
        let client = VllmRenderClient::new(
            &format!("{base_url}/gateway/vllm/"),
            TEST_TIMEOUT,
            TEST_MAX_RESPONSE_BYTES,
        )
        .unwrap();

        let token_ids = client.render_chat(Bytes::from_static(b"{}")).await.unwrap();

        assert_eq!(token_ids, vec![4, 5]);
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
        let client = VllmRenderClient::new(&base_url, timeout, TEST_MAX_RESPONSE_BYTES).unwrap();

        let error = client
            .render_chat(Bytes::from_static(b"{}"))
            .await
            .unwrap_err();

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
        let server = tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.unwrap();
        });
        let client = VllmRenderClient::new(
            &format!("http://{address}"),
            TEST_TIMEOUT,
            TEST_MAX_RESPONSE_BYTES,
        )
        .unwrap();

        let error = client
            .render_chat(Bytes::from_static(b"{}"))
            .await
            .unwrap_err();

        assert!(matches!(error, VllmRenderError::Unavailable { .. }));
        server.await.unwrap();
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
        let client =
            VllmRenderClient::new(&base_url, TEST_TIMEOUT, TEST_MAX_RESPONSE_BYTES).unwrap();

        let error = client
            .render_chat(Bytes::from_static(b"{}"))
            .await
            .unwrap_err();

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
    async fn rejects_declared_response_larger_than_limit() {
        const RESPONSE_BODY: &[u8] = br#"{"token_ids":[1,2,3]}"#;
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(|| async {
                Response::builder()
                    .header(CONTENT_TYPE, "application/json")
                    .header("content-length", RESPONSE_BODY.len())
                    .body(Body::from(RESPONSE_BODY))
                    .unwrap()
            }),
        );
        let (base_url, server) = spawn_server(router).await;
        let limit = RESPONSE_BODY.len() - 1;
        let client = VllmRenderClient::new(&base_url, TEST_TIMEOUT, limit).unwrap();

        let error = client
            .render_chat(Bytes::from_static(b"{}"))
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            VllmRenderError::ResponseTooLarge {
                limit: actual_limit,
                received,
            } if actual_limit == limit && received == RESPONSE_BODY.len() as u64
        ));
        server.abort();
    }

    #[tokio::test]
    async fn rejects_chunked_response_larger_than_limit() {
        const FIRST_CHUNK: &[u8] = br#"{"token_ids"#;
        const SECOND_CHUNK: &[u8] = br#":[1,2,3]}"#;
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(|| async {
                Response::builder()
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from_stream(stream::iter(vec![
                        Ok::<Bytes, Infallible>(Bytes::from_static(FIRST_CHUNK)),
                        Ok::<Bytes, Infallible>(Bytes::from_static(SECOND_CHUNK)),
                    ])))
                    .unwrap()
            }),
        );
        let (base_url, server) = spawn_server(router).await;
        let limit = FIRST_CHUNK.len();
        let client = VllmRenderClient::new(&base_url, TEST_TIMEOUT, limit).unwrap();

        let error = client
            .render_chat(Bytes::from_static(b"{}"))
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            VllmRenderError::ResponseTooLarge {
                limit: actual_limit,
                received,
            } if actual_limit == limit
                && received > limit as u64
                && received <= (FIRST_CHUNK.len() + SECOND_CHUNK.len()) as u64
        ));
        server.abort();
    }

    #[tokio::test]
    async fn accepts_response_at_limit() {
        const RESPONSE_BODY: &[u8] = br#"{"token_ids":[1,2,3]}"#;
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(|| async {
                Response::builder()
                    .header(CONTENT_TYPE, "application/json")
                    .header("content-length", RESPONSE_BODY.len())
                    .body(Body::from(RESPONSE_BODY))
                    .unwrap()
            }),
        );
        let (base_url, server) = spawn_server(router).await;
        let client = VllmRenderClient::new(&base_url, TEST_TIMEOUT, RESPONSE_BODY.len()).unwrap();

        let token_ids = client.render_chat(Bytes::from_static(b"{}")).await.unwrap();

        assert_eq!(token_ids, vec![1, 2, 3]);
        server.abort();
    }

    #[tokio::test]
    async fn classifies_invalid_success_response() {
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(|| async { Json(json!({"prompt": "missing token_ids"})) }),
        );
        let (base_url, server) = spawn_server(router).await;
        let client =
            VllmRenderClient::new(&base_url, TEST_TIMEOUT, TEST_MAX_RESPONSE_BYTES).unwrap();

        let error = client
            .render_chat(Bytes::from_static(b"{}"))
            .await
            .unwrap_err();

        assert!(matches!(error, VllmRenderError::InvalidResponse { .. }));
        server.abort();
    }

    #[test]
    fn rejects_invalid_client_config() {
        assert!(
            VllmRenderClient::new(
                "unix:///tmp/vllm.sock",
                Duration::from_secs(1),
                TEST_MAX_RESPONSE_BYTES
            )
            .is_err()
        );
        assert!(
            VllmRenderClient::new(
                "http://127.0.0.1:8000",
                Duration::ZERO,
                TEST_MAX_RESPONSE_BYTES
            )
            .is_err()
        );
        assert!(VllmRenderClient::new("http://127.0.0.1:8000", TEST_TIMEOUT, 0).is_err());
    }
}
