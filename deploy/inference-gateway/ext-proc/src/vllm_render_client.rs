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
use reqwest::header::CONTENT_TYPE;
use reqwest::{Client, Url};
use serde::Deserialize;

use crate::render_http::{
    RenderError, check_content_length, classify_transport_error, read_error_body, read_success_body,
};

const CHAT_RENDER_PATH: &str = "/v1/chat/completions/render";

/// A reusable client for vLLM's `/v1/chat/completions/render` endpoint.
#[derive(Clone, Debug)]
pub struct VllmRenderClient {
    client: Client,
    endpoint: Url,
    timeout: Duration,
    max_response_bytes: usize,
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

        let mut endpoint = crate::render_http::parse_render_base_url(base_url)?;
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
    pub async fn render_chat(&self, request_body: Bytes) -> Result<Vec<u32>, RenderError> {
        let response = self
            .client
            .post(self.endpoint.clone())
            .header(CONTENT_TYPE, "application/json")
            .body(request_body)
            .send()
            .await
            .map_err(|e| classify_transport_error(e, self.timeout))?;

        let status = response.status();
        if !status.is_success() {
            return Err(RenderError::UpstreamStatus {
                status,
                body: read_error_body(response).await,
            });
        }

        check_content_length(&response, self.max_response_bytes)?;
        let body = read_success_body(response, self.max_response_bytes, self.timeout).await?;
        serde_json::from_slice::<VllmRenderResponse>(&body)
            .map(|r| r.token_ids)
            .map_err(|source| RenderError::InvalidResponse { source })
    }
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
    use crate::render_http::RenderError;

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
            RenderError::Timeout {
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

        assert!(matches!(error, RenderError::Unavailable { .. }));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn classifies_upstream_status_and_bounds_body() {
        let response_body = Arc::new("x".repeat(TEST_MAX_RESPONSE_BYTES * 2));
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(move || {
                let response_body = response_body.clone();
                async move {
                    (
                        axum::http::StatusCode::SERVICE_UNAVAILABLE,
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
            RenderError::UpstreamStatus { status, body } => {
                assert_eq!(status, reqwest::StatusCode::SERVICE_UNAVAILABLE);
                assert_eq!(body.len(), TEST_MAX_RESPONSE_BYTES);
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
            RenderError::ResponseTooLarge {
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
            RenderError::ResponseTooLarge {
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

        assert!(matches!(error, RenderError::InvalidResponse { .. }));
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
