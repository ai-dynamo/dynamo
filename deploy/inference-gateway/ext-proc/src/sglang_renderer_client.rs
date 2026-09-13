// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP client for SGLang's chat render endpoint.
//!
//! The client targets the `sglang-renderer` binary running in render-only mode
//! (launched without `--engine-url`), which exposes `/v1/chat/completions/render`
//! and performs CPU-only chat template application and tokenization.
//!
//! The endpoint accepts an OpenAI chat-completions JSON body and returns a
//! `GenerateRequest` object; this client extracts the `input_ids` field.
//!
//! The client sends no `Authorization` header, targeting unauthenticated
//! in-cluster sidecars. The request body is forwarded unchanged so the renderer
//! remains responsible for chat template application.

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

/// A reusable client for the sglang renderer's `/v1/chat/completions/render` endpoint.
#[derive(Clone, Debug)]
pub struct SglangRendererClient {
    client: Client,
    endpoint: Url,
    timeout: Duration,
    max_response_bytes: usize,
}

/// Subset of the sglang renderer's `GenerateRequest` response.
///
/// The renderer returns a full `GenerateRequest` object; we extract only the
/// token IDs needed for routing. The field is `input_ids` (sglang's internal
/// name) rather than vLLM render's `token_ids`.
#[derive(Debug, Deserialize)]
struct SglangRenderResponse {
    input_ids: Vec<u32>,
}

impl SglangRendererClient {
    /// Build a pooled HTTP client from the sglang renderer's base URL.
    ///
    /// The base URL selects either a local sidecar (for example,
    /// `http://127.0.0.1:30000`) or an external Service. The sglang-specific
    /// chat render path is appended by this client.
    pub fn new(
        base_url: &str,
        timeout: Duration,
        max_response_bytes: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !timeout.is_zero(),
            "SGLang render timeout must be greater than zero"
        );
        anyhow::ensure!(
            max_response_bytes > 0,
            "SGLang render maximum response bytes must be greater than zero"
        );

        let mut endpoint = crate::render_http::parse_render_base_url(base_url)?;
        {
            let mut path_segments = endpoint.path_segments_mut().map_err(|_| {
                anyhow::anyhow!("SGLang renderer base URL cannot be used as a base URL")
            })?;
            path_segments.pop_if_empty();
            path_segments.extend(CHAT_RENDER_PATH.trim_start_matches('/').split('/'));
        }
        endpoint.set_query(None);
        endpoint.set_fragment(None);

        let client = Client::builder()
            .timeout(timeout)
            .build()
            .context("building SGLang renderer HTTP client")?;

        Ok(Self {
            client,
            endpoint,
            timeout,
            max_response_bytes,
        })
    }

    /// Forward an OpenAI chat-completions JSON body and return its prompt token IDs.
    ///
    /// The body is sent unchanged so the sglang renderer remains responsible for
    /// chat template application and tokenization.
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
        serde_json::from_slice::<SglangRenderResponse>(&body)
            .map(|r| r.input_ids)
            .map_err(|source| RenderError::InvalidResponse { source })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::Json;
    use axum::routing::post;
    use axum::{Router, http::StatusCode, response::Response};
    use serde_json::json;
    use tokio::net::TcpListener;
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
    async fn extracts_input_ids_from_generate_request_response() {
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(|| async {
                Json(json!({
                    "rid": "test-rid",
                    "input_ids": [1, 2, 3],
                    "sampling_params": {},
                    "stream": false,
                    "return_logprob": false,
                    "logprob_start_len": 0,
                    "top_logprobs_num": 0,
                    "return_hidden_states": false
                }))
            }),
        );
        let (base_url, server) = spawn_server(router).await;
        let client =
            SglangRendererClient::new(&base_url, TEST_TIMEOUT, TEST_MAX_RESPONSE_BYTES).unwrap();

        let token_ids = client
            .render_chat(Bytes::from_static(
                br#"{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hello"}]}"#,
            ))
            .await
            .unwrap();

        assert_eq!(token_ids, vec![1, 2, 3]);
        server.abort();
    }

    #[tokio::test]
    async fn classifies_upstream_error_status() {
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(|| async { (StatusCode::SERVICE_UNAVAILABLE, "renderer not ready") }),
        );
        let (base_url, server) = spawn_server(router).await;
        let client =
            SglangRendererClient::new(&base_url, TEST_TIMEOUT, TEST_MAX_RESPONSE_BYTES).unwrap();

        let error = client
            .render_chat(Bytes::from_static(b"{}"))
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            RenderError::UpstreamStatus {
                status,
                ..
            } if status == reqwest::StatusCode::SERVICE_UNAVAILABLE
        ));
        server.abort();
    }

    #[tokio::test]
    async fn classifies_invalid_response() {
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(|| async { Json(json!({"token_ids": [1, 2, 3]})) }),
        );
        let (base_url, server) = spawn_server(router).await;
        let client =
            SglangRendererClient::new(&base_url, TEST_TIMEOUT, TEST_MAX_RESPONSE_BYTES).unwrap();

        // vLLM response schema (token_ids) must fail — sglang uses input_ids
        let error = client
            .render_chat(Bytes::from_static(b"{}"))
            .await
            .unwrap_err();

        assert!(matches!(error, RenderError::InvalidResponse { .. }));
        server.abort();
    }

    #[tokio::test]
    async fn classifies_timeout() {
        use std::time::Duration;
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(|| async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Response::new(axum::body::Body::empty())
            }),
        );
        let (base_url, server) = spawn_server(router).await;
        let timeout = Duration::from_millis(10);
        let client =
            SglangRendererClient::new(&base_url, timeout, TEST_MAX_RESPONSE_BYTES).unwrap();

        let error = client
            .render_chat(Bytes::from_static(b"{}"))
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            RenderError::Timeout { timeout: actual, .. } if actual == timeout
        ));
        server.abort();
    }

    #[test]
    fn rejects_invalid_client_config() {
        assert!(
            SglangRendererClient::new(
                "unix:///tmp/sglang.sock",
                Duration::from_secs(1),
                TEST_MAX_RESPONSE_BYTES
            )
            .is_err()
        );
        assert!(
            SglangRendererClient::new(
                "http://127.0.0.1:30000",
                Duration::ZERO,
                TEST_MAX_RESPONSE_BYTES
            )
            .is_err()
        );
        assert!(
            SglangRendererClient::new("http://127.0.0.1:30000", Duration::from_secs(1), 0).is_err()
        );
    }
}
