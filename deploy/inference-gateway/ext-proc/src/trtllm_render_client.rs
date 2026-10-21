// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP client for TensorRT-LLM's chat render endpoint.
//!
//! The client targets `POST /v1/chat/completions/render`, served either by the
//! CPU-only `trtllm-render` process (`python -m tensorrt_llm.serve.render`,
//! which loads only the tokenizer and model configuration) or by a
//! `trtllm-serve` worker started with `TRTLLM_ENABLE_RENDER_ENDPOINTS=1`.
//! The endpoint accepts an OpenAI chat-completions JSON body and returns a
//! `RenderedRequest`; this client extracts its `token_ids` field.
//!
//! Two protocol details matter to the EPP:
//!
//! - The renderer echoes the original request body in the response's `request`
//!   field, so a success response is at least as large as the request it
//!   rendered. Size `DYN_EPP_TOKENIZER_MAX_RESPONSE_BYTES` with that in mind.
//! - The renderer strips the Dynamo-only top-level fields (`nvext`,
//!   `cache_namespace`) before validating the body, so a Dynamo-style request
//!   renders without the EPP rewriting it.
//!
//! The client sends no `Authorization` header, targeting unauthenticated
//! in-cluster renderers. The body is forwarded unchanged so the renderer
//! remains responsible for chat template application and tokenization.

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

/// A reusable client for TensorRT-LLM's `/v1/chat/completions/render` endpoint.
#[derive(Clone, Debug)]
pub struct TrtllmRenderClient {
    client: Client,
    endpoint: Url,
    timeout: Duration,
    max_response_bytes: usize,
}

/// Subset of TensorRT-LLM's `RenderedRequest` response.
///
/// The renderer returns the rendered prompt plus bookkeeping (`kind`,
/// `request_id`, `cache_salt`, reasoning fields, and the echoed `request`).
/// Only the token IDs are needed for routing; the rest is ignored.
#[derive(Debug, Deserialize)]
struct TrtllmRenderResponse {
    token_ids: Vec<u32>,
}

impl TrtllmRenderClient {
    /// Build a pooled HTTP client from the TensorRT-LLM renderer's base URL.
    ///
    /// The base URL selects either a local sidecar (for example,
    /// `http://127.0.0.1:8100`) or an external Service. The chat render path
    /// is appended by this client.
    pub fn new(
        base_url: &str,
        timeout: Duration,
        max_response_bytes: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !timeout.is_zero(),
            "TensorRT-LLM render timeout must be greater than zero"
        );
        anyhow::ensure!(
            max_response_bytes > 0,
            "TensorRT-LLM render maximum response bytes must be greater than zero"
        );

        let mut endpoint = crate::render_http::parse_render_base_url(base_url)?;
        {
            let mut path_segments = endpoint.path_segments_mut().map_err(|_| {
                anyhow::anyhow!("TensorRT-LLM renderer base URL cannot be used as a base URL")
            })?;
            path_segments.pop_if_empty();
            path_segments.extend(CHAT_RENDER_PATH.trim_start_matches('/').split('/'));
        }
        endpoint.set_query(None);
        endpoint.set_fragment(None);

        let client = Client::builder()
            .timeout(timeout)
            .build()
            .context("building TensorRT-LLM renderer HTTP client")?;

        Ok(Self {
            client,
            endpoint,
            timeout,
            max_response_bytes,
        })
    }

    /// Forward an OpenAI chat-completions JSON body and return its prompt token IDs.
    ///
    /// The body is sent unchanged so the renderer remains responsible for
    /// validating engine-specific request fields and applying the chat template.
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
        serde_json::from_slice::<TrtllmRenderResponse>(&body)
            .map(|r| r.token_ids)
            .map_err(|source| RenderError::InvalidResponse { source })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::body::Bytes;
    use axum::http::HeaderMap;
    use axum::routing::post;
    use axum::{Json, Router, http::StatusCode};
    use serde_json::json;
    use tokio::net::TcpListener;
    use tokio::sync::mpsc;
    use tokio::task::JoinHandle;

    use super::*;
    use crate::render_http::RenderError;

    const TEST_TIMEOUT: Duration = Duration::from_secs(5);
    const TEST_MAX_RESPONSE_BYTES: usize = 4096;

    async fn spawn_server(router: Router) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        (format!("http://{address}"), task)
    }

    #[tokio::test]
    async fn extracts_token_ids_from_rendered_request_response() {
        let (request_tx, mut request_rx) = mpsc::unbounded_channel();
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(move |headers: HeaderMap, body: Bytes| {
                let request_tx = request_tx.clone();
                async move {
                    request_tx.send((headers, body)).unwrap();
                    // The documented `RenderedRequest` shape: ids plus bookkeeping
                    // and the echoed request body, all of which must be ignored.
                    Json(json!({
                        "schema_version": 1,
                        "kind": "chat",
                        "request_id": "chatcmpl-9651e194058642d9b538b695bb179c68",
                        "token_ids": [151644, 8948, 198, 2610],
                        "model": "Qwen/Qwen3-0.6B",
                        "reasoning_ended": null,
                        "reasoning_parser_kwargs": {"chat_template_kwargs": {}},
                        "cache_salt": null,
                        "request": {
                            "model": "Qwen/Qwen3-0.6B",
                            "messages": [{"role": "user", "content": "hello"}]
                        }
                    }))
                }
            }),
        );
        let (base_url, server) = spawn_server(router).await;
        let client =
            TrtllmRenderClient::new(&base_url, TEST_TIMEOUT, TEST_MAX_RESPONSE_BYTES).unwrap();
        let request = Bytes::from_static(
            br#"{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hello"}],"nvext":{"agent_hints":{"osl":32}}}"#,
        );

        let token_ids = client.render_chat(request.clone()).await.unwrap();

        assert_eq!(token_ids, vec![151644, 8948, 198, 2610]);
        // The body (including Dynamo-only fields) is forwarded unchanged; the
        // renderer, not the EPP, strips `nvext`.
        let (headers, body) = request_rx.recv().await.unwrap();
        assert_eq!(body, request);
        assert_eq!(headers.get(CONTENT_TYPE).unwrap(), "application/json");
        server.abort();
    }

    #[tokio::test]
    async fn preserves_base_url_path_prefix() {
        const PREFIXED_CHAT_RENDER_PATH: &str = "/gateway/trtllm/v1/chat/completions/render";

        let router = Router::new().route(
            PREFIXED_CHAT_RENDER_PATH,
            post(|| async { Json(json!({"token_ids": [4, 5], "kind": "chat"})) }),
        );
        let (base_url, server) = spawn_server(router).await;
        let client = TrtllmRenderClient::new(
            &format!("{base_url}/gateway/trtllm/"),
            TEST_TIMEOUT,
            TEST_MAX_RESPONSE_BYTES,
        )
        .unwrap();

        let token_ids = client.render_chat(Bytes::from_static(b"{}")).await.unwrap();

        assert_eq!(token_ids, vec![4, 5]);
        server.abort();
    }

    #[tokio::test]
    async fn classifies_upstream_error_status() {
        // The renderer answers a body its request model rejects with an
        // OpenAI-shaped 400 error.
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(|| async {
                (
                    StatusCode::BAD_REQUEST,
                    Json(json!({
                        "message": "1 validation error for ChatCompletionRequest",
                        "type": "BadRequestError",
                        "code": 400
                    })),
                )
            }),
        );
        let (base_url, server) = spawn_server(router).await;
        let client =
            TrtllmRenderClient::new(&base_url, TEST_TIMEOUT, TEST_MAX_RESPONSE_BYTES).unwrap();

        let error = client
            .render_chat(Bytes::from_static(b"{}"))
            .await
            .unwrap_err();

        match error {
            RenderError::UpstreamStatus { status, body } => {
                assert_eq!(status, reqwest::StatusCode::BAD_REQUEST);
                assert!(body.contains("BadRequestError"), "{body}");
            }
            other => panic!("expected upstream status error, got {other:?}"),
        }
        server.abort();
    }

    #[tokio::test]
    async fn classifies_invalid_response() {
        // The SGLang renderer's `input_ids` shape is not this contract.
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(|| async { Json(json!({"input_ids": [1, 2, 3]})) }),
        );
        let (base_url, server) = spawn_server(router).await;
        let client =
            TrtllmRenderClient::new(&base_url, TEST_TIMEOUT, TEST_MAX_RESPONSE_BYTES).unwrap();

        let error = client
            .render_chat(Bytes::from_static(b"{}"))
            .await
            .unwrap_err();

        assert!(matches!(error, RenderError::InvalidResponse { .. }));
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
        let client = TrtllmRenderClient::new(&base_url, timeout, TEST_MAX_RESPONSE_BYTES).unwrap();

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
            TrtllmRenderClient::new(
                "unix:///tmp/trtllm.sock",
                Duration::from_secs(1),
                TEST_MAX_RESPONSE_BYTES
            )
            .is_err()
        );
        assert!(
            TrtllmRenderClient::new(
                "http://127.0.0.1:8100",
                Duration::ZERO,
                TEST_MAX_RESPONSE_BYTES
            )
            .is_err()
        );
        assert!(TrtllmRenderClient::new("http://127.0.0.1:8100", TEST_TIMEOUT, 0).is_err());
    }
}
