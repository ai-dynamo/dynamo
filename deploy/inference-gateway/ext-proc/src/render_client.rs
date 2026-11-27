// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP client for the engines' `POST /v1/chat/completions/render` endpoints.
//!
//! vLLM, SGLang's renderer, and TensorRT-LLM share the transport, URL handling,
//! bounded reads, and error mapping; they differ only in the response field
//! that carries the token IDs, selected by [`RendererProtocol`]. The request
//! body is forwarded unchanged and no `Authorization` header is sent.

use std::time::Duration;

use anyhow::Context;
use bytes::Bytes;
use reqwest::header::CONTENT_TYPE;
use reqwest::{Client, Url};
use serde::Deserialize;

use crate::epp_standalone_config::RendererProtocol;
use crate::render_http::{
    RenderError, check_content_length, classify_transport_error, parse_render_base_url,
    read_error_body, read_success_body,
};

const CHAT_RENDER_PATH: &str = "/v1/chat/completions/render";

/// Pooled client for one renderer.
#[derive(Clone, Debug)]
pub struct RenderClient {
    client: Client,
    endpoint: Url,
    timeout: Duration,
    max_response_bytes: usize,
    protocol: RendererProtocol,
}

impl RenderClient {
    /// Any path prefix on `base_url` is kept; the render path is appended to it.
    pub fn new(
        protocol: RendererProtocol,
        base_url: &str,
        timeout: Duration,
        max_response_bytes: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !timeout.is_zero(),
            "render timeout must be greater than zero"
        );
        anyhow::ensure!(
            max_response_bytes > 0,
            "render maximum response bytes must be greater than zero"
        );

        let mut endpoint = parse_render_base_url(base_url)?;
        {
            let mut path_segments = endpoint
                .path_segments_mut()
                .map_err(|_| anyhow::anyhow!("renderer base URL cannot be used as a base URL"))?;
            path_segments.pop_if_empty();
            path_segments.extend(CHAT_RENDER_PATH.trim_start_matches('/').split('/'));
        }
        endpoint.set_query(None);
        endpoint.set_fragment(None);

        let client = Client::builder()
            .timeout(timeout)
            .build()
            .context("building renderer HTTP client")?;

        Ok(Self {
            client,
            endpoint,
            timeout,
            max_response_bytes,
            protocol,
        })
    }

    /// Returns the prompt token IDs for an OpenAI chat-completions body.
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
        parse_token_ids(self.protocol, &body)
            .map_err(|source| RenderError::InvalidResponse { source })
    }
}

/// Typed per protocol so a TensorRT-LLM response, which echoes the whole
/// request, is never parsed into a `Value`.
fn parse_token_ids(protocol: RendererProtocol, body: &[u8]) -> Result<Vec<u32>, serde_json::Error> {
    /// vLLM's render response and TensorRT-LLM's `RenderedRequest`.
    #[derive(Deserialize)]
    struct TokenIds {
        token_ids: Vec<u32>,
    }
    /// SGLang's `GenerateRequest`.
    #[derive(Deserialize)]
    struct InputIds {
        input_ids: Vec<u32>,
    }
    match protocol {
        RendererProtocol::VllmRender | RendererProtocol::TrtllmRender => {
            serde_json::from_slice::<TokenIds>(body).map(|r| r.token_ids)
        }
        RendererProtocol::SglangRenderer => {
            serde_json::from_slice::<InputIds>(body).map(|r| r.input_ids)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::Arc;

    use axum::body::{Body, Bytes};
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::Response;
    use axum::routing::post;
    use axum::{Json, Router};
    use futures::stream;
    use serde_json::{Value, json};
    use tokio::net::TcpListener;
    use tokio::sync::mpsc;
    use tokio::task::JoinHandle;

    use super::*;

    const TEST_TIMEOUT: Duration = Duration::from_secs(5);
    const TEST_MAX_RESPONSE_BYTES: usize = 4096;

    const CHAT_BODY: &[u8] = br#"{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hello"}],"nvext":{"agent_hints":{"osl":32}}}"#;

    fn vllm_fixture() -> Value {
        json!({"token_ids": [1, 2, 3], "features": {"ignored": true}})
    }

    /// SGLang returns its full `GenerateRequest`.
    fn sglang_fixture() -> Value {
        json!({
            "rid": "test-rid",
            "input_ids": [4, 5, 6],
            "sampling_params": {},
            "stream": false,
            "return_logprob": false,
            "logprob_start_len": 0,
            "top_logprobs_num": 0,
            "return_hidden_states": false
        })
    }

    /// TensorRT-LLM's `RenderedRequest` echoes the request it rendered.
    fn trtllm_fixture() -> Value {
        json!({
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
        })
    }

    fn engine_fixtures() -> Vec<(RendererProtocol, Value, Vec<u32>)> {
        vec![
            (RendererProtocol::VllmRender, vllm_fixture(), vec![1, 2, 3]),
            (
                RendererProtocol::SglangRenderer,
                sglang_fixture(),
                vec![4, 5, 6],
            ),
            (
                RendererProtocol::TrtllmRender,
                trtllm_fixture(),
                vec![151644, 8948, 198, 2610],
            ),
        ]
    }

    async fn spawn_server(router: Router) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        (format!("http://{address}"), task)
    }

    fn fixture_router(response: Value) -> (Router, mpsc::UnboundedReceiver<(HeaderMap, Bytes)>) {
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        let router = Router::new().route(
            CHAT_RENDER_PATH,
            post(move |headers: HeaderMap, body: Bytes| {
                let request_tx = request_tx.clone();
                let response = response.clone();
                async move {
                    request_tx.send((headers, body)).unwrap();
                    Json(response)
                }
            }),
        );
        (router, request_rx)
    }

    fn client(protocol: RendererProtocol, base_url: &str) -> RenderClient {
        RenderClient::new(protocol, base_url, TEST_TIMEOUT, TEST_MAX_RESPONSE_BYTES).unwrap()
    }

    #[tokio::test]
    async fn each_protocol_extracts_token_ids_from_its_engine_response() {
        for (protocol, fixture, expected) in engine_fixtures() {
            let (router, mut request_rx) = fixture_router(fixture);
            let (base_url, server) = spawn_server(router).await;
            let request = Bytes::from_static(CHAT_BODY);

            let token_ids = client(protocol, &base_url)
                .render_chat(request.clone())
                .await
                .unwrap_or_else(|e| panic!("{protocol:?}: {e}"));

            assert_eq!(token_ids, expected, "{protocol:?}");
            let (headers, body) = request_rx.recv().await.unwrap();
            assert_eq!(body, request, "{protocol:?}");
            assert_eq!(headers.get(CONTENT_TYPE).unwrap(), "application/json");
            server.abort();
        }
    }

    #[tokio::test]
    async fn rejects_another_engines_response_shape() {
        let (router, _rx) = fixture_router(sglang_fixture());
        let (base_url, server) = spawn_server(router).await;

        let error = client(RendererProtocol::TrtllmRender, &base_url)
            .render_chat(Bytes::from_static(b"{}"))
            .await
            .unwrap_err();

        assert!(
            matches!(error, RenderError::InvalidResponse { .. }),
            "{error:?}"
        );
        server.abort();
    }

    #[tokio::test]
    async fn preserves_base_url_path_prefix() {
        const PREFIXED_CHAT_RENDER_PATH: &str = "/gateway/render/v1/chat/completions/render";

        let router = Router::new().route(
            PREFIXED_CHAT_RENDER_PATH,
            post(|| async { Json(json!({"token_ids": [4, 5]})) }),
        );
        let (base_url, server) = spawn_server(router).await;

        let token_ids = client(
            RendererProtocol::VllmRender,
            &format!("{base_url}/gateway/render/"),
        )
        .render_chat(Bytes::from_static(b"{}"))
        .await
        .unwrap();

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
        let client = RenderClient::new(
            RendererProtocol::TrtllmRender,
            &base_url,
            timeout,
            TEST_MAX_RESPONSE_BYTES,
        )
        .unwrap();

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

    #[tokio::test]
    async fn classifies_unavailable_renderer() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.unwrap();
        });

        let error = client(
            RendererProtocol::SglangRenderer,
            &format!("http://{address}"),
        )
        .render_chat(Bytes::from_static(b"{}"))
        .await
        .unwrap_err();

        assert!(matches!(error, RenderError::Unavailable { .. }));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn classifies_upstream_status_and_bounds_error_body() {
        let response_body = Arc::new("x".repeat(TEST_MAX_RESPONSE_BYTES * 2));
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

        let error = client(RendererProtocol::VllmRender, &base_url)
            .render_chat(Bytes::from_static(b"{}"))
            .await
            .unwrap_err();

        match error {
            RenderError::UpstreamStatus { status, body } => {
                assert_eq!(status, reqwest::StatusCode::SERVICE_UNAVAILABLE);
                // Error bodies are capped independently of the success limit.
                assert_eq!(body.len(), 1024);
            }
            other => panic!("expected upstream status error, got {other:?}"),
        }
        server.abort();
    }

    #[tokio::test]
    async fn surfaces_renderer_validation_error_body() {
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

        let error = client(RendererProtocol::TrtllmRender, &base_url)
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
        let client =
            RenderClient::new(RendererProtocol::VllmRender, &base_url, TEST_TIMEOUT, limit)
                .unwrap();

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
        let client =
            RenderClient::new(RendererProtocol::VllmRender, &base_url, TEST_TIMEOUT, limit)
                .unwrap();

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
        let client = RenderClient::new(
            RendererProtocol::TrtllmRender,
            &base_url,
            TEST_TIMEOUT,
            RESPONSE_BODY.len(),
        )
        .unwrap();

        let token_ids = client.render_chat(Bytes::from_static(b"{}")).await.unwrap();

        assert_eq!(token_ids, vec![1, 2, 3]);
        server.abort();
    }

    #[test]
    fn rejects_invalid_client_config() {
        let protocol = RendererProtocol::TrtllmRender;
        assert!(
            RenderClient::new(
                protocol,
                "unix:///tmp/render.sock",
                Duration::from_secs(1),
                TEST_MAX_RESPONSE_BYTES
            )
            .is_err()
        );
        assert!(
            RenderClient::new(
                protocol,
                "http://127.0.0.1:8000",
                Duration::ZERO,
                TEST_MAX_RESPONSE_BYTES
            )
            .is_err()
        );
        assert!(RenderClient::new(protocol, "http://127.0.0.1:8000", TEST_TIMEOUT, 0).is_err());
    }
}
