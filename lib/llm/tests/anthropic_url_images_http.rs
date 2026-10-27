// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real Messages HTTP handlers with a scripted backend using the shared media
//! fetcher and decoder. No GPU, NIXL transfer, public-network fixture, or model inference.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use dynamo_llm::preprocessor::media::{Decoder, EncodedMediaData, ImageDecoder, MediaFetcher};
use dynamo_llm::protocols::{
    Annotated,
    openai::chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
    },
};
use dynamo_runtime::config::environment_names::llm::{
    DYN_ENABLE_ANTHROPIC_API, DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
};
use dynamo_runtime::error::{DynamoError, ErrorType};
use dynamo_runtime::pipeline::{AsyncEngine, ManyOut, SingleIn, async_trait};
use serde_json::{Value, json};
use serial_test::serial;
use tokio::sync::Mutex;

#[allow(dead_code)]
#[path = "common/http_harness.rs"]
mod http_harness;
#[path = "common/ports.rs"]
mod ports;
#[allow(dead_code)]
#[path = "common/scripted_chat_engine.rs"]
mod scripted_chat_engine;

use http_harness::{HarnessService, MODEL, load_agent_fixture};
use scripted_chat_engine::ScriptedChatEngine;

const ENV: [(&str, Option<&str>); 2] = [
    (DYN_ENABLE_ANTHROPIC_API, Some("1")),
    (DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0")),
];
const PNG: &[u8] = include_bytes!("data/media/llm-optimize-deploy-graphic.png");

struct MediaEngine {
    next: Arc<ScriptedChatEngine>,
    fetcher: MediaFetcher,
    client: reqwest::Client,
    decoder: ImageDecoder,
    images: Mutex<Vec<Vec<u8>>>,
}

fn invalid_image() -> anyhow::Error {
    // As in the production media-loader boundary, do not expose URLs or payloads.
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(
            "Unable to load image: check URL policy, availability, format, and configured limits",
        )
        .build()
        .into()
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        anyhow::Error,
    > for MediaEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>> {
        // Inspect the actual converted chat message shape, including tool messages.
        let messages = serde_json::to_value(&request.inner.messages)?;
        for message in messages.as_array().unwrap() {
            if let Some(parts) = message["content"].as_array() {
                for part in parts {
                    if part["type"] != "image_url" {
                        continue;
                    }
                    let url = url::Url::parse(part["image_url"]["url"].as_str().unwrap())?;
                    self.fetcher
                        .check_if_url_allowed_with_dns(&url)
                        .await
                        .map_err(|_| invalid_image())?;
                    let bytes = EncodedMediaData::from_url(&url, &self.client)
                        .await
                        .map_err(|_| invalid_image())?
                        .into_bytes()
                        .map_err(|_| invalid_image())?;
                    self.decoder
                        .decode_async(EncodedMediaData::from_bytes(bytes.clone()))
                        .await
                        .map_err(|_| invalid_image())?;
                    self.images.lock().await.push(bytes);
                }
            }
        }
        self.next.generate(request).await
    }
}

async fn service(
    fetcher: MediaFetcher,
    decoder: ImageDecoder,
    scripts: usize,
) -> (HarnessService, Arc<MediaEngine>) {
    let script = load_agent_fixture("text.sse").await.unwrap();
    let next = Arc::new(ScriptedChatEngine::new(
        std::iter::repeat_n(script, scripts).map(Ok),
    ));
    let engine = Arc::new(MediaEngine {
        next: next.clone(),
        client: fetcher.build_http_client().unwrap(),
        fetcher,
        decoder,
        images: Mutex::new(Vec::new()),
    });
    (
        HarnessService::start_with_chat_engine(next, engine.clone()).await,
        engine,
    )
}

fn local_policy() -> MediaFetcher {
    MediaFetcher {
        allow_direct_ip: true,
        allow_direct_port: true,
        allow_private_ips: true,
        allowed_media_domains: Some(["127.0.0.1".to_string()].into()),
        timeout: Some(Duration::from_secs(1)),
        ..Default::default()
    }
}

fn body(source: Value, stream: bool, tool: bool) -> Value {
    let image = json!({"type": "image", "source": source});
    let content =
        json!([{"type": "text", "text": "before"}, image, {"type": "text", "text": "after"}]);
    let messages = if tool {
        json!([
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_image", "name": "read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_image", "content": content}]}
        ])
    } else {
        json!([{"role": "user", "content": content}])
    };
    json!({"model": MODEL, "max_tokens": 64, "stream": stream, "messages": messages})
}

async fn post(svc: &HarnessService, body: Value) -> reqwest::Response {
    svc.client
        .post(format!("{}/v1/messages", svc.base_url))
        .timeout(Duration::from_secs(10))
        .json(&body)
        .send()
        .await
        .unwrap()
}

#[tokio::test]
#[serial]
async fn url_and_base64_sources_deliver_the_same_image_in_messages_and_tool_results() {
    temp_env::async_with_vars(ENV, async {
        let mut server = mockito::Server::new_async().await;
        let image = server.mock("GET", "/image.png").with_status(200).with_body(PNG).expect(4).create_async().await;
        let (svc, media) = service(local_policy(), ImageDecoder::default(), 8).await;
        let url = format!("{}/image.png", server.url());
        for stream in [false, true] {
            for tool in [false, true] {
                for source in [json!({"type": "url", "url": url}), json!({"type": "base64", "media_type": "image/png", "data": STANDARD.encode(PNG)})] {
                    let response = post(&svc, body(source, stream, tool)).await;
                    assert_eq!(response.status(), reqwest::StatusCode::OK);
                    let _ = response.text().await.unwrap();
                    let requests = svc.engine.take_requests().await;
                    assert_eq!(requests.len(), 1);
                    let messages = serde_json::to_value(&requests[0].inner.messages).unwrap();
                    let message = &messages[usize::from(tool)];
                    let parts = message["content"].as_array().unwrap();
                    assert_eq!(parts.len(), 3);
                    assert_eq!(parts[0]["text"], "before");
                    assert_eq!(parts[1]["type"], "image_url");
                    assert_eq!(parts[2]["text"], "after");
                    if tool { assert_eq!(message["tool_call_id"], "toolu_image"); }
                }
            }
        }
        let loaded = media.images.lock().await;
        assert_eq!(loaded.len(), 8);
        assert!(loaded.iter().all(|bytes| bytes == PNG));
        drop(loaded);
        image.assert_async().await;
        assert_eq!(svc.engine.remaining_scripts().await, 0);
        svc.shutdown().await;
    }).await;
}

#[tokio::test]
#[serial]
async fn malformed_sources_reject_before_media_or_generation_without_exposing_urls() {
    temp_env::async_with_vars(ENV, async {
        let (svc, media) = service(local_policy(), ImageDecoder::default(), 1).await;
        let cases = [
            json!({"type": "url"}), json!({"type": "url", "url": null}),
            json!({"type": "url", "url": 42}), json!({"type": "url", "url": "/private-token"}),
            json!({"type": "url", "url": "file:///private-token"}),
            json!({"type": "url", "url": "data:image/png;base64,private-token"}),
            json!({"type": "url", "url": "https://user:private-token@example.test/image.png"}),
            json!({"type": "url", "url": "https://example.test/image.png?private-token", "data": "mixed"}),
            json!({"type": "base64", "url": "https://example.test/private-token"}),
            json!({"type": "file", "file_id": "private-token"}),
        ];
        for stream in [false, true] {
            for tool in [false, true] {
                for source in &cases {
                    let response = post(&svc, body(source.clone(), stream, tool)).await;
                    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST, "{source}");
                    assert!(!response.text().await.unwrap().contains("private-token"));
                }
            }
        }
        assert!(media.images.lock().await.is_empty());
        assert!(svc.engine.take_requests().await.is_empty());
        assert_eq!(svc.engine.remaining_scripts().await, 1);
        svc.shutdown().await;
    }).await;
}

#[tokio::test]
#[serial]
async fn media_policy_fetch_decode_and_size_failures_reject_unary_and_streaming() {
    temp_env::async_with_vars(ENV, async {
        let mut server = mockito::Server::new_async().await;
        let missing = server
            .mock("GET", "/missing")
            .with_status(404)
            .expect(4)
            .create_async()
            .await;
        let invalid = server
            .mock("GET", "/invalid")
            .with_status(200)
            .with_body("not an image")
            .expect(4)
            .create_async()
            .await;
        let redirect = server
            .mock("GET", "/redirect")
            .with_status(302)
            .with_header("location", "http://169.254.169.254/private-token")
            .expect(4)
            .create_async()
            .await;
        let oversized = server
            .mock("GET", "/large")
            .with_status(200)
            .with_body(PNG)
            .expect(4)
            .create_async()
            .await;
        let stalled = server
            .mock("GET", "/stalled")
            .with_status(200)
            .with_chunked_body(|writer| {
                std::thread::sleep(Duration::from_secs(2));
                writer.write_all(PNG)
            })
            .expect(4)
            .create_async()
            .await;
        let (svc, media) = service(
            local_policy(),
            serde_json::from_value(json!({"limits": {"max_image_width": 1}})).unwrap(),
            1,
        )
        .await;
        for stream in [false, true] {
            for tool in [false, true] {
                for path in ["missing", "invalid", "redirect", "large", "stalled"] {
                    let response = post(
                        &svc,
                        body(
                            json!({"type": "url", "url": format!("{}/{path}", server.url())}),
                            stream,
                            tool,
                        ),
                    )
                    .await;
                    assert_eq!(
                        response.status(),
                        reqwest::StatusCode::BAD_REQUEST,
                        "{path}"
                    );
                    let error = response.text().await.unwrap();
                    assert!(error.contains("invalid_request_error"), "{error}");
                    assert!(
                        !error.contains(&server.url()) && !error.contains("private-token"),
                        "{error}"
                    );
                }
            }
        }
        for mock in [missing, invalid, redirect, oversized, stalled] {
            mock.assert_async().await;
        }
        assert!(media.images.lock().await.is_empty());
        assert!(svc.engine.take_requests().await.is_empty());
        assert_eq!(svc.engine.remaining_scripts().await, 1);
        svc.shutdown().await;

        let (svc, _) = service(MediaFetcher::default(), ImageDecoder::default(), 1).await;
        for stream in [false, true] {
            let response = post(
                &svc,
                body(
                    json!({"type": "url", "url": "http://169.254.169.254/private-token"}),
                    stream,
                    false,
                ),
            )
            .await;
            assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
            assert!(!response.text().await.unwrap().contains("private-token"));
        }
        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}
