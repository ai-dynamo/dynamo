// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end HTTP test for the `/v1/responses` deserialization path against
// real Axum routing — pins the AIPerf-shaped multimodal payload (issue #9468).
//
// Pre-patch (before EasyInputMessage / EasyInputContent were shadowed in
// `dynamo_protocols::types::responses`), the JSON body below would land on
// upstream's strict `InputImageContent` via the `EasyMessage` fallback variant
// and Axum's `Json` extractor would return 400 with
// "data did not match any variant of untagged enum InputItem".
//
// Post-patch, the body deserializes cleanly and the request flows into the
// handler. The stub chat engine then short-circuits with a controlled error,
// proving the request shape was accepted *and* routed end-to-end.

use anyhow::Error;
use async_stream::stream;
use dynamo_llm::{
    http::service::service_v2::HttpService,
    model_card::ModelDeploymentCard,
    protocols::{
        Annotated,
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
        },
    },
};
use dynamo_runtime::{
    CancellationToken,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn, async_trait,
    },
};
use reqwest::StatusCode;
use std::sync::Arc;

#[path = "common/ports.rs"]
mod ports;
use ports::get_random_port;

/// Trivial chat engine: returns one canned choice and exits. We don't care
/// about the response body — only that the request body was accepted.
struct EchoEngine;

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for EchoEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();
        let mut generator = request.response_generator(ctx.id().to_string());
        let stream = stream! {
            let output = generator.create_choice(0, Some("ok".to_string()), None, None, None, None);
            yield Annotated::from_data(output);
        };
        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

/// reqwest client that bypasses any system HTTP_PROXY — local-loopback
/// requests through a forward proxy come back as 405, masking real test
/// behavior.
fn client_no_proxy() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("reqwest client build")
}

async fn boot_service_with_model(model_name: &str) -> (u16, CancellationToken) {
    let port = get_random_port().await;
    let service = HttpService::builder()
        .port(port)
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(true)
        .build()
        .expect("HttpService build");

    let state = service.state_clone();
    let manager = state.manager();

    let token = CancellationToken::new();
    let run_token = token.clone();
    tokio::spawn(async move {
        let _ = service.run(run_token).await;
    });

    // Wait for the service to be ready by checking /health (mirrors the
    // existing wait_for_service_ready helper in http-service.rs).
    let start = tokio::time::Instant::now();
    let deadline = std::time::Duration::from_secs(5);
    let probe = client_no_proxy();
    loop {
        if probe
            .get(format!("http://localhost:{port}/health"))
            .send()
            .await
            .is_ok()
        {
            break;
        }
        assert!(
            start.elapsed() < deadline,
            "service on port {port} never came up"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    let card = ModelDeploymentCard::with_name_only(model_name);
    manager
        .add_chat_completions_model(model_name, card.mdcsum(), Arc::new(EchoEngine))
        .expect("register chat model");

    (port, token)
}

#[tokio::test]
async fn aiperf_multimodal_pre_pr931_payload_does_not_get_400() {
    // Exactly the body `_create_input_items` in
    // src/aiperf/endpoints/openai_responses.py was producing on AIPerf main
    // before PR #931 — no top-level `type` on either input item.
    let body = serde_json::json!({
        "model": "test-model",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe"},
                    {"type": "input_image", "image_url": "data:image/png;base64,abc"}
                ]
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "What about a different one?"}]
            }
        ]
    });

    let (port, token) = boot_service_with_model("test-model").await;
    let client = client_no_proxy();
    let resp = client
        .post(format!("http://localhost:{port}/v1/responses"))
        .json(&body)
        .send()
        .await
        .expect("POST /v1/responses");

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    token.cancel();

    assert_ne!(
        status,
        StatusCode::BAD_REQUEST,
        "Got 400 BAD_REQUEST — the EasyInputMessage path likely regressed. Body: {text}"
    );
    assert!(
        !text.contains("did not match any variant of untagged enum InputItem"),
        "Body still mentions the untagged-enum failure: {text}",
    );
}

#[tokio::test]
async fn aiperf_multimodal_with_type_message_still_works() {
    // Post-PR-931 shape — `type: "message"` present on every item. This
    // already worked before the EasyMessage shadow; pinning it ensures the
    // strict `Item::Message` path didn't regress.
    let body = serde_json::json!({
        "model": "test-model",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,abc"}
                ]
            }
        ]
    });

    let (port, token) = boot_service_with_model("test-model").await;
    let client = client_no_proxy();
    let resp = client
        .post(format!("http://localhost:{port}/v1/responses"))
        .json(&body)
        .send()
        .await
        .expect("POST /v1/responses");

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    token.cancel();

    assert_ne!(
        status,
        StatusCode::BAD_REQUEST,
        "with-type path regressed: {text}"
    );
}
