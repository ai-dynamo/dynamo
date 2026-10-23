// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A replacement controller serves the existing HTTP contract without registering
//! an engine or model with Dynamo's ModelManager.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use dynamo_llm::{
    controller::chat::{ChatController, ChatError, ChatModel, ChatOutput},
    http::service::openapi_docs::generate_openapi_spec,
    http::service::service_v2::HttpService,
};
use dynamo_llm::{
    protocols::{
        Annotated,
        openai::{
            ParsingOptions,
            chat_completions::{
                NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
            },
        },
    },
    request_template::RequestTemplate,
};
use dynamo_runtime::{
    CancellationToken,
    pipeline::{AsyncEngineContext, AsyncEngineContextProvider, Context, ResponseStream},
};
use serde_json::json;

#[path = "common/ports.rs"]
mod ports;

#[derive(Default)]
struct ExampleController {
    requests: Mutex<Vec<NvCreateChatCompletionRequest>>,
    contexts: Mutex<Vec<Arc<dyn AsyncEngineContext>>>,
}

#[async_trait]
impl ChatController for ExampleController {
    fn resolve_model(&self, model: &str) -> ChatModel {
        ChatModel {
            name: if model == "alias" {
                "custom-model"
            } else {
                model
            }
            .into(),
            metric_name: "custom-model".into(),
        }
    }

    fn check_model_ready(&self, model: &str) -> Result<(), ChatError> {
        if model == "warming" {
            return Err(ChatError::ModelUnavailable(model.into()));
        }
        Ok(())
    }

    async fn generate(
        &self,
        request: Context<NvCreateChatCompletionRequest>,
    ) -> Result<ChatOutput, ChatError> {
        let (request, context) = request.transfer(());
        let ctx = context.context();
        let model = request.inner.model.clone();
        self.requests.lock().unwrap().push(request);
        self.contexts.lock().unwrap().push(ctx.clone());
        if model == "missing" {
            return Err(ChatError::ModelNotFound);
        }
        if model == "broken" {
            return Err(ChatError::Generation(anyhow::anyhow!(
                "private backend failure"
            )));
        }
        let chunk: NvCreateChatCompletionStreamResponse = serde_json::from_value(json!({
            "id": "chat-custom", "object": "chat.completion.chunk", "created": 0,
            "model": model, "choices": [{"index": 0,
                "delta": {"role": "assistant", "content": "Hello from another controller"},
                "finish_reason": if model == "wait" { None } else { Some("stop") }
            }]
        }))
        .unwrap();
        let stream = async_stream::stream! {
            yield Annotated::from_data(chunk);
            if model == "wait" {
                std::future::pending::<()>().await;
            }
        };
        Ok(ChatOutput {
            stream: ResponseStream::new(Box::pin(stream), ctx),
            parsing_options: ParsingOptions::default(),
            can_defer_all_output: false,
        })
    }
}

#[tokio::test]
async fn replacement_controller_reuses_http_contract_and_cancellation() {
    let controller = Arc::new(ExampleController::default());
    let service = HttpService::builder()
        .enable_chat_endpoints(true)
        .chat_controller(controller.clone())
        .request_template(Some(RequestTemplate {
            model: "alias".into(),
            temperature: 0.5,
            max_completion_tokens: 32,
        }))
        .build()
        .unwrap();
    assert!(!service.model_manager().has_registered_model("custom-model"));

    // Controller injection must not change route metadata or public schemas.
    let default_service = HttpService::builder()
        .enable_chat_endpoints(true)
        .build()
        .unwrap();
    assert_eq!(
        generate_openapi_spec(service.route_docs()).unwrap(),
        generate_openapi_spec(default_service.route_docs()).unwrap(),
    );
    drop(default_service);

    let state = service.state_clone();
    let (listener, port) = ports::bind_random_port().await;
    let cancel = CancellationToken::new();
    let join = service.spawn_with_listener(cancel.clone(), listener).await;
    tokio::time::timeout(Duration::from_secs(5), async {
        while !state.is_ready() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let body = |model: &str, streaming: bool| {
        json!({
            "model": model, "messages": [{"role": "user", "content": "Hello"}],
            "stream": streaming,
        })
    };

    for streaming in [false, true] {
        // Empty model exercises the shared request template and alias resolution.
        let response = client
            .post(&url)
            .json(&body("", streaming))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        if streaming {
            assert!(
                response.headers()["content-type"]
                    .to_str()
                    .unwrap()
                    .starts_with("text/event-stream")
            );
            let text = response.text().await.unwrap();
            assert!(text.contains("Hello from another controller"));
            assert!(text.contains("data: [DONE]"));
            assert!(text.contains("\"model\":\"custom-model\""));
        } else {
            let response: serde_json::Value = response.json().await.unwrap();
            assert_eq!(response["model"], "custom-model");
            assert_eq!(
                response["choices"][0]["message"]["content"],
                "Hello from another controller"
            );
        }
    }
    {
        let requests = controller.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        for request in requests.iter() {
            assert_eq!(request.inner.model, "custom-model");
            assert_eq!(request.inner.temperature, Some(0.5));
            assert_eq!(request.inner.max_completion_tokens, Some(32));
        }
    }

    // Shared API validation rejects input before it reaches any controller.
    let invalid = client
        .post(&url)
        .json(&json!({"model": "alias", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(invalid.status(), 400);
    assert_eq!(controller.requests.lock().unwrap().len(), 2);
    for (model, status) in [("missing", 404), ("warming", 503), ("broken", 500)] {
        let response = client
            .post(&url)
            .json(&body(model, false))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), status);
        let error: serde_json::Value = response.json().await.unwrap();
        assert_eq!(error["code"], status);
        if model == "broken" {
            assert_ne!(error["message"], "private backend failure");
        }
    }

    // Dropping an active SSE response must cancel the controller's input context.
    let mut response = client
        .post(&url)
        .json(&body("wait", true))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert!(response.chunk().await.unwrap().is_some());
    let context = controller.contexts.lock().unwrap().last().unwrap().clone();
    drop(response);
    tokio::time::timeout(Duration::from_secs(5), async {
        while !context.is_killed() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("HTTP disconnect did not cancel the controller request");

    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), join)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}
