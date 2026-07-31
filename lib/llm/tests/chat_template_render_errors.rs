// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for chat template rendering failures over HTTP.
//!
//! Rendering only consumes the request, so a template that refuses to render is a client
//! error: the frontend must answer 400 with a JSON body on both the streaming and the
//! non-streaming path, and must not reject inputs the model's own template accepts.

use std::sync::Arc;

use anyhow::Error;
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::{BackendOutput, OpenAIPreprocessor, PreprocessedRequest};
use dynamo_llm::protocols::Annotated;
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_runtime::CancellationToken;
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, Operator, ResponseStream, SingleIn,
    async_trait,
};
use reqwest::StatusCode;

#[path = "common/ports.rs"]
mod ports;

use ports::bind_random_port;

/// mock-llama carries a chat template in tokenizer_config.json, which the preprocessor
/// needs, and that template renders any message list without inspecting roles.
const MODEL_PATH: &str = "tests/data/sample-models/mock-llama-3.1-8b-instruct";

const ACCEPTING_MODEL: &str = "accepting-template";
const REJECTING_MODEL: &str = "rejecting-template";

/// Mirrors the guard published chat templates use to refuse conversations they cannot
/// encode. `raise_exception` is minijinja's abort hook, so rendering fails mid-template.
const REQUIRES_USER_TEMPLATE: &str = "\
    {% set ns = namespace(has_user=false) %}\
    {% for message in messages %}\
        {% if message['role'] == 'user' %}{% set ns.has_user = true %}{% endif %}\
    {% endfor %}\
    {% if not ns.has_user %}{{ raise_exception('No user query found in messages.') }}{% endif %}\
    {{ messages[0]['content'] }}";

/// Terminal engine for the pipeline. The rejecting-template requests never reach it.
struct EchoBackend;

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>
    for EchoBackend
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<BackendOutput>>, Error> {
        let (_request, context) = request.transfer(());
        let ctx = context.context();

        let output = BackendOutput {
            token_ids: vec![],
            tokens: vec![],
            text: Some("ok".to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Stop),
            stop_reason: None,
            index: Some(0),
            completion_usage: None,
            disaggregated_params: None,
            encoder_result: None,
            worker_trace_link: None,
            engine_data: None,
            routing_data: None,
        };

        Ok(ResponseStream::new(
            Box::pin(futures::stream::once(async move {
                Annotated::from_data(output)
            })),
            ctx,
        ))
    }
}

/// Wires the preprocessor ahead of a backend the way the frontend does in production, so
/// chat template rendering happens inside the engine the HTTP service calls.
struct PreprocessingChatEngine {
    preprocessor: Arc<OpenAIPreprocessor>,
    backend: Arc<
        dyn AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>,
    >,
}

impl PreprocessingChatEngine {
    fn new(mdc: ModelDeploymentCard) -> Self {
        Self {
            preprocessor: OpenAIPreprocessor::new(mdc).expect("failed to build preprocessor"),
            backend: Arc::new(EchoBackend),
        }
    }
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for PreprocessingChatEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        Operator::generate(self.preprocessor.as_ref(), request, self.backend.clone()).await
    }
}

struct TestService {
    port: u16,
    client: reqwest::Client,
    cancel: CancellationToken,
    join: tokio::task::JoinHandle<Result<(), Error>>,
    // Holds the custom template on disk for the lifetime of the service.
    _template: tempfile::NamedTempFile,
}

impl TestService {
    /// Registers the same model twice, differing only in chat template: one that renders
    /// any history, and one that refuses a history without a user turn.
    async fn start() -> Self {
        let mut template = tempfile::Builder::new()
            .suffix(".jinja")
            .tempfile()
            .expect("failed to create custom template file");
        std::io::Write::write_all(&mut template, REQUIRES_USER_TEMPLATE.as_bytes())
            .expect("failed to write custom template file");

        let (listener, port) = bind_random_port().await;
        let service = HttpService::builder()
            .port(port)
            .host("127.0.0.1")
            .enable_chat_endpoints(true)
            .build()
            .expect("failed to build HTTP service");

        for (model, custom_template) in [
            (ACCEPTING_MODEL, None),
            (REJECTING_MODEL, Some(template.path())),
        ] {
            let mut mdc = ModelDeploymentCard::load_from_disk(MODEL_PATH, custom_template)
                .expect("failed to load model deployment card");
            mdc.set_name(model);
            service
                .model_manager()
                .add_chat_completions_model(
                    model,
                    mdc.mdcsum(),
                    Arc::new(PreprocessingChatEngine::new(mdc.clone())),
                )
                .expect("failed to register model");
        }

        let cancel = CancellationToken::new();
        let join = service.spawn_with_listener(cancel.clone(), listener).await;
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("failed to build HTTP client");

        let service = Self {
            port,
            client,
            cancel,
            join,
            _template: template,
        };
        service.wait_for_health().await;
        service
    }

    async fn wait_for_health(&self) {
        let url = format!("http://127.0.0.1:{}/health", self.port);
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                if self
                    .client
                    .get(&url)
                    .send()
                    .await
                    .is_ok_and(|response| response.status().is_success())
                {
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("HTTP service did not become healthy");
    }

    async fn post_assistant_only(&self, model: &str, stream: bool) -> reqwest::Response {
        self.client
            .post(format!(
                "http://127.0.0.1:{}/v1/chat/completions",
                self.port
            ))
            .json(&serde_json::json!({
                "model": model,
                "stream": stream,
                "max_tokens": 1,
                "messages": [{"role": "assistant", "content": "prefill"}]
            }))
            .send()
            .await
            .expect("POST /v1/chat/completions failed")
    }

    async fn shutdown(self) {
        self.cancel.cancel();
        self.join
            .await
            .expect("HTTP service task panicked")
            .expect("HTTP service returned an error");
    }
}

#[tokio::test]
async fn template_render_failure_returns_400_json_not_sse() {
    let service = TestService::start().await;

    // Streaming matters here: the frontend commits to HTTP 200 before the first SSE frame,
    // so a render failure has to surface as a status code rather than an error event.
    for stream in [false, true] {
        let response = service.post_assistant_only(REJECTING_MODEL, stream).await;

        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "stream={stream}"
        );
        assert_eq!(
            response.headers().get(reqwest::header::CONTENT_TYPE),
            Some(&reqwest::header::HeaderValue::from_static(
                "application/json"
            )),
            "stream={stream}"
        );

        let body: serde_json::Value = response.json().await.expect("error body was not JSON");
        assert_eq!(body["code"], 400, "stream={stream}");
        assert!(
            body["message"]
                .as_str()
                .is_some_and(|message| message.contains("No user query found in messages.")),
            "stream={stream}, body={body}"
        );
    }

    service.shutdown().await;
}

#[tokio::test]
async fn assistant_only_history_succeeds_when_template_accepts_it() {
    let service = TestService::start().await;

    let response = service.post_assistant_only(ACCEPTING_MODEL, false).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = response.json().await.expect("response body was not JSON");
    assert_eq!(body["choices"][0]["message"]["content"], "ok", "{body}");

    service.shutdown().await;
}
