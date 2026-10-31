// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real HTTP requests with tiny local tokenizer/template fixtures and a backend
//! that captures the generation prompt, then stops before model execution.

use super::*;
use std::sync::Mutex;
use std::time::Duration;

use dynamo_runtime::CancellationToken;
use dynamo_runtime::engine::AsyncEngine;
use dynamo_runtime::pipeline::{ManyOut, Operator, SingleIn};
use serde_json::{Value, json};
use tokio::sync::watch;

use crate::discovery::WorkerSet;
use crate::local_model::runtime_config::{TOKEN_BUDGET_RUNTIME_KEY, TokenBudget};
use crate::model_card::ModelDeploymentCard;
use crate::preprocessor::OpenAIPreprocessor;
use crate::protocols::common::{llm_backend::BackendOutput, preprocessor::PreprocessedRequest};
use crate::worker_type::WorkerType;

const TEMPLATE: &str = concat!(
    "{% if tools %}tools: {{ tools | tojson }}\n{% endif %}",
    "{% for message in messages %}{{ message.role }}: {{ message.content or '' }}",
    "{% if message.tool_calls %} {{ message.tool_calls | tojson }}{% endif %}",
    "{% if message.reasoning_content %} {{ message.reasoning_content }}{% endif %}{{ '\n' }}{% endfor %}",
    "assistant:{% if enable_thinking %} thinking{% endif %}"
);

#[derive(Default)]
struct RecordingBackend(Mutex<Vec<PreprocessedRequest>>);

#[async_trait::async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, anyhow::Error>
    for RecordingBackend
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> anyhow::Result<ManyOut<Annotated<BackendOutput>>> {
        self.0.lock().unwrap().push(request.into_parts().0);
        anyhow::bail!("fixture stops before inference")
    }
}

struct ChatPipeline {
    preprocessor: Arc<OpenAIPreprocessor>,
    backend: Arc<RecordingBackend>,
}

#[async_trait::async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        anyhow::Error,
    > for ChatPipeline
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> anyhow::Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>> {
        Operator::generate(self.preprocessor.as_ref(), request, self.backend.clone()).await
    }
}

struct Fixture {
    _directory: tempfile::TempDir,
    card: ModelDeploymentCard,
    preprocessor: Arc<OpenAIPreprocessor>,
}

impl Fixture {
    fn new(
        name: &str,
        prefix: &str,
        pre_tokenizer: &str,
        configure: impl FnOnce(&mut ModelDeploymentCard),
    ) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        std::fs::copy(
            source.join("config.json"),
            directory.path().join("config.json"),
        )
        .unwrap();
        std::fs::write(directory.path().join("tokenizer.json"), json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [], "normalizer": null,
            "pre_tokenizer": {"type": pre_tokenizer}, "post_processor": null, "decoder": null,
            "model": {"type":"WordLevel", "vocab": {"[UNK]":0, "hello":1, "你好":2, "👩‍💻":3, "café":4}, "unk_token":"[UNK]"}
        }).to_string()).unwrap();
        std::fs::write(
            directory.path().join("tokenizer_config.json"),
            json!({
                "chat_template": format!("{prefix}{TEMPLATE}")
            })
            .to_string(),
        )
        .unwrap();
        let mut card = ModelDeploymentCard::load_from_disk(directory.path(), None).unwrap();
        card.set_name(name);
        card.worker_type = Some(WorkerType::Aggregated);
        card.needs.clear();
        card.kv_cache_block_size = 16;
        configure(&mut card);
        let preprocessor = OpenAIPreprocessor::new(card.clone()).unwrap();
        Self {
            _directory: directory,
            card,
            preprocessor,
        }
    }
}

struct Server {
    url: String,
    client: reqwest::Client,
    backend: Arc<RecordingBackend>,
    cancel: CancellationToken,
    task: tokio::task::JoinHandle<()>,
}

impl Server {
    async fn start(
        fixtures: &[(&Fixture, bool)],
        alias: bool,
    ) -> (Self, Vec<watch::Sender<Vec<u64>>>) {
        Self::start_with_registration(fixtures, alias, false).await
    }

    async fn start_with_registration(
        fixtures: &[(&Fixture, bool)],
        alias: bool,
        in_process: bool,
    ) -> (Self, Vec<watch::Sender<Vec<u64>>>) {
        let service = service_v2::HttpService::builder().build().unwrap();
        let backend = Arc::new(RecordingBackend::default());
        let mut workers = Vec::new();
        for (fixture, countable) in fixtures {
            let name = fixture.card.name();
            let engine = Arc::new(ChatPipeline {
                preprocessor: fixture.preprocessor.clone(),
                backend: backend.clone(),
            });
            if in_process {
                service
                    .model_manager()
                    .add_chat_completions_model_with_preprocessor(
                        name,
                        fixture.card.mdcsum(),
                        engine,
                        fixture.preprocessor.clone(),
                    )
                    .unwrap();
                continue;
            }
            let mut set = WorkerSet::new(
                name.into(),
                fixture.card.mdcsum().into(),
                fixture.card.clone(),
            );
            set.chat_engine = Some(engine);
            set.chat_preprocessor = countable.then(|| fixture.preprocessor.clone());
            let (sender, receiver) = watch::channel(vec![1]);
            set.set_instance_watcher(receiver);
            assert!(service.model_manager().add_worker_set(name, name, set));
            workers.push(sender);
        }
        if alias {
            assert!(
                service
                    .model_manager()
                    .register_alias("alias", fixtures[0].0.card.name())
            );
        }
        let (_, router) = anthropic_messages_router(service.state_clone(), None, None);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let cancel = CancellationToken::new();
        let stop = cancel.clone();
        let task = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(stop.cancelled_owned())
                .await
                .unwrap();
        });
        (
            Self {
                url,
                backend,
                cancel,
                task,
                client: reqwest::Client::builder()
                    .no_proxy()
                    .timeout(Duration::from_secs(5))
                    .build()
                    .unwrap(),
            },
            workers,
        )
    }

    async fn post(&self, path: &str, body: &Value) -> (StatusCode, Value) {
        let response = self
            .client
            .post(format!("{}{path}", self.url))
            .json(body)
            .send()
            .await
            .unwrap();
        let status = response.status();
        (status, response.json().await.unwrap())
    }

    async fn count(&self, body: &Value) -> u64 {
        let before = self.backend.0.lock().unwrap().len();
        let (status, response) = self.post("/v1/messages/count_tokens", body).await;
        assert_eq!(status, StatusCode::OK, "{response}");
        assert_eq!(
            self.backend.0.lock().unwrap().len(),
            before,
            "counting must not call an engine"
        );
        response["input_tokens"].as_u64().unwrap()
    }

    async fn assert_generation_parity(&self, mut body: Value) -> u64 {
        let count = self.count(&body).await;
        body["max_tokens"] = json!(1);
        body["stream"] = json!(false);
        let before = self.backend.0.lock().unwrap().len();
        let (status, _) = self.post("/v1/messages", &body).await;
        assert_eq!(
            status,
            StatusCode::INTERNAL_SERVER_ERROR,
            "fixture must stop at captured prompt"
        );
        let requests = self.backend.0.lock().unwrap();
        assert_eq!(
            requests.len(),
            before + 1,
            "generation must reach the capture backend"
        );
        let request = requests.last().unwrap();
        let expected = request
            .mm_routing_info
            .as_ref()
            .map_or(request.token_ids.len(), |mm| mm.expanded_prompt_len);
        assert_eq!(count, expected as u64);
        count
    }

    async fn shutdown(self) {
        self.cancel.cancel();
        tokio::time::timeout(Duration::from_secs(2), self.task)
            .await
            .unwrap()
            .unwrap();
    }
}

fn body(model: &str, text: &str) -> Value {
    json!({"model": model, "messages": [{"role":"user", "content":text}]})
}

#[tokio::test]
async fn count_tokens_uses_model_tokenizer_template_and_alias() {
    let a = Fixture::new("model-a", "", "WhitespaceSplit", |_| {});
    let b = Fixture::new("model-b", "prefix prefix ", "WhitespaceSplit", |_| {});
    let c = Fixture::new("model-c", "", "Whitespace", |_| {});
    let (server, _workers) = Server::start(&[(&a, true), (&b, true), (&c, true)], true).await;
    assert_eq!(
        server
            .assert_generation_parity(body("model-a", "hello"))
            .await,
        3
    );
    assert_eq!(server.count(&body("alias", "hello")).await, 3);
    assert_eq!(
        server
            .assert_generation_parity(body("model-b", "hello"))
            .await,
        5
    );
    assert_eq!(
        server
            .assert_generation_parity(body("model-c", "hello"))
            .await,
        5
    );
    assert_eq!(
        server
            .assert_generation_parity(body("model-a", "你好 👩‍💻 café"))
            .await,
        5
    );
    server.shutdown().await;
}

#[tokio::test]
async fn count_tokens_includes_system_tools_and_history() {
    let fixture = Fixture::new("model", "", "WhitespaceSplit", |_| {});
    let (server, _workers) = Server::start(&[(&fixture, true)], false).await;
    let simple = server.count(&body("model", "hello")).await;
    let request = json!({
        "model":"model", "system":[{"type":"text", "text":"System instructions."}],
        "tools":[{"name":"read", "description":"Read the file", "input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}],
        "messages":[
            {"role":"user","content":"Read it."},
            {"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"read","input":{"path":"example.txt"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"File contents."}]}
        ]
    });
    assert!(server.assert_generation_parity(request).await > simple);
    server.shutdown().await;
}

#[tokio::test]
async fn count_tokens_honors_thinking_and_tool_choice() {
    let fixture = Fixture::new("model", "", "WhitespaceSplit", |card| {
        card.runtime_config.reasoning_parser = Some("deepseek_r1".into());
        card.runtime_config.exclude_tools_when_tool_choice_none = true;
    });
    let (server, _workers) = Server::start(&[(&fixture, true)], false).await;
    let mut request = body("model", "hello");
    let enabled = server.assert_generation_parity(request.clone()).await;
    request["thinking"] = json!({"type":"disabled"});
    assert_eq!(
        server.assert_generation_parity(request.clone()).await + 1,
        enabled
    );
    request["tools"] = json!([{"name":"read","input_schema":{"type":"object","properties":{}}}]);
    let with_tools = server.assert_generation_parity(request.clone()).await;
    request["tool_choice"] = json!({"type":"none"});
    assert!(server.assert_generation_parity(request).await < with_tools);
    server.shutdown().await;
}

#[tokio::test]
async fn count_tokens_supports_in_process_registration_without_changing_thinking_defaults() {
    let fixture = Fixture::new("model", "", "WhitespaceSplit", |card| {
        card.runtime_config.reasoning_parser = Some("deepseek_r1".into());
        card.runtime_config
            .set_engine_specific(
                crate::preprocessor::DEFAULT_THINKING_MODE_RUNTIME_KEY,
                "disabled",
            )
            .unwrap();
    });
    let (server, _) = Server::start_with_registration(&[(&fixture, true)], false, true).await;
    assert_eq!(
        server
            .assert_generation_parity(body("model", "hello"))
            .await,
        3
    );
    server.shutdown().await;
}

#[tokio::test]
async fn count_tokens_rejects_unknown_unavailable_and_backend_owned_tokenizers() {
    let fixture = Fixture::new("model", "", "WhitespaceSplit", |_| {});
    let (server, workers) = Server::start(&[(&fixture, false)], true).await;
    for (model, expected, kind) in [
        ("missing", StatusCode::NOT_FOUND, "not_found_error"),
        ("model", StatusCode::BAD_REQUEST, "invalid_request_error"),
        ("alias", StatusCode::BAD_REQUEST, "invalid_request_error"),
    ] {
        let (status, response) = server
            .post("/v1/messages/count_tokens", &body(model, "hello"))
            .await;
        assert_eq!(status, expected, "{response}");
        assert_eq!(response["type"], "error");
        assert_eq!(response["error"]["type"], kind);
    }
    workers[0].send(vec![]).unwrap();
    let (status, response) = server
        .post("/v1/messages/count_tokens", &body("alias", "hello"))
        .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{response}");
    assert_eq!(response["error"]["type"], "overloaded_error");
    assert!(server.backend.0.lock().unwrap().is_empty());
    server.shutdown().await;
}

#[tokio::test]
async fn count_tokens_counts_prompts_exceeding_generation_limit() {
    let fixture = Fixture::new("model", "", "WhitespaceSplit", |card| {
        card.runtime_config
            .set_engine_specific(
                TOKEN_BUDGET_RUNTIME_KEY,
                TokenBudget {
                    combined_limit: 2,
                    reject_prompt_overflow: true,
                    reject_total_overflow: true,
                },
            )
            .unwrap();
    });
    let (server, _workers) = Server::start(&[(&fixture, true)], false).await;
    let mut request = body("model", "hello hello hello hello");
    assert_eq!(server.count(&request).await, 6);
    request["max_tokens"] = json!(1);
    let (status, _) = server.post("/v1/messages", &request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(server.backend.0.lock().unwrap().is_empty());
    server.shutdown().await;
}

#[tokio::test]
async fn count_tokens_rejects_unaccounted_media_and_server_tools() {
    let fixture = Fixture::new("model", "", "WhitespaceSplit", |_| {});
    let (server, _workers) = Server::start(&[(&fixture, true)], false).await;
    let mut request = body("model", "hello");
    request["messages"][0]["content"] = json!([{
        "type":"image", "source":{"type":"base64","media_type":"image/png", "data":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aO/8AAAAASUVORK5CYII="}
    }]);
    let (status, response) = server.post("/v1/messages/count_tokens", &request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("media inputs"),
        "{response}"
    );
    let mut request = body("model", "hello");
    request["tools"] = json!([{"type":"web_search_20250305", "name":"web_search"}]);
    let (status, response) = server.post("/v1/messages/count_tokens", &request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("server tool")
    );
    assert!(server.backend.0.lock().unwrap().is_empty());
    server.shutdown().await;
}

#[tokio::test]
async fn count_tokens_rejects_duplicate_model_keys() {
    let fixture = Fixture::new("model", "", "WhitespaceSplit", |_| {});
    let (server, _workers) = Server::start(&[(&fixture, true)], false).await;
    let response = server
        .client
        .post(format!("{}/v1/messages/count_tokens", server.url))
        .header("content-type", "application/json")
        .body(
            r#"{"model":"missing","model":"model","messages":[{"role":"user","content":"hello"}]}"#,
        )
        .send()
        .await
        .unwrap();
    assert!(!response.status().is_success());
    assert!(server.backend.0.lock().unwrap().is_empty());
    server.shutdown().await;
}

#[cfg(feature = "mm-routing")]
#[tokio::test]
async fn count_tokens_includes_expanded_image_tokens() {
    use base64::Engine;
    let mut fixture = Fixture::new("vision", "", "WhitespaceSplit", |_| {});
    let directory = fixture._directory.path();
    let mut config: Value =
        serde_json::from_slice(&std::fs::read(directory.join("config.json")).unwrap()).unwrap();
    config["model_type"] = json!("qwen3_vl");
    config["image_token_id"] = json!(5);
    std::fs::write(directory.join("config.json"), config.to_string()).unwrap();
    std::fs::write(
        directory.join("preprocessor_config.json"),
        json!({
            "patch_size":16, "merge_size":2, "min_pixels":3136,
            "max_pixels":12845056, "temporal_patch_size":2
        })
        .to_string(),
    )
    .unwrap();
    let mut tokenizer: Value =
        serde_json::from_slice(&std::fs::read(directory.join("tokenizer.json")).unwrap()).unwrap();
    tokenizer["model"]["vocab"]["<|image_pad|>"] = json!(5);
    tokenizer["added_tokens"] = json!([{
        "id":5,"content":"<|image_pad|>","single_word":false,
        "lstrip":false,"rstrip":false,"normalized":false,"special":true
    }]);
    std::fs::write(directory.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    std::fs::write(
        directory.join("tokenizer_config.json"),
        json!({"chat_template": concat!(
            "{% for message in messages %}{{ message.role }}: ",
            "{% for part in message.content %}",
            "{% if part.type == 'image' or part.type == 'image_url' %}<|image_pad|>",
            "{% else %}{{ part.text }}{% endif %}{% endfor %}\n{% endfor %}assistant:"
        )})
        .to_string(),
    )
    .unwrap();
    fixture.card = ModelDeploymentCard::load_from_disk(directory, None).unwrap();
    fixture.card.set_name("vision");
    fixture.card.worker_type = Some(WorkerType::Aggregated);
    fixture.card.needs.clear();
    fixture.card.kv_cache_block_size = 16;
    fixture.preprocessor = OpenAIPreprocessor::new(fixture.card.clone()).unwrap();
    let (server, _workers) = Server::start(&[(&fixture, true)], false).await;
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(640, 480)
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    let image = json!({"type":"image", "source": {
        "type":"base64", "media_type":"image/png",
        "data":base64::engine::general_purpose::STANDARD.encode(png.into_inner())
    }});
    let mut request = body("vision", "");
    request["messages"][0]["content"] = json!([image]);
    // Qwen3-VL: (640/16)*(480/16)/2^2 = 300 image tokens, plus two role tokens.
    assert_eq!(server.assert_generation_parity(request.clone()).await, 302);
    request["messages"][0]["content"] = json!([image, image]);
    assert_eq!(server.assert_generation_parity(request).await, 602);
    server.shutdown().await;
}
