// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `POST /tokenize` and `POST /detokenize`.
//!
//! Boots an `HttpService` with a sample-model card registered on the `ModelManager`.
//! No engine is needed: both endpoints only read the card and its tokenizer file.

use dynamo_llm::{
    http::service::service_v2::HttpService,
    model_card::ModelDeploymentCard,
    preprocessor::{OpenAIPreprocessor, prompt::prompt_formatter_from_mdc},
};
use dynamo_renderer::PromptFormatter;
use dynamo_runtime::CancellationToken;
use serde_json::{Value, json};

#[path = "common/ports.rs"]
mod ports;
use ports::bind_random_port;

/// No chat template; `max_position_embeddings` is 2048 and BOS is token id 1.
const FIXTURE_COMPLETION: &str = "tests/data/sample-models/TinyLlama_v1.1";
const MODEL_COMPLETION: &str = "tinyllama";

/// Carries a chat template in `tokenizer_config.json`.
const FIXTURE_CHAT: &str = "tests/data/sample-models/mock-llama-3.1-8b-instruct";
const MODEL_CHAT: &str = "llama3-chat";

/// tiktoken tokenizer — exercises the non-HuggingFace encoding path.
const FIXTURE_TIKTOKEN: &str = "tests/data/sample-models/mock-tiktoken";
const MODEL_TIKTOKEN: &str = "tiktoken-model";

struct Service {
    /// Kept alive for the service's lifetime: the chat template is read lazily at
    /// render time, so dropping the temp file early makes /tokenize 501.
    _chat_template: Option<tempfile::TempPath>,
    port: u16,
    cancel: CancellationToken,
    join: tokio::task::JoinHandle<anyhow::Result<()>>,
}

impl Service {
    async fn start(fixture: &str, display_name: &str) -> Self {
        Self::start_with(fixture, display_name, None, |_| {}).await
    }

    async fn start_with(
        fixture: &str,
        display_name: &str,
        chat_template: Option<tempfile::TempPath>,
        tweak: impl FnOnce(&mut ModelDeploymentCard),
    ) -> Self {
        let (listener, port) = bind_random_port().await;
        let service = HttpService::builder()
            .port(port)
            .host("127.0.0.1")
            .build()
            .expect("failed to build HTTP service");

        let mut card = ModelDeploymentCard::load_from_disk(fixture, chat_template.as_deref())
            .expect("load_from_disk");
        card.display_name = display_name.to_string();
        tweak(&mut card);

        // What the discovery watcher builds for a real model. Registering only the card
        // would not exercise the objects production uses.
        let tokenizer = card.tokenizer().expect("tokenizer");
        let chat = prompt_formatter_from_mdc(&card).ok().map(|formatter| {
            let PromptFormatter::OAI(formatter) = formatter;
            OpenAIPreprocessor::new_with_parts(card.clone(), formatter, tokenizer.clone())
                .expect("chat preprocessor")
        });
        let PromptFormatter::OAI(no_op) = PromptFormatter::no_op();
        let completions =
            OpenAIPreprocessor::new_with_parts(card.clone(), no_op, tokenizer.clone())
                .expect("completions preprocessor");

        let checksum = card.mdcsum().to_string();
        let manager = service.model_manager();
        manager
            .save_model_card("test-instance-key", card)
            .expect("save_model_card");
        manager
            .add_model_preprocessors(display_name, &checksum, chat, Some(completions))
            .expect("add_model_preprocessors");

        let cancel = CancellationToken::new();
        let join = service.spawn_with_listener(cancel.clone(), listener).await;
        Self {
            _chat_template: chat_template,
            port,
            cancel,
            join,
        }
    }

    async fn post(&self, path: &str, body: Value) -> reqwest::Response {
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client")
            .post(format!("http://127.0.0.1:{}{path}", self.port))
            .json(&body)
            .send()
            .await
            .expect("request send failed")
    }

    /// POST expecting 200, returning the decoded body.
    async fn post_ok(&self, path: &str, body: Value) -> Value {
        let resp = self.post(path, body).await;
        let status = resp.status();
        let body: Value = resp.json().await.expect("json body");
        assert_eq!(status, 200, "body: {body}");
        body
    }

    async fn shutdown(self) {
        self.cancel.cancel();
        let _ = self.join.await;
    }
}

fn tokens(body: &Value) -> Vec<u64> {
    body["tokens"]
        .as_array()
        .expect("tokens array")
        .iter()
        .map(|t| t.as_u64().expect("token id"))
        .collect()
}

#[tokio::test]
async fn tokenize_completion_defaults_to_no_special_tokens() {
    let svc = Service::start(FIXTURE_COMPLETION, MODEL_COMPLETION).await;

    let body = svc
        .post_ok(
            "/tokenize",
            json!({"model": MODEL_COMPLETION, "prompt": "Hello, world!"}),
        )
        .await;

    let tokens = tokens(&body);
    assert_eq!(body["count"].as_u64().unwrap() as usize, tokens.len());
    assert_eq!(body["max_model_len"].as_u64(), Some(2048));
    // Dynamo's /v1/completions adds no BOS, so neither does the count reported for it.
    assert_ne!(tokens.first(), Some(&1), "TinyLlama BOS must not be added");
    assert!(body.get("token_strs").is_none_or(Value::is_null));

    svc.shutdown().await;
}

#[tokio::test]
async fn tokenize_completion_rejects_add_special_tokens() {
    let svc = Service::start(FIXTURE_COMPLETION, MODEL_COMPLETION).await;

    let resp = svc
        .post(
            "/tokenize",
            json!({"model": MODEL_COMPLETION, "prompt": "hi", "add_special_tokens": true}),
        )
        .await;
    assert_eq!(resp.status(), 501);

    svc.shutdown().await;
}

#[tokio::test]
async fn tokenize_returns_token_strs() {
    let svc = Service::start(FIXTURE_COMPLETION, MODEL_COMPLETION).await;

    let body = svc
        .post_ok(
            "/tokenize",
            json!({"model": MODEL_COMPLETION, "prompt": "Hello", "return_token_strs": true}),
        )
        .await;

    let strs = body["token_strs"].as_array().expect("token_strs present");
    assert_eq!(tokens(&body).len(), strs.len());
    assert!(strs.iter().all(Value::is_string));
    // Decoded text, not the byte-level vocabulary spelling.
    assert_eq!(strs.first().and_then(Value::as_str), Some("Hello"));

    svc.shutdown().await;
}

#[tokio::test]
async fn tokenize_tiktoken_model() {
    let svc = Service::start(FIXTURE_TIKTOKEN, MODEL_TIKTOKEN).await;

    let body = svc
        .post_ok(
            "/tokenize",
            json!({"model": MODEL_TIKTOKEN, "prompt": "Hello, world!", "return_token_strs": true}),
        )
        .await;

    let tokens = tokens(&body);
    assert!(!tokens.is_empty());
    assert_eq!(
        body["token_strs"].as_array().expect("token_strs").len(),
        tokens.len(),
        "the tiktoken path decodes each id to build token_strs"
    );

    svc.shutdown().await;
}

#[tokio::test]
async fn tokenize_chat_form_applies_template() {
    let svc = Service::start(FIXTURE_CHAT, MODEL_CHAT).await;

    let chat = svc
        .post_ok(
            "/tokenize",
            json!({
                "model": MODEL_CHAT,
                "messages": [{"role": "user", "content": "Hello"}],
            }),
        )
        .await;
    let bare = svc
        .post_ok(
            "/tokenize",
            json!({"model": MODEL_CHAT, "prompt": "Hello", "add_special_tokens": false}),
        )
        .await;

    assert!(
        tokens(&chat).len() > tokens(&bare).len(),
        "chat form should add the template's tokens on top of the bare prompt"
    );

    svc.shutdown().await;
}

#[tokio::test]
async fn tokenize_chat_honors_add_generation_prompt() {
    let svc = Service::start(FIXTURE_CHAT, MODEL_CHAT).await;

    let body = json!({"model": MODEL_CHAT, "messages": [{"role": "user", "content": "Hello"}]});
    let with = svc.post_ok("/tokenize", body.clone()).await;

    let mut without = body;
    without["add_generation_prompt"] = json!(false);
    let without = svc.post_ok("/tokenize", without).await;

    assert!(
        tokens(&with).len() > tokens(&without).len(),
        "add_generation_prompt=true must append the assistant header"
    );

    svc.shutdown().await;
}

#[tokio::test]
async fn tokenize_chat_rejects_unsupported_fields() {
    let svc = Service::start(FIXTURE_CHAT, MODEL_CHAT).await;

    for extra in [
        json!({"chat_template": "{{ messages[0].content }}"}),
        json!({"mm_processor_kwargs": {"foo": 1}}),
        json!({"media_io_kwargs": {"image": {}}}),
    ] {
        let mut body =
            json!({"model": MODEL_CHAT, "messages": [{"role": "user", "content": "hi"}]});
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        let resp = svc.post("/tokenize", body).await;
        assert_eq!(resp.status(), 501, "unsupported field {extra} must not 200");
    }

    svc.shutdown().await;
}

#[tokio::test]
async fn tokenize_chat_follows_the_formatter_fallback() {
    let svc = Service::start(FIXTURE_COMPLETION, MODEL_COMPLETION).await;

    // TinyLlama's card carries no chat template. The chat pipeline still builds a
    // formatter for it, so the chat form renders rather than erroring — /tokenize
    // reports whatever that path produces instead of second-guessing it.
    let body = svc
        .post_ok(
            "/tokenize",
            json!({
                "model": MODEL_COMPLETION,
                "messages": [{"role": "user", "content": "hi"}],
            }),
        )
        .await;
    assert!(!tokens(&body).is_empty());

    svc.shutdown().await;
}

#[tokio::test]
async fn tokenize_chat_honors_continue_final_message() {
    let svc = Service::start(FIXTURE_CHAT, MODEL_CHAT).await;

    // Leaves the final turn open, so the template stops short of closing it.
    let body = json!({
        "model": MODEL_CHAT,
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "par"}],
    });
    let closed = svc.post_ok("/tokenize", body.clone()).await;

    let mut open = body;
    open["continue_final_message"] = json!(true);
    open["add_generation_prompt"] = json!(false);
    let open = svc.post_ok("/tokenize", open).await;

    assert_ne!(
        tokens(&closed),
        tokens(&open),
        "continue_final_message must reach the renderer"
    );

    svc.shutdown().await;
}

#[tokio::test]
async fn tokenize_defaults_to_the_only_registered_model() {
    let svc = Service::start(FIXTURE_COMPLETION, MODEL_COMPLETION).await;

    let body = svc.post_ok("/tokenize", json!({"prompt": "hi"})).await;
    assert!(!tokens(&body).is_empty());

    svc.shutdown().await;
}

#[tokio::test]
async fn tokenize_unknown_model_is_404() {
    let svc = Service::start(FIXTURE_COMPLETION, MODEL_COMPLETION).await;

    for path in ["/tokenize", "/detokenize"] {
        let body = match path {
            "/tokenize" => json!({"model": "does-not-exist", "prompt": "hi"}),
            _ => json!({"model": "does-not-exist", "tokens": [1, 2, 3]}),
        };
        assert_eq!(svc.post(path, body).await.status(), 404, "{path}");
    }

    svc.shutdown().await;
}

#[tokio::test]
async fn detokenize_round_trips() {
    let svc = Service::start(FIXTURE_COMPLETION, MODEL_COMPLETION).await;

    let encoded = svc
        .post_ok(
            "/tokenize",
            json!({
                "model": MODEL_COMPLETION,
                "prompt": "Hello, world!",
                "add_special_tokens": false,
            }),
        )
        .await;
    let decoded = svc
        .post_ok(
            "/detokenize",
            json!({"model": MODEL_COMPLETION, "tokens": encoded["tokens"]}),
        )
        .await;

    assert_eq!(decoded["prompt"].as_str(), Some("Hello, world!"));

    svc.shutdown().await;
}

#[tokio::test]
async fn malformed_body_is_a_json_400() {
    let svc = Service::start(FIXTURE_COMPLETION, MODEL_COMPLETION).await;

    // Matches neither arm of the untagged union.
    let resp = svc
        .post(
            "/tokenize",
            json!({"model": MODEL_COMPLETION, "text": "hi"}),
        )
        .await;
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.expect("errors must stay JSON");
    let message = body["message"].as_str().expect("message field");
    assert!(
        message.contains("`prompt`") && message.contains("`messages`"),
        "the error should name the two accepted shapes: {message}"
    );

    assert_eq!(
        svc.post("/detokenize", json!({"model": MODEL_COMPLETION}))
            .await
            .status(),
        400
    );

    svc.shutdown().await;
}

#[tokio::test]
async fn tokenize_chat_rejects_multimodal_content() {
    let svc = Service::start(FIXTURE_CHAT, MODEL_CHAT).await;

    // The template's single placeholder is nowhere near what the image expands to.
    let resp = svc
        .post(
            "/tokenize",
            json!({
                "model": MODEL_CHAT,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                ]}],
            }),
        )
        .await;
    assert_eq!(resp.status(), 501);

    // A content array that is only text parts is ordinary text and must still work.
    let body = svc
        .post_ok(
            "/tokenize",
            json!({
                "model": MODEL_CHAT,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            }),
        )
        .await;
    assert!(!tokens(&body).is_empty());

    svc.shutdown().await;
}

#[tokio::test]
async fn token_strs_is_null_rather_than_absent() {
    let svc = Service::start(FIXTURE_COMPLETION, MODEL_COMPLETION).await;

    // The field is always present, so clients can read it without a key check.
    let body = svc
        .post_ok(
            "/tokenize",
            json!({"model": MODEL_COMPLETION, "prompt": "hi"}),
        )
        .await;
    assert_eq!(body.get("token_strs"), Some(&Value::Null));

    // An explicit null must parse as "not requested".
    let body = svc
        .post_ok(
            "/tokenize",
            json!({"model": MODEL_COMPLETION, "prompt": "hi", "return_token_strs": null}),
        )
        .await;
    assert_eq!(body.get("token_strs"), Some(&Value::Null));

    svc.shutdown().await;
}

#[tokio::test]
async fn detokenize_keeps_special_tokens() {
    let svc = Service::start(FIXTURE_CHAT, MODEL_CHAT).await;

    // Special tokens are not skipped, so a chat prompt's own specials have to survive
    // the round-trip.
    let encoded = svc
        .post_ok(
            "/tokenize",
            json!({
                "model": MODEL_CHAT,
                "messages": [{"role": "user", "content": "hi"}],
                "return_token_strs": true,
            }),
        )
        .await;
    let first = encoded["token_strs"][0]
        .as_str()
        .expect("leading token str")
        .to_string();
    assert!(
        first.starts_with('<'),
        "expected a special token, got {first:?}"
    );

    let decoded = svc
        .post_ok(
            "/detokenize",
            json!({"model": MODEL_CHAT, "tokens": encoded["tokens"]}),
        )
        .await;
    assert!(
        decoded["prompt"]
            .as_str()
            .expect("prompt")
            .starts_with(&first),
        "the template's special token must survive detokenize"
    );

    svc.shutdown().await;
}

/// A chat template whose output length depends on `enable_thinking`, so a token-count
/// difference is proof the flag reached the renderer.
const THINKING_TEMPLATE: &str = r#"{% for m in messages %}{{ m['content'] }}{% endfor %}{% if enable_thinking %} thinking thinking thinking{% endif %}"#;

fn thinking_template_file() -> tempfile::TempPath {
    use std::io::Write;
    let mut f = tempfile::Builder::new()
        .suffix(".jinja")
        .tempfile()
        .expect("tempfile");
    f.write_all(THINKING_TEMPLATE.as_bytes()).expect("write");
    f.into_temp_path()
}

async fn thinking_service(default_mode: &str) -> Service {
    let mode = default_mode.to_string();
    Service::start_with(
        FIXTURE_COMPLETION,
        MODEL_COMPLETION,
        Some(thinking_template_file()),
        move |card| {
            card.runtime_config
                .runtime_data
                .insert("default_thinking_mode".to_string(), json!(mode));
        },
    )
    .await
}

#[tokio::test]
async fn tokenize_chat_applies_the_models_default_thinking_mode() {
    let body = json!({"model": MODEL_COMPLETION, "messages": [{"role": "user", "content": "hi"}]});

    // The generate path injects the model's default thinking mode when the client sends no
    // thinking control; /tokenize has to render the same prompt the model would be sent.
    let enabled = thinking_service("enabled").await;
    let on = enabled.post_ok("/tokenize", body.clone()).await;
    enabled.shutdown().await;

    let disabled = thinking_service("disabled").await;
    let off = disabled.post_ok("/tokenize", body.clone()).await;

    assert!(
        tokens(&on).len() > tokens(&off).len(),
        "default_thinking_mode must reach the template: on={} off={}",
        tokens(&on).len(),
        tokens(&off).len()
    );

    // An explicit client value still wins over the model default.
    let mut explicit = body;
    explicit["chat_template_kwargs"] = json!({"enable_thinking": true});
    let overridden = disabled.post_ok("/tokenize", explicit).await;
    assert_eq!(tokens(&overridden).len(), tokens(&on).len());

    disabled.shutdown().await;
}

#[tokio::test]
async fn tokenize_requires_the_models_pipeline_preprocessor() {
    // A card alone is not enough. Falling back to a tokenizer built here is exactly the
    // drift this plumbing exists to prevent, so the absence has to be an error.
    let (listener, port) = bind_random_port().await;
    let service = HttpService::builder()
        .port(port)
        .host("127.0.0.1")
        .build()
        .expect("build");
    let mut card = ModelDeploymentCard::load_from_disk(FIXTURE_COMPLETION, None).expect("card");
    card.display_name = MODEL_COMPLETION.to_string();
    service
        .model_manager()
        .save_model_card("card-only", card)
        .expect("save_model_card");
    let cancel = CancellationToken::new();
    let join = service.spawn_with_listener(cancel.clone(), listener).await;

    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("client");
    for (path, body) in [
        (
            "/tokenize",
            json!({"model": MODEL_COMPLETION, "prompt": "hi"}),
        ),
        (
            "/detokenize",
            json!({"model": MODEL_COMPLETION, "tokens": [1]}),
        ),
    ] {
        let resp = client
            .post(format!("http://127.0.0.1:{port}{path}"))
            .json(&body)
            .send()
            .await
            .expect("send");
        assert_eq!(resp.status(), 501, "{path}");
    }

    cancel.cancel();
    let _ = join.await;
}
