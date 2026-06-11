// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `POST /tokenize` and `POST /detokenize`.
//!
//! Spins up an `HttpService`, registers the TinyLlama sample-model card on the
//! `ModelManager` (no engine required — `/tokenize` only needs the card and its
//! tokenizer file), and exercises both request shapes plus the round-trip.

use dynamo_llm::{http::service::service_v2::HttpService, model_card::ModelDeploymentCard};
use dynamo_runtime::CancellationToken;
use serde_json::{Value, json};

#[path = "common/ports.rs"]
mod ports;
use ports::get_random_port;

const HF_PATH: &str = "tests/data/sample-models/TinyLlama_v1.1";
const MODEL_NAME: &str = "tinyllama";

const HF_PATH_CHAT: &str = "tests/data/sample-models/mock-llama-3.1-8b-instruct";
const MODEL_NAME_CHAT: &str = "llama3-chat";

async fn wait_for_service_ready(port: u16) {
    let start = tokio::time::Instant::now();
    let timeout = tokio::time::Duration::from_secs(5);
    loop {
        if let Ok(resp) = reqwest::get(&format!("http://localhost:{port}/health")).await
            && resp.status().is_success()
        {
            break;
        }
        if start.elapsed() >= timeout {
            panic!("HTTP service did not become ready");
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
}

/// Boot a service, register a real MDC for TinyLlama, return (port, cancel_token, join_handle).
async fn spawn_service_with_tinyllama() -> (u16, CancellationToken, tokio::task::JoinHandle<()>) {
    let port = get_random_port().await;
    let service = HttpService::builder().port(port).build().unwrap();
    let manager = service.state_clone().manager_clone();

    // Load the MDC from the on-disk fixture and register the display name we'll use in requests.
    let mut mdc = ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap();
    mdc.display_name = MODEL_NAME.to_string();
    manager
        .save_model_card("test-instance-key", mdc)
        .expect("save_model_card");

    let token = CancellationToken::new();
    let cancel_token = token.clone();
    let handle = tokio::spawn(async move {
        let _ = service.run(token).await;
    });

    wait_for_service_ready(port).await;
    (port, cancel_token, handle)
}

/// Boot a service registered with a model that has a chat template.
async fn spawn_service_with_chat_model() -> (u16, CancellationToken, tokio::task::JoinHandle<()>) {
    let port = get_random_port().await;
    let service = HttpService::builder().port(port).build().unwrap();
    let manager = service.state_clone().manager_clone();

    let mut mdc = ModelDeploymentCard::load_from_disk(HF_PATH_CHAT, None).unwrap();
    mdc.display_name = MODEL_NAME_CHAT.to_string();
    manager
        .save_model_card("test-chat-key", mdc)
        .expect("save_model_card");

    let token = CancellationToken::new();
    let cancel_token = token.clone();
    let handle = tokio::spawn(async move {
        let _ = service.run(token).await;
    });

    wait_for_service_ready(port).await;
    (port, cancel_token, handle)
}

/// Helper: POST JSON to a path on the running service.
async fn post_json(port: u16, path: &str, body: Value) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("http://localhost:{port}{path}"))
        .json(&body)
        .send()
        .await
        .expect("request send failed")
}

#[tokio::test]
async fn test_tokenize_completion_basic() {
    let (port, cancel, handle) = spawn_service_with_tinyllama().await;

    let resp = post_json(
        port,
        "/tokenize",
        json!({"model": MODEL_NAME, "prompt": "Hello, world!"}),
    )
    .await;
    assert_eq!(resp.status(), 200, "body: {:?}", resp.text().await);
    let body: Value = post_json(
        port,
        "/tokenize",
        json!({"model": MODEL_NAME, "prompt": "Hello, world!"}),
    )
    .await
    .json()
    .await
    .unwrap();

    let tokens = body["tokens"].as_array().expect("tokens array");
    assert!(!tokens.is_empty(), "tokenized output must be non-empty");
    assert_eq!(body["count"].as_u64().unwrap() as usize, tokens.len());
    assert_eq!(
        body["max_model_len"].as_u64(),
        Some(2048),
        "TinyLlama max_position_embeddings"
    );
    // add_special_tokens defaults to true, so the first token should be BOS (1 for TinyLlama).
    assert_eq!(tokens[0].as_u64(), Some(1), "expected BOS prefix");
    assert!(body.get("token_strs").map(|v| v.is_null()).unwrap_or(true));

    cancel.cancel();
    let _ = handle.await;
}

#[tokio::test]
async fn test_tokenize_no_special_tokens() {
    let (port, cancel, handle) = spawn_service_with_tinyllama().await;

    let body: Value = post_json(
        port,
        "/tokenize",
        json!({
            "model": MODEL_NAME,
            "prompt": "Hello, world!",
            "add_special_tokens": false,
        }),
    )
    .await
    .json()
    .await
    .unwrap();

    let tokens = body["tokens"].as_array().unwrap();
    // Without BOS, the leading token must not be 1.
    assert_ne!(tokens[0].as_u64(), Some(1));

    cancel.cancel();
    let _ = handle.await;
}

#[tokio::test]
async fn test_tokenize_returns_token_strs() {
    let (port, cancel, handle) = spawn_service_with_tinyllama().await;

    let body: Value = post_json(
        port,
        "/tokenize",
        json!({
            "model": MODEL_NAME,
            "prompt": "Hello",
            "return_token_strs": true,
        }),
    )
    .await
    .json()
    .await
    .unwrap();

    let tokens = body["tokens"].as_array().unwrap();
    let strs = body["token_strs"].as_array().expect("token_strs present");
    assert_eq!(tokens.len(), strs.len());
    for s in strs {
        assert!(s.is_string());
    }

    cancel.cancel();
    let _ = handle.await;
}

#[tokio::test]
async fn test_tokenize_unknown_model_404() {
    let (port, cancel, handle) = spawn_service_with_tinyllama().await;

    let resp = post_json(
        port,
        "/tokenize",
        json!({"model": "does-not-exist", "prompt": "hi"}),
    )
    .await;
    assert_eq!(resp.status(), 404);

    cancel.cancel();
    let _ = handle.await;
}

#[tokio::test]
async fn test_tokenize_chat_form() {
    let (port, cancel, handle) = spawn_service_with_chat_model().await;

    let resp = post_json(
        port,
        "/tokenize",
        json!({
            "model": MODEL_NAME_CHAT,
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
            "add_generation_prompt": true,
        }),
    )
    .await;
    assert_eq!(resp.status(), 200, "body: {:?}", resp.text().await);

    let body: Value = post_json(
        port,
        "/tokenize",
        json!({
            "model": MODEL_NAME_CHAT,
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
            "add_generation_prompt": true,
        }),
    )
    .await
    .json()
    .await
    .unwrap();

    let tokens = body["tokens"].as_array().unwrap();
    assert!(
        tokens.len() >= 5,
        "chat-templated prompt should produce more tokens than the bare text: {tokens:?}"
    );

    // Compare to the bare-prompt token count to confirm the template wrapped the message.
    let bare: Value = post_json(
        port,
        "/tokenize",
        json!({"model": MODEL_NAME_CHAT, "prompt": "Hello"}),
    )
    .await
    .json()
    .await
    .unwrap();
    let bare_count = bare["tokens"].as_array().unwrap().len();
    assert!(
        tokens.len() > bare_count,
        "chat form should include template tokens beyond the bare prompt"
    );

    cancel.cancel();
    let _ = handle.await;
}

#[tokio::test]
async fn test_tokenize_chat_continue_final_message_501() {
    let (port, cancel, handle) = spawn_service_with_chat_model().await;

    let resp = post_json(
        port,
        "/tokenize",
        json!({
            "model": MODEL_NAME_CHAT,
            "messages": [{"role": "user", "content": "hi"}],
            "continue_final_message": true,
        }),
    )
    .await;
    assert_eq!(resp.status(), 501);

    cancel.cancel();
    let _ = handle.await;
}

#[tokio::test]
async fn test_tokenize_chat_template_override_501() {
    let (port, cancel, handle) = spawn_service_with_chat_model().await;

    let resp = post_json(
        port,
        "/tokenize",
        json!({
            "model": MODEL_NAME_CHAT,
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template": "{{ messages[0].content }}",
        }),
    )
    .await;
    assert_eq!(resp.status(), 501);

    cancel.cancel();
    let _ = handle.await;
}

#[tokio::test]
async fn test_tokenize_chat_no_template_501() {
    // TinyLlama fixture has no chat template — chat-form requests should fail with 501.
    let (port, cancel, handle) = spawn_service_with_tinyllama().await;

    let resp = post_json(
        port,
        "/tokenize",
        json!({
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(resp.status(), 501);

    cancel.cancel();
    let _ = handle.await;
}

#[tokio::test]
async fn test_detokenize_roundtrip() {
    let (port, cancel, handle) = spawn_service_with_tinyllama().await;

    let original = "Hello, world!";
    let tok_body: Value = post_json(
        port,
        "/tokenize",
        json!({"model": MODEL_NAME, "prompt": original, "add_special_tokens": false}),
    )
    .await
    .json()
    .await
    .unwrap();
    let tokens = tok_body["tokens"].clone();

    let detok_body: Value = post_json(
        port,
        "/detokenize",
        json!({"model": MODEL_NAME, "tokens": tokens}),
    )
    .await
    .json()
    .await
    .unwrap();

    // Tokenizers don't always preserve whitespace exactly; compare normalized.
    let prompt = detok_body["prompt"].as_str().unwrap();
    assert!(
        prompt.trim().contains("Hello") && prompt.contains("world"),
        "round-tripped prompt should contain the original substrings, got: {prompt:?}"
    );

    cancel.cancel();
    let _ = handle.await;
}

#[tokio::test]
async fn test_detokenize_unknown_model_404() {
    let (port, cancel, handle) = spawn_service_with_tinyllama().await;

    let resp = post_json(
        port,
        "/detokenize",
        json!({"model": "does-not-exist", "tokens": [1, 2, 3]}),
    )
    .await;
    assert_eq!(resp.status(), 404);

    cancel.cancel();
    let _ = handle.await;
}

#[tokio::test]
async fn test_default_model_when_only_one_registered() {
    let (port, cancel, handle) = spawn_service_with_tinyllama().await;

    // No `model` field — should fall through to the only registered card.
    let resp = post_json(port, "/tokenize", json!({"prompt": "hi"})).await;
    assert_eq!(resp.status(), 200, "body: {:?}", resp.text().await);
    let body: Value = post_json(port, "/tokenize", json!({"prompt": "hi"}))
        .await
        .json()
        .await
        .unwrap();
    assert!(!body["tokens"].as_array().unwrap().is_empty());

    cancel.cancel();
    let _ = handle.await;
}
