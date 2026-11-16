// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_protocols::types::ReasoningEffort;
use dynamo_runtime::config::environment_names::llm::{
    DYN_ENABLE_ANTHROPIC_API, DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
};
use serde_json::{Value, json};
use serial_test::serial;

#[allow(dead_code)]
#[path = "common/http_harness.rs"]
mod http_harness;
#[path = "common/mistral_model.rs"]
mod mistral_model;
#[path = "common/ports.rs"]
mod ports;
#[allow(dead_code)]
#[path = "common/scripted_chat_engine.rs"]
mod scripted_chat_engine;

use http_harness::{HarnessService, MODEL, load_agent_fixture, parse_anthropic_sse};
use scripted_chat_engine::ScriptedChatEngine;

const ENV: [(&str, Option<&str>); 2] = [
    (DYN_ENABLE_ANTHROPIC_API, Some("1")),
    (DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0")),
];

fn assert_effort(request: &NvCreateChatCompletionRequest, expected: Option<&str>) {
    assert_eq!(
        request
            .chat_template_args
            .as_ref()
            .and_then(|args| args.get("reasoning_effort"))
            .and_then(Value::as_str),
        expected,
    );
    if let Some(effort) = expected {
        let enabled = effort != "none";
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(args["thinking"], enabled);
        assert_eq!(args["enable_thinking"], enabled);
        assert_eq!(
            args["thinking_mode"],
            if enabled { "enabled" } else { "disabled" }
        );
    }
}

#[tokio::test]
#[serial]
async fn both_checkpoint_layouts_normalize_chat_and_anthropic_through_http() {
    temp_env::async_with_vars(ENV, async {
        for with_hf_config in [false, true] {
            for streaming in [false, true] {
                let dir = mistral_model::model_dir(with_hf_config, "v15", &["low", "medium", "high"]);
                let card = ModelDeploymentCard::load_from_disk(dir.path(), None).unwrap();
                let dynamo_renderer::PromptFormatter::OAI(formatter) =
                    dynamo_llm::preprocessor::prompt::prompt_formatter_from_mdc(&card).unwrap();
                let cases = [
                    ("chat/completions", json!({"reasoning_effort":"none"}), Some("low")),
                    ("chat/completions", json!({"chat_template_kwargs":{"reasoning_effort":"none"}}), Some("low")),
                    ("chat/completions", json!({"reasoning_effort":"high","chat_template_kwargs":{"reasoning_effort":"none"}}), Some("high")),
                    ("chat/completions", json!({"reasoning_effort":"none","chat_template_kwargs":{"reasoning_effort":"high"}}), Some("low")),
                    ("chat/completions", json!({}), None),
                    ("chat/completions", json!({"reasoning_effort":null}), None),
                    ("messages", json!({"thinking":{"type":"disabled"},"output_config":{"effort":"high"}}), Some("low")),
                    ("messages", json!({"output_config":{"effort":"medium"}}), Some("medium")),
                    ("messages", json!({}), None),
                ];
                let script = load_agent_fixture("text.sse").await.unwrap();
                let engine = Arc::new(ScriptedChatEngine::new(cases.iter().map(|_| Ok(script.clone()))));
                let svc = HarnessService::start_with_engine_and_card(engine, card).await;
                // Policy metadata must not be reopened on the request path.
                std::fs::remove_file(dir.path().join("tekken.json")).unwrap();
                for (route, controls, expected) in cases {
                    let mut body = json!({"model":MODEL,"max_tokens":32,"stream":streaming,
                        "messages":[{"role":"user","content":"hi"}]});
                    body.as_object_mut().unwrap().extend(controls.as_object().unwrap().clone());
                    let response = svc.client.post(format!("{}/v1/{route}", svc.base_url))
                        .json(&body).send().await.unwrap();
                    assert_eq!(response.status(), reqwest::StatusCode::OK, "{body}");
                    let text = response.text().await.unwrap();
                    if streaming && route == "messages" { parse_anthropic_sse(&text).await.unwrap(); }
                    if streaming && route == "chat/completions" { assert!(text.contains("[DONE]")); }
                    let requests = svc.engine.take_requests().await;
                    assert_eq!(requests.len(), 1);
                    assert_effort(&requests[0], expected);
                    let prompt = formatter.render(&requests[0]).unwrap();
                    assert!(prompt.contains(&format!(r#""reasoning_effort": "{}""#, expected.unwrap_or("high"))));
                }
                // An invalid Anthropic grade stays an input error, even with disabled thinking.
                let response = svc.client.post(format!("{}/v1/messages", svc.base_url)).json(&json!({
                    "model":MODEL,"max_tokens":32,"thinking":{"type":"disabled"},
                    "output_config":{"effort":"none"},"messages":[{"role":"user","content":"hi"}]
                })).send().await.unwrap();
                assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
                assert!(svc.engine.take_requests().await.is_empty());
                assert_eq!(svc.engine.remaining_scripts().await, 0);
                svc.shutdown().await;
            }
        }
    }).await;
}

#[tokio::test]
#[serial]
async fn inline_system_after_tool_is_merged_before_native_rendering() {
    temp_env::async_with_vars(ENV, async {
        for with_hf_config in [false, true] {
            let dir = mistral_model::model_dir(with_hf_config, "v15", &["low", "medium", "high"]);
            let card = ModelDeploymentCard::load_from_disk(dir.path(), None).unwrap();
            let dynamo_renderer::PromptFormatter::OAI(formatter) =
                dynamo_llm::preprocessor::prompt::prompt_formatter_from_mdc(&card).unwrap();
            let script = load_agent_fixture("text.sse").await.unwrap();
            let svc = HarnessService::start_with_engine_and_card(
                Arc::new(ScriptedChatEngine::new([Ok(script)])), card,
            ).await;
            let response = svc.client.post(format!("{}/v1/messages", svc.base_url)).json(&json!({
                "model":MODEL,"max_tokens":32,"system":"Initial instruction", "messages":[
                    {"role":"user","content":"read notes"},
                    {"role":"assistant","content":[{"type":"tool_use","id":"call00001","name":"Bash","input":{"command":"cat notes"}}]},
                    {"role":"user","content":[{"type":"tool_result","tool_use_id":"call00001","content":"hello"}]},
                    {"role":"system","content":"Budget reminder"}
                ]
            })).send().await.unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            response.bytes().await.unwrap();
            let requests = svc.engine.take_requests().await;
            let prompt = formatter.render(&requests[0]).unwrap();
            assert!(prompt.starts_with("<s>[SYSTEM_PROMPT]Initial instruction\nBudget reminder[/SYSTEM_PROMPT]"));
            assert!(prompt.contains("[TOOL_RESULTS]hello[/TOOL_RESULTS]"));
            svc.shutdown().await;
        }
    }).await;
}

#[tokio::test]
#[serial]
async fn supported_none_and_standard_models_keep_none() {
    temp_env::async_with_vars(ENV, async {
        let dir = mistral_model::model_dir(true, "v15", &["none", "high"]);
        for card in [ModelDeploymentCard::load_from_disk(dir.path(), None).unwrap(),
            ModelDeploymentCard::with_name_only(MODEL)] {
            let script = load_agent_fixture("text.sse").await.unwrap();
            let svc = HarnessService::start_with_engine_and_card(
                Arc::new(ScriptedChatEngine::new([Ok(script)])), card,
            ).await;
            let response = svc.client.post(format!("{}/v1/chat/completions", svc.base_url)).json(&json!({
                "model":MODEL,"reasoning_effort":"none","messages":[{"role":"user","content":"hi"}]
            })).send().await.unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            response.bytes().await.unwrap();
            let requests = svc.engine.take_requests().await;
            assert_eq!(requests[0].inner.reasoning_effort, Some(ReasoningEffort::None));
            assert_effort(&requests[0], Some("none"));
            svc.shutdown().await;
        }
    }).await;
}
