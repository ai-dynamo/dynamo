// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_runtime::config::environment_names::llm::{
    DYN_ENABLE_ANTHROPIC_API, DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
};
use serde_json::json;
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

use http_harness::{HarnessService, MODEL, load_agent_fixture};
use scripted_chat_engine::ScriptedChatEngine;

const ENV: [(&str, Option<&str>); 2] = [
    (DYN_ENABLE_ANTHROPIC_API, Some("1")),
    (DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0")),
];

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
