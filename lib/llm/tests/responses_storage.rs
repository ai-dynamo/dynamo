// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::http::service::response_store::{ResponseStorage, StoreConfig};
use serde_json::{Value, json};
use std::sync::Arc;

#[path = "common/http_harness.rs"]
mod http_harness;
#[path = "common/ports.rs"]
mod ports;
#[path = "common/scripted_chat_engine.rs"]
mod scripted_chat_engine;
use http_harness::{HarnessService, MODEL, load_agent_fixture, parse_json_sse};
use scripted_chat_engine::ScriptedChatEngine;

async fn service(fixtures: &[&str]) -> HarnessService {
    let mut scripts = Vec::new();
    for fixture in fixtures {
        scripts.push(Ok(load_agent_fixture(fixture).await.unwrap()));
    }
    HarnessService::start_with_storage(
        Arc::new(ScriptedChatEngine::new(scripts)),
        Some(ResponseStorage::memory(StoreConfig::default()).unwrap()),
    )
    .await
}

async fn create(svc: &HarnessService, body: Value) -> Value {
    let response = svc
        .client
        .post(format!("{}/v1/responses", svc.base_url))
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body: Value = response.json().await.unwrap();
    assert!(status.is_success(), "{status}: {body}");
    body
}

#[tokio::test]
async fn retrieve_continue_delete_and_keep_descendant_independent() {
    let svc = service(&["text.sse", "text.sse", "text.sse"]).await;
    let first = create(
        &svc,
        json!({"model":MODEL,"input":"Remember Alice", "instructions":"Only this turn"}),
    )
    .await;
    let first_url = format!(
        "{}/v1/responses/{}",
        svc.base_url,
        first["id"].as_str().unwrap()
    );
    let fetched: Value = svc
        .client
        .get(&first_url)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(first, fetched);
    let second = create(
        &svc,
        json!({"model":MODEL,"input":"Who?","previous_response_id":first["id"]}),
    )
    .await;
    assert_eq!(second["previous_response_id"], first["id"]);
    assert!(second["instructions"].is_null());
    assert_eq!(
        svc.client.delete(&first_url).send().await.unwrap().status(),
        200
    );
    assert_eq!(
        svc.client.get(&first_url).send().await.unwrap().status(),
        404
    );
    create(
        &svc,
        json!({"model":MODEL,"input":"Again?","previous_response_id":second["id"],"store":false}),
    )
    .await;
    let requests = svc.engine.take_requests().await;
    assert_eq!(requests[1].inner.messages.len(), 3);
    assert_eq!(requests[2].inner.messages.len(), 5);
    svc.shutdown().await;
}

#[tokio::test]
async fn streaming_record_matches_terminal_event_and_continues_tool_call() {
    let svc = service(&["fragmented-tool.sse", "text.sse"]).await;
    let response = svc.client.post(format!("{}/v1/responses", svc.base_url)).json(&json!({
        "model":MODEL,"input":"List files", "stream":true,
        "tools":[{"type":"function","name":"list_directory","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}]
    })).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let events = parse_json_sse(&response.text().await.unwrap())
        .await
        .unwrap();
    let terminal = events
        .iter()
        .find(|event| event.event == "response.completed")
        .unwrap();
    let response = &terminal.data["response"];
    let fetched: Value = svc
        .client
        .get(format!(
            "{}/v1/responses/{}",
            svc.base_url,
            response["id"].as_str().unwrap()
        ))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(&fetched, response);
    let call = response["output"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "function_call")
        .unwrap();
    create(&svc,json!({"model":MODEL,"previous_response_id":response["id"],"input":[{"type":"function_call_output","call_id":call["call_id"],"output":"[]"}]})).await;
    assert_eq!(svc.engine.take_requests().await[1].inner.messages.len(), 3);
    svc.shutdown().await;
}

#[tokio::test]
async fn stateless_and_credential_scoped_records() {
    let svc = service(&["text.sse", "text.sse"]).await;
    let first = create(&svc, json!({"model":MODEL,"input":"hello","store":false})).await;
    assert_eq!(first["store"], false);
    assert_eq!(
        svc.client
            .get(format!(
                "{}/v1/responses/{}",
                svc.base_url,
                first["id"].as_str().unwrap()
            ))
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
    let response = svc
        .client
        .post(format!("{}/v1/responses", svc.base_url))
        .bearer_auth("scope-a")
        .json(&json!({"model":MODEL,"input":"hello"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let response: Value = response.json().await.unwrap();
    let url = format!(
        "{}/v1/responses/{}",
        svc.base_url,
        response["id"].as_str().unwrap()
    );
    assert_eq!(
        svc.client
            .get(&url)
            .bearer_auth("scope-a")
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    assert_eq!(
        svc.client
            .get(&url)
            .bearer_auth("scope-b")
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
    assert_eq!(svc.client.delete(&url).send().await.unwrap().status(), 404);
    let response = svc
        .client
        .post(format!("{}/v1/responses", svc.base_url))
        .bearer_auth("scope-b")
        .json(&json!({"model":MODEL,"input":"hello","previous_response_id":response["id"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 404);
    assert_eq!(svc.engine.remaining_scripts().await, 0);
    svc.shutdown().await;
}

#[tokio::test]
async fn capacity_failure_never_acknowledges_storage_and_unknown_history_skips_engine() {
    let config = StoreConfig {
        max_entries: 1,
        ..StoreConfig::default()
    };
    let script = load_agent_fixture("text.sse").await.unwrap();
    let svc = HarnessService::start_with_storage(
        Arc::new(ScriptedChatEngine::new(vec![
            Ok(script.clone()),
            Ok(script),
        ])),
        Some(ResponseStorage::memory(config).unwrap()),
    )
    .await;
    let first = create(&svc, json!({"model":MODEL,"input":"first"})).await;
    let response = svc
        .client
        .post(format!("{}/v1/responses", svc.base_url))
        .json(&json!({"model":MODEL,"input":"second"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 507);
    let response = svc
        .client
        .post(format!("{}/v1/responses", svc.base_url))
        .json(&json!({"model":MODEL,"input":"third","previous_response_id":"resp_missing"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 404);
    assert_eq!(svc.engine.take_requests().await.len(), 2);
    let retained: Value = svc
        .client
        .get(format!(
            "{}/v1/responses/{}",
            svc.base_url,
            first["id"].as_str().unwrap()
        ))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(retained, first);
    svc.shutdown().await;
}
