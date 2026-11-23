// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full-HTTP coverage for server-side Responses storage: `store`,
//! `previous_response_id` continuation, and the `GET` / `DELETE` /
//! `input_items` lifecycle routes.

use std::sync::Arc;

use dynamo_llm::http::service::responses_store::{
    MemoryResponseStore, MemoryStoreConfig, ResponseStore,
};
use dynamo_protocols::types::{
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
    ChatCompletionRequestUserMessageContent,
};
use dynamo_runtime::config::environment_names::llm::DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS;
use serde_json::{Value, json};
use serial_test::serial;

#[allow(dead_code)]
#[path = "common/http_harness.rs"]
mod http_harness;
#[path = "common/ports.rs"]
mod ports;
#[allow(dead_code)]
#[path = "common/scripted_chat_engine.rs"]
mod scripted_chat_engine;

use http_harness::{HarnessService, MODEL, load_agent_fixture, parse_responses_sse};

const ENV: [(&str, Option<&str>); 1] = [(DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0"))];

fn memory_store() -> Arc<dyn ResponseStore> {
    Arc::new(MemoryResponseStore::new(MemoryStoreConfig::default()))
}

async fn post_responses(svc: &HarnessService, body: &Value) -> reqwest::Response {
    svc.client
        .post(format!("{}/v1/responses", svc.base_url))
        .json(body)
        .send()
        .await
        .expect("POST /v1/responses failed")
}

fn assistant_text(message: &ChatCompletionRequestMessage) -> &str {
    match message {
        ChatCompletionRequestMessage::Assistant(assistant) => match &assistant.content {
            Some(ChatCompletionRequestAssistantMessageContent::Text(text)) => text,
            other => panic!("unexpected assistant content: {other:#?}"),
        },
        other => panic!("expected assistant message, got {other:#?}"),
    }
}

fn user_text(message: &ChatCompletionRequestMessage) -> &str {
    match message {
        ChatCompletionRequestMessage::User(user) => match &user.content {
            ChatCompletionRequestUserMessageContent::Text(text) => text,
            other => panic!("unexpected user content: {other:#?}"),
        },
        other => panic!("expected user message, got {other:#?}"),
    }
}

#[tokio::test]
async fn store_true_without_a_store_is_echoed_false_and_continuation_is_501() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([load_agent_fixture("text.sse").await.unwrap()]).await;
        let response = post_responses(
            &svc,
            &json!({"model": MODEL, "input": "ping", "store": true, "stream": false}),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["store"], false, "nothing was stored: {body}");
        let id = body["id"].as_str().unwrap().to_string();

        let response = post_responses(
            &svc,
            &json!({"model": MODEL, "input": "pong", "previous_response_id": id}),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::NOT_IMPLEMENTED);
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
async fn continuation_replays_the_stored_chain_for_unary_responses() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start_with_store(
            [
                load_agent_fixture("text.sse").await.unwrap(),
                load_agent_fixture("text.sse").await.unwrap(),
                load_agent_fixture("text.sse").await.unwrap(),
            ],
            memory_store(),
        )
        .await;

        let first = post_responses(
            &svc,
            &json!({"model": MODEL, "input": "ping", "store": true, "stream": false}),
        )
        .await;
        assert_eq!(first.status(), reqwest::StatusCode::OK);
        let first: Value = first.json().await.unwrap();
        assert_eq!(first["store"], true, "{first}");
        assert_eq!(first["previous_response_id"], Value::Null);
        let first_id = first["id"].as_str().unwrap().to_string();

        let second = post_responses(
            &svc,
            &json!({
                "model": MODEL,
                "input": "again",
                "store": true,
                "stream": false,
                "previous_response_id": first_id,
            }),
        )
        .await;
        assert_eq!(second.status(), reqwest::StatusCode::OK);
        let second: Value = second.json().await.unwrap();
        assert_eq!(second["previous_response_id"], first_id);
        let second_id = second["id"].as_str().unwrap().to_string();

        // A continuation that is itself not stored still replays the chain.
        let third = post_responses(
            &svc,
            &json!({
                "model": MODEL,
                "input": [{"role": "user", "content": "third"}],
                "stream": false,
                "previous_response_id": second_id,
            }),
        )
        .await;
        assert_eq!(third.status(), reqwest::StatusCode::OK);
        let third: Value = third.json().await.unwrap();
        assert_eq!(third["store"], false);

        let requests = svc.engine.take_requests().await;
        assert_eq!(requests.len(), 3);
        let m = &requests[0].inner.messages;
        assert_eq!(m.len(), 1);
        assert_eq!(user_text(&m[0]), "ping");
        let m = &requests[1].inner.messages;
        assert_eq!(m.len(), 3, "{m:#?}");
        assert_eq!(user_text(&m[0]), "ping");
        assert_eq!(assistant_text(&m[1]), "Pong.");
        assert_eq!(user_text(&m[2]), "again");
        let m = &requests[2].inner.messages;
        assert_eq!(m.len(), 5, "{m:#?}");
        assert_eq!(user_text(&m[0]), "ping");
        assert_eq!(assistant_text(&m[1]), "Pong.");
        assert_eq!(user_text(&m[2]), "again");
        assert_eq!(assistant_text(&m[3]), "Pong.");
        assert_eq!(user_text(&m[4]), "third");

        // The third response was not stored, so it cannot be continued.
        let missing = post_responses(
            &svc,
            &json!({
                "model": MODEL,
                "input": "x",
                "previous_response_id": third["id"],
            }),
        )
        .await;
        assert_eq!(missing.status(), reqwest::StatusCode::NOT_FOUND);
        assert_eq!(svc.engine.remaining_scripts().await, 0);
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn streaming_responses_are_stored_before_the_terminal_event() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start_with_store(
            [
                load_agent_fixture("text.sse").await.unwrap(),
                load_agent_fixture("text.sse").await.unwrap(),
            ],
            memory_store(),
        )
        .await;

        let response = post_responses(
            &svc,
            &json!({"model": MODEL, "input": "ping", "store": true, "stream": true}),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let raw = response.text().await.unwrap();
        let events = parse_responses_sse(&raw).await.unwrap();
        let completed = events
            .iter()
            .find(|event| event.event == "response.completed")
            .expect("response.completed");
        assert_eq!(completed.data["response"]["store"], true);
        let id = completed.data["response"]["id"]
            .as_str()
            .unwrap()
            .to_string();

        // The streamed turn is already stored when the terminal event arrives,
        // so an immediate continuation replays it.
        let second = post_responses(
            &svc,
            &json!({
                "model": MODEL,
                "input": "again",
                "stream": false,
                "previous_response_id": id,
            }),
        )
        .await;
        assert_eq!(second.status(), reqwest::StatusCode::OK);
        let second: Value = second.json().await.unwrap();
        assert_eq!(second["previous_response_id"], id);

        let requests = svc.engine.take_requests().await;
        assert_eq!(requests.len(), 2);
        let m = &requests[1].inner.messages;
        assert_eq!(m.len(), 3, "{m:#?}");
        assert_eq!(user_text(&m[0]), "ping");
        assert_eq!(assistant_text(&m[1]), "Pong.");
        assert_eq!(user_text(&m[2]), "again");

        assert_eq!(svc.engine.remaining_scripts().await, 0);
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
async fn lifecycle_routes_return_501_without_a_store() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([]).await;
        for request in [
            svc.client
                .get(format!("{}/v1/responses/resp_x", svc.base_url)),
            svc.client
                .delete(format!("{}/v1/responses/resp_x", svc.base_url)),
            svc.client
                .get(format!("{}/v1/responses/resp_x/input_items", svc.base_url)),
        ] {
            let response = request.send().await.unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::NOT_IMPLEMENTED);
        }
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn lifecycle_routes_serve_stored_responses() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start_with_store(
            [
                load_agent_fixture("text.sse").await.unwrap(),
                load_agent_fixture("text.sse").await.unwrap(),
            ],
            memory_store(),
        )
        .await;

        let first = post_responses(
            &svc,
            &json!({"model": MODEL, "input": "ping", "store": true, "stream": false}),
        )
        .await;
        assert_eq!(first.status(), reqwest::StatusCode::OK);
        let first: Value = first.json().await.unwrap();
        let id = first["id"].as_str().unwrap().to_string();

        // GET returns the object exactly as it was returned to the client.
        let fetched = svc
            .client
            .get(format!("{}/v1/responses/{id}", svc.base_url))
            .send()
            .await
            .unwrap();
        assert_eq!(fetched.status(), reqwest::StatusCode::OK);
        let fetched: Value = fetched.json().await.unwrap();
        assert_eq!(fetched, first);

        let second = post_responses(
            &svc,
            &json!({
                "model": MODEL,
                "input": "again",
                "store": true,
                "stream": false,
                "previous_response_id": id,
            }),
        )
        .await;
        assert_eq!(second.status(), reqwest::StatusCode::OK);
        let second: Value = second.json().await.unwrap();
        let second_id = second["id"].as_str().unwrap().to_string();

        // input_items lists the replayed chain and then the turn's own input.
        let items = svc
            .client
            .get(format!(
                "{}/v1/responses/{second_id}/input_items",
                svc.base_url
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(items.status(), reqwest::StatusCode::OK);
        let items: Value = items.json().await.unwrap();
        assert_eq!(items["object"], "list");
        assert_eq!(items["has_more"], false);
        let data = items["data"].as_array().unwrap();
        assert_eq!(data.len(), 3, "{items}");
        assert_eq!(data[0]["role"], "user");
        assert_eq!(data[0]["content"], "ping");
        assert_eq!(data[1]["type"], "message");
        assert_eq!(data[1]["role"], "assistant");
        assert_eq!(data[2]["content"], "again");

        // Deleting the first node breaks the chain for later continuations.
        let deleted = svc
            .client
            .delete(format!("{}/v1/responses/{id}", svc.base_url))
            .send()
            .await
            .unwrap();
        assert_eq!(deleted.status(), reqwest::StatusCode::OK);
        let deleted: Value = deleted.json().await.unwrap();
        assert_eq!(
            deleted,
            json!({"id": id, "object": "response", "deleted": true})
        );
        for request in [
            svc.client
                .get(format!("{}/v1/responses/{id}", svc.base_url)),
            svc.client
                .delete(format!("{}/v1/responses/{id}", svc.base_url)),
            svc.client
                .get(format!("{}/v1/responses/{id}/input_items", svc.base_url)),
        ] {
            let response = request.send().await.unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
        }
        let broken = post_responses(
            &svc,
            &json!({"model": MODEL, "input": "x", "previous_response_id": second_id}),
        )
        .await;
        assert_eq!(broken.status(), reqwest::StatusCode::NOT_FOUND);

        assert_eq!(svc.engine.remaining_scripts().await, 0);
        svc.shutdown().await;
    })
    .await;
}
