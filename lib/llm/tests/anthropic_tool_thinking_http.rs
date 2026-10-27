// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool failure status and reasoning display survive the HTTP conversion boundary.

use std::time::Duration;

use dynamo_runtime::config::environment_names::llm::{
    DYN_ENABLE_ANTHROPIC_API, DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
};
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

use http_harness::{HarnessService, MODEL, load_agent_fixture, parse_json_sse_events};

const ENV: [(&str, Option<&str>); 2] = [
    (DYN_ENABLE_ANTHROPIC_API, Some("1")),
    (DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0")),
];

async fn post(svc: &HarnessService, body: &Value) -> reqwest::Response {
    svc.client
        .post(format!("{}/v1/messages", svc.base_url))
        .timeout(Duration::from_secs(10))
        .json(body)
        .send()
        .await
        .unwrap()
}

#[tokio::test]
#[serial]
async fn thinking_display_filters_unary_and_streaming_without_losing_other_output() {
    temp_env::async_with_vars(ENV, async {
        let mut script = load_agent_fixture("thinking-tool.sse").await.unwrap();
        script.insert(1, script[0].clone()); // Exercise more than the first reasoning chunk.
        let svc = HarnessService::start(std::iter::repeat_n(script, 6)).await;
        for stream in [false, true] {
            for display in [None, Some("summarized"), Some("omitted")] {
                let mut thinking = json!({"type": "adaptive"});
                if let Some(display) = display {
                    thinking["display"] = json!(display);
                }
                let response = post(
                    &svc,
                    &json!({"model": MODEL, "max_tokens": 4096,
                        "messages": [{"role": "user", "content": "List /tmp"}],
                        "stream": stream, "thinking": thinking,
                        "tools": [{"name": "list_directory", "input_schema": {"type": "object"}}]
                    }),
                )
                .await;
                assert_eq!(response.status(), reqwest::StatusCode::OK);
                let body = response.text().await.unwrap();
                let omitted = display == Some("omitted");
                assert_eq!(
                    body.contains("I should inspect the directory."),
                    !omitted,
                    "{body}"
                );
                assert!(body.contains("I will list it."), "{body}");
                if stream {
                    let events = parse_json_sse_events(&body).await.unwrap();
                    assert_eq!(events.first().unwrap().event, "message_start");
                    assert_eq!(events.last().unwrap().event, "message_stop");
                    let deltas: Vec<_> = events
                        .iter()
                        .filter(|e| e.data["delta"]["type"] == "thinking_delta")
                        .collect();
                    assert_eq!(deltas.len(), 2);
                    for event in deltas {
                        assert_eq!(
                            event.data["delta"]["thinking"],
                            if omitted {
                                ""
                            } else {
                                "I should inspect the directory."
                            }
                        );
                    }
                    assert!(
                        events
                            .iter()
                            .any(|e| e.data["delta"]["type"] == "signature_delta")
                    );
                    let mut open = None;
                    for event in &events {
                        match event.event.as_str() {
                            "content_block_start" => {
                                assert!(open.is_none());
                                open = event.data["index"].as_u64();
                            }
                            "content_block_delta" => assert_eq!(event.data["index"].as_u64(), open),
                            "content_block_stop" => {
                                assert_eq!(event.data["index"].as_u64(), open);
                                open = None;
                            }
                            _ => {}
                        }
                    }
                    assert!(open.is_none());
                    let tool = events
                        .iter()
                        .find(|e| e.data["content_block"]["type"] == "tool_use")
                        .unwrap();
                    assert!(
                        tool.data["content_block"]["id"]
                            .as_str()
                            .unwrap()
                            .starts_with("toolu_")
                    );
                    assert_eq!(tool.data["content_block"]["name"], "list_directory");
                    let arguments: String = events
                        .iter()
                        .filter_map(|e| e.data["delta"]["partial_json"].as_str())
                        .collect();
                    assert_eq!(
                        serde_json::from_str::<Value>(&arguments).unwrap(),
                        json!({"path": "/tmp"})
                    );
                    let tail = events.iter().find(|e| e.event == "message_delta").unwrap();
                    assert_eq!(tail.data["delta"]["stop_reason"], "tool_use");
                    assert_eq!(tail.data["usage"]["output_tokens"], 18);
                } else {
                    let body: Value = serde_json::from_str(&body).unwrap();
                    let blocks = body["content"].as_array().unwrap();
                    let thinking = blocks.iter().find(|b| b["type"] == "thinking").unwrap();
                    assert_eq!(
                        thinking["thinking"],
                        if omitted {
                            ""
                        } else {
                            "I should inspect the directory.I should inspect the directory."
                        }
                    );
                    assert!(thinking["signature"].is_string());
                    let tool = blocks.iter().find(|b| b["type"] == "tool_use").unwrap();
                    assert!(tool["id"].as_str().unwrap().starts_with("toolu_"));
                    assert_eq!(tool["name"], "list_directory");
                    assert_eq!(tool["input"], json!({"path": "/tmp"}));
                    assert_eq!(body["stop_reason"], "tool_use");
                    assert_eq!(body["usage"]["input_tokens"], 30);
                    assert_eq!(body["usage"]["output_tokens"], 18);
                }
            }
        }
        assert_eq!(svc.engine.take_requests().await.len(), 6);
        assert_eq!(svc.engine.remaining_scripts().await, 0);
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn invalid_display_mode_combinations_reject_before_generation() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([load_agent_fixture("text.sse").await.unwrap()]).await;
        for mode in ["disabled", "between_tools", "unknown"] {
            for stream in [false, true] {
                let response = post(
                    &svc,
                    &json!({"model": MODEL, "max_tokens": 4096,
                        "messages": [{"role": "user", "content": "ping"}], "stream": stream,
                        "thinking": {"type": mode, "display": "omitted"}
                    }),
                )
                .await;
                assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
                let error: Value = response.json().await.unwrap();
                assert_eq!(error["error"]["type"], "invalid_request_error");
                assert!(
                    error["error"]["message"]
                        .as_str()
                        .unwrap()
                        .contains("thinking.display")
                );
            }
        }
        assert!(svc.engine.take_requests().await.is_empty());
        assert_eq!(svc.engine.remaining_scripts().await, 1);
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn failed_tool_results_keep_identity_content_and_image_order() {
    temp_env::async_with_vars(ENV, async {
        let script = load_agent_fixture("text.sse").await.unwrap();
        let svc = HarnessService::start(std::iter::repeat_n(script, 24)).await;
        let image = json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}});
        let contents = [Value::Null, json!("permission denied"), json!([{ "type": "text", "text": "permission denied" }]), json!([{ "type": "text", "text": "before" }, image, { "type": "text", "text": "after" }])];
        for stream in [false, true] {
            for content in &contents {
                for status in [None, Some(false), Some(true)] {
                    let mut result = json!({"type": "tool_result", "tool_use_id": "toolu_original", "content": content});
                    if let Some(status) = status { result["is_error"] = json!(status); }
                    let response = post(&svc, &json!({"model": MODEL, "max_tokens": 64, "stream": stream,
                        "messages": [
                            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_original", "name": "read", "input": {"path": "/tmp"}}]},
                            {"role": "user", "content": [result]}
                        ]
                    })).await;
                    assert_eq!(response.status(), reqwest::StatusCode::OK);
                    let _ = response.text().await.unwrap();
                    let requests = svc.engine.take_requests().await;
                    assert_eq!(requests.len(), 1);
                    let messages = serde_json::to_value(&requests[0].inner.messages).unwrap();
                    assert_eq!(messages[0]["tool_calls"][0]["id"], "toolu_original");
                    let tool = &messages[1];
                    assert_eq!(tool["role"], "tool");
                    assert_eq!(tool["tool_call_id"], "toolu_original");
                    let failed = status == Some(true);
                    if content.as_array().is_some_and(|a| a.len() == 3) {
                        let parts = tool["content"].as_array().unwrap();
                        let offset = usize::from(failed);
                        assert_eq!(parts.len(), 3 + offset);
                        if failed { assert_eq!(parts[0]["text"], "[Tool execution failed]\n"); }
                        assert_eq!(parts[offset]["text"], "before");
                        assert_eq!(parts[offset + 1]["image_url"]["url"], "data:image/png;base64,aW1hZ2U=");
                        assert_eq!(parts[offset + 2]["text"], "after");
                    } else {
                        let original = if content.is_null() { "" } else { "permission denied" };
                        let expected = if failed { format!("[Tool execution failed]\n{original}") } else { original.to_string() };
                        assert_eq!(tool["content"], expected);
                    }
                }
            }
        }
        assert_eq!(svc.engine.remaining_scripts().await, 0);
        svc.shutdown().await;
    }).await;
}
