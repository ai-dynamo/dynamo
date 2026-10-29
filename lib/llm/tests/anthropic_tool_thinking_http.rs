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
                    assert_eq!(deltas.len(), if omitted { 1 } else { 2 });
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
            let response = post(
                &svc,
                &json!({"model": MODEL, "max_tokens": 4096,
                    "messages": [{"role": "user", "content": "ping"}], "stream": false,
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
                    .contains("thinking.")
            );
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
        let svc = HarnessService::start(std::iter::repeat_n(script, 12)).await;
        let image = json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}});
        let contents = [Value::Null, json!("permission denied"), json!([{ "type": "text", "text": "permission denied" }]), json!([{ "type": "text", "text": "before" }, image, { "type": "text", "text": "after" }])];
        for content in &contents {
            for status in [None, Some(false), Some(true)] {
                let mut result = json!({"type": "tool_result", "tool_use_id": "toolu_original", "content": content});
                if let Some(status) = status { result["is_error"] = json!(status); }
                let response = post(&svc, &json!({"model": MODEL, "max_tokens": 64, "stream": false,
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
        assert_eq!(svc.engine.remaining_scripts().await, 0);
        svc.shutdown().await;
    }).await;
}

#[tokio::test]
#[serial]
async fn updates_display_and_malformed_tool_error_flags_reject_before_generation() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([]).await;
        let mut cases = vec![json!({"model": MODEL, "max_tokens": 4096,
            "messages": [{"role": "user", "content": "ping"}],
            "thinking": {"type": "adaptive", "display": "updates"}})];
        for is_error in [json!("true"), json!(0), json!({})] {
            cases.push(json!({"model": MODEL, "max_tokens": 4096,
                "messages": [{"role": "user", "content": [{"type": "tool_result",
                    "tool_use_id": "tool_1", "content": "failed", "is_error": is_error}]}]}));
        }
        for body in cases {
            let response = svc
                .client
                .post(format!("{}/v1/messages", svc.base_url))
                .header("anthropic-beta", "thinking-display-updates-2026-08-18")
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
            let error: Value = response.json().await.unwrap();
            assert_eq!(error["error"]["type"], "invalid_request_error");
        }
        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn resumed_thinking_keeps_sequential_blocks_and_pending_tool_order() {
    temp_env::async_with_vars(ENV, async {
        let fixture = load_agent_fixture("thinking-tool.sse").await.unwrap();
        let mut tool_start = fixture[2].clone();
        tool_start.data.as_mut().unwrap().inner.choices[0]
            .delta
            .tool_calls
            .as_mut()
            .unwrap()[0]
            .function
            .as_mut()
            .unwrap()
            .arguments = Some("{\"path\":".into());
        let mut tool_end = fixture[2].clone();
        let tool = &mut tool_end.data.as_mut().unwrap().inner.choices[0]
            .delta
            .tool_calls
            .as_mut()
            .unwrap()[0];
        tool.id = None;
        tool.function.as_mut().unwrap().name = None;
        tool.function.as_mut().unwrap().arguments = Some("\"/tmp\"}".into());
        let mut second_tool = fixture[2].clone();
        let second_call = &mut second_tool.data.as_mut().unwrap().inner.choices[0]
            .delta
            .tool_calls
            .as_mut()
            .unwrap()[0];
        second_call.index = 1;
        second_call.id = Some("call_second".into());
        let late_parallel = vec![
            tool_start.clone(),
            fixture[0].clone(),
            second_tool,
            fixture[3].clone(),
            tool_end.clone(),
        ];
        let scripts = [
            (late_parallel, vec!["tool_use", "thinking", "tool_use"]),
            (
                vec![
                    fixture[1].clone(),
                    fixture[0].clone(),
                    fixture[1].clone(),
                    fixture[0].clone(),
                ],
                vec!["text", "thinking", "text", "thinking"],
            ),
            (
                vec![fixture[0].clone(), fixture[1].clone(), fixture[0].clone()],
                vec!["thinking", "text", "thinking"],
            ),
            (
                vec![
                    fixture[0].clone(),
                    tool_start,
                    fixture[1].clone(),
                    fixture[0].clone(),
                    fixture[0].clone(),
                    tool_end,
                ],
                vec!["thinking", "tool_use", "text", "thinking"],
            ),
        ];
        for display in ["summarized", "omitted"] {
            for (script, expected) in &scripts {
                let mut script = script.clone();
                script.extend(fixture[3..].iter().cloned());
                let svc = HarnessService::start([script]).await;
                let response = post(
                    &svc,
                    &json!({
                        "model": MODEL, "max_tokens": 4096, "stream": true,
                        "messages": [{"role": "user", "content": "ping"}],
                        "thinking": {"type": "adaptive", "display": display},
                        "tools": [{"name": "list_directory", "input_schema": {"type": "object"}}]
                    }),
                )
                .await;
                assert_eq!(response.status(), reqwest::StatusCode::OK);
                let body = response.text().await.unwrap();
                assert_eq!(
                    body.contains("I should inspect"),
                    display == "summarized",
                    "{body}"
                );
                let events = parse_json_sse_events(&body).await.unwrap();
                let mut open = None;
                let mut kinds = Vec::new();
                let mut thinking_deltas = 0;
                let mut signature_deltas = 0;
                let mut arguments = std::collections::BTreeMap::<u64, String>::new();
                for event in &events {
                    match event.event.as_str() {
                        "content_block_start" => {
                            assert!(open.is_none(), "nested block: {body}");
                            let index = event.data["index"].as_u64().unwrap();
                            assert_eq!(index as usize, kinds.len());
                            open = Some(index);
                            kinds.push(event.data["content_block"]["type"].as_str().unwrap());
                        }
                        "content_block_delta" => {
                            assert_eq!(event.data["index"].as_u64(), open, "{body}");
                            match event.data["delta"]["type"].as_str().unwrap() {
                                "thinking_delta" => {
                                    thinking_deltas += 1;
                                    if display == "omitted" {
                                        assert_eq!(event.data["delta"]["thinking"], "");
                                    }
                                }
                                "signature_delta" => signature_deltas += 1,
                                "input_json_delta" => {
                                    arguments.entry(open.unwrap()).or_default().push_str(
                                        event.data["delta"]["partial_json"].as_str().unwrap(),
                                    )
                                }
                                _ => {}
                            }
                        }
                        "content_block_stop" => {
                            assert_eq!(event.data["index"].as_u64(), open);
                            open = None;
                        }
                        _ => {}
                    }
                }
                assert!(open.is_none());
                assert_eq!(&kinds, expected, "{body}");
                let expected_thinking = expected.iter().filter(|kind| **kind == "thinking").count();
                assert_eq!(signature_deltas, expected_thinking);
                if display == "omitted" {
                    assert_eq!(thinking_deltas, expected_thinking);
                }
                assert_eq!(
                    arguments.len(),
                    expected.iter().filter(|kind| **kind == "tool_use").count()
                );
                for arguments in arguments.values() {
                    assert_eq!(
                        serde_json::from_str::<Value>(arguments).unwrap(),
                        json!({"path": "/tmp"})
                    );
                }
                assert_eq!(events.last().unwrap().event, "message_stop");
                svc.shutdown().await;
            }
        }
    })
    .await;
}

#[tokio::test]
#[serial]
async fn omitted_thinking_enables_default_http_heartbeats() {
    use dynamo_runtime::config::environment_names::llm::DYN_HTTP_SSE_KEEP_ALIVE_INTERVAL_MS;
    let mut env = ENV.to_vec();
    env.push((DYN_HTTP_SSE_KEEP_ALIVE_INTERVAL_MS, None));
    temp_env::async_with_vars(env, async {
        let script = load_agent_fixture("thinking-tool.sse").await.unwrap();
        let (svc, gate) = HarnessService::start_with_gated_tail(script, 1).await;
        let mut response = svc
            .client
            .post(format!("{}/v1/messages", svc.base_url))
            .timeout(Duration::from_secs(25))
            .json(&json!({"model": MODEL, "max_tokens": 4096, "stream": true,
                "messages": [{"role": "user", "content": "ping"}],
                "thinking": {"type": "adaptive", "display": "omitted"}}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let mut body = String::new();
        tokio::time::timeout(Duration::from_secs(18), async {
            loop {
                let chunk = response
                    .chunk()
                    .await
                    .unwrap()
                    .expect("stream ended before heartbeat");
                body.push_str(std::str::from_utf8(&chunk).unwrap());
                if body.lines().any(|line| line.starts_with(':')) {
                    break;
                }
            }
        })
        .await
        .expect("omitted thinking must send the default 15-second heartbeat");
        assert!(!body.contains("I should inspect"));
        gate.release();
        body.push_str(&response.text().await.unwrap());
        assert!(body.contains("message_stop"));
        svc.shutdown().await;
    })
    .await;
}
