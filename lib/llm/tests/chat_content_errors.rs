// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::config::environment_names::llm::DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS;
use reqwest::StatusCode;
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

use http_harness::{HarnessService, MODEL};

#[tokio::test]
#[serial]
async fn unsupported_chat_content_types_have_bounded_actionable_errors_before_dispatch() {
    temp_env::async_with_vars(
        [(DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0"))],
        async {
            let svc = HarnessService::start(Vec::new()).await;
            for streaming in [false, true] {
                for role in ["user", "tool", "assistant", "system", "developer"] {
                    for tag in [
                        "file".to_string(),
                        "future_media".to_string(),
                        "secret\n".repeat(1000),
                    ] {
                        let body = json!({
                            "model": MODEL, "stream": streaming, "messages": [
                                {"role": "system", "content": "instructions"},
                                {"role": role, "tool_call_id": "call-1", "content": [
                                    {"type": "text", "text": "summarize"},
                                    {"type": tag, "file": {
                                        "file_data": "private-document", "filename": "report.pdf"
                                    }}
                                ]}
                            ]
                        });
                        let response = svc
                            .client
                            .post(format!("{}/v1/chat/completions", svc.base_url))
                            .json(&body)
                            .send()
                            .await
                            .unwrap();
                        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
                        let bytes = response.bytes().await.unwrap();
                        assert!(bytes.len() < 512);
                        let error: Value = serde_json::from_slice(&bytes).unwrap();
                        assert_eq!(error["type"], "unsupported_content_type", "{error}");
                        assert_eq!(error["code"], 400);
                        let message = error["message"].as_str().unwrap();
                        assert!(message.contains("messages[1].content[1]"), "{error}");
                        if tag.len() < 64 {
                            assert!(message.contains(&tag));
                        }
                        assert!(!message.contains("secret"));
                        assert!(!message.contains("private-document"));
                    }
                }
            }
            assert!(svc.engine.take_requests().await.is_empty());
            svc.shutdown().await;
        },
    )
    .await;
}

#[tokio::test]
#[serial]
async fn malformed_chat_content_keeps_the_existing_error_classification() {
    temp_env::async_with_vars(
        [(DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0"))],
        async {
            let svc = HarnessService::start(Vec::new()).await;
            let messages = [
                json!({"role": "user", "content": [{"type": "image_url", "image_url": 7}]}),
                json!({"role": "user", "content": [{"type": 7}]}),
                json!({"role": "unknown", "content": [{"type": "file"}]}),
            ];
            let duplicate = r#"{"role":"user","content":[{"type":"text","type":"file"}]}"#;
            let valid = json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": [{"type": "file"}]}]
            })
            .to_string();
            let bodies = messages
                .into_iter()
                .map(|message| json!({"model": MODEL, "messages": [message]}).to_string())
                .chain([
                    format!(r#"{{"model":"{MODEL}","messages":[{duplicate}]}}"#),
                    valid[..valid.len() - 1].to_string(),
                ]);
            for body in bodies {
                let response = svc
                    .client
                    .post(format!("{}/v1/chat/completions", svc.base_url))
                    .header("content-type", "application/json")
                    .body(body)
                    .send()
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::BAD_REQUEST);
                let error: Value = response.json().await.unwrap();
                assert_eq!(error["type"], "Bad Request", "{error}");
            }
            assert!(svc.engine.take_requests().await.is_empty());
            svc.shutdown().await;
        },
    )
    .await;
}
