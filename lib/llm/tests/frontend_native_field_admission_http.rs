// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Separate binary because the migration switch is read once per process.
//! This proves rejection before dispatch even when unknown-field dropping is enabled.

use dynamo_llm::protocols::{
    common::llm_backend::BackendOutput,
    openai::{
        DeltaGeneratorExt, chat_completions::NvCreateChatCompletionRequest,
        completions::NvCreateCompletionRequest,
    },
};
use dynamo_runtime::config::environment_names::llm::{
    DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS,
};
use serde_json::{Value, json};

#[allow(dead_code)]
#[path = "common/http_harness.rs"]
mod http_harness;
#[path = "common/ports.rs"]
mod ports;
#[allow(dead_code)]
#[path = "common/scripted_chat_engine.rs"]
mod scripted_chat_engine;

use http_harness::{HarnessService, MODEL, load_agent_fixture};

#[tokio::test]
async fn known_native_fields_are_rejected_before_dispatch_with_ignore_mode_enabled() {
    temp_env::async_with_vars(
        [
            (DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS, Some("1")),
            (DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0")),
        ],
        async {
            let script = load_agent_fixture("text.sse").await.unwrap();
            let svc = HarnessService::start_with_completion_validation([script]).await;
            for stream in [false, true] {
                for value in [json!(null), json!(false), json!(0), json!([]), json!({})] {
                    let response = svc
                        .client
                        .post(format!("{}/v1/chat/completions", svc.base_url))
                        .json(&json!({
                            "model": MODEL,
                            "messages": [{"role": "user", "content": "ping"}],
                            "stream": stream,
                            "watermarking": value,
                        }))
                        .send()
                        .await
                        .unwrap();
                    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
                    let body: Value = response.json().await.unwrap();
                    assert_eq!(body["error"]["code"], 400);
                    assert_eq!(body["error"]["param"], "watermarking");
                    assert_eq!(body["error"]["details"]["field"], "watermarking");
                    assert_eq!(body["error"]["details"]["kind"], "unsupported_field");
                    assert_eq!(body["error"]["details"]["stage"], "request_validation");
                    assert!(body["error"]["details"]["profile"].is_null());
                    let message = body["error"]["message"].as_str().unwrap();
                    assert!(message.contains("watermarking"), "{body}");
                    assert!(message.contains("cannot be ignored"), "{body}");
                    assert!(svc.engine.take_requests().await.is_empty());
                }
            }

            // Positive control proves the switch is on and the route is healthy:
            // an unknown migration field can still reach the scripted engine.
            let response = svc
                .client
                .post(format!("{}/v1/chat/completions", svc.base_url))
                .json(&json!({
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "ping"}],
                    "migration_only_unknown_field": "ignored",
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            let body: Value = response.json().await.unwrap();
            assert!(body["choices"].is_array(), "{body}");
            assert_eq!(svc.engine.take_requests().await.len(), 1);

            // Completion validation runs before completion-engine lookup, even
            // when this ready model only has a chat engine. Misplaced keys must be
            // counted individually, without going through migration ignore.
            for stream in [false, true] {
                for value in [json!(null), json!(false), json!(0), json!([]), json!({})] {
                    let response = svc.client
                        .post(format!("{}/v1/completions", svc.base_url))
                        .json(&json!({
                            "model": MODEL, "prompt": "ping", "stream": stream,
                            "add_generation_prompt": value,
                            "continue_final_message": value,
                        }))
                        .send().await.unwrap();
                    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
                    let body: Value = response.json().await.unwrap();
                    assert!(body["error"]["message"].as_str().unwrap().contains("only supported on /v1/chat/completions"), "{body}");
                    assert!(svc.engine.take_requests().await.is_empty());
                }
            }

            // Exercise production response converters directly, then inspect
            // their counters through the real metrics endpoint. This is not a
            // backend transport or native-server test.
            for stream in [false, true] {
                for root in [false, true] {
                    for nvext in [false, true] {
                        let body = json!({
                            "model": "private-model-label", "messages": [], "prompt": "ping",
                            "stream": stream,
                            "prompt_logprobs": if root { json!(0) } else { Value::Null },
                            "nvext": {"extra_fields": if nvext { vec!["prompt_logprobs"] } else { vec![] }},
                        });
                        let chat: NvCreateChatCompletionRequest = serde_json::from_value(body.clone()).unwrap();
                        let completion: NvCreateCompletionRequest = serde_json::from_value(body).unwrap();
                        let mut chat = chat.response_generator("private-request-label".into());
                        let mut completion = completion.response_generator("private-request-label".into());
                        for payload in [Value::Null, json!([null]), json!("private-invalid-payload")] {
                            let output: BackendOutput = serde_json::from_value(json!({
                                "token_ids": [0], "tokens": ["!"], "text": "!",
                                "engine_data": {"prompt_logprobs": payload},
                            })).unwrap();
                            let expected_error = (root || nvext) && payload.is_string();
                            assert_eq!(chat.choice_from_postprocessor(output.clone()).is_err(), expected_error);
                            assert_eq!(completion.choice_from_postprocessor(output).is_err(), expected_error);
                        }
                    }
                }
            }

            let response = svc
                .client
                .get(format!("{}/metrics", svc.base_url))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            let metrics = response.text().await.unwrap();
            let decisions: Vec<_> = metrics
                .lines()
                .filter(|line| {
                    !line.starts_with('#') && line.contains("_protocol_decisions_total{")
                })
                .collect();
            let rejected = decisions
                .iter()
                .find(|line| line.contains("field=\"watermarking\""))
                .expect("known-field rejection must be exported");
            assert!(rejected.contains("decision=\"reject\""));
            assert!(rejected.contains("endpoint=\"/v1/chat/completions\""));
            assert!(rejected.contains("target=\"unresolved\""));
            assert!(rejected.ends_with(" 10"), "{rejected}");
            let ignored = decisions
                .iter()
                .find(|line| line.contains("decision=\"ignore\""))
                .expect("migration ignore must be exported");
            assert!(ignored.contains("field=\"unknown\""));
            assert!(ignored.ends_with(" 1"), "{ignored}");
            for field in ["add_generation_prompt", "continue_final_message"] {
                let rejected = decisions.iter().find(|line| line.contains(&format!("field=\"{field}\""))).unwrap();
                assert!(rejected.contains("reason=\"wrong_endpoint\""));
                assert!(rejected.contains("endpoint=\"/v1/completions\""));
                assert!(rejected.ends_with(" 10"), "{rejected}");
            }
            for endpoint in ["/v1/chat/completions", "/v1/completions"] {
                let malformed = decisions.iter().find(|line| {
                    line.contains("reason=\"malformed_backend_payload\"")
                        && line.contains(&format!("endpoint=\"{endpoint}\""))
                }).unwrap();
                assert!(malformed.contains("stage=\"response_decode\""));
                assert!(malformed.contains("decision=\"error\""));
                assert!(malformed.contains("target=\"unresolved\""));
                assert!(malformed.contains("field=\"prompt_logprobs\""));
                assert!(malformed.ends_with(" 6"), "{malformed}");
            }
            assert!(!decisions.join("\n").contains("private-"));
            assert!(
                !decisions
                    .join("\n")
                    .contains("migration_only_unknown_field")
            );
            svc.shutdown().await;
        },
    )
    .await;
}
