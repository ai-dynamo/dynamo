// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::Method;
use dynamo_llm::{
    http::service::{RouteDoc, openapi_docs},
    protocols::openai::{
        chat_completions::{NvCreateChatCompletionResponse, NvCreateChatCompletionStreamResponse},
        completions::NvCreateCompletionResponse,
    },
    reasoning_field::{ReasoningField, RoutedReasoning},
};
use serde_json::{Value, json};

fn document(field: ReasoningField) -> Value {
    serde_json::to_value(openapi_docs::generate_openapi_spec_with_reasoning_field(
        &[
            RouteDoc::new(Method::POST, "/v1/chat/completions"),
            RouteDoc::new(Method::POST, "/v1/completions"),
            RouteDoc::new(Method::GET, "/v1/chat/completions"),
            RouteDoc::new(Method::POST, "/v1/embeddings"),
        ],
        field,
    ))
    .unwrap()
}

#[test]
fn successful_media_types_reference_registered_payloads_only() {
    let spec = document(ReasoningField::DEFAULT);
    for (path, unary, stream) in [
        (
            "/v1/chat/completions",
            "NvCreateChatCompletionResponse",
            "NvCreateChatCompletionStreamResponse",
        ),
        (
            "/v1/completions",
            "NvCreateCompletionResponse",
            "NvCreateCompletionResponse",
        ),
    ] {
        let success = &spec["paths"][path]["post"]["responses"]["200"];
        assert!(success["description"].as_str().unwrap().contains("[DONE]"));
        for (media, name) in [("application/json", unary), ("text/event-stream", stream)] {
            assert_eq!(
                success["content"][media]["schema"]["$ref"],
                format!("#/components/schemas/{name}")
            );
            assert!(spec["components"]["schemas"].get(name).is_some());
        }
    }
    assert!(
        spec["paths"]["/v1/chat/completions"]["get"]["responses"]["200"]
            .get("content")
            .is_none()
    );
    assert!(
        spec["paths"]["/v1/embeddings"]["post"]["responses"]["200"]
            .get("content")
            .is_none()
    );
    let stream = spec["components"]["schemas"]["NvCreateChatCompletionStreamResponse"].to_string();
    for internal in ["llm_metrics", "tool_call_completion", "prompt_logprobs"] {
        assert!(!stream.contains(internal), "internal field {internal}");
    }
    assert!(
        spec["components"]["schemas"]["NvCreateChatCompletionResponse"]
            .to_string()
            .contains("prompt_logprobs")
    );
    assert_eq!(
        spec["components"]["schemas"]["async_openai.CreateCompletionResponse"]["x-dynamo-schema-import"]
            ["type"],
        "CreateCompletionResponse"
    );
}

#[test]
fn always_serialized_nullable_response_fields_are_required() {
    let spec = document(ReasoningField::DEFAULT);
    for (name, fields) in [
        ("ChatChoice", &["finish_reason", "logprobs"][..]),
        ("ChatChoiceStream", &["finish_reason", "logprobs"][..]),
        ("ChatCompletionResponseMessage", &["content", "refusal"][..]),
        ("ChatChoiceLogprobs", &["content", "refusal"][..]),
        ("ChatCompletionTokenLogprob", &["bytes"][..]),
    ] {
        let schema = &spec["components"]["schemas"][format!("dynamo_protocols.chat.{name}")];
        for field in fields {
            assert!(
                schema["required"]
                    .as_array()
                    .unwrap()
                    .contains(&json!(field)),
                "{name}.{field}"
            );
        }
    }
}

#[test]
fn configured_reasoning_name_matches_unary_and_stream_serialization() {
    let spec = document(ReasoningField::Reasoning);
    for name in [
        "ChatCompletionResponseMessage",
        "ChatCompletionStreamResponseDelta",
    ] {
        let properties =
            &spec["components"]["schemas"][format!("dynamo_protocols.chat.{name}")]["properties"];
        assert!(properties.get("reasoning").is_some());
        assert!(properties.get("reasoning_content").is_none());
    }
    let raw = json!({"choices": [{"message": {"reasoning_content": "think"}, "delta": {"reasoning_content": "think"}}]});
    let routed =
        serde_json::to_value(RoutedReasoning::new(raw, ReasoningField::Reasoning)).unwrap();
    assert_eq!(routed["choices"][0]["message"]["reasoning"], "think");
    assert_eq!(routed["choices"][0]["delta"]["reasoning"], "think");
}

#[test]
fn shared_response_fixtures_are_actual_serialized_output() {
    let cases: Value =
        serde_json::from_str(include_str!("fixtures/openapi/responses.json")).unwrap();
    for case in cases["cases"].as_array().unwrap() {
        let input = case["input"].clone();
        let serialized = match case["kind"].as_str().unwrap() {
            "chat" => serde_json::to_value(
                serde_json::from_value::<NvCreateChatCompletionResponse>(input).unwrap(),
            ),
            "stream" => serde_json::to_value(
                serde_json::from_value::<NvCreateChatCompletionStreamResponse>(input).unwrap(),
            ),
            "completion" => serde_json::to_value(
                serde_json::from_value::<NvCreateCompletionResponse>(input).unwrap(),
            ),
            other => panic!("unknown response case kind: {other}"),
        }
        .unwrap();
        assert_eq!(serialized, case["output"], "{}", case["name"]);
    }
}
