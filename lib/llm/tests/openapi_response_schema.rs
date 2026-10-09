// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Guard native response-schema wiring and the JSON serialization it describes.
//!
//! Structural assertions cover endpoint/media-type registration, required nullable
//! fields, internal-field exclusion, and the configured reasoning key. Shared fixtures
//! separately pin actual Serde output; this suite does not validate that output with a
//! JSON Schema validator. Dependency import slots are resolved by offline composition.
//! These checks do not prove framework compatibility, inference, or SSE ordering and
//! termination: the streaming schema describes successful JSON data payloads only.
//!
//! Run: `cargo test -p dynamo-llm --no-default-features --test openapi_response_schema`.

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
    // A derived component is not sufficient: each supported success response must
    // reference the correct registered type for unary JSON and streaming payloads.
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
        // A machine-readable marker distinguishes a successful SSE data payload
        // schema from a schema for the whole transport body. Unary JSON is unmarked.
        assert_eq!(
            success["content"]["text/event-stream"]["x-dynamo-sse-data-schema"],
            true
        );
        assert!(
            success["content"]["application/json"]
                .get("x-dynamo-sse-data-schema")
                .is_none()
        );
        for (media, name) in [("application/json", unary), ("text/event-stream", stream)] {
            assert_eq!(
                success["content"][media]["schema"]["$ref"],
                format!("#/components/schemas/{name}")
            );
            assert!(spec["components"]["schemas"].get(name).is_some());
        }
    }
    // Matching only the path would incorrectly attach chat schemas to other methods;
    // unrelated endpoints must not inherit either completion response contract.
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
    // Worker/frontend transport fields are not part of the client-facing stream.
    // Prompt logprobs belong to the unary response, not streaming JSON chunks.
    let stream = spec["components"]["schemas"]["NvCreateChatCompletionStreamResponse"].to_string();
    for internal in ["llm_metrics", "tool_call_completion", "prompt_logprobs"] {
        assert!(!stream.contains(internal), "internal field {internal}");
    }
    assert!(
        spec["components"]["schemas"]["NvCreateChatCompletionResponse"]
            .to_string()
            .contains("prompt_logprobs")
    );
    // An explicit unresolved dependency slot must not masquerade as a complete
    // legacy completion contract; offline composition owns its resolution.
    assert_eq!(
        spec["components"]["schemas"]["async_openai.CreateCompletionResponse"]["x-dynamo-schema-import"]
            ["type"],
        "CreateCompletionResponse"
    );
}

#[test]
fn always_serialized_nullable_response_fields_are_required() {
    // Serde emits these keys even for None. Nullable permits null, not omission.
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
    // The export must follow the same wire-name option as the response wrapper,
    // rather than always publishing the canonical Rust field name.
    for (field, absent) in [
        (ReasoningField::ReasoningContent, "reasoning"),
        (ReasoningField::Reasoning, "reasoning_content"),
    ] {
        let spec = document(field);
        for name in [
            "ChatCompletionResponseMessage",
            "ChatCompletionStreamResponseDelta",
        ] {
            let properties = &spec["components"]["schemas"]
                [format!("dynamo_protocols.chat.{name}")]["properties"];
            assert!(properties.get(field.as_str()).is_some());
            assert!(properties.get(absent).is_none());
        }
    }
    let raw = json!({"choices": [{"message": {"reasoning_content": "think"}, "delta": {"reasoning_content": "think"}}]});
    let routed =
        serde_json::to_value(RoutedReasoning::new(raw, ReasoningField::Reasoning)).unwrap();
    assert_eq!(routed["choices"][0]["message"]["reasoning"], "think");
    assert_eq!(routed["choices"][0]["delta"]["reasoning"], "think");
}

#[test]
fn shared_response_fixtures_are_actual_serialized_output() {
    // Inputs construct real response values; outputs pin serialization, including
    // null-versus-omitted fields. They are not captured inference/SSE transcripts.
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
