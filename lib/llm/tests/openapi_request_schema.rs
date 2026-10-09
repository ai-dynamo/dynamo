// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Guard request-schema declarations that derives alone cannot express: input
//! aliases and endpoint-specific extensions. Alias claims are checked against real
//! Serde parsing, not against a framework server or backend execution.
//!
//! Run: `cargo test -p dynamo-llm --no-default-features --test openapi_request_schema`.

use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use utoipa::ToSchema;

#[test]
fn http_export_describes_request_aliases() {
    use dynamo_llm::http::service::{RouteDoc, openapi_docs::generate_openapi_spec};
    let routes = [
        RouteDoc::new(axum::http::Method::POST, "/v1/chat/completions"),
        RouteDoc::new(axum::http::Method::POST, "/v1/completions"),
    ];
    let spec = serde_json::to_value(generate_openapi_spec(&routes)).unwrap();
    fn assert_alias(node: &serde_json::Value, field: &str, alias: &str) -> bool {
        if let Some(schema) = node.get("properties").and_then(|p| p.get(field)) {
            assert_eq!(schema["x-dynamo-input-aliases"], serde_json::json!([alias]));
            assert!(schema.get("x-dynamo-alias-conflict").is_none());
            return true;
        }
        node.get("allOf")
            .and_then(|v| v.as_array())
            .is_some_and(|parts| parts.iter().any(|part| assert_alias(part, field, alias)))
    }
    // Request roots share their registered components, so annotations have one
    // source of truth instead of a second inline copy that could drift.
    for (path, name) in [
        ("/v1/chat/completions", "NvCreateChatCompletionRequest"),
        ("/v1/completions", "NvCreateCompletionRequest"),
    ] {
        let schema =
            &spec["paths"][path]["post"]["requestBody"]["content"]["application/json"]["schema"];
        assert_eq!(
            schema,
            &serde_json::json!({"$ref": format!("#/components/schemas/{name}")})
        );
        assert!(
            spec.pointer(schema["$ref"].as_str().unwrap().strip_prefix('#').unwrap())
                .is_some()
        );
    }
    assert!(assert_alias(
        &spec["components"]["schemas"]["NvCreateChatCompletionRequest"],
        "chat_template_args",
        "chat_template_kwargs"
    ));
    assert!(assert_alias(
        &spec["components"]["schemas"]["dynamo_protocols.chat.ChatCompletionRequestAssistantMessage"],
        "reasoning_content",
        "reasoning",
    ));
}

#[test]
fn advertised_aliases_populate_the_same_fields_and_reject_dual_names() {
    use serde_json::json;
    for name in ["chat_template_args", "chat_template_kwargs"] {
        let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "test", "messages": [], (name): {"enable_thinking": false},
        }))
        .unwrap();
        assert_eq!(
            request.chat_template_args.unwrap()["enable_thinking"],
            false
        );
    }
    // Duplicate spellings are invalid even when their values are equal.
    for alternate in [json!({}), json!({"enable_thinking": false})] {
        assert!(
            serde_json::from_value::<NvCreateChatCompletionRequest>(json!({
                "model": "test", "messages": [], "chat_template_args": {}, "chat_template_kwargs": alternate,
            }))
            .is_err()
        );
    }
    for name in ["reasoning_content", "reasoning"] {
        let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "test", "messages": [{"role": "assistant", "content": "answer", (name): "thought"}],
        })).unwrap();
        let serialized = serde_json::to_value(request).unwrap();
        assert_eq!(serialized["messages"][0]["reasoning_content"], "thought");
        assert!(serialized["messages"][0].get("reasoning").is_none());
    }
    for alternate in ["a", "b"] {
        assert!(serde_json::from_value::<NvCreateChatCompletionRequest>(json!({
            "model": "test", "messages": [{"role": "assistant", "content": "answer", "reasoning": alternate, "reasoning_content": "a"}],
        })).is_err());
    }
}

#[test]
fn chat_only_fields_are_exported_without_advertising_them_for_completions() {
    // The runtime shares CommonExt, but completions rejects these chat options.
    // Exporting the shared runtime shape verbatim would overstate its contract.
    let mut schemas = Vec::new();
    NvCreateChatCompletionRequest::schemas(&mut schemas);
    let chat_common = schemas
        .iter()
        .find(|(name, _)| name == "ChatCommonExt")
        .unwrap();
    let chat_schema = serde_json::to_value(&chat_common.1).unwrap();
    let properties = &chat_schema["allOf"][1]["properties"];
    for field in ["add_generation_prompt", "continue_final_message"] {
        assert_eq!(
            properties[field]["type"],
            serde_json::json!(["boolean", "null"])
        );
    }
    let shared = schemas
        .iter()
        .find(|(name, _)| name == "CommonExt")
        .unwrap();
    let shared = serde_json::to_value(&shared.1).unwrap();
    assert!(shared["properties"].get("add_generation_prompt").is_none());
    assert!(shared["properties"].get("continue_final_message").is_none());
}
