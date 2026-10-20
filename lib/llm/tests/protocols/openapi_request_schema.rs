// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Guard request-schema declarations that derives alone cannot express: input
//! aliases, omission policies and endpoint-specific extensions. Claims are checked
//! against real Serde parsing and sampling extraction, not backend execution.
//!
//! Run: `cargo test -p dynamo-llm --no-default-features --test protocols openapi_request_schema::`.

use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use utoipa::ToSchema;

/// Return the unique inline/flattened `n` field, failing on missing or duplicate declarations.
fn n_schema(node: &serde_json::Value) -> &serde_json::Value {
    /// Collect only this object's inline/allOf `n` fields, not nested request objects.
    fn collect<'a>(node: &'a serde_json::Value, fields: &mut Vec<&'a serde_json::Value>) {
        if let Some(field) = node
            .get("properties")
            .and_then(|properties| properties.get("n"))
        {
            fields.push(field);
        }
        if let Some(parts) = node.get("allOf").and_then(|parts| parts.as_array()) {
            for part in parts {
                collect(part, fields);
            }
        }
    }
    let mut fields = Vec::new();
    collect(node, &mut fields);
    assert_eq!(fields.len(), 1, "expected one canonical n field");
    fields[0]
}

/// The HTTP/file export declares the handlers' default without changing other field constraints.
#[test]
fn http_export_declares_one_choice_by_default() {
    use dynamo_llm::http::service::{
        OpenApiRouteKind, RouteDoc, openapi_docs::generate_openapi_spec,
    };
    use dynamo_llm::protocols::openai::completions::NvCreateCompletionRequest;

    let routes = [
        RouteDoc::new(axum::http::Method::POST, "/v1/chat/completions")
            .with_openapi_kind(OpenApiRouteKind::ChatCompletions),
        RouteDoc::new(axum::http::Method::POST, "/v1/completions")
            .with_openapi_kind(OpenApiRouteKind::Completions),
    ];
    let spec = generate_openapi_spec(&routes).unwrap();
    let mut schemas = Vec::new();
    NvCreateChatCompletionRequest::schemas(&mut schemas);
    NvCreateCompletionRequest::schemas(&mut schemas);
    for (path, wrapper, name) in [
        (
            "/v1/chat/completions",
            "NvCreateChatCompletionRequest",
            "dynamo_protocols.chat.CreateChatCompletionRequest",
        ),
        (
            "/v1/completions",
            "NvCreateCompletionRequest",
            "dynamo_protocols.completion.CreateCompletionRequest",
        ),
    ] {
        assert_eq!(
            spec["paths"][path]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"],
            format!("#/components/schemas/{wrapper}"),
        );
        let reference = format!("#/components/schemas/{name}");
        assert_eq!(
            spec["components"]["schemas"][wrapper]["allOf"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|part| part["$ref"] == reference)
                .count(),
            1,
            "the endpoint must use the annotated component exactly once",
        );
        let original = &schemas.iter().find(|(key, _)| key == name).unwrap().1;
        let original = serde_json::to_value(original).unwrap();
        let original = n_schema(&original);
        let exported = n_schema(&spec["components"]["schemas"][name]);
        assert!(exported.get("x-dynamo-default-policy").is_none());
        assert_eq!(exported["default"], 1);
        let description = exported["description"].as_str().unwrap();
        assert!(description.contains("When omitted or null, Dynamo sets n to 1"));
        let mut unchanged = exported.clone();
        if let Some(default) = original.get("default") {
            unchanged["default"] = default.clone();
        } else {
            unchanged.as_object_mut().unwrap().remove("default");
        }
        if let Some(description) = original.get("description") {
            assert!(
                exported["description"]
                    .as_str()
                    .unwrap()
                    .starts_with(description.as_str().unwrap())
            );
            unchanged["description"] = description.clone();
        } else {
            unchanged.as_object_mut().unwrap().remove("description");
        }
        assert_eq!(&unchanged, original);
    }
}

#[test]
fn http_export_describes_request_aliases() {
    use dynamo_llm::http::service::{
        OpenApiRouteKind, RouteDoc, openapi_docs::generate_openapi_spec,
    };
    let routes = [
        RouteDoc::new(axum::http::Method::POST, "/v1/chat/completions")
            .with_openapi_kind(OpenApiRouteKind::ChatCompletions),
        RouteDoc::new(axum::http::Method::POST, "/v1/completions")
            .with_openapi_kind(OpenApiRouteKind::Completions),
        RouteDoc::new(axum::http::Method::POST, "/v1/embeddings")
            .with_openapi_kind(OpenApiRouteKind::Embeddings),
        RouteDoc::new(axum::http::Method::POST, "/v1/responses")
            .with_openapi_kind(OpenApiRouteKind::Responses),
    ];
    let spec = generate_openapi_spec(&routes).unwrap();
    /// Check an alias on an inline/flattened field and return whether it was found.
    fn assert_alias(node: &serde_json::Value, field: &str, alias: &str) -> bool {
        if let Some(schema) = node.get("properties").and_then(|p| p.get(field)) {
            assert_eq!(schema["x-dynamo-input-aliases"], serde_json::json!([alias]));
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
        ("/v1/embeddings", "NvCreateEmbeddingRequest"),
        ("/v1/responses", "NvCreateResponse"),
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
    assert_eq!(properties["add_generation_prompt"]["default"], true);
    assert!(
        properties["add_generation_prompt"]["description"]
            .as_str()
            .unwrap()
            .contains("continue_final_message=true")
    );
    assert!(
        properties["continue_final_message"]["description"]
            .as_str()
            .unwrap()
            .contains("add_generation_prompt=false")
    );
    let shared = schemas
        .iter()
        .find(|(name, _)| name == "CommonExt")
        .unwrap();
    let shared = serde_json::to_value(&shared.1).unwrap();
    assert!(shared["properties"].get("add_generation_prompt").is_none());
    assert!(shared["properties"].get("continue_final_message").is_none());
}
