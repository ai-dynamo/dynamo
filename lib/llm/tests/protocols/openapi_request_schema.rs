// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Guard request-schema declarations that derives alone cannot express: input
//! aliases and endpoint-specific extensions. Alias claims are checked against real
//! Serde parsing, not against a framework server or backend execution.
//!
//! Run: `cargo test -p dynamo-llm --no-default-features --test protocols openapi_request_schema::`.

use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use utoipa::ToSchema;

/// Find a root field through the export's known acyclic allOf/local-ref chain.
/// Returns None for absence; never matches the same name in nested request values.
fn root_property<'a>(
    spec: &'a serde_json::Value,
    node: &'a serde_json::Value,
    name: &str,
) -> Option<&'a serde_json::Value> {
    if let Some(field) = node.get("properties").and_then(|fields| fields.get(name)) {
        return Some(field);
    }
    if let Some(reference) = node.get("$ref").and_then(|value| value.as_str()) {
        return root_property(spec, spec.pointer(reference.strip_prefix('#')?)?, name);
    }
    node.get("allOf")?
        .as_array()?
        .iter()
        .find_map(|part| root_property(spec, part, name))
}

/// Extra-map token controls must be declared without changing their runtime owner.
/// Preserve null/omission, integer items and the actual admission count limit;
/// inference and vocabulary/error semantics are checked separately by real servers.
#[test]
fn passthrough_token_controls_match_admission_shape() {
    use dynamo_llm::http::service::{RouteDoc, openapi_docs::generate_openapi_spec};
    use dynamo_llm::protocols::openai::{completions::NvCreateCompletionRequest, validate};
    let spec = generate_openapi_spec(&[
        RouteDoc::new(axum::http::Method::POST, "/v1/chat/completions"),
        RouteDoc::new(axum::http::Method::POST, "/v1/completions"),
    ]);
    let controls = &spec["components"]["schemas"]["PassthroughTokenControls"];
    for name in ["allowed_token_ids", "stop_token_ids"] {
        let field = &controls["properties"][name];
        assert_eq!(field["type"], serde_json::json!(["array", "null"]));
        assert_eq!(field["items"]["type"], "integer");
        assert!(
            !controls["required"]
                .as_array()
                .is_some_and(|fields| fields.iter().any(|field| field == name))
        );
        for root in ["NvCreateChatCompletionRequest", "NvCreateCompletionRequest"] {
            assert_eq!(
                root_property(&spec, &spec["components"]["schemas"][root], name),
                Some(field)
            );
        }
    }
    assert_eq!(
        controls["properties"]["stop_token_ids"]["maxItems"],
        validate::MAX_STOP_SEQUENCES
    );
    let request: NvCreateCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "test", "prompt": "hello", "allowed_token_ids": [22], "stop_token_ids": null
    }))
    .unwrap();
    assert_eq!(
        request.unsupported_fields["allowed_token_ids"],
        serde_json::json!([22])
    );
    assert!(!request.unsupported_fields.contains_key("stop_token_ids"));
    validate::validate_no_unsupported_fields(&request.unsupported_fields).unwrap();
}

/// Word exclusions are optional lists, not nullable or opaque extra fields.
/// Shape is exported here; runtime tests separately establish sampling effects.
#[test]
fn bad_words_schema_is_an_optional_nonnullable_string_array() {
    use dynamo_llm::http::service::{RouteDoc, openapi_docs::generate_openapi_spec};
    let routes = [
        RouteDoc::new(axum::http::Method::POST, "/v1/chat/completions"),
        RouteDoc::new(axum::http::Method::POST, "/v1/completions"),
    ];
    let spec = serde_json::to_value(generate_openapi_spec(&routes)).unwrap();
    let common = &spec["components"]["schemas"]["CommonExt"];
    let words = &common["properties"]["bad_words"];
    assert_eq!(words["type"], "array");
    assert_eq!(words["items"]["type"], "string");
    assert!(
        !common["required"]
            .as_array()
            .is_some_and(|required| required.iter().any(|name| name == "bad_words"))
    );
    for root in ["NvCreateChatCompletionRequest", "NvCreateCompletionRequest"] {
        assert_eq!(
            root_property(&spec, &spec["components"]["schemas"][root], "bad_words"),
            Some(words)
        );
    }
}

/// The native HTTP export must expose completion's full format union and its
/// required metadata/tag members. This checks declaration, not grammar execution.
#[test]
fn completion_response_format_exports_json_and_both_structural_tag_forms() {
    use dynamo_llm::http::service::{RouteDoc, openapi_docs::generate_openapi_spec};
    let spec = serde_json::to_value(generate_openapi_spec(&[RouteDoc::new(
        axum::http::Method::POST,
        "/v1/completions",
    )]))
    .unwrap();
    let schemas = &spec["components"]["schemas"];
    assert!(
        schemas["NvCreateCompletionRequest"]
            .to_string()
            .contains("#/components/schemas/CompletionResponseFormat")
    );
    let format = &schemas["CompletionResponseFormat"]["oneOf"];
    assert_eq!(format.as_array().unwrap().len(), 4);
    for kind in ["text", "json_object", "json_schema", "structural_tag"] {
        assert!(format.to_string().contains(kind), "missing {kind}");
    }
    assert_eq!(
        schemas["CompletionJsonSchema"]["required"],
        serde_json::json!(["name"])
    );
    let tags = &schemas["CompletionStructuralTag"]["anyOf"];
    assert_eq!(tags.as_array().unwrap().len(), 2);
    assert_eq!(tags[0]["required"], serde_json::json!(["format"]));
    assert_eq!(
        tags[1]["required"],
        serde_json::json!(["structures", "triggers"])
    );
    // Both shapes can be present and must reach the backend without dropping
    // either grammar. A oneOf export would misdescribe this accepted overlap.
    let request: dynamo_llm::protocols::openai::completions::NvCreateCompletionRequest =
        serde_json::from_value(serde_json::json!({
            "model": "test", "prompt": "hello", "response_format": {
                "type": "structural_tag", "format": {"type":"const_string", "value":"violet"},
                "structures": [], "triggers": []
            }
        }))
        .unwrap();
    let encoded = serde_json::to_value(request.response_format).unwrap();
    assert_eq!(encoded["structures"], serde_json::json!([]));
    assert_eq!(encoded["triggers"], serde_json::json!([]));
    assert_eq!(
        encoded["format"],
        serde_json::json!({"type":"const_string", "value":"violet"})
    );
}

/// Both endpoints advertise the nested constraint without claiming that every
/// JSON value is a schema. Semantic grammar validity still belongs to the backend.
#[test]
fn structured_outputs_is_exported_with_its_json_input_union() {
    use dynamo_llm::http::service::{RouteDoc, openapi_docs::generate_openapi_spec};
    let routes = [
        RouteDoc::new(axum::http::Method::POST, "/v1/chat/completions"),
        RouteDoc::new(axum::http::Method::POST, "/v1/completions"),
    ];
    let spec = serde_json::to_value(generate_openapi_spec(&routes)).unwrap();
    let schemas = &spec["components"]["schemas"];
    let nested = &schemas["CommonExt"]["properties"]["structured_outputs"];
    assert!(
        nested
            .to_string()
            .contains("#/components/schemas/StructuredOutputs")
    );
    let json_input = &schemas["StructuredOutputs"]["properties"]["json"];
    assert_eq!(
        json_input["type"],
        serde_json::json!(["object", "string", "null"])
    );
    for root in ["NvCreateChatCompletionRequest", "NvCreateCompletionRequest"] {
        let request = schemas[root].to_string();
        // ChatCommonExt derives from the shared CommonExt; completion uses it directly.
        assert!(request.contains("CommonExt"), "{root} lost its extensions");
    }
}

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
