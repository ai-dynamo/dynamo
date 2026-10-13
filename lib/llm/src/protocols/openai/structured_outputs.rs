// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Normalize vLLM's HTTP structured_outputs into the existing worker constraint.
//!
//! This is admission and transport, not a grammar compiler. Backend configuration
//! owns grammar support, reasoning-phase activation and modifier effectiveness.

use dynamo_protocols::types::ResponseFormat;
use dynamo_runtime::error::{DynamoError, ErrorType};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use utoipa::ToSchema;

use crate::protocols::common::GuidedDecodingOptions;

/// A vLLM-style generation constraint; exactly one constraint must be non-null.
///
/// Chat response_format overrides the matching constraint member, then the same
/// exclusivity rule applies. Legacy guided_* fields cannot be mixed with this
/// object. Modifiers are forwarded to the backend, whose configuration determines
/// their effect; admission does not promise a per-request backend override.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, Default)]
pub struct StructuredOutputs {
    /// JSON Schema object or encoded schema string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(schema_with = json_constraint_schema)]
    pub json: Option<Value>,
    /// Regular expression constraining the generated content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,
    /// Allowed complete strings; an empty list is invalid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub choice: Option<Vec<String>>,
    /// Backend-supported grammar text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,
    /// Request an arbitrary JSON object. False does not define a constraint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_object: Option<bool>,
    /// JSON-encoded structural-tag grammar.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structural_tag: Option<String>,
    /// Backend whitespace modifier; support depends on the engine configuration.
    #[serde(default)]
    pub disable_any_whitespace: bool,
    /// Backend object-property modifier; support depends on the engine configuration.
    #[serde(default)]
    pub disable_additional_properties: bool,
    /// Backend-specific whitespace pattern.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub whitespace_pattern: Option<String>,
}

/// Describe the accepted outer JSON types, not an unconstrained serde_json::Value.
/// Encoded-schema validity and constraint exclusivity are checked during admission.
fn json_constraint_schema() -> utoipa::openapi::RefOr<utoipa::openapi::Schema> {
    use utoipa::openapi::schema::{ObjectBuilder, SchemaType, Type};

    ObjectBuilder::new()
        .schema_type(SchemaType::Array(vec![
            Type::Object,
            Type::String,
            Type::Null,
        ]))
        .into()
}

/// Build a bounded, typed client error without including a potentially large schema.
pub(super) fn invalid(message: &'static str) -> anyhow::Error {
    DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message)
        .build()
        .into()
}

impl StructuredOutputs {
    /// Validate one constraint and normalize it for worker dispatch.
    ///
    /// An optional chat response_format replaces the corresponding member, matching
    /// vLLM's override order; it cannot erase a different active constraint. Returns
    /// owned canonical options without mutating the request. Empty/conflicting or
    /// malformed inputs return InvalidArgument. Actual grammar compilation remains
    /// backend-owned; a syntactically valid JSON object is not proof of its validity.
    pub(crate) fn to_guided(
        &self,
        response_format: Option<&ResponseFormat>,
    ) -> anyhow::Result<GuidedDecodingOptions> {
        self.validate_selection()?;
        let mut effective = self.clone();
        match response_format {
            Some(ResponseFormat::JsonObject) => effective.json_object = Some(true),
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                effective.json = Some(json_schema.schema.clone());
            }
            Some(ResponseFormat::Text) | None => {}
        }
        effective.validate_shape()?;
        let json_schema = if effective.json_object == Some(true) {
            Some(json!({"type": "object"}))
        } else {
            effective.json
        };
        let mut result = GuidedDecodingOptions::validated(
            json_schema,
            effective.regex,
            effective.choice,
            effective.grammar,
            None,
            effective.whitespace_pattern,
            effective.structural_tag.map(Value::String),
        )?;
        // Keep default/legacy wire output unchanged. Present modifiers are optional
        // additions; the pinned vLLM backend currently uses its server-level policy.
        result.disable_any_whitespace = Some(self.disable_any_whitespace);
        result.disable_additional_properties = Some(self.disable_additional_properties);
        Ok(result)
    }

    /// Check the original selection before overrides, without compiling its content.
    /// A same-kind response_format may replace invalid content, but cannot repair
    /// missing/multiple constraints or an invalid JSON field's outer type.
    fn validate_selection(&self) -> anyhow::Result<()> {
        let count = [
            self.json.is_some(),
            self.regex.is_some(),
            self.choice.is_some(),
            self.grammar.is_some(),
            self.json_object.is_some(),
            self.structural_tag.is_some(),
        ]
        .into_iter()
        .filter(|active| *active)
        .count();
        if count != 1 {
            return Err(invalid(
                "structured_outputs requires exactly one non-null constraint",
            ));
        }
        if self
            .json
            .as_ref()
            .is_some_and(|schema| !schema.is_object() && !schema.is_string())
        {
            return Err(invalid(
                "structured_outputs.json must be a schema object or encoded schema string",
            ));
        }
        Ok(())
    }

    /// Reject malformed effective constraints after response_format replacement.
    /// Backend-specific grammar compilation still occurs in the worker.
    fn validate_shape(&self) -> anyhow::Result<()> {
        self.validate_selection()?;
        if self.json_object == Some(false) {
            return Err(invalid(
                "structured_outputs.json_object must be true to define a constraint",
            ));
        }
        if self.choice.as_ref().is_some_and(Vec::is_empty) {
            return Err(invalid("structured_outputs.choice must not be empty"));
        }
        if let Some(schema) = &self.json {
            let valid = match schema {
                Value::Object(_) => true,
                Value::String(encoded) => serde_json::from_str::<Value>(encoded).is_ok(),
                _ => false,
            };
            if !valid {
                return Err(invalid(
                    "structured_outputs.json must be a schema object or valid JSON schema string",
                ));
            }
        }
        if let Some(encoded) = &self.structural_tag {
            let value: Value = serde_json::from_str(encoded).map_err(|_| {
                invalid("structured_outputs.structural_tag must encode a JSON object")
            })?;
            if !value.is_object() || value["type"] != "structural_tag" {
                return Err(invalid(
                    "structured_outputs.structural_tag must encode a structural_tag object",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::ValidateRequest;
    use crate::protocols::common::SamplingOptionsProvider;
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use crate::protocols::openai::completions::NvCreateCompletionRequest;

    /// The real request types must consume the nested field and deliver its constraint.
    #[test]
    fn request_admission_and_sampling_share_nested_constraints() {
        for nested in [
            json!({"json": {"type": "object"}}),
            json!({"regex": "xyz"}),
            json!({"choice": ["xyz"]}),
            json!({"grammar": "root ::= \"xyz\""}),
            json!({"json_object": true}),
        ] {
            let expected: StructuredOutputs = serde_json::from_value(nested.clone()).unwrap();
            let expected = serde_json::to_value(expected.to_guided(None).unwrap()).unwrap();
            let chat: NvCreateChatCompletionRequest = serde_json::from_value(json!({
                "model": "test", "messages": [{"role": "user", "content": "hello"}],
                "structured_outputs": nested,
            }))
            .unwrap();
            let completion: NvCreateCompletionRequest = serde_json::from_value(json!({
                "model": "test", "prompt": "hello", "structured_outputs": nested,
            }))
            .unwrap();
            chat.validate().unwrap();
            completion.validate().unwrap();
            assert_eq!(
                serde_json::to_value(
                    chat.extract_sampling_options()
                        .unwrap()
                        .guided_decoding
                        .unwrap()
                )
                .unwrap(),
                expected
            );
            assert_eq!(
                serde_json::to_value(
                    completion
                        .extract_sampling_options()
                        .unwrap()
                        .guided_decoding
                        .unwrap()
                )
                .unwrap(),
                expected
            );
        }
    }

    /// Null is omission, but empty/conflicting constraints fail before sampling.
    #[test]
    fn null_and_invalid_constraints_keep_distinct_admission() {
        let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "test", "messages": [{"role": "user", "content": "hello"}],
            "structured_outputs": null,
        }))
        .unwrap();
        request.validate().unwrap();
        assert!(
            request
                .extract_sampling_options()
                .unwrap()
                .guided_decoding
                .is_none()
        );
        for nested in [
            json!({}),
            json!({"json": {}, "regex": "x"}),
            json!({"choice": []}),
            json!({"json": 1}),
        ] {
            let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
                "model": "test", "messages": [{"role": "user", "content": "hello"}],
                "structured_outputs": nested,
            }))
            .unwrap();
            assert!(request.validate().is_err());
            assert!(request.extract_sampling_options().is_err());
        }
    }

    /// A response-format schema replaces nested JSON, not another constraint kind.
    #[test]
    fn response_format_overrides_only_matching_constraint() {
        let format: ResponseFormat = serde_json::from_value(json!({
            "type": "json_schema", "json_schema": {"name": "value", "schema": {"const": 9}}
        }))
        .unwrap();
        for schema in [json!({"const": 7}), json!("{")] {
            let nested: StructuredOutputs =
                serde_json::from_value(json!({"json": schema})).unwrap();
            assert_eq!(
                nested.to_guided(Some(&format)).unwrap().json,
                Some(json!({"const": 9}))
            );
        }
        let nested: StructuredOutputs =
            serde_json::from_value(json!({"json_object": false})).unwrap();
        assert_eq!(
            nested
                .to_guided(Some(&ResponseFormat::JsonObject))
                .unwrap()
                .json,
            Some(json!({"type": "object"}))
        );
        for malformed in [
            json!({}),
            json!({"json": 1}),
            json!({"json": {}, "regex": "x"}),
        ] {
            let nested: StructuredOutputs = serde_json::from_value(malformed).unwrap();
            assert!(nested.to_guided(Some(&format)).is_err());
        }
        let nested: StructuredOutputs = serde_json::from_value(json!({"regex": "x"})).unwrap();
        assert!(nested.to_guided(Some(&format)).is_err());
    }

    /// The actual chat wrapper must supply response_format to both admission and dispatch.
    #[test]
    fn chat_request_dispatches_the_response_format_override() {
        let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "test", "messages": [{"role": "user", "content": "hello"}],
            "structured_outputs": {"json": {"const": 7}},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "value", "schema": {"const": 9}
            }},
        }))
        .unwrap();
        request.validate().unwrap();
        assert_eq!(
            request
                .extract_sampling_options()
                .unwrap()
                .guided_decoding
                .unwrap()
                .json,
            Some(json!({"const": 9}))
        );
    }

    /// A nested constraint cannot silently replace a caller's legacy guided constraint.
    #[test]
    fn nested_and_legacy_constraints_are_rejected_together() {
        let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "test", "messages": [{"role": "user", "content": "hello"}],
            "structured_outputs": {"regex": "violet"}, "guided_regex": "seven",
        }))
        .unwrap();
        assert!(request.validate().is_err());
        assert!(request.extract_sampling_options().is_err());
    }

    /// New modifiers survive the canonical wire while defaults preserve the legacy shape.
    #[test]
    fn modifiers_are_optional_wire_additions() {
        let nested: StructuredOutputs = serde_json::from_value(json!({
            "json": {}, "disable_any_whitespace": true, "disable_additional_properties": true,
            "whitespace_pattern": "",
        }))
        .unwrap();
        let wire = serde_json::to_value(nested.to_guided(None).unwrap()).unwrap();
        assert_eq!(wire["disable_any_whitespace"], true);
        assert_eq!(wire["disable_additional_properties"], true);
        assert_eq!(wire["whitespace_pattern"], "");
        let nested: StructuredOutputs = serde_json::from_value(json!({
            "json": {}, "disable_any_whitespace": false, "disable_additional_properties": false,
        }))
        .unwrap();
        let wire = serde_json::to_value(nested.to_guided(None).unwrap()).unwrap();
        assert_eq!(wire["disable_any_whitespace"], false);
        assert_eq!(wire["disable_additional_properties"], false);
        let legacy = serde_json::to_value(GuidedDecodingOptions::default()).unwrap();
        assert!(legacy.get("disable_any_whitespace").is_none());
        assert!(legacy.get("disable_additional_properties").is_none());
    }
}
