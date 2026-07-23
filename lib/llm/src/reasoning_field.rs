// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-deployment selector for the reasoning field name emitted by the OpenAI
//! chat-completions HTTP API. Dynamo keeps `reasoning_content` as its internal
//! canonical field; this module changes only the serialized response boundary.

use std::sync::OnceLock;

use serde::{Serialize, Serializer, ser::Error as _};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasoningField {
    Reasoning,
    ReasoningContent,
}

fn reasoning_field_selection() -> ReasoningField {
    static FIELD: OnceLock<ReasoningField> = OnceLock::new();
    *FIELD.get_or_init(|| match std::env::var("DYN_REASONING_FIELD_NAME").as_deref() {
        Ok("reasoning") => ReasoningField::Reasoning,
        Ok("reasoning_content") | Err(_) => ReasoningField::ReasoningContent,
        Ok(other) => {
            tracing::warn!(
                "DYN_REASONING_FIELD_NAME={other:?} is not \"reasoning\" or \"reasoning_content\"; defaulting to reasoning_content"
            );
            ReasoningField::ReasoningContent
        }
    })
}

/// Serialization wrapper that routes chat-completion reasoning to the field
/// selected by `DYN_REASONING_FIELD_NAME`.
///
/// Keeping routing at the HTTP boundary is intentional: reasoning parsers,
/// tool-call parsers, aggregators, request tracing, and gRPC continue to use
/// Dynamo's canonical `reasoning_content` representation. That avoids
/// teaching every internal producer about a wire-format compatibility option.
#[derive(Debug, Clone)]
pub struct RoutedReasoning<T>(T);

impl<T> RoutedReasoning<T> {
    pub fn new(inner: T) -> Self {
        Self(inner)
    }
}

impl<T: Serialize> Serialize for RoutedReasoning<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut value = serde_json::to_value(&self.0).map_err(S::Error::custom)?;
        route_serialized_reasoning(&mut value, reasoning_field_selection());
        value.serialize(serializer)
    }
}

fn route_serialized_reasoning(value: &mut Value, field: ReasoningField) {
    if field == ReasoningField::ReasoningContent {
        return;
    }

    let Some(choices) = value.get_mut("choices").and_then(Value::as_array_mut) else {
        return;
    };

    for choice in choices {
        let Some(choice) = choice.as_object_mut() else {
            continue;
        };

        // Unary chat completions use `message`; streamed chunks use `delta`.
        for container_name in ["message", "delta"] {
            let Some(container) = choice
                .get_mut(container_name)
                .and_then(Value::as_object_mut)
            else {
                continue;
            };

            if let Some(reasoning) = container.remove("reasoning_content") {
                container.insert("reasoning".to_string(), reasoning);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn routes_unary_and_streaming_reasoning() {
        let mut unary = json!({
            "choices": [{
                "message": {"content": null, "reasoning_content": "thinking"}
            }]
        });
        route_serialized_reasoning(&mut unary, ReasoningField::Reasoning);
        assert_eq!(unary["choices"][0]["message"]["reasoning"], "thinking");
        assert!(
            unary["choices"][0]["message"]
                .get("reasoning_content")
                .is_none()
        );

        let mut stream = json!({
            "choices": [{
                "delta": {"reasoning_content": "think"}
            }]
        });
        route_serialized_reasoning(&mut stream, ReasoningField::Reasoning);
        assert_eq!(stream["choices"][0]["delta"]["reasoning"], "think");
        assert!(
            stream["choices"][0]["delta"]
                .get("reasoning_content")
                .is_none()
        );
    }

    #[test]
    fn reasoning_content_mode_preserves_upstream_json() {
        let mut value = json!({
            "choices": [{
                "message": {"reasoning_content": "thinking"}
            }],
            "nvext": {"reasoning_content": "unrelated"}
        });
        let original = value.clone();
        route_serialized_reasoning(&mut value, ReasoningField::ReasoningContent);
        assert_eq!(value, original);
    }

    #[test]
    fn routing_does_not_rewrite_extension_fields_or_add_missing_fields() {
        let mut value = json!({
            "choices": [{"delta": {"content": "answer"}}],
            "nvext": {"reasoning_content": "extension metadata"}
        });
        route_serialized_reasoning(&mut value, ReasoningField::Reasoning);
        assert!(value["choices"][0]["delta"].get("reasoning").is_none());
        assert_eq!(value["nvext"]["reasoning_content"], "extension metadata");
    }
}
