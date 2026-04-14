// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Re-exports upstream async-openai completion types and defines
// inference-serving extensions.

use std::collections::HashMap;
use std::pin::Pin;

use derive_builder::Builder;
use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::error::OpenAIError;

use super::{ChatCompletionStreamOptions, Choice, CompletionUsage, Prompt, Stop};

// Re-export response type from upstream (identical)
pub use async_openai::types::completions::CreateCompletionResponse;

/// Custom deserializer for the echo parameter that only accepts booleans.
/// Rejects integers and strings with clear error messages.
fn deserialize_echo_bool<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct StrictBoolVisitor;

    impl<'de> serde::de::Visitor<'de> for StrictBoolVisitor {
        type Value = Option<bool>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("echo parameter to be a boolean (true or false) or null")
        }

        fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            deserializer.deserialize_any(BoolOnlyVisitor)
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }
    }

    struct BoolOnlyVisitor;

    impl<'de> serde::de::Visitor<'de> for BoolOnlyVisitor {
        type Value = Option<bool>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("echo parameter to be a boolean (true or false) or null")
        }

        fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(Some(value))
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Err(E::invalid_type(
                serde::de::Unexpected::Str(value),
                &"echo parameter to be a boolean (true or false) or null",
            ))
        }
    }

    deserializer.deserialize_option(StrictBoolVisitor)
}

/// Completion request with inference-serving extensions.
///
/// Extends upstream `CreateCompletionRequest` with:
/// - `prompt_embeds`: base64-encoded PyTorch tensor for pre-computed embeddings
/// - `echo`: strict bool validation (rejects integers/strings)
/// - `stream_options`: uses our extended `ChatCompletionStreamOptions` (with `continuous_usage_stats`)
#[derive(Clone, Serialize, Deserialize, Default, Debug, Builder, PartialEq)]
#[builder(name = "CreateCompletionRequestArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct CreateCompletionRequest {
    pub model: String,
    pub prompt: Prompt,
    /// Base64-encoded PyTorch tensor containing pre-computed embeddings.
    /// At least one of prompt or prompt_embeds is required.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_embeds: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u8>,
    /// Echo back the prompt in addition to the completion.
    /// Strict bool validation -- rejects integers and strings.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default, deserialize_with = "deserialize_echo_bool")]
    pub echo: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Stop>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

/// Parsed server side events stream until an \[DONE\] is received from server.
pub type CompletionResponseStream =
    Pin<Box<dyn Stream<Item = Result<CreateCompletionResponse, OpenAIError>> + Send>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn echo_rejects_integer() {
        let json = r#"{"model": "test_model", "prompt": "test", "echo": 1}"#;
        let result: Result<CreateCompletionRequest, _> = serde_json::from_str(json);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("invalid type"));
        assert!(err_msg.contains("integer"));
        assert!(err_msg.contains("echo parameter"));
    }

    #[test]
    fn echo_rejects_string() {
        let json = r#"{"model": "test_model", "prompt": "test", "echo": "null"}"#;
        let result: Result<CreateCompletionRequest, _> = serde_json::from_str(json);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("invalid type"));
        assert!(err_msg.contains("string"));
        assert!(err_msg.contains("echo parameter"));
    }
}
