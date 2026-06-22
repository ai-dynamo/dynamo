// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

use crate::engines::ValidateRequest;
use crate::preprocessor::media::MediaDecoder;

use super::{
    OpenAIOutputOptionsProvider, OpenAISamplingOptionsProvider, OpenAIStopConditionsProvider,
    common_ext::{CommonExt, CommonExtProvider},
    validate,
};
use crate::protocols::common::extensions::{
    NvExt, NvExtProvider, validate_completion_token_ids_single_choice, validate_nvext_semantics,
};

pub mod aggregator;
mod delta;
pub mod jail;

pub use aggregator::DeltaAggregator;
pub use delta::DeltaGenerator;

use dynamo_parsers::tool_calling::{ToolCallResponse, ToolCallResponseChunk};
use dynamo_protocols::types::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallChunk, FunctionCall,
    FunctionCallStream, FunctionType,
};

/// Map a parser-native [`ToolCallResponse`] onto the protocol/wire
/// [`ChatCompletionMessageToolCall`].
///
/// `dynamo-parsers` is decoupled from `dynamo-protocols`, so this consumer —
/// which already depends on both — owns the mapping between the parser-native
/// types and the OpenAI wire types. The field shapes are identical, so this is
/// a straight re-map that preserves the previous wire output.
pub(crate) fn tool_call_response_to_protocol(
    parsed: ToolCallResponse,
) -> ChatCompletionMessageToolCall {
    ChatCompletionMessageToolCall {
        id: parsed.id,
        r#type: FunctionType::Function,
        function: FunctionCall {
            name: parsed.function.name,
            arguments: parsed.function.arguments,
        },
    }
}

/// Map a parser-native [`ToolCallResponseChunk`] onto the protocol/wire
/// [`ChatCompletionMessageToolCallChunk`]. See
/// [`tool_call_response_to_protocol`] for the rationale.
///
/// Exposed so consumers of the decoupled streaming parser entrypoint
/// ([`dynamo_parsers::tool_calling::try_tool_call_parse_stream`]) can recover
/// the wire type without `dynamo-parsers` depending on `dynamo-protocols`.
#[allow(dead_code)]
pub(crate) fn tool_call_response_chunk_to_protocol(
    parsed: ToolCallResponseChunk,
) -> ChatCompletionMessageToolCallChunk {
    ChatCompletionMessageToolCallChunk {
        index: parsed.index,
        id: parsed.id,
        r#type: parsed.tp.map(|_| FunctionType::Function),
        function: parsed.function.map(|f| FunctionCallStream {
            name: f.name,
            arguments: f.arguments,
        }),
    }
}

/// A request structure for creating a chat completion, extending OpenAI's
/// `CreateChatCompletionRequest` with [`NvExt`] extensions and common fields.
///
/// # Fields
/// - `inner`: The base OpenAI chat completion request, embedded using `serde(flatten)`.
/// - `common`: Common extension fields (ignore_eos, min_tokens) at root level, embedded using `serde(flatten)`.
/// - `nvext`: The optional NVIDIA extension field. See [`NvExt`] for more details.
///   Note: If ignore_eos is specified in both common and nvext, the common (root-level) value takes precedence.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateChatCompletionRequest {
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub inner: dynamo_protocols::types::CreateChatCompletionRequest,

    #[serde(flatten, default)]
    pub common: CommonExt,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Object)]
    pub nvext: Option<NvExt>,

    /// Extra args to pass to the chat template rendering context
    /// Also accepts "chat_template_kwargs" as an alias for compatibility
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "chat_template_kwargs"
    )]
    pub chat_template_args: Option<std::collections::HashMap<String, serde_json::Value>>,

    /// OpenAI-style thinking control from client request payloads.
    /// Normalized into `chat_template_args.thinking` before preprocessing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<serde_json::Value>,

    /// Runtime media decoding parameters.
    /// When provided, these override the MDC defaults
    /// Example: `{"video": {"num_frames": 16}}`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_io_kwargs: Option<MediaDecoder>,

    /// When true, logprob token fields are returned as "token_id:<id>" instead
    /// of decoded text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub return_tokens_as_token_ids: Option<bool>,

    /// Catch-all for unsupported fields - checked during validation
    #[serde(flatten, default, skip_serializing)]
    pub unsupported_fields: std::collections::HashMap<String, serde_json::Value>,
}

impl NvCreateChatCompletionRequest {
    /// Normalize OpenAI-style DS-V4 reasoning controls into the template kwargs
    /// consumed by the SGLang/DeepSeek-V4 prompt formatter.
    pub fn normalize_reasoning_template_args(&mut self) -> anyhow::Result<()> {
        let thinking_enabled = self
            .thinking
            .as_ref()
            .map(openai_thinking_enabled)
            .transpose()?
            .flatten();
        let reasoning_effort = self
            .inner
            .reasoning_effort
            .as_ref()
            .and_then(|effort| serde_json::to_value(effort).ok());

        if thinking_enabled.is_none() && reasoning_effort.is_none() {
            return Ok(());
        }

        let args = self.chat_template_args.get_or_insert_with(HashMap::new);
        if let Some(enabled) = thinking_enabled {
            args.entry("thinking".to_string())
                .or_insert(serde_json::Value::Bool(enabled));
        }
        if let Some(effort) = reasoning_effort {
            args.entry("reasoning_effort".to_string()).or_insert(effort);
        }

        // The raw `thinking` payload has been folded into `chat_template_args`;
        // drop it so it isn't double-shipped downstream (and so it can't be
        // re-interpreted with different precedence by the worker preprocessor).
        self.thinking = None;
        Ok(())
    }
}

fn openai_thinking_enabled(value: &serde_json::Value) -> anyhow::Result<Option<bool>> {
    if let Some(enabled) = value.as_bool() {
        return Ok(Some(enabled));
    }

    let Some(thinking_object) = value.as_object() else {
        anyhow::bail!(
            "`thinking` must be a boolean or an object with `type` set to `enabled` or `disabled`"
        );
    };
    let Some(thinking_type) = thinking_object.get("type").and_then(|v| v.as_str()) else {
        anyhow::bail!("`thinking.type` must be `enabled` or `disabled`");
    };
    match thinking_type {
        "enabled" => Ok(Some(true)),
        "disabled" => Ok(Some(false)),
        _ => anyhow::bail!("`thinking.type` must be `enabled` or `disabled`"),
    }
}

/// A response structure for unary chat completion responses, embedding OpenAI's
/// `CreateChatCompletionResponse` with optional NVIDIA extension metadata.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NvCreateChatCompletionResponse {
    #[serde(flatten)]
    pub inner: dynamo_protocols::types::CreateChatCompletionResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<serde_json::Value>,
}

/// A response structure for streamed chat completions, embedding OpenAI's
/// `CreateChatCompletionStreamResponse` with optional NVIDIA extension metadata.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NvCreateChatCompletionStreamResponse {
    #[serde(flatten)]
    pub inner: dynamo_protocols::types::CreateChatCompletionStreamResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<serde_json::Value>,
}

/// Implements `NvExtProvider` for `NvCreateChatCompletionRequest`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateChatCompletionRequest {
    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    /// Returns `None`, as raw prompt extraction is not implemented.
    fn raw_prompt(&self) -> Option<String> {
        None
    }

    fn unsupported_fields(&self) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
        Some(&self.unsupported_fields)
    }
}

/// Implements `AnnotationsProvider` for `NvCreateChatCompletionRequest`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the list of annotations from `NvExt`, if present.
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    /// Checks whether a specific annotation exists in the request.
    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}

/// Implements `OpenAISamplingOptionsProvider` for `NvCreateChatCompletionRequest`,
/// exposing OpenAI's sampling parameters for chat completion.
impl OpenAISamplingOptionsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the temperature parameter for sampling, if set.
    fn get_temperature(&self) -> Option<f32> {
        self.inner.temperature
    }

    /// Retrieves the top-p (nucleus sampling) parameter, if set.
    fn get_top_p(&self) -> Option<f32> {
        self.inner.top_p
    }

    /// Retrieves the frequency penalty parameter, if set.
    fn get_frequency_penalty(&self) -> Option<f32> {
        self.inner.frequency_penalty
    }

    /// Retrieves the presence penalty parameter, if set.
    fn get_presence_penalty(&self) -> Option<f32> {
        self.inner.presence_penalty
    }

    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
    /// Retrieves the seed value for random number generation, if set.
    fn get_seed(&self) -> Option<i64> {
        self.inner.seed
    }

    /// Retrieves the number of completions to generate for each prompt, if set.
    fn get_n(&self) -> Option<u8> {
        self.inner.n
    }

    /// Retrieves the best_of parameter, if set.
    fn get_best_of(&self) -> Option<u8> {
        None // Not supported in chat completions
    }
}

/// Implements `CommonExtProvider` for `NvCreateChatCompletionRequest`,
/// providing access to common extension fields.
impl CommonExtProvider for NvCreateChatCompletionRequest {
    /// Returns a reference to the CommonExt struct.
    fn common_ext(&self) -> Option<&CommonExt> {
        Some(&self.common)
    }

    /// Guided Decoding Options
    fn get_guided_json(&self) -> Option<serde_json::Value> {
        if let Some(value) = self.common.guided_json.clone() {
            return Some(value);
        }

        if let Some(response_format) = self.inner.response_format.as_ref() {
            use dynamo_protocols::types::ResponseFormat;
            match response_format {
                ResponseFormat::Text => {}
                ResponseFormat::JsonObject => {
                    // Minimal JSON Schema for "any JSON object"
                    return Some(serde_json::json!({
                        "type": "object"
                    }));
                }
                ResponseFormat::JsonSchema { json_schema } => {
                    // validate_response_format ensures schema is present when type=json_schema
                    if let Some(schema) = json_schema.schema.clone() {
                        return Some(schema);
                    }
                }
            }
        }

        None
    }

    fn get_guided_regex(&self) -> Option<String> {
        self.common.guided_regex.clone()
    }

    fn get_guided_grammar(&self) -> Option<String> {
        self.common.guided_grammar.clone()
    }

    fn get_guided_choice(&self) -> Option<Vec<String>> {
        self.common.guided_choice.clone()
    }

    fn get_guided_decoding_backend(&self) -> Option<String> {
        self.common.guided_decoding_backend.clone()
    }

    fn get_guided_whitespace_pattern(&self) -> Option<String> {
        self.common.guided_whitespace_pattern.clone()
    }

    fn get_top_k(&self) -> Option<i32> {
        self.common.top_k
    }

    fn get_min_p(&self) -> Option<f32> {
        self.common.min_p
    }

    fn get_repetition_penalty(&self) -> Option<f32> {
        self.common.repetition_penalty
    }

    fn get_include_stop_str_in_output(&self) -> Option<bool> {
        self.common.include_stop_str_in_output
    }

    fn get_skip_special_tokens(&self) -> Option<bool> {
        self.common.skip_special_tokens
    }

    fn get_prompt_logprobs_count(&self) -> Option<u32> {
        self.common.prompt_logprobs
    }
}

/// Implements `OpenAIStopConditionsProvider` for `NvCreateChatCompletionRequest`,
/// providing access to stop conditions that control chat completion behavior.
impl OpenAIStopConditionsProvider for NvCreateChatCompletionRequest {
    /// Retrieves the maximum number of tokens allowed in the response.
    #[allow(deprecated)]
    fn get_max_tokens(&self) -> Option<u32> {
        self.inner.max_completion_tokens.or(self.inner.max_tokens)
    }

    /// Retrieves the minimum number of tokens required in the response.
    /// Returns `min_tokens` Value
    /// `min_tokens` is not an OpenAI-supported parameter.
    fn get_min_tokens(&self) -> Option<u32> {
        self.common.min_tokens
    }

    /// Retrieves the stop conditions that terminate the chat completion response.
    ///
    /// Converts OpenAI's `Stop` enum to a `Vec<String>`, normalizing the representation.
    ///
    /// # Returns
    /// * `Some(Vec<String>)` if stop conditions are set.
    /// * `None` if no stop conditions are defined.
    fn get_stop(&self) -> Option<Vec<String>> {
        self.inner.stop.as_ref().and_then(|stop| stop.strings())
    }

    fn get_stop_token_ids(&self) -> Option<Vec<crate::types::TokenIdType>> {
        // Token IDs may be provided in the standard OpenAI `stop` array.
        if let Some(ids) = self.inner.stop.as_ref().and_then(|stop| stop.token_ids()) {
            return Some(ids);
        }
        // Also accept top-level `stop_token_ids` from passthrough clients.
        self.unsupported_fields
            .get("stop_token_ids")
            .and_then(|v| serde_json::from_value::<Vec<crate::types::TokenIdType>>(v.clone()).ok())
    }

    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    /// Get ignore_eos from CommonExt.
    fn get_common_ignore_eos(&self) -> Option<bool> {
        self.common.ignore_eos
    }

    /// Get the effective ignore_eos value from CommonExt.
    fn get_ignore_eos(&self) -> Option<bool> {
        self.common.ignore_eos
    }
}

impl OpenAIOutputOptionsProvider for NvCreateChatCompletionRequest {
    fn get_logprobs(&self) -> Option<u32> {
        match self.inner.logprobs {
            Some(true) => match self.inner.top_logprobs {
                Some(top_logprobs) => Some(top_logprobs as u32),
                None => Some(1_u32),
            },
            Some(false) => None,
            None => None,
        }
    }

    fn get_prompt_logprobs(&self) -> Option<u32> {
        // Top-level `prompt_logprobs` is carried through CommonExt.
        self.common.prompt_logprobs
    }

    fn get_skip_special_tokens(&self) -> Option<bool> {
        CommonExtProvider::get_skip_special_tokens(self)
    }

    fn get_formatted_prompt(&self) -> Option<bool> {
        None
    }

    fn get_return_tokens_as_token_ids(&self) -> Option<bool> {
        self.return_tokens_as_token_ids
    }
}

/// Implements `ValidateRequest` for `NvCreateChatCompletionRequest`,
/// allowing us to validate the data.
impl ValidateRequest for NvCreateChatCompletionRequest {
    fn validate(&self) -> Result<(), anyhow::Error> {
        validate::validate_no_unsupported_fields(&self.unsupported_fields)?;
        validate::validate_messages(&self.inner.messages)?;
        validate::validate_model(&self.inner.model)?;
        // none for store
        validate::validate_reasoning_effort(&self.inner.reasoning_effort)?;
        // none for metadata
        validate::validate_frequency_penalty(self.inner.frequency_penalty)?;
        validate::validate_logit_bias(&self.inner.logit_bias)?;
        // none for logprobs
        validate::validate_top_logprobs(self.inner.top_logprobs)?;
        // validate::validate_max_tokens(self.inner.max_tokens)?; // warning depricated field
        validate::validate_max_completion_tokens(self.inner.max_completion_tokens)?;
        validate::validate_n(self.inner.n)?;
        validate_completion_token_ids_single_choice(
            self.inner.n.unwrap_or(1) as usize,
            self.nvext.as_ref(),
        )?;
        validate_nvext_semantics(self.nvext.as_ref())?;
        // none for modalities
        // none for prediction
        // none for audio
        validate::validate_presence_penalty(self.inner.presence_penalty)?;
        validate::validate_response_format(&self.inner.response_format)?;
        // none for seed
        validate::validate_service_tier(&self.inner.service_tier)?;
        validate::validate_stop(&self.inner.stop)?;
        // none for stream
        // none for stream_options
        validate::validate_temperature(self.inner.temperature)?;
        validate::validate_top_p(self.inner.top_p)?;
        validate::validate_tools(&self.inner.tools.as_deref())?;
        validate::validate_tool_choice(&self.inner.tool_choice, self.inner.tools.as_deref())?;
        // none for parallel_tool_calls
        validate::validate_user(self.inner.user.as_deref())?;
        // none for function call
        // none for functions
        // Common Ext
        validate::validate_repetition_penalty(self.get_repetition_penalty())?;
        validate::validate_min_p(self.get_min_p())?;
        validate::validate_top_k(self.get_top_k())?;
        // Cross-field validation
        validate::validate_n_with_temperature(self.inner.n, self.inner.temperature)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::ValidateRequest;
    use crate::protocols::common::{OutputOptionsProvider, StopConditionsProvider};
    use dynamo_protocols::types::{ChatCompletionTool, ChatCompletionToolType, FunctionObject};
    use serde_json::json;

    #[test]
    fn test_skip_special_tokens_none() {
        let json_str = json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        });

        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");

        assert_eq!(request.common.skip_special_tokens, None);

        let output_options = request
            .extract_output_options()
            .expect("Failed to extract output options");

        assert_eq!(output_options.skip_special_tokens, None);
    }

    #[test]
    fn test_skip_special_tokens_propagates() {
        for skip_value in [true, false] {
            let json_str = json!({
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "skip_special_tokens": skip_value
            });

            let request: NvCreateChatCompletionRequest =
                serde_json::from_value(json_str).expect("Failed to deserialize request");

            let output_options = request
                .extract_output_options()
                .expect("Failed to extract output options");

            assert_eq!(output_options.skip_special_tokens, Some(skip_value));
        }
    }

    #[test]
    fn test_stop_contract() {
        let one_stop = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": " The"
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(one_stop).expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), Some(vec![" The".to_string()]));
        assert_eq!(request.get_stop_token_ids(), None);

        let many_stops = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": ["A", "B"]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(many_stops).expect("Failed to deserialize request");
        assert_eq!(
            request.get_stop(),
            Some(vec!["A".to_string(), "B".to_string()])
        );
        assert_eq!(request.get_stop_token_ids(), None);

        let token_id_stops = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": [32, 34]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(token_id_stops).expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), None);
        assert_eq!(request.get_stop_token_ids(), Some(vec![32, 34]));

        let stop_conditions = request
            .extract_stop_conditions()
            .expect("extract stop conditions");
        assert_eq!(stop_conditions.stop, None);
        assert_eq!(stop_conditions.stop_token_ids, Some(vec![32, 34]));

        let token_id_display_string_stop = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": "token_id:576"
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(token_id_display_string_stop)
                .expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), Some(vec!["token_id:576".to_string()]));
        assert_eq!(request.get_stop_token_ids(), None);

        let token_id_display_string_array_stop = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": ["token_id:576"]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(token_id_display_string_array_stop)
                .expect("Failed to deserialize request");
        assert_eq!(request.get_stop(), Some(vec!["token_id:576".to_string()]));
        assert_eq!(request.get_stop_token_ids(), None);

        let scalar_token_id_stop = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": 576
        });
        let result: Result<NvCreateChatCompletionRequest, _> =
            serde_json::from_value(scalar_token_id_stop);
        assert!(result.is_err());

        // `stop_token_ids` is accepted and plumbed by the provider trait.
        let whitelisted_stop_token_ids = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop_token_ids": [576]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(whitelisted_stop_token_ids)
                .expect("Failed to deserialize request");
        assert_eq!(request.get_stop_token_ids(), Some(vec![576]));
        assert!(
            ValidateRequest::validate(&request).is_ok(),
            "stop_token_ids must be accepted via PASSTHROUGH_EXTRA_FIELDS"
        );

        let invalid_stop_token_ids = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop_token_ids": "bad"
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(invalid_stop_token_ids).expect("Failed to deserialize request");
        let err = ValidateRequest::validate(&request).expect_err("invalid stop_token_ids");
        assert!(err.to_string().contains("stop_token_ids"));
    }

    #[test]
    fn test_passthrough_token_constraints_validate() {
        let request_json = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "allowed_token_ids": [10, 11],
            "bad_words_token_ids": [[12, 13]]
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        assert_eq!(
            request.unsupported_fields.get("allowed_token_ids"),
            Some(&serde_json::json!([10, 11]))
        );
        assert_eq!(
            request.unsupported_fields.get("bad_words_token_ids"),
            Some(&serde_json::json!([[12, 13]]))
        );
        assert!(ValidateRequest::validate(&request).is_ok());
    }

    #[test]
    fn test_completion_token_ids_rejected_for_multi_choice() {
        let request_json = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "n": 2,
            "nvext": {
                "extra_fields": ["completion_token_ids"]
            }
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        let err = ValidateRequest::validate(&request).expect_err("multi-choice token ids");
        assert!(err.to_string().contains("completion_token_ids"));
    }

    #[test]
    fn test_validate_tool_choice_required_rejects_empty_tools() {
        let request_json = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "tool_choice": "required"
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        let err = ValidateRequest::validate(&request).expect_err("required needs tools");
        assert!(
            err.to_string()
                .contains("tool_choice is \"required\" but tools is empty")
        );
    }

    #[test]
    fn test_validate_tool_choice_named_rejects_missing_tool() {
        let request_json = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "search"}
            }
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        let err = ValidateRequest::validate(&request).expect_err("named tool must exist");
        assert!(
            err.to_string()
                .contains("tool named \"search\" in tool_choice is not present in tools")
        );
    }

    #[test]
    fn test_truncate_prompt_tokens_rejected_until_supported() {
        let request_json = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "truncate_prompt_tokens": 2
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("Failed to deserialize request");

        assert!(ValidateRequest::validate(&request).is_err());
    }

    // -----------------------------------------------------------------------
    // Parser -> protocol mapping (decoupling guard).
    //
    // `dynamo-parsers` no longer depends on `dynamo-protocols`; the mapping
    // moved into this consumer. These tests pin the mapper output to the
    // *exact* struct + serialized JSON the old protocol-typed parser path
    // produced, proving the wire output is unchanged.
    // -----------------------------------------------------------------------
    use dynamo_parsers::tool_calling::{
        CalledFunction, CalledFunctionStream, ToolCallResponse, ToolCallResponseChunk, ToolCallType,
    };

    fn native_call(id: &str, name: &str, args: &str) -> ToolCallResponse {
        ToolCallResponse {
            id: id.to_string(),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        }
    }

    fn native_chunk(index: u32, id: &str, name: &str, args: &str) -> ToolCallResponseChunk {
        ToolCallResponseChunk {
            index,
            id: Some(id.to_string()),
            tp: Some(ToolCallType::Function),
            function: Some(CalledFunctionStream {
                name: Some(name.to_string()),
                arguments: Some(args.to_string()),
            }),
        }
    }

    /// Reference reconstruction of the pre-decoupling unary mapping that lived
    /// inside `dynamo-parsers`. Kept inline so a divergence in the live mapper
    /// fails the test.
    fn legacy_unary(id: &str, name: &str, args: &str) -> ChatCompletionMessageToolCall {
        ChatCompletionMessageToolCall {
            id: id.to_string(),
            r#type: FunctionType::Function,
            function: FunctionCall {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        }
    }

    /// Reference reconstruction of the pre-decoupling streaming mapping.
    fn legacy_chunk(
        index: u32,
        id: &str,
        name: &str,
        args: &str,
    ) -> ChatCompletionMessageToolCallChunk {
        ChatCompletionMessageToolCallChunk {
            index,
            id: Some(id.to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some(name.to_string()),
                arguments: Some(args.to_string()),
            }),
        }
    }

    #[test]
    fn unary_mapping_matches_legacy_struct_and_json() {
        for (id, name, args) in [
            (
                "call_1",
                "get_weather",
                r#"{"location":"SF","unit":"celsius"}"#,
            ),
            ("call_2", "ping", "{}"), // empty arguments
        ] {
            let mapped = tool_call_response_to_protocol(native_call(id, name, args));
            let legacy = legacy_unary(id, name, args);
            assert_eq!(mapped, legacy, "struct mismatch for {name}");
            assert_eq!(
                serde_json::to_string(&mapped).unwrap(),
                serde_json::to_string(&legacy).unwrap(),
                "serialized JSON mismatch for {name}"
            );
        }
    }

    #[test]
    fn unary_mapping_multi_call_matches_legacy() {
        let inputs = [
            ("a", "first", r#"{"k":"v1"}"#),
            ("b", "second", r#"{"k":"v2"}"#),
        ];
        let mapped: Vec<_> = inputs
            .iter()
            .map(|(id, n, a)| tool_call_response_to_protocol(native_call(id, n, a)))
            .collect();
        let legacy: Vec<_> = inputs
            .iter()
            .map(|(id, n, a)| legacy_unary(id, n, a))
            .collect();
        assert_eq!(mapped, legacy);
        assert_eq!(
            serde_json::to_string(&mapped).unwrap(),
            serde_json::to_string(&legacy).unwrap()
        );
    }

    #[test]
    fn stream_mapping_matches_legacy_struct_and_json() {
        for (idx, id, name, args) in [
            (0u32, "call_1", "get_weather", r#"{"location":"SF"}"#),
            (1u32, "call_2", "ping", "{}"), // empty arguments
        ] {
            let mapped = tool_call_response_chunk_to_protocol(native_chunk(idx, id, name, args));
            let legacy = legacy_chunk(idx, id, name, args);
            assert_eq!(mapped, legacy, "struct mismatch for {name}");
            assert_eq!(
                serde_json::to_string(&mapped).unwrap(),
                serde_json::to_string(&legacy).unwrap(),
                "serialized JSON mismatch for {name}"
            );
        }
    }

    #[test]
    fn stream_mapping_multi_call_indexes_and_matches_legacy() {
        let inputs = [
            (0u32, "a", "first", r#"{"k":"v1"}"#),
            (1u32, "b", "second", r#"{"k":"v2"}"#),
        ];
        let mapped: Vec<_> = inputs
            .iter()
            .map(|(i, id, n, a)| tool_call_response_chunk_to_protocol(native_chunk(*i, id, n, a)))
            .collect();
        let legacy: Vec<_> = inputs
            .iter()
            .map(|(i, id, n, a)| legacy_chunk(*i, id, n, a))
            .collect();
        assert_eq!(mapped, legacy);
        assert_eq!(
            serde_json::to_string(&mapped).unwrap(),
            serde_json::to_string(&legacy).unwrap()
        );
    }

    #[test]
    fn test_validate_tools_valid_names() {
        fn make_tool(name: &str) -> ChatCompletionTool {
            ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name: name.to_string(),
                    description: None,
                    parameters: Some(json!({"type": "object", "properties": {}})),
                    strict: None,
                },
            }
        }

        let tools = vec![
            make_tool("func_name"),
            make_tool("func-name_v2"),
            make_tool("FuncName"),
            make_tool("Func_Name-123"),
        ];
        assert!(validate::validate_tools(&Some(&tools)).is_ok());
    }

    #[test]
    fn test_validate_tools_invalid_names() {
        for name in ["<func_name>", "func name", "func@name", "func,name", ""] {
            let tools = vec![ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name: name.to_string(),
                    description: None,
                    parameters: Some(json!({"type": "object", "properties": {}})),
                    strict: None,
                },
            }];
            assert!(
                validate::validate_tools(&Some(&tools)).is_err(),
                "expected error for name: {name:?}"
            );
        }
    }

    #[test]
    fn test_openai_thinking_payload_normalizes_to_template_args() {
        let json_str = json!({
            "model": "deepseek-ai/DeepSeek-V4-Pro",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "reasoning_effort": "max",
            "thinking": {"type": "enabled"}
        });

        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(json_str).expect("Failed to deserialize request");
        request
            .normalize_reasoning_template_args()
            .expect("thinking payload should normalize");

        let args = request
            .chat_template_args
            .as_ref()
            .expect("chat_template_args should be populated");
        assert_eq!(args.get("thinking"), Some(&json!(true)));
        assert_eq!(args.get("reasoning_effort"), Some(&json!("max")));
    }

    #[test]
    fn test_invalid_openai_thinking_payload_is_rejected() {
        for invalid_thinking in [
            json!("enabled"),
            json!({"type": "auto"}),
            json!({"type": true}),
            json!({}),
        ] {
            let json_str = json!({
                "model": "deepseek-ai/DeepSeek-V4-Pro",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "thinking": invalid_thinking
            });

            let mut request: NvCreateChatCompletionRequest =
                serde_json::from_value(json_str).expect("Failed to deserialize request");
            assert!(request.normalize_reasoning_template_args().is_err());
        }
    }
}
