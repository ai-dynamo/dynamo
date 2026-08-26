// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

/// Common extensions for OpenAI API requests that are not part of the standard OpenAI spec
/// but are commonly needed across different request types.
#[derive(ToSchema, Serialize, Deserialize, Builder, Validate, Debug, Clone, Default)]
pub struct CommonExt {
    /// If true, the model will ignore the end of string token and generate to max_tokens.
    /// This field can also be specified in nvext, but the root-level value takes precedence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub ignore_eos: Option<bool>,

    /// The minimum number of tokens to generate.
    /// This is a common parameter needed across different request types.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub min_tokens: Option<u32>,

    /// Integer that controls the number of top tokens to consider. Set to -1 or 0 to consider all
    /// tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub top_k: Option<i32>,

    /// Relative probability floor
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub min_p: Option<f32>,

    /// How much to penalize tokens based on how frequently they occur in the text.
    /// A value of 1 means no penalty, while values larger than 1 discourage and values smaller encourage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub repetition_penalty: Option<f32>,

    /// include_stop_str_in_output
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub include_stop_str_in_output: Option<bool>,

    /// Guided Decoding Options
    /// If specified, the output will be a JSON object. Can be a string, an object, or null.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub guided_json: Option<serde_json::Value>,

    /// If specified, the output will follow the regex pattern. Can be a string or null.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub guided_regex: Option<String>,

    /// If specified, the output will follow the context-free grammar. Can be a string or null.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub guided_grammar: Option<String>,

    /// If specified, the output will be exactly one of the choices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub guided_choice: Option<Vec<String>>,

    /// If specified, the backend to use for guided decoding, can be backends like xgrammar or custom guided decoding backend
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub guided_decoding_backend: Option<String>,

    /// If specified, the output will follow the whitespace pattern. Can be a string or null.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    #[allow(unused)] // Not used
    pub guided_whitespace_pattern: Option<String>,

    /// Whether to skip special tokens in the decoded output.
    /// When true, special tokens (like EOS, BOS, PAD) are removed from the output text.
    /// When false, special tokens are included in the output text.
    /// Defaults to false if not specified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub skip_special_tokens: Option<bool>,

    /// Number of log probabilities to return per prompt token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub prompt_logprobs: Option<u32>,

    /// If true, append the assistant generation prompt after the last message.
    /// Defaults to true when omitted, matching vLLM / HuggingFace `apply_chat_template`.
    /// Incompatible with `continue_final_message`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub add_generation_prompt: Option<bool>,

    /// If true, leave the last assistant message open so the model continues it
    /// instead of starting a new turn. Requires the last message to be assistant.
    /// Incompatible with `add_generation_prompt: true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub continue_final_message: Option<bool>,
}

impl CommonExt {
    pub fn builder() -> CommonExtBuilder {
        CommonExtBuilder::default()
    }
}

/// Trait for types that provide CommonExt fields
pub trait CommonExtProvider {
    /// Get a reference to the CommonExt struct if available
    fn common_ext(&self) -> Option<&CommonExt>;

    /// Guided Decoding Options
    fn get_guided_json(&self) -> Option<serde_json::Value>;
    fn get_guided_regex(&self) -> Option<String>;
    fn get_guided_grammar(&self) -> Option<String>;
    fn get_guided_choice(&self) -> Option<Vec<String>>;
    fn get_guided_decoding_backend(&self) -> Option<String>;
    #[allow(unused)] // Not used
    fn get_guided_whitespace_pattern(&self) -> Option<String>;

    /// Other sampling Options
    fn get_top_k(&self) -> Option<i32>;
    fn get_min_p(&self) -> Option<f32>;
    fn get_repetition_penalty(&self) -> Option<f32>;
    fn get_include_stop_str_in_output(&self) -> Option<bool>;

    /// Output Options
    fn get_skip_special_tokens(&self) -> Option<bool>;

    /// Number of prompt logprobs to request from the engine.
    fn get_prompt_logprobs_count(&self) -> Option<u32> {
        None
    }

    /// Whether to append the assistant generation prompt. Unset means the vLLM default (`true`).
    fn get_add_generation_prompt(&self) -> Option<bool> {
        self.common_ext()
            .and_then(|common| common.add_generation_prompt)
    }

    /// Whether to continue the last assistant message instead of starting a new turn.
    fn get_continue_final_message(&self) -> Option<bool> {
        self.common_ext()
            .and_then(|common| common.continue_final_message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json;

    #[test]
    fn test_validation_min_tokens() {
        // Test that min_tokens with 0 is valid
        let common_ext = CommonExt {
            ignore_eos: None,
            min_tokens: Some(0), // Should be valid (min = 0)
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            include_stop_str_in_output: None,
            guided_json: None,
            guided_regex: None,
            guided_grammar: None,
            guided_choice: None,
            guided_decoding_backend: None,
            guided_whitespace_pattern: None,
            skip_special_tokens: None,
            prompt_logprobs: None,
            add_generation_prompt: None,
            continue_final_message: None,
        };
        assert!(common_ext.validate().is_ok());
    }

    #[test]
    fn test_skip_special_tokens_serialization() {
        // Test that skip_special_tokens can be serialized and deserialized
        let common_ext = CommonExt::builder()
            .skip_special_tokens(true)
            .build()
            .unwrap();

        let json = serde_json::to_string(&common_ext).unwrap();
        let deserialized: CommonExt = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.skip_special_tokens, Some(true));

        // Test with false value
        let common_ext = CommonExt::builder()
            .skip_special_tokens(false)
            .build()
            .unwrap();

        let json = serde_json::to_string(&common_ext).unwrap();
        let deserialized: CommonExt = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.skip_special_tokens, Some(false));

        // Test that None is not serialized (skip_serializing_if = "Option::is_none")
        let common_ext = CommonExt::builder().build().unwrap();
        let json = serde_json::to_string(&common_ext).unwrap();
        assert!(!json.contains("skip_special_tokens"));
    }
}
