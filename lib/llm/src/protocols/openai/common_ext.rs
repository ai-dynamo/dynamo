// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

use crate::protocols::common::GuidedDecodingOptions;

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
    pub guided_whitespace_pattern: Option<String>,

    /// vLLM word exclusions, tokenized and enforced by the engine.
    /// Omission adds no restriction; an empty list explicitly clears exclusions.
    /// Null and empty string entries are invalid. Requires a vLLM token-input
    /// worker and cannot combine
    /// nonempty words with Dynamo's nonempty `bad_words_token_ids` extension.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_bad_words"
    )]
    #[schema(value_type = Vec<String>, required = false)]
    #[builder(default, setter(strip_option))]
    pub bad_words: Option<Vec<String>>,

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
    /// Defaults to true when omitted, matching vLLM 0.27.1 and the Python
    /// frontend (HuggingFace Transformers defaults this flag to false).
    /// Incompatible with `continue_final_message`. Chat-only: runtime-rejected
    /// on `/v1/completions`. Hidden from the shared OpenAPI schema so
    /// completions does not advertise these fields.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(ignore)]
    #[builder(default, setter(strip_option))]
    pub add_generation_prompt: Option<bool>,

    /// If true, leave the last message open so the model continues that turn
    /// instead of starting a new one. Any final message role can be continued.
    /// Incompatible with omitted or `true` `add_generation_prompt` (vLLM 0.27.1
    /// finalizes the omitted field to true). Chat-only: runtime-rejected on
    /// `/v1/completions`. Hidden from the shared OpenAPI schema so completions
    /// does not advertise these fields.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(ignore)]
    #[builder(default, setter(strip_option))]
    pub continue_final_message: Option<bool>,
}

/// Schema-only view of common extensions for the chat endpoint.
///
/// The runtime type remains `CommonExt`. Its shared schema omits these two
/// fields because completions rejects them; the chat request must expose them.
#[derive(ToSchema)]
#[allow(dead_code)]
pub(crate) struct ChatCommonExt {
    #[serde(flatten)]
    common: CommonExt,
    /// Append the assistant generation prompt after the last message.
    /// Defaults to true when omitted or null. Incompatible with
    /// `continue_final_message=true`; set this explicitly to false to continue.
    #[schema(default = true)]
    add_generation_prompt: Option<bool>,
    /// Continue the final message rather than starting a new assistant turn.
    /// Any final message role can be continued. Requires an explicit
    /// `add_generation_prompt=false`; omission or null defaults that flag to true.
    continue_final_message: Option<bool>,
}

pub(crate) fn extract_guided_decoding_options(
    request: &impl CommonExtProvider,
) -> anyhow::Result<Option<GuidedDecodingOptions>> {
    let guided_whitespace_pattern = request.get_guided_whitespace_pattern();
    GuidedDecodingOptions::from_optional(
        request.get_guided_json(),
        request.get_guided_regex(),
        request.get_guided_choice(),
        request.get_guided_grammar(),
        request.get_guided_decoding_backend(),
        guided_whitespace_pattern.clone(),
        None,
    )
}

impl CommonExt {
    pub fn builder() -> CommonExtBuilder {
        CommonExtBuilder::default()
    }
}

/// Preserve a supplied list, including empty, but reject null and non-string items.
/// Serde's field default handles omission without calling this deserializer.
fn deserialize_bad_words<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Vec::<String>::deserialize(deserializer).map(Some)
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

    /// Whether to continue the last message instead of starting a new turn.
    fn get_continue_final_message(&self) -> Option<bool> {
        self.common_ext()
            .and_then(|common| common.continue_final_message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json;

    /// API omission and empty lists differ; null and malformed list items never
    /// become an accepted unconstrained request. Both endpoints share this type.
    #[test]
    fn bad_words_preserves_presence_and_rejects_null() {
        let omitted: CommonExt = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(omitted.bad_words.is_none());
        assert!(
            serde_json::to_value(omitted)
                .unwrap()
                .get("bad_words")
                .is_none()
        );
        for words in [
            serde_json::json!([]),
            serde_json::json!(["hello", " hello", "hello"]),
        ] {
            let parsed: CommonExt =
                serde_json::from_value(serde_json::json!({"bad_words": words})).unwrap();
            assert_eq!(serde_json::to_value(parsed).unwrap()["bad_words"], words);
        }
        for invalid in [
            serde_json::Value::Null,
            serde_json::json!("hello"),
            serde_json::json!([1]),
        ] {
            assert!(
                serde_json::from_value::<CommonExt>(serde_json::json!({"bad_words": invalid}))
                    .is_err()
            );
        }
    }

    /// Request parsing must extract words into the worker-bound sampling object,
    /// not leave a newly admitted field in an ignored extra-body map.
    #[test]
    fn bad_words_reaches_sampling_on_both_endpoints() {
        use crate::protocols::common::SamplingOptionsProvider;
        use crate::protocols::openai::{
            chat_completions::NvCreateChatCompletionRequest, completions::NvCreateCompletionRequest,
        };
        let chat: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test", "messages": [{"role":"user", "content":"hello"}], "bad_words": ["hello"]
        })).unwrap();
        let completion: NvCreateCompletionRequest = serde_json::from_value(serde_json::json!({
            "model":"test", "prompt":"hello", "bad_words": []
        }))
        .unwrap();
        assert_eq!(
            chat.extract_sampling_options().unwrap().bad_words,
            Some(vec!["hello".into()])
        );
        assert_eq!(
            completion.extract_sampling_options().unwrap().bad_words,
            Some(vec![])
        );
        assert!(!chat.unsupported_fields.contains_key("bad_words"));
        assert!(!completion.unsupported_fields.contains_key("bad_words"));
    }

    /// Older/default worker messages omit this opt-in field; newer readers must
    /// still accept them and serialization must not add an absent constraint.
    #[test]
    fn bad_words_wire_is_optional_without_changing_default_messages() {
        use crate::protocols::common::SamplingOptions;
        let legacy: SamplingOptions = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(legacy.bad_words.is_none());
        assert!(
            serde_json::to_value(legacy)
                .unwrap()
                .get("bad_words")
                .is_none()
        );
        let enabled = SamplingOptions {
            bad_words: Some(vec![]),
            ..Default::default()
        };
        assert_eq!(
            serde_json::to_value(enabled).unwrap()["bad_words"],
            serde_json::json!([])
        );
    }

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
            bad_words: None,
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
