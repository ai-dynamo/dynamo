// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use validator::Validate;

/// Common extensions for OpenAI API requests that are not part of the standard OpenAI spec
/// but are commonly needed across different request types.
#[derive(Serialize, Deserialize, Builder, Validate, Debug, Clone, Default)]
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

    /// Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json;

    #[test]
    fn test_common_ext_builder_default() {
        let common_ext = CommonExt::builder().build().unwrap();
        assert_eq!(common_ext.ignore_eos, None);
        assert_eq!(common_ext.min_tokens, None);
        assert_eq!(common_ext.top_k, None);
        assert_eq!(common_ext.repetition_penalty, None);
        assert_eq!(common_ext.guided_json, None);
        assert_eq!(common_ext.guided_regex, None);
        assert_eq!(common_ext.guided_grammar, None);
        assert_eq!(common_ext.guided_choice, None);
        assert_eq!(common_ext.guided_decoding_backend, None);
        assert_eq!(common_ext.include_stop_str_in_output, None);
        assert_eq!(common_ext.skip_special_tokens, None);
    }

    #[test]
    fn test_common_ext_builder_with_values() {
        let common_ext = CommonExt::builder()
            .ignore_eos(true)
            .min_tokens(10)
            .top_k(50)
            .repetition_penalty(1.2)
            .include_stop_str_in_output(true)
            .guided_json(serde_json::json!({"key": "value"}))
            .guided_regex("regex".to_string())
            .guided_grammar("grammar".to_string())
            .guided_choice(vec!["choice1".to_string(), "choice2".to_string()])
            .guided_decoding_backend("backend".to_string())
            .skip_special_tokens(false)
            .build()
            .unwrap();

        assert_eq!(common_ext.ignore_eos, Some(true));
        assert_eq!(common_ext.min_tokens, Some(10));
        assert_eq!(common_ext.top_k, Some(50));
        assert_eq!(common_ext.repetition_penalty, Some(1.2));
        assert_eq!(common_ext.include_stop_str_in_output, Some(true));
        assert_eq!(
            common_ext.guided_json.as_ref(),
            Some(&serde_json::json!({"key": "value"}))
        );
        assert_eq!(common_ext.guided_regex, Some("regex".to_string()));
        assert_eq!(common_ext.guided_grammar, Some("grammar".to_string()));
        assert_eq!(
            common_ext.guided_choice,
            Some(vec!["choice1".to_string(), "choice2".to_string()])
        );
        assert_eq!(
            common_ext.guided_decoding_backend,
            Some("backend".to_string())
        );
        assert_eq!(common_ext.skip_special_tokens, Some(false));
    }

    #[test]
    fn test_common_ext_fields() {
        // Test that CommonExt fields can be set and retrieved correctly
        let common_ext = CommonExt::builder()
            .ignore_eos(false)
            .min_tokens(5)
            .include_stop_str_in_output(true)
            .build()
            .unwrap();

        assert_eq!(common_ext.ignore_eos, Some(false));
        assert_eq!(common_ext.min_tokens, Some(5));
        assert_eq!(common_ext.include_stop_str_in_output, Some(true));
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
            skip_special_tokens: None,
        };
        assert!(common_ext.validate().is_ok());
    }

    #[test]
    fn test_common_ext_neither_specified() {
        // Test that neither ignore_eos nor min_tokens specified works
        let common_ext = CommonExt::builder().build().unwrap();

        assert_eq!(common_ext.ignore_eos, None);
        assert_eq!(common_ext.min_tokens, None);
        assert_eq!(common_ext.top_k, None);
        assert_eq!(common_ext.repetition_penalty, None);
        assert_eq!(common_ext.include_stop_str_in_output, None);
        assert!(common_ext.validate().is_ok());
    }

    #[test]
    fn test_common_ext_default() {
        // Test that Default trait implementation works correctly
        let common_ext = CommonExt::default();

        assert_eq!(common_ext.ignore_eos, None);
        assert_eq!(common_ext.min_tokens, None);
        assert_eq!(common_ext.top_k, None);
        assert_eq!(common_ext.repetition_penalty, None);
        assert_eq!(common_ext.include_stop_str_in_output, None);
        assert!(common_ext.validate().is_ok());
    }

    #[test]
    fn test_skip_special_tokens_field() {
        // Test that skip_special_tokens can be set and retrieved
        let common_ext = CommonExt::builder()
            .skip_special_tokens(true)
            .build()
            .unwrap();

        assert_eq!(common_ext.skip_special_tokens, Some(true));

        let common_ext = CommonExt::builder()
            .skip_special_tokens(false)
            .build()
            .unwrap();

        assert_eq!(common_ext.skip_special_tokens, Some(false));
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
