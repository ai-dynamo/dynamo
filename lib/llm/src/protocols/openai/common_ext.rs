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
    fn get_guided_json(&self) -> Option<&serde_json::Value>;
    fn get_guided_regex(&self) -> Option<String>;
    fn get_guided_grammar(&self) -> Option<String>;
    fn get_guided_choice(&self) -> Option<Vec<String>>;
    fn get_guided_decoding_backend(&self) -> Option<String>;
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
        assert_eq!(common_ext.guided_json, None);
        assert_eq!(common_ext.guided_regex, None);
        assert_eq!(common_ext.guided_grammar, None);
        assert_eq!(common_ext.guided_choice, None);
        assert_eq!(common_ext.guided_decoding_backend, None);
    }

    #[test]
    fn test_common_ext_builder_with_values() {
        let common_ext = CommonExt::builder()
            .ignore_eos(true)
            .min_tokens(10)
            .guided_json(serde_json::json!({"key": "value"}))
            .guided_regex("regex".to_string())
            .guided_grammar("grammar".to_string())
            .guided_choice(vec!["choice1".to_string(), "choice2".to_string()])
            .guided_decoding_backend("backend".to_string())
            .build()
            .unwrap();

        assert_eq!(common_ext.ignore_eos, Some(true));
        assert_eq!(common_ext.min_tokens, Some(10));
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
    }

    #[test]
    fn test_common_ext_fields() {
        // Test that CommonExt fields can be set and retrieved correctly
        let common_ext = CommonExt::builder()
            .ignore_eos(false)
            .min_tokens(5)
            .build()
            .unwrap();

        assert_eq!(common_ext.ignore_eos, Some(false));
        assert_eq!(common_ext.min_tokens, Some(5));
    }

    #[test]
    fn test_validation_min_tokens() {
        // Test that min_tokens with 0 is valid
        let common_ext = CommonExt {
            ignore_eos: None,
            min_tokens: Some(0), // Should be valid (min = 0)
            guided_json: None,
            guided_regex: None,
            guided_grammar: None,
            guided_choice: None,
            guided_decoding_backend: None,
        };
        assert!(common_ext.validate().is_ok());
    }

    #[test]
    fn test_common_ext_neither_specified() {
        // Test that neither ignore_eos nor min_tokens specified works
        let common_ext = CommonExt::builder().build().unwrap();

        assert_eq!(common_ext.ignore_eos, None);
        assert_eq!(common_ext.min_tokens, None);
        assert!(common_ext.validate().is_ok());
    }

    #[test]
    fn test_common_ext_default() {
        // Test that Default trait implementation works correctly
        let common_ext = CommonExt::default();

        assert_eq!(common_ext.ignore_eos, None);
        assert_eq!(common_ext.min_tokens, None);
        assert!(common_ext.validate().is_ok());
    }
}
