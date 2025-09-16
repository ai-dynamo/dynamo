// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

pub trait NvExtProvider {
    fn nvext(&self) -> Option<&NvExt>;
    fn raw_prompt(&self) -> Option<String>;
}

/// NVIDIA LLM extensions to the OpenAI API
#[derive(Serialize, Deserialize, Builder, Validate, Debug, Clone)]
#[validate(schema(function = "validate_nv_ext"))]
pub struct NvExt {
    /// If true, the model will ignore the end of string token and generate to max_tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub ignore_eos: Option<bool>,

    #[builder(default, setter(strip_option))] // NIM LLM might default to -1
    #[validate(custom(function = "validate_top_k"))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,

    /// How much to penalize tokens based on how frequently they occur in the text.
    /// A value of 1 means no penalty, while values larger than 1 discourage and values smaller encourage.
    #[builder(default, setter(strip_option))]
    #[validate(range(exclusive_min = 0.0, max = 2.0))]
    pub repetition_penalty: Option<f32>,

    /// If true, sampling will be forced to be greedy.
    /// The backend is responsible for selecting the correct backend-specific options to
    /// implement this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub greed_sampling: Option<bool>,

    /// If true, the preproessor will try to bypass the prompt template and pass the prompt directly to
    /// to the tokenizer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub use_raw_prompt: Option<bool>,

    /// Annotations
    /// User requests triggers which result in the request issue back out-of-band information in the SSE
    /// stream using the `event:` field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub annotations: Option<Vec<String>>,

    /// Targeted backend instance ID for the request
    /// If set, the request will be routed to backend instance with the given ID.
    /// If not set, the request will be routed to the best matching instance.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_instance_id: Option<i64>,

    /// Pre-tokenized data to use instead of tokenizing the prompt
    /// If provided along with backend_instance_id, these tokens will be used directly
    /// and tokenization will be skipped.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_data: Option<Vec<u32>>,
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

    /// Maximum number of thinking tokens allowed
    /// NOTE: Currently passed through to backends as a no-op for future implementation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub max_thinking_tokens: Option<u32>,
}

impl Default for NvExt {
    fn default() -> Self {
        NvExt::builder().build().unwrap()
    }
}

impl NvExt {
    pub fn builder() -> NvExtBuilder {
        NvExtBuilder::default()
    }
}

fn validate_nv_ext(_nv_ext: &NvExt) -> Result<(), ValidationError> {
    Ok(())
}

pub fn validate_top_k(top_k: i32) -> Result<(), ValidationError> {
    if top_k == -1 || (top_k >= 1) {
        return Ok(());
    }
    let mut error = ValidationError::new("top_k");
    error.message = Some("top_k must be -1 or greater than or equal to 1".into());
    Err(error)
}

impl NvExtBuilder {
    pub fn add_annotation(&mut self, annotation: impl Into<String>) -> &mut Self {
        self.annotations
            .get_or_insert_with(|| Some(vec![]))
            .as_mut()
            .expect("stop should always be Some(Vec)")
            .push(annotation.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use validator::Validate;

    use super::*;

    // Test default builder configuration
    #[test]
    fn test_nv_ext_builder_default() {
        let nv_ext = NvExt::builder().build().unwrap();
        assert_eq!(nv_ext.ignore_eos, None);
        assert_eq!(nv_ext.top_k, None);
        assert_eq!(nv_ext.repetition_penalty, None);
        assert_eq!(nv_ext.greed_sampling, None);
        assert_eq!(nv_ext.guided_json, None);
        assert_eq!(nv_ext.guided_regex, None);
        assert_eq!(nv_ext.guided_grammar, None);
        assert_eq!(nv_ext.guided_choice, None);
        assert_eq!(nv_ext.max_thinking_tokens, None);
    }

    // Test valid builder configurations
    #[test]
    fn test_nv_ext_builder_custom() {
        let nv_ext = NvExt::builder()
            .ignore_eos(true)
            .top_k(10)
            .repetition_penalty(1.5)
            .greed_sampling(true)
            .guided_json(serde_json::json!({"type": "object"}))
            .guided_regex("^[0-9]+$".to_string())
            .guided_grammar("S -> 'a' S 'b' | 'c'".to_string())
            .guided_choice(vec!["choice1".to_string(), "choice2".to_string()])
            .guided_decoding_backend("xgrammar".to_string())
            .max_thinking_tokens(1024)
            .build()
            .unwrap();

        assert_eq!(nv_ext.ignore_eos, Some(true));
        assert_eq!(nv_ext.top_k, Some(10));
        assert_eq!(nv_ext.repetition_penalty, Some(1.5));
        assert_eq!(nv_ext.greed_sampling, Some(true));
        assert_eq!(
            nv_ext.guided_json,
            Some(serde_json::json!({"type": "object"}))
        );
        assert_eq!(nv_ext.guided_regex, Some("^[0-9]+$".to_string()));
        assert_eq!(
            nv_ext.guided_grammar,
            Some("S -> 'a' S 'b' | 'c'".to_string())
        );
        assert_eq!(
            nv_ext.guided_choice,
            Some(vec!["choice1".to_string(), "choice2".to_string()])
        );
        assert_eq!(nv_ext.guided_decoding_backend, Some("xgrammar".to_string()));
        assert_eq!(nv_ext.max_thinking_tokens, Some(1024));
        // Validate the built struct
        assert!(nv_ext.validate().is_ok());
    }

    // Test invalid `top_k` validation using proptest
    proptest! {
        #[test]
        fn test_invalid_top_k_value(top_k in any::<i32>().prop_filter("Invalid top_k", |&k| k < -1 || (k > 0 && k < 1))) {
            let nv_ext = NvExt::builder()
                .top_k(top_k)
                .build()
                .unwrap();

            let validation_result = nv_ext.validate();
            assert!(validation_result.is_err(), "top_k should fail validation if less than -1 or in the invalid range 0 < top_k < 1");
        }
    }

    // Test valid `top_k` values
    #[test]
    fn test_valid_top_k_values() {
        let nv_ext = NvExt::builder().top_k(-1).build().unwrap();
        assert!(nv_ext.validate().is_ok());

        let nv_ext = NvExt::builder().top_k(1).build().unwrap();
        assert!(nv_ext.validate().is_ok());

        let nv_ext = NvExt::builder().top_k(10).build().unwrap();
        assert!(nv_ext.validate().is_ok());
    }

    // Test valid repetition_penalty values
    proptest! {
        #[test]
        fn test_valid_repetition_penalty_values(repetition_penalty in 0.01f32..=2.0f32) {
            let nv_ext = NvExt::builder()
                .repetition_penalty(repetition_penalty)
                .build()
                .unwrap();

            let validation_result = nv_ext.validate();
            assert!(validation_result.is_ok(), "repetition_penalty should be valid within the range (0, 2]");
        }
    }

    // Test invalid repetition_penalty values
    proptest! {
        #[test]
        fn test_invalid_repetition_penalty_values(repetition_penalty in -10.0f32..0.0f32) {
            let nv_ext = NvExt::builder()
                .repetition_penalty(repetition_penalty)
                .build()
                .unwrap();

            let validation_result = nv_ext.validate();
            assert!(validation_result.is_err(), "repetition_penalty should fail validation when outside the range (0, 2]");
        }
    }
}
