// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    pub top_k: Option<i64>,

    /// How much to penalize tokens based on how frequently they occur in the text.
    /// A value of 1 means no penalty, while values larger than 1 discourage and values smaller encourage.
    #[builder(default, setter(strip_option))]
    #[validate(range(exclusive_min = 0.0, max = 2.0))]
    pub repetition_penalty: Option<f64>,

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

    /// Url for multimodal model asset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub multimodal_url: Option<String>,

    /// Modality of the multimodal url.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    #[validate(custom(function = "validate_modality"))]
    pub modality: Option<String>,
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

fn validate_top_k(top_k: i64) -> Result<(), ValidationError> {
    if top_k == -1 || (top_k >= 1) {
        return Ok(());
    }
    let mut error = ValidationError::new("top_k");
    error.message = Some("top_k must be -1 or greater than or equal to 1".into());
    Err(error)
}

fn validate_modality(modality: &str) -> Result<(), ValidationError> {
    if modality == "image" {
        Ok(())
    } else {
        let mut error = ValidationError::new("modality");
        error.message = Some("modality must be 'image'".into());
        Err(error)
    }
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
        assert_eq!(nv_ext.multimodal_url, None);
        assert_eq!(nv_ext.modality, None);
    }

    // Test valid builder configurations
    #[test]
    fn test_nv_ext_builder_custom() {
        let nv_ext = NvExt::builder()
            .ignore_eos(true)
            .top_k(10)
            .repetition_penalty(1.5)
            .greed_sampling(true)
            .multimodal_url("https://example.com/multimodal.url".to_string())
            .modality("image".to_string())
            .build()
            .unwrap();

        assert_eq!(nv_ext.ignore_eos, Some(true));
        assert_eq!(nv_ext.top_k, Some(10));
        assert_eq!(nv_ext.repetition_penalty, Some(1.5));
        assert_eq!(nv_ext.greed_sampling, Some(true));
        assert_eq!(nv_ext.multimodal_url, Some("https://example.com/multimodal.url".to_string()));
        assert_eq!(nv_ext.modality, Some("image".to_string()));

        // Validate the built struct
        assert!(nv_ext.validate().is_ok());
    }

    // Test invalid `top_k` validation using proptest
    proptest! {
        #[test]
        fn test_invalid_top_k_value(top_k in any::<i64>().prop_filter("Invalid top_k", |&k| k < -1 || (k > 0 && k < 1))) {
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
        fn test_valid_repetition_penalty_values(repetition_penalty in 0.01f64..=2.0f64) {
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
        fn test_invalid_repetition_penalty_values(repetition_penalty in -10.0f64..0.0f64) {
            let nv_ext = NvExt::builder()
                .repetition_penalty(repetition_penalty)
                .build()
                .unwrap();

            let validation_result = nv_ext.validate();
            assert!(validation_result.is_err(), "repetition_penalty should fail validation when outside the range (0, 2]");
        }
    }

    // Test modality values
    proptest! {
        #[test]
        fn test_modality_validation(modality in any::<String>()) {
            let nv_ext = NvExt::builder().modality(modality.clone()).build().unwrap();
            let validation_result = nv_ext.validate();

            if modality == "image" {
                assert!(validation_result.is_ok(), "modality 'image' should be valid");
            } else {
                assert!(validation_result.is_err(), "modality '{}' should be invalid", modality);
            }
        }
    }
}
