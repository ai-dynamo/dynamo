// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

/// Common extensions for OpenAI API requests that are not part of the standard OpenAI spec
/// but are commonly needed across different request types.
#[derive(Serialize, Deserialize, Builder, Validate, Debug, Clone, Default)]
pub struct CommonExt {
    /// If true, the model will ignore the end of string token and generate to max_tokens.
    /// This field can also be specified in nvext, where the nvext value takes precedence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub ignore_eos: Option<bool>,

    /// The minimum number of tokens to generate.
    /// This is a common parameter needed across different request types.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    #[validate(range(min = 0))]
    pub min_tokens: Option<u32>,
}

impl CommonExt {
    pub fn builder() -> CommonExtBuilder {
        CommonExtBuilder::default()
    }

    /// Merge values from this CommonExt with values from NvExt, giving precedence to NvExt values.
    /// This allows backward compatibility where nvext values override root-level values.
    pub fn merge_with_nvext(&self, nvext: Option<&super::nvext::NvExt>) -> (Option<bool>, Option<u32>) {
        let ignore_eos = nvext
            .and_then(|nv| nv.ignore_eos)
            .or(self.ignore_eos);

        // min_tokens is only available at root level (not in nvext currently)
        let min_tokens = self.min_tokens;

        (ignore_eos, min_tokens)
    }
}

/// Trait for types that provide CommonExt fields
pub trait CommonExtProvider {
    /// Get a reference to the CommonExt struct if available
    fn common_ext(&self) -> Option<&CommonExt>;

    /// Get the effective ignore_eos value, considering both CommonExt and NvExt
    fn effective_ignore_eos(&self) -> Option<bool>;

    /// Get the effective min_tokens value
    fn effective_min_tokens(&self) -> Option<u32>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::nvext::NvExt;

    #[test]
    fn test_common_ext_builder_default() {
        let common_ext = CommonExt::builder().build().unwrap();
        assert_eq!(common_ext.ignore_eos, None);
        assert_eq!(common_ext.min_tokens, None);
    }

    #[test]
    fn test_common_ext_builder_with_values() {
        let common_ext = CommonExt::builder()
            .ignore_eos(true)
            .min_tokens(10)
            .build()
            .unwrap();

        assert_eq!(common_ext.ignore_eos, Some(true));
        assert_eq!(common_ext.min_tokens, Some(10));
    }

    #[test]
    fn test_merge_with_nvext_precedence() {
        // Test that nvext values take precedence over CommonExt values
        let common_ext = CommonExt::builder()
            .ignore_eos(false)
            .min_tokens(5)
            .build()
            .unwrap();

        let nvext = NvExt::builder()
            .ignore_eos(true)  // This should override CommonExt's false value
            .build()
            .unwrap();

        let (ignore_eos, min_tokens) = common_ext.merge_with_nvext(Some(&nvext));

        assert_eq!(ignore_eos, Some(true));  // nvext value takes precedence
        assert_eq!(min_tokens, Some(5));     // min_tokens from CommonExt
    }

    #[test]
    fn test_merge_with_nvext_fallback() {
        // Test that CommonExt values are used when nvext doesn't specify them
        let common_ext = CommonExt::builder()
            .ignore_eos(true)
            .min_tokens(10)
            .build()
            .unwrap();

        let nvext = NvExt::builder()
            // No ignore_eos specified in nvext
            .build()
            .unwrap();

        let (ignore_eos, min_tokens) = common_ext.merge_with_nvext(Some(&nvext));

        assert_eq!(ignore_eos, Some(true));  // Falls back to CommonExt value
        assert_eq!(min_tokens, Some(10));
    }

    #[test]
    fn test_merge_with_no_nvext() {
        // Test behavior when nvext is None
        let common_ext = CommonExt::builder()
            .ignore_eos(true)
            .min_tokens(15)
            .build()
            .unwrap();

        let (ignore_eos, min_tokens) = common_ext.merge_with_nvext(None);

        assert_eq!(ignore_eos, Some(true));
        assert_eq!(min_tokens, Some(15));
    }

    #[test]
    fn test_validation_min_tokens() {
        // Test that negative min_tokens fails validation
        let common_ext = CommonExt {
            ignore_eos: None,
            min_tokens: Some(0),  // Should be valid (min = 0)
        };
        assert!(common_ext.validate().is_ok());
    }
}
