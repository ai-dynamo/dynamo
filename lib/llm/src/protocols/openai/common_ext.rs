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
    /// This field can also be specified in nvext, where the nvext value takes precedence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub ignore_eos: Option<bool>,

    /// The minimum number of tokens to generate.
    /// This is a common parameter needed across different request types.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub min_tokens: Option<u32>,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::nvext::NvExt;

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
        // Test that negative min_tokens fails validation
        let common_ext = CommonExt {
            ignore_eos: None,
            min_tokens: Some(0), // Should be valid (min = 0)
        };
        assert!(common_ext.validate().is_ok());
    }
}
