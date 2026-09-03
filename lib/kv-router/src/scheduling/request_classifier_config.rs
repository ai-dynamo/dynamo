// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! YAML schema for one process-wide request-classifier plugin.

use serde::Deserialize;

use super::policy_config::{RouterPolicyConfigError, validate_identifier};

#[derive(Debug, Clone, PartialEq)]
pub struct RequestClassifierConfig {
    classifier_type: String,
    parameters: serde_yaml::Value,
}

impl RequestClassifierConfig {
    /// The classifier type registered by a linked plugin crate.
    pub fn classifier_type(&self) -> &str {
        &self.classifier_type
    }

    /// YAML parameters owned and validated by the linked plugin crate.
    pub fn parameters(&self) -> &serde_yaml::Value {
        &self.parameters
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RawRequestClassifierConfig {
    #[serde(rename = "type")]
    classifier_type: String,
    #[serde(default = "empty_parameters")]
    parameters: serde_yaml::Value,
}

impl RawRequestClassifierConfig {
    pub(super) fn resolve(self) -> Result<RequestClassifierConfig, RouterPolicyConfigError> {
        validate_identifier(
            &self.classifier_type,
            "classifier type",
            "request_classifier",
        )?;
        if self.classifier_type == "default" {
            return Err(RouterPolicyConfigError::Validation(
                "request_classifier type 'default' is reserved; omit request_classifier to use Dynamo's pass-through classifier"
                    .to_string(),
            ));
        }
        if !matches!(self.parameters, serde_yaml::Value::Mapping(_)) {
            return Err(RouterPolicyConfigError::Validation(
                "request_classifier parameters must be a YAML mapping".to_string(),
            ));
        }
        Ok(RequestClassifierConfig {
            classifier_type: self.classifier_type,
            parameters: self.parameters,
        })
    }
}

fn empty_parameters() -> serde_yaml::Value {
    serde_yaml::Value::Mapping(Default::default())
}
