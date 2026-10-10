// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! YAML schema for one process-wide KV-hint policy plugin.

use serde::Deserialize;

use crate::scheduling::policy_config::{RouterPolicyConfigError, validate_identifier};

/// Startup selection and plugin-owned parameters for one KV-hint policy type.
#[derive(Debug, Clone, PartialEq)]
pub struct KvHintPolicyConfig {
    policy_type: String,
    parameters: serde_yaml::Value,
}

impl KvHintPolicyConfig {
    /// The policy type registered by a linked plugin crate.
    pub fn policy_type(&self) -> &str {
        &self.policy_type
    }

    /// YAML parameters owned and validated by the linked plugin crate.
    pub fn parameters(&self) -> &serde_yaml::Value {
        &self.parameters
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawKvHintPolicyConfig {
    #[serde(rename = "type")]
    policy_type: String,
    #[serde(default = "empty_parameters")]
    parameters: serde_yaml::Value,
}

impl RawKvHintPolicyConfig {
    pub(crate) fn resolve(self) -> Result<KvHintPolicyConfig, RouterPolicyConfigError> {
        validate_identifier(&self.policy_type, "policy type", "kv_hint_policy")?;
        if self.policy_type == "default" {
            return Err(RouterPolicyConfigError::Validation(
                "kv_hint_policy type 'default' is reserved; omit kv_hint_policy to disable custom hint formulation"
                    .to_string(),
            ));
        }
        if !matches!(self.parameters, serde_yaml::Value::Mapping(_)) {
            return Err(RouterPolicyConfigError::Validation(
                "kv_hint_policy parameters must be a YAML mapping".to_string(),
            ));
        }
        Ok(KvHintPolicyConfig {
            policy_type: self.policy_type,
            parameters: self.parameters,
        })
    }
}

fn empty_parameters() -> serde_yaml::Value {
    serde_yaml::Value::Mapping(Default::default())
}
