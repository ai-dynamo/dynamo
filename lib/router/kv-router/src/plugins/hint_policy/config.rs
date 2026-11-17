// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! YAML schema for one hint in a worker-selection role's `hints` list.

use std::collections::HashSet;

use serde::Deserialize;

use crate::scheduling::policy_config::{RouterPolicyConfigError, validate_identifier};

/// One hint policy in a role's list.
#[derive(Debug, Clone, PartialEq)]
pub struct HintPolicyEntry {
    name: String,
    policy_type: String,
    parameters: serde_yaml::Value,
}

impl HintPolicyEntry {
    /// The entry's name in logs: its `name`, or its type when it has none.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The policy type registered by Dynamo or a linked plugin crate.
    pub fn policy_type(&self) -> &str {
        &self.policy_type
    }

    /// YAML parameters owned and validated by the policy type's provider.
    pub fn parameters(&self) -> &serde_yaml::Value {
        &self.parameters
    }
}

/// Validate one role's `hints` list, read at `location`, in order.
pub(crate) fn resolve_hints(
    location: &str,
    entries: Vec<RawHintPolicyEntry>,
) -> Result<Vec<HintPolicyEntry>, RouterPolicyConfigError> {
    let mut names = HashSet::with_capacity(entries.len());
    let mut resolved = Vec::with_capacity(entries.len());
    for raw in entries {
        validate_identifier(&raw.policy_type, "policy type", location)?;
        if let Some(name) = &raw.name {
            validate_identifier(name, "entry", location)?;
        }
        if !matches!(raw.parameters, serde_yaml::Value::Mapping(_)) {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} entry {:?} parameters must be a YAML mapping",
                raw.name.as_deref().unwrap_or(&raw.policy_type)
            )));
        }
        let name = raw.name.unwrap_or_else(|| raw.policy_type.clone());
        if !names.insert(name.clone()) {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} lists {name:?} more than once; give each entry of the same type a distinct name"
            )));
        }
        resolved.push(HintPolicyEntry {
            name,
            policy_type: raw.policy_type,
            parameters: raw.parameters,
        });
    }
    Ok(resolved)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawHintPolicyEntry {
    #[serde(rename = "type")]
    policy_type: String,
    #[serde(default = "empty_parameters")]
    parameters: serde_yaml::Value,
    #[serde(default)]
    name: Option<String>,
}

fn empty_parameters() -> serde_yaml::Value {
    serde_yaml::Value::Mapping(Default::default())
}
