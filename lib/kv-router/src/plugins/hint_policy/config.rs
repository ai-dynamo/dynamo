// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! YAML schema for the ordered hint policies each worker role runs.

use std::collections::HashSet;

use serde::Deserialize;

use crate::WorkerType;
use crate::scheduling::policy_config::{RouterPolicyConfigError, validate_identifier};

/// Process-wide hint-policy configuration: the ordered hint policies each worker role runs.
///
/// A role the config doesn't list runs the registry's default hint policies. Listing a role, even
/// as an empty list, replaces the defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct HintPolicyConfig {
    aggregated: Option<Vec<HintPolicyEntry>>,
    prefill: Option<Vec<HintPolicyEntry>>,
    decode: Option<Vec<HintPolicyEntry>>,
    encode: Option<Vec<HintPolicyEntry>>,
}

impl HintPolicyConfig {
    /// The hint policies a worker role lists, in order; `None` when the role isn't listed.
    pub fn role(&self, worker_type: WorkerType) -> Option<&[HintPolicyEntry]> {
        match worker_type {
            WorkerType::Aggregated => self.aggregated.as_deref(),
            WorkerType::Prefill => self.prefill.as_deref(),
            WorkerType::Decode => self.decode.as_deref(),
            WorkerType::Encode => self.encode.as_deref(),
        }
    }
}

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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawHintPolicyConfig {
    #[serde(default)]
    aggregated: Option<Vec<RawHintPolicyEntry>>,
    #[serde(default)]
    prefill: Option<Vec<RawHintPolicyEntry>>,
    #[serde(default)]
    decode: Option<Vec<RawHintPolicyEntry>>,
    #[serde(default)]
    encode: Option<Vec<RawHintPolicyEntry>>,
}

impl RawHintPolicyConfig {
    pub(crate) fn resolve(self) -> Result<HintPolicyConfig, RouterPolicyConfigError> {
        Ok(HintPolicyConfig {
            aggregated: resolve_role("aggregated", self.aggregated)?,
            prefill: resolve_role("prefill", self.prefill)?,
            decode: resolve_role("decode", self.decode)?,
            encode: resolve_role("encode", self.encode)?,
        })
    }
}

fn resolve_role(
    role: &str,
    entries: Option<Vec<RawHintPolicyEntry>>,
) -> Result<Option<Vec<HintPolicyEntry>>, RouterPolicyConfigError> {
    let Some(entries) = entries else {
        return Ok(None);
    };
    let location = format!("hint_policies {role}");
    let mut names = HashSet::with_capacity(entries.len());
    let mut resolved = Vec::with_capacity(entries.len());
    for raw in entries {
        validate_identifier(&raw.policy_type, "policy type", &location)?;
        if let Some(name) = &raw.name {
            validate_identifier(name, "entry", &location)?;
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
    Ok(Some(resolved))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHintPolicyEntry {
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
