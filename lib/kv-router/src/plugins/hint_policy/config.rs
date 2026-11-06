// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! YAML schema for process-wide hint-policy instances and their per-role order.

use std::collections::{HashMap, HashSet};

use serde::Deserialize;

use crate::WorkerType;
use crate::scheduling::policy_config::{RouterPolicyConfigError, validate_identifier};

/// Process-wide hint-policy configuration: named instances and the ordered list each worker role
/// runs. Hint policies are opt-in; a role without a list runs none.
#[derive(Debug, Clone, PartialEq)]
pub struct HintPolicyConfig {
    aggregated: Vec<String>,
    prefill: Vec<String>,
    decode: Vec<String>,
    encode: Vec<String>,
    instances: HashMap<String, HintPolicyInstanceConfig>,
}

impl HintPolicyConfig {
    /// The ordered instance names a worker role runs.
    pub fn role(&self, worker_type: WorkerType) -> &[String] {
        match worker_type {
            WorkerType::Aggregated => &self.aggregated,
            WorkerType::Prefill => &self.prefill,
            WorkerType::Decode => &self.decode,
            WorkerType::Encode => &self.encode,
        }
    }

    /// Look up one named instance.
    pub fn instance(&self, name: &str) -> Option<&HintPolicyInstanceConfig> {
        self.instances.get(name)
    }

    /// Every configured instance name, including ones no role lists.
    pub(crate) fn instance_names(&self) -> impl Iterator<Item = &str> {
        self.instances.keys().map(String::as_str)
    }
}

/// One named, parameterized hint-policy instance.
#[derive(Debug, Clone, PartialEq)]
pub struct HintPolicyInstanceConfig {
    policy_type: String,
    parameters: serde_yaml::Value,
}

impl HintPolicyInstanceConfig {
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
    aggregated: Vec<String>,
    #[serde(default)]
    prefill: Vec<String>,
    #[serde(default)]
    decode: Vec<String>,
    #[serde(default)]
    encode: Vec<String>,
    #[serde(default)]
    instances: Vec<RawHintPolicyInstance>,
}

impl RawHintPolicyConfig {
    pub(crate) fn resolve(self) -> Result<HintPolicyConfig, RouterPolicyConfigError> {
        let mut instances = HashMap::with_capacity(self.instances.len());
        for raw in self.instances {
            validate_identifier(&raw.name, "instance", "hint_policies")?;
            validate_identifier(&raw.policy_type, "policy type", "hint_policies")?;
            if !matches!(raw.parameters, serde_yaml::Value::Mapping(_)) {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "hint_policies instance {:?} parameters must be a YAML mapping",
                    raw.name
                )));
            }
            let instance = HintPolicyInstanceConfig {
                policy_type: raw.policy_type,
                parameters: raw.parameters,
            };
            if instances.insert(raw.name.clone(), instance).is_some() {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "hint_policies contains duplicate instance {:?}",
                    raw.name
                )));
            }
        }

        for (role, selected) in [
            ("aggregated", &self.aggregated),
            ("prefill", &self.prefill),
            ("decode", &self.decode),
            ("encode", &self.encode),
        ] {
            let mut seen = HashSet::new();
            for name in selected {
                if !instances.contains_key(name) {
                    return Err(RouterPolicyConfigError::Validation(format!(
                        "hint_policies {role} entry {name:?} does not name a configured instance"
                    )));
                }
                if !seen.insert(name.as_str()) {
                    return Err(RouterPolicyConfigError::Validation(format!(
                        "hint_policies {role} lists {name:?} more than once"
                    )));
                }
            }
        }

        Ok(HintPolicyConfig {
            aggregated: self.aggregated,
            prefill: self.prefill,
            decode: self.decode,
            encode: self.encode,
            instances,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHintPolicyInstance {
    name: String,
    #[serde(rename = "type")]
    policy_type: String,
    #[serde(default = "empty_parameters")]
    parameters: serde_yaml::Value,
}

fn empty_parameters() -> serde_yaml::Value {
    serde_yaml::Value::Mapping(Default::default())
}
