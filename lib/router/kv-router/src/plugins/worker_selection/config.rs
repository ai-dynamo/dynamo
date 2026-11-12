// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! YAML schema for process-wide worker-selection policy selections.
//!
//! Each worker role (`aggregated`, `prefill`, `decode`, `encode`) selects its policy inline:
//!
//! ```yaml
//! worker_selection:
//!   aggregated: dynamo-two-tier-cost-fn        # a policy type with default parameters
//!   decode:
//!     type: dynamo-default-cost-fn             # a policy type with parameters
//!     parameters:
//!       router_temperature: 0.5
//! ```
//!
//! The older `instances:` list, which named a `{type, parameters}` pair once and referenced it
//! from each role by name, is deprecated and still accepted.

use std::collections::HashMap;

use serde::Deserialize;

use crate::scheduling::policy_config::RouterPolicyConfigError;
use crate::scheduling::policy_config::validate_identifier;

/// Process-wide configuration for worker selection.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerSelectionConfig {
    aggregated: Option<WorkerSelectionPolicySelection>,
    prefill: Option<WorkerSelectionPolicySelection>,
    decode: Option<WorkerSelectionPolicySelection>,
    encode: Option<WorkerSelectionPolicySelection>,
    instances: HashMap<String, WorkerSelectionInstance>,
}

impl WorkerSelectionConfig {
    /// The policy selected for a full-request worker pool.
    pub fn aggregated(&self) -> Option<&WorkerSelectionPolicySelection> {
        self.aggregated.as_ref()
    }

    pub(crate) fn prefill(&self) -> Option<&WorkerSelectionPolicySelection> {
        self.prefill.as_ref()
    }

    pub(crate) fn decode(&self) -> Option<&WorkerSelectionPolicySelection> {
        self.decode.as_ref()
    }

    pub(crate) fn encode(&self) -> Option<&WorkerSelectionPolicySelection> {
        self.encode.as_ref()
    }

    /// Look up one named instance from the deprecated `instances:` list.
    pub fn instance(&self, name: &str) -> Option<&WorkerSelectionInstance> {
        self.instances.get(name)
    }

    /// Return configured instance names in stable order for diagnostics.
    pub fn instance_names(&self) -> Vec<String> {
        let mut names = self.instances.keys().cloned().collect::<Vec<_>>();
        names.sort_unstable();
        names
    }
}

/// One role's worker-selection policy selection.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerSelectionPolicySelection {
    /// A bare string: the reserved `default`, a registered policy type with default parameters,
    /// or (deprecated) the name of an entry in `instances:`.
    Named(String),
    /// An inline `{type, parameters}` spec.
    Inline(WorkerSelectionInstance),
}

impl WorkerSelectionPolicySelection {
    /// Whether this selection is the reserved built-in policy.
    pub fn is_default(&self) -> bool {
        matches!(self, Self::Named(name) if name == "default")
    }

    /// A short label for diagnostics: the name for a bare string, the type for an inline spec.
    pub fn label(&self) -> &str {
        match self {
            Self::Named(name) => name,
            Self::Inline(instance) => instance.policy_type(),
        }
    }
}

/// One parameterized worker-selection policy: a registered type plus its YAML parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerSelectionInstance {
    policy_type: String,
    parameters: serde_yaml::Value,
}

impl WorkerSelectionInstance {
    /// The policy type registered by a linked policy crate.
    pub fn policy_type(&self) -> &str {
        &self.policy_type
    }

    /// YAML parameters owned and validated by the linked policy crate.
    pub fn parameters(&self) -> &serde_yaml::Value {
        &self.parameters
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawWorkerSelectionConfig {
    aggregated: Option<serde_yaml::Value>,
    prefill: Option<serde_yaml::Value>,
    decode: Option<serde_yaml::Value>,
    encode: Option<serde_yaml::Value>,
    #[serde(default)]
    instances: Vec<RawWorkerSelectionInstance>,
}

enum RawRoleSelection {
    Named(String),
    Inline(RawInlineSelection),
}

impl RawRoleSelection {
    // Parsed per role rather than with `#[serde(untagged)]`, which discards the inner error
    // (for example an unknown field) and reports only that no variant matched.
    fn from_value(role: &str, value: serde_yaml::Value) -> Result<Self, RouterPolicyConfigError> {
        let location = format!("worker_selection.{role}");
        match value {
            serde_yaml::Value::String(name) => Ok(Self::Named(name)),
            serde_yaml::Value::Mapping(_) => serde_yaml::from_value(value)
                .map(Self::Inline)
                .map_err(|source| {
                    RouterPolicyConfigError::Validation(format!("{location}: {source}"))
                }),
            _ => Err(RouterPolicyConfigError::Validation(format!(
                "{location} must be a policy type name or a {{type, parameters}} mapping"
            ))),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawInlineSelection {
    #[serde(rename = "type")]
    policy_type: String,
    #[serde(default = "empty_parameters")]
    parameters: serde_yaml::Value,
}

impl RawWorkerSelectionConfig {
    pub(crate) fn resolve(self) -> Result<WorkerSelectionConfig, RouterPolicyConfigError> {
        if self.instances.is_empty()
            && self.aggregated.is_none()
            && self.prefill.is_none()
            && self.decode.is_none()
            && self.encode.is_none()
        {
            return Err(RouterPolicyConfigError::Validation(
                "worker_selection must select a policy for aggregated, prefill, decode, or encode"
                    .to_string(),
            ));
        }

        if !self.instances.is_empty() {
            tracing::warn!(
                "worker_selection.instances is deprecated for removal in v1.8; write each role's policy inline as a type name or a {{type, parameters}} mapping"
            );
        }

        let mut instances = HashMap::with_capacity(self.instances.len());
        for raw in self.instances {
            validate_identifier(&raw.name, "instance", "worker_selection")?;
            if raw.name == "default" {
                return Err(RouterPolicyConfigError::Validation(
                    "worker_selection instance name 'default' is reserved for Dynamo's built-in selector".to_string(),
                ));
            }
            let instance = resolve_instance(
                raw.policy_type,
                raw.parameters,
                &format!("worker_selection instance {:?}", raw.name),
            )?;
            if instances.insert(raw.name.clone(), instance).is_some() {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "worker_selection contains duplicate instance {:?}",
                    raw.name
                )));
            }
        }

        let resolve_role = |role: &str,
                            raw: Option<serde_yaml::Value>|
         -> Result<
            Option<WorkerSelectionPolicySelection>,
            RouterPolicyConfigError,
        > {
            let location = format!("worker_selection.{role}");
            let raw = raw
                .map(|value| RawRoleSelection::from_value(role, value))
                .transpose()?;
            Ok(match raw {
                None => None,
                Some(RawRoleSelection::Named(name)) => {
                    validate_identifier(&name, "policy", &location)?;
                    Some(WorkerSelectionPolicySelection::Named(name))
                }
                Some(RawRoleSelection::Inline(inline)) => {
                    if inline.policy_type == "default" {
                        return Err(RouterPolicyConfigError::Validation(format!(
                            "{location} type 'default' takes no parameters; write `{role}: default`"
                        )));
                    }
                    Some(WorkerSelectionPolicySelection::Inline(resolve_instance(
                        inline.policy_type,
                        inline.parameters,
                        &location,
                    )?))
                }
            })
        };

        Ok(WorkerSelectionConfig {
            aggregated: resolve_role("aggregated", self.aggregated)?,
            prefill: resolve_role("prefill", self.prefill)?,
            decode: resolve_role("decode", self.decode)?,
            encode: resolve_role("encode", self.encode)?,
            instances,
        })
    }
}

fn resolve_instance(
    policy_type: String,
    parameters: serde_yaml::Value,
    location: &str,
) -> Result<WorkerSelectionInstance, RouterPolicyConfigError> {
    validate_identifier(&policy_type, "policy type", location)?;
    if !matches!(parameters, serde_yaml::Value::Mapping(_)) {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} parameters must be a YAML mapping"
        )));
    }
    Ok(WorkerSelectionInstance {
        policy_type,
        parameters,
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWorkerSelectionInstance {
    name: String,
    #[serde(rename = "type")]
    policy_type: String,
    #[serde(default = "empty_parameters")]
    parameters: serde_yaml::Value,
}

fn empty_parameters() -> serde_yaml::Value {
    serde_yaml::Value::Mapping(Default::default())
}
