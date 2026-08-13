// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! YAML schema for process-wide custom worker-selection policy instances.

use std::collections::HashMap;

use serde::Deserialize;

use super::policy_config::{RouterPolicyConfigError, validate_identifier};

/// Process-wide configuration for worker selection in a custom Dynamo image.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerSelectionConfig {
    aggregated: Option<String>,
    prefill: Option<String>,
    decode: Option<String>,
    has_explicit_stage_selection: bool,
    instances: HashMap<String, WorkerSelectionInstance>,
}

impl WorkerSelectionConfig {
    /// The selected instance for a full-request worker pool.
    pub fn aggregated_instance(&self) -> Option<&str> {
        self.aggregated.as_deref()
    }

    pub(crate) fn prefill_instance(&self) -> Option<&str> {
        self.prefill.as_deref()
    }

    pub(crate) fn decode_instance(&self) -> Option<&str> {
        self.decode.as_deref()
    }

    pub(crate) fn has_explicit_stage_selection(&self) -> bool {
        self.has_explicit_stage_selection
    }

    /// Look up one named instance.
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

/// One named, parameterized worker-selection policy instance.
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
pub(super) struct RawWorkerSelectionConfig {
    // Compatibility with v1.4 policy files during the N-2 upgrade window.
    // TODO(v1.7): Remove after v1.4 falls outside the compatibility window.
    #[serde(default, rename = "default")]
    legacy_default: Option<String>,
    #[serde(default)]
    aggregated: Option<String>,
    #[serde(default)]
    prefill: Option<String>,
    #[serde(default)]
    decode: Option<String>,
    #[serde(default)]
    instances: Vec<RawWorkerSelectionInstance>,
}

impl RawWorkerSelectionConfig {
    pub(super) fn resolve(self) -> Result<WorkerSelectionConfig, RouterPolicyConfigError> {
        if self.instances.is_empty()
            && self.legacy_default.is_none()
            && self.aggregated.is_none()
            && self.prefill.is_none()
            && self.decode.is_none()
        {
            return Err(RouterPolicyConfigError::Validation(
                "worker_selection must define an instance or a default, aggregated, prefill, or decode selection"
                    .to_string(),
            ));
        }

        if self.legacy_default.is_some() && self.aggregated.is_some() {
            return Err(RouterPolicyConfigError::Validation(
                "worker_selection cannot define both deprecated default and aggregated selections"
                    .to_string(),
            ));
        }

        let mut instances = HashMap::with_capacity(self.instances.len());
        for raw in self.instances {
            validate_identifier(&raw.name, "instance", "worker_selection")?;
            if raw.name == "default" {
                return Err(RouterPolicyConfigError::Validation(
                    "worker_selection instance name 'default' is reserved for Dynamo's built-in selector".to_string(),
                ));
            }
            validate_identifier(&raw.policy_type, "policy type", "worker_selection")?;
            if !matches!(raw.parameters, serde_yaml::Value::Mapping(_)) {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "worker_selection instance {:?} parameters must be a YAML mapping",
                    raw.name
                )));
            }
            let instance = WorkerSelectionInstance {
                policy_type: raw.policy_type,
                parameters: raw.parameters,
            };
            if instances.insert(raw.name.clone(), instance).is_some() {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "worker_selection contains duplicate instance {:?}",
                    raw.name
                )));
            }
        }

        for (stage, selected) in [
            ("default", self.legacy_default.as_deref()),
            ("aggregated", self.aggregated.as_deref()),
            ("prefill", self.prefill.as_deref()),
            ("decode", self.decode.as_deref()),
        ] {
            if let Some(selected) = selected
                && selected != "default"
                && !instances.contains_key(selected)
            {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "worker_selection {stage} {selected:?} does not name a configured instance"
                )));
            }
        }

        if self.legacy_default.is_some() {
            tracing::warn!(
                "worker_selection.default is deprecated; use explicit worker_selection.aggregated, worker_selection.prefill, and worker_selection.decode selections"
            );
        }

        let has_explicit_stage_selection = self.prefill.is_some() || self.decode.is_some();
        let aggregated = self.aggregated.or_else(|| self.legacy_default.clone());
        let prefill = self.prefill.or_else(|| self.legacy_default.clone());
        let decode = self.decode.or(self.legacy_default);

        Ok(WorkerSelectionConfig {
            aggregated,
            prefill,
            decode,
            has_explicit_stage_selection,
            instances,
        })
    }
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
