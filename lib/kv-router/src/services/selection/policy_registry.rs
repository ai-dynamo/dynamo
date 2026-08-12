// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup registration for statically linked worker-selection policies.

use std::collections::HashMap;
use std::sync::Arc;

use serde::de::DeserializeOwned;
use thiserror::Error;

use crate::WorkerSelectionPolicyFactory;
use crate::config::KvRouterConfig;
pub use crate::scheduling::config::DYN_ROUTER_WORKER_SELECTION_POLICY;

/// Parses one policy instance's YAML parameters and creates its partition factory.
pub type WorkerSelectionPolicyProvider = Arc<
    dyn Fn(
            &WorkerSelectionPolicyParameters,
        ) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError>
        + Send
        + Sync,
>;

/// YAML parameters passed to a linked worker-selection policy provider.
#[derive(Debug, Clone)]
pub struct WorkerSelectionPolicyParameters(serde_yaml::Value);

impl WorkerSelectionPolicyParameters {
    fn new(value: serde_yaml::Value) -> Self {
        Self(value)
    }

    /// Deserialize this policy instance's parameter mapping into a policy-owned type.
    pub fn deserialize<T: DeserializeOwned>(
        &self,
    ) -> Result<T, WorkerSelectionPolicyProviderError> {
        serde_yaml::from_value(self.0.clone())
            .map_err(|source| WorkerSelectionPolicyProviderError::new(source.to_string()))
    }
}

/// A policy-owned validation failure while creating a parameterized policy factory.
#[derive(Debug, Error)]
#[error("{message}")]
pub struct WorkerSelectionPolicyProviderError {
    message: String,
}

impl WorkerSelectionPolicyProviderError {
    /// Construct an error suitable for user-facing startup diagnostics.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// A startup-only registry of policy types linked into a custom Dynamo image.
#[derive(Clone, Default)]
pub struct WorkerSelectionPolicyRegistry {
    providers: HashMap<String, WorkerSelectionPolicyProvider>,
}

#[derive(Debug, Error)]
pub enum WorkerSelectionPolicyRegistryError {
    #[error("worker-selection policy type must not be empty")]
    EmptyName,
    #[error("worker-selection policy type 'default' is reserved for Dynamo's built-in selector")]
    ReservedDefault,
    #[error("worker-selection policy type {name:?} is already registered")]
    Duplicate { name: String },
    #[error(transparent)]
    Selection(#[from] crate::scheduling::config::WorkerSelectionPolicyConfigError),
    #[error("unknown worker-selection instance {name:?}; configured instances: {available}")]
    UnknownInstance { name: String, available: String },
    #[error("unknown worker-selection policy type {name:?}; linked policy types: {available}")]
    UnknownType { name: String, available: String },
    #[error("invalid parameters for worker-selection policy type {policy_type:?}: {source}")]
    Provider {
        policy_type: String,
        #[source]
        source: WorkerSelectionPolicyProviderError,
    },
}

impl WorkerSelectionPolicyRegistry {
    /// Register a policy type supplied by a linked policy crate.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        provider: WorkerSelectionPolicyProvider,
    ) -> Result<(), WorkerSelectionPolicyRegistryError> {
        let name = name.into();
        if name.is_empty() {
            return Err(WorkerSelectionPolicyRegistryError::EmptyName);
        }
        if name == "default" {
            return Err(WorkerSelectionPolicyRegistryError::ReservedDefault);
        }
        if self.providers.contains_key(&name) {
            return Err(WorkerSelectionPolicyRegistryError::Duplicate { name });
        }
        self.providers.insert(name, provider);
        Ok(())
    }

    /// Resolve the configured policy instance once at process startup.
    ///
    /// `DYN_ROUTER_WORKER_SELECTION_POLICY`, when set, overrides the YAML default and names an
    /// instance. The reserved value `default` uses Dynamo's built-in selector.
    pub fn resolve(
        &self,
        config: &KvRouterConfig,
    ) -> Result<Option<WorkerSelectionPolicyFactory>, WorkerSelectionPolicyRegistryError> {
        let selected = config.selected_worker_selection_policy_instance()?;
        let policy_config = config.worker_selection_config().map_err(|source| {
            crate::scheduling::config::WorkerSelectionPolicyConfigError::Config { source }
        })?;
        self.resolve_selected(policy_config, selected.as_deref())
    }

    fn resolve_selected(
        &self,
        config: Option<&crate::scheduling::WorkerSelectionConfig>,
        selected: Option<&str>,
    ) -> Result<Option<WorkerSelectionPolicyFactory>, WorkerSelectionPolicyRegistryError> {
        let Some(selected) = selected else {
            return Ok(None);
        };
        if selected == "default" {
            return Ok(None);
        }
        let instance = config
            .and_then(|config| config.instance(selected))
            .ok_or_else(|| WorkerSelectionPolicyRegistryError::UnknownInstance {
                name: selected.to_owned(),
                available: config
                    .map(crate::scheduling::WorkerSelectionConfig::instance_names)
                    .filter(|names| !names.is_empty())
                    .map(|names| names.join(", "))
                    .unwrap_or_else(|| "<none>".to_owned()),
            })?;
        let provider = self.providers.get(instance.policy_type()).ok_or_else(|| {
            WorkerSelectionPolicyRegistryError::UnknownType {
                name: instance.policy_type().to_owned(),
                available: self.available_policy_types(),
            }
        })?;
        provider(&WorkerSelectionPolicyParameters::new(
            instance.parameters().clone(),
        ))
        .map(Some)
        .map_err(|source| WorkerSelectionPolicyRegistryError::Provider {
            policy_type: instance.policy_type().to_owned(),
            source,
        })
    }

    fn available_policy_types(&self) -> String {
        let mut available = self.providers.keys().cloned().collect::<Vec<_>>();
        available.sort_unstable();
        if available.is_empty() {
            "<none>".to_owned()
        } else {
            available.join(", ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::RoutingPartitionRef;
    use crate::scheduling::selector::WorkerSelectionPolicy;

    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Parameters {
        threshold: usize,
    }

    fn policy(
        _config: &KvRouterConfig,
        _worker_type: &'static str,
        _partition: RoutingPartitionRef<'_>,
    ) -> WorkerSelectionPolicy {
        unreachable!("registry tests never invoke policy factories")
    }

    fn provider(
        parameters: &WorkerSelectionPolicyParameters,
    ) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
        let parameters: Parameters = parameters.deserialize()?;
        if parameters.threshold == 0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "threshold must be greater than zero",
            ));
        }
        Ok(Arc::new(policy))
    }

    fn config() -> crate::scheduling::WorkerSelectionConfig {
        crate::scheduling::RouterPolicyConfig::from_yaml(
            r#"
worker_selection:
  default: first
  instances:
    - name: first
      type: alpha
      parameters:
        threshold: 1
    - name: second
      type: beta
      parameters:
        threshold: 2
"#,
        )
        .unwrap()
        .worker_selection()
        .unwrap()
        .clone()
    }

    #[test]
    fn resolves_default_and_multiple_custom_instances() {
        let mut registry = WorkerSelectionPolicyRegistry::default();
        registry.register("alpha", Arc::new(provider)).unwrap();
        registry.register("beta", Arc::new(provider)).unwrap();
        let config = config();

        assert!(
            registry
                .resolve_selected(Some(&config), Some("default"))
                .unwrap()
                .is_none()
        );
        assert!(
            registry
                .resolve_selected(Some(&config), Some("first"))
                .unwrap()
                .is_some()
        );
        assert!(
            registry
                .resolve_selected(Some(&config), Some("second"))
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn rejects_invalid_and_unknown_names() {
        let mut registry = WorkerSelectionPolicyRegistry::default();
        let provider: WorkerSelectionPolicyProvider = Arc::new(provider);

        assert!(matches!(
            registry.resolve_selected(None, Some("missing")),
            Err(WorkerSelectionPolicyRegistryError::UnknownInstance { available, .. }) if available == "<none>"
        ));
        assert!(matches!(
            registry.register("", provider.clone()),
            Err(WorkerSelectionPolicyRegistryError::EmptyName)
        ));
        assert!(matches!(
            registry.register("default", provider.clone()),
            Err(WorkerSelectionPolicyRegistryError::ReservedDefault)
        ));
        registry.register("alpha", provider.clone()).unwrap();
        assert!(matches!(
            registry.register("alpha", provider),
            Err(WorkerSelectionPolicyRegistryError::Duplicate { name }) if name == "alpha"
        ));

        let missing_type = crate::scheduling::RouterPolicyConfig::from_yaml(
            r#"
worker_selection:
  default: missing-type
  instances:
    - name: missing-type
      type: missing
      parameters: {}
"#,
        )
        .unwrap()
        .worker_selection()
        .unwrap()
        .clone();
        assert!(matches!(
            registry.resolve_selected(Some(&missing_type), Some("missing-type")),
            Err(WorkerSelectionPolicyRegistryError::UnknownType { name, available })
                if name == "missing" && available == "alpha"
        ));

        let config = crate::scheduling::RouterPolicyConfig::from_yaml(
            r#"
worker_selection:
  default: invalid-parameters
  instances:
    - name: invalid-parameters
      type: alpha
      parameters:
        threshold: 0
"#,
        )
        .unwrap()
        .worker_selection()
        .unwrap()
        .clone();
        assert!(matches!(
            registry.resolve_selected(Some(&config), Some("invalid-parameters")),
            Err(WorkerSelectionPolicyRegistryError::Provider { policy_type, .. })
                if policy_type == "alpha"
        ));
    }
}
