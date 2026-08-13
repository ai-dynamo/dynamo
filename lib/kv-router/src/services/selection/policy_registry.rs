// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup registration for statically linked worker-selection policies.

use std::collections::HashMap;
use std::sync::Arc;

use serde::de::DeserializeOwned;
use thiserror::Error;

use crate::WorkerSelectionPolicyFactory;
use crate::WorkerType;
use crate::config::KvRouterConfig;
use crate::scheduling::config::WorkerSelectionPolicySelections;
pub use crate::scheduling::config::{
    DYN_ROUTER_DECODE_POLICY, DYN_ROUTER_PREFILL_POLICY, DYN_ROUTER_WORKER_SELECTION_POLICY,
};
use crate::scheduling::selector::WorkerSelectionPolicy;

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

    /// Resolve the configured policy instances once at process startup.
    ///
    /// Aggregated, prefill, and decode pools can select separate instances. The returned factory
    /// dispatches by the typed role that the host supplies when it constructs each worker pool.
    pub fn resolve(
        &self,
        config: &KvRouterConfig,
    ) -> Result<Option<WorkerSelectionPolicyFactory>, WorkerSelectionPolicyRegistryError> {
        let selected = config.selected_worker_selection_policy_instances()?;
        let policy_config = config.worker_selection_config().map_err(|source| {
            crate::scheduling::config::WorkerSelectionPolicyConfigError::Config { source }
        })?;
        self.resolve_selections(policy_config, selected)
    }

    fn resolve_selections(
        &self,
        policy_config: Option<&crate::scheduling::WorkerSelectionConfig>,
        selected: WorkerSelectionPolicySelections,
    ) -> Result<Option<WorkerSelectionPolicyFactory>, WorkerSelectionPolicyRegistryError> {
        let mut resolved = HashMap::new();
        let aggregated = self.resolve_selected_cached(
            policy_config,
            selected.aggregated.as_deref(),
            &mut resolved,
        )?;
        let prefill = self.resolve_selected_cached(
            policy_config,
            selected.prefill.as_deref(),
            &mut resolved,
        )?;
        let decode =
            self.resolve_selected_cached(policy_config, selected.decode.as_deref(), &mut resolved)?;

        if aggregated.is_none() && prefill.is_none() && decode.is_none() {
            return Ok(None);
        }

        Ok(Some(Arc::new(move |config, worker_type, partition| {
            let selected = match worker_type {
                WorkerType::Aggregated | WorkerType::Encode => aggregated.as_ref(),
                WorkerType::Prefill => prefill.as_ref(),
                WorkerType::Decode => decode.as_ref(),
            };
            match selected {
                Some(factory) => factory(config, worker_type, partition),
                None => WorkerSelectionPolicy::default(
                    config.clone(),
                    worker_type.default_selector_label(),
                ),
            }
        })))
    }

    fn resolve_selected_cached(
        &self,
        config: Option<&crate::scheduling::WorkerSelectionConfig>,
        selected: Option<&str>,
        resolved: &mut HashMap<String, WorkerSelectionPolicyFactory>,
    ) -> Result<Option<WorkerSelectionPolicyFactory>, WorkerSelectionPolicyRegistryError> {
        let Some(selected) = selected else {
            return Ok(None);
        };
        if let Some(factory) = resolved.get(selected) {
            return Ok(Some(factory.clone()));
        }
        let factory = self.resolve_selected(config, Some(selected))?;
        if let Some(factory) = factory {
            resolved.insert(selected.to_owned(), factory.clone());
            Ok(Some(factory))
        } else {
            Ok(None)
        }
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
    use crate::WorkerSelectionPolicyError;
    use crate::identity::RoutingPartitionRef;
    use crate::scheduling::selector::{
        WorkerInputView, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy,
    };
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Parameters {
        threshold: usize,
    }

    fn policy(
        _config: &KvRouterConfig,
        _worker_type: WorkerType,
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

    struct NeverPicker;

    impl WorkerPicker for NeverPicker {
        fn pick(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            _input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            unreachable!("registry tests only construct policies")
        }
    }

    fn recording_provider(
        provider_calls: Arc<AtomicUsize>,
        factory_calls: Arc<Mutex<Vec<(usize, String)>>>,
    ) -> WorkerSelectionPolicyProvider {
        Arc::new(move |parameters| {
            let parameters: Parameters = parameters.deserialize()?;
            provider_calls.fetch_add(1, Ordering::Relaxed);
            let factory_calls = factory_calls.clone();
            Ok(Arc::new(move |config, worker_type, _partition| {
                factory_calls
                    .lock()
                    .unwrap()
                    .push((parameters.threshold, worker_type.to_string()));
                WorkerSelectionPolicy::new(
                    config.clone(),
                    worker_type.as_str(),
                    Vec::new(),
                    Box::new(NeverPicker),
                )
            }))
        })
    }

    fn config() -> crate::scheduling::WorkerSelectionConfig {
        crate::scheduling::RouterPolicyConfig::from_yaml(
            r#"
worker_selection:
  aggregated: first
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
    fn resolves_builtin_default_and_multiple_custom_instances() {
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
    fn resolved_factory_dispatches_role_specific_instances() {
        let provider_calls = Arc::new(AtomicUsize::new(0));
        let factory_calls = Arc::new(Mutex::new(Vec::new()));
        let mut registry = WorkerSelectionPolicyRegistry::default();
        registry
            .register(
                "alpha",
                recording_provider(provider_calls.clone(), factory_calls.clone()),
            )
            .unwrap();
        registry
            .register(
                "beta",
                recording_provider(provider_calls.clone(), factory_calls.clone()),
            )
            .unwrap();
        let config = config();
        let factory = registry
            .resolve_selections(
                Some(&config),
                WorkerSelectionPolicySelections {
                    aggregated: Some("first".to_string()),
                    prefill: Some("first".to_string()),
                    decode: Some("second".to_string()),
                },
            )
            .unwrap()
            .unwrap();
        let router_config = KvRouterConfig::default();
        let partition = RoutingPartitionRef::new("model", "default");

        factory(&router_config, WorkerType::Prefill, partition);
        factory(&router_config, WorkerType::Decode, partition);
        factory(&router_config, WorkerType::Aggregated, partition);
        factory(&router_config, WorkerType::Encode, partition);

        assert_eq!(provider_calls.load(Ordering::Relaxed), 2);
        assert_eq!(
            *factory_calls.lock().unwrap(),
            vec![
                (1, "prefill".to_string()),
                (2, "decode".to_string()),
                (1, "aggregated".to_string()),
                (1, "encode".to_string()),
            ]
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
  aggregated: missing-type
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
  aggregated: invalid-parameters
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
