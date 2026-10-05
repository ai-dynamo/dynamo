// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup registration for statically linked KV-hint policy plugins.

use std::collections::HashMap;
use std::sync::Arc;

use serde::de::DeserializeOwned;
use thiserror::Error;

use super::KvHintPolicy;
use crate::{KvRouterConfig, RoutingPartitionRef, WorkerType};

/// Constructs one configured policy per routing partition.
pub type KvHintPolicyConstructor = Arc<
    dyn for<'a> Fn(
            &KvHintPolicyParameters,
            &KvRouterConfig,
            WorkerType,
            RoutingPartitionRef<'a>,
        ) -> Result<Box<dyn KvHintPolicy>, KvHintPolicyConstructorError>
        + Send
        + Sync,
>;

/// YAML parameters passed to a linked KV-hint policy constructor.
#[derive(Debug, Clone)]
pub struct KvHintPolicyParameters(serde_yaml::Value);

impl KvHintPolicyParameters {
    /// Deserialize the parameter mapping into the plugin's own configuration type.
    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T, KvHintPolicyConstructorError> {
        serde_yaml::from_value(self.0.clone())
            .map_err(|source| KvHintPolicyConstructorError::new(source.to_string()))
    }
}

/// A plugin-owned construction failure reported during startup.
#[derive(Debug, Error)]
#[error("{message}")]
pub struct KvHintPolicyConstructorError {
    message: String,
}

impl KvHintPolicyConstructorError {
    /// Construct an error suitable for user-facing startup diagnostics.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Clone, Default)]
pub(crate) struct KvHintPolicyRegistry {
    constructors: HashMap<String, KvHintPolicyConstructor>,
}

#[derive(Clone)]
pub(crate) struct ConfiguredKvHintPolicy {
    policy_type: String,
    parameters: KvHintPolicyParameters,
    constructor: KvHintPolicyConstructor,
}

impl ConfiguredKvHintPolicy {
    pub(crate) fn construct(
        &self,
        config: &KvRouterConfig,
        worker_type: WorkerType,
        partition: RoutingPartitionRef<'_>,
    ) -> Result<Box<dyn KvHintPolicy>, KvHintPolicyRegistryError> {
        (self.constructor)(&self.parameters, config, worker_type, partition).map_err(|source| {
            KvHintPolicyRegistryError::Constructor {
                policy_type: self.policy_type.clone(),
                source,
            }
        })
    }
}

/// An error from KV-hint policy registration or startup resolution.
#[derive(Debug, Error)]
pub enum KvHintPolicyRegistryError {
    #[error("KV-hint policy type must not be empty")]
    EmptyName,
    #[error("KV-hint policy type 'default' is reserved for no custom policy")]
    ReservedDefault,
    #[error("KV-hint policy type {name:?} is already registered")]
    Duplicate { name: String },
    #[error("could not load kv_hint_policy from router_policy_config: {source}")]
    Config {
        #[source]
        source: crate::scheduling::RouterPolicyConfigError,
    },
    #[error("unknown KV-hint policy type {name:?}; linked policy types: {available}")]
    UnknownType { name: String, available: String },
    #[error("could not construct KV-hint policy type {policy_type:?}: {source}")]
    Constructor {
        policy_type: String,
        #[source]
        source: KvHintPolicyConstructorError,
    },
}

impl KvHintPolicyRegistry {
    pub fn is_empty(&self) -> bool {
        self.constructors.is_empty()
    }

    pub fn register(
        &mut self,
        name: impl Into<String>,
        constructor: KvHintPolicyConstructor,
    ) -> Result<(), KvHintPolicyRegistryError> {
        let name = name.into();
        if name.is_empty() {
            return Err(KvHintPolicyRegistryError::EmptyName);
        }
        if name == "default" {
            return Err(KvHintPolicyRegistryError::ReservedDefault);
        }
        if self.constructors.contains_key(&name) {
            return Err(KvHintPolicyRegistryError::Duplicate { name });
        }
        self.constructors.insert(name, constructor);
        Ok(())
    }

    pub fn resolve(
        &self,
        config: &KvRouterConfig,
    ) -> Result<Option<ConfiguredKvHintPolicy>, KvHintPolicyRegistryError> {
        let Some(policy) = config
            .kv_hint_policy_config()
            .map_err(|source| KvHintPolicyRegistryError::Config { source })?
        else {
            return Ok(None);
        };
        let policy_type = policy.policy_type();
        let constructor = self.constructors.get(policy_type).ok_or_else(|| {
            KvHintPolicyRegistryError::UnknownType {
                name: policy_type.to_owned(),
                available: self.available_policy_types(),
            }
        })?;
        Ok(Some(ConfiguredKvHintPolicy {
            policy_type: policy_type.to_owned(),
            parameters: KvHintPolicyParameters(policy.parameters().clone()),
            constructor: Arc::clone(constructor),
        }))
    }

    fn available_policy_types(&self) -> String {
        let mut available = self.constructors.keys().cloned().collect::<Vec<_>>();
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
    use crate::kv_hints::KvHintAction;
    use crate::plugins::kv_hint::{KvHintPolicyContext, KvHintPolicyError};

    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Parameters {
        enabled: bool,
    }

    struct EmptyPolicy;

    impl KvHintPolicy for EmptyPolicy {
        fn formulate(
            &self,
            _context: &KvHintPolicyContext<'_>,
        ) -> Result<Vec<KvHintAction>, Box<KvHintPolicyError>> {
            Ok(Vec::new())
        }
    }

    fn constructor(
        parameters: &KvHintPolicyParameters,
        _config: &KvRouterConfig,
        _worker_type: WorkerType,
        _partition: RoutingPartitionRef<'_>,
    ) -> Result<Box<dyn KvHintPolicy>, KvHintPolicyConstructorError> {
        let parameters: Parameters = parameters.deserialize()?;
        if !parameters.enabled {
            return Err(KvHintPolicyConstructorError::new("policy must be enabled"));
        }
        Ok(Box::new(EmptyPolicy))
    }

    fn config(yaml: &str) -> (tempfile::NamedTempFile, KvRouterConfig) {
        let policy = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(policy.path(), yaml).unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(policy.path().display().to_string()),
            ..Default::default()
        };
        (policy, config)
    }

    const VALID_CONFIG: &str = r#"
kv_hint_policy:
  type: test
  parameters:
    enabled: true
"#;

    #[test]
    fn resolves_linked_policy_and_validates_parameters() {
        let mut registry = KvHintPolicyRegistry::default();
        registry.register("test", Arc::new(constructor)).unwrap();
        let (_policy, valid_config) = config(VALID_CONFIG);
        let configured = registry.resolve(&valid_config).unwrap().unwrap();
        let policy = configured
            .construct(
                &valid_config,
                WorkerType::Aggregated,
                RoutingPartitionRef::new("model", "default"),
            )
            .unwrap();
        let _: Box<dyn KvHintPolicy> = policy;

        let (_policy, invalid) = config(
            r#"
kv_hint_policy:
  type: test
  parameters:
    enabled: false
"#,
        );
        let configured = registry.resolve(&invalid).unwrap().unwrap();
        assert!(matches!(
            configured.construct(
                &invalid,
                WorkerType::Aggregated,
                RoutingPartitionRef::new("model", "default"),
            ),
            Err(KvHintPolicyRegistryError::Constructor { policy_type, .. })
                if policy_type == "test"
        ));
    }

    #[test]
    fn rejects_duplicate_reserved_and_unknown_types() {
        let constructor: KvHintPolicyConstructor = Arc::new(constructor);
        let mut registry = KvHintPolicyRegistry::default();
        assert!(matches!(
            registry.register("", constructor.clone()),
            Err(KvHintPolicyRegistryError::EmptyName)
        ));
        assert!(matches!(
            registry.register("default", constructor.clone()),
            Err(KvHintPolicyRegistryError::ReservedDefault)
        ));
        registry.register("test", constructor.clone()).unwrap();
        assert!(matches!(
            registry.register("test", constructor),
            Err(KvHintPolicyRegistryError::Duplicate { .. })
        ));

        let (_policy, config) = config(VALID_CONFIG);
        assert!(matches!(
            KvHintPolicyRegistry::default().resolve(&config),
            Err(KvHintPolicyRegistryError::UnknownType { name, .. }) if name == "test"
        ));
    }
}
