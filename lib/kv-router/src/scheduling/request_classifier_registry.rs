// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup registration for statically linked request-classifier plugins.

use std::collections::HashMap;
use std::sync::Arc;

use serde::de::DeserializeOwned;
use thiserror::Error;

use super::{RequestClassifierFactory, config::KvRouterConfig};

pub type RequestClassifierProvider = Arc<
    dyn Fn(
            &RequestClassifierParameters,
        ) -> Result<RequestClassifierFactory, RequestClassifierProviderError>
        + Send
        + Sync,
>;

#[derive(Debug, Clone)]
pub struct RequestClassifierParameters(serde_yaml::Value);

impl RequestClassifierParameters {
    fn new(value: serde_yaml::Value) -> Self {
        Self(value)
    }

    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T, RequestClassifierProviderError> {
        serde_yaml::from_value(self.0.clone())
            .map_err(|source| RequestClassifierProviderError::new(source.to_string()))
    }
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct RequestClassifierProviderError {
    message: String,
}

impl RequestClassifierProviderError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Clone, Default)]
pub struct RequestClassifierRegistry {
    providers: HashMap<String, RequestClassifierProvider>,
}

#[derive(Debug, Error)]
pub enum RequestClassifierRegistryError {
    #[error("request-classifier type must not be empty")]
    EmptyName,
    #[error("request-classifier type 'default' is reserved for Dynamo's pass-through behavior")]
    ReservedDefault,
    #[error("request-classifier type {name:?} is already registered")]
    Duplicate { name: String },
    #[error("could not load request_classifier from router_policy_config: {source}")]
    Config {
        #[source]
        source: super::RouterPolicyConfigError,
    },
    #[error("unknown request-classifier type {name:?}; linked classifier types: {available}")]
    UnknownType { name: String, available: String },
    #[error("invalid parameters for request-classifier type {classifier_type:?}: {source}")]
    Provider {
        classifier_type: String,
        #[source]
        source: RequestClassifierProviderError,
    },
}

impl RequestClassifierRegistry {
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    pub fn register(
        &mut self,
        name: impl Into<String>,
        provider: RequestClassifierProvider,
    ) -> Result<(), RequestClassifierRegistryError> {
        let name = name.into();
        if name.is_empty() {
            return Err(RequestClassifierRegistryError::EmptyName);
        }
        if name == "default" {
            return Err(RequestClassifierRegistryError::ReservedDefault);
        }
        if self.providers.contains_key(&name) {
            return Err(RequestClassifierRegistryError::Duplicate { name });
        }
        self.providers.insert(name, provider);
        Ok(())
    }

    pub fn resolve(
        &self,
        config: &KvRouterConfig,
    ) -> Result<Option<RequestClassifierFactory>, RequestClassifierRegistryError> {
        let Some(classifier) = config
            .request_classifier_config()
            .map_err(|source| RequestClassifierRegistryError::Config { source })?
        else {
            return Ok(None);
        };
        let classifier_type = classifier.classifier_type();
        let provider = self.providers.get(classifier_type).ok_or_else(|| {
            RequestClassifierRegistryError::UnknownType {
                name: classifier_type.to_owned(),
                available: self.available_classifier_types(),
            }
        })?;
        provider(&RequestClassifierParameters::new(
            classifier.parameters().clone(),
        ))
        .map(Some)
        .map_err(|source| RequestClassifierRegistryError::Provider {
            classifier_type: classifier_type.to_owned(),
            source,
        })
    }

    fn available_classifier_types(&self) -> String {
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
    use crate::scheduling::RequestClassifier;

    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Parameters {
        limit: usize,
    }

    struct PassThrough;

    impl RequestClassifier for PassThrough {}

    fn provider(
        parameters: &RequestClassifierParameters,
    ) -> Result<RequestClassifierFactory, RequestClassifierProviderError> {
        let parameters: Parameters = parameters.deserialize()?;
        if parameters.limit == 0 {
            return Err(RequestClassifierProviderError::new(
                "limit must be greater than zero",
            ));
        }
        Ok(Arc::new(|_| Box::new(PassThrough)))
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
request_classifier:
  type: test
  parameters:
    limit: 4
"#;

    #[test]
    fn resolves_linked_classifier_and_validates_parameters() {
        let mut registry = RequestClassifierRegistry::default();
        registry.register("test", Arc::new(provider)).unwrap();
        let (_policy, valid_config) = config(VALID_CONFIG);
        let factory = registry.resolve(&valid_config).unwrap().unwrap();
        let classifier = factory(super::super::RequestClassifierContext::new(16, Vec::new));
        let _: Box<dyn RequestClassifier> = classifier;

        let (_policy, invalid) = config(
            r#"
request_classifier:
  type: test
  parameters:
    limit: 0
"#,
        );
        assert!(matches!(
            registry.resolve(&invalid),
            Err(RequestClassifierRegistryError::Provider {
                classifier_type,
                ..
            }) if classifier_type == "test"
        ));
    }

    #[test]
    fn rejects_duplicate_reserved_and_unknown_types() {
        let provider: RequestClassifierProvider = Arc::new(provider);
        let mut registry = RequestClassifierRegistry::default();
        assert!(matches!(
            registry.register("", provider.clone()),
            Err(RequestClassifierRegistryError::EmptyName)
        ));
        assert!(matches!(
            registry.register("default", provider.clone()),
            Err(RequestClassifierRegistryError::ReservedDefault)
        ));
        registry.register("test", provider.clone()).unwrap();
        assert!(matches!(
            registry.register("test", provider),
            Err(RequestClassifierRegistryError::Duplicate { .. })
        ));

        let (_policy, config) = config(VALID_CONFIG);
        let empty = RequestClassifierRegistry::default();
        assert!(matches!(
            empty.resolve(&config),
            Err(RequestClassifierRegistryError::UnknownType { name, .. }) if name == "test"
        ));
    }
}
