// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup registration and resolution for statically linked hint-policy plugins.

use std::sync::Arc;

use serde::de::DeserializeOwned;
use thiserror::Error;

use super::{HintHost, HintPolicyConstructor, HintPolicyEntry, HintPolicyFactory};
use crate::WorkerType;
use crate::config::KvRouterConfig;
use crate::plugins::worker_selection::{
    WorkerSelectionConfig, WorkerSelectionInstance, WorkerSelectionPolicySelection,
};

/// Parses one entry's YAML parameters and returns its constructor.
pub type HintPolicyProvider = Arc<
    dyn Fn(&HintPolicyParameters) -> Result<HintPolicyConstructor, HintPolicyProviderError>
        + Send
        + Sync,
>;

/// YAML parameters passed to a hint-policy provider.
#[derive(Debug, Clone)]
pub struct HintPolicyParameters(serde_yaml::Value);

impl HintPolicyParameters {
    pub(super) fn new(value: serde_yaml::Value) -> Self {
        Self(value)
    }

    /// Deserialize this entry's parameter mapping into a policy-owned type.
    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T, HintPolicyProviderError> {
        serde_yaml::from_value(self.0.clone())
            .map_err(|source| HintPolicyProviderError::new(source.to_string()))
    }
}

/// A policy-owned validation failure while creating a hint-policy constructor.
#[derive(Debug, Error)]
#[error("{message}")]
pub struct HintPolicyProviderError {
    message: String,
}

impl HintPolicyProviderError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// An error from hint-policy registration or resolution.
#[derive(Debug, Error)]
pub enum HintPolicyRegistryError {
    #[error("hint-policy type must not be empty")]
    EmptyName,
    #[error("hint-policy type {name:?} is already registered")]
    Duplicate { name: String },
    #[error("could not load worker_selection hints: {source}")]
    Config {
        #[source]
        source: crate::scheduling::config::WorkerSelectionPolicyConfigError,
    },
    #[error("unknown hint-policy type {name:?}; linked hint-policy types: {available}")]
    UnknownType { name: String, available: String },
    #[error("invalid parameters for hint-policy type {policy_type:?}: {source}")]
    Provider {
        policy_type: String,
        #[source]
        source: HintPolicyProviderError,
    },
    #[error("hint-policy type {name:?} lists host {host:?} more than once")]
    DuplicateHost { name: String, host: HintHost },
    #[error("hint-policy type {name:?} names host {host:?}, which no linked plugin registers")]
    UnknownHost { name: String, host: HintHost },
}

const WORKER_TYPES: [WorkerType; 4] = [
    WorkerType::Aggregated,
    WorkerType::Prefill,
    WorkerType::Decode,
    WorkerType::Encode,
];

/// One linked hint-policy type: its provider, and the plugins it runs under.
#[derive(Clone)]
struct RegisteredHintPolicy {
    name: String,
    provider: HintPolicyProvider,
    hosts: Vec<HintHost>,
}

impl RegisteredHintPolicy {
    /// Whether a role under the worker-selection type `host` runs this hint when its entry lists
    /// no `hints`.
    fn on_by_default_under(&self, host: &str) -> bool {
        self.hosts.iter().any(|candidate| match candidate {
            HintHost::AnyWorkerSelection => true,
            HintHost::WorkerSelection(policy_type) => policy_type == host,
        })
    }

    fn construct(
        &self,
        parameters: serde_yaml::Value,
    ) -> Result<(Arc<str>, HintPolicyConstructor), HintPolicyRegistryError> {
        let constructor =
            (self.provider)(&HintPolicyParameters::new(parameters)).map_err(|source| {
                HintPolicyRegistryError::Provider {
                    policy_type: self.name.clone(),
                    source,
                }
            })?;
        Ok((Arc::from(self.name.as_str()), constructor))
    }
}

/// Linked hint-policy types, in registration order, with the plugins each runs under by default.
#[derive(Clone, Default)]
pub(crate) struct HintPolicyRegistry {
    policies: Vec<RegisteredHintPolicy>,
}

impl HintPolicyRegistry {
    pub(crate) fn is_empty(&self) -> bool {
        self.policies.is_empty()
    }

    pub(crate) fn register(
        &mut self,
        name: impl Into<String>,
        provider: HintPolicyProvider,
        hosts: impl IntoIterator<Item = HintHost>,
    ) -> Result<(), HintPolicyRegistryError> {
        let name = name.into();
        if name.is_empty() {
            return Err(HintPolicyRegistryError::EmptyName);
        }
        if self.get(&name).is_some() {
            return Err(HintPolicyRegistryError::Duplicate { name });
        }
        let mut registered: Vec<HintHost> = Vec::new();
        for host in hosts {
            if registered.contains(&host) {
                return Err(HintPolicyRegistryError::DuplicateHost { name, host });
            }
            registered.push(host);
        }
        self.policies.push(RegisteredHintPolicy {
            name,
            provider,
            hosts: registered,
        });
        Ok(())
    }

    /// Reject a host that no linked plugin registers, as `is_linked` decides.
    pub(crate) fn check_hosts(
        &self,
        is_linked: impl Fn(&HintHost) -> bool,
    ) -> Result<(), HintPolicyRegistryError> {
        for policy in &self.policies {
            if let Some(host) = policy.hosts.iter().find(|host| !is_linked(host)) {
                return Err(HintPolicyRegistryError::UnknownHost {
                    name: policy.name.clone(),
                    host: host.clone(),
                });
            }
        }
        Ok(())
    }

    /// Resolve each role's hints once at startup, from the `worker_selection` entry the role runs.
    ///
    /// A role runs the hints its entry lists, of any registered type. A role whose entry lists no
    /// `hints` runs the hints whose hosts include its worker-selection type, in registration order,
    /// each built from empty parameters.
    pub(crate) fn resolve(
        &self,
        config: &KvRouterConfig,
    ) -> Result<HintPolicyFactory, HintPolicyRegistryError> {
        let selected = config
            .selected_worker_selection_policy_instances()
            .map_err(|source| HintPolicyRegistryError::Config { source })?;
        let worker_selection =
            config
                .worker_selection_config()
                .map_err(|source| HintPolicyRegistryError::Config {
                    source: crate::scheduling::config::WorkerSelectionPolicyConfigError::Config {
                        source,
                    },
                })?;
        let mut roles = Vec::with_capacity(WORKER_TYPES.len());
        for worker_type in WORKER_TYPES {
            let selection = match worker_type {
                WorkerType::Aggregated => &selected.aggregated,
                WorkerType::Prefill => &selected.prefill,
                WorkerType::Decode => &selected.decode,
                WorkerType::Encode => &selected.encode,
            };
            let host = host_type(worker_selection, selection.as_ref());
            roles.push(
                match selection
                    .as_ref()
                    .and_then(WorkerSelectionPolicySelection::hints)
                {
                    Some(entries) => (
                        worker_type,
                        entries
                            .iter()
                            .map(|entry| self.listed(entry))
                            .collect::<Result<_, _>>()?,
                        false,
                    ),
                    None => (
                        worker_type,
                        self.policies
                            .iter()
                            .filter(|policy| policy.on_by_default_under(host))
                            .map(|policy| {
                                policy.construct(serde_yaml::Value::Mapping(Default::default()))
                            })
                            .collect::<Result<_, _>>()?,
                        true,
                    ),
                },
            );
        }
        Ok(HintPolicyFactory::from_roles(roles))
    }

    /// Resolve one hint a role lists: any registered type.
    fn listed(
        &self,
        entry: &HintPolicyEntry,
    ) -> Result<(Arc<str>, HintPolicyConstructor), HintPolicyRegistryError> {
        let policy =
            self.get(entry.policy_type())
                .ok_or_else(|| HintPolicyRegistryError::UnknownType {
                    name: entry.policy_type().to_owned(),
                    available: self.available_policy_types(),
                })?;
        let (_, constructor) = policy.construct(entry.parameters().clone())?;
        Ok((Arc::from(entry.name()), constructor))
    }

    fn get(&self, policy_type: &str) -> Option<&RegisteredHintPolicy> {
        self.policies
            .iter()
            .find(|policy| policy.name == policy_type)
    }

    fn available_policy_types(&self) -> String {
        let mut available = self
            .policies
            .iter()
            .map(|policy| policy.name.as_str())
            .collect::<Vec<_>>();
        available.sort_unstable();
        available.join(", ")
    }
}

/// The worker-selection type a role's hints run under: `default`, the built-in selector, when
/// the role selects no policy, and a deprecated instance's type for that instance's name.
fn host_type<'a>(
    config: Option<&'a WorkerSelectionConfig>,
    selection: Option<&'a WorkerSelectionPolicySelection>,
) -> &'a str {
    match selection {
        None => "default",
        Some(WorkerSelectionPolicySelection::Named(name)) => config
            .and_then(|config| config.instance(name))
            .map_or(name.as_str(), WorkerSelectionInstance::policy_type),
        Some(WorkerSelectionPolicySelection::Inline(instance)) => instance.policy_type(),
    }
}
