// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup registration and resolution for statically linked hint-policy plugins.

use std::collections::HashMap;
use std::sync::Arc;

use serde::de::DeserializeOwned;
use thiserror::Error;

use super::HintPolicy;
use crate::config::KvRouterConfig;
use crate::{RoutingPartitionRef, WorkerType};

type Construct = dyn for<'a> Fn(&KvRouterConfig, WorkerType, RoutingPartitionRef<'a>) -> Arc<dyn HintPolicy>
    + Send
    + Sync;

/// Creates the instance of one configured hint policy that each routing partition runs.
#[derive(Clone)]
pub struct HintPolicyConstructor(Arc<Construct>);

impl HintPolicyConstructor {
    /// Construct a fresh instance per partition, for a policy that keeps per-partition state.
    pub fn new<F>(construct: F) -> Self
    where
        F: for<'a> Fn(&KvRouterConfig, WorkerType, RoutingPartitionRef<'a>) -> Arc<dyn HintPolicy>
            + Send
            + Sync
            + 'static,
    {
        Self(Arc::new(construct))
    }

    /// Share one instance across every partition, for a stateless policy.
    pub fn shared<P: HintPolicy>(policy: P) -> Self {
        let policy: Arc<dyn HintPolicy> = Arc::new(policy);
        Self::new(move |_, _, _| Arc::clone(&policy))
    }

    fn construct(
        &self,
        config: &KvRouterConfig,
        worker_type: WorkerType,
        partition: RoutingPartitionRef<'_>,
    ) -> Arc<dyn HintPolicy> {
        (self.0)(config, worker_type, partition)
    }
}

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
    #[error("could not load hint_policies from router_policy_config: {source}")]
    Config {
        #[source]
        source: crate::scheduling::RouterPolicyConfigError,
    },
    #[error("hint policy {name:?} is listed more than once")]
    DuplicateName { name: String },
    #[error("unknown hint-policy type {name:?}; linked hint-policy types: {available}")]
    UnknownType { name: String, available: String },
    #[error("invalid parameters for hint-policy type {policy_type:?}: {source}")]
    Provider {
        policy_type: String,
        #[source]
        source: HintPolicyProviderError,
    },
}

#[derive(Clone)]
struct ResolvedHintPolicy {
    name: Arc<str>,
    constructor: HintPolicyConstructor,
}

/// One role's resolved hint policies.
#[derive(Clone, Default)]
struct ResolvedRole {
    policies: Vec<ResolvedHintPolicy>,
    /// `hint_policies` doesn't list the role, so it runs the registry's default hint policies.
    defaults: bool,
}

/// The resolved, ordered hint policies for every worker role.
///
/// Cloning shares the resolved constructors. Each call to [`Self::instances`] constructs one
/// partition's instances, so a policy that keeps per-partition state never shares it.
#[derive(Clone)]
pub struct HintPolicyFactory {
    roles: Arc<[ResolvedRole; 4]>,
}

impl HintPolicyFactory {
    /// No hint policies for any role.
    pub fn none() -> Self {
        Self {
            roles: Arc::new(Default::default()),
        }
    }

    /// The same ordered policies for every worker role, for hosts that build hint policies in
    /// code rather than from router configuration. They count as configured, not as defaults.
    pub fn for_all_roles(
        policies: impl IntoIterator<Item = (String, HintPolicyConstructor)>,
    ) -> Result<Self, HintPolicyRegistryError> {
        let policies = named(policies)?;
        let role = ResolvedRole {
            policies,
            defaults: false,
        };
        Ok(Self {
            roles: Arc::new([role.clone(), role.clone(), role.clone(), role]),
        })
    }

    /// The configured policy names for a worker role, in order.
    pub fn names(&self, worker_type: WorkerType) -> Vec<&str> {
        self.role(worker_type)
            .policies
            .iter()
            .map(|policy| policy.name.as_ref())
            .collect()
    }

    /// Whether a worker role runs the registry's default hint policies, because `hint_policies`
    /// doesn't list it.
    pub fn uses_defaults(&self, worker_type: WorkerType) -> bool {
        self.role(worker_type).defaults
    }

    /// Construct one partition's instances in configured order.
    pub fn instances(
        &self,
        config: &KvRouterConfig,
        worker_type: WorkerType,
        partition: RoutingPartitionRef<'_>,
    ) -> RoleHintPolicies {
        let role = self.role(worker_type);
        RoleHintPolicies {
            policies: role
                .policies
                .iter()
                .map(|policy| {
                    (
                        Arc::clone(&policy.name),
                        policy.constructor.construct(config, worker_type, partition),
                    )
                })
                .collect(),
            defaults: role.defaults,
        }
    }

    fn role(&self, worker_type: WorkerType) -> &ResolvedRole {
        &self.roles[role_index(worker_type)]
    }
}

/// A hint-policy instance and its name in logs.
pub type NamedHintPolicy = (Arc<str>, Arc<dyn HintPolicy>);

/// One role's hint-policy instances for one routing partition, in configured order.
pub struct RoleHintPolicies {
    policies: Vec<NamedHintPolicy>,
    defaults: bool,
}

impl RoleHintPolicies {
    /// No hint policies, as configured rather than defaulted.
    pub fn none() -> Self {
        Self {
            policies: Vec::new(),
            defaults: false,
        }
    }

    /// The named instances, in configured order.
    pub fn policies(&self) -> &[NamedHintPolicy] {
        &self.policies
    }

    /// Whether these are the registry's default hint policies, which are passive: they start no
    /// resources and don't require the KV path.
    pub fn are_defaults(&self) -> bool {
        self.defaults
    }

    pub(crate) fn into_parts(self) -> (Vec<NamedHintPolicy>, bool) {
        (self.policies, self.defaults)
    }
}

fn named(
    policies: impl IntoIterator<Item = (String, HintPolicyConstructor)>,
) -> Result<Vec<ResolvedHintPolicy>, HintPolicyRegistryError> {
    let policies: Vec<_> = policies
        .into_iter()
        .map(|(name, constructor)| ResolvedHintPolicy {
            name: Arc::from(name),
            constructor,
        })
        .collect();
    for (index, policy) in policies.iter().enumerate() {
        if policies[..index]
            .iter()
            .any(|earlier| earlier.name == policy.name)
        {
            return Err(HintPolicyRegistryError::DuplicateName {
                name: policy.name.to_string(),
            });
        }
    }
    Ok(policies)
}

fn role_index(worker_type: WorkerType) -> usize {
    match worker_type {
        WorkerType::Aggregated => 0,
        WorkerType::Prefill => 1,
        WorkerType::Decode => 2,
        WorkerType::Encode => 3,
    }
}

const WORKER_TYPES: [WorkerType; 4] = [
    WorkerType::Aggregated,
    WorkerType::Prefill,
    WorkerType::Decode,
    WorkerType::Encode,
];

/// Linked hint-policy providers, and the default hint policies a role runs when `hint_policies`
/// doesn't list it.
#[derive(Clone, Default)]
pub(crate) struct HintPolicyRegistry {
    providers: HashMap<String, HintPolicyProvider>,
    defaults: Vec<ResolvedHintPolicy>,
}

impl HintPolicyRegistry {
    pub(crate) fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    pub(crate) fn register(
        &mut self,
        name: impl Into<String>,
        provider: HintPolicyProvider,
    ) -> Result<(), HintPolicyRegistryError> {
        let name = name.into();
        if name.is_empty() {
            return Err(HintPolicyRegistryError::EmptyName);
        }
        if self.providers.contains_key(&name) {
            return Err(HintPolicyRegistryError::Duplicate { name });
        }
        self.providers.insert(name, provider);
        Ok(())
    }

    /// Replace the default hint policies.
    pub(crate) fn set_defaults(
        &mut self,
        policies: impl IntoIterator<Item = (String, HintPolicyConstructor)>,
    ) -> Result<(), HintPolicyRegistryError> {
        self.defaults = named(policies)?;
        Ok(())
    }

    /// Resolve the configured hint policies once at startup. A role that `hint_policies` doesn't
    /// list runs the default hint policies.
    pub(crate) fn resolve(
        &self,
        config: &KvRouterConfig,
    ) -> Result<HintPolicyFactory, HintPolicyRegistryError> {
        let hint_config = config
            .hint_policy_config()
            .map_err(|source| HintPolicyRegistryError::Config { source })?;
        let mut roles: [ResolvedRole; 4] = Default::default();
        for worker_type in WORKER_TYPES {
            let Some(entries) = hint_config.and_then(|config| config.role(worker_type)) else {
                roles[role_index(worker_type)] = ResolvedRole {
                    policies: self.defaults.clone(),
                    defaults: true,
                };
                continue;
            };
            let policies = entries
                .iter()
                .map(|entry| {
                    Ok(ResolvedHintPolicy {
                        name: Arc::from(entry.name()),
                        constructor: self.constructor(
                            entry.policy_type(),
                            HintPolicyParameters::new(entry.parameters().clone()),
                        )?,
                    })
                })
                .collect::<Result<_, HintPolicyRegistryError>>()?;
            roles[role_index(worker_type)] = ResolvedRole {
                policies,
                defaults: false,
            };
        }
        Ok(HintPolicyFactory {
            roles: Arc::new(roles),
        })
    }

    fn constructor(
        &self,
        policy_type: &str,
        parameters: HintPolicyParameters,
    ) -> Result<HintPolicyConstructor, HintPolicyRegistryError> {
        let provider = self.providers.get(policy_type).ok_or_else(|| {
            HintPolicyRegistryError::UnknownType {
                name: policy_type.to_owned(),
                available: self.available_policy_types(),
            }
        })?;
        provider(&parameters).map_err(|source| HintPolicyRegistryError::Provider {
            policy_type: policy_type.to_owned(),
            source,
        })
    }

    fn available_policy_types(&self) -> String {
        let mut available = self.providers.keys().cloned().collect::<Vec<_>>();
        available.sort_unstable();
        available.join(", ")
    }
}
