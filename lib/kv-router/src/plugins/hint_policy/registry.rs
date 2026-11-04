// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup registration and resolution for statically linked hint-policy plugins.

use std::collections::HashMap;
use std::sync::Arc;

use serde::de::DeserializeOwned;
use thiserror::Error;

use super::{HintPolicyInstance, HintRequirements};
use crate::config::KvRouterConfig;
use crate::{RoutingPartitionRef, WorkerType};

/// Creates one instance of a configured hint policy per routing partition.
pub type HintPolicyConstructor = Arc<
    dyn for<'a> Fn(&KvRouterConfig, WorkerType, RoutingPartitionRef<'a>) -> HintPolicyInstance
        + Send
        + Sync,
>;

/// A configured hint policy: the requirements every instance declares, and its constructor.
///
/// The host unions requirements across the configured policies before it starts indexes and
/// subscriptions, independently of the worker-selection policy, and rejects an instance whose
/// requirements differ from the declared ones.
#[derive(Clone)]
pub struct HintPolicyDefinition {
    requirements: HintRequirements,
    constructor: HintPolicyConstructor,
}

impl HintPolicyDefinition {
    pub fn new(requirements: HintRequirements, constructor: HintPolicyConstructor) -> Self {
        Self {
            requirements,
            constructor,
        }
    }

    pub fn requirements(&self) -> HintRequirements {
        self.requirements
    }
}

/// Parses one instance's YAML parameters and returns its definition.
pub type HintPolicyProvider = Arc<
    dyn Fn(&HintPolicyParameters) -> Result<HintPolicyDefinition, HintPolicyProviderError>
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

    /// Deserialize this instance's parameter mapping into a policy-owned type.
    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T, HintPolicyProviderError> {
        serde_yaml::from_value(self.0.clone())
            .map_err(|source| HintPolicyProviderError::new(source.to_string()))
    }
}

/// A policy-owned validation failure while creating a hint-policy definition.
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

/// An error from hint-policy registration, resolution, or instance construction.
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
    DuplicateInstance { name: String },
    #[error("unknown hint-policy type {name:?}; linked hint-policy types: {available}")]
    UnknownType { name: String, available: String },
    #[error("invalid parameters for hint-policy type {policy_type:?}: {source}")]
    Provider {
        policy_type: String,
        #[source]
        source: HintPolicyProviderError,
    },
    #[error(
        "hint policy {name:?} constructed an instance whose requirements {actual:?} differ from its declared {declared:?}"
    )]
    RequirementsChanged {
        name: String,
        declared: HintRequirements,
        actual: HintRequirements,
    },
}

#[derive(Clone)]
struct ResolvedHintPolicy {
    name: Arc<str>,
    definition: HintPolicyDefinition,
}

/// The resolved, ordered hint policies for every worker role.
///
/// Cloning shares the resolved definitions. Each call to [`Self::instances`] constructs fresh
/// instances for one partition, so mutable policy state is never shared across partitions.
#[derive(Clone)]
pub struct HintPolicyFactory {
    roles: Arc<[Vec<ResolvedHintPolicy>; 4]>,
}

impl HintPolicyFactory {
    /// No hint policies for any role.
    pub fn none() -> Self {
        Self {
            roles: Arc::new(Default::default()),
        }
    }

    /// The same ordered policies for every worker role, for hosts that build hint policies in
    /// code rather than from router configuration.
    pub fn for_all_roles(
        policies: impl IntoIterator<Item = (String, HintPolicyDefinition)>,
    ) -> Result<Self, HintPolicyRegistryError> {
        let policies: Vec<_> = policies
            .into_iter()
            .map(|(name, definition)| ResolvedHintPolicy {
                name: Arc::from(name),
                definition,
            })
            .collect();
        for (index, policy) in policies.iter().enumerate() {
            if policies[..index]
                .iter()
                .any(|earlier| earlier.name == policy.name)
            {
                return Err(HintPolicyRegistryError::DuplicateInstance {
                    name: policy.name.to_string(),
                });
            }
        }
        Ok(Self {
            roles: Arc::new([
                policies.clone(),
                policies.clone(),
                policies.clone(),
                policies,
            ]),
        })
    }

    /// The union of the requirements of the policies a worker role runs. Hosts start the
    /// resources these require, such as the KV index and its event subscription, independently
    /// of the worker-selection policy.
    pub fn requirements(&self, worker_type: WorkerType) -> HintRequirements {
        self.role(worker_type)
            .iter()
            .fold(HintRequirements::new(), |requirements, policy| {
                requirements.union(policy.definition.requirements)
            })
    }

    /// The configured policy names for a worker role, in order.
    pub fn names(&self, worker_type: WorkerType) -> Vec<&str> {
        self.role(worker_type)
            .iter()
            .map(|policy| policy.name.as_ref())
            .collect()
    }

    /// Construct one partition's instances in configured order.
    pub fn instances(
        &self,
        config: &KvRouterConfig,
        worker_type: WorkerType,
        partition: RoutingPartitionRef<'_>,
    ) -> Result<Vec<HintPolicyInstance>, HintPolicyRegistryError> {
        self.role(worker_type)
            .iter()
            .map(|policy| {
                let instance = (policy.definition.constructor)(config, worker_type, partition);
                if instance.requirements() != policy.definition.requirements {
                    return Err(HintPolicyRegistryError::RequirementsChanged {
                        name: policy.name.to_string(),
                        declared: policy.definition.requirements,
                        actual: instance.requirements(),
                    });
                }
                Ok(instance)
            })
            .collect()
    }

    fn role(&self, worker_type: WorkerType) -> &[ResolvedHintPolicy] {
        &self.roles[role_index(worker_type)]
    }
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

/// Linked hint-policy providers.
#[derive(Clone, Default)]
pub(crate) struct HintPolicyRegistry {
    providers: HashMap<String, HintPolicyProvider>,
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

    /// Resolve the configured hint policies once at startup. Hint policies are opt-in: a role
    /// without a configured list runs none.
    pub(crate) fn resolve(
        &self,
        config: &KvRouterConfig,
    ) -> Result<HintPolicyFactory, HintPolicyRegistryError> {
        let Some(hint_config) = config
            .hint_policy_config()
            .map_err(|source| HintPolicyRegistryError::Config { source })?
        else {
            return Ok(HintPolicyFactory::none());
        };
        let mut resolved: HashMap<&str, ResolvedHintPolicy> = HashMap::new();
        let mut roles: [Vec<ResolvedHintPolicy>; 4] = Default::default();
        for worker_type in WORKER_TYPES {
            let names = hint_config.role(worker_type);
            let mut policies = Vec::with_capacity(names.len());
            for name in names.iter().map(String::as_str) {
                if let Some(policy) = resolved.get(name) {
                    policies.push(policy.clone());
                    continue;
                }
                let instance = hint_config
                    .instance(name)
                    .expect("hint_policies role entries name configured instances");
                let policy = ResolvedHintPolicy {
                    name: Arc::from(name),
                    definition: self.definition(
                        instance.policy_type(),
                        HintPolicyParameters::new(instance.parameters().clone()),
                    )?,
                };
                resolved.insert(name, policy.clone());
                policies.push(policy);
            }
            roles[role_index(worker_type)] = policies;
        }
        // Validate instances no role lists too, so a typo in one fails at startup.
        let mut unlisted: Vec<_> = hint_config
            .instance_names()
            .filter(|name| !resolved.contains_key(name))
            .collect();
        unlisted.sort_unstable();
        for name in unlisted {
            let instance = hint_config
                .instance(name)
                .expect("instance names come from the config");
            self.definition(
                instance.policy_type(),
                HintPolicyParameters::new(instance.parameters().clone()),
            )?;
        }
        Ok(HintPolicyFactory {
            roles: Arc::new(roles),
        })
    }

    fn definition(
        &self,
        policy_type: &str,
        parameters: HintPolicyParameters,
    ) -> Result<HintPolicyDefinition, HintPolicyRegistryError> {
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
