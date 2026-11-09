// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup registration and resolution for statically linked hint-policy plugins.

use std::sync::Arc;

use serde::de::DeserializeOwned;
use thiserror::Error;

use super::{HintDefault, HintHost, HintPolicy, HintPolicyEntry};
use crate::config::KvRouterConfig;
use crate::plugins::worker_selection::{
    WorkerSelectionConfig, WorkerSelectionInstance, WorkerSelectionPolicySelection,
};
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
    #[error("could not load worker_selection hints: {source}")]
    Config {
        #[source]
        source: crate::scheduling::config::WorkerSelectionPolicyConfigError,
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
    #[error("hint-policy type {name:?} registers no hosts")]
    NoHosts { name: String },
    #[error("hint-policy type {name:?} lists host {host:?} more than once")]
    DuplicateHost { name: String, host: HintHost },
    #[error("hint-policy type {name:?} names host {host:?}, which no linked plugin registers")]
    UnknownHost { name: String, host: HintHost },
    #[error(
        "hint-policy type {hint:?} doesn't run under worker-selection type {host:?} ({role}); hints for that type: {supported}"
    )]
    Unsupported {
        role: &'static str,
        host: String,
        hint: String,
        supported: String,
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
    /// The role's `worker_selection` entry lists no `hints`, so it runs the default hint policies.
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

    /// Whether a worker role runs the default hint policies, because its `worker_selection` entry
    /// lists no `hints`.
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

/// One linked hint-policy type: its provider, and the plugins it runs under.
#[derive(Clone)]
struct RegisteredHintPolicy {
    name: String,
    provider: HintPolicyProvider,
    hosts: Vec<(HintHost, HintDefault)>,
}

impl RegisteredHintPolicy {
    /// Whether this hint runs under the worker-selection type `host`, and if so, whether it runs
    /// there by default. A host that names `host` overrides [`HintHost::AnyWorkerSelection`].
    fn runs_under(&self, host: &str) -> Option<HintDefault> {
        let mut any = None;
        for (candidate, default) in &self.hosts {
            match candidate {
                HintHost::WorkerSelection(policy_type) if policy_type == host => {
                    return Some(*default);
                }
                HintHost::AnyWorkerSelection => any = Some(*default),
                _ => {}
            }
        }
        any
    }

    fn construct(
        &self,
        parameters: serde_yaml::Value,
    ) -> Result<ResolvedHintPolicy, HintPolicyRegistryError> {
        let constructor =
            (self.provider)(&HintPolicyParameters::new(parameters)).map_err(|source| {
                HintPolicyRegistryError::Provider {
                    policy_type: self.name.clone(),
                    source,
                }
            })?;
        Ok(ResolvedHintPolicy {
            name: Arc::from(self.name.as_str()),
            constructor,
        })
    }
}

/// Linked hint-policy types, in registration order, with the plugins each runs under.
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
        hosts: impl IntoIterator<Item = (HintHost, HintDefault)>,
    ) -> Result<(), HintPolicyRegistryError> {
        let name = name.into();
        if name.is_empty() {
            return Err(HintPolicyRegistryError::EmptyName);
        }
        if self.get(&name).is_some() {
            return Err(HintPolicyRegistryError::Duplicate { name });
        }
        let mut registered: Vec<(HintHost, HintDefault)> = Vec::new();
        for (host, default) in hosts {
            if registered.iter().any(|(known, _)| *known == host) {
                return Err(HintPolicyRegistryError::DuplicateHost { name, host });
            }
            registered.push((host, default));
        }
        if registered.is_empty() {
            return Err(HintPolicyRegistryError::NoHosts { name });
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
            if let Some((host, _)) = policy.hosts.iter().find(|(host, _)| !is_linked(host)) {
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
    /// Listed hints must run under the role's worker-selection type. A role whose entry lists no
    /// `hints` runs the hints registered [`HintDefault::On`] for that type, in registration order,
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
        let mut roles: [ResolvedRole; 4] = Default::default();
        for worker_type in WORKER_TYPES {
            let selection = match worker_type {
                WorkerType::Aggregated => &selected.aggregated,
                WorkerType::Prefill => &selected.prefill,
                WorkerType::Decode => &selected.decode,
                WorkerType::Encode => &selected.encode,
            };
            let host = host_type(worker_selection, selection.as_ref());
            roles[role_index(worker_type)] = match selection
                .as_ref()
                .and_then(WorkerSelectionPolicySelection::hints)
            {
                Some(entries) => ResolvedRole {
                    policies: entries
                        .iter()
                        .map(|entry| self.listed(worker_type, host, entry))
                        .collect::<Result<_, _>>()?,
                    defaults: false,
                },
                None => ResolvedRole {
                    policies: self
                        .policies
                        .iter()
                        .filter(|policy| policy.runs_under(host) == Some(HintDefault::On))
                        .map(|policy| {
                            policy.construct(serde_yaml::Value::Mapping(Default::default()))
                        })
                        .collect::<Result<_, _>>()?,
                    defaults: true,
                },
            };
        }
        Ok(HintPolicyFactory {
            roles: Arc::new(roles),
        })
    }

    /// Resolve one hint a role lists: a registered type that runs under the role's `host`.
    fn listed(
        &self,
        worker_type: WorkerType,
        host: &str,
        entry: &HintPolicyEntry,
    ) -> Result<ResolvedHintPolicy, HintPolicyRegistryError> {
        let policy =
            self.get(entry.policy_type())
                .ok_or_else(|| HintPolicyRegistryError::UnknownType {
                    name: entry.policy_type().to_owned(),
                    available: self.available_policy_types(),
                })?;
        if policy.runs_under(host).is_none() {
            return Err(HintPolicyRegistryError::Unsupported {
                role: worker_type.as_str(),
                host: host.to_owned(),
                hint: entry.policy_type().to_owned(),
                supported: self.policy_types_under(host),
            });
        }
        let mut resolved = policy.construct(entry.parameters().clone())?;
        resolved.name = Arc::from(entry.name());
        Ok(resolved)
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

    /// The hint types that run under the worker-selection type `host`.
    fn policy_types_under(&self, host: &str) -> String {
        let mut supported = self
            .policies
            .iter()
            .filter(|policy| policy.runs_under(host).is_some())
            .map(|policy| policy.name.as_str())
            .collect::<Vec<_>>();
        supported.sort_unstable();
        if supported.is_empty() {
            "<none>".to_owned()
        } else {
            supported.join(", ")
        }
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
