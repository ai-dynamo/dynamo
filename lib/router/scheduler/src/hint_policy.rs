// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental hint-policy plugins: engine-directed KV actions for a booked worker.
//!
//! A [`HintPolicy`] turns the selection's commitments and request signals into actions in the
//! engine's hint format. The host calls [`HintPolicy::build_hints`] once per booked selection,
//! after booking and outside the scheduler actor, with the booked worker. Worker choice stays
//! with the configured worker-selection policy; hint policies never change it.
//!
//! The router composes one [`dynamo_kv_hints::KvHint`] per booking: its own fetch, if any, then
//! the policy that declares [`Commitments::KV_COPY`], then every other policy in configured
//! order. It numbers the actions and never reads their payloads. Only workers that advertise
//! `router_hint` receive hints. Hosts own eligibility, booking, and delivery. Policies never
//! reserve capacity, and the engine may still ignore or miss any action.
//!
//! This module is experimental: names and signatures can change without a deprecation period.

mod commitments;
pub(crate) mod runtime;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use crate::identity::RoutingPartitionRef;
use crate::kv_hints::{KvCopyLocations, KvCopySource};
use crate::types::SessionContext;
use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_router_protocol::worker_type::WorkerType;
use dynamo_worker_selection::{SelectionConfig, WorkerInputs};

pub use crate::resolve::{PluginConfigError, check_role};
pub use commitments::Commitments;
pub use dynamo_kv_hints::KvHintAction;
use runtime::BookedCopy;

/// A statically linked hint policy. One instance serves one routing partition.
///
/// The router reads [`Self::required_worker_inputs`] and [`Self::commitments`] once, when it
/// builds the router for a role. Implementing the trait is the hook: the router calls
/// [`Self::build_hints`] for every booking. Callbacks are synchronous, run on shared router
/// tasks, and must be bounded: no I/O, awaits, or unbounded index traversal.
pub trait HintPolicy: Send + Sync + 'static {
    /// Router data this policy reads, from the documented [`WorkerInputs`] list. Reading
    /// [`WorkerInputs::KV_COPY`] provides [`HintContext::kv_copy_source`]. The router rejects
    /// a policy that declares any other input, which [`HintContext`] can't serve.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// Commitments this policy turns into hints. Today the only kind is a KV copy. At most one
    /// policy per role may declare [`Commitments::KV_COPY`].
    fn commitments(&self) -> Commitments {
        Commitments::NONE
    }

    /// Hints for the booked worker, called once per booking and never for previews.
    ///
    /// An error drops only this policy's actions, and the booking proceeds with the other
    /// policies' actions. The exception is a committed copy: when
    /// [`HintContext::committed_kv_copy`] is set, returning an error or no action fails the
    /// booking, because the selection's cost assumed that copy.
    fn build_hints(&self, context: &HintContext<'_>) -> Result<Vec<KvHintAction>, HintPolicyError>;
}

/// Values available to [`HintPolicy::build_hints`] for one booking.
///
/// The request values cost the router nothing, so every policy receives them. The copy
/// accessors return `None` unless the policy declared what they need.
pub struct HintContext<'a> {
    pub(crate) request_id: &'a str,
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) isl_tokens: usize,
    pub(crate) block_size: u32,
    pub(crate) session_context: Option<&'a SessionContext>,
    pub(crate) committed_kv_copy: Option<&'a BookedCopy<'a>>,
    pub(crate) kv_copy_source: Option<&'a BookedCopy<'a>>,
}

impl<'a> HintContext<'a> {
    /// The routing request ID, which the composed hint also uses as its message ID.
    pub fn request_id(&self) -> &'a str {
        self.request_id
    }

    /// The booked worker and rank every action targets.
    pub fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    /// Exact incoming prompt length in tokens.
    pub fn isl_tokens(&self) -> usize {
        self.isl_tokens
    }

    /// Tokens per KV block in this partition.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// The request's agent session, with its harness headers. `None` for a request without a
    /// recognized agent session.
    pub fn session_context(&self) -> Option<&'a SessionContext> {
        self.session_context
    }

    /// The KV copy the worker-selection policy's cost counted for the booked worker. Set only for
    /// a policy that declares [`Commitments::KV_COPY`], when the selection committed a copy. The
    /// policy must send it: see [`HintPolicy::build_hints`].
    pub fn committed_kv_copy(&self) -> Option<&'a KvCopySource> {
        self.committed_kv_copy.map(BookedCopy::copy)
    }

    /// The booked worker's best copy source, when the selection committed none. Set only for a
    /// policy that reads [`WorkerInputs::KV_COPY`], when the partition's workers advertise
    /// `router_hint` and the KV index keeps the request's block chain.
    pub fn kv_copy_source(&self) -> Option<&'a KvCopySource> {
        self.kv_copy_source.map(BookedCopy::copy)
    }

    /// Where the blocks of `copy`, from [`Self::committed_kv_copy`] or [`Self::kv_copy_source`],
    /// are: the source's current address, the request's block chain, and both prefixes. The view
    /// is borrowed and the policy picks the range its action sends. `None` when the source no
    /// longer advertises a control endpoint, or for any other copy.
    pub fn kv_copy_locations(&self, copy: &KvCopySource) -> Option<&'a KvCopyLocations<'a>> {
        [self.committed_kv_copy, self.kv_copy_source]
            .into_iter()
            .flatten()
            .find(|booked| booked.copy() == copy)
            .and_then(BookedCopy::locations)
    }
}

/// A policy failure reported to the router. The message is included in logs and errors.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct HintPolicyError {
    message: String,
}

impl HintPolicyError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

type Construct = dyn for<'a> Fn(&SelectionConfig, WorkerType, RoutingPartitionRef<'a>) -> Arc<dyn HintPolicy>
    + Send
    + Sync;

/// Creates the instance of one configured hint policy that each routing partition runs.
#[derive(Clone)]
pub struct HintPolicyConstructor(Arc<Construct>);

impl HintPolicyConstructor {
    /// Construct a fresh instance per partition, for a policy that keeps per-partition state.
    pub fn new<F>(construct: F) -> Self
    where
        F: for<'a> Fn(&SelectionConfig, WorkerType, RoutingPartitionRef<'a>) -> Arc<dyn HintPolicy>
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
        config: &SelectionConfig,
        worker_type: WorkerType,
        partition: RoutingPartitionRef<'_>,
    ) -> Arc<dyn HintPolicy> {
        (self.0)(config, worker_type, partition)
    }
}

/// A hint-policy name listed more than once for a role.
#[derive(Debug, thiserror::Error)]
#[error("hint policy {name:?} is listed more than once")]
pub struct DuplicateHintPolicyName {
    pub name: String,
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
    ) -> Result<Self, DuplicateHintPolicyName> {
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
                return Err(DuplicateHintPolicyName {
                    name: policy.name.to_string(),
                });
            }
        }
        let role = ResolvedRole {
            policies,
            defaults: false,
        };
        Ok(Self {
            roles: Arc::new([role.clone(), role.clone(), role.clone(), role]),
        })
    }

    /// Each listed role's ordered policies, and whether they are the registry's defaults. A role
    /// not listed runs none.
    pub fn from_roles(
        roles: impl IntoIterator<Item = (WorkerType, Vec<(Arc<str>, HintPolicyConstructor)>, bool)>,
    ) -> Self {
        let mut resolved: [ResolvedRole; 4] = Default::default();
        for (worker_type, policies, defaults) in roles {
            resolved[role_index(worker_type)] = ResolvedRole {
                policies: policies
                    .into_iter()
                    .map(|(name, constructor)| ResolvedHintPolicy { name, constructor })
                    .collect(),
                defaults,
            };
        }
        Self {
            roles: Arc::new(resolved),
        }
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
        config: &SelectionConfig,
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

fn role_index(worker_type: WorkerType) -> usize {
    match worker_type {
        WorkerType::Aggregated => 0,
        WorkerType::Prefill => 1,
        WorkerType::Decode => 2,
        WorkerType::Encode => 3,
    }
}
