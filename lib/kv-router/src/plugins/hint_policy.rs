// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental hint-policy plugins: engine-directed KV actions for a booked worker.
//!
//! A [`HintPolicy`] maps request signals and serving state to engine-directed KV actions. The host
//! calls [`HintPolicy::hint_kv_locality`] once per booked selection, after booking and outside the
//! scheduler actor, with the booked worker. Worker choice stays with the configured
//! worker-selection policy; hint policies never change it.
//!
//! The router composes its own fetch, if any, and then every policy's actions in configured order
//! into one [`crate::kv_hints::KvHint`]. Hosts own eligibility, booking, and delivery. Policies
//! never reserve capacity, and the engine may still ignore or miss any action.
//!
//! This module is experimental: names and signatures can change without a deprecation period.

mod config;
mod registry;
#[cfg_attr(not(feature = "standalone-selection"), allow(dead_code))]
pub(crate) mod runtime;
#[cfg(test)]
mod tests;

use std::fmt;
use std::sync::Arc;

use crate::kv_hints::KvHintAction;
use crate::plugins::worker_selection::WorkerInputs;
use crate::protocols::WorkerWithDpRank;

pub(crate) use config::RawHintPolicyConfig;
pub use config::{HintPolicyConfig, HintPolicyInstanceConfig};
pub(crate) use registry::HintPolicyRegistry;
pub use registry::{
    HintPolicyConstructor, HintPolicyDefinition, HintPolicyFactory, HintPolicyParameters,
    HintPolicyProvider, HintPolicyProviderError, HintPolicyRegistryError,
};

/// A statically linked hint policy. One instance serves one routing partition.
///
/// [`Self::requirements`] declares whether the host calls the policy and which resources it
/// starts. Callbacks are synchronous, run on shared router tasks, and must be bounded: no I/O,
/// awaits, or unbounded index traversal.
pub trait HintPolicy: Send + Sync + 'static {
    /// The stage and inputs this policy uses. Read once when the instance is constructed.
    fn requirements(&self) -> HintRequirements;

    /// Return KV actions for the booked worker.
    ///
    /// Called once per booked selection, never for previews. An error drops only this policy's
    /// actions; the booking proceeds with the other policies' actions.
    fn hint_kv_locality(
        &self,
        _context: &HintActionContext<'_>,
    ) -> Result<Vec<KvHintAction>, HintPolicyError> {
        Ok(Vec::new())
    }
}

/// Whether the host calls a hint policy and the worker inputs whose resources it starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HintRequirements {
    inputs: WorkerInputs,
    actions: bool,
}

impl Default for HintRequirements {
    fn default() -> Self {
        Self::new()
    }
}

impl HintRequirements {
    /// No stage and no inputs: a policy that the host never calls.
    pub const fn new() -> Self {
        Self {
            inputs: WorkerInputs::NONE,
            actions: false,
        }
    }

    /// Start the resources these worker input groups need, such as the KV index for
    /// [`WorkerInputs::CACHE`], even when the worker-selection policy reads none.
    pub fn with_inputs(mut self, inputs: WorkerInputs) -> Self {
        self.inputs = self.inputs | inputs;
        self
    }

    /// Call [`HintPolicy::hint_kv_locality`] for booked selections.
    pub fn with_actions(mut self) -> Self {
        self.actions = true;
        self
    }

    pub fn uses_actions(&self) -> bool {
        self.actions
    }

    /// Requirements a host must satisfy to run every policy in a list.
    pub(crate) fn union(self, other: Self) -> Self {
        Self {
            inputs: self.inputs | other.inputs,
            actions: self.actions || other.actions,
        }
    }

    /// Worker inputs a host must provide for these requirements. Hosts union this with the
    /// worker-selection policy's inputs before starting indexes and subscriptions.
    pub fn worker_inputs(&self) -> WorkerInputs {
        self.inputs
    }
}

/// Values available to [`HintPolicy::hint_kv_locality`] for one booked selection.
pub struct HintActionContext<'a> {
    request_id: &'a str,
    worker: WorkerWithDpRank,
    prompt_tokens: usize,
    block_size: u32,
}

impl HintActionContext<'_> {
    /// The routing request ID, which the composed hint also uses as its message ID.
    pub fn request_id(&self) -> &str {
        self.request_id
    }

    /// The booked worker and rank every action targets.
    pub fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    /// Exact incoming prompt length in tokens.
    pub fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// Tokens per KV block in this partition.
    pub fn block_size(&self) -> u32 {
        self.block_size
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

/// One named policy instance for one routing partition, created by a provider's factory.
pub struct HintPolicyInstance {
    name: Arc<str>,
    requirements: HintRequirements,
    #[cfg_attr(not(feature = "standalone-selection"), allow(dead_code))]
    policy: Arc<dyn HintPolicy>,
}

impl HintPolicyInstance {
    pub fn new<P: HintPolicy>(name: impl Into<Arc<str>>, policy: P) -> Self {
        Self {
            name: name.into(),
            requirements: policy.requirements(),
            policy: Arc::new(policy),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn requirements(&self) -> HintRequirements {
        self.requirements
    }

    /// The name, and the policy when its KV-action stage runs.
    #[cfg_attr(not(feature = "standalone-selection"), allow(dead_code))]
    pub(crate) fn into_parts(self) -> (Arc<str>, Option<Arc<dyn HintPolicy>>) {
        let policy = self.requirements.actions.then_some(self.policy);
        (self.name, policy)
    }
}

impl fmt::Debug for HintPolicyInstance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HintPolicyInstance")
            .field("name", &self.name)
            .field("requirements", &self.requirements)
            .finish_non_exhaustive()
    }
}
