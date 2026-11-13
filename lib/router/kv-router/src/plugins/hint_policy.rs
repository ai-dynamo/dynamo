// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental hint-policy plugins: engine-directed KV actions for a booked worker.
//!
//! The contract and the stage that runs it live in `dynamo-scheduler`; this module re-exports
//! them and adds what only the composed router knows: registration, the hosts a hint runs under
//! by default, and the `hints` entries of the YAML policy file.
//!
//! This module is experimental: names and signatures can change without a deprecation period.

mod config;
mod registry;
#[cfg(test)]
mod tests;

pub use config::HintPolicyEntry;
pub(crate) use config::{RawHintPolicyEntry, resolve_hints};
pub(crate) use registry::HintPolicyRegistry;
pub use registry::{
    HintPolicyParameters, HintPolicyProvider, HintPolicyProviderError, HintPolicyRegistryError,
};

pub use dynamo_scheduler::hint_policy::{
    Commitments, DuplicateHintPolicyName, HintContext, HintPolicy, HintPolicyConstructor,
    HintPolicyError, HintPolicyFactory, KvHintAction, NamedHintPolicy, RoleHintPolicies,
};

/// A plugin under which a hint runs by default, when the role's entry lists no `hints`. A hint
/// names these hosts when it registers. Any role can still list any hint.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HintHost {
    /// Every worker-selection policy, including the built-in selector (`default`).
    AnyWorkerSelection,
    /// One worker-selection type.
    WorkerSelection(String),
}

impl HintHost {
    /// The worker-selection type `policy_type`; `default` is the built-in selector.
    pub fn worker_selection(policy_type: impl Into<String>) -> Self {
        Self::WorkerSelection(policy_type.into())
    }
}
