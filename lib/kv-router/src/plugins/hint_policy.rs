// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental hint-policy plugins: engine-directed KV actions for a booked worker.
//!
//! A [`HintPolicy`] turns the selection's commitments and request signals into actions in the
//! engine's hint format. The host calls [`HintPolicy::build_hints`] once per booked selection,
//! after booking and outside the scheduler actor, with the booked worker. Worker choice stays
//! with the configured worker-selection policy; hint policies never change it.
//!
//! The router composes one [`crate::kv_hints::KvHint`] per booking: its own fetch, if any, then
//! the policy that declares [`Commitments::KV_COPY`], then every other policy in configured
//! order. It numbers the actions and never reads their payloads. Only workers that advertise
//! `router_hint` receive hints. Hosts own eligibility, booking, and delivery. Policies never
//! reserve capacity, and the engine may still ignore or miss any action.
//!
//! This module is experimental: names and signatures can change without a deprecation period.

mod config;
mod registry;
#[cfg_attr(not(feature = "standalone-selection"), allow(dead_code))]
pub(crate) mod runtime;
#[cfg(test)]
mod tests;

use crate::plugins::worker_selection::{Commitments, KvCopyLocations, KvCopySource, WorkerInputs};
use crate::protocols::WorkerWithDpRank;
use crate::scheduling::SessionContext;

pub use crate::kv_hints::KvHintAction;
pub use config::HintPolicyEntry;
pub(crate) use config::{RawHintPolicyEntry, resolve_hints};
pub(crate) use registry::HintPolicyRegistry;
pub use registry::{
    HintPolicyConstructor, HintPolicyFactory, HintPolicyParameters, HintPolicyProvider,
    HintPolicyProviderError, HintPolicyRegistryError, NamedHintPolicy, RoleHintPolicies,
};
use runtime::BookedCopy;

/// A statically linked hint policy. One instance serves one routing partition.
///
/// The router reads [`Self::required_worker_inputs`] and [`Self::commitments`] once, when it
/// builds the router for a role. Implementing the trait is the hook: the router calls
/// [`Self::build_hints`] for every booking. Callbacks are synchronous, run on shared router
/// tasks, and must be bounded: no I/O, awaits, or unbounded index traversal.
pub trait HintPolicy: Send + Sync + 'static {
    /// Router data this policy reads, from the documented [`WorkerInputs`] list. Reading
    /// [`WorkerInputs::KV_TRANSFER`] provides [`HintContext::kv_copy_source`].
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
    /// policy that reads [`WorkerInputs::KV_TRANSFER`], when the partition's workers advertise
    /// `router_hint` and the KV index keeps the request's block chain.
    pub fn kv_copy_source(&self) -> Option<&'a KvCopySource> {
        self.kv_copy_source.map(BookedCopy::copy)
    }

    /// Where the blocks of `copy`, from [`Self::committed_kv_copy`] or [`Self::kv_copy_source`],
    /// are: the source's current address, the request's block chain, and both prefixes. The view
    /// is borrowed and the policy picks the range its action sends. `None` when the source no
    /// longer advertises a control endpoint, or for any other copy.
    pub fn kv_copy_locations(&self, copy: &KvCopySource) -> Option<KvCopyLocations<'a>> {
        [self.committed_kv_copy, self.kv_copy_source]
            .into_iter()
            .flatten()
            .find(|booked| booked.copy() == copy)
            .and_then(BookedCopy::locations)
    }
}

/// A plugin a hint runs under. A hint names its hosts when it registers.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HintHost {
    /// Every worker-selection policy, including the built-in selector (`default`).
    AnyWorkerSelection,
    /// One worker-selection type.
    WorkerSelection(String),
    /// One request-classifier type. Checked at startup; matches no role yet.
    RequestClassifier(String),
}

impl HintHost {
    /// The worker-selection type `policy_type`; `default` is the built-in selector.
    pub fn worker_selection(policy_type: impl Into<String>) -> Self {
        Self::WorkerSelection(policy_type.into())
    }

    /// The request-classifier type `classifier_type`.
    pub fn request_classifier(classifier_type: impl Into<String>) -> Self {
        Self::RequestClassifier(classifier_type.into())
    }
}

/// Whether a host runs the hint when the role's entry has no `hints`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HintDefault {
    On,
    Off,
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
