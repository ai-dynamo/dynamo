// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public filter, scorer, and picker contracts and their input signals.

mod config;
mod context;
pub(crate) mod inputs;
mod runtime;

pub(crate) use runtime::RuntimePolicyState;
pub use runtime::{
    RuntimePolicyBuilder, RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyUpdate,
};

pub use context::WorkerSelectionContext;
pub(crate) use inputs::{CacheSnapshot, CandidateData, WorkerCacheData, WorkerInputRows};
pub use inputs::{
    ScoredWorkerCandidate, WorkerCacheInput, WorkerCacheInputs, WorkerCandidate, WorkerCandidates,
    WorkerCapacity, WorkerInfo, WorkerInputView, WorkerInputs, WorkerLoadInput,
    WorkerTransferInput,
};

pub use crate::kv_hints::KvFetchSource;

pub use super::registry::{
    WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};
pub use crate::scheduling::selector::WorkerSelectionPolicy;
pub use crate::scheduling::{
    SessionContext, WorkerSelectionInputTrigger, WorkerSelectionPolicyError,
};
pub(crate) use config::RawWorkerSelectionConfig;
pub use config::{WorkerSelectionConfig, WorkerSelectionInstance};

use crate::{KvRouterConfig, RoutingPartitionRef, WorkerType};
use std::sync::Arc;

/// Factory that creates one worker-selection policy per routing partition.
pub type WorkerSelectionPolicyFactory = Arc<
    dyn for<'a> Fn(&KvRouterConfig, WorkerType, RoutingPartitionRef<'a>) -> WorkerSelectionPolicy
        + Send
        + Sync,
>;

/// Adds one finite cost contribution to each eligible worker.
pub trait WorkerScorer: Send {
    /// Declare the worker-signal groups needed by this scorer.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// Score every candidate that survived host eligibility and policy filters in one call.
    /// The borrowed snapshot and its row order remain fixed for this selection. Each scorer
    /// can access only its own declared optional inputs, even when other components request more. Empty candidate
    /// sets skip scoring. Scorers run in declaration order; an error stops before picking.
    ///
    /// `costs` has the same length as `candidates` and is initialized to NaN before each call.
    /// Write one finite, lower-is-better contribution to every slot. The host validates and adds
    /// each contribution to that row's total. The candidate view and cost slice may not be retained after this call.
    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError>;
}

/// Scores like a [`WorkerScorer`] and commits each candidate to the KV fetch its cost assumes.
///
/// Use it when a cost credits blocks a worker would fetch from another worker or cache owner.
/// Only a `KvFetchScorer` receives [`WorkerInputs::KV_TRANSFER`]: the host strips it from
/// filters, scorers, and pickers, so a policy that credits a fetch always commits to one. A
/// policy has at most one; it runs after the [`WorkerScorer`]s, and its costs add to theirs.
pub trait KvFetchScorer: Send {
    /// Declare the worker-signal groups needed by this scorer. The host adds
    /// [`WorkerInputs::KV_TRANSFER`].
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// Score every candidate and choose the fetch each cost assumes.
    ///
    /// `costs` follows [`WorkerScorer::score`]: it starts as NaN, and every slot needs one finite,
    /// lower-is-better contribution. `fetches` starts with the source the host offers each
    /// candidate, the one behind [`WorkerTransferInput::source_prefix_blocks`], or `None` when it
    /// offers none. Keep a slot when that candidate's cost assumes the fetch, and set it to
    /// `None` otherwise; any other value fails selection. Only the picked candidate's fetch leaves
    /// selection, and the router delivers it with the booking.
    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
        fetches: &mut [Option<KvFetchSource>],
    ) -> Result<(), WorkerSelectionPolicyError>;
}

/// Filters run in declaration order for each candidate. Callback order across different
/// candidates and scorers is unspecified; implementations must not depend on filters and scorers
/// being interleaved.
pub trait WorkerFilter: Send {
    /// Declare the worker-signal groups needed by this filter.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// Return `true` to keep an eligible worker in the policy candidate set.
    fn keep(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidate: WorkerCandidate<'_>,
    ) -> Result<bool, WorkerSelectionPolicyError>;
}

/// Selects one row after all filters and scorers run.
pub trait WorkerPicker: Send {
    /// Declare the optional worker-signal columns needed by this picker.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// Return one row index from the host-owned eligible candidate table. Row order is
    /// unspecified; inspect candidate data instead of relying on a stable position.
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError>;
}
