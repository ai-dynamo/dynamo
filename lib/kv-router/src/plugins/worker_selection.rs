// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public filter, scorer, and picker contracts and their input signals.

mod config;
mod context;
mod error;
pub(crate) mod inputs;
#[cfg(any(test, feature = "bench"))]
mod reference_picker;
mod runtime;
pub(crate) mod selection_config;
mod shared_cache;
mod trace;

pub(crate) use runtime::RuntimePolicyState;
pub use runtime::{
    RuntimePolicyBuilder, RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyUpdate,
};

pub use context::WorkerSelectionContext;
pub use error::WorkerSelectionPolicyError;
pub use inputs::{
    CacheSnapshot, CandidateData, ScoredWorkerCandidate, WorkerCacheData, WorkerCacheInput,
    WorkerCacheInputs, WorkerCandidate, WorkerCandidates, WorkerCapacity, WorkerInfo,
    WorkerInputRows, WorkerInputView, WorkerInputs, WorkerLoadInput, WorkerTransferInput,
};
#[cfg(any(test, feature = "bench"))]
pub use reference_picker::{DefaultPick, DefaultWorkerPicker};
pub use selection_config::{SelectionConfig, SharedCacheType};
pub use shared_cache::SharedCacheHits;
pub use trace::{SelectionTrace, SelectionTraceRow, SelectionTraceSpec};

pub use super::registry::{
    WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};
pub use crate::scheduling::selector::{
    CandidatePreparer, CandidateRank, CandidateRowSource, PreparedRows, RankVisitor, RowPick,
    SelectionFailure, WorkerCapacityExt, WorkerInfoExt, WorkerLoadInputExt, WorkerSelectionPolicy,
};
pub use crate::scheduling::{SessionContext, WorkerSelectionInputTrigger};
pub(crate) use config::RawWorkerSelectionConfig;
pub use config::{WorkerSelectionConfig, WorkerSelectionInstance};

use crate::{RoutingPartitionRef, WorkerType};
use std::sync::Arc;

/// Factory that creates one worker-selection policy per routing partition.
pub type WorkerSelectionPolicyFactory = Arc<
    dyn for<'a> Fn(&SelectionConfig, WorkerType, RoutingPartitionRef<'a>) -> WorkerSelectionPolicy
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
