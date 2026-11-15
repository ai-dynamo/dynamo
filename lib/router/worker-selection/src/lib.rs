// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Candidate rows in, a pick out: the Dynamo router's worker selection stage.
//!
//! The kernel walks its eligible ranks as a [`CandidateRowSource`]; a
//! [`WorkerSelectionPolicy`] pulls each rank's row at the columns its filters, scorers,
//! and picker declared, and returns a [`RowPick`] or a [`SelectionFailure`]. Everything a
//! plugin names lives here: the [`WorkerFilter`], [`WorkerScorer`], and [`WorkerPicker`]
//! traits, the [`WorkerSelectionContext`] view of the request, the candidate rows and
//! their views, runtime parameter updates, the selection slice of the router config, the
//! conditional-disaggregation policies, and the LoRA narrowing filter. The kv-transfer source
//! summary a policy credits fetches with lives in `dynamo-kv-hints`; the kernel builds the
//! [`WorkerCopyInput`] column from it. The kernel's `WorkerSelector`, its request-to-row
//! materialization, and the config- and projection-derived columns stay in `lib/router/kv-router`.

pub mod conditional_disagg;
mod context;
mod error;
pub mod experimental;
pub mod host;
mod hosted;
mod inputs;
pub mod lora_filter;
mod policy;
#[cfg(any(test, feature = "testing", feature = "bench"))]
mod reference_picker;
mod runtime;
/// The selection slice of the router config, with the defaults and validators the flat
/// router config in `lib/router/kv-router` shares.
pub mod selection_config;
mod shared_cache;
mod trace;

pub use context::WorkerSelectionContext;
pub use error::WorkerSelectionPolicyError;
pub use hosted::{HostedSelectionRequest, HostedWorkerPicker};
#[allow(deprecated)]
pub use inputs::WorkerTransferInput;
pub use inputs::{
    CacheSnapshot, CandidateData, ResidentBlockSource, ScoredWorkerCandidate, WorkerCacheData,
    WorkerCacheInput, WorkerCacheInputs, WorkerCandidate, WorkerCandidates, WorkerCapacity,
    WorkerCopyInput, WorkerInfo, WorkerInputRows, WorkerInputView, WorkerInputs, WorkerLoadInput,
};
pub use policy::{
    CandidatePreparer, CandidateRank, CandidateRowSource, PreparedRows, RankVisitor, RowPick,
    SelectionFailure, WorkerSelectionPolicy,
};
#[cfg(any(test, feature = "testing", feature = "bench"))]
pub use reference_picker::{DefaultPick, DefaultWorkerPicker};
pub(crate) use runtime::RuntimePolicyState;
pub use runtime::{
    RuntimePolicyBuilder, RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyUpdate,
};
pub use selection_config::{SelectionConfig, SharedCacheType};
pub use shared_cache::SharedCacheHits;
pub use trace::{SelectionTrace, SelectionTraceRow, SelectionTraceSpec};

pub use dynamo_router_protocol::session::SessionContext;

/// Adds one finite cost contribution to each eligible worker.
///
/// One instance may be called from several threads at once, and successive calls may run on
/// different threads. Keep any state that changes between calls thread-safe.
pub trait WorkerScorer: Send + Sync {
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
        &self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError>;
}

/// Filters run in declaration order for each candidate. Callback order across different
/// candidates and scorers is unspecified; implementations must not depend on filters and scorers
/// being interleaved.
///
/// One instance may be called from several threads at once, and successive calls may run on
/// different threads. Keep any state that changes between calls thread-safe.
pub trait WorkerFilter: Send + Sync {
    /// Declare the worker-signal groups needed by this filter.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// Return `true` to keep an eligible worker in the policy candidate set.
    fn keep(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidate: WorkerCandidate<'_>,
    ) -> Result<bool, WorkerSelectionPolicyError>;
}

/// Selects one row after all filters and scorers run.
///
/// One instance may be called from several threads at once, and successive calls may run on
/// different threads. Keep any state that changes between calls thread-safe.
pub trait WorkerPicker: Send + Sync {
    /// Declare the optional worker-signal columns needed by this picker.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// Return one row index from the host-owned eligible candidate table. Row order is
    /// unspecified unless [`WorkerInputView`] says otherwise; inspect candidate data instead
    /// of relying on a stable position.
    fn pick(
        &self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError>;
}
