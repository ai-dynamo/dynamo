// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What an in-tree host names from the worker-selection stage beyond the plugin contract it
//! reaches through `dynamo_kv_router::plugins::v1alpha1`: the configuration slices, the
//! conditional-disaggregation decision inputs, the LoRA filter, the trace types it renders, and
//! the host half of the plugin contract types: how the kernel builds the request view and the
//! candidate rows a policy reads, and how it runs a policy. Those are extension traits and
//! functions here, not methods on the contract types, so a plugin, which does not name this
//! module, cannot call them. This list is the host contract.

use dynamo_router_protocol::session::SessionContext;
use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_session_affinity::WorkerAffinityTarget;
use dynamo_slot_tracker::WorkerLoadProjection;

use crate::context::WorkerSelectionContext;
use crate::inputs::{ResidentBlockSource, WorkerCopyInput, WorkerLoadInput};
use crate::policy::{
    CandidatePreparer, CandidateRowSource, PreparedRows, RowPick, SelectionFailure,
};

pub use crate::conditional_disagg::{
    ConditionalDisaggConfig, ConditionalDisaggDecisionInput, ConditionalDisaggPolicyKind,
};
pub use crate::error::WorkerSelectionPolicyError;
pub use crate::hosted::{HostedSelectionRequest, HostedWorkerPicker};
pub use crate::lora_filter::LoraWorkerFilter;
pub use crate::policy::WorkerSelectionPolicy;
pub use crate::runtime::RuntimePolicyError;
pub use crate::selection_config::{SelectionConfig, SharedCacheType};
pub use crate::shared_cache::SharedCacheHits;
pub use crate::trace::{SelectionTrace, SelectionTraceSpec};

// What the frontend host (lib/llm) names beyond the lists above.
pub use crate::conditional_disagg::{ConditionalDisaggPolicy, make_conditional_disagg_policy};

/// A live lookup lent by the scheduler for one selection. Pickers reach it through
/// `experimental::decode_progress`; the returned summary belongs to this frontend.
pub type DecodeProgressSource<'a> = dyn Fn(
        dynamo_router_protocol::worker::WorkerWithDpRank,
    ) -> Option<dynamo_slot_tracker::host::DecodeProgress>
    + 'a;

/// How the kernel builds the request view a policy reads. A plugin only reads a
/// [`WorkerSelectionContext`]; its tests get one through `dynamo_kv_router::plugins::testing`.
pub trait WorkerSelectionContextHostExt<'a>: Sized {
    /// A view of a request of `isl_tokens` routed in blocks of `block_size`, with every other
    /// field at its default. A `block_size` of 0, as hosted selection without the KV path
    /// supplies, reads as zero request blocks.
    fn new(isl_tokens: usize, block_size: u32) -> Self;

    /// Whether this selection only previews a worker, as a peek does.
    fn with_preview(self, preview: bool) -> Self;

    fn with_request_id(self, request_id: Option<&'a str>) -> Self;

    fn with_prefill_tracking(self, track_prefill_tokens: bool) -> Self;

    /// The exact worker and rank the host imposes: an explicit pin or an eligible
    /// exclusive-affinity target.
    fn with_pinned_worker(self, pinned_worker: Option<WorkerWithDpRank>) -> Self;

    fn with_router_temperature_override(self, temperature: Option<f64>) -> Self;

    fn with_session_context(self, session_context: Option<&'a SessionContext>) -> Self;

    fn with_affinity_target(self, affinity_target: Option<WorkerAffinityTarget>) -> Self;

    fn with_expected_output_tokens(self, expected_output_tokens: Option<u32>) -> Self;

    fn with_priority(self, priority_jump: f64, strict_priority: u32) -> Self;

    fn with_shared_cache_hits(self, shared_cache_hits: Option<&'a SharedCacheHits>) -> Self;

    /// Whether any rank reported per-tier overlap, so cache columns read device overlap
    /// from the tier blocks rather than the effective estimate.
    fn with_tier_matches(self, has_tier_matches: bool) -> Self;

    /// Per-rank resident block counts the host lends to this selection, read only by
    /// components that declared `WorkerInputs::RESIDENT_BLOCKS`.
    fn with_resident_blocks(self, resident_blocks: Option<&'a dyn ResidentBlockSource>) -> Self;

    /// Lend a live worker lookup to pickers declaring the experimental decode-progress input.
    /// The lookup is called only when the picker reads a candidate's progress.
    fn with_decode_progress(self, progress: Option<&'a DecodeProgressSource<'a>>) -> Self;
}

impl<'a> WorkerSelectionContextHostExt<'a> for WorkerSelectionContext<'a> {
    #[inline]
    fn new(isl_tokens: usize, block_size: u32) -> Self {
        WorkerSelectionContext::new(isl_tokens, block_size)
    }

    #[inline]
    fn with_preview(self, preview: bool) -> Self {
        WorkerSelectionContext::with_preview(self, preview)
    }

    #[inline]
    fn with_request_id(self, request_id: Option<&'a str>) -> Self {
        WorkerSelectionContext::with_request_id(self, request_id)
    }

    #[inline]
    fn with_prefill_tracking(self, track_prefill_tokens: bool) -> Self {
        WorkerSelectionContext::with_prefill_tracking(self, track_prefill_tokens)
    }

    #[inline]
    fn with_pinned_worker(self, pinned_worker: Option<WorkerWithDpRank>) -> Self {
        WorkerSelectionContext::with_pinned_worker(self, pinned_worker)
    }

    #[inline]
    fn with_router_temperature_override(self, temperature: Option<f64>) -> Self {
        WorkerSelectionContext::with_router_temperature_override(self, temperature)
    }

    #[inline]
    fn with_session_context(self, session_context: Option<&'a SessionContext>) -> Self {
        WorkerSelectionContext::with_session_context(self, session_context)
    }

    #[inline]
    fn with_affinity_target(self, affinity_target: Option<WorkerAffinityTarget>) -> Self {
        WorkerSelectionContext::with_affinity_target(self, affinity_target)
    }

    #[inline]
    fn with_expected_output_tokens(self, expected_output_tokens: Option<u32>) -> Self {
        WorkerSelectionContext::with_expected_output_tokens(self, expected_output_tokens)
    }

    #[inline]
    fn with_priority(self, priority_jump: f64, strict_priority: u32) -> Self {
        WorkerSelectionContext::with_priority(self, priority_jump, strict_priority)
    }

    #[inline]
    fn with_shared_cache_hits(self, shared_cache_hits: Option<&'a SharedCacheHits>) -> Self {
        WorkerSelectionContext::with_shared_cache_hits(self, shared_cache_hits)
    }

    #[inline]
    fn with_tier_matches(self, has_tier_matches: bool) -> Self {
        WorkerSelectionContext::with_tier_matches(self, has_tier_matches)
    }

    #[inline]
    fn with_resident_blocks(self, resident_blocks: Option<&'a dyn ResidentBlockSource>) -> Self {
        WorkerSelectionContext::with_resident_blocks(self, resident_blocks)
    }

    #[inline]
    fn with_decode_progress(self, progress: Option<&'a DecodeProgressSource<'a>>) -> Self {
        WorkerSelectionContext::with_decode_progress(self, progress)
    }
}

/// How the kernel runs a [`WorkerSelectionPolicy`] a plugin built: selection over candidate
/// rows, row preparation, and the hosted picker.
pub trait WorkerSelectionPolicyHostExt: Sized {
    /// A preparer for this policy's rows, when its rows depend only on the request and
    /// worker configuration until load inputs are filled. Filters are policy callbacks, so a
    /// policy with filters builds its rows at selection.
    fn candidate_preparer(&self) -> Option<CandidatePreparer>;

    /// Whether `prepared` rows are usable by this policy: built by one that would build the
    /// same rows, with the same inputs and without filters, which run as policy callbacks.
    fn accepts_prepared(&self, prepared: &PreparedRows) -> bool;

    /// Filter, score, and pick over the rows `source` yields.
    ///
    /// `prepared` rows, which must satisfy [`Self::accepts_prepared`], stand in for the
    /// source's rows. Their load column is filled from `source` here, on the actor, so it
    /// reflects bookings since preparation; after [`PreparedRows::keep_filled_loads`]
    /// only the ranks it named are read from `source`. When `trace` is set, the pick
    /// carries the retained per-candidate costs.
    fn select_rows<S: CandidateRowSource>(
        &self,
        context: &WorkerSelectionContext<'_>,
        source: &S,
        prepared: Option<PreparedRows>,
        trace: Option<SelectionTraceSpec>,
    ) -> Result<RowPick, SelectionFailure>;

    /// Convert a policy that needs no KV-path input into the picker a host runs without the
    /// KV path, or return the policy unchanged when it needs that path.
    fn into_hosted_picker(self) -> Result<HostedWorkerPicker, Self>;
}

impl WorkerSelectionPolicyHostExt for WorkerSelectionPolicy {
    #[inline]
    fn candidate_preparer(&self) -> Option<CandidatePreparer> {
        WorkerSelectionPolicy::candidate_preparer(self)
    }

    #[inline]
    fn accepts_prepared(&self, prepared: &PreparedRows) -> bool {
        WorkerSelectionPolicy::accepts_prepared(self, prepared)
    }

    #[inline]
    fn select_rows<S: CandidateRowSource>(
        &self,
        context: &WorkerSelectionContext<'_>,
        source: &S,
        prepared: Option<PreparedRows>,
        trace: Option<SelectionTraceSpec>,
    ) -> Result<RowPick, SelectionFailure> {
        WorkerSelectionPolicy::select_rows(self, context, source, prepared, trace)
    }

    #[inline]
    fn into_hosted_picker(self) -> Result<HostedWorkerPicker, Self> {
        WorkerSelectionPolicy::into_hosted_picker(self)
    }
}

/// The `LOAD` row for one rank from its load projection. `observed` is false for a rank
/// the host has no projection for, which a policy reads through
/// [`WorkerLoadInput::is_available`].
#[inline]
pub fn worker_load_input(load: WorkerLoadProjection, observed: bool) -> WorkerLoadInput {
    WorkerLoadInput {
        available: observed,
        active_prefill_tokens: load.active_prefill_tokens,
        active_decode_blocks: load.active_decode_blocks,
        decode_cost_blocks: load.potential_decode_blocks() as f64,
        active_requests: load.active_requests,
    }
}

/// The `KV_COPY` offer for one rank: the prefix, in blocks, the copy source chosen for it
/// holds, or `None` when no source was chosen.
#[inline]
pub fn worker_copy_input(source_prefix_blocks: Option<u32>) -> WorkerCopyInput {
    WorkerCopyInput {
        source_prefix_blocks,
    }
}
