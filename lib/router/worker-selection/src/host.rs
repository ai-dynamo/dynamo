// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What an in-tree host names from the worker-selection stage beyond the plugin contract it
//! reaches through `dynamo_kv_router::plugins::worker_selection`: the configuration slices, the
//! conditional-disaggregation decision inputs, the LoRA filter, the trace types it renders, and
//! the host half of the plugin contract types: how the kernel builds the request view, the
//! selection config, and the candidate rows a policy reads, and how it runs a policy. Those are extension traits and
//! functions here, not methods on the contract types, so a plugin, which does not name this
//! module, cannot call them. This list is the host contract.

use dynamo_router_protocol::session::SessionContext;
use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_session_affinity::WorkerAffinityTarget;
use dynamo_slot_tracker::WorkerLoadProjection;

use crate::inputs::{ResidentBlockSource, WorkerCopyInput, WorkerLoadInput};
use crate::policy::{
    CandidatePreparer, CandidateRowSource, PreparedRows, RowPick, SelectionFailure,
};
use crate::request::{RequestInfo, WorkerSelectionRequest};

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

mod sealed {
    /// Keeps the host traits implementable only here, so a method can be added to one.
    pub trait Sealed {}
}

impl sealed::Sealed for WorkerSelectionRequest<'_> {}
impl sealed::Sealed for WorkerSelectionPolicy {}
impl sealed::Sealed for SelectionConfig {}

/// How the kernel builds the request view a policy reads. A plugin only reads a
/// [`WorkerSelectionRequest`]; its tests get one through `dynamo_kv_router::plugins::testing`.
pub trait WorkerSelectionRequestHostExt<'a>: sealed::Sealed + Sized {
    /// A view of the request `info` describes, routed in blocks of `block_size`, with every
    /// other field at its default. A `block_size` of 0, as hosted selection without the KV
    /// path supplies, reads as zero request blocks.
    fn new(info: RequestInfo<'a>, block_size: u32) -> Self;

    /// Whether this selection only previews a worker, as a peek does.
    fn with_preview(self, preview: bool) -> Self;

    fn with_prefill_tracking(self, track_prefill_tokens: bool) -> Self;

    fn with_router_temperature_override(self, temperature: Option<f64>) -> Self;

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

/// How the kernel runs a [`WorkerSelectionPolicy`] a plugin built: selection over candidate
/// rows, row preparation, and the hosted picker.
pub trait WorkerSelectionPolicyHostExt: sealed::Sealed + Sized {
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
        request: &WorkerSelectionRequest<'_>,
        source: &S,
        prepared: Option<PreparedRows>,
        trace: Option<SelectionTraceSpec>,
    ) -> Result<RowPick, SelectionFailure>;

    /// Take out this policy's picker for hosted selection, when the policy can run there:
    /// it has no filters, no scorers and no runtime config, and its picker declares at
    /// most [`WorkerInputs::ACTIVE_REQUESTS`](crate::WorkerInputs::ACTIVE_REQUESTS). Any
    /// other policy comes back unchanged.
    ///
    /// Passing this check doesn't make a policy run hosted: for now the hosts call it only
    /// for the `--router-mode` types (random, round-robin, least-loaded), in the routing
    /// host and as `PushRouter`'s instance selectors. A picker-only type named in a policy
    /// file still runs on the KV path.
    ///
    /// The picker then serves every request for its role, so each factory call must build
    /// a fresh one. Pins and session affinity are applied by the host before selection, so
    /// [`WorkerSelectionPolicy::with_exclusive_affinity`] has no effect there.
    fn into_hosted_picker(self) -> Result<HostedWorkerPicker, Self>;
}

/// How a host sets the values a policy reads from a [`SelectionConfig`], starting from
/// `SelectionConfig::default()`. Each builder sets the value the accessor of the same name
/// returns. A plugin only reads a [`SelectionConfig`].
pub trait SelectionConfigHostExt: sealed::Sealed + Sized {
    fn with_overlap_score_credit(self, overlap_score_credit: f64) -> Self;

    fn with_overlap_score_credit_decay(self, overlap_score_credit_decay: f64) -> Self;

    fn with_prefill_load_scale(self, prefill_load_scale: f64) -> Self;

    fn with_decode_active_request_weight(self, decode_active_request_weight: f64) -> Self;

    fn with_host_cache_hit_weight(self, host_cache_hit_weight: f64) -> Self;

    fn with_disk_cache_hit_weight(self, disk_cache_hit_weight: f64) -> Self;

    fn with_router_temperature(self, router_temperature: f64) -> Self;

    fn with_shared_cache_multiplier(self, shared_cache_multiplier: Option<f64>) -> Self;

    fn with_shared_cache_type(self, shared_cache_type: SharedCacheType) -> Self;

    fn with_conditional_disagg_enabled(self, conditional_disagg_enabled: bool) -> Self;
}

/// The facts about a request that every stage reads. It takes every value, so a place that
/// builds a [`RequestInfo`] fails to compile until it supplies a fact added later.
/// `pinned_worker` is the exact worker and rank the request's host requires, the same value
/// at every stage. An affinity target is not a pin, even when the selector treats it as
/// exclusive.
#[inline]
pub fn request_info<'a>(
    request_id: Option<&'a str>,
    isl_tokens: usize,
    session_context: Option<&'a SessionContext>,
    pinned_worker: Option<WorkerWithDpRank>,
) -> RequestInfo<'a> {
    RequestInfo::new(request_id, isl_tokens, session_context, pinned_worker)
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
