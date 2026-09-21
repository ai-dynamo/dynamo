// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request metadata and the borrowed, CACHE-gated lookup snapshot.

use super::{SessionContext, WorkerInputs};
use crate::protocols::{SharedCacheHits, WorkerAffinityTarget, WorkerWithDpRank};
use crate::scheduling::SchedulingRequest;

/// Request-level values available to custom filters, scorers, and pickers.
pub struct WorkerSelectionContext<'a> {
    pub(crate) request: &'a SchedulingRequest,
    pub(crate) request_blocks: u64,
    pub(crate) block_size: u32,
    pub(crate) track_prefill_tokens: bool,
    pub(crate) has_tier_matches: bool,
    pub(crate) inputs: WorkerInputs,
    pub(crate) pinned_worker: Option<WorkerWithDpRank>,
    pub(crate) router_temperature_override: Option<f64>,
}

/// Request-wide cache facts from the host's current lookup snapshot.
/// Borrowed for the selection callback; no worker data is copied or looked up here.
#[derive(Clone, Copy)]
pub struct RequestCacheInput<'a> {
    shared_hits: Option<&'a SharedCacheHits>,
    has_tier_matches: bool,
}

impl<'a> RequestCacheInput<'a> {
    /// Unweighted shared-cache ranges in KV block positions, or None if no result was supplied.
    /// The host owns this snapshot. It does not change during the callback, and may lag engine state.
    /// Use `hits_beyond(prefix)` to exclude hits already covered by the policy's chosen prefix.
    pub fn shared_hits(self) -> Option<&'a SharedCacheHits> {
        self.shared_hits
    }

    /// Whether the request snapshot contains any tier-specific matches before worker filtering.
    /// False means only accounting estimates (or no cache data) were supplied. This describes
    /// observed matches, not worker cache capacity or whether a particular worker has a match.
    pub fn has_tier_matches(self) -> bool {
        self.has_tier_matches
    }
}

impl WorkerSelectionContext<'_> {
    /// The exact worker/rank imposed by the host for this selection, if any.
    /// Includes explicit pins and eligible exclusive-affinity targets. This is
    /// read-only routing metadata, not permission to change eligibility.
    pub fn pinned_worker(&self) -> Option<WorkerWithDpRank> {
        self.pinned_worker
    }

    /// Exact incoming prompt length in tokens. Borrowed from this request; no rounding,
    /// cache weighting, or additional storage is involved.
    pub fn prompt_tokens(&self) -> usize {
        self.request.isl_tokens
    }

    /// Borrow request-wide cache facts when this component declared [`WorkerInputs::CACHE`].
    /// This is the existing lookup snapshot; accessing it adds no lookup or allocation.
    pub fn cache(&self) -> Option<RequestCacheInput<'_>> {
        self.inputs
            .contains(WorkerInputs::CACHE)
            .then_some(RequestCacheInput {
                shared_hits: self.request.shared_cache_hits.as_ref(),
                has_tier_matches: self.has_tier_matches,
            })
    }

    /// Restrict request-level signals to this component's startup input declaration.
    pub(crate) fn with_inputs(&self, inputs: WorkerInputs) -> Self {
        Self {
            request: self.request,
            request_blocks: self.request_blocks,
            block_size: self.block_size,
            track_prefill_tokens: self.track_prefill_tokens,
            has_tier_matches: self.has_tier_matches,
            inputs,
            pinned_worker: self.pinned_worker,
            router_temperature_override: self.router_temperature_override,
        }
    }

    /// Return the incoming prompt size in KV blocks.
    pub fn request_blocks(&self) -> u64 {
        self.request_blocks
    }

    /// Return the number of tokens in one KV block.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Return whether this request contributes to prefill-load tracking.
    pub fn tracks_prefill_tokens(&self) -> bool {
        self.track_prefill_tokens
    }

    /// Return the session metadata available to worker selection.
    pub fn session_context(&self) -> Option<&SessionContext> {
        self.request.session_context.as_ref()
    }

    /// Return the session-affinity target resolved by the request host.
    ///
    /// The default selector treats an eligible target as exclusive. Custom policies receive it as
    /// advisory context; it may be absent from their candidate set when unavailable or filtered.
    pub fn affinity_target(&self) -> Option<WorkerAffinityTarget> {
        self.request.affinity_target
    }

    /// Return the expected output length, if the request supplies one.
    pub fn expected_output_tokens(&self) -> Option<u32> {
        self.request.expected_output_tokens
    }

    /// Return the request's scheduler priority boost.
    pub fn priority_jump(&self) -> f64 {
        self.request.priority_jump
    }

    /// Return the request's strict integer priority.
    pub fn strict_priority(&self) -> u32 {
        self.request.strict_priority
    }

    /// Return the request-level router temperature override, if present.
    pub fn router_temperature_override(&self) -> Option<f64> {
        self.router_temperature_override
    }
}
