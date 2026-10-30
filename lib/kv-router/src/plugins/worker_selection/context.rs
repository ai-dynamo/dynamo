// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_router_protocol::session::SessionContext;
use dynamo_router_protocol::worker::{WorkerAffinityTarget, WorkerWithDpRank};

use super::inputs::CacheSnapshot;
use super::shared_cache::SharedCacheHits;

/// The view of one scheduling request that filters, scorers, and pickers may read.
///
/// Holds values, not the request: the kernel builds it once per selection from the
/// request it owns, so a policy sees exactly the fields the accessors expose.
pub struct WorkerSelectionContext<'a> {
    pub(crate) request_id: Option<&'a str>,
    pub(crate) isl_tokens: usize,
    pub(crate) request_blocks: u64,
    pub(crate) block_size: u32,
    pub(crate) track_prefill_tokens: bool,
    pub(crate) pinned_worker: Option<WorkerWithDpRank>,
    pub(crate) router_temperature_override: Option<f64>,
    pub(crate) session_context: Option<&'a SessionContext>,
    pub(crate) affinity_target: Option<WorkerAffinityTarget>,
    pub(crate) expected_output_tokens: Option<u32>,
    pub(crate) priority_jump: f64,
    pub(crate) strict_priority: u32,
    pub(crate) shared_cache_hits: Option<&'a SharedCacheHits>,
    pub(crate) has_tier_matches: bool,
}

impl<'a> WorkerSelectionContext<'a> {
    /// A view of a request of `isl_tokens` routed in blocks of `block_size`, with every other
    /// field at its default: no request id, no pin, no session, no affinity target, no
    /// expected output length, zero priority, no shared-cache hits, no tier matches, and
    /// prefill-load tracking off. The kernel sets what the request carries with the `with_*`
    /// methods; a test builds exactly the view it needs.
    pub fn new(isl_tokens: usize, block_size: u32) -> Self {
        Self {
            request_id: None,
            isl_tokens,
            request_blocks: isl_tokens.div_ceil(block_size as usize) as u64,
            block_size,
            track_prefill_tokens: false,
            pinned_worker: None,
            router_temperature_override: None,
            session_context: None,
            affinity_target: None,
            expected_output_tokens: None,
            priority_jump: 0.0,
            strict_priority: 0,
            shared_cache_hits: None,
            has_tier_matches: false,
        }
    }

    pub fn with_request_id(mut self, request_id: Option<&'a str>) -> Self {
        self.request_id = request_id;
        self
    }

    pub fn with_prefill_tracking(mut self, track_prefill_tokens: bool) -> Self {
        self.track_prefill_tokens = track_prefill_tokens;
        self
    }

    /// The exact worker and rank the host imposes: an explicit pin or an eligible
    /// exclusive-affinity target.
    pub fn with_pinned_worker(mut self, pinned_worker: Option<WorkerWithDpRank>) -> Self {
        self.pinned_worker = pinned_worker;
        self
    }

    pub fn with_router_temperature_override(mut self, temperature: Option<f64>) -> Self {
        self.router_temperature_override = temperature;
        self
    }

    pub fn with_session_context(mut self, session_context: Option<&'a SessionContext>) -> Self {
        self.session_context = session_context;
        self
    }

    pub fn with_affinity_target(mut self, affinity_target: Option<WorkerAffinityTarget>) -> Self {
        self.affinity_target = affinity_target;
        self
    }

    pub fn with_expected_output_tokens(mut self, expected_output_tokens: Option<u32>) -> Self {
        self.expected_output_tokens = expected_output_tokens;
        self
    }

    pub fn with_priority(mut self, priority_jump: f64, strict_priority: u32) -> Self {
        self.priority_jump = priority_jump;
        self.strict_priority = strict_priority;
        self
    }

    pub fn with_shared_cache_hits(
        mut self,
        shared_cache_hits: Option<&'a SharedCacheHits>,
    ) -> Self {
        self.shared_cache_hits = shared_cache_hits;
        self
    }

    /// Whether any rank reported per-tier overlap, so cache columns read device overlap
    /// from the tier blocks rather than the effective estimate.
    pub fn with_tier_matches(mut self, has_tier_matches: bool) -> Self {
        self.has_tier_matches = has_tier_matches;
        self
    }

    /// Request-wide cache facts shared by every candidate row.
    pub(crate) fn cache_snapshot(&self) -> CacheSnapshot<'a> {
        CacheSnapshot {
            shared_hits: self.shared_cache_hits,
            has_tier_matches: self.has_tier_matches,
        }
    }

    /// The request id for tracked requests; `None` for query-only selections.
    pub fn request_id(&self) -> Option<&str> {
        self.request_id
    }

    /// The exact worker/rank imposed by the host for this selection, if any.
    /// Includes explicit pins and eligible exclusive-affinity targets. This is
    /// read-only routing metadata, not permission to change eligibility.
    pub fn pinned_worker(&self) -> Option<WorkerWithDpRank> {
        self.pinned_worker
    }

    /// Input sequence length in tokens, the same value classifiers read as
    /// `ClassifyRequest::isl_tokens`. No rounding or cache weighting is applied.
    /// Zero when the request carries no prompt token IDs, such as embeddings-only input.
    pub fn isl_tokens(&self) -> usize {
        self.isl_tokens
    }

    /// Return the input size in KV blocks. Zero when
    /// [`isl_tokens`](Self::isl_tokens) is zero.
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
        self.session_context
    }

    /// Return the session-affinity target resolved by the request host.
    ///
    /// The default selector treats an eligible target as exclusive. Custom policies receive it as
    /// advisory context; it may be absent from their candidate set when unavailable or filtered.
    pub fn affinity_target(&self) -> Option<WorkerAffinityTarget> {
        self.affinity_target
    }

    /// Return the expected output length, if the request supplies one.
    pub fn expected_output_tokens(&self) -> Option<u32> {
        self.expected_output_tokens
    }

    /// Return the request's scheduler priority boost.
    pub fn priority_jump(&self) -> f64 {
        self.priority_jump
    }

    /// Return the request's strict integer priority.
    pub fn strict_priority(&self) -> u32 {
        self.strict_priority
    }

    /// Return the request-level router temperature override, if present.
    pub fn router_temperature_override(&self) -> Option<f64> {
        self.router_temperature_override
    }

    /// Block positions the external shared cache holds for this request, when a shared
    /// cache is configured and answered.
    pub fn shared_cache_hits(&self) -> Option<&SharedCacheHits> {
        self.shared_cache_hits
    }
}
