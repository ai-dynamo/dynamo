// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_router_protocol::session::SessionContext;
use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_session_affinity::WorkerAffinityTarget;

use super::inputs::{CacheSnapshot, ResidentBlockSource};
use super::shared_cache::SharedCacheHits;

/// What is true about a request at every stage, read by a request classifier, a dispatch
/// gate, and a worker-selection policy alike through `info()` on the request each receives.
///
/// A value that exists only at one stage stays on that stage's type: a classifier's
/// overrides and timing on `ClassifyRequest`, the affinity target and the shared-cache hits
/// on [`WorkerSelectionRequest`].
#[derive(Clone, Copy, Debug)]
pub struct RequestInfo<'a> {
    request_id: Option<&'a str>,
    isl_tokens: usize,
    session_context: Option<&'a SessionContext>,
    pinned_worker: Option<WorkerWithDpRank>,
}

impl<'a> RequestInfo<'a> {
    #[inline]
    pub(crate) fn new(
        request_id: Option<&'a str>,
        isl_tokens: usize,
        session_context: Option<&'a SessionContext>,
        pinned_worker: Option<WorkerWithDpRank>,
    ) -> Self {
        Self {
            request_id,
            isl_tokens,
            session_context,
            pinned_worker,
        }
    }

    /// The request id for tracked requests; `None` for query-only selections.
    #[inline]
    pub fn request_id(&self) -> Option<&'a str> {
        self.request_id
    }

    /// Input sequence length in tokens, the size the scheduler uses. No rounding or cache
    /// weighting is applied. Zero when the request carries no prompt token IDs, such as
    /// embeddings-only input.
    #[inline]
    pub fn isl_tokens(&self) -> usize {
        self.isl_tokens
    }

    /// Return the session metadata of the request, when it carries any.
    #[inline]
    pub fn session_context(&self) -> Option<&'a SessionContext> {
        self.session_context
    }

    /// The exact worker/rank the request's host requires, if any, the same value at every
    /// stage, such as an explicit target on the request, a session binding the host resolved
    /// to one rank before selection, or the worker a route preview chose. At worker selection
    /// the policy is offered only this worker. This is read-only routing metadata, not
    /// permission to change eligibility; a classifier's soft selection target and a dispatch
    /// gate's `Go` target cannot override it. A session-affinity target the host passes
    /// without pinning is not a pin, even when the selector treats it as exclusive: a policy
    /// reads it through [`WorkerSelectionRequest::affinity_target`].
    #[inline]
    pub fn pinned_worker(&self) -> Option<WorkerWithDpRank> {
        self.pinned_worker
    }
}

/// The view of one scheduling request that filters, scorers, and pickers may read.
///
/// Holds values, not the request: the kernel builds it once per selection from the
/// request it owns, so a policy sees exactly the fields the accessors expose.
pub struct WorkerSelectionRequest<'a> {
    pub(crate) decode_progress: Option<&'a super::host::DecodeProgressSource<'a>>,
    pub(crate) info: RequestInfo<'a>,
    pub(crate) request_blocks: u64,
    pub(crate) block_size: u32,
    pub(crate) track_prefill_tokens: bool,
    pub(crate) router_temperature_override: Option<f64>,
    pub(crate) affinity_target: Option<WorkerAffinityTarget>,
    pub(crate) expected_output_tokens: Option<u32>,
    pub(crate) priority_jump: f64,
    pub(crate) strict_priority: u32,
    pub(crate) shared_cache_hits: Option<&'a SharedCacheHits>,
    pub(crate) has_tier_matches: bool,
    pub(crate) resident_blocks: Option<&'a dyn ResidentBlockSource>,
    pub(crate) preview: bool,
}

impl<'a> WorkerSelectionRequest<'a> {
    /// Request-wide cache facts shared by every candidate row.
    #[inline]
    pub(crate) fn cache_snapshot(&self) -> CacheSnapshot<'a> {
        CacheSnapshot {
            shared_hits: self.shared_cache_hits,
            has_tier_matches: self.has_tier_matches,
            resident_blocks: self.resident_blocks,
        }
    }

    /// The request facts every stage reads: the request id, the input length, the session,
    /// and the pinned worker.
    #[inline]
    pub fn info(&self) -> RequestInfo<'a> {
        self.info
    }

    /// Return the input size in KV blocks. Zero when
    /// [`isl_tokens`](RequestInfo::isl_tokens) is zero.
    #[inline]
    pub fn request_blocks(&self) -> u64 {
        self.request_blocks
    }

    /// Return the number of tokens in one KV block.
    #[inline]
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Return whether this request contributes to prefill-load tracking.
    #[inline]
    pub fn tracks_prefill_tokens(&self) -> bool {
        self.track_prefill_tokens
    }

    /// Return the session-affinity target resolved by the request host.
    ///
    /// The default selector treats an eligible target as exclusive. Custom policies receive it as
    /// advisory context; it may be absent from their candidate set when unavailable or filtered.
    #[inline]
    pub fn affinity_target(&self) -> Option<WorkerAffinityTarget> {
        self.affinity_target
    }

    /// Return the expected output length, if the request supplies one.
    #[inline]
    pub fn expected_output_tokens(&self) -> Option<u32> {
        self.expected_output_tokens
    }

    /// Return the request's scheduler priority boost.
    #[inline]
    pub fn priority_jump(&self) -> f64 {
        self.priority_jump
    }

    /// Return the request's strict integer priority.
    #[inline]
    pub fn strict_priority(&self) -> u32 {
        self.strict_priority
    }

    /// Return the request-level router temperature override, if present.
    #[inline]
    pub fn router_temperature_override(&self) -> Option<f64> {
        self.router_temperature_override
    }

    /// Block positions the external shared cache holds for this request, when a shared
    /// cache is configured and answered.
    #[inline]
    pub fn shared_cache_hits(&self) -> Option<&SharedCacheHits> {
        self.shared_cache_hits
    }

    /// Whether this selection only previews a worker, as a peek does, without routing the
    /// request to it. A picker that keeps state between selections, such as a rotation
    /// cursor, should read that state without advancing it. Always `false` when the router
    /// routes with KV state; today only the hosted path, which runs the builtin router-mode
    /// pickers, sets it.
    #[inline]
    pub fn is_preview(&self) -> bool {
        self.preview
    }
}

impl<'a> crate::host::WorkerSelectionRequestHostExt<'a> for WorkerSelectionRequest<'a> {
    #[inline]
    fn new(info: RequestInfo<'a>, block_size: u32) -> Self {
        Self {
            decode_progress: None,
            info,
            request_blocks: match block_size {
                0 => 0,
                block_size => info.isl_tokens().div_ceil(block_size as usize) as u64,
            },
            block_size,
            track_prefill_tokens: false,
            router_temperature_override: None,
            affinity_target: None,
            expected_output_tokens: None,
            priority_jump: 0.0,
            strict_priority: 0,
            shared_cache_hits: None,
            has_tier_matches: false,
            resident_blocks: None,
            preview: false,
        }
    }

    #[inline]
    fn with_preview(mut self, preview: bool) -> Self {
        self.preview = preview;
        self
    }

    #[inline]
    fn with_prefill_tracking(mut self, track_prefill_tokens: bool) -> Self {
        self.track_prefill_tokens = track_prefill_tokens;
        self
    }

    #[inline]
    fn with_router_temperature_override(mut self, temperature: Option<f64>) -> Self {
        self.router_temperature_override = temperature;
        self
    }

    #[inline]
    fn with_affinity_target(mut self, affinity_target: Option<WorkerAffinityTarget>) -> Self {
        self.affinity_target = affinity_target;
        self
    }

    #[inline]
    fn with_expected_output_tokens(mut self, expected_output_tokens: Option<u32>) -> Self {
        self.expected_output_tokens = expected_output_tokens;
        self
    }

    #[inline]
    fn with_priority(mut self, priority_jump: f64, strict_priority: u32) -> Self {
        self.priority_jump = priority_jump;
        self.strict_priority = strict_priority;
        self
    }

    #[inline]
    fn with_shared_cache_hits(mut self, shared_cache_hits: Option<&'a SharedCacheHits>) -> Self {
        self.shared_cache_hits = shared_cache_hits;
        self
    }

    #[inline]
    fn with_tier_matches(mut self, has_tier_matches: bool) -> Self {
        self.has_tier_matches = has_tier_matches;
        self
    }

    #[inline]
    fn with_resident_blocks(
        mut self,
        resident_blocks: Option<&'a dyn ResidentBlockSource>,
    ) -> Self {
        self.resident_blocks = resident_blocks;
        self
    }

    #[inline]
    fn with_decode_progress(
        mut self,
        progress: Option<&'a super::host::DecodeProgressSource<'a>>,
    ) -> Self {
        self.decode_progress = progress;
        self
    }
}
