// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::SessionContext;
use crate::protocols::{SharedCacheHits, WorkerAffinityTarget, WorkerWithDpRank};
use crate::scheduling::SchedulingRequest;

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
}

impl<'a> WorkerSelectionContext<'a> {
    /// Kernel-side constructor: copy the fields a policy may read out of `request`.
    pub(crate) fn from_request(request: &'a SchedulingRequest, block_size: u32) -> Self {
        Self {
            request_id: request.mode.request_id(),
            isl_tokens: request.isl_tokens,
            request_blocks: request.request_blocks(block_size),
            block_size,
            track_prefill_tokens: request.track_prefill_tokens,
            pinned_worker: request.pinned_worker,
            router_temperature_override: request
                .router_config_override
                .as_ref()
                .and_then(|config| config.router_temperature),
            session_context: request.session_context.as_ref(),
            affinity_target: request.affinity_target,
            expected_output_tokens: request.expected_output_tokens,
            priority_jump: request.priority_jump,
            strict_priority: request.strict_priority,
            shared_cache_hits: request.shared_cache_hits.as_ref(),
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
