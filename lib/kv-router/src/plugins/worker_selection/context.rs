// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request metadata available to worker-selection components.

use super::SessionContext;
use crate::protocols::{WorkerAffinityTarget, WorkerWithDpRank};
use crate::scheduling::SchedulingRequest;

/// Request-level values available to custom filters, scorers, and pickers.
pub struct WorkerSelectionContext<'a> {
    pub(crate) request: &'a SchedulingRequest,
    pub(crate) request_blocks: u64,
    pub(crate) block_size: u32,
    pub(crate) track_prefill_tokens: bool,
    pub(crate) pinned_worker: Option<WorkerWithDpRank>,
    pub(crate) router_temperature_override: Option<f64>,
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
