// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request metadata available to worker-selection components.

use super::{SessionContext, WorkerCapacityInput};
use crate::protocols::{WorkerAffinityTarget, WorkerId, WorkerWithDpRank};
use crate::scheduling::SchedulingRequest;

/// Request-level values available to custom filters, scorers, and pickers.
pub struct WorkerSelectionContext<'a> {
    pub(crate) request: &'a SchedulingRequest,
    /// Looks up a worker's runtime config, so rows carry no capacity a policy may never read.
    pub(crate) worker_capacity: &'a dyn Fn(WorkerId) -> Option<WorkerCapacityInput>,
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
    /// cache weighting, or additional storage is involved. Zero when the request carries
    /// no prompt token IDs, such as embeddings-only input.
    pub fn prompt_tokens(&self) -> usize {
        self.request.isl_tokens
    }

    /// Return the incoming prompt size in KV blocks. Zero when
    /// [`prompt_tokens`](Self::prompt_tokens) is zero.
    pub fn request_blocks(&self) -> u64 {
        self.request_blocks
    }

    /// Return the number of tokens in one KV block.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Chained hashes of the prompt's complete KV blocks, borrowed from this request.
    ///
    /// Entry `i` identifies prompt blocks `0..=i`, so two requests share their first `i + 1`
    /// blocks exactly when their entry `i` is equal. LoRA adapters, cache namespaces, and
    /// multimodal content hash into separate domains. These are the host's active-sequence
    /// tracking hashes; do not compare them with engine KV-event hashes. None when the host
    /// does not track active blocks, as in disaggregated prefill pools. When the host does not
    /// assume KV reuse, as in disaggregated decode pools, every entry is unique to the request.
    pub fn prefix_hashes(&self) -> Option<&[u64]> {
        self.request.token_seq.as_deref()
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

    /// Return the capacity advertised in `worker`'s runtime config, or None if the host does not
    /// know the worker. Every data-parallel rank of a worker shares one config. Each call looks
    /// the worker up in the host's worker table.
    pub fn worker_capacity(&self, worker: WorkerWithDpRank) -> Option<WorkerCapacityInput> {
        (self.worker_capacity)(worker.worker_id)
    }

    /// Return `worker`'s modeled prefill backlog in milliseconds: a derived estimate of the time
    /// its active prefills still need, from the host's prefill-load model predictions. It excludes
    /// this request's own prefill.
    ///
    /// None unless a component of this policy declares [`WorkerInputs::PREFILL_TIME`], and None
    /// for a worker when the host runs without a prefill-load model
    /// (`router_prefill_load_model: ais`), a prediction failed, or any active prefill on the
    /// worker is unmodeled. Values synced from router replicas are anchored at receive time, not
    /// at the producer's time.
    ///
    /// [`WorkerInputs::PREFILL_TIME`]: super::WorkerInputs::PREFILL_TIME
    pub fn modeled_prefill_backlog_ms(&self, worker: WorkerWithDpRank) -> Option<u64> {
        self.request
            .modeled_prefill_backlog_ms
            .get(&worker)
            .copied()
    }

    /// Return the policy class requested for this request: the caller's value, replaced by a
    /// request classifier's override. This is the requested name; a profile with class
    /// families may still resolve it to a family member for queueing.
    pub fn policy_class(&self) -> Option<&str> {
        self.request.policy_class.as_deref()
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
