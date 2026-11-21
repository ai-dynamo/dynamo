// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What an in-tree host names from the admission stage: the classifier runtime it drives, the
//! queue rejection it renders, the queue policy and profiles it configures, the queue
//! snapshot replay reads, and the host half of the classifier contract types: how the kernel
//! builds the request a classifier reads and takes back what the classifier set. That half is
//! an extension trait and a function here, not methods on the contract types, so a plugin,
//! which does not name this module, cannot call them. This list is the host contract.

use dynamo_router_protocol::session::SessionContext;
use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_slot_tracker::BusyThresholds;
use dynamo_tokens::SequenceHash;
use tokio::time::Instant;

use crate::classifier::{ClassificationOverrides, ClassifyRequest, RequestProgressUpdater};

pub use crate::classifier::{AbortCause, RequestLifecycle, RequestProgress, WorkerView};
pub use crate::placement::LaneKey;
pub use crate::profile::{PolicyClassConfig, PolicyProfile, RouterQueuePolicy};
pub use crate::queue::{
    BlockReason, LaneCheck, PolicyQueue, QueueLane, QueueMetadata, QueueRejection, QueueSnapshot,
    RecheckScope,
};

// What the frontend host (lib/llm) names beyond the lists above.
pub use crate::queue::QueueLimitKind;

/// How the kernel builds the request a classifier reads, and what it reads back. A plugin
/// reads a [`ClassifyRequest`] and calls its `set_*` methods; its tests get one through
/// `dynamo_kv_router::plugins::testing`.
pub trait ClassifyRequestHostExt: Sized {
    /// A request of `input_tokens` prompt tokens, `initial_cached_tokens` of them cached,
    /// that reached the router at `ingress_at`. Every other field starts unset.
    fn with_timing(input_tokens: usize, initial_cached_tokens: usize, ingress_at: Instant) -> Self;

    fn with_request_id(self, request_id: impl Into<String>) -> Self;

    fn with_initial_policy_class(self, policy_class: impl Into<String>) -> Self;

    fn with_session_context(self, session_context: SessionContext) -> Self;

    fn with_pinned_worker(self, pinned_worker: Option<WorkerWithDpRank>) -> Self;

    /// Copy tracking hashes into storage owned by this classification attempt.
    fn with_sequence_hashes(self, hashes: Option<&[SequenceHash]>) -> Self;

    /// The router-wide busy thresholds read when classification started.
    fn with_busy_thresholds(self, reject: BusyThresholds) -> Self;

    /// Router-wide busy thresholds, read when classification started. Selection skips a
    /// worker past them and rejects the request when none remains.
    fn reject_thresholds(&self) -> BusyThresholds;

    /// The lane the classifier chose for the request within its class, when it chose one.
    fn lane_key(&self) -> Option<LaneKey>;

    /// What the classifier set on the request. Only these explicit overrides feed the queue:
    /// cache eligibility is recomputed from the current workers at enqueue, because worker
    /// state may have changed while the classification was pending.
    fn into_queue_inputs(self) -> ClassificationOverrides;
}

impl ClassifyRequestHostExt for ClassifyRequest {
    #[inline]
    fn with_timing(input_tokens: usize, initial_cached_tokens: usize, ingress_at: Instant) -> Self {
        ClassifyRequest::with_timing(input_tokens, initial_cached_tokens, ingress_at)
    }

    #[inline]
    fn with_request_id(self, request_id: impl Into<String>) -> Self {
        ClassifyRequest::with_request_id(self, request_id)
    }

    #[inline]
    fn with_initial_policy_class(self, policy_class: impl Into<String>) -> Self {
        ClassifyRequest::with_initial_policy_class(self, policy_class)
    }

    #[inline]
    fn with_session_context(self, session_context: SessionContext) -> Self {
        ClassifyRequest::with_session_context(self, session_context)
    }

    #[inline]
    fn with_pinned_worker(self, pinned_worker: Option<WorkerWithDpRank>) -> Self {
        ClassifyRequest::with_pinned_worker(self, pinned_worker)
    }

    #[inline]
    fn with_sequence_hashes(self, hashes: Option<&[SequenceHash]>) -> Self {
        ClassifyRequest::with_sequence_hashes(self, hashes)
    }

    #[inline]
    fn with_busy_thresholds(self, reject: BusyThresholds) -> Self {
        ClassifyRequest::with_busy_thresholds(self, reject)
    }

    #[inline]
    fn reject_thresholds(&self) -> BusyThresholds {
        ClassifyRequest::reject_thresholds(self)
    }

    #[inline]
    fn lane_key(&self) -> Option<LaneKey> {
        ClassifyRequest::lane_key(self)
    }

    #[inline]
    fn into_queue_inputs(self) -> ClassificationOverrides {
        ClassifyRequest::into_queue_inputs(self)
    }
}

/// A request's progress reading and the write handle the host updates it through. A
/// classifier only reads a [`RequestProgress`].
#[inline]
pub fn request_progress(
    initial_context_tokens: usize,
) -> (RequestProgress, RequestProgressUpdater) {
    RequestProgress::new(initial_context_tokens)
}
