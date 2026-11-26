// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What an in-tree host names from the admission stage: the abort cause it reports and the
//! request progress it reads and updates, the queue rejection it renders, the queue policy and
//! profiles it configures, the policy queue offline replay drives, and the host half of the
//! classifier contract types: how the kernel builds the request a classifier reads and takes
//! back what the classifier set. That half is an extension trait here, not methods on the
//! contract type, so a plugin, which does not name this module, cannot call them. This list is
//! the host contract.

use dynamo_router_protocol::session::SessionContext;
use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_slot_tracker::BusyThresholds;
use dynamo_tokens::SequenceHash;
use tokio::time::Instant;

use crate::classifier::{ClassificationOverrides, ClassifyRequest};

pub use crate::classifier::{AbortCause, RequestProgress, RequestProgressUpdater};
pub use crate::lane_key::LaneKey;
pub use crate::profile::{PolicyClassConfig, PolicyProfile, RouterQueuePolicy};
pub use crate::queue::{
    BlockReason, LaneCheck, PolicyQueue, QueueLane, QueueLimitKind, QueueMetadata, QueueRejection,
    QueueSnapshot, RecheckScope,
};

mod sealed {
    /// Keeps the host traits implementable only here, so a method can be added to one.
    pub trait Sealed {}
}

impl sealed::Sealed for ClassifyRequest {}

/// A request's progress reading and the write capability paired with it, starting at
/// `initial_context_tokens`. The classifier runtime creates the pair for each request it
/// classifies; a host calls this where it has no runtime, as in its tests. A plugin only
/// reads a [`RequestProgress`]; its tests get one through
/// `dynamo_kv_router::plugins::testing::request_progress`.
pub fn request_progress(
    initial_context_tokens: usize,
) -> (RequestProgress, RequestProgressUpdater) {
    RequestProgress::new(initial_context_tokens)
}

/// How the kernel builds the request a classifier reads, and what it reads back. A plugin
/// reads a [`ClassifyRequest`] and calls its `set_*` methods; its tests get one through
/// `dynamo_kv_router::plugins::testing`.
pub trait ClassifyRequestHostExt: sealed::Sealed + Sized {
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

    /// The lane the classifier chose for the request within its class, when it chose one.
    fn lane_key(&self) -> Option<LaneKey>;

    /// What the classifier set on the request. Only these explicit overrides feed the queue:
    /// cache eligibility is recomputed from the current workers at enqueue, because worker
    /// state may have changed while the classification was pending.
    fn into_queue_inputs(self) -> ClassificationOverrides;
}
