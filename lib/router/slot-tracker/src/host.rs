// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What an in-tree host names from the slot-tracker stage: the load snapshots and projections
//! it reads, the fullness rule it configures, the publisher and subscriber traits it
//! implements, the replication wire it forwards, and the booking descriptor the kernel hands
//! it. Every item but [`BusyThresholdsHostExt`] is re-exported from its owning module; this
//! list is the host contract.

pub use crate::booking::SchedulerBookingDescriptor;
pub use crate::busy::{BusyRanksExt, BusyThresholds, RankLoad};
pub use crate::decode_progress::{DecodeProgress, DecodeProgressUpdate};
pub use crate::multi_worker::{
    ActiveSequencesMultiWorker, LocalWorkerLoad, NoopSequencePublisher,
    ReplicaRequestLeaseObserver, SchedulerLoadSnapshot, SequenceError, SequencePublishQueueError,
    SequencePublisher, SequenceRequest, SequenceSubscriber,
};
pub use crate::prompt_registry::WorkerLoadProjection;
pub use crate::topology::WorkerDpRange;
pub use crate::wire::{
    ActiveSequenceEvent, ActiveSequenceEventData, ActiveSequenceEventOrigin, PrefillLoadHint,
    SchedulerGroup, SchedulerLoad,
};

// What the frontend host (lib/llm) names beyond the lists above.
pub use crate::multi_worker::{ReplicaWorkerPolicy, active_request_expiry_duration};
pub use crate::single::RequestId;
pub use crate::topology::MAX_DATA_PARALLEL_RANKS_PER_WORKER;
pub use crate::wire::{
    ActiveSequenceEventBatch, MAX_REPLICA_BATCH_DURATION, MAX_REPLICA_BATCH_EVENTS,
};

mod sealed {
    /// Keeps the host trait implementable only here, so a method can be added to it.
    pub trait Sealed {}
}

impl sealed::Sealed for BusyThresholds {}

/// How a host builds a [`BusyThresholds`], starting from `BusyThresholds::default()`, and
/// reads the limits it holds. A plugin only asks [`BusyThresholds::has_room`], so these are
/// not methods on the type: a plugin does not name this module and cannot call them.
pub trait BusyThresholdsHostExt: sealed::Sealed + Sized {
    /// Set the booked prefill tokens above which a rank is full.
    // TODO(v1.7): Remove the absolute threshold; use `prefill_busy_threshold_frac`.
    fn with_prefill_busy_threshold(self, prefill_busy_threshold: Option<usize>) -> Self;

    /// Set the fraction of the rank's `max_num_batched_tokens` above which it is full.
    fn with_prefill_busy_threshold_frac(self, prefill_busy_threshold_frac: Option<f64>) -> Self;

    /// Set the fraction of the rank's `total_kv_blocks` above which its active requests'
    /// KV blocks make it full.
    fn with_decode_busy_threshold_frac(self, decode_busy_threshold_frac: Option<f64>) -> Self;

    /// Booked prefill tokens above which a rank is full.
    fn prefill_busy_threshold(&self) -> Option<usize>;

    /// Fraction of the rank's `max_num_batched_tokens` above which it is full.
    fn prefill_busy_threshold_frac(&self) -> Option<f64>;

    /// Fraction of the rank's `total_kv_blocks` above which it is full.
    fn decode_busy_threshold_frac(&self) -> Option<f64>;
}
