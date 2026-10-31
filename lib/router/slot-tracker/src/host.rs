// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What an in-tree host names from the slot-tracker stage: the load snapshots and projections
//! it reads, the publisher and subscriber traits it implements, the replication wire it
//! forwards, and the booking descriptor the kernel hands it. Every item is re-exported from its
//! owning module; this list is the host contract (hosts inventory v2, section 3).

pub use crate::booking::SchedulerBookingDescriptor;
pub use crate::multi_worker::{
    ActiveSequencesMultiWorker, LocalWorkerLoad, NoopSequencePublisher,
    ReplicaRequestLeaseObserver, SchedulerLoadSnapshot, SequenceError, SequencePublishQueueError,
    SequencePublisher, SequenceRequest, SequenceSubscriber,
};
pub use crate::prompt_registry::WorkerLoadProjection;
pub use crate::single::ActiveSequences;
pub use crate::topology::WorkerDpRange;
pub use crate::wire::{ActiveSequenceEvent, ActiveSequenceEventData, PrefillLoadHint};
