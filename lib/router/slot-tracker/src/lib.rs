// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bookings per rank in, load and fullness per request out: the Dynamo router's slot
//! tracker stage.
//!
//! Holds the active-sequence trackers (`single`, `multi_worker`), the prompt registry with
//! its membership trie and load projection, the block and prefill trackers, the request
//! index that mints attempt ids, worker topology, the replication effects, the replication
//! wire (`wire`, with the scheduler-load report), per-rank fullness and its rule (`busy`), and
//! the booking descriptor the scheduler hands in (`booking`). Admission and worker selection
//! depend on this crate for the fullness rule. The replica mesh, the registry, and the HTTP server are `dynamo-kv-router`
//! adapters. Depends only on `dynamo-router-protocol` among router crates.

mod block_tracker;
mod booking;
mod busy;
mod cleanup;
mod compressed_path_arena;
pub mod host;
mod multi_worker;
mod prefill_tracker;
mod prelude;
mod prompt_membership_trie;
mod prompt_registry;
mod replica_sync;
mod request_maps;
mod sharded_lock;
mod single;
mod topology;
mod wire;

pub use booking::SchedulerBookingDescriptor;
pub use busy::{BusyRanksExt, BusyThresholds, FullRankCounts, RankLoad};
pub use multi_worker::{
    ActiveSequencesMultiWorker, LifecycleMutationOutcome, LocalWorkerLoad, NoopSequencePublisher,
    RateLimitedFailureLog, ReplicaRequestLeaseObserver, ReplicaWorkerPolicy, SchedulerLoadSnapshot,
    SequenceError, SequencePublishQueueError, SequencePublisher, SequenceRequest,
    SequenceSubscriber, SequenceTrackerOptions, active_request_expiry_duration,
};
pub use prefill_tracker::PrefillTokenDeltas;
pub use prompt_registry::WorkerLoadReader;
pub use prompt_registry::{PotentialLoadMaps, WorkerLoadProjection};
pub use single::{
    ActiveSequences, DEFAULT_ACTIVE_REQUEST_EXPIRY_DURATION, PrefillCompletion, RequestId,
};
pub use topology::{MAX_DATA_PARALLEL_RANKS_PER_WORKER, WorkerDpRange, WorkerTopologyError};
pub use wire::{
    ActiveBlockEvent, ActiveBlockEventData, ActiveSequenceEvent, ActiveSequenceEventBatch,
    ActiveSequenceEventData, MAX_REPLICA_BATCH_DURATION, MAX_REPLICA_BATCH_EVENTS, PrefillEvent,
    PrefillEventData, PrefillLoadHint, SchedulerGroup, SchedulerLoad,
};
