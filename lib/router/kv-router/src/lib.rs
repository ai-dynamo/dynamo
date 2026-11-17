// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The composition crate of the router: it assembles the scheduler kernel and the stage
//! crates into a running router and is the one crate plugins and standalone consumers name.
//!
//! - [`services::selection`]: `SelectionService` and `SelectionServiceBuilder`, the composed
//!   router an embedding host or the standalone service runs.
//! - [`plugins`]: the plugin registry (`RouterPluginRegistry`, `RouterPlugins`) and the
//!   worker-selection and request-classifier contracts; [`plugins::prelude`] is the single
//!   import a statically linked plugin needs.
//! - [`config`]: the flat `KvRouterConfig` with its CLI, environment, and policy-file parsers,
//!   converted once into the kernel's validated configuration; [`scheduling::policy_config`]
//!   is the YAML policy file and its resolver.
//! - [`tracking_hash`]: the tracking-hash key file read and the context built from the flat
//!   configuration.
//! - Adapters behind features: the HTTP servers (`http`), the ZMQ KV-event listener and its
//!   registry (`ingest-zmq`), the replica mesh (`replica-sync-zmq`), the indexer peer-recovery
//!   client (`peer-recovery`), and Prometheus exposition (`metrics`), under [`services`].
//!
//! Hosts import the owning crate's `host` module (`dynamo_scheduler::host`,
//! `dynamo_kv_indexer::host`, `dynamo_slot_tracker::host`, `dynamo_session_affinity::host`,
//! `dynamo_worker_selection::host`, `dynamo_admission::host`) for stage and kernel items. The
//! pre-split paths of this crate keep resolving through 1.9 as deprecated re-exports (`compat.rs`;
//! the README's migration map names each owner) and are removed in 1.10.

pub mod plugins;
pub mod scheduling;
pub mod services;
pub mod tracking_hash;

pub use scheduling::config;

mod compat;
// Pre-split paths kept through 1.9 and removed in 1.10; the named list is the inventory in compat.rs.
#[cfg(any(test, feature = "testing", feature = "bench"))]
#[allow(deprecated)]
pub use compat::DefaultWorkerSelector;
#[allow(deprecated)]
pub use compat::{
    ActiveSequences, ActiveSequencesMultiWorker, AnchorAwareBranchShardedIndexer, AnchorRef,
    AnchorTask, BranchShardedIndexer, ClusterId, ConcurrentRadixTreeCompressed,
    ConditionalDisaggPolicyKind, DEFAULT_ROUTING_GROUP, FcfsPolicy, KvCacheEventError,
    KvRouterConfig, KvSchedulerError, KvTransferEnforcement, LocalBlockHash, LocalScheduler,
    LogicalNode, LoraWorkerFilter, LowerTierContinuation, LowerTierIndexer, MaybeError, NodeId,
    NoopSequencePublisher, OverlapScores, PositionalIndexer, PotentialLoad, PrefillLoadEstimator,
    PrefillTokenDeltas, RadixTree, ReplicaWorkerPolicy, RequestId, RouterConfigOverride,
    RouterEvent, RouterEventSink, RouterPrefillLoadModel, RouterQueuePolicy,
    RouterSchedulingPolicy, RoutingPartitionId, RoutingPartitionRef, SchedulerQueue,
    SchedulingPolicy, SchedulingRequest, SchedulingResponse, SequenceError, SequencePublisher,
    SequenceRequest, SequenceSubscriber, SessionContext, SessionId, SessionPrefixIndexError,
    SessionPrefixIndexer, SessionResidencyOwner, SharedCacheBackend, SharedCacheHits,
    SharedCacheType, SharedKvCache, SyncIndexer, ThreadPoolIndexer, TrackingHashAlgorithm,
    TrackingHashContext, TrackingHashScope, WorkerConfigLike, WorkerId, WorkerLoadProjection,
    WorkerSelectionInput, WorkerSelectionInputTrigger, WorkerSelectionPolicyError, WorkerSelector,
    WorkerType, WsptPolicy, approx, compute_block_hash_for_seq, concurrent_radix_tree_compressed,
    conditional_disagg, identity, indexer, kv_hints, multi_worker_sequence, nested_map, protocols,
    queue, radix_tree, recovery, selector, sequence, sequences, session_prefix_index, worker_type,
    zmq_wire,
};

#[cfg(any(test, feature = "bench"))]
pub mod test_utils;
