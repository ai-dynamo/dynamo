// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV Router - Radix tree data structures for LLM KV cache routing.
//!
//! This crate provides the core radix tree implementation and protocols for
//! efficient KV cache lookup and routing in distributed LLM inference systems.

#[deprecated(
    since = "1.7.0",
    note = "the stage owns it: dynamo_worker_selection::host; see lib/kv-router/README.md, Deprecated paths"
)]
pub mod conditional_disagg;

#[deprecated(
    since = "1.7.0",
    note = "the vocabulary crate owns it: dynamo_router_protocol::identity; see lib/kv-router/README.md, Deprecated paths"
)]
pub mod identity;
#[deprecated(
    since = "1.7.0",
    note = "the stage owns it: dynamo_kv_indexer (host, cuckoo, api); see lib/kv-router/README.md, Deprecated paths"
)]
pub mod indexer;
#[deprecated(
    since = "1.7.0",
    note = "owned by dynamo_kv_hints, dynamo_kv_indexer::kv_hints, and dynamo_scheduler::kv_hints; see lib/kv-router/README.md, Deprecated paths"
)]
pub mod kv_hints;
pub mod plugins;
#[deprecated(
    since = "1.7.0",
    note = "the owners are dynamo_router_protocol, dynamo_kv_events, dynamo_kv_hashing, dynamo_kv_hints, and dynamo_scheduler::protocols; see lib/kv-router/README.md, Deprecated paths"
)]
pub mod protocols;
#[deprecated(
    since = "1.7.0",
    note = "owned by dynamo_kv_events::cursor; see lib/kv-router/README.md, Deprecated paths"
)]
pub mod recovery;
pub mod scheduling;
#[deprecated(
    since = "1.7.0",
    note = "depend on dynamo_slot_tracker directly (host module); see lib/kv-router/README.md, Deprecated paths"
)]
pub use dynamo_slot_tracker as sequences;
pub mod services;
#[deprecated(
    since = "1.7.0",
    note = "depend on dynamo_kv_indexer directly; see lib/kv-router/README.md, Deprecated paths"
)]
pub use dynamo_kv_indexer::session_prefix_index;
pub mod tracking_hash;
#[deprecated(
    since = "1.7.0",
    note = "owned by dynamo_router_protocol::WorkerType; see lib/kv-router/README.md, Deprecated paths"
)]
pub mod worker_type;
#[deprecated(
    since = "1.7.0",
    note = "depend on dynamo_kv_events directly; see lib/kv-router/README.md, Deprecated paths"
)]
pub use dynamo_kv_events::zmq_wire;

// Backward-compat re-exports: old top-level module paths still work
#[deprecated(
    since = "1.7.0",
    note = "dynamo_kv_indexer::concurrent_radix_tree_compressed; see lib/kv-router/README.md, Deprecated paths"
)]
pub use indexer::concurrent_radix_tree_compressed;
#[deprecated(
    since = "1.7.0",
    note = "dynamo_kv_indexer::positional; see lib/kv-router/README.md, Deprecated paths"
)]
pub use indexer::positional as nested_map;
#[deprecated(
    since = "1.7.0",
    note = "dynamo_kv_indexer::pruning; see lib/kv-router/README.md, Deprecated paths"
)]
pub use indexer::pruning as approx;
#[deprecated(
    since = "1.7.0",
    note = "dynamo_kv_indexer::radix_tree; see lib/kv-router/README.md, Deprecated paths"
)]
pub use indexer::radix_tree;

#[deprecated(
    since = "1.7.0",
    note = "dynamo_scheduler::queue (hosts: dynamo_scheduler::host); see lib/kv-router/README.md, Deprecated paths"
)]
pub use dynamo_scheduler::queue;
#[deprecated(
    since = "1.7.0",
    note = "dynamo_scheduler::selector (hosts: dynamo_scheduler::host); plugins: plugins::prelude; see lib/kv-router/README.md, Deprecated paths"
)]
pub use dynamo_scheduler::selector;
pub use scheduling::config;
#[deprecated(
    since = "1.7.0",
    note = "dynamo_slot_tracker::multi_worker; see lib/kv-router/README.md, Deprecated paths"
)]
pub use sequences::multi_worker as multi_worker_sequence;
#[deprecated(
    since = "1.7.0",
    note = "dynamo_slot_tracker::single; see lib/kv-router/README.md, Deprecated paths"
)]
pub use sequences::single as sequence;

#[cfg(any(test, feature = "bench"))]
pub mod test_utils;

// Re-export key types for convenience
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use self::multi_worker_sequence::{
    ActiveSequencesMultiWorker, NoopSequencePublisher, ReplicaWorkerPolicy, SequenceError,
    SequencePublisher, SequenceRequest, SequenceSubscriber,
};
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use self::sequence::{ActiveSequences, RequestId};
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use self::sequences::{PrefillTokenDeltas, WorkerLoadProjection};
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use config::{
    ConditionalDisaggConfig, ConditionalDisaggPolicyKind, KvRouterConfig, RouterConfigOverride,
    RouterPrefillLoadModel, RouterQueuePolicy, SelectionConfig, SharedCacheType,
};
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use identity::{ClusterId, DEFAULT_ROUTING_GROUP, RoutingPartitionId, RoutingPartitionRef};
#[allow(deprecated)]
pub use indexer::{
    AnchorAwareBranchShardedIndexer, AnchorRef, AnchorTask, BranchShardedIndexer,
    LowerTierContinuation, LowerTierIndexer, MaybeError, SharedKvCache, SyncIndexer,
    ThreadPoolIndexer,
};
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use nested_map::PositionalIndexer;
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use protocols::{
    KvCacheEventError, KvTransferEnforcement, LocalBlockHash, OverlapScores, RouterEvent,
    RouterEventSink, SharedCacheHits, WorkerConfigLike, WorkerId, compute_block_hash_for_seq,
};
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use queue::SchedulerQueue;
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use radix_tree::RadixTree;
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use scheduling::LocalScheduler;
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use scheduling::LoraWorkerFilter;
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use scheduling::PrefillLoadEstimator;
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use scheduling::{
    KvSchedulerError, PotentialLoad, SchedulingRequest, SchedulingResponse, SessionContext,
    WorkerSelectionInputTrigger, WorkerSelectionPolicyError,
};
#[cfg(any(test, feature = "bench"))]
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use selector::DefaultWorkerSelector;
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use selector::{WorkerSelectionInput, WorkerSelector};
// TODO(v1.7): Remove these compatibility re-exports; use crate::plugins instead.
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use plugins::worker_selection::{
    ScoredWorkerCandidate, WorkerCacheInput, WorkerCacheInputs, WorkerCandidate, WorkerCandidates,
    WorkerFilter, WorkerInputView, WorkerInputs, WorkerLoadInput, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicy,
};
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use session_prefix_index::{
    LogicalNode, NodeId, SessionId, SessionPrefixIndexError, SessionPrefixIndexer,
    SessionResidencyOwner, SharedCacheBackend,
};
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use tracking_hash::{
    TrackingHashAlgorithm, TrackingHashContext, TrackingHashScope, tracking_hash_context,
};
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use worker_type::WorkerType;

// TODO(v1.7): Remove these compatibility re-exports; use crate::plugins instead.
#[deprecated(
    since = "1.7.0",
    note = "import from the owner's host module, the vocabulary crate, plugins::prelude, or config::; see lib/kv-router/README.md, Deprecated paths"
)]
pub use plugins::worker_selection::WorkerSelectionPolicyFactory;
