// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV events in, prefix knowledge out: the Dynamo router's KV indexer stage.
//!
//! The crate holds the radix trees (`radix_tree`, `positional`, the concurrent compressed
//! tree), the thread-pool and branch-sharded wrappers, `KvIndexer` and `LocalKvIndexer`,
//! approximate LRU retention and pruning, the cuckoo-filter pool index (`cuckoo`), the hash
//! lifecycle delegate (`delegate`), the indexer counters (`metrics`), the residency
//! projection (`residency`), overlap scores (`overlap`), KV transfer candidates (`kv_hints`),
//! the lower-tier continuation index with its per-tier registry, the ingest backend
//! (`backend`, `lookup`, `recording`, `session_updates`), the session prefix index, the
//! request-plane DTOs (`api`), explicit identity material (`identity`), and its configuration
//! slice (`config`). `lib/router/kv-router` re-exports every item here at its old path and keeps the
//! HTTP, ZMQ, peer-recovery, and Prometheus adapters.
//!
//! Depends only on `dynamo-router-protocol`, `dynamo-kv-events`, and `dynamo-kv-hashing`. It
//! owns what it emits with a match: the transfer candidate vocabulary
//! (`kv_hints::KvTransferCandidateSource`, `label_key`, `topology_keys`) and the residency
//! owner key (`residency::ResidencyOwnerKey`); `dynamo-kv-hints` imports them from here, never
//! the reverse.
//! The `export` feature (Prometheus counter vectors) is transitional: the design replaces it
//! with atomics and a `stats()` snapshot in the metrics behavior change.
//!
//! # KV RadixTree
//!
//! This module implements a key-value (KV) store using a Radix Tree structure to efficiently manage and retrieve data blocks.
//! It is designed to support LLM (Large Language Model) inference by re-using a global KV cache.
//!
//! # Overview
//!
//! The main components of this module include:
//!
//! - **Radix Tree Structure**:
//!   - The `RadixTree` struct represents the main data structure, with nodes (`RadixBlock`) containing children and associated worker IDs.
//!   - It allows efficient storage and retrieval of data blocks based on their hashes.
//!
//! - **Event Handling**:
//!   - The `RouterEvent` struct represents events emitted by LLM workers, which can be applied to the Radix Tree to update its state.
//!   - The `KvIndexer` struct manages these events and match requests asynchronously using Tokio channels.
//!
//! - **Hash Computation**:
//!   - Functions like `compute_block_hash` and `compute_block_hash_for_seq` compute hashes for data blocks and sequences of tokens, facilitating quick lookups.
//!
//! - **Concurrency and Asynchronous Operations**:
//!   - The `KvIndexer` uses a single-threaded Tokio runtime to handle events and match requests concurrently, ensuring efficient processing without blocking.
//!
//! - **Match Requests**:
//!   - The `MatchRequest` struct represents requests to find matches in the Radix Tree, returning overlap scores indicating the best matches.
//!
//! # Purpose
//!
//! This module provides a scalable and efficient way to manage and retrieve data blocks for LLM inference, leveraging a global KV cache to optimize performance.

mod active_set;
mod api;
mod approximate_lru;
mod backend;
mod branch_sharded;
mod compressed_radix;
mod concurrent_radix_tree_compressed;
mod config;
pub mod cuckoo;
mod delegate;
pub mod host;
mod identity;
mod kv_hints;
mod kv_indexer;
mod local;
mod lookup;
mod lower_tier;
mod lower_tier_indexers;
mod metrics;
#[cfg(test)]
mod mm_exclusion_tests;
#[cfg(feature = "bench")]
mod observation;
mod overlap;
mod positional;
mod prelude;
mod pruning;
mod radix_tree;
mod recording;
mod registry;
mod residency;
mod session_prefix_index;
mod session_updates;
mod shard_handle;
#[cfg(any(test, feature = "testing", feature = "bench"))]
pub mod testing;
mod thread_pool;
mod traits;
mod types;

#[cfg(test)]
mod delegate_tests;
#[cfg(test)]
mod local_tests;
#[cfg(test)]
mod tests;

use std::any::Any;

pub(crate) fn panic_payload_message(panic_payload: &(dyn Any + Send)) -> String {
    if let Some(s) = panic_payload.downcast_ref::<&str>() {
        return s.to_string();
    }

    if let Some(s) = panic_payload.downcast_ref::<String>() {
        return s.clone();
    }

    "Unknown panic payload".to_string()
}

fn warn_on_unit_block_size(indexer_type: &'static str, kv_block_size: u32) {
    if kv_block_size == 1 {
        tracing::warn!(
            indexer_type,
            kv_block_size,
            "block_size=1 is supported for KV indexers, but consider avoiding it because KV events may saturate network bandwidth",
        );
    }
}

pub use api::{
    IndexerQueryRequest, IndexerQueryResponse, IndexerRecordRoutingDecisionRequest,
    IndexerRecordRoutingDecisionResponse, KV_INDEXER_QUERY_ENDPOINT,
    KV_INDEXER_RECORD_ROUTING_DECISION_ENDPOINT, KvStateAgentIdentity, KvStateAgentStatus,
    KvStateAttachmentStatus, KvStateProtocolVersion, KvStateRecoveryReceipt, MaybeError,
    WorkerKvQueryKind, WorkerKvQueryRequest, WorkerKvQueryResponse,
};
pub use approximate_lru::{
    ApproximateAcquireMode, ApproximateLruBlock, ApproximateLruClient, ApproximateLruIncarnation,
    ApproximateLruLease, ApproximateLruReleaseAck, ApproximateLruReply, ApproximateLruStats,
    ApproximateLruTask, ApproximateRetentionConfig,
};
pub use backend::{
    Indexer, IndexerPolicy, PrimaryRetention, RemotePrimary, SideIndexer, create_indexer,
    create_indexer_with_metrics, create_indexer_with_policy,
};
// The deprecated alias stays exported for the pre-split paths in `dynamo-kv-router`'s compat list.
#[allow(deprecated)]
pub use branch_sharded::AnchorAwareBranchShardedIndexer;
pub use branch_sharded::BranchShardedIndexer;
pub use concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
pub use config::{ApproximateCachePolicyKind, IndexerConfig};
pub use delegate::{HashLifecycle, KvIndexerDelegate};
pub use identity::{
    CanonicalIdentityMaterial, ExplicitIdentityMap, IdentitySpecError, IndexerIdentitySpec,
    MAX_EXPLICIT_IDENTITY_ENTRIES, MAX_EXPLICIT_IDENTITY_KEY_BYTES,
    MAX_EXPLICIT_IDENTITY_VALUE_BYTES,
};
pub use kv_hints::{
    CacheOwnerKvTransferKeys, KvTransferCandidateSource, KvTransferCandidates,
    KvTransferSourceInterval, label_key, topology_keys,
};
pub use kv_indexer::{KvEventSender, KvIndexer, KvIndexerBuilder};
pub use local::LocalKvIndexer;
pub use lookup::HashInput;
pub use lower_tier::{
    KvTransferExtensions, LowerTierContinuation, LowerTierIndexer, LowerTierMatchDetails,
};
pub use lower_tier_indexers::{
    LowerTierIndexers, LowerTierQueryOptions, TieredMatchDetails, TieredMatchProvider,
    lower_tier_query_order, query_lower_tiers_with_options,
    query_lower_tiers_with_options_and_snapshot,
};
pub use metrics::{
    CkfMutationKind, EventKind, EventWarningKind, KvIndexerMetrics,
    METRIC_CKF_MUTATION_CAPACITY_EXHAUSTED, METRIC_CKF_MUTATION_UNKNOWN_REMOVE,
    METRIC_EVENT_CLEARED, METRIC_EVENT_REMOVED, METRIC_EVENT_STORED,
    METRIC_STATUS_ALLOCATION_FAILED, METRIC_STATUS_BLOCK_NOT_FOUND,
    METRIC_STATUS_CAPACITY_EXHAUSTED, METRIC_STATUS_INDEXER_INVARIANT_VIOLATION,
    METRIC_STATUS_INVALID_BLOCK, METRIC_STATUS_OK, METRIC_STATUS_OWNERSHIP_DEGREE_OVERFLOW,
    METRIC_STATUS_PARENT_NOT_FOUND, METRIC_STATUS_UNSUPPORTED_RESIDENCY_DOMAIN,
    METRIC_WARNING_DUPLICATE_STORE, PreBoundEventCounters, record_unsupported_residency_event,
};
#[cfg(feature = "bench")]
pub use observation::{
    EventCompletion, EventCompletionBuffer, EventCompletionWriter, ObservationError,
    ObservationSeal, ObservedEnqueueReceipt, ThreadPoolObservationPlan,
    ThreadPoolObservationSnapshot, WorkerObservationState,
};
pub use overlap::OverlapScores;
pub use positional::{
    DYN_ROUTER_POSITIONAL_SEARCH_MODE, LevelIndex, PositionalIndexer, SearchMode,
};
pub use pruning::{BlockEntry, PruneConfig, PruneManager, WorkerPruneManager};
pub use radix_tree::RadixTree;
pub use recording::ApproximateRequestLease;
pub use registry::{IndexClaimError, IndexRegistry, IndexerEntry, LifecycleGuard};
pub use residency::{
    ResidencyOwnerKey, ResidencyOwnerKeyExt, ResidencyProjection, ResidencyProjectionError,
    ResidencyRoutingSnapshot, ResolvedRouterHintSource, RouterHintSourceMetadata,
};
pub use session_prefix_index::{
    LogicalNode, NodeId, SessionId, SessionPrefixIndexError, SessionPrefixIndexer,
    SessionResidencyOwner, SharedCacheBackend,
};
pub use session_updates::SessionUpdateSender;
pub use thread_pool::ThreadPoolIndexer;
#[cfg(feature = "bench")]
pub use thread_pool::{
    ThreadPoolObservationDrain, ThreadPoolObservationSession, ThreadPoolSealedObservation,
};
pub use traits::{AnchorCapableSyncIndexer, KvIndexerInterface, ShardSizeSnapshot, SyncIndexer};
pub use types::{
    AnchorRef, AnchorTask, ContainsWorkerBlockRequest, DumpRequest, FlushRequest,
    GetWorkersRequest, KvRouterError, MatchDetails, MatchDetailsRequest, MatchRequest,
    RoutingDecisionHashes, WireLowerTierMatchDetails, WireOverlapScores, WireTieredMatchDetails,
    WorkerLookupStats, WorkerTask,
};
