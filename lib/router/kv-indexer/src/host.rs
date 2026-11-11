// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What an in-tree host names from the KV indexer stage, in one place: the indexer handles it
//! builds and feeds, the registry it opens partitions in, the lookup and recording types it
//! reads back, and the service DTOs. Every item is re-exported from its owning module; this
//! list is the host contract (hosts inventory v2, section 3). Tier 1 never imports it. The DC
//! relay has its own contract: the re-export list at the root of the `cuckoo` module, whose
//! submodules are private.

pub use crate::api::{KvStateProtocolVersion, WorkerKvQueryResponse};
pub use crate::approximate_lru::{
    ApproximateAcquireMode, ApproximateLruBlock, ApproximateLruLease, ApproximateLruStats,
    ApproximateRetentionConfig,
};
pub use crate::backend::{Indexer, IndexerPolicy, RemotePrimary, SideIndexer};
pub use crate::branch_sharded::BranchShardedIndexer;
pub use crate::concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
pub use crate::config::ApproximateCachePolicyKind;
pub use crate::kv_hints::{KvTransferCandidateSource, KvTransferCandidates};
pub use crate::kv_indexer::KvIndexer;
pub use crate::local::LocalKvIndexer;
pub use crate::lower_tier::LowerTierMatchDetails;
pub use crate::lower_tier_indexers::{
    LowerTierIndexers, LowerTierQueryOptions, TieredMatchDetails, query_lower_tiers_with_options,
};
#[cfg(feature = "export")]
pub use crate::metrics::{
    CKF_MUTATION_HELP, CKF_MUTATION_LABELS, KV_CACHE_EVENT_WARNINGS_HELP,
    KV_CACHE_EVENT_WARNINGS_LABELS, KV_CACHE_EVENTS_APPLIED_HELP, KV_CACHE_EVENTS_APPLIED_LABELS,
};
pub use crate::metrics::{
    KvIndexerMetrics, METRIC_EVENT_STORED, METRIC_STATUS_PARENT_NOT_FOUND,
    METRIC_WARNING_DUPLICATE_STORE,
};
#[cfg(feature = "bench")]
pub use crate::observation::{
    ObservationError, ThreadPoolObservationPlan, ThreadPoolObservationSnapshot,
};
pub use crate::overlap::OverlapScores;
pub use crate::positional::PositionalIndexer;
pub use crate::pruning::PruneConfig;
pub use crate::radix_tree::RadixTree;
pub use crate::recording::ApproximateRequestLease;
pub use crate::registry::IndexRegistry;
pub use crate::residency::{
    ResidencyOwnerKeyExt, ResidencyProjection, ResidencyRoutingSnapshot, RouterHintSourceMetadata,
};
pub use crate::session_prefix_index::{
    SessionPrefixIndexer, SessionResidencyOwner, SharedCacheBackend,
};
pub use crate::session_updates::SessionUpdateSender;
pub use crate::thread_pool::ThreadPoolIndexer;
pub use crate::traits::{KvIndexerInterface, SyncIndexer};
pub use crate::types::{
    KvRouterError, MatchDetails, RoutingDecisionHashes, WireTieredMatchDetails,
};

// What the frontend host (lib/llm) names beyond the lists above.
pub use crate::api::{
    IndexerQueryRequest, IndexerQueryResponse, IndexerRecordRoutingDecisionRequest,
    IndexerRecordRoutingDecisionResponse, KV_INDEXER_QUERY_ENDPOINT,
    KV_INDEXER_RECORD_ROUTING_DECISION_ENDPOINT, KvStateAgentIdentity, KvStateAgentStatus,
    KvStateAttachmentStatus, KvStateRecoveryReceipt, WorkerKvQueryKind, WorkerKvQueryRequest,
};
pub use crate::approximate_lru::ApproximateLruIncarnation;
pub use crate::identity::{CanonicalIdentityMaterial, ExplicitIdentityMap, IndexerIdentitySpec};
pub use crate::residency::ResidencyProjectionError;
pub use crate::types::GetWorkersRequest;
