// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pre-split import paths of `dynamo_kv_router`, kept for Dynamo 1.8 and 1.9 and removed in 1.10.
//!
//! Dynamo 1.7 shipped the router as one crate. 1.8 splits it into the vocabulary, stage, kernel,
//! and composition crates under `lib/router`, and every item below now lives in another crate or
//! in another module of this one. This file keeps the 1.7 paths resolving so out-of-tree Rust code
//! keeps compiling through 1.9; the schedule and the migration table are on the deprecations page,
//! <https://docs.nvidia.com/dynamo/reference/releases/deprecations>.
//!
//! Shape: one `pub mod` block per pre-split module, each a named list of re-exports from the owner
//! crate (the comment at the top of a block names the owners). A plain struct is a
//! `#[deprecated]` type alias, so a use site warns with the new path; traits, functions,
//! constants, enums, generic and tuple types, and modules are plain `pub use` lines, because an
//! alias would break `Enum::*` imports, tuple-struct constructors, and generic parameters. An item
//! that exists only behind a feature carries the same `#[cfg]` as its owner. `lib.rs` mounts the
//! root modules and the root re-exports; `scheduling`, `scheduling::config`, `services`,
//! `services::selection`, and `test_utils` still exist and mount their blocks from their own
//! files.
//!
//! Gone with no replacement here: the dummy shared-cache server (`services::shared_cache`) and
//! `protocols::ActiveLoad`, which lives in `dynamo_llm::kv_router::ActiveLoad` (this crate cannot
//! depend on `dynamo-llm`).

/// Pre-split `dynamo_kv_router::conditional_disagg`; its items live in `dynamo_worker_selection` now.
pub mod conditional_disagg {
    // Owner: dynamo_worker_selection (9)
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_worker_selection::conditional_disagg::ConditionalDisaggDecisionInput; removed in 1.10.0"
    )]
    pub type ConditionalDisaggDecisionInput =
        dynamo_worker_selection::conditional_disagg::ConditionalDisaggDecisionInput;
    pub use dynamo_worker_selection::conditional_disagg::ConditionalDisaggPolicy;
    pub use dynamo_worker_selection::conditional_disagg::DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD;
    pub use dynamo_worker_selection::conditional_disagg::DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_worker_selection::conditional_disagg::IslBoundingPolicy; removed in 1.10.0"
    )]
    pub type IslBoundingPolicy = dynamo_worker_selection::conditional_disagg::IslBoundingPolicy;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_worker_selection::conditional_disagg::IslOrLoadPolicy; removed in 1.10.0"
    )]
    pub type IslOrLoadPolicy = dynamo_worker_selection::conditional_disagg::IslOrLoadPolicy;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_worker_selection::conditional_disagg::PrefillLoadPolicy; removed in 1.10.0"
    )]
    pub type PrefillLoadPolicy = dynamo_worker_selection::conditional_disagg::PrefillLoadPolicy;
    pub use dynamo_worker_selection::conditional_disagg::make_conditional_disagg_policy;
    pub use dynamo_worker_selection::conditional_disagg::policy_needs_prefill_worker_busy;
}

/// Pre-split `dynamo_kv_router::identity`; its items live in `dynamo_router_protocol` and `dynamo_kv_indexer` now.
pub mod identity {
    // Owner: dynamo_router_protocol (12), dynamo_kv_indexer (7)
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_router_protocol::identity::CacheOwnerId; removed in 1.10.0"
    )]
    pub type CacheOwnerId = dynamo_router_protocol::identity::CacheOwnerId;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_router_protocol::identity::CacheSemanticsId; removed in 1.10.0"
    )]
    pub type CacheSemanticsId = dynamo_router_protocol::identity::CacheSemanticsId;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::identity::CanonicalIdentityMaterial; removed in 1.10.0"
    )]
    pub type CanonicalIdentityMaterial = dynamo_kv_indexer::identity::CanonicalIdentityMaterial;
    pub use dynamo_router_protocol::identity::ClusterId;
    pub use dynamo_router_protocol::identity::DEFAULT_ROUTING_GROUP;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::identity::ExplicitIdentityMap; removed in 1.10.0"
    )]
    pub type ExplicitIdentityMap = dynamo_kv_indexer::identity::ExplicitIdentityMap;
    pub use dynamo_kv_indexer::identity::IdentitySpecError;
    pub use dynamo_router_protocol::identity::IdentityParseError;
    pub use dynamo_router_protocol::identity::IdentitySource;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_router_protocol::identity::IndexerDomainId; removed in 1.10.0"
    )]
    pub type IndexerDomainId = dynamo_router_protocol::identity::IndexerDomainId;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::identity::IndexerIdentitySpec; removed in 1.10.0"
    )]
    pub type IndexerIdentitySpec = dynamo_kv_indexer::identity::IndexerIdentitySpec;
    pub use dynamo_kv_indexer::identity::MAX_EXPLICIT_IDENTITY_ENTRIES;
    pub use dynamo_kv_indexer::identity::MAX_EXPLICIT_IDENTITY_KEY_BYTES;
    pub use dynamo_kv_indexer::identity::MAX_EXPLICIT_IDENTITY_VALUE_BYTES;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_router_protocol::identity::PoolId; removed in 1.10.0"
    )]
    pub type PoolId = dynamo_router_protocol::identity::PoolId;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_router_protocol::identity::RoutingPartitionId; removed in 1.10.0"
    )]
    pub type RoutingPartitionId = dynamo_router_protocol::identity::RoutingPartitionId;
    pub use dynamo_router_protocol::identity::RoutingPartitionRef;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_router_protocol::identity::RoutingScopeId; removed in 1.10.0"
    )]
    pub type RoutingScopeId = dynamo_router_protocol::identity::RoutingScopeId;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_router_protocol::identity::StableDpSlotId; removed in 1.10.0"
    )]
    pub type StableDpSlotId = dynamo_router_protocol::identity::StableDpSlotId;
}

/// Pre-split `dynamo_kv_router::indexer`; its items live in `dynamo_kv_indexer` and `dynamo_scheduler` now.
pub mod indexer {
    // Owner: dynamo_kv_indexer (37), dynamo_scheduler (1)
    /// Pre-split `dynamo_kv_router::indexer::concurrent_radix_tree_compressed`; its items live in `dynamo_kv_indexer` now.
    pub mod concurrent_radix_tree_compressed {
        // Owner: dynamo_kv_indexer (1)
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_kv_indexer::concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed; removed in 1.10.0"
        )]
        pub type ConcurrentRadixTreeCompressed =
            dynamo_kv_indexer::concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
    }

    /// Pre-split `dynamo_kv_router::indexer::positional`; its items live in `dynamo_kv_indexer` now.
    pub mod positional {
        // Owner: dynamo_kv_indexer (4)
        pub use dynamo_kv_indexer::positional::DYN_ROUTER_POSITIONAL_SEARCH_MODE;
        pub use dynamo_kv_indexer::positional::LevelIndex;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_kv_indexer::positional::PositionalIndexer; removed in 1.10.0"
        )]
        pub type PositionalIndexer = dynamo_kv_indexer::positional::PositionalIndexer;
        pub use dynamo_kv_indexer::positional::SearchMode;
    }

    /// Pre-split `dynamo_kv_router::indexer::pruning`; its items live in `dynamo_kv_indexer` now.
    pub mod pruning {
        // Owner: dynamo_kv_indexer (4)
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_kv_indexer::pruning::BlockEntry; removed in 1.10.0"
        )]
        pub type BlockEntry = dynamo_kv_indexer::pruning::BlockEntry;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_kv_indexer::pruning::PruneConfig; removed in 1.10.0"
        )]
        pub type PruneConfig = dynamo_kv_indexer::pruning::PruneConfig;
        pub use dynamo_kv_indexer::pruning::PruneManager;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_kv_indexer::pruning::WorkerPruneManager; removed in 1.10.0"
        )]
        pub type WorkerPruneManager = dynamo_kv_indexer::pruning::WorkerPruneManager;
    }

    /// Pre-split `dynamo_kv_router::indexer::radix_tree`; its items live in `dynamo_kv_indexer` now.
    pub mod radix_tree {
        // Owner: dynamo_kv_indexer (1)
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_kv_indexer::radix_tree::RadixTree; removed in 1.10.0"
        )]
        pub type RadixTree = dynamo_kv_indexer::radix_tree::RadixTree;
    }

    #[allow(deprecated)]
    pub use dynamo_kv_indexer::AnchorAwareBranchShardedIndexer;
    pub use dynamo_kv_indexer::AnchorCapableSyncIndexer;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::AnchorRef; removed in 1.10.0"
    )]
    pub type AnchorRef = dynamo_kv_indexer::AnchorRef;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::AnchorTask; removed in 1.10.0"
    )]
    pub type AnchorTask = dynamo_kv_indexer::AnchorTask;
    pub use dynamo_kv_indexer::ApproximateAcquireMode;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::ApproximateLruBlock; removed in 1.10.0"
    )]
    pub type ApproximateLruBlock = dynamo_kv_indexer::ApproximateLruBlock;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::ApproximateLruClient; removed in 1.10.0"
    )]
    pub type ApproximateLruClient = dynamo_kv_indexer::ApproximateLruClient;
    pub use dynamo_kv_indexer::ApproximateLruIncarnation;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::ApproximateLruLease; removed in 1.10.0"
    )]
    pub type ApproximateLruLease = dynamo_kv_indexer::ApproximateLruLease;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::ApproximateLruReleaseAck; removed in 1.10.0"
    )]
    pub type ApproximateLruReleaseAck = dynamo_kv_indexer::ApproximateLruReleaseAck;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::ApproximateLruStats; removed in 1.10.0"
    )]
    pub type ApproximateLruStats = dynamo_kv_indexer::ApproximateLruStats;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::ApproximateLruTask; removed in 1.10.0"
    )]
    pub type ApproximateLruTask = dynamo_kv_indexer::ApproximateLruTask;
    pub use dynamo_kv_indexer::ApproximateRetentionConfig;
    pub use dynamo_kv_indexer::BranchShardedIndexer;
    pub use dynamo_kv_indexer::CkfMutationKind;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::ContainsWorkerBlockRequest; removed in 1.10.0"
    )]
    pub type ContainsWorkerBlockRequest = dynamo_kv_indexer::ContainsWorkerBlockRequest;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::DumpRequest; removed in 1.10.0"
    )]
    pub type DumpRequest = dynamo_kv_indexer::DumpRequest;
    #[cfg(feature = "bench")]
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::EventCompletion; removed in 1.10.0"
    )]
    pub type EventCompletion = dynamo_kv_indexer::EventCompletion;
    #[cfg(feature = "bench")]
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::EventCompletionBuffer; removed in 1.10.0"
    )]
    pub type EventCompletionBuffer = dynamo_kv_indexer::EventCompletionBuffer;
    #[cfg(feature = "bench")]
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::EventCompletionWriter; removed in 1.10.0"
    )]
    pub type EventCompletionWriter = dynamo_kv_indexer::EventCompletionWriter;
    pub use dynamo_kv_indexer::EventKind;
    pub use dynamo_kv_indexer::EventWarningKind;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::FlushRequest; removed in 1.10.0"
    )]
    pub type FlushRequest = dynamo_kv_indexer::FlushRequest;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::GetWorkersRequest; removed in 1.10.0"
    )]
    pub type GetWorkersRequest = dynamo_kv_indexer::GetWorkersRequest;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::IndexerQueryRequest; removed in 1.10.0"
    )]
    pub type IndexerQueryRequest = dynamo_kv_indexer::IndexerQueryRequest;
    pub use dynamo_kv_indexer::IndexerQueryResponse;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::IndexerRecordRoutingDecisionRequest; removed in 1.10.0"
    )]
    pub type IndexerRecordRoutingDecisionRequest =
        dynamo_kv_indexer::IndexerRecordRoutingDecisionRequest;
    pub use dynamo_kv_indexer::IndexerRecordRoutingDecisionResponse;
    pub use dynamo_kv_indexer::KV_INDEXER_QUERY_ENDPOINT;
    pub use dynamo_kv_indexer::KV_INDEXER_RECORD_ROUTING_DECISION_ENDPOINT;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::KvEventSender; removed in 1.10.0"
    )]
    pub type KvEventSender = dynamo_kv_indexer::KvEventSender;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::KvIndexer; removed in 1.10.0"
    )]
    pub type KvIndexer = dynamo_kv_indexer::KvIndexer;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::KvIndexerBuilder; removed in 1.10.0"
    )]
    pub type KvIndexerBuilder = dynamo_kv_indexer::KvIndexerBuilder;
    pub use dynamo_kv_indexer::KvIndexerInterface;
    pub use dynamo_kv_indexer::delegate::KvIndexerDelegate;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::KvIndexerMetrics; removed in 1.10.0"
    )]
    pub type KvIndexerMetrics = dynamo_kv_indexer::KvIndexerMetrics;
    pub use dynamo_kv_indexer::KvRouterError;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::KvStateAgentIdentity; removed in 1.10.0"
    )]
    pub type KvStateAgentIdentity = dynamo_kv_indexer::KvStateAgentIdentity;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::KvStateAgentStatus; removed in 1.10.0"
    )]
    pub type KvStateAgentStatus = dynamo_kv_indexer::KvStateAgentStatus;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::KvStateAttachmentStatus; removed in 1.10.0"
    )]
    pub type KvStateAttachmentStatus = dynamo_kv_indexer::KvStateAttachmentStatus;
    pub use dynamo_kv_indexer::KvStateProtocolVersion;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::KvStateRecoveryReceipt; removed in 1.10.0"
    )]
    pub type KvStateRecoveryReceipt = dynamo_kv_indexer::KvStateRecoveryReceipt;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::KvTransferExtensions; removed in 1.10.0"
    )]
    pub type KvTransferExtensions = dynamo_kv_indexer::KvTransferExtensions;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::LocalKvIndexer; removed in 1.10.0"
    )]
    pub type LocalKvIndexer = dynamo_kv_indexer::LocalKvIndexer;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::LowerTierContinuation; removed in 1.10.0"
    )]
    pub type LowerTierContinuation = dynamo_kv_indexer::LowerTierContinuation;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::LowerTierIndexer; removed in 1.10.0"
    )]
    pub type LowerTierIndexer = dynamo_kv_indexer::LowerTierIndexer;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::LowerTierIndexers; removed in 1.10.0"
    )]
    pub type LowerTierIndexers = dynamo_kv_indexer::LowerTierIndexers;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::LowerTierMatchDetails; removed in 1.10.0"
    )]
    pub type LowerTierMatchDetails = dynamo_kv_indexer::LowerTierMatchDetails;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::LowerTierQueryOptions; removed in 1.10.0"
    )]
    pub type LowerTierQueryOptions = dynamo_kv_indexer::LowerTierQueryOptions;
    pub use dynamo_kv_indexer::METRIC_CKF_MUTATION_CAPACITY_EXHAUSTED;
    pub use dynamo_kv_indexer::METRIC_CKF_MUTATION_UNKNOWN_REMOVE;
    pub use dynamo_kv_indexer::METRIC_EVENT_CLEARED;
    pub use dynamo_kv_indexer::METRIC_EVENT_REMOVED;
    pub use dynamo_kv_indexer::METRIC_EVENT_STORED;
    pub use dynamo_kv_indexer::METRIC_STATUS_ALLOCATION_FAILED;
    pub use dynamo_kv_indexer::METRIC_STATUS_BLOCK_NOT_FOUND;
    pub use dynamo_kv_indexer::METRIC_STATUS_CAPACITY_EXHAUSTED;
    pub use dynamo_kv_indexer::METRIC_STATUS_INDEXER_INVARIANT_VIOLATION;
    pub use dynamo_kv_indexer::METRIC_STATUS_INVALID_BLOCK;
    pub use dynamo_kv_indexer::METRIC_STATUS_OK;
    pub use dynamo_kv_indexer::METRIC_STATUS_OWNERSHIP_DEGREE_OVERFLOW;
    pub use dynamo_kv_indexer::METRIC_STATUS_PARENT_NOT_FOUND;
    pub use dynamo_kv_indexer::METRIC_STATUS_UNSUPPORTED_RESIDENCY_DOMAIN;
    pub use dynamo_kv_indexer::METRIC_WARNING_DUPLICATE_STORE;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::MatchDetails; removed in 1.10.0"
    )]
    pub type MatchDetails = dynamo_kv_indexer::MatchDetails;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::MatchDetailsRequest; removed in 1.10.0"
    )]
    pub type MatchDetailsRequest = dynamo_kv_indexer::MatchDetailsRequest;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::MatchRequest; removed in 1.10.0"
    )]
    pub type MatchRequest = dynamo_kv_indexer::MatchRequest;
    pub use dynamo_kv_indexer::MaybeError;
    #[cfg(feature = "bench")]
    pub use dynamo_kv_indexer::ObservationError;
    #[cfg(feature = "bench")]
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::ObservationSeal; removed in 1.10.0"
    )]
    pub type ObservationSeal = dynamo_kv_indexer::ObservationSeal;
    #[cfg(feature = "bench")]
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::ObservedEnqueueReceipt; removed in 1.10.0"
    )]
    pub type ObservedEnqueueReceipt = dynamo_kv_indexer::ObservedEnqueueReceipt;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::PreBoundEventCounters; removed in 1.10.0"
    )]
    pub type PreBoundEventCounters = dynamo_kv_indexer::PreBoundEventCounters;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::radix_tree::RadixTree; removed in 1.10.0"
    )]
    pub type RadixTree = dynamo_kv_indexer::radix_tree::RadixTree;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::RoutingDecisionHashes; removed in 1.10.0"
    )]
    pub type RoutingDecisionHashes = dynamo_kv_indexer::RoutingDecisionHashes;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::ShardSizeSnapshot; removed in 1.10.0"
    )]
    pub type ShardSizeSnapshot = dynamo_kv_indexer::ShardSizeSnapshot;
    pub use dynamo_kv_indexer::SyncIndexer;
    pub use dynamo_kv_indexer::ThreadPoolIndexer;
    #[cfg(feature = "bench")]
    pub use dynamo_kv_indexer::ThreadPoolObservationDrain;
    pub use dynamo_scheduler::shared_cache::SharedKvCache;
    #[cfg(feature = "bench")]
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::ThreadPoolObservationPlan; removed in 1.10.0"
    )]
    pub type ThreadPoolObservationPlan = dynamo_kv_indexer::ThreadPoolObservationPlan;
    #[cfg(feature = "bench")]
    pub use dynamo_kv_indexer::ThreadPoolObservationSession;
    #[cfg(feature = "bench")]
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::ThreadPoolObservationSnapshot; removed in 1.10.0"
    )]
    pub type ThreadPoolObservationSnapshot = dynamo_kv_indexer::ThreadPoolObservationSnapshot;
    #[cfg(feature = "bench")]
    pub use dynamo_kv_indexer::ThreadPoolSealedObservation;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::TieredMatchDetails; removed in 1.10.0"
    )]
    pub type TieredMatchDetails = dynamo_kv_indexer::TieredMatchDetails;
    pub use dynamo_kv_indexer::TieredMatchProvider;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::WireLowerTierMatchDetails; removed in 1.10.0"
    )]
    pub type WireLowerTierMatchDetails = dynamo_kv_indexer::WireLowerTierMatchDetails;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::WireOverlapScores; removed in 1.10.0"
    )]
    pub type WireOverlapScores = dynamo_kv_indexer::WireOverlapScores;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::WireTieredMatchDetails; removed in 1.10.0"
    )]
    pub type WireTieredMatchDetails = dynamo_kv_indexer::WireTieredMatchDetails;
    pub use dynamo_kv_indexer::WorkerKvQueryKind;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::WorkerKvQueryRequest; removed in 1.10.0"
    )]
    pub type WorkerKvQueryRequest = dynamo_kv_indexer::WorkerKvQueryRequest;
    pub use dynamo_kv_indexer::WorkerKvQueryResponse;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::WorkerLookupStats; removed in 1.10.0"
    )]
    pub type WorkerLookupStats = dynamo_kv_indexer::WorkerLookupStats;
    #[cfg(feature = "bench")]
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::WorkerObservationState; removed in 1.10.0"
    )]
    pub type WorkerObservationState = dynamo_kv_indexer::WorkerObservationState;
    pub use dynamo_kv_indexer::WorkerTask;
    pub use dynamo_kv_indexer::lower_tier_query_order;
    pub use dynamo_kv_indexer::query_lower_tiers_with_options;
    pub use dynamo_kv_indexer::query_lower_tiers_with_options_and_snapshot;
    pub use dynamo_kv_indexer::record_unsupported_residency_event;
}

/// Pre-split `dynamo_kv_router::kv_hints`; its items live in `dynamo_kv_hints` and `dynamo_kv_indexer` now.
pub mod kv_hints {
    // Owner: dynamo_kv_hints (12), dynamo_kv_indexer (4)
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::kv_hints::CacheOwnerKvTransferKeys; removed in 1.10.0"
    )]
    pub type CacheOwnerKvTransferKeys = dynamo_kv_indexer::kv_hints::CacheOwnerKvTransferKeys;
    pub use dynamo_kv_hints::wire::KV_HINT_TRANSFER_CAPABILITY_KEY;
    pub use dynamo_kv_hints::wire::KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY;
    pub use dynamo_kv_hints::wire::KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_hints::wire::KvHint; removed in 1.10.0"
    )]
    pub type KvHint = dynamo_kv_hints::wire::KvHint;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_hints::wire::KvHintAction; removed in 1.10.0"
    )]
    pub type KvHintAction = dynamo_kv_hints::wire::KvHintAction;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_hints::wire::KvSourceLocationsPayload; removed in 1.10.0"
    )]
    pub type KvSourceLocationsPayload = dynamo_kv_hints::wire::KvSourceLocationsPayload;
    pub use dynamo_kv_indexer::kv_hints::KvTransferCandidateSource;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::kv_hints::KvTransferCandidates; removed in 1.10.0"
    )]
    pub type KvTransferCandidates = dynamo_kv_indexer::kv_hints::KvTransferCandidates;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_hints::transfer::KvTransferScope; removed in 1.10.0"
    )]
    pub type KvTransferScope = dynamo_kv_hints::transfer::KvTransferScope;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_hints::transfer::KvTransferSourceChoice; removed in 1.10.0"
    )]
    pub type KvTransferSourceChoice = dynamo_kv_hints::transfer::KvTransferSourceChoice;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::kv_hints::KvTransferSourceInterval; removed in 1.10.0"
    )]
    pub type KvTransferSourceInterval = dynamo_kv_indexer::kv_hints::KvTransferSourceInterval;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_hints::transfer::KvTransferSourceSummary; removed in 1.10.0"
    )]
    pub type KvTransferSourceSummary = dynamo_kv_hints::transfer::KvTransferSourceSummary;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_hints::transfer::KvTransferTarget; removed in 1.10.0"
    )]
    pub type KvTransferTarget = dynamo_kv_hints::transfer::KvTransferTarget;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_hints::transfer::WorkerKvTransferKeys; removed in 1.10.0"
    )]
    pub type WorkerKvTransferKeys = dynamo_kv_hints::transfer::WorkerKvTransferKeys;
    pub use dynamo_kv_hints::keys::WorkerKvTransferKeysCache;
}

/// Pre-split `dynamo_kv_router::protocols`; its items live in `dynamo_kv_events`, `dynamo_kv_hashing`, `dynamo_slot_tracker`, `dynamo_scheduler`, `dynamo_kv_indexer`, `dynamo_worker_selection`, `dynamo_router_protocol`, `dynamo_kv_hints` and `dynamo_session_affinity` now.
pub mod protocols {
    // Owner: dynamo_kv_events (18), dynamo_kv_hashing (17), dynamo_slot_tracker (12), dynamo_scheduler (8), dynamo_kv_indexer (7), dynamo_worker_selection (4), dynamo_router_protocol (3), dynamo_kv_hints (2), dynamo_session_affinity (1)
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_slot_tracker::wire::ActiveBlockEvent; removed in 1.10.0"
    )]
    pub type ActiveBlockEvent = dynamo_slot_tracker::wire::ActiveBlockEvent;
    pub use dynamo_slot_tracker::wire::ActiveBlockEventData;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_slot_tracker::wire::ActiveSequenceEvent; removed in 1.10.0"
    )]
    pub type ActiveSequenceEvent = dynamo_slot_tracker::wire::ActiveSequenceEvent;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_slot_tracker::wire::ActiveSequenceEventBatch; removed in 1.10.0"
    )]
    pub type ActiveSequenceEventBatch = dynamo_slot_tracker::wire::ActiveSequenceEventBatch;
    pub use dynamo_slot_tracker::wire::ActiveSequenceEventData;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_hashing::multimodal::BlockExtraInfo; removed in 1.10.0"
    )]
    pub type BlockExtraInfo = dynamo_kv_hashing::multimodal::BlockExtraInfo;
    pub use dynamo_kv_hashing::routing::BlockHashOptions;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_hashing::multimodal::BlockMmObjectInfo; removed in 1.10.0"
    )]
    pub type BlockMmObjectInfo = dynamo_kv_hashing::multimodal::BlockMmObjectInfo;
    pub use dynamo_kv_events::event::KV_EVENT_SUBJECT;
    pub use dynamo_kv_hashing::ExternalSequenceBlockHash;
    pub use dynamo_router_protocol::worker::DpRank;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_events::event::KvCacheEvent; removed in 1.10.0"
    )]
    pub type KvCacheEvent = dynamo_kv_events::event::KvCacheEvent;
    pub use dynamo_kv_events::event::KvCacheEventData;
    pub use dynamo_kv_events::event::KvCacheEventError;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_events::event::KvCacheEvents; removed in 1.10.0"
    )]
    pub type KvCacheEvents = dynamo_kv_events::event::KvCacheEvents;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_events::event::KvCacheRemoveData; removed in 1.10.0"
    )]
    pub type KvCacheRemoveData = dynamo_kv_events::event::KvCacheRemoveData;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_events::event::KvCacheStoreData; removed in 1.10.0"
    )]
    pub type KvCacheStoreData = dynamo_kv_events::event::KvCacheStoreData;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_events::event::KvCacheStoredBlockData; removed in 1.10.0"
    )]
    pub type KvCacheStoredBlockData = dynamo_kv_events::event::KvCacheStoredBlockData;
    pub use dynamo_kv_hashing::LocalBlockHash;
    pub use dynamo_kv_hashing::multimodal::MM_PAD_HASH_MASK;
    pub use dynamo_kv_hashing::multimodal::MM_PAD_SHIFT_VALUE;
    pub use dynamo_kv_hints::keys::KvHintTransferWorkerMetadata;
    pub use dynamo_kv_hints::keys::KvTransferEnforcement;
    pub use dynamo_slot_tracker::wire::MAX_REPLICA_BATCH_DURATION;
    pub use dynamo_slot_tracker::wire::MAX_REPLICA_BATCH_EVENTS;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::overlap::OverlapScores; removed in 1.10.0"
    )]
    pub type OverlapScores = dynamo_kv_indexer::overlap::OverlapScores;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_events::placement::Placement; removed in 1.10.0"
    )]
    pub type Placement = dynamo_kv_events::placement::Placement;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_events::placement::PlacementEvent; removed in 1.10.0"
    )]
    pub type PlacementEvent = dynamo_kv_events::placement::PlacementEvent;
    pub use dynamo_kv_events::placement::PlacementOwner;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::protocols::PotentialLoad; removed in 1.10.0"
    )]
    pub type PotentialLoad = dynamo_scheduler::protocols::PotentialLoad;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_slot_tracker::wire::PrefillEvent; removed in 1.10.0"
    )]
    pub type PrefillEvent = dynamo_slot_tracker::wire::PrefillEvent;
    pub use dynamo_slot_tracker::wire::PrefillEventData;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_slot_tracker::wire::PrefillLoadHint; removed in 1.10.0"
    )]
    pub type PrefillLoadHint = dynamo_slot_tracker::wire::PrefillLoadHint;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_hashing::multimodal::RequestExtraInfo; removed in 1.10.0"
    )]
    pub type RequestExtraInfo = dynamo_kv_hashing::multimodal::RequestExtraInfo;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_hashing::multimodal::RequestMmObjectInfo; removed in 1.10.0"
    )]
    pub type RequestMmObjectInfo = dynamo_kv_hashing::multimodal::RequestMmObjectInfo;
    pub use dynamo_kv_events::residency::ResetScope;
    pub use dynamo_kv_events::residency::ResidencyDomain;
    pub use dynamo_kv_events::residency::ResidencyOwner;
    pub use dynamo_kv_indexer::residency::ResidencyOwnerKey;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::residency::ResidencyProjection; removed in 1.10.0"
    )]
    pub type ResidencyProjection = dynamo_kv_indexer::residency::ResidencyProjection;
    pub use dynamo_kv_indexer::residency::ResidencyProjectionError;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::residency::ResidencyRoutingSnapshot; removed in 1.10.0"
    )]
    pub type ResidencyRoutingSnapshot = dynamo_kv_indexer::residency::ResidencyRoutingSnapshot;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::residency::ResolvedRouterHintSource; removed in 1.10.0"
    )]
    pub type ResolvedRouterHintSource = dynamo_kv_indexer::residency::ResolvedRouterHintSource;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_events::event::RouterEvent; removed in 1.10.0"
    )]
    pub type RouterEvent = dynamo_kv_events::event::RouterEvent;
    pub use dynamo_scheduler::protocols::RouterEventSink;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::residency::RouterHintSourceMetadata; removed in 1.10.0"
    )]
    pub type RouterHintSourceMetadata = dynamo_kv_indexer::residency::RouterHintSourceMetadata;
    pub use dynamo_scheduler::protocols::RouterRequest;
    pub use dynamo_scheduler::protocols::RouterResponse;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::protocols::RoutingConstraints; removed in 1.10.0"
    )]
    pub type RoutingConstraints = dynamo_scheduler::protocols::RoutingConstraints;
    pub use dynamo_slot_tracker::wire::SchedulerGroup;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_slot_tracker::wire::SchedulerLoad; removed in 1.10.0"
    )]
    pub type SchedulerLoad = dynamo_slot_tracker::wire::SchedulerLoad;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::protocols::SelectionLogFields; removed in 1.10.0"
    )]
    pub type SelectionLogFields = dynamo_scheduler::protocols::SelectionLogFields;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_worker_selection::SelectionTrace; removed in 1.10.0"
    )]
    pub type SelectionTrace = dynamo_worker_selection::SelectionTrace;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_worker_selection::SelectionTraceRow; removed in 1.10.0"
    )]
    pub type SelectionTraceRow = dynamo_worker_selection::SelectionTraceRow;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_worker_selection::SelectionTraceSpec; removed in 1.10.0"
    )]
    pub type SelectionTraceSpec = dynamo_worker_selection::SelectionTraceSpec;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_worker_selection::SharedCacheHits; removed in 1.10.0"
    )]
    pub type SharedCacheHits = dynamo_worker_selection::SharedCacheHits;
    pub use dynamo_kv_events::residency::StorageTier;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_hashing::routing::TokensWithHashes; removed in 1.10.0"
    )]
    pub type TokensWithHashes = dynamo_kv_hashing::routing::TokensWithHashes;
    pub use dynamo_kv_events::residency::UnsupportedResidencyDomain;
    pub use dynamo_kv_events::residency::WireResidencyDomain;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_session_affinity::WorkerAffinityTarget; removed in 1.10.0"
    )]
    pub type WorkerAffinityTarget = dynamo_session_affinity::WorkerAffinityTarget;
    pub use dynamo_router_protocol::worker::WorkerId;
    pub use dynamo_scheduler::protocols::WorkerConfigLike;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::protocols::WorkerSelectionResult; removed in 1.10.0"
    )]
    pub type WorkerSelectionResult = dynamo_scheduler::protocols::WorkerSelectionResult;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_router_protocol::worker::WorkerWithDpRank; removed in 1.10.0"
    )]
    pub type WorkerWithDpRank = dynamo_router_protocol::worker::WorkerWithDpRank;
    pub use dynamo_kv_hashing::multimodal::hash_mm_identifier;
    pub use dynamo_kv_hashing::multimodal::pad_value_for_mm_hash;
    pub use dynamo_kv_hashing::routing::XXH3_SEED;
    pub use dynamo_kv_hashing::routing::compute_block_hash;
    pub use dynamo_kv_hashing::routing::compute_block_hash_for_seq;
    pub use dynamo_kv_hashing::routing::compute_next_seq_hash;
    pub use dynamo_kv_hashing::routing::compute_seq_hash_for_block;
}

/// Pre-split `dynamo_kv_router::recovery`; its items live in `dynamo_kv_events` now.
pub mod recovery {
    // Owner: dynamo_kv_events (2)
    pub use dynamo_kv_events::cursor::CursorObservation;
    pub use dynamo_kv_events::cursor::CursorState;
}

/// Pre-split `dynamo_kv_router::sequences`; its items live in `dynamo_slot_tracker` now.
pub mod sequences {
    // Owner: dynamo_slot_tracker (19)
    /// Pre-split `dynamo_kv_router::sequences::multi_worker`; its items live in `dynamo_slot_tracker` now.
    pub mod multi_worker {
        // Owner: dynamo_slot_tracker (13)
        pub use dynamo_slot_tracker::ActiveSequencesMultiWorker;
        pub use dynamo_slot_tracker::LifecycleMutationOutcome;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_slot_tracker::LocalWorkerLoad; removed in 1.10.0"
        )]
        pub type LocalWorkerLoad = dynamo_slot_tracker::LocalWorkerLoad;
        pub use dynamo_slot_tracker::NoopSequencePublisher;
        pub use dynamo_slot_tracker::ReplicaRequestLeaseObserver;
        pub use dynamo_slot_tracker::ReplicaWorkerPolicy;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_slot_tracker::SchedulerLoadSnapshot; removed in 1.10.0"
        )]
        pub type SchedulerLoadSnapshot = dynamo_slot_tracker::SchedulerLoadSnapshot;
        pub use dynamo_slot_tracker::SequenceError;
        pub use dynamo_slot_tracker::SequencePublishQueueError;
        pub use dynamo_slot_tracker::SequencePublisher;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_slot_tracker::SequenceRequest; removed in 1.10.0"
        )]
        pub type SequenceRequest = dynamo_slot_tracker::SequenceRequest;
        pub use dynamo_slot_tracker::SequenceSubscriber;
        pub use dynamo_slot_tracker::active_request_expiry_duration;
    }

    /// Pre-split `dynamo_kv_router::sequences::single`; its items live in `dynamo_slot_tracker` now.
    pub mod single {
        // Owner: dynamo_slot_tracker (3)
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_slot_tracker::ActiveSequences; removed in 1.10.0"
        )]
        pub type ActiveSequences = dynamo_slot_tracker::ActiveSequences;
        pub use dynamo_slot_tracker::DEFAULT_ACTIVE_REQUEST_EXPIRY_DURATION;
        pub use dynamo_slot_tracker::RequestId;
    }

    /// Pre-split `dynamo_kv_router::sequences::topology`; its items live in `dynamo_slot_tracker` now.
    pub mod topology {
        // Owner: dynamo_slot_tracker (3)
        pub use dynamo_slot_tracker::topology::MAX_DATA_PARALLEL_RANKS_PER_WORKER;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_slot_tracker::topology::WorkerDpRange; removed in 1.10.0"
        )]
        pub type WorkerDpRange = dynamo_slot_tracker::topology::WorkerDpRange;
        pub use dynamo_slot_tracker::topology::WorkerTopologyError;
    }

    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_slot_tracker::ActiveSequences; removed in 1.10.0"
    )]
    pub type ActiveSequences = dynamo_slot_tracker::ActiveSequences;
    pub use dynamo_slot_tracker::ActiveSequencesMultiWorker;
    pub use dynamo_slot_tracker::DEFAULT_ACTIVE_REQUEST_EXPIRY_DURATION;
    pub use dynamo_slot_tracker::LifecycleMutationOutcome;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_slot_tracker::LocalWorkerLoad; removed in 1.10.0"
    )]
    pub type LocalWorkerLoad = dynamo_slot_tracker::LocalWorkerLoad;
    pub use dynamo_slot_tracker::NoopSequencePublisher;
    pub use dynamo_slot_tracker::PotentialLoadMaps;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_slot_tracker::PrefillTokenDeltas; removed in 1.10.0"
    )]
    pub type PrefillTokenDeltas = dynamo_slot_tracker::PrefillTokenDeltas;
    pub use dynamo_slot_tracker::ReplicaRequestLeaseObserver;
    pub use dynamo_slot_tracker::ReplicaWorkerPolicy;
    pub use dynamo_slot_tracker::RequestId;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_slot_tracker::SchedulerLoadSnapshot; removed in 1.10.0"
    )]
    pub type SchedulerLoadSnapshot = dynamo_slot_tracker::SchedulerLoadSnapshot;
    pub use dynamo_slot_tracker::SequenceError;
    pub use dynamo_slot_tracker::SequencePublishQueueError;
    pub use dynamo_slot_tracker::SequencePublisher;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_slot_tracker::SequenceRequest; removed in 1.10.0"
    )]
    pub type SequenceRequest = dynamo_slot_tracker::SequenceRequest;
    pub use dynamo_slot_tracker::SequenceSubscriber;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_slot_tracker::WorkerLoadProjection; removed in 1.10.0"
    )]
    pub type WorkerLoadProjection = dynamo_slot_tracker::WorkerLoadProjection;
    pub use dynamo_slot_tracker::active_request_expiry_duration;
}

/// Pre-split `dynamo_kv_router::session_prefix_index`; its items live in `dynamo_kv_indexer` now.
pub mod session_prefix_index {
    // Owner: dynamo_kv_indexer (7)
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::session_prefix_index::LogicalNode; removed in 1.10.0"
    )]
    pub type LogicalNode = dynamo_kv_indexer::session_prefix_index::LogicalNode;
    pub use dynamo_kv_indexer::session_prefix_index::NodeId;
    pub use dynamo_kv_indexer::session_prefix_index::SessionId;
    pub use dynamo_kv_indexer::session_prefix_index::SessionPrefixIndexError;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_indexer::session_prefix_index::SessionPrefixIndexer; removed in 1.10.0"
    )]
    pub type SessionPrefixIndexer = dynamo_kv_indexer::session_prefix_index::SessionPrefixIndexer;
    pub use dynamo_kv_indexer::session_prefix_index::SessionResidencyOwner;
    pub use dynamo_kv_indexer::session_prefix_index::SharedCacheBackend;
}

/// Pre-split `dynamo_kv_router::worker_type`; its items live in `dynamo_router_protocol` now.
pub mod worker_type {
    // Owner: dynamo_router_protocol (2)
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_router_protocol::worker_type::ParseWorkerTypeError; removed in 1.10.0"
    )]
    pub type ParseWorkerTypeError = dynamo_router_protocol::worker_type::ParseWorkerTypeError;
    pub use dynamo_router_protocol::worker_type::WorkerType;
}

/// Pre-split `dynamo_kv_router::zmq_wire`; its items live in `dynamo_kv_events` and `dynamo_kv_hashing` now.
pub mod zmq_wire {
    // Owner: dynamo_kv_events (22), dynamo_kv_hashing (2)
    pub use dynamo_kv_events::zmq_wire::BlockHashValue;
    pub use dynamo_kv_events::zmq_wire::DYNAMO_CACHE_SALT_PREFIX;
    pub use dynamo_kv_events::zmq_wire::ExtraKeyItem;
    pub use dynamo_kv_events::zmq_wire::IMAGE_ROUTING_UUID_PREFIX;
    pub use dynamo_kv_events::zmq_wire::KvCacheSpecKind;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_events::zmq_wire::KvEventBatch; removed in 1.10.0"
    )]
    pub type KvEventBatch = dynamo_kv_events::zmq_wire::KvEventBatch;
    pub use dynamo_kv_events::zmq_wire::KvEventOwnership;
    pub use dynamo_kv_events::zmq_wire::KvTokenIds;
    pub use dynamo_kv_events::zmq_wire::Locality;
    pub use dynamo_kv_events::zmq_wire::RawKvEvent;
    pub use dynamo_kv_events::zmq_wire::StoredBlockOptions;
    pub use dynamo_kv_events::zmq_wire::WIDE_TOKEN_SENTINEL;
    pub use dynamo_kv_events::zmq_wire::ZmqEventFilterReason;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_kv_events::zmq_wire::ZmqEventNormalizer; removed in 1.10.0"
    )]
    pub type ZmqEventNormalizer = dynamo_kv_events::zmq_wire::ZmqEventNormalizer;
    pub use dynamo_kv_events::zmq_wire::convert_event;
    pub use dynamo_kv_events::zmq_wire::create_stored_block_from_parts;
    pub use dynamo_kv_events::zmq_wire::create_stored_blocks;
    pub use dynamo_kv_events::zmq_wire::decode_event_batch;
    pub use dynamo_kv_events::zmq_wire::extra_keys_to_block_mm_infos;
    pub use dynamo_kv_events::zmq_wire::extra_keys_to_cache_namespace;
    pub use dynamo_kv_events::zmq_wire::normalize_mm_placeholder_runs;
    pub use dynamo_kv_events::zmq_wire::normalize_mm_token_runs;
    pub use dynamo_kv_hashing::multimodal::mark_mm_hash_for_extra_key;
    pub use dynamo_kv_hashing::multimodal::parse_mm_hash_from_extra_key;
}

// Root module aliases of the pre-split lib.rs.
pub use indexer::{
    concurrent_radix_tree_compressed, positional as nested_map, pruning as approx, radix_tree,
};
pub use scheduling::{queue, selector};
pub use sequences::{multi_worker as multi_worker_sequence, single as sequence};

// Root re-exports of the pre-split lib.rs.
// Owner: dynamo_scheduler (22), dynamo_kv_indexer (20), dynamo_slot_tracker (11), dynamo_router_protocol (8), dynamo_worker_selection (5), dynamo_kv_events (2), dynamo_kv_hashing (2), dynamo_admission (1), dynamo_kv_hints (1), dynamo_kv_router (1)
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_slot_tracker::ActiveSequences; removed in 1.10.0"
)]
pub type ActiveSequences = dynamo_slot_tracker::ActiveSequences;
#[allow(deprecated)]
pub use dynamo_kv_indexer::AnchorAwareBranchShardedIndexer;
pub use dynamo_slot_tracker::ActiveSequencesMultiWorker;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_kv_indexer::AnchorRef; removed in 1.10.0"
)]
pub type AnchorRef = dynamo_kv_indexer::AnchorRef;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_kv_indexer::AnchorTask; removed in 1.10.0"
)]
pub type AnchorTask = dynamo_kv_indexer::AnchorTask;
pub use dynamo_kv_indexer::BranchShardedIndexer;
pub use dynamo_router_protocol::identity::ClusterId;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_kv_indexer::concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed; removed in 1.10.0"
)]
pub type ConcurrentRadixTreeCompressed =
    dynamo_kv_indexer::concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
pub use dynamo_router_protocol::identity::DEFAULT_ROUTING_GROUP;
pub use dynamo_worker_selection::conditional_disagg::ConditionalDisaggPolicyKind;
#[cfg(any(test, feature = "testing", feature = "bench"))]
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_scheduler::host::DefaultWorkerSelector; removed in 1.10.0"
)]
pub type DefaultWorkerSelector = dynamo_scheduler::host::DefaultWorkerSelector;
pub use dynamo_kv_events::event::KvCacheEventError;
pub use dynamo_scheduler::policy::FcfsPolicy;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_kv_router::config::KvRouterConfig; removed in 1.10.0"
)]
pub type KvRouterConfig = crate::config::KvRouterConfig;
pub use dynamo_kv_hashing::LocalBlockHash;
pub use dynamo_kv_hints::keys::KvTransferEnforcement;
pub use dynamo_scheduler::KvSchedulerError;
pub use dynamo_scheduler::local::LocalScheduler;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_kv_indexer::session_prefix_index::LogicalNode; removed in 1.10.0"
)]
pub type LogicalNode = dynamo_kv_indexer::session_prefix_index::LogicalNode;
pub use dynamo_worker_selection::lora_filter::LoraWorkerFilter;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_kv_indexer::LowerTierContinuation; removed in 1.10.0"
)]
pub type LowerTierContinuation = dynamo_kv_indexer::LowerTierContinuation;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_kv_indexer::LowerTierIndexer; removed in 1.10.0"
)]
pub type LowerTierIndexer = dynamo_kv_indexer::LowerTierIndexer;
pub use dynamo_kv_indexer::MaybeError;
pub use dynamo_kv_indexer::session_prefix_index::NodeId;
pub use dynamo_slot_tracker::NoopSequencePublisher;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_kv_indexer::overlap::OverlapScores; removed in 1.10.0"
)]
pub type OverlapScores = dynamo_kv_indexer::overlap::OverlapScores;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_kv_indexer::positional::PositionalIndexer; removed in 1.10.0"
)]
pub type PositionalIndexer = dynamo_kv_indexer::positional::PositionalIndexer;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_scheduler::protocols::PotentialLoad; removed in 1.10.0"
)]
pub type PotentialLoad = dynamo_scheduler::protocols::PotentialLoad;
pub use dynamo_scheduler::prefill_load::PrefillLoadEstimator;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_slot_tracker::PrefillTokenDeltas; removed in 1.10.0"
)]
pub type PrefillTokenDeltas = dynamo_slot_tracker::PrefillTokenDeltas;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_kv_indexer::radix_tree::RadixTree; removed in 1.10.0"
)]
pub type RadixTree = dynamo_kv_indexer::radix_tree::RadixTree;
pub use dynamo_slot_tracker::ReplicaWorkerPolicy;
pub use dynamo_slot_tracker::RequestId;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_scheduler::config::RouterConfigOverride; removed in 1.10.0"
)]
pub type RouterConfigOverride = dynamo_scheduler::config::RouterConfigOverride;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_kv_events::event::RouterEvent; removed in 1.10.0"
)]
pub type RouterEvent = dynamo_kv_events::event::RouterEvent;
pub use dynamo_admission::profile::RouterQueuePolicy;
pub use dynamo_scheduler::config::RouterPrefillLoadModel;
pub use dynamo_scheduler::policy::RouterSchedulingPolicy;
pub use dynamo_scheduler::protocols::RouterEventSink;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_router_protocol::identity::RoutingPartitionId; removed in 1.10.0"
)]
pub type RoutingPartitionId = dynamo_router_protocol::identity::RoutingPartitionId;
pub use dynamo_router_protocol::identity::RoutingPartitionRef;
pub use dynamo_scheduler::policy::SchedulingPolicy;
pub use dynamo_scheduler::queue::SchedulerQueue;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_scheduler::SchedulingRequest; removed in 1.10.0"
)]
pub type SchedulingRequest = dynamo_scheduler::SchedulingRequest;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_scheduler::SchedulingResponse; removed in 1.10.0"
)]
pub type SchedulingResponse = dynamo_scheduler::SchedulingResponse;
pub use dynamo_slot_tracker::SequenceError;
pub use dynamo_slot_tracker::SequencePublisher;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_slot_tracker::SequenceRequest; removed in 1.10.0"
)]
pub type SequenceRequest = dynamo_slot_tracker::SequenceRequest;
pub use dynamo_slot_tracker::SequenceSubscriber;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_router_protocol::session::SessionContext; removed in 1.10.0"
)]
pub type SessionContext = dynamo_router_protocol::session::SessionContext;
pub use dynamo_kv_indexer::session_prefix_index::SessionId;
pub use dynamo_kv_indexer::session_prefix_index::SessionPrefixIndexError;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_kv_indexer::session_prefix_index::SessionPrefixIndexer; removed in 1.10.0"
)]
pub type SessionPrefixIndexer = dynamo_kv_indexer::session_prefix_index::SessionPrefixIndexer;
pub use dynamo_kv_indexer::session_prefix_index::SessionResidencyOwner;
pub use dynamo_kv_indexer::session_prefix_index::SharedCacheBackend;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_worker_selection::SharedCacheHits; removed in 1.10.0"
)]
pub type SharedCacheHits = dynamo_worker_selection::SharedCacheHits;
pub use dynamo_kv_indexer::SyncIndexer;
pub use dynamo_kv_indexer::ThreadPoolIndexer;
pub use dynamo_scheduler::shared_cache::SharedKvCache;
pub use dynamo_scheduler::tracking_hash::TrackingHashAlgorithm;
pub use dynamo_worker_selection::selection_config::SharedCacheType;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_scheduler::tracking_hash::TrackingHashContext; removed in 1.10.0"
)]
pub type TrackingHashContext = dynamo_scheduler::tracking_hash::TrackingHashContext;
pub use dynamo_router_protocol::worker::WorkerId;
pub use dynamo_scheduler::protocols::WorkerConfigLike;
pub use dynamo_scheduler::tracking_hash::TrackingHashScope;
#[deprecated(
    since = "1.8.0",
    note = "moved to dynamo_slot_tracker::WorkerLoadProjection; removed in 1.10.0"
)]
pub type WorkerLoadProjection = dynamo_slot_tracker::WorkerLoadProjection;
pub use dynamo_kv_hashing::routing::compute_block_hash_for_seq;
pub use dynamo_router_protocol::session::WorkerSelectionInputTrigger;
pub use dynamo_router_protocol::worker_type::WorkerType;
pub use dynamo_scheduler::policy::WsptPolicy;
pub use dynamo_scheduler::selector::WorkerSelectionInput;
pub use dynamo_scheduler::selector::WorkerSelector;
pub use dynamo_worker_selection::WorkerSelectionPolicyError;

// Blocks mounted from modules that still exist in this crate (named lists in the module files).
/// Pre-split `dynamo_kv_router::scheduling` items that moved; mounted by `src/scheduling/mod.rs`.
pub mod scheduling {
    // Owner: dynamo_scheduler (36), dynamo_admission (8), dynamo_router_protocol (3), dynamo_worker_selection (2)
    /// Pre-split `dynamo_kv_router::scheduling::config` items that moved; mounted by `src/scheduling/config.rs`.
    pub mod config {
        // Owner: dynamo_scheduler (2)
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::config::RouterConfigOverrideBuilder; removed in 1.10.0"
        )]
        pub type RouterConfigOverrideBuilder =
            dynamo_scheduler::config::RouterConfigOverrideBuilder;
        pub use dynamo_scheduler::config::RouterConfigOverrideBuilderError;
    }

    /// Pre-split `dynamo_kv_router::scheduling::overlap`; its items live in `dynamo_scheduler` now.
    pub mod overlap {
        // Owner: dynamo_scheduler (10)
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::overlap::CacheHitEstimates; removed in 1.10.0"
        )]
        pub type CacheHitEstimates = dynamo_scheduler::overlap::CacheHitEstimates;
        pub use dynamo_scheduler::overlap::OverlapAnalysis;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::overlap::OverlapScoresResponse; removed in 1.10.0"
        )]
        pub type OverlapScoresResponse = dynamo_scheduler::overlap::OverlapScoresResponse;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::overlap::OverlapSignals; removed in 1.10.0"
        )]
        pub type OverlapSignals = dynamo_scheduler::overlap::OverlapSignals;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::overlap::SelectedWorkerTierSnapshot; removed in 1.10.0"
        )]
        pub type SelectedWorkerTierSnapshot = dynamo_scheduler::overlap::SelectedWorkerTierSnapshot;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::overlap::SharedCacheOverlapScore; removed in 1.10.0"
        )]
        pub type SharedCacheOverlapScore = dynamo_scheduler::overlap::SharedCacheOverlapScore;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::overlap::WorkerOverlapScore; removed in 1.10.0"
        )]
        pub type WorkerOverlapScore = dynamo_scheduler::overlap::WorkerOverlapScore;
        pub use dynamo_scheduler::overlap::build_overlap_scores_response;
        pub use dynamo_scheduler::overlap::cache_hit_estimates_from_tiered_matches;
        pub use dynamo_scheduler::overlap::tier_overlap_blocks_from_tiered_matches;
    }

    /// Pre-split `dynamo_kv_router::scheduling::overlap_refresh`; its items live in `dynamo_scheduler` now.
    pub mod overlap_refresh {
        // Owner: dynamo_scheduler (9)
        pub use dynamo_scheduler::overlap_refresh::DEFAULT_OVERLAP_REFRESH_AFTER_SECS;
        pub use dynamo_scheduler::overlap_refresh::NoopOverlapScoresRefresh;
        pub use dynamo_scheduler::overlap_refresh::OVERLAP_REFRESH_TIMEOUT;
        pub use dynamo_scheduler::overlap_refresh::OverlapScoresRefresh;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::overlap_refresh::RefreshedOverlap; removed in 1.10.0"
        )]
        pub type RefreshedOverlap = dynamo_scheduler::overlap_refresh::RefreshedOverlap;
        pub use dynamo_scheduler::overlap_refresh::TieredOverlapRefresher;
        pub use dynamo_scheduler::overlap_refresh::read_overlap_refresh_after;
        pub use dynamo_scheduler::overlap_refresh::refresh_overlap;
        pub use dynamo_scheduler::overlap_refresh::should_refresh_overlap;
    }

    /// Pre-split `dynamo_kv_router::scheduling::policy`; its items live in `dynamo_scheduler` now.
    pub mod policy {
        // Owner: dynamo_scheduler (5)
        pub use dynamo_scheduler::policy::FcfsPolicy;
        pub use dynamo_scheduler::policy::LcfsPolicy;
        pub use dynamo_scheduler::policy::RouterSchedulingPolicy;
        pub use dynamo_scheduler::policy::SchedulingPolicy;
        pub use dynamo_scheduler::policy::WsptPolicy;
    }

    /// Pre-split `dynamo_kv_router::scheduling::policy_queue`; its items live in `dynamo_admission` now.
    pub mod policy_queue {
        // Owner: dynamo_admission (7)
        pub use dynamo_admission::queue::PolicyQueue;
        pub use dynamo_admission::queue::PolicyQueueEntry;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_admission::queue::PolicyQueueStats; removed in 1.10.0"
        )]
        pub type PolicyQueueStats = dynamo_admission::queue::PolicyQueueStats;
        pub use dynamo_admission::queue::QueueLimitKind;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_admission::queue::QueueMetadata; removed in 1.10.0"
        )]
        pub type QueueMetadata = dynamo_admission::queue::QueueMetadata;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_admission::queue::QueueRejection; removed in 1.10.0"
        )]
        pub type QueueRejection = dynamo_admission::queue::QueueRejection;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_admission::queue::QueueSnapshot; removed in 1.10.0"
        )]
        pub type QueueSnapshot = dynamo_admission::queue::QueueSnapshot;
    }

    /// Pre-split `dynamo_kv_router::scheduling::prefill_load`; its items live in `dynamo_scheduler` now.
    pub mod prefill_load {
        // Owner: dynamo_scheduler (4)
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::prefill_load::InvalidEffectivePrefillTokens; removed in 1.10.0"
        )]
        pub type InvalidEffectivePrefillTokens =
            dynamo_scheduler::prefill_load::InvalidEffectivePrefillTokens;
        pub use dynamo_scheduler::prefill_load::PrefillLoadEstimator;
        pub use dynamo_scheduler::prefill_load::effective_prefill_tokens;
        pub use dynamo_scheduler::prefill_load::prefill_load_hint_from_effective_tokens;
    }

    /// Pre-split `dynamo_kv_router::scheduling::queue`; its items live in `dynamo_scheduler`, `dynamo_router_protocol` and `dynamo_slot_tracker` now.
    pub mod queue {
        // Owner: dynamo_scheduler (6), dynamo_router_protocol (1), dynamo_slot_tracker (1)
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::queue::BookingHandle; removed in 1.10.0"
        )]
        pub type BookingHandle = dynamo_scheduler::queue::BookingHandle;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::queue::ClassQueueStats; removed in 1.10.0"
        )]
        pub type ClassQueueStats = dynamo_scheduler::queue::ClassQueueStats;
        pub use dynamo_router_protocol::worker::DEFAULT_MAX_BATCHED_TOKENS;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::queue::RuntimePolicyControl; removed in 1.10.0"
        )]
        pub type RuntimePolicyControl = dynamo_scheduler::queue::RuntimePolicyControl;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::queue::SchedulerBookingCleanup; removed in 1.10.0"
        )]
        pub type SchedulerBookingCleanup = dynamo_scheduler::queue::SchedulerBookingCleanup;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_slot_tracker::booking::SchedulerBookingDescriptor; removed in 1.10.0"
        )]
        pub type SchedulerBookingDescriptor =
            dynamo_slot_tracker::booking::SchedulerBookingDescriptor;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::queue::SchedulerCleanupAck; removed in 1.10.0"
        )]
        pub type SchedulerCleanupAck = dynamo_scheduler::queue::SchedulerCleanupAck;
        pub use dynamo_scheduler::queue::SchedulerQueue;
    }

    /// Pre-split `dynamo_kv_router::scheduling::request_classifier`; its items live in `dynamo_admission` now.
    pub mod request_classifier {
        // Owner: dynamo_admission (1)
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_admission::classifier::runtime::RequestLifecycle; removed in 1.10.0"
        )]
        pub type RequestLifecycle = dynamo_admission::classifier::runtime::RequestLifecycle;
    }

    /// Pre-split `dynamo_kv_router::scheduling::selector`; its items live in `dynamo_scheduler` and `dynamo_worker_selection` now.
    pub mod selector {
        // Owner: dynamo_scheduler (4), dynamo_worker_selection (3)
        pub use dynamo_scheduler::selector::ConfiguredSelectionInputs;
        #[cfg(any(test, feature = "testing", feature = "bench"))]
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::host::DefaultWorkerSelector; removed in 1.10.0"
        )]
        pub type DefaultWorkerSelector = dynamo_scheduler::host::DefaultWorkerSelector;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_worker_selection::HostedSelectionRequest; removed in 1.10.0"
        )]
        pub type HostedSelectionRequest = dynamo_worker_selection::HostedSelectionRequest;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_worker_selection::HostedWorkerPicker; removed in 1.10.0"
        )]
        pub type HostedWorkerPicker = dynamo_worker_selection::HostedWorkerPicker;
        pub use dynamo_scheduler::selector::WorkerSelectionInput;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_worker_selection::WorkerSelectionPolicy; removed in 1.10.0"
        )]
        pub type WorkerSelectionPolicy = dynamo_worker_selection::WorkerSelectionPolicy;
        pub use dynamo_scheduler::selector::WorkerSelector;
    }

    pub use dynamo_scheduler::AccountingAttribution;
    pub use dynamo_scheduler::AdmissionAttempt;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::AdmittedSchedulingResponse; removed in 1.10.0"
    )]
    pub type AdmittedSchedulingResponse = dynamo_scheduler::AdmittedSchedulingResponse;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::AdvisorySchedulingResponse; removed in 1.10.0"
    )]
    pub type AdvisorySchedulingResponse = dynamo_scheduler::AdvisorySchedulingResponse;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::AdvisoryWorkerLoad; removed in 1.10.0"
    )]
    pub type AdvisoryWorkerLoad = dynamo_scheduler::AdvisoryWorkerLoad;
    pub use dynamo_router_protocol::attempt::AttemptId;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::overlap::CacheHitEstimates; removed in 1.10.0"
    )]
    pub type CacheHitEstimates = dynamo_scheduler::overlap::CacheHitEstimates;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::prefill_load::InvalidEffectivePrefillTokens; removed in 1.10.0"
    )]
    pub type InvalidEffectivePrefillTokens =
        dynamo_scheduler::prefill_load::InvalidEffectivePrefillTokens;
    pub use dynamo_scheduler::KvSchedulerError;
    pub use dynamo_scheduler::local::LocalScheduler;
    pub use dynamo_worker_selection::lora_filter::LoraWorkerFilter;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::NonMaxOverlapSelection; removed in 1.10.0"
    )]
    pub type NonMaxOverlapSelection = dynamo_scheduler::NonMaxOverlapSelection;
    pub use dynamo_scheduler::NonMaxOverlapSelectionObserver;
    pub use dynamo_scheduler::overlap::OverlapAnalysis;
    pub use dynamo_scheduler::overlap_refresh::NoopOverlapScoresRefresh;
    pub use dynamo_scheduler::overlap_refresh::OverlapScoresRefresh;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::overlap::OverlapScoresResponse; removed in 1.10.0"
    )]
    pub type OverlapScoresResponse = dynamo_scheduler::overlap::OverlapScoresResponse;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::overlap::OverlapSignals; removed in 1.10.0"
    )]
    pub type OverlapSignals = dynamo_scheduler::overlap::OverlapSignals;
    pub use dynamo_admission::queue::PolicyQueue;
    pub use dynamo_admission::queue::PolicyQueueEntry;
    pub use dynamo_scheduler::OverloadedWorkerProvider;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::protocols::PotentialLoad; removed in 1.10.0"
    )]
    pub type PotentialLoad = dynamo_scheduler::protocols::PotentialLoad;
    pub use dynamo_admission::queue::QueueLimitKind;
    pub use dynamo_scheduler::prefill_load::PrefillLoadEstimator;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_admission::queue::QueueMetadata; removed in 1.10.0"
    )]
    pub type QueueMetadata = dynamo_admission::queue::QueueMetadata;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_admission::queue::QueueRejection; removed in 1.10.0"
    )]
    pub type QueueRejection = dynamo_admission::queue::QueueRejection;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_admission::queue::QueueSnapshot; removed in 1.10.0"
    )]
    pub type QueueSnapshot = dynamo_admission::queue::QueueSnapshot;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::overlap_refresh::RefreshedOverlap; removed in 1.10.0"
    )]
    pub type RefreshedOverlap = dynamo_scheduler::overlap_refresh::RefreshedOverlap;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_admission::classifier::runtime::RequestLifecycle; removed in 1.10.0"
    )]
    pub type RequestLifecycle = dynamo_admission::classifier::runtime::RequestLifecycle;
    pub use dynamo_scheduler::RoutingEligibility;
    pub use dynamo_scheduler::ScheduleMode;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::ScheduleRequest; removed in 1.10.0"
    )]
    pub type ScheduleRequest = dynamo_scheduler::ScheduleRequest;
    pub use dynamo_scheduler::SchedulingContext;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::SchedulingRequest; removed in 1.10.0"
    )]
    pub type SchedulingRequest = dynamo_scheduler::SchedulingRequest;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::SchedulingResponse; removed in 1.10.0"
    )]
    pub type SchedulingResponse = dynamo_scheduler::SchedulingResponse;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::overlap::SelectedWorkerTierSnapshot; removed in 1.10.0"
    )]
    pub type SelectedWorkerTierSnapshot = dynamo_scheduler::overlap::SelectedWorkerTierSnapshot;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_router_protocol::session::SessionContext; removed in 1.10.0"
    )]
    pub type SessionContext = dynamo_router_protocol::session::SessionContext;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::overlap::SharedCacheOverlapScore; removed in 1.10.0"
    )]
    pub type SharedCacheOverlapScore = dynamo_scheduler::overlap::SharedCacheOverlapScore;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::TierOverlapBlocks; removed in 1.10.0"
    )]
    pub type TierOverlapBlocks = dynamo_scheduler::TierOverlapBlocks;
    pub use dynamo_scheduler::WorkerAvailabilityProvider;
    pub use dynamo_scheduler::WorkerEligibilityError;
    pub use dynamo_scheduler::WorkerIdSet;
    pub use dynamo_scheduler::overlap_refresh::TieredOverlapRefresher;
    #[deprecated(
        since = "1.8.0",
        note = "moved to dynamo_scheduler::overlap::WorkerOverlapScore; removed in 1.10.0"
    )]
    pub type WorkerOverlapScore = dynamo_scheduler::overlap::WorkerOverlapScore;
    pub use dynamo_admission::placement::WorkerPlacement;
    pub use dynamo_router_protocol::session::WorkerSelectionInputTrigger;
    pub use dynamo_scheduler::prefill_load::effective_prefill_tokens;
    pub use dynamo_scheduler::prefill_load::prefill_load_hint_from_effective_tokens;
    pub use dynamo_worker_selection::WorkerSelectionPolicyError;
}

/// Pre-split `dynamo_kv_router::services` items that moved; mounted by `src/services/mod.rs`.
pub mod services {
    /// Pre-split `dynamo_kv_router::services::selection` items that moved; mounted by `src/services/selection/mod.rs`.
    pub mod selection {
        // Owner: dynamo_scheduler (50), dynamo_kv_router (8), dynamo_session_affinity (1)
        /// Pre-split `dynamo_kv_router::services::selection::affinity`; its items live in `dynamo_session_affinity` now.
        pub mod affinity {
            // Owner: dynamo_session_affinity (19)
            pub use dynamo_session_affinity::AcquireStep;
            pub use dynamo_session_affinity::AffinityError;
            #[deprecated(
                since = "1.8.0",
                note = "moved to dynamo_session_affinity::AffinityInitialization; removed in 1.10.0"
            )]
            pub type AffinityInitialization = dynamo_session_affinity::AffinityInitialization;
            #[deprecated(
                since = "1.8.0",
                note = "moved to dynamo_session_affinity::AffinityLease; removed in 1.10.0"
            )]
            pub type AffinityLease = dynamo_session_affinity::AffinityLease;
            pub use dynamo_session_affinity::AffinityReplicaSink;
            pub use dynamo_session_affinity::AffinityTarget;
            #[deprecated(
                since = "1.8.0",
                note = "moved to dynamo_session_affinity::AffinityVersion; removed in 1.10.0"
            )]
            pub type AffinityVersion = dynamo_session_affinity::AffinityVersion;
            pub use dynamo_session_affinity::Hold;
            pub use dynamo_session_affinity::MAX_SESSION_AFFINITY_ENTRIES;
            pub use dynamo_session_affinity::MAX_SESSION_AFFINITY_ID_BYTES;
            pub use dynamo_session_affinity::MAX_SESSION_AFFINITY_TTL_SECS;
            pub use dynamo_session_affinity::ReplicaApplyOutcome;
            #[deprecated(
                since = "1.8.0",
                note = "moved to dynamo_session_affinity::SessionAffinity; removed in 1.10.0"
            )]
            pub type SessionAffinity = dynamo_session_affinity::SessionAffinity;
            #[deprecated(
                since = "1.8.0",
                note = "moved to dynamo_session_affinity::SessionAffinityConfig; removed in 1.10.0"
            )]
            pub type SessionAffinityConfig = dynamo_session_affinity::SessionAffinityConfig;
            pub use dynamo_session_affinity::SessionAffinityMode;
            #[deprecated(
                since = "1.8.0",
                note = "moved to dynamo_session_affinity::WeakSessionAffinity; removed in 1.10.0"
            )]
            pub type WeakSessionAffinity = dynamo_session_affinity::WeakSessionAffinity;
            pub use dynamo_session_affinity::subagent_group_affinity_id;
            pub use dynamo_session_affinity::validate_bound_target;
            pub use dynamo_session_affinity::validate_dispatch_target;
        }

        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_session_affinity::wire::AffinityBindingEvent; removed in 1.10.0"
        )]
        pub type AffinityBindingEvent = dynamo_session_affinity::wire::AffinityBindingEvent;
        pub use dynamo_scheduler::selection::membership::CatalogObserver;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::membership::CatalogReconciler; removed in 1.10.0"
        )]
        pub type CatalogReconciler = dynamo_scheduler::selection::membership::CatalogReconciler;
        pub use crate::config::DYN_ROUTER_DECODE_POLICY;
        pub use crate::config::DYN_ROUTER_PREFILL_POLICY;
        pub use crate::config::DYN_ROUTER_WORKER_SELECTION_POLICY;
        pub use dynamo_scheduler::selection::types::DEFAULT_MODEL_NAME;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::core::HostCache; removed in 1.10.0"
        )]
        pub type HostCache = dynamo_scheduler::selection::core::HostCache;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::core::HostEligibility; removed in 1.10.0"
        )]
        pub type HostEligibility = dynamo_scheduler::selection::core::HostEligibility;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::core::HostLoad; removed in 1.10.0"
        )]
        pub type HostLoad = dynamo_scheduler::selection::core::HostLoad;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::replication::HostReplicaChannels; removed in 1.10.0"
        )]
        pub type HostReplicaChannels = dynamo_scheduler::replication::HostReplicaChannels;
        pub use dynamo_scheduler::replication::HostReplicaSyncFactory;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::core::HostReplication; removed in 1.10.0"
        )]
        pub type HostReplication = dynamo_scheduler::selection::core::HostReplication;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::core::HostTelemetry; removed in 1.10.0"
        )]
        pub type HostTelemetry = dynamo_scheduler::selection::core::HostTelemetry;
        pub use dynamo_scheduler::selection::core::KvIndexSource;
        pub use dynamo_scheduler::selection::ingress::KvEventIngress;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::types::ModelLoadResponse; removed in 1.10.0"
        )]
        pub type ModelLoadResponse = dynamo_scheduler::selection::types::ModelLoadResponse;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::types::OutputBlockRequest; removed in 1.10.0"
        )]
        pub type OutputBlockRequest = dynamo_scheduler::selection::types::OutputBlockRequest;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::types::OverlapScoresRequest; removed in 1.10.0"
        )]
        pub type OverlapScoresRequest = dynamo_scheduler::selection::types::OverlapScoresRequest;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::overlap::OverlapScoresResponse; removed in 1.10.0"
        )]
        pub type OverlapScoresResponse = dynamo_scheduler::overlap::OverlapScoresResponse;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::types::PotentialLoadsRequest; removed in 1.10.0"
        )]
        pub type PotentialLoadsRequest = dynamo_scheduler::selection::types::PotentialLoadsRequest;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::input::PromptRequest; removed in 1.10.0"
        )]
        pub type PromptRequest = dynamo_scheduler::selection::input::PromptRequest;
        pub use dynamo_scheduler::selection::input::PromptView;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::types::ReadyResponse; removed in 1.10.0"
        )]
        pub type ReadyResponse = dynamo_scheduler::selection::types::ReadyResponse;
        pub use dynamo_scheduler::replication::ReplicaIngressObserver;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::types::ReservationRequest; removed in 1.10.0"
        )]
        pub type ReservationRequest = dynamo_scheduler::selection::types::ReservationRequest;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::types::ReservationResponse; removed in 1.10.0"
        )]
        pub type ReservationResponse = dynamo_scheduler::selection::types::ReservationResponse;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_kv_router::plugins::RouterPluginRegistry; removed in 1.10.0"
        )]
        pub type RouterPluginRegistry = crate::plugins::RouterPluginRegistry;
        pub use dynamo_scheduler::replication::SchedulerLoadSink;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::types::SelectAndReserveRequest; removed in 1.10.0"
        )]
        pub type SelectAndReserveRequest =
            dynamo_scheduler::selection::types::SelectAndReserveRequest;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::types::SelectRequest; removed in 1.10.0"
        )]
        pub type SelectRequest = dynamo_scheduler::selection::types::SelectRequest;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::types::SelectResponse; removed in 1.10.0"
        )]
        pub type SelectResponse = dynamo_scheduler::selection::types::SelectResponse;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::host::Selected; removed in 1.10.0"
        )]
        pub type Selected = dynamo_scheduler::host::Selected;
        pub use dynamo_scheduler::host::SelectionAdmission;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::pending::SelectionCacheConfig; removed in 1.10.0"
        )]
        pub type SelectionCacheConfig = dynamo_scheduler::selection::pending::SelectionCacheConfig;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::core::SelectionCore; removed in 1.10.0"
        )]
        pub type SelectionCore = dynamo_scheduler::selection::core::SelectionCore;
        pub use dynamo_scheduler::selection::error::SelectionError;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::core::SelectionHost; removed in 1.10.0"
        )]
        pub type SelectionHost = dynamo_scheduler::selection::core::SelectionHost;
        pub use dynamo_scheduler::host::SelectionOperation;
        pub use dynamo_scheduler::host::SelectionOutcome;
        pub use dynamo_scheduler::selection::core::SelectionPartition;
        pub use dynamo_scheduler::selection::types::SelectionInputTrigger;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::host::SelectionRun; removed in 1.10.0"
        )]
        pub type SelectionRun = dynamo_scheduler::host::SelectionRun;
        pub use dynamo_scheduler::selection::core::SelectionScheduler;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::types::SelectionSessionContext; removed in 1.10.0"
        )]
        pub type SelectionSessionContext =
            dynamo_scheduler::selection::types::SelectionSessionContext;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::types::SelectionWorkerConfig; removed in 1.10.0"
        )]
        pub type SelectionWorkerConfig = dynamo_scheduler::selection::types::SelectionWorkerConfig;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::types::SelectionWorkerLoad; removed in 1.10.0"
        )]
        pub type SelectionWorkerLoad = dynamo_scheduler::selection::types::SelectionWorkerLoad;
        pub use dynamo_scheduler::host::SessionBinding;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::overlap::SharedCacheOverlapScore; removed in 1.10.0"
        )]
        pub type SharedCacheOverlapScore = dynamo_scheduler::overlap::SharedCacheOverlapScore;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::types::WorkerCatalogRecord; removed in 1.10.0"
        )]
        pub type WorkerCatalogRecord = dynamo_scheduler::selection::types::WorkerCatalogRecord;
        pub use dynamo_scheduler::selection::membership::WorkerCatalogSource;
        pub use dynamo_scheduler::selection::types::WorkerLifecycle;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::overlap::WorkerOverlapScore; removed in 1.10.0"
        )]
        pub type WorkerOverlapScore = dynamo_scheduler::overlap::WorkerOverlapScore;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::types::WorkerPatchRequest; removed in 1.10.0"
        )]
        pub type WorkerPatchRequest = dynamo_scheduler::selection::types::WorkerPatchRequest;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::selection::types::WorkerRequest; removed in 1.10.0"
        )]
        pub type WorkerRequest = dynamo_scheduler::selection::types::WorkerRequest;
        pub use crate::plugins::WorkerSelectionPolicyParameters;
        pub use crate::plugins::WorkerSelectionPolicyProvider;
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_kv_router::plugins::WorkerSelectionPolicyProviderError; removed in 1.10.0"
        )]
        pub type WorkerSelectionPolicyProviderError =
            crate::plugins::WorkerSelectionPolicyProviderError;
        pub use crate::plugins::WorkerSelectionPolicyRegistryError;
    }

    /// Pre-split `dynamo_kv_router::services::overlap`; its items live in `dynamo_scheduler` now.
    pub mod overlap {
        // Owner: dynamo_scheduler (1)
        #[deprecated(
            since = "1.8.0",
            note = "moved to dynamo_scheduler::overlap_summary::MooncakeOverlapSummary; removed in 1.10.0"
        )]
        pub type MooncakeOverlapSummary = dynamo_scheduler::overlap_summary::MooncakeOverlapSummary;
    }
}

/// Pre-split `dynamo_kv_router::test_utils` items that moved; mounted by `src/test_utils.rs`.
#[cfg(any(test, feature = "bench"))]
pub mod test_utils {
    // Owner: dynamo_kv_indexer (7)
    pub use dynamo_kv_indexer::testing::assert_exact_scores;
    pub use dynamo_kv_indexer::testing::assert_no_scores;
    pub use dynamo_kv_indexer::testing::assert_overlap_scores_eq;
    pub use dynamo_kv_indexer::testing::assert_score;
    pub use dynamo_kv_indexer::testing::flush_and_settle;
    pub use dynamo_kv_indexer::testing::query_scores;
    pub use dynamo_kv_indexer::testing::snapshot_tree;
}
