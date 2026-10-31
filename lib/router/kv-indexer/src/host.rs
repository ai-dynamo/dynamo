// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What an in-tree host names from the KV indexer stage, in one place: the indexer handles it
//! builds and feeds, the registry it opens partitions in, the lookup and recording types it
//! reads back, and the service DTOs. Every item is re-exported from its owning module; this
//! list is the host contract (hosts inventory v2, section 3). Tier 1 never imports it, and the
//! DC relay's `cuckoo` module stays a public module of its own.

pub use crate::api::{KvStateProtocolVersion, WorkerKvQueryResponse};
pub use crate::approximate_lru::ApproximateLruStats;
pub use crate::backend::{Indexer, IndexerPolicy, RemotePrimary, SideIndexer};
pub use crate::concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
pub use crate::local::LocalKvIndexer;
pub use crate::lower_tier_indexers::TieredMatchDetails;
pub use crate::metrics::KvIndexerMetrics;
#[cfg(feature = "bench")]
pub use crate::observation::ThreadPoolObservationSnapshot;
pub use crate::overlap::OverlapScores;
pub use crate::pruning::PruneConfig;
pub use crate::radix_tree::RadixTree;
pub use crate::recording::ApproximateRequestLease;
pub use crate::registry::IndexRegistry;
pub use crate::residency::ResidencyOwnerKeyExt;
pub use crate::session_updates::SessionUpdateSender;
pub use crate::traits::{KvIndexerInterface, SyncIndexer};
pub use crate::types::{KvRouterError, RoutingDecisionHashes, WireTieredMatchDetails};
