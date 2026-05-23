// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV indexer feature.
//!
//! A hub-side, router-like index of which worker instances hold which KV
//! blocks. Workers publish block create/remove events
//! ([`kvbm_logical::events::KvbmCacheEvents`]) to the ZMQ ingest endpoint
//! advertised by `GET /v1/features/indexer/config`; the hub buckets them by
//! block position and resolves "who holds this sequence?" queries to the
//! deepest matching block's holders.
//!
//! The feature owns its whole HTTP namespace via
//! [`FeatureManager::route_prefix`](crate::features::FeatureManager::route_prefix)
//! — it never piggybacks on routes owned by another manager.

/// `kvbmctl` client CLI for this feature. Gated behind the `kvbmctl` feature.
#[cfg(feature = "kvbmctl")]
pub mod cli;
pub mod client;
pub mod handlers;
pub mod index;
pub mod ingest;
pub mod manager;
pub mod protocol;
pub mod zmq;

pub use client::IndexerLookupClient;
pub use handlers::create_query_handler;
pub use index::PositionalIndex;
pub use manager::IndexerManager;
pub use protocol::{
    ByPositionResponse, FindBlocksHit, IndexEntry, IndexerConfigResponse, InstancesResponse,
    QUERY_HANDLER, QueryRequest, QueryResponse, ROUTE_PREFIX,
};
