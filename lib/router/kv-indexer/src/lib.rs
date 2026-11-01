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
//! slice (`config`). `lib/kv-router` re-exports every item here at its old path and keeps the
//! HTTP, ZMQ, peer-recovery, and Prometheus adapters.
//!
//! Depends only on `dynamo-router-protocol`, `dynamo-kv-events`, `dynamo-kv-hashing`, and
//! `dynamo-kv-hints` (the hint vocabulary and label hashing its transfer candidates use).
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
pub mod api;
mod approximate_lru;
pub mod backend;
mod branch_sharded;
mod compressed_radix;
pub mod concurrent_radix_tree_compressed;
pub mod config;
pub mod cuckoo;
pub mod delegate;
pub mod host;
pub mod identity;
pub mod kv_hints;
mod kv_indexer;
mod local;
pub mod lookup;
mod lower_tier;
mod lower_tier_indexers;
pub mod metrics;
#[cfg(feature = "bench")]
mod observation;
pub mod overlap;
pub mod positional;
mod prelude;
pub mod pruning;
pub mod radix_tree;
pub mod recording;
pub mod registry;
pub mod residency;
pub mod session_prefix_index;
pub mod session_updates;
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

pub use api::*;
pub use approximate_lru::*;
pub use branch_sharded::*;
pub use concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
pub use delegate::{HashLifecycle, KvIndexerDelegate};
pub use kv_indexer::*;
pub use local::*;
pub use lower_tier::*;
pub use lower_tier_indexers::*;
pub use metrics::*;
#[cfg(feature = "bench")]
pub use observation::*;
pub use radix_tree::RadixTree;
pub use session_prefix_index::{SessionPrefixIndexer, SessionResidencyOwner, SharedCacheBackend};
pub use thread_pool::*;
pub use traits::*;
pub use types::*;
