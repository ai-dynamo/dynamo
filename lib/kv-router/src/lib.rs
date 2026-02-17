// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV Router - Radix tree data structures for LLM KV cache routing.
//!
//! This crate provides the core radix tree implementation and protocols for
//! efficient KV cache lookup and routing in distributed LLM inference systems.

pub mod approx;
#[cfg(feature = "bench")]
pub mod bench_utils;
pub mod concurrent_radix_tree;
pub mod indexer;
#[cfg(feature = "bench")]
pub mod naive_indexers;
pub mod nested_map;
pub mod protocols;
pub mod radix_tree;

#[cfg(test)]
pub(crate) mod test_utils;

// Re-export key types for convenience
pub use concurrent_radix_tree::ConcurrentRadixTree;
pub use indexer::{MaybeError, SyncIndexer, ThreadPoolIndexer};
#[cfg(feature = "bench")]
pub use naive_indexers::{InvertedIndex, NaiveNestedMap};
pub use nested_map::PositionalIndexer;
pub use protocols::{
    KvCacheEventError, LocalBlockHash, OverlapScores, RouterEvent, WorkerId,
    compute_block_hash_for_seq,
};
pub use radix_tree::RadixTree;
