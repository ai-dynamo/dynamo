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
pub mod scheduling;
pub mod sequences;

// Backward-compat re-exports: preserve old module paths for external consumers
pub use scheduling::config;
pub use scheduling::queue;
pub use scheduling::selector;
pub use sequences::multi_worker as multi_worker_sequence;
pub use sequences::single as sequence;

#[cfg(test)]
pub(crate) mod test_utils;

// Re-export key types for convenience
pub use self::multi_worker_sequence::{
    ActiveSequencesMultiWorker, SequenceError, SequencePublisher, SequenceRequest,
    SequenceSubscriber,
};
pub use self::sequence::{ActiveSequences, RequestId};
pub use concurrent_radix_tree::ConcurrentRadixTree;
pub use config::{KvRouterConfig, RouterConfigOverride};
pub use indexer::{MaybeError, SyncIndexer, ThreadPoolIndexer};
#[cfg(feature = "bench")]
pub use naive_indexers::{InvertedIndex, NaiveNestedMap};
pub use nested_map::PositionalIndexer;
pub use protocols::{
    KvCacheEventError, LocalBlockHash, OverlapScores, RouterEvent, WorkerConfigLike, WorkerId,
    compute_block_hash_for_seq,
};
pub use queue::SchedulerQueue;
pub use radix_tree::RadixTree;
pub use scheduling::{KvSchedulerError, PotentialLoad, SchedulingRequest, SchedulingResponse};
pub use selector::{DefaultWorkerSelector, WorkerSelector};
