// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV Router - Radix tree data structures for LLM KV cache routing.
//!
//! This crate provides the core radix tree implementation and protocols for
//! efficient KV cache lookup and routing in distributed LLM inference systems.

pub mod event_sink;
pub mod indexer;
pub mod protocols;
pub mod scheduling;
pub mod sequences;
pub mod zmq_wire;

// Backward-compat re-exports: old top-level module paths still work
pub use indexer::concurrent_radix_tree;
pub use indexer::concurrent_radix_tree_compressed;
#[cfg(feature = "bench")]
pub use indexer::naive as naive_indexers;
pub use indexer::positional as nested_map;
pub use indexer::pruning as approx;
pub use indexer::radix_tree;

pub use scheduling::config;
pub use scheduling::queue;
pub use scheduling::selector;
pub use sequences::multi_worker as multi_worker_sequence;
pub use sequences::single as sequence;

#[cfg(any(test, feature = "bench"))]
pub mod test_utils;

// Re-export key types for convenience
pub use self::multi_worker_sequence::{
    ActiveSequencesMultiWorker, SequenceError, SequencePublisher, SequenceRequest,
    SequenceSubscriber,
};
pub use self::sequence::{ActiveSequences, RequestId};
pub use concurrent_radix_tree::ConcurrentRadixTree;
pub use concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
pub use config::{KvRouterConfig, RouterConfigOverride};
pub use event_sink::EventSink;
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
