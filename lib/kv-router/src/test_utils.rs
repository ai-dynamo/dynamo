// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test utilities. The indexer event builders and assertions live in
//! `dynamo_kv_indexer::testing` and are re-exported here at their old paths.

use std::collections::HashSet;

use crate::protocols::WorkerConfigLike;
pub use crate::sequences::NoopSequencePublisher;
pub use dynamo_kv_indexer::testing::*;

/// Minimal [`WorkerConfigLike`] for scheduler/queue tests and benchmarks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimpleWorkerConfig {
    pub data_parallel_start_rank: u32,
    pub data_parallel_size: u32,
    pub max_num_batched_tokens: Option<u64>,
    pub total_kv_blocks: Option<u64>,
    pub taints: HashSet<String>,
}

impl Default for SimpleWorkerConfig {
    fn default() -> Self {
        Self {
            data_parallel_start_rank: 0,
            data_parallel_size: 1,
            max_num_batched_tokens: None,
            total_kv_blocks: None,
            taints: HashSet::new(),
        }
    }
}

impl WorkerConfigLike for SimpleWorkerConfig {
    fn data_parallel_start_rank(&self) -> u32 {
        self.data_parallel_start_rank
    }

    fn data_parallel_size(&self) -> u32 {
        self.data_parallel_size
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        self.max_num_batched_tokens
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        self.total_kv_blocks
    }

    fn taints(&self) -> &HashSet<String> {
        &self.taints
    }
}
