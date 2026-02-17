// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Benchmark utilities for kv-router benchmarks.
//!
//! This module provides shared data structures for benchmarking:
//! - `SequenceData`: Pre-generated sequence data for benchmarking

use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, RouterEvent, WorkerId, compute_seq_hash_for_block,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Duration;

/// Pre-generated sequence data for benchmarking.
#[derive(Clone)]
pub struct SequenceData {
    pub worker_id: WorkerId,
    pub local_hashes: Vec<LocalBlockHash>,
    pub external_hashes: Vec<ExternalSequenceBlockHash>,
}

impl SequenceData {
    /// Create a new sequence with synthetic hashes based on sequence ID.
    pub fn new(seq_id: u64, worker_id: WorkerId, depth: usize) -> Self {
        let local_hashes: Vec<LocalBlockHash> = (0..depth)
            .map(|block_idx| LocalBlockHash((seq_id << 32) | (block_idx as u64)))
            .collect();

        let external_hashes: Vec<ExternalSequenceBlockHash> = (0..depth)
            .map(|block_idx| ExternalSequenceBlockHash((seq_id << 32) | (block_idx as u64)))
            .collect();

        Self {
            worker_id,
            local_hashes,
            external_hashes,
        }
    }

    /// Create a sequence from local hashes, computing external hashes using cumulative hash.
    ///
    /// This ensures FlatHashMap can correctly identify block positions.
    pub fn from_local_hashes(worker_id: WorkerId, local_hashes: Vec<LocalBlockHash>) -> Self {
        let seq_hashes = compute_seq_hash_for_block(&local_hashes);
        let external_hashes = seq_hashes
            .into_iter()
            .map(ExternalSequenceBlockHash)
            .collect();

        Self {
            worker_id,
            local_hashes,
            external_hashes,
        }
    }

    /// Convert to a store event.
    pub fn to_store_event(&self, event_id: u64) -> RouterEvent {
        RouterEvent {
            worker_id: self.worker_id,
            event: KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    blocks: self
                        .local_hashes
                        .iter()
                        .zip(self.external_hashes.iter())
                        .map(|(local, ext)| KvCacheStoredBlockData {
                            tokens_hash: *local,
                            block_hash: *ext,
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
                dp_rank: 0,
            },
        }
    }

    /// Convert to a remove event.
    pub fn to_remove_event(&self, event_id: u64) -> RouterEvent {
        RouterEvent {
            worker_id: self.worker_id,
            event: KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: self.external_hashes.clone(),
                }),
                dp_rank: 0,
            },
        }
    }
}

/// Generate sequences with shared prefix prompts.
///
/// # Arguments
/// * `num_sequences` - Number of sequences to generate
/// * `depth` - Number of blocks per sequence
/// * `num_workers` - Number of workers to distribute sequences across
/// * `prefix_ratio` - Ratio of blocks that share a prefix (0.0 to 1.0)
/// * `num_prefix_groups` - Number of distinct prefix groups
/// * `seed` - Random seed for reproducibility
/// * `use_cumulative_hash` - If true, use `from_local_hashes` for proper cumulative hashes
pub fn generate_sequences(
    num_sequences: usize,
    depth: usize,
    num_workers: usize,
    prefix_ratio: f64,
    num_prefix_groups: usize,
    seed: u64,
    use_cumulative_hash: bool,
) -> Vec<SequenceData> {
    let mut sequences = Vec::with_capacity(num_sequences);
    let prefix_length = (depth as f64 * prefix_ratio).round() as usize;
    let mut rng: StdRng = StdRng::seed_from_u64(seed);

    for seq_id in 0..num_sequences {
        let seq_id_u64 = seq_id as u64;
        let worker_id = (seq_id % num_workers) as WorkerId;

        // Determine prefix group for this sequence
        let group_id = if num_prefix_groups > 0 && prefix_length > 0 {
            Some(rng.random_range(0..num_prefix_groups) as u64)
        } else {
            None
        };

        // Build local_hashes: shared prefix (if applicable) + unique suffix
        let local_hashes: Vec<LocalBlockHash> = (0..depth)
            .map(|block_idx| {
                let block_idx_u64 = block_idx as u64;
                if let Some(gid) = group_id
                    && block_idx < prefix_length
                {
                    // Shared prefix based on group_id
                    return LocalBlockHash(0xDEAD_BEEF_0000_0000 | (gid << 32) | block_idx_u64);
                }
                // Unique suffix (or no shared prefix)
                LocalBlockHash((seq_id_u64 << 32) | block_idx_u64)
            })
            .collect();

        if use_cumulative_hash {
            sequences.push(SequenceData::from_local_hashes(worker_id, local_hashes));
        } else {
            let external_hashes: Vec<ExternalSequenceBlockHash> = (0..depth)
                .map(|block_idx| {
                    let block_idx_u64 = block_idx as u64;
                    if let Some(gid) = group_id
                        && block_idx < prefix_length
                    {
                        return ExternalSequenceBlockHash(
                            0xDEAD_BEEF_0000_0000 | (gid << 32) | block_idx_u64,
                        );
                    }
                    ExternalSequenceBlockHash((seq_id_u64 << 32) | block_idx_u64)
                })
                .collect();

            sequences.push(SequenceData {
                worker_id,
                local_hashes,
                external_hashes,
            });
        }
    }

    sequences
}

/// Compute median of durations.
pub fn median(durations: &[Duration]) -> Duration {
    if durations.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted = durations.to_vec();
    sorted.sort();
    sorted[sorted.len() / 2]
}
