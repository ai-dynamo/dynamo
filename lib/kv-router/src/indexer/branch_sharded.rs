// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use async_trait::async_trait;
use dashmap::DashMap;
use rustc_hash::FxBuildHasher;
use tokio_util::sync::CancellationToken;

use super::{KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvRouterError};
use crate::indexer::pruning::PruneConfig;
use crate::protocols::*;

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BranchKey {
    depth: u32,
    hash: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct WorkerSequenceKey {
    worker: WorkerWithDpRank,
    sequence_hash: ExternalSequenceBlockHash,
}

#[derive(Clone, Debug)]
struct BranchState {
    branch_key: BranchKey,
    primary_shard: usize,
    resident_shards: Vec<usize>,
    prefix_blocks: Vec<KvCacheStoredBlockData>,
}

#[derive(Clone)]
pub struct BranchShardedIndexer {
    kv_block_size: u32,
    prefix_depth: u32,
    shards: Vec<KvIndexer>,
    branch_to_shards: DashMap<BranchKey, Vec<usize>, FxBuildHasher>,
    sequence_states: DashMap<WorkerSequenceKey, BranchState, FxBuildHasher>,
}

impl BranchShardedIndexer {
    pub fn new_with_frequency(
        token: CancellationToken,
        num_shards: usize,
        prefix_depth: u32,
        expiration_duration: Option<Duration>,
        kv_block_size: u32,
        metrics: std::sync::Arc<KvIndexerMetrics>,
        prune_config: Option<PruneConfig>,
    ) -> Self {
        assert!(
            num_shards > 0,
            "BranchShardedIndexer requires at least one shard"
        );

        let shards = (0..num_shards)
            .map(|_| {
                KvIndexer::new_with_frequency(
                    token.clone(),
                    expiration_duration,
                    kv_block_size,
                    metrics.clone(),
                    prune_config.clone(),
                )
            })
            .collect();

        Self {
            kv_block_size,
            prefix_depth,
            shards,
            branch_to_shards: DashMap::with_hasher(FxBuildHasher),
            sequence_states: DashMap::with_hasher(FxBuildHasher),
        }
    }

    pub fn new(
        token: CancellationToken,
        num_shards: usize,
        prefix_depth: u32,
        kv_block_size: u32,
        metrics: std::sync::Arc<KvIndexerMetrics>,
    ) -> Self {
        Self::new_with_frequency(
            token,
            num_shards,
            prefix_depth,
            None,
            kv_block_size,
            metrics,
            None,
        )
    }

    fn assign_shard(&self, key: BranchKey) -> usize {
        (key.hash as usize) % self.shards.len()
    }

    fn branch_key_for_sequence(&self, sequence: &[LocalBlockHash]) -> Option<BranchKey> {
        let depth = (sequence.len() as u32).min(self.prefix_depth);
        (depth > 0).then(|| BranchKey {
            depth,
            hash: fnv_branch_hash(&sequence[..depth as usize]),
        })
    }

    fn register_shard_for_key(&self, key: BranchKey, shard: usize) {
        let mut entry = self.branch_to_shards.entry(key).or_default();
        if !entry.contains(&shard) {
            entry.push(shard);
        }
    }

    fn register_prefix_keys(&self, blocks: &[KvCacheStoredBlockData], shard: usize) {
        let mut rolling_hash = FNV_OFFSET_BASIS;
        for (idx, block) in blocks.iter().take(self.prefix_depth as usize).enumerate() {
            rolling_hash = fnv_extend(rolling_hash, block.tokens_hash);
            self.register_shard_for_key(
                BranchKey {
                    depth: (idx + 1) as u32,
                    hash: rolling_hash,
                },
                shard,
            );
        }
    }

    fn targeted_shards(&self, sequence: &[LocalBlockHash]) -> Option<Vec<usize>> {
        let key = self.branch_key_for_sequence(sequence)?;
        self.branch_to_shards.get(&key).map(|shards| shards.clone())
    }

    pub async fn get_workers(&self) -> Result<Vec<WorkerId>, KvRouterError> {
        let mut workers = std::collections::BTreeSet::new();
        for shard in &self.shards {
            for event in shard.dump_events().await? {
                workers.insert(event.worker_id);
            }
        }
        Ok(workers.into_iter().collect())
    }

    fn compute_store_plan(
        &self,
        worker: WorkerWithDpRank,
        store_data: &KvCacheStoreData,
    ) -> Option<StorePlan> {
        let parent_state = store_data.parent_hash.and_then(|parent_hash| {
            self.sequence_states
                .get(&WorkerSequenceKey {
                    worker,
                    sequence_hash: parent_hash,
                })
                .map(|state| state.clone())
        });

        let mut prefix_blocks = parent_state
            .as_ref()
            .map(|state| state.prefix_blocks.clone())
            .unwrap_or_default();
        let mut rolling_hash = if let Some(parent_state) = &parent_state {
            parent_state.branch_key.hash
        } else {
            FNV_OFFSET_BASIS
        };
        let mut depth = prefix_blocks.len() as u32;

        for block in store_data.blocks.iter() {
            if depth < self.prefix_depth {
                rolling_hash = fnv_extend(rolling_hash, block.tokens_hash);
                prefix_blocks.push(block.clone());
                depth += 1;
            }
        }

        let target_key = if depth > 0 {
            BranchKey {
                depth,
                hash: rolling_hash,
            }
        } else {
            return None;
        };
        let target_shard = self.assign_shard(target_key);

        let replay_blocks = parent_state.as_ref().and_then(|parent_state| {
            (parent_state.primary_shard != target_shard
                && parent_state.branch_key.depth < self.prefix_depth
                && !parent_state.prefix_blocks.is_empty())
            .then(|| parent_state.prefix_blocks.clone())
        });

        let mut ancestor_updates = Vec::new();
        if replay_blocks.is_some() {
            if let Some(parent_state) = &parent_state {
                for ancestor in prefix_blocks.iter().take(parent_state.prefix_blocks.len()) {
                    ancestor_updates.push(WorkerSequenceKey {
                        worker,
                        sequence_hash: ancestor.block_hash,
                    });
                }
            }
        }

        let mut state_updates = Vec::with_capacity(store_data.blocks.len());
        let mut effective_prefix = parent_state
            .as_ref()
            .map(|state| state.prefix_blocks.clone())
            .unwrap_or_default();
        let mut effective_hash = if let Some(parent_state) = &parent_state {
            parent_state.branch_key.hash
        } else {
            FNV_OFFSET_BASIS
        };
        let mut effective_depth = effective_prefix.len() as u32;

        for block in store_data.blocks.iter() {
            if effective_depth < self.prefix_depth {
                effective_hash = fnv_extend(effective_hash, block.tokens_hash);
                effective_prefix.push(block.clone());
                effective_depth += 1;
            }

            state_updates.push((
                WorkerSequenceKey {
                    worker,
                    sequence_hash: block.block_hash,
                },
                BranchState {
                    branch_key: BranchKey {
                        depth: effective_depth,
                        hash: effective_hash,
                    },
                    primary_shard: target_shard,
                    resident_shards: vec![target_shard],
                    prefix_blocks: effective_prefix.clone(),
                },
            ));
        }

        Some(StorePlan {
            target_shard,
            replay_blocks,
            ancestor_updates,
            state_updates,
            registered_prefix_blocks: prefix_blocks,
        })
    }

    async fn query_shards(
        &self,
        sequence: Vec<LocalBlockHash>,
        mut shard_indices: Vec<usize>,
    ) -> Result<OverlapScores, KvRouterError> {
        shard_indices.sort_unstable();
        shard_indices.dedup();

        let mut merged = OverlapScores::new();
        for shard in shard_indices {
            let scores = self.shards[shard].find_matches(sequence.clone()).await?;
            merge_overlap_scores(&mut merged, scores);
        }
        Ok(merged)
    }

    fn remove_sequence_state(&self, worker: WorkerWithDpRank, hash: ExternalSequenceBlockHash) {
        self.sequence_states.remove(&WorkerSequenceKey {
            worker,
            sequence_hash: hash,
        });
    }

    fn remove_worker_states(&self, worker: WorkerId, dp_rank: Option<DpRank>) {
        let keys: Vec<_> = self
            .sequence_states
            .iter()
            .filter_map(|entry| {
                let key = *entry.key();
                (key.worker.worker_id == worker
                    && dp_rank.is_none_or(|rank| key.worker.dp_rank == rank))
                .then_some(key)
            })
            .collect();

        for key in keys {
            self.sequence_states.remove(&key);
        }
    }
}

#[async_trait]
impl KvIndexerInterface for BranchShardedIndexer {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        if let Some(shards) = self.targeted_shards(&sequence) {
            return self.query_shards(sequence, shards).await;
        }

        self.query_shards(sequence, (0..self.shards.len()).collect())
            .await
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
        lora_name: Option<&str>,
    ) -> Result<OverlapScores, KvRouterError> {
        let sequence = compute_block_hash_for_seq(tokens, self.kv_block_size, None, lora_name);
        self.find_matches(sequence).await
    }

    async fn apply_event(&self, event: RouterEvent) {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);

        match &event.event.data {
            KvCacheEventData::Stored(store_data) => {
                let Some(plan) = self.compute_store_plan(worker, store_data) else {
                    for shard in &self.shards {
                        shard.apply_event(event.clone()).await;
                    }
                    return;
                };

                if let Some(replay_blocks) = &plan.replay_blocks {
                    self.register_prefix_keys(replay_blocks, plan.target_shard);
                    let replay_event = RouterEvent::new(
                        event.worker_id,
                        KvCacheEvent {
                            event_id: event.event.event_id,
                            data: KvCacheEventData::Stored(KvCacheStoreData {
                                parent_hash: None,
                                blocks: replay_blocks.clone(),
                            }),
                            dp_rank: event.event.dp_rank,
                        },
                    );
                    self.shards[plan.target_shard]
                        .apply_event(replay_event)
                        .await;
                }

                self.register_prefix_keys(&plan.registered_prefix_blocks, plan.target_shard);
                self.shards[plan.target_shard].apply_event(event).await;

                for ancestor_key in plan.ancestor_updates {
                    if let Some(mut state) = self.sequence_states.get_mut(&ancestor_key) {
                        state.primary_shard = plan.target_shard;
                        if !state.resident_shards.contains(&plan.target_shard) {
                            state.resident_shards.push(plan.target_shard);
                        }
                    }
                }

                for (key, state) in plan.state_updates {
                    self.sequence_states.insert(key, state);
                }
            }
            KvCacheEventData::Removed(remove_data) => {
                let mut shard_indices = Vec::with_capacity(remove_data.block_hashes.len());
                for hash in &remove_data.block_hashes {
                    let resident_shards = self
                        .sequence_states
                        .get(&WorkerSequenceKey {
                            worker,
                            sequence_hash: *hash,
                        })
                        .map(|state| state.resident_shards.clone())
                        .unwrap_or_default();
                    self.remove_sequence_state(worker, *hash);
                    shard_indices.extend(resident_shards);
                }

                if shard_indices.is_empty() {
                    for shard in &self.shards {
                        shard.apply_event(event.clone()).await;
                    }
                } else {
                    shard_indices.sort_unstable();
                    shard_indices.dedup();
                    for shard in shard_indices {
                        self.shards[shard].apply_event(event.clone()).await;
                    }
                }
            }
            KvCacheEventData::Cleared => {
                self.remove_worker_states(event.worker_id, Some(event.event.dp_rank));
                for shard in &self.shards {
                    shard.apply_event(event.clone()).await;
                }
            }
        }
    }

    async fn remove_worker(&self, worker: WorkerId) {
        self.remove_worker_states(worker, None);
        for shard in &self.shards {
            shard.remove_worker(worker).await;
        }
    }

    async fn remove_worker_dp_rank(&self, worker: WorkerId, dp_rank: DpRank) {
        self.remove_worker_states(worker, Some(dp_rank));
        for shard in &self.shards {
            shard.remove_worker_dp_rank(worker, dp_rank).await;
        }
    }

    fn shutdown(&self) {
        if let Some(shard) = self.shards.first() {
            shard.shutdown();
        }
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        let mut all_events = Vec::new();
        for shard in &self.shards {
            all_events.extend(shard.dump_events().await?);
        }
        Ok(all_events)
    }

    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        let local_hashes = tokens_with_hashes.get_or_compute_block_hashes().to_vec();
        let sequence_hashes = tokens_with_hashes.get_or_compute_seq_hashes().to_vec();
        let shard = self
            .branch_key_for_sequence(&local_hashes)
            .map(|key| self.assign_shard(key))
            .unwrap_or_default();

        self.shards[shard]
            .process_routing_decision_for_request(tokens_with_hashes, worker)
            .await?;

        for (idx, seq_hash) in sequence_hashes.iter().enumerate() {
            let prefix_len = ((idx + 1) as u32).min(self.prefix_depth) as usize;
            if prefix_len == 0 {
                continue;
            }
            let key = BranchKey {
                depth: prefix_len as u32,
                hash: fnv_branch_hash(&local_hashes[..prefix_len]),
            };
            self.register_shard_for_key(key, shard);
            self.sequence_states.insert(
                WorkerSequenceKey {
                    worker,
                    sequence_hash: ExternalSequenceBlockHash(*seq_hash),
                },
                BranchState {
                    branch_key: key,
                    primary_shard: shard,
                    resident_shards: vec![shard],
                    prefix_blocks: sequence_hashes
                        .iter()
                        .zip(local_hashes.iter())
                        .take(prefix_len)
                        .map(|(seq_hash, local_hash)| KvCacheStoredBlockData {
                            tokens_hash: *local_hash,
                            block_hash: ExternalSequenceBlockHash(*seq_hash),
                            mm_extra_info: None,
                        })
                        .collect(),
                },
            );
        }

        Ok(())
    }

    async fn flush(&self) -> usize {
        let mut pending = 0;
        for shard in &self.shards {
            pending += shard.flush().await;
        }
        pending
    }
}

struct StorePlan {
    target_shard: usize,
    replay_blocks: Option<Vec<KvCacheStoredBlockData>>,
    ancestor_updates: Vec<WorkerSequenceKey>,
    state_updates: Vec<(WorkerSequenceKey, BranchState)>,
    registered_prefix_blocks: Vec<KvCacheStoredBlockData>,
}

fn merge_overlap_scores(into: &mut OverlapScores, from: OverlapScores) {
    for (worker, score) in from.scores {
        into.scores
            .entry(worker)
            .and_modify(|existing| *existing = (*existing).max(score))
            .or_insert(score);
    }

    for (worker, tree_size) in from.tree_sizes {
        *into.tree_sizes.entry(worker).or_insert(0) += tree_size;
    }

    if into.frequencies.len() < from.frequencies.len() {
        into.frequencies.resize(from.frequencies.len(), 0);
    }
    for (idx, frequency) in from.frequencies.into_iter().enumerate() {
        into.frequencies[idx] = into.frequencies[idx].max(frequency);
    }
}

fn fnv_branch_hash(sequence: &[LocalBlockHash]) -> u64 {
    sequence
        .iter()
        .fold(FNV_OFFSET_BASIS, |hash, block| fnv_extend(hash, *block))
}

fn fnv_extend(hash: u64, block: LocalBlockHash) -> u64 {
    let mut hash = hash;
    for byte in block.0.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
