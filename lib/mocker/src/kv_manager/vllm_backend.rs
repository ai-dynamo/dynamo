// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Manual backend for the mocker's KV manager.
//!
//! Uses [`HashCache`] for O(1) block lookups with active/inactive pool management
//! and manual reference counting.

use crate::cache::HashCache;
use crate::common::protocols::{KvCacheEventSink, MoveBlock};
use crate::kv_manager::KvBackend;
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash,
};
use dynamo_runtime::config::environment_names::mocker;
use dynamo_tokens::blocks::UniqueBlock;
use dynamo_tokens::{BlockHash, PositionalLineageHash, SequenceHash};
use std::collections::HashMap;
use std::env;
use std::sync::{Arc, LazyLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Check the env var to enable KV cache allocation/eviction trace logs.
static KV_CACHE_TRACE_ENABLED: LazyLock<bool> = LazyLock::new(|| {
    env::var(mocker::DYN_MOCKER_KV_CACHE_TRACE)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
});

pub struct ManualKvManager {
    cache: HashCache,
    block_size: usize,
    kv_event_sink: Option<Arc<dyn KvCacheEventSink>>,
    dp_rank: u32,
    next_event_id: u64,
}

impl ManualKvManager {
    pub fn new(max_capacity: usize, block_size: usize) -> Self {
        Self::new_with_event_sink(max_capacity, block_size, None, 0)
    }

    pub fn new_with_event_sink(
        max_capacity: usize,
        block_size: usize,
        kv_event_sink: Option<Arc<dyn KvCacheEventSink>>,
        dp_rank: u32,
    ) -> Self {
        debug_assert!(max_capacity > 0, "max_capacity must be > 0");
        if kv_event_sink.is_some() {
            tracing::info!(
                "ManualKvManager initialized with event sink for DP rank {dp_rank} with block_size {block_size}"
            );
        }

        ManualKvManager {
            cache: HashCache::new(max_capacity),
            block_size,
            kv_event_sink,
            dp_rank,
            next_event_id: 0,
        }
    }

    /// Converts stored/removed blocks into KvCacheEventData and publishes if sink is available.
    fn publish_kv_event(
        &mut self,
        full_blocks: Vec<SequenceHash>,
        local_hashes: &[BlockHash],
        parent_hash: Option<u64>,
        is_store: bool,
    ) {
        if full_blocks.is_empty() {
            return;
        }

        if *KV_CACHE_TRACE_ENABLED {
            let active_len = self.cache.num_active();
            let inactive_len = self.cache.num_inactive();
            let free_blocks = self
                .cache
                .max_capacity()
                .saturating_sub(active_len)
                .saturating_sub(inactive_len);
            let event = if is_store { "allocation" } else { "eviction" };
            let timestamp_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            tracing::info!(
                event,
                timestamp_ms,
                block_ids = ?&full_blocks,
                block_size = self.block_size,
                free_blocks_after = free_blocks,
                active_blocks = active_len,
                inactive_blocks = inactive_len,
                total_blocks = self.cache.max_capacity(),
                dp_rank = self.dp_rank,
                "KV cache trace"
            );
        }

        let Some(ref sink) = self.kv_event_sink else {
            return;
        };

        let event_data = if is_store {
            let num_blocks = full_blocks.len();
            let local_hashes_slice = &local_hashes[local_hashes
                .len()
                .checked_sub(num_blocks)
                .expect("local hashes fewer than stored blocks")..];

            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                blocks: full_blocks
                    .into_iter()
                    .zip(local_hashes_slice.iter())
                    .map(|(global_hash, local_hash)| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(global_hash),
                        tokens_hash: LocalBlockHash(*local_hash),
                        mm_extra_info: None,
                    })
                    .collect(),
            })
        } else {
            KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: full_blocks
                    .into_iter()
                    .map(ExternalSequenceBlockHash)
                    .collect(),
            })
        };

        let event_id = self.next_event_id;
        self.next_event_id += 1;

        let event = KvCacheEvent {
            event_id,
            data: event_data,
            dp_rank: self.dp_rank,
        };

        if let Err(e) = sink.publish(event) {
            tracing::warn!("Failed to publish KV event: {e}");
        }
    }

    /// Get the keys of inactive blocks
    pub fn get_inactive_blocks(&self) -> Vec<&UniqueBlock> {
        self.cache.inactive_keys().collect()
    }

    /// Get the keys of active blocks
    pub fn get_active_blocks(&self) -> Vec<&UniqueBlock> {
        self.cache.active_keys().collect()
    }

    /// Direct access to active blocks map (for tests).
    pub fn active_blocks(&self) -> &HashMap<UniqueBlock, usize> {
        self.cache.active_blocks()
    }
}

impl KvBackend for ManualKvManager {
    fn process(&mut self, event: &MoveBlock) -> bool {
        match event {
            MoveBlock::Use(hashes, local_hashes, _plhs) => {
                let mut blocks_stored = Vec::<u64>::new();

                let mut parent_block: Option<&UniqueBlock> = None;
                for hash in hashes {
                    if self.cache.contains_active(hash) {
                        self.cache.increment_ref(hash);
                        parent_block = Some(hash);
                        continue;
                    }

                    if self.cache.reactivate(hash) {
                        parent_block = Some(hash);
                        continue;
                    }

                    if self.cache.is_at_capacity() {
                        let Some(evicted) = self.cache.evict_inactive() else {
                            return false;
                        };
                        tracing::trace!(
                            "Evicting block from inactive pool: {evicted:?}, dp_rank={}",
                            self.dp_rank
                        );
                        if let UniqueBlock::FullBlock(evicted_full_block) = evicted {
                            self.publish_kv_event(vec![evicted_full_block], &[], None, false);
                        }
                    }

                    self.cache.insert_active(hash.clone(), 1);
                    if let UniqueBlock::FullBlock(stored_full_block) = hash {
                        blocks_stored.push(*stored_full_block);
                    }
                }

                let parent_hash = match parent_block {
                    None => None,
                    Some(UniqueBlock::FullBlock(block)) => Some(*block),
                    Some(UniqueBlock::PartialBlock(_)) => panic!("parent block cannot be partial"),
                };
                self.publish_kv_event(blocks_stored, local_hashes, parent_hash, true);
            }

            MoveBlock::Destroy(hashes) => {
                let mut blocks_destroyed = Vec::<u64>::new();
                for hash in hashes.iter() {
                    self.cache.remove_active(hash).unwrap();
                    if let UniqueBlock::FullBlock(destroyed_full_block) = hash {
                        blocks_destroyed.push(*destroyed_full_block);
                    }
                }
                self.publish_kv_event(blocks_destroyed, &[], None, false);
            }

            MoveBlock::Deref(hashes) => {
                for hash in hashes.iter() {
                    if let Some(ref_count) = self.cache.get_active_ref_count(hash) {
                        if ref_count == 0 {
                            panic!("Negative reference count would be encountered after Deref.");
                        }
                        if ref_count == 1 {
                            self.cache.deactivate(hash);
                        } else {
                            self.cache.decrement_ref(hash);
                        }
                    }
                }
            }

            MoveBlock::Promote(uuid, hash, parent_hash, local_hash, _plh) => {
                let uuid_block = UniqueBlock::PartialBlock(*uuid);
                let hash_block = UniqueBlock::FullBlock(*hash);

                assert_eq!(
                    self.cache.remove_active(&uuid_block),
                    Some(1),
                    "uuid_block {uuid_block:?} should exist and be unique with ref_count=1"
                );

                let hash_ref_count = self.cache.get_active_ref_count(&hash_block);
                let is_new = if hash_ref_count.is_some() {
                    false
                } else {
                    !self.cache.remove_inactive(&hash_block)
                };

                self.cache
                    .insert_active(hash_block, hash_ref_count.unwrap_or(0) + 1);

                if is_new {
                    self.publish_kv_event(vec![*hash], &[*local_hash], *parent_hash, true);
                }
            }
        }

        true
    }

    fn max_capacity(&self) -> usize {
        self.cache.max_capacity()
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn num_active_blocks(&self) -> usize {
        self.cache.num_active()
    }

    fn num_inactive_blocks(&self) -> usize {
        self.cache.num_inactive()
    }

    fn current_capacity(&self) -> usize {
        self.cache.current_capacity()
    }

    fn probe_new_blocks(&self, blocks: &[UniqueBlock]) -> usize {
        blocks
            .iter()
            .filter(|&block| !self.cache.contains(block))
            .count()
    }

    fn is_block_cached(&self, seq_hash: u64, _plh: Option<PositionalLineageHash>) -> bool {
        let block = UniqueBlock::FullBlock(seq_hash);
        self.cache.contains(&block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_failure_on_max_capacity() {
        let mut manager = ManualKvManager::new(10, 16);

        fn use_blocks(manager: &mut ManualKvManager, ids: Vec<u64>) -> bool {
            let blocks: Vec<_> = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            let hashes: Vec<_> = ids.into_iter().collect();
            manager.process(&MoveBlock::Use(blocks, hashes, vec![]))
        }

        let response = use_blocks(&mut manager, (0..10).collect());
        assert!(response, "Expected success response");
        assert_eq!(manager.current_capacity(), 10);

        let response = use_blocks(&mut manager, vec![10]);
        assert!(
            !response,
            "Expected failure response when exceeding max capacity"
        );
    }

    #[test]
    fn test_block_lifecycle_stringent() {
        let mut manager = ManualKvManager::new(10, 16);

        fn use_blocks(manager: &mut ManualKvManager, ids: Vec<u64>) {
            let blocks: Vec<_> = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            let hashes: Vec<_> = ids.into_iter().collect();
            manager.process(&MoveBlock::Use(blocks, hashes, vec![]));
        }

        fn destroy_blocks(manager: &mut ManualKvManager, ids: Vec<u64>) {
            let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
            manager.process(&MoveBlock::Destroy(blocks));
        }

        fn deref_blocks(manager: &mut ManualKvManager, ids: Vec<u64>) {
            let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
            manager.process(&MoveBlock::Deref(blocks));
        }

        fn assert_active_blocks(manager: &ManualKvManager, expected_blocks: &[(u64, usize)]) {
            assert_eq!(
                manager.active_blocks().len(),
                expected_blocks.len(),
                "Active blocks count doesn't match expected"
            );
            for &(id, ref_count) in expected_blocks {
                let block = UniqueBlock::FullBlock(id);
                assert!(
                    manager.active_blocks().contains_key(&block),
                    "Block {id} not found in active blocks",
                );
                assert_eq!(
                    manager.active_blocks().get(&block),
                    Some(&ref_count),
                    "Block {id} has wrong reference count",
                );
            }
        }

        fn assert_inactive_blocks(
            manager: &ManualKvManager,
            expected_size: usize,
            expected_blocks: &[u64],
        ) {
            let inactive_blocks = manager.get_inactive_blocks();
            let inactive_blocks_count = manager.num_inactive_blocks();
            assert_eq!(
                inactive_blocks_count, expected_size,
                "Inactive blocks count doesn't match expected"
            );
            for &id in expected_blocks {
                let block = UniqueBlock::FullBlock(id);
                assert!(
                    inactive_blocks.iter().any(|&b| *b == block),
                    "Block {id} not found in inactive blocks",
                );
            }
        }

        use_blocks(&mut manager, (0..5).collect());
        use_blocks(&mut manager, vec![0, 1, 5, 6]);
        assert_active_blocks(
            &manager,
            &[(0, 2), (1, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],
        );

        destroy_blocks(&mut manager, vec![4]);
        deref_blocks(&mut manager, vec![0, 1, 2, 3]);
        assert_inactive_blocks(&manager, 2, &[3, 2]);
        assert_active_blocks(&manager, &[(0, 1), (1, 1), (5, 1), (6, 1)]);

        destroy_blocks(&mut manager, vec![6]);
        deref_blocks(&mut manager, vec![0, 1, 5]);
        assert_inactive_blocks(&manager, 5, &[0, 1, 2, 3, 5]);
        assert_active_blocks(&manager, &[]);

        use_blocks(&mut manager, vec![0, 1, 2, 7, 8, 9]);
        assert_inactive_blocks(&manager, 2, &[3, 5]);
        assert_active_blocks(&manager, &[(0, 1), (1, 1), (2, 1), (7, 1), (8, 1), (9, 1)]);

        let blocks_to_check: Vec<UniqueBlock> = vec![0, 1, 2, 3, 4]
            .into_iter()
            .map(UniqueBlock::FullBlock)
            .collect();
        assert_eq!(manager.probe_new_blocks(&blocks_to_check), 1);

        use_blocks(&mut manager, vec![10, 11, 12]);
        assert_inactive_blocks(&manager, 1, &[5]);

        use_blocks(&mut manager, vec![13]);
    }
}
