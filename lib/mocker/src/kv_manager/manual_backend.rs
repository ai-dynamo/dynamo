// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Manual backend for the mocker's KV manager.
//!
//! Original HashMap + LRUEvictor reference-counting implementation that tracks
//! active/inactive block pools and publishes KV cache events.

use std::collections::HashMap;
use std::env;
use std::sync::{Arc, LazyLock};
use std::time::{SystemTime, UNIX_EPOCH};

use derive_getters::Getters;
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash,
};
use dynamo_runtime::config::environment_names::mocker;
use dynamo_tokens::blocks::UniqueBlock;
use dynamo_tokens::{BlockHash, SequenceHash};

use crate::evictor::LRUEvictor;
use crate::kv_manager::KvBackend;
use crate::protocols::{KvCacheEventSink, MoveBlock};
use dynamo_tokens::PositionalLineageHash;

/// Check the env var to enable KV cache allocation/eviction trace logs.
static KV_CACHE_TRACE_ENABLED: LazyLock<bool> = LazyLock::new(|| {
    env::var(mocker::DYN_MOCKER_KV_CACHE_TRACE)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
});

#[derive(Getters)]
pub struct ManualKvManager {
    #[getter(copy)]
    max_capacity: usize,

    #[getter(copy)]
    block_size: usize,

    active_blocks: HashMap<UniqueBlock, usize>,

    inactive_blocks: LRUEvictor<UniqueBlock>,

    kv_event_sink: Option<Arc<dyn KvCacheEventSink>>,

    #[getter(copy)]
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
        let active_blocks = HashMap::new();
        let inactive_blocks = LRUEvictor::default();

        if kv_event_sink.is_some() {
            tracing::info!(
                "ManualKvManager initialized with event sink for DP rank {dp_rank} with block_size {block_size}"
            );
        }

        ManualKvManager {
            max_capacity,
            block_size,
            active_blocks,
            inactive_blocks,
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
            let timestamp_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            let active_len = self.active_blocks.len();
            let inactive_len = self.inactive_blocks.len();
            let free_blocks = self
                .max_capacity
                .saturating_sub(active_len)
                .saturating_sub(inactive_len);
            let event = if is_store { "allocation" } else { "eviction" };
            tracing::info!(
                event,
                timestamp_ms,
                block_ids = ?&full_blocks,
                block_size = self.block_size,
                free_blocks_after = free_blocks,
                active_blocks = active_len,
                inactive_blocks = inactive_len,
                total_blocks = self.max_capacity,
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

        // Use incremental event ID starting from 0
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
        self.inactive_blocks.keys().collect()
    }

    /// Get the keys of active blocks
    pub fn get_active_blocks(&self) -> Vec<&UniqueBlock> {
        self.active_blocks.keys().collect()
    }
}

impl KvBackend for ManualKvManager {
    fn process(&mut self, event: &MoveBlock) -> bool {
        match event {
            MoveBlock::Use(hashes, local_hashes, _plhs) => {
                let mut blocks_stored = Vec::<u64>::new();

                let mut parent_block: Option<&UniqueBlock> = None;
                for hash in hashes {
                    // First check if it already exists in active blocks
                    if let Some(ref_count) = self.active_blocks.get_mut(hash) {
                        // Block already active, just increment reference count
                        *ref_count += 1;
                        parent_block = Some(hash);
                        continue;
                    }

                    // Then check if it exists in inactive and move it to active if found
                    if self.inactive_blocks.remove(hash) {
                        // Insert into active with reference count 1
                        self.active_blocks.insert(hash.clone(), 1);
                        parent_block = Some(hash);
                        continue;
                    }

                    // Get counts for capacity check
                    let active_count = self.active_blocks.len();
                    let inactive_count = self.inactive_blocks.len();

                    // If at max capacity, evict the oldest entry from inactive blocks
                    if active_count + inactive_count >= self.max_capacity {
                        let Some(evicted) = self.inactive_blocks.evict() else {
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

                    // Now insert the new block in active blocks with reference count 1
                    self.active_blocks.insert(hash.clone(), 1);
                    // Track blocks for trace/event
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

                // Process blocks in order (already reversed by caller if needed)
                for hash in hashes.iter() {
                    self.active_blocks.remove(hash).unwrap();

                    // Track blocks for batch sending
                    if let UniqueBlock::FullBlock(destroyed_full_block) = hash {
                        blocks_destroyed.push(*destroyed_full_block);
                    }
                }

                self.publish_kv_event(blocks_destroyed, &[], None, false);
            }

            MoveBlock::Deref(hashes) => {
                // Process blocks in order (already reversed by caller if needed)
                for hash in hashes.iter() {
                    // Decrement reference count and check if we need to move to inactive
                    if let Some(ref_count) = self.active_blocks.get_mut(hash) {
                        if *ref_count == 0 {
                            panic!("Negative reference count would be encountered after Deref.");
                        }
                        *ref_count -= 1;

                        // If reference count reaches zero, remove from active and move to inactive
                        if *ref_count == 0 {
                            self.active_blocks.remove(hash);
                            // Use the LRUEvictor's timing functionality
                            self.inactive_blocks.insert(hash.clone());
                        }
                    }
                }
            }

            MoveBlock::Promote(uuid, hash, parent_hash, local_hash, _plh) => {
                let uuid_block = UniqueBlock::PartialBlock(*uuid);
                let hash_block = UniqueBlock::FullBlock(*hash);

                assert_eq!(
                    self.active_blocks.remove(&uuid_block),
                    Some(1),
                    "uuid_block {uuid_block:?} should exist and be unique with ref_count=1"
                );

                let hash_ref_count = self.active_blocks.get(&hash_block).copied();
                let is_new = hash_ref_count.is_none() && !self.inactive_blocks.remove(&hash_block);

                self.active_blocks
                    .insert(hash_block.clone(), hash_ref_count.unwrap_or(0) + 1);

                if is_new {
                    self.publish_kv_event(vec![*hash], &[*local_hash], *parent_hash, true);
                }
            }
        }

        // Return true if we made it this far
        true
    }

    fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn num_active_blocks(&self) -> usize {
        self.active_blocks.len()
    }

    fn num_inactive_blocks(&self) -> usize {
        self.inactive_blocks.len()
    }

    fn current_capacity(&self) -> usize {
        self.active_blocks.len() + self.inactive_blocks.len()
    }

    fn probe_new_blocks(&self, blocks: &[UniqueBlock]) -> usize {
        blocks
            .iter()
            .filter(|&block| {
                !self.active_blocks.contains_key(block) && !self.inactive_blocks.contains(block)
            })
            .count()
    }

    fn is_block_cached(&self, seq_hash: u64, _plh: Option<PositionalLineageHash>) -> bool {
        let block = UniqueBlock::FullBlock(seq_hash);
        self.active_blocks.contains_key(&block) || self.inactive_blocks.contains(&block)
    }
}
