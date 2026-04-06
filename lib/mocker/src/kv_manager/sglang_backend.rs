// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang KV manager — wraps [`RadixCache`] with request-level lifecycle
//! operations and KV event publishing.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::cache::radix_cache::{NodeId, RadixCache};
use crate::common::kv_cache_trace;
use crate::common::protocols::KvEventPublishers;
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData,
};

/// Result of `allocate_for_request`.
pub struct AllocResult {
    /// Number of tokens matched from the prefix cache.
    pub prefix_len: usize,
    /// Pool token indices for the allocated input (1 per token).
    pub kv_indices: Vec<usize>,
    /// The deepest matched node in the radix tree (used for lock/unlock).
    /// This is the prefix match point, not the new tokens — new tokens are
    /// only in kv_indices and get inserted into the tree on completion.
    pub last_node: NodeId,
}

pub struct SglangKvManager {
    cache: RadixCache,
    kv_event_publishers: KvEventPublishers,
    dp_rank: u32,
    next_event_id: u64,
    /// Maps pool_idx → block_hash assigned during Stored events,
    /// so Removed events can use the same block_hash.
    idx_to_block_hash: HashMap<usize, ExternalSequenceBlockHash>,
    /// Tracks how many live pool slots currently advertise the same logical
    /// block hash so router events reflect logical block visibility, not
    /// transient slot ownership.
    block_hash_refcounts: HashMap<ExternalSequenceBlockHash, usize>,
}

impl SglangKvManager {
    pub fn new(
        total_tokens: usize,
        page_size: usize,
        kv_event_publishers: KvEventPublishers,
        dp_rank: u32,
    ) -> Self {
        Self {
            cache: RadixCache::new(total_tokens, page_size),
            kv_event_publishers,
            dp_rank,
            next_event_id: 0,
            idx_to_block_hash: HashMap::new(),
            block_hash_refcounts: HashMap::new(),
        }
    }

    pub fn cache(&self) -> &RadixCache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut RadixCache {
        &mut self.cache
    }

    /// Try to allocate KV cache for a new request.
    /// Returns `None` if the pool doesn't have enough token slots (OOM).
    pub fn allocate_for_request(&mut self, token_ids: &[u64]) -> Option<AllocResult> {
        let (prefix_len, last_node) = self.cache.match_prefix(token_ids);

        let new_tokens = token_ids.len() - prefix_len;

        let prefix_indices = self.collect_path_indices(last_node);

        let new_indices = self.cache.token_pool.allocate(new_tokens)?;

        let mut kv_indices = prefix_indices;
        kv_indices.extend_from_slice(&new_indices);

        self.cache.inc_lock_ref(last_node);

        // Chain from prefix's last block_hash (if any)
        let parent_hash = kv_indices
            .get(prefix_len.wrapping_sub(1))
            .and_then(|&idx| self.idx_to_block_hash.get(&idx).copied());
        self.publish_stored_event(&token_ids[prefix_len..], &new_indices, parent_hash);

        self.log_trace("allocation", new_tokens);

        Some(AllocResult {
            prefix_len,
            kv_indices,
            last_node,
        })
    }

    /// Continue an in-flight request from an already materialized prefix.
    ///
    /// This is used by chunked-prefill continuation where the request still
    /// owns token slots for a prefix that may extend past the radix-tree's
    /// page-aligned cached prefix.
    pub fn allocate_after_prefix(
        &mut self,
        token_ids: &[u64],
        prefix_len: usize,
        prefix_indices: &[usize],
        last_node: NodeId,
    ) -> Option<AllocResult> {
        let new_tokens = token_ids.len().saturating_sub(prefix_len);
        let new_indices = self.cache.token_pool.allocate(new_tokens)?;

        let mut kv_indices = prefix_indices[..prefix_len].to_vec();
        kv_indices.extend_from_slice(&new_indices);

        self.cache.inc_lock_ref(last_node);

        let parent_hash = kv_indices
            .get(prefix_len.wrapping_sub(1))
            .and_then(|&idx| self.idx_to_block_hash.get(&idx).copied());
        self.publish_stored_event(&token_ids[prefix_len..], &new_indices, parent_hash);
        self.log_trace("allocation", new_tokens);

        Some(AllocResult {
            prefix_len,
            kv_indices,
            last_node,
        })
    }

    /// Cache a completed request's full sequence into the radix tree.
    ///
    /// Inserts the full token sequence so future requests can reuse it,
    /// then unlocks the path.
    pub fn cache_finished_req(
        &mut self,
        token_ids: &[u64],
        kv_indices: &[usize],
        last_node: NodeId,
    ) {
        self.cache.insert(token_ids, kv_indices);
        self.cache.dec_lock_ref(last_node);
    }

    /// Cache a partial sequence after a chunked prefill step.
    ///
    /// Inserts the partial sequence, then transfers the lock from the old
    /// path to the new (extended) path. The request is still active, so the
    /// new deepest node stays locked.
    ///
    /// Returns the new `last_node` that the caller should use for
    /// subsequent calls.
    pub fn cache_unfinished_req(
        &mut self,
        token_ids: &[u64],
        kv_indices: &[usize],
        last_node: NodeId,
    ) -> NodeId {
        self.cache.insert(token_ids, kv_indices);

        // Find the new deepest node after insert
        let (_, new_last_node) = self.cache.match_prefix(token_ids);

        // Transfer lock: release old path, protect new path
        self.cache.dec_lock_ref(last_node);
        self.cache.inc_lock_ref(new_last_node);

        new_last_node
    }

    /// Allocate a single token slot for decode output and publish a BlockStored event.
    /// `last_idx` is the request's previous pool index for chaining block_hash.
    pub fn allocate_decode_token(&mut self, last_idx: Option<usize>) -> Option<usize> {
        let indices = self.cache.token_pool.allocate(1)?;
        let idx = indices[0];
        let parent_hash = last_idx.and_then(|i| self.idx_to_block_hash.get(&i).copied());
        self.publish_stored_event(&[], &[idx], parent_hash);
        self.log_trace("allocation", 1);
        Some(idx)
    }

    /// Free a request without caching (e.g., aborted request).
    ///
    /// Unlocks the path without inserting into the tree.
    pub fn free_request(&mut self, last_node: NodeId) {
        self.cache.dec_lock_ref(last_node);
    }

    /// Return request-owned token slots to the free pool and publish matching
    /// removal events for any slots that were previously advertised to the router.
    pub fn free_indices(&mut self, indices: &[usize]) {
        if indices.is_empty() {
            return;
        }

        self.cache.token_pool.free(indices);
        self.publish_removed_event(indices);
        self.log_trace("free", indices.len());
    }

    /// Collect token indices from the matched prefix path by walking root→last_node.
    fn collect_path_indices(&self, last_node: NodeId) -> Vec<usize> {
        if last_node == self.cache.root() {
            return Vec::new();
        }

        // Walk from last_node to root, collecting node IDs
        let mut path = Vec::new();
        let mut current = last_node;
        loop {
            let node = self.cache.node(current);
            if node.parent.is_none() {
                break;
            }
            path.push(current);
            current = node.parent.unwrap();
        }
        path.reverse();

        // Collect token indices from each node's value
        let mut indices = Vec::new();
        for node_id in path {
            indices.extend_from_slice(&self.cache.node(node_id).value);
        }
        indices
    }

    /// Evict tokens from the cache, publish BlockRemoved events, and log a trace.
    pub fn evict(&mut self, num_tokens: usize) {
        let (evicted, evicted_indices) = self.cache.evict(num_tokens);
        if !evicted_indices.is_empty() {
            self.publish_removed_event(&evicted_indices);
        }
        self.log_trace("eviction", evicted);
    }

    fn log_trace(&self, event: &str, num_tokens: usize) {
        kv_cache_trace::log_sglang_trace(&kv_cache_trace::SglangCacheState {
            event,
            dp_rank: self.dp_rank,
            num_tokens,
            page_size: self.cache.page_size(),
            available_tokens: self.cache.available_tokens(),
            evictable_tokens: self.cache.evictable_size,
            protected_tokens: self.cache.protected_size,
            total_tokens: self.cache.total_tokens(),
        });
    }

    fn publish_stored_event(
        &mut self,
        token_ids: &[u64],
        indices: &[usize],
        parent_hash: Option<ExternalSequenceBlockHash>,
    ) {
        if indices.is_empty() {
            return;
        }
        let mut computed_blocks = Vec::with_capacity(indices.len());
        let mut running_hash = parent_hash.map_or(0u64, |h| h.0);
        for (i, &idx) in indices.iter().enumerate() {
            // tokens_hash: per-token content hash for router prefix matching
            let token = token_ids.get(i).copied().unwrap_or(idx as u64);
            let token_bytes = token.to_le_bytes();
            let tokens_hash = dynamo_kv_router::protocols::compute_block_hash(&token_bytes);

            // block_hash: cumulative hash (parent_hash, token_id) so it's unique
            // per position and uniform across workers with the same token sequence.
            let mut hasher = DefaultHasher::new();
            running_hash.hash(&mut hasher);
            tokens_hash.0.hash(&mut hasher);
            running_hash = hasher.finish();
            let block_hash = ExternalSequenceBlockHash(running_hash);

            self.idx_to_block_hash.insert(idx, block_hash);
            *self.block_hash_refcounts.entry(block_hash).or_default() += 1;
            computed_blocks.push(KvCacheStoredBlockData {
                block_hash,
                tokens_hash,
                mm_extra_info: None,
            });
        }

        if self.kv_event_publishers.is_empty() {
            return;
        }

        let first_new = computed_blocks.iter().position(|block| {
            self.block_hash_refcounts
                .get(&block.block_hash)
                .copied()
                .unwrap_or_default()
                == 1
        });
        let Some(first_new) = first_new else {
            return;
        };

        let parent_hash = if first_new == 0 {
            parent_hash
        } else {
            Some(computed_blocks[first_new - 1].block_hash)
        };
        let blocks = computed_blocks.into_iter().skip(first_new).collect();

        let event = KvCacheEvent {
            event_id: self.next_event_id,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash,
                blocks,
            }),
            dp_rank: self.dp_rank,
        };
        self.next_event_id += 1;

        if let Err(e) = self.kv_event_publishers.publish(event, None) {
            tracing::warn!("Failed to publish SGLang KV event: {e}");
        }
    }

    fn publish_removed_event(&mut self, evicted_indices: &[usize]) {
        if self.kv_event_publishers.is_empty() {
            return;
        }

        let mut block_hashes = Vec::new();
        for &idx in evicted_indices {
            let Some(block_hash) = self.idx_to_block_hash.remove(&idx) else {
                continue;
            };
            let Some(refcount) = self.block_hash_refcounts.get_mut(&block_hash) else {
                continue;
            };
            if *refcount > 1 {
                *refcount -= 1;
                continue;
            }
            self.block_hash_refcounts.remove(&block_hash);
            block_hashes.push(block_hash);
        }

        if block_hashes.is_empty() {
            return;
        }

        let event = KvCacheEvent {
            event_id: self.next_event_id,
            data: KvCacheEventData::Removed(KvCacheRemoveData { block_hashes }),
            dp_rank: self.dp_rank,
        };
        self.next_event_id += 1;

        if let Err(e) = self.kv_event_publishers.publish(event, None) {
            tracing::warn!("Failed to publish SGLang KV remove event: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::Mutex;

    use crate::common::protocols::KvCacheEventSink;

    struct MockSink {
        events: Mutex<Vec<KvCacheEvent>>,
    }

    impl MockSink {
        fn new() -> Self {
            Self {
                events: Mutex::new(Vec::new()),
            }
        }

        fn event_count(&self) -> usize {
            self.events.lock().unwrap().len()
        }

        fn clone_events(&self) -> Vec<KvCacheEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl KvCacheEventSink for MockSink {
        fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()> {
            self.events.lock().unwrap().push(event);
            Ok(())
        }
    }

    #[test]
    fn test_allocate_cache_miss() {
        let mut mgr = SglangKvManager::new(100, 1, KvEventPublishers::default(), 0);

        let result = mgr.allocate_for_request(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(result.prefix_len, 0);
        assert_eq!(result.kv_indices.len(), 5);
        assert_eq!(mgr.cache().token_pool.available(), 95);
    }

    #[test]
    fn test_allocate_cache_hit() {
        let mut mgr = SglangKvManager::new(100, 1, KvEventPublishers::default(), 0);

        // First request: allocate and cache
        let r1 = mgr.allocate_for_request(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(r1.kv_indices.len(), 5); // 5 pages (page_size=1)
        mgr.cache_finished_req(&[1, 2, 3, 4, 5], &r1.kv_indices, r1.last_node);

        // Second request with shared prefix
        let r2 = mgr.allocate_for_request(&[1, 2, 3, 4, 5, 6, 7]).unwrap();
        assert_eq!(r2.prefix_len, 5);
        assert_eq!(r2.kv_indices.len(), 7); // 5 reused + 2 new pages
        assert_eq!(mgr.cache().token_pool.available(), 93); // 100 - 5 - 2
    }

    #[test]
    fn test_free_request_without_caching() {
        let mut mgr = SglangKvManager::new(100, 1, KvEventPublishers::default(), 0);

        let result = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        mgr.free_request(result.last_node);

        // Path is unlocked, tokens still allocated in pool
        assert_eq!(mgr.cache().protected_size, 0);
    }

    #[test]
    fn test_event_publishing() {
        let sink = Arc::new(MockSink::new());
        let mut mgr =
            SglangKvManager::new(100, 1, KvEventPublishers::new(Some(sink.clone()), None), 0);

        let r = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        assert_eq!(sink.event_count(), 1); // BlockStored for 3 new pages

        mgr.cache_finished_req(&[1, 2, 3], &r.kv_indices, r.last_node);

        // Second request with full cache hit → no new events
        let r2 = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        assert_eq!(r2.prefix_len, 3);
        assert_eq!(sink.event_count(), 1); // no new event
    }

    #[test]
    fn test_duplicate_logical_blocks_publish_once_and_remove_once() {
        let sink = Arc::new(MockSink::new());
        let mut mgr =
            SglangKvManager::new(100, 1, KvEventPublishers::new(Some(sink.clone()), None), 0);

        let req1 = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        let req2 = mgr.allocate_for_request(&[1, 2, 3]).unwrap();

        let events = sink.clone_events();
        assert_eq!(events.len(), 1);
        let KvCacheEventData::Stored(store) = &events[0].data else {
            panic!("expected stored event");
        };
        assert_eq!(store.blocks.len(), 3);

        mgr.free_indices(&req1.kv_indices);
        assert_eq!(sink.event_count(), 1);

        mgr.free_indices(&req2.kv_indices);
        let events = sink.clone_events();
        assert_eq!(events.len(), 2);
        let KvCacheEventData::Removed(remove) = &events[1].data else {
            panic!("expected removed event");
        };
        assert_eq!(remove.block_hashes.len(), 3);
    }

    #[test]
    fn test_allocate_oom() {
        let mut mgr = SglangKvManager::new(3, 1, KvEventPublishers::default(), 0);

        let _r = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        // Pool is full
        let result = mgr.allocate_for_request(&[4, 5, 6]);
        assert!(result.is_none());
    }

    #[test]
    fn test_chunked_prefill_parent_hash() {
        let sink = Arc::new(MockSink::new());
        let mut mgr =
            SglangKvManager::new(32, 1, KvEventPublishers::new(Some(sink.clone()), None), 0);
        let tokens = [11, 22, 33, 44, 55, 66];
        let chunk1_len = 3;
        let chunk2_len = 6;

        let alloc1 = mgr.allocate_for_request(&tokens[..chunk1_len]).unwrap();
        let new_last =
            mgr.cache_unfinished_req(&tokens[..chunk1_len], &alloc1.kv_indices, alloc1.last_node);

        let alloc2 = mgr.allocate_for_request(&tokens[..chunk2_len]).unwrap();
        mgr.free_request(new_last);

        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 2, "expected two stored events");

        let KvCacheEventData::Stored(store1) = &events[0].data else {
            panic!("expected first event to be Stored");
        };
        let KvCacheEventData::Stored(store2) = &events[1].data else {
            panic!("expected second event to be Stored");
        };

        assert!(
            store1.parent_hash.is_none(),
            "first chunk should start from the root"
        );

        let last_block_hash = store1
            .blocks
            .last()
            .expect("first chunk should store at least one block")
            .block_hash;
        assert_eq!(
            store2.parent_hash,
            Some(last_block_hash),
            "second chunk should chain from the last block of chunk 1"
        );
        assert_eq!(
            store2.blocks.len(),
            chunk2_len - chunk1_len,
            "second chunk should only emit new blocks"
        );
        assert_eq!(
            alloc2.prefix_len, chunk1_len,
            "second chunk should reuse the cached partial prefix"
        );
    }
}
