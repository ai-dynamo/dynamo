// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang KV manager — wraps [`RadixCache`] with request-level lifecycle
//! operations and KV event publishing.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::cache::radix_cache::{NodeId, RadixCache};
use crate::common::kv_cache_trace;
use crate::common::protocols::KvCacheEventSink;
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash,
};

/// Result of `allocate_for_request`.
pub struct AllocResult {
    /// Number of tokens matched from the prefix cache.
    pub prefix_len: usize,
    /// Pool page indices for the allocated input (1 per page, not per token).
    pub kv_indices: Vec<usize>,
    /// The deepest matched node in the radix tree (used for lock/unlock).
    pub last_node: NodeId,
}

pub struct SglangKvManager {
    cache: RadixCache,
    kv_event_sink: Option<Arc<dyn KvCacheEventSink>>,
    dp_rank: u32,
    next_event_id: u64,
}

impl SglangKvManager {
    pub fn new(
        total_tokens: usize,
        page_size: usize,
        kv_event_sink: Option<Arc<dyn KvCacheEventSink>>,
        dp_rank: u32,
    ) -> Self {
        Self {
            cache: RadixCache::new(total_tokens, page_size),
            kv_event_sink,
            dp_rank,
            next_event_id: 0,
        }
    }

    pub fn cache(&self) -> &RadixCache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut RadixCache {
        &mut self.cache
    }

    /// Try to allocate KV cache for a new request.
    ///
    /// 1. `match_prefix` to find cached tokens
    /// 2. Allocate new pages from the pool for uncached tokens
    /// 3. `inc_lock_ref` to protect the matched path
    /// 4. Publish BlockStored events
    ///
    /// Returns `None` if the pool doesn't have enough pages (OOM).
    pub fn allocate_for_request(&mut self, token_ids: &[u64]) -> Option<AllocResult> {
        let (prefix_len, last_node) = self.cache.match_prefix(token_ids);

        let new_tokens = token_ids.len() - prefix_len;
        let page_size = self.cache.page_size();
        let new_pages = new_tokens.div_ceil(page_size);

        // Collect reused page indices from the matched prefix path
        let prefix_page_indices = self.collect_path_indices(last_node);

        // Allocate new pages for uncached tokens
        let new_indices = self.cache.token_pool.allocate(new_pages)?;

        // Build full kv_indices (page-level): prefix reuse + new allocations
        let mut kv_indices = prefix_page_indices;
        kv_indices.extend_from_slice(&new_indices);

        // inc_lock_ref is a no-op when last_node == root (skips root in walk),
        // which is correct: a full cache miss has no path to protect.
        self.cache.inc_lock_ref(last_node);

        self.publish_stored_event(&token_ids[prefix_len..], &new_indices);
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
    /// Cache a completed request's full sequence into the radix tree.
    ///
    /// `insert` may split/extend nodes, but `last_node` remains valid for
    /// `dec_lock_ref` because: (1) split preserves the original node ID for
    /// the suffix, and (2) lock_ref is walked to root, so structural changes
    /// above last_node don't affect the unlock path.
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

    /// Allocate a single page for decode output and publish a BlockStored event.
    /// Returns the page index, or None if pool is empty.
    pub fn allocate_decode_page(&mut self) -> Option<usize> {
        let indices = self.cache.token_pool.allocate(1)?;
        let idx = indices[0];
        self.publish_stored_event(&[0], &[idx]); // token ID doesn't matter for hash
        Some(idx)
    }

    /// Free a request without caching (e.g., aborted request).
    ///
    /// Unlocks the path without inserting into the tree.
    pub fn free_request(&mut self, last_node: NodeId) {
        self.cache.dec_lock_ref(last_node);
    }

    /// Collect page indices from the matched prefix path by walking root→last_node.
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

        // Collect page indices (node.value is already page-level)
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

    fn publish_stored_event(&mut self, new_token_ids: &[u64], page_indices: &[usize]) {
        if new_token_ids.is_empty() {
            return;
        }
        let Some(ref sink) = self.kv_event_sink else {
            return;
        };

        let blocks: Vec<KvCacheStoredBlockData> = page_indices
            .iter()
            .map(|&idx| {
                // Hash only idx (not event_id) so Removed events can
                // reconstruct the same hash to correlate with Stored.
                let mut hasher = DefaultHasher::new();
                idx.hash(&mut hasher);
                let hash = hasher.finish();

                KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(hash),
                    tokens_hash: LocalBlockHash(hash),
                    mm_extra_info: None,
                }
            })
            .collect();

        let event = KvCacheEvent {
            event_id: self.next_event_id,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                blocks,
            }),
            dp_rank: self.dp_rank,
        };
        self.next_event_id += 1;

        if let Err(e) = sink.publish(event, None) {
            tracing::warn!("Failed to publish SGLang KV event: {e}");
        }
    }

    fn publish_removed_event(&mut self, evicted_indices: &[usize]) {
        let Some(ref sink) = self.kv_event_sink else {
            return;
        };

        let block_hashes: Vec<ExternalSequenceBlockHash> = evicted_indices
            .iter()
            .map(|&idx| {
                let mut hasher = DefaultHasher::new();
                idx.hash(&mut hasher);
                ExternalSequenceBlockHash(hasher.finish())
            })
            .collect();

        let event = KvCacheEvent {
            event_id: self.next_event_id,
            data: KvCacheEventData::Removed(KvCacheRemoveData { block_hashes }),
            dp_rank: self.dp_rank,
        };
        self.next_event_id += 1;

        if let Err(e) = sink.publish(event, None) {
            tracing::warn!("Failed to publish SGLang KV remove event: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

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
    }

    impl KvCacheEventSink for MockSink {
        fn publish(
            &self,
            event: KvCacheEvent,
            _block_token_ids: Option<&[Vec<u32>]>,
        ) -> anyhow::Result<()> {
            self.events.lock().unwrap().push(event);
            Ok(())
        }
    }

    #[test]
    fn test_allocate_cache_miss() {
        let mut mgr = SglangKvManager::new(100, 1, None, 0);

        let result = mgr.allocate_for_request(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(result.prefix_len, 0);
        assert_eq!(result.kv_indices.len(), 5);
        assert_eq!(mgr.cache().token_pool.available(), 95);
    }

    #[test]
    fn test_allocate_cache_hit() {
        let mut mgr = SglangKvManager::new(100, 1, None, 0);

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
        let mut mgr = SglangKvManager::new(100, 1, None, 0);

        let result = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        mgr.free_request(result.last_node);

        // Path is unlocked, tokens still allocated in pool
        assert_eq!(mgr.cache().protected_size, 0);
    }

    #[test]
    fn test_event_publishing() {
        let sink = Arc::new(MockSink::new());
        let mut mgr = SglangKvManager::new(100, 1, Some(sink.clone()), 0);

        let r = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        assert_eq!(sink.event_count(), 1); // BlockStored for 3 new pages

        mgr.cache_finished_req(&[1, 2, 3], &r.kv_indices, r.last_node);

        // Second request with full cache hit → no new events
        let r2 = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        assert_eq!(r2.prefix_len, 3);
        assert_eq!(sink.event_count(), 1); // no new event
    }

    #[test]
    fn test_allocate_oom() {
        let mut mgr = SglangKvManager::new(3, 1, None, 0);

        let _r = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        // Pool is full
        let result = mgr.allocate_for_request(&[4, 5, 6]);
        assert!(result.is_none());
    }
}
