// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang KV manager — wraps [`RadixCache`] with request-level lifecycle
//! operations and KV event publishing.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::cache::radix_cache::{NodeId, RadixCache};
use crate::common::kv_cache_trace;
use crate::common::protocols::KvCacheEventSink;
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
    kv_event_sink: Option<Arc<dyn KvCacheEventSink>>,
    dp_rank: u32,
    next_event_id: u64,
    /// Maps pool_idx → block_hash assigned during Stored events,
    /// so Removed events can use the same block_hash.
    idx_to_block_hash: HashMap<usize, ExternalSequenceBlockHash>,
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
            idx_to_block_hash: HashMap::new(),
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
        let Some(ref sink) = self.kv_event_sink else {
            return;
        };

        let mut blocks = Vec::with_capacity(indices.len());
        let mut running_hash = parent_hash.map_or(0u64, |h| h.0);
        for (i, &idx) in indices.iter().enumerate() {
            // tokens_hash: per-token content hash for router prefix matching
            let token_bytes: Vec<u8> = token_ids
                .get(i)
                .unwrap_or(&(idx as u64))
                .to_le_bytes()
                .to_vec();
            let tokens_hash = dynamo_kv_router::protocols::compute_block_hash(&token_bytes);

            // block_hash: cumulative hash (parent_hash, token_id) so it's unique
            // per position and uniform across workers with the same token sequence.
            let mut hasher = DefaultHasher::new();
            running_hash.hash(&mut hasher);
            tokens_hash.0.hash(&mut hasher);
            running_hash = hasher.finish();
            let block_hash = ExternalSequenceBlockHash(running_hash);

            self.idx_to_block_hash.insert(idx, block_hash);

            blocks.push(KvCacheStoredBlockData {
                block_hash,
                tokens_hash,
                mm_extra_info: None,
            });
        }

        let event = KvCacheEvent {
            event_id: self.next_event_id,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash,
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
            .filter_map(|&idx| self.idx_to_block_hash.remove(&idx))
            .collect();

        if block_hashes.is_empty() {
            return;
        }

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
