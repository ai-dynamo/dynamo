// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang KV manager — wraps [`RadixCache`] with request-level lifecycle
//! operations and KV event publishing.

use std::collections::HashMap;
use std::collections::VecDeque;

use crate::cache::radix_cache::{NodeId, RadixCache};
use crate::common::kv_cache_trace;
use crate::common::protocols::KvEventPublishers;
use dynamo_kv_router::protocols::{
    BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
    KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, compute_block_hash_for_seq,
    compute_next_seq_hash,
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
    /// Maps each complete block's terminal pool_idx → block_hash assigned
    /// during Stored events, so Removed events can use the same block_hash.
    idx_to_block_hash: HashMap<usize, ExternalSequenceBlockHash>,
    /// Tracks how many live pool slots currently advertise the same logical
    /// block hash so router events reflect logical block visibility, not
    /// transient slot ownership.
    block_hash_refcounts: HashMap<ExternalSequenceBlockHash, usize>,
}

pub struct DecodeTokenReservation {
    indices: VecDeque<usize>,
}

pub struct SglangDestinationReservation {
    pub(crate) prefix_len: usize,
    prefix_indices: Vec<usize>,
    last_node: NodeId,
    unpublished_indices: Vec<usize>,
    pub(crate) allocated_tokens: usize,
}

impl SglangDestinationReservation {
    pub(crate) fn transferable_prompt_tokens(&self) -> usize {
        self.unpublished_indices.len()
    }

    #[cfg(test)]
    pub(crate) fn indices(&self) -> Vec<usize> {
        self.prefix_indices
            .iter()
            .chain(&self.unpublished_indices)
            .copied()
            .collect()
    }
}

impl DecodeTokenReservation {
    pub fn take(&mut self) -> usize {
        self.indices
            .pop_front()
            .expect("reserved decode token allocation must be infallible")
    }

    pub(crate) fn len(&self) -> usize {
        self.indices.len()
    }
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

        // Router-visible KV events are complete-block only.
        self.publish_stored_event(token_ids, &kv_indices, prefix_len);

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

        self.publish_stored_event(token_ids, &kv_indices, prefix_len);
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
        first_new_token: usize,
    ) {
        self.publish_stored_event(token_ids, kv_indices, first_new_token);
        self.cache.insert(token_ids, kv_indices);
        self.release_unretained_finished_indices(token_ids, kv_indices);
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
        first_new_token: usize,
    ) -> NodeId {
        self.publish_stored_event(token_ids, kv_indices, first_new_token);
        self.cache.insert(token_ids, kv_indices);

        // Find the new deepest node after insert
        let (_, new_last_node) = self.cache.match_prefix(token_ids);

        // Acquire the extended path before releasing the old prefix so
        // destination activation never leaves valid transferred KV unprotected.
        self.cache.inc_lock_ref(new_last_node);
        self.cache.dec_lock_ref(last_node);

        new_last_node
    }

    /// Allocate a single token slot for decode output.
    /// Router-visible BlockStored events are published once a full block exists.
    pub fn allocate_decode_token(&mut self, last_idx: Option<usize>) -> Option<usize> {
        let indices = self.cache.token_pool.allocate(1)?;
        let idx = indices[0];
        self.publish_decode_token(idx, last_idx);
        Some(idx)
    }

    pub fn reserve_decode_tokens(&mut self, count: usize) -> Option<DecodeTokenReservation> {
        self.cache
            .token_pool
            .allocate(count)
            .map(|indices| DecodeTokenReservation {
                indices: indices.into(),
            })
    }

    pub(crate) fn reserve_destination(
        &mut self,
        token_ids: &[u64],
    ) -> Option<SglangDestinationReservation> {
        let (prefix_len, last_node) = self.cache.match_prefix(token_ids);
        let mut prefix_indices = self.collect_path_indices(last_node);
        prefix_indices.truncate(prefix_len);
        self.cache.inc_lock_ref(last_node);

        let allocated_tokens = if token_ids.is_empty() {
            0
        } else {
            token_ids.len().div_ceil(self.cache.page_size()) * self.cache.page_size()
        };
        let fresh_tokens = allocated_tokens.saturating_sub(prefix_len);
        let reservable = self.cache.token_pool.available() + self.cache.evictable_size;
        if fresh_tokens > reservable {
            self.cache.dec_lock_ref(last_node);
            return None;
        }
        let available = self.cache.token_pool.available();
        if fresh_tokens > available {
            self.evict(fresh_tokens - available);
        }
        let Some(unpublished_indices) = self.cache.token_pool.allocate(fresh_tokens) else {
            self.cache.dec_lock_ref(last_node);
            return None;
        };
        self.log_trace("reserve_destination", fresh_tokens);
        Some(SglangDestinationReservation {
            prefix_len,
            prefix_indices,
            last_node,
            unpublished_indices,
            allocated_tokens,
        })
    }

    pub(crate) fn activate_destination(
        &mut self,
        reservation: SglangDestinationReservation,
        token_ids: &[u64],
    ) -> AllocResult {
        let SglangDestinationReservation {
            prefix_len,
            mut prefix_indices,
            last_node,
            mut unpublished_indices,
            allocated_tokens: _,
        } = reservation;
        let missing_tokens = token_ids.len().saturating_sub(prefix_len);
        let surplus = unpublished_indices.split_off(missing_tokens);
        self.release_unpublished_indices(surplus);
        prefix_indices.append(&mut unpublished_indices);
        let new_last_node =
            self.cache_unfinished_req(token_ids, &prefix_indices, last_node, prefix_len);
        self.log_trace("activate_destination", missing_tokens);
        AllocResult {
            prefix_len,
            kv_indices: prefix_indices,
            last_node: new_last_node,
        }
    }

    pub(crate) fn cancel_destination(&mut self, reservation: SglangDestinationReservation) {
        self.cache.dec_lock_ref(reservation.last_node);
        self.release_unpublished_indices(reservation.unpublished_indices);
    }

    pub fn publish_decode_token(&mut self, idx: usize, last_idx: Option<usize>) {
        let _ = (idx, last_idx);
        self.log_trace("allocation", 1);
    }

    pub fn release_decode_reservation(&mut self, reservation: DecodeTokenReservation) {
        let indices = reservation.indices.into_iter().collect::<Vec<_>>();
        self.release_unpublished_indices(indices);
    }

    fn release_unpublished_indices(&mut self, indices: Vec<usize>) {
        if indices.is_empty() {
            return;
        }
        self.cache.token_pool.free(&indices);
        self.log_trace("release_unpublished", indices.len());
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

    fn release_unretained_finished_indices(&mut self, token_ids: &[u64], kv_indices: &[usize]) {
        let block_size = self.cache.page_size();
        let complete_len = token_ids.len().min(kv_indices.len()) / block_size * block_size;
        if complete_len == 0 {
            return;
        }

        let (matched_len, last_node) = self.cache.match_prefix(&token_ids[..complete_len]);
        debug_assert_eq!(
            matched_len, complete_len,
            "completed SGLang sequence should be fully cached after insert"
        );
        if matched_len < complete_len {
            return;
        }

        let canonical_indices = self.collect_path_indices(last_node);
        debug_assert!(
            canonical_indices.len() >= complete_len,
            "cached SGLang sequence path should carry complete KV indices"
        );
        if canonical_indices.len() < complete_len {
            return;
        }

        let mut unretained_indices = Vec::new();
        for block_start in (0..complete_len).step_by(block_size) {
            let block_end = block_start + block_size;
            if canonical_indices[block_end - 1] != kv_indices[block_end - 1] {
                unretained_indices.extend_from_slice(&kv_indices[block_start..block_end]);
            }
        }

        self.free_indices(&unretained_indices);
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
        first_new_token: usize,
    ) -> usize {
        if self.kv_event_publishers.is_empty() {
            return 0;
        }

        let block_size = self.cache.page_size();
        let complete_len = token_ids.len().min(indices.len()) / block_size * block_size;
        if complete_len == 0 || first_new_token >= complete_len {
            return 0;
        }

        let mut computed_blocks = Vec::new();
        let first_block_start = first_new_token / block_size * block_size;
        let local_hashes = self.local_hashes_for_range(&token_ids[first_block_start..complete_len]);

        for (block_idx, tokens_hash) in local_hashes.iter().copied().enumerate() {
            let block_start = first_block_start + block_idx * block_size;
            let block_end = block_start + block_size;
            let representative_idx = indices[block_end - 1];
            if self.idx_to_block_hash.contains_key(&representative_idx) {
                continue;
            }

            let parent_hash = if block_start == 0 {
                None
            } else {
                self.idx_to_block_hash
                    .get(&indices[block_start - 1])
                    .copied()
            };
            let block_hash = match parent_hash {
                Some(parent_hash) => {
                    ExternalSequenceBlockHash(compute_next_seq_hash(parent_hash.0, tokens_hash))
                }
                None => ExternalSequenceBlockHash(tokens_hash.0),
            };

            self.idx_to_block_hash
                .insert(representative_idx, block_hash);
            let refcount = self.block_hash_refcounts.entry(block_hash).or_default();
            *refcount += 1;
            computed_blocks.push((
                parent_hash,
                KvCacheStoredBlockData {
                    block_hash,
                    tokens_hash,
                    mm_extra_info: None,
                },
                *refcount,
            ));
        }

        let hashed_blocks = local_hashes.len();

        let first_new = computed_blocks
            .iter()
            .position(|(_, _, refcount)| *refcount == 1);
        let Some(first_new) = first_new else {
            return hashed_blocks;
        };

        let parent_hash = computed_blocks[first_new].0;
        let blocks = computed_blocks
            .into_iter()
            .skip(first_new)
            .map(|(_, block, _)| block)
            .collect();

        let event = KvCacheEvent {
            event_id: self.next_event_id,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash,
                start_position: None,
                blocks,
            }),
            dp_rank: self.dp_rank,
        };
        self.next_event_id += 1;

        if let Err(e) = self.kv_event_publishers.publish(event, None) {
            tracing::warn!("Failed to publish SGLang KV event: {e}");
        }

        hashed_blocks
    }

    fn local_hashes_for_range(&self, token_ids: &[u64]) -> Vec<LocalBlockHash> {
        let tokens = token_ids
            .iter()
            .map(|&token| {
                u32::try_from(token).unwrap_or_else(|_| {
                    panic!("local_hashes_for_range: token {token} exceeds router u32 token domain")
                })
            })
            .collect::<Vec<_>>();
        compute_block_hash_for_seq(
            &tokens,
            self.cache.page_size() as u32,
            BlockHashOptions::default(),
        )
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
    use crate::scheduler::capture_router_event_sink;
    use crate::scheduler::test_utils::{RouterIndexerHarness, stored_hashes};
    use dynamo_kv_router::protocols::{RouterEvent, WorkerId, compute_seq_hash_for_block};

    const ROUTER_TEST_WORKER_ID: WorkerId = 31;

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

    fn stored_event_count(events: &[RouterEvent]) -> usize {
        events
            .iter()
            .filter(|event| matches!(event.event.data, KvCacheEventData::Stored(_)))
            .count()
    }

    fn removed_event_count(events: &[RouterEvent]) -> usize {
        events
            .iter()
            .filter(|event| matches!(event.event.data, KvCacheEventData::Removed(_)))
            .count()
    }

    fn removed_block_count(events: &[RouterEvent]) -> usize {
        events
            .iter()
            .filter_map(|event| match &event.event.data {
                KvCacheEventData::Removed(remove) => Some(remove.block_hashes.len()),
                _ => None,
            })
            .sum()
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
        mgr.cache_finished_req(&[1, 2, 3, 4, 5], &r1.kv_indices, r1.last_node, 0);

        // Second request with shared prefix
        let r2 = mgr.allocate_for_request(&[1, 2, 3, 4, 5, 6, 7]).unwrap();
        assert_eq!(r2.prefix_len, 5);
        assert_eq!(r2.kv_indices.len(), 7); // 5 reused + 2 new pages
        assert_eq!(mgr.cache().token_pool.available(), 93); // 100 - 5 - 2
    }

    #[test]
    fn destination_transfer_footprint_uses_missing_physical_pages() {
        let mut mgr = SglangKvManager::new(64, 4, KvEventPublishers::default(), 0);
        let prompt = (0..10).collect::<Vec<_>>();

        let cold = mgr
            .reserve_destination(&prompt)
            .expect("cold destination reservation should fit");
        assert_eq!(cold.transferable_prompt_tokens(), 12);
        mgr.cancel_destination(cold);

        let prefix_tokens = &prompt[..4];
        let prefix = mgr
            .allocate_for_request(prefix_tokens)
            .expect("prefix allocation should fit");
        mgr.cache_finished_req(
            prefix_tokens,
            &prefix.kv_indices,
            prefix.last_node,
            prefix.prefix_len,
        );
        let partial = mgr
            .reserve_destination(&prompt)
            .expect("partially cached destination reservation should fit");
        assert_eq!(partial.transferable_prompt_tokens(), 8);
        mgr.cancel_destination(partial);

        let aligned_tokens = (20..28).collect::<Vec<_>>();
        let aligned = mgr
            .allocate_for_request(&aligned_tokens)
            .expect("aligned prompt allocation should fit");
        mgr.cache_finished_req(
            &aligned_tokens,
            &aligned.kv_indices,
            aligned.last_node,
            aligned.prefix_len,
        );
        let full_hit = mgr
            .reserve_destination(&aligned_tokens)
            .expect("fully cached destination reservation should fit");
        assert_eq!(full_hit.transferable_prompt_tokens(), 0);
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

        mgr.cache_finished_req(&[1, 2, 3], &r.kv_indices, r.last_node, 0);

        // Second request with full cache hit → no new events
        let r2 = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        assert_eq!(r2.prefix_len, 3);
        assert_eq!(sink.event_count(), 1); // no new event
    }

    #[test]
    fn test_event_publishing_uses_router_block_hashes() {
        let sink = Arc::new(MockSink::new());
        let mut mgr =
            SglangKvManager::new(100, 4, KvEventPublishers::new(Some(sink.clone()), None), 0);

        let r = mgr.allocate_for_request(&[1, 2, 3, 4, 5, 6]).unwrap();
        mgr.cache_finished_req(&[1, 2, 3, 4, 5, 6], &r.kv_indices, r.last_node, 0);

        let events = sink.clone_events();
        assert_eq!(events.len(), 1);
        let KvCacheEventData::Stored(store) = &events[0].data else {
            panic!("expected stored event");
        };
        assert_eq!(store.blocks.len(), 1);

        let expected_local =
            compute_block_hash_for_seq(&[1, 2, 3, 4], 4, BlockHashOptions::default());
        let expected_sequence = compute_seq_hash_for_block(&expected_local);
        assert_eq!(store.blocks[0].tokens_hash, expected_local[0]);
        assert_eq!(
            store.blocks[0].block_hash,
            ExternalSequenceBlockHash(expected_sequence[0])
        );
    }

    #[test]
    fn test_cache_materialization_processes_only_newly_completed_blocks() {
        let sink = Arc::new(MockSink::new());
        let mut mgr = SglangKvManager::new(100, 2, KvEventPublishers::default(), 0);
        let out_of_domain_token = u32::MAX as u64 + 1;
        let tokens = [out_of_domain_token, 2, 3, 4, 5, 6];

        let alloc = mgr.allocate_for_request(&tokens[..2]).unwrap();
        let first_last_node =
            mgr.cache_unfinished_req(&tokens[..2], &alloc.kv_indices, alloc.last_node, 0);

        let mut kv_indices = alloc.kv_indices;
        kv_indices.extend_from_slice(&mgr.cache_mut().token_pool.allocate(4).unwrap());
        mgr.kv_event_publishers = KvEventPublishers::new(Some(sink.clone()), None);

        let last_after_first_cache =
            mgr.cache_unfinished_req(&tokens[..4], &kv_indices[..4], first_last_node, 2);
        let events = sink.clone_events();
        assert_eq!(events.len(), 1);
        let KvCacheEventData::Stored(first_store) = &events[0].data else {
            panic!("expected first cache event to be Stored");
        };
        assert_eq!(
            first_store.blocks.len(),
            1,
            "first unfinished cache should store only the newly completed block"
        );

        mgr.cache_finished_req(&tokens, &kv_indices, last_after_first_cache, 4);
        let events = sink.clone_events();
        assert_eq!(events.len(), 2);
        let KvCacheEventData::Stored(final_store) = &events[1].data else {
            panic!("expected final cache event to be Stored");
        };
        assert_eq!(
            final_store.blocks.len(),
            1,
            "finished cache should store only the newly completed block"
        );
    }

    #[test]
    #[should_panic(
        expected = "local_hashes_for_range: token 4294967296 exceeds router u32 token domain"
    )]
    fn test_local_hashes_reject_out_of_domain_tokens() {
        let mgr = SglangKvManager::new(100, 1, KvEventPublishers::default(), 0);
        let _ = mgr.local_hashes_for_range(&[u32::MAX as u64 + 1]);
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

    #[tokio::test]
    async fn test_duplicate_completion_releases_unretained_indices_and_removes_on_eviction() {
        let (buffer, sink) = capture_router_event_sink(ROUTER_TEST_WORKER_ID);
        let harness = RouterIndexerHarness::new(1, ROUTER_TEST_WORKER_ID);
        let mut mgr = SglangKvManager::new(100, 1, KvEventPublishers::new(Some(sink), None), 0);
        let tokens = [1, 2, 3];

        let req1 = mgr.allocate_for_request(&tokens).unwrap();
        let req2 = mgr.allocate_for_request(&tokens).unwrap();
        assert_eq!(
            mgr.cache().token_pool.available(),
            94,
            "both identical requests should allocate before either is cached"
        );

        let allocation_events = buffer.drain();
        assert_eq!(
            stored_event_count(&allocation_events),
            1,
            "duplicate allocation should emit only one logical Stored event"
        );
        let query_hashes = stored_hashes(&allocation_events);
        assert_eq!(query_hashes.len(), tokens.len());
        harness.apply_events(allocation_events).await;

        mgr.cache_finished_req(&tokens, &req1.kv_indices, req1.last_node, 0);
        let req1_completion_events = buffer.drain();
        assert_eq!(
            stored_event_count(&req1_completion_events),
            0,
            "first completion should not re-emit Stored blocks"
        );
        assert_eq!(
            removed_event_count(&req1_completion_events),
            0,
            "canonical completion should not emit Removed blocks"
        );
        assert_eq!(
            mgr.cache().token_pool.available(),
            94,
            "canonical completion should retain the first request's slots"
        );

        mgr.cache_finished_req(&tokens, &req2.kv_indices, req2.last_node, 0);
        let req2_completion_events = buffer.drain();
        assert_eq!(
            stored_event_count(&req2_completion_events),
            0,
            "duplicate completion should not re-emit Stored blocks"
        );
        assert_eq!(
            removed_event_count(&req2_completion_events),
            0,
            "duplicate completion should only decrement duplicate refcounts"
        );
        assert_eq!(
            mgr.cache().token_pool.available(),
            97,
            "duplicate completion should return unretained request slots"
        );
        assert_eq!(harness.overlap_for_hashes(query_hashes.clone()).await, 3);

        mgr.evict(tokens.len());
        let eviction_events = buffer.drain();
        assert_eq!(
            removed_event_count(&eviction_events),
            1,
            "evicting the canonical sequence should emit one logical Removed event"
        );
        assert_eq!(
            removed_block_count(&eviction_events),
            tokens.len(),
            "Removed event should cover every cached block"
        );
        harness.apply_events(eviction_events).await;
        assert_eq!(harness.overlap_for_hashes(query_hashes).await, 0);
        harness.shutdown();
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
        let new_last = mgr.cache_unfinished_req(
            &tokens[..chunk1_len],
            &alloc1.kv_indices,
            alloc1.last_node,
            0,
        );

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
