// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang KV manager — wraps [`RadixCache`] with request-level lifecycle
//! operations and KV event publishing.

use crate::cache::radix_cache::{KvPageId, NodeId, RadixCache};
use crate::common::kv_cache_trace;
use crate::common::protocols::KvEventPublishers;
use dynamo_kv_router::protocols::{
    BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
    KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, compute_block_hash_for_seq,
    compute_next_seq_hash,
};
use rustc_hash::FxHashMap;

/// Move-only ownership of an active request's KV slots and protected radix path.
#[derive(Debug, Default)]
#[must_use = "an active KV lease must be finished, aborted, or retracted"]
pub(crate) struct ActiveKvLease {
    kv_indices: Vec<usize>,
    cached_tokens: usize,
    last_node: Option<NodeId>,
}

impl ActiveKvLease {
    #[cfg(test)]
    pub(crate) fn indices(&self) -> &[usize] {
        &self.kv_indices
    }

    pub(crate) fn len(&self) -> usize {
        self.kv_indices.len()
    }

    pub(crate) fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    pub(crate) fn last_index(&self) -> Option<usize> {
        self.kv_indices.last().copied()
    }

    pub(crate) fn is_active(&self) -> bool {
        self.last_node.is_some()
    }

    fn last_node(&self) -> NodeId {
        self.last_node
            .expect("active KV lease must retain a radix path")
    }

    #[cfg(test)]
    pub(crate) fn from_parts(
        kv_indices: Vec<usize>,
        cached_tokens: usize,
        last_node: NodeId,
    ) -> Self {
        Self {
            kv_indices,
            cached_tokens,
            last_node: Some(last_node),
        }
    }
}

/// Result of `allocate_for_request`.
pub(crate) struct AllocResult {
    /// Number of tokens matched from the prefix cache.
    pub(crate) prefix_len: usize,
    pub(crate) lease: ActiveKvLease,
}

pub struct SglangKvManager {
    cache: RadixCache,
    kv_event_publishers: KvEventPublishers,
    dp_rank: u32,
    next_event_id: u64,
    /// Maps each complete block's terminal pool_idx → block_hash assigned
    /// during Stored events, so Removed events can use the same block_hash.
    idx_to_block_hash: FxHashMap<usize, ExternalSequenceBlockHash>,
    /// Tracks how many live pool slots currently advertise the same logical
    /// block hash so router events reflect logical block visibility, not
    /// transient slot ownership.
    block_hash_refcounts: FxHashMap<ExternalSequenceBlockHash, usize>,
}

pub struct DecodeTokenReservation {
    pages: Vec<KvPageId>,
    next: usize,
    page_size: usize,
}

pub struct SglangDestinationReservation {
    pub(crate) prefix_len: usize,
    prefix_indices: Vec<usize>,
    last_node: NodeId,
    unpublished_pages: Vec<KvPageId>,
    page_size: usize,
    missing_tokens: usize,
    pub(crate) allocated_tokens: usize,
}

impl SglangDestinationReservation {
    pub(crate) fn transferable_prompt_tokens(&self) -> usize {
        self.unpublished_pages.len() * self.page_size
    }

    #[cfg(test)]
    pub(crate) fn indices(&self) -> Vec<usize> {
        self.prefix_indices
            .iter()
            .copied()
            .chain(
                self.unpublished_pages
                    .iter()
                    .flat_map(|page| {
                        let start = page.first_token_index(self.page_size);
                        start..start + self.page_size
                    })
                    .take(self.missing_tokens),
            )
            .collect()
    }
}

impl DecodeTokenReservation {
    fn take(&mut self, last_idx: Option<usize>) -> usize {
        if let Some(last_idx) = last_idx
            && (last_idx + 1) % self.page_size != 0
        {
            return last_idx + 1;
        }
        let page = *self
            .pages
            .get(self.next)
            .expect("reserved decode page allocation must be infallible");
        self.next += 1;
        page.first_token_index(self.page_size)
    }

    pub(crate) fn len(&self) -> usize {
        self.pages.len() - self.next
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
            idx_to_block_hash: FxHashMap::default(),
            block_hash_refcounts: FxHashMap::default(),
        }
    }

    pub fn cache(&self) -> &RadixCache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut RadixCache {
        &mut self.cache
    }

    /// Match and protect a reusable prefix, evict other cached pages if needed,
    /// then allocate KV pages for a new request.
    ///
    /// Returns `None` if protected and free capacity cannot satisfy the request.
    pub(crate) fn allocate_for_request(&mut self, token_ids: &[u32]) -> Option<AllocResult> {
        let (prefix_len, last_node) = self.cache.match_prefix(token_ids);
        let new_tokens = token_ids.len() - prefix_len;
        let required_tokens = new_tokens.div_ceil(self.cache.page_size()) * self.cache.page_size();

        // Protect the matched path before making room. Otherwise an LRU
        // eviction can remove the prefix used to size this allocation, and a
        // second match would require more pages than were freed.
        self.cache.inc_lock_ref(last_node);
        let reservable = self.cache.available_tokens() + self.cache.evictable_size;
        if required_tokens > reservable {
            self.cache.dec_lock_ref(last_node);
            return None;
        }
        let available = self.cache.available_tokens();
        if required_tokens > available {
            self.evict(required_tokens - available);
        }
        let prefix_indices = self.collect_path_indices(last_node);

        let mut kv_indices = prefix_indices;
        let available_before = self.cache.available_tokens();
        if !self
            .cache
            .page_pool
            .allocate_indices_into(new_tokens, &mut kv_indices)
        {
            self.cache.dec_lock_ref(last_node);
            return None;
        }
        let allocated_tokens = available_before - self.cache.available_tokens();

        // Router-visible KV events are complete-block only.
        self.publish_stored_event(token_ids, &kv_indices, prefix_len);

        self.log_trace("allocation", allocated_tokens);

        Some(AllocResult {
            prefix_len,
            lease: ActiveKvLease {
                kv_indices,
                cached_tokens: prefix_len,
                last_node: Some(last_node),
            },
        })
    }

    /// Continue an in-flight request from an already materialized prefix.
    ///
    /// This is used by chunked-prefill continuation where the request still
    /// owns token slots for a prefix that may extend past the radix-tree's
    /// page-aligned cached prefix.
    pub(crate) fn extend_allocation(
        &mut self,
        token_ids: &[u32],
        lease: &mut ActiveKvLease,
    ) -> bool {
        let prefix_len = lease.kv_indices.len();
        assert!(
            lease.is_active() && prefix_len <= token_ids.len(),
            "invalid SGLang KV lease extension: active={}, owned_tokens={prefix_len}, target_tokens={}",
            lease.is_active(),
            token_ids.len()
        );
        let new_tokens = token_ids.len() - prefix_len;
        let available_before = self.cache.available_tokens();
        if !self
            .cache
            .page_pool
            .allocate_indices_into(new_tokens, &mut lease.kv_indices)
        {
            return false;
        }
        let allocated_tokens = available_before - self.cache.available_tokens();

        self.publish_stored_event(token_ids, &lease.kv_indices, prefix_len);
        self.log_trace("allocation", allocated_tokens);
        true
    }

    pub(crate) fn extend_cached_prefix(&mut self, token_ids: &[u32], lease: &mut ActiveKvLease) {
        let complete_len = token_ids.len() / self.cache.page_size() * self.cache.page_size();
        if complete_len <= lease.cached_tokens {
            return;
        }
        assert!(
            lease.is_active() && complete_len <= lease.len(),
            "invalid SGLang KV lease cache extension: active={}, cached_tokens={}, complete_tokens={complete_len}, owned_tokens={}",
            lease.is_active(),
            lease.cached_tokens,
            lease.len()
        );
        let last_node = lease.last_node();
        let new_last_node = self.cache_unfinished_req(
            token_ids,
            &mut lease.kv_indices[..complete_len],
            last_node,
            lease.cached_tokens,
        );
        lease.last_node = Some(new_last_node);
        lease.cached_tokens = complete_len;
    }

    pub(crate) fn extend_decode(
        &mut self,
        lease: &mut ActiveKvLease,
        reservation: &mut DecodeTokenReservation,
    ) {
        debug_assert!(lease.is_active());
        let new_idx = reservation.take(lease.last_index());
        lease.kv_indices.push(new_idx);
    }

    pub(crate) fn finish(&mut self, token_ids: &[u32], mut lease: ActiveKvLease) {
        let Some(last_node) = lease.last_node.take() else {
            debug_assert!(lease.kv_indices.is_empty());
            debug_assert_eq!(lease.cached_tokens, 0);
            return;
        };
        let complete_len =
            token_ids.len().min(lease.len()) / self.cache.page_size() * self.cache.page_size();
        assert!(
            lease.cached_tokens <= complete_len,
            "invalid SGLang KV lease finish: cached_tokens={}, complete_tokens={complete_len}, owned_tokens={}",
            lease.cached_tokens,
            lease.len()
        );
        self.free_indices(&lease.kv_indices[complete_len..]);
        lease.kv_indices.truncate(complete_len);

        if complete_len == 0 {
            self.cache.dec_lock_ref(last_node);
            return;
        }
        self.cache_finished_req(
            &token_ids[..complete_len],
            &lease.kv_indices,
            last_node,
            lease.cached_tokens,
        );
    }

    pub(crate) fn abort(&mut self, lease: ActiveKvLease) -> bool {
        self.release_active_lease(lease)
    }

    pub(crate) fn retract(&mut self, lease: ActiveKvLease) -> bool {
        self.release_active_lease(lease)
    }

    /// Cache a completed request's full sequence into the radix tree.
    ///
    /// Inserts the full token sequence so future requests can reuse it,
    /// then unlocks the path.
    fn cache_finished_req(
        &mut self,
        token_ids: &[u32],
        kv_indices: &[usize],
        last_node: NodeId,
        first_new_token: usize,
    ) {
        self.publish_stored_event(token_ids, kv_indices, first_new_token);
        let new_last_node =
            self.cache
                .insert_from_node(last_node, first_new_token, token_ids, kv_indices);
        let complete_len =
            token_ids.len().min(kv_indices.len()) / self.cache.page_size() * self.cache.page_size();
        self.release_unretained_finished_indices(
            kv_indices,
            new_last_node,
            first_new_token,
            complete_len,
        );
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
    fn cache_unfinished_req(
        &mut self,
        token_ids: &[u32],
        kv_indices: &mut [usize],
        last_node: NodeId,
        first_new_token: usize,
    ) -> NodeId {
        let block_size = self.cache.page_size();
        let complete_len = token_ids.len() / block_size * block_size;
        assert!(
            first_new_token.is_multiple_of(block_size)
                && first_new_token <= complete_len
                && complete_len <= kv_indices.len(),
            "invalid SGLang canonicalization range: first_new_token={first_new_token}, complete_len={complete_len}, kv_indices={}",
            kv_indices.len()
        );

        self.publish_stored_event(token_ids, kv_indices, first_new_token);
        let new_last_node =
            self.cache
                .insert_from_node(last_node, first_new_token, token_ids, kv_indices);

        // An interleaved insert can retain different physical pages for the same prefix.
        // Move the active request to canonical pages before releasing its duplicates.
        // Acquire the extended path before releasing the old prefix so
        // destination activation never leaves valid transferred KV unprotected.
        if new_last_node != last_node {
            self.cache.inc_lock_ref(new_last_node);
        }
        self.canonicalize_unfinished_indices(
            kv_indices,
            new_last_node,
            first_new_token,
            complete_len,
        );
        if new_last_node != last_node {
            self.cache.dec_lock_ref(last_node);
        }

        new_last_node
    }

    /// Allocate a decode token, consuming a new page only at a page boundary.
    /// Router-visible BlockStored events are published once a full block exists.
    pub fn allocate_decode_token(&mut self, last_idx: Option<usize>) -> Option<usize> {
        let mut reservation = self.reserve_decode_pages(usize::from(
            last_idx.is_none_or(|idx| (idx + 1) % self.cache.page_size() == 0),
        ))?;
        let idx = reservation.take(last_idx);
        Some(idx)
    }

    pub fn reserve_decode_pages(&mut self, count: usize) -> Option<DecodeTokenReservation> {
        let pages = self.cache.page_pool.allocate_pages(count)?;
        if !pages.is_empty() {
            self.log_trace("allocation", pages.len() * self.cache.page_size());
        }
        Some(DecodeTokenReservation {
            pages,
            next: 0,
            page_size: self.cache.page_size(),
        })
    }

    pub(crate) fn reserve_destination(
        &mut self,
        token_ids: &[u32],
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
        let fresh_pages = fresh_tokens / self.cache.page_size();
        let reservable = self.cache.available_tokens() + self.cache.evictable_size;
        if fresh_tokens > reservable {
            self.cache.dec_lock_ref(last_node);
            return None;
        }
        let available = self.cache.available_tokens();
        if fresh_tokens > available {
            self.evict(fresh_tokens - available);
        }
        let Some(unpublished_pages) = self.cache.page_pool.allocate_pages(fresh_pages) else {
            self.cache.dec_lock_ref(last_node);
            return None;
        };
        self.log_trace("reserve_destination", fresh_tokens);
        Some(SglangDestinationReservation {
            prefix_len,
            prefix_indices,
            last_node,
            unpublished_pages,
            page_size: self.cache.page_size(),
            missing_tokens: token_ids.len().saturating_sub(prefix_len),
            allocated_tokens,
        })
    }

    pub(crate) fn activate_destination(
        &mut self,
        reservation: SglangDestinationReservation,
        token_ids: &[u32],
    ) -> AllocResult {
        let SglangDestinationReservation {
            prefix_len,
            mut prefix_indices,
            last_node,
            unpublished_pages,
            page_size: _,
            missing_tokens,
            allocated_tokens: _,
        } = reservation;
        let mut unpublished_indices = self
            .cache
            .page_pool
            .expand_pages(&unpublished_pages, missing_tokens);
        prefix_indices.append(&mut unpublished_indices);
        let new_last_node =
            self.cache_unfinished_req(token_ids, &mut prefix_indices, last_node, prefix_len);
        self.log_trace("activate_destination", missing_tokens);
        AllocResult {
            prefix_len,
            lease: ActiveKvLease {
                kv_indices: prefix_indices,
                cached_tokens: token_ids.len() / self.cache.page_size() * self.cache.page_size(),
                last_node: Some(new_last_node),
            },
        }
    }

    pub(crate) fn cancel_destination(&mut self, reservation: SglangDestinationReservation) {
        self.cache.dec_lock_ref(reservation.last_node);
        self.release_unpublished_pages(reservation.unpublished_pages);
    }

    pub fn release_decode_reservation(&mut self, reservation: DecodeTokenReservation) {
        let pages = &reservation.pages[reservation.next..];
        if pages.is_empty() {
            return;
        }
        self.cache.page_pool.free_pages(pages);
        self.log_trace("release_unpublished", pages.len() * self.cache.page_size());
    }

    fn release_unpublished_pages(&mut self, pages: Vec<KvPageId>) {
        if pages.is_empty() {
            return;
        }
        self.cache.page_pool.free_pages(&pages);
        self.log_trace("release_unpublished", pages.len() * self.cache.page_size());
    }

    #[cfg(test)]
    fn free_request(&mut self, last_node: NodeId) {
        self.cache.dec_lock_ref(last_node);
    }

    /// Return request-owned token slots to the free pool and publish matching
    /// removal events for any slots that were previously advertised to the router.
    fn free_indices(&mut self, indices: &[usize]) {
        if indices.is_empty() {
            return;
        }

        let pages = self.cache.page_pool.free_indices(indices);
        self.publish_removed_pages(&pages);
        self.log_trace("free", pages.len() * self.cache.page_size());
    }

    fn release_active_lease(&mut self, mut lease: ActiveKvLease) -> bool {
        let Some(last_node) = lease.last_node.take() else {
            debug_assert!(lease.kv_indices.is_empty());
            debug_assert_eq!(lease.cached_tokens, 0);
            return false;
        };
        assert!(
            lease.cached_tokens <= lease.len(),
            "invalid SGLang KV lease release: cached_tokens={}, owned_tokens={}",
            lease.cached_tokens,
            lease.len()
        );
        let owned_suffix = &lease.kv_indices[lease.cached_tokens..];
        let capacity_improved = !owned_suffix.is_empty() || last_node != self.cache.root();
        self.free_indices(owned_suffix);
        self.cache.dec_lock_ref(last_node);
        capacity_improved
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

        // Expand cached page IDs only for the bounded active request.
        let mut indices = Vec::new();
        for node_id in path {
            let node = self.cache.node(node_id);
            indices.extend(
                self.cache
                    .page_pool
                    .expand_pages(&node.value, node.value.len() * self.cache.page_size()),
            );
        }
        indices
    }

    fn release_unretained_finished_indices(
        &mut self,
        kv_indices: &[usize],
        last_node: NodeId,
        first_new_token: usize,
        complete_len: usize,
    ) {
        let block_size = self.cache.page_size();
        if complete_len == 0 {
            return;
        }

        let mut unretained_pages = Vec::new();
        let mut current = last_node;
        let first_new_page = first_new_token / block_size;
        let mut path_end = complete_len / block_size;

        while path_end > first_new_page {
            debug_assert_ne!(current, self.cache.root());
            if current == self.cache.root() {
                tracing::error!(
                    path_end,
                    first_new_token,
                    complete_len,
                    "SGLang radix path ended before finished-request reconciliation"
                );
                break;
            }

            let node = self.cache.node(current);
            let node_len = node.value.len();
            debug_assert!(node_len <= path_end);
            if node_len > path_end {
                tracing::error!(
                    node_len,
                    path_end,
                    complete_len,
                    "SGLang radix node exceeds finished materialized prefix"
                );
                break;
            }
            let path_start = path_end - node_len;
            let reconcile_start = path_start.max(first_new_page);

            for page_idx in reconcile_start..path_end {
                let token_start = page_idx * block_size;
                let incoming_page = KvPageId::from_token_index(kv_indices[token_start], block_size);
                let canonical_page = node.value[page_idx - path_start];
                if incoming_page != canonical_page {
                    unretained_pages.push(incoming_page);
                }
            }

            path_end = path_start;
            current = node.parent.unwrap_or(self.cache.root());
        }

        self.free_pages(&unretained_pages);
    }

    fn canonicalize_unfinished_indices(
        &mut self,
        kv_indices: &mut [usize],
        last_node: NodeId,
        first_new_token: usize,
        complete_len: usize,
    ) {
        let block_size = self.cache.page_size();
        debug_assert_eq!(complete_len % block_size, 0);
        debug_assert_eq!(first_new_token % block_size, 0);
        debug_assert!(complete_len <= kv_indices.len());
        debug_assert!(first_new_token <= complete_len);

        assert!(
            first_new_token.is_multiple_of(block_size)
                && complete_len.is_multiple_of(block_size)
                && complete_len <= kv_indices.len()
                && first_new_token <= complete_len
                && self.radix_path_covers(last_node, first_new_token, complete_len),
            "invalid SGLang canonicalization range or radix path: first_new_token={first_new_token}, complete_len={complete_len}, kv_indices={}",
            kv_indices.len()
        );

        let mut unretained_pages = Vec::new();
        let mut current = last_node;
        let first_new_page = first_new_token / block_size;
        let mut path_end = complete_len / block_size;

        while path_end > first_new_page {
            let node = self.cache.node(current);
            let node_len = node.value.len();
            let path_start = path_end - node_len;
            let reconcile_start = path_start.max(first_new_page);

            for page_idx in reconcile_start..path_end {
                let token_start = page_idx * block_size;
                let token_end = token_start + block_size;
                let incoming_page = KvPageId::from_token_index(kv_indices[token_start], block_size);
                let canonical_page = node.value[page_idx - path_start];
                if incoming_page != canonical_page {
                    unretained_pages.push(incoming_page);
                    let canonical = self
                        .cache
                        .page_pool
                        .expand_pages(&[canonical_page], block_size);
                    kv_indices[token_start..token_end].copy_from_slice(&canonical);
                }
            }

            path_end = path_start;
            current = node.parent.unwrap_or_else(|| self.cache.root());
        }

        self.free_pages(&unretained_pages);
    }

    fn radix_path_covers(
        &self,
        mut current: NodeId,
        first_new_token: usize,
        path_end: usize,
    ) -> bool {
        let page_size = self.cache.page_size();
        let first_new_page = first_new_token / page_size;
        let mut path_end = path_end / page_size;
        while path_end > first_new_page {
            if current == self.cache.root() {
                return false;
            }
            let node = self.cache.node(current);
            if node.value.len() > path_end {
                return false;
            }
            path_end -= node.value.len();
            current = node.parent.unwrap_or_else(|| self.cache.root());
        }
        true
    }

    /// Evict tokens from the cache, publish BlockRemoved events, and log a trace.
    pub fn evict(&mut self, num_tokens: usize) {
        let (evicted, evicted_pages) = self.cache.evict(num_tokens);
        if !evicted_pages.is_empty() {
            self.publish_removed_pages(&evicted_pages);
        }
        self.log_trace("eviction", evicted);
    }

    fn free_pages(&mut self, pages: &[KvPageId]) {
        if pages.is_empty() {
            return;
        }
        self.cache.page_pool.free_pages(pages);
        self.publish_removed_pages(pages);
        self.log_trace("free", pages.len() * self.cache.page_size());
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
        token_ids: &[u32],
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

        let first_block_start = first_new_token / block_size * block_size;
        let Some(first_unpublished_block) = (first_block_start..complete_len)
            .step_by(block_size)
            .find(|&block_start| {
                let representative_idx = indices[block_start + block_size - 1];
                !self.idx_to_block_hash.contains_key(&representative_idx)
            })
        else {
            return 0;
        };

        let mut computed_blocks = Vec::new();
        let local_hashes =
            self.local_hashes_for_range(&token_ids[first_unpublished_block..complete_len]);

        for (block_idx, tokens_hash) in local_hashes.iter().copied().enumerate() {
            let block_start = first_unpublished_block + block_idx * block_size;
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

    fn local_hashes_for_range(&self, token_ids: &[u32]) -> Vec<LocalBlockHash> {
        compute_block_hash_for_seq(
            token_ids,
            self.cache.page_size() as u32,
            BlockHashOptions::default(),
        )
    }

    fn publish_removed_pages(&mut self, evicted_pages: &[KvPageId]) {
        if self.kv_event_publishers.is_empty() {
            return;
        }

        let mut block_hashes = Vec::new();
        for &page in evicted_pages {
            let idx = page.terminal_token_index(self.cache.page_size());
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
    use dynamo_kv_router::RadixTree;
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
    fn active_kv_lease_has_no_space_overhead() {
        let previous_fields = std::mem::size_of::<Vec<usize>>()
            + std::mem::size_of::<usize>()
            + std::mem::size_of::<Option<NodeId>>();

        assert_eq!(std::mem::size_of::<ActiveKvLease>(), previous_fields);
    }

    #[test]
    fn active_partial_page_owns_full_capacity_and_extends_in_place() {
        let mut mgr = SglangKvManager::new(12, 4, KvEventPublishers::default(), 0);
        let mut alloc = mgr.allocate_for_request(&[1]).unwrap();
        assert_eq!(mgr.cache().available_tokens(), 8);

        assert!(mgr.extend_allocation(&[1, 2, 3, 4], &mut alloc.lease));
        assert_eq!(
            mgr.cache().available_tokens(),
            8,
            "filling an owned partial page must not allocate another page"
        );

        assert!(mgr.extend_allocation(&[1, 2, 3, 4, 5], &mut alloc.lease));
        assert_eq!(mgr.cache().available_tokens(), 4);
        assert!(mgr.retract(alloc.lease));
        assert_eq!(mgr.cache().available_tokens(), 12);
    }

    #[test]
    fn fresh_allocation_protects_matched_prefix_before_eviction() {
        let mut mgr = SglangKvManager::new(12, 4, KvEventPublishers::default(), 0);
        let prefix = [1, 2, 3, 4];
        let other = [9, 10, 11, 12, 13, 14, 15, 16];

        let prefix_alloc = mgr.allocate_for_request(&prefix).unwrap();
        mgr.finish(&prefix, prefix_alloc.lease);
        let other_alloc = mgr.allocate_for_request(&other).unwrap();
        mgr.finish(&other, other_alloc.lease);
        // Make `prefix` the LRU victim. The new request must protect it before
        // evicting one of `other`'s pages to satisfy its suffix allocation.
        assert_eq!(mgr.cache_mut().match_prefix(&other).0, other.len());
        assert_eq!(mgr.cache().available_tokens(), 0);

        let extended = [1, 2, 3, 4, 5, 6, 7, 8];
        let alloc = mgr
            .allocate_for_request(&extended)
            .expect("protected prefix plus one evicted page should fit");

        assert_eq!(alloc.prefix_len, prefix.len());
        assert_eq!(mgr.cache().prefix_match_len(&prefix), prefix.len());
        assert_eq!(mgr.cache().available_tokens(), 0);
    }

    #[test]
    fn finish_drops_partial_tail_page_and_caches_compact_complete_page() {
        let mut mgr = SglangKvManager::new(12, 4, KvEventPublishers::default(), 0);
        let tokens = [1, 2, 3, 4, 5];
        let alloc = mgr.allocate_for_request(&tokens).unwrap();
        assert_eq!(mgr.cache().available_tokens(), 4);

        mgr.finish(&tokens, alloc.lease);

        assert_eq!(mgr.cache().available_tokens(), 8);
        let (matched, node) = mgr.cache_mut().match_prefix(&tokens);
        assert_eq!(matched, 4);
        assert_eq!(mgr.cache().node(node).key.len(), 1);
        assert_eq!(mgr.cache().node(node).value.len(), 1);
    }

    #[test]
    fn partially_consumed_decode_reservation_releases_only_unused_slots() {
        let mut mgr = SglangKvManager::new(4, 1, KvEventPublishers::default(), 0);
        let mut reservation = mgr.reserve_decode_pages(3).unwrap();
        let consumed = reservation.take(None);
        assert_eq!(reservation.len(), 2);
        let mut expected_unused = reservation.pages[reservation.next..].to_vec();

        mgr.release_decode_reservation(reservation);
        assert_eq!(mgr.cache().available_tokens(), 3);

        let mut reallocated = mgr
            .cache_mut()
            .page_pool
            .allocate_pages(expected_unused.len())
            .unwrap();
        expected_unused.sort_unstable();
        reallocated.sort_unstable();
        assert_eq!(reallocated, expected_unused);
        assert!(reallocated.windows(2).all(|pair| pair[0] != pair[1]));
        assert!(!reallocated.contains(&KvPageId::from_token_index(consumed, 1)));
    }

    #[test]
    fn retract_lease_releases_only_the_uncached_suffix() {
        let mut mgr = SglangKvManager::new(16, 4, KvEventPublishers::default(), 0);
        let mut alloc = mgr.allocate_for_request(&[1, 2, 3, 4]).unwrap();
        mgr.extend_cached_prefix(&[1, 2, 3, 4], &mut alloc.lease);
        assert!(mgr.extend_allocation(&[1, 2, 3, 4, 5, 6], &mut alloc.lease));

        assert_eq!(alloc.lease.cached_tokens(), 4);
        assert_eq!(alloc.lease.len(), 6);
        assert!(mgr.retract(alloc.lease));

        assert_eq!(mgr.cache().page_pool.available(), 12);
        assert_eq!(mgr.cache().protected_size, 0);
        assert_eq!(mgr.cache().evictable_size, 4);
        assert_eq!(mgr.cache().prefix_match_len(&[1, 2, 3, 4]), 4);
    }

    #[test]
    fn retained_tail_split_releases_leases_before_eviction() {
        let (buffer, sink) = capture_router_event_sink(ROUTER_TEST_WORKER_ID);
        let mut mgr = SglangKvManager::new(16, 4, KvEventPublishers::new(Some(sink), None), 0);
        let mut indexer = RadixTree::new();

        let first_tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut first = mgr.allocate_for_request(&first_tokens[..4]).unwrap();
        mgr.extend_cached_prefix(&first_tokens[..4], &mut first.lease);
        let retained_tail = first.lease.last_node();
        assert!(mgr.extend_allocation(&first_tokens, &mut first.lease));
        mgr.extend_cached_prefix(&first_tokens, &mut first.lease);

        assert_eq!(first.lease.last_node(), retained_tail);
        assert_eq!(mgr.cache().num_nodes(), 2);
        for event in buffer.drain() {
            indexer.apply_event(event).unwrap();
        }

        let second_tokens = [1, 2, 3, 4, 9, 10, 11, 12];
        let mut second = mgr.allocate_for_request(&second_tokens).unwrap();
        assert_eq!(second.prefix_len, 4);
        assert_eq!(first.lease.last_node(), retained_tail);
        assert_eq!(mgr.cache().num_nodes(), 3);
        mgr.extend_cached_prefix(&second_tokens, &mut second.lease);
        assert_eq!(mgr.cache().num_nodes(), 4);
        for event in buffer.drain() {
            indexer.apply_event(event).unwrap();
        }

        mgr.finish(&first_tokens, first.lease);
        assert!(mgr.retract(second.lease));
        assert_eq!(mgr.cache().protected_size, 0);
        assert_eq!(mgr.cache().evictable_size, 12);

        mgr.evict(12);
        for event in buffer.drain() {
            indexer.apply_event(event).unwrap();
        }
        assert_eq!(mgr.cache().page_pool.available(), 16);
        assert_eq!(mgr.cache().protected_size, 0);
        assert_eq!(mgr.cache().evictable_size, 0);
        assert_eq!(mgr.cache().num_nodes(), 1);
    }

    #[test]
    fn test_allocate_cache_miss() {
        let mut mgr = SglangKvManager::new(100, 1, KvEventPublishers::default(), 0);

        let result = mgr.allocate_for_request(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(result.prefix_len, 0);
        assert_eq!(result.lease.kv_indices.len(), 5);
        assert_eq!(mgr.cache().page_pool.available(), 95);
    }

    #[test]
    fn test_allocate_cache_hit() {
        let mut mgr = SglangKvManager::new(100, 1, KvEventPublishers::default(), 0);

        // First request: allocate and cache
        let r1 = mgr.allocate_for_request(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(r1.lease.kv_indices.len(), 5); // 5 pages (page_size=1)
        mgr.cache_finished_req(
            &[1, 2, 3, 4, 5],
            &r1.lease.kv_indices,
            r1.lease.last_node(),
            0,
        );

        // Second request with shared prefix
        let r2 = mgr.allocate_for_request(&[1, 2, 3, 4, 5, 6, 7]).unwrap();
        assert_eq!(r2.prefix_len, 5);
        assert_eq!(r2.lease.kv_indices.len(), 7); // 5 reused + 2 new pages
        assert_eq!(mgr.cache().page_pool.available(), 93); // 100 - 5 - 2
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
            &prefix.lease.kv_indices,
            prefix.lease.last_node(),
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
            &aligned.lease.kv_indices,
            aligned.lease.last_node(),
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
        mgr.free_request(result.lease.last_node());

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

        mgr.cache_finished_req(&[1, 2, 3], &r.lease.kv_indices, r.lease.last_node(), 0);

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
        mgr.cache_finished_req(
            &[1, 2, 3, 4, 5, 6],
            &r.lease.kv_indices,
            r.lease.last_node(),
            0,
        );

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
    fn test_published_prefix_hashes_only_unseen_suffix() {
        let sink = Arc::new(MockSink::new());
        let mut mgr =
            SglangKvManager::new(16, 4, KvEventPublishers::new(Some(sink.clone()), None), 0);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let indices = mgr.cache_mut().page_pool.allocate(tokens.len()).unwrap();

        assert_eq!(mgr.publish_stored_event(&tokens[..4], &indices[..4], 0), 1);
        assert_eq!(mgr.publish_stored_event(&tokens, &indices, 0), 1);
        assert_eq!(mgr.publish_stored_event(&tokens, &indices, 0), 0);

        let events = sink.clone_events();
        assert_eq!(events.len(), 2);
        let KvCacheEventData::Stored(first) = &events[0].data else {
            panic!("expected first stored event");
        };
        let KvCacheEventData::Stored(second) = &events[1].data else {
            panic!("expected suffix stored event");
        };
        assert_eq!(first.blocks.len(), 1);
        assert_eq!(second.blocks.len(), 1);
        assert_eq!(second.parent_hash, Some(first.blocks[0].block_hash));
    }

    #[test]
    fn test_cache_materialization_processes_only_newly_completed_blocks() {
        let sink = Arc::new(MockSink::new());
        let mut mgr = SglangKvManager::new(100, 2, KvEventPublishers::default(), 0);
        let tokens = [1, 2, 3, 4, 5, 6];

        let mut alloc = mgr.allocate_for_request(&tokens[..2]).unwrap();
        let alloc_last_node = alloc.lease.last_node();
        let first_last_node = mgr.cache_unfinished_req(
            &tokens[..2],
            &mut alloc.lease.kv_indices,
            alloc_last_node,
            0,
        );

        let mut kv_indices = alloc.lease.kv_indices;
        kv_indices.extend_from_slice(&mgr.cache_mut().page_pool.allocate(4).unwrap());
        mgr.kv_event_publishers = KvEventPublishers::new(Some(sink.clone()), None);

        let last_after_first_cache =
            mgr.cache_unfinished_req(&tokens[..4], &mut kv_indices[..4], first_last_node, 2);
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

        mgr.free_indices(&req1.lease.kv_indices);
        assert_eq!(sink.event_count(), 1);

        mgr.free_indices(&req2.lease.kv_indices);
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
            mgr.cache().page_pool.available(),
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

        mgr.cache_finished_req(&tokens, &req1.lease.kv_indices, req1.lease.last_node(), 0);
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
            mgr.cache().page_pool.available(),
            94,
            "canonical completion should retain the first request's slots"
        );

        mgr.cache_finished_req(&tokens, &req2.lease.kv_indices, req2.lease.last_node(), 0);
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
            mgr.cache().page_pool.available(),
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

    #[tokio::test]
    async fn retained_tail_eviction_preserves_page_granular_prefix_reuse() {
        let (buffer, sink) = capture_router_event_sink(ROUTER_TEST_WORKER_ID);
        let harness = RouterIndexerHarness::new(4, ROUTER_TEST_WORKER_ID);
        let mut mgr = SglangKvManager::new(8, 4, KvEventPublishers::new(Some(sink), None), 0);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];

        let mut request = mgr.allocate_for_request(&tokens[..4]).unwrap();
        mgr.extend_cached_prefix(&tokens[..4], &mut request.lease);
        assert!(mgr.extend_allocation(&tokens, &mut request.lease));
        mgr.extend_cached_prefix(&tokens, &mut request.lease);
        mgr.finish(&tokens, request.lease);

        let stored_events = buffer.drain();
        let query_hashes = stored_hashes(&stored_events);
        assert_eq!(query_hashes.len(), 2);
        harness.apply_events(stored_events).await;
        assert_eq!(harness.overlap_for_hashes(query_hashes.clone()).await, 2);
        assert_eq!(mgr.cache().evictable_size, 8);
        assert_eq!(mgr.cache().page_pool.available(), 0);

        mgr.evict(4);
        let eviction_events = buffer.drain();
        assert_eq!(removed_event_count(&eviction_events), 1);
        assert_eq!(removed_block_count(&eviction_events), 1);
        harness.apply_events(eviction_events).await;

        assert_eq!(mgr.cache().prefix_match_len(&tokens), 4);
        assert_eq!(mgr.cache().evictable_size, 4);
        assert_eq!(mgr.cache().protected_size, 0);
        assert_eq!(mgr.cache().page_pool.available(), 4);
        assert_eq!(harness.overlap_for_hashes(query_hashes).await, 1);
        harness.shutdown();
    }

    #[test]
    fn unfinished_duplicate_canonicalization_prevents_missing_parent() {
        let (buffer, sink) = capture_router_event_sink(ROUTER_TEST_WORKER_ID);
        let mut mgr = SglangKvManager::new(32, 4, KvEventPublishers::new(Some(sink), None), 0);
        let mut indexer = RadixTree::new();

        let seed_tokens = [1, 2, 3, 4];
        let seed = mgr.allocate_for_request(&seed_tokens).unwrap();
        mgr.finish(&seed_tokens, seed.lease);
        for event in buffer.drain() {
            indexer.apply_event(event).unwrap();
        }

        // Two requests miss the same suffix before either inserts it. Their
        // physical suffix pages are distinct even though the logical block is
        // identical.
        let shared_tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut first = mgr.allocate_for_request(&shared_tokens).unwrap();
        let mut duplicate = mgr.allocate_for_request(&shared_tokens).unwrap();
        let duplicate_suffix = duplicate.lease.indices()[seed_tokens.len()..].to_vec();
        assert_ne!(
            first.lease.indices()[seed_tokens.len()..],
            duplicate.lease.indices()[seed_tokens.len()..]
        );
        for event in buffer.drain() {
            indexer.apply_event(event).unwrap();
        }

        mgr.extend_cached_prefix(&shared_tokens, &mut first.lease);
        mgr.extend_cached_prefix(&shared_tokens, &mut duplicate.lease);
        assert_eq!(
            duplicate.lease.indices(),
            first.lease.indices(),
            "the active duplicate must switch to radix-owned canonical pages"
        );
        assert_eq!(
            mgr.cache().page_pool.available(),
            24,
            "every duplicate slot in the four-token page must return to the pool"
        );
        assert!(
            duplicate_suffix
                .iter()
                .all(|idx| !duplicate.lease.indices().contains(idx)),
            "no duplicate physical slot may remain attached to the active request"
        );
        for event in buffer.drain() {
            indexer.apply_event(event).unwrap();
        }

        // Mirror retracting the duplicate after its full prefix was cached,
        // then finish and evict the canonical request.
        assert!(mgr.retract(duplicate.lease));
        mgr.finish(&shared_tokens, first.lease);
        mgr.evict(seed_tokens.len());
        mgr.evict(seed_tokens.len());
        for event in buffer.drain() {
            indexer.apply_event(event).unwrap();
        }

        // Restore only the first block, then extend through the formerly
        // duplicated block. A leaked duplicate publisher refcount would
        // suppress re-storing block 2 and emit block 3 with a missing parent.
        let restored = mgr.allocate_for_request(&seed_tokens).unwrap();
        mgr.finish(&seed_tokens, restored.lease);
        for event in buffer.drain() {
            indexer.apply_event(event).unwrap();
        }

        let extended = mgr
            .allocate_for_request(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            .unwrap();
        let extension_events = buffer.drain();
        assert_eq!(stored_event_count(&extension_events), 1);
        let store = extension_events
            .iter()
            .find_map(|event| match &event.event.data {
                KvCacheEventData::Stored(store) => Some(store),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            store.blocks.len(),
            2,
            "both missing descendants must be stored"
        );
        for event in extension_events {
            indexer.apply_event(event).unwrap();
        }

        assert!(mgr.abort(extended.lease));
    }

    #[test]
    #[should_panic(expected = "invalid SGLang canonicalization range or radix path")]
    fn invalid_canonical_path_is_fatal() {
        let mut mgr = SglangKvManager::new(8, 4, KvEventPublishers::default(), 0);
        let mut indices = mgr.cache_mut().page_pool.allocate(4).unwrap();
        let root = mgr.cache().root();

        mgr.canonicalize_unfinished_indices(&mut indices, root, 0, 4);
    }

    #[test]
    fn cache_unfinished_rejects_invalid_range_before_publishing() {
        let sink = Arc::new(MockSink::new());
        let mut mgr =
            SglangKvManager::new(8, 4, KvEventPublishers::new(Some(sink.clone()), None), 0);
        let tokens = [1, 2, 3, 4];
        let mut alloc = mgr.allocate_for_request(&tokens).unwrap();
        let events_before = sink.event_count();
        let last_node = alloc.lease.last_node();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            mgr.cache_unfinished_req(&tokens, &mut alloc.lease.kv_indices, last_node, 2)
        }));

        assert!(result.is_err());
        assert_eq!(sink.event_count(), events_before);
    }

    #[test]
    fn cache_unfinished_rejects_short_indices_before_publishing() {
        let sink = Arc::new(MockSink::new());
        let mut mgr =
            SglangKvManager::new(8, 4, KvEventPublishers::new(Some(sink.clone()), None), 0);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut alloc = mgr.allocate_for_request(&tokens).unwrap();
        alloc.lease.kv_indices.truncate(4);
        let events_before = sink.event_count();
        let last_node = alloc.lease.last_node();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            mgr.cache_unfinished_req(&tokens, &mut alloc.lease.kv_indices, last_node, 0)
        }));

        assert!(result.is_err());
        assert_eq!(sink.event_count(), events_before);
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

        let mut alloc1 = mgr.allocate_for_request(&tokens[..chunk1_len]).unwrap();
        let previous_last = alloc1.lease.last_node();
        let new_last = mgr.cache_unfinished_req(
            &tokens[..chunk1_len],
            &mut alloc1.lease.kv_indices,
            previous_last,
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
