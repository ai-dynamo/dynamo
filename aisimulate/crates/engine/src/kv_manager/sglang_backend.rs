// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang KV manager — wraps [`RadixCache`] with request-level lifecycle
//! operations and KV event publishing.

use crate::cache::radix_cache::{KvPageId, NodeId, RadixCache};
use crate::common::hashing::{
    LocalBlockHash, SequenceHash, compute_block_hash_for_seq, compute_next_seq_hash,
};
use crate::common::kv_cache_trace;
use crate::common::protocols::KvEventPublishers;
use crate::{KvBlock, KvEvent, KvEventData, StoredBlocks};
use rustc_hash::FxHashMap;

/// Move-only ownership of a request's SGLang KV state.
///
/// Logical page hashes survive retraction so re-admission never rescans the
/// prompt. Physical pages and the radix lock exist only while the lease is
/// active.
#[derive(Debug, Default)]
#[must_use = "an active KV lease must be finished, aborted, or retracted"]
pub(crate) struct RadixRequestLease {
    pages: Vec<KvPageId>,
    materialized_tokens: usize,
    cached_tokens: usize,
    page_hashes: Vec<LocalBlockHash>,
    last_node: Option<NodeId>,
}

impl RadixRequestLease {
    #[cfg(test)]
    pub(crate) fn pages(&self) -> &[KvPageId] {
        &self.pages
    }

    pub(crate) fn len(&self) -> usize {
        self.materialized_tokens
    }

    #[cfg(any(test, debug_assertions))]
    pub(crate) fn page_count(&self) -> usize {
        self.pages.len()
    }

    pub(crate) fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    pub(crate) fn page_hashes(&self) -> &[LocalBlockHash] {
        &self.page_hashes
    }

    pub(crate) fn reserve_page_hashes(&mut self, complete_pages: usize) {
        self.page_hashes
            .reserve_exact(complete_pages.saturating_sub(self.page_hashes.len()));
    }

    #[cfg(test)]
    pub(crate) fn page_hash_capacity(&self) -> usize {
        self.page_hashes.capacity()
    }

    pub(crate) fn ensure_page_hashes(&mut self, token_ids: &[u32], page_size: usize) {
        let complete_pages = token_ids.len() / page_size;
        if self.page_hashes.len() >= complete_pages {
            return;
        }
        let first_new_token = self.page_hashes.len() * page_size;
        self.page_hashes.extend(compute_block_hash_for_seq(
            &token_ids[first_new_token..complete_pages * page_size],
            page_size,
        ));
    }

    fn page_hashes_through(&self, token_count: usize, page_size: usize) -> &[LocalBlockHash] {
        &self.page_hashes[..token_count / page_size]
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
        pages: Vec<KvPageId>,
        materialized_tokens: usize,
        cached_tokens: usize,
        last_node: NodeId,
    ) -> Self {
        Self {
            pages,
            materialized_tokens,
            cached_tokens,
            page_hashes: Vec::new(),
            last_node: Some(last_node),
        }
    }
}

/// Result of `allocate_for_request`.
#[cfg(test)]
pub(crate) struct AllocResult {
    /// Number of tokens matched from the prefix cache.
    pub(crate) prefix_len: usize,
    pub(crate) lease: RadixRequestLease,
}

pub struct SglangKvManager {
    cache: RadixCache,
    enable_prefix_caching: bool,
    kv_event_publishers: KvEventPublishers,
    dp_rank: u32,
    next_event_id: u64,
    /// Maps each dense physical page ID to the block hash assigned during
    /// Stored events, so Removed events can use the same block hash.
    page_to_block_hash: Vec<Option<SequenceHash>>,
    /// Tracks how many live pool slots currently advertise the same logical
    /// block hash so router events reflect logical block visibility, not
    /// transient slot ownership.
    block_hash_refcounts: FxHashMap<SequenceHash, usize>,
}

pub struct DecodeTokenReservation {
    pages: Vec<KvPageId>,
    next: usize,
}

pub struct SglangDestinationReservation {
    pub(crate) prefix_len: usize,
    prefix_pages: Vec<KvPageId>,
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
    pub(crate) fn pages(&self) -> Vec<KvPageId> {
        self.prefix_pages
            .iter()
            .chain(&self.unpublished_pages)
            .copied()
            .collect()
    }
}

impl DecodeTokenReservation {
    fn take_page(&mut self) -> KvPageId {
        let page = *self
            .pages
            .get(self.next)
            .expect("reserved decode page allocation must be infallible");
        self.next += 1;
        page
    }

    pub(crate) fn len(&self) -> usize {
        self.pages.len() - self.next
    }
}

impl SglangKvManager {
    #[cfg(test)]
    pub fn new(
        total_tokens: usize,
        page_size: usize,
        kv_event_publishers: KvEventPublishers,
        dp_rank: u32,
    ) -> Self {
        Self::new_with_prefix_caching(total_tokens, page_size, kv_event_publishers, dp_rank, true)
    }

    pub(crate) fn new_with_prefix_caching(
        total_tokens: usize,
        page_size: usize,
        kv_event_publishers: KvEventPublishers,
        dp_rank: u32,
        enable_prefix_caching: bool,
    ) -> Self {
        let page_to_block_hash = if kv_event_publishers.is_empty() {
            Vec::new()
        } else {
            vec![None; total_tokens / page_size]
        };
        Self {
            cache: RadixCache::new(total_tokens, page_size),
            enable_prefix_caching,
            kv_event_publishers,
            dp_rank,
            next_event_id: 0,
            page_to_block_hash,
            block_hash_refcounts: FxHashMap::default(),
        }
    }

    #[cfg(test)]
    fn page_metadata_len(&self) -> usize {
        self.page_to_block_hash.len()
    }

    pub fn cache(&self) -> &RadixCache {
        &self.cache
    }

    #[cfg(test)]
    pub fn cache_mut(&mut self) -> &mut RadixCache {
        &mut self.cache
    }

    /// Match and protect a reusable prefix, evict other cached pages if needed,
    /// then allocate KV pages for a new request.
    ///
    /// Returns `None` if protected and free capacity cannot satisfy the request.
    pub(crate) fn allocate_for_request_lease(
        &mut self,
        token_ids: &[u32],
        lease: &mut RadixRequestLease,
    ) -> Option<usize> {
        assert!(!lease.is_active(), "request KV lease is already active");
        let page_size = self.cache.page_size();
        lease.ensure_page_hashes(token_ids, page_size);
        let materialized_hashes = lease.page_hashes_through(token_ids.len(), page_size);
        let (prefix_len, last_node) = self.match_prefix_hashes_and_lock(materialized_hashes);
        let required_pages = token_ids.len().div_ceil(page_size) - prefix_len / page_size;
        let required_tokens = required_pages * page_size;

        // Protect the matched path before making room. Otherwise an LRU
        // eviction can remove the prefix used to size this allocation, and a
        // second match would require more pages than were freed.
        let reservable = self.cache.available_tokens() + self.cache.evictable_size;
        if required_tokens > reservable {
            self.cache.dec_lock_ref(last_node);
            return None;
        }
        let available = self.cache.available_tokens();
        if required_tokens > available {
            self.evict(required_tokens - available);
        }
        let mut pages = self.collect_path_pages_through(last_node, prefix_len);

        let available_before = self.cache.available_tokens();
        let Some(mut new_pages) = self.cache.page_pool.allocate_pages(required_pages) else {
            self.cache.dec_lock_ref(last_node);
            return None;
        };
        pages.append(&mut new_pages);
        let allocated_tokens = available_before - self.cache.available_tokens();

        // Observer-visible KV events are complete-block only.
        self.publish_stored_hashes(materialized_hashes, &pages, token_ids.len(), prefix_len);

        self.log_trace("allocation", allocated_tokens);

        lease.pages = pages;
        lease.materialized_tokens = token_ids.len();
        lease.cached_tokens = prefix_len;
        lease.last_node = Some(last_node);
        Some(prefix_len)
    }

    #[cfg(test)]
    pub(crate) fn allocate_for_request(&mut self, token_ids: &[u32]) -> Option<AllocResult> {
        let mut lease = RadixRequestLease::default();
        let prefix_len = self.allocate_for_request_lease(token_ids, &mut lease)?;
        Some(AllocResult { prefix_len, lease })
    }

    /// Continue an in-flight request from an already materialized prefix.
    ///
    /// This is used by chunked-prefill continuation where the request still
    /// owns token slots for a prefix that may extend past the radix-tree's
    /// page-aligned cached prefix.
    pub(crate) fn extend_allocation(
        &mut self,
        token_ids: &[u32],
        lease: &mut RadixRequestLease,
    ) -> bool {
        let prefix_len = lease.materialized_tokens;
        assert!(
            lease.is_active() && prefix_len <= token_ids.len(),
            "invalid SGLang KV lease extension: active={}, owned_tokens={prefix_len}, target_tokens={}",
            lease.is_active(),
            token_ids.len()
        );
        let page_size = self.cache.page_size();
        let target_pages = token_ids.len().div_ceil(page_size);
        let new_pages = target_pages.saturating_sub(lease.pages.len());
        let available_before = self.cache.available_tokens();
        let Some(mut allocated_pages) = self.cache.page_pool.allocate_pages(new_pages) else {
            return false;
        };
        lease.pages.append(&mut allocated_pages);
        lease.materialized_tokens = token_ids.len();
        lease.ensure_page_hashes(token_ids, page_size);
        let allocated_tokens = available_before - self.cache.available_tokens();

        self.publish_stored_hashes(
            lease.page_hashes_through(token_ids.len(), page_size),
            &lease.pages,
            lease.materialized_tokens,
            prefix_len,
        );
        self.log_trace("allocation", allocated_tokens);
        true
    }

    pub(crate) fn extend_cached_prefix(
        &mut self,
        token_ids: &[u32],
        lease: &mut RadixRequestLease,
    ) {
        if !self.enable_prefix_caching {
            return;
        }
        lease.ensure_page_hashes(token_ids, self.cache.page_size());
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
        let complete_pages = complete_len / self.cache.page_size();
        let page_hashes = &lease.page_hashes[..complete_pages];
        let pages = &mut lease.pages[..complete_pages];
        let new_last_node =
            self.cache_unfinished_hashes(page_hashes, pages, last_node, lease.cached_tokens);
        lease.last_node = Some(new_last_node);
        lease.cached_tokens = complete_len;
    }

    pub(crate) fn extend_decode(
        &mut self,
        lease: &mut RadixRequestLease,
        reservation: &mut DecodeTokenReservation,
    ) {
        debug_assert!(lease.is_active());
        if lease
            .materialized_tokens
            .is_multiple_of(self.cache.page_size())
        {
            lease.pages.push(reservation.take_page());
        }
        lease.materialized_tokens += 1;
    }

    pub(crate) fn finish(&mut self, token_ids: &[u32], mut lease: RadixRequestLease) {
        let Some(last_node) = lease.last_node.take() else {
            debug_assert!(lease.pages.is_empty());
            debug_assert_eq!(lease.materialized_tokens, 0);
            debug_assert_eq!(lease.cached_tokens, 0);
            return;
        };
        if !self.enable_prefix_caching {
            self.free_pages(&lease.pages);
            self.cache.dec_lock_ref(last_node);
            return;
        }
        let complete_len =
            token_ids.len().min(lease.len()) / self.cache.page_size() * self.cache.page_size();
        assert!(
            lease.cached_tokens <= complete_len,
            "invalid SGLang KV lease finish: cached_tokens={}, complete_tokens={complete_len}, owned_tokens={}",
            lease.cached_tokens,
            lease.len()
        );
        let complete_pages = complete_len / self.cache.page_size();
        self.free_pages(&lease.pages[complete_pages..]);
        lease.pages.truncate(complete_pages);
        lease.materialized_tokens = complete_len;

        if complete_len == 0 {
            self.cache.dec_lock_ref(last_node);
            return;
        }
        self.cache_finished_hashes(
            lease.page_hashes_through(complete_len, self.cache.page_size()),
            &lease.pages,
            last_node,
            lease.cached_tokens,
        );
    }

    pub(crate) fn abort(&mut self, mut lease: RadixRequestLease) -> bool {
        self.release_active_lease(&mut lease)
    }

    pub(crate) fn retract_in_place(&mut self, lease: &mut RadixRequestLease) -> bool {
        self.release_active_lease(lease)
    }

    #[cfg(test)]
    pub(crate) fn retract(&mut self, mut lease: RadixRequestLease) -> bool {
        self.release_active_lease(&mut lease)
    }

    /// Cache a completed request's full sequence into the radix tree.
    ///
    /// Inserts the full token sequence so future requests can reuse it,
    /// then unlocks the path.
    fn cache_finished_hashes(
        &mut self,
        page_hashes: &[LocalBlockHash],
        pages: &[KvPageId],
        last_node: NodeId,
        first_new_token: usize,
    ) {
        let complete_len = page_hashes.len() * self.cache.page_size();
        self.publish_stored_hashes(page_hashes, pages, complete_len, first_new_token);
        let new_last_node =
            self.cache
                .insert_page_hashes_from_node(last_node, first_new_token, page_hashes, pages);
        self.release_unretained_finished_pages(pages, new_last_node, first_new_token, complete_len);
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
    fn cache_unfinished_hashes(
        &mut self,
        page_hashes: &[LocalBlockHash],
        pages: &mut [KvPageId],
        last_node: NodeId,
        first_new_token: usize,
    ) -> NodeId {
        let block_size = self.cache.page_size();
        let complete_len = page_hashes.len() * block_size;
        assert!(
            first_new_token.is_multiple_of(block_size)
                && first_new_token <= complete_len
                && complete_len / block_size <= pages.len(),
            "invalid SGLang canonicalization range: first_new_token={first_new_token}, complete_len={complete_len}, pages={}",
            pages.len()
        );

        self.publish_stored_hashes(page_hashes, pages, complete_len, first_new_token);
        let new_last_node =
            self.cache
                .insert_page_hashes_from_node(last_node, first_new_token, page_hashes, pages);

        // An interleaved insert can retain different physical pages for the same prefix.
        // Move the active request to canonical pages before releasing its duplicates.
        // Acquire the extended path before releasing the old prefix so
        // destination activation never leaves valid transferred KV unprotected.
        if new_last_node != last_node {
            self.cache.inc_lock_ref(new_last_node);
        }
        self.canonicalize_unfinished_pages(pages, new_last_node, first_new_token, complete_len);
        if new_last_node != last_node {
            self.cache.dec_lock_ref(last_node);
        }

        new_last_node
    }

    pub fn reserve_decode_pages(&mut self, count: usize) -> Option<DecodeTokenReservation> {
        let pages = self.cache.page_pool.allocate_pages(count)?;
        if !pages.is_empty() {
            self.log_trace("allocation", pages.len() * self.cache.page_size());
        }
        Some(DecodeTokenReservation { pages, next: 0 })
    }

    pub(crate) fn reserve_destination_lease(
        &mut self,
        page_hashes: &[LocalBlockHash],
        token_count: usize,
    ) -> Option<SglangDestinationReservation> {
        let (prefix_len, last_node) = self.match_prefix_hashes_and_lock(page_hashes);
        let prefix_pages = self.collect_path_pages_through(last_node, prefix_len);

        let allocated_tokens = if token_count == 0 {
            0
        } else {
            token_count.div_ceil(self.cache.page_size()) * self.cache.page_size()
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
            prefix_pages,
            last_node,
            unpublished_pages,
            page_size: self.cache.page_size(),
            missing_tokens: token_count.saturating_sub(prefix_len),
            allocated_tokens,
        })
    }

    pub(crate) fn activate_destination_lease(
        &mut self,
        reservation: SglangDestinationReservation,
        token_count: usize,
        lease: &mut RadixRequestLease,
    ) -> usize {
        let SglangDestinationReservation {
            prefix_len,
            mut prefix_pages,
            last_node,
            mut unpublished_pages,
            page_size: _,
            missing_tokens,
            allocated_tokens: _,
        } = reservation;
        prefix_pages.append(&mut unpublished_pages);
        let (new_last_node, cached_tokens) = if self.enable_prefix_caching {
            (
                self.cache_unfinished_hashes(
                    lease.page_hashes_through(token_count, self.cache.page_size()),
                    &mut prefix_pages,
                    last_node,
                    prefix_len,
                ),
                token_count / self.cache.page_size() * self.cache.page_size(),
            )
        } else {
            (last_node, 0)
        };
        self.log_trace("activate_destination", missing_tokens);
        lease.pages = prefix_pages;
        lease.materialized_tokens = token_count;
        lease.cached_tokens = cached_tokens;
        lease.last_node = Some(new_last_node);
        prefix_len
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

    fn release_active_lease(&mut self, lease: &mut RadixRequestLease) -> bool {
        let Some(last_node) = lease.last_node.take() else {
            debug_assert!(lease.pages.is_empty());
            debug_assert_eq!(lease.materialized_tokens, 0);
            debug_assert_eq!(lease.cached_tokens, 0);
            return false;
        };
        assert!(
            lease.cached_tokens <= lease.len(),
            "invalid SGLang KV lease release: cached_tokens={}, owned_tokens={}",
            lease.cached_tokens,
            lease.len()
        );
        let first_owned_page = lease.cached_tokens / self.cache.page_size();
        let owned_suffix = &lease.pages[first_owned_page..];
        let capacity_improved = !owned_suffix.is_empty() || last_node != self.cache.root();
        self.free_pages(owned_suffix);
        self.cache.dec_lock_ref(last_node);
        lease.pages.clear();
        lease.materialized_tokens = 0;
        lease.cached_tokens = 0;
        capacity_improved
    }

    fn match_prefix_hashes_and_lock(&mut self, page_hashes: &[LocalBlockHash]) -> (usize, NodeId) {
        if self.enable_prefix_caching {
            self.cache.match_prefix_hashes_and_lock(page_hashes)
        } else {
            (0, self.cache.root())
        }
    }

    /// Collect physical pages from the matched prefix path by walking root→last_node.
    fn collect_path_pages(&self, last_node: NodeId) -> Vec<KvPageId> {
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

        let mut pages = Vec::new();
        for node_id in path {
            pages.extend_from_slice(&self.cache.node(node_id).value);
        }
        pages
    }

    fn collect_path_pages_through(&self, last_node: NodeId, prefix_len: usize) -> Vec<KvPageId> {
        assert_eq!(
            prefix_len % self.cache.page_size(),
            0,
            "matched SGLang prefix must be page-aligned"
        );
        let expected_pages = prefix_len / self.cache.page_size();
        let mut pages = self.collect_path_pages(last_node);
        assert!(
            pages.len() >= expected_pages,
            "SGLang radix path returned {} pages for a {expected_pages}-page prefix",
            pages.len()
        );
        pages.truncate(expected_pages);
        pages
    }

    fn release_unretained_finished_pages(
        &mut self,
        pages: &[KvPageId],
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

            for (page_idx, &incoming_page) in pages
                .iter()
                .enumerate()
                .take(path_end)
                .skip(reconcile_start)
            {
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

    fn canonicalize_unfinished_pages(
        &mut self,
        pages: &mut [KvPageId],
        last_node: NodeId,
        first_new_token: usize,
        complete_len: usize,
    ) {
        let block_size = self.cache.page_size();
        debug_assert_eq!(complete_len % block_size, 0);
        debug_assert_eq!(first_new_token % block_size, 0);
        debug_assert!(complete_len / block_size <= pages.len());
        debug_assert!(first_new_token <= complete_len);

        assert!(
            first_new_token.is_multiple_of(block_size)
                && complete_len.is_multiple_of(block_size)
                && complete_len / block_size <= pages.len()
                && first_new_token <= complete_len
                && self.radix_path_covers(last_node, first_new_token, complete_len),
            "invalid SGLang canonicalization range or radix path: first_new_token={first_new_token}, complete_len={complete_len}, pages={}",
            pages.len()
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

            for (page_idx, incoming_page) in pages
                .iter_mut()
                .enumerate()
                .take(path_end)
                .skip(reconcile_start)
            {
                let canonical_page = node.value[page_idx - path_start];
                if *incoming_page != canonical_page {
                    unretained_pages.push(*incoming_page);
                    *incoming_page = canonical_page;
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

    fn publish_stored_hashes(
        &mut self,
        page_hashes: &[LocalBlockHash],
        pages: &[KvPageId],
        num_tokens: usize,
        first_new_token: usize,
    ) -> usize {
        if self.kv_event_publishers.is_empty() {
            return 0;
        }
        if self.page_to_block_hash.is_empty() {
            self.page_to_block_hash
                .resize(self.cache.total_tokens() / self.cache.page_size(), None);
        }

        let block_size = self.cache.page_size();
        let complete_len =
            (page_hashes.len() * block_size).min(num_tokens) / block_size * block_size;
        if complete_len == 0 || first_new_token >= complete_len {
            return 0;
        }
        let complete_pages = complete_len / block_size;
        assert!(
            pages.len() >= complete_pages,
            "not enough KV pages for Stored event: need {complete_pages}, got {}",
            pages.len()
        );

        let first_page = first_new_token / block_size;
        let Some(first_unpublished_page) = (first_page..complete_pages)
            .find(|&page_idx| self.page_to_block_hash[pages[page_idx].index()].is_none())
        else {
            return 0;
        };

        let local_hashes = &page_hashes[first_unpublished_page..complete_pages];
        let mut parent_hash = None;
        let mut blocks = Vec::new();
        let mut publishing = false;

        for (block_idx, tokens_hash) in local_hashes.iter().copied().enumerate() {
            let page_idx = first_unpublished_page + block_idx;
            let page = pages[page_idx];
            let page_slot = page.index();
            if self.page_to_block_hash[page_slot].is_some() {
                continue;
            }

            let block_parent_hash = if page_idx == 0 {
                None
            } else {
                self.page_to_block_hash[pages[page_idx - 1].index()]
            };
            let block_hash = match block_parent_hash {
                Some(parent_hash) => compute_next_seq_hash(parent_hash, tokens_hash),
                None => tokens_hash.0,
            };

            self.page_to_block_hash[page_slot] = Some(block_hash);
            let refcount = self.block_hash_refcounts.entry(block_hash).or_default();
            *refcount += 1;
            if *refcount == 1 && !publishing {
                publishing = true;
                parent_hash = block_parent_hash;
            }
            if publishing {
                blocks.push(KvBlock {
                    block_hash,
                    tokens_hash: tokens_hash.0,
                    token_ids: None,
                });
            }
        }

        let hashed_blocks = local_hashes.len();
        if blocks.is_empty() {
            return hashed_blocks;
        }

        let event = KvEvent {
            event_id: self.next_event_id,
            data: KvEventData::Stored(StoredBlocks {
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

    fn publish_removed_pages(&mut self, evicted_pages: &[KvPageId]) {
        if self.kv_event_publishers.is_empty() {
            return;
        }

        let mut block_hashes = Vec::new();
        for (page_idx, &page) in evicted_pages.iter().enumerate() {
            let Some(block_hash) = self.page_to_block_hash[page.index()].take() else {
                continue;
            };
            if let std::collections::hash_map::Entry::Occupied(mut entry) =
                self.block_hash_refcounts.entry(block_hash)
            {
                if *entry.get() > 1 {
                    *entry.get_mut() -= 1;
                } else {
                    entry.remove();
                    if block_hashes.is_empty() {
                        block_hashes.reserve_exact(evicted_pages.len() - page_idx);
                    }
                    block_hashes.push(block_hash);
                }
            }
        }

        if block_hashes.is_empty() {
            return;
        }

        let event = KvEvent {
            event_id: self.next_event_id,
            data: KvEventData::Removed { block_hashes },
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

    use crate::common::hashing::{compute_block_hash_for_seq, compute_seq_hash_for_block};
    use crate::common::protocols::KvCacheEventSink;
    use crate::{KvEvent, KvEventData};

    fn lease_with_hashes(tokens: &[u32], page_size: usize) -> RadixRequestLease {
        let mut lease = RadixRequestLease::default();
        lease.ensure_page_hashes(tokens, page_size);
        lease
    }

    struct MockSink {
        events: Mutex<Vec<KvEvent>>,
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

        fn clone_events(&self) -> Vec<KvEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl KvCacheEventSink for MockSink {
        fn publish(&self, event: KvEvent) -> anyhow::Result<()> {
            self.events.lock().unwrap().push(event);
            Ok(())
        }
    }

    #[test]
    fn dense_page_metadata_is_allocated_only_for_kv_events() {
        let disabled = SglangKvManager::new(64, 4, KvEventPublishers::default(), 0);
        assert_eq!(disabled.page_metadata_len(), 0);

        let sink = Arc::new(MockSink::new());
        let enabled = SglangKvManager::new(64, 4, KvEventPublishers::new(Some(sink)), 0);
        assert_eq!(enabled.page_metadata_len(), 16);
    }

    #[test]
    fn active_kv_lease_retains_one_id_per_physical_page() {
        let mut mgr = SglangKvManager::new(64, 16, KvEventPublishers::default(), 0);
        let mut tokens = vec![1; 33];
        let mut alloc = mgr.allocate_for_request(&tokens).unwrap();

        assert_eq!(alloc.lease.len(), 33);
        assert_eq!(alloc.lease.page_count(), 3);
        assert_eq!(mgr.cache().available_tokens(), 16);

        tokens.resize(48, 2);
        assert!(mgr.extend_allocation(&tokens, &mut alloc.lease));
        assert_eq!(alloc.lease.page_count(), 3);
        assert_eq!(mgr.cache().available_tokens(), 16);

        tokens.push(3);
        assert!(mgr.extend_allocation(&tokens, &mut alloc.lease));
        assert_eq!(alloc.lease.page_count(), 4);
        assert_eq!(mgr.cache().available_tokens(), 0);
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
    fn page_native_extension_oom_is_atomic() {
        let mut mgr = SglangKvManager::new(8, 4, KvEventPublishers::default(), 0);
        let mut alloc = mgr.allocate_for_request(&[1]).unwrap();
        let blocker = mgr.allocate_for_request(&[9]).unwrap();
        let pages_before = alloc.lease.pages().to_vec();

        assert!(!mgr.extend_allocation(&[1, 2, 3, 4, 5], &mut alloc.lease));
        assert_eq!(alloc.lease.pages(), pages_before);
        assert_eq!(alloc.lease.len(), 1);
        assert_eq!(mgr.cache().available_tokens(), 0);

        mgr.abort(alloc.lease);
        mgr.abort(blocker.lease);
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
        let consumed = reservation.take_page();
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
        assert!(!reallocated.contains(&consumed));
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
    fn destination_activation_bounds_hashes_to_materialized_tokens() {
        let mut mgr = SglangKvManager::new(8, 4, KvEventPublishers::default(), 0);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut lease = RadixRequestLease::default();
        lease.ensure_page_hashes(&tokens, 4);

        let reservation = mgr
            .reserve_destination_lease(lease.page_hashes(), 4)
            .unwrap();
        assert_eq!(
            mgr.activate_destination_lease(reservation, 4, &mut lease),
            0
        );

        assert_eq!(lease.page_count(), 1);
        assert_eq!(lease.cached_tokens(), 4);
        assert_eq!(mgr.cache().prefix_match_len(&tokens[..4]), 4);
        assert_eq!(mgr.cache().prefix_match_len(&tokens), 4);
        mgr.abort(lease);
    }

    #[test]
    fn retract_and_readmit_reuses_cached_page_hashes() {
        let mut mgr = SglangKvManager::new(8, 4, KvEventPublishers::default(), 0);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut lease = RadixRequestLease::default();
        assert_eq!(mgr.allocate_for_request_lease(&tokens, &mut lease), Some(0));
        mgr.extend_cached_prefix(&tokens, &mut lease);

        let hashes = lease.page_hashes().to_vec();
        let hash_storage = lease.page_hashes().as_ptr();
        let hash_capacity = lease.page_hashes.capacity();
        assert!(mgr.retract_in_place(&mut lease));
        assert_eq!(lease.page_hashes(), hashes);

        assert_eq!(
            mgr.allocate_for_request_lease(&tokens, &mut lease),
            Some(tokens.len())
        );
        assert_eq!(lease.page_hashes(), hashes);
        assert_eq!(lease.page_hashes().as_ptr(), hash_storage);
        assert_eq!(lease.page_hashes.capacity(), hash_capacity);
        mgr.abort(lease);
    }

    #[test]
    fn retained_tail_split_releases_leases_before_eviction() {
        let mut mgr = SglangKvManager::new(16, 4, KvEventPublishers::default(), 0);

        let first_tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut first = mgr.allocate_for_request(&first_tokens[..4]).unwrap();
        mgr.extend_cached_prefix(&first_tokens[..4], &mut first.lease);
        let retained_tail = first.lease.last_node();
        assert!(mgr.extend_allocation(&first_tokens, &mut first.lease));
        mgr.extend_cached_prefix(&first_tokens, &mut first.lease);

        assert_eq!(first.lease.last_node(), retained_tail);
        assert_eq!(mgr.cache().num_nodes(), 2);
        let second_tokens = [1, 2, 3, 4, 9, 10, 11, 12];
        let mut second = mgr.allocate_for_request(&second_tokens).unwrap();
        assert_eq!(second.prefix_len, 4);
        assert_eq!(first.lease.last_node(), retained_tail);
        assert_eq!(mgr.cache().num_nodes(), 3);
        mgr.extend_cached_prefix(&second_tokens, &mut second.lease);
        assert_eq!(mgr.cache().num_nodes(), 4);
        mgr.finish(&first_tokens, first.lease);
        assert!(mgr.retract(second.lease));
        assert_eq!(mgr.cache().protected_size, 0);
        assert_eq!(mgr.cache().evictable_size, 12);

        mgr.evict(12);
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
        assert_eq!(result.lease.pages.len(), 5);
        assert_eq!(mgr.cache().page_pool.available(), 95);
    }

    #[test]
    fn test_allocate_cache_hit() {
        let mut mgr = SglangKvManager::new(100, 1, KvEventPublishers::default(), 0);

        // First request: allocate and cache
        let r1 = mgr.allocate_for_request(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(r1.lease.pages.len(), 5);
        mgr.finish(&[1, 2, 3, 4, 5], r1.lease);

        // Second request with shared prefix
        let r2 = mgr.allocate_for_request(&[1, 2, 3, 4, 5, 6, 7]).unwrap();
        assert_eq!(r2.prefix_len, 5);
        assert_eq!(r2.lease.pages.len(), 7);
        assert_eq!(mgr.cache().page_pool.available(), 93); // 100 - 5 - 2
    }

    #[test]
    fn destination_transfer_footprint_uses_missing_physical_pages() {
        let mut mgr = SglangKvManager::new(64, 4, KvEventPublishers::default(), 0);
        let prompt = (0..10).collect::<Vec<_>>();

        let cold_lease = lease_with_hashes(&prompt, 4);
        let cold = mgr
            .reserve_destination_lease(cold_lease.page_hashes(), prompt.len())
            .expect("cold destination reservation should fit");
        assert_eq!(cold.transferable_prompt_tokens(), 12);
        mgr.cancel_destination(cold);

        let prefix_tokens = &prompt[..4];
        let prefix = mgr
            .allocate_for_request(prefix_tokens)
            .expect("prefix allocation should fit");
        mgr.finish(prefix_tokens, prefix.lease);
        let partial_lease = lease_with_hashes(&prompt, 4);
        let partial = mgr
            .reserve_destination_lease(partial_lease.page_hashes(), prompt.len())
            .expect("partially cached destination reservation should fit");
        assert_eq!(partial.transferable_prompt_tokens(), 8);
        mgr.cancel_destination(partial);

        let aligned_tokens = (20..28).collect::<Vec<_>>();
        let aligned = mgr
            .allocate_for_request(&aligned_tokens)
            .expect("aligned prompt allocation should fit");
        mgr.finish(&aligned_tokens, aligned.lease);
        let full_hit_lease = lease_with_hashes(&aligned_tokens, 4);
        let full_hit = mgr
            .reserve_destination_lease(full_hit_lease.page_hashes(), aligned_tokens.len())
            .expect("fully cached destination reservation should fit");
        assert_eq!(full_hit.transferable_prompt_tokens(), 0);
    }

    #[test]
    fn test_free_request_without_caching() {
        let mut mgr = SglangKvManager::new(100, 1, KvEventPublishers::default(), 0);

        let result = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        assert!(mgr.abort(result.lease));

        // The request path is unlocked and its private pages are returned.
        assert_eq!(mgr.cache().protected_size, 0);
    }

    #[test]
    fn test_event_publishing() {
        let sink = Arc::new(MockSink::new());
        let mut mgr = SglangKvManager::new(100, 1, KvEventPublishers::new(Some(sink.clone())), 0);

        let r = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        assert_eq!(sink.event_count(), 1); // BlockStored for 3 new pages

        mgr.finish(&[1, 2, 3], r.lease);

        // Second request with full cache hit → no new events
        let r2 = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        assert_eq!(r2.prefix_len, 3);
        assert_eq!(sink.event_count(), 1); // no new event
    }

    #[test]
    fn reused_physical_page_replaces_dense_event_metadata() {
        let sink = Arc::new(MockSink::new());
        let mut mgr = SglangKvManager::new(1, 1, KvEventPublishers::new(Some(sink.clone())), 0);

        let first = mgr.allocate_for_request(&[1]).unwrap();
        let page = first.lease.pages()[0];
        assert!(mgr.abort(first.lease));

        let second = mgr.allocate_for_request(&[2]).unwrap();
        assert_eq!(second.lease.pages(), &[page]);

        let events = sink.clone_events();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event_id, 0);
        assert_eq!(events[1].event_id, 1);
        assert_eq!(events[2].event_id, 2);
        let KvEventData::Stored(first_store) = &events[0].data else {
            panic!("expected first stored event");
        };
        let KvEventData::Removed { block_hashes } = &events[1].data else {
            panic!("expected removal before page reuse");
        };
        let KvEventData::Stored(second_store) = &events[2].data else {
            panic!("expected replacement stored event");
        };
        assert_eq!(block_hashes, &vec![first_store.blocks[0].block_hash]);
        assert_ne!(
            first_store.blocks[0].block_hash,
            second_store.blocks[0].block_hash
        );
    }

    #[test]
    fn test_event_publishing_uses_native_block_hashes() {
        let sink = Arc::new(MockSink::new());
        let mut mgr = SglangKvManager::new(100, 4, KvEventPublishers::new(Some(sink.clone())), 0);

        let r = mgr.allocate_for_request(&[1, 2, 3, 4, 5, 6]).unwrap();
        mgr.finish(&[1, 2, 3, 4, 5, 6], r.lease);

        let events = sink.clone_events();
        assert_eq!(events.len(), 1);
        let KvEventData::Stored(store) = &events[0].data else {
            panic!("expected stored event");
        };
        assert_eq!(store.blocks.len(), 1);

        let expected_local = compute_block_hash_for_seq(&[1, 2, 3, 4], 4);
        let expected_sequence = compute_seq_hash_for_block(&expected_local);
        assert_eq!(store.blocks[0].tokens_hash, expected_local[0].0);
        assert_eq!(store.blocks[0].block_hash, expected_sequence[0]);
    }

    #[test]
    fn test_published_prefix_hashes_only_unseen_suffix() {
        let sink = Arc::new(MockSink::new());
        let mut mgr = SglangKvManager::new(16, 4, KvEventPublishers::new(Some(sink.clone())), 0);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let pages = mgr.cache_mut().page_pool.allocate_pages(2).unwrap();
        let lease = lease_with_hashes(&tokens, 4);

        assert_eq!(
            mgr.publish_stored_hashes(lease.page_hashes_through(4, 4), &pages[..1], 4, 0),
            1
        );
        assert_eq!(
            mgr.publish_stored_hashes(lease.page_hashes_through(8, 4), &pages, 8, 0),
            1
        );
        assert_eq!(
            mgr.publish_stored_hashes(lease.page_hashes_through(8, 4), &pages, 8, 0),
            0
        );

        let events = sink.clone_events();
        assert_eq!(events.len(), 2);
        let KvEventData::Stored(first) = &events[0].data else {
            panic!("expected first stored event");
        };
        let KvEventData::Stored(second) = &events[1].data else {
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
        mgr.extend_cached_prefix(&tokens[..2], &mut alloc.lease);
        mgr.kv_event_publishers = KvEventPublishers::new(Some(sink.clone()));

        assert!(mgr.extend_allocation(&tokens[..4], &mut alloc.lease));
        mgr.extend_cached_prefix(&tokens[..4], &mut alloc.lease);
        let events = sink.clone_events();
        assert_eq!(events.len(), 1);
        let KvEventData::Stored(first_store) = &events[0].data else {
            panic!("expected first cache event to be Stored");
        };
        assert_eq!(
            first_store.blocks.len(),
            1,
            "first unfinished cache should store only the newly completed block"
        );

        assert!(mgr.extend_allocation(&tokens, &mut alloc.lease));
        mgr.finish(&tokens, alloc.lease);
        let events = sink.clone_events();
        assert_eq!(events.len(), 2);
        let KvEventData::Stored(final_store) = &events[1].data else {
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
        let mut mgr = SglangKvManager::new(100, 1, KvEventPublishers::new(Some(sink.clone())), 0);

        let req1 = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        let req2 = mgr.allocate_for_request(&[1, 2, 3]).unwrap();

        let events = sink.clone_events();
        assert_eq!(events.len(), 1);
        let KvEventData::Stored(store) = &events[0].data else {
            panic!("expected stored event");
        };
        assert_eq!(store.blocks.len(), 3);

        mgr.free_pages(&req1.lease.pages);
        assert_eq!(sink.event_count(), 1);

        mgr.free_pages(&req2.lease.pages);
        let events = sink.clone_events();
        assert_eq!(events.len(), 2);
        let KvEventData::Removed { block_hashes } = &events[1].data else {
            panic!("expected removed event");
        };
        assert_eq!(block_hashes.len(), 3);
    }

    #[test]
    #[should_panic(expected = "invalid SGLang canonicalization range or radix path")]
    fn invalid_canonical_path_is_fatal() {
        let mut mgr = SglangKvManager::new(8, 4, KvEventPublishers::default(), 0);
        let mut pages = mgr.cache_mut().page_pool.allocate_pages(1).unwrap();
        let root = mgr.cache().root();

        mgr.canonicalize_unfinished_pages(&mut pages, root, 0, 4);
    }

    #[test]
    fn cache_unfinished_rejects_invalid_range_before_publishing() {
        let sink = Arc::new(MockSink::new());
        let mut mgr = SglangKvManager::new(8, 4, KvEventPublishers::new(Some(sink.clone())), 0);
        let tokens = [1, 2, 3, 4];
        let mut alloc = mgr.allocate_for_request(&tokens).unwrap();
        let events_before = sink.event_count();
        let last_node = alloc.lease.last_node();
        let page_hashes = alloc.lease.page_hashes().to_vec();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            mgr.cache_unfinished_hashes(&page_hashes, &mut alloc.lease.pages, last_node, 2)
        }));

        assert!(result.is_err());
        assert_eq!(sink.event_count(), events_before);
    }

    #[test]
    fn cache_unfinished_rejects_short_page_list_before_publishing() {
        let sink = Arc::new(MockSink::new());
        let mut mgr = SglangKvManager::new(8, 4, KvEventPublishers::new(Some(sink.clone())), 0);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut alloc = mgr.allocate_for_request(&tokens).unwrap();
        alloc.lease.pages.truncate(1);
        let events_before = sink.event_count();
        let last_node = alloc.lease.last_node();
        let page_hashes = alloc.lease.page_hashes().to_vec();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            mgr.cache_unfinished_hashes(&page_hashes, &mut alloc.lease.pages, last_node, 0)
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
        let mut mgr = SglangKvManager::new(32, 1, KvEventPublishers::new(Some(sink.clone())), 0);
        let tokens = [11, 22, 33, 44, 55, 66];
        let chunk1_len = 3;
        let chunk2_len = 6;

        let mut alloc1 = mgr.allocate_for_request(&tokens[..chunk1_len]).unwrap();
        mgr.extend_cached_prefix(&tokens[..chunk1_len], &mut alloc1.lease);

        let alloc2 = mgr.allocate_for_request(&tokens[..chunk2_len]).unwrap();
        assert!(mgr.abort(alloc1.lease));

        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 2, "expected two stored events");

        let KvEventData::Stored(store1) = &events[0].data else {
            panic!("expected first event to be Stored");
        };
        let KvEventData::Stored(store2) = &events[1].data else {
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
