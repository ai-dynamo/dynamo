// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Radix-tree KV cache for SGLang engine simulation.
//!
//! Reference: sglang/python/sglang/srt/mem_cache/radix_cache.py

use crate::engine::common::hashing::LocalBlockHash;
#[cfg(test)]
use crate::engine::common::hashing::compute_block_hash_for_seq;
use rustc_hash::FxHashMap;
use slotmap::{SlotMap, new_key_type};
use std::collections::BTreeSet;
use std::time::Instant;

new_key_type! {
    /// Stable identifier for a tree node inside the [`RadixCache`].
    pub struct NodeId;
}

/// Physical page identifier in the simulated SGLang KV pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KvPageId(usize);

impl KvPageId {
    pub(crate) fn index(self) -> usize {
        self.0
    }

    #[cfg(test)]
    pub(crate) fn from_token_index(index: usize, page_size: usize) -> Self {
        Self(index / page_size)
    }

    #[cfg(test)]
    pub(crate) fn first_token_index(self, page_size: usize) -> usize {
        self.0 * page_size
    }
}

/// Manages free / allocated pages for the simulated SGLang KV cache.
///
/// SGLang's paged allocator owns and frees whole pages in production.
pub struct PagePool {
    next_fresh: usize,
    free: Vec<KvPageId>,
    total_pages: usize,
    page_size: usize,
}

impl PagePool {
    pub fn new(total_tokens: usize, page_size: usize) -> Self {
        assert!(page_size >= 1, "page_size must be >= 1");
        Self {
            next_fresh: 0,
            free: Vec::new(),
            total_pages: total_tokens / page_size,
            page_size,
        }
    }

    pub fn allocate_pages(&mut self, count: usize) -> Option<Vec<KvPageId>> {
        if self.available_pages() < count {
            return None;
        }

        let recycled = count.min(self.free.len());
        let fresh = count - recycled;
        let mut pages = Vec::with_capacity(count);
        pages.extend(self.free.drain(self.free.len() - recycled..));
        pages.extend((self.next_fresh..self.next_fresh + fresh).map(KvPageId));
        self.next_fresh += fresh;
        Some(pages)
    }

    #[cfg(test)]
    pub fn allocate(&mut self, token_count: usize) -> Option<Vec<usize>> {
        let mut indices = Vec::new();
        self.allocate_indices_into(token_count, &mut indices)
            .then_some(indices)
    }

    /// Append flattened indices for `new_tokens`, allocating whole pages only
    /// when the request's current final page has no remaining slots.
    #[cfg(test)]
    pub fn allocate_indices_into(&mut self, new_tokens: usize, indices: &mut Vec<usize>) -> bool {
        if new_tokens == 0 {
            return true;
        }

        let available_in_last_page = indices
            .last()
            .map_or(0, |last| self.page_size - 1 - (last % self.page_size));
        let tokens_requiring_pages = new_tokens.saturating_sub(available_in_last_page);
        let required_pages = tokens_requiring_pages.div_ceil(self.page_size);
        if self.available_pages() < required_pages {
            return false;
        }

        indices.reserve(new_tokens);
        let from_existing = new_tokens.min(available_in_last_page);
        if from_existing > 0 {
            let start = indices.last().copied().expect("last page must exist") + 1;
            indices.extend(start..start + from_existing);
        }

        let remaining = new_tokens - from_existing;
        let Some(pages) = self.allocate_pages(required_pages) else {
            return false;
        };
        for (page_idx, page) in pages.into_iter().enumerate() {
            let take = remaining
                .saturating_sub(page_idx * self.page_size)
                .min(self.page_size);
            let start = page.first_token_index(self.page_size);
            indices.extend(start..start + take);
        }
        true
    }

    pub fn free_pages(&mut self, pages: &[KvPageId]) {
        self.free.extend_from_slice(pages);
    }

    /// Free every distinct page represented by a contiguous request-index list.
    #[cfg(test)]
    pub fn free_indices(&mut self, indices: &[usize]) -> Vec<KvPageId> {
        let mut pages = Vec::with_capacity(indices.len().div_ceil(self.page_size));
        for &index in indices {
            let page = KvPageId::from_token_index(index, self.page_size);
            if pages.last().copied() != Some(page) {
                pages.push(page);
            }
        }
        self.free_pages(&pages);
        pages
    }

    #[cfg(test)]
    pub fn free(&mut self, indices: &[usize]) {
        self.free_indices(indices);
    }

    pub fn available_pages(&self) -> usize {
        self.free.len() + self.total_pages - self.next_fresh
    }

    pub fn available(&self) -> usize {
        self.available_pages() * self.page_size
    }

    pub fn total(&self) -> usize {
        self.total_pages * self.page_size
    }
}

/// A single node in the radix tree.
pub struct TreeNode {
    /// Children keyed by the first complete page on the child edge.
    pub children: FxHashMap<LocalBlockHash, NodeId>,
    pub parent: Option<NodeId>,
    /// One content identity per complete page stored on this compressed edge.
    ///
    /// The mocker intentionally uses the router's 64-bit local block hash as
    /// page identity so completed radix state does not retain token IDs.
    /// Consequently, as in router-side indexing, hash collisions are treated
    /// as identical pages rather than guarded by an exact-token comparison.
    pub key: Vec<LocalBlockHash>,
    /// One physical page ID per key. Length = `key.len()`.
    pub value: Vec<KvPageId>,
    /// Walk-to-root reference count (protected when > 0).
    pub lock_ref: usize,
    /// Monotonic timestamp for LRU eviction.
    pub last_access_time: Instant,
}

/// Radix tree for SGLang KV cache simulation.
pub struct RadixCache {
    nodes: SlotMap<NodeId, TreeNode>,
    root: NodeId,
    pub page_pool: PagePool,
    page_size: usize,
    #[cfg(test)]
    test_now: Option<Instant>,
    /// Unlocked leaves ordered by their last access time for O(log n) updates
    /// and O(1) oldest-victim lookup.
    evictable_leaves: BTreeSet<(Instant, NodeId)>,
    /// Total token count in evictable nodes.
    pub evictable_size: usize,
    /// Total token count in protected (locked) nodes.
    pub protected_size: usize,
}

impl RadixCache {
    pub fn new(total_tokens: usize, page_size: usize) -> Self {
        assert!(page_size >= 1, "page_size must be >= 1");
        let mut nodes = SlotMap::with_key();
        let root = nodes.insert(TreeNode {
            children: FxHashMap::default(),
            parent: None,
            key: Vec::new(),
            value: Vec::new(),
            lock_ref: 0,
            last_access_time: Instant::now(),
        });
        Self {
            nodes,
            root,
            page_pool: PagePool::new(total_tokens, page_size),
            page_size,
            #[cfg(test)]
            test_now: None,
            evictable_leaves: BTreeSet::new(),
            evictable_size: 0,
            protected_size: 0,
        }
    }

    pub fn root(&self) -> NodeId {
        self.root
    }
    pub fn node(&self, id: NodeId) -> &TreeNode {
        &self.nodes[id]
    }
    pub fn page_size(&self) -> usize {
        self.page_size
    }
    #[cfg(test)]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    fn now(&self) -> Instant {
        #[cfg(test)]
        if let Some(now) = self.test_now {
            return now;
        }
        Instant::now()
    }

    #[cfg(test)]
    fn set_test_now(&mut self, now: Instant) {
        self.test_now = Some(now);
    }

    fn evictable_key(&self, id: NodeId) -> (Instant, NodeId) {
        (self.nodes[id].last_access_time, id)
    }

    fn is_evictable_leaf(&self, id: NodeId) -> bool {
        id != self.root && self.nodes[id].lock_ref == 0 && self.is_leaf(id)
    }

    fn insert_evictable_leaf(&mut self, id: NodeId) {
        debug_assert!(
            self.is_evictable_leaf(id),
            "only unlocked non-root leaves are directly evictable"
        );
        let inserted = self.evictable_leaves.insert(self.evictable_key(id));
        debug_assert!(inserted, "evictable leaf was already indexed");
    }

    fn remove_evictable_leaf(&mut self, id: NodeId) -> bool {
        if !self.is_evictable_leaf(id) {
            return false;
        }
        self.evictable_leaves.remove(&self.evictable_key(id))
    }

    fn touch_node(&mut self, id: NodeId, now: Instant) {
        let was_indexed = self.remove_evictable_leaf(id);
        self.nodes[id].last_access_time = now;
        if was_indexed {
            self.insert_evictable_leaf(id);
        }
    }

    #[cfg(test)]
    fn debug_assert_evictable_index(&self) {
        let expected = self
            .nodes
            .iter()
            .filter(|(id, node)| *id != self.root && node.lock_ref == 0 && node.children.is_empty())
            .map(|(id, node)| (node.last_access_time, id))
            .collect::<BTreeSet<_>>();
        assert_eq!(
            self.evictable_leaves, expected,
            "SGLang evictable-leaf index drifted from radix nodes"
        );
    }

    #[cfg(test)]
    fn evictable_leaf_is_indexed(&self, id: NodeId) -> bool {
        self.evictable_leaves.contains(&self.evictable_key(id))
    }

    #[cfg(test)]
    pub(crate) fn page_hashes(&self, tokens: &[u32]) -> Vec<LocalBlockHash> {
        compute_block_hash_for_seq(tokens, self.page_size)
    }

    #[cfg(test)]
    fn page_ids(&self, indices: &[usize], page_count: usize) -> Vec<KvPageId> {
        assert!(
            indices.len() >= page_count * self.page_size,
            "not enough token indices for {page_count} complete pages"
        );
        indices
            .chunks_exact(self.page_size)
            .take(page_count)
            .map(|chunk| {
                let page = KvPageId::from_token_index(chunk[0], self.page_size);
                let start = page.first_token_index(self.page_size);
                assert!(
                    chunk.iter().copied().eq(start..start + self.page_size),
                    "SGLang cached pages must contain contiguous page-aligned indices"
                );
                page
            })
            .collect()
    }

    fn key_match(key0: &[LocalBlockHash], key1: &[LocalBlockHash]) -> usize {
        key0.iter().zip(key1).take_while(|(a, b)| a == b).count()
    }

    #[cfg(test)]
    pub fn match_prefix(&mut self, key: &[u32]) -> (usize, NodeId) {
        let page_keys = self.page_hashes(key);
        self.match_prefix_hashes(&page_keys)
    }

    /// Match a prefix using page hashes already owned by the request.
    #[cfg(test)]
    pub(crate) fn match_prefix_hashes(&mut self, page_keys: &[LocalBlockHash]) -> (usize, NodeId) {
        self.match_prefix_hashes_impl(page_keys, false)
    }

    /// Match and protect a prefix using page hashes already owned by the request.
    pub(crate) fn match_prefix_hashes_and_lock(
        &mut self,
        page_keys: &[LocalBlockHash],
    ) -> (usize, NodeId) {
        self.match_prefix_hashes_impl(page_keys, true)
    }

    fn match_prefix_hashes_impl(
        &mut self,
        page_keys: &[LocalBlockHash],
        lock_match: bool,
    ) -> (usize, NodeId) {
        let now = self.now();
        self.nodes[self.root].last_access_time = now;

        let mut current = self.root;
        let mut matched_pages: usize = 0;
        // Delay the most recently matched node's touch until we know whether it
        // is the terminal match. The match-and-lock path can then remove an
        // unlocked terminal leaf once without reinserting it before locking.
        let mut pending_touch = None;

        while matched_pages < page_keys.len() {
            let child_id = match self.nodes[current]
                .children
                .get(&page_keys[matched_pages])
                .copied()
            {
                Some(id) => id,
                None => break,
            };

            if let Some(id) = pending_touch.take() {
                self.touch_node(id, now);
            }

            let (common_len, child_len) = {
                let child_key = &self.nodes[child_id].key;
                (
                    Self::key_match(child_key, &page_keys[matched_pages..]),
                    child_key.len(),
                )
            };

            if common_len < child_len {
                if common_len > 0 {
                    let intermediate = self.split_node(child_id, common_len);
                    current = intermediate;
                }
                matched_pages += common_len;
                break;
            }

            matched_pages += common_len;
            current = child_id;
            pending_touch = Some(current);
        }

        if let Some(id) = pending_touch {
            if lock_match {
                self.touch_and_inc_lock_ref(id, now);
            } else {
                self.touch_node(id, now);
            }
        } else if lock_match {
            // A partial edge match keeps the split node's inherited timestamp.
            self.inc_lock_ref(current);
        }

        #[cfg(test)]
        self.debug_assert_evictable_index();
        (matched_pages * self.page_size, current)
    }

    /// Read-only prefix match length (does not mutate timestamps or split nodes).
    /// Used for LPM scheduling scoring.
    #[cfg(test)]
    pub fn prefix_match_len(&self, key: &[u32]) -> usize {
        let page_keys = self.page_hashes(key);
        self.prefix_match_hashes_len(&page_keys)
    }

    /// Read-only prefix match using page hashes already owned by the request.
    pub(crate) fn prefix_match_hashes_len(&self, page_keys: &[LocalBlockHash]) -> usize {
        let mut current = self.root;
        let mut matched_pages: usize = 0;

        while matched_pages < page_keys.len() {
            let child_id = match self.nodes[current]
                .children
                .get(&page_keys[matched_pages])
                .copied()
            {
                Some(id) => id,
                None => break,
            };

            let child_key = &self.nodes[child_id].key;
            let common_len = Self::key_match(child_key, &page_keys[matched_pages..]);

            if common_len < child_key.len() {
                matched_pages += common_len;
                break;
            }

            matched_pages += common_len;
            current = child_id;
        }

        matched_pages * self.page_size
    }

    /// Insert a token sequence into the tree. Key is page-aligned before insertion.
    #[cfg(test)]
    pub fn insert(&mut self, key: &[u32], value: &[usize]) -> NodeId {
        let aligned_len = key.len() / self.page_size * self.page_size;
        assert!(
            value.len() >= aligned_len,
            "not enough token indices: need {aligned_len}, got {}",
            value.len()
        );
        let page_ids = self.page_ids(&value[..aligned_len], aligned_len / self.page_size);
        let inserted = self.insert_page_suffix(self.root, 0, key, &page_ids, false);
        #[cfg(test)]
        self.debug_assert_evictable_index();
        inserted
    }

    /// Insert only the suffix after a retained, page-aligned prefix.
    ///
    /// `prefix_node` must be the locked terminal node for `prefix_len`. Keeping
    /// that handle lets decode growth avoid walking the full sequence from the
    /// root on every completed page.
    #[cfg(test)]
    pub fn insert_from_node(
        &mut self,
        prefix_node: NodeId,
        prefix_len: usize,
        key: &[u32],
        value: &[usize],
    ) -> NodeId {
        let aligned_len = key.len() / self.page_size * self.page_size;
        assert_eq!(
            prefix_len % self.page_size,
            0,
            "prefix length must be page-aligned"
        );
        assert!(
            prefix_len <= aligned_len,
            "prefix length {prefix_len} exceeds aligned key length {aligned_len}"
        );
        assert!(
            value.len() >= aligned_len,
            "not enough token indices: need {aligned_len}, got {}",
            value.len()
        );
        let page_ids = self.page_ids(
            &value[prefix_len..aligned_len],
            (aligned_len - prefix_len) / self.page_size,
        );
        let inserted = self.insert_page_suffix(prefix_node, prefix_len, key, &page_ids, true);
        #[cfg(test)]
        self.debug_assert_evictable_index();
        inserted
    }

    /// Insert page identities already materialized by the request lease.
    pub(crate) fn insert_page_hashes_from_node(
        &mut self,
        prefix_node: NodeId,
        prefix_len: usize,
        page_keys: &[LocalBlockHash],
        pages: &[KvPageId],
    ) -> NodeId {
        assert_eq!(
            prefix_len % self.page_size,
            0,
            "prefix length must be page-aligned"
        );
        assert!(
            pages.len() >= page_keys.len(),
            "not enough KV pages: need {}, got {}",
            page_keys.len(),
            pages.len()
        );
        let prefix_pages = prefix_len / self.page_size;
        assert!(
            prefix_pages <= page_keys.len(),
            "prefix pages {prefix_pages} exceed hashed pages {}",
            page_keys.len()
        );
        let inserted = self.insert_page_hash_suffix(
            prefix_node,
            &page_keys[prefix_pages..],
            &pages[prefix_pages..page_keys.len()],
            true,
        );
        #[cfg(test)]
        self.debug_assert_evictable_index();
        inserted
    }

    #[cfg(test)]
    fn insert_page_suffix(
        &mut self,
        start_node: NodeId,
        prefix_len: usize,
        key: &[u32],
        page_ids: &[KvPageId],
        allow_locked_tail_extension: bool,
    ) -> NodeId {
        let aligned_len = key.len() / self.page_size * self.page_size;
        assert_eq!(
            prefix_len % self.page_size,
            0,
            "prefix length must be page-aligned"
        );
        assert!(
            prefix_len <= aligned_len,
            "prefix length {prefix_len} exceeds aligned key length {aligned_len}"
        );
        if aligned_len == prefix_len {
            return start_node;
        }
        let page_keys = self.page_hashes(&key[prefix_len..aligned_len]);
        self.insert_page_hash_suffix(
            start_node,
            &page_keys,
            page_ids,
            allow_locked_tail_extension,
        )
    }

    fn insert_page_hash_suffix(
        &mut self,
        start_node: NodeId,
        page_keys: &[LocalBlockHash],
        page_ids: &[KvPageId],
        allow_locked_tail_extension: bool,
    ) -> NodeId {
        assert!(
            page_ids.len() >= page_keys.len(),
            "not enough KV pages: need {}, got {}",
            page_keys.len(),
            page_ids.len()
        );
        if page_keys.is_empty() {
            return start_node;
        }
        let now = self.now();
        self.touch_path(start_node, now);

        let mut current = start_node;
        let mut key_offset = 0;

        while key_offset < page_keys.len() {
            let can_extend_leaf = current != self.root
                && self.nodes[current].children.is_empty()
                && (self.nodes[current].lock_ref == 0
                    || (allow_locked_tail_extension
                        && current == start_node
                        && self.nodes[current].lock_ref == 1));
            if can_extend_leaf {
                return self.extend_leaf(
                    current,
                    &page_keys[key_offset..],
                    &page_ids[key_offset..],
                    now,
                );
            }

            let child_id = match self.nodes[current]
                .children
                .get(&page_keys[key_offset])
                .copied()
            {
                Some(id) => id,
                None => {
                    return self.create_child(
                        current,
                        &page_keys[key_offset..],
                        &page_ids[key_offset..],
                    );
                }
            };

            let (common_len, child_len) = {
                let child_key = &self.nodes[child_id].key;
                (
                    Self::key_match(child_key, &page_keys[key_offset..]),
                    child_key.len(),
                )
            };

            if common_len == child_len {
                key_offset += common_len;
                current = child_id;
                self.touch_node(current, now);
            } else {
                if common_len > 0 {
                    let intermediate = self.split_node(child_id, common_len);
                    key_offset += common_len;
                    if key_offset < page_keys.len() {
                        return self.create_child(
                            intermediate,
                            &page_keys[key_offset..],
                            &page_ids[key_offset..],
                        );
                    }
                    return intermediate;
                }
                return current;
            }
        }

        current
    }

    fn touch_path(&mut self, node_id: NodeId, now: Instant) {
        let mut current = Some(node_id);
        while let Some(id) = current {
            let parent = self.nodes[id].parent;
            self.touch_node(id, now);
            current = parent;
        }
    }

    fn extend_leaf(
        &mut self,
        node_id: NodeId,
        key: &[LocalBlockHash],
        value: &[KvPageId],
        now: Instant,
    ) -> NodeId {
        self.touch_node(node_id, now);
        let node = &mut self.nodes[node_id];
        debug_assert!(node.children.is_empty());
        debug_assert!(node.lock_ref <= 1);
        node.key.extend_from_slice(key);
        node.value.extend_from_slice(value);
        if node.lock_ref == 0 {
            self.evictable_size += key.len() * self.page_size;
        } else {
            self.protected_size += key.len() * self.page_size;
        }
        node_id
    }

    fn split_node(&mut self, child_id: NodeId, split_pos: usize) -> NodeId {
        // Keep child_id as the suffix node so its last-access timestamp and
        // evictable-index key remain valid across the split.
        let (child_parent, original_ck, prefix_key, prefix_value, suffix_ck, lock_ref, accessed) = {
            let child = &mut self.nodes[child_id];
            let child_parent = child.parent;
            let original_ck = child.key[0];
            let suffix_key = child.key.split_off(split_pos);
            let prefix_key = std::mem::replace(&mut child.key, suffix_key);
            let suffix_value = child.value.split_off(split_pos);
            let prefix_value = std::mem::replace(&mut child.value, suffix_value);
            let suffix_ck = child.key[0];
            (
                child_parent,
                original_ck,
                prefix_key,
                prefix_value,
                suffix_ck,
                child.lock_ref,
                child.last_access_time,
            )
        };

        let mut inter_children = FxHashMap::default();
        inter_children.insert(suffix_ck, child_id);

        let intermediate = TreeNode {
            children: inter_children,
            parent: child_parent,
            key: prefix_key,
            value: prefix_value,
            lock_ref,
            last_access_time: accessed,
        };
        let inter_id = self.nodes.insert(intermediate);

        let child = &mut self.nodes[child_id];
        child.parent = Some(inter_id);

        if let Some(parent_id) = child_parent {
            self.nodes[parent_id].children.insert(original_ck, inter_id);
        }

        // Both size totals are unchanged: the intermediate and suffix split
        // the original edge without changing its lock state or token count.

        inter_id
    }

    fn create_child(
        &mut self,
        parent_id: NodeId,
        key: &[LocalBlockHash],
        value: &[KvPageId],
    ) -> NodeId {
        let now = self.now();
        let new_node = TreeNode {
            children: FxHashMap::default(),
            parent: Some(parent_id),
            key: key.to_vec(),
            value: value.to_vec(),
            lock_ref: 0,
            last_access_time: now,
        };
        let ck = key[0];
        let new_id = self.nodes.insert(new_node);

        self.remove_evictable_leaf(parent_id);

        self.nodes[parent_id].children.insert(ck, new_id);

        self.insert_evictable_leaf(new_id);
        self.evictable_size += key.len() * self.page_size;

        new_id
    }

    pub fn is_leaf(&self, id: NodeId) -> bool {
        self.nodes[id].children.is_empty()
    }

    fn touch_and_inc_lock_ref(&mut self, node_id: NodeId, now: Instant) {
        let terminal_unindexed = self.remove_evictable_leaf(node_id);
        self.nodes[node_id].last_access_time = now;
        self.inc_lock_ref_impl(node_id, terminal_unindexed);
    }

    pub fn inc_lock_ref(&mut self, node_id: NodeId) {
        self.inc_lock_ref_impl(node_id, false);
    }

    fn inc_lock_ref_impl(&mut self, node_id: NodeId, terminal_unindexed: bool) {
        let mut current = Some(node_id);
        let mut is_terminal = true;
        while let Some(id) = current {
            if id == self.root {
                break;
            }
            let was_unlocked = self.nodes[id].lock_ref == 0;
            if was_unlocked && !(is_terminal && terminal_unindexed) {
                self.remove_evictable_leaf(id);
            }
            let node = &mut self.nodes[id];
            let tokens = node.key.len() * self.page_size;
            node.lock_ref += 1;
            if was_unlocked {
                self.evictable_size -= tokens;
                self.protected_size += tokens;
            }
            current = self.nodes[id].parent;
            is_terminal = false;
        }
        #[cfg(test)]
        self.debug_assert_evictable_index();
    }

    pub fn dec_lock_ref(&mut self, node_id: NodeId) {
        let mut current = Some(node_id);
        while let Some(id) = current {
            if id == self.root {
                break;
            }
            let node = &mut self.nodes[id];
            if node.lock_ref == 0 {
                tracing::warn!("dec_lock_ref on node with lock_ref == 0, skipping");
                break;
            }
            node.lock_ref -= 1;
            if node.lock_ref == 0 {
                let tokens = node.key.len() * self.page_size;
                self.protected_size -= tokens;
                self.evictable_size += tokens;
                if self.is_leaf(id) {
                    self.insert_evictable_leaf(id);
                }
            }
            current = self.nodes[id].parent;
        }
        #[cfg(test)]
        self.debug_assert_evictable_index();
    }

    /// Evict tokens from the cache by LRU order, rounding partial leaves to full pages.
    /// Returns `(num_tokens_evicted, evicted_page_ids)`.
    pub fn evict(&mut self, num_tokens: usize) -> (usize, Vec<KvPageId>) {
        let mut evicted = 0;
        let mut evicted_indices =
            Vec::with_capacity(num_tokens.min(self.evictable_size).div_ceil(self.page_size));
        while evicted < num_tokens {
            let Some((last_access_time, victim_id)) = self.evictable_leaves.pop_first() else {
                break;
            };
            debug_assert_eq!(
                last_access_time, self.nodes[victim_id].last_access_time,
                "eviction index timestamp drifted from radix node"
            );

            let victim_pages = self.nodes[victim_id].key.len();
            let victim_tokens = victim_pages * self.page_size;
            let remaining = num_tokens - evicted;
            let eviction_pages = remaining.div_ceil(self.page_size).min(victim_pages);
            let eviction_len = eviction_pages * self.page_size;

            // A compressed leaf may span pages. Preserve its indexed prefix when
            // only the newest suffix pages are needed to satisfy this eviction.
            if eviction_len < victim_tokens {
                let split_pos = victim_pages - eviction_pages;
                let (nodes, page_pool) = (&mut self.nodes, &mut self.page_pool);
                let victim_node = &mut nodes[victim_id];
                victim_node.key.truncate(split_pos);
                let evicted_values = &victim_node.value[split_pos..];
                page_pool.free_pages(evicted_values);
                evicted_indices.extend_from_slice(evicted_values);
                victim_node.value.truncate(split_pos);

                self.evictable_size -= eviction_len;
                evicted += eviction_len;
                self.insert_evictable_leaf(victim_id);
                continue;
            }

            let victim_node = self
                .nodes
                .remove(victim_id)
                .expect("evictable leaf disappeared before removal");
            let tokens = victim_node.key.len() * self.page_size;
            let parent_id = victim_node.parent;

            self.evictable_size -= tokens;
            evicted += tokens;

            evicted_indices.extend_from_slice(&victim_node.value);
            self.page_pool.free_pages(&victim_node.value);

            if let Some(pid) = parent_id {
                self.nodes[pid].children.remove(&victim_node.key[0]);

                if pid != self.root
                    && self.nodes[pid].children.is_empty()
                    && self.nodes[pid].lock_ref == 0
                {
                    self.insert_evictable_leaf(pid);
                }
            }
        }
        #[cfg(test)]
        self.debug_assert_evictable_index();
        (evicted, evicted_indices)
    }

    pub fn available_tokens(&self) -> usize {
        self.page_pool.available()
    }

    pub fn total_tokens(&self) -> usize {
        self.page_pool.total()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_pool_allocate_extend_and_free() {
        let mut pool = PagePool::new(12, 4);
        assert_eq!(pool.available(), 12);
        assert!(pool.allocate(usize::MAX).is_none());
        let a = pool.allocate(3).unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(pool.available(), 8);
        let mut extended = a.clone();
        assert!(pool.allocate_indices_into(1, &mut extended));
        assert_eq!(extended, vec![0, 1, 2, 3]);
        assert_eq!(pool.available(), 8);
        let b = pool.allocate(5).unwrap();
        assert_eq!(pool.available(), 0);
        assert!(pool.allocate(1).is_none());
        pool.free(&a);
        assert_eq!(pool.available(), 4);
        pool.free(&b);
        assert_eq!(pool.available(), 12);
    }

    #[test]
    fn test_allocate_indices_into_failure_is_atomic() {
        let mut pool = PagePool::new(8, 4);
        let mut destination = pool.allocate(4).unwrap();
        let _other = pool.allocate(4).unwrap();
        let available_before = pool.available();
        let destination_before = destination.clone();

        assert!(!pool.allocate_indices_into(1, &mut destination));
        assert_eq!(destination, destination_before);
        assert_eq!(pool.available(), available_before);
    }

    #[test]
    fn test_match_prefix() {
        let mut cache = RadixCache::new(100, 1);

        // Empty tree
        let (len, node) = cache.match_prefix(&[1, 2, 3]);
        assert_eq!(len, 0);
        assert_eq!(node, cache.root());

        // Full match
        cache.insert(&[1, 2, 3, 4, 5], &[10, 20, 30, 40, 50]);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5]).0, 5);

        // Partial match with split
        cache.insert(&[1, 2, 3, 4, 5, 6, 7], &[10, 20, 30, 40, 50, 60, 70]);
        let (len, node) = cache.match_prefix(&[1, 2, 3, 4, 5, 9, 9]);
        assert_eq!(len, 5);
        let n = cache.node(node);
        assert_eq!(n.key, cache.page_hashes(&[1, 2, 3, 4, 5]));
        assert_eq!(
            n.value,
            vec![
                KvPageId(10),
                KvPageId(20),
                KvPageId(30),
                KvPageId(40),
                KvPageId(50)
            ]
        );
        let suffix_key = cache.page_hashes(&[6])[0];
        let &suffix_id = n.children.get(&suffix_key).unwrap();
        assert_eq!(
            cache.node(suffix_id).value,
            vec![KvPageId(60), KvPageId(70)]
        );
    }

    #[test]
    fn test_insert() {
        let mut cache = RadixCache::new(100, 1);

        // Shared prefix splits the tree
        cache.insert(&[1, 2, 3, 4, 5], &[10, 20, 30, 40, 50]);
        cache.insert(&[1, 2, 3, 6, 7], &[10, 20, 30, 60, 70]);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5]).0, 5);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 6, 7]).0, 5);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 9]).0, 3);

        // Extend existing prefix
        let mut cache = RadixCache::new(100, 1);
        cache.insert(&[1, 2, 3], &[10, 20, 30]);
        cache.insert(&[1, 2, 3, 4, 5], &[10, 20, 30, 40, 50]);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5]).0, 5);

        // Duplicate insert is idempotent
        cache.insert(&[1, 2, 3], &[10, 20, 30]);

        // Match then insert suffix
        let mut cache = RadixCache::new(100, 1);
        cache.insert(&[1, 2, 3, 4, 5], &[10, 20, 30, 40, 50]);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]).0, 5);
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &[10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]).0, 8);
    }

    #[test]
    fn test_retained_tail_extends_unique_leaf_in_place() {
        let mut cache = RadixCache::new(100, 4);
        cache.insert(&[1, 2, 3, 4], &[0, 1, 2, 3]);
        let (_, tail) = cache.match_prefix(&[1, 2, 3, 4]);
        cache.inc_lock_ref(tail);
        let nodes_before = cache.num_nodes();

        let extended = cache.insert_from_node(
            tail,
            4,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[0, 1, 2, 3, 4, 5, 6, 7],
        );

        assert_eq!(extended, tail);
        assert_eq!(cache.num_nodes(), nodes_before);
        assert_eq!(
            cache.node(tail).key,
            cache.page_hashes(&[1, 2, 3, 4, 5, 6, 7, 8])
        );
        assert_eq!(cache.protected_size, 8);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]).0, 8);
    }

    #[test]
    fn test_retained_tail_does_not_extend_shared_leaf_in_place() {
        let mut cache = RadixCache::new(100, 4);
        cache.insert(&[1, 2, 3, 4], &[0, 1, 2, 3]);
        let (_, tail) = cache.match_prefix(&[1, 2, 3, 4]);
        cache.inc_lock_ref(tail);
        cache.inc_lock_ref(tail);
        let nodes_before = cache.num_nodes();

        let extended = cache.insert_from_node(
            tail,
            4,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[0, 1, 2, 3, 4, 5, 6, 7],
        );

        assert_ne!(extended, tail);
        assert_eq!(cache.num_nodes(), nodes_before + 1);
        assert_eq!(cache.node(tail).key, cache.page_hashes(&[1, 2, 3, 4]));
        assert_eq!(cache.node(extended).key, cache.page_hashes(&[5, 6, 7, 8]));
    }

    #[test]
    fn test_page_size() {
        // Insert and match with page_size=4
        let mut cache = RadixCache::new(100, 4);
        assert_eq!(cache.page_pool.total(), 100);
        cache.insert(&[1, 2, 3, 4, 5, 6, 7], &[0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4]).0, 4);
        let (_, node) = cache.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(cache.node(node).value, vec![KvPageId(0)]);

        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]).0, 8);

        // Children disambiguated by first page_size tokens
        let mut cache = RadixCache::new(100, 4);
        cache.insert(&[1, 2, 3, 4], &[0, 1, 2, 3]);
        cache.insert(&[1, 2, 3, 5], &[4, 5, 6, 7]);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4]).0, 4);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 5]).0, 4);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 6]).0, 0);

        // Split at page boundary preserves value
        let mut cache = RadixCache::new(100, 4);
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &[0, 1, 2, 3, 4, 5, 6, 7]);
        cache.match_prefix(&[1, 2, 3, 4, 9, 9, 9, 9]);
        let (_, node) = cache.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(cache.node(node).value, vec![KvPageId(0)]);
    }

    #[test]
    fn completed_edges_store_one_key_and_page_id_per_page() {
        let mut cache = RadixCache::new(256, 64);
        let tokens = (0..128).collect::<Vec<u32>>();
        let indices = cache.page_pool.allocate(tokens.len()).unwrap();
        let node = cache.insert(&tokens, &indices);

        assert_eq!(cache.node(node).key.len(), 2);
        assert_eq!(cache.node(node).value.len(), 2);
        assert_eq!(cache.node(node).value, vec![KvPageId(0), KvPageId(1)]);
        assert_eq!(cache.match_prefix(&tokens).0, tokens.len());
    }

    #[test]
    fn test_lock_unlock_shared_prefix() {
        let mut cache = RadixCache::new(100, 1);
        cache.insert(&[1, 2, 3, 4, 5], &[0, 1, 2, 3, 4]);
        cache.insert(&[1, 2, 3, 6, 7], &[0, 1, 2, 5, 6]);

        let (_, node_a) = cache.match_prefix(&[1, 2, 3, 4, 5]);
        let (_, node_b) = cache.match_prefix(&[1, 2, 3, 6, 7]);

        cache.inc_lock_ref(node_a);
        cache.inc_lock_ref(node_b);
        assert_eq!(cache.protected_size, 7); // 2+2+3

        cache.dec_lock_ref(node_a);
        assert!(cache.evictable_leaf_is_indexed(node_a));
        cache.dec_lock_ref(node_b);
        assert_eq!(cache.protected_size, 0);
    }

    #[test]
    fn test_evict() {
        // LRU order: oldest evicted first
        let mut cache = RadixCache::new(100, 1);
        let base = Instant::now();
        cache.set_test_now(base);
        cache.insert(&[1, 2, 3], &[0, 1, 2]);
        let (_, n1) = cache.match_prefix(&[1, 2, 3]);
        cache.inc_lock_ref(n1);
        cache.dec_lock_ref(n1);

        cache.set_test_now(base + std::time::Duration::from_secs(1));
        cache.insert(&[4, 5, 6], &[3, 4, 5]);
        let (_, n2) = cache.match_prefix(&[4, 5, 6]);
        cache.inc_lock_ref(n2);
        cache.dec_lock_ref(n2);

        let (evicted_count, evicted_indices) = cache.evict(3);
        assert_eq!(evicted_count, 3);
        // Evicted indices should match the pool indices originally inserted for [1,2,3]
        let mut sorted_evicted = evicted_indices.clone();
        sorted_evicted.sort();
        let mut expected_indices = vec![KvPageId(0), KvPageId(1), KvPageId(2)];
        expected_indices.sort();
        assert_eq!(
            sorted_evicted, expected_indices,
            "evicted indices should match inserted indices"
        );
        assert_eq!(cache.match_prefix(&[1, 2, 3]).0, 0); // oldest evicted
        assert_eq!(cache.match_prefix(&[4, 5, 6]).0, 3); // newer kept

        // Locked nodes are not evicted
        let mut cache = RadixCache::new(100, 1);
        cache.insert(&[1, 2, 3], &[0, 1, 2]);
        cache.insert(&[4, 5, 6], &[3, 4, 5]);
        let (_, locked) = cache.match_prefix(&[1, 2, 3]);
        cache.inc_lock_ref(locked);
        let (_, unlocked) = cache.match_prefix(&[4, 5, 6]);
        cache.inc_lock_ref(unlocked);
        cache.dec_lock_ref(unlocked);
        let (evicted_count, evicted_indices) = cache.evict(6);
        assert_eq!(evicted_count, 3); // only unlocked evicted
        let mut sorted_evicted = evicted_indices;
        sorted_evicted.sort();
        assert_eq!(
            sorted_evicted,
            vec![KvPageId(3), KvPageId(4), KvPageId(5)],
            "should evict unlocked [4,5,6] indices"
        );
        assert_eq!(cache.match_prefix(&[1, 2, 3]).0, 3);
    }

    #[test]
    fn touching_an_evictable_leaf_updates_order() {
        let mut cache = RadixCache::new(100, 1);
        let base = Instant::now();
        cache.set_test_now(base);
        cache.insert(&[1, 2, 3], &[0, 1, 2]);
        cache.set_test_now(base + std::time::Duration::from_secs(1));
        cache.insert(&[4, 5, 6], &[3, 4, 5]);
        cache.set_test_now(base + std::time::Duration::from_secs(2));
        let (_, extended) = cache.match_prefix(&[1, 2, 3]);
        cache.insert_from_node(extended, 3, &[1, 2, 3, 7], &[0, 1, 2, 6]);

        assert_eq!(cache.evict(3).0, 3);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 7]).0, 4);
        assert_eq!(cache.match_prefix(&[4, 5, 6]).0, 0);
    }

    #[test]
    fn unlocking_preserves_last_access_order() {
        let mut cache = RadixCache::new(100, 1);
        let base = Instant::now();
        cache.set_test_now(base);
        cache.insert(&[1, 2, 3], &[0, 1, 2]);
        let older_hashes = cache.page_hashes(&[1, 2, 3]);
        let (_, older) = cache.match_prefix_hashes_and_lock(&older_hashes);

        cache.set_test_now(base + std::time::Duration::from_secs(1));
        cache.insert(&[4, 5, 6], &[3, 4, 5]);
        let newer_hashes = cache.page_hashes(&[4, 5, 6]);
        let (_, newer) = cache.match_prefix_hashes_and_lock(&newer_hashes);
        cache.dec_lock_ref(newer);
        cache.dec_lock_ref(older);

        assert_eq!(cache.evict(3).0, 3);
        assert_eq!(cache.match_prefix(&[1, 2, 3]).0, 0);
        assert_eq!(cache.match_prefix(&[4, 5, 6]).0, 3);
    }

    #[test]
    fn partial_eviction_and_parent_promotion_keep_index_consistent() {
        let mut cache = RadixCache::new(100, 1);
        let base = Instant::now();
        cache.set_test_now(base);
        cache.insert(&[1, 2, 3, 4], &[0, 1, 2, 3]);
        let (_, leaf) = cache.match_prefix(&[1, 2, 3, 4]);
        cache.set_test_now(base + std::time::Duration::from_secs(1));
        cache.insert(&[5, 6, 7], &[4, 5, 6]);

        assert_eq!(cache.evict(2).0, 2);
        assert_eq!(cache.node(leaf).key.len(), 2);
        assert!(cache.evictable_leaf_is_indexed(leaf));
        assert_eq!(cache.evict(2).0, 2);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4]).0, 0);
        assert_eq!(cache.match_prefix(&[5, 6, 7]).0, 3);

        let mut branched = RadixCache::new(100, 1);
        branched.set_test_now(base);
        branched.insert(&[10, 11, 12], &[0, 1, 2]);
        branched.set_test_now(base + std::time::Duration::from_secs(1));
        branched.insert(&[10, 11, 13], &[0, 1, 3]);
        branched.set_test_now(base + std::time::Duration::from_secs(2));
        branched.insert(&[20, 21], &[4, 5]);
        let parent_hash = branched.page_hashes(&[10])[0];
        let parent = *branched.nodes[branched.root]
            .children
            .get(&parent_hash)
            .unwrap();
        assert!(!branched.is_leaf(parent));

        assert_eq!(branched.evict(2).0, 2);
        assert!(branched.is_leaf(parent));
        assert!(branched.evictable_leaf_is_indexed(parent));
        assert_eq!(branched.evict(2).0, 2);
        assert_eq!(branched.match_prefix(&[10, 11, 12]).0, 0);
        assert_eq!(branched.match_prefix(&[20, 21]).0, 2);
    }

    #[test]
    fn test_evictable_size_includes_unlocked_internal_prefix() {
        let mut cache = RadixCache::new(16, 4);
        let first = cache.page_pool.allocate(8).unwrap();
        cache.insert(&[1; 8], &first);
        let mut branch = first[..4].to_vec();
        branch.extend(cache.page_pool.allocate(4).unwrap());
        cache.insert(&[1, 1, 1, 1, 2, 2, 2, 2], &branch);

        assert_eq!(cache.evictable_size, 12);
        assert_eq!(cache.evict(12).0, 12);
        assert_eq!(cache.available_tokens(), 16);
    }

    #[test]
    fn test_query_methods() {
        let cache = RadixCache::new(100, 1);
        assert_eq!(cache.available_tokens(), 100);
        assert_eq!(cache.total_tokens(), 100);

        let cache4 = RadixCache::new(100, 4);
        assert_eq!(cache4.available_tokens(), 100);
        assert_eq!(cache4.total_tokens(), 100);
    }
}
