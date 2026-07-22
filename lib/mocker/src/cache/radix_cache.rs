// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Radix-tree KV cache for SGLang engine simulation.
//!
//! Reference: sglang/python/sglang/srt/mem_cache/radix_cache.py

use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::{SlotMap, new_key_type};
use std::time::Instant;

new_key_type! {
    /// Stable identifier for a tree node inside the [`RadixCache`].
    pub struct NodeId;
}

/// Manages free / allocated token slot indices for the KV cache pool.
pub struct TokenPool {
    next_fresh: usize,
    free: Vec<usize>,
    total: usize,
}

impl TokenPool {
    pub fn new(total: usize) -> Self {
        Self {
            next_fresh: 0,
            free: Vec::new(),
            total,
        }
    }

    /// Allocate `n` token slots. Returns `None` if not enough free slots.
    pub fn allocate(&mut self, n: usize) -> Option<Vec<usize>> {
        if self.available() < n {
            return None;
        }

        let recycled = n.min(self.free.len());
        let fresh = n - recycled;
        let mut indices = Vec::with_capacity(n);
        indices.extend(self.free.drain(self.free.len() - recycled..));
        indices.extend(self.next_fresh..self.next_fresh + fresh);
        self.next_fresh += fresh;
        Some(indices)
    }

    /// Return token slots to the free pool.
    pub fn free(&mut self, indices: &[usize]) {
        self.free.extend(indices);
    }

    pub fn available(&self) -> usize {
        self.free.len() + self.total - self.next_fresh
    }

    pub fn total(&self) -> usize {
        self.total
    }
}

/// A single node in the radix tree.
pub struct TreeNode {
    /// Children keyed by `child.key[..page_size]` (a "child key").
    pub children: FxHashMap<Vec<u64>, NodeId>,
    pub parent: Option<NodeId>,
    /// Token IDs stored at this edge.
    pub key: Vec<u64>,
    /// KV cache pool token indices. Length = `key.len()`.
    pub value: Vec<usize>,
    /// Walk-to-root reference count (protected when > 0).
    pub lock_ref: usize,
    /// Monotonic timestamp for LRU eviction.
    pub last_access_time: Instant,
}

/// Radix tree for SGLang KV cache simulation.
pub struct RadixCache {
    nodes: SlotMap<NodeId, TreeNode>,
    root: NodeId,
    pub token_pool: TokenPool,
    page_size: usize,
    /// Total token count in evictable nodes.
    pub evictable_leaves: FxHashSet<NodeId>,
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
            token_pool: TokenPool::new(total_tokens),
            page_size,
            evictable_leaves: FxHashSet::default(),
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
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    fn child_key<'a>(&self, key: &'a [u64]) -> &'a [u64] {
        &key[..self.page_size.min(key.len())]
    }

    fn page_align(&self, len: usize) -> usize {
        len / self.page_size * self.page_size
    }

    fn key_match(&self, key0: &[u64], key1: &[u64]) -> usize {
        if self.page_size == 1 {
            key0.iter().zip(key1).take_while(|(a, b)| a == b).count()
        } else {
            let min_len = key0.len().min(key1.len());
            let mut i = 0;
            while i + self.page_size <= min_len {
                if key0[i..i + self.page_size] != key1[i..i + self.page_size] {
                    break;
                }
                i += self.page_size;
            }
            i
        }
    }

    pub fn match_prefix(&mut self, key: &[u64]) -> (usize, NodeId) {
        let now = Instant::now();
        self.nodes[self.root].last_access_time = now;

        let mut current = self.root;
        let mut matched: usize = 0;

        while matched < key.len() {
            let ck = self.child_key(&key[matched..]);
            let child_id = match self.nodes[current].children.get(ck).copied() {
                Some(id) => id,
                None => break,
            };

            let (common_len, child_len) = {
                let child_key = &self.nodes[child_id].key;
                (self.key_match(child_key, &key[matched..]), child_key.len())
            };

            if common_len < child_len {
                if common_len > 0 {
                    let intermediate = self.split_node(child_id, common_len);
                    current = intermediate;
                }
                matched += common_len;
                break;
            }

            matched += common_len;
            current = child_id;
            self.nodes[current].last_access_time = now;
        }

        (matched, current)
    }

    /// Read-only prefix match length (does not mutate timestamps or split nodes).
    /// Used for LPM scheduling scoring.
    pub fn prefix_match_len(&self, key: &[u64]) -> usize {
        let mut current = self.root;
        let mut matched: usize = 0;

        while matched < key.len() {
            let ck = self.child_key(&key[matched..]);
            let child_id = match self.nodes[current].children.get(ck).copied() {
                Some(id) => id,
                None => break,
            };

            let child_key = &self.nodes[child_id].key;
            let common_len = self.key_match(child_key, &key[matched..]);

            if common_len < child_key.len() {
                matched += common_len;
                break;
            }

            matched += common_len;
            current = child_id;
        }

        // Round down to page boundary
        matched / self.page_size * self.page_size
    }

    /// Insert a token sequence into the tree. Key is page-aligned before insertion.
    pub fn insert(&mut self, key: &[u64], value: &[usize]) -> NodeId {
        self.insert_from(self.root, 0, key, value, false)
    }

    /// Insert only the suffix after a retained, page-aligned prefix.
    ///
    /// `prefix_node` must be the locked terminal node for `prefix_len`. Keeping
    /// that handle lets decode growth avoid walking the full sequence from the
    /// root on every completed page.
    pub fn insert_from_node(
        &mut self,
        prefix_node: NodeId,
        prefix_len: usize,
        key: &[u64],
        value: &[usize],
    ) -> NodeId {
        self.insert_from(prefix_node, prefix_len, key, value, true)
    }

    fn insert_from(
        &mut self,
        start_node: NodeId,
        prefix_len: usize,
        key: &[u64],
        value: &[usize],
        allow_locked_tail_extension: bool,
    ) -> NodeId {
        let aligned_len = self.page_align(key.len());
        assert_eq!(
            prefix_len,
            self.page_align(prefix_len),
            "prefix length must be page-aligned"
        );
        assert!(
            prefix_len <= aligned_len,
            "prefix length {prefix_len} exceeds aligned key length {aligned_len}"
        );
        if aligned_len == prefix_len {
            return start_node;
        }
        assert!(
            value.len() >= aligned_len,
            "not enough token indices: need {aligned_len}, got {}",
            value.len()
        );
        let key = &key[..aligned_len];
        let value = &value[..aligned_len];

        let now = Instant::now();
        self.touch_path(start_node, now);

        let mut current = start_node;
        let mut key_offset = prefix_len;

        while key_offset < key.len() {
            let can_extend_leaf = current != self.root
                && self.nodes[current].children.is_empty()
                && (self.nodes[current].lock_ref == 0
                    || (allow_locked_tail_extension
                        && current == start_node
                        && self.nodes[current].lock_ref == 1));
            if can_extend_leaf {
                return self.extend_leaf(current, &key[key_offset..], &value[key_offset..], now);
            }

            let ck = self.child_key(&key[key_offset..]);
            let child_id = match self.nodes[current].children.get(ck).copied() {
                Some(id) => id,
                None => {
                    return self.create_child(current, &key[key_offset..], &value[key_offset..]);
                }
            };

            let (common_len, child_len) = {
                let child_key = &self.nodes[child_id].key;
                (
                    self.key_match(child_key, &key[key_offset..]),
                    child_key.len(),
                )
            };

            if common_len == child_len {
                key_offset += common_len;
                current = child_id;
                self.nodes[current].last_access_time = now;
            } else {
                if common_len > 0 {
                    let intermediate = self.split_node(child_id, common_len);
                    key_offset += common_len;
                    if key_offset < key.len() {
                        return self.create_child(
                            intermediate,
                            &key[key_offset..],
                            &value[key_offset..],
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
            self.nodes[id].last_access_time = now;
            current = self.nodes[id].parent;
        }
    }

    fn extend_leaf(
        &mut self,
        node_id: NodeId,
        key: &[u64],
        value: &[usize],
        now: Instant,
    ) -> NodeId {
        let node = &mut self.nodes[node_id];
        debug_assert!(node.children.is_empty());
        debug_assert!(node.lock_ref <= 1);
        node.key.extend_from_slice(key);
        node.value.extend_from_slice(value);
        node.last_access_time = now;
        if node.lock_ref == 0 {
            self.evictable_size += key.len();
        } else {
            self.protected_size += key.len();
        }
        node_id
    }

    fn split_node(&mut self, child_id: NodeId, split_pos: usize) -> NodeId {
        let (child_parent, original_ck, prefix_key, prefix_value, suffix_ck, lock_ref, accessed) = {
            let child = &mut self.nodes[child_id];
            let child_parent = child.parent;
            let original_ck = child.key[..self.page_size].to_vec();
            let suffix_key = child.key.split_off(split_pos);
            let prefix_key = std::mem::replace(&mut child.key, suffix_key);
            let suffix_value = child.value.split_off(split_pos);
            let prefix_value = std::mem::replace(&mut child.value, suffix_value);
            let suffix_ck = child.key[..self.page_size].to_vec();
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

    fn create_child(&mut self, parent_id: NodeId, key: &[u64], value: &[usize]) -> NodeId {
        let new_node = TreeNode {
            children: FxHashMap::default(),
            parent: Some(parent_id),
            key: key.to_vec(),
            value: value.to_vec(),
            lock_ref: 0,
            last_access_time: Instant::now(),
        };
        let ck = self.child_key(key).to_vec();
        let new_id = self.nodes.insert(new_node);

        self.evictable_leaves.remove(&parent_id);

        self.nodes[parent_id].children.insert(ck, new_id);

        self.evictable_leaves.insert(new_id);
        self.evictable_size += key.len();

        new_id
    }

    pub fn is_leaf(&self, id: NodeId) -> bool {
        self.nodes[id].children.is_empty()
    }

    pub fn inc_lock_ref(&mut self, node_id: NodeId) {
        let mut current = Some(node_id);
        while let Some(id) = current {
            if id == self.root {
                break;
            }
            let node = &mut self.nodes[id];
            let tokens = node.key.len();
            node.lock_ref += 1;
            if node.lock_ref == 1 {
                self.evictable_leaves.remove(&id);
                self.evictable_size -= tokens;
                self.protected_size += tokens;
            }
            current = self.nodes[id].parent;
        }
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
                let tokens = node.key.len();
                self.protected_size -= tokens;
                self.evictable_size += tokens;
                if self.is_leaf(id) {
                    self.evictable_leaves.insert(id);
                }
            }
            current = self.nodes[id].parent;
        }
    }

    /// Evict tokens from the cache by LRU order, rounding partial leaves to full pages.
    /// Returns `(num_tokens_evicted, evicted_page_indices)`.
    pub fn evict(&mut self, num_tokens: usize) -> (usize, Vec<usize>) {
        let mut evicted = 0;
        let mut evicted_indices = Vec::with_capacity(num_tokens.min(self.evictable_size));
        while evicted < num_tokens {
            let victim = self
                .evictable_leaves
                .iter()
                .min_by_key(|&&id| self.nodes[id].last_access_time)
                .copied();

            let Some(victim_id) = victim else {
                break;
            };

            let victim_tokens = self.nodes[victim_id].key.len();
            let remaining = num_tokens - evicted;
            let eviction_len = remaining
                .div_ceil(self.page_size)
                .saturating_mul(self.page_size)
                .min(victim_tokens);

            // A compressed leaf may span pages. Preserve its indexed prefix when
            // only the newest suffix pages are needed to satisfy this eviction.
            if eviction_len < victim_tokens {
                let split_pos = victim_tokens - eviction_len;
                let (nodes, token_pool) = (&mut self.nodes, &mut self.token_pool);
                let victim_node = &mut nodes[victim_id];
                victim_node.key.truncate(split_pos);
                let evicted_values = &victim_node.value[split_pos..];
                token_pool.free(evicted_values);
                evicted_indices.extend_from_slice(evicted_values);
                victim_node.value.truncate(split_pos);

                self.evictable_size -= eviction_len;
                evicted += eviction_len;
                continue;
            }

            let victim_node = self
                .nodes
                .remove(victim_id)
                .expect("evictable leaf disappeared before removal");
            let tokens = victim_node.key.len();
            let parent_id = victim_node.parent;

            self.evictable_leaves.remove(&victim_id);
            self.evictable_size -= tokens;
            evicted += tokens;

            evicted_indices.extend_from_slice(&victim_node.value);
            self.token_pool.free(&victim_node.value);

            if let Some(pid) = parent_id {
                let ck = self.child_key(&victim_node.key);
                self.nodes[pid].children.remove(ck);

                if pid != self.root
                    && self.nodes[pid].children.is_empty()
                    && self.nodes[pid].lock_ref == 0
                {
                    self.evictable_leaves.insert(pid);
                }
            }
        }
        (evicted, evicted_indices)
    }

    pub fn available_tokens(&self) -> usize {
        self.token_pool.available()
    }

    pub fn total_tokens(&self) -> usize {
        self.token_pool.total()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_pool_allocate_and_free() {
        let mut pool = TokenPool::new(10);
        assert_eq!(pool.available(), 10);
        let a = pool.allocate(3).unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(pool.available(), 7);
        let b = pool.allocate(7).unwrap();
        assert_eq!(pool.available(), 0);
        assert!(pool.allocate(1).is_none());
        pool.free(&a);
        assert_eq!(pool.available(), 3);
        pool.free(&b);
        assert_eq!(pool.available(), 10);
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
        assert_eq!(n.key, vec![1, 2, 3, 4, 5]);
        assert_eq!(n.value, vec![10, 20, 30, 40, 50]);
        let &suffix_id = n.children.get(&vec![6]).unwrap();
        assert_eq!(cache.node(suffix_id).value, vec![60, 70]);
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
        assert_eq!(cache.node(tail).key, vec![1, 2, 3, 4, 5, 6, 7, 8]);
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
        assert_eq!(cache.node(tail).key, vec![1, 2, 3, 4]);
        assert_eq!(cache.node(extended).key, vec![5, 6, 7, 8]);
    }

    #[test]
    fn test_page_size() {
        // Insert and match with page_size=4
        let mut cache = RadixCache::new(100, 4);
        assert_eq!(cache.token_pool.total(), 100);
        cache.insert(&[1, 2, 3, 4, 5, 6, 7], &[0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4]).0, 4);
        let (_, node) = cache.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(cache.node(node).value, vec![0, 1, 2, 3]);

        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &[0, 1, 2, 3, 10, 11, 12, 13]);
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
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &[0, 1, 2, 3, 10, 11, 12, 13]);
        cache.match_prefix(&[1, 2, 3, 4, 9, 9, 9, 9]);
        let (_, node) = cache.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(cache.node(node).value, vec![0, 1, 2, 3]);
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
        assert!(cache.evictable_leaves.contains(&node_a));
        cache.dec_lock_ref(node_b);
        assert_eq!(cache.protected_size, 0);
    }

    #[test]
    fn test_evict() {
        // LRU order: oldest evicted first
        let mut cache = RadixCache::new(100, 1);
        cache.insert(&[1, 2, 3], &[0, 1, 2]);
        let (_, n1) = cache.match_prefix(&[1, 2, 3]);
        cache.inc_lock_ref(n1);
        cache.dec_lock_ref(n1);

        std::thread::sleep(std::time::Duration::from_millis(1));
        cache.insert(&[4, 5, 6], &[3, 4, 5]);
        let (_, n2) = cache.match_prefix(&[4, 5, 6]);
        cache.inc_lock_ref(n2);
        cache.dec_lock_ref(n2);

        let (evicted_count, evicted_indices) = cache.evict(3);
        assert_eq!(evicted_count, 3);
        // Evicted indices should match the pool indices originally inserted for [1,2,3]
        let mut sorted_evicted = evicted_indices.clone();
        sorted_evicted.sort();
        let mut expected_indices = vec![0, 1, 2];
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
            vec![3, 4, 5],
            "should evict unlocked [4,5,6] indices"
        );
        assert_eq!(cache.match_prefix(&[1, 2, 3]).0, 3);
    }

    #[test]
    fn test_evictable_size_includes_unlocked_internal_prefix() {
        let mut cache = RadixCache::new(16, 4);
        let first = cache.token_pool.allocate(8).unwrap();
        cache.insert(&[1; 8], &first);
        let mut branch = first[..4].to_vec();
        branch.extend(cache.token_pool.allocate(4).unwrap());
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
