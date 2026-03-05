// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Radix-tree KV cache for SGLang engine simulation.
//!
//! Supports both token-level (`page_size=1`, default) and page-level
//! (`page_size>1`) granularity, matching SGLang's RadixCache behaviour.
//!
//! Node values store **one pool index per page** (not per token).
//! When `page_size=1`, `value.len() == key.len()`.
//! When `page_size>1`, `value.len() == key.len() / page_size`.
//!
//! Reference: sglang/python/sglang/srt/mem_cache/radix_cache.py

use slotmap::{SlotMap, new_key_type};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

new_key_type! {
    /// Stable identifier for a tree node inside the [`RadixCache`].
    pub struct NodeId;
}

/// Manages free / allocated page slot indices for the KV cache pool.
///
/// Each slot represents one *page* of KV cache (= `page_size` tokens).
/// Note: SGLang uses VecDeque (FIFO allocation). We use Vec (LIFO) for
/// simplicity; allocation order doesn't affect simulation correctness.
pub struct TokenPool {
    free: Vec<usize>,
    total: usize,
}

impl TokenPool {
    pub fn new(total: usize) -> Self {
        let free: Vec<usize> = (0..total).rev().collect();
        Self { free, total }
    }

    /// Allocate `n` page slots. Returns `None` if not enough free slots.
    pub fn allocate(&mut self, n: usize) -> Option<Vec<usize>> {
        if self.free.len() < n {
            return None;
        }
        let start = self.free.len() - n;
        let indices: Vec<usize> = self.free.drain(start..).collect();
        Some(indices)
    }

    /// Return page slots to the free pool.
    pub fn free(&mut self, indices: &[usize]) {
        self.free.extend(indices);
    }

    pub fn available(&self) -> usize {
        self.free.len()
    }

    pub fn total(&self) -> usize {
        self.total
    }
}

/// A single node in the radix tree.
pub struct TreeNode {
    /// Children keyed by `child.key[..page_size]` (a "child key").
    pub children: HashMap<Vec<u64>, NodeId>,
    pub parent: Option<NodeId>,
    /// Token IDs stored at this edge.
    pub key: Vec<u64>,
    /// KV cache pool page indices. Length = `key.len() / page_size`.
    pub value: Vec<usize>,
    /// Walk-to-root reference count (protected when > 0).
    pub lock_ref: usize,
    /// Monotonic timestamp for LRU eviction.
    pub last_access_time: Instant,
}

/// Radix tree for SGLang KV cache simulation.
///
/// `page_size == 1`: token-level granularity (default).
/// `page_size > 1`: page-level granularity — pool manages pages, matching
/// and splitting happen at page boundaries, children are keyed by the first
/// `page_size` tokens.
pub struct RadixCache {
    nodes: SlotMap<NodeId, TreeNode>,
    root: NodeId,
    pub token_pool: TokenPool,
    page_size: usize,
    /// Total token count in evictable nodes.
    pub evictable_leaves: HashSet<NodeId>,
    pub evictable_size: usize,
    /// Total token count in protected (locked) nodes.
    pub protected_size: usize,
}

impl RadixCache {
    pub fn new(total_tokens: usize, page_size: usize) -> Self {
        assert!(page_size >= 1, "page_size must be >= 1");
        let num_pages = total_tokens / page_size;
        let mut nodes = SlotMap::with_key();
        let root = nodes.insert(TreeNode {
            children: HashMap::new(),
            parent: None,
            key: Vec::new(),
            value: Vec::new(),
            lock_ref: 0,
            last_access_time: Instant::now(),
        });
        Self {
            nodes,
            root,
            token_pool: TokenPool::new(num_pages),
            page_size,
            evictable_leaves: HashSet::new(),
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

    fn child_key(&self, key: &[u64]) -> Vec<u64> {
        key[..self.page_size.min(key.len())].to_vec()
    }

    fn page_align(&self, len: usize) -> usize {
        len / self.page_size * self.page_size
    }

    /// Convert token count to page count.
    fn token_to_page(&self, tokens: usize) -> usize {
        tokens / self.page_size
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
            let child_id = match self.nodes[current].children.get(&ck).copied() {
                Some(id) => id,
                None => break,
            };

            let child_key = self.nodes[child_id].key.clone();
            let common_len = self.key_match(&child_key, &key[matched..]);

            if common_len < child_key.len() {
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
            let child_id = match self.nodes[current].children.get(&ck).copied() {
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

    /// Insert a token sequence into the tree.
    ///
    /// `key` contains token IDs, `value` contains page indices
    /// (1 per `page_size` tokens). Both are page-aligned before insertion.
    pub fn insert(&mut self, key: &[u64], value: &[usize]) {
        let aligned_key_len = self.page_align(key.len());
        if aligned_key_len == 0 {
            return;
        }
        let aligned_value_len = self.token_to_page(aligned_key_len);
        assert!(
            value.len() >= aligned_value_len,
            "not enough page indices: need {aligned_value_len}, got {}",
            value.len()
        );
        let key = &key[..aligned_key_len];
        let value = &value[..aligned_value_len];

        let now = Instant::now();
        self.nodes[self.root].last_access_time = now;

        let mut current = self.root;
        let mut key_offset: usize = 0;

        while key_offset < key.len() {
            let ck = self.child_key(&key[key_offset..]);
            let val_offset = self.token_to_page(key_offset);
            let child_id = match self.nodes[current].children.get(&ck).copied() {
                Some(id) => id,
                None => {
                    self.create_child(current, &key[key_offset..], &value[val_offset..]);
                    return;
                }
            };

            let child_key = self.nodes[child_id].key.clone();
            let common_len = self.key_match(&child_key, &key[key_offset..]);

            if common_len == child_key.len() {
                key_offset += common_len;
                current = child_id;
                self.nodes[current].last_access_time = now;
            } else {
                if common_len > 0 {
                    let intermediate = self.split_node(child_id, common_len);
                    key_offset += common_len;
                    let val_offset = self.token_to_page(key_offset);
                    if key_offset < key.len() {
                        self.create_child(intermediate, &key[key_offset..], &value[val_offset..]);
                    }
                }
                return;
            }
        }
    }

    /// Split `child_id` at `split_pos` (token-level, page-aligned when page_size>1).
    fn split_node(&mut self, child_id: NodeId, split_pos: usize) -> NodeId {
        let child = &self.nodes[child_id];
        let child_parent = child.parent;
        let child_key = child.key.clone();
        let child_value = child.value.clone();
        let child_lock_ref = child.lock_ref;
        let child_last_access = child.last_access_time;

        let val_split = self.token_to_page(split_pos);

        let suffix_ck = self.child_key(&child_key[split_pos..]);
        let mut inter_children = HashMap::new();
        inter_children.insert(suffix_ck, child_id);

        let intermediate = TreeNode {
            children: inter_children,
            parent: child_parent,
            key: child_key[..split_pos].to_vec(),
            value: child_value[..val_split].to_vec(),
            lock_ref: child_lock_ref,
            last_access_time: child_last_access,
        };
        let inter_id = self.nodes.insert(intermediate);

        let child = &mut self.nodes[child_id];
        child.key = child_key[split_pos..].to_vec();
        child.value = child_value[val_split..].to_vec();
        child.parent = Some(inter_id);

        let original_ck = self.child_key(&child_key);
        if let Some(parent_id) = child_parent {
            self.nodes[parent_id].children.insert(original_ck, inter_id);
        }

        // Size tracking: intermediate inherits child's lock_ref, so
        // protected_size is unchanged (split_pos + remainder = original).
        // For evictable: intermediate is NOT a leaf (has child), so only
        // the child's contribution changes.
        if self.evictable_leaves.contains(&child_id) {
            let old_tokens = child_key.len();
            let new_tokens = child_key.len() - split_pos;
            self.evictable_size = self.evictable_size - old_tokens + new_tokens;
        }

        inter_id
    }

    fn create_child(&mut self, parent_id: NodeId, key: &[u64], value: &[usize]) {
        let new_node = TreeNode {
            children: HashMap::new(),
            parent: Some(parent_id),
            key: key.to_vec(),
            value: value.to_vec(),
            lock_ref: 0,
            last_access_time: Instant::now(),
        };
        let ck = self.child_key(key);
        let new_id = self.nodes.insert(new_node);

        if self.evictable_leaves.remove(&parent_id) {
            let parent_tokens = self.nodes[parent_id].key.len();
            self.evictable_size -= parent_tokens;
        }

        self.nodes[parent_id].children.insert(ck, new_id);

        self.evictable_leaves.insert(new_id);
        self.evictable_size += key.len();
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
            node.lock_ref += 1;
            if node.lock_ref == 1 {
                let tokens = node.key.len();
                if self.evictable_leaves.remove(&id) {
                    self.evictable_size -= tokens;
                }
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
                if self.is_leaf(id) {
                    self.evictable_leaves.insert(id);
                    self.evictable_size += tokens;
                }
            }
            current = self.nodes[id].parent;
        }
    }

    /// Evict tokens from the cache by LRU order.
    /// Returns `(num_tokens_evicted, evicted_page_indices)`.
    pub fn evict(&mut self, num_tokens: usize) -> (usize, Vec<usize>) {
        let mut evicted = 0;
        let mut evicted_indices = Vec::new();
        while evicted < num_tokens {
            let victim = self
                .evictable_leaves
                .iter()
                .min_by_key(|&&id| self.nodes[id].last_access_time)
                .copied();

            let Some(victim_id) = victim else {
                break;
            };

            let victim_node = &self.nodes[victim_id];
            let tokens = victim_node.key.len();
            let pool_indices = victim_node.value.clone();
            let parent_id = victim_node.parent;
            let victim_key = victim_node.key.clone();

            self.evictable_leaves.remove(&victim_id);
            self.evictable_size -= tokens;
            evicted += tokens;

            evicted_indices.extend_from_slice(&pool_indices);
            self.token_pool.free(&pool_indices);

            if let Some(pid) = parent_id {
                let ck = self.child_key(&victim_key);
                self.nodes[pid].children.remove(&ck);

                if pid != self.root
                    && self.nodes[pid].children.is_empty()
                    && self.nodes[pid].lock_ref == 0
                {
                    let parent_tokens = self.nodes[pid].key.len();
                    self.evictable_leaves.insert(pid);
                    self.evictable_size += parent_tokens;
                }
            }

            self.nodes.remove(victim_id);
        }
        (evicted, evicted_indices)
    }

    pub fn available_tokens(&self) -> usize {
        self.token_pool.available() * self.page_size
    }

    pub fn total_tokens(&self) -> usize {
        self.token_pool.total() * self.page_size
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
    fn test_match_prefix_empty_tree() {
        let mut cache = RadixCache::new(100, 1);
        let (len, node) = cache.match_prefix(&[1, 2, 3]);
        assert_eq!(len, 0);
        assert_eq!(node, cache.root());
    }

    #[test]
    fn test_match_prefix_full_match() {
        let mut cache = RadixCache::new(100, 1);
        cache.insert(&[1, 2, 3, 4, 5], &[10, 20, 30, 40, 50]);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5]).0, 5);
    }

    #[test]
    fn test_match_prefix_with_split() {
        let mut cache = RadixCache::new(100, 1);
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
    fn test_insert_shared_prefix() {
        let mut cache = RadixCache::new(100, 1);
        cache.insert(&[1, 2, 3, 4, 5], &[10, 20, 30, 40, 50]);
        cache.insert(&[1, 2, 3, 6, 7], &[10, 20, 30, 60, 70]);

        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5]).0, 5);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 6, 7]).0, 5);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 9]).0, 3);
    }

    #[test]
    fn test_insert_extends_existing() {
        let mut cache = RadixCache::new(100, 1);
        cache.insert(&[1, 2, 3], &[10, 20, 30]);
        cache.insert(&[1, 2, 3, 4, 5], &[10, 20, 30, 40, 50]);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5]).0, 5);
    }

    #[test]
    fn test_insert_duplicate() {
        let mut cache = RadixCache::new(100, 1);
        cache.insert(&[1, 2, 3], &[10, 20, 30]);
        cache.insert(&[1, 2, 3], &[10, 20, 30]);
        assert_eq!(cache.num_nodes(), 2);
    }

    #[test]
    fn test_match_then_insert_suffix() {
        let mut cache = RadixCache::new(100, 1);
        cache.insert(&[1, 2, 3, 4, 5], &[10, 20, 30, 40, 50]);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]).0, 5);
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &[10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]).0, 8);
    }

    // page_size > 1: value has 1 index per page

    #[test]
    fn test_page_insert_and_match() {
        let mut cache = RadixCache::new(100, 4);
        assert_eq!(cache.token_pool.total(), 25);

        // 7 tokens → truncate to 4 (1 page), value needs 1 entry
        cache.insert(&[1, 2, 3, 4, 5, 6, 7], &[0, 1]);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4]).0, 4);
        let (_, node) = cache.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(cache.node(node).value, vec![0]); // 1 page index

        // 8 tokens = 2 pages, value = [idx0, idx1]
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &[0, 1]);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]).0, 8);
    }

    #[test]
    fn test_page_children_disambiguate() {
        let mut cache = RadixCache::new(100, 4);
        cache.insert(&[1, 2, 3, 4], &[0]);
        cache.insert(&[1, 2, 3, 5], &[1]);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4]).0, 4);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 5]).0, 4);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 6]).0, 0);
    }

    #[test]
    fn test_page_split_value() {
        let mut cache = RadixCache::new(100, 4);
        // 2 pages: value = [idx0, idx1]
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &[0, 1]);
        // Split at page boundary → intermediate gets value [0], suffix gets value [1]
        cache.match_prefix(&[1, 2, 3, 4, 9, 9, 9, 9]);
        let (_, node) = cache.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(cache.node(node).value, vec![0]);
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
    fn test_evict_lru_order() {
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

        assert_eq!(cache.evict(3).0, 3);
        assert_eq!(cache.match_prefix(&[1, 2, 3]).0, 0); // oldest evicted
        assert_eq!(cache.match_prefix(&[4, 5, 6]).0, 3); // newer kept
    }

    #[test]
    fn test_evict_respects_locks() {
        let mut cache = RadixCache::new(100, 1);
        cache.insert(&[1, 2, 3], &[0, 1, 2]);
        cache.insert(&[4, 5, 6], &[3, 4, 5]);

        let (_, locked) = cache.match_prefix(&[1, 2, 3]);
        cache.inc_lock_ref(locked);
        let (_, unlocked) = cache.match_prefix(&[4, 5, 6]);
        cache.inc_lock_ref(unlocked);
        cache.dec_lock_ref(unlocked);

        assert_eq!(cache.evict(6).0, 3); // only unlocked evicted
        assert_eq!(cache.match_prefix(&[1, 2, 3]).0, 3);
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
