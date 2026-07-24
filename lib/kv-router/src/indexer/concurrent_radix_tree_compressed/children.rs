// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

use arc_swap::ArcSwap;
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use rustc_hash::{FxBuildHasher, FxHashMap};

use super::types::SharedNode;
use crate::protocols::LocalBlockHash;

const SMALL_CHILD_LIMIT: usize = 4;

type ShardedChildren = DashMap<LocalBlockHash, SharedNode, FxBuildHasher>;

#[derive(Clone)]
struct ChildEntry {
    hash: LocalBlockHash,
    node: SharedNode,
}

enum ChildrenState {
    Empty,
    Singleton(ChildEntry),
    Small(Box<[ChildEntry]>),
    Sharded(ShardedChildren),
}

pub(super) enum ChildInsertResult {
    Existing(SharedNode),
    Inserted(SharedNode),
}

pub(super) struct NodeChildren {
    state: ArcSwap<ChildrenState>,
}

impl fmt::Debug for NodeChildren {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeChildren")
            .field("len", &self.len())
            .finish()
    }
}

impl NodeChildren {
    pub(super) fn from_map(children: FxHashMap<LocalBlockHash, SharedNode>) -> Self {
        let state = if children.len() > SMALL_CHILD_LIMIT {
            let sharded = ShardedChildren::with_hasher(FxBuildHasher);
            for (key, child) in children {
                sharded.insert(key, child);
            }
            ChildrenState::Sharded(sharded)
        } else {
            let mut entries: Vec<_> = children
                .into_iter()
                .map(|(hash, node)| ChildEntry { hash, node })
                .collect();
            entries.sort_unstable_by_key(|entry| entry.hash);
            Self::compact_state(entries)
        };
        Self::from_state(Arc::new(state))
    }

    fn from_state(state: Arc<ChildrenState>) -> Self {
        Self {
            state: ArcSwap::new(state),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        match &**self.state.load() {
            ChildrenState::Empty => true,
            ChildrenState::Singleton(_) => false,
            ChildrenState::Small(children) => children.is_empty(),
            ChildrenState::Sharded(children) => children.is_empty(),
        }
    }

    fn len(&self) -> usize {
        match &**self.state.load() {
            ChildrenState::Empty => 0,
            ChildrenState::Singleton(_) => 1,
            ChildrenState::Small(children) => children.len(),
            ChildrenState::Sharded(children) => children.len(),
        }
    }

    pub(super) fn get(&self, key: &LocalBlockHash) -> Option<SharedNode> {
        match &**self.state.load() {
            ChildrenState::Empty => None,
            ChildrenState::Singleton(entry) => (entry.hash == *key).then(|| entry.node.clone()),
            ChildrenState::Small(children) => children
                .binary_search_by_key(key, |entry| entry.hash)
                .ok()
                .map(|index| children[index].node.clone()),
            ChildrenState::Sharded(children) => {
                children.get(key).map(|entry| entry.value().clone())
            }
        }
    }

    pub(super) fn values_snapshot(&self) -> Vec<SharedNode> {
        match &**self.state.load() {
            ChildrenState::Empty => Vec::new(),
            ChildrenState::Singleton(entry) => vec![entry.node.clone()],
            ChildrenState::Small(children) => {
                children.iter().map(|entry| entry.node.clone()).collect()
            }
            ChildrenState::Sharded(children) => {
                children.iter().map(|entry| entry.value().clone()).collect()
            }
        }
    }

    pub(super) fn entries_snapshot(&self) -> Vec<(LocalBlockHash, SharedNode)> {
        match &**self.state.load() {
            ChildrenState::Empty => Vec::new(),
            ChildrenState::Singleton(entry) => vec![(entry.hash, entry.node.clone())],
            ChildrenState::Small(children) => children
                .iter()
                .map(|entry| (entry.hash, entry.node.clone()))
                .collect(),
            ChildrenState::Sharded(children) => children
                .iter()
                .map(|entry| (*entry.key(), entry.value().clone()))
                .collect(),
        }
    }

    pub(super) fn extend_values(&self, queue: &mut VecDeque<SharedNode>) {
        match &**self.state.load() {
            ChildrenState::Empty => {}
            ChildrenState::Singleton(entry) => queue.push_back(entry.node.clone()),
            ChildrenState::Small(children) => {
                queue.extend(children.iter().map(|entry| entry.node.clone()));
            }
            ChildrenState::Sharded(children) => {
                queue.extend(children.iter().map(|entry| entry.value().clone()));
            }
        }
    }

    pub(super) fn insert(&self, key: LocalBlockHash, child: SharedNode) -> Option<SharedNode> {
        loop {
            let current = self.state.load_full();
            if let ChildrenState::Sharded(children) = current.as_ref() {
                return children.insert(key, child);
            }

            let mut entries = Self::clone_compact_entries(current.as_ref());
            let previous = match entries.binary_search_by_key(&key, |entry| entry.hash) {
                Ok(index) => Some(std::mem::replace(&mut entries[index].node, child.clone())),
                Err(index) => {
                    entries.insert(
                        index,
                        ChildEntry {
                            hash: key,
                            node: child.clone(),
                        },
                    );
                    None
                }
            };

            if self.compare_and_swap(&current, Self::state_from_entries(entries)) {
                return previous;
            }
        }
    }

    pub(super) fn insert_if_absent(
        &self,
        key: LocalBlockHash,
        child: SharedNode,
    ) -> ChildInsertResult {
        loop {
            let current = self.state.load_full();
            let insert_index = match current.as_ref() {
                ChildrenState::Empty => 0,
                ChildrenState::Singleton(entry) if entry.hash == key => {
                    return ChildInsertResult::Existing(entry.node.clone());
                }
                ChildrenState::Singleton(entry) => usize::from(entry.hash < key),
                ChildrenState::Small(entries) => {
                    match entries.binary_search_by_key(&key, |entry| entry.hash) {
                        Ok(index) => {
                            return ChildInsertResult::Existing(entries[index].node.clone());
                        }
                        Err(index) => index,
                    }
                }
                ChildrenState::Sharded(children) => {
                    return match children.entry(key) {
                        Entry::Occupied(entry) => ChildInsertResult::Existing(entry.get().clone()),
                        Entry::Vacant(entry) => {
                            entry.insert(child.clone());
                            ChildInsertResult::Inserted(child)
                        }
                    };
                }
            };

            let mut entries = Self::clone_compact_entries(current.as_ref());
            entries.insert(
                insert_index,
                ChildEntry {
                    hash: key,
                    node: child.clone(),
                },
            );
            if self.compare_and_swap(&current, Self::state_from_entries(entries)) {
                return ChildInsertResult::Inserted(child);
            }
        }
    }

    pub(super) fn remove(&self, key: &LocalBlockHash) -> Option<SharedNode> {
        loop {
            let current = self.state.load_full();
            let remove_index = match current.as_ref() {
                ChildrenState::Empty => return None,
                ChildrenState::Singleton(entry) => {
                    if entry.hash != *key {
                        return None;
                    }
                    0
                }
                ChildrenState::Small(entries) => {
                    let Ok(index) = entries.binary_search_by_key(key, |entry| entry.hash) else {
                        return None;
                    };
                    index
                }
                ChildrenState::Sharded(children) => {
                    return children.remove(key).map(|(_, child)| child);
                }
            };

            let mut entries = Self::clone_compact_entries(current.as_ref());
            let removed = entries.remove(remove_index);
            if self.compare_and_swap(&current, Self::compact_state(entries)) {
                return Some(removed.node);
            }
        }
    }

    pub(super) fn clear(&self) -> bool {
        loop {
            let current = self.state.load_full();
            match current.as_ref() {
                ChildrenState::Empty => return false,
                ChildrenState::Sharded(children) => {
                    let had_children = !children.is_empty();
                    children.clear();
                    return had_children;
                }
                ChildrenState::Singleton(_) | ChildrenState::Small(_) => {
                    if self.compare_and_swap(&current, ChildrenState::Empty) {
                        return true;
                    }
                }
            }
        }
    }

    /// Transfers the current state while the owning node's exclusive shape gate is held.
    pub(super) fn transfer_for_split(&self) -> Self {
        loop {
            let current = self.state.load_full();
            let replacement = match current.as_ref() {
                ChildrenState::Sharded(_) => {
                    ChildrenState::Sharded(ShardedChildren::with_hasher(FxBuildHasher))
                }
                ChildrenState::Empty | ChildrenState::Singleton(_) | ChildrenState::Small(_) => {
                    ChildrenState::Empty
                }
            };
            if self.compare_and_swap(&current, replacement) {
                return Self::from_state(current);
            }
        }
    }

    fn clone_compact_entries(state: &ChildrenState) -> Vec<ChildEntry> {
        match state {
            ChildrenState::Empty => Vec::new(),
            ChildrenState::Singleton(entry) => vec![entry.clone()],
            ChildrenState::Small(children) => children.to_vec(),
            ChildrenState::Sharded(_) => unreachable!("sharded children are mutated in place"),
        }
    }

    fn state_from_entries(entries: Vec<ChildEntry>) -> ChildrenState {
        if entries.len() <= SMALL_CHILD_LIMIT {
            return Self::compact_state(entries);
        }

        let sharded = ShardedChildren::with_hasher(FxBuildHasher);
        for entry in entries {
            sharded.insert(entry.hash, entry.node);
        }
        ChildrenState::Sharded(sharded)
    }

    fn compact_state(mut entries: Vec<ChildEntry>) -> ChildrenState {
        debug_assert!(entries.windows(2).all(|pair| pair[0].hash < pair[1].hash));
        if entries.is_empty() {
            return ChildrenState::Empty;
        }
        if entries.len() == 1 {
            if let Some(entry) = entries.pop() {
                return ChildrenState::Singleton(entry);
            }
            return ChildrenState::Empty;
        }
        if entries.len() <= SMALL_CHILD_LIMIT {
            return ChildrenState::Small(entries.into_boxed_slice());
        }

        let sharded = ShardedChildren::with_hasher(FxBuildHasher);
        for entry in entries {
            sharded.insert(entry.hash, entry.node);
        }
        ChildrenState::Sharded(sharded)
    }

    fn compare_and_swap(&self, current: &Arc<ChildrenState>, next: ChildrenState) -> bool {
        let previous = self.state.compare_and_swap(current, Arc::new(next));
        Arc::ptr_eq(current, &*previous)
    }

    #[cfg(test)]
    fn kind(&self) -> ChildrenKind {
        match &**self.state.load() {
            ChildrenState::Empty => ChildrenKind::Empty,
            ChildrenState::Singleton(_) => ChildrenKind::Singleton,
            ChildrenState::Small(_) => ChildrenKind::Small,
            ChildrenState::Sharded(_) => ChildrenKind::Sharded,
        }
    }
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
enum ChildrenKind {
    Empty,
    Singleton,
    Small,
    Sharded,
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    use super::*;
    use crate::indexer::concurrent_radix_tree_compressed::node::Node;

    fn child() -> SharedNode {
        Arc::new(Node::new())
    }

    #[test]
    fn compact_removals_shrink_to_singleton_then_empty() {
        let children = NodeChildren::from_map(FxHashMap::default());
        let nodes: Vec<_> = (0..SMALL_CHILD_LIMIT).map(|_| child()).collect();

        for (key, node) in nodes.iter().enumerate() {
            let result = children.insert_if_absent(LocalBlockHash(key as u64), node.clone());
            assert!(matches!(result, ChildInsertResult::Inserted(_)));
        }
        assert_eq!(children.kind(), ChildrenKind::Small);

        for key in (1..SMALL_CHILD_LIMIT).rev() {
            let removed = children.remove(&LocalBlockHash(key as u64)).unwrap();
            assert!(Arc::ptr_eq(&removed, &nodes[key]));
        }
        assert_eq!(children.kind(), ChildrenKind::Singleton);
        assert!(Arc::ptr_eq(
            &children.get(&LocalBlockHash(0)).unwrap(),
            &nodes[0]
        ));

        children.remove(&LocalBlockHash(0)).unwrap();
        assert_eq!(children.kind(), ChildrenKind::Empty);
        assert!(children.is_empty());
    }

    #[test]
    fn fifth_distinct_child_promotes_and_sharded_never_demotes() {
        let children = NodeChildren::from_map(FxHashMap::default());

        for key in 0..=SMALL_CHILD_LIMIT {
            let result = children.insert_if_absent(LocalBlockHash(key as u64), child());
            assert!(matches!(result, ChildInsertResult::Inserted(_)));
        }
        assert_eq!(children.kind(), ChildrenKind::Sharded);
        assert_eq!(children.len(), SMALL_CHILD_LIMIT + 1);

        for key in 0..=SMALL_CHILD_LIMIT {
            children.remove(&LocalBlockHash(key as u64)).unwrap();
        }
        assert!(children.is_empty());
        assert_eq!(children.kind(), ChildrenKind::Sharded);

        children.insert(LocalBlockHash(99), child());
        children.clear();
        assert!(children.is_empty());
        assert_eq!(children.kind(), ChildrenKind::Sharded);
    }

    #[test]
    fn split_transfer_preserves_compact_and_sharded_history() {
        let compact = NodeChildren::from_map(FxHashMap::default());
        compact.insert(LocalBlockHash(1), child());
        compact.insert(LocalBlockHash(2), child());
        let compact_suffix = compact.transfer_for_split();
        assert_eq!(compact.kind(), ChildrenKind::Empty);
        assert_eq!(compact_suffix.kind(), ChildrenKind::Small);
        assert_eq!(compact_suffix.len(), 2);
        compact.insert(LocalBlockHash(3), child());
        assert_eq!(compact.kind(), ChildrenKind::Singleton);

        let sharded = NodeChildren::from_map(FxHashMap::default());
        for key in 0..=SMALL_CHILD_LIMIT {
            sharded.insert(LocalBlockHash(key as u64), child());
        }
        let sharded_suffix = sharded.transfer_for_split();
        assert_eq!(sharded.kind(), ChildrenKind::Sharded);
        assert!(sharded.is_empty());
        assert_eq!(sharded_suffix.kind(), ChildrenKind::Sharded);
        assert_eq!(sharded_suffix.len(), SMALL_CHILD_LIMIT + 1);
        sharded.insert(LocalBlockHash(99), child());
        assert_eq!(sharded.kind(), ChildrenKind::Sharded);
    }

    #[test]
    fn stale_compact_remove_replace_cannot_overwrite_promotion() {
        let children = Arc::new(NodeChildren::from_map(FxHashMap::default()));
        let originals: Vec<_> = (0..SMALL_CHILD_LIMIT).map(|_| child()).collect();
        for (key, node) in originals.iter().enumerate() {
            children.insert(LocalBlockHash(key as u64), node.clone());
        }

        let stale = children.state.load_full();
        let mut stale_entries = NodeChildren::clone_compact_entries(stale.as_ref());
        stale_entries.remove(0);
        let replacement = child();
        let replacement_index = stale_entries
            .binary_search_by_key(&LocalBlockHash(1), |entry| entry.hash)
            .unwrap();
        stale_entries[replacement_index].node = replacement.clone();

        let start = Arc::new(Barrier::new(2));
        let promoted = Arc::new(Barrier::new(2));
        let promote_children = children.clone();
        let promote_start = start.clone();
        let promote_done = promoted.clone();
        let fifth = child();
        let fifth_for_thread = fifth.clone();
        let handle = thread::spawn(move || {
            promote_start.wait();
            let result = promote_children
                .insert_if_absent(LocalBlockHash(SMALL_CHILD_LIMIT as u64), fifth_for_thread);
            promote_done.wait();
            result
        });

        start.wait();
        promoted.wait();
        assert_eq!(children.kind(), ChildrenKind::Sharded);
        assert!(!children.compare_and_swap(&stale, NodeChildren::compact_state(stale_entries)));
        assert!(matches!(
            handle.join().unwrap(),
            ChildInsertResult::Inserted(_)
        ));

        assert!(Arc::ptr_eq(
            &children.get(&LocalBlockHash(0)).unwrap(),
            &originals[0]
        ));
        assert!(Arc::ptr_eq(
            &children.get(&LocalBlockHash(1)).unwrap(),
            &originals[1]
        ));
        assert!(!Arc::ptr_eq(
            &children.get(&LocalBlockHash(1)).unwrap(),
            &replacement
        ));
        assert!(Arc::ptr_eq(
            &children
                .get(&LocalBlockHash(SMALL_CHILD_LIMIT as u64))
                .unwrap(),
            &fifth
        ));
    }

    #[test]
    fn concurrent_duplicate_insert_has_one_winner() {
        const THREADS: usize = 16;
        let children = Arc::new(NodeChildren::from_map(FxHashMap::default()));
        let barrier = Arc::new(Barrier::new(THREADS + 1));
        let inserted = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::with_capacity(THREADS);

        for _ in 0..THREADS {
            let children = children.clone();
            let barrier = barrier.clone();
            let inserted = inserted.clone();
            handles.push(thread::spawn(move || {
                let candidate = child();
                barrier.wait();
                let result = children.insert_if_absent(LocalBlockHash(7), candidate);
                match result {
                    ChildInsertResult::Inserted(node) => {
                        inserted.fetch_add(1, Ordering::Relaxed);
                        node
                    }
                    ChildInsertResult::Existing(node) => node,
                }
            }));
        }

        barrier.wait();
        let returned: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        let stored = children.get(&LocalBlockHash(7)).unwrap();
        assert_eq!(inserted.load(Ordering::Relaxed), 1);
        assert_eq!(children.len(), 1);
        assert!(returned.iter().all(|node| Arc::ptr_eq(node, &stored)));
    }

    #[test]
    fn concurrent_distinct_inserts_survive_promotion_races() {
        const THREADS: usize = 32;
        let children = Arc::new(NodeChildren::from_map(FxHashMap::default()));
        let barrier = Arc::new(Barrier::new(THREADS + 1));
        let mut handles = Vec::with_capacity(THREADS);

        for key in 0..THREADS {
            let children = children.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                children.insert_if_absent(LocalBlockHash(key as u64), child())
            }));
        }

        barrier.wait();
        for handle in handles {
            assert!(matches!(
                handle.join().unwrap(),
                ChildInsertResult::Inserted(_)
            ));
        }
        assert_eq!(children.kind(), ChildrenKind::Sharded);
        assert_eq!(children.len(), THREADS);
        for key in 0..THREADS {
            assert!(children.get(&LocalBlockHash(key as u64)).is_some());
        }
    }
}
