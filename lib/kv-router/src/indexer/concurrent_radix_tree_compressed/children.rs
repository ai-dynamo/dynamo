// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use arc_swap::ArcSwap;
use dashmap::DashMap;
use rustc_hash::{FxBuildHasher, FxHashMap};

use super::SharedNode;
#[cfg(test)]
use super::node::Node;
use crate::protocols::LocalBlockHash;

type ShardedChildren = DashMap<LocalBlockHash, SharedNode, FxBuildHasher>;

pub(super) struct NodeChildren {
    state: ArcSwap<CompactChildrenState>,
}

#[derive(Debug)]
enum CompactChildrenState {
    Inline(Option<(LocalBlockHash, SharedNode)>),
    Sharded(ShardedChildren),
}

impl std::fmt::Debug for NodeChildren {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NodeChildren")
            .field("len", &self.len())
            .field("promoted", &self.is_promoted())
            .finish()
    }
}

pub(super) enum ChildInsertResult {
    Existing(SharedNode),
    Inserted,
}

impl NodeChildren {
    pub(super) fn from_entries(children: FxHashMap<LocalBlockHash, SharedNode>) -> Self {
        match children.len() {
            0 => Self {
                state: ArcSwap::from_pointee(CompactChildrenState::Inline(None)),
            },
            1 => Self {
                state: ArcSwap::from_pointee(CompactChildrenState::Inline(
                    children.into_iter().next(),
                )),
            },
            _ => {
                let map = DashMap::with_hasher(FxBuildHasher);
                for (key, child) in children {
                    map.insert(key, child);
                }
                Self {
                    state: ArcSwap::from_pointee(CompactChildrenState::Sharded(map)),
                }
            }
        }
    }

    fn publish_if_current(
        &self,
        current: &Arc<CompactChildrenState>,
        next: CompactChildrenState,
    ) -> bool {
        let previous = self.state.compare_and_swap(current, Arc::new(next));
        Arc::ptr_eq(&previous, current)
    }

    pub(super) fn get_cloned(&self, key: LocalBlockHash) -> Option<SharedNode> {
        let state = self.state.load();
        match &**state {
            CompactChildrenState::Inline(Some((entry_key, child))) if *entry_key == key => {
                Some(child.clone())
            }
            CompactChildrenState::Inline(_) => None,
            CompactChildrenState::Sharded(map) => map.get(&key).map(|entry| entry.value().clone()),
        }
    }

    pub(super) fn insert_if_absent(
        &self,
        key: LocalBlockHash,
        child: SharedNode,
    ) -> ChildInsertResult {
        loop {
            let current = self.state.load_full();
            match &*current {
                CompactChildrenState::Inline(None) => {
                    if self.publish_if_current(
                        &current,
                        CompactChildrenState::Inline(Some((key, child.clone()))),
                    ) {
                        return ChildInsertResult::Inserted;
                    }
                }
                CompactChildrenState::Inline(Some((entry_key, existing))) if *entry_key == key => {
                    return ChildInsertResult::Existing(existing.clone());
                }
                CompactChildrenState::Inline(Some((entry_key, existing))) => {
                    let map = DashMap::with_hasher(FxBuildHasher);
                    map.insert(*entry_key, existing.clone());
                    map.insert(key, child.clone());
                    if self.publish_if_current(&current, CompactChildrenState::Sharded(map)) {
                        return ChildInsertResult::Inserted;
                    }
                }
                CompactChildrenState::Sharded(map) => match map.entry(key) {
                    dashmap::mapref::entry::Entry::Occupied(entry) => {
                        return ChildInsertResult::Existing(entry.get().clone());
                    }
                    dashmap::mapref::entry::Entry::Vacant(entry) => {
                        entry.insert(child);
                        return ChildInsertResult::Inserted;
                    }
                },
            }
        }
    }

    pub(super) fn insert_or_replace(&self, key: LocalBlockHash, child: SharedNode) {
        loop {
            let current = self.state.load_full();
            match &*current {
                CompactChildrenState::Inline(None) => {
                    if self.publish_if_current(
                        &current,
                        CompactChildrenState::Inline(Some((key, child.clone()))),
                    ) {
                        return;
                    }
                }
                CompactChildrenState::Inline(Some((entry_key, _))) if *entry_key == key => {
                    if self.publish_if_current(
                        &current,
                        CompactChildrenState::Inline(Some((key, child.clone()))),
                    ) {
                        return;
                    }
                }
                CompactChildrenState::Inline(Some((entry_key, existing))) => {
                    let map = DashMap::with_hasher(FxBuildHasher);
                    map.insert(*entry_key, existing.clone());
                    map.insert(key, child.clone());
                    if self.publish_if_current(&current, CompactChildrenState::Sharded(map)) {
                        return;
                    }
                }
                CompactChildrenState::Sharded(map) => {
                    map.insert(key, child);
                    return;
                }
            }
        }
    }

    pub(super) fn remove_if_same(&self, key: LocalBlockHash, child: &SharedNode) -> bool {
        loop {
            let current = self.state.load_full();
            match &*current {
                CompactChildrenState::Inline(Some((entry_key, existing)))
                    if *entry_key == key && Arc::ptr_eq(existing, child) =>
                {
                    if self.publish_if_current(&current, CompactChildrenState::Inline(None)) {
                        return true;
                    }
                }
                CompactChildrenState::Inline(_) => return false,
                CompactChildrenState::Sharded(map) => {
                    return match map.entry(key) {
                        dashmap::mapref::entry::Entry::Occupied(entry)
                            if Arc::ptr_eq(entry.get(), child) =>
                        {
                            entry.remove();
                            true
                        }
                        _ => false,
                    };
                }
            }
        }
    }

    pub(super) fn clear_children(&self) -> bool {
        loop {
            let current = self.state.load_full();
            match &*current {
                CompactChildrenState::Inline(None) => return false,
                CompactChildrenState::Inline(Some(_)) => {
                    if self.publish_if_current(&current, CompactChildrenState::Inline(None)) {
                        return true;
                    }
                }
                CompactChildrenState::Sharded(map) => {
                    if map.is_empty() {
                        return false;
                    }
                    map.clear();
                    return true;
                }
            }
        }
    }

    pub(super) fn entries_snapshot(&self) -> Vec<(LocalBlockHash, SharedNode)> {
        let state = self.state.load();
        match &**state {
            CompactChildrenState::Inline(None) => Vec::new(),
            CompactChildrenState::Inline(Some((key, child))) => {
                vec![(*key, child.clone())]
            }
            CompactChildrenState::Sharded(map) => map
                .iter()
                .map(|entry| (*entry.key(), entry.value().clone()))
                .collect(),
        }
    }

    pub(super) fn len(&self) -> usize {
        let state = self.state.load();
        match &**state {
            CompactChildrenState::Inline(None) => 0,
            CompactChildrenState::Inline(Some(_)) => 1,
            CompactChildrenState::Sharded(map) => map.len(),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(super) fn values_snapshot(&self) -> Vec<SharedNode> {
        self.entries_snapshot()
            .into_iter()
            .map(|(_, child)| child)
            .collect()
    }

    #[cfg(feature = "bench")]
    pub(super) fn capacity(&self) -> usize {
        let state = self.state.load();
        match &**state {
            CompactChildrenState::Inline(_) => 1,
            CompactChildrenState::Sharded(map) => map.capacity(),
        }
    }

    pub(super) fn is_promoted(&self) -> bool {
        matches!(&**self.state.load(), CompactChildrenState::Sharded(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node() -> SharedNode {
        Arc::new(Node::new())
    }

    #[test]
    fn second_distinct_child_promotes_and_clear_does_not_demote() {
        let children = NodeChildren::from_entries(FxHashMap::default());
        assert!(matches!(
            children.insert_if_absent(LocalBlockHash(1), node()),
            ChildInsertResult::Inserted
        ));
        assert!(!children.is_promoted());
        assert!(matches!(
            children.insert_if_absent(LocalBlockHash(2), node()),
            ChildInsertResult::Inserted
        ));
        assert!(children.is_promoted());
        assert!(children.clear_children());
        assert!(children.is_promoted());
    }

    #[test]
    fn concurrent_distinct_inserts_survive_singleton_promotion() {
        let children = Arc::new(NodeChildren::from_entries(FxHashMap::default()));
        let mut threads = Vec::new();
        for key in 0..8 {
            let children = children.clone();
            threads.push(std::thread::spawn(move || {
                children.insert_if_absent(LocalBlockHash(key), node());
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(children.len(), 8);
        assert!(children.is_promoted());
    }

    #[test]
    fn concurrent_duplicate_inserts_keep_one_unpromoted_child() {
        let children = Arc::new(NodeChildren::from_entries(FxHashMap::default()));
        let mut threads = Vec::new();
        for _ in 0..8 {
            let children = children.clone();
            threads.push(std::thread::spawn(move || {
                children.insert_if_absent(LocalBlockHash(1), node());
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(children.len(), 1);
        assert!(!children.is_promoted());
    }
}
