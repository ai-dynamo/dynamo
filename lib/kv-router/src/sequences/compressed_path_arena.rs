// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_tokens::SequenceHash;
use rustc_hash::FxHashMap;
use slotmap::{SlotMap, new_key_type};

new_key_type! {
    pub(super) struct CompressedNodeId;
}

#[derive(Debug)]
pub(super) struct CompressedPathNode<M> {
    pub(super) edge: Vec<SequenceHash>,
    pub(super) parent: Option<CompressedNodeId>,
    pub(super) children: FxHashMap<SequenceHash, CompressedNodeId>,
    pub(super) metadata: M,
}

/// A single-owner compressed path forest with generational node handles.
///
/// The existing node ID always stays with the suffix during a split. During a
/// unary merge the child ID survives. Those two rules keep request-held tail
/// handles valid across unrelated insertions and removals.
#[derive(Debug)]
pub(super) struct CompressedPathArena<M> {
    pub(super) nodes: SlotMap<CompressedNodeId, CompressedPathNode<M>>,
    pub(super) roots: FxHashMap<SequenceHash, CompressedNodeId>,
}

impl<M> Default for CompressedPathArena<M> {
    fn default() -> Self {
        Self {
            nodes: SlotMap::with_key(),
            roots: FxHashMap::default(),
        }
    }
}

impl<M> CompressedPathArena<M> {
    pub(super) fn root(&self, first_hash: SequenceHash) -> Option<CompressedNodeId> {
        self.roots.get(&first_hash).copied()
    }

    pub(super) fn insert_root(&mut self, edge: Vec<SequenceHash>, metadata: M) -> CompressedNodeId {
        assert!(!edge.is_empty(), "compressed root edge cannot be empty");
        let first = edge[0];
        let id = self.nodes.insert(CompressedPathNode {
            edge,
            parent: None,
            children: FxHashMap::default(),
            metadata,
        });
        assert!(
            self.roots.insert(first, id).is_none(),
            "compressed root already exists"
        );
        id
    }

    pub(super) fn insert_child(
        &mut self,
        parent: CompressedNodeId,
        edge: Vec<SequenceHash>,
        metadata: M,
    ) -> CompressedNodeId {
        assert!(!edge.is_empty(), "compressed child edge cannot be empty");
        assert!(
            self.nodes.contains_key(parent),
            "compressed child parent is missing"
        );
        let first = edge[0];
        let id = self.nodes.insert(CompressedPathNode {
            edge,
            parent: Some(parent),
            children: FxHashMap::default(),
            metadata,
        });
        let previous = self.nodes[parent].children.insert(first, id);
        assert!(previous.is_none(), "compressed child already exists");
        id
    }

    /// Split `node_id` at `split_at`, preserving `node_id` for the suffix.
    pub(super) fn split_keep_suffix(
        &mut self,
        node_id: CompressedNodeId,
        split_at: usize,
        prefix_metadata: M,
    ) -> CompressedNodeId {
        let (old_parent, old_first, prefix_edge, suffix_first) = {
            let node = self
                .nodes
                .get_mut(node_id)
                .expect("compressed split node is missing");
            assert!(
                split_at > 0 && split_at < node.edge.len(),
                "compressed split must be inside the edge"
            );
            let old_parent = node.parent;
            let old_first = node.edge[0];
            let suffix = node.edge.split_off(split_at);
            let prefix = std::mem::replace(&mut node.edge, suffix);
            let suffix_first = node.edge[0];
            (old_parent, old_first, prefix, suffix_first)
        };

        let prefix_id = self.nodes.insert(CompressedPathNode {
            edge: prefix_edge,
            parent: old_parent,
            children: FxHashMap::from_iter([(suffix_first, node_id)]),
            metadata: prefix_metadata,
        });
        self.nodes[node_id].parent = Some(prefix_id);

        if let Some(parent_id) = old_parent {
            let replaced = self.nodes[parent_id].children.insert(old_first, prefix_id);
            assert_eq!(
                replaced,
                Some(node_id),
                "compressed parent did not reference the split node"
            );
        } else {
            let replaced = self.roots.insert(old_first, prefix_id);
            assert_eq!(
                replaced,
                Some(node_id),
                "compressed roots did not reference the split node"
            );
        }

        prefix_id
    }

    /// Remove an empty leaf and detach it from its parent/root map.
    pub(super) fn remove_leaf(&mut self, node_id: CompressedNodeId) -> CompressedPathNode<M> {
        let node = self
            .nodes
            .remove(node_id)
            .expect("compressed leaf is missing");
        assert!(node.children.is_empty(), "cannot remove a non-leaf node");
        let first = node.edge[0];
        if let Some(parent_id) = node.parent {
            let removed = self.nodes[parent_id].children.remove(&first);
            assert_eq!(
                removed,
                Some(node_id),
                "compressed parent did not reference the removed leaf"
            );
        } else {
            let removed = self.roots.remove(&first);
            assert_eq!(
                removed,
                Some(node_id),
                "compressed roots did not reference the removed leaf"
            );
        }
        node
    }

    /// Merge a nonterminal unary parent into its child, preserving `child_id`.
    pub(super) fn merge_parent_into_child(
        &mut self,
        parent_id: CompressedNodeId,
        child_id: CompressedNodeId,
    ) {
        let parent = self
            .nodes
            .remove(parent_id)
            .expect("compressed merge parent is missing");
        assert_eq!(parent.children.len(), 1, "merge parent must be unary");
        assert_eq!(
            parent.children.values().next().copied(),
            Some(child_id),
            "merge parent does not reference the selected child"
        );

        let parent_parent = parent.parent;
        let old_first = parent.edge[0];
        let child = self
            .nodes
            .get_mut(child_id)
            .expect("compressed merge child is missing");
        assert_eq!(
            child.parent,
            Some(parent_id),
            "compressed merge child has a different parent"
        );
        let mut merged_edge = parent.edge;
        merged_edge.append(&mut child.edge);
        child.edge = merged_edge;
        child.parent = parent_parent;

        if let Some(grandparent_id) = parent_parent {
            let replaced = self.nodes[grandparent_id]
                .children
                .insert(old_first, child_id);
            assert_eq!(
                replaced,
                Some(parent_id),
                "compressed grandparent did not reference the merge parent"
            );
        } else {
            let replaced = self.roots.insert(old_first, child_id);
            assert_eq!(
                replaced,
                Some(parent_id),
                "compressed roots did not reference the merge parent"
            );
        }
    }

    pub(super) fn path_from_root(&self, tail: CompressedNodeId) -> Vec<CompressedNodeId> {
        let mut path = Vec::new();
        let mut current = Some(tail);
        while let Some(node_id) = current {
            let node = self
                .nodes
                .get(node_id)
                .expect("compressed path references a missing node");
            path.push(node_id);
            current = node.parent;
        }
        path.reverse();
        path
    }

    #[cfg(test)]
    pub(super) fn hash_count(&self) -> usize {
        self.nodes.values().map(|node| node.edge.len()).sum()
    }

    #[cfg(any(test, debug_assertions))]
    pub(super) fn assert_topology(&self) {
        for (&first, &root_id) in &self.roots {
            let root = self
                .nodes
                .get(root_id)
                .expect("compressed roots reference a missing node");
            assert_eq!(root.parent, None, "compressed root has a parent");
            assert_eq!(
                root.edge.first(),
                Some(&first),
                "compressed root key mismatch"
            );
        }

        for (node_id, node) in &self.nodes {
            assert!(!node.edge.is_empty(), "compressed node edge is empty");
            match node.parent {
                Some(parent_id) => {
                    let parent = self
                        .nodes
                        .get(parent_id)
                        .expect("compressed node parent is missing");
                    assert_eq!(
                        parent.children.get(&node.edge[0]),
                        Some(&node_id),
                        "compressed parent-child link mismatch"
                    );
                }
                None => assert_eq!(
                    self.roots.get(&node.edge[0]),
                    Some(&node_id),
                    "compressed root link mismatch"
                ),
            }
            for (&first, &child_id) in &node.children {
                let child = self
                    .nodes
                    .get(child_id)
                    .expect("compressed child link is stale");
                assert_eq!(
                    child.parent,
                    Some(node_id),
                    "compressed child parent mismatch"
                );
                assert_eq!(
                    child.edge.first(),
                    Some(&first),
                    "compressed child key mismatch"
                );
            }
        }
    }
}
