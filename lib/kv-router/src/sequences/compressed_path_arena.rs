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

/// A single-owner compressed path forest with stable generational handles.
#[derive(Debug)]
pub(super) struct CompressedPathArena<M> {
    pub(super) nodes: SlotMap<CompressedNodeId, CompressedPathNode<M>>,
    pub(super) roots: FxHashMap<SequenceHash, CompressedNodeId>,
    pub(super) splits: u64,
    pub(super) merges: u64,
}

impl<M> Default for CompressedPathArena<M> {
    fn default() -> Self {
        Self {
            nodes: SlotMap::with_key(),
            roots: FxHashMap::default(),
            splits: 0,
            merges: 0,
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
            "compressed parent is missing"
        );
        let first = edge[0];
        let id = self.nodes.insert(CompressedPathNode {
            edge,
            parent: Some(parent),
            children: FxHashMap::default(),
            metadata,
        });
        assert!(
            self.nodes[parent].children.insert(first, id).is_none(),
            "compressed child already exists"
        );
        id
    }

    /// Split `node_id` while retaining that ID for the suffix.
    pub(super) fn split_keep_suffix(
        &mut self,
        node_id: CompressedNodeId,
        split_at: usize,
        prefix_metadata: M,
    ) -> CompressedNodeId {
        let (old_parent, old_first, prefix_edge, suffix_first) = {
            let node = self.nodes.get_mut(node_id).expect("split node is missing");
            assert!(
                split_at > 0 && split_at < node.edge.len(),
                "split must be inside the edge"
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
        self.splits = self.splits.saturating_add(1);
        if let Some(parent_id) = old_parent {
            assert_eq!(
                self.nodes[parent_id].children.insert(old_first, prefix_id),
                Some(node_id),
                "parent did not reference split node"
            );
        } else {
            assert_eq!(
                self.roots.insert(old_first, prefix_id),
                Some(node_id),
                "roots did not reference split node"
            );
        }
        prefix_id
    }

    pub(super) fn remove_leaf(&mut self, node_id: CompressedNodeId) -> CompressedPathNode<M> {
        let node = self.nodes.remove(node_id).expect("leaf is missing");
        assert!(node.children.is_empty(), "cannot remove a non-leaf");
        let first = node.edge[0];
        if let Some(parent_id) = node.parent {
            assert_eq!(
                self.nodes[parent_id].children.remove(&first),
                Some(node_id),
                "parent did not reference removed leaf"
            );
        } else {
            assert_eq!(
                self.roots.remove(&first),
                Some(node_id),
                "roots did not reference removed leaf"
            );
        }
        node
    }

    /// Merge a unary parent into its child while retaining the child ID.
    pub(super) fn merge_parent_into_child(
        &mut self,
        parent_id: CompressedNodeId,
        child_id: CompressedNodeId,
    ) {
        let parent = self
            .nodes
            .remove(parent_id)
            .expect("merge parent is missing");
        assert_eq!(parent.children.len(), 1, "merge parent must be unary");
        assert_eq!(
            parent.children.values().next().copied(),
            Some(child_id),
            "merge parent references another child"
        );
        let parent_parent = parent.parent;
        let old_first = parent.edge[0];
        let child = self
            .nodes
            .get_mut(child_id)
            .expect("merge child is missing");
        assert_eq!(child.parent, Some(parent_id), "merge child parent mismatch");
        let mut merged_edge = parent.edge;
        merged_edge.append(&mut child.edge);
        child.edge = merged_edge;
        child.parent = parent_parent;
        self.merges = self.merges.saturating_add(1);
        if let Some(grandparent_id) = parent_parent {
            assert_eq!(
                self.nodes[grandparent_id]
                    .children
                    .insert(old_first, child_id),
                Some(parent_id),
                "grandparent did not reference merge parent"
            );
        } else {
            assert_eq!(
                self.roots.insert(old_first, child_id),
                Some(parent_id),
                "roots did not reference merge parent"
            );
        }
    }

    pub(super) fn path_from_root(&self, tail: CompressedNodeId) -> Vec<CompressedNodeId> {
        let mut path = Vec::new();
        let mut current = Some(tail);
        while let Some(node_id) = current {
            let node = self.nodes.get(node_id).expect("path node is missing");
            path.push(node_id);
            current = node.parent;
        }
        path.reverse();
        path
    }

    #[cfg(any(test, debug_assertions))]
    pub(super) fn assert_topology(&self) {
        for (&first, &root_id) in &self.roots {
            let root = self.nodes.get(root_id).expect("root node is missing");
            assert_eq!(root.parent, None, "root has a parent");
            assert_eq!(root.edge.first(), Some(&first), "root key mismatch");
        }
        for (node_id, node) in &self.nodes {
            assert!(!node.edge.is_empty(), "compressed edge is empty");
            match node.parent {
                Some(parent_id) => assert_eq!(
                    self.nodes[parent_id].children.get(&node.edge[0]),
                    Some(&node_id),
                    "parent-child link mismatch"
                ),
                None => assert_eq!(
                    self.roots.get(&node.edge[0]),
                    Some(&node_id),
                    "root link mismatch"
                ),
            }
            for (&first, &child_id) in &node.children {
                let child = self.nodes.get(child_id).expect("child link is stale");
                assert_eq!(child.parent, Some(node_id), "child parent mismatch");
                assert_eq!(child.edge.first(), Some(&first), "child key mismatch");
            }
        }
    }
}
