// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_tokens::{PositionalLineageHash, SequenceHash};
use rustc_hash::FxHashMap;
#[cfg(debug_assertions)]
use rustc_hash::FxHashSet;
use slotmap::{SlotMap, new_key_type};
use std::collections::hash_map::Entry;
use std::num::NonZeroU32;

new_key_type! {
    struct BlockNodeId;
}

/// One node in a persistent request block chain.
///
/// Positional lineage hashes distinguish the same sequence hash at different
/// positions or beneath different parent-hash fragments. Parent IDs are
/// generational arena keys, while `incoming` counts direct request-tail and
/// child-to-parent edges.
#[derive(Debug)]
struct BlockNode {
    lineage_hash: PositionalLineageHash,
    parent: Option<BlockNodeId>,
    incoming: NonZeroU32,
}

impl BlockNode {
    #[inline]
    fn sequence_hash(&self) -> SequenceHash {
        self.lineage_hash.current_sequence_hash()
    }

    #[inline]
    fn depth(&self) -> usize {
        usize::try_from(self.lineage_hash.position())
            .expect("block position does not fit usize")
            .checked_add(1)
            .expect("block chain depth overflowed")
    }
}

/// The single block-chain tail retained by a request.
///
/// This type intentionally does not implement `Clone`. New request owners must
/// be acquired through [`BlockTracker::acquire_prompt`], which also maintains
/// the arena ownership counts and block index.
#[derive(Debug, Default)]
#[must_use = "request block chains must be retained by a request and released through BlockTracker"]
pub(super) struct RequestBlockChain {
    tail: Option<BlockNodeId>,
    prompt_depth: usize,
}

impl RequestBlockChain {
    fn new(tail: Option<BlockNodeId>, prompt_depth: usize) -> Self {
        Self { tail, prompt_depth }
    }
}

#[derive(Debug, Default)]
pub(super) struct BlockTracker {
    nodes: SlotMap<BlockNodeId, BlockNode>,
    unique_blocks: FxHashMap<PositionalLineageHash, BlockNodeId>,
    fractional_blocks: FxHashMap<PositionalLineageHash, f64>,
}

impl BlockTracker {
    /// Acquire one request tail for a prompt and return the first newly present
    /// prompt index, if any.
    ///
    /// The internal index augments each [`SequenceHash`] with its position and
    /// parent-hash fragment. Public membership events continue to use the raw
    /// sequence hashes.
    pub(super) fn acquire_prompt(
        &mut self,
        sequence: &[SequenceHash],
    ) -> (RequestBlockChain, Option<usize>) {
        let Some(tail_idx) = sequence.len().checked_sub(1) else {
            return (RequestBlockChain::default(), None);
        };
        let tail_hash = Self::prompt_lineage_hash(sequence, tail_idx);

        if let Some(tail) = self.live_node_id(tail_hash) {
            self.debug_assert_lineage(tail, sequence);
            self.increment_incoming(tail);
            return (RequestBlockChain::new(Some(tail), sequence.len()), None);
        }

        let mut parent = None;
        let mut first_new_idx = 0;

        // Prompt liveness is prefix-closed. Search backward for the deepest
        // live prefix, then construct only the missing suffix.
        for idx in (0..sequence.len().saturating_sub(1)).rev() {
            let lineage_hash = Self::prompt_lineage_hash(sequence, idx);
            if let Some(node_id) = self.live_node_id(lineage_hash) {
                self.debug_assert_lineage(node_id, &sequence[..=idx]);
                parent = Some(node_id);
                first_new_idx = idx + 1;
                break;
            }
        }

        let missing = sequence.len() - first_new_idx;
        self.nodes.reserve(missing);
        self.unique_blocks.reserve(missing);
        self.debug_assert_new_hashes(sequence, first_new_idx);

        // Adding the first new child contributes one new incoming edge to the
        // reused prefix. Subsequent construction transfers the provisional
        // request-tail edge into a child-to-parent edge without changing it.
        if let Some(parent_id) = parent {
            self.increment_incoming(parent_id);
        }

        for idx in first_new_idx..sequence.len() {
            let lineage_hash = Self::prompt_lineage_hash(sequence, idx);
            let node_id = self.nodes.insert(BlockNode {
                lineage_hash,
                parent,
                incoming: NonZeroU32::MIN,
            });
            let previous = self.unique_blocks.insert(lineage_hash, node_id);
            debug_assert!(
                previous.is_none(),
                "new block unexpectedly replaced a live index entry"
            );
            parent = Some(node_id);
        }

        (
            RequestBlockChain::new(parent, sequence.len()),
            Some(first_new_idx),
        )
    }

    /// Append a unique output block to an existing request chain.
    pub(super) fn append_output(&mut self, chain: &mut RequestBlockChain, hash: SequenceHash) {
        let parent = chain.tail;
        let (parent_hash, position) = parent.map_or((None, 0), |node_id| {
            let node = self
                .nodes
                .get(node_id)
                .expect("request tail references a missing block node");
            let position = node
                .lineage_hash
                .position()
                .checked_add(1)
                .expect("block position overflowed");
            (Some(node.sequence_hash()), position)
        });
        let lineage_hash = PositionalLineageHash::new(hash, parent_hash, position);

        debug_assert!(
            !self.unique_blocks.contains_key(&lineage_hash),
            "random output hash unexpectedly collided with a live block"
        );

        let node_id = self.nodes.insert(BlockNode {
            lineage_hash,
            parent,
            incoming: NonZeroU32::MIN,
        });
        let previous = self.unique_blocks.insert(lineage_hash, node_id);
        debug_assert!(
            previous.is_none(),
            "new output unexpectedly replaced a live index entry"
        );
        chain.tail = Some(node_id);
    }

    /// Release a request chain and return the prompt suffix that became absent
    /// from the worker.
    pub(super) fn release(&mut self, chain: RequestBlockChain) -> Vec<SequenceHash> {
        let RequestBlockChain { tail, prompt_depth } = chain;
        let mut prompt_remove = Vec::new();
        let mut current = tail;

        while let Some(node_id) = current {
            let node = self
                .nodes
                .get(node_id)
                .expect("request chain references a missing block node");
            let incoming = node.incoming.get();

            if incoming > 1 {
                self.nodes
                    .get_mut(node_id)
                    .expect("request chain references a missing block node")
                    .incoming = NonZeroU32::new(incoming - 1)
                    .expect("shared block ownership count cannot become zero");
                break;
            }

            let node = self
                .nodes
                .get(node_id)
                .expect("request chain references a missing block node");
            let lineage_hash = node.lineage_hash;
            let hash = node.sequence_hash();
            let depth = node.depth();
            let parent = node.parent;

            match self.unique_blocks.entry(lineage_hash) {
                Entry::Occupied(entry) => {
                    assert_eq!(
                        *entry.get(),
                        node_id,
                        "block hash index references a different live node"
                    );
                    entry.remove();
                }
                Entry::Vacant(_) => panic!("live block node is missing from the hash index"),
            }
            self.fractional_blocks.remove(&lineage_hash);
            self.nodes
                .remove(node_id)
                .expect("validated block node disappeared before removal");

            if depth <= prompt_depth {
                prompt_remove.push(hash);
            }
            current = parent;
        }

        prompt_remove.reverse();
        prompt_remove
    }

    /// Mark the request's exclusively owned suffix as fractional.
    pub(super) fn set_unique_suffix_fractional(
        &mut self,
        chain: &RequestBlockChain,
        fraction: f64,
    ) {
        let mut current = chain.tail;
        while let Some(node_id) = current {
            let node = self
                .nodes
                .get(node_id)
                .expect("request chain references a missing block node");
            if node.incoming.get() != 1 {
                break;
            }
            self.fractional_blocks.insert(node.lineage_hash, fraction);
            current = node.parent;
        }
    }

    pub(super) fn active_blocks(&self) -> usize {
        let mut count = self.unique_blocks.len() as f64;
        for (hash, frac) in &self.fractional_blocks {
            if self.unique_blocks.contains_key(hash) {
                count = count - 1.0 + frac;
            }
        }
        count.round() as usize
    }

    pub(super) fn contains_prompt_block(&self, sequence: &[SequenceHash], index: usize) -> bool {
        self.live_node_id(Self::prompt_lineage_hash(sequence, index))
            .is_some()
    }

    #[cfg(any(test, debug_assertions))]
    pub(super) fn assert_consistent<'a>(
        &self,
        chains: impl IntoIterator<Item = &'a RequestBlockChain>,
    ) {
        assert_eq!(
            self.nodes.len(),
            self.unique_blocks.len(),
            "arena and live block index must have identical cardinality"
        );

        let mut expected_incoming = FxHashMap::default();
        for (node_id, node) in &self.nodes {
            assert!(
                expected_incoming.insert(node_id, 0_u32).is_none(),
                "arena yielded a duplicate node ID"
            );
            assert_eq!(
                self.unique_blocks.get(&node.lineage_hash),
                Some(&node_id),
                "live arena node is missing its exact lineage-index entry"
            );

            if let Some(parent_id) = node.parent {
                let parent = self
                    .nodes
                    .get(parent_id)
                    .expect("block node references a missing parent");
                assert_eq!(
                    parent.depth() + 1,
                    node.depth(),
                    "child block depth must immediately follow its parent"
                );
            } else {
                assert_eq!(node.depth(), 1, "root block depth must be one");
            }
        }

        for (&lineage_hash, &node_id) in &self.unique_blocks {
            let node = self
                .nodes
                .get(node_id)
                .expect("lineage index references a missing arena node");
            assert_eq!(
                node.lineage_hash, lineage_hash,
                "lineage index key does not match its arena node"
            );
        }

        for (_, node) in &self.nodes {
            if let Some(parent_id) = node.parent {
                let count = expected_incoming
                    .get_mut(&parent_id)
                    .expect("block node references a missing parent");
                *count = count
                    .checked_add(1)
                    .expect("reconstructed child ownership count overflowed");
            }
        }

        for chain in chains {
            if let Some(tail_id) = chain.tail {
                let tail = self
                    .nodes
                    .get(tail_id)
                    .expect("request tail references a missing arena node");
                assert!(
                    chain.prompt_depth <= tail.depth(),
                    "request prompt depth cannot exceed its full block-chain depth"
                );
                let count = expected_incoming
                    .get_mut(&tail_id)
                    .expect("request tail references a missing arena node");
                *count = count
                    .checked_add(1)
                    .expect("reconstructed request ownership count overflowed");
            } else {
                assert_eq!(
                    chain.prompt_depth, 0,
                    "an empty request chain cannot retain prompt depth"
                );
            }
        }

        for (node_id, node) in &self.nodes {
            assert_eq!(
                expected_incoming[&node_id],
                node.incoming.get(),
                "stored incoming ownership count differs from reconstructed edges"
            );
        }

        assert!(
            self.fractional_blocks
                .keys()
                .all(|hash| self.unique_blocks.contains_key(hash)),
            "fractional blocks cannot reference non-active blocks"
        );
    }

    #[cfg(test)]
    pub(super) fn active_hashes(&self) -> impl Iterator<Item = SequenceHash> + '_ {
        self.unique_blocks
            .keys()
            .map(PositionalLineageHash::current_sequence_hash)
    }

    #[cfg(test)]
    pub(super) fn prompt_hashes<'a>(
        &'a self,
        chain: &'a RequestBlockChain,
    ) -> impl Iterator<Item = SequenceHash> + 'a {
        self.node_ids_from_tail(chain).filter_map(|node_id| {
            let node = &self.nodes[node_id];
            (node.depth() <= chain.prompt_depth).then(|| node.sequence_hash())
        })
    }

    fn live_node_id(&self, lineage_hash: PositionalLineageHash) -> Option<BlockNodeId> {
        let &node_id = self.unique_blocks.get(&lineage_hash)?;
        let node = self
            .nodes
            .get(node_id)
            .expect("live lineage index references a stale arena ID");
        assert_eq!(
            node.lineage_hash, lineage_hash,
            "live lineage index key does not match its arena node"
        );
        Some(node_id)
    }

    #[inline]
    fn prompt_lineage_hash(sequence: &[SequenceHash], index: usize) -> PositionalLineageHash {
        let current = sequence[index];
        let parent = index.checked_sub(1).map(|parent_idx| sequence[parent_idx]);
        let position = u64::try_from(index).expect("block position does not fit u64");
        PositionalLineageHash::new(current, parent, position)
    }

    fn increment_incoming(&mut self, node_id: BlockNodeId) {
        let node = self
            .nodes
            .get_mut(node_id)
            .expect("cannot acquire ownership of a missing block node");
        let incoming = node
            .incoming
            .get()
            .checked_add(1)
            .expect("block ownership count overflowed");
        node.incoming = NonZeroU32::new(incoming).expect("incremented ownership cannot be zero");
    }

    #[cfg(debug_assertions)]
    fn debug_assert_new_hashes(&self, sequence: &[SequenceHash], first_new_idx: usize) {
        let mut seen = FxHashSet::default();
        for idx in first_new_idx..sequence.len() {
            let lineage_hash = Self::prompt_lineage_hash(sequence, idx);
            assert!(
                !self.unique_blocks.contains_key(&lineage_hash),
                "sequence lineage hash unexpectedly aliases a live block"
            );
            assert!(
                seen.insert(lineage_hash),
                "sequence lineage repeats a hash in one missing suffix"
            );
        }
    }

    #[cfg(not(debug_assertions))]
    #[inline]
    fn debug_assert_new_hashes(&self, _sequence: &[SequenceHash], _first_new_idx: usize) {}

    #[cfg(any(test, debug_assertions))]
    fn debug_assert_lineage(&self, tail: BlockNodeId, sequence: &[SequenceHash]) {
        let tail_node = self
            .nodes
            .get(tail)
            .expect("live sequence tail references a missing arena node");
        assert_eq!(
            tail_node.depth(),
            sequence.len(),
            "sequence tail depth mismatch"
        );
        let mut current = Some(tail);
        for (expected_depth, &expected_hash) in sequence.iter().enumerate().rev() {
            let node_id = current.expect("live sequence chain ended before its expected root");
            let node = self
                .nodes
                .get(node_id)
                .expect("live sequence chain references a missing arena node");
            assert_eq!(node.depth(), expected_depth + 1, "sequence depth mismatch");
            assert_eq!(
                node.lineage_hash,
                Self::prompt_lineage_hash(sequence, expected_depth),
                "positional sequence lineage hash mismatch"
            );
            assert_eq!(
                node.sequence_hash(),
                expected_hash,
                "sequence lineage hash mismatch"
            );
            current = node.parent;
        }
    }

    #[cfg(not(any(test, debug_assertions)))]
    #[inline]
    fn debug_assert_lineage(&self, _tail: BlockNodeId, _sequence: &[SequenceHash]) {}

    #[cfg(test)]
    fn node_ids_from_tail<'a>(
        &'a self,
        chain: &'a RequestBlockChain,
    ) -> impl Iterator<Item = BlockNodeId> + 'a {
        let mut current = chain.tail;
        std::iter::from_fn(move || {
            let node_id = current?;
            let node = self
                .nodes
                .get(node_id)
                .expect("request chain references a missing arena node");
            current = node.parent;
            Some(node_id)
        })
    }

    #[cfg(test)]
    fn incoming_for(&self, hash: SequenceHash) -> u32 {
        let node_id = self.node_id_for(hash);
        self.nodes[node_id].incoming.get()
    }

    #[cfg(test)]
    fn node_id_for(&self, hash: SequenceHash) -> BlockNodeId {
        let mut matches = self.unique_blocks.iter().filter_map(|(lineage, &node_id)| {
            (lineage.current_sequence_hash() == hash).then_some(node_id)
        });
        let node_id = matches.next().expect("expected live block hash");
        assert!(
            matches.next().is_none(),
            "expected raw block hash to identify exactly one live lineage"
        );
        node_id
    }

    #[cfg(test)]
    fn node_id_for_prompt(&self, sequence: &[SequenceHash], index: usize) -> BlockNodeId {
        self.live_node_id(Self::prompt_lineage_hash(sequence, index))
            .expect("expected live prompt lineage")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rustc_hash::FxHashSet;

    #[test]
    fn first_acquire_and_last_release_report_presence_transitions() {
        let mut tracker = BlockTracker::default();

        let (first, first_new) = tracker.acquire_prompt(&[1]);
        let (second, second_new) = tracker.acquire_prompt(&[1]);

        assert_eq!(first_new, Some(0));
        assert_eq!(second_new, None);
        assert_eq!(tracker.incoming_for(1), 2);
        tracker.assert_consistent([&first, &second]);
        assert_eq!(tracker.active_blocks(), 1);

        assert!(tracker.release(first).is_empty());
        assert_eq!(tracker.incoming_for(1), 1);
        tracker.assert_consistent([&second]);
        assert_eq!(tracker.active_blocks(), 1);

        assert_eq!(tracker.release(second), vec![1]);
        tracker.assert_consistent(std::iter::empty());
        assert_eq!(tracker.active_blocks(), 0);
    }

    #[test]
    fn shorter_request_adds_only_a_tail_edge() {
        let mut tracker = BlockTracker::default();
        let (longer, _) = tracker.acquire_prompt(&[1, 2, 3]);
        let (shorter, first_new) = tracker.acquire_prompt(&[1, 2]);

        assert_eq!(first_new, None);
        assert_eq!(tracker.incoming_for(1), 1);
        assert_eq!(tracker.incoming_for(2), 2);
        assert_eq!(tracker.incoming_for(3), 1);
        tracker.assert_consistent([&longer, &shorter]);

        assert!(tracker.release(shorter).is_empty());
        tracker.assert_consistent([&longer]);
        assert_eq!(tracker.release(longer), vec![1, 2, 3]);
    }

    #[test]
    fn longer_request_adds_one_child_edge_to_a_shorter_tail() {
        let mut tracker = BlockTracker::default();
        let (shorter, _) = tracker.acquire_prompt(&[1, 2]);
        let (longer, first_new) = tracker.acquire_prompt(&[1, 2, 3]);

        assert_eq!(first_new, Some(2));
        assert_eq!(tracker.incoming_for(1), 1);
        assert_eq!(tracker.incoming_for(2), 2);
        assert_eq!(tracker.incoming_for(3), 1);
        tracker.assert_consistent([&shorter, &longer]);

        assert_eq!(tracker.release(longer), vec![3]);
        tracker.assert_consistent([&shorter]);
        assert_eq!(tracker.release(shorter), vec![1, 2]);
    }

    #[test]
    fn release_only_removes_unique_branch_suffix() {
        let mut tracker = BlockTracker::default();
        let (left, _) = tracker.acquire_prompt(&[1, 2, 3]);
        let (right, first_new) = tracker.acquire_prompt(&[1, 2, 4]);

        assert_eq!(first_new, Some(2));
        assert_eq!(tracker.incoming_for(1), 1);
        assert_eq!(tracker.incoming_for(2), 2);
        tracker.assert_consistent([&left, &right]);
        assert_eq!(tracker.active_blocks(), 4);

        assert_eq!(tracker.release(left), vec![3]);
        tracker.assert_consistent([&right]);
        assert_eq!(tracker.active_blocks(), 3);
        assert_eq!(tracker.release(right), vec![1, 2, 4]);
        tracker.assert_consistent(std::iter::empty());
        assert_eq!(tracker.active_blocks(), 0);
    }

    #[test]
    fn output_append_transfers_tail_ownership_to_the_child_edge() {
        let mut tracker = BlockTracker::default();
        let (mut chain, _) = tracker.acquire_prompt(&[1, 2]);

        let before = tracker.incoming_for(2);
        tracker.append_output(&mut chain, 42);

        assert_eq!(tracker.incoming_for(2), before);
        assert_eq!(tracker.incoming_for(42), 1);
        let output = &tracker.nodes[tracker.node_id_for(42)];
        assert_eq!(output.lineage_hash.position(), 2);
        assert_eq!(output.lineage_hash.current_sequence_hash(), 42);
        assert_eq!(output.lineage_hash.parent_hash_fragment(), 2);
        tracker.assert_consistent([&chain]);
        assert_eq!(tracker.release(chain), vec![1, 2]);
    }

    #[test]
    fn same_tail_hash_under_different_parents_has_independent_lineages() {
        let left_sequence = [1, 9];
        let right_sequence = [2, 9];
        let mut tracker = BlockTracker::default();
        let (left, left_first_new) = tracker.acquire_prompt(&left_sequence);
        let (right, right_first_new) = tracker.acquire_prompt(&right_sequence);

        assert_eq!(left_first_new, Some(0));
        assert_eq!(right_first_new, Some(0));
        assert_ne!(
            tracker.node_id_for_prompt(&left_sequence, 1),
            tracker.node_id_for_prompt(&right_sequence, 1)
        );
        assert_eq!(tracker.active_blocks(), 4);

        tracker.set_unique_suffix_fractional(&left, 0.5);
        assert_eq!(tracker.active_blocks(), 3);
        tracker.assert_consistent([&left, &right]);

        assert_eq!(tracker.release(left), vec![1, 9]);
        assert_eq!(tracker.active_blocks(), 2);
        tracker.assert_consistent([&right]);
        assert_eq!(tracker.release(right), vec![2, 9]);
        tracker.assert_consistent(std::iter::empty());
    }

    #[test]
    fn same_hash_at_different_depths_has_independent_lineages() {
        let root_sequence = [9];
        let nested_sequence = [1, 9];
        let mut tracker = BlockTracker::default();
        let (root, _) = tracker.acquire_prompt(&root_sequence);
        let (nested, nested_first_new) = tracker.acquire_prompt(&nested_sequence);

        assert_eq!(nested_first_new, Some(0));
        assert_ne!(
            tracker.node_id_for_prompt(&root_sequence, 0),
            tracker.node_id_for_prompt(&nested_sequence, 1)
        );
        assert_eq!(tracker.active_blocks(), 3);
        tracker.assert_consistent([&root, &nested]);

        assert_eq!(tracker.release(root), vec![9]);
        tracker.assert_consistent([&nested]);
        assert_eq!(tracker.release(nested), vec![1, 9]);
    }

    #[test]
    fn output_append_preserves_a_shared_tail_count() {
        let mut tracker = BlockTracker::default();
        let (mut generating, _) = tracker.acquire_prompt(&[1, 2]);
        let (shared, _) = tracker.acquire_prompt(&[1, 2]);

        assert_eq!(tracker.incoming_for(2), 2);
        tracker.append_output(&mut generating, 42);
        assert_eq!(tracker.incoming_for(2), 2);
        tracker.assert_consistent([&generating, &shared]);

        assert!(tracker.release(generating).is_empty());
        assert_eq!(tracker.incoming_for(2), 1);
        tracker.assert_consistent([&shared]);
        assert_eq!(tracker.release(shared), vec![1, 2]);
    }

    #[test]
    fn either_branch_can_be_released_first() {
        let mut tracker = BlockTracker::default();
        let (left, _) = tracker.acquire_prompt(&[1, 2, 3]);
        let (right, _) = tracker.acquire_prompt(&[1, 2, 4]);

        assert_eq!(tracker.release(right), vec![4]);
        tracker.assert_consistent([&left]);
        assert_eq!(tracker.release(left), vec![1, 2, 3]);
    }

    #[test]
    fn fractional_blocks_adjust_active_block_count() {
        let mut tracker = BlockTracker::default();
        let (chain, _) = tracker.acquire_prompt(&[1, 2]);

        tracker.set_unique_suffix_fractional(&chain, 0.5);
        tracker.assert_consistent([&chain]);
        assert_eq!(tracker.active_blocks(), 1);

        assert_eq!(tracker.release(chain), vec![1, 2]);
        assert!(tracker.fractional_blocks.is_empty());
        tracker.assert_consistent(std::iter::empty());
        assert_eq!(tracker.active_blocks(), 0);
    }

    #[test]
    fn release_cleans_fractional_unique_suffix_above_shared_prefix() {
        let mut tracker = BlockTracker::default();
        let (longer, _) = tracker.acquire_prompt(&[1, 2, 3]);
        let (shared_prefix, _) = tracker.acquire_prompt(&[1, 2]);

        tracker.set_unique_suffix_fractional(&longer, 0.5);
        let sequence = [1, 2, 3];
        assert_eq!(
            tracker
                .fractional_blocks
                .get(&BlockTracker::prompt_lineage_hash(&sequence, 2)),
            Some(&0.5)
        );
        assert!(
            !tracker
                .fractional_blocks
                .contains_key(&BlockTracker::prompt_lineage_hash(&sequence, 0))
        );
        assert!(
            !tracker
                .fractional_blocks
                .contains_key(&BlockTracker::prompt_lineage_hash(&sequence, 1))
        );
        tracker.assert_consistent([&longer, &shared_prefix]);

        assert_eq!(tracker.release(longer), vec![3]);
        assert!(tracker.fractional_blocks.is_empty());
        tracker.assert_consistent([&shared_prefix]);
        assert_eq!(tracker.active_blocks(), 2);

        assert_eq!(tracker.release(shared_prefix), vec![1, 2]);
        assert_eq!(tracker.active_blocks(), 0);
    }

    #[test]
    fn output_only_chain_releases_without_prompt_membership() {
        let mut tracker = BlockTracker::default();
        let (mut chain, first_new) = tracker.acquire_prompt(&[]);

        assert_eq!(first_new, None);
        tracker.append_output(&mut chain, 42);
        tracker.assert_consistent([&chain]);
        assert_eq!(tracker.active_blocks(), 1);
        assert!(tracker.release(chain).is_empty());
        tracker.assert_consistent(std::iter::empty());
        assert_eq!(tracker.active_blocks(), 0);
    }

    #[test]
    fn removed_slot_generation_cannot_resolve_after_reuse() {
        let mut tracker = BlockTracker::default();
        let (first, _) = tracker.acquire_prompt(&[1]);
        let old_id = tracker.node_id_for(1);
        assert_eq!(tracker.release(first), vec![1]);

        let (second, _) = tracker.acquire_prompt(&[2]);
        assert!(tracker.nodes.get(old_id).is_none());
        assert_ne!(old_id, tracker.node_id_for(2));
        tracker.assert_consistent([&second]);
        assert_eq!(tracker.release(second), vec![2]);
    }

    #[test]
    #[should_panic(expected = "live lineage index references a stale arena ID")]
    fn stale_generation_in_index_is_rejected() {
        let mut tracker = BlockTracker::default();
        let (first, _) = tracker.acquire_prompt(&[1]);
        let old_id = tracker.node_id_for(1);
        assert_eq!(tracker.release(first), vec![1]);
        let (second, _) = tracker.acquire_prompt(&[2]);

        let wrong_lineage = BlockTracker::prompt_lineage_hash(&[99], 0);
        tracker.unique_blocks.insert(wrong_lineage, old_id);
        let _ = tracker.live_node_id(wrong_lineage);
        drop(second);
    }

    #[test]
    #[should_panic(expected = "live lineage index key does not match its arena node")]
    fn live_id_with_the_wrong_hash_is_rejected() {
        let mut tracker = BlockTracker::default();
        let (chain, _) = tracker.acquire_prompt(&[1]);
        let node_id = tracker.node_id_for(1);

        let wrong_lineage = BlockTracker::prompt_lineage_hash(&[2], 0);
        tracker.unique_blocks.insert(wrong_lineage, node_id);
        let _ = tracker.live_node_id(wrong_lineage);
        drop(chain);
    }

    #[test]
    fn positional_lineage_modes_cover_prompt_size_boundaries() {
        let sequence = (1..=65_537_u64).collect::<Vec<_>>();

        assert_eq!(BlockTracker::prompt_lineage_hash(&sequence, 255).mode(), 0);
        assert_eq!(BlockTracker::prompt_lineage_hash(&sequence, 256).mode(), 1);
        assert_eq!(
            BlockTracker::prompt_lineage_hash(&sequence, 65_535).mode(),
            1
        );
        assert_eq!(
            BlockTracker::prompt_lineage_hash(&sequence, 65_536).mode(),
            2
        );
    }

    #[test]
    #[should_panic(expected = "block ownership count overflowed")]
    fn ownership_count_overflow_panics() {
        let mut tracker = BlockTracker::default();
        let (chain, _) = tracker.acquire_prompt(&[1]);
        let node_id = tracker.node_id_for(1);
        tracker.nodes[node_id].incoming = NonZeroU32::MAX;
        let _ = tracker.acquire_prompt(&[1]);
        drop(chain);
    }

    #[test]
    fn randomized_lifecycle_matches_reference_model() {
        const SLOTS: usize = 32;
        const STEPS: usize = 10_000;
        let prompts = [
            vec![1, 2, 3, 4],
            vec![1, 2, 3, 5],
            vec![1, 2, 6],
            vec![1, 7],
            vec![8, 9, 10],
            vec![8, 9, 11],
            vec![12],
        ];
        let mut rng = StdRng::seed_from_u64(0x5eed_51a7);
        let mut tracker = BlockTracker::default();
        let mut chains = (0..SLOTS).map(|_| None).collect::<Vec<_>>();
        let mut reference = (0..SLOTS)
            .map(|_| None::<(Vec<SequenceHash>, usize)>)
            .collect::<Vec<_>>();
        let mut counts = FxHashMap::<SequenceHash, usize>::default();
        let mut next_output = 1_000_000_u64;

        for _ in 0..STEPS {
            let slot = rng.random_range(0..SLOTS);
            match (&mut chains[slot], &mut reference[slot]) {
                (chain @ None, reference @ None) => {
                    let prompt = &prompts[rng.random_range(0..prompts.len())];
                    let expected_first_new =
                        prompt.iter().position(|hash| !counts.contains_key(hash));
                    let (new_chain, first_new) = tracker.acquire_prompt(prompt);
                    assert_eq!(first_new, expected_first_new);
                    for &hash in prompt {
                        *counts.entry(hash).or_default() += 1;
                    }
                    *chain = Some(new_chain);
                    *reference = Some((prompt.clone(), prompt.len()));
                }
                (Some(chain), Some((blocks, _prompt_depth))) if rng.random_bool(0.35) => {
                    let hash = next_output;
                    next_output += 1;
                    tracker.append_output(chain, hash);
                    blocks.push(hash);
                    counts.insert(hash, 1);
                }
                (chain @ Some(_), reference @ Some(_)) => {
                    let chain = chain.take().expect("matched active chain");
                    let (blocks, prompt_depth) = reference.take().expect("matched reference");
                    let expected_remove = blocks[..prompt_depth]
                        .iter()
                        .copied()
                        .filter(|hash| counts[hash] == 1)
                        .collect::<Vec<_>>();
                    assert_eq!(tracker.release(chain), expected_remove);
                    for hash in blocks {
                        let count = counts.get_mut(&hash).expect("reference block is active");
                        *count -= 1;
                        if *count == 0 {
                            counts.remove(&hash);
                        }
                    }
                }
                _ => unreachable!("tracker and reference occupancy diverged"),
            }

            tracker.assert_consistent(chains.iter().filter_map(Option::as_ref));
            let actual = tracker.active_hashes().collect::<FxHashSet<_>>();
            let expected = counts.keys().copied().collect::<FxHashSet<_>>();
            assert_eq!(actual, expected);
        }

        for slot in 0..SLOTS {
            if let Some(chain) = chains[slot].take() {
                let (blocks, prompt_depth) = reference[slot].take().expect("active reference");
                let expected_remove = blocks[..prompt_depth]
                    .iter()
                    .copied()
                    .filter(|hash| counts[hash] == 1)
                    .collect::<Vec<_>>();
                assert_eq!(tracker.release(chain), expected_remove);
                for hash in blocks {
                    let count = counts.get_mut(&hash).expect("reference block is active");
                    *count -= 1;
                    if *count == 0 {
                        counts.remove(&hash);
                    }
                }
            }
        }

        tracker.assert_consistent(std::iter::empty());
        assert!(counts.is_empty());
        assert_eq!(tracker.active_blocks(), 0);
    }

    #[test]
    fn long_chain_release_is_iterative() {
        const DEPTH: usize = 65_536;
        let sequence = (1..=DEPTH as u64).collect::<Vec<_>>();
        let mut tracker = BlockTracker::default();
        let (chain, first_new) = tracker.acquire_prompt(&sequence);

        assert_eq!(first_new, Some(0));
        tracker.assert_consistent([&chain]);
        assert_eq!(tracker.active_blocks(), DEPTH);
        assert_eq!(tracker.release(chain), sequence);
        tracker.assert_consistent(std::iter::empty());
        assert_eq!(tracker.active_blocks(), 0);
    }
}
