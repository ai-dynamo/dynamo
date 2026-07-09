// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_tokens::SequenceHash;
use rustc_hash::FxHashMap;
use std::sync::{Arc, Weak};

/// One node in a persistent request block chain.
///
/// Sequence hashes identify their lineage, so a live tail transitively owns the
/// entire prompt prefix through these parent references.
#[derive(Debug)]
struct BlockNode {
    hash: SequenceHash,
    depth: usize,
    parent: Option<Arc<BlockNode>>,
}

/// The single strong block owner retained by a request.
///
/// This type intentionally does not implement `Clone`. New request owners must
/// be acquired through [`BlockTracker::acquire_prompt`], which also maintains
/// the weak block index and membership transitions.
#[derive(Debug, Default)]
pub(super) struct RequestBlockChain {
    tail: Option<Arc<BlockNode>>,
    prompt_depth: usize,
}

impl RequestBlockChain {
    fn new(tail: Option<Arc<BlockNode>>, prompt_depth: usize) -> Self {
        Self { tail, prompt_depth }
    }

    #[cfg(any(test, debug_assertions))]
    fn nodes_from_tail(&self) -> impl Iterator<Item = &BlockNode> {
        std::iter::successors(self.tail.as_deref(), |node| node.parent.as_deref())
    }

    #[cfg(test)]
    pub(super) fn prompt_hashes(&self) -> impl Iterator<Item = SequenceHash> + '_ {
        self.nodes_from_tail()
            .filter(|node| node.depth <= self.prompt_depth)
            .map(|node| node.hash)
    }
}

impl Drop for RequestBlockChain {
    fn drop(&mut self) {
        let Some(mut current) = self.tail.take() else {
            return;
        };

        // A block-size-1 prompt can contain tens of thousands of nodes. Unwrap
        // an exclusively owned suffix iteratively so dropping the parent chain
        // cannot overflow the stack.
        loop {
            match Arc::try_unwrap(current) {
                Ok(mut node) => match node.parent.take() {
                    Some(parent) => current = parent,
                    None => break,
                },
                Err(shared) => {
                    drop(shared);
                    break;
                }
            }
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct BlockTracker {
    unique_blocks: FxHashMap<SequenceHash, Weak<BlockNode>>,
    fractional_blocks: FxHashMap<SequenceHash, f64>,
}

impl BlockTracker {
    /// Acquire one strong tail for a prompt and return the first newly present
    /// prompt index, if any.
    ///
    /// # Lineage contract
    ///
    /// Each [`SequenceHash`] must identify exactly one prefix ancestry and
    /// depth. As with the KV indexer, this tracker trusts callers to maintain
    /// that invariant and does not validate the hash recurrence in production.
    pub(super) fn acquire_prompt(
        &mut self,
        sequence: &[SequenceHash],
    ) -> (RequestBlockChain, Option<usize>) {
        let Some(&tail_hash) = sequence.last() else {
            return (RequestBlockChain::default(), None);
        };

        if let Some(tail) = self.upgrade_live_node(tail_hash) {
            Self::debug_assert_lineage(&tail, sequence);
            return (RequestBlockChain::new(Some(tail), sequence.len()), None);
        }

        let mut parent = None;
        let mut first_new_idx = 0;

        // Prompt liveness is prefix-closed. Search backward for the deepest
        // live prefix, then construct only the missing suffix.
        for idx in (0..sequence.len().saturating_sub(1)).rev() {
            if let Some(node) = self.upgrade_live_node(sequence[idx]) {
                Self::debug_assert_lineage(&node, &sequence[..=idx]);
                parent = Some(node);
                first_new_idx = idx + 1;
                break;
            }
        }

        for (idx, &hash) in sequence[first_new_idx..].iter().enumerate() {
            let node = Arc::new(BlockNode {
                hash,
                depth: first_new_idx + idx + 1,
                parent,
            });
            self.unique_blocks.insert(hash, Arc::downgrade(&node));
            parent = Some(node);
        }

        (
            RequestBlockChain::new(parent, sequence.len()),
            Some(first_new_idx),
        )
    }

    /// Append a unique output block to an existing request chain.
    pub(super) fn append_output(&mut self, chain: &mut RequestBlockChain, hash: SequenceHash) {
        debug_assert!(
            self.unique_blocks
                .get(&hash)
                .and_then(Weak::upgrade)
                .is_none(),
            "random output hash unexpectedly collided with a live block"
        );

        let parent = chain.tail.take();
        let depth = parent.as_ref().map_or(1, |node| node.depth + 1);
        let node = Arc::new(BlockNode {
            hash,
            depth,
            parent,
        });
        self.unique_blocks.insert(hash, Arc::downgrade(&node));
        chain.tail = Some(node);
    }

    /// Release a request chain and return the prompt suffix that became absent
    /// from the worker.
    pub(super) fn release(&mut self, chain: RequestBlockChain) -> Vec<SequenceHash> {
        let mut dead_nodes = Vec::new();
        let mut current = chain.tail.as_ref();

        while let Some(node) = current {
            if Arc::strong_count(node) != 1 {
                break;
            }
            dead_nodes.push((node.hash, node.depth));
            current = node.parent.as_ref();
        }

        for &(hash, _) in &dead_nodes {
            self.unique_blocks.remove(&hash);
            self.fractional_blocks.remove(&hash);
        }

        let mut prompt_remove = dead_nodes
            .into_iter()
            .filter_map(|(hash, depth)| (depth <= chain.prompt_depth).then_some(hash))
            .collect::<Vec<_>>();
        prompt_remove.reverse();
        prompt_remove
    }

    /// Mark the request's exclusively owned suffix as fractional.
    pub(super) fn set_unique_suffix_fractional(
        &mut self,
        chain: &RequestBlockChain,
        fraction: f64,
    ) {
        let mut current = chain.tail.as_ref();
        while let Some(node) = current {
            if Arc::strong_count(node) != 1 {
                break;
            }
            self.fractional_blocks.insert(node.hash, fraction);
            current = node.parent.as_ref();
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

    pub(super) fn contains_block(&self, hash: &SequenceHash) -> bool {
        self.unique_blocks.contains_key(hash)
    }

    #[cfg(any(test, debug_assertions))]
    pub(super) fn fractional_hashes_are_active(&self) -> bool {
        self.fractional_blocks
            .keys()
            .all(|hash| self.unique_blocks.contains_key(hash))
    }

    #[cfg(any(test, debug_assertions))]
    pub(super) fn contains_chain(&self, chain: &RequestBlockChain) -> bool {
        chain
            .nodes_from_tail()
            .all(|node| self.unique_blocks.contains_key(&node.hash))
    }

    #[cfg(test)]
    pub(super) fn active_hashes(&self) -> impl Iterator<Item = SequenceHash> + '_ {
        self.unique_blocks.keys().copied()
    }

    fn upgrade_live_node(&mut self, hash: SequenceHash) -> Option<Arc<BlockNode>> {
        let weak = self.unique_blocks.get(&hash)?;
        let node = weak.upgrade();
        if node.is_none() {
            self.unique_blocks.remove(&hash);
            self.fractional_blocks.remove(&hash);
        }
        node
    }

    #[cfg(any(test, debug_assertions))]
    fn debug_assert_lineage(tail: &Arc<BlockNode>, sequence: &[SequenceHash]) {
        assert_eq!(tail.depth, sequence.len(), "sequence tail depth mismatch");
        let mut current = Some(tail.as_ref());
        for (expected_depth, &expected_hash) in sequence.iter().enumerate().rev() {
            let node = current.expect("live sequence chain ended before its expected root");
            assert_eq!(node.depth, expected_depth + 1, "sequence depth mismatch");
            assert_eq!(node.hash, expected_hash, "sequence lineage hash mismatch");
            current = node.parent.as_deref();
        }
    }

    #[cfg(not(any(test, debug_assertions)))]
    #[inline]
    fn debug_assert_lineage(_tail: &Arc<BlockNode>, _sequence: &[SequenceHash]) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_acquire_and_last_release_report_presence_transitions() {
        let mut tracker = BlockTracker::default();

        let (first, first_new) = tracker.acquire_prompt(&[1]);
        let (second, second_new) = tracker.acquire_prompt(&[1]);

        assert_eq!(first_new, Some(0));
        assert_eq!(second_new, None);
        assert_eq!(tracker.active_blocks(), 1);

        assert!(tracker.release(first).is_empty());
        assert_eq!(tracker.active_blocks(), 1);

        assert_eq!(tracker.release(second), vec![1]);
        assert_eq!(tracker.active_blocks(), 0);
    }

    #[test]
    fn release_only_removes_unique_branch_suffix() {
        let mut tracker = BlockTracker::default();
        let (left, _) = tracker.acquire_prompt(&[1, 2, 3]);
        let (right, first_new) = tracker.acquire_prompt(&[1, 2, 4]);

        assert_eq!(first_new, Some(2));
        assert_eq!(tracker.active_blocks(), 4);
        assert_eq!(tracker.release(left), vec![3]);
        assert_eq!(tracker.active_blocks(), 3);
        assert_eq!(tracker.release(right), vec![1, 2, 4]);
        assert_eq!(tracker.active_blocks(), 0);
    }

    #[test]
    fn fractional_blocks_adjust_active_block_count() {
        let mut tracker = BlockTracker::default();
        let (chain, _) = tracker.acquire_prompt(&[1, 2]);

        tracker.set_unique_suffix_fractional(&chain, 0.5);
        assert_eq!(tracker.active_blocks(), 1);

        assert_eq!(tracker.release(chain), vec![1, 2]);
        assert!(tracker.fractional_blocks.is_empty());
        assert_eq!(tracker.active_blocks(), 0);
    }

    #[test]
    fn release_cleans_fractional_unique_suffix_above_shared_prefix() {
        let mut tracker = BlockTracker::default();
        let (longer, _) = tracker.acquire_prompt(&[1, 2, 3]);
        let (shared_prefix, _) = tracker.acquire_prompt(&[1, 2]);

        tracker.set_unique_suffix_fractional(&longer, 0.5);
        assert_eq!(tracker.fractional_blocks.get(&3), Some(&0.5));
        assert!(!tracker.fractional_blocks.contains_key(&1));
        assert!(!tracker.fractional_blocks.contains_key(&2));

        assert_eq!(tracker.release(longer), vec![3]);
        assert!(tracker.fractional_blocks.is_empty());
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
        assert_eq!(tracker.active_blocks(), 1);
        assert!(tracker.release(chain).is_empty());
        assert_eq!(tracker.active_blocks(), 0);
    }

    #[test]
    fn long_chain_release_is_iterative() {
        const DEPTH: usize = 65_536;
        let sequence = (1..=DEPTH as u64).collect::<Vec<_>>();
        let mut tracker = BlockTracker::default();
        let (chain, first_new) = tracker.acquire_prompt(&sequence);

        assert_eq!(first_new, Some(0));
        assert_eq!(tracker.active_blocks(), DEPTH);
        assert_eq!(tracker.release(chain), sequence);
        assert_eq!(tracker.active_blocks(), 0);
    }
}
