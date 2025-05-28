// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Offload Queue
//!
//! This module defines [`OffloadTreeQueue`], a forest-based data structure
//! that tracks blocks waiting to be off-loaded.  Each node is addressed
//! exclusively via its [`SequenceHash`] as to ensure we never
//! prolong the lifetime of the underlying blocks.
//!
//! ## Key Insights
//! - When offloading, we always want to offload from the leaves back down to the root.
//! - This is because the leaves get evicted first, so there's much less of a margin to offload them
//!   before they're evicted. Blocks close to the root are more likely to stick around for longer, so they
//!   don't need to be offloaded with as much urgency.
//! - This also applies to multi-level caching. The same blocks which are offloaded first from G1 to G2
//!   should also be offloaded first from G2 to G3, and so on.
//!
//! ## Semantics
//! --------
//! - We define the idea of a "sequence" as a maximal chain of blocks.
//! - A block is a "sequence end" if it has no children in the queue.
//! - When removing blocks, we only ever want to remove sequence ends.
//!  - We do this by maintaining a set of sequence ends.
//!  - Within this set, we do a round-robin removal based on offload priority.
//!  - When we remove a sequence end, its parent (if it exists in the queue) automatically
//!    becomes a new sequence end, regardless of whether the parent has other children.
//! - We also must handle the case where two sequences are forked from a common prefix.
//!  - When a sequence end is forked, we can proceed with the forked sequence as normal.
//!  - All other sequences that use the prefix are detached from the prefix.
//!  - These other sequences will proceed with dequeueing until they reach the point where they diverge from the
//!    forked sequence, at which point they are completed.
//!
//! ## Block Registration
//! - Blocks are offloaded in the order in which they are registered.
//! - There are three steps:
//!   1. A prefill is completed, and a large list of blocks is added to the queue all at once.
//!     - In this case, we want to start offloading from the end of the prefill back towards the root.
//!   2. A block is completed in the decode phase, and is added to the queue.
//!     - In this case, we want to offload this block first, then proceed with its parents,
//!       and eventually back to the remaining prefill blocks.
//!   3. A block is offloaded, and registered in its target pool.
//!     - By steps 1 and 2, blocks get offloaded from leaf to root (approximately).
//!     - Because of this, it's possible that blocks we add to the queue are parents of other blocks already in the queue.

use super::request::OffloadRequest;
use crate::block_manager::block::BlockMetadata;
use crate::block_manager::storage::Storage;
use crate::tokens::SequenceHash;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

/// Lightweight node stored inside [`OffloadTreeQueue::entries`].
///
/// Only hash relationships are stored – within the request, we deliberately avoid holding a
/// strong reference to the [`MutableBlock`] itself so that the queue does not
/// extend the lifetime of the block data.
struct OffloadTreeNode<S: Storage, M: BlockMetadata> {
    request: OffloadRequest<S, M>,
    parent: Option<SequenceHash>,
}

/// A collection of trees (*forest*) that indexes blocks currently waiting in
/// the offload queue.
pub struct OffloadTreeQueue<S: Storage, M: BlockMetadata> {
    entries: HashMap<SequenceHash, OffloadTreeNode<S, M>>,
    // Current sequence ends (may converge later).
    ends: HashSet<SequenceHash>,
    /// Round-robin queues of sequence ends. Stale hashes are skipped lazily.
    end_queues: BTreeMap<u64, VecDeque<SequenceHash>>,
    /// A map of unregistered sequence hashes to their waiting children.
    waiting_children: HashMap<SequenceHash, Vec<SequenceHash>>,
}

impl<S: Storage, M: BlockMetadata> OffloadTreeQueue<S, M> {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            ends: HashSet::new(),
            end_queues: BTreeMap::new(),
            waiting_children: HashMap::new(),
        }
    }

    // TODO: Incorporate request priority into this.
    pub fn insert(&mut self, request: OffloadRequest<S, M>) -> anyhow::Result<()> {
        // `sequence_hash` uniquely identifies the *current* block, whereas
        // `parent` (if `Some`) points to its immediate predecessor.
        // Note: The parent might not yet exist in the queue. This is fine, and is handled below.
        let sequence_hash = request.sequence_hash;
        let parent = request.parent_sequence_hash;

        if self.entries.contains_key(&sequence_hash) {
            return Ok(());
        }

        let entry = OffloadTreeNode::<S, M> { parent, request };

        if let Some(parent_hash) = parent {
            // Parent already present: append ourselves to its `children` list
            // and ensure the parent is no longer considered a leaf.
            if self.entries.get_mut(&parent_hash).is_some() {
                self.ends.remove(&parent_hash);
            } else {
                // Otherwise, we need to wait for the parent to be registered.
                self.waiting_children
                    .entry(parent_hash)
                    .or_default()
                    .push(sequence_hash);
            }
        }

        let mut is_end = true;

        // If there are waiting children, and any of the children are present, this is not a sequence end.
        if let Some(children) = self.waiting_children.remove(&sequence_hash) {
            if children
                .into_iter()
                .any(|child| self.entries.contains_key(&child))
            {
                is_end = false;
            }
        }

        if is_end && self.ends.insert(sequence_hash) {
            self.end_queues
                .entry(entry.request.priority)
                .or_default()
                .push_back(sequence_hash);
        }

        self.entries.insert(sequence_hash, entry);

        Ok(())
    }

    /// Remove and return a sequence end from the queue.
    /// Use the end_queue to enforce a FIFO order.
    pub fn remove(&mut self) -> Option<OffloadRequest<S, M>> {
        // Pop until we find a still-valid sequence end.
        let end_hash = loop {
            let end_queue = self.end_queues.last_entry()?.into_mut();

            let hash = end_queue.pop_front().unwrap();

            if end_queue.is_empty() {
                self.end_queues.pop_last();
            }

            if self.ends.contains(&hash) {
                break hash;
            }
            // stale entry, skip
        };

        // Remove the end from auxiliary sets.
        self.ends.remove(&end_hash);

        // Remove the node from the main map.
        let node = self.entries.remove(&end_hash)?;

        // The parent is now a sequence end.
        if let Some(parent_hash) = node.parent {
            if self.entries.get_mut(&parent_hash).is_some() && self.ends.insert(parent_hash) {
                self.end_queues
                    .entry(node.request.priority)
                    .or_default()
                    .push_back(parent_hash);
            }
        }

        Some(node.request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::{
        block::{BlockExt, MutableBlock},
        layout::{FullyContiguous, LayoutConfig},
        storage::tests::{NullDeviceAllocator, NullDeviceStorage},
        BasicMetadata,
    };
    use crate::tokens::{TokenBlockSequence, Tokens};

    use anyhow::Result;
    use std::sync::Arc;
    use tokio::sync::mpsc;

    fn make_blocks(
        num_blocks: usize,
    ) -> Result<Vec<MutableBlock<NullDeviceStorage, BasicMetadata>>> {
        let layout_config = LayoutConfig::builder()
            .num_blocks(num_blocks)
            .num_layers(1)
            .outer_dim(1)
            .page_size(4)
            .inner_dim(1)
            .build()?;

        let layout = FullyContiguous::allocate(layout_config, &NullDeviceAllocator)?;
        let blocks = crate::block_manager::block::Blocks::<_, BasicMetadata>::new(layout, 0, 0)?
            .into_blocks()?;

        let (tx, _) = mpsc::unbounded_channel();

        Ok(blocks
            .into_iter()
            .map(|b| MutableBlock::new(b, tx.clone()))
            .collect())
    }

    fn extract(request: Option<OffloadRequest<NullDeviceStorage, BasicMetadata>>) -> SequenceHash {
        request.unwrap().sequence_hash
    }

    type BlockList = Vec<Arc<MutableBlock<NullDeviceStorage, BasicMetadata>>>;

    fn tokens_to_blocks(
        tokens: Vec<u32>,
        block_pool: &mut Vec<MutableBlock<NullDeviceStorage, BasicMetadata>>,
    ) -> Result<(BlockList, Vec<SequenceHash>)> {
        let sequence = TokenBlockSequence::new(Tokens::from(tokens.to_vec()), 4, None);

        let mut blocks = Vec::new();
        let mut sequence_hashes = Vec::new();

        for token_block in sequence.blocks() {
            let mut block = block_pool.pop().unwrap();
            block.apply_token_block(token_block.clone())?;
            sequence_hashes.push(block.sequence_hash()?);
            blocks.push(Arc::new(block));
        }

        Ok((blocks, sequence_hashes))
    }

    fn make_request_with_priority(
        block: &Arc<MutableBlock<NullDeviceStorage, BasicMetadata>>,
        priority: u64,
    ) -> Result<OffloadRequest<NullDeviceStorage, BasicMetadata>> {
        Ok(OffloadRequest {
            block: Arc::downgrade(block),
            sequence_hash: block.sequence_hash()?,
            priority,
            parent_sequence_hash: block.parent_sequence_hash()?,
        })
    }

    fn make_request(
        block: &Arc<MutableBlock<NullDeviceStorage, BasicMetadata>>,
    ) -> Result<OffloadRequest<NullDeviceStorage, BasicMetadata>> {
        make_request_with_priority(block, 0)
    }

    #[test]
    fn test_basic_dequeue_order() -> Result<()> {
        let mut blocks = make_blocks(3)?;
        let mut queue = OffloadTreeQueue::new();

        let (block_sequence, sequence_hashes) = tokens_to_blocks(vec![0; 13], &mut blocks)?;

        for block in &block_sequence {
            queue.insert(make_request(block)?)?;
        }

        for i in 0..3 {
            assert_eq!(extract(queue.remove()), sequence_hashes[2 - i]);
        }

        for i in (0..3).rev() {
            queue.insert(make_request(&block_sequence[i])?)?;
        }

        for i in 0..3 {
            assert_eq!(extract(queue.remove()), sequence_hashes[2 - i]);
        }

        Ok(())
    }

    #[test]
    fn test_forked_dequeue_order() -> Result<()> {
        let mut blocks = make_blocks(5)?;
        let mut queue = OffloadTreeQueue::new();

        let base_tokens = vec![0; 5];

        let seq1_tokens = base_tokens.clone().into_iter().chain(vec![1; 4]).collect();
        let seq2_tokens = base_tokens.clone().into_iter().chain(vec![2; 8]).collect();

        let (block_sequence1, sequence1_hashes) = tokens_to_blocks(seq1_tokens, &mut blocks)?;
        let (block_sequence2, sequence2_hashes) = tokens_to_blocks(seq2_tokens, &mut blocks)?;

        for block in &block_sequence1 {
            queue.insert(make_request(block)?)?;
        }

        queue.insert(make_request(&block_sequence2[1])?)?;
        queue.insert(make_request(&block_sequence2[2])?)?;

        assert_eq!(extract(queue.remove()), sequence1_hashes[1]);
        assert_eq!(extract(queue.remove()), sequence2_hashes[2]);
        assert_eq!(extract(queue.remove()), sequence1_hashes[0]);
        assert_eq!(extract(queue.remove()), sequence2_hashes[1]);
        assert!(queue.remove().is_none());

        Ok(())
    }

    #[test]
    fn test_prefill_decode() -> Result<()> {
        let mut blocks = make_blocks(4)?;
        let mut queue = OffloadTreeQueue::new();

        let tokens = vec![0; 17];

        let (mut prefill_blocks, sequence_hashes) = tokens_to_blocks(tokens, &mut blocks)?;

        let decode_block = prefill_blocks.pop().unwrap();

        assert_eq!(prefill_blocks.len(), 3);

        for block in &prefill_blocks {
            queue.insert(make_request(block)?)?;
        }

        assert_eq!(extract(queue.remove()), sequence_hashes[2]);

        queue.insert(make_request(&decode_block)?)?;

        assert_eq!(extract(queue.remove()), sequence_hashes[1]);
        assert_eq!(extract(queue.remove()), decode_block.sequence_hash()?);
        assert_eq!(extract(queue.remove()), sequence_hashes[0]);
        assert!(queue.remove().is_none());

        Ok(())
    }

    #[test]
    fn test_backfill() -> Result<()> {
        let mut blocks = make_blocks(4)?;
        let mut queue = OffloadTreeQueue::new();

        let tokens = vec![0; 17];

        let (sequence_blocks, sequence_hashes) = tokens_to_blocks(tokens, &mut blocks)?;

        queue.insert(make_request(&sequence_blocks[3])?)?;
        queue.insert(make_request(&sequence_blocks[2])?)?;

        assert_eq!(extract(queue.remove()), sequence_hashes[3]);

        queue.insert(make_request(&sequence_blocks[0])?)?;
        queue.insert(make_request(&sequence_blocks[1])?)?;

        assert_eq!(extract(queue.remove()), sequence_hashes[2]);
        assert_eq!(extract(queue.remove()), sequence_hashes[1]);
        assert_eq!(extract(queue.remove()), sequence_hashes[0]);
        assert!(queue.remove().is_none());

        Ok(())
    }

    #[test]
    fn test_duplicate_ignore() -> Result<()> {
        // Ensures that attempting to add the *same* block twice does not
        // change dequeue semantics (i.e. the duplicate is silently ignored).
        let mut blocks = make_blocks(3)?;
        let mut queue = OffloadTreeQueue::new();

        // Build a simple three-block sequence.
        let (block_sequence, sequence_hashes) = tokens_to_blocks(vec![0; 13], &mut blocks)?;

        // Add all blocks once …
        for block in &block_sequence {
            queue.insert(make_request(block)?)?;
        }
        // … and then try to add the last block a second time.
        queue.insert(make_request(&block_sequence[2])?)?;

        // We still expect exactly three dequeue operations.
        for i in 0..3 {
            assert_eq!(extract(queue.remove()), sequence_hashes[2 - i]);
        }
        assert!(queue.remove().is_none());

        Ok(())
    }

    #[test]
    fn test_multiple_roots() -> Result<()> {
        // Verifies that the queue can handle several independent sequences
        // (a *forest*) without mixing them up.
        let mut blocks = make_blocks(6)?;
        let mut queue = OffloadTreeQueue::new();

        // Create two disjoint sequences of three blocks each.
        let (seq1_blocks, seq1_hashes) = tokens_to_blocks(vec![1; 13], &mut blocks)?;
        let (seq2_blocks, seq2_hashes) = tokens_to_blocks(vec![2; 13], &mut blocks)?;

        for block in &seq1_blocks {
            queue.insert(make_request(block)?)?;
        }
        for block in &seq2_blocks {
            queue.insert(make_request(block)?)?;
        }

        // Collect all hashes removed from the queue.
        use std::collections::HashSet;
        let mut removed = HashSet::new();
        while let Some(block) = queue.remove() {
            removed.insert(extract(Some(block)));
        }

        let expected: HashSet<_> = seq1_hashes.into_iter().chain(seq2_hashes).collect();
        assert_eq!(removed, expected);

        Ok(())
    }

    #[test]
    fn test_empty_queue() -> Result<()> {
        // Calling `remove` on an empty queue should immediately return `None`.
        let mut queue: OffloadTreeQueue<NullDeviceStorage, BasicMetadata> = OffloadTreeQueue::new();
        assert!(queue.remove().is_none());
        Ok(())
    }

    #[test]
    fn test_re_add() -> Result<()> {
        let mut blocks = make_blocks(8)?;

        let (block_sequence, sequence_hashes) = tokens_to_blocks(vec![0; 13], &mut blocks)?;

        let mut queue = OffloadTreeQueue::new();

        for block in &block_sequence {
            queue.insert(make_request(block)?)?;
        }

        assert_eq!(extract(queue.remove()), sequence_hashes[2]);

        queue.insert(make_request(&block_sequence[2])?)?;

        assert_eq!(extract(queue.remove()), sequence_hashes[2]);
        assert_eq!(extract(queue.remove()), sequence_hashes[1]);
        assert_eq!(extract(queue.remove()), sequence_hashes[0]);

        assert!(queue.remove().is_none());

        Ok(())
    }

    #[test]
    fn test_sequence_end_rebuild() -> Result<()> {
        let mut blocks = make_blocks(8)?;

        let (block_sequence, sequence_hashes) = tokens_to_blocks(vec![0; 13], &mut blocks)?;

        let mut queue = OffloadTreeQueue::new();

        for block in block_sequence.iter().rev() {
            queue.insert(make_request(block)?)?;
        }

        let (new_sequence_blocks, new_sequence_hashes) =
            tokens_to_blocks(vec![0, 0, 0, 0, 1, 1, 1, 1, 1], &mut blocks)?;

        queue.insert(make_request(&new_sequence_blocks[1])?)?;

        assert_eq!(extract(queue.remove()), sequence_hashes[2]);
        assert_eq!(extract(queue.remove()), new_sequence_hashes[1]);
        assert_eq!(extract(queue.remove()), sequence_hashes[1]);
        assert_eq!(extract(queue.remove()), sequence_hashes[0]);

        Ok(())
    }

    #[test]
    fn test_priority_order() -> Result<()> {
        let mut blocks = make_blocks(8)?;
        let mut queue = OffloadTreeQueue::new();

        let (block_sequence1, sequence1_hashes) = tokens_to_blocks(vec![0; 13], &mut blocks)?;

        for block in &block_sequence1 {
            queue.insert(make_request_with_priority(block, 1)?)?;
        }

        let (block_sequence2, sequence2_hashes) = tokens_to_blocks(vec![1; 13], &mut blocks)?;

        for block in &block_sequence2 {
            queue.insert(make_request(block)?)?;
        }
        assert_eq!(extract(queue.remove()), sequence1_hashes[2]);
        assert_eq!(extract(queue.remove()), sequence1_hashes[1]);
        assert_eq!(extract(queue.remove()), sequence1_hashes[0]);
        assert_eq!(extract(queue.remove()), sequence2_hashes[2]);
        assert_eq!(extract(queue.remove()), sequence2_hashes[1]);
        assert_eq!(extract(queue.remove()), sequence2_hashes[0]);

        Ok(())
    }
}
