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

use super::request::OffloadRequest;
use crate::block_manager::block::BlockMetadata;
use crate::block_manager::storage::Storage;
use crate::tokens::SequenceHash;
use std::collections::{HashMap, HashSet, VecDeque};

/// Lightweight node stored inside [`OffloadTreeQueue::entries`].
///
/// Only hash relationships are stored – within the request, we deliberately avoid holding a
/// strong reference to the [`MutableBlock`] itself so that the queue does not
/// extend the lifetime of the block data.
struct OffloadTreeNode<S: Storage, M: BlockMetadata> {
    request: OffloadRequest<S, M>,
    parent: Option<SequenceHash>,
    children: Vec<SequenceHash>,
}

/// A collection of trees (*forest*) that indexes blocks currently waiting in
/// the offload queue.
///
/// Internal data-structures:
/// * `entries` – mapping `sequence_hash → OffloadTreeNode`.
/// * `ends`    – `HashSet` satisfying **Invariant A**.
/// * `end_queue` – `VecDeque` to store sequence ends for round-robin consumption.
/// * `waiting` – `HashMap` to store children that arrived *before* their parent.
///   - This can happen with blocks registered after an offload request.
pub struct OffloadTreeQueue<S: Storage, M: BlockMetadata> {
    entries: HashMap<SequenceHash, OffloadTreeNode<S, M>>,
    // Current sequence ends (may converge later).
    ends: HashSet<SequenceHash>,
    /// Round-robin queue of sequence ends. Stale hashes are skipped lazily.
    end_queue: VecDeque<SequenceHash>,
    /// Children that appeared *before* their parent.  Keyed by the missing
    /// `parent_sequence_hash` and containing the hashes of its children.
    waiting: HashMap<SequenceHash, Vec<SequenceHash>>,
}

impl<S: Storage, M: BlockMetadata> OffloadTreeQueue<S, M> {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            ends: HashSet::new(),
            end_queue: VecDeque::new(),
            waiting: HashMap::new(),
        }
    }

    // TODO: Incorporate priorities into this.
    pub fn insert(&mut self, request: OffloadRequest<S, M>) -> anyhow::Result<()> {
        // `sequence_hash` uniquely identifies the *current* block, whereas
        // `parent` (if `Some`) points to its immediate predecessor.
        // Note: The parent might not yet exist in the queue. This is fine, and is handled below.
        let sequence_hash = request.sequence_hash;
        let parent = request.parent_sequence_hash;

        if self.entries.contains_key(&sequence_hash) {
            return Ok(());
        }

        // Create the node, and fill in the children (if any) later.
        let mut entry = OffloadTreeNode::<S, M> {
            parent,
            children: Vec::new(),
            request,
        };

        // A brand-new node starts as a leaf by definition.
        if self.ends.insert(sequence_hash) {
            self.end_queue.push_back(sequence_hash);
        }

        if let Some(parent_hash) = parent {
            // Parent already present: append ourselves to its `children` list
            // and ensure the parent is no longer considered a leaf.
            if let Some(parent_node) = self.entries.get_mut(&parent_hash) {
                parent_node.children.push(sequence_hash);
                // If the sequence hasn't been forked (i.e. This is the only child), the parent is no longer a sequence end.
                if parent_node.children.len() == 1 {
                    self.ends.remove(&parent_hash);
                }
            } else {
                // Parent not yet present: mark this node as a temporary root
                self.waiting
                    .entry(parent_hash)
                    .or_default()
                    .push(sequence_hash);
            }
        } else {
            // No parent => unconditional root.
            // root-less sequence end already inserted above
        }

        // Any children that arrived *before* their parent were stored in the
        // `waiting` map.  Retrieve and re-link them.
        if let Some(orphans) = self.waiting.remove(&sequence_hash) {
            // Drain the orphan list into the parent's `children` vector.
            entry.children.extend(orphans.iter().copied());

            if !orphans.is_empty() {
                // Parent now definitely has at least one child.
                self.ends.remove(&sequence_hash);
            }
        }

        self.entries.insert(sequence_hash, entry);

        Ok(())
    }

    /// Remove and return any leaf block from the queue.
    ///
    /// The current implementation picks the first leaf returned by the
    /// iteration order of the underlying `HashSet` which is effectively
    /// deterministic within a single process execution but otherwise
    /// arbitrary.  This keeps the API simple while still ensuring that *only*
    /// leaves are removed and all invariants (`roots`, `leaves`, `waiting`)
    /// remain valid.
    pub fn remove(&mut self) -> Option<OffloadRequest<S, M>> {
        // Pop until we find a still-valid leaf.
        let leaf_hash = loop {
            let hash = self.end_queue.pop_front()?; // None => queue empty
            if self.ends.contains(&hash) {
                break hash;
            }
            // stale entry, skip
        };

        // Remove the end from auxiliary sets.
        self.ends.remove(&leaf_hash);

        // Remove the node from the main map.
        let node = self.entries.remove(&leaf_hash)?;

        // Detach from parent if we know it.
        if let Some(parent_hash) = node.parent {
            if let Some(parent_node) = self.entries.get_mut(&parent_hash) {
                // Remove this child from the parent's list.
                parent_node.children.retain(|c| *c != leaf_hash);

                if self.ends.insert(parent_hash) {
                    self.end_queue.push_back(parent_hash);
                }
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

    fn make_request(
        block: &Arc<MutableBlock<NullDeviceStorage, BasicMetadata>>,
    ) -> Result<OffloadRequest<NullDeviceStorage, BasicMetadata>> {
        Ok(OffloadRequest {
            block: Arc::downgrade(block),
            sequence_hash: block.sequence_hash()?,
            priority: 0,
            parent_sequence_hash: block.parent_sequence_hash()?,
        })
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

        let mut queue = OffloadTreeQueue::new();

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
}
