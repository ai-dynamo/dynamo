// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Weak;

use super::*;

type ReservedBlockMap = Arc<RwLock<HashMap<SequenceHash, Weak<ReservedBlockInner>>>>;

#[derive(Clone)]
pub struct ReservedBlock {
    inner: Arc<ReservedBlockInner>,
}

impl ReservedBlock {
    fn new(inner: Arc<ReservedBlockInner>) -> Self {
        Self { inner }
    }

    pub fn inflight_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

impl std::ops::Deref for ReservedBlock {
    type Target = SharedBlock;

    fn deref(&self) -> &Self::Target {
        &self.inner.block
    }
}

struct ReservedBlockInner {
    block: SharedBlock,
    map: ReservedBlockMap,
}

impl Drop for ReservedBlockInner {
    fn drop(&mut self) {
        let sequence_hash = self.block.token_block.sequence_hash();
        let mut map = self.map.write().unwrap();
        let val = map.remove(&sequence_hash);

        if let Some(inner) = val {
            if inner.strong_count() > 0 {
                // this was not the weak pointer we were looking for
                map.insert(sequence_hash, inner);
            }
        }
    }
}

/// [ReservedBlocks] is a collection of inflight blocks that are actively being used
pub struct ReservedBlocks {
    block_size: usize,
    blocks: ReservedBlockMap,
}

impl ReservedBlocks {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            blocks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn match_sequence_hashes(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<Vec<ReservedBlock>> {
        let mut inflight_blocks = Vec::new();
        let map = self.blocks.read().unwrap();
        for sequence_hash in sequence_hashes {
            if let Some(inner) = map.get(sequence_hash) {
                if let Some(inner) = inner.upgrade() {
                    inflight_blocks.push(ReservedBlock::new(inner.clone()));
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        Ok(inflight_blocks)
    }

    /// Match the list of blocks to inflight blocks
    ///
    /// This will return a [Vec<ReservedBlock>] that match the sequence hashes
    /// in the order of the token blocks.
    ///
    /// The matching is done in order, with the first block in the list being the first
    /// block in the token blocks list.
    ///
    /// If a block is not found, the function will return the list of matched blocks
    /// and the remaining blocks will not be included.
    pub fn match_token_blocks(&self, token_blocks: &[TokenBlock]) -> Result<Vec<ReservedBlock>> {
        let mut inflight_blocks = Vec::new();
        let map = self.blocks.read().unwrap();
        for token_block in token_blocks {
            let sequence_hash = token_block.sequence_hash();
            if let Some(inner) = map.get(&sequence_hash) {
                if let Some(inner) = inner.upgrade() {
                    inflight_blocks.push(ReservedBlock::new(inner.clone()));
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        Ok(inflight_blocks)
    }

    pub fn register(&mut self, block: UniqueBlock) -> Result<ReservedBlock> {
        let sequence_hash = block.token_block.sequence_hash();
        let shared = block.into_shared();

        if shared.token_block.tokens().len() != self.block_size {
            raise!("Block size mismatch");
        }

        // if the block already exists, we drop the block the user passed in and return the existing block
        // this should return the passed in block to the free pool
        let mut map = self.blocks.write().unwrap();
        if let Some(existing_block) = map.get(&sequence_hash) {
            // return an ReservedBlock with the existing block
            // the passed in block will be dropped and returned to the pool
            // this could happen if two sequences are building the same block at the same time,
            // the first sequence to finish and register the block will insert it into the map
            if let Some(inner) = existing_block.upgrade() {
                return Ok(ReservedBlock::new(inner.clone()));
            }
        }

        // Insert the new block and create an ReservedBlock from it
        let inner = Arc::new(ReservedBlockInner {
            block: shared,
            map: self.blocks.clone(),
        });

        map.insert(sequence_hash, Arc::downgrade(&inner));

        Ok(ReservedBlock::new(inner))
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use super::reuse::tests::{create_blocks, create_token_sequence};
    use super::reuse::AvailableBlocks;

    #[tokio::test]
    async fn test_reserved_blocks() {
        let available_blocks = AvailableBlocks::new().await;
        let mut reserved_blocks = ReservedBlocks::new(2);

        // Create two sequences with different priorities
        let seq1 = create_token_sequence(&[1, 2, 3, 4]);
        let seq2 = create_token_sequence(&[5, 6, 7, 8]);

        // This is creating new KvBlock; this is will be done when the block manager is initialized
        // but since we are not using the block manager in this test, we need to create them manually
        let blocks1 = create_blocks(seq1, 2);
        let blocks2 = create_blocks(seq2, 2);

        // Insert Sequence 2
        for block in blocks2.into_iter().rev() {
            available_blocks.insert(block).await.unwrap();
        }

        // Insert Sequence 1
        for block in blocks1.into_iter().rev() {
            available_blocks.insert(block).await.unwrap();
        }

        available_blocks.fence().await.unwrap();
        assert_eq!(available_blocks.total_blocks(), 4);
        assert_eq!(available_blocks.available_blocks(), 4);

        // Initialize of the KvBlocks is complete - there are 4 blocks with state in the available pool

        // Mimic a request for 2 tokens and test the block matching sequence
        // This pattern will be used in the KvBlockManager
        let req1 = create_token_sequence(&[1, 2]);
        let seq1 = req1.into_sequence(2);
        let (blocks, tail_block) = seq1.into_parts();
        assert_eq!(blocks.len(), 1);
        assert_eq!(tail_block.tokens().len(), 0);

        let matched = reserved_blocks.match_token_blocks(&blocks).unwrap();
        assert_eq!(matched.len(), 0);

        let matched = available_blocks.match_token_blocks(&blocks).await.unwrap();
        assert_eq!(matched.len(), 1);

        // possible update the api to take a vec of unique blocks and return a vec of reserved blocks
        let reserved: Vec<ReservedBlock> = matched
            .into_iter()
            .map(|unique_block| reserved_blocks.register(unique_block).unwrap())
            .collect();

        assert_eq!(reserved.len(), 1);
        assert_eq!(reserved[0].inflight_count(), 1);
        assert_eq!(available_blocks.available_blocks(), 3);

        // request 2
        // reuse blocks
        // match blocks to the reserved blocks get a new reserved block which should have a ref count of 2

        let reserved2 = reserved_blocks.match_token_blocks(&blocks).unwrap();
        assert_eq!(reserved2.len(), 1);
        assert_eq!(reserved2[0].inflight_count(), 2);
        assert_eq!(available_blocks.available_blocks(), 3);

        drop(reserved2);
        available_blocks.fence().await.unwrap();

        assert_eq!(reserved[0].inflight_count(), 1);
        assert_eq!(available_blocks.available_blocks(), 3);

        drop(reserved);
        available_blocks.fence().await.unwrap();

        assert_eq!(available_blocks.available_blocks(), 4);
    }
}
