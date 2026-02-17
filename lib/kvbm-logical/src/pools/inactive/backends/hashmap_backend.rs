// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend trait for InactivePool storage strategies.

use std::collections::HashMap;

use super::{Block, BlockMetadata, InactiveBlock, Registered, SequenceHash};

use super::super::InactivePoolBackend;
use super::ReusePolicy;

pub struct HashMapBackend<T: BlockMetadata> {
    blocks: HashMap<SequenceHash, Block<T, Registered>>,
    reuse_policy: Box<dyn ReusePolicy>,
}

impl<T: BlockMetadata> HashMapBackend<T> {
    pub fn new(reuse_policy: Box<dyn ReusePolicy>) -> Self {
        Self {
            blocks: HashMap::new(),
            reuse_policy,
        }
    }
}

impl<T: BlockMetadata> InactivePoolBackend<T> for HashMapBackend<T> {
    fn find_matches(&mut self, hashes: &[SequenceHash], _touch: bool) -> Vec<Block<T, Registered>> {
        let mut matches = Vec::with_capacity(hashes.len());

        for hash in hashes {
            if let Some(block) = self.blocks.remove(hash) {
                let _ = self.reuse_policy.remove(block.block_id());
                matches.push(block);
            } else {
                break;
            }
        }

        matches
    }

    fn scan_matches(
        &mut self,
        hashes: &[SequenceHash],
        _touch: bool,
    ) -> Vec<(SequenceHash, Block<T, Registered>)> {
        let mut matches = Vec::new();

        for hash in hashes {
            if let Some(block) = self.blocks.remove(hash) {
                let _ = self.reuse_policy.remove(block.block_id());
                matches.push((*hash, block));
            }
            // Unlike find_matches: NO break on miss - continue scanning
        }

        matches
    }

    fn allocate(&mut self, count: usize) -> Vec<Block<T, Registered>> {
        let mut allocated = Vec::with_capacity(count);

        for _ in 0..count {
            if let Some(InactiveBlock { seq_hash, .. }) = self.reuse_policy.next_free() {
                if let Some(block) = self.blocks.remove(&seq_hash) {
                    allocated.push(block);
                } else {
                    debug_assert!(
                        false,
                        "reuse_policy yielded seq_hash {:?} not found in blocks",
                        seq_hash
                    );
                }
            } else {
                break;
            }
        }

        allocated
    }

    fn insert(&mut self, block: Block<T, Registered>) {
        let seq_hash = block.sequence_hash();
        let _ = self.reuse_policy.insert(InactiveBlock {
            block_id: block.block_id(),
            seq_hash,
        });
        self.blocks.insert(seq_hash, block);
    }

    fn len(&self) -> usize {
        self.blocks.len()
    }

    fn has_block(&self, seq_hash: SequenceHash) -> bool {
        self.blocks.contains_key(&seq_hash)
    }

    fn allocate_all(&mut self) -> Vec<Block<T, Registered>> {
        // Drain reuse policy by consuming all entries
        while self.reuse_policy.next_free().is_some() {}
        // Drain and return all blocks
        self.blocks.drain().map(|(_, block)| block).collect()
    }
}
