// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HashMap-backed inactive index.
//!
//! Maps `SequenceHash → BlockId` and delegates eviction order to a pluggable
//! [`ReusePolicy`] (FIFO is the default).

use std::collections::HashMap;

use crate::BlockId;
use crate::blocks::SequenceHash;
use crate::pools::InactiveBlock;
use crate::pools::store::InactiveIndex;

use super::ReusePolicy;

pub(crate) struct HashMapBackend {
    blocks: HashMap<SequenceHash, BlockId>,
    reuse_policy: Box<dyn ReusePolicy>,
}

impl HashMapBackend {
    pub(crate) fn new(reuse_policy: Box<dyn ReusePolicy>) -> Self {
        Self {
            blocks: HashMap::new(),
            reuse_policy,
        }
    }
}

impl InactiveIndex for HashMapBackend {
    fn find_matches(
        &mut self,
        hashes: &[SequenceHash],
        _touch: bool,
    ) -> Vec<(SequenceHash, BlockId)> {
        let mut matches = Vec::with_capacity(hashes.len());
        for hash in hashes {
            if let Some(block_id) = self.blocks.remove(hash) {
                let _ = self.reuse_policy.remove(block_id);
                matches.push((*hash, block_id));
            } else {
                break;
            }
        }
        matches
    }

    fn find_match(&mut self, hash: SequenceHash, _touch: bool) -> Option<(SequenceHash, BlockId)> {
        let block_id = self.blocks.remove(&hash)?;
        let _ = self.reuse_policy.remove(block_id);
        Some((hash, block_id))
    }

    fn scan_matches(
        &mut self,
        hashes: &[SequenceHash],
        _touch: bool,
    ) -> Vec<(SequenceHash, BlockId)> {
        let mut matches = Vec::new();
        for hash in hashes {
            if let Some(block_id) = self.blocks.remove(hash) {
                let _ = self.reuse_policy.remove(block_id);
                matches.push((*hash, block_id));
            }
        }
        matches
    }

    fn allocate(&mut self, count: usize) -> Vec<(SequenceHash, BlockId)> {
        let mut allocated = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(InactiveBlock { seq_hash, block_id }) = self.reuse_policy.next_free() {
                if self.blocks.remove(&seq_hash).is_some() {
                    allocated.push((seq_hash, block_id));
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

    fn insert(&mut self, seq_hash: SequenceHash, block_id: BlockId) {
        let _ = self
            .reuse_policy
            .insert(InactiveBlock { block_id, seq_hash });
        self.blocks.insert(seq_hash, block_id);
    }

    fn len(&self) -> usize {
        self.blocks.len()
    }

    fn has(&self, seq_hash: SequenceHash) -> bool {
        self.blocks.contains_key(&seq_hash)
    }

    fn take(&mut self, seq_hash: SequenceHash, block_id: BlockId) -> bool {
        if self.blocks.get(&seq_hash) == Some(&block_id) {
            self.blocks.remove(&seq_hash);
            let _ = self.reuse_policy.remove(block_id);
            true
        } else {
            false
        }
    }

    fn allocate_all(&mut self) -> Vec<(SequenceHash, BlockId)> {
        // Drain reuse policy
        while self.reuse_policy.next_free().is_some() {}
        // Drain map
        self.blocks.drain().collect()
    }
}
