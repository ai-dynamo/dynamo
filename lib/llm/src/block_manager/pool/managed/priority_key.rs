// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PriorityKey<M: BlockMetadata> {
    metadata: M,
    sequence_hash: SequenceHash,
}

impl<M: BlockMetadata> PriorityKey<M> {
    pub(crate) fn new(metadata: M, sequence_hash: SequenceHash) -> Self {
        Self {
            metadata,
            sequence_hash,
        }
    }

    pub fn sequence_hash(&self) -> SequenceHash {
        self.sequence_hash
    }

    #[allow(dead_code)]
    pub fn metadata(&self) -> &M {
        &self.metadata
    }

    #[allow(dead_code)]
    pub fn update_metadata(&mut self, metadata: M) {
        self.metadata = metadata;
    }
}

// customize ord and partial ord for to store first by priority (lowest to highest),
// then by return_tick (lowest to highest)

impl<M: BlockMetadata> PartialOrd for PriorityKey<M> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<M: BlockMetadata> Ord for PriorityKey<M> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.metadata
            .cmp(&other.metadata)
            .then(self.sequence_hash.cmp(&other.sequence_hash))
    }
}
