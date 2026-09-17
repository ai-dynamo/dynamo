// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::protocols::{LocalBlockHash, compute_next_seq_hash};

/// Dynamo's canonical rolling sequence hash for one KV-cache block.
///
/// Engine-specific block identifiers must be translated to this type before
/// they enter CKF ownership, residency, or publication state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CanonicalSequenceBlockHash(u64);

impl CanonicalSequenceBlockHash {
    /// Start a canonical sequence. The first sequence hash is the block's local token hash.
    pub const fn root(tokens_hash: LocalBlockHash) -> Self {
        Self(tokens_hash.0)
    }

    /// Extend a canonical sequence with the next block's local token hash.
    pub fn child(self, tokens_hash: LocalBlockHash) -> Self {
        Self(compute_next_seq_hash(self.0, tokens_hash))
    }

    /// Return the wire-compatible `u64` fingerprint input.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use crate::protocols::{LocalBlockHash, compute_next_seq_hash};

    use super::CanonicalSequenceBlockHash;

    #[test]
    fn root_and_child_match_dynamo_sequence_hashing() {
        let first = LocalBlockHash(17);
        let second = LocalBlockHash(23);
        let root = CanonicalSequenceBlockHash::root(first);

        assert_eq!(root.as_u64(), first.0);
        assert_eq!(
            root.child(second).as_u64(),
            compute_next_seq_hash(first.0, second)
        );
    }
}
