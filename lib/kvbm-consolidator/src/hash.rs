// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Projection of [`kvbm_logical::SequenceHash`] (128-bit [`dynamo_tokens::PositionalLineageHash`])
//! onto the kv-router u64 wire format.
//!
//! The router expects a u64 per block-edge. The 128-bit PLH packs `(mode | position |
//! parent_fragment | current_fragment)`. The u64 the router consumes is what a *child*
//! at `position+1` would see as its `parent_hash_fragment` — produced by
//! [`PositionalLineageHash::parent_fragment_for_child_position`], which handles the
//! 0→1 (pos 256) and 1→2 (pos 65536) mode-boundary alignments.

use kvbm_logical::SequenceHash;

/// Project the block's u64 hash for the kv-router wire format.
///
/// Returns the fragment a child at `self.position() + 1` would observe as its
/// `parent_hash_fragment` — the load-bearing invariant for chain alignment across mode
/// boundaries.
#[must_use]
pub fn router_block_hash(hash: SequenceHash) -> u64 {
    hash.parent_fragment_for_child_position(hash.position() + 1)
}

/// Project the parent block's u64 hash for the kv-router wire format.
/// Returns `None` at position 0 (root block).
#[must_use]
pub fn router_parent_hash(hash: SequenceHash) -> Option<u64> {
    if hash.position() == 0 {
        None
    } else {
        Some(hash.parent_hash_fragment())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_hashing::Request;
    use dynamo_tokens::PositionalLineageHash;

    /// Reference chain via the canonical kv-hashing path.
    fn build_chain(tokens: &[u32], block_size: u32) -> Vec<PositionalLineageHash> {
        Request::builder()
            .tokens(tokens.to_vec())
            .build()
            .expect("valid request")
            .into_blocks(block_size)
            .expect("into_blocks")
            .into_iter()
            .map(|b| b.plh)
            .collect()
    }

    #[test]
    fn router_parent_hash_is_none_at_position_zero() {
        let h = PositionalLineageHash::new(0x1234, None, 0);
        assert_eq!(router_parent_hash(h), None);
    }

    #[test]
    fn router_parent_hash_equals_parent_router_block_hash_at_pos_one() {
        let chain = build_chain(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
        assert_eq!(
            router_parent_hash(chain[1]),
            Some(router_block_hash(chain[0])),
        );
    }

    /// Cross-mode-boundary alignment: at 255→256 and 65535→65536, child's parent fragment
    /// must equal parent's router_block_hash even though the PLH packing changes mode.
    #[test]
    fn fragments_correct_at_mode_boundary_255_to_256() {
        // Build 257 single-token blocks so block 255 is at pos 255 (mode 0) and block 256
        // is at pos 256 (mode 1).
        let tokens: Vec<u32> = (0u32..257).collect();
        let chain = build_chain(&tokens, 1);
        assert_eq!(chain[256].position(), 256);
        assert_eq!(
            router_parent_hash(chain[256]),
            Some(router_block_hash(chain[255])),
            "cross-mode boundary 255→256 must align"
        );
    }
}
