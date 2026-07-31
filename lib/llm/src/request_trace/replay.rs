// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replay-oriented request hash capture for live traces.

use bytemuck::cast_slice;
use dynamo_kv_router::protocols::{
    BlockHashOptions, LocalBlockHash, XXH3_SEED, compute_block_hash_for_seq,
    compute_seq_hash_for_block,
};
use dynamo_tokens::compute_hash_v2;

use crate::protocols::TokenIdType;

use super::RequestReplayMetrics;

pub(crate) fn replay_metrics(
    token_ids: &[TokenIdType],
    trace_block_size: usize,
) -> Option<RequestReplayMetrics> {
    if trace_block_size == 0 {
        return None;
    }

    Some(RequestReplayMetrics {
        trace_block_size,
        input_length: token_ids.len(),
        input_sequence_hashes: input_sequence_hashes(token_ids, trace_block_size),
    })
}

pub(crate) fn input_sequence_hashes(
    token_ids: &[TokenIdType],
    trace_block_size: usize,
) -> Vec<u64> {
    assert!(
        trace_block_size > 0,
        "request trace replay block size must be positive"
    );

    // Keep this identical to the router/mocker sequence-aware hashing path so
    // replay preserves shared-prefix identity.
    let block_size = trace_block_size as u32;
    let mut block_hashes =
        compute_block_hash_for_seq(token_ids, block_size, BlockHashOptions::default());

    let full_token_count = block_hashes.len() * trace_block_size;
    if full_token_count < token_ids.len() {
        block_hashes.push(partial_local_block_hash(&token_ids[full_token_count..]));
    }

    compute_seq_hash_for_block(&block_hashes)
}

fn partial_local_block_hash(tokens: &[TokenIdType]) -> LocalBlockHash {
    LocalBlockHash(compute_hash_v2(cast_slice(tokens), XXH3_SEED))
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::input_sequence_hashes;

    #[test]
    fn shared_prefix_has_same_leading_sequence_hashes() {
        let prefix = vec![1_u32, 2, 3, 4];
        let extended = vec![1_u32, 2, 3, 4, 5, 6];

        let prefix_hashes = input_sequence_hashes(&prefix, 2);
        let extended_hashes = input_sequence_hashes(&extended, 2);

        assert_eq!(prefix_hashes.len(), 2);
        assert_eq!(extended_hashes.len(), 3);
        assert_eq!(extended_hashes[..2], prefix_hashes[..]);
    }

    #[test]
    fn same_tokens_at_different_positions_have_different_sequence_hashes() {
        let hashes = input_sequence_hashes(&[1_u32, 2, 1, 2], 2);

        assert_eq!(hashes.len(), 2);
        assert_ne!(hashes[0], hashes[1]);
    }

    #[test]
    fn empty_input_has_empty_sequence_hashes() {
        assert!(input_sequence_hashes(&[], 64).is_empty());
    }

    #[test]
    fn long_input_hashes_cover_every_token() {
        let tokens = (0..131_072_u32).collect::<Vec<_>>();
        let started = Instant::now();
        let hashes = input_sequence_hashes(&tokens, 64);
        eprintln!(
            "hashed {} input tokens into {} sequence hashes in {:?}",
            tokens.len(),
            hashes.len(),
            started.elapsed()
        );

        assert_eq!(hashes.len(), tokens.len() / 64);
    }
}
