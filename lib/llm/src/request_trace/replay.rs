// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replay-oriented request hash capture for live traces.

use bytemuck::cast_slice;
use std::sync::{Arc, Mutex};

use dynamo_kv_router::protocols::{
    BlockHashOptions, LocalBlockHash, XXH3_SEED, compute_block_hash_for_seq, compute_next_seq_hash,
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
        output_sequence_hashes: Vec::new(),
    })
}

pub(crate) type SharedOutputSequenceHashCapture = Arc<Mutex<OutputSequenceHashCapture>>;

/// Captures a generated continuation as sequence hashes while retaining no more
/// than one unfinished KV block of raw token IDs.
pub(crate) struct OutputSequenceHashCapture {
    trace_block_size: usize,
    pending_tokens: Vec<TokenIdType>,
    parent_sequence_hash: Option<u64>,
    output_sequence_hashes: Vec<u64>,
    output_tokens_seen: bool,
}

impl OutputSequenceHashCapture {
    fn new(input_tokens: &[TokenIdType], replay: &RequestReplayMetrics) -> Self {
        let input_remainder = input_tokens.len() % replay.trace_block_size;
        let full_input_blocks = input_tokens.len() / replay.trace_block_size;
        let parent_sequence_hash = if input_remainder == 0 {
            replay.input_sequence_hashes.last().copied()
        } else {
            full_input_blocks
                .checked_sub(1)
                .and_then(|index| replay.input_sequence_hashes.get(index).copied())
        };

        Self {
            trace_block_size: replay.trace_block_size,
            pending_tokens: input_tokens[input_tokens.len() - input_remainder..].to_vec(),
            parent_sequence_hash,
            output_sequence_hashes: Vec::new(),
            output_tokens_seen: false,
        }
    }

    pub(crate) fn record(&mut self, token_ids: &[TokenIdType]) {
        self.output_tokens_seen |= !token_ids.is_empty();
        for &token_id in token_ids {
            self.pending_tokens.push(token_id);
            if self.pending_tokens.len() == self.trace_block_size {
                let block = std::mem::take(&mut self.pending_tokens);
                self.push_block(&block);
            }
        }
    }

    pub(crate) fn sequence_hashes(&self) -> Vec<u64> {
        if !self.output_tokens_seen {
            return Vec::new();
        }

        let mut hashes = self.output_sequence_hashes.clone();
        if !self.pending_tokens.is_empty() {
            let sequence_hash =
                self.next_sequence_hash(partial_local_block_hash(&self.pending_tokens));
            hashes.push(sequence_hash);
        }
        hashes
    }

    fn push_block(&mut self, tokens: &[TokenIdType]) {
        let local_hash = compute_block_hash_for_seq(
            tokens,
            self.trace_block_size as u32,
            BlockHashOptions::default(),
        )[0];
        let sequence_hash = self.next_sequence_hash(local_hash);
        self.parent_sequence_hash = Some(sequence_hash);
        self.output_sequence_hashes.push(sequence_hash);
    }

    fn next_sequence_hash(&self, local_hash: LocalBlockHash) -> u64 {
        self.parent_sequence_hash
            .map(|parent| compute_next_seq_hash(parent, local_hash))
            .unwrap_or(local_hash.0)
    }
}

pub(crate) fn output_sequence_hash_capture(
    input_tokens: &[TokenIdType],
    replay: &RequestReplayMetrics,
) -> SharedOutputSequenceHashCapture {
    Arc::new(Mutex::new(OutputSequenceHashCapture::new(
        input_tokens,
        replay,
    )))
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

    use super::{input_sequence_hashes, output_sequence_hash_capture};

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

    #[test]
    fn output_hashes_extend_the_input_chain_across_a_partial_block() {
        let input = vec![1_u32, 2, 3];
        let replay = super::replay_metrics(&input, 2).unwrap();
        let capture = output_sequence_hash_capture(&input, &replay);
        capture.lock().unwrap().record(&[4, 5, 6]);

        let mut expected = input;
        expected.extend([4, 5, 6]);
        let expected = input_sequence_hashes(&expected, 2);
        assert_eq!(capture.lock().unwrap().sequence_hashes(), expected[1..]);
    }

    #[test]
    fn output_hashes_keep_no_raw_tokens_after_a_full_block() {
        let input = vec![1_u32, 2];
        let replay = super::replay_metrics(&input, 2).unwrap();
        let capture = output_sequence_hash_capture(&input, &replay);
        capture.lock().unwrap().record(&[3, 4]);

        let capture = capture.lock().unwrap();
        assert!(capture.pending_tokens.is_empty());
        assert_eq!(capture.sequence_hashes().len(), 1);
    }

    #[test]
    fn output_hashes_do_not_depend_on_backend_chunk_boundaries() {
        let input = vec![1_u32, 2, 3];
        let replay = super::replay_metrics(&input, 2).unwrap();
        let one_chunk = output_sequence_hash_capture(&input, &replay);
        one_chunk.lock().unwrap().record(&[4, 5, 6, 7]);

        let many_chunks = output_sequence_hash_capture(&input, &replay);
        many_chunks.lock().unwrap().record(&[4]);
        many_chunks.lock().unwrap().record(&[5, 6]);
        many_chunks.lock().unwrap().record(&[7]);

        assert_eq!(
            one_chunk.lock().unwrap().sequence_hashes(),
            many_chunks.lock().unwrap().sequence_hashes()
        );
    }
}
