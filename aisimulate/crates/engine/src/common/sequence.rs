// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::common::hashing::{
    BlockHash, SequenceHash, compute_block_hash_for_tokens, compute_next_sequence_hash,
};
use rand::random;

const MOCKER_SALT_HASH: u64 = 1337;

#[derive(Debug)]
struct FlatTokens {
    retained: Vec<u32>,
    retained_start: usize,
}

impl FlatTokens {
    fn new(
        mut tokens: Vec<u32>,
        output_capacity_hint: usize,
        block_size: usize,
        retain_history: bool,
    ) -> Self {
        if retain_history {
            tokens.reserve_exact(output_capacity_hint);
            return Self {
                retained: tokens,
                retained_start: 0,
            };
        }

        let retained_start = tokens.len() - (tokens.len() % block_size);
        // Lazy promotion runs after the next block's first token is pushed, so
        // the retained window must hold one complete block plus that token.
        let mut retained = Vec::with_capacity(
            block_size
                .checked_add(1)
                .expect("flat token tail capacity overflow"),
        );
        retained.extend_from_slice(&tokens[retained_start..]);
        Self {
            retained,
            retained_start,
        }
    }

    fn len(&self) -> usize {
        self.retained_start
            .checked_add(self.retained.len())
            .expect("flat token length overflow")
    }

    fn push(&mut self, token: u32) {
        self.retained.push(token);
    }

    fn complete_block(&self, position: usize, block_size: usize) -> Option<&[u32]> {
        let start = position.checked_mul(block_size)?;
        debug_assert!(
            start >= self.retained_start,
            "promoted block precedes retained flat-token window"
        );
        let end = start.checked_add(block_size)?;
        let local_start = start.checked_sub(self.retained_start)?;
        let local_end = end.checked_sub(self.retained_start)?;
        self.retained.get(local_start..local_end)
    }

    fn discard_through(&mut self, absolute_end: usize) {
        let local_end = absolute_end
            .checked_sub(self.retained_start)
            .expect("promoted block precedes retained flat-token window");
        assert!(
            local_end <= self.retained.len(),
            "promoted block extends beyond retained flat-token window"
        );
        self.retained.drain(..local_end);
        self.retained_start = absolute_end;
    }
}

/// Logical identity for one native-G1 request block.
///
/// A missing sequence hash denotes the request's mutable partial tail. Native
/// G1 does not need a positional-lineage hash or external-tier metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockIdentity {
    pub(crate) sequence_hash: Option<SequenceHash>,
    pub(crate) local_hash: Option<BlockHash>,
}

impl BlockIdentity {
    pub(crate) fn partial() -> Self {
        Self {
            sequence_hash: None,
            local_hash: None,
        }
    }
}

/// Lightweight request-owned token and generation progress for native G1.
///
/// Physical block ownership and cache visibility live in the attached
/// `BlockRequestLease`; this type deliberately contains no physical IDs,
/// positional lineage, or allocation bookkeeping.
#[derive(Debug)]
pub(crate) struct RequestSequence {
    tokens: FlatTokens,
    block_size: usize,
    max_output_tokens: usize,
    generated_tokens: usize,
    planned_output_ids: Option<Vec<u32>>,
    num_input_tokens: usize,
    enable_prefix_caching: bool,
    emit_token_ids: bool,
    retain_local_hashes: bool,
}

impl RequestSequence {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        tokens: Vec<u32>,
        max_output_tokens: usize,
        output_capacity_hint: usize,
        block_size: usize,
        enable_prefix_caching: bool,
        retain_local_hashes: bool,
        emit_token_ids: bool,
        planned_output_ids: Option<Vec<u32>>,
    ) -> (Self, Vec<BlockIdentity>) {
        assert!(block_size >= 2, "block_size must be at least two");
        let num_input_tokens = tokens.len();
        let output_capacity_hint = output_capacity_hint.min(max_output_tokens);
        let completion_blocks = num_input_tokens
            .checked_add(output_capacity_hint)
            .expect("native request completion length overflow")
            .div_ceil(block_size);
        let emit_token_ids = emit_token_ids && enable_prefix_caching;
        let retain_local_hashes = retain_local_hashes && enable_prefix_caching;

        let mut identities = Vec::with_capacity(completion_blocks);
        let mut parent_hash = None;
        for block in tokens.chunks_exact(block_size) {
            let (sequence_hash, local_hash) = if enable_prefix_caching {
                let local_hash = compute_block_hash_for_tokens(block, MOCKER_SALT_HASH);
                let sequence_hash = parent_hash
                    .map(|parent| compute_next_sequence_hash(parent, local_hash))
                    .unwrap_or(local_hash);
                (sequence_hash, retain_local_hashes.then_some(local_hash))
            } else {
                (random::<u64>(), None)
            };
            identities.push(BlockIdentity {
                sequence_hash: Some(sequence_hash),
                local_hash,
            });
            parent_hash = Some(sequence_hash);
        }
        if !tokens.len().is_multiple_of(block_size) {
            identities.push(BlockIdentity::partial());
        }

        let sequence = Self {
            tokens: FlatTokens::new(tokens, output_capacity_hint, block_size, emit_token_ids),
            block_size,
            max_output_tokens,
            generated_tokens: 0,
            planned_output_ids,
            num_input_tokens,
            enable_prefix_caching,
            emit_token_ids,
            retain_local_hashes,
        };
        sequence.debug_assert_invariants(identities.iter().copied());
        (sequence, identities)
    }

    pub(crate) fn len(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn max_output_tokens(&self) -> usize {
        self.max_output_tokens
    }

    pub(crate) fn generated_tokens(&self) -> usize {
        self.generated_tokens
    }

    pub(crate) fn num_input_tokens(&self) -> usize {
        self.num_input_tokens
    }

    pub(crate) fn enable_prefix_caching(&self) -> bool {
        self.enable_prefix_caching
    }

    pub(crate) fn current_known_blocks(&self) -> usize {
        self.len().div_ceil(self.block_size)
    }

    pub(crate) fn to_completion_blocks(&self) -> usize {
        (self.num_input_tokens + self.max_output_tokens).div_ceil(self.block_size)
    }

    /// Append the next planned or synthetic token.
    ///
    /// Returns the token and whether this append opened a new partial block.
    pub(crate) fn generate_token(&mut self) -> (u32, bool) {
        assert!(
            self.generated_tokens < self.max_output_tokens,
            "Cannot generate more tokens: reached max_output_tokens limit"
        );
        let token = self
            .planned_output_ids
            .as_ref()
            .and_then(|ids| ids.get(self.generated_tokens).copied())
            .unwrap_or_else(random::<u32>);
        let opened_partial = self.len().is_multiple_of(self.block_size);
        self.tokens.push(token);
        self.generated_tokens += 1;
        (token, opened_partial)
    }

    pub(crate) fn complete_block_identity(
        &self,
        position: usize,
        parent_hash: Option<SequenceHash>,
    ) -> BlockIdentity {
        // Validate that the retained tail still contains the complete block
        // even when prefix caching is disabled. Finalization discards that
        // tail immediately afterward, so skipping this lookup would hide
        // request/lease progress drift.
        let tokens = self
            .tokens
            .complete_block(position, self.block_size)
            .unwrap_or_else(|| {
                panic!("native partial tail cannot be promoted without a complete token block")
            });
        if !self.enable_prefix_caching {
            return BlockIdentity {
                sequence_hash: Some(random::<u64>()),
                local_hash: None,
            };
        }
        let local_hash = compute_block_hash_for_tokens(tokens, MOCKER_SALT_HASH);
        let sequence_hash = parent_hash
            .map(|parent| compute_next_sequence_hash(parent, local_hash))
            .unwrap_or(local_hash);
        BlockIdentity {
            sequence_hash: Some(sequence_hash),
            local_hash: self.retain_local_hashes.then_some(local_hash),
        }
    }

    /// Materialize one complete block's token IDs when event emission requires it.
    ///
    /// # Panics
    ///
    /// Panics if token-ID event mode did not retain the full request history.
    pub(crate) fn block_token_ids(&self, position: usize) -> Option<Vec<u32>> {
        if !self.emit_token_ids {
            return None;
        }
        assert_eq!(
            self.tokens.retained_start, 0,
            "token-ID event mode must retain full request history"
        );
        self.tokens
            .complete_block(position, self.block_size)
            .map(<[u32]>::to_vec)
    }

    /// Compact the bounded native tail after its completed block is published.
    pub(crate) fn discard_completed_block(&mut self, position: usize) {
        if self.emit_token_ids {
            return;
        }
        let promoted_end = position
            .checked_add(1)
            .and_then(|blocks| blocks.checked_mul(self.block_size))
            .expect("native promoted-block boundary overflow");
        if promoted_end <= self.tokens.retained_start {
            return;
        }
        self.tokens.discard_through(promoted_end);
    }

    pub(crate) fn debug_assert_invariants(
        &self,
        _identities: impl ExactSizeIterator<Item = BlockIdentity>,
    ) {
        // The underscore keeps release builds warning-free; debug builds use
        // the iterator for the full logical-identity consistency check.
        #[cfg(debug_assertions)]
        {
            let identity_count = _identities.len();
            let aligned = self.len().is_multiple_of(self.block_size);
            debug_assert_eq!(identity_count, self.current_known_blocks());
            debug_assert!(
                _identities.enumerate().all(|(position, identity)| {
                    identity.sequence_hash.is_some() || (position + 1 == identity_count && !aligned)
                }),
                "only the final native block may be partial"
            );
            self.debug_assert_storage_invariants();
        }
    }

    /// Check only identities changed by one finalization plus the final tail.
    #[cfg(debug_assertions)]
    pub(crate) fn debug_assert_finalized_range(
        &self,
        identity_count: usize,
        finalized: impl IntoIterator<Item = BlockIdentity>,
        final_identity: Option<BlockIdentity>,
    ) {
        debug_assert_eq!(identity_count, self.current_known_blocks());
        debug_assert!(
            finalized
                .into_iter()
                .all(|identity| identity.sequence_hash.is_some()),
            "finalized native blocks must have sequence hashes"
        );
        let aligned = self.len().is_multiple_of(self.block_size);
        debug_assert!(
            final_identity.is_none_or(|identity| identity.sequence_hash.is_some() || !aligned),
            "only an unaligned final native block may be partial"
        );
        self.debug_assert_storage_invariants();
    }

    #[cfg(debug_assertions)]
    fn debug_assert_storage_invariants(&self) {
        if self.emit_token_ids {
            debug_assert_eq!(self.tokens.retained_start, 0);
        } else {
            debug_assert!(self.tokens.retained.len() <= self.block_size + 1);
        }
    }

    #[cfg(test)]
    pub(crate) fn token_capacity(&self) -> usize {
        self.tokens.retained.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sequence(
        tokens: Vec<u32>,
        max_output_tokens: usize,
        retain_local_hashes: bool,
        emit_token_ids: bool,
        planned_output_ids: Option<Vec<u32>>,
    ) -> (RequestSequence, Vec<BlockIdentity>) {
        RequestSequence::new(
            tokens,
            max_output_tokens,
            max_output_tokens,
            4,
            true,
            retain_local_hashes,
            emit_token_ids,
            planned_output_ids,
        )
    }

    #[test]
    fn new_sequence_creates_an_initial_partial_identity() {
        let (sequence, identities) = sequence(vec![0, 1, 2], 2, false, false, None);
        assert_eq!(sequence.num_input_tokens(), 3);
        assert_eq!(sequence.len(), 3);
        assert_eq!(identities, vec![BlockIdentity::partial()]);
    }

    #[test]
    fn crossing_a_block_boundary_opens_one_new_partial_identity() {
        let (mut sequence, identities) = sequence(vec![0, 1, 2], 3, false, false, None);
        assert_eq!(identities.len(), 1);
        let (_, opened_partial) = sequence.generate_token();
        assert!(
            !opened_partial,
            "completing the current block reuses its identity"
        );
        let completed = sequence.complete_block_identity(0, None);
        assert!(completed.sequence_hash.is_some());
        let (_, opened_partial) = sequence.generate_token();
        assert!(
            opened_partial,
            "the first token in the next block needs a new lease entry"
        );
        assert_eq!(sequence.current_known_blocks(), 2);
    }

    #[test]
    fn equivalent_token_histories_preserve_full_block_identity() {
        let (_, first_identities) = sequence(vec![0, 1, 2, 3], 1, true, false, None);
        let (mut second, _) = sequence(vec![0, 1, 2], 2, true, false, Some(vec![3, 4]));
        second.generate_token();
        assert_eq!(first_identities[0], second.complete_block_identity(0, None));
    }

    #[test]
    fn chained_identity_uses_the_previous_complete_block_as_parent() {
        let (_, identities) = sequence((0..8).collect(), 1, true, false, None);
        let first = identities[0]
            .sequence_hash
            .expect("first block is complete");
        let second_local = identities[1].local_hash.expect("local hashes are retained");
        assert_eq!(
            identities[1].sequence_hash,
            Some(compute_next_sequence_hash(first, second_local))
        );
        assert_ne!(identities[1].sequence_hash, Some(second_local));
    }

    #[test]
    fn planned_output_tokens_are_generated_exactly() {
        let (mut sequence, _) = sequence(vec![0, 1], 3, false, false, Some(vec![7, 8, 9]));
        assert_eq!(sequence.generate_token().0, 7);
        assert_eq!(sequence.generate_token().0, 8);
        assert_eq!(sequence.generate_token().0, 9);
        assert_eq!(sequence.generated_tokens(), 3);
        assert_eq!(sequence.len(), 5);
    }

    #[test]
    fn bounded_native_storage_keeps_only_one_complete_block_and_tail() {
        let (mut sequence, _) = sequence(vec![0, 1, 2], 32, false, false, None);
        for position in 0..4 {
            while sequence.len() < (position + 1) * 4 {
                sequence.generate_token();
            }
            let _ = sequence.complete_block_identity(position, None);
            sequence.discard_completed_block(position);
            assert!(sequence.token_capacity() <= 5);
        }
    }

    #[test]
    fn token_id_event_mode_retains_materialized_block_tokens() {
        let (mut sequence, _) = sequence(vec![0, 1, 2, 3], 4, true, true, Some(vec![4, 5, 6, 7]));
        assert_eq!(sequence.block_token_ids(0), Some(vec![0, 1, 2, 3]));
        for _ in 0..4 {
            sequence.generate_token();
        }
        assert_eq!(sequence.block_token_ids(1), Some(vec![4, 5, 6, 7]));
        sequence.discard_completed_block(1);
        assert_eq!(sequence.block_token_ids(0), Some(vec![0, 1, 2, 3]));
    }

    #[test]
    fn completion_footprint_is_bounded_by_requested_output() {
        let (sequence, identities) =
            RequestSequence::new(vec![0, 1, 2, 3, 4], 100, 3, 4, false, false, false, None);
        assert_eq!(sequence.current_known_blocks(), 2);
        assert_eq!(sequence.to_completion_blocks(), 27);
        assert_eq!(
            identities.len(),
            2,
            "storage hint, not declared OSL, bounds initial lease entries"
        );
    }
}
