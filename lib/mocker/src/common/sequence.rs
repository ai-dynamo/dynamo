// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::common::protocols::MoveBlock;
use derive_getters::Getters;
use dynamo_tokens::blocks::UniqueBlock;
use dynamo_tokens::{
    BlockHash, PositionalLineageHash, SaltHash, SequenceHash, TokenBlockSequence, Tokens,
    compute_block_hash_for_tokens, compute_next_sequence_hash,
};
use rand::random;
use validator::Validate;

const MOCKER_SALT_HASH: SaltHash = 1337;

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
/// G1 never needs a positional-lineage hash; KVBM continues to use
/// [`ActiveSequence`] and its legacy metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NativeBlockIdentity {
    pub(crate) sequence_hash: Option<SequenceHash>,
    pub(crate) local_hash: Option<BlockHash>,
}

impl NativeBlockIdentity {
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
    ) -> (Self, Vec<NativeBlockIdentity>) {
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
            identities.push(NativeBlockIdentity {
                sequence_hash: Some(sequence_hash),
                local_hash,
            });
            parent_hash = Some(sequence_hash);
        }
        if !tokens.len().is_multiple_of(block_size) {
            identities.push(NativeBlockIdentity::partial());
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

    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn block_size(&self) -> usize {
        self.block_size
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
    ) -> NativeBlockIdentity {
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
            return NativeBlockIdentity {
                sequence_hash: Some(random::<u64>()),
                local_hash: None,
            };
        }
        let local_hash = compute_block_hash_for_tokens(tokens, MOCKER_SALT_HASH);
        let sequence_hash = parent_hash
            .map(|parent| compute_next_sequence_hash(parent, local_hash))
            .unwrap_or(local_hash);
        NativeBlockIdentity {
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
        _identities: impl ExactSizeIterator<Item = NativeBlockIdentity>,
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
        finalized: impl IntoIterator<Item = NativeBlockIdentity>,
        final_identity: Option<NativeBlockIdentity>,
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

/// Create unique blocks, block hashes, and positional-lineage hashes from a
/// [`TokenBlockSequence`].
fn create_sequence_cache(
    tokens: &TokenBlockSequence,
    block_size: usize,
    enable_prefix_caching: bool,
) -> (Vec<UniqueBlock>, Vec<BlockHash>, Vec<PositionalLineageHash>) {
    let mut unique_blocks = Vec::with_capacity(tokens.blocks().len() + 1);
    let mut block_hashes = Vec::new();
    if enable_prefix_caching {
        block_hashes.reserve(tokens.blocks().len());
    }
    let mut plhs = Vec::with_capacity(tokens.blocks().len());

    for (pos, block) in tokens.blocks().iter().enumerate() {
        if enable_prefix_caching {
            block_hashes.push(block.block_hash());
            unique_blocks.push(UniqueBlock::FullBlock(block.sequence_hash()));
            plhs.push(block.positional_lineage_hash());
        } else {
            unique_blocks.push(UniqueBlock::FullBlock(random::<u64>()));
            plhs.push(PositionalLineageHash::new(
                random::<u64>(),
                None,
                pos as u64,
            ));
        }
    }

    // Only push the partial block if tokens count isn't a multiple of block_size
    if !tokens.total_tokens().is_multiple_of(block_size) {
        unique_blocks.push(UniqueBlock::default());
    }
    (unique_blocks, block_hashes, plhs)
}

/// A sequence that is actively being built, with the ability to add tokens and commit to hashes
/// TODO: reuse tokens
#[derive(Debug, Getters, Validate)]
pub struct ActiveSequence {
    unique_blocks: Vec<UniqueBlock>,
    block_hashes: Vec<BlockHash>,
    plhs: Vec<PositionalLineageHash>,

    #[getter(skip)]
    tokens: TokenBlockSequence,

    #[getter(copy)]
    #[validate(range(min = 2))]
    block_size: usize,

    #[getter(copy)]
    max_output_tokens: usize,

    #[getter(copy)]
    generated_tokens: usize,

    planned_output_ids: Option<Vec<u32>>,

    #[getter(copy)]
    num_input_tokens: usize,

    #[getter(copy)]
    num_allocated_tokens: usize,

    #[getter(copy)]
    enable_prefix_caching: bool,

    #[getter(copy)]
    emit_token_ids: bool,
}

impl ActiveSequence {
    fn promote_last_partial(&mut self) -> Option<MoveBlock> {
        let UniqueBlock::PartialBlock(uuid) = self.unique_blocks.last().cloned()? else {
            return None;
        };

        let parent_hash = self.unique_blocks[..self.unique_blocks.len() - 1]
            .last()
            .map(|block| match block {
                UniqueBlock::FullBlock(hash) => *hash,
                UniqueBlock::PartialBlock(_) => panic!("partial block cannot be a parent"),
            });
        let position = self.plhs.len();
        debug_assert_eq!(position + 1, self.len() / self.block_size);
        let last_complete = self.tokens.last_complete_block().unwrap_or_else(|| {
            panic!("partial sequence tail cannot be promoted without a complete token block")
        });
        let last_seq_hash = if self.enable_prefix_caching {
            last_complete.sequence_hash()
        } else {
            random::<u64>()
        };
        let last_block_hash = self
            .enable_prefix_caching
            .then(|| last_complete.block_hash());
        let last_plh = if self.enable_prefix_caching {
            last_complete.positional_lineage_hash()
        } else {
            PositionalLineageHash::new(random::<u64>(), None, position as u64)
        };
        let promote_token_ids = if self.emit_token_ids {
            Some(last_complete.tokens().to_vec())
        } else {
            None
        };
        if let Some(last_block_hash) = last_block_hash {
            self.block_hashes.push(last_block_hash);
        }
        self.plhs.push(last_plh);
        self.unique_blocks.pop();

        self.unique_blocks
            .push(UniqueBlock::FullBlock(last_seq_hash));

        Some(MoveBlock::Promote(
            uuid,
            last_seq_hash,
            parent_hash,
            last_block_hash,
            last_plh,
            promote_token_ids,
        ))
    }

    /// Create a new ActiveSequence instance with the provided tokens
    pub fn new(
        tokens: Vec<u32>,
        max_output_tokens: usize,
        block_size: Option<usize>,
        enable_prefix_caching: bool,
        emit_token_ids: bool,
    ) -> Self {
        Self::new_with_planned_output_ids(
            tokens,
            max_output_tokens,
            block_size,
            enable_prefix_caching,
            emit_token_ids,
            None,
        )
    }

    pub fn new_with_planned_output_ids(
        tokens: Vec<u32>,
        max_output_tokens: usize,
        block_size: Option<usize>,
        enable_prefix_caching: bool,
        emit_token_ids: bool,
        planned_output_ids: Option<Vec<u32>>,
    ) -> Self {
        let block_size = block_size.unwrap_or(64);
        let num_input_tokens = tokens.len();

        let tokens = Tokens::from(tokens).into_sequence(block_size as u32, Some(MOCKER_SALT_HASH));
        let (unique_blocks, block_hashes, plhs) =
            create_sequence_cache(&tokens, block_size, enable_prefix_caching);

        let seq = Self {
            unique_blocks,
            block_hashes,
            plhs,
            tokens,
            block_size,
            max_output_tokens,
            generated_tokens: 0,
            planned_output_ids,
            num_input_tokens,
            num_allocated_tokens: 0,
            enable_prefix_caching,
            emit_token_ids: emit_token_ids && enable_prefix_caching,
        };
        seq.validate().expect("invalid ActiveSequence");
        seq
    }

    pub fn extra_tokens(&self) -> u32 {
        (self.len() % self.block_size) as u32
    }

    pub fn len(&self) -> usize {
        self.tokens.total_tokens()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Current known sequence footprint in blocks: prompt plus generated tokens.
    pub(crate) fn current_known_blocks(&self) -> usize {
        self.len().div_ceil(self.block_size)
    }

    /// To-completion footprint in blocks: `ceil((prompt + max_output) / block_size)`.
    ///
    /// The full physical residency a request needs to run end to end, with no
    /// prefix-reuse or already-allocated discount. Callers deciding "can it be
    /// admitted now?" apply their own discounts on top of this primitive.
    pub(crate) fn to_completion_blocks(&self) -> usize {
        (self.num_input_tokens + self.max_output_tokens).div_ceil(self.block_size)
    }

    /// Build a `MoveBlock::Use` signal for blocks up to `cumulative_tokens`
    /// without updating internal state. Returns `None` if no new blocks are needed.
    /// Call `commit_allocation` after the signal is successfully processed.
    pub fn prepare_allocation(&self, cumulative_tokens: usize) -> Option<MoveBlock> {
        let prev_blocks = self
            .num_allocated_tokens
            .div_ceil(self.block_size)
            .min(self.unique_blocks.len());
        let target_blocks = cumulative_tokens
            .div_ceil(self.block_size)
            .min(self.unique_blocks.len());
        if target_blocks <= prev_blocks {
            return None;
        }

        let range = prev_blocks..target_blocks;
        let blocks = self.unique_blocks[range.clone()].to_vec();

        let hash_start = prev_blocks.min(self.block_hashes.len());
        let hash_end = target_blocks.min(self.block_hashes.len());
        let hashes = self.block_hashes[hash_start..hash_end].to_vec();
        // Cached per-sequence PLHs (stable across calls).
        let plh_start = prev_blocks.min(self.plhs.len());
        let plh_end = target_blocks.min(self.plhs.len());
        let plhs = self.plhs[plh_start..plh_end].to_vec();

        let token_ids = if self.emit_token_ids && hash_start < hash_end {
            Some(self.block_token_ids_in(hash_start, hash_end))
        } else {
            None
        };

        let parent = if prev_blocks > 0 {
            Some(self.unique_blocks[prev_blocks - 1].clone())
        } else {
            None
        };
        Some(MoveBlock::Use(blocks, hashes, plhs, token_ids, parent))
    }

    /// Positional lineage hashes for all fully-tokenised blocks in the sequence.
    /// Mirrors `block_hashes()` but returns the PLH identity used by kvbm-logical.
    pub fn positional_lineage_hashes(&self) -> &[PositionalLineageHash] {
        &self.plhs
    }

    fn block_token_ids_in(&self, start: usize, end: usize) -> Vec<Vec<u32>> {
        self.tokens.blocks()[start..end]
            .iter()
            .map(|block| block.tokens().to_vec())
            .collect()
    }

    /// Materialize every complete block's token IDs.
    pub fn block_token_ids(&self) -> Vec<Vec<u32>> {
        self.block_token_ids_in(0, self.len() / self.block_size)
    }

    /// Commit a successful allocation by advancing `num_allocated_tokens`.
    pub fn commit_allocation(&mut self, cumulative_tokens: usize) {
        self.num_allocated_tokens = cumulative_tokens;
    }

    /// Prepare + commit in one call (convenience for paths where failure is impossible).
    pub fn allocate_blocks_for_chunk(&mut self, cumulative_tokens: usize) -> Option<MoveBlock> {
        let signal = self.prepare_allocation(cumulative_tokens);
        self.commit_allocation(cumulative_tokens);
        signal
    }

    /// Allocate all remaining blocks at once (backward compat).
    pub fn take_creation_signal(&mut self) -> Option<MoveBlock> {
        self.allocate_blocks_for_chunk(self.len())
    }

    /// Create a new ActiveSequence instance and return the creation signal
    pub fn new_with_signal(
        tokens: Vec<u32>,
        max_output_tokens: usize,
        block_size: Option<usize>,
        enable_prefix_caching: bool,
    ) -> (Self, Option<MoveBlock>) {
        let mut sequence = Self::new(
            tokens,
            max_output_tokens,
            block_size,
            enable_prefix_caching,
            false,
        );
        let signal = sequence.take_creation_signal();
        (sequence, signal)
    }

    /// Push a token to the sequence
    #[cfg_attr(feature = "profile", inline(never))]
    pub fn push(&mut self, token: u32) -> Option<Vec<MoveBlock>> {
        self.tokens.append(token).expect("Token push failed.");
        self.generated_tokens += 1;

        if self.len() % self.block_size != 1 {
            return None;
        }

        // Add a partial block for the first token in a new partial sequence
        // Send Use signal (to allocate space for this new generation block)
        let mut signals = Vec::new();

        // The scheduler may already have promoted this block at its computed
        // boundary. Retain this fallback for callers that have not.
        if let Some(promote) = self.promote_last_partial() {
            signals.push(promote);
        }

        let new_partial_block = UniqueBlock::default();
        self.unique_blocks.push(new_partial_block.clone());
        signals.push(MoveBlock::Use(
            vec![new_partial_block],
            vec![],
            vec![],
            None,
            None,
        ));
        Some(signals)
    }

    /// Generate a random token, push it to the sequence, and increment generation count.
    ///
    /// This function:
    /// - Generates a random token and adds it to the current sequence
    /// - Acquires a new partial block if needed or promotes an existing partial block to a full block
    /// - Returns appropriate signals for the G1 manager to process
    ///
    /// # Panics
    ///
    /// Calling this function when max_output_tokens has already been reached will cause a panic.
    /// Always check `generated_tokens < max_output_tokens` before calling this method.
    #[cfg_attr(feature = "profile", inline(never))]
    pub fn generate(&mut self) -> Vec<MoveBlock> {
        self.generate_token().1
    }

    /// Generate the next output token, push it to the sequence, and return the
    /// token alongside any KV movement signals.
    #[cfg_attr(feature = "profile", inline(never))]
    pub fn generate_token(&mut self) -> (u32, Vec<MoveBlock>) {
        // Assert that we haven't reached the maximum output tokens
        assert!(
            self.generated_tokens < self.max_output_tokens,
            "Cannot generate more tokens: reached max_output_tokens limit"
        );

        let token = self
            .planned_output_ids
            .as_ref()
            .and_then(|ids| ids.get(self.generated_tokens).copied())
            .unwrap_or_else(random::<u32>);

        // Collect signals
        let mut signals = Vec::new();

        // Push the token to the sequence and collect any signals
        if let Some(move_blocks) = self.push(token) {
            signals.extend(move_blocks);
        }

        // Check if we've reached the limit after pushing
        if self.generated_tokens != self.max_output_tokens {
            return (token, signals);
        }

        // Free all blocks when we reach max tokens
        signals.extend(self.terminal_signals());
        (token, signals)
    }

    /// Release the full sequence footprint after an independent terminal
    /// condition, such as the model context-length limit, is reached.
    pub(crate) fn terminal_signals(&self) -> Vec<MoveBlock> {
        self.free_signal_for_tokens(self.len())
    }

    fn free_signal_for_tokens(&self, active_tokens: usize) -> Vec<MoveBlock> {
        let active_blocks = active_tokens
            .div_ceil(self.block_size)
            .min(self.unique_blocks.len());
        if active_blocks == 0 {
            return Vec::new();
        }

        let blocks = self.unique_blocks[..active_blocks]
            .iter()
            .rev()
            .cloned()
            .collect();
        vec![MoveBlock::Deref(blocks)]
    }

    /// Free the currently active allocation footprint.
    pub fn free_signal(&self) -> Vec<MoveBlock> {
        self.free_signal_for_tokens(self.num_allocated_tokens)
    }

    /// Move the request to a preempted state and return the free signals from freeing current blocks.
    /// Upon preemption, the sequence retains the tokens generated during the decode phase (if any).
    /// Resets `num_allocated_tokens` so re-admission will re-allocate from scratch.
    pub fn reset_with_signal(&mut self) -> Vec<MoveBlock> {
        let free_signal = self.free_signal();
        self.num_allocated_tokens = 0;
        free_signal
    }

    /// Pops the last token in the sequence.
    ///
    /// This is only used to undo a freshly generated decode token after a failed
    /// allocation/preemption path. Under that invariant, the token being removed
    /// must be in the current partial block, so we only need to drop the trailing
    /// partial `UniqueBlock` when the sequence length returns to an exact block
    /// boundary. Using this to unwind arbitrary prompt history would be incorrect.
    ///
    /// If this contract is violated in release builds, legacy token storage
    /// preserves its historical no-op on an empty buffer.
    pub fn pop(&mut self) {
        debug_assert!(
            self.generated_tokens > 0,
            "sequence rollback requires a freshly generated token"
        );
        self.tokens.pop();
        self.generated_tokens = self.generated_tokens.saturating_sub(1);

        // Reverts to the last full block
        if self.len().is_multiple_of(self.block_size) {
            self.unique_blocks.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block_hashes_from_tokens(seq: &ActiveSequence) -> Vec<BlockHash> {
        seq.tokens
            .blocks()
            .iter()
            .map(|block| block.block_hash())
            .collect()
    }

    fn assert_cached_hashes_match_promoted_blocks(seq: &ActiveSequence) {
        let num_full_unique_blocks = seq
            .unique_blocks()
            .iter()
            .filter(|block| matches!(block, UniqueBlock::FullBlock(_)))
            .count();
        assert_eq!(
            seq.block_hashes().as_slice(),
            &block_hashes_from_tokens(seq)[..num_full_unique_blocks],
            "cached block hashes should match the promoted full blocks"
        );
    }

    fn assert_use_signal(
        signal: &MoveBlock,
        expected_blocks: &[UniqueBlock],
        expected_hashes: &[BlockHash],
    ) {
        match signal {
            MoveBlock::Use(blocks, hashes, ..) => {
                assert_eq!(blocks, expected_blocks);
                assert_eq!(hashes, expected_hashes);
            }
            _ => panic!("Expected MoveBlock::Use"),
        }
    }

    fn assert_single_partial_use(signal: &MoveBlock) {
        match signal {
            MoveBlock::Use(blocks, hashes, ..) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UniqueBlock::PartialBlock(_)));
                assert!(hashes.is_empty());
            }
            _ => panic!("Expected MoveBlock::Use with a single partial block"),
        }
    }

    fn assert_promote_parent(signal: &MoveBlock, expected_parent: Option<u64>) {
        match signal {
            MoveBlock::Promote(_, _, parent_hash, _hash, ..) => {
                assert_eq!(*parent_hash, expected_parent);
            }
            _ => panic!("Expected MoveBlock::Promote"),
        }
    }

    fn assert_deref_blocks(signal: &MoveBlock, expected: &[UniqueBlock]) {
        match signal {
            MoveBlock::Deref(blocks) => {
                assert_eq!(blocks, expected);
            }
            _ => panic!("Expected MoveBlock::Deref"),
        }
    }

    #[test]
    fn test_new_with_signal_creates_initial_partial_block() {
        let initial_tokens: Vec<u32> = (0..15).collect();
        let (seq, signal) = ActiveSequence::new_with_signal(initial_tokens, 100, Some(16), true);

        assert_eq!(seq.num_input_tokens(), 15);
        assert_eq!(seq.len(), 15);
        assert_single_partial_use(signal.as_ref().expect("Expected initial Use signal"));
    }

    #[test]
    fn test_push_across_block_boundary_promotes_and_allocates_partial() {
        let initial_tokens: Vec<u32> = (0..15).collect();
        let (mut seq, _) = ActiveSequence::new_with_signal(initial_tokens, 100, Some(16), true);

        let signal_15 = seq.push(15);
        assert!(
            signal_15.is_none(),
            "Completing a block should not trigger signals"
        );

        let signal_16 = seq.push(16).expect("Expected boundary crossing signals");
        assert_eq!(signal_16.len(), 2);
        assert_promote_parent(&signal_16[0], None);
        assert_single_partial_use(&signal_16[1]);

        assert_eq!(
            seq.unique_blocks().len(),
            2,
            "sequence should have one full block and one partial block"
        );
        assert_eq!(
            seq.len() % seq.block_size(),
            1,
            "sequence should have one token in the new partial block"
        );
    }

    #[test]
    fn test_equivalent_histories_preserve_full_block_identity() {
        let initial_tokens: Vec<u32> = (0..15).collect();
        let (mut seq1, _) = ActiveSequence::new_with_signal(initial_tokens, 100, Some(16), true);
        seq1.push(15);
        seq1.push(16);

        let extended_tokens: Vec<u32> = (0..16).collect();
        let (mut seq2, _) = ActiveSequence::new_with_signal(extended_tokens, 100, Some(16), true);
        seq2.push(16);
        seq2.pop();
        seq2.push(16);

        assert_eq!(seq1.unique_blocks()[0], seq2.unique_blocks()[0]);
        assert_ne!(seq1.unique_blocks()[1], seq2.unique_blocks()[1]);
    }

    #[test]
    fn test_promote_uses_previous_full_block_as_parent() {
        let initial_tokens: Vec<u32> = (0..15).collect();
        let (mut seq, _) = ActiveSequence::new_with_signal(initial_tokens, 100, Some(16), true);
        seq.push(15);
        seq.push(16);

        seq.push(17);
        seq.pop();
        seq.pop();
        seq.push(16);

        let extended_tokens: Vec<u32> = (0..16).collect();
        let (mut seq_equiv, _) =
            ActiveSequence::new_with_signal(extended_tokens, 100, Some(16), true);
        seq_equiv.push(16);
        seq_equiv.pop();
        seq_equiv.push(16);
        for token in 17..33 {
            seq.push(token);
            seq_equiv.push(token);
        }

        assert_eq!(
            &seq.unique_blocks()[0..2],
            &seq_equiv.unique_blocks()[0..2],
            "first two full blocks should remain identical"
        );

        for token in 33..48 {
            seq.push(token);
        }

        let signal = seq
            .push(48)
            .expect("Expected promote when opening next partial");

        let UniqueBlock::FullBlock(expected_hash) = seq.unique_blocks()[1] else {
            panic!("unique_blocks[1] should be a full block");
        };
        assert_promote_parent(&signal[0], Some(expected_hash));
        assert_single_partial_use(&signal[1]);
    }

    #[test]
    fn test_reset_with_signal_frees_blocks_and_resets_allocation() {
        let initial_tokens: Vec<u32> = (0..15).collect();
        let (mut seq, _) = ActiveSequence::new_with_signal(initial_tokens, 100, Some(16), true);
        seq.push(15);
        seq.push(16);
        seq.commit_allocation(seq.len());

        let free_signals = seq.reset_with_signal();

        assert_eq!(free_signals.len(), 1);
        let expected = seq
            .unique_blocks()
            .iter()
            .rev()
            .cloned()
            .collect::<Vec<_>>();
        assert_deref_blocks(&free_signals[0], &expected);
        assert_eq!(seq.num_allocated_tokens(), 0);
        assert_eq!(seq.generated_tokens(), 2);
    }

    #[test]
    fn test_free_signal_is_empty_without_an_active_allocation() {
        let seq = ActiveSequence::new((0..10).collect(), 4, Some(4), true, false);

        assert!(seq.free_signal().is_empty());
    }

    #[test]
    fn test_free_signal_batches_allocated_blocks_in_reverse_order() {
        let mut seq = ActiveSequence::new((0..10).collect(), 4, Some(4), true, false);
        seq.commit_allocation(seq.len());

        let expected = seq
            .unique_blocks()
            .iter()
            .rev()
            .cloned()
            .collect::<Vec<_>>();
        let signals = seq.free_signal();

        assert_eq!(signals.len(), 1);
        assert_deref_blocks(&signals[0], &expected);
    }

    #[test]
    fn test_active_sequence_generate_signals() {
        // Create a sequence with block size 16, max_output_tokens 4, initialized with tokens [0..14)
        let initial_tokens: Vec<u32> = (0..14).collect();
        let (mut seq, signal) = ActiveSequence::new_with_signal(initial_tokens, 5, Some(16), true);

        // Initial signal - should have received a Use signal for the partial block
        assert_single_partial_use(signal.as_ref().expect("Expected initial Use signal"));

        // Generate first two tokens - should not trigger new signals
        seq.generate();
        let signals_first = seq.generate();
        assert_eq!(signals_first.len(), 0);

        // Generate third token - this fills the block and should trigger both Promote and Use signals
        let signals_second = seq.generate();
        assert_eq!(signals_second.len(), 2);

        // First signal should be Promote
        assert_promote_parent(&signals_second[0], None);

        // Second signal should be Use for new partial block
        assert_single_partial_use(&signals_second[1]);

        // Generate fourth token - should not trigger new signals as it's adding to partial block
        let signals_third = seq.generate();
        assert_eq!(signals_third.len(), 0);

        // Generate last token - we reach max_output_tokens, so all blocks should
        // be dereferenced in one reverse-ordered batch.
        let expected = seq
            .unique_blocks()
            .iter()
            .rev()
            .cloned()
            .collect::<Vec<_>>();
        let signals_last = seq.generate();
        assert_eq!(signals_last.len(), 1);
        assert_deref_blocks(&signals_last[0], &expected);
    }

    #[test]
    fn test_prepare_allocation_slices_full_and_partial_blocks() {
        let tokens: Vec<u32> = (0..10).collect();
        let seq = ActiveSequence::new(tokens, 4, Some(4), true, false);

        let first = seq.prepare_allocation(4).unwrap();
        assert_use_signal(
            &first,
            &seq.unique_blocks()[0..1],
            &seq.block_hashes()[0..1],
        );

        let second = seq.prepare_allocation(8).unwrap();
        assert_use_signal(
            &second,
            &seq.unique_blocks()[0..2],
            &seq.block_hashes()[0..2],
        );

        let third = seq.prepare_allocation(10).unwrap();
        assert_use_signal(
            &third,
            &seq.unique_blocks()[0..3],
            &seq.block_hashes()[0..2],
        );
    }

    #[test]
    fn test_prepare_allocation_is_stable_until_commit() {
        let tokens: Vec<u32> = (0..10).collect();
        let mut seq = ActiveSequence::new(tokens, 4, Some(4), true, false);

        let first = seq.prepare_allocation(4).unwrap();
        let second = seq.prepare_allocation(4).unwrap();
        assert_eq!(first, second);

        seq.commit_allocation(4);
        let next = seq.prepare_allocation(8).unwrap();
        assert_use_signal(&next, &seq.unique_blocks()[1..2], &seq.block_hashes()[1..2]);
    }

    #[test]
    fn test_block_hash_cache_stays_in_sync_after_promote_and_pop() {
        let initial_tokens: Vec<u32> = (0..15).collect();
        let (mut seq, _) = ActiveSequence::new_with_signal(initial_tokens, 4, Some(16), true);

        assert_cached_hashes_match_promoted_blocks(&seq);

        seq.push(15);
        assert_cached_hashes_match_promoted_blocks(&seq);

        let promote_signals = seq.push(16).unwrap();
        assert_eq!(promote_signals.len(), 2);
        assert_cached_hashes_match_promoted_blocks(&seq);

        // `pop()` is only valid for undoing a freshly generated token from the
        // current partial block; this is the replay/preemption path we rely on.
        seq.pop();
        assert_cached_hashes_match_promoted_blocks(&seq);
    }
}
