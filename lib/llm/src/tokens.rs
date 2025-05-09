// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![allow(dead_code)]

//! Types and utilities for handling sequences of tokens, including block creation and hashing.

use bytemuck::cast_slice;
use derive_getters::Dissolve;
use rayon::prelude::*;
use std::ops::Range;

/// A token is represented as a 32-bit unsigned integer.
pub type Token = u32;

/// A salt used for hashing, represented as a vector of bytes.
/// This might encode model architecture, weights, PEFT info, etc.
pub type Salt = Vec<u8>;

/// A 64-bit hash of the salt, computed using [`compute_hash_v2`] with a seed of 0.
/// Used as the initial seed for subsequent block hashes.
pub type SaltHash = u64;

/// A 64-bit hash computed only from the tokens within a single block.
/// It uses [`compute_hash_v2`] with the [`SaltHash`] as the seed.
pub type BlockHash = u64;

/// A 64-bit sequence-aware hash.
/// It combines the previous block's [`SequenceHash`] (or the [`SaltHash`] for the first block)
/// with the current block's [`BlockHash`] using [`compute_hash_v2`] and the [`SaltHash`] as the seed.
pub type SequenceHash = u64;

/// Computes a hash of the data using the given seed.
pub fn compute_hash_v2(data: &[u8], seed: u64) -> u64 {
    xxhash_rust::xxh3::xxh3_64_with_seed(data, seed)
}

/// A collection of tokens, represented as a `Vec<Token>`.
///
/// Provides convenience methods for conversion and manipulation.
#[derive(Debug, Clone, Dissolve, Default, Eq)]
pub struct Tokens(Vec<Token>);

impl AsRef<[Token]> for Tokens {
    fn as_ref(&self) -> &[Token] {
        &self.0
    }
}

impl std::ops::Deref for Tokens {
    type Target = [Token];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::borrow::Borrow<[Token]> for Tokens {
    fn borrow(&self) -> &[Token] {
        &self.0
    }
}

impl From<Vec<Token>> for Tokens {
    fn from(tokens: Vec<Token>) -> Self {
        Tokens(tokens)
    }
}

impl From<&[Token]> for Tokens {
    fn from(tokens: &[Token]) -> Self {
        Tokens(tokens.to_vec())
    }
}

impl From<Vec<i32>> for Tokens {
    /// Converts `Vec<i32>` to `Tokens`, casting each `i32` to `u32`.
    fn from(tokens: Vec<i32>) -> Self {
        Tokens(tokens.into_iter().map(|t| t as u32).collect())
    }
}

impl From<&[i32]> for Tokens {
    /// Converts `&[i32]` to `Tokens`, casting each `i32` to `u32`.
    fn from(tokens: &[i32]) -> Self {
        Tokens(tokens.iter().map(|&t| t as u32).collect())
    }
}

impl From<Tokens> for Vec<Token> {
    fn from(tokens: Tokens) -> Self {
        tokens.0
    }
}

// PartialEq implementations for comparing Tokens with Vec<Token> and &[Token]
// (Generated implementations are usually sufficient, but explicit ones can be clearer)
impl PartialEq<Vec<Token>> for Tokens {
    fn eq(&self, other: &Vec<Token>) -> bool {
        self.0 == *other
    }
}

impl PartialEq<Tokens> for Vec<Token> {
    fn eq(&self, other: &Tokens) -> bool {
        *self == other.0
    }
}

impl PartialEq<[Token]> for Tokens {
    fn eq(&self, other: &[Token]) -> bool {
        self.0.as_slice() == other
    }
}

impl PartialEq<Tokens> for &[Token] {
    fn eq(&self, other: &Tokens) -> bool {
        *self == other.0.as_slice()
    }
}

impl PartialEq for Tokens {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

// Add PartialEq<&[T]> where T: Into<Token> + Copy could be more general,
// but specifically implementing for &[Token] is sufficient for the tests.
impl PartialEq<&[Token]> for Tokens {
    fn eq(&self, other: &&[Token]) -> bool {
        self.0.as_slice() == *other
    }
}

impl Tokens {
    /// Consumes the [`Tokens`] object and creates a [`TokenBlockSequence`].
    ///
    /// The sequence is initialized with the provided tokens, splitting them into blocks
    /// of the specified `block_size` using the given `salt_hash` (or 0 if `None`).
    ///
    /// # Arguments
    ///
    /// * `block_size` - The fixed size for each [`TokenBlock`].
    /// * `salt_hash` - An optional [`SaltHash`] used as the base seed for hashing. Defaults to 0.
    pub fn into_sequence(
        self,
        block_size: usize,
        salt_hash: Option<SaltHash>,
    ) -> TokenBlockSequence {
        TokenBlockSequence::new(self, block_size, salt_hash)
    }
}

/// Errors that can occur during [`PartialTokenBlock`] operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TokenBlockError {
    /// The operation could not be completed because the block is full.
    #[error("TokenBlock is full")]
    Full,

    /// The operation requires a full block, but the block is incomplete.
    #[error("TokenBlock is incomplete")]
    Incomplete,

    /// The operation could not be completed because the block is empty.
    #[error("TokenBlock is empty")]
    Empty,

    /// The operation requires more tokens than are currently in the block.
    #[error("TokenBlock has insufficient tokens")]
    InsufficientTokens,
}

/// Represents a partially filled block of tokens within a sequence.
///
/// This structure accumulates tokens until it reaches the specified `block_size`,
/// at which point it can be [`commit`](PartialTokenBlock::commit)ted into a full [`TokenBlock`].
#[derive(Debug)] // No Clone: intended to be unique within a sequence
pub struct PartialTokenBlock {
    tokens: Tokens,
    block_size: usize,
    salt_hash: SaltHash,
    parent_sequence_hash: Option<SequenceHash>,
}

impl PartialTokenBlock {
    /// Creates the first partial block (root) for a new sequence.
    ///
    /// # Arguments
    ///
    /// * `block_size` - The fixed size for blocks in this sequence.
    /// * `salt_hash` - The [`SaltHash`] for the sequence.
    pub(crate) fn create_sequence_root(block_size: usize, salt_hash: SaltHash) -> Self {
        Self {
            tokens: Tokens::default(),
            block_size,
            salt_hash,
            parent_sequence_hash: None, // Root has no parent
        }
    }

    /// Attempts to push a single token onto the block.
    ///
    /// # Arguments
    ///
    /// * `token` - The [`Token`] to push.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the token was successfully added.
    /// * `Err(TokenBlockError::Full)` - If the block already contains `block_size` tokens.
    pub(crate) fn push_token(&mut self, token: Token) -> Result<(), TokenBlockError> {
        if self.tokens.0.len() >= self.block_size {
            return Err(TokenBlockError::Full);
        }
        self.tokens.0.push(token);
        Ok(())
    }

    /// Attempts to push multiple tokens onto the block from a [`Tokens`] object.
    ///
    /// Tokens are added until the block is full or all input tokens are consumed.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The [`Tokens`] to push.
    ///
    /// # Returns
    ///
    /// A new [`Tokens`] object containing any tokens that did not fit,
    /// if all tokens were added, the returned object will be empty.
    pub(crate) fn push_tokens(&mut self, tokens: Tokens) -> Tokens {
        let remaining_space = self.remaining();

        if remaining_space == 0 {
            return tokens; // Block is already full
        }

        if tokens.0.len() <= remaining_space {
            // All tokens fit
            self.tokens.0.extend(tokens.0);
            Tokens::default() // No remaining tokens
        } else {
            // Only some tokens fit
            let (to_add, remaining) = tokens.0.split_at(remaining_space);
            self.tokens.0.extend_from_slice(to_add);
            Tokens(remaining.to_vec()) // Return the leftover tokens
        }
    }

    /// Attempts to remove the last token from the block.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If a token was successfully removed.
    /// * `Err(TokenBlockError::Empty)` - If the block was already empty.
    pub(crate) fn pop_token(&mut self) -> Result<(), TokenBlockError> {
        if self.tokens.0.is_empty() {
            return Err(TokenBlockError::Empty);
        }
        self.tokens.0.pop();
        Ok(())
    }

    /// Attempts to remove the last `count` tokens from the block.
    ///
    /// # Arguments
    ///
    /// * `count` - The number of tokens to remove.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the specified number of tokens were successfully removed.
    /// * `Err(TokenBlockError::InsufficientTokens)` - If `count` is greater than the number of tokens in the block.
    pub(crate) fn pop_tokens(&mut self, count: usize) -> Result<(), TokenBlockError> {
        if self.tokens.0.len() < count {
            return Err(TokenBlockError::InsufficientTokens);
        }
        self.tokens.0.truncate(self.tokens.0.len() - count);
        Ok(())
    }

    /// Attempts to commit the current partial block into a full [`TokenBlock`].
    ///
    /// This operation consumes the tokens within the partial block.
    /// After a successful commit, this `PartialTokenBlock` instance is reset
    /// to represent the *next* partial block in the sequence, inheriting the
    /// sequence hash from the block just committed.
    ///
    /// # Returns
    ///
    /// * `Ok(TokenBlock)` - The newly created full [`TokenBlock`].
    /// * `Err(TokenBlockError::Incomplete)` - If the block does not contain exactly `block_size` tokens.
    pub(crate) fn commit(&mut self) -> Result<TokenBlock, TokenBlockError> {
        if self.tokens.0.len() != self.block_size {
            // Check for exact size match for committing
            return Err(TokenBlockError::Incomplete);
        }

        // Take ownership of the tokens, leaving the internal tokens empty
        let tokens = std::mem::take(&mut self.tokens);

        let chunk = TokenBlockChunk::new(tokens, self.salt_hash);
        let block = TokenBlock::from_chunk(chunk, self.parent_sequence_hash);

        // Reset self to be the next block in the sequence
        self.parent_sequence_hash = Some(block.sequence_hash());
        // self.tokens is already empty due to mem::take
        // self.block_size and self.salt_hash remain the same

        Ok(block)
    }

    /// Returns the number of additional tokens required to fill the block.
    pub fn remaining(&self) -> usize {
        // Use saturating_sub to prevent underflow if len somehow exceeds block_size
        self.block_size.saturating_sub(self.tokens.0.len())
    }

    /// Returns the number of tokens currently in the block.
    pub fn len(&self) -> usize {
        self.tokens.0.len()
    }

    /// Returns `true` if the block contains no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.0.is_empty()
    }

    /// Returns a reference to the tokens currently in the block.
    pub fn tokens(&self) -> &Tokens {
        &self.tokens
    }
}

// Deref allows treating &PartialTokenBlock like &Tokens for read-only access.
impl std::ops::Deref for PartialTokenBlock {
    type Target = Tokens;

    fn deref(&self) -> &Self::Target {
        &self.tokens
    }
}

/// An intermediate structure holding a chunk of tokens destined to become a [`TokenBlock`].
///
/// This calculates the [`BlockHash`] but does not compute the final [`SequenceHash`],
/// allowing chunks to be processed independently (e.g., in parallel).
#[derive(Debug)] // No Clone: temporary intermediate value
struct TokenBlockChunk {
    tokens: Tokens,
    salt_hash: SaltHash,
    block_hash: BlockHash,
}

impl TokenBlockChunk {
    /// Creates a new chunk from [`Tokens`], calculating the [`BlockHash`].
    fn new(tokens: Tokens, salt_hash: SaltHash) -> Self {
        let block_hash = compute_hash_v2(cast_slice(&tokens), salt_hash);
        Self {
            tokens,
            salt_hash,
            block_hash,
        }
    }

    /// Creates a new chunk from a slice of `&[Token]`, calculating the [`BlockHash`].
    fn from_tokens(tokens: &[Token], salt_hash: SaltHash) -> Self {
        let block_hash = compute_hash_v2(cast_slice(tokens), salt_hash);
        Self {
            tokens: tokens.into(), // Converts slice to owned Tokens
            salt_hash,
            block_hash,
        }
    }
}

/// Represents a completed, immutable block of tokens with associated hashes.
///
/// Contains exactly `block_size` tokens and includes the [`SaltHash`], [`BlockHash`],
/// [`SequenceHash`], and optionally the parent's [`SequenceHash`].
#[derive(Debug, Clone, Default, PartialEq)] // Add PartialEq for tests
pub struct TokenBlock {
    tokens: Tokens,
    salt_hash: SaltHash,
    block_hash: BlockHash,
    sequence_hash: SequenceHash,
    parent_sequence_hash: Option<SequenceHash>,
}

impl TokenBlock {
    /// Creates a new [`PartialTokenBlock`] representing the block immediately following this one.
    ///
    /// The new partial block will have the correct `parent_sequence_hash` set.
    pub fn next_block(&self) -> PartialTokenBlock {
        PartialTokenBlock {
            tokens: Tokens::default(),
            block_size: self.tokens.len(), // Should be == self.block_size
            salt_hash: self.salt_hash,
            parent_sequence_hash: Some(self.sequence_hash), // Link to this block
        }
    }

    /// Finalizes a [`TokenBlock`] from a [`TokenBlockChunk`] and the parent's sequence hash.
    ///
    /// This computes the final [`SequenceHash`] for the block.
    fn from_chunk(chunk: TokenBlockChunk, parent_sequence_hash: Option<SequenceHash>) -> Self {
        let sequence_hash = match parent_sequence_hash {
            Some(parent) => {
                // Combine parent sequence hash and current block hash
                compute_hash_v2(cast_slice(&[parent, chunk.block_hash]), chunk.salt_hash)
            }
            None => {
                // First block: sequence hash is just the block hash
                chunk.block_hash
            }
        };

        Self {
            tokens: chunk.tokens,
            salt_hash: chunk.salt_hash,
            block_hash: chunk.block_hash,
            sequence_hash,
            parent_sequence_hash,
        }
    }

    /// Returns a reference to the tokens in this block.
    pub fn tokens(&self) -> &Tokens {
        &self.tokens
    }

    /// Returns the salt hash used for this block's hashing.
    pub fn salt_hash(&self) -> SaltHash {
        self.salt_hash
    }

    /// Returns the hash of only the tokens within this block.
    pub fn block_hash(&self) -> BlockHash {
        self.block_hash
    }

    /// Returns the sequence-aware hash for this block.
    pub fn sequence_hash(&self) -> SequenceHash {
        self.sequence_hash
    }

    /// Returns the sequence hash of the preceding block, if any.
    pub fn parent_sequence_hash(&self) -> Option<SequenceHash> {
        self.parent_sequence_hash
    }
}

/// Represents a sequence of tokens, segmented into fixed-size, hashed blocks.
///
/// This structure manages a series of completed [`TokenBlock`]s and one
/// [`PartialTokenBlock`] for accumulating incoming tokens.
/// It provides methods for appending tokens (`append`, `extend`), removing tokens
/// (`pop`, `truncate`, `unwind`), and accessing sequence information.
///
/// Hashing incorporates an initial [`SaltHash`] to ensure uniqueness across different
/// contexts (e.g., different models, PEFTs).
///
/// Key Hashes:
/// - [`BlockHash`]: Hash of tokens within a single block (seeded by [`SaltHash`]).
/// - [`SequenceHash`]: Hash combining the previous block's [`SequenceHash`] and the current
///   block's [`BlockHash`] (also seeded by [`SaltHash`]).
#[derive(Debug)]
pub struct TokenBlockSequence {
    blocks: Vec<TokenBlock>,
    current_block: PartialTokenBlock,
    salt_hash: SaltHash,
}

impl TokenBlockSequence {
    /// Creates a new [`TokenBlockSequence`] from an initial set of tokens.
    ///
    /// The tokens are split into blocks of `block_size`. Any remaining tokens
    /// form the initial `current_block`.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The initial [`Tokens`] for the sequence.
    /// * `block_size` - The fixed size for each [`TokenBlock`]. Must be greater than 0.
    /// * `salt_hash` - An optional [`SaltHash`]. Defaults to 0 if `None`.
    ///
    /// # Panics
    ///
    /// Panics if `block_size` is 0.
    pub fn new(tokens: Tokens, block_size: usize, salt_hash: Option<SaltHash>) -> Self {
        assert!(block_size > 0, "block_size must be greater than 0");
        let salt_hash = salt_hash.unwrap_or(0);
        let (blocks, current_block) = Self::split_tokens(&tokens, block_size, salt_hash);

        Self {
            blocks,
            current_block,
            salt_hash,
        }
    }

    /// Extends the sequence with the given tokens, potentially completing multiple blocks.
    ///
    /// This method processes all tokens from the input [`Tokens`] object.
    /// If adding tokens causes one or more blocks to become full, they are committed
    /// and added to the internal list of completed blocks.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The [`Tokens`] object containing the tokens to extend the sequence with.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(Range<usize>))` - The range of indices in the `blocks` vector corresponding
    ///   to the blocks completed during this `extend` operation.
    /// * `Ok(None)` - If no blocks were completed.
    /// * `Err(TokenBlockError)` - If an internal error occurs during commit.
    pub fn extend(&mut self, tokens: Tokens) -> Result<Option<Range<usize>>, TokenBlockError> {
        let start_block_index = self.blocks.len();
        let mut tokens_to_append = tokens;

        while !tokens_to_append.is_empty() {
            let remaining_in_current = self.current_block.remaining();

            if remaining_in_current == 0 {
                // Current block is full, commit it first
                let new_block = self.current_block.commit()?;
                self.blocks.push(new_block);
                // Continue loop to add tokens to the *new* current_block
            }

            // Push as many tokens as possible into the current (potentially new) block
            let available_tokens = tokens_to_append;
            tokens_to_append = self.current_block.push_tokens(available_tokens);

            // Check if the current block *became* full after pushing tokens
            if self.current_block.remaining() == 0 && !tokens_to_append.is_empty() {
                // If it became full AND there are still more tokens to append,
                // commit it now so the next loop iteration starts with a fresh block.
                let new_block = self.current_block.commit()?;
                self.blocks.push(new_block);
            }
            // If it became full and there are NO more tokens, the loop will exit,
            // and the block remains partial but full, ready for the next append/commit.
        }

        let end_block_index = self.blocks.len();
        if start_block_index == end_block_index {
            Ok(None) // No blocks were completed
        } else {
            Ok(Some(start_block_index..end_block_index))
        }
    }

    /// Appends a single token to the sequence.
    ///
    /// If adding this token completes the current partial block, the block is committed,
    /// and the index of the newly completed block is returned.
    ///
    /// This method is equivalent to calling [`extend`] with a single-token [`Tokens`] object.
    ///
    /// # Arguments
    ///
    /// * `token` - The [`Token`] to append.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(usize))` - The index of the block that was just completed.
    /// * `Ok(None)` - No block was completed by adding this token.
    /// * `Err(TokenBlockError)` - If an internal error occurs during processing.
    pub fn append(&mut self, token: Token) -> Result<Option<usize>, TokenBlockError> {
        // Create a single-token Tokens object
        let tokens = Tokens::from(vec![token]);

        // Call extend
        let range_option = self.extend(tokens)?;

        // Convert the range to Option<usize>
        match range_option {
            None => Ok(None),
            Some(range) => {
                // Since we only added one token, the range can only be empty or have one element.
                // If it's not empty, it must be `n..(n+1)`.
                assert_eq!(range.len(), 1, "Appending a single token completed more than one block, which should be impossible.");
                Ok(Some(range.start))
            }
        }
    }

    /// Shortens the sequence, keeping the first `len` tokens and removing the rest.
    ///
    /// If `len` is greater than the sequence's current length, this has no effect.
    ///
    /// This operation is analogous to `Vec::truncate`.
    /// It may involve removing tokens from the current partial block, removing entire
    /// completed blocks, and adjusting the current partial block
    /// to reflect the new end of the sequence.
    ///
    /// # Arguments
    ///
    /// * `len` - The number of tokens to keep.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the sequence was successfully truncated.
    /// * `Err(TokenBlockError::InsufficientTokens)` - This error should ideally not occur if `len`
    ///   is correctly checked against `total_tokens`, but the underlying `pop_tokens` might return it.
    pub fn truncate(&mut self, len: usize) -> Result<(), TokenBlockError> {
        let current_total_len = self.total_tokens();
        if len >= current_total_len {
            return Ok(()); // Nothing to truncate
        }

        let n = current_total_len - len; // Number of tokens to remove

        // This inner block handles the actual removal logic based on `n` tokens to remove.
        {
            let current_len = self.current_block.len();
            // Avoid division by zero if block_size is somehow 0 (though asserted in new)
            let block_size = self.current_block.block_size.max(1);

            if n <= current_len {
                // Only need to pop from the current partial block
                self.current_block.pop_tokens(n)?;
            } else {
                // Need to pop from full blocks as well
                let tokens_to_pop_from_blocks = n - current_len;

                // Calculate how many blocks are affected (including the one partially popped)
                let num_blocks_to_affect = tokens_to_pop_from_blocks.div_ceil(block_size);

                // Check if we need to pop more blocks than available (should be prevented by initial len check)
                if num_blocks_to_affect > self.blocks.len() {
                    // This indicates an inconsistency between total_tokens() and internal state.
                    debug_assert!(
                        false,
                        "Truncate calculation error: trying to pop too many blocks."
                    );
                    return Err(TokenBlockError::InsufficientTokens);
                }

                // Determine the index of the block that will be the source for the new partial block
                let source_block_index = self.blocks.len() - num_blocks_to_affect;

                // Calculate how many tokens to keep from that source block
                let num_full_blocks_completely_popped = num_blocks_to_affect - 1;
                let num_tokens_to_pop_from_source_block =
                    tokens_to_pop_from_blocks - num_full_blocks_completely_popped * block_size;
                let num_tokens_to_keep_in_new_partial =
                    block_size.saturating_sub(num_tokens_to_pop_from_source_block);

                // Get the tokens for the new partial block
                let new_partial_tokens = if num_tokens_to_keep_in_new_partial > 0 {
                    self.blocks[source_block_index].tokens().as_ref()
                        [..num_tokens_to_keep_in_new_partial]
                        .to_vec()
                } else {
                    Vec::new()
                };

                // Truncate the blocks vector to remove popped blocks
                self.blocks.truncate(source_block_index);

                // Update the current_block state
                self.current_block.tokens = Tokens(new_partial_tokens);
                // Correctly set the parent hash based on the *new* last block
                self.current_block.parent_sequence_hash =
                    self.blocks.last().map(|b| b.sequence_hash());
                // salt_hash and block_size remain the same for current_block
            }
        }
        Ok(())
    }

    /// Removes the last `count` tokens from the sequence.
    ///
    /// This is a convenience method that calculates the required length and calls [`truncate`].
    ///
    /// # Arguments
    ///
    /// * `count` - The number of tokens to remove from the end.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the tokens were successfully removed.
    /// * `Err(TokenBlockError::InsufficientTokens)` - If `count` is greater than or equal to
    ///   the total number of tokens in the sequence.
    pub fn unwind(&mut self, count: usize) -> Result<(), TokenBlockError> {
        let current_total_len = self.total_tokens();
        if count > current_total_len {
            // Allow count == current_total_len, which truncates to 0.
            return Err(TokenBlockError::InsufficientTokens);
        }

        // number of tokens remaining in the sequence after undoing the given count
        let len = current_total_len - count;
        self.truncate(len)
    }

    /// Removes the last token from the sequence and returns it, or [`None`] if it is empty.
    ///
    /// This operation is analogous to `Vec::pop`.
    ///
    /// # Returns
    ///
    /// * `Some(Token)` - The last token, if the sequence was not empty.
    /// * `None` - If the sequence was empty.
    pub fn pop(&mut self) -> Option<Token> {
        let current_total_len = self.total_tokens();
        if current_total_len == 0 {
            return None;
        }

        // Determine the last token. It must be in the current_block if current_block is not empty.
        // If current_block is empty, it must be the last token of the last full block.
        let last_token = if !self.current_block.tokens.is_empty() {
            // Last token is in the partial block
            *self
                .current_block
                .tokens
                .last()
                .expect("Current block checked for non-empty")
        } else {
            // Current block is empty, sequence is not. Must be in the last full block.
            let last_block = self
                .blocks
                .last()
                .expect("Sequence is not empty but has no blocks and empty current block?");
            *last_block
                .tokens()
                .last()
                .expect("Last block cannot be empty")
        };

        // Truncate the sequence by one element.
        // We expect this to succeed since we know the length > 0.
        match self.truncate(current_total_len - 1) {
            Ok(_) => Some(last_token),
            Err(_) => {
                // This should be logically impossible if total_tokens() and truncate() are correct.
                // Panic in debug, return None in release as a fallback, though it indicates a bug.
                debug_assert!(
                    false,
                    "truncate failed unexpectedly after checking length in pop"
                );
                None
            }
        }
    }

    /// Returns a slice containing all the completed [`TokenBlock`]s in the sequence.
    pub fn blocks(&self) -> &[TokenBlock] {
        &self.blocks
    }

    /// Returns a reference to the last completed [`TokenBlock`] in the sequence, if any.
    pub fn last_complete_block(&self) -> Option<&TokenBlock> {
        self.blocks.last()
    }

    /// Returns a reference to the current [`PartialTokenBlock`] where new tokens are added.
    pub fn current_block(&self) -> &PartialTokenBlock {
        &self.current_block
    }

    /// Consumes the sequence and returns its parts: a `Vec` of completed blocks and the final partial block.
    pub fn into_parts(self) -> (Vec<TokenBlock>, PartialTokenBlock) {
        (self.blocks, self.current_block)
    }

    /// Returns the [`SaltHash`] used for this sequence.
    pub fn salt_hash(&self) -> SaltHash {
        self.salt_hash
    }

    /// Returns the total number of tokens in the sequence (sum of tokens in all completed blocks
    /// plus tokens in the current partial block).
    pub fn total_tokens(&self) -> usize {
        let block_size = self.current_block.block_size;
        (self.blocks.len() * block_size) + self.current_block.len()
    }

    /// Splits a [`Tokens`] object into a vector of completed blocks and a final partial block.
    ///
    /// This is primarily used internally by [`TokenBlockSequence::new`] but can be used externally.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The [`Tokens`] to split.
    /// * `block_size` - The size of each block.
    /// * `salt_hash` - The [`SaltHash`] to use for hashing.
    ///
    /// # Returns
    ///
    /// A tuple containing `(Vec<TokenBlock>, PartialTokenBlock)`.
    ///
    /// # Panics
    ///
    /// Panics if `block_size` is 0.
    pub fn split_tokens(
        tokens: &[Token],
        block_size: usize,
        salt_hash: u64,
    ) -> (Vec<TokenBlock>, PartialTokenBlock) {
        assert!(block_size > 0, "block_size must be greater than 0");
        // Use Rayon for parallel computation of block chunks (hashes)
        let chunks: Vec<TokenBlockChunk> = tokens
            .as_ref()
            .par_chunks_exact(block_size)
            .map(|chunk| TokenBlockChunk::from_tokens(chunk, salt_hash))
            .collect();

        let mut result_blocks = Vec::with_capacity(chunks.len());
        let mut last_sequence_hash: Option<SequenceHash> = None;

        // Sequentially combine chunks to compute sequence hashes
        for chunk in chunks {
            let new_block = TokenBlock::from_chunk(chunk, last_sequence_hash);
            last_sequence_hash = Some(new_block.sequence_hash());
            result_blocks.push(new_block);
        }

        // Handle any remaining tokens
        let remainder = tokens.as_ref().chunks_exact(block_size).remainder();

        let current_block = PartialTokenBlock {
            tokens: remainder.into(),
            block_size,
            salt_hash,
            // Parent hash is the sequence hash of the last *full* block computed
            parent_sequence_hash: last_sequence_hash,
        };

        (result_blocks, current_block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::cast_slice;

    // Helper to create a sequence for testing
    fn create_test_sequence(
        initial_tokens: &[Token],
        block_size: usize,
        salt_hash: Option<SaltHash>,
    ) -> TokenBlockSequence {
        TokenBlockSequence::new(Tokens::from(initial_tokens), block_size, salt_hash)
    }

    // Helper to get expected hashes (replace with actual calculated values if needed)
    const TEST_SALT_HASH: SaltHash = 1337;
    const HASH_1_4: BlockHash = 14643705804678351452; // hash([1,2,3,4], 1337)
    const SEQ_HASH_1_4: SequenceHash = HASH_1_4;
    const HASH_5_8: BlockHash = 16777012769546811212; // hash([5,6,7,8], 1337)
    const SEQ_HASH_5_8: SequenceHash = 4945711292740353085; // hash([SEQ_HASH_1_4, HASH_5_8], 1337)
    const HASH_9_12: BlockHash = 483935686894639516; // hash([9,10,11,12], 1337)
    const SEQ_HASH_9_12: SequenceHash = 12583592247330656132; // hash([SEQ_HASH_5_8, HASH_9_12], 1337)

    #[test]
    fn test_validate_hash_constants() {
        let salt = TEST_SALT_HASH;

        // Block 1: [1, 2, 3, 4]
        let tokens_1_4 = &[1u32, 2, 3, 4];
        let computed_hash_1_4 = compute_hash_v2(cast_slice(tokens_1_4), salt);
        assert_eq!(computed_hash_1_4, HASH_1_4, "Mismatch for HASH_1_4");
        // First block's sequence hash is its block hash
        assert_eq!(computed_hash_1_4, SEQ_HASH_1_4, "Mismatch for SEQ_HASH_1_4");

        // Block 2: [5, 6, 7, 8]
        let tokens_5_8 = &[5u32, 6, 7, 8];
        let computed_hash_5_8 = compute_hash_v2(cast_slice(tokens_5_8), salt);
        assert_eq!(computed_hash_5_8, HASH_5_8, "Mismatch for HASH_5_8");
        let computed_seq_hash_5_8 = compute_hash_v2(cast_slice(&[SEQ_HASH_1_4, HASH_5_8]), salt);
        assert_eq!(
            computed_seq_hash_5_8, SEQ_HASH_5_8,
            "Mismatch for SEQ_HASH_5_8"
        );

        // Block 3: [9, 10, 11, 12]
        let tokens_9_12 = &[9u32, 10, 11, 12];
        let computed_hash_9_12 = compute_hash_v2(cast_slice(tokens_9_12), salt);
        assert_eq!(computed_hash_9_12, HASH_9_12, "Mismatch for HASH_9_12");
        let computed_seq_hash_9_12 = compute_hash_v2(cast_slice(&[SEQ_HASH_5_8, HASH_9_12]), salt);
        assert_eq!(
            computed_seq_hash_9_12, SEQ_HASH_9_12,
            "Mismatch for SEQ_HASH_9_12"
        );
    }

    #[test]
    fn test_tokens_from() {
        let vec_u32: Vec<u32> = vec![1, 2, 3];
        let tokens_u32: Tokens = vec_u32.clone().into();
        assert_eq!(tokens_u32.0, vec_u32);

        let slice_u32: &[u32] = &[4, 5];
        let tokens_slice_u32: Tokens = slice_u32.into();
        assert_eq!(tokens_slice_u32.0, vec![4, 5]);

        let vec_i32: Vec<i32> = vec![-1, 0, 1]; // Note: -1 becomes large u32
        let tokens_i32: Tokens = vec_i32.into();
        assert_eq!(tokens_i32.0, vec![u32::MAX, 0, 1]);

        let slice_i32: &[i32] = &[100, 200];
        let tokens_slice_i32: Tokens = slice_i32.into();
        assert_eq!(tokens_slice_i32.0, vec![100, 200]);

        let into_vec: Vec<u32> = tokens_slice_i32.into();
        assert_eq!(into_vec, vec![100, 200]);
    }

    #[test]
    fn test_tokens_equality() {
        let tokens = Tokens::from(vec![1, 2, 3]);
        assert_eq!(tokens, vec![1, 2, 3]);
        assert_eq!(vec![1, 2, 3], tokens);
        assert_eq!(tokens, &[1, 2, 3][..]);
        assert_eq!(&[1, 2, 3][..], tokens);
        assert_eq!(tokens, Tokens::from(vec![1, 2, 3]));
        assert_ne!(tokens, Tokens::from(vec![1, 2, 4]));
    }

    #[test]
    fn test_tokens_deref_asref() {
        let tokens = Tokens::from(vec![10, 20, 30]);

        // Deref to &[Token]
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[1], 20);
        let slice: &[Token] = &tokens;
        assert_eq!(slice, &[10, 20, 30]);

        // AsRef<[Token]>
        let as_ref_slice: &[Token] = tokens.as_ref();
        assert_eq!(as_ref_slice, &[10, 20, 30]);

        // Borrow<[Token]>
        let borrowed_slice: &[Token] = std::borrow::Borrow::borrow(&tokens);
        assert_eq!(borrowed_slice, &[10, 20, 30]);
    }

    #[test]
    fn test_tokens_into_sequence() {
        let tokens = Tokens::from(vec![1, 2, 3, 4, 5]);
        let seq = tokens.into_sequence(3, Some(TEST_SALT_HASH));
        assert_eq!(seq.blocks().len(), 1);
        assert_eq!(seq.blocks[0].tokens().as_ref(), &[1, 2, 3]);
        assert_eq!(seq.current_block().tokens().as_ref(), &[4, 5]);
        assert_eq!(seq.salt_hash(), TEST_SALT_HASH);
    }

    #[test]
    fn test_partial_block_ops() {
        let mut partial = PartialTokenBlock::create_sequence_root(3, TEST_SALT_HASH);
        assert_eq!(partial.len(), 0);
        assert_eq!(partial.remaining(), 3);
        assert!(partial.is_empty());

        // Push tokens
        assert!(partial.push_token(1).is_ok());
        assert_eq!(partial.len(), 1);
        assert_eq!(partial.remaining(), 2);
        let remaining = partial.push_tokens(Tokens::from(vec![2, 3, 4]));
        assert_eq!(partial.len(), 3);
        assert_eq!(partial.remaining(), 0);
        assert_eq!(remaining.as_ref(), &[4]); // Token 4 didn't fit
        assert_eq!(partial.tokens().as_ref(), &[1, 2, 3]);

        // Push when full
        assert_eq!(partial.push_token(5), Err(TokenBlockError::Full));
        let remaining_full = partial.push_tokens(Tokens::from(vec![5]));
        assert_eq!(remaining_full.as_ref(), &[5]);

        // Pop tokens
        assert!(partial.pop_token().is_ok());
        assert_eq!(partial.len(), 2);
        assert_eq!(partial.tokens().as_ref(), &[1, 2]);
        assert!(partial.pop_tokens(2).is_ok());
        assert!(partial.is_empty());

        // Pop when empty
        assert_eq!(partial.pop_token(), Err(TokenBlockError::Empty));
        assert_eq!(
            partial.pop_tokens(1),
            Err(TokenBlockError::InsufficientTokens)
        );

        // Commit incomplete
        assert!(partial.push_token(10).is_ok());
        assert_eq!(partial.commit(), Err(TokenBlockError::Incomplete));

        // Commit complete
        assert!(partial.push_token(11).is_ok());
        assert!(partial.push_token(12).is_ok());
        assert_eq!(partial.len(), 3);
        let commit_result = partial.commit();
        assert!(commit_result.is_ok());
        let committed_block = commit_result.unwrap();
        assert_eq!(committed_block.tokens().as_ref(), &[10, 11, 12]);

        // Check state after commit (partial block is now the next one)
        assert!(partial.is_empty());
        assert_eq!(
            partial.parent_sequence_hash,
            Some(committed_block.sequence_hash())
        );
        assert_eq!(partial.block_size, 3);
    }

    #[test]
    fn test_token_block_creation_and_hashes() {
        let salt = TEST_SALT_HASH;
        let tokens1 = Tokens::from(vec![1, 2, 3, 4]);
        let chunk1 = TokenBlockChunk::new(tokens1.clone(), salt);
        let block1 = TokenBlock::from_chunk(chunk1, None);

        assert_eq!(block1.tokens(), &tokens1);
        assert_eq!(block1.salt_hash(), salt);
        assert_eq!(block1.parent_sequence_hash(), None);
        assert_eq!(block1.block_hash(), HASH_1_4);
        assert_eq!(block1.sequence_hash(), SEQ_HASH_1_4); // First block seq_hash == block_hash

        let tokens2 = Tokens::from(vec![5, 6, 7, 8]);
        let chunk2 = TokenBlockChunk::new(tokens2.clone(), salt);
        let block2 = TokenBlock::from_chunk(chunk2, block1.parent_sequence_hash()); // Incorrect parent
                                                                                    // Sequence hash should differ if parent is wrong
        assert_ne!(block2.sequence_hash(), SEQ_HASH_5_8);

        let chunk2_correct = TokenBlockChunk::new(tokens2.clone(), salt);
        let block2_correct = TokenBlock::from_chunk(chunk2_correct, Some(block1.sequence_hash()));

        assert_eq!(block2_correct.tokens(), &tokens2);
        assert_eq!(block2_correct.salt_hash(), salt);
        assert_eq!(
            block2_correct.parent_sequence_hash(),
            Some(block1.sequence_hash())
        );
        assert_eq!(block2_correct.block_hash(), HASH_5_8);
        assert_eq!(block2_correct.sequence_hash(), SEQ_HASH_5_8);
    }

    #[test]
    fn test_new_sequence() {
        // Empty initial tokens
        let seq_empty = create_test_sequence(&[], 4, Some(TEST_SALT_HASH));
        assert!(seq_empty.blocks().is_empty());
        assert!(seq_empty.current_block().is_empty());
        assert_eq!(seq_empty.total_tokens(), 0);
        assert_eq!(seq_empty.salt_hash(), TEST_SALT_HASH);
        assert_eq!(seq_empty.current_block().parent_sequence_hash, None);

        // Less than one block
        let seq_partial = create_test_sequence(&[1, 2], 4, Some(TEST_SALT_HASH));
        assert!(seq_partial.blocks().is_empty());
        assert_eq!(seq_partial.current_block().tokens().as_ref(), &[1, 2]);
        assert_eq!(seq_partial.total_tokens(), 2);
        assert_eq!(seq_partial.current_block().parent_sequence_hash, None);

        // Exactly one block
        let seq_one_block = create_test_sequence(&[1, 2, 3, 4], 4, Some(TEST_SALT_HASH));
        assert_eq!(seq_one_block.blocks().len(), 1);
        assert!(seq_one_block.current_block().is_empty());
        assert_eq!(seq_one_block.total_tokens(), 4);
        assert_eq!(seq_one_block.blocks[0].tokens().as_ref(), &[1, 2, 3, 4]);
        assert_eq!(seq_one_block.blocks[0].sequence_hash(), SEQ_HASH_1_4);
        assert_eq!(
            seq_one_block.current_block().parent_sequence_hash,
            Some(SEQ_HASH_1_4)
        );

        // More than one block
        let seq_multi = create_test_sequence(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 4, Some(TEST_SALT_HASH));
        assert_eq!(seq_multi.blocks().len(), 2);
        assert_eq!(seq_multi.current_block().tokens().as_ref(), &[9]);
        assert_eq!(seq_multi.total_tokens(), 9);
        assert_eq!(seq_multi.blocks[0].sequence_hash(), SEQ_HASH_1_4);
        assert_eq!(seq_multi.blocks[1].sequence_hash(), SEQ_HASH_5_8);
        assert_eq!(
            seq_multi.current_block().parent_sequence_hash,
            Some(SEQ_HASH_5_8)
        );

        // No salt hash
        let seq_no_salt = create_test_sequence(&[1, 2, 3, 4, 5], 4, None);
        assert_eq!(seq_no_salt.salt_hash(), 0);
        assert_eq!(seq_no_salt.blocks().len(), 1);
        assert_ne!(seq_no_salt.blocks[0].block_hash(), HASH_1_4); // Hash differs with salt 0
        assert_eq!(seq_no_salt.current_block().tokens().as_ref(), &[5]);
    }

    #[test]
    #[should_panic]
    fn test_new_sequence_zero_block_size() {
        let _ = create_test_sequence(&[1], 0, None);
    }

    #[test]
    fn test_append_single_token() {
        let mut sequence =
            create_test_sequence(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4, Some(TEST_SALT_HASH));
        assert_eq!(sequence.blocks().len(), 2);
        assert_eq!(sequence.current_block().tokens.len(), 2);
        assert_eq!(sequence.current_block().tokens, vec![9, 10]);
        assert_eq!(
            sequence.current_block().parent_sequence_hash,
            Some(SEQ_HASH_5_8)
        );

        // Append token 11 - should not complete a block
        let completed_idx = sequence.append(11).unwrap();
        assert_eq!(completed_idx, None);
        assert_eq!(sequence.blocks().len(), 2);
        assert_eq!(sequence.current_block().tokens.as_ref(), &[9, 10, 11]);

        // Append token 12 - should complete block 2 (index 2)
        let completed_idx = sequence.append(12).unwrap();
        assert_eq!(completed_idx, None); // Lazy commit: extend returns None
        assert_eq!(sequence.blocks().len(), 2); // Block 2 not added yet
        assert_eq!(sequence.current_block.tokens.as_ref(), &[9, 10, 11, 12]); // Current block is now full
        assert_eq!(sequence.current_block.remaining(), 0);
        assert_eq!(
            sequence.current_block().parent_sequence_hash,
            Some(SEQ_HASH_5_8)
        ); // Still linked to block 1

        // Append token 13 - should not complete a block
        // NOW appending 13 should first commit block 2, then add 13 to the new current
        let completed_idx_13 = sequence.append(13).unwrap();
        assert_eq!(completed_idx_13, Some(2)); // Block 2 (index 2) was completed by this append
        assert_eq!(sequence.blocks.len(), 3); // Now 3 blocks committed
        assert_eq!(sequence.blocks[2].tokens().as_ref(), &[9, 10, 11, 12]); // Verify committed block 2
        assert_eq!(sequence.blocks[2].sequence_hash(), SEQ_HASH_9_12);
        assert_eq!(sequence.current_block.tokens.as_ref(), &[13]); // New current block has 13
        assert_eq!(sequence.current_block.remaining(), 3);
        assert_eq!(
            sequence.current_block.parent_sequence_hash,
            Some(SEQ_HASH_9_12)
        ); // Linked to new block 2
    }

    #[test]
    fn test_extend() {
        let block_size = 4;
        let salt_hash = Some(TEST_SALT_HASH);

        // Case 1: Extend less than block size
        let mut seq1 = create_test_sequence(&[], block_size, salt_hash);
        let tokens1 = Tokens::from(vec![1, 2]);
        let completed1 = seq1.extend(tokens1).unwrap();
        assert_eq!(completed1, None); // No blocks completed
        assert_eq!(seq1.blocks.len(), 0);
        assert_eq!(seq1.current_block.tokens.as_ref(), &[1, 2]);
        assert_eq!(seq1.current_block.remaining(), 2);

        // Case 2: Extend exactly block size
        let mut seq2 = create_test_sequence(&[], block_size, salt_hash);
        let tokens2 = Tokens::from(vec![1, 2, 3, 4]);
        let completed2 = seq2.extend(tokens2).unwrap();
        assert_eq!(completed2, None); // Block is full but not committed yet
        assert_eq!(seq2.blocks.len(), 0); // No blocks committed
        assert_eq!(seq2.current_block.tokens.as_ref(), &[1, 2, 3, 4]); // Current block is full
        assert_eq!(seq2.current_block.remaining(), 0);
        assert_eq!(seq2.current_block.parent_sequence_hash, None); // Still the root block

        // Case 3: Extend more than block size, less than two blocks
        let mut seq3 = create_test_sequence(&[], block_size, salt_hash);
        let tokens3 = Tokens::from(vec![1, 2, 3, 4, 5, 6]);
        let completed3 = seq3.extend(tokens3).unwrap();
        assert_eq!(completed3, Some(0..1)); // Block at index 0 completed
        assert_eq!(seq3.blocks.len(), 1);
        assert_eq!(seq3.current_block.tokens.as_ref(), &[5, 6]); // Partial block has remainder
        assert_eq!(seq3.blocks[0].tokens().as_ref(), &[1, 2, 3, 4]);
        assert_eq!(seq3.current_block.parent_sequence_hash, Some(SEQ_HASH_1_4));
        assert_eq!(seq3.current_block.remaining(), 2);

        // Case 4: Extend exactly two blocks
        let mut seq4 = create_test_sequence(&[], block_size, salt_hash);
        let tokens4 = Tokens::from(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let completed4 = seq4.extend(tokens4).unwrap();
        assert_eq!(completed4, Some(0..1)); // Only block 0 is committed
        assert_eq!(seq4.blocks.len(), 1); // Only 1 block committed
        assert_eq!(seq4.current_block.tokens.as_ref(), &[5, 6, 7, 8]); // Current block holds the second block's tokens
        assert_eq!(seq4.current_block.remaining(), 0); // Current block is full
        assert_eq!(seq4.blocks[0].tokens().as_ref(), &[1, 2, 3, 4]);
        assert_eq!(seq4.blocks[0].sequence_hash(), SEQ_HASH_1_4);
        assert_eq!(seq4.current_block.parent_sequence_hash, Some(SEQ_HASH_1_4)); // Parent is the first block

        // Case 5: Extend multiple times, completing blocks across calls
        let mut seq5 = create_test_sequence(&[], block_size, salt_hash);
        let tokens5a = Tokens::from(vec![1, 2]);
        let completed5a = seq5.extend(tokens5a).unwrap();
        assert_eq!(completed5a, None);
        assert_eq!(seq5.blocks.len(), 0);
        assert_eq!(seq5.current_block.tokens.as_ref(), &[1, 2]);

        let tokens5b = Tokens::from(vec![3, 4, 5]);
        let completed5b = seq5.extend(tokens5b).unwrap();
        assert_eq!(completed5b, Some(0..1)); // Block at index 0 completed
        assert_eq!(seq5.blocks.len(), 1);
        assert_eq!(seq5.current_block.tokens.as_ref(), &[5]);
        assert_eq!(seq5.blocks[0].tokens().as_ref(), &[1, 2, 3, 4]);
        assert_eq!(seq5.current_block.parent_sequence_hash, Some(SEQ_HASH_1_4));
        assert_eq!(seq5.current_block.remaining(), 3);

        let tokens5c = Tokens::from(vec![6, 7, 8, 9, 10]);
        let completed5c = seq5.extend(tokens5c).unwrap();
        assert_eq!(completed5c, Some(1..2)); // Block at index 1 completed
        assert_eq!(seq5.blocks.len(), 2);
        assert_eq!(seq5.current_block.tokens.as_ref(), &[9, 10]);
        assert_eq!(seq5.blocks[1].tokens().as_ref(), &[5, 6, 7, 8]);
        assert_eq!(seq5.current_block.parent_sequence_hash, Some(SEQ_HASH_5_8));
        assert_eq!(seq5.current_block.remaining(), 2);

        // Case 6: Extend empty tokens
        let mut seq6 = create_test_sequence(&[1], block_size, salt_hash);
        let completed6 = seq6.extend(Tokens::default()).unwrap();
        assert_eq!(completed6, None);
        assert_eq!(seq6.blocks.len(), 0);
        assert_eq!(seq6.current_block.tokens.as_ref(), &[1]);
        assert_eq!(seq6.total_tokens(), 1);

        // Case 7: Extend fills current exactly, no remainder
        let mut seq7 = create_test_sequence(&[1, 2], block_size, salt_hash);
        let tokens7 = Tokens::from(vec![3, 4]);
        let completed7 = seq7.extend(tokens7).unwrap();
        assert_eq!(completed7, None); // Block is full but not committed yet
        assert_eq!(seq7.blocks.len(), 0);
        assert_eq!(seq7.current_block.tokens.as_ref(), &[1, 2, 3, 4]); // Current block is full
        assert_eq!(seq7.current_block.remaining(), 0);
        assert_eq!(seq7.total_tokens(), 4);
        assert_eq!(seq7.current_block.parent_sequence_hash, None); // Still the root block
    }

    #[test]
    fn test_truncate() {
        let block_size = 4;
        let salt_hash = Some(TEST_SALT_HASH);
        let initial_tokens = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // 10 tokens

        // Case 1: Truncate within current block (len 9)
        let mut seq1 = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq1.truncate(9).is_ok());
        assert_eq!(seq1.total_tokens(), 9);
        assert_eq!(seq1.blocks().len(), 2);
        assert_eq!(seq1.current_block().tokens.as_ref(), &[9]);
        assert_eq!(
            seq1.current_block().parent_sequence_hash,
            Some(SEQ_HASH_5_8)
        );

        // Case 2: Truncate to exact block boundary (len 8)
        let mut seq2 = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq2.truncate(8).is_ok());
        assert_eq!(seq2.total_tokens(), 8);
        assert_eq!(seq2.blocks().len(), 2);
        assert!(seq2.current_block().tokens.is_empty());
        assert_eq!(
            seq2.current_block().parent_sequence_hash,
            Some(SEQ_HASH_5_8)
        );

        // Case 3: Truncate into last full block (len 7)
        let mut seq3 = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq3.truncate(7).is_ok());
        assert_eq!(seq3.total_tokens(), 7);
        assert_eq!(seq3.blocks().len(), 1); // Block [5,6,7,8] removed conceptually
        assert_eq!(seq3.current_block().tokens.as_ref(), &[5, 6, 7]); // Kept 3 from [5,6,7,8]
        assert_eq!(
            seq3.current_block().parent_sequence_hash,
            Some(SEQ_HASH_1_4)
        ); // Parent is hash of [1,2,3,4]
        assert_eq!(seq3.blocks()[0].tokens().as_ref(), &[1, 2, 3, 4]);

        // Case 4: Truncate removing full block(s) exactly (len 4)
        let mut seq4 = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq4.truncate(4).is_ok());
        assert_eq!(seq4.total_tokens(), 4);
        assert_eq!(seq4.blocks().len(), 1); // Block [5,6,7,8] removed
        assert!(seq4.current_block().tokens.is_empty()); // New partial based on block [1,2,3,4]
        assert_eq!(
            seq4.current_block().parent_sequence_hash,
            Some(SEQ_HASH_1_4)
        );
        assert_eq!(seq4.blocks()[0].tokens().as_ref(), &[1, 2, 3, 4]);

        // Case 5: Truncate into first block (len 3)
        let mut seq5 = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq5.truncate(3).is_ok());
        assert_eq!(seq5.total_tokens(), 3);
        assert!(seq5.blocks().is_empty()); // Both blocks removed conceptually
        assert_eq!(seq5.current_block().tokens.as_ref(), &[1, 2, 3]); // Kept 3 from [1,2,3,4]
        assert_eq!(seq5.current_block().parent_sequence_hash, None); // No parent

        // Case 6: Truncate to zero length (len 0)
        let mut seq6 = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq6.truncate(0).is_ok());
        assert_eq!(seq6.total_tokens(), 0);
        assert!(seq6.blocks().is_empty());
        assert!(seq6.current_block().tokens.is_empty());
        assert_eq!(seq6.current_block().parent_sequence_hash, None);

        // Case 7: Truncate to length greater than current (len 11)
        let mut seq7 = create_test_sequence(initial_tokens, block_size, salt_hash);
        let original_state = (seq7.blocks.clone(), seq7.current_block.tokens.clone()); // Clone for state check
        assert!(seq7.truncate(11).is_ok()); // Should have no effect
        assert_eq!(seq7.total_tokens(), 10);
        assert_eq!(seq7.blocks, original_state.0);
        assert_eq!(seq7.current_block.tokens, original_state.1);

        // Case 8: Truncate to current length (len 10)
        let mut seq8 = create_test_sequence(initial_tokens, block_size, salt_hash);
        let original_state = (seq8.blocks.clone(), seq8.current_block.tokens.clone());
        assert!(seq8.truncate(10).is_ok());
        assert_eq!(seq8.total_tokens(), 10);
        assert_eq!(seq8.blocks, original_state.0);
        assert_eq!(seq8.current_block.tokens, original_state.1);

        // Case 9: Truncate an empty sequence to 0
        let mut seq9 = create_test_sequence(&[], block_size, salt_hash);
        assert!(seq9.truncate(0).is_ok());
        assert_eq!(seq9.total_tokens(), 0);
        assert!(seq9.blocks().is_empty());
        assert!(seq9.current_block().tokens.is_empty());

        // Case 10: Truncate on exact block boundary when current is empty (len 4)
        let tokens10 = &[1, 2, 3, 4, 5, 6, 7, 8]; // 8 tokens
        let mut seq10 = create_test_sequence(tokens10, block_size, salt_hash);
        assert_eq!(seq10.total_tokens(), 8);
        assert!(seq10.current_block().is_empty());
        assert!(seq10.truncate(4).is_ok()); // Remove block [5, 6, 7, 8]
        assert_eq!(seq10.total_tokens(), 4);
        assert_eq!(seq10.blocks().len(), 1);
        assert!(seq10.current_block().tokens.is_empty());
        assert_eq!(
            seq10.current_block().parent_sequence_hash,
            Some(SEQ_HASH_1_4)
        );

        // Case 11: Truncate into first block when current is empty (len 3)
        let tokens11 = &[1, 2, 3, 4, 5, 6, 7, 8]; // 8 tokens
        let mut seq11 = create_test_sequence(tokens11, block_size, salt_hash);
        assert!(seq11.truncate(3).is_ok()); // Pop block [5,6,7,8] + 1 from [1,2,3,4]
        assert_eq!(seq11.total_tokens(), 3);
        assert!(seq11.blocks().is_empty());
        assert_eq!(seq11.current_block().tokens.as_ref(), &[1, 2, 3]); // Kept 3 from [1,2,3,4]
        assert_eq!(seq11.current_block().parent_sequence_hash, None);
    }

    #[test]
    fn test_unwind() {
        let block_size = 4;
        let salt_hash = Some(TEST_SALT_HASH);
        let initial_tokens = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // 10 tokens

        // Unwind 0
        let mut seq = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq.unwind(0).is_ok());
        assert_eq!(seq.total_tokens(), 10);

        // Unwind 1
        let mut seq = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq.unwind(1).is_ok());
        assert_eq!(seq.total_tokens(), 9);
        assert_eq!(seq.current_block.tokens.as_ref(), &[9]);

        // Unwind 3 (crosses boundary)
        let mut seq = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq.unwind(3).is_ok());
        assert_eq!(seq.total_tokens(), 7);
        assert_eq!(seq.blocks.len(), 1);
        assert_eq!(seq.current_block.tokens.as_ref(), &[5, 6, 7]);

        // Unwind all (10)
        let mut seq = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq.unwind(10).is_ok());
        assert_eq!(seq.total_tokens(), 0);
        assert!(seq.blocks.is_empty());
        assert!(seq.current_block.is_empty());

        // Unwind more than available (11)
        let mut seq = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert_eq!(seq.unwind(11), Err(TokenBlockError::InsufficientTokens));
        assert_eq!(seq.total_tokens(), 10); // State unchanged

        // Unwind from empty
        let mut seq_empty = create_test_sequence(&[], block_size, salt_hash);
        assert_eq!(
            seq_empty.unwind(1),
            Err(TokenBlockError::InsufficientTokens)
        );
    }

    #[test]
    fn test_pop() {
        let block_size = 4;
        let salt_hash = Some(TEST_SALT_HASH);
        let initial_tokens = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // 10 tokens

        let mut seq = create_test_sequence(initial_tokens, block_size, salt_hash);

        // Pop 10
        assert_eq!(seq.pop(), Some(10));
        assert_eq!(seq.total_tokens(), 9);
        assert_eq!(seq.current_block.tokens.as_ref(), &[9]);
        assert_eq!(seq.blocks.len(), 2);

        // Pop 9
        assert_eq!(seq.pop(), Some(9));
        assert_eq!(seq.total_tokens(), 8);
        assert!(seq.current_block.is_empty());
        assert_eq!(seq.blocks.len(), 2);
        assert_eq!(seq.current_block.parent_sequence_hash, Some(SEQ_HASH_5_8));

        // Pop 8 (crosses boundary)
        assert_eq!(seq.pop(), Some(8));
        assert_eq!(seq.total_tokens(), 7);
        assert_eq!(seq.current_block.tokens.as_ref(), &[5, 6, 7]);
        assert_eq!(seq.blocks.len(), 1);
        assert_eq!(seq.current_block.parent_sequence_hash, Some(SEQ_HASH_1_4));

        // Pop remaining partial (7, 6, 5)
        assert_eq!(seq.pop(), Some(7));
        assert_eq!(seq.pop(), Some(6));
        assert_eq!(seq.pop(), Some(5));
        assert_eq!(seq.total_tokens(), 4);
        assert!(seq.current_block.is_empty());
        assert_eq!(seq.blocks.len(), 1);
        assert_eq!(seq.current_block.parent_sequence_hash, Some(SEQ_HASH_1_4));

        // Pop 4 (crosses boundary)
        assert_eq!(seq.pop(), Some(4));
        assert_eq!(seq.total_tokens(), 3);
        assert_eq!(seq.current_block.tokens.as_ref(), &[1, 2, 3]);
        assert!(seq.blocks.is_empty());
        assert_eq!(seq.current_block.parent_sequence_hash, None);

        // Pop 3, 2, 1
        assert_eq!(seq.pop(), Some(3));
        assert_eq!(seq.pop(), Some(2));
        assert_eq!(seq.pop(), Some(1));
        assert_eq!(seq.total_tokens(), 0);
        assert!(seq.current_block.is_empty());
        assert!(seq.blocks.is_empty());

        // Pop from empty
        assert_eq!(seq.pop(), None);
        assert_eq!(seq.total_tokens(), 0);
    }

    #[test]
    fn test_total_tokens() {
        let block_size = 3;
        let salt_hash = Some(TEST_SALT_HASH);

        let mut seq = create_test_sequence(&[], block_size, salt_hash);
        assert_eq!(seq.total_tokens(), 0);

        seq.extend(Tokens::from(vec![1, 2])).unwrap();
        assert_eq!(seq.total_tokens(), 2);

        seq.append(3).unwrap(); // Completes block 0
        assert_eq!(seq.total_tokens(), 3);

        seq.extend(Tokens::from(vec![4, 5, 6, 7])).unwrap(); // Completes block 1, partial [7]
        assert_eq!(seq.total_tokens(), 7);

        seq.pop().unwrap(); // Removes 7
        assert_eq!(seq.total_tokens(), 6);

        seq.truncate(4).unwrap(); // Keep [1,2,3,4]
        assert_eq!(seq.total_tokens(), 4);

        seq.unwind(2).unwrap(); // Keep [1,2]
        assert_eq!(seq.total_tokens(), 2);
    }

    #[test]
    fn test_push_tokens_partial_block() {
        let mut partial = PartialTokenBlock::create_sequence_root(4, 1337);

        let tokens = Tokens(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let remaining = partial.push_tokens(tokens);
        assert_eq!(partial.tokens.len(), 4);
        assert_eq!(remaining.len(), 6);
    }
}
