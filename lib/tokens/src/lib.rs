// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(missing_docs)]

//! Types and utilities for handling sequences of tokens, including block creation and hashing.

use bytemuck::cast_slice;
use derive_getters::Dissolve;
use std::ops::Range;

pub mod blocks;
mod radix;
pub use radix::PositionalRadixTree;

/// Trait for hashes that include position information.
pub trait PositionalHash {
    /// Returns the position associated with the hash.
    fn position(&self) -> u64;
}

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

/// A 128-bit positional sequence hash combining traditional sequence hash with positional information.
///
/// Layout:
/// - Lower 64 bits: Traditional SequenceHash
/// - Upper 64 bits: 2-bit mode + position + LocalBlockHash (BlockHash)
///
/// Modes (automatically selected based on position):
/// - Mode 00: 8-bit position (max 255) + 54-bit LBH
/// - Mode 01: 16-bit position (max 65,535) + 46-bit LBH
/// - Mode 10: 24-bit position (max 16,777,215) + 38-bit LBH
/// - Mode 11: 31-bit position (max 2,147,483,647) + 31-bit LBH
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
pub struct PositionalSequenceHash(u128);

impl PositionalSequenceHash {
    /// Creates a new PositionalSequenceHash from components.
    ///
    /// The mode is automatically selected based on the position value to use the minimal
    /// representation that can fit the position.
    pub fn new(sequence_hash: SequenceHash, position: u64, local_block_hash: BlockHash) -> Self {
        let mode = Self::select_mode(position);
        let upper = Self::encode_upper(mode, position, local_block_hash);
        let value = ((upper as u128) << 64) | (sequence_hash as u128);
        PositionalSequenceHash(value)
    }

    /// Returns the sequence hash component (lower 64 bits).
    pub fn sequence_hash(&self) -> SequenceHash {
        (self.0 & 0xFFFF_FFFF_FFFF_FFFF) as u64
    }

    /// Returns the block position.
    pub fn position(&self) -> u64 {
        let (_, position, _) = self.decode_upper();
        position
    }

    /// Returns the local block hash (BlockHash) component.
    pub fn local_block_hash(&self) -> BlockHash {
        let (_, _, lbh) = self.decode_upper();
        lbh
    }

    /// Returns the mode used for encoding (0, 1, 2, or 3).
    pub fn mode(&self) -> u8 {
        let (mode, _, _) = self.decode_upper();
        mode
    }

    /// Returns the inner 128-bit value.
    #[inline(always)]
    pub fn as_u128(&self) -> u128 {
        self.0
    }

    /// Selects the minimal mode that can represent the given position.
    fn select_mode(position: u64) -> u8 {
        if position < (1u64 << 8) {
            0 // Mode 00: 8-bit position
        } else if position < (1u64 << 16) {
            1 // Mode 01: 16-bit position
        } else if position < (1u64 << 24) {
            2 // Mode 10: 24-bit position
        } else if position < (1u64 << 31) {
            3 // Mode 11: 31-bit position
        } else {
            panic!(
                "Position {} exceeds maximum supported value (2^31 - 1)",
                position
            );
        }
    }

    /// Encodes the upper 64 bits from mode, position, and local block hash.
    fn encode_upper(mode: u8, position: u64, local_block_hash: u64) -> u64 {
        let (position_bits, lbh_bits) = match mode {
            0 => (8, 54),  // 2 + 8 + 54 = 64
            1 => (16, 46), // 2 + 16 + 46 = 64
            2 => (24, 38), // 2 + 24 + 38 = 64
            3 => (31, 31), // 2 + 31 + 31 = 64
            _ => unreachable!(
                "Invalid mode {} when encoding PositionalSequenceHash; mode must be 0, 1, 2, or 3",
                mode
            ),
        };

        // Create masks for extracting the relevant bits
        let position_mask = (1u64 << position_bits) - 1;
        let lbh_mask = (1u64 << lbh_bits) - 1;

        // Extract and position components
        let position_part = position & position_mask;
        let lbh_part = local_block_hash & lbh_mask;

        // Combine: [mode (2 bits)][position (X bits)][lbh (R bits)]
        ((mode as u64) << 62) | (position_part << lbh_bits) | lbh_part
    }

    /// Decodes the upper 64 bits into (mode, position, local_block_hash).
    fn decode_upper(&self) -> (u8, u64, u64) {
        let upper = (self.0 >> 64) as u64;

        // Extract mode from top 2 bits
        let mode = (upper >> 62) as u8;

        let (position_bits, lbh_bits) = match mode {
            0 => (8, 54),
            1 => (16, 46),
            2 => (24, 38),
            3 => (31, 31),
            _ => unreachable!(
                "Invalid mode {} in PositionalSequenceHash - value may be corrupted",
                mode
            ),
        };

        // Create masks
        let lbh_mask = (1u64 << lbh_bits) - 1;
        let position_mask = (1u64 << position_bits) - 1;

        // Extract components
        let lbh = upper & lbh_mask;
        let position = (upper >> lbh_bits) & position_mask;

        (mode, position, lbh)
    }
}

impl std::fmt::Debug for PositionalSequenceHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PositionalSequenceHash")
            .field("sequence_hash", &self.sequence_hash())
            .field("local_block_hash", &self.local_block_hash())
            .field("position", &self.position())
            .finish()
    }
}

/// A 128-bit positional lineage hash encoding parental lineage for tree traversal.
///
/// Layout (using full 128 bits):
/// - Mode (2 bits): Determines position field size
/// - Position (8/16/24 bits): Block position in sequence
/// - Parent Fragment (variable bits): Fragment of parent's sequence hash
/// - Current Fragment (variable bits): Fragment of current sequence hash
///
/// Modes (automatically selected based on position):
/// - Mode 00: 8-bit position (max 255) + 59-bit parent + 59-bit current
/// - Mode 01: 16-bit position (max 65,535) + 55-bit parent + 55-bit current
/// - Mode 10: 24-bit position (max 16,777,215) + 51-bit parent + 51-bit current
///
/// This encoding enables backward traversal through the radix tree by matching
/// parent fragments at position-1.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
pub struct PositionalLineageHash(u128);

impl PositionalLineageHash {
    /// Creates a new PositionalLineageHash from components.
    ///
    /// The mode is automatically selected based on the position value to use the minimal
    /// representation that can fit the position.
    ///
    /// # Arguments
    ///
    /// * `current_seq_hash` - The sequence hash of the current block
    /// * `parent_seq_hash` - The sequence hash of the parent block (None for root)
    /// * `position` - The block position in the sequence
    ///
    /// # Panics
    ///
    /// Panics if position >= 2^24 (16,777,216).
    pub fn new(
        current_seq_hash: SequenceHash,
        parent_seq_hash: Option<SequenceHash>,
        position: u64,
    ) -> Self {
        if position >= (1u64 << 24) {
            panic!(
                "Position {} exceeds maximum supported value (2^24 - 1 = 16,777,215)",
                position
            );
        }

        let mode = Self::select_mode(position);
        let (position_bits, parent_bits, current_bits) = Self::bit_layout(mode);

        // CRITICAL: For cross-mode boundary matching, we need to align the current hash
        // to the bits available at the next position. This ensures that when position+1
        // stores our current hash as its parent, the fragments will match.
        let next_mode = Self::select_mode(position + 1);
        let (_, next_parent_bits, _) = Self::bit_layout(next_mode);

        // Use the minimum of current_bits and next_parent_bits to ensure alignment
        let aligned_current_bits = current_bits.min(next_parent_bits);

        // Create masks
        let position_mask = (1u128 << position_bits) - 1;
        let parent_mask = (1u128 << parent_bits) - 1;
        let current_mask = (1u128 << aligned_current_bits) - 1;

        // Extract fragments (LSB-aligned for subset compatibility across modes)
        let position_part = (position as u128) & position_mask;
        let parent_part = (parent_seq_hash.unwrap_or(0) as u128) & parent_mask;
        let current_part = (current_seq_hash as u128) & current_mask;

        // Pack: [mode (2)][position (P)][parent (M)][current (N)]
        // Note: We still allocate current_bits in the layout, but only use aligned_current_bits
        let value = ((mode as u128) << 126)
            | (position_part << (parent_bits + current_bits))
            | (parent_part << current_bits)
            | current_part;

        PositionalLineageHash(value)
    }

    /// Returns the block position.
    pub fn position(&self) -> u64 {
        let mode = self.mode();
        let (position_bits, parent_bits, current_bits) = Self::bit_layout(mode);
        let position_mask = (1u128 << position_bits) - 1;
        ((self.0 >> (parent_bits + current_bits)) & position_mask) as u64
    }

    /// Returns the current sequence hash fragment.
    pub fn current_hash_fragment(&self) -> u64 {
        let mode = self.mode();
        let (_, _, current_bits) = Self::bit_layout(mode);
        let current_mask = (1u128 << current_bits) - 1;
        (self.0 & current_mask) as u64
    }

    /// Returns the parent sequence hash fragment.
    pub fn parent_hash_fragment(&self) -> u64 {
        let mode = self.mode();
        let (_, parent_bits, current_bits) = Self::bit_layout(mode);
        let parent_mask = (1u128 << parent_bits) - 1;
        ((self.0 >> current_bits) & parent_mask) as u64
    }

    /// Returns the mode used for encoding (0, 1, or 2).
    pub fn mode(&self) -> u8 {
        (self.0 >> 126) as u8
    }

    /// Returns the inner 128-bit value.
    #[inline(always)]
    pub fn as_u128(&self) -> u128 {
        self.0
    }

    /// Selects the minimal mode that can represent the given position.
    fn select_mode(position: u64) -> u8 {
        if position < (1u64 << 8) {
            0 // Mode 00: 8-bit position
        } else if position < (1u64 << 16) {
            1 // Mode 01: 16-bit position
        } else {
            2 // Mode 10: 24-bit position
        }
    }

    /// Returns the bit layout for a given mode: (position_bits, parent_bits, current_bits).
    fn bit_layout(mode: u8) -> (u32, u32, u32) {
        match mode {
            0 => (8, 59, 59),  // 2 + 8 + 59 + 59 = 128
            1 => (16, 55, 55), // 2 + 16 + 55 + 55 = 128
            2 => (24, 51, 51), // 2 + 24 + 51 + 51 = 128
            _ => unreachable!(
                "Invalid mode {} in PositionalLineageHash; mode must be 0, 1, or 2",
                mode
            ),
        }
    }
}

impl PositionalLineageHash {
    fn format_impl(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let position = self.position();
        let current_hash = self.current_hash_fragment();
        let current_hash_b58 = bs58::encode(current_hash.to_be_bytes()).into_string();

        if position == 0 {
            write!(f, "{}:{}", position, current_hash_b58)
        } else {
            let parent_hash = self.parent_hash_fragment();
            let parent_hash_b58 = bs58::encode(parent_hash.to_be_bytes()).into_string();
            write!(f, "{}:{}:{}", position, current_hash_b58, parent_hash_b58)
        }
    }
}

impl std::fmt::Debug for PositionalLineageHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format_impl(f)
    }
}

impl std::fmt::Display for PositionalLineageHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format_impl(f)
    }
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

impl From<Vec<usize>> for Tokens {
    fn from(tokens: Vec<usize>) -> Self {
        Tokens(
            tokens
                .into_iter()
                .map(|t| t.try_into().expect("Token ID exceeds u32::MAX"))
                .collect(),
        )
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
    pub fn into_sequence(self, block_size: u32, salt_hash: Option<SaltHash>) -> TokenBlockSequence {
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
#[derive(Debug, PartialEq)] // No Clone: intended to be unique within a sequence
pub struct PartialTokenBlock {
    tokens: Tokens,
    block_size: u32,
    salt_hash: SaltHash,
    parent_sequence_hash: Option<SequenceHash>,
    position: usize, // The position this block will have when committed
}

impl PartialTokenBlock {
    /// Creates the first partial block (root) for a new sequence.
    ///
    /// # Arguments
    ///
    /// * `block_size` - The fixed size for blocks in this sequence.
    /// * `salt_hash` - The [`SaltHash`] for the sequence.
    pub(crate) fn create_sequence_root(block_size: u32, salt_hash: SaltHash) -> Self {
        Self {
            tokens: Tokens::default(),
            block_size,
            salt_hash,
            parent_sequence_hash: None, // Root has no parent
            position: 0,                // First block is at position 0
        }
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
    pub fn commit(&mut self) -> Result<TokenBlock, TokenBlockError> {
        if self.tokens.0.len() != self.block_size as usize {
            // Check for exact size match for committing
            return Err(TokenBlockError::Incomplete);
        }

        // Take ownership of the tokens, leaving the internal tokens empty
        let tokens = std::mem::take(&mut self.tokens);

        let chunk = TokenBlockChunk::new(tokens, self.salt_hash);
        let block = TokenBlock::from_chunk(chunk, self.parent_sequence_hash, self.position);

        // Reset self to be the next block in the sequence
        self.parent_sequence_hash = Some(block.sequence_hash());
        self.position += 1; // Increment position for the next block
        // self.tokens is already empty due to mem::take
        // self.block_size and self.salt_hash remain the same

        Ok(block)
    }

    /// Returns the number of additional tokens required to fill the block.
    pub fn remaining(&self) -> usize {
        // Use saturating_sub to prevent underflow if len somehow exceeds block_size
        (self.block_size as usize).saturating_sub(self.tokens.0.len())
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
/// [`SequenceHash`], [`PositionalSequenceHash`], [`PositionalLineageHash`], and optionally the parent's [`SequenceHash`].
#[derive(Debug, Clone, Default, PartialEq)] // Add PartialEq for tests
pub struct TokenBlock {
    tokens: Tokens,
    salt_hash: SaltHash,
    block_hash: BlockHash,
    sequence_hash: SequenceHash,
    parent_sequence_hash: Option<SequenceHash>,
    positional_sequence_hash: PositionalSequenceHash,
    positional_lineage_hash: PositionalLineageHash,
}

impl TokenBlock {
    /// Creates a new [`PartialTokenBlock`] representing the block immediately following this one.
    ///
    /// The new partial block will have the correct `parent_sequence_hash` and `position` set.
    pub fn next_block(&self) -> PartialTokenBlock {
        PartialTokenBlock {
            tokens: Tokens::default(),
            block_size: self.tokens.len() as u32, // Should be == self.block_size
            salt_hash: self.salt_hash,
            parent_sequence_hash: Some(self.sequence_hash), // Link to this block
            position: self.position() as usize + 1,         // Next position
        }
    }

    /// Finalizes a [`TokenBlock`] from a [`TokenBlockChunk`], parent's sequence hash, and position.
    ///
    /// This computes the final [`SequenceHash`], [`PositionalSequenceHash`], and [`PositionalLineageHash`] for the block.
    fn from_chunk(
        chunk: TokenBlockChunk,
        parent_sequence_hash: Option<SequenceHash>,
        position: usize,
    ) -> Self {
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

        let positional_sequence_hash = PositionalSequenceHash::new(
            sequence_hash,
            position as u64,
            chunk.block_hash, // LocalBlockHash is the same as BlockHash
        );

        let positional_lineage_hash =
            PositionalLineageHash::new(sequence_hash, parent_sequence_hash, position as u64);

        Self {
            tokens: chunk.tokens,
            salt_hash: chunk.salt_hash,
            block_hash: chunk.block_hash,
            sequence_hash,
            parent_sequence_hash,
            positional_sequence_hash,
            positional_lineage_hash,
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

    /// Returns the number of tokens in the block.
    pub fn block_size(&self) -> usize {
        self.tokens.0.len()
    }

    /// Returns the positional sequence hash for this block.
    pub fn positional_sequence_hash(&self) -> PositionalSequenceHash {
        self.positional_sequence_hash
    }

    /// Returns the positional lineage hash for this block.
    pub fn positional_lineage_hash(&self) -> PositionalLineageHash {
        self.positional_lineage_hash
    }

    /// Returns the position of this block in the sequence.
    pub fn position(&self) -> u64 {
        self.positional_sequence_hash.position()
    }
}

impl PositionalHash for PositionalSequenceHash {
    fn position(&self) -> u64 {
        self.position()
    }
}

impl PositionalHash for PositionalLineageHash {
    fn position(&self) -> u64 {
        self.position()
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
#[derive(Debug, PartialEq)]
pub struct TokenBlockSequence {
    blocks: Vec<TokenBlock>,
    current_block: PartialTokenBlock,
    salt_hash: SaltHash,
    block_size: usize,
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
    pub fn new(tokens: Tokens, block_size: u32, salt_hash: Option<SaltHash>) -> Self {
        assert!(block_size > 0, "block_size must be greater than 0");
        let salt_hash = salt_hash.unwrap_or(0);
        let (blocks, current_block) = Self::split_tokens(&tokens, block_size, salt_hash);

        Self {
            blocks,
            current_block,
            salt_hash,
            block_size: block_size as usize,
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
            if self.current_block.remaining() == 0 {
                // If it became full AND there are still more tokens to append,
                // commit it now so the next loop iteration starts with a fresh block.
                let new_block = self.current_block.commit()?;
                self.blocks.push(new_block);
            }
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
                assert_eq!(
                    range.len(),
                    1,
                    "Appending a single token completed more than one block, which should be impossible."
                );
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
                let num_blocks_to_affect = tokens_to_pop_from_blocks.div_ceil(block_size as usize);

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
                let num_tokens_to_pop_from_source_block = tokens_to_pop_from_blocks
                    - num_full_blocks_completely_popped * block_size as usize;
                let num_tokens_to_keep_in_new_partial =
                    (block_size as usize).saturating_sub(num_tokens_to_pop_from_source_block);

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
                // Update position to match the number of complete blocks
                self.current_block.position = self.blocks.len();
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

    /// Resets the sequence to the initial state.
    pub fn reset(&mut self) {
        self.blocks.clear();
        self.current_block =
            PartialTokenBlock::create_sequence_root(self.block_size as u32, self.salt_hash);
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

    /// Returns the block size used for this sequence.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the [`SaltHash`] used for this sequence.
    pub fn salt_hash(&self) -> SaltHash {
        self.salt_hash
    }

    /// Returns the total number of tokens in the sequence (sum of tokens in all completed blocks
    /// plus tokens in the current partial block).
    pub fn total_tokens(&self) -> usize {
        let block_size = self.current_block.block_size as usize;
        (self.blocks.len() * block_size) + self.current_block.len()
    }

    /// Extract the token with the range
    pub fn tokens_at(&self, range: Range<usize>) -> Tokens {
        let total = self.total_tokens();

        // Validate range - return empty tokens for invalid ranges
        if range.start > range.end || range.end > total {
            return Tokens::default();
        }

        // Handle empty range
        if range.is_empty() {
            return Tokens::default();
        }

        let mut result = Vec::with_capacity(range.len());

        for i in range {
            if i < self.blocks.len() * self.block_size {
                // Token is in a completed block
                let block_index = i / self.block_size;
                let token_index = i % self.block_size;
                result.push(self.blocks[block_index].tokens()[token_index]);
            } else {
                // Token is in the current partial block
                let current_block_index = i - (self.blocks.len() * self.block_size);
                result.push(self.current_block.tokens()[current_block_index]);
            }
        }

        Tokens::from(result)
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
        block_size: u32,
        salt_hash: u64,
    ) -> (Vec<TokenBlock>, PartialTokenBlock) {
        assert!(block_size > 0, "block_size must be greater than 0");
        let chunks: Vec<TokenBlockChunk> = tokens
            .as_ref()
            .chunks_exact(block_size as usize)
            .map(|chunk| TokenBlockChunk::from_tokens(chunk, salt_hash))
            .collect();

        let mut result_blocks = Vec::with_capacity(chunks.len());
        let mut last_sequence_hash: Option<SequenceHash> = None;

        // Sequentially combine chunks to compute sequence hashes
        for (position, chunk) in chunks.into_iter().enumerate() {
            let new_block = TokenBlock::from_chunk(chunk, last_sequence_hash, position);
            last_sequence_hash = Some(new_block.sequence_hash());
            result_blocks.push(new_block);
        }

        // Handle any remaining tokens
        let remainder = tokens
            .as_ref()
            .chunks_exact(block_size as usize)
            .remainder();

        let next_position = result_blocks.len(); // Position for the next block to be committed

        let current_block = PartialTokenBlock {
            tokens: remainder.into(),
            block_size,
            salt_hash,
            // Parent hash is the sequence hash of the last *full* block computed
            parent_sequence_hash: last_sequence_hash,
            position: next_position,
        };

        (result_blocks, current_block)
    }

    /// Creates a new [`TokenBlockSequence`] from a slice of tokens.
    ///
    /// The tokens are split into blocks of `block_size`. Any remaining tokens
    /// form the initial `current_block`.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The slice of tokens to create the sequence from.
    /// * `block_size` - The size of each block.
    /// * `salt_hash` - The [`SaltHash`] to use for hashing.
    pub fn from_slice(tokens: &[Token], block_size: u32, salt_hash: Option<SaltHash>) -> Self {
        assert!(block_size > 0, "block_size must be greater than 0");
        let salt_hash = salt_hash.unwrap_or(0);
        let (blocks, current_block) = Self::split_tokens(tokens, block_size, salt_hash);

        Self {
            blocks,
            current_block,
            salt_hash,
            block_size: block_size as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::cast_slice;

    // Helper to create a sequence for testing
    fn create_test_sequence(
        initial_tokens: &[Token],
        block_size: u32,
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

    impl PartialTokenBlock {
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
        pub fn push_token(&mut self, token: Token) -> Result<(), TokenBlockError> {
            if self.tokens.0.len() >= self.block_size as usize {
                return Err(TokenBlockError::Full);
            }
            self.tokens.0.push(token);
            Ok(())
        }

        /// Attempts to remove the last token from the block.
        ///
        /// # Returns
        ///
        /// * `Ok(())` - If a token was successfully removed.
        /// * `Err(TokenBlockError::Empty)` - If the block was already empty.
        pub fn pop_token(&mut self) -> Result<(), TokenBlockError> {
            if self.tokens.0.is_empty() {
                return Err(TokenBlockError::Empty);
            }
            self.tokens.0.pop();
            Ok(())
        }
    }

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
    fn test_positional_sequence_hash_encoding_decoding() {
        // Test Mode 0: position fits in 8 bits (< 256)
        let seq_hash_0 = 0x1234567890ABCDEF;
        let position_0 = 100;
        let lbh_0 = 0xFEDCBA9876543210;
        let psh_0 = PositionalSequenceHash::new(seq_hash_0, position_0, lbh_0);

        assert_eq!(psh_0.mode(), 0, "Position 100 should use mode 0");
        assert_eq!(psh_0.sequence_hash(), seq_hash_0);
        assert_eq!(psh_0.position(), position_0);
        // LBH is truncated to 54 bits in mode 0
        assert_eq!(
            psh_0.local_block_hash(),
            lbh_0 & ((1u64 << 54) - 1),
            "LBH should be truncated to 54 bits"
        );

        // Test Mode 1: position fits in 16 bits (256 <= pos < 65536)
        let position_1 = 1000;
        let psh_1 = PositionalSequenceHash::new(seq_hash_0, position_1, lbh_0);

        assert_eq!(psh_1.mode(), 1, "Position 1000 should use mode 1");
        assert_eq!(psh_1.sequence_hash(), seq_hash_0);
        assert_eq!(psh_1.position(), position_1);
        // LBH is truncated to 46 bits in mode 1
        assert_eq!(
            psh_1.local_block_hash(),
            lbh_0 & ((1u64 << 46) - 1),
            "LBH should be truncated to 46 bits"
        );

        // Test Mode 2: position fits in 24 bits (65536 <= pos < 16777216)
        let position_2 = 100_000;
        let psh_2 = PositionalSequenceHash::new(seq_hash_0, position_2, lbh_0);

        assert_eq!(psh_2.mode(), 2, "Position 100,000 should use mode 2");
        assert_eq!(psh_2.sequence_hash(), seq_hash_0);
        assert_eq!(psh_2.position(), position_2);
        // LBH is truncated to 38 bits in mode 2
        assert_eq!(
            psh_2.local_block_hash(),
            lbh_0 & ((1u64 << 38) - 1),
            "LBH should be truncated to 38 bits"
        );

        // Test Mode 3: position fits in 31 bits (16777216 <= pos < 2^31)
        let position_3 = 20_000_000;
        let psh_3 = PositionalSequenceHash::new(seq_hash_0, position_3, lbh_0);

        assert_eq!(psh_3.mode(), 3, "Position 20,000,000 should use mode 3");
        assert_eq!(psh_3.sequence_hash(), seq_hash_0);
        assert_eq!(psh_3.position(), position_3);
        // LBH is truncated to 31 bits in mode 3
        assert_eq!(
            psh_3.local_block_hash(),
            lbh_0 & ((1u64 << 31) - 1),
            "LBH should be truncated to 31 bits"
        );

        // Test edge case: position at boundary
        let position_255 = 255;
        let psh_255 = PositionalSequenceHash::new(seq_hash_0, position_255, lbh_0);
        assert_eq!(psh_255.mode(), 0, "Position 255 should use mode 0");
        assert_eq!(psh_255.position(), position_255);

        let position_256 = 256;
        let psh_256 = PositionalSequenceHash::new(seq_hash_0, position_256, lbh_0);
        assert_eq!(psh_256.mode(), 1, "Position 256 should use mode 1");
        assert_eq!(psh_256.position(), position_256);
    }

    #[test]
    fn test_positional_lineage_hash() {
        // Test Mode 0: position fits in 8 bits (< 256)
        let current_hash_0 = 0x1234567890ABCDEF;
        let parent_hash_0 = 0xFEDCBA9876543210;
        let position_0 = 100;
        let plh_0 = PositionalLineageHash::new(current_hash_0, Some(parent_hash_0), position_0);

        assert_eq!(plh_0.mode(), 0, "Position 100 should use mode 0");
        assert_eq!(plh_0.position(), position_0);
        // Current and parent are truncated to 59 bits in mode 0
        assert_eq!(
            plh_0.current_hash_fragment(),
            current_hash_0 & ((1u64 << 59) - 1),
            "Current hash should be truncated to 59 bits"
        );
        assert_eq!(
            plh_0.parent_hash_fragment(),
            parent_hash_0 & ((1u64 << 59) - 1),
            "Parent hash should be truncated to 59 bits"
        );

        // Test Mode 1: position fits in 16 bits (256 <= pos < 65536)
        let position_1 = 1000;
        let plh_1 = PositionalLineageHash::new(current_hash_0, Some(parent_hash_0), position_1);

        assert_eq!(plh_1.mode(), 1, "Position 1000 should use mode 1");
        assert_eq!(plh_1.position(), position_1);
        // Current and parent are truncated to 55 bits in mode 1
        assert_eq!(
            plh_1.current_hash_fragment(),
            current_hash_0 & ((1u64 << 55) - 1),
            "Current hash should be truncated to 55 bits"
        );
        assert_eq!(
            plh_1.parent_hash_fragment(),
            parent_hash_0 & ((1u64 << 55) - 1),
            "Parent hash should be truncated to 55 bits"
        );

        // Test Mode 2: position fits in 24 bits (65536 <= pos < 16777216)
        let position_2 = 100_000;
        let plh_2 = PositionalLineageHash::new(current_hash_0, Some(parent_hash_0), position_2);

        assert_eq!(plh_2.mode(), 2, "Position 100,000 should use mode 2");
        assert_eq!(plh_2.position(), position_2);
        // Current and parent are truncated to 51 bits in mode 2
        assert_eq!(
            plh_2.current_hash_fragment(),
            current_hash_0 & ((1u64 << 51) - 1),
            "Current hash should be truncated to 51 bits"
        );
        assert_eq!(
            plh_2.parent_hash_fragment(),
            parent_hash_0 & ((1u64 << 51) - 1),
            "Parent hash should be truncated to 51 bits"
        );

        // Test edge cases: position at boundaries
        let position_255 = 255;
        let plh_255 = PositionalLineageHash::new(current_hash_0, Some(parent_hash_0), position_255);
        assert_eq!(plh_255.mode(), 0, "Position 255 should use mode 0");
        assert_eq!(plh_255.position(), position_255);

        let position_256 = 256;
        let plh_256 = PositionalLineageHash::new(current_hash_0, Some(parent_hash_0), position_256);
        assert_eq!(plh_256.mode(), 1, "Position 256 should use mode 1");
        assert_eq!(plh_256.position(), position_256);

        let position_65535 = 65535;
        let plh_65535 =
            PositionalLineageHash::new(current_hash_0, Some(parent_hash_0), position_65535);
        assert_eq!(plh_65535.mode(), 1, "Position 65535 should use mode 1");
        assert_eq!(plh_65535.position(), position_65535);

        let position_65536 = 65536;
        let plh_65536 =
            PositionalLineageHash::new(current_hash_0, Some(parent_hash_0), position_65536);
        assert_eq!(plh_65536.mode(), 2, "Position 65536 should use mode 2");
        assert_eq!(plh_65536.position(), position_65536);

        // Test with None parent (root block)
        let plh_root = PositionalLineageHash::new(current_hash_0, None, 0);
        assert_eq!(plh_root.mode(), 0);
        assert_eq!(plh_root.position(), 0);
        assert_eq!(
            plh_root.parent_hash_fragment(),
            0,
            "Root should have zero parent hash"
        );
        assert_eq!(
            plh_root.current_hash_fragment(),
            current_hash_0 & ((1u64 << 59) - 1)
        );

        // Test LSB alignment: verify that smaller mode fragments are subsets of larger mode fragments
        let position_small = 100; // Mode 0: 59 bits
        let position_large = 1000; // Mode 1: 55 bits
        let plh_small =
            PositionalLineageHash::new(current_hash_0, Some(parent_hash_0), position_small);
        let plh_large =
            PositionalLineageHash::new(current_hash_0, Some(parent_hash_0), position_large);

        // The 55-bit fragment from mode 1 should match the lower 55 bits of the 59-bit fragment from mode 0
        let mask_55 = (1u64 << 55) - 1;
        assert_eq!(
            plh_large.current_hash_fragment(),
            plh_small.current_hash_fragment() & mask_55,
            "LSB alignment: mode 1 fragment should be subset of mode 0 fragment"
        );
    }

    #[test]
    #[should_panic(expected = "Position 16777216 exceeds maximum supported value")]
    fn test_positional_lineage_hash_panic_on_large_position() {
        let current_hash = 0x1234567890ABCDEF;
        let parent_hash = 0xFEDCBA9876543210;
        let position = 1u64 << 24; // 2^24 = 16,777,216
        let _ = PositionalLineageHash::new(current_hash, Some(parent_hash), position);
    }

    #[test]
    fn test_positional_lineage_hash_mode_boundary_alignment() {
        // Test that hash fragments align correctly across mode boundaries
        // This is critical for backward traversal in the radix tree

        let parent_hash = 0xFEDCBA9876543210;
        let current_hash_255 = 0x1234567890ABCDEF;
        let current_hash_256 = 0xABCDEF0123456789;

        // Position 255: Mode 0 (last position before boundary)
        let plh_255 = PositionalLineageHash::new(current_hash_255, Some(parent_hash), 255);
        assert_eq!(plh_255.mode(), 0);

        // Position 256: Mode 1 (first position after boundary)
        // This should store position 255's current_hash as its parent
        let plh_256 = PositionalLineageHash::new(current_hash_256, Some(current_hash_255), 256);
        assert_eq!(plh_256.mode(), 1);

        // CRITICAL TEST: The parent fragment at position 256 should match
        // the current fragment at position 255, allowing backward traversal
        // Both should be truncated to 55 bits (the minimum available at the boundary)
        let mask_55 = (1u64 << 55) - 1;
        assert_eq!(
            plh_256.parent_hash_fragment(),
            plh_255.current_hash_fragment() & mask_55,
            "Mode boundary: position 256's parent fragment should match position 255's current fragment (55 bits)"
        );

        // Verify that position 255's current fragment is already truncated to 55 bits
        // (not the full 59 bits that Mode 0 could theoretically support)
        assert_eq!(
            plh_255.current_hash_fragment(),
            current_hash_255 & mask_55,
            "Position 255 should pre-truncate current hash to 55 bits for next mode compatibility"
        );

        // Test the other boundary: 65535 -> 65536 (Mode 1 -> Mode 2)
        let current_hash_65535 = 0x1111222233334444;
        let current_hash_65536 = 0x5555666677778888;

        let plh_65535 = PositionalLineageHash::new(current_hash_65535, Some(parent_hash), 65535);
        assert_eq!(plh_65535.mode(), 1);

        let plh_65536 =
            PositionalLineageHash::new(current_hash_65536, Some(current_hash_65535), 65536);
        assert_eq!(plh_65536.mode(), 2);

        // Both should align to 51 bits (Mode 2's capacity)
        let mask_51 = (1u64 << 51) - 1;
        assert_eq!(
            plh_65536.parent_hash_fragment(),
            plh_65535.current_hash_fragment() & mask_51,
            "Mode boundary: position 65536's parent fragment should match position 65535's current fragment (51 bits)"
        );

        assert_eq!(
            plh_65535.current_hash_fragment(),
            current_hash_65535 & mask_51,
            "Position 65535 should pre-truncate current hash to 51 bits for next mode compatibility"
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
        let block1 = TokenBlock::from_chunk(chunk1, None, 0);

        assert_eq!(block1.tokens(), &tokens1);
        assert_eq!(block1.salt_hash(), salt);
        assert_eq!(block1.parent_sequence_hash(), None);
        assert_eq!(block1.block_hash(), HASH_1_4);
        assert_eq!(block1.sequence_hash(), SEQ_HASH_1_4); // First block seq_hash == block_hash
        assert_eq!(block1.position(), 0); // First block is at position 0

        // Verify positional lineage hash for block 1
        let plh1 = block1.positional_lineage_hash();
        assert_eq!(plh1.position(), 0);
        assert_eq!(plh1.parent_hash_fragment(), 0); // Root has no parent
        assert_eq!(
            plh1.current_hash_fragment(),
            SEQ_HASH_1_4 & ((1u64 << 59) - 1)
        ); // Mode 0: 59 bits

        let tokens2 = Tokens::from(vec![5, 6, 7, 8]);
        let chunk2 = TokenBlockChunk::new(tokens2.clone(), salt);
        let block2 = TokenBlock::from_chunk(chunk2, block1.parent_sequence_hash(), 1); // Incorrect parent
        // Sequence hash should differ if parent is wrong
        assert_ne!(block2.sequence_hash(), SEQ_HASH_5_8);

        let chunk2_correct = TokenBlockChunk::new(tokens2.clone(), salt);
        let block2_correct =
            TokenBlock::from_chunk(chunk2_correct, Some(block1.sequence_hash()), 1);

        assert_eq!(block2_correct.tokens(), &tokens2);
        assert_eq!(block2_correct.salt_hash(), salt);
        assert_eq!(
            block2_correct.parent_sequence_hash(),
            Some(block1.sequence_hash())
        );
        assert_eq!(block2_correct.block_hash(), HASH_5_8);
        assert_eq!(block2_correct.sequence_hash(), SEQ_HASH_5_8);
        assert_eq!(block2_correct.position(), 1); // Second block is at position 1

        // Verify positional lineage hash for block 2
        let plh2 = block2_correct.positional_lineage_hash();
        assert_eq!(plh2.position(), 1);
        assert_eq!(
            plh2.parent_hash_fragment(),
            SEQ_HASH_1_4 & ((1u64 << 59) - 1)
        ); // Parent fragment matches block1's sequence hash
        assert_eq!(
            plh2.current_hash_fragment(),
            SEQ_HASH_5_8 & ((1u64 << 59) - 1)
        ); // Mode 0: 59 bits
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

        // Test tokens_at across blocks and partial block
        assert_eq!(seq_multi.tokens_at(0..4).as_ref(), &[1, 2, 3, 4]); // First complete block
        assert_eq!(seq_multi.tokens_at(4..8).as_ref(), &[5, 6, 7, 8]); // Second complete block
        assert_eq!(seq_multi.tokens_at(8..9).as_ref(), &[9]); // Current partial block
        assert_eq!(seq_multi.tokens_at(2..6).as_ref(), &[3, 4, 5, 6]); // Spanning blocks
        assert_eq!(seq_multi.tokens_at(6..9).as_ref(), &[7, 8, 9]); // Spanning to partial
        assert_eq!(seq_multi.tokens_at(5..5).as_ref(), &[0u32; 0]); // Empty range
        assert_eq!(seq_multi.tokens_at(10..15).as_ref(), &[0u32; 0]); // Out of bounds

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
        // This will also commit block 2
        let completed_idx = sequence.append(12).unwrap();
        assert_eq!(completed_idx, Some(2));
        assert_eq!(sequence.blocks().len(), 3);
        assert_eq!(sequence.current_block.tokens.as_ref(), &[0u32; 0]);
        assert_eq!(sequence.current_block.remaining(), 4);
        assert_eq!(
            sequence.current_block().parent_sequence_hash,
            Some(SEQ_HASH_9_12)
        ); // Still linked to block 1

        // Append token 13 - should not complete a block
        let completed_idx_13 = sequence.append(13).unwrap();
        assert_eq!(completed_idx_13, None);
        assert_eq!(sequence.blocks().len(), 3);
        assert_eq!(sequence.blocks[2].tokens().as_ref(), &[9, 10, 11, 12]);
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
        assert_eq!(seq1.current_block.parent_sequence_hash, None); // Still the root block

        // Case 2: Extend exactly block size
        let mut seq2 = create_test_sequence(&[], block_size, salt_hash);
        let tokens2 = Tokens::from(vec![1, 2, 3, 4]);
        let completed2 = seq2.extend(tokens2).unwrap();
        assert_eq!(completed2, Some(0..1));
        assert_eq!(seq2.blocks.len(), 1);
        assert_eq!(seq2.current_block.tokens.as_ref(), &[0u32; 0]); // Current block is empty
        assert_eq!(seq2.current_block.remaining(), 4);
        assert_eq!(seq2.current_block.parent_sequence_hash, Some(SEQ_HASH_1_4)); // Still the root block

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
        assert_eq!(completed4, Some(0..2)); // Only block 0 is committed
        assert_eq!(seq4.blocks.len(), 2); // Only 1 block committed
        assert_eq!(seq4.current_block.tokens.as_ref(), &[0u32; 0]);
        assert_eq!(seq4.current_block.remaining(), 4);
        assert_eq!(seq4.blocks[0].tokens().as_ref(), &[1, 2, 3, 4]);
        assert_eq!(seq4.blocks[0].sequence_hash(), SEQ_HASH_1_4);
        assert_eq!(seq4.current_block.parent_sequence_hash, Some(SEQ_HASH_5_8)); // Parent is the first block

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
        assert_eq!(completed7, Some(0..1)); // Block is full but not committed yet
        assert_eq!(seq7.blocks.len(), 1);
        assert_eq!(seq7.current_block.tokens.as_ref(), &[0u32; 0]); // Current block is full
        assert_eq!(seq7.current_block.remaining(), 4);
        assert_eq!(seq7.total_tokens(), 4);
        assert_eq!(seq7.current_block.parent_sequence_hash, Some(SEQ_HASH_1_4)); // Still the root block

        // Test tokens_at extraction
        assert_eq!(seq7.tokens_at(0..2).as_ref(), &[1, 2]);
        assert_eq!(seq7.tokens_at(1..3).as_ref(), &[2, 3]);
        assert_eq!(seq7.tokens_at(0..4).as_ref(), &[1, 2, 3, 4]);
        assert_eq!(seq7.tokens_at(2..2).as_ref(), &[0u32; 0]); // Empty range
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

    // ========== Additional tests for coverage improvement ==========

    // === PositionalRadixTree Tests ===

    #[test]
    fn test_positional_radix_tree_basic_operations() {
        use crate::PositionalRadixTree;

        // Test new() and is_empty()
        let tree: PositionalRadixTree<String> = PositionalRadixTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);

        // Test default()
        let tree2: PositionalRadixTree<i32> = PositionalRadixTree::default();
        assert!(tree2.is_empty());

        // Test prefix() and insertion
        let psh1 = PositionalSequenceHash::new(0x1234, 0, 0xABCD);
        let psh2 = PositionalSequenceHash::new(0x5678, 0, 0xEF01);
        let psh3 = PositionalSequenceHash::new(0x9ABC, 1, 0x2345);

        tree.prefix(&psh1).insert(psh1, "value1".to_string());
        assert!(!tree.is_empty());
        assert_eq!(tree.len(), 1);

        tree.prefix(&psh2).insert(psh2, "value2".to_string());
        assert_eq!(tree.len(), 2);

        tree.prefix(&psh3).insert(psh3, "value3".to_string());
        assert_eq!(tree.len(), 3);

        // Test retrieval
        assert_eq!(
            tree.prefix(&psh1).get(&psh1).map(|v| v.clone()),
            Some("value1".to_string())
        );
    }

    #[test]
    fn test_positional_radix_tree_with_lineage_hash() {
        use crate::PositionalRadixTree;

        // Test generic usage with PositionalLineageHash
        let tree: PositionalRadixTree<u32, PositionalLineageHash> = PositionalRadixTree::new();
        assert!(tree.is_empty());

        let plh1 = PositionalLineageHash::new(0x1234, None, 0);
        let plh2 = PositionalLineageHash::new(0x5678, Some(0x1234), 1);

        tree.prefix(&plh1).insert(plh1, 100);
        tree.prefix(&plh2).insert(plh2, 200);

        assert_eq!(tree.len(), 2);
        assert_eq!(tree.prefix(&plh1).get(&plh1).map(|v| *v), Some(100));
        assert_eq!(tree.prefix(&plh2).get(&plh2).map(|v| *v), Some(200));
    }

    #[test]
    fn test_positional_radix_tree_position_lookup() {
        use crate::PositionalRadixTree;

        let tree: PositionalRadixTree<String> = PositionalRadixTree::new();

        // Insert at different positions
        let psh0 = PositionalSequenceHash::new(0x1111, 0, 0xAAAA);
        let psh1 = PositionalSequenceHash::new(0x2222, 1, 0xBBBB);
        let psh2 = PositionalSequenceHash::new(0x3333, 2, 0xCCCC);

        tree.prefix(&psh0).insert(psh0, "pos0".to_string());
        tree.prefix(&psh1).insert(psh1, "pos1".to_string());
        tree.prefix(&psh2).insert(psh2, "pos2".to_string());

        // Test position() method
        assert!(tree.position(0).is_some());
        assert!(tree.position(1).is_some());
        assert!(tree.position(2).is_some());
        assert!(tree.position(3).is_none()); // No entries at position 3

        // Verify position lookup returns correct submap
        let pos0_map = tree.position(0).unwrap();
        assert_eq!(pos0_map.len(), 1);
    }

    // === PositionalSequenceHash Additional Tests ===

    #[test]
    fn test_positional_sequence_hash_mode_2_and_3() {
        // Mode 2: position fits in 24 bits (65536 <= pos < 16777216)
        let position_mode2 = 100_000u64;
        let seq_hash = 0x1234567890ABCDEF;
        let block_hash = 0xFEDCBA9876543210;

        let psh_mode2 = PositionalSequenceHash::new(seq_hash, position_mode2, block_hash);
        assert_eq!(psh_mode2.mode(), 2, "Position 100,000 should use mode 2");
        assert_eq!(psh_mode2.position(), position_mode2);
        assert_eq!(psh_mode2.sequence_hash(), seq_hash);
        // Local block hash truncated to 38 bits in mode 2
        assert_eq!(
            psh_mode2.local_block_hash(),
            block_hash & ((1u64 << 38) - 1)
        );

        // Mode 3: position fits in 31 bits (16777216 <= pos < 2147483648)
        let position_mode3 = 100_000_000u64;
        let psh_mode3 = PositionalSequenceHash::new(seq_hash, position_mode3, block_hash);
        assert_eq!(
            psh_mode3.mode(),
            3,
            "Position 100,000,000 should use mode 3"
        );
        assert_eq!(psh_mode3.position(), position_mode3);
        assert_eq!(psh_mode3.sequence_hash(), seq_hash);
        // Local block hash truncated to 31 bits in mode 3
        assert_eq!(
            psh_mode3.local_block_hash(),
            block_hash & ((1u64 << 31) - 1)
        );
    }

    #[test]
    fn test_positional_sequence_hash_as_u128() {
        let psh = PositionalSequenceHash::new(0x1234, 100, 0xABCD);
        let raw = psh.as_u128();

        // Verify we can reconstruct from raw value
        assert_eq!(raw & 0xFFFF_FFFF_FFFF_FFFF, 0x1234);
        assert!(raw > 0); // Non-zero

        // Create another and compare
        let psh2 = PositionalSequenceHash::new(0x1234, 100, 0xABCD);
        assert_eq!(psh.as_u128(), psh2.as_u128());
    }

    #[test]
    fn test_positional_sequence_hash_debug() {
        let psh = PositionalSequenceHash::new(0x1234567890ABCDEF, 42, 0xFEDCBA98);
        let debug_str = format!("{:?}", psh);

        // Debug should contain field names and values
        assert!(debug_str.contains("PositionalSequenceHash"));
        assert!(debug_str.contains("sequence_hash"));
        assert!(debug_str.contains("local_block_hash"));
        assert!(debug_str.contains("position"));
    }

    // === PositionalLineageHash Additional Tests ===

    #[test]
    fn test_positional_lineage_hash_debug_and_display() {
        // Test position 0 (no parent shown)
        let plh_root = PositionalLineageHash::new(0x123456789ABCDEF0, None, 0);
        let debug_root = format!("{:?}", plh_root);
        let display_root = format!("{}", plh_root);

        // Debug and Display should show position 0
        assert!(debug_root.starts_with("0:"));
        assert!(display_root.starts_with("0:"));
        // Position 0 should not show parent
        assert_eq!(debug_root.matches(':').count(), 1);
        assert_eq!(display_root.matches(':').count(), 1);

        // Test position > 0 (parent shown)
        let plh_child = PositionalLineageHash::new(0xABCDEF0123456789, Some(0x123456789ABCDEF0), 5);
        let debug_child = format!("{:?}", plh_child);
        let display_child = format!("{}", plh_child);

        // Should show position:current:parent
        assert!(debug_child.starts_with("5:"));
        assert!(display_child.starts_with("5:"));
        // Position > 0 should show parent (3 parts)
        assert_eq!(debug_child.matches(':').count(), 2);
        assert_eq!(display_child.matches(':').count(), 2);
    }

    #[test]
    fn test_positional_lineage_hash_as_u128() {
        let plh = PositionalLineageHash::new(0x1234, Some(0x5678), 10);
        let raw = plh.as_u128();

        assert!(raw > 0);

        // Create another with same params and compare
        let plh2 = PositionalLineageHash::new(0x1234, Some(0x5678), 10);
        assert_eq!(plh.as_u128(), plh2.as_u128());

        // Different params should give different hash
        let plh3 = PositionalLineageHash::new(0x1234, Some(0x5678), 11);
        assert_ne!(plh.as_u128(), plh3.as_u128());
    }

    // === Tokens From Impls ===

    #[test]
    fn test_tokens_from_vec_usize() {
        let usize_vec: Vec<usize> = vec![1, 2, 3, 4, 5];
        let tokens = Tokens::from(usize_vec);

        assert_eq!(tokens.as_ref(), &[1u32, 2, 3, 4, 5]);
        assert_eq!(tokens.len(), 5);
    }

    #[test]
    fn test_tokens_partial_eq_slice_ref() {
        let tokens = Tokens::from(vec![1u32, 2, 3, 4]);
        let slice: &[Token] = &[1, 2, 3, 4];

        // Test PartialEq<&[Token]> for Tokens
        assert!(tokens == slice);

        let different_slice: &[Token] = &[1, 2, 3, 5];
        assert!(tokens != different_slice);
    }

    // === TokenBlock Accessors ===

    #[test]
    fn test_token_block_accessors() {
        let tokens = Tokens::from(vec![1u32, 2, 3, 4]);
        let seq = TokenBlockSequence::new(tokens, 4, Some(1337));

        let block = &seq.blocks()[0];

        // Test block_size()
        assert_eq!(block.block_size(), 4);

        // Test positional_sequence_hash()
        let psh = block.positional_sequence_hash();
        assert_eq!(psh.position(), 0);

        // Test positional_lineage_hash()
        let plh = block.positional_lineage_hash();
        assert_eq!(plh.position(), 0);
        assert_eq!(plh.parent_hash_fragment(), 0); // Root has no parent
    }

    #[test]
    fn test_positional_hash_trait_impls() {
        use crate::PositionalHash;

        // Test PositionalHash for PositionalSequenceHash
        let psh = PositionalSequenceHash::new(0x1234, 42, 0xABCD);
        assert_eq!(PositionalHash::position(&psh), 42);

        // Test PositionalHash for PositionalLineageHash
        let plh = PositionalLineageHash::new(0x1234, None, 99);
        assert_eq!(PositionalHash::position(&plh), 99);
    }

    // === TokenBlockSequence Edge Cases ===

    #[test]
    fn test_sequence_pop_from_full_block() {
        // Test pop when current partial block is empty (must pop from full block)
        let tokens = Tokens::from(vec![1u32, 2, 3, 4, 5, 6, 7, 8]);
        let mut seq = TokenBlockSequence::new(tokens, 4, Some(TEST_SALT_HASH));

        // Current block should be empty, all tokens in completed blocks
        assert!(seq.current_block().is_empty());
        assert_eq!(seq.blocks().len(), 2);
        assert_eq!(seq.total_tokens(), 8);

        // Pop should remove from last full block
        let popped = seq.pop();
        assert_eq!(popped, Some(8));
        assert_eq!(seq.total_tokens(), 7);
        assert_eq!(seq.blocks().len(), 1);
        assert_eq!(seq.current_block().tokens.as_ref(), &[5, 6, 7]);
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)] // so we can explicitly test invalid ranges
    fn test_sequence_tokens_at_edge_cases() {
        let tokens = Tokens::from(vec![1u32, 2, 3, 4, 5]);
        let seq = TokenBlockSequence::new(tokens, 4, Some(TEST_SALT_HASH));

        // Start > end (invalid range
        assert!(seq.tokens_at(3..2).is_empty());

        // End > total (out of bounds)
        assert!(seq.tokens_at(0..10).is_empty());

        // Valid edge case: exact boundaries
        assert_eq!(seq.tokens_at(0..4).as_ref(), &[1, 2, 3, 4]);
        assert_eq!(seq.tokens_at(4..5).as_ref(), &[5]);
    }

    #[test]
    fn test_sequence_next_block() {
        let tokens = Tokens::from(vec![1u32, 2, 3, 4]);
        let seq = TokenBlockSequence::new(tokens, 4, Some(1337));

        let block = &seq.blocks()[0];
        let next_partial = block.next_block();

        // next_block should create a partial block linked to this block
        assert!(next_partial.is_empty());
        assert_eq!(next_partial.remaining(), 4);
        assert_eq!(
            next_partial.parent_sequence_hash,
            Some(block.sequence_hash())
        );
        assert_eq!(next_partial.position, 1);
    }

    #[test]
    fn test_sequence_reset() {
        let tokens = Tokens::from(vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9]);
        let mut seq = TokenBlockSequence::new(tokens, 4, Some(1337));

        assert_eq!(seq.blocks().len(), 2);
        assert_eq!(seq.total_tokens(), 9);

        seq.reset();

        assert!(seq.blocks().is_empty());
        assert!(seq.current_block().is_empty());
        assert_eq!(seq.total_tokens(), 0);
        assert_eq!(seq.current_block().parent_sequence_hash, None);
    }

    #[test]
    fn test_sequence_into_parts() {
        let tokens = Tokens::from(vec![1u32, 2, 3, 4, 5]);
        let seq = TokenBlockSequence::new(tokens, 4, Some(1337));

        let (blocks, partial) = seq.into_parts();

        assert_eq!(blocks.len(), 1);
        assert_eq!(partial.tokens.as_ref(), &[5]);
    }

    #[test]
    fn test_sequence_last_complete_block() {
        // Empty sequence
        let seq_empty = TokenBlockSequence::new(Tokens::default(), 4, None);
        assert!(seq_empty.last_complete_block().is_none());

        // With blocks
        let tokens = Tokens::from(vec![1u32, 2, 3, 4, 5, 6, 7, 8]);
        let seq = TokenBlockSequence::new(tokens, 4, Some(1337));
        let last = seq.last_complete_block();
        assert!(last.is_some());
        assert_eq!(last.unwrap().tokens().as_ref(), &[5, 6, 7, 8]);
    }
}
