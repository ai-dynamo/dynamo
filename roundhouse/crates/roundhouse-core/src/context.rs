// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Context assembly and incremental tokenization.
//!
//! Routing on cache locality means knowing the exact block hashes of the prompt
//! before dispatching it. Recomputing those from scratch every turn would cost
//! O(context) per turn and O(context * turns) over a session — which for a long
//! agentic run is more work than the routing decision can possibly save.
//!
//! Because the conversation is append-only and Dynamo's block hashes are
//! computed per fixed-size block from tokens alone, only the newly completed
//! blocks need hashing on each turn. [`TokenBuffer`] keeps the running token
//! sequence, the block hashes, and the rolling sequence-hash chain, and extends
//! all three in place.
//!
//! A buffer is valid for exactly one tokenizer: block hashes are over token
//! ids, so a session that can be served by two different tokenizer families
//! needs one buffer per family. Frontier providers expose no hashes at all, and
//! are costed from the routing ledger instead.

use dynamo_kv_router::protocols::{
    BlockHashOptions, LocalBlockHash, compute_block_hash_for_seq, compute_next_seq_hash,
};
use dynamo_tokens::SequenceHash;

use crate::item::Item;

/// Minimal tokenizer seam.
///
/// Kept as a trait so the core stays free of model assets and tests stay
/// deterministic. A real deployment supplies an implementation backed by the
/// model's own tokenizer — the block hashes are only meaningful to a worker if
/// the token ids match what that worker would produce.
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<u32>;
}

/// Byte-level tokenizer used in tests.
///
/// Not a real BPE: it exists so the buffer's incremental behavior can be tested
/// without shipping a vocabulary.
#[derive(Debug, Default, Clone, Copy)]
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.as_bytes().iter().map(|b| *b as u32).collect()
    }
}

/// Append-only token sequence with incrementally maintained hashes.
pub struct TokenBuffer {
    tokens: Vec<u32>,
    block_size: u32,
    block_hashes: Vec<LocalBlockHash>,
    sequence_hashes: Vec<SequenceHash>,
}

impl TokenBuffer {
    pub fn new(block_size: u32) -> Self {
        assert!(block_size > 0, "block size must be non-zero");
        Self {
            tokens: Vec::new(),
            block_size,
            block_hashes: Vec::new(),
            sequence_hashes: Vec::new(),
        }
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Total tokens, i.e. the input sequence length of the next request.
    pub fn isl_tokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn block_hashes(&self) -> &[LocalBlockHash] {
        &self.block_hashes
    }

    /// Rolling sequence hashes, the form the router matches prefixes on.
    pub fn sequence_hashes(&self) -> &[SequenceHash] {
        &self.sequence_hashes
    }

    /// Number of complete blocks for a given token count.
    ///
    /// Eagle speculative decoding shifts block boundaries by one token; the
    /// skeleton supports only the standard layout, and a session configured for
    /// Eagle would need this and [`BlockHashOptions::is_eagle`] threaded
    /// through together.
    fn complete_blocks(&self, token_count: usize) -> usize {
        token_count / self.block_size as usize
    }

    /// Append tokens, hashing only the blocks this append completes.
    ///
    /// Returns the number of newly completed blocks. The already-hashed prefix
    /// is never revisited, which is what keeps per-turn cost proportional to
    /// the delta rather than the conversation.
    pub fn append(&mut self, new_tokens: &[u32]) -> usize {
        if new_tokens.is_empty() {
            return 0;
        }

        let blocks_before = self.block_hashes.len();
        debug_assert_eq!(
            blocks_before,
            self.complete_blocks(self.tokens.len()),
            "block hashes must stay in step with the token buffer"
        );

        self.tokens.extend_from_slice(new_tokens);
        let blocks_after = self.complete_blocks(self.tokens.len());
        if blocks_after == blocks_before {
            return 0;
        }

        // Hash only the newly completed, block-aligned region. Block hashes
        // depend on the block's tokens and the salt seed alone -- not on
        // position -- so hashing this window matches hashing the whole
        // sequence and discarding the prefix. `sequence_hashes` is what carries
        // position, via the chain below.
        let stride = self.block_size as usize;
        let start = blocks_before * stride;
        let end = blocks_after * stride;
        let fresh = compute_block_hash_for_seq(
            &self.tokens[start..end],
            self.block_size,
            BlockHashOptions::default(),
        );
        debug_assert_eq!(fresh.len(), blocks_after - blocks_before);

        // Extend the rolling chain: the first block's sequence hash is its
        // block hash, and every later one folds in its parent.
        for block_hash in &fresh {
            let next = match self.sequence_hashes.last() {
                Some(parent) => compute_next_seq_hash(*parent, *block_hash),
                None => block_hash.0,
            };
            self.sequence_hashes.push(next);
        }
        self.block_hashes.extend(fresh);

        blocks_after - blocks_before
    }

    /// Recompute every hash from scratch. Reference implementation for tests.
    #[cfg(test)]
    fn recomputed_from_scratch(&self) -> (Vec<LocalBlockHash>, Vec<SequenceHash>) {
        let blocks = compute_block_hash_for_seq(
            &self.tokens,
            self.block_size,
            BlockHashOptions::default(),
        );
        let mut chain: Vec<SequenceHash> = Vec::with_capacity(blocks.len());
        for block_hash in &blocks {
            let next = match chain.last() {
                Some(parent) => compute_next_seq_hash(*parent, *block_hash),
                None => block_hash.0,
            };
            chain.push(next);
        }
        (blocks, chain)
    }
}

/// Owns the canonical item list and its derived token buffer.
pub struct ContextAssembler<T: Tokenizer> {
    tokenizer: T,
    items: Vec<Item>,
    buffer: TokenBuffer,
}

impl<T: Tokenizer> ContextAssembler<T> {
    pub fn new(tokenizer: T, block_size: u32) -> Self {
        Self {
            tokenizer,
            items: Vec::new(),
            buffer: TokenBuffer::new(block_size),
        }
    }

    /// Rebuild from a replayed item list, e.g. after a failover.
    pub fn rehydrate(tokenizer: T, block_size: u32, items: Vec<Item>) -> Self {
        let mut assembler = Self::new(tokenizer, block_size);
        for item in items {
            assembler.push(item);
        }
        assembler
    }

    /// Commit an item and extend the token buffer.
    ///
    /// Returns how many new blocks this completed — the quantity that bounds
    /// the routing work this turn will need.
    pub fn push(&mut self, item: Item) -> usize {
        let tokens = self.tokenizer.encode(&item.render());
        self.items.push(item);
        self.buffer.append(&tokens)
    }

    pub fn items(&self) -> &[Item] {
        &self.items
    }

    pub fn buffer(&self) -> &TokenBuffer {
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: u32 = 16;

    fn tokens(n: usize, seed: u32) -> Vec<u32> {
        (0..n).map(|i| seed.wrapping_add(i as u32) % 1000).collect()
    }

    #[test]
    fn incremental_hashing_matches_a_full_recompute() {
        let mut buffer = TokenBuffer::new(BLOCK);
        // Appends deliberately unaligned to the block size, so most of them
        // leave a partial block that the next append must complete.
        for (step, size) in [7usize, 13, 5, 40, 1, 22, 9].iter().enumerate() {
            buffer.append(&tokens(*size, step as u32 * 31));
        }

        let (expected_blocks, expected_chain) = buffer.recomputed_from_scratch();
        assert_eq!(buffer.block_hashes(), expected_blocks.as_slice());
        assert_eq!(buffer.sequence_hashes(), expected_chain.as_slice());
        assert_eq!(buffer.block_hashes().len(), buffer.isl_tokens() / BLOCK as usize);
    }

    #[test]
    fn appending_in_one_shot_equals_appending_piecewise() {
        let whole = tokens(200, 5);

        let mut one_shot = TokenBuffer::new(BLOCK);
        one_shot.append(&whole);

        let mut piecewise = TokenBuffer::new(BLOCK);
        for chunk in whole.chunks(7) {
            piecewise.append(chunk);
        }

        assert_eq!(one_shot.block_hashes(), piecewise.block_hashes());
        assert_eq!(one_shot.sequence_hashes(), piecewise.sequence_hashes());
    }

    #[test]
    fn a_partial_block_produces_no_hashes_until_it_completes() {
        let mut buffer = TokenBuffer::new(BLOCK);
        assert_eq!(buffer.append(&tokens(15, 0)), 0);
        assert!(buffer.block_hashes().is_empty());

        // One more token closes the first block.
        assert_eq!(buffer.append(&tokens(1, 99)), 1);
        assert_eq!(buffer.block_hashes().len(), 1);
    }

    #[test]
    fn a_shared_prefix_yields_a_shared_sequence_hash_chain() {
        let prefix = tokens(64, 1);

        let mut a = TokenBuffer::new(BLOCK);
        a.append(&prefix);
        a.append(&tokens(32, 700));

        let mut b = TokenBuffer::new(BLOCK);
        b.append(&prefix);
        b.append(&tokens(32, 900));

        // The shared prefix must hash identically -- this is exactly what the
        // router matches on -- while the divergent tails must not.
        let shared = prefix.len() / BLOCK as usize;
        assert_eq!(a.sequence_hashes()[..shared], b.sequence_hashes()[..shared]);
        assert_ne!(a.sequence_hashes()[shared], b.sequence_hashes()[shared]);
    }

    #[test]
    fn assembler_grows_the_buffer_monotonically_across_turns() {
        let mut assembler = ContextAssembler::new(ByteTokenizer, BLOCK);
        assembler.push(Item::system_text("you are a careful assistant"));

        let mut previous = assembler.buffer().isl_tokens();
        for turn in 0..10 {
            assembler.push(Item::user_text(format!("question number {turn}")));
            let current = assembler.buffer().isl_tokens();
            assert!(current > previous, "context must grow every turn");
            previous = current;
        }
        assert_eq!(assembler.items().len(), 11);
    }

    #[test]
    fn rehydrating_from_items_reproduces_the_buffer_exactly() {
        let mut original = ContextAssembler::new(ByteTokenizer, BLOCK);
        for turn in 0..6 {
            original.push(Item::user_text(format!("turn {turn} with some padding text")));
        }

        let restored =
            ContextAssembler::rehydrate(ByteTokenizer, BLOCK, original.items().to_vec());

        // This is the failover path: a successor node replays the item log and
        // must arrive at byte-identical routing inputs.
        assert_eq!(restored.buffer().tokens(), original.buffer().tokens());
        assert_eq!(
            restored.buffer().sequence_hashes(),
            original.buffer().sequence_hashes()
        );
    }
}
