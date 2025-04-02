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

use crate::kv_router::indexer::compute_hash;
use bytemuck::cast_slice;
use derive_getters::{Dissolve, Getters};
use rayon::prelude::*;

pub type Token = u32;

/// A hash of the only the tokens within a block computed from [compute_hash].
pub type BlockHash = u64;

/// A sequence aware hash that combines the previous block's sequence hash with the current block's hash.
pub type SequenceHash = u64;

#[derive(Debug, Clone, Dissolve, Default)]
pub struct Tokens(Vec<Token>);

impl AsRef<[Token]> for Tokens {
    fn as_ref(&self) -> &[Token] {
        &self.0
    }
}

impl AsMut<[Token]> for Tokens {
    fn as_mut(&mut self) -> &mut [Token] {
        &mut self.0
    }
}

impl std::ops::Deref for Tokens {
    type Target = [Token];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Tokens {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl std::borrow::Borrow<[Token]> for Tokens {
    fn borrow(&self) -> &[Token] {
        &self.0
    }
}

impl IntoIterator for Tokens {
    type Item = Token;

    type IntoIter = std::vec::IntoIter<Token>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
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
    fn from(tokens: Vec<i32>) -> Self {
        Tokens(tokens.into_iter().map(|t| t as u32).collect())
    }
}

impl From<&[i32]> for Tokens {
    fn from(tokens: &[i32]) -> Self {
        Tokens(tokens.iter().map(|&t| t as u32).collect())
    }
}

impl From<Tokens> for Vec<Token> {
    fn from(tokens: Tokens) -> Self {
        tokens.0
    }
}

impl Tokens {
    pub fn into_sequence(self, block_size: usize) -> TokenSequence {
        TokenSequence::new(self, block_size)
    }

    pub fn compute_block_hash(tokens: &[Token], block_size: usize) -> Vec<BlockHash> {
        tokens
            .par_chunks_exact(block_size)
            .map(|chunk| compute_hash(cast_slice(chunk)))
            .collect()
    }
}

pub struct PartialTokenBlock {
    tokens: Tokens,
    block_size: usize,
    parent_sequence_hash: Option<SequenceHash>,
}

impl PartialTokenBlock {
    /// Push a token onto the block, if the block is full, return a new [TokenBlock]
    /// and reset the incomplete block
    pub fn push_token(&mut self, token: Token) -> Option<TokenBlock> {
        self.tokens.0.push(token);
        if self.tokens.0.len() == self.block_size {
            let block = std::mem::take(&mut self.tokens);
            let block_hash = compute_hash(cast_slice(&block));
            let sequence_hash = compute_hash(bytemuck::cast_slice(&[
                self.parent_sequence_hash.unwrap_or_default(),
                block_hash,
            ]));
            Some(TokenBlock {
                tokens: block,
                sequence_hash,
                block_hash,
                parent_sequence_hash: self.parent_sequence_hash,
            })
        } else {
            None
        }
    }

    pub fn tokens(&self) -> &Tokens {
        &self.tokens
    }
}

impl std::ops::Deref for PartialTokenBlock {
    type Target = Tokens;

    fn deref(&self) -> &Self::Target {
        &self.tokens
    }
}

#[derive(Debug, Clone, Getters, Default)]
pub struct TokenBlock {
    tokens: Tokens,

    #[getter(copy)]
    block_hash: BlockHash,

    #[getter(copy)]
    sequence_hash: SequenceHash,

    #[getter(copy)]
    parent_sequence_hash: Option<SequenceHash>,
}

pub struct TokenSequence {
    blocks: Vec<TokenBlock>,
    current_block: PartialTokenBlock,
}

impl TokenSequence {
    pub fn new(tokens: Tokens, block_size: usize) -> Self {
        let (blocks, current_block) = Self::split_tokens(tokens, block_size);

        Self {
            blocks,
            current_block,
        }
    }

    pub fn push_token(&mut self, token: Token) -> Option<&TokenBlock> {
        if let Some(block) = self.current_block.push_token(token) {
            self.blocks.push(block);
            self.blocks.last()
        } else {
            None
        }
    }

    pub fn blocks(&self) -> &[TokenBlock] {
        &self.blocks
    }

    pub fn current_block(&self) -> &PartialTokenBlock {
        &self.current_block
    }

    pub fn into_parts(self) -> (Vec<TokenBlock>, PartialTokenBlock) {
        (self.blocks, self.current_block)
    }

    pub fn split_tokens(tokens: Tokens, block_size: usize) -> (Vec<TokenBlock>, PartialTokenBlock) {
        // Use rayon's parallel iterator to process chunks in parallel
        let mut blocks: Vec<TokenBlock> = tokens
            .par_chunks_exact(block_size)
            .map(|chunk| TokenBlock {
                tokens: chunk.to_vec().into(),
                sequence_hash: 0,
                block_hash: compute_hash(cast_slice(chunk)),
                parent_sequence_hash: None,
            })
            .collect();

        blocks[0].sequence_hash = blocks[0].block_hash;

        // compute the sequence hash for each block
        // this is the sequence hash of the previous block with the current block's hash
        for i in 1..blocks.len() {
            let previous_block = &blocks[i - 1];
            let parent_sequence_hash = previous_block.sequence_hash;
            let vals = &[parent_sequence_hash, blocks[i].block_hash];
            blocks[i].sequence_hash = compute_hash(bytemuck::cast_slice(vals));
            blocks[i].parent_sequence_hash = Some(parent_sequence_hash);
        }

        let remainder = tokens.chunks_exact(block_size).remainder();

        let next_block = PartialTokenBlock {
            tokens: remainder.into(),
            block_size,
            parent_sequence_hash: blocks.last().map(|b| b.sequence_hash),
        };

        (blocks, next_block)
    }
}

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

impl PartialEq<Vec<Token>> for &Tokens {
    fn eq(&self, other: &Vec<Token>) -> bool {
        self.0 == *other
    }
}

impl<'a> PartialEq<&'a Tokens> for Vec<Token> {
    fn eq(&self, other: &&'a Tokens) -> bool {
        *self == other.0
    }
}

impl PartialEq<[Token]> for &Tokens {
    fn eq(&self, other: &[Token]) -> bool {
        self.0.as_slice() == other
    }
}

impl<'a> PartialEq<&'a [Token]> for Tokens {
    fn eq(&self, other: &&'a [Token]) -> bool {
        self.0.as_slice() == *other
    }
}

impl PartialEq for Tokens {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Tokens {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokens_slice_operations() {
        let tokens = Tokens(vec![1, 2, 3, 4, 5]);

        // Test AsRef<[Token]>
        let slice: &[Token] = tokens.as_ref();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);

        // Test Deref
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0], 1);
        assert_eq!(tokens[4], 5);

        // Test iteration
        let sum: u32 = tokens.iter().sum();
        assert_eq!(sum, 15);

        // Test slicing
        let slice = &tokens[1..4];
        assert_eq!(slice, &[2, 3, 4]);

        // Test Borrow
        let borrowed: &[Token] = std::borrow::Borrow::borrow(&tokens);
        assert_eq!(borrowed, &[1, 2, 3, 4, 5]);

        // Test with functions that accept &[Token]
        fn takes_slice(slice: &[Token]) -> usize {
            slice.len()
        }

        assert_eq!(takes_slice(&tokens), 5);
    }

    #[test]
    fn test_tokens_conversions() {
        // Test From<Vec<Token>> for Tokens
        let vec = vec![1, 2, 3, 4, 5];
        let tokens: Tokens = vec.clone().into();
        assert_eq!(tokens.0, vec);

        // Test Into<Vec<Token>> for Tokens
        let tokens = Tokens(vec![6, 7, 8, 9, 10]);
        let vec: Vec<Token> = tokens.into();
        assert_eq!(vec, vec![6, 7, 8, 9, 10]);

        // Test From<&[Token]> for Tokens
        let slice: &[Token] = &[11, 12, 13];
        let tokens: Tokens = slice.into();
        assert_eq!(tokens.0, vec![11, 12, 13]);

        // Test From<Vec<i32>> for Tokens
        let i32_values = vec![100_i32, 200_i32, 300_i32];
        let tokens: Tokens = i32_values.into();
        assert_eq!(tokens.0, vec![100, 200, 300]);

        // Test From<&[i32]> for Tokens
        let i32_slice: &[i32] = &[400_i32, 500_i32, 600_i32];
        let tokens: Tokens = i32_slice.into();
        assert_eq!(tokens.0, vec![400, 500, 600]);
    }

    #[test]
    fn test_tokens_blocks() {
        let tokens = Tokens(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let sequence = TokenSequence::new(tokens, 4);

        assert_eq!(sequence.blocks().len(), 2);
        assert_eq!(sequence.current_block().len(), 2);

        assert_eq!(sequence.blocks()[0].tokens(), vec![1, 2, 3, 4]);
        assert_eq!(sequence.blocks()[0].block_hash(), 14643705804678351452);
        assert_eq!(sequence.blocks()[0].sequence_hash(), 14643705804678351452);
        println!("blocks[0]: {:?}", sequence.blocks()[0]);

        assert_eq!(sequence.blocks()[1].tokens(), vec![5, 6, 7, 8]);
        assert_eq!(sequence.blocks()[1].block_hash(), 16777012769546811212);
        assert_eq!(sequence.blocks()[1].sequence_hash(), 4945711292740353085);
        println!("blocks[1]: {:?}", sequence.blocks()[1]);

        assert_eq!(sequence.current_block().tokens(), vec![9, 10]);

        let mut sequence = sequence;

        let new_block = sequence.push_token(11);
        assert!(new_block.is_none());
        assert_eq!(sequence.blocks().len(), 2);

        let new_block = sequence.push_token(12);
        assert!(new_block.is_some());
        assert_eq!(sequence.blocks().len(), 3);
        assert_eq!(sequence.current_block().tokens().len(), 0);
        println!("blocks[2]: {:?}", sequence.blocks()[2]);

        let (blocks, mut current_block) = sequence.into_parts();

        let new_block = current_block.push_token(13);
        assert!(new_block.is_none());
        assert_eq!(current_block.tokens().len(), 1);

        let new_block = current_block.push_token(14);
        assert!(new_block.is_none());
        assert_eq!(current_block.tokens().len(), 2);

        let new_block = current_block.push_token(15);
        assert!(new_block.is_none());
        assert_eq!(current_block.tokens().len(), 3);

        let new_block = current_block.push_token(16);
        assert!(new_block.is_some());
        assert_eq!(blocks.len(), 3);
        assert_eq!(current_block.tokens().len(), 0);
    }
}
