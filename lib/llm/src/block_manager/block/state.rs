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

use std::sync::Arc;

use derive_getters::Getters;

use super::registry::RegistrationHandle;
use super::Result;
use crate::tokens::{PartialTokenBlock, SaltHash, Token, TokenBlock, Tokens};

#[derive(Debug, thiserror::Error)]
#[error("Block state is invalid: {0}")]
pub struct BlockStateInvalid(pub String);

#[derive(Debug)]
pub enum BlockState {
    Reset,
    Partial(PartialState),
    Complete(CompleteState),
    Registered(Arc<RegistrationHandle>, Arc<()>),
}

impl BlockState {
    pub fn initialize_sequence(
        &mut self,
        page_size: usize,
        salt_hash: SaltHash,
    ) -> Result<(), BlockStateInvalid> {
        if !matches!(self, BlockState::Reset) {
            return Err(BlockStateInvalid("Block is not reset".to_string()));
        }

        let block = PartialTokenBlock::create_sequence_root(page_size, salt_hash);
        *self = BlockState::Partial(PartialState::new(block));
        Ok(())
    }

    pub fn add_token(&mut self, token: Token) -> Result<()> {
        match self {
            BlockState::Partial(state) => Ok(state.block.push_token(token)?),
            _ => Err(BlockStateInvalid("Block is not partial".to_string()))?,
        }
    }

    pub fn add_tokens(&mut self, tokens: Tokens) -> Result<Tokens> {
        match self {
            BlockState::Partial(state) => Ok(state.block.push_tokens(tokens)),
            _ => Err(BlockStateInvalid("Block is not partial".to_string()))?,
        }
    }

    pub fn pop_token(&mut self) -> Result<()> {
        match self {
            BlockState::Partial(state) => {
                state.block.pop_token()?;
                Ok(())
            }
            _ => Err(BlockStateInvalid("Block is not partial".to_string()))?,
        }
    }

    pub fn pop_tokens(&mut self, count: usize) -> Result<()> {
        match self {
            BlockState::Partial(state) => {
                state.block.pop_tokens(count)?;
                Ok(())
            }
            _ => Err(BlockStateInvalid("Block is not partial".to_string()))?,
        }
    }

    pub fn commit(&mut self) -> Result<()> {
        match self {
            BlockState::Partial(state) => {
                let token_block = state.block.commit()?;
                *self = BlockState::Complete(CompleteState::new(token_block));
                Ok(())
            }
            _ => Err(BlockStateInvalid("Block is not partial".to_string()))?,
        }
    }

    pub fn apply_token_block(&mut self, token_block: TokenBlock) -> Result<()> {
        match self {
            BlockState::Reset => {
                *self = BlockState::Complete(CompleteState::new(token_block));
                Ok(())
            }
            _ => Err(BlockStateInvalid("Block is not reset".to_string()))?,
        }
    }

    /// Returns the number of tokens currently in the block.
    pub fn len(&self) -> Option<usize> {
        match self {
            BlockState::Reset => Some(0),
            BlockState::Partial(state) => Some(state.block.len()),
            BlockState::Complete(state) => Some(state.token_block.tokens().len()),
            BlockState::Registered(_, _) => None,
        }
    }

    /// Returns the number of additional tokens that can be added.
    pub fn remaining(&self) -> usize {
        match self {
            BlockState::Partial(state) => state.block.remaining(),
            _ => 0, // Reset, Complete, Registered have 0 remaining capacity
        }
    }

    /// Returns true if the block contains no tokens.
    pub fn is_empty(&self) -> bool {
        match self {
            BlockState::Reset => true,
            BlockState::Partial(state) => state.block.is_empty(),
            BlockState::Complete(_) => false,      // Always full
            BlockState::Registered(_, _) => false, // Always full
        }
    }

    /// Returns a reference to the underlying TokenBlock if the state is Complete or Registered.
    pub fn tokens(&self) -> Option<&Tokens> {
        match self {
            BlockState::Reset | BlockState::Registered(_, _) => None,
            BlockState::Partial(state) => Some(state.block.tokens()),
            BlockState::Complete(state) => Some(state.token_block.tokens()),
        }
    }

    /// Returns true if the block is empty
    pub fn is_reset(&self) -> bool {
        matches!(self, BlockState::Reset)
    }

    /// Returns true if the block is in the complete or registered state
    pub fn is_complete(&self) -> bool {
        matches!(self, BlockState::Complete(_) | BlockState::Registered(_, _))
    }

    /// Returns true if the block is in the registered state
    pub fn is_registered(&self) -> bool {
        matches!(self, BlockState::Registered(_state, _))
    }
}

#[derive(Debug)]
pub struct PartialState {
    block: PartialTokenBlock,
}

impl PartialState {
    pub fn new(block: PartialTokenBlock) -> Self {
        Self { block }
    }
}

#[derive(Debug, Getters)]
pub struct CompleteState {
    token_block: TokenBlock,
}

impl CompleteState {
    pub fn new(token_block: TokenBlock) -> Self {
        Self { token_block }
    }
}
