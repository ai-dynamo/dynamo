// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block-hash computation for a [`crate::Request`].

use dynamo_tokens::{
    BlockHash, PositionalLineageHash, SaltHash, SequenceHash, TokenBlockSequence, Tokens,
};

use crate::block::UniversalBlock;
use crate::error::KvHashingError;
use crate::request::Request;
use crate::salt::compute_salt_hash;

impl Request {
    /// Returns the canonical [`SaltHash`] for this request.
    pub fn salt_hash(&self) -> Result<SaltHash, KvHashingError> {
        compute_salt_hash(self.salt(), self.lora_name())
    }

    /// Returns the rich per-block result.
    ///
    /// One [`UniversalBlock`] per *complete* `block_size`-sized window in the request's
    /// token stream (placeholder slots count toward `block_size`). A trailing partial
    /// block — fewer than `block_size` slots — is not hashed and not returned.
    pub fn into_blocks(&self, block_size: u32) -> Result<Vec<UniversalBlock>, KvHashingError> {
        let salt_hash = self.salt_hash()?;
        let token_mm = self.token_mm_info();
        let seq = TokenBlockSequence::new_with_mm(
            Tokens::from(self.tokens.clone()),
            &token_mm,
            block_size,
            Some(salt_hash),
        )?;
        Ok(seq.blocks().iter().map(UniversalBlock::from).collect())
    }

    /// Projection: per-block [`BlockHash`].
    pub fn block_hashes(&self, block_size: u32) -> Result<Vec<BlockHash>, KvHashingError> {
        Ok(self
            .into_blocks(block_size)?
            .into_iter()
            .map(|b| b.block_hash)
            .collect())
    }

    /// Projection: per-block [`SequenceHash`] (parent-chained, derived from PLH).
    pub fn sequence_hashes(&self, block_size: u32) -> Result<Vec<SequenceHash>, KvHashingError> {
        Ok(self
            .into_blocks(block_size)?
            .into_iter()
            .map(|b| b.sequence_hash())
            .collect())
    }

    /// Projection: per-block [`PositionalLineageHash`] (the universal identifier).
    pub fn positional_lineage_hashes(
        &self,
        block_size: u32,
    ) -> Result<Vec<PositionalLineageHash>, KvHashingError> {
        Ok(self
            .into_blocks(block_size)?
            .into_iter()
            .map(|b| b.plh)
            .collect())
    }
}
