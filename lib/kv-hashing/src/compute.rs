// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block-hash computation for a [`crate::Request`].

use dynamo_tokens::{
    BlockHash, PositionalLineageHash, SaltHash, SequenceHash, hash_complete_blocks,
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

    fn block_lineages(
        &self,
        block_size: u32,
    ) -> Result<Vec<dynamo_tokens::BlockLineage>, KvHashingError> {
        let salt_hash = self.salt_hash()?;
        let token_mm = self.token_mm_info();
        Ok(hash_complete_blocks(
            &self.tokens,
            &token_mm,
            block_size,
            salt_hash,
        ))
    }

    /// Returns the rich per-block result.
    ///
    /// One [`UniversalBlock`] per *complete* `block_size`-sized window in the request's
    /// token stream (placeholder slots count toward `block_size`). A trailing partial
    /// block — fewer than `block_size` slots — is not hashed and not returned.
    ///
    /// Tokens are borrowed. Text windows are hashed in place. Only a window that
    /// overlaps a multimodal run allocates the 13-byte tagged frame.
    pub fn into_blocks(&self, block_size: u32) -> Result<Vec<UniversalBlock>, KvHashingError> {
        Ok(self
            .block_lineages(block_size)?
            .into_iter()
            .map(UniversalBlock::from)
            .collect())
    }

    /// Projection: per-block [`BlockHash`].
    pub fn block_hashes(&self, block_size: u32) -> Result<Vec<BlockHash>, KvHashingError> {
        Ok(self
            .block_lineages(block_size)?
            .into_iter()
            .map(|lineage| lineage.block_hash)
            .collect())
    }

    /// Projection: per-block [`SequenceHash`] (parent-chained, derived from PLH).
    pub fn sequence_hashes(&self, block_size: u32) -> Result<Vec<SequenceHash>, KvHashingError> {
        Ok(self
            .block_lineages(block_size)?
            .into_iter()
            .map(|lineage| lineage.sequence_hash)
            .collect())
    }

    /// Consuming projection: per-block [`SequenceHash`].
    ///
    /// Hashing borrows the token slice, so this does the same work as
    /// [`Self::sequence_hashes`]. It remains for callers that already own the
    /// request and do not need it afterward.
    pub fn into_sequence_hashes(
        self,
        block_size: u32,
    ) -> Result<Vec<SequenceHash>, KvHashingError> {
        self.sequence_hashes(block_size)
    }

    /// Projection: per-block [`PositionalLineageHash`] (the universal identifier).
    pub fn positional_lineage_hashes(
        &self,
        block_size: u32,
    ) -> Result<Vec<PositionalLineageHash>, KvHashingError> {
        Ok(self
            .block_lineages(block_size)?
            .into_iter()
            .map(|lineage| lineage.positional_lineage_hash)
            .collect())
    }
}
