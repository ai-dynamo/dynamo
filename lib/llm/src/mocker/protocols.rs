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

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash,
};

pub type Token = u32;
pub type GlobalHash = u64;
pub type NumBlocks = usize;

/// Represents an active block in the cache with a reference count
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum UniqueBlock {
    /// Block identified by UUID
    PartialBlock(Uuid),
    /// Block identified by hash
    FullBlock(GlobalHash),
}

impl Default for UniqueBlock {
    fn default() -> Self {
        // Generate a random UUID when default is used
        Self::PartialBlock(Uuid::new_v4())
    }
}

/// Represents different block movement operations in the cache
/// For Use and Promote variants, parent hash is the second field
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MoveBlock {
    Use(Vec<UniqueBlock>),
    Destroy(Vec<UniqueBlock>),
    Deref(Vec<UniqueBlock>),
    Promote(Uuid, GlobalHash, Option<u64>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MoveBlockResponse {
    Store(Vec<GlobalHash>, Option<u64>),
    Remove(Vec<GlobalHash>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectRequest {
    pub tokens: Vec<Token>,
    pub max_output_tokens: usize,
    pub uuid: Option<Uuid>,
    pub dp_rank: Option<u32>,
}

/// Represents the cost of prefilling content in the cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillCost {
    pub new_blocks: usize,
    pub new_tokens: usize,
    pub prefill_compute: f64,
}

/// Signal for output token generation with completion status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSignal {
    pub uuid: Uuid,
    pub completed: bool,
}

/// Configuration arguments for MockVllmEngine
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(pattern = "owned", build_fn(public))]
pub struct MockEngineArgs {
    #[builder(default = "16384")]
    pub num_gpu_blocks: usize,

    #[builder(default = "64")]
    pub block_size: usize,

    #[builder(default = None)]
    pub max_num_seqs: Option<usize>,

    // default for open api server, for llm class it's 16384
    #[builder(default = Some(8192))]
    pub max_num_batched_tokens: Option<usize>,

    #[builder(default = "0.01")]
    pub watermark: f64,

    #[builder(default = "1.0")]
    pub speedup_ratio: f64,

    #[builder(default = "1")]
    pub dp_size: u32,
}

impl MockEngineArgs {
    pub fn builder() -> MockEngineArgsBuilder {
        MockEngineArgsBuilder::default()
    }
}

/// Note: This assumes block_hash and tokens_hash are the same, which is not correct in rare cases
/// where the sequence-aware hash differs from the token content hash.
pub fn block_response_to_kv_event(response: MoveBlockResponse) -> KvCacheEventData {
    match response {
        MoveBlockResponse::Store(full_blocks, parent_hash) => {
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                blocks: full_blocks
                    .into_iter()
                    .map(|block| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(block),
                        tokens_hash: LocalBlockHash(block),
                    })
                    .collect(),
            })
        }
        MoveBlockResponse::Remove(full_blocks) => KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: full_blocks
                .into_iter()
                .map(ExternalSequenceBlockHash)
                .collect(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique_block_default_uniqueness() {
        // Create 10 default UniqueBlock instances
        let blocks: Vec<UniqueBlock> = (0..10).map(|_| UniqueBlock::default()).collect();

        // Extract UUIDs from each block
        let mut uuids = Vec::new();
        for block in blocks {
            match block {
                UniqueBlock::PartialBlock(uuid) => uuids.push(uuid),
                _ => panic!("Expected UuidIdentifier variant"),
            }
        }

        // Check that all UUIDs are unique by comparing each with every other
        for i in 0..uuids.len() {
            for j in i + 1..uuids.len() {
                assert_ne!(
                    uuids[i], uuids[j],
                    "UUID at index {} and {} are identical",
                    i, j
                );
            }
        }
    }
}
