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

use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type Token = u32;
pub type LocalBlockHash = u64;
/// A global hash identifier for blocks
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MoveBlock {
    Use(Vec<UniqueBlock>, Option<f64>),
    Destroy(Vec<UniqueBlock>),
    Deref(Vec<UniqueBlock>),
    Promote(Uuid, GlobalHash),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectRequest {
    pub tokens: Vec<Token>,
    pub max_output_tokens: usize,
    pub uuid: Option<Uuid>,
}

/// Represents the cost of prefilling content in the cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillCost {
    pub new_tokens: usize,
    pub prefill_compute: f64,
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
