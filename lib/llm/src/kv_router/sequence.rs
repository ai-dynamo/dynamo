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

use crate::mocker::protocols::UniqueBlock;
use crate::tokens::{TokenBlockSequence, Tokens};
use derive_getters::Getters;
use std::collections::{HashMap, HashSet};
use uuid;

// TODO: use the common request_id if it exists in the repo
pub type RequestId = String;

/// Create unique blocks from a TokenBlockSequence
fn create_unique_blocks_from_sequence(
    tokens: &TokenBlockSequence,
    uuid: Option<uuid::Uuid>,
    block_size: usize,
) -> Vec<UniqueBlock> {
    let mut unique_blocks: Vec<UniqueBlock> = tokens
        .blocks()
        .iter()
        .map(|block| UniqueBlock::FullBlock(block.sequence_hash()))
        .collect();

    // Only push the partial block if tokens count isn't a multiple of block_size
    if tokens.total_tokens() % block_size != 0 {
        unique_blocks.push(match uuid {
            Some(uuid) => UniqueBlock::PartialBlock(uuid),
            None => UniqueBlock::default(),
        });
    }
    unique_blocks
}

/// A multi-request sequence manager that handles multiple active sequences with shared KV cache
#[derive(Debug, Getters)]
pub struct SharedSequenceManager {
    active_seqs: HashMap<RequestId, TokenBlockSequence>,

    partial_blocks: HashMap<RequestId, UniqueBlock>,

    unique_blocks: HashMap<UniqueBlock, HashSet<RequestId>>,

    #[getter(copy)]
    block_size: usize,

    #[getter(copy)]
    active_blocks: usize,
}

impl SharedSequenceManager {
    /// Create a new SharedSequenceManager instance
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 1, "block_size must be greater than 1");

        Self {
            active_seqs: HashMap::new(),
            partial_blocks: HashMap::new(),
            unique_blocks: HashMap::new(),
            block_size,
            active_blocks: 0,
        }
    }

    fn add_block(&mut self, request_id: RequestId, block: &UniqueBlock) {
        let is_new_block = !self.unique_blocks.contains_key(block);

        self.unique_blocks
            .entry(block.clone())
            .or_default()
            .insert(request_id.clone());

        if is_new_block {
            self.active_blocks += 1;
        }

        if matches!(block, UniqueBlock::PartialBlock(_)) {
            self.partial_blocks.insert(request_id, block.clone());
        };
    }

    fn remove_block(&mut self, request_id: &RequestId, block: &UniqueBlock) {
        let Some(request_ids) = self.unique_blocks.get_mut(block) else {
            panic!("Cannot remove a block that does not exist.")
        };

        // Remove the unique block if no more requests using it
        request_ids.retain(|w| w != request_id);
        if request_ids.is_empty() {
            self.active_blocks -= 1;
            self.unique_blocks.remove(block);
        }
    }

    /// Add a new request with its initial tokens
    pub fn add_request(&mut self, request_id: RequestId, tokens: Vec<u32>) {
        let token_sequence = Tokens::from(tokens).into_sequence(self.block_size as u32, None);
        let blocks = create_unique_blocks_from_sequence(&token_sequence, None, self.block_size);

        for block in &blocks {
            self.add_block(request_id.clone(), block);
        }

        self.active_seqs.insert(request_id.clone(), token_sequence);
    }

    /// Free all blocks associated with a request
    pub fn free(&mut self, request_id: &RequestId) {
        let Some(token_seq) = self.active_seqs.get(request_id) else {
            panic!("Cannot free non-existent request {request_id}");
        };

        let blocks = create_unique_blocks_from_sequence(token_seq, None, self.block_size);
        for block in blocks {
            if matches!(block, UniqueBlock::FullBlock(_)) {
                self.remove_block(request_id, &block);
            }
        }
        if let Some(partial_block) = self.partial_blocks.remove(request_id) {
            self.remove_block(request_id, &partial_block);
        }

        self.active_seqs.remove(request_id).unwrap();
    }

    /// Push a token to a specific request's sequence
    pub fn push(&mut self, request_id: &RequestId, token: u32) {
        let token_seq = self
            .active_seqs
            .get_mut(request_id)
            .expect("Request ID not found for token push");
        token_seq.append(token).expect("Token push failed.");

        // No need to update anything
        if token_seq.total_tokens() % self.block_size != 1 {
            return;
        }

        let last_seq_hash = token_seq
            .last_complete_block()
            .map(|block| block.sequence_hash());

        // Promote a partial block into a full block if not already
        if let Some(partial_block) = self.partial_blocks.get(request_id).cloned() {
            self.remove_block(request_id, &partial_block);
        }
        if let Some(full_block) = last_seq_hash {
            self.add_block(request_id.clone(), &UniqueBlock::FullBlock(full_block));
        }

        self.add_block(request_id.clone(), &UniqueBlock::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_sequence_manager_operations() {
        let mut manager = SharedSequenceManager::new(4);

        // Step 1: Add request 0 with tokens [0, 1, 2], then push 3 and 4
        manager.add_request("0".to_string(), vec![0, 1, 2]);
        manager.push(&"0".to_string(), 3);
        manager.push(&"0".to_string(), 4);

        assert_eq!(manager.active_blocks(), 2);
        assert_eq!(manager.partial_blocks.len(), 1);

        // Step 2: Add request 1 with tokens [0, 1, 2, 3, 4, 5, 6]
        manager.add_request("1".to_string(), vec![0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(manager.active_blocks(), 3);

        // Check that only one key is FullBlock with both requests sharing it
        let mut full_block_count = 0;
        let mut shared_block_requests = None;
        for (block, requests) in &manager.unique_blocks {
            if let UniqueBlock::FullBlock(_) = block {
                full_block_count += 1;
                if requests.len() == 2 {
                    shared_block_requests = Some(requests.clone());
                }
            }
        }
        assert_eq!(full_block_count, 1);
        assert!(shared_block_requests.is_some());
        let shared_requests = shared_block_requests.unwrap();
        assert!(shared_requests.contains(&"0".to_string()));
        assert!(shared_requests.contains(&"1".to_string()));

        // Step 3: Free request 1
        manager.free(&"1".to_string());
        assert_eq!(manager.active_blocks(), 2);

        // Step 4: Free request 0
        manager.free(&"0".to_string());
        assert_eq!(manager.active_blocks(), 0);
        assert_eq!(manager.unique_blocks.len(), 0);
        assert_eq!(manager.partial_blocks.len(), 0);
        assert_eq!(manager.active_seqs.len(), 0);
    }

    #[test]
    #[should_panic(expected = "Cannot free non-existent request 0")]
    fn test_double_free_panic() {
        let mut manager = SharedSequenceManager::new(4);
        manager.add_request("0".to_string(), vec![0, 1, 2]);
        manager.free(&"0".to_string());
        // This should panic
        manager.free(&"0".to_string());
    }
}
