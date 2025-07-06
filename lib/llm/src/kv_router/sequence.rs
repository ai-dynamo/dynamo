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

use crate::kv_router::indexer::WorkerId;
use crate::mocker::protocols::UniqueBlock;
use crate::tokens::TokenBlockSequence;
use derive_getters::Getters;
use std::collections::{HashMap, HashSet};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;
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
pub struct ActiveSequences {
    active_seqs: HashMap<RequestId, TokenBlockSequence>,

    partial_blocks: HashMap<RequestId, UniqueBlock>,

    unique_blocks: HashMap<UniqueBlock, HashSet<RequestId>>,

    #[getter(copy)]
    block_size: usize,

    #[getter(copy)]
    active_blocks: usize,
}

impl ActiveSequences {
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
    pub fn add_request(
        &mut self,
        request_id: RequestId,
        token_sequence: TokenBlockSequence,
    ) -> usize {
        let blocks = create_unique_blocks_from_sequence(&token_sequence, None, self.block_size);

        for block in &blocks {
            self.add_block(request_id.clone(), block);
        }

        self.active_seqs.insert(request_id.clone(), token_sequence);

        self.active_blocks
    }

    /// Match a request against existing blocks and return the number of new blocks that would be added
    pub fn new_blocks(&self, token_sequence: &TokenBlockSequence) -> usize {
        let blocks = create_unique_blocks_from_sequence(token_sequence, None, self.block_size);

        blocks
            .iter()
            .filter(|block| !self.unique_blocks.contains_key(block))
            .count()
    }

    /// Return the total number of blocks that would be used if the token sequence was added
    /// This is the sum of new blocks that would be added plus the current active blocks
    pub fn potential_blocks(&self, token_sequence: &TokenBlockSequence) -> usize {
        self.new_blocks(token_sequence) + self.active_blocks
    }

    /// Free all blocks associated with a request
    pub fn free(&mut self, request_id: &RequestId) -> usize {
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

        self.active_blocks
    }

    /// Push a token to a specific request's sequence
    pub fn push(&mut self, request_id: &RequestId, token: u32) -> usize {
        let token_seq = self
            .active_seqs
            .get_mut(request_id)
            .expect("Request ID not found for token push");
        token_seq.append(token).expect("Token push failed.");

        // No need to update anything
        if token_seq.total_tokens() % self.block_size != 1 {
            return self.active_blocks;
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

        self.active_blocks
    }
}

#[derive(Debug)]
enum UpdateSequences {
    AddRequest {
        request_id: RequestId,
        token_sequence: TokenBlockSequence,
    },
    Free {
        request_id: RequestId,
    },
    Push {
        request_id: RequestId,
        token: u32,
    },
    NewBlocks {
        token_sequence: Arc<TokenBlockSequence>,
        resp_tx: mpsc::SyncSender<usize>,
    },
    PotentialBlocks {
        token_sequence: Arc<TokenBlockSequence>,
        resp_tx: mpsc::SyncSender<usize>,
    },
    ActiveBlocks {
        resp_tx: mpsc::SyncSender<usize>,
    },
    Shutdown,
}

/// Multi-worker wrapper around ActiveSequences that distributes requests across multiple threads
pub struct ActiveSequencesMultiWorker {
    senders: HashMap<WorkerId, mpsc::Sender<UpdateSequences>>,
    request_to_worker: HashMap<RequestId, WorkerId>,
    handles: HashMap<WorkerId, thread::JoinHandle<()>>,
    block_size: usize,
}

impl ActiveSequencesMultiWorker {
    pub fn new(block_size: usize, worker_ids: Vec<WorkerId>) -> Self {
        assert!(block_size > 1, "block_size must be greater than 1");

        let mut senders = HashMap::new();
        let mut handles = HashMap::new();

        for worker_id in worker_ids {
            let (sender, handle) = Self::start_worker(block_size);
            senders.insert(worker_id, sender);
            handles.insert(worker_id, handle);
        }

        Self {
            senders,
            request_to_worker: HashMap::new(),
            handles,
            block_size,
        }
    }

    /// Helper method to start a worker thread
    fn start_worker(block_size: usize) -> (mpsc::Sender<UpdateSequences>, thread::JoinHandle<()>) {
        let (request_tx, request_rx) = mpsc::channel::<UpdateSequences>();

        let handle = thread::spawn(move || {
            let mut active_sequences = ActiveSequences::new(block_size);

            while let Ok(command) = request_rx.recv() {
                match command {
                    UpdateSequences::AddRequest {
                        request_id,
                        token_sequence,
                    } => {
                        active_sequences.add_request(request_id, token_sequence);
                    }
                    UpdateSequences::Free { request_id } => {
                        active_sequences.free(&request_id);
                    }
                    UpdateSequences::Push { request_id, token } => {
                        active_sequences.push(&request_id, token);
                    }
                    UpdateSequences::NewBlocks {
                        token_sequence,
                        resp_tx,
                    } => {
                        let new_blocks = active_sequences.new_blocks(&token_sequence);
                        let _ = resp_tx.send(new_blocks);
                    }
                    UpdateSequences::PotentialBlocks {
                        token_sequence,
                        resp_tx,
                    } => {
                        let potential_blocks = active_sequences.potential_blocks(&token_sequence);
                        let _ = resp_tx.send(potential_blocks);
                    }
                    UpdateSequences::ActiveBlocks { resp_tx } => {
                        let active_blocks = active_sequences.active_blocks();
                        let _ = resp_tx.send(active_blocks);
                    }
                    UpdateSequences::Shutdown => {
                        break;
                    }
                }
            }
        });

        (request_tx, handle)
    }

    /// Update the set of workers, adding and removing as needed
    pub fn update_workers(&mut self, new_worker_ids: Vec<WorkerId>) -> HashMap<WorkerId, usize> {
        let current_workers: HashSet<WorkerId> = self.senders.keys().copied().collect();
        let new_workers: HashSet<WorkerId> = new_worker_ids.into_iter().collect();

        let workers_to_remove: Vec<WorkerId> =
            current_workers.difference(&new_workers).copied().collect();
        let workers_to_add: Vec<WorkerId> =
            new_workers.difference(&current_workers).copied().collect();

        // Remove workers
        for worker_id in &workers_to_remove {
            tracing::warn!("Removing worker {}", worker_id);

            // Send shutdown command to the worker
            if let Some(sender) = self.senders.remove(worker_id) {
                let _ = sender.send(UpdateSequences::Shutdown);
            }
            if let Some(handle) = self.handles.remove(worker_id) {
                let _ = handle.join();
            }
        }

        // Add new workers
        for worker_id in &workers_to_add {
            tracing::warn!("Adding worker {}", worker_id);

            let (sender, handle) = Self::start_worker(self.block_size);
            self.senders.insert(*worker_id, sender);
            self.handles.insert(*worker_id, handle);
        }

        // Return active blocks for all workers
        self.active_blocks()
    }

    pub fn add_request(
        &mut self,
        request_id: RequestId,
        token_sequence: TokenBlockSequence,
        worker_id: WorkerId,
    ) {
        if !self.senders.contains_key(&worker_id) {
            panic!("Worker ID {worker_id} not found");
        }

        self.request_to_worker.insert(request_id.clone(), worker_id);

        self.senders[&worker_id]
            .send(UpdateSequences::AddRequest {
                request_id,
                token_sequence,
            })
            .expect("Failed to send add_request command to worker");
    }

    pub fn free(&mut self, request_id: &RequestId) {
        let worker_id = self
            .request_to_worker
            .get(request_id)
            .copied()
            .expect("Request ID not found in request_to_worker mapping");

        self.senders[&worker_id]
            .send(UpdateSequences::Free {
                request_id: request_id.clone(),
            })
            .expect("Failed to send free command to worker");

        self.request_to_worker.remove(request_id);
    }

    pub fn push(&mut self, request_id: &RequestId, token: u32) {
        let worker_id = self
            .request_to_worker
            .get(request_id)
            .copied()
            .expect("Request ID not found in request_to_worker mapping");
        self.senders[&worker_id]
            .send(UpdateSequences::Push {
                request_id: request_id.clone(),
                token,
            })
            .expect("Failed to send push command to worker");
    }

    /// Get the number of workers
    pub fn num_workers(&self) -> usize {
        self.senders.len()
    }

    /// Generic method to query all workers with a given command
    fn query_workers(
        &self,
        token_sequence: Option<TokenBlockSequence>,
        command_fn: impl Fn(Option<Arc<TokenBlockSequence>>, mpsc::SyncSender<usize>) -> UpdateSequences,
    ) -> HashMap<WorkerId, usize> {
        let mut results = HashMap::new();
        let token_sequence_shared = token_sequence.map(Arc::new);
        let mut receivers = Vec::new();

        // Send queries to all workers in parallel
        for (worker_id, sender) in &self.senders {
            let (resp_tx, resp_rx) = mpsc::sync_channel(0);
            receivers.push((worker_id, resp_rx));
            sender
                .send(command_fn(token_sequence_shared.clone(), resp_tx))
                .expect("Failed to send command to worker");
        }

        // Collect results from all workers
        for (worker_id, receiver) in receivers {
            let result = receiver
                .recv_timeout(Duration::from_secs(1))
                .expect("Failed to receive response from worker");
            results.insert(*worker_id, result);
        }

        results
    }

    /// Query all workers for the number of new blocks that would be added by a token sequence
    pub fn new_blocks(&self, token_sequence: TokenBlockSequence) -> HashMap<WorkerId, usize> {
        self.query_workers(Some(token_sequence), |ts, resp_tx| match ts {
            Some(ts) => UpdateSequences::NewBlocks {
                token_sequence: ts,
                resp_tx,
            },
            None => unreachable!("token_sequence should always be Some for new_blocks"),
        })
    }

    /// Query all workers for the total number of blocks (new + active) that would be used by a token sequence
    pub fn potential_blocks(&self, token_sequence: TokenBlockSequence) -> HashMap<WorkerId, usize> {
        self.query_workers(Some(token_sequence), |ts, resp_tx| match ts {
            Some(ts) => UpdateSequences::PotentialBlocks {
                token_sequence: ts,
                resp_tx,
            },
            None => unreachable!("token_sequence should always be Some for potential_blocks"),
        })
    }

    /// Query all workers for their current number of active blocks
    pub fn active_blocks(&self) -> HashMap<WorkerId, usize> {
        self.query_workers(None, |_, resp_tx| UpdateSequences::ActiveBlocks { resp_tx })
    }
}

impl Drop for ActiveSequencesMultiWorker {
    fn drop(&mut self) {
        // Send shutdown command to all workers
        for sender in self.senders.values() {
            let _ = sender.send(UpdateSequences::Shutdown);
        }

        // Wait for all threads to finish
        for (_, handle) in self.handles.drain() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::Tokens;

    #[test]
    fn test_shared_sequence_manager_operations() {
        let block_size = 4;
        let mut manager = ActiveSequences::new(block_size);
        let to_sequence =
            |tokens: Vec<u32>| Tokens::from(tokens).into_sequence(block_size as u32, None);

        // Step 1: Add request 0 with tokens [0, 1, 2], then push 3 and 4
        manager.add_request("0".to_string(), to_sequence(vec![0, 1, 2]));
        manager.push(&"0".to_string(), 3);
        manager.push(&"0".to_string(), 4);

        assert_eq!(manager.active_blocks(), 2);
        assert_eq!(manager.partial_blocks.len(), 1);

        // Step 2: Add request 1 with tokens [0, 1, 2, 3, 4, 5, 6]
        manager.add_request("1".to_string(), to_sequence(vec![0, 1, 2, 3, 4, 5, 6]));
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
        assert!(shared_requests.contains("0"));
        assert!(shared_requests.contains("1"));

        let new_blocks = manager.new_blocks(&to_sequence(vec![0, 1, 2, 3, 4, 5]));
        assert_eq!(new_blocks, 1);

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
        let block_size = 4;
        let mut manager = ActiveSequences::new(block_size);
        let to_sequence =
            |tokens: Vec<u32>| Tokens::from(tokens).into_sequence(block_size as u32, None);

        manager.add_request("0".to_string(), to_sequence(vec![0, 1, 2]));
        manager.free(&"0".to_string());
        // This should panic
        manager.free(&"0".to_string());
    }

    #[test]
    fn test_active_sequences_multi_worker() {
        let block_size = 4;
        let worker_ids = vec![0, 1, 2];
        let mut manager = ActiveSequencesMultiWorker::new(block_size, worker_ids);
        let to_sequence =
            |tokens: Vec<u32>| Tokens::from(tokens).into_sequence(block_size as u32, None);

        // Send request [0, 1, 2, 3] to worker 0
        manager.add_request("req0".to_string(), to_sequence(vec![0, 1, 2, 3]), 0);

        // Send request [0, 1, 2] to worker 1, then push 3 and push 4
        manager.add_request("req1".to_string(), to_sequence(vec![0, 1, 2]), 1);
        manager.push(&"req1".to_string(), 3);
        manager.push(&"req1".to_string(), 4);

        // Send request [0, 1, 2] to worker 2
        manager.add_request("req2".to_string(), to_sequence(vec![0, 1, 2]), 2);

        // Check new_blocks on tokens [0, 1, 2, 3, 4]
        let new_blocks_map = manager.new_blocks(to_sequence(vec![0, 1, 2, 3, 4]));

        assert_eq!(new_blocks_map[&0], 1); // Worker 0 would have 1 new block
        assert_eq!(new_blocks_map[&1], 1); // Worker 1 would have 1 new block
        assert_eq!(new_blocks_map[&2], 2); // Worker 2 would have 2 new blocks

        manager.update_workers(vec![0, 1]);
        manager.update_workers(vec![0, 1, 3]);

        let new_blocks_map = manager.new_blocks(to_sequence(vec![0, 1, 2, 3, 4]));

        assert_eq!(new_blocks_map.len(), 3);
        assert_eq!(new_blocks_map[&3], 2);
    }
}
