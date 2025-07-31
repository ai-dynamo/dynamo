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

//! KV Cache Sequence Management for LLM Inference
//!
//! This module provides efficient management of token sequences and their associated KV cache blocks
//! for distributed LLM inference. It implements a shared block system where multiple requests can
//! reuse the same KV cache blocks for common token prefixes, significantly reducing memory usage.
//!
//! # Key Components
//!
//! - [`ActiveSequences`]: Single-threaded sequence manager that tracks active requests and their
//!   token sequences, managing shared KV cache blocks efficiently.
//!
//! - [`ActiveSequencesMultiWorker`]: Multi-threaded extension that distributes sequence management
//!   across multiple worker threads, enabling parallel processing of requests while maintaining
//!   consistency.
//!
//! # Architecture
//!
//! The system uses a block-based approach where token sequences are divided into fixed-size blocks.
//! Each block is identified by a hash of its contents, allowing for deduplication when multiple
//! requests share common prefixes (e.g., system prompts, few-shot examples).

use crate::kv_router::indexer::OverlapScores;
use crate::kv_router::indexer::WorkerId;
use crate::tokens::SequenceHash;
use derive_getters::Getters;
use std::collections::{HashMap, HashSet};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

// TODO: use the common request_id if it exists in the repo
pub type RequestId = String;

/// A multi-request sequence manager that handles multiple active sequences with shared KV cache
#[derive(Debug, Getters)]
pub struct ActiveSequences {
    active_seqs: HashMap<RequestId, Vec<SequenceHash>>,

    prefill_tokens: HashMap<RequestId, usize>,

    unique_blocks: HashMap<SequenceHash, HashSet<RequestId>>,

    #[getter(copy)]
    block_size: usize,

    #[getter(copy)]
    active_blocks: usize,

    #[getter(copy)]
    active_tokens: usize,
}

impl ActiveSequences {
    /// Create a new SharedSequenceManager instance
    pub fn new(block_size: usize) -> Self {
        // TODO: make this not a hard req
        assert!(block_size > 1, "block_size must be greater than 1");

        Self {
            active_seqs: HashMap::new(),
            prefill_tokens: HashMap::new(),
            unique_blocks: HashMap::new(),
            block_size,
            active_blocks: 0,
            active_tokens: 0,
        }
    }

    fn add_block(&mut self, request_id: RequestId, block: &SequenceHash) {
        let is_new_block = !self.unique_blocks.contains_key(block);

        self.unique_blocks
            .entry(*block)
            .or_default()
            .insert(request_id.clone());

        if is_new_block {
            self.active_blocks += 1;
        }
    }

    fn remove_block(&mut self, request_id: &RequestId, block: &SequenceHash) {
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
        token_sequence: Vec<SequenceHash>,
        isl: usize,
        overlap: u32,
    ) -> usize {
        let prefill_tokens = self.new_tokens(isl, overlap);
        self.prefill_tokens
            .insert(request_id.clone(), prefill_tokens);
        self.active_tokens += prefill_tokens;

        for block in &token_sequence {
            self.add_block(request_id.clone(), block);
        }

        self.active_seqs.insert(request_id.clone(), token_sequence);

        self.active_blocks
    }

    /// Mark prefill as completed for a request, removing it from prefill_tokens tracking
    pub fn mark_prefill_completed(&mut self, request_id: &RequestId) {
        if let Some(tokens) = self.prefill_tokens.remove(request_id) {
            self.active_tokens = self
                .active_tokens
                .checked_sub(tokens.saturating_sub(1)) // Keep 1 token for decoding
                .expect("active_tokens underflow");
        }
    }

    pub fn new_tokens(&self, isl: usize, overlap: u32) -> usize {
        isl.checked_sub((overlap as usize) * self.block_size)
            .unwrap_or_else(|| panic!("prefill_tokens < 0 with overlap {overlap} and ISL {isl}"))
    }

    pub fn potential_blocks_and_tokens(
        &self,
        token_sequence: &[SequenceHash],
        isl: usize,
        overlap: u32,
    ) -> (usize, usize) {
        let potential_blocks = self.new_blocks(token_sequence) + self.active_blocks;
        let potential_tokens = self.new_tokens(isl, overlap) + self.active_tokens;
        (potential_blocks, potential_tokens)
    }

    /// Match a request against existing blocks and return the number of new blocks that would be added
    pub fn new_blocks(&self, token_sequence: &[SequenceHash]) -> usize {
        token_sequence
            .iter()
            .filter(|block| !self.unique_blocks.contains_key(block))
            .count()
    }

    /// Return the total number of blocks that would be used if the token sequence was added
    /// This is the sum of new blocks that would be added plus the current active blocks
    pub fn potential_blocks(&self, token_sequence: &[SequenceHash]) -> usize {
        self.new_blocks(token_sequence) + self.active_blocks
    }

    /// Free all blocks associated with a request
    pub fn free(&mut self, request_id: &RequestId) -> usize {
        // decoding has one active token
        self.active_tokens = self
            .active_tokens
            .checked_sub(self.prefill_tokens.remove(request_id).unwrap_or(1))
            .expect("active_tokens < 0");

        let Some(token_seq) = self.active_seqs.get(request_id) else {
            tracing::warn!("Trying to free free non-existent request {request_id}");
            return 0;
        };

        for block in token_seq.clone() {
            self.remove_block(request_id, &block)
        }

        self.active_seqs.remove(request_id).unwrap();

        self.active_blocks
    }
}

enum UpdateSequences {
    AddRequest {
        request_id: RequestId,
        token_sequence: Vec<SequenceHash>,
        isl: usize,
        overlap: u32,
    },
    Free {
        request_id: RequestId,
    },
    MarkPrefillCompleted {
        request_id: RequestId,
    },
    NewBlocks {
        token_sequence: Arc<Vec<SequenceHash>>,
        resp_tx: mpsc::SyncSender<usize>,
    },
    PotentialBlocks {
        token_sequence: Arc<Vec<SequenceHash>>,
        resp_tx: mpsc::SyncSender<usize>,
    },
    PotentialBlocksAndTokens {
        token_sequence: Arc<Vec<SequenceHash>>,
        isl: usize,
        overlap: u32,
        resp_tx: mpsc::SyncSender<(usize, usize)>,
    },
    ActiveBlocks {
        resp_tx: mpsc::SyncSender<usize>,
    },
    ActiveTokens {
        resp_tx: mpsc::SyncSender<usize>,
    },
    Shutdown,
}

/// Multi-worker extension of ActiveSequences that distributes requests across multiple threads
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
                        isl,
                        overlap,
                    } => {
                        active_sequences.add_request(request_id, token_sequence, isl, overlap);
                    }
                    UpdateSequences::Free { request_id } => {
                        active_sequences.free(&request_id);
                    }
                    UpdateSequences::MarkPrefillCompleted { request_id } => {
                        active_sequences.mark_prefill_completed(&request_id);
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
                    UpdateSequences::PotentialBlocksAndTokens {
                        token_sequence,
                        isl,
                        overlap,
                        resp_tx,
                    } => {
                        let potential_tokens = active_sequences.potential_blocks_and_tokens(
                            &token_sequence,
                            isl,
                            overlap,
                        );
                        let _ = resp_tx.send(potential_tokens);
                    }
                    UpdateSequences::ActiveBlocks { resp_tx } => {
                        let active_blocks = active_sequences.active_blocks();
                        let _ = resp_tx.send(active_blocks);
                    }
                    UpdateSequences::ActiveTokens { resp_tx } => {
                        let active_tokens = active_sequences.active_tokens();
                        let _ = resp_tx.send(active_tokens);
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
        token_sequence: Vec<SequenceHash>,
        isl: usize,
        overlap: u32,
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
                isl,
                overlap,
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

    /// Mark prefill as completed for a request
    pub fn mark_prefill_completed(&mut self, request_id: &RequestId) {
        let worker_id = self
            .request_to_worker
            .get(request_id)
            .copied()
            .expect("Request ID not found in request_to_worker mapping");

        self.senders[&worker_id]
            .send(UpdateSequences::MarkPrefillCompleted {
                request_id: request_id.clone(),
            })
            .expect("Failed to send mark_prefill_completed command to worker");
    }

    /// Get the number of workers
    pub fn num_workers(&self) -> usize {
        self.senders.len()
    }

    /// Generic method to query all workers with a given command
    fn query_workers(
        &self,
        token_sequence: Option<Vec<SequenceHash>>,
        command_fn: impl Fn(Option<Arc<Vec<SequenceHash>>>, mpsc::SyncSender<usize>) -> UpdateSequences,
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
    pub fn new_blocks(&self, token_sequence: Vec<SequenceHash>) -> HashMap<WorkerId, usize> {
        self.query_workers(Some(token_sequence), |ts, resp_tx| match ts {
            Some(ts) => UpdateSequences::NewBlocks {
                token_sequence: ts,
                resp_tx,
            },
            None => unreachable!("token_sequence should always be Some for new_blocks"),
        })
    }

    /// Query all workers for the total number of blocks (new + active) that would be used by a token sequence
    pub fn potential_blocks(&self, token_sequence: Vec<SequenceHash>) -> HashMap<WorkerId, usize> {
        self.query_workers(Some(token_sequence), |ts, resp_tx| match ts {
            Some(ts) => UpdateSequences::PotentialBlocks {
                token_sequence: ts,
                resp_tx,
            },
            None => unreachable!("token_sequence should always be Some for potential_blocks"),
        })
    }

    /// Query all workers for the potential tokens (new + active) that would be used by a token sequence with overlap
    pub fn potential_blocks_and_tokens(
        &self,
        token_sequence: Vec<SequenceHash>,
        isl: usize,
        overlaps: OverlapScores,
    ) -> (HashMap<WorkerId, usize>, HashMap<WorkerId, usize>) {
        let mut potential_blocks = HashMap::new();
        let mut potential_tokens = HashMap::new();
        let token_sequence_shared = Arc::new(token_sequence);
        let mut receivers = Vec::new();

        // Send queries to all workers in parallel
        for (worker_id, sender) in &self.senders {
            let (resp_tx, resp_rx) = mpsc::sync_channel(0);
            receivers.push((worker_id, resp_rx));

            sender
                .send(UpdateSequences::PotentialBlocksAndTokens {
                    token_sequence: token_sequence_shared.clone(),
                    isl,
                    overlap: overlaps.scores.get(worker_id).copied().unwrap_or(0),
                    resp_tx,
                })
                .expect("Failed to send potential_tokens command to worker");
        }

        // Collect results from all workers
        for (worker_id, receiver) in receivers {
            let (blocks, tokens) = receiver
                .recv_timeout(Duration::from_secs(1))
                .expect("Failed to receive response from worker");
            potential_blocks.insert(*worker_id, blocks);
            potential_tokens.insert(*worker_id, tokens);
        }

        (potential_blocks, potential_tokens)
    }

    /// Query all workers for their current number of active blocks
    pub fn active_blocks(&self) -> HashMap<WorkerId, usize> {
        self.query_workers(None, |_, resp_tx| UpdateSequences::ActiveBlocks { resp_tx })
    }

    /// Query all workers for their current number of active tokens
    pub fn active_tokens(&self) -> HashMap<WorkerId, usize> {
        self.query_workers(None, |_, resp_tx| UpdateSequences::ActiveTokens { resp_tx })
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

    #[test]
    fn test_multi_worker_block_sharing() {
        // Create multi-worker sequence manager with 3 workers
        let block_size = 4; // arbitrary block size
        let worker_ids = vec![0, 1, 2];
        let mut seq_manager = ActiveSequencesMultiWorker::new(block_size, worker_ids);

        // Add requests to each worker
        // Worker 0: sequence [0, 1, 2]
        seq_manager.add_request(
            "request_0".to_string(),
            vec![0, 1, 2],
            12, // ISL (3 blocks * 4 block_size)
            0,  // no overlap
            0,  // worker_id
        );

        // Worker 1: sequence [3, 4]
        seq_manager.add_request(
            "request_1".to_string(),
            vec![3, 4],
            8, // ISL (2 blocks * 4 block_size)
            0, // no overlap
            1, // worker_id
        );

        // Worker 2: sequence [0, 1, 2, 3]
        seq_manager.add_request(
            "request_2".to_string(),
            vec![0, 1, 2, 3],
            16, // ISL (4 blocks * 4 block_size)
            0,  // no overlap
            2,  // worker_id
        );

        // Verify active tokens after adding requests
        let tokens_after_add = seq_manager.active_tokens();
        assert_eq!(
            tokens_after_add[&0], 12,
            "Worker 0 should have 12 active tokens"
        );
        assert_eq!(
            tokens_after_add[&1], 8,
            "Worker 1 should have 8 active tokens"
        );
        assert_eq!(
            tokens_after_add[&2], 16,
            "Worker 2 should have 16 active tokens"
        );

        // Test potential blocks for sequence [0, 1]
        let potential_blocks = seq_manager.potential_blocks(vec![0, 1]);

        // Worker 0 should return 3 (already has blocks 0, 1, 2, so no new blocks needed for [0, 1])
        assert_eq!(
            potential_blocks[&0], 3,
            "Worker 0 should have 3 potential blocks"
        );

        // Worker 1 should return 4 (has blocks 3, 4, would need to add blocks 0, 1)
        assert_eq!(
            potential_blocks[&1], 4,
            "Worker 1 should have 4 potential blocks"
        );

        // Worker 2 should return 4 (already has blocks 0, 1, 2, 3, so no new blocks needed for [0, 1])
        assert_eq!(
            potential_blocks[&2], 4,
            "Worker 2 should have 4 potential blocks"
        );

        // Free all original requests
        seq_manager.free(&"request_0".to_string());
        seq_manager.free(&"request_1".to_string());
        seq_manager.free(&"request_2".to_string());

        // Verify active blocks are zero for all workers
        let active_blocks = seq_manager.active_blocks();
        assert_eq!(active_blocks[&0], 0, "Worker 0 should have 0 active blocks");
        assert_eq!(active_blocks[&1], 0, "Worker 1 should have 0 active blocks");
        assert_eq!(active_blocks[&2], 0, "Worker 2 should have 0 active blocks");

        // Verify active tokens are zero for all workers
        let final_tokens = seq_manager.active_tokens();
        assert_eq!(
            final_tokens[&0], 0,
            "Worker 0 should have 0 active tokens after freeing all"
        );
        assert_eq!(
            final_tokens[&1], 0,
            "Worker 1 should have 0 active tokens after freeing all"
        );
        assert_eq!(
            final_tokens[&2], 0,
            "Worker 2 should have 0 active tokens after freeing all"
        );
    }
}
