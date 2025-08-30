// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
use anyhow::Result;
use dashmap::DashMap;
use derive_getters::Getters;
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::traits::events::{EventPublisher, EventSubscriber};
use futures::StreamExt;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

use super::protocols::{ActiveSequenceEvent, ActiveSequenceEventData};
use crate::kv_router::ACTIVE_SEQUENCES_SUBJECT;
use dynamo_runtime::CancellationToken;

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
                .checked_sub(tokens)
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
        self.mark_prefill_completed(request_id);

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
        resp_tx: tokio::sync::oneshot::Sender<usize>,
    },
    PotentialBlocks {
        token_sequence: Arc<Vec<SequenceHash>>,
        resp_tx: tokio::sync::oneshot::Sender<usize>,
    },
    PotentialBlocksAndTokens {
        token_sequence: Arc<Vec<SequenceHash>>,
        isl: usize,
        overlap: u32,
        resp_tx: tokio::sync::oneshot::Sender<(usize, usize)>,
    },
    ActiveBlocks {
        resp_tx: tokio::sync::oneshot::Sender<usize>,
    },
    ActiveTokens {
        resp_tx: tokio::sync::oneshot::Sender<usize>,
    },
    Shutdown,
}

/// Multi-worker extension of ActiveSequences that distributes requests across multiple threads
pub struct ActiveSequencesMultiWorker {
    senders: Arc<DashMap<WorkerId, tokio::sync::mpsc::UnboundedSender<UpdateSequences>>>,
    request_to_worker: Arc<DashMap<RequestId, WorkerId>>,
    handles: Arc<DashMap<WorkerId, tokio::task::JoinHandle<()>>>,
    block_size: usize,
    component: Component,
    router_id: Uuid,
    replica_sync: bool,
}

impl ActiveSequencesMultiWorker {
    pub fn new(
        component: Component,
        block_size: usize,
        worker_ids: Vec<WorkerId>,
        replica_sync: bool,
    ) -> Self {
        assert!(block_size > 1, "block_size must be greater than 1");

        let senders = Arc::new(DashMap::new());
        let handles = Arc::new(DashMap::new());
        let request_to_worker = Arc::new(DashMap::new());
        let router_id = Uuid::new_v4();

        for worker_id in worker_ids {
            // Create a child cancellation token from the component's runtime
            let cancel_token = component.drt().runtime().child_token();
            let (sender, handle) = Self::start_worker(block_size, cancel_token);
            senders.insert(worker_id, sender);
            handles.insert(worker_id, handle);
        }

        let multi_worker = Self {
            senders: senders.clone(),
            request_to_worker: request_to_worker.clone(),
            handles,
            block_size,
            component: component.clone(),
            router_id,
            replica_sync,
        };

        // Start the subscription loop only if replica_sync is enabled
        if replica_sync {
            let senders_clone = senders.clone();
            let request_to_worker_clone = request_to_worker.clone();
            let component_clone = component.clone();
            let router_id_clone = router_id;

            component.drt().runtime().secondary().spawn(async move {
                // NATS subscription loop
                if let Err(e) = Self::subscribe_to_events(
                    senders_clone,
                    request_to_worker_clone,
                    component_clone,
                    router_id_clone,
                )
                .await
                {
                    tracing::error!("Error in active sequences events subscription: {}", e);
                }
            });
        }

        multi_worker
    }

    /// Helper method to start a worker task
    fn start_worker(
        block_size: usize,
        cancel_token: CancellationToken, // Add cancellation token parameter
    ) -> (
        tokio::sync::mpsc::UnboundedSender<UpdateSequences>,
        tokio::task::JoinHandle<()>,
    ) {
        let (request_tx, mut request_rx) = tokio::sync::mpsc::unbounded_channel();

        let handle = tokio::spawn(async move {
            let mut active_sequences = ActiveSequences::new(block_size);

            loop {
                tokio::select! {
                    // Handle incoming commands
                    command = request_rx.recv() => {
                        match command {
                            Some(command) => {
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
                            None => {
                                // Channel closed, exit
                                break;
                            }
                        }
                    }
                    // Handle cancellation
                    _ = cancel_token.cancelled() => {
                        tracing::debug!("Worker task cancelled");
                        break;
                    }
                }
            }
        });

        (request_tx, handle)
    }

    /// Background task to subscribe to active sequence events and update all workers
    async fn subscribe_to_events(
        senders: Arc<DashMap<WorkerId, tokio::sync::mpsc::UnboundedSender<UpdateSequences>>>,
        request_to_worker: Arc<DashMap<RequestId, WorkerId>>,
        component: Component,
        router_id: Uuid,
    ) -> Result<()> {
        let mut subscriber = component
            .subscribe_with_type::<ActiveSequenceEvent>(ACTIVE_SEQUENCES_SUBJECT)
            .await?;

        while let Some(result) = subscriber.next().await {
            let Ok(event) = result else {
                tracing::error!(
                    "Error receiving active sequence event: {}",
                    result.unwrap_err()
                );
                continue;
            };

            // Skip events emitted by itself
            if event.router_id == router_id {
                continue;
            }

            match &event.data {
                ActiveSequenceEventData::AddRequest {
                    token_sequence,
                    isl,
                    overlap,
                } => {
                    request_to_worker.insert(event.request_id.clone(), event.worker_id);

                    if let Some(sender) = senders.get(&event.worker_id) {
                        let _ = sender.send(UpdateSequences::AddRequest {
                            request_id: event.request_id.clone(),
                            token_sequence: token_sequence.clone(),
                            isl: *isl,
                            overlap: *overlap,
                        });
                    } else {
                        tracing::warn!(
                            "Worker {} not found, cannot process AddRequest",
                            event.worker_id
                        );
                    }
                }
                ActiveSequenceEventData::Free => {
                    if let Some((_, worker_id)) = request_to_worker.remove(&event.request_id)
                        && let Some(sender) = senders.get(&worker_id)
                    {
                        let _ = sender.send(UpdateSequences::Free {
                            request_id: event.request_id.clone(),
                        });
                    }
                }
                ActiveSequenceEventData::MarkPrefillCompleted => {
                    if let Some(worker_id) = request_to_worker.get(&event.request_id)
                        && let Some(sender) = senders.get(&*worker_id)
                    {
                        let _ = sender.send(UpdateSequences::MarkPrefillCompleted {
                            request_id: event.request_id.clone(),
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Update the set of workers, adding and removing as needed
    pub fn update_workers(&self, new_worker_ids: Vec<WorkerId>) {
        let current_workers: HashSet<WorkerId> =
            self.senders.iter().map(|entry| *entry.key()).collect();
        let new_workers: HashSet<WorkerId> = new_worker_ids.into_iter().collect();

        let workers_to_remove: Vec<WorkerId> =
            current_workers.difference(&new_workers).copied().collect();
        let workers_to_add: Vec<WorkerId> =
            new_workers.difference(&current_workers).copied().collect();

        // Remove workers
        for worker_id in &workers_to_remove {
            tracing::warn!("Removing worker {}", worker_id);

            // Send shutdown command to the worker
            if let Some((_, sender)) = self.senders.remove(worker_id) {
                let _ = sender.send(UpdateSequences::Shutdown);
            }
            if let Some((_, handle)) = self.handles.remove(worker_id) {
                handle.abort();
            }
        }

        // Add new workers
        for worker_id in &workers_to_add {
            tracing::warn!("Adding worker {}", worker_id);

            let (sender, handle) = Self::start_worker(
                self.block_size,
                self.component.drt().runtime().child_token(),
            );
            self.senders.insert(*worker_id, sender);
            self.handles.insert(*worker_id, handle);
        }
    }

    pub async fn add_request(
        &self,
        request_id: RequestId,
        token_sequence: Vec<SequenceHash>,
        isl: usize,
        overlap: u32,
        worker_id: WorkerId,
    ) -> Result<()> {
        if !self.senders.contains_key(&worker_id) {
            return Err(anyhow::anyhow!("Worker ID {worker_id} not found"));
        }

        // Publish event only if replica_sync is enabled
        if self.replica_sync {
            let event = ActiveSequenceEvent {
                request_id: request_id.clone(),
                worker_id,
                data: ActiveSequenceEventData::AddRequest {
                    token_sequence: token_sequence.clone(),
                    isl,
                    overlap,
                },
                router_id: self.router_id,
            };
            self.component
                .publish(ACTIVE_SEQUENCES_SUBJECT, &event)
                .await?;
        }

        // Update local state
        self.request_to_worker.insert(request_id.clone(), worker_id);

        self.senders
            .get(&worker_id)
            .unwrap()
            .send(UpdateSequences::AddRequest {
                request_id,
                token_sequence,
                isl,
                overlap,
            })
            .map_err(|_| anyhow::anyhow!("Failed to send add_request command to worker"))?;

        Ok(())
    }

    pub async fn free(&self, request_id: &RequestId) -> Result<()> {
        let worker_id = self
            .request_to_worker
            .get(request_id)
            .map(|entry| *entry)
            .ok_or_else(|| anyhow::anyhow!("Request ID not found in request_to_worker mapping"))?;

        // Publish event only if replica_sync is enabled
        if self.replica_sync {
            let event = ActiveSequenceEvent {
                request_id: request_id.clone(),
                worker_id,
                data: ActiveSequenceEventData::Free,
                router_id: self.router_id,
            };
            self.component
                .publish(ACTIVE_SEQUENCES_SUBJECT, &event)
                .await?;
        }

        // Update local state
        self.senders
            .get(&worker_id)
            .unwrap()
            .send(UpdateSequences::Free {
                request_id: request_id.clone(),
            })
            .map_err(|_| anyhow::anyhow!("Failed to send free command to worker"))?;

        self.request_to_worker.remove(request_id);

        Ok(())
    }

    /// Mark prefill as completed for a request
    pub async fn mark_prefill_completed(&self, request_id: &RequestId) -> Result<()> {
        let worker_id = self
            .request_to_worker
            .get(request_id)
            .map(|entry| *entry)
            .ok_or_else(|| anyhow::anyhow!("Request ID not found in request_to_worker mapping"))?;

        // Publish event only if replica_sync is enabled
        if self.replica_sync {
            let event = ActiveSequenceEvent {
                request_id: request_id.clone(),
                worker_id,
                data: ActiveSequenceEventData::MarkPrefillCompleted,
                router_id: self.router_id,
            };
            self.component
                .publish(ACTIVE_SEQUENCES_SUBJECT, &event)
                .await?;
        }

        // Update local state
        self.senders
            .get(&worker_id)
            .unwrap()
            .send(UpdateSequences::MarkPrefillCompleted {
                request_id: request_id.clone(),
            })
            .map_err(|_| {
                anyhow::anyhow!("Failed to send mark_prefill_completed command to worker")
            })?;

        Ok(())
    }

    /// Get the number of workers
    pub fn num_workers(&self) -> usize {
        self.senders.len()
    }

    /// Generic method to query all workers with a given command
    async fn query_workers<T: Send + 'static>(
        &self,
        token_sequence: Option<Vec<SequenceHash>>,
        command_fn: impl Fn(
            Option<Arc<Vec<SequenceHash>>>,
            tokio::sync::oneshot::Sender<T>,
        ) -> UpdateSequences,
    ) -> HashMap<WorkerId, T> {
        let mut results = HashMap::new();
        let token_sequence_shared = token_sequence.map(Arc::new);
        let mut receivers = Vec::new();

        // Send queries to all workers in parallel
        for entry in self.senders.iter() {
            let worker_id = *entry.key();
            let sender = entry.value();
            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            receivers.push((worker_id, resp_rx));
            if let Err(e) = sender.send(command_fn(token_sequence_shared.clone(), resp_tx)) {
                tracing::error!("Failed to send command to worker {}: {}", worker_id, e);
            }
        }

        // Collect results from all workers
        for (worker_id, receiver) in receivers {
            match tokio::time::timeout(tokio::time::Duration::from_secs(1), receiver).await {
                Ok(Ok(result)) => {
                    results.insert(worker_id, result);
                }
                Ok(Err(_)) => {
                    tracing::error!("Worker {} dropped response channel", worker_id);
                }
                Err(_) => {
                    tracing::error!("Timeout waiting for response from worker {}", worker_id);
                }
            }
        }

        results
    }

    /// Query all workers for the number of new blocks that would be added by a token sequence
    pub async fn new_blocks(&self, token_sequence: Vec<SequenceHash>) -> HashMap<WorkerId, usize> {
        self.query_workers(Some(token_sequence), |ts, resp_tx| match ts {
            Some(ts) => UpdateSequences::NewBlocks {
                token_sequence: ts,
                resp_tx,
            },
            None => unreachable!("token_sequence should always be Some for new_blocks"),
        })
        .await
    }

    /// Query all workers for the total number of blocks (new + active) that would be used by a token sequence
    pub async fn potential_blocks(
        &self,
        token_sequence: Vec<SequenceHash>,
    ) -> HashMap<WorkerId, usize> {
        self.query_workers(Some(token_sequence), |ts, resp_tx| match ts {
            Some(ts) => UpdateSequences::PotentialBlocks {
                token_sequence: ts,
                resp_tx,
            },
            None => unreachable!("token_sequence should always be Some for potential_blocks"),
        })
        .await
    }

    /// Query all workers for the potential tokens (new + active) that would be used by a token sequence with overlap
    pub async fn potential_blocks_and_tokens(
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
        for entry in self.senders.iter() {
            let worker_id = *entry.key();
            let sender = entry.value();
            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            receivers.push((worker_id, resp_rx));

            if let Err(e) = sender.send(UpdateSequences::PotentialBlocksAndTokens {
                token_sequence: token_sequence_shared.clone(),
                isl,
                overlap: overlaps.scores.get(&worker_id).copied().unwrap_or(0),
                resp_tx,
            }) {
                tracing::error!(
                    "Failed to send potential_tokens command to worker {}: {}",
                    worker_id,
                    e
                );
            }
        }

        // Collect results from all workers
        for (worker_id, receiver) in receivers {
            match tokio::time::timeout(tokio::time::Duration::from_secs(1), receiver).await {
                Ok(Ok((blocks, tokens))) => {
                    potential_blocks.insert(worker_id, blocks);
                    potential_tokens.insert(worker_id, tokens);
                }
                Ok(Err(_)) => {
                    tracing::error!("Worker {} dropped response channel", worker_id);
                }
                Err(_) => {
                    tracing::error!("Timeout waiting for response from worker {}", worker_id);
                }
            }
        }

        (potential_blocks, potential_tokens)
    }

    /// Query all workers for their current number of active blocks
    pub async fn active_blocks(&self) -> HashMap<WorkerId, usize> {
        self.query_workers(None, |_, resp_tx| UpdateSequences::ActiveBlocks { resp_tx })
            .await
    }

    /// Query all workers for their current number of active tokens
    pub async fn active_tokens(&self) -> HashMap<WorkerId, usize> {
        self.query_workers(None, |_, resp_tx| UpdateSequences::ActiveTokens { resp_tx })
            .await
    }
}

impl Drop for ActiveSequencesMultiWorker {
    fn drop(&mut self) {
        // Send shutdown to all workers
        for entry in self.senders.iter() {
            let _ = entry.value().send(UpdateSequences::Shutdown);
        }

        // Abort all tasks
        for entry in self.handles.iter() {
            entry.value().abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[test]
    #[ignore]
    fn test_multi_worker_block_sharing() -> Result<()> {
        // Initialize logging once
        dynamo_runtime::logging::init();

        let block_size = 4; // arbitrary block size

        // Shared state for collecting results from both threads
        let active_tokens_after_add = Arc::new(Mutex::new(HashMap::new()));
        let potential_blocks_result = Arc::new(Mutex::new(HashMap::new()));
        let active_blocks_after_free = Arc::new(Mutex::new(HashMap::new()));
        let active_tokens_after_free = Arc::new(Mutex::new(HashMap::new()));

        let active_tokens_after_add_clone = active_tokens_after_add.clone();
        let potential_blocks_result_clone = potential_blocks_result.clone();
        let active_blocks_after_free_clone = active_blocks_after_free.clone();
        let active_tokens_after_free_clone = active_tokens_after_free.clone();

        // Clone again for the second thread
        let active_tokens_after_add_clone2 = active_tokens_after_add.clone();
        let potential_blocks_result_clone2 = potential_blocks_result.clone();
        let active_blocks_after_free_clone2 = active_blocks_after_free.clone();
        let active_tokens_after_free_clone2 = active_tokens_after_free.clone();

        // Thread 1: First runtime with workers 0 and 1
        let handle1 = thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();

            rt.block_on(async {
                // Create runtime and distributed runtime
                let runtime = Runtime::from_current()?;
                let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

                // Create namespace and component with same names as thread 2
                let namespace = distributed.namespace("test_multiworker_sequences")?;
                let component = namespace
                    .component("sequences")?
                    .service_builder()
                    .create()
                    .await?;

                // Create multi-worker sequence manager with workers 0 and 1
                let worker_ids = vec![0, 1];
                let seq_manager =
                    ActiveSequencesMultiWorker::new(component, block_size, worker_ids, true);

                // Give some time for the subscription loop to start
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                // Add requests to workers
                // Worker 0: sequence [0, 1, 2]
                seq_manager
                    .add_request(
                        "request_0".to_string(),
                        vec![0, 1, 2],
                        12, // ISL (3 blocks * 4 block_size)
                        0,  // no overlap
                        0,  // worker_id
                    )
                    .await?;

                // Worker 1: sequence [3, 4]
                seq_manager
                    .add_request(
                        "request_1".to_string(),
                        vec![3, 4],
                        8, // ISL (2 blocks * 4 block_size)
                        0, // no overlap
                        1, // worker_id
                    )
                    .await?;

                // Give some time for the commands to be processed and synchronization
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                // Get active tokens from workers 0 and 1
                let tokens = seq_manager.active_tokens().await;
                active_tokens_after_add_clone
                    .lock()
                    .unwrap()
                    .insert(0, tokens.get(&0).copied().unwrap_or(0));
                active_tokens_after_add_clone
                    .lock()
                    .unwrap()
                    .insert(1, tokens.get(&1).copied().unwrap_or(0));

                // Test potential blocks for sequence [0, 1]
                let potential = seq_manager.potential_blocks(vec![0, 1]).await;
                potential_blocks_result_clone
                    .lock()
                    .unwrap()
                    .insert(0, potential.get(&0).copied().unwrap_or(0));
                potential_blocks_result_clone
                    .lock()
                    .unwrap()
                    .insert(1, potential.get(&1).copied().unwrap_or(0));

                // Wait for second thread to process its requests
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                // Free requests from workers 0 and 1
                seq_manager.free(&"request_0".to_string()).await?;
                seq_manager.free(&"request_1".to_string()).await?;

                // Give some time for the commands to be processed
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                // Get final active blocks and tokens
                let blocks = seq_manager.active_blocks().await;
                let tokens = seq_manager.active_tokens().await;

                active_blocks_after_free_clone
                    .lock()
                    .unwrap()
                    .insert(0, blocks.get(&0).copied().unwrap_or(0));
                active_blocks_after_free_clone
                    .lock()
                    .unwrap()
                    .insert(1, blocks.get(&1).copied().unwrap_or(0));
                active_tokens_after_free_clone
                    .lock()
                    .unwrap()
                    .insert(0, tokens.get(&0).copied().unwrap_or(0));
                active_tokens_after_free_clone
                    .lock()
                    .unwrap()
                    .insert(1, tokens.get(&1).copied().unwrap_or(0));

                // Keep runtime alive a bit longer for synchronization
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                // Shutdown runtime
                runtime.shutdown();

                Ok::<(), anyhow::Error>(())
            })
        });

        // Thread 2: Second runtime with worker 2
        let handle2 = thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();

            rt.block_on(async {
                // Create runtime and distributed runtime
                let runtime = Runtime::from_current()?;
                let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

                // Create namespace and component with same names as thread 1
                let namespace = distributed.namespace("test_multiworker_sequences")?;
                let component = namespace
                    .component("sequences")?
                    .service_builder()
                    .create()
                    .await?;

                // Create multi-worker sequence manager with worker 2
                let worker_ids = vec![2];
                let seq_manager =
                    ActiveSequencesMultiWorker::new(component, block_size, worker_ids, true);

                // Give some time for the subscription loop to start
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                // Wait a bit to ensure thread 1 has started
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                // Worker 2: sequence [0, 1, 2, 3]
                seq_manager
                    .add_request(
                        "request_2".to_string(),
                        vec![0, 1, 2, 3],
                        16, // ISL (4 blocks * 4 block_size)
                        0,  // no overlap
                        2,  // worker_id
                    )
                    .await?;

                // Give some time for the commands to be processed and synchronization
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                // Get active tokens from worker 2
                let tokens = seq_manager.active_tokens().await;
                active_tokens_after_add_clone2
                    .lock()
                    .unwrap()
                    .insert(2, tokens.get(&2).copied().unwrap_or(0));

                // Test potential blocks for sequence [0, 1]
                let potential = seq_manager.potential_blocks(vec![0, 1]).await;
                potential_blocks_result_clone2
                    .lock()
                    .unwrap()
                    .insert(2, potential.get(&2).copied().unwrap_or(0));

                // Wait for first thread to free its requests
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                // Free request from worker 2
                seq_manager.free(&"request_2".to_string()).await?;

                // Give some time for the commands to be processed
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                // Get final active blocks and tokens
                let blocks = seq_manager.active_blocks().await;
                let tokens = seq_manager.active_tokens().await;

                active_blocks_after_free_clone2
                    .lock()
                    .unwrap()
                    .insert(2, blocks.get(&2).copied().unwrap_or(0));
                active_tokens_after_free_clone2
                    .lock()
                    .unwrap()
                    .insert(2, tokens.get(&2).copied().unwrap_or(0));

                // Keep runtime alive a bit longer for synchronization
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                // Shutdown runtime
                runtime.shutdown();

                Ok::<(), anyhow::Error>(())
            })
        });

        // Wait for both threads to complete
        handle1.join().unwrap()?;
        handle2.join().unwrap()?;

        // Extract results
        let tokens_after_add = active_tokens_after_add.lock().unwrap();
        let potential_blocks = potential_blocks_result.lock().unwrap();
        let blocks_after_free = active_blocks_after_free.lock().unwrap();
        let tokens_after_free = active_tokens_after_free.lock().unwrap();

        // Verify active tokens after adding requests
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

        // Verify active blocks are zero for all workers
        assert_eq!(
            blocks_after_free[&0], 0,
            "Worker 0 should have 0 active blocks"
        );
        assert_eq!(
            blocks_after_free[&1], 0,
            "Worker 1 should have 0 active blocks"
        );
        assert_eq!(
            blocks_after_free[&2], 0,
            "Worker 2 should have 0 active blocks"
        );

        // Verify active tokens are zero for all workers
        assert_eq!(
            tokens_after_free[&0], 0,
            "Worker 0 should have 0 active tokens after freeing all"
        );
        assert_eq!(
            tokens_after_free[&1], 0,
            "Worker 1 should have 0 active tokens after freeing all"
        );
        assert_eq!(
            tokens_after_free[&2], 0,
            "Worker 2 should have 0 active tokens after freeing all"
        );

        Ok(())
    }
}
