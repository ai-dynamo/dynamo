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
use crate::tokens::SequenceHash;
use anyhow::Result;
use dashmap::DashMap;
use derive_getters::Getters;
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::traits::events::{EventPublisher, EventSubscriber};
use futures::StreamExt;
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;
use uuid::Uuid;

use super::protocols::{ActiveSequenceEvent, ActiveSequenceEventData, WorkerWithDpRank};
use crate::kv_router::ACTIVE_SEQUENCES_SUBJECT;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_runtime::CancellationToken;

/// Errors that can occur during sequence management operations
#[derive(Debug, thiserror::Error)]
pub enum SequenceError {
    #[error("Worker {worker:?} not found")]
    WorkerNotFound { worker: WorkerWithDpRank },

    #[error("Request {request_id} already exists (assigned to worker {worker:?})")]
    DuplicateRequest {
        request_id: String,
        worker: WorkerWithDpRank,
    },

    #[error("Request {request_id} not found")]
    RequestNotFound { request_id: String },

    #[error("Failed to publish event: {0}")]
    PublishFailed(#[from] anyhow::Error),

    #[error("Failed to send command to worker: channel closed")]
    WorkerChannelClosed,
}

/// Duration after which stale requests are forcibly expired (5 minutes)
const EXPIRY_DURATION: Duration = Duration::from_secs(300);

// TODO: use the common request_id if it exists in the repo
pub type RequestId = String;

/// A multi-request sequence manager that handles multiple active sequences with shared KV cache
#[derive(Debug, Getters)]
pub struct ActiveSequences {
    active_seqs: HashMap<RequestId, Vec<(SequenceHash, Rc<()>)>>,

    prefill_tokens: HashMap<RequestId, usize>,

    unique_blocks: HashMap<SequenceHash, Weak<()>>,

    #[getter(copy)]
    block_size: usize,

    #[getter(copy)]
    active_tokens: usize,

    /// Timer for when to force expiry of stale requests
    expiry_timer: Instant,

    /// Set of request IDs to check for expiry
    expiry_requests: HashSet<RequestId>,
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
            active_tokens: 0,
            expiry_timer: Instant::now() + EXPIRY_DURATION,
            expiry_requests: HashSet::new(),
        }
    }

    fn touch_block(&mut self, block: &SequenceHash) -> Rc<()> {
        if let Some(weak) = self.unique_blocks.get(block)
            && let Some(rc) = weak.upgrade()
        {
            return rc;
        }

        let rc = Rc::new(());
        self.unique_blocks.insert(*block, Rc::downgrade(&rc));
        rc
    }

    fn try_remove_block(&mut self, block: &SequenceHash) {
        if let Some(weak) = self.unique_blocks.get(block)
            && weak.strong_count() == 0
        {
            self.unique_blocks.remove(block);
        }
    }

    pub fn active_blocks(&self) -> usize {
        self.unique_blocks.len()
    }

    /// Add a new request with its initial tokens
    /// Returns the set of expired request IDs that were removed during cleanup
    pub fn add_request(
        &mut self,
        request_id: RequestId,
        token_sequence: Option<Vec<SequenceHash>>,
        isl: usize,
        overlap: u32,
    ) -> HashSet<RequestId> {
        // Check for double-add and log error, returning early
        if self.active_seqs.contains_key(&request_id) {
            tracing::error!("Request {request_id} is already active. Ignoring duplicate add.");
            return HashSet::new();
        }

        // Lazily check and clean up expired requests, capturing removed IDs
        let removed_requests = self.force_expiry();

        let prefill_tokens = self.new_tokens(isl, overlap);
        self.prefill_tokens
            .insert(request_id.clone(), prefill_tokens);
        self.active_tokens += prefill_tokens;

        if let Some(sequence) = token_sequence {
            let sequence_with_refs: Vec<(SequenceHash, Rc<()>)> = sequence
                .iter()
                .map(|block| (*block, self.touch_block(block)))
                .collect();
            self.active_seqs
                .insert(request_id.clone(), sequence_with_refs);
        } else {
            // dummy empty sequence
            self.active_seqs.insert(request_id.clone(), Vec::new());
        }

        removed_requests
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
        let cached_tokens = (overlap as usize) * self.block_size;
        isl.checked_sub(cached_tokens)
            .unwrap_or_else(|| {
                tracing::error!(
                    "prefill_tokens < 0 with ISL {isl} < cached_tokens {cached_tokens} (overlap {overlap} * block_size {}), returning 0",
                    self.block_size
                );
                0
            })
    }

    pub fn potential_blocks_and_tokens(
        &self,
        token_sequence: Option<&[SequenceHash]>,
        isl: usize,
        overlap: u32,
    ) -> (usize, usize) {
        let potential_blocks = if let Some(token_seq) = token_sequence {
            self.new_blocks(token_seq) + self.active_blocks()
        } else {
            self.active_blocks()
        };
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
        self.new_blocks(token_sequence) + self.active_blocks()
    }

    /// Free all blocks associated with a request
    pub fn free(&mut self, request_id: &RequestId) -> usize {
        self.mark_prefill_completed(request_id);

        self.expiry_requests.remove(request_id);

        // Remove from active_seqs and get the token sequence
        let token_seq = match self.active_seqs.remove(request_id) {
            Some(seq) => seq,
            None => {
                tracing::warn!("Trying to free non-existent request {request_id}");
                return self.active_blocks();
            }
        };

        // Drop each Rc reference, then clean up the corresponding weak reference
        for (block_hash, rc) in token_seq {
            drop(rc);
            self.try_remove_block(&block_hash);
        }

        self.active_blocks()
    }

    /// Force expiry of stale requests if the timer has elapsed
    /// Returns the set of expired request IDs that were removed
    pub fn force_expiry(&mut self) -> HashSet<RequestId> {
        let now = Instant::now();

        // Early return if timer hasn't expired yet
        if now < self.expiry_timer {
            return HashSet::new();
        }

        // Process expired requests - drain to avoid clone
        let expired_requests: HashSet<RequestId> = self.expiry_requests.drain().collect();
        for request_id in &expired_requests {
            tracing::warn!("Force expiring stale request: {}", request_id);
            self.free(request_id);
        }

        self.expiry_timer = now + EXPIRY_DURATION;
        self.expiry_requests = self.active_seqs.keys().cloned().collect();

        expired_requests
    }
}

enum UpdateSequences {
    AddRequest {
        request_id: RequestId,
        token_sequence: Option<Vec<SequenceHash>>,
        isl: usize,
        overlap: u32,
        resp_tx: tokio::sync::oneshot::Sender<HashSet<RequestId>>,
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
        token_sequence: Option<Arc<Vec<SequenceHash>>>,
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
    senders: Arc<DashMap<WorkerWithDpRank, tokio::sync::mpsc::UnboundedSender<UpdateSequences>>>,
    request_to_worker: Arc<DashMap<RequestId, WorkerWithDpRank>>,
    handles: Arc<DashMap<WorkerWithDpRank, std::thread::JoinHandle<()>>>,
    block_size: usize,
    component: Component,
    router_id: Uuid,
    replica_sync: bool,
}

impl ActiveSequencesMultiWorker {
    pub fn new(
        component: Component,
        block_size: usize,
        workers_with_configs: HashMap<u64, Option<ModelRuntimeConfig>>,
        replica_sync: bool,
        router_uuid: String,
    ) -> Self {
        assert!(block_size > 1, "block_size must be greater than 1");

        let senders = Arc::new(DashMap::new());
        let handles = Arc::new(DashMap::new());
        let request_to_worker = Arc::new(DashMap::new());
        let router_id = Uuid::parse_str(&router_uuid).unwrap_or_else(|e| {
            tracing::warn!(
                "Failed to parse router UUID '{}': {}, using new UUID",
                router_uuid,
                e
            );
            Uuid::new_v4()
        });

        // Expand workers by their dp_rank
        for (worker_id, config) in workers_with_configs {
            let dp_size = config.as_ref().map(|c| c.data_parallel_size).unwrap_or(1);

            for dp_rank in 0..dp_size {
                let worker = WorkerWithDpRank::new(worker_id, dp_rank);
                // Create a child cancellation token from the component's runtime
                let cancel_token = component.drt().runtime().child_token();
                let (sender, handle) = Self::start_worker(block_size, cancel_token);
                senders.insert(worker, sender);
                handles.insert(worker, handle);
            }
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
            let cancel_token = component.drt().runtime().child_token();

            tokio::spawn(async move {
                // NATS subscription loop
                if let Err(e) = Self::subscribe_to_events(
                    senders_clone,
                    request_to_worker_clone,
                    component_clone,
                    router_id_clone,
                    cancel_token,
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
        cancel_token: CancellationToken,
    ) -> (
        tokio::sync::mpsc::UnboundedSender<UpdateSequences>,
        std::thread::JoinHandle<()>,
    ) {
        let (request_tx, request_rx) = tokio::sync::mpsc::unbounded_channel();

        let handle = std::thread::spawn(move || {
            // Create a single-threaded tokio runtime
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            runtime.block_on(async move {
                let mut active_sequences = ActiveSequences::new(block_size);
                let mut request_rx = request_rx;

                loop {
                    tokio::select! {
                        command = request_rx.recv() => {
                            let Some(command) = command else {
                                break;
                            };

                            match command {
                                UpdateSequences::AddRequest {
                                    request_id,
                                    token_sequence,
                                    isl,
                                    overlap,
                                    resp_tx,
                                } => {
                                    let removed = active_sequences.add_request(request_id, token_sequence, isl, overlap);
                                    let _ = resp_tx.send(removed);
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
                                        token_sequence.as_ref().map(|v| v.as_slice()),
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
                        // Handle cancellation
                        _ = cancel_token.cancelled() => {
                            tracing::debug!("Worker task cancelled");
                            break;
                        }
                    }
                }
            });

            tracing::debug!("ActiveSequences worker task completed");
        });

        (request_tx, handle)
    }

    /// Background task to subscribe to active sequence events and update all workers
    async fn subscribe_to_events(
        senders: Arc<
            DashMap<WorkerWithDpRank, tokio::sync::mpsc::UnboundedSender<UpdateSequences>>,
        >,
        request_to_worker: Arc<DashMap<RequestId, WorkerWithDpRank>>,
        component: Component,
        router_id: Uuid,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        let mut subscriber = component
            .subscribe_with_type::<ActiveSequenceEvent>(ACTIVE_SEQUENCES_SUBJECT)
            .await?;

        loop {
            tokio::select! {
                // Handle incoming events
                result = subscriber.next() => {
                    let Some(result) = result else {
                        // Stream ended
                        break;
                    };

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
                            request_to_worker.insert(event.request_id.clone(), event.worker);

                            if let Some(sender) = senders.get(&event.worker) {
                                // For replicated events, we create a dummy response channel since we don't need to handle expired requests
                                let (resp_tx, _) = tokio::sync::oneshot::channel();
                                let _ = sender.send(UpdateSequences::AddRequest {
                                    request_id: event.request_id.clone(),
                                    token_sequence: token_sequence.clone(),
                                    isl: *isl,
                                    overlap: *overlap,
                                    resp_tx,
                                });
                            } else {
                                tracing::warn!(
                                    "Worker {:?} not found, cannot process AddRequest",
                                    event.worker
                                );
                            }
                        }
                        ActiveSequenceEventData::Free => {
                            if let Some((_, worker)) = request_to_worker.remove(&event.request_id)
                                && let Some(sender) = senders.get(&worker)
                            {
                                let _ = sender.send(UpdateSequences::Free {
                                    request_id: event.request_id.clone(),
                                });
                            }
                        }
                        ActiveSequenceEventData::MarkPrefillCompleted => {
                            if let Some(worker) = request_to_worker.get(&event.request_id)
                                && let Some(sender) = senders.get(&*worker)
                            {
                                let _ = sender.send(UpdateSequences::MarkPrefillCompleted {
                                    request_id: event.request_id.clone(),
                                });
                            }
                        }
                    }
                }
                // Handle cancellation
                _ = cancel_token.cancelled() => {
                    tracing::debug!("Subscription task cancelled");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Update the set of workers, adding and removing as needed
    pub fn update_workers(
        &self,
        new_workers_with_configs: HashMap<u64, Option<ModelRuntimeConfig>>,
    ) {
        let current_workers: HashSet<WorkerWithDpRank> =
            self.senders.iter().map(|entry| *entry.key()).collect();

        // Expand new workers by their dp_rank
        let mut new_workers: HashSet<WorkerWithDpRank> = HashSet::new();
        for (worker_id, config) in &new_workers_with_configs {
            let dp_size = config.as_ref().map(|c| c.data_parallel_size).unwrap_or(1);

            for dp_rank in 0..dp_size {
                new_workers.insert(WorkerWithDpRank::new(*worker_id, dp_rank));
            }
        }

        let workers_to_remove: Vec<WorkerWithDpRank> =
            current_workers.difference(&new_workers).copied().collect();
        let workers_to_add: Vec<WorkerWithDpRank> =
            new_workers.difference(&current_workers).copied().collect();

        // Remove workers (this will naturally remove all dp ranks for a worker_id)
        for worker in &workers_to_remove {
            tracing::warn!("Removing worker {:?}", worker);

            // Send shutdown command to the worker
            if let Some((_, sender)) = self.senders.remove(worker) {
                let _ = sender.send(UpdateSequences::Shutdown);
            }
            self.handles.remove(worker);

            // Clean up request_to_worker mappings for this worker
            self.request_to_worker
                .retain(|_request_id, mapped_worker| mapped_worker != worker);
        }

        // Add new workers
        for worker in &workers_to_add {
            tracing::warn!("Adding worker {:?}", worker);

            let (sender, handle) = Self::start_worker(
                self.block_size,
                self.component.drt().runtime().child_token(),
            );
            self.senders.insert(*worker, sender);
            self.handles.insert(*worker, handle);
        }
    }

    pub async fn add_request(
        &self,
        request_id: RequestId,
        token_sequence: Option<Vec<SequenceHash>>,
        isl: usize,
        overlap: u32,
        worker: WorkerWithDpRank,
    ) -> Result<(), SequenceError> {
        // Check for worker existence
        if !self.senders.contains_key(&worker) {
            return Err(SequenceError::WorkerNotFound { worker });
        }

        // Check for duplicate request
        if let Some(existing_worker) = self.request_to_worker.get(&request_id) {
            return Err(SequenceError::DuplicateRequest {
                request_id,
                worker: *existing_worker,
            });
        }

        // Create response channel
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();

        // Publish event only if replica_sync is enabled
        if self.replica_sync {
            let event = ActiveSequenceEvent {
                request_id: request_id.clone(),
                worker,
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

        // Update local state with full WorkerWithDpRank
        self.request_to_worker.insert(request_id.clone(), worker);

        self.senders
            .get(&worker)
            .unwrap()
            .send(UpdateSequences::AddRequest {
                request_id,
                token_sequence,
                isl,
                overlap,
                resp_tx,
            })
            .map_err(|_| SequenceError::WorkerChannelClosed)?;

        // Wait for response and handle removed requests
        let removed_requests = resp_rx
            .await
            .map_err(|_| SequenceError::WorkerChannelClosed)?;

        // Remove expired requests from request_to_worker mapping
        for expired_id in &removed_requests {
            self.request_to_worker.remove(expired_id);
        }

        Ok(())
    }

    /// Free all blocks associated with a request
    ///
    /// Note: This operation is idempotent. Calling it multiple times for the same request
    /// will log a warning but not return an error (double free is allowed).
    pub async fn free(&self, request_id: &RequestId) -> Result<(), SequenceError> {
        // Check if request exists - if not, it's already been freed (idempotent)
        let Some(worker) = self.request_to_worker.get(request_id).map(|entry| *entry) else {
            tracing::debug!("Request {request_id} not found, already freed (idempotent)");
            return Ok(());
        };

        // Verify worker still exists
        if !self.senders.contains_key(&worker) {
            return Err(SequenceError::WorkerNotFound { worker });
        }

        // Publish event only if replica_sync is enabled
        if self.replica_sync {
            let event = ActiveSequenceEvent {
                request_id: request_id.clone(),
                worker,
                data: ActiveSequenceEventData::Free,
                router_id: self.router_id,
            };
            self.component
                .publish(ACTIVE_SEQUENCES_SUBJECT, &event)
                .await?;
        }

        // Update local state
        self.senders
            .get(&worker)
            .unwrap()
            .send(UpdateSequences::Free {
                request_id: request_id.clone(),
            })
            .map_err(|_| SequenceError::WorkerChannelClosed)?;

        self.request_to_worker.remove(request_id);

        Ok(())
    }

    /// Mark prefill as completed for a request
    ///
    /// Note: Calling this multiple times for the same request is allowed and will be a no-op
    /// after the first call (idempotent).
    pub async fn mark_prefill_completed(
        &self,
        request_id: &RequestId,
    ) -> Result<(), SequenceError> {
        let worker = self
            .request_to_worker
            .get(request_id)
            .map(|entry| *entry)
            .ok_or_else(|| SequenceError::RequestNotFound {
                request_id: request_id.clone(),
            })?;

        // Verify worker still exists
        if !self.senders.contains_key(&worker) {
            return Err(SequenceError::WorkerNotFound { worker });
        }

        // Publish event only if replica_sync is enabled
        if self.replica_sync {
            let event = ActiveSequenceEvent {
                request_id: request_id.clone(),
                worker,
                data: ActiveSequenceEventData::MarkPrefillCompleted,
                router_id: self.router_id,
            };
            self.component
                .publish(ACTIVE_SEQUENCES_SUBJECT, &event)
                .await?;
        }

        // Update local state
        self.senders
            .get(&worker)
            .unwrap()
            .send(UpdateSequences::MarkPrefillCompleted {
                request_id: request_id.clone(),
            })
            .map_err(|_| SequenceError::WorkerChannelClosed)?;

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
    ) -> HashMap<WorkerWithDpRank, T> {
        let mut results = HashMap::new();
        let token_sequence_shared = token_sequence.map(Arc::new);
        let mut receivers = Vec::new();

        // Send queries to all workers in parallel
        for entry in self.senders.iter() {
            let worker = *entry.key();
            let sender = entry.value();
            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            receivers.push((worker, resp_rx));
            if let Err(e) = sender.send(command_fn(token_sequence_shared.clone(), resp_tx)) {
                tracing::error!("Failed to send command to worker {:?}: {}", worker, e);
            }
        }

        // Collect results from all workers
        for (worker, receiver) in receivers {
            match tokio::time::timeout(tokio::time::Duration::from_secs(1), receiver).await {
                Ok(Ok(result)) => {
                    results.insert(worker, result);
                }
                Ok(Err(_)) => {
                    tracing::error!("Worker {:?} dropped response channel", worker);
                }
                Err(_) => {
                    tracing::error!("Timeout waiting for response from worker {:?}", worker);
                }
            }
        }

        results
    }

    /// Query all workers for the number of new blocks that would be added by a token sequence
    pub async fn new_blocks(
        &self,
        token_sequence: Vec<SequenceHash>,
    ) -> HashMap<WorkerWithDpRank, usize> {
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
    ) -> HashMap<WorkerWithDpRank, usize> {
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
        token_sequence: Option<Vec<SequenceHash>>,
        isl: usize,
        overlaps: OverlapScores,
    ) -> (
        HashMap<WorkerWithDpRank, usize>,
        HashMap<WorkerWithDpRank, usize>,
    ) {
        let mut potential_blocks = HashMap::new();
        let mut potential_tokens = HashMap::new();
        let token_sequence_shared = token_sequence.map(Arc::new);
        let mut receivers = Vec::new();

        // Iterate through all workers, not just those with overlap
        // This ensures we properly account for active tokens/blocks on all workers
        for sender_entry in self.senders.iter() {
            let worker = *sender_entry.key();
            let sender = sender_entry.value();

            // Get overlap for this worker (defaults to 0 if not in overlaps)
            let overlap = *overlaps.scores.get(&worker).unwrap_or(&0);

            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            receivers.push((worker, resp_rx));

            if let Err(e) = sender.send(UpdateSequences::PotentialBlocksAndTokens {
                token_sequence: token_sequence_shared.clone(),
                isl,
                overlap,
                resp_tx,
            }) {
                tracing::error!(
                    "Failed to send potential_tokens command to worker {:?}: {}",
                    worker,
                    e
                );
            }
        }

        // Collect results from all workers
        for (worker, receiver) in receivers {
            match tokio::time::timeout(tokio::time::Duration::from_secs(1), receiver).await {
                Ok(Ok((blocks, tokens))) => {
                    potential_blocks.insert(worker, blocks);
                    potential_tokens.insert(worker, tokens);
                }
                Ok(Err(_)) => {
                    tracing::error!("Worker {:?} dropped response channel", worker);
                }
                Err(_) => {
                    tracing::error!("Timeout waiting for response from worker {:?}", worker);
                }
            }
        }

        (potential_blocks, potential_tokens)
    }

    /// Query all workers for their current number of active blocks
    pub async fn active_blocks(&self) -> HashMap<WorkerWithDpRank, usize> {
        self.query_workers(None, |_, resp_tx| UpdateSequences::ActiveBlocks { resp_tx })
            .await
    }

    /// Query all workers for their current number of active tokens
    pub async fn active_tokens(&self) -> HashMap<WorkerWithDpRank, usize> {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    use std::sync::Arc;

    #[test]
    fn test_active_sequences_shared_blocks() {
        let block_size = 4;
        let mut seq_manager = ActiveSequences::new(block_size);

        seq_manager.add_request("request_1".to_string(), Some(vec![1, 2, 3]), 12, 0);
        assert_eq!(seq_manager.active_blocks(), 3);
        assert_eq!(seq_manager.active_tokens(), 12);

        seq_manager.add_request("request_2".to_string(), Some(vec![4]), 4, 0);
        assert_eq!(seq_manager.active_blocks(), 4);
        assert_eq!(seq_manager.active_tokens(), 16);

        seq_manager.add_request("request_3".to_string(), Some(vec![1, 2, 3, 4]), 16, 4);
        assert_eq!(seq_manager.active_blocks(), 4);
        assert_eq!(seq_manager.active_tokens(), 16);

        seq_manager.free(&"request_2".to_string());
        assert_eq!(seq_manager.active_blocks(), 4);
        assert_eq!(seq_manager.active_tokens(), 12);

        seq_manager.free(&"request_3".to_string());
        assert_eq!(seq_manager.active_blocks(), 3);
        assert_eq!(seq_manager.active_tokens(), 12);

        seq_manager.free(&"request_1".to_string());
        assert_eq!(seq_manager.active_blocks(), 0);
        assert_eq!(seq_manager.active_tokens(), 0);
    }

    #[tokio::test]
    #[ignore]
    async fn test_multi_worker_cross_instance_sync() -> Result<()> {
        // Initialize logging once
        dynamo_runtime::logging::init();

        let block_size = 4; // arbitrary block size

        // Create runtime and distributed runtime
        let runtime = Runtime::from_current()?;
        let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

        // Create namespace and shared component for both seq_managers
        let namespace = distributed.namespace("test_cross_instance_sync")?;
        let component = namespace.component("sequences")?;

        // Create multi-worker sequence managers with:
        // - Worker 0 with dp_size=2 (dp_ranks 0 and 1)
        // - Worker 1 with dp_size=1 (dp_rank 0)
        // This gives us 3 effective workers total to test dp_rank effect
        // Both seq_managers use the same component to ensure event synchronization works
        let mut workers_with_configs = HashMap::new();

        // Create runtime config for worker 0 with dp_size=2
        let mut config_worker_0 = crate::local_model::runtime_config::ModelRuntimeConfig::new();
        config_worker_0.data_parallel_size = 2;
        workers_with_configs.insert(0, Some(config_worker_0));

        // Create runtime config for worker 1 with dp_size=1 (default)
        let config_worker_1 = crate::local_model::runtime_config::ModelRuntimeConfig::new();
        workers_with_configs.insert(1, Some(config_worker_1));

        let seq_manager_1 = Arc::new(ActiveSequencesMultiWorker::new(
            component.clone(),
            block_size,
            workers_with_configs.clone(),
            true,
            Uuid::new_v4().to_string(),
        ));
        let seq_manager_2 = Arc::new(ActiveSequencesMultiWorker::new(
            component,
            block_size,
            workers_with_configs,
            true,
            Uuid::new_v4().to_string(),
        ));

        // Give some time for the subscription loops to start
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // PHASE 1: Add requests using both seq_manager_1 and seq_manager_2

        // Add request_0 to worker 0, dp_rank 0: sequence [0, 1, 2]
        seq_manager_1
            .add_request(
                "request_0".to_string(),
                Some(vec![0, 1, 2]),
                12, // ISL (3 blocks * 4 block_size)
                0,  // no overlap
                WorkerWithDpRank::new(0, 0),
            )
            .await?;

        // Add request_1 to worker 0, dp_rank 1: sequence [3, 4]
        seq_manager_1
            .add_request(
                "request_1".to_string(),
                Some(vec![3, 4]),
                8, // ISL (2 blocks * 4 block_size)
                0, // no overlap
                WorkerWithDpRank::new(0, 1),
            )
            .await?;

        // Add request_2 to worker 1, dp_rank 0: sequence [0, 1, 2, 3] using seq_manager_2
        seq_manager_2
            .add_request(
                "request_2".to_string(),
                Some(vec![0, 1, 2, 3]),
                16, // ISL (4 blocks * 4 block_size)
                0,  // no overlap
                WorkerWithDpRank::new(1, 0),
            )
            .await?;

        // Give some time for synchronization
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        // Query seq_manager_1 to verify it sees all requests including request_2 from seq_manager_2
        let blocks_phase1 = seq_manager_1.active_blocks().await;
        let tokens_phase1 = seq_manager_1.active_tokens().await;

        // Verify that seq_manager_1 sees all requests including request_2 from seq_manager_2
        // We now have:
        // - Worker 0, dp_rank 0: request_0
        // - Worker 0, dp_rank 1: request_1
        // - Worker 1, dp_rank 0: request_2
        let worker_0_dp0 = WorkerWithDpRank::new(0, 0);
        let worker_0_dp1 = WorkerWithDpRank::new(0, 1);
        let worker_1_dp0 = WorkerWithDpRank::new(1, 0);

        assert_eq!(
            blocks_phase1[&worker_0_dp0], 3,
            "Worker 0 dp_rank 0 should have 3 active blocks (from request_0)"
        );
        assert_eq!(
            blocks_phase1[&worker_0_dp1], 2,
            "Worker 0 dp_rank 1 should have 2 active blocks (from request_1)"
        );
        assert_eq!(
            blocks_phase1[&worker_1_dp0], 4,
            "Worker 1 dp_rank 0 should have 4 active blocks (from request_2 added by seq_manager_2)"
        );
        assert_eq!(
            tokens_phase1[&worker_0_dp0], 12,
            "Worker 0 dp_rank 0 should have 12 active tokens"
        );
        assert_eq!(
            tokens_phase1[&worker_0_dp1], 8,
            "Worker 0 dp_rank 1 should have 8 active tokens"
        );
        assert_eq!(
            tokens_phase1[&worker_1_dp0], 16,
            "Worker 1 dp_rank 0 should have 16 active tokens (from request_2 added by seq_manager_2)"
        );

        // PHASE 2: Free requests using opposite sequence managers, verify on seq_manager_2

        // Free request_2 (which was added by seq_manager_2) using seq_manager_1
        seq_manager_1.free(&"request_2".to_string()).await?;

        // Free request_0 and request_1 (which were added by seq_manager_1) using seq_manager_2
        seq_manager_2.free(&"request_0".to_string()).await?;
        seq_manager_2.free(&"request_1".to_string()).await?;

        // Give some time for synchronization
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        // Query seq_manager_2 to verify everything is empty
        let blocks_phase2 = seq_manager_2.active_blocks().await;
        let tokens_phase2 = seq_manager_2.active_tokens().await;

        // Verify phase 2 results - everything should be empty for all 3 workers
        let all_workers = vec![
            WorkerWithDpRank::new(0, 0),
            WorkerWithDpRank::new(0, 1),
            WorkerWithDpRank::new(1, 0),
        ];

        for worker in all_workers {
            assert_eq!(
                blocks_phase2[&worker], 0,
                "Worker (id={}, dp_rank={}) should have 0 active blocks after all requests freed",
                worker.worker_id, worker.dp_rank
            );
            assert_eq!(
                tokens_phase2[&worker], 0,
                "Worker (id={}, dp_rank={}) should have 0 active tokens after all requests freed",
                worker.worker_id, worker.dp_rank
            );
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_multi_worker_no_token_sequence_sync() -> Result<()> {
        // Initialize logging once
        dynamo_runtime::logging::init();

        let block_size = 4; // arbitrary block size

        // Create runtime and distributed runtime
        let runtime = Runtime::from_current()?;
        let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

        // Create namespace and shared component for both seq_managers
        let namespace = distributed.namespace("test_no_token_seq_sync")?;
        let component = namespace.component("sequences")?;

        // Create multi-worker sequence managers with ALL workers [0, 1, 2]
        // Both use the same component to ensure event synchronization works
        let mut workers_with_configs = HashMap::new();
        workers_with_configs.insert(0, None);
        workers_with_configs.insert(1, None);
        workers_with_configs.insert(2, None);

        let seq_manager_1 = Arc::new(ActiveSequencesMultiWorker::new(
            component.clone(),
            block_size,
            workers_with_configs.clone(),
            true,
            Uuid::new_v4().to_string(),
        ));
        let seq_manager_2 = Arc::new(ActiveSequencesMultiWorker::new(
            component,
            block_size,
            workers_with_configs,
            true,
            Uuid::new_v4().to_string(),
        ));

        // Give some time for the subscription loops to start
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // PHASE 1: Add requests (without token sequences) using both seq_managers

        // Add request_0 to worker 0 with no token sequence
        seq_manager_1
            .add_request(
                "request_0".to_string(),
                None, // No token sequence
                12,   // ISL (12 tokens)
                0,    // no overlap
                WorkerWithDpRank::from_worker_id(0),
            )
            .await?;

        // Add request_1 to worker 1 with no token sequence
        seq_manager_1
            .add_request(
                "request_1".to_string(),
                None, // No token sequence
                8,    // ISL (8 tokens)
                0,    // no overlap
                WorkerWithDpRank::from_worker_id(1),
            )
            .await?;

        // Add request_2 to worker 2 with no token sequence using seq_manager_2
        seq_manager_2
            .add_request(
                "request_2".to_string(),
                None, // No token sequence
                16,   // ISL (16 tokens)
                0,    // no overlap
                WorkerWithDpRank::from_worker_id(2),
            )
            .await?;

        // Give some time for synchronization
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        // Query seq_manager_1 to verify it sees all requests including request_2 from seq_manager_2
        let tokens_phase1 = seq_manager_1.active_tokens().await;

        // Verify that seq_manager_1 sees all requests including request_2 from thread 2
        let worker_0 = WorkerWithDpRank::from_worker_id(0);
        let worker_1 = WorkerWithDpRank::from_worker_id(1);
        let worker_2 = WorkerWithDpRank::from_worker_id(2);

        assert_eq!(
            tokens_phase1[&worker_0], 12,
            "Worker 0 should have 12 active tokens"
        );
        assert_eq!(
            tokens_phase1[&worker_1], 8,
            "Worker 1 should have 8 active tokens"
        );
        assert_eq!(
            tokens_phase1[&worker_2], 16,
            "Worker 2 should have 16 active tokens (from request_2 added by seq_manager_2)"
        );

        // PHASE 2: Free requests using opposite sequence managers, verify on seq_manager_2

        // Mark prefill completed and free request_2 (which was added by seq_manager_2) using seq_manager_1
        seq_manager_1
            .mark_prefill_completed(&"request_2".to_string())
            .await?;
        seq_manager_1.free(&"request_2".to_string()).await?;

        // Mark prefill completed and free requests 0 and 1 (which were added by seq_manager_1) using seq_manager_2
        seq_manager_2
            .mark_prefill_completed(&"request_0".to_string())
            .await?;
        seq_manager_2
            .mark_prefill_completed(&"request_1".to_string())
            .await?;
        seq_manager_2.free(&"request_0".to_string()).await?;
        seq_manager_2.free(&"request_1".to_string()).await?;

        // Give some time for synchronization
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        // Query seq_manager_2 to verify everything is empty
        let tokens_phase2 = seq_manager_2.active_tokens().await;

        // Verify phase 2 results - everything should be empty
        for worker_id in 0..=2 {
            let worker = WorkerWithDpRank::from_worker_id(worker_id);
            assert_eq!(
                tokens_phase2[&worker], 0,
                "Worker {} should have 0 active tokens after all requests freed",
                worker_id
            );
        }

        Ok(())
    }
}
