// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core scheduler implementation.

use super::config::SchedulerConfig;
use super::kv_cache::KVCacheManager;
use super::policy::{FCFSPolicy, SchedulingPolicy};
use super::queues::{RunningRequests, WaitingQueue};
use super::request::{RequestStatus, SchedulerRequest};
use crate::v2::integrations::common::{Request, SchedulerConnectorState, SchedulerOutput};

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

/// The main scheduler for G1 block management.
///
/// This scheduler manages the allocation of GPU (G1) blocks to requests,
/// handling scheduling decisions, preemption, and request lifecycle.
///
/// # Block Management
///
/// The scheduler uses `KVCacheManager` to allocate real RAII blocks from
/// `BlockManager<G1>`. Blocks are stored in each `SchedulerRequest`'s
/// `block_state` field, which manages pending (mutable) and registered
/// (immutable) blocks.
///
/// # Block Lifecycle
///
/// 1. **Scheduling**: `KVCacheManager::allocate()` returns `MutableBlock<G1>`
///    which are stored in `request.block_state.pending`.
///
/// 2. **Forward Pass**: After the model computes token data, blocks are
///    transitioned via `KVCacheManager::complete_and_register()`.
///
/// 3. **Cleanup**: When requests finish or are preempted, blocks are dropped
///    via RAII, returning them to the appropriate pools.
///
/// # Integration with Connector
///
/// When `shared_state` is set, the scheduler can communicate with the
/// ConnectorLeader for G2+ tier offloading. This is completely optional -
/// the scheduler works independently without it.
pub struct Scheduler {
    /// Scheduler configuration.
    config: SchedulerConfig,

    /// KV cache manager for block allocation.
    kv_cache: KVCacheManager,

    /// Queue of requests waiting to be scheduled.
    waiting: WaitingQueue,

    /// Currently running requests.
    running: RunningRequests,

    /// Scheduling policy for request prioritization.
    policy: Box<dyn SchedulingPolicy>,

    /// Optional shared state with connector (completely optional).
    shared_state: Option<Arc<Mutex<dyn SchedulerConnectorState>>>,

    /// Current iteration number.
    iteration: usize,
}

impl Scheduler {
    /// Create a new scheduler with the given configuration and KV cache manager.
    pub fn new(config: SchedulerConfig, kv_cache: KVCacheManager) -> Self {
        let policy = Box::new(FCFSPolicy::new(config.max_num_seqs));
        Self {
            config,
            kv_cache,
            waiting: WaitingQueue::new(),
            running: RunningRequests::new(),
            policy,
            shared_state: None,
            iteration: 0,
        }
    }

    /// Set a custom scheduling policy.
    pub fn with_policy(mut self, policy: Box<dyn SchedulingPolicy>) -> Self {
        self.policy = policy;
        self
    }

    /// Attach optional shared state for connector communication.
    ///
    /// When set, the scheduler can communicate with the connector via this
    /// shared state. When None, the scheduler operates independently.
    pub fn with_shared_state(mut self, state: Arc<Mutex<dyn SchedulerConnectorState>>) -> Self {
        self.shared_state = Some(state);
        self
    }

    /// Get the current iteration number.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Get the number of waiting requests.
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Get the number of running requests.
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Get the KV cache usage as a fraction.
    pub fn cache_usage(&self) -> f32 {
        self.kv_cache.usage()
    }

    /// Add a new request to the scheduler.
    pub fn add_request(&mut self, request: Request) {
        let scheduler_request = SchedulerRequest::new(request);
        self.waiting.push_back(scheduler_request);
    }

    /// Abort a request by ID.
    ///
    /// The request will be removed from whichever queue it's in.
    // todo: this is very wrong. there is no interaction with the connector here.
    // if the request is running, we need to inform to ask the connector's request_finished method
    // and then handle the return value. if there are outstanding operations on the blocks, we need
    // to wait to clean up the internals (the held G1 blocks) until the connector is finished with the blocks.
    // we get this signal from the update_scheduler_output method in the connector.
    pub fn abort_request(&mut self, request_id: &str) {
        // Try to remove from waiting queue
        if let Some(mut request) = self.waiting.remove(request_id) {
            request.finish(RequestStatus::FinishedAborted);
            return;
        }

        // Try to remove from running
        if let Some(mut request) = self.running.remove(request_id) {
            request.finish(RequestStatus::FinishedAborted);
        }
    }

    /// Finish requests by ID with the given status.
    pub fn finish_requests(&mut self, request_ids: &[String], status: RequestStatus) {
        for request_id in request_ids {
            if let Some(mut request) = self.running.remove(request_id) {
                request.finish(status);
            }
        }
    }

    /// Run the scheduler to produce a scheduling decision.
    ///
    /// This is the main scheduling loop that:
    /// 1. Allocates blocks to running requests that need more
    /// 2. Schedules new requests from the waiting queue
    /// 3. Handles preemption if memory pressure occurs
    pub fn schedule(&mut self) -> SchedulerOutput {
        self.iteration += 1;
        let mut output = SchedulerOutput::new(self.iteration);
        let mut num_scheduled_tokens: HashMap<String, usize> = HashMap::new();

        // Phase 1: Allocate blocks for running requests (decode phase)
        self.allocate_for_running(&mut output, &mut num_scheduled_tokens);

        // Phase 2: Schedule new requests from waiting queue (prefill phase)
        self.schedule_waiting(&mut output, &mut num_scheduled_tokens);

        // Update totals
        output.set_num_scheduled_tokens(num_scheduled_tokens);

        output
    }

    /// Allocate blocks for running requests (decode phase).
    fn allocate_for_running(
        &mut self,
        output: &mut SchedulerOutput,
        num_scheduled_tokens: &mut HashMap<String, usize>,
    ) {
        // Collect request IDs first to avoid borrow issues
        let request_ids: Vec<String> = self.running.request_ids().cloned().collect();

        for request_id in request_ids {
            // First, get the info we need without holding the mutable borrow
            let (blocks_needed, tokens_to_compute, resumed, all_tokens, computed, output_count) = {
                let request = match self.running.get(&request_id) {
                    Some(r) => r,
                    None => continue,
                };
                (
                    request.num_new_blocks_needed(self.config.block_size),
                    request.num_tokens_to_compute(),
                    request.resumed_from_preemption,
                    if request.resumed_from_preemption {
                        Some(request.request.tokens.to_vec())
                    } else {
                        None
                    },
                    request.num_computed_tokens,
                    request.num_output_tokens,
                )
            };

            if blocks_needed > 0 {
                // Try to allocate new blocks from the KV cache manager
                if let Some(new_blocks) = self.kv_cache.allocate(blocks_needed) {
                    // Extract block IDs before moving blocks into request
                    let new_block_ids: Vec<_> = new_blocks.iter().map(|b| b.block_id()).collect();

                    // Now get mutable access to update the request
                    if let Some(request) = self.running.get_mut(&request_id) {
                        request.add_pending_blocks(new_blocks);
                        request.clear_resumed_flag();
                    }

                    // Record in output
                    num_scheduled_tokens.insert(request_id.clone(), tokens_to_compute);

                    output.add_cached_request(
                        request_id.clone(),
                        resumed,
                        vec![], // new_token_ids - populated by model output
                        all_tokens,
                        new_block_ids,
                        computed,
                        output_count,
                    );
                }
                // else: Need to preempt - handled in preemption phase
            } else {
                // No new blocks needed, just decode one token
                let tokens_to_schedule = 1; // Single decode token
                num_scheduled_tokens.insert(request_id.clone(), tokens_to_schedule);

                // Clear resumed flag
                if let Some(request) = self.running.get_mut(&request_id) {
                    request.clear_resumed_flag();
                }

                output.add_cached_request(
                    request_id.clone(),
                    resumed,
                    vec![],
                    None,
                    vec![],
                    computed,
                    output_count,
                );
            }
        }
    }

    /// Schedule new requests from the waiting queue.
    fn schedule_waiting(
        &mut self,
        output: &mut SchedulerOutput,
        num_scheduled_tokens: &mut HashMap<String, usize>,
    ) {
        let mut total_scheduled = output.total_num_scheduled_tokens;

        while !self.waiting.is_empty() {
            // Check budget limits
            if total_scheduled >= self.config.max_num_batched_tokens {
                break;
            }
            if self.running.len() >= self.config.max_num_seqs {
                break;
            }

            // Calculate available blocks for policy
            let available_blocks = self.kv_cache.free_blocks();

            // Collect waiting requests as references for policy
            let waiting_refs: Vec<&SchedulerRequest> = self.waiting.iter().collect();

            // Ask policy which request to schedule next
            let next_idx = self.policy.select_next(
                &waiting_refs,
                self.running.len(),
                available_blocks,
                self.config.block_size,
            );

            let Some(idx) = next_idx else {
                // Policy says don't schedule anything
                break;
            };

            // Remove the selected request from waiting queue
            // Note: We need to handle index correctly since we're using a VecDeque
            let mut request = match self.waiting.pop_front() {
                Some(r) => r,
                None => break,
            };

            // If not the first one, we need to re-add and pop the right one
            // For simplicity, FCFS always returns 0, so this works
            if idx != 0 {
                // Put it back and skip for now (complex case)
                self.waiting.push_front(request);
                break;
            }

            // Calculate blocks needed and tokens to schedule
            let blocks_needed = request.num_new_blocks_needed(self.config.block_size);
            let tokens_to_schedule = self.calculate_prefill_tokens(&request, total_scheduled);

            if tokens_to_schedule == 0 {
                self.waiting.push_front(request);
                break;
            }

            // Allocate blocks
            let blocks_to_allocate =
                (tokens_to_schedule + self.config.block_size - 1) / self.config.block_size;
            let blocks_to_allocate = blocks_to_allocate.min(blocks_needed);

            // Try to allocate blocks from KV cache manager
            let allocated_blocks = if blocks_to_allocate > 0 {
                match self.kv_cache.allocate(blocks_to_allocate) {
                    Some(blocks) => blocks,
                    None => {
                        // Not enough blocks - try preemption
                        if !self.try_preempt(blocks_to_allocate) {
                            // Can't preempt enough, put request back
                            self.waiting.push_front(request);
                            break;
                        }
                        // Try allocation again after preemption
                        match self.kv_cache.allocate(blocks_to_allocate) {
                            Some(blocks) => blocks,
                            None => {
                                // Still not enough, put request back
                                self.waiting.push_front(request);
                                break;
                            }
                        }
                    }
                }
            } else {
                Vec::new()
            };

            // Extract block IDs for output
            let block_ids: Vec<_> = allocated_blocks.iter().map(|b| b.block_id()).collect();

            // Add blocks to request
            request.add_pending_blocks(allocated_blocks);
            request.start_running();

            // Record in output
            output.add_new_request(
                request.request_id().to_string(),
                request.request.tokens.to_vec(),
                block_ids,
                request.num_computed_tokens,
            );

            num_scheduled_tokens.insert(request.request_id().to_string(), tokens_to_schedule);
            total_scheduled += tokens_to_schedule;

            // Move to running
            self.running.insert(request);
        }
    }

    /// Calculate how many tokens to prefill for a request.
    fn calculate_prefill_tokens(&self, request: &SchedulerRequest, current_total: usize) -> usize {
        let remaining_budget = self
            .config
            .max_num_batched_tokens
            .saturating_sub(current_total);
        let tokens_to_compute = request.num_tokens_to_compute();

        if self.config.enable_chunked_prefill {
            let max_chunk = self
                .config
                .max_prefill_chunk_size
                .unwrap_or(self.config.max_num_batched_tokens);
            tokens_to_compute.min(remaining_budget).min(max_chunk)
        } else {
            // Without chunked prefill, we need to fit the whole prefill
            if tokens_to_compute <= remaining_budget {
                tokens_to_compute
            } else {
                0 // Can't fit, don't schedule
            }
        }
    }

    /// Try to preempt running requests to free up blocks.
    ///
    /// This preempts the lowest priority running request(s) to free up blocks.
    /// When a request is preempted, its RAII blocks are dropped, returning
    /// them to the appropriate pools.
    fn try_preempt(&mut self, blocks_needed: usize) -> bool {
        let mut freed_blocks = 0;

        while freed_blocks < blocks_needed {
            // Collect running requests for policy
            let running_refs: Vec<&SchedulerRequest> =
                self.running.iter().map(|(_, r)| r).collect();

            if running_refs.is_empty() {
                return false;
            }

            // Ask policy which request to preempt
            let victim_id = match self
                .policy
                .select_victim(&running_refs, blocks_needed - freed_blocks)
            {
                Some(id) => id.to_string(),
                None => return false,
            };

            // Preempt the victim
            if let Some(mut victim) = self.running.remove(&victim_id) {
                // Count blocks before clearing (RAII will return them to pools)
                let victim_blocks = victim.block_state.total_blocks();
                freed_blocks += victim_blocks;
                // Preempt clears block_state, RAII returns blocks to pools
                victim.preempt();
                victim.resume();
                self.waiting.push_front(victim);
            } else {
                return false;
            }
        }

        true
    }

    /// Update state after model output is received.
    ///
    /// This should be called after each forward pass to update computed tokens
    /// and handle finished requests.
    pub fn update_from_output(
        &mut self,
        finished_ids: &[String],
        output_tokens: &HashMap<String, Vec<u32>>,
    ) {
        // Handle finished requests
        for request_id in finished_ids {
            if let Some(mut request) = self.running.remove(request_id) {
                request.finish(RequestStatus::FinishedStopped);
            }
        }

        // Update running requests with output tokens
        for (request_id, tokens) in output_tokens {
            if let Some(request) = self.running.get_mut(request_id) {
                request.add_output_tokens(tokens.len());
                // Update computed tokens to match total tokens
                request.update_computed_tokens(request.total_tokens());
            }
        }
    }
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler")
            .field("iteration", &self.iteration)
            .field("waiting", &self.waiting.len())
            .field("running", &self.running.len())
            .field("kv_cache", &self.kv_cache)
            .field("has_shared_state", &self.shared_state.is_some())
            .finish()
    }
}
