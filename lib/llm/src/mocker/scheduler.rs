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

//! Asynchronous Scheduler for LLM Request Management
//!
//! This module implements an asynchronous scheduler that handles three main functions:
//! 1. Receiving new requests and placing them in the waiting queue
//! 2. Scheduling waiting requests against available KV cache resources
//! 3. Simulating the execution of running requests with realistic timing
//!
//! ## Scheduling Process
//! The scheduler uses a watermark-based approach to determine if there's sufficient
//! KV cache space for new requests. It also enforces a batched tokens budget to prevent
//! oversubscription of computational resources. Only requests that can be allocated
//! these resources are moved from waiting to running state.
//!
//! ## Request Simulation
//! The simulation models two key phases:
//! - Prefill phase: Uses a quadratic cost function: (cached_tokens + new_tokens) * new_tokens
//! - Decode phase: Uses a cost function proportional to active KV blocks (linear)
//!
//! ## Resource Management
//! The scheduler communicates with the KvManager through MoveBlock signals at each
//! stage of request processing. When resources become constrained, it employs an
//! LRU-based preemption strategy where the oldest running request is evicted and
//! placed at the back of the waiting queue to be rescheduled later.
//!
//! ## NOTE
//! The current prefill and decoding time simulations are not scientific at all and are WIP

use crate::kv_router::protocols::ForwardPassMetrics;
use crate::mocker::evictor::LRUEvictor;
use crate::mocker::kv_manager::KvManager;
use crate::mocker::protocols::DirectRequest;
use crate::mocker::protocols::{MoveBlock, PrefillCost, UniqueBlock};
use crate::mocker::sequence::ActiveSequence;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::time::{interval, Duration};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Enum representing either a direct request or an active sequence
pub enum Request {
    Direct(DirectRequest),
    Active(ActiveSequence),
}

#[derive(Default)]
struct SchedulerState {
    waiting: VecDeque<Uuid>,
    ready: VecDeque<Uuid>,
    running: LRUEvictor<Uuid>,
    requests: HashMap<Uuid, Request>,
    prefill_costs: HashMap<Uuid, Option<PrefillCost>>,
}

impl SchedulerState {
    /// Create a new UUID for a DirectRequest, add it to requests, and push the UUID to waiting.
    fn receive(&mut self, request: DirectRequest) -> Uuid {
        // Use the provided UUID if available, otherwise generate a new one
        let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);

        // Add the request to the map and waiting queue
        self.requests.insert(uuid, Request::Direct(request));
        self.waiting.push_back(uuid);
        uuid
    }

    /// Get the next UUID from ready or waiting queue and its associated Request.
    /// Returns from ready if not empty, otherwise from waiting, or None if both are empty.
    /// Also removes the Request from the requests HashMap.
    fn next(&mut self) -> Option<(Uuid, Request)> {
        let uuid = self
            .ready
            .pop_front()
            .or_else(|| self.waiting.pop_front())?;
        let request = self.requests.remove(&uuid)?;
        Some((uuid, request))
    }

    /// Move a UUID and its Request to the ready queue.
    fn make_ready(&mut self, uuid: Uuid, active_seq: ActiveSequence) {
        self.requests.insert(uuid, Request::Active(active_seq));
        self.ready.push_back(uuid);
    }

    /// Schedule the request with the given UUID.
    /// Returns the creation signal from the ActiveSequence.
    fn run(&mut self, uuid: Uuid, active_seq: ActiveSequence) -> MoveBlock {
        // Insert the request into the map
        self.requests.insert(uuid, Request::Active(active_seq));

        // Get the creation signal
        let Some(Request::Active(sequence)) = self.requests.get(&uuid) else {
            panic!("Failed to get ActiveSequence for UUID");
        };
        let Some(signal) = sequence.creation_signal() else {
            panic!("Failed to get creation signal from ActiveSequence");
        };

        // Add to running requests
        self.running.insert(uuid);
        signal.clone()
    }

    /// Set the prefill cost for a UUID
    fn set_prefill_cost(&mut self, uuid: Uuid, cost: Option<PrefillCost>) {
        self.prefill_costs.insert(uuid, cost);
    }

    /// Get the prefill compute value for a UUID if available
    fn get_prefill_compute(&self, uuid: &Uuid) -> Option<f64> {
        self.prefill_costs
            .get(uuid)
            .and_then(|cost| cost.as_ref())
            .map(|cost| cost.prefill_compute)
    }

    /// Calculate the current running batched tokens
    fn num_batched_tokens(&self) -> usize {
        self.prefill_costs
            .values()
            .map(|cost| match cost {
                Some(cost) => cost.new_tokens,
                None => 1,
            })
            .sum()
    }

    /// Remove a UUID and its associated Request from collections.
    fn complete(&mut self, uuid: &Uuid) {
        // println!("Request {} will complete", uuid);
        self.running.remove(uuid);
        self.requests.remove(uuid);
        self.prefill_costs.remove(uuid);
    }

    /// Preempt the oldest running request by evicting it from running, resetting the sequence,
    /// and adding it back to the waiting queue.
    /// Returns the signal from reset_with_signal or None if no requests are running.
    fn preempt(&mut self) -> Option<Vec<MoveBlock>> {
        // Evict the oldest UUID from running
        let uuid = self.running.evict()?;
        eprintln!("Request {} will be preempted", uuid);

        // Remove the request from the requests HashMap and ensure it's an ActiveSequence
        let request = self.requests.remove(&uuid)?;

        // Remove the prefill cost to force recomputation
        self.prefill_costs.remove(&uuid);

        // Extract the ActiveSequence from the Request enum
        let Request::Active(mut active_sequence) = request else {
            panic!("Expected ActiveSequence in running queue")
        };

        // Reset the sequence and get the new sequence and signal
        let signals = active_sequence.reset_with_signal();

        // Insert the new sequence back into the requests map and add to waiting queue
        self.requests.insert(uuid, Request::Active(active_sequence));
        self.waiting.push_back(uuid);

        Some(signals)
    }
}

/// Manages scheduling of requests using KvManager resources
#[derive(Clone)]
pub struct Scheduler {
    state: Arc<Mutex<SchedulerState>>,
    kv_manager: Arc<Mutex<KvManager>>,
    request_tx: mpsc::Sender<DirectRequest>,
}

impl Scheduler {
    /// Create a new Scheduler with the given parameters
    pub fn new(
        kv_capacity: usize,
        watermark: f64,
        block_size: usize,
        chunk_size: Option<usize>,
        output_tx: Option<mpsc::Sender<Uuid>>,
        cancellation_token: Option<CancellationToken>,
    ) -> Self {
        // Create KvManager internally
        let kv_manager = KvManager::new(kv_capacity, block_size);

        let token_capacity: usize = 8192;
        let state = Arc::new(Mutex::new(SchedulerState::default()));

        let kv_manager = Arc::new(Mutex::new(kv_manager));
        let chunk_size = chunk_size.unwrap_or(256);

        // Create channel for request handling
        let (request_tx, mut request_rx) = mpsc::channel::<DirectRequest>(1024);

        // Use provided cancellation token or create new one
        let cancellation_token = cancellation_token.unwrap_or_default();
        let token_clone = cancellation_token.clone();

        // Create a clone for the background task
        let state_clone = state.clone();
        let kv_manager_clone = kv_manager.clone();
        let output_tx_clone = output_tx.clone();

        // Spawn main background task with cancellation token
        tokio::spawn(async move {
            let mut schedule_interval = interval(Duration::from_millis(5));
            let mut simulate_interval = interval(Duration::from_millis(1));

            loop {
                tokio::select! {
                    biased;

                    // Enqueue new request
                    Some(request) = request_rx.recv() => {
                        let mut state = state_clone.lock().await;
                        state.receive(request);
                    }

                    // Try Scheduling Requests
                    _ = schedule_interval.tick() => {
                        let mut state_guard = state_clone.lock().await;
                        let mut kv_manager_guard = kv_manager_clone.lock().await;

                        // Process DirectRequests, converting them to ActiveSequence and scheduling them until we can't
                        // schedule anymore.
                        while let Some((uuid, request)) = state_guard.next() {
                            let active_sequence = get_active_sequence(request, block_size, chunk_size);

                            // Calculate token budget using new_tokens from PrefillCost
                            let total_prefill_tokens = state_guard.num_batched_tokens();
                            let tokens_budget = token_capacity.saturating_sub(total_prefill_tokens);

                            // Check if it can be scheduled
                            let Some(prefill_cost) = kv_manager_guard.try_schedule(&active_sequence, watermark, tokens_budget) else {
                                state_guard.make_ready(uuid, active_sequence);
                                break;
                            };

                            // Get creation signal and schedule the request
                            let signal = state_guard.run(uuid, active_sequence);
                            kv_manager_guard.process(&signal);
                            state_guard.set_prefill_cost(uuid, Some(prefill_cost));
                        }
                    }

                    // Check for cancellation
                    _ = token_clone.cancelled() => {
                        break;
                    }

                    // Simulate running requests (prefill + decode)
                    _ = simulate_interval.tick() => {
                        let mut state_guard = state_clone.lock().await;
                        let mut kv_manager_guard = kv_manager_clone.lock().await;

                        // Base time needed for decoding (assumed memory bound on KV cache)
                        let active_tokens = kv_manager_guard.num_active_blocks() * block_size;
                        // TODO: 2 is a dummy / magic scaling factor
                        let mut generation_time = Duration::from_micros((active_tokens / 2) as u64);

                        // Process each running request
                        let uuids: Vec<Uuid> = state_guard.running.keys().cloned().collect();
                        for uuid in uuids {
                            // Check if UUID is still in running_requests, if not skip this iteration
                            if !state_guard.running.contains(&uuid) {
                                continue;
                            }

                            // Get prefill compute value first
                            let prefill_compute = state_guard.get_prefill_compute(&uuid);

                            // Get the active sequence for this UUID
                            let sequence = state_guard.requests.get_mut(&uuid)
                                .and_then(|req| if let Request::Active(seq) = req { Some(seq) } else { None })
                                .expect("UUID in running_requests must have a corresponding active sequence");

                            // Generate token and get signals
                            let signals = sequence.generate();

                            // Accumulate sleep duration based on prefill_compute if available
                            // prefill compute = (cached_tokens + new_tokens) * new_tokens
                            let sleep_ms = if let Some(compute) = prefill_compute {
                                // TODO: 1024 is a dummy / magic scaling factor
                                (compute / 1024.0) as u64
                            } else { 0 };
                            generation_time += Duration::from_micros(sleep_ms);

                            // Process all signals with the KvManager
                            // Handling of preemption on failure
                            if !process_signals(&mut kv_manager_guard, &signals) {
                                sequence.pop();  // revert the failed generation op

                                // free_signal derefs the preempted blocks
                                let Some(free_signal) = state_guard.preempt() else {
                                    panic!("Failed to acquire signal to free KV blocks from preemption");
                                };

                                for signal in free_signal {
                                    kv_manager_guard.process(&signal);
                                }
                                continue;
                            }

                            // Send UUID notification for each generated token
                            // TODO: hook this up to an AsyncEngine
                            if let Some(tx) = &output_tx_clone {
                                let _ = tx.try_send(uuid);
                            }

                            // Check if we're done after generating
                            if sequence.generated_tokens() >= sequence.max_output_tokens() {
                                state_guard.complete(&uuid);
                                continue;
                            }

                            // Transition to decode (no prefill cost)
                            if sequence.generated_tokens() == 1 {
                                state_guard.set_prefill_cost(uuid, None);
                            }
                        }

                        // Sleep once for the accumulated duration
                        if generation_time.as_millis() > 0 {
                            tokio::time::sleep(generation_time).await;
                        }
                    }
                }
            }
        });

        Self {
            state,
            kv_manager,
            request_tx,
        }
    }

    /// Add a new request to the waiting queue
    pub async fn receive(&self, request: DirectRequest) {
        let _ = self.request_tx.send(request).await;
    }

    /// Get the count of waiting requests
    pub async fn waiting_count(&self) -> usize {
        let state = self.state.lock().await;
        state.waiting.len()
    }

    /// Get the count of running requests
    pub async fn running_count(&self) -> usize {
        let state = self.state.lock().await;
        state.running.len()
    }

    /// Get the current capacity of the KvManager
    pub async fn kv_usage_perc(&self) -> f64 {
        let kv_manager = self.kv_manager.lock().await;
        kv_manager.current_capacity_perc()
    }

    /// Returns forward pass metrics for monitoring purposes
    pub async fn get_forward_pass_metrics(&self) -> ForwardPassMetrics {
        let state = self.state.lock().await;
        let kv_manager = self.kv_manager.lock().await;

        // Get the active blocks and total capacity from KvManager
        let active_blocks_count = kv_manager.active_blocks().len() as u64;
        let total_capacity = kv_manager.max_capacity() as u64;

        // Calculate GPU cache usage percentage
        let gpu_cache_usage_perc = if total_capacity > 0 {
            active_blocks_count as f32 / total_capacity as f32
        } else {
            0.0
        };

        ForwardPassMetrics {
            request_active_slots: state.running.len() as u64,
            request_total_slots: 420, // Dummy value as specified
            kv_active_blocks: active_blocks_count,
            kv_total_blocks: total_capacity,
            num_requests_waiting: state.waiting.len() as u64,
            gpu_cache_usage_perc,
            gpu_prefix_cache_hit_rate: 0.0, // Placeholder value as specified
        }
    }
}

/// Convert a Request to an ActiveSequence
fn get_active_sequence(request: Request, block_size: usize, chunk_size: usize) -> ActiveSequence {
    if let Request::Active(active_seq) = request {
        return active_seq;
    }

    let Request::Direct(direct_request) = request else {
        unreachable!("Request must be either Direct or Active");
    };

    ActiveSequence::new(
        direct_request.tokens,
        direct_request.max_output_tokens,
        Some(block_size),
        Some(chunk_size),
    )
}

/// Processes MoveBlock signals with the KvManager.
///
/// When a signal fails, this function verifies that the failure is for an expected case:
/// specifically a single signal attempting to create a single partial (generation) block.
/// This validation is important because in normal operation, the only legitimate failure
/// case should be when trying to acquire a new generation block - any other failures would
/// indicate an unexpected state in the system.
fn process_signals(
    kv_manager_guard: &mut tokio::sync::MutexGuard<'_, KvManager>,
    signals: &[MoveBlock],
) -> bool {
    for signal in signals {
        if kv_manager_guard.process(signal) {
            continue;
        }

        // Check we have a Use signal with blocks
        let MoveBlock::Use(blocks, _) = signal else {
            panic!("Failed signal is Invalid. Has to fail on generation signal.");
        };

        // Verify the signal contains exactly one block
        if blocks.len() != 1 {
            panic!("Failed signal is Invalid. Can have only one generation signal.");
        }

        // Verify the block is a PartialBlock (generation block)
        if !matches!(blocks[0], UniqueBlock::PartialBlock(_)) {
            panic!("Failed signal is Invalid. Generation block has to be partial.");
        }

        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use std::time::Duration;

    #[rstest]
    #[case::random(false)]
    #[case::caching(true)]
    #[tokio::test]
    async fn test_scheduler_token_generation_patterns(#[case] use_shared_tokens: bool) {
        std::env::set_var("RUST_LOG", "debug");

        let kv_capacity: usize = 500;
        let watermark: f64 = 0.01; // 1% watermark
        let block_size: usize = 64;
        let chunk_size: usize = 256;
        let num_requests: usize = 100;
        let input_len: usize = 1000;
        let max_output_tokens: usize = 100;

        // Create channel for token output
        let (output_tx, mut output_rx) = mpsc::channel::<Uuid>(1024);

        // Create scheduler with internal KvManager
        let scheduler = Scheduler::new(
            kv_capacity,
            watermark,
            block_size,
            Some(chunk_size),
            Some(output_tx),
            None,
        );

        // Create shared tokens for caching case
        let shared_tokens = if use_shared_tokens {
            Some(
                (0..input_len / 2)
                    .map(|_| rand::random::<u32>() % 50000)
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        // Create test requests
        for _ in 0..num_requests {
            let input_tokens = if let Some(ref shared) = shared_tokens {
                // For caching case: use shared tokens for first half, random for second half
                let mut tokens = shared.clone();
                tokens.extend((0..input_len / 2).map(|_| rand::random::<u32>() % 50000));
                tokens
            } else {
                // For random case: create unique random token vector for each request
                (0..input_len)
                    .map(|_| rand::random::<u32>() % 50000)
                    .collect::<Vec<_>>()
            };

            let request = DirectRequest {
                tokens: input_tokens,
                max_output_tokens,
                uuid: None,
            };
            scheduler.receive(request).await;
        }

        let start_time = std::time::Instant::now();

        // Collect all generated tokens (should be num_requests * max_output_tokens)
        let expected_tokens = num_requests * max_output_tokens;
        let mut received_tokens = 0;

        // Set up a timeout that causes the test to panic if no tokens are received for 2 seconds
        let timeout = tokio::time::sleep(Duration::from_secs(2));
        tokio::pin!(timeout);

        // Set up debug ticker interval
        let mut debug_interval = interval(Duration::from_millis(500));

        loop {
            tokio::select! {
                biased;

                // Manual debug ticker that prints forward pass metrics
                _ = debug_interval.tick() => {
                    let _metrics = scheduler.get_forward_pass_metrics().await;
                    // println!("Forward Pass Metrics: {:#?}", _metrics);
                }

                Some(_) = output_rx.recv() => {
                    received_tokens += 1;
                    // Reset timeout whenever we receive a token
                    timeout.set(tokio::time::sleep(Duration::from_secs(2)));
                }

                _ = &mut timeout => {
                    // Break instead of panicking when timeout occurs
                    break;
                }
            }
        }

        // Calculate and print elapsed time
        let elapsed = start_time.elapsed();
        println!(
            "Test completed in: {:?} for {} case",
            elapsed,
            if use_shared_tokens {
                "caching"
            } else {
                "random"
            }
        );

        // Assert that we received the expected number of tokens
        assert!(
            received_tokens > expected_tokens,
            "Received {} tokens but expected more than {}",
            received_tokens,
            expected_tokens
        );
    }
}
