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

use crate::kv_router::protocols::ForwardPassMetrics;
use crate::mocker::kv_manager::KvManager;
use crate::mocker::protocols::DirectRequest;
use crate::mocker::protocols::PrefillCost;
use crate::mocker::sequence::ActiveSequence;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio::time::{interval, Duration};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Enum representing either a direct request or an active sequence
pub enum Request {
    Direct(DirectRequest),
    Active(ActiveSequence),
}

struct SchedulerState {
    waiting_requests: VecDeque<Uuid>,
    running_requests: HashSet<Uuid>,
    requests: HashMap<Uuid, Request>,
}

/// Manages scheduling of requests using KvManager resources
pub struct Scheduler {
    state: Arc<Mutex<SchedulerState>>,
    kv_manager: Arc<Mutex<KvManager>>, // Now need to protect KvManager with Mutex for thread safety
    prefill_costs: Arc<Mutex<HashMap<Uuid, Option<PrefillCost>>>>,
    token_capacity: usize,
    watermark: f64,
    block_size: usize,
    chunk_size: usize,
    background_handle: Option<JoinHandle<()>>,
    request_tx: mpsc::Sender<DirectRequest>,
    cancellation_token: CancellationToken,
}

impl Scheduler {
    /// Create a new Scheduler with the given parameters
    pub fn new(
        kv_capacity: usize,
        watermark: f64,
        block_size: usize,
        chunk_size: Option<usize>,
        output_tx: Option<mpsc::Sender<Uuid>>,
    ) -> Self {
        // Create KvManager internally
        let kv_manager = KvManager::new(kv_capacity, block_size);

        let token_capacity: usize = 8192;
        let state = Arc::new(Mutex::new(SchedulerState {
            waiting_requests: VecDeque::new(),
            running_requests: HashSet::new(),
            requests: HashMap::new(),
        }));

        let kv_manager = Arc::new(Mutex::new(kv_manager));
        let chunk_size = chunk_size.unwrap_or(256);

        let prefill_costs = Arc::new(Mutex::new(HashMap::<Uuid, Option<PrefillCost>>::new()));

        // Create channel for request handling
        let (request_tx, mut request_rx) = mpsc::channel::<DirectRequest>(1024);

        // Create cancellation token
        let cancellation_token = CancellationToken::new();
        let token_clone = cancellation_token.clone();

        // Create a clone for the background task
        let state_clone = state.clone();
        let kv_manager_clone = kv_manager.clone();
        let watermark_clone = watermark;
        let block_size_clone = block_size;
        let chunk_size_clone = chunk_size;
        let prefill_costs_clone = prefill_costs.clone();
        let output_tx_clone = output_tx.clone();

        // Spawn main background task with cancellation token
        let background_handle = tokio::spawn(async move {
            let mut schedule_interval = interval(Duration::from_millis(5));
            let mut simulate_interval = interval(Duration::from_millis(1));

            loop {
                tokio::select! {
                    biased;

                    // Enqueue new request
                    Some(request) = request_rx.recv() => {
                        let uuid = Uuid::new_v4();
                        let mut state = state_clone.lock().await;
                        state.requests.insert(uuid, Request::Direct(request));
                        state.waiting_requests.push_back(uuid);
                    }

                    // Try Scheduling Requests
                    _ = schedule_interval.tick() => {
                        let mut state_guard = state_clone.lock().await;
                        let mut prefill_costs_guard = prefill_costs_clone.lock().await;
                        let mut kv_manager_guard = kv_manager_clone.lock().await;

                        // Process DirectRequests, converting them to ActiveSequence and scheduling them until we can't
                        // schedule anymore. Once a request can't be scheduled, it remains at the front of the waiting queue
                        // as an ActiveSequence for future scheduling attempts. This implements First-Come-First-Served (FCFS)
                        // scheduling, as future requests cannot be processed until the request at the front of the queue is scheduled.
                        while let Some(uuid) = state_guard.waiting_requests.front().copied() {
                            // Get the request for this UUID
                            let request = state_guard.requests.get(&uuid).expect("UUID in waiting queue must have a corresponding request");

                            // Process the request regardless of its type
                            let active_sequence = match request {
                                Request::Direct(direct_request) => {
                                    // Convert DirectRequest to ActiveSequence
                                    ActiveSequence::new(
                                        direct_request.tokens.clone(),
                                        direct_request.max_output_tokens,
                                        Some(block_size_clone),
                                        Some(chunk_size_clone),
                                    )
                                },
                                Request::Active(active_seq) => {
                                    // Use existing ActiveSequence
                                    active_seq.clone()
                                }
                            };

                            // Update the requests map with the ActiveSequence
                            state_guard.requests.insert(uuid, Request::Active(active_sequence.clone()));

                            // Calculate token budget using new_tokens from PrefillCost or treating None as 1
                            let prefill_tokens = prefill_costs_guard.values().map(|cost| {
                                match cost {
                                    Some(cost) => cost.new_tokens,
                                    None => 0,
                                }
                            }).sum::<usize>();
                            let tokens_budget = token_capacity.saturating_sub(prefill_tokens);

                            // Check if it can be scheduled
                            if let Some(prefill_cost) = kv_manager_guard.try_schedule(&active_sequence, watermark_clone, tokens_budget) {
                                // Remove from waiting queue
                                state_guard.waiting_requests.pop_front();

                                // Send create signal to KvManager
                                if let Some(signal) = active_sequence.creation_signal() {
                                    kv_manager_guard.process(signal);
                                }

                                // Add to running requests with the PrefillCost
                                state_guard.running_requests.insert(uuid);
                                // Store the PrefillCost in active_tokens
                                prefill_costs_guard.insert(uuid, Some(prefill_cost));
                            } else {
                                break;
                            }
                        }
                    }

                    // Check for cancellation
                    _ = token_clone.cancelled() => {
                        break;
                    }

                    // Simulate running requests (prefill + decode)
                    _ = simulate_interval.tick() => {
                        let mut state_guard = state_clone.lock().await;
                        let mut prefill_costs_guard = prefill_costs_clone.lock().await;
                        let mut kv_manager_guard = kv_manager_clone.lock().await;

                        // Process each running request
                        let mut uuids_to_remove = Vec::new();
                        // Accumulate total sleep duration
                        let mut total_sleep_duration = Duration::from_millis(0);

                        for uuid in state_guard.running_requests.clone() {
                            // Get the active sequence for this UUID
                            let sequence = state_guard.requests.get_mut(&uuid)
                                .and_then(|req| if let Request::Active(seq) = req { Some(seq) } else { None })
                                .expect("UUID in running_requests must have a corresponding active sequence");

                            // Get the prefill cost for sleep calculation if available
                            let prefill_cost = prefill_costs_guard.get(&uuid).and_then(|cost| cost.as_ref());

                            // Generate token and get signals
                            let signals = sequence.generate();

                            // Accumulate sleep duration based on prefill_compute if available
                            if let Some(cost) = prefill_cost {
                                let sleep_ms = (cost.prefill_compute / 131072.0) as u64;
                                total_sleep_duration += Duration::from_millis(sleep_ms);
                            } else {
                                total_sleep_duration += Duration::from_millis(1);
                            }

                            // Process all signals with the KvManager
                            for signal in signals {
                                kv_manager_guard.process(&signal);
                            }

                            // Send UUID notification for each generated token
                            if let Some(tx) = &output_tx_clone {
                                let _ = tx.try_send(uuid);
                            }

                            // Set active_tokens to None after a token is generated
                            if sequence.generated_tokens == 1 {
                                prefill_costs_guard.insert(uuid, None);
                            }

                            // Check if we're done after generating
                            if sequence.generated_tokens >= sequence.max_output_tokens {
                                uuids_to_remove.push(uuid);
                            }
                        }

                        // Remove completed sequences
                        for uuid in uuids_to_remove {
                            state_guard.running_requests.remove(&uuid);
                            state_guard.requests.remove(&uuid);
                            prefill_costs_guard.remove(&uuid);
                        }

                        // Sleep once for the accumulated duration
                        if total_sleep_duration.as_millis() > 0 {
                            tokio::time::sleep(total_sleep_duration).await;
                        }
                    }
                }
            }
        });

        Self {
            state,
            kv_manager,
            prefill_costs,
            token_capacity,
            watermark,
            block_size,
            chunk_size,
            background_handle: Some(background_handle),
            request_tx,
            cancellation_token,
        }
    }

    /// Add a new request to the waiting queue
    pub async fn receive_request(&self, request: DirectRequest) {
        let _ = self.request_tx.send(request).await;
    }

    /// Get the count of waiting requests
    pub async fn waiting_count(&self) -> usize {
        let state = self.state.lock().await;
        state.waiting_requests.len()
    }

    /// Get the count of running requests
    pub async fn running_count(&self) -> usize {
        let state = self.state.lock().await;
        state.running_requests.len()
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
        let active_blocks_count = kv_manager.active_blocks.len() as u64;
        let total_capacity = kv_manager.max_capacity as u64;

        // Calculate GPU cache usage percentage
        let gpu_cache_usage_perc = if total_capacity > 0 {
            active_blocks_count as f32 / total_capacity as f32
        } else {
            0.0
        };

        ForwardPassMetrics {
            request_active_slots: state.running_requests.len() as u64,
            request_total_slots: 420, // Dummy value as specified
            kv_active_blocks: active_blocks_count,
            kv_total_blocks: total_capacity,
            num_requests_waiting: state.waiting_requests.len() as u64,
            gpu_cache_usage_perc,
            gpu_prefix_cache_hit_rate: 0.0, // Placeholder value as specified
        }
    }
}

// Implement Clone for Scheduler to support sharing between tasks
impl Clone for Scheduler {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            kv_manager: self.kv_manager.clone(),
            prefill_costs: self.prefill_costs.clone(),
            token_capacity: self.token_capacity,
            watermark: self.watermark,
            block_size: self.block_size,
            chunk_size: self.chunk_size,
            background_handle: None,
            request_tx: self.request_tx.clone(),
            cancellation_token: self.cancellation_token.clone(),
        }
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        // Only the original instance should cancel
        if self.background_handle.is_some() {
            self.cancellation_token.cancel();
        }

        // Abort the background task if this is the original instance
        if let Some(handle) = self.background_handle.take() {
            handle.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_scheduler_token_generation() {
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
        );

        // Create test requests
        for _ in 0..num_requests {
            // Create unique random token vector for each request
            let input_tokens = (0..input_len)
                .map(|_| rand::random::<u32>() % 50000)
                .collect::<Vec<_>>();

            let request = DirectRequest {
                tokens: input_tokens,
                max_output_tokens,
            };
            scheduler.receive_request(request).await;
        }
        let start_time = std::time::Instant::now();

        // Collect all generated tokens (should be num_requests * max_output_tokens)
        let expected_tokens = num_requests * max_output_tokens as usize;
        let mut received_tokens = 0;

        // Set up a timeout that causes the test to panic if no tokens are received for 5 consecutive seconds.
        let timeout = tokio::time::sleep(Duration::from_secs(5));
        tokio::pin!(timeout);

        // Set up debug ticker interval
        let mut debug_interval = interval(Duration::from_millis(500));

        loop {
            tokio::select! {
                biased;

                // Manual debug ticker that prints forward pass metrics
                _ = debug_interval.tick() => {
                    let metrics = scheduler.get_forward_pass_metrics().await;
                    println!("Forward Pass Metrics: {:#?}", metrics);
                }

                Some(_) = output_rx.recv() => {
                    received_tokens += 1;
                    if received_tokens == expected_tokens {
                        // Calculate and print elapsed time
                        let elapsed = start_time.elapsed();
                        println!("Test completed in: {:?}", elapsed);
                        break;
                    }
                    // Reset timeout whenever we receive a token
                    timeout.set(tokio::time::sleep(Duration::from_secs(5)));
                }

                _ = &mut timeout => {
                    panic!("Test timed out after 5 seconds of inactivity! Received only {} of {} expected tokens",
                           received_tokens, expected_tokens);
                }
            }
        }

        // Wait to ensure all sequences are cleaned up
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Verify no running requests remain
        assert_eq!(scheduler.running_count().await, 0);
    }
}
